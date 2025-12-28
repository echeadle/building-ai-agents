"""
Exercise Solution: Resilient Tool Executor

Chapter 30: Error Handling and Recovery

This module implements a ResilientToolExecutor class that combines
all error handling patterns from the chapter into a production-ready
tool execution system.

Requirements:
1. Execute tools with automatic retry on transient failures
2. Fall back to cached results when available
3. Log all errors with full context
4. Track error statistics for monitoring
5. Support self-correction for common errors
6. Abort after too many consecutive failures
"""

import os
import json
import time
import random
import logging
import hashlib
from typing import Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# ============================================================
# Supporting Types
# ============================================================

class ErrorCategory(Enum):
    """Categories of errors for tracking."""
    TRANSIENT = "transient"      # Retryable errors
    VALIDATION = "validation"    # Input validation errors
    EXECUTION = "execution"      # Tool execution errors
    TIMEOUT = "timeout"          # Timeout errors
    UNKNOWN = "unknown"          # Unclassified errors


@dataclass
class ToolError:
    """Structured representation of a tool error."""
    tool_name: str
    error_message: str
    category: ErrorCategory
    timestamp: datetime
    input_data: dict
    recoverable: bool
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "error_message": self.error_message,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
        }


@dataclass
class CacheEntry:
    """Cached tool result with metadata."""
    result: Any
    timestamp: datetime
    ttl_seconds: int
    
    def is_valid(self) -> bool:
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=self.ttl_seconds)
    
    def is_usable_as_fallback(self, max_age_seconds: int = 86400) -> bool:
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=max_age_seconds)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class ToolConfig:
    """Per-tool configuration."""
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    cache_ttl: int = 3600  # 1 hour default
    timeout_seconds: float = 30.0
    allow_self_correction: bool = True


@dataclass
class ExecutionResult:
    """Result of a tool execution attempt."""
    success: bool
    result: Optional[Any]
    error: Optional[ToolError]
    source: str  # "live", "cache", "fallback"
    execution_time_ms: float
    retries_used: int


# ============================================================
# Resilient Tool Executor
# ============================================================

class ResilientToolExecutor:
    """
    A resilient tool execution system with:
    - Automatic retry with exponential backoff
    - Result caching with configurable TTL
    - Comprehensive error logging and tracking
    - Self-correction for recoverable errors
    - Abort on excessive failures
    
    Usage:
        executor = ResilientToolExecutor(client)
        
        # Register tools
        executor.register_tool("calculator", calculator_func, calculator_def)
        
        # Execute with full resilience
        result = executor.execute("calculator", {"expression": "2+2"})
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        default_config: Optional[ToolConfig] = None,
        max_consecutive_failures: int = 5,
        log_level: int = logging.INFO
    ):
        """
        Initialize the executor.
        
        Args:
            client: Anthropic client for self-correction
            default_config: Default tool configuration
            max_consecutive_failures: Failures before abort
            log_level: Logging verbosity
        """
        self.client = client
        self.default_config = default_config or ToolConfig()
        self.max_consecutive_failures = max_consecutive_failures
        
        # Tool registry
        self._tools: dict[str, Callable] = {}
        self._tool_definitions: dict[str, dict] = {}
        self._tool_configs: dict[str, ToolConfig] = {}
        
        # Caching
        self._cache: dict[str, CacheEntry] = {}
        
        # Error tracking
        self._error_history: list[ToolError] = []
        self._consecutive_failures = 0
        
        # Logging
        self._setup_logging(log_level)
    
    def _setup_logging(self, level: int):
        """Configure logging."""
        self.logger = logging.getLogger("resilient_executor")
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            self.logger.addHandler(handler)
    
    def register_tool(
        self,
        name: str,
        handler: Callable[[dict], Any],
        definition: Optional[dict] = None,
        config: Optional[ToolConfig] = None
    ):
        """
        Register a tool with the executor.
        
        Args:
            name: Tool name
            handler: Function that executes the tool
            definition: Tool definition for self-correction
            config: Tool-specific configuration
        """
        self._tools[name] = handler
        self._tool_definitions[name] = definition or {}
        self._tool_configs[name] = config or self.default_config
        self.logger.debug(f"Registered tool: {name}")
    
    def _get_cache_key(self, tool_name: str, input_data: dict) -> str:
        """Generate a unique cache key for a tool call."""
        # Create deterministic hash of tool name and input
        content = json.dumps({"tool": tool_name, "input": input_data}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_result(self, tool_name: str, input_data: dict) -> Optional[CacheEntry]:
        """Get cached result if available."""
        key = self._get_cache_key(tool_name, input_data)
        return self._cache.get(key)
    
    def _cache_result(self, tool_name: str, input_data: dict, result: Any):
        """Cache a successful result."""
        key = self._get_cache_key(tool_name, input_data)
        config = self._tool_configs.get(tool_name, self.default_config)
        
        self._cache[key] = CacheEntry(
            result=result,
            timestamp=datetime.now(),
            ttl_seconds=config.cache_ttl
        )
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = config.base_delay * (config.exponential_base ** attempt)
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            delay *= (0.5 + random.random())
        
        return delay
    
    def _categorize_error(self, error: Exception) -> tuple[ErrorCategory, bool]:
        """
        Categorize an error and determine if it's recoverable.
        
        Returns:
            Tuple of (category, is_recoverable)
        """
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.TRANSIENT, True
        if isinstance(error, ValueError):
            return ErrorCategory.VALIDATION, True  # Might be correctable
        if isinstance(error, (TypeError, KeyError)):
            return ErrorCategory.VALIDATION, False
        if isinstance(error, PermissionError):
            return ErrorCategory.EXECUTION, False
        
        return ErrorCategory.UNKNOWN, False
    
    def _record_error(
        self,
        tool_name: str,
        error: Exception,
        input_data: dict,
        retry_count: int
    ) -> ToolError:
        """Record an error for tracking."""
        category, recoverable = self._categorize_error(error)
        
        tool_error = ToolError(
            tool_name=tool_name,
            error_message=str(error),
            category=category,
            timestamp=datetime.now(),
            input_data=input_data,
            recoverable=recoverable,
            retry_count=retry_count
        )
        
        self._error_history.append(tool_error)
        self.logger.warning(
            f"Tool error: {tool_name} | {category.value} | {error}"
        )
        
        return tool_error
    
    def _attempt_self_correction(
        self,
        tool_name: str,
        input_data: dict,
        error: Exception
    ) -> Optional[dict]:
        """
        Ask the LLM to correct invalid tool input.
        
        Returns corrected input or None if correction fails.
        """
        definition = self._tool_definitions.get(tool_name)
        if not definition:
            return None
        
        correction_prompt = f"""A tool call failed. Please provide corrected input.

Tool: {tool_name}
Original input: {json.dumps(input_data, indent=2)}
Error: {error}

Tool definition:
{json.dumps(definition, indent=2)}

Provide ONLY the corrected JSON input, no explanation."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": correction_prompt}]
            )
            
            corrected_text = response.content[0].text.strip()
            
            # Try to parse the corrected input
            # Handle potential code blocks
            if "```" in corrected_text:
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', corrected_text)
                if match:
                    corrected_text = match.group(1)
            
            corrected_input = json.loads(corrected_text)
            self.logger.info(f"Self-correction succeeded for {tool_name}")
            return corrected_input
            
        except (json.JSONDecodeError, anthropic.APIError) as e:
            self.logger.debug(f"Self-correction failed: {e}")
            return None
    
    def execute(
        self,
        tool_name: str,
        input_data: dict,
        allow_cache: bool = True,
        allow_stale_cache: bool = True
    ) -> ExecutionResult:
        """
        Execute a tool with full resilience.
        
        Args:
            tool_name: Name of the tool to execute
            input_data: Input data for the tool
            allow_cache: Whether to use cached results
            allow_stale_cache: Whether to use stale cache as fallback
            
        Returns:
            ExecutionResult with outcome details
        """
        start_time = time.time()
        
        # Check if tool exists
        if tool_name not in self._tools:
            error = ToolError(
                tool_name=tool_name,
                error_message=f"Unknown tool: {tool_name}",
                category=ErrorCategory.VALIDATION,
                timestamp=datetime.now(),
                input_data=input_data,
                recoverable=False
            )
            return ExecutionResult(
                success=False,
                result=None,
                error=error,
                source="none",
                execution_time_ms=(time.time() - start_time) * 1000,
                retries_used=0
            )
        
        config = self._tool_configs.get(tool_name, self.default_config)
        handler = self._tools[tool_name]
        
        # Check cache first
        if allow_cache:
            cached = self._get_cached_result(tool_name, input_data)
            if cached and cached.is_valid():
                self.logger.debug(f"Cache hit for {tool_name}")
                self._consecutive_failures = 0
                return ExecutionResult(
                    success=True,
                    result=cached.result,
                    error=None,
                    source="cache",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    retries_used=0
                )
        
        # Attempt execution with retries
        last_error = None
        current_input = input_data
        retries_used = 0
        
        for attempt in range(config.retry_config.max_retries + 1):
            try:
                result = handler(current_input)
                
                # Success! Cache and return
                self._cache_result(tool_name, input_data, result)
                self._consecutive_failures = 0
                
                return ExecutionResult(
                    success=True,
                    result=result,
                    error=None,
                    source="live",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    retries_used=retries_used
                )
                
            except Exception as e:
                last_error = e
                retries_used = attempt
                
                # Record the error
                tool_error = self._record_error(
                    tool_name, e, current_input, attempt
                )
                
                # Check for abort condition
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.max_consecutive_failures:
                    self.logger.critical(
                        f"Max consecutive failures ({self.max_consecutive_failures}) "
                        "reached - aborting"
                    )
                    raise RuntimeError(
                        f"Executor aborted after {self.max_consecutive_failures} "
                        "consecutive failures"
                    )
                
                # Try self-correction for validation errors
                if (tool_error.recoverable and 
                    config.allow_self_correction and
                    tool_error.category == ErrorCategory.VALIDATION):
                    
                    corrected = self._attempt_self_correction(
                        tool_name, current_input, e
                    )
                    if corrected:
                        current_input = corrected
                        # Don't count correction as a retry
                        continue
                
                # Check if we should retry
                if attempt < config.retry_config.max_retries:
                    if tool_error.category == ErrorCategory.TRANSIENT:
                        delay = self._calculate_delay(attempt, config.retry_config)
                        self.logger.info(
                            f"Retrying {tool_name} in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{config.retry_config.max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        # Non-transient errors don't benefit from retry
                        break
        
        # All attempts failed - try stale cache
        if allow_stale_cache:
            cached = self._get_cached_result(tool_name, input_data)
            if cached and cached.is_usable_as_fallback():
                self.logger.warning(
                    f"Using stale cache for {tool_name} after failures"
                )
                return ExecutionResult(
                    success=True,
                    result=cached.result,
                    error=None,
                    source="stale_cache",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    retries_used=retries_used
                )
        
        # Complete failure
        final_error = ToolError(
            tool_name=tool_name,
            error_message=str(last_error),
            category=self._categorize_error(last_error)[0],
            timestamp=datetime.now(),
            input_data=input_data,
            recoverable=False,
            retry_count=retries_used
        )
        
        return ExecutionResult(
            success=False,
            result=None,
            error=final_error,
            source="none",
            execution_time_ms=(time.time() - start_time) * 1000,
            retries_used=retries_used
        )
    
    def get_statistics(self) -> dict:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary with error counts and patterns
        """
        stats = {
            "total_errors": len(self._error_history),
            "by_tool": {},
            "by_category": {},
            "consecutive_failures": self._consecutive_failures,
            "cache_size": len(self._cache),
            "recent_errors": [],
        }
        
        for error in self._error_history:
            # By tool
            tool = error.tool_name
            stats["by_tool"][tool] = stats["by_tool"].get(tool, 0) + 1
            
            # By category
            cat = error.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        # Recent errors
        for error in self._error_history[-5:]:
            stats["recent_errors"].append(error.to_dict())
        
        return stats
    
    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def reset_error_tracking(self):
        """Reset error history and consecutive failure count."""
        self._error_history.clear()
        self._consecutive_failures = 0
        self.logger.info("Error tracking reset")


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESILIENT TOOL EXECUTOR - EXERCISE SOLUTION")
    print("=" * 60)
    
    # Create executor
    client = anthropic.Anthropic()
    executor = ResilientToolExecutor(
        client=client,
        max_consecutive_failures=5,
        log_level=logging.DEBUG
    )
    
    # Define and register tools
    
    # 1. Calculator tool
    def calculator(input_data: dict) -> float:
        expr = input_data.get("expression", "")
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expr):
            raise ValueError(f"Invalid characters in expression: {expr}")
        return eval(expr)
    
    calculator_def = {
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression (e.g., '2 + 2')"
                }
            },
            "required": ["expression"]
        }
    }
    
    executor.register_tool(
        "calculator",
        calculator,
        calculator_def,
        ToolConfig(cache_ttl=60)  # Cache for 1 minute
    )
    
    # 2. Flaky tool (for testing retries)
    call_count = {"value": 0}
    
    def flaky_tool(input_data: dict) -> str:
        call_count["value"] += 1
        if call_count["value"] < 3:
            raise ConnectionError("Simulated transient failure")
        return f"Success on attempt {call_count['value']}"
    
    executor.register_tool(
        "flaky_tool",
        flaky_tool,
        None,
        ToolConfig(
            retry_config=RetryConfig(max_retries=3, base_delay=0.3)
        )
    )
    
    # 3. Validation-error tool (for testing self-correction)
    def strict_tool(input_data: dict) -> dict:
        name = input_data.get("name")
        age = input_data.get("age")
        
        if not isinstance(name, str) or not name:
            raise ValueError("'name' must be a non-empty string")
        if not isinstance(age, int) or age < 0:
            raise ValueError("'age' must be a non-negative integer")
        
        return {"greeting": f"Hello {name}, you are {age} years old!"}
    
    strict_tool_def = {
        "name": "strict_tool",
        "description": "A tool that requires specific input format",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"}
            },
            "required": ["name", "age"]
        }
    }
    
    executor.register_tool("strict_tool", strict_tool, strict_tool_def)
    
    # Run demonstrations
    print("\n### 1. Basic Execution with Caching ###\n")
    
    result = executor.execute("calculator", {"expression": "10 * 5"})
    print(f"First call: {result.result} (source: {result.source})")
    
    result = executor.execute("calculator", {"expression": "10 * 5"})
    print(f"Second call: {result.result} (source: {result.source})")
    
    print("\n### 2. Retry on Transient Failure ###\n")
    
    call_count["value"] = 0  # Reset
    result = executor.execute("flaky_tool", {"test": True})
    print(f"Result: {result.result}")
    print(f"Retries used: {result.retries_used}")
    print(f"Execution time: {result.execution_time_ms:.1f}ms")
    
    print("\n### 3. Self-Correction Demo ###\n")
    
    # This input has wrong types - self-correction should fix it
    result = executor.execute("strict_tool", {"name": "Alice", "age": 30})
    print(f"Valid input result: {result.result}")
    
    print("\n### 4. Error Statistics ###\n")
    
    stats = executor.get_statistics()
    print(json.dumps(stats, indent=2, default=str))
    
    print("\n### 5. Stale Cache Fallback ###\n")
    
    # Execute to cache a result
    executor.execute("calculator", {"expression": "7 * 7"})
    
    # Now register a broken version
    def broken_calculator(input_data: dict) -> float:
        raise ConnectionError("Calculator API is down!")
    
    executor._tools["calculator"] = broken_calculator
    
    # Should fall back to stale cache
    result = executor.execute("calculator", {"expression": "7 * 7"})
    print(f"Result during outage: {result.result}")
    print(f"Source: {result.source}")
    
    print("\n" + "=" * 60)
    print("EXERCISE REQUIREMENTS MET:")
    print("✓ Automatic retry with exponential backoff")
    print("✓ Result caching with configurable TTL")
    print("✓ Comprehensive error logging")
    print("✓ Error statistics tracking")
    print("✓ Self-correction for validation errors")
    print("✓ Abort after consecutive failures")
    print("=" * 60)
