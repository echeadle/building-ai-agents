"""
Exercise Solution: Resilient Tool Executor

This solution demonstrates all five Python concepts from Appendix A:
1. Async/await for concurrent execution
2. Type hints and Pydantic for validation
3. Context managers for resource management
4. Decorators for retry and timing
5. Dataclasses for configuration

Appendix A: Python Refresher for Agent Development
"""

import os
import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# ============================================================================
# 1. DATACLASSES FOR CONFIGURATION
# ============================================================================

@dataclass
class ToolConfig:
    """Configuration for tool executor."""
    
    timeout: float = 30.0  # seconds
    max_retries: int = 3
    cache_ttl: float = 300.0  # seconds
    backoff_factor: float = 2.0
    
    # Computed fields
    created_at: datetime = field(default_factory=datetime.now, init=False)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")


@dataclass
class ToolResult:
    """Result from a tool execution."""
    
    tool_name: str
    result: Any
    duration_ms: float
    cached: bool = False
    attempts: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# 2. PYDANTIC FOR VALIDATION
# ============================================================================

class ToolInput(BaseModel):
    """Input for tool execution with validation."""
    
    tool_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the tool to execute"
    )
    
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters"
    )
    
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Execution priority (1-10)"
    )
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is valid."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Tool name must be alphanumeric (underscores allowed)")
        return v.lower()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tool_name": "calculator",
                    "parameters": {"expression": "2 + 2"},
                    "priority": 8
                }
            ]
        }
    }


# ============================================================================
# 3. DECORATORS FOR RETRY AND TIMING
# ============================================================================

def retry_async(config: ToolConfig):
    """
    Retry async functions with exponential backoff.
    
    Uses configuration for max_retries and backoff_factor.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        wait_time = config.backoff_factor ** (attempt - 1)
                        print(f"  ⚠️  Attempt {attempt} failed: {str(e)[:50]}")
                        print(f"  ⏳ Retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  ❌ All {config.max_retries} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


def measure_time_async(func: Callable):
    """Measure async function execution time."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Attach timing to result if it's a ToolResult
        if isinstance(result, ToolResult):
            result.duration_ms = elapsed_ms
        
        return result
    
    return wrapper


# ============================================================================
# 4. CONTEXT MANAGERS FOR RESOURCE MANAGEMENT
# ============================================================================

@asynccontextmanager
async def tool_execution_context(tool_input: ToolInput) -> AsyncGenerator[dict, None]:
    """
    Context manager for tool execution.
    
    Provides setup and cleanup for tool execution, including:
    - Logging start/end
    - Resource allocation
    - Error handling
    """
    execution_id = hashlib.md5(
        f"{tool_input.tool_name}-{time.time()}".encode()
    ).hexdigest()[:8]
    
    context = {
        "execution_id": execution_id,
        "start_time": time.perf_counter(),
        "tool_name": tool_input.tool_name,
    }
    
    print(f"\n🔧 Starting execution [{execution_id}]: {tool_input.tool_name}")
    
    try:
        yield context
    except Exception as e:
        print(f"❌ Execution [{execution_id}] failed: {e}")
        raise
    finally:
        elapsed = time.perf_counter() - context["start_time"]
        print(f"✅ Execution [{execution_id}] completed in {elapsed:.2f}s")


# ============================================================================
# 5. TOOL EXECUTOR (COMBINING ALL CONCEPTS)
# ============================================================================

class ToolExecutor:
    """
    Production-ready tool executor demonstrating all Python concepts.
    
    Features:
    - Async/await for concurrent execution
    - Pydantic validation for inputs
    - Context managers for resource management
    - Decorators for retry and timing
    - Dataclasses for configuration and results
    """
    
    def __init__(self, config: ToolConfig):
        self.config = config
        self.tools: dict[str, Callable] = {}
        self.cache: dict[str, tuple[Any, float]] = {}
        self.execution_count = 0
    
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function (sync or async)."""
        self.tools[name.lower()] = func
        print(f"📝 Registered tool: {name}")
    
    def _get_cache_key(self, tool_input: ToolInput) -> str:
        """Generate cache key from tool input."""
        cache_data = {
            "tool": tool_input.tool_name,
            "params": json.dumps(tool_input.parameters, sort_keys=True)
        }
        return hashlib.md5(json.dumps(cache_data).encode()).hexdigest()
    
    def _check_cache(self, tool_input: ToolInput) -> Optional[Any]:
        """Check if result is in cache and still valid."""
        cache_key = self._get_cache_key(tool_input)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            age = time.time() - timestamp
            
            if age < self.config.cache_ttl:
                print(f"  💾 Cache hit (age: {age:.1f}s)")
                return result
            else:
                print(f"  🗑️  Cache expired (age: {age:.1f}s)")
                del self.cache[cache_key]
        
        return None
    
    def _store_cache(self, tool_input: ToolInput, result: Any) -> None:
        """Store result in cache."""
        cache_key = self._get_cache_key(tool_input)
        self.cache[cache_key] = (result, time.time())
        print(f"  💾 Cached result for {self.config.cache_ttl}s")
    
    @measure_time_async
    async def _execute_single(
        self,
        tool_input: ToolInput,
        context: dict
    ) -> ToolResult:
        """Execute a single tool with timeout protection."""
        tool_name = tool_input.tool_name
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Check cache
        cached_result = self._check_cache(tool_input)
        if cached_result is not None:
            return ToolResult(
                tool_name=tool_name,
                result=cached_result,
                duration_ms=0,
                cached=True,
                attempts=0
            )
        
        # Execute tool
        func = self.tools[tool_name]
        
        try:
            # Apply timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**tool_input.parameters),
                    timeout=self.config.timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**tool_input.parameters)),
                    timeout=self.config.timeout
                )
            
            # Cache the result
            self._store_cache(tool_input, result)
            
            return ToolResult(
                tool_name=tool_name,
                result=result,
                duration_ms=0,  # Will be set by measure_time_async
                cached=False,
                attempts=context.get("attempt", 1)
            )
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool {tool_name} timed out after {self.config.timeout}s")
    
    async def execute(self, tool_input: ToolInput) -> ToolResult:
        """
        Execute a tool with full resilience.
        
        Combines:
        - Context manager for setup/cleanup
        - Retry decorator for resilience
        - Timeout protection
        - Caching for performance
        """
        # Validate input with Pydantic
        validated_input = ToolInput.model_validate(tool_input.model_dump())
        
        async with tool_execution_context(validated_input) as context:
            # Create retry-wrapped executor
            @retry_async(self.config)
            async def execute_with_retry():
                return await self._execute_single(validated_input, context)
            
            # Execute with retry
            result = await execute_with_retry()
            self.execution_count += 1
            
            return result
    
    async def execute_batch(self, tools: list[ToolInput]) -> list[ToolResult]:
        """Execute multiple tools in parallel."""
        print(f"\n🚀 Executing batch of {len(tools)} tools in parallel...")
        
        tasks = [self.execute(tool) for tool in tools]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successes from failures
        successes = [r for r in results if isinstance(r, ToolResult)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        print(f"\n📊 Batch complete: {len(successes)} succeeded, {len(failures)} failed")
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_executions": self.execution_count,
            "registered_tools": len(self.tools),
            "cache_size": len(self.cache),
            "cache_ttl": self.config.cache_ttl,
            "max_retries": self.config.max_retries,
        }


# ============================================================================
# EXAMPLE TOOLS
# ============================================================================

async def calculator_tool(expression: str) -> dict:
    """Simple calculator tool (async)."""
    await asyncio.sleep(0.1)  # Simulate work
    
    try:
        result = eval(expression)  # In production, use safe eval!
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def uppercase_tool(text: str) -> dict:
    """Uppercase conversion tool (sync)."""
    time.sleep(0.05)  # Simulate work
    return {"original": text, "uppercase": text.upper()}


async def slow_tool(seconds: float) -> dict:
    """Tool that simulates slow operation."""
    await asyncio.sleep(seconds)
    return {"waited": seconds, "message": f"Waited {seconds}s"}


# ============================================================================
# MAIN DEMO
# ============================================================================

async def main():
    """Demonstrate the complete tool executor."""
    print("=" * 70)
    print("RESILIENT TOOL EXECUTOR - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Create configuration
    config = ToolConfig(
        timeout=5.0,
        max_retries=3,
        cache_ttl=10.0,
        backoff_factor=2.0
    )
    
    print(f"\nConfiguration:")
    print(f"  Timeout: {config.timeout}s")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Cache TTL: {config.cache_ttl}s")
    
    # Create executor
    executor = ToolExecutor(config)
    
    # Register tools
    executor.register_tool("calculator", calculator_tool)
    executor.register_tool("uppercase", uppercase_tool)
    executor.register_tool("slow", slow_tool)
    
    # Example 1: Single execution
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Tool Execution")
    print("=" * 70)
    
    tool1 = ToolInput(
        tool_name="calculator",
        parameters={"expression": "2 + 2"},
        priority=10
    )
    
    result1 = await executor.execute(tool1)
    print(f"\nResult: {result1.result}")
    print(f"Duration: {result1.duration_ms:.2f}ms")
    print(f"Cached: {result1.cached}")
    
    # Example 2: Cache hit
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Cache Hit (Same Tool)")
    print("=" * 70)
    
    result2 = await executor.execute(tool1)  # Same input - should hit cache
    print(f"\nResult: {result2.result}")
    print(f"Cached: {result2.cached}")
    
    # Example 3: Parallel execution
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Parallel Batch Execution")
    print("=" * 70)
    
    batch = [
        ToolInput(tool_name="calculator", parameters={"expression": "10 * 5"}),
        ToolInput(tool_name="uppercase", parameters={"text": "hello world"}),
        ToolInput(tool_name="calculator", parameters={"expression": "100 / 4"}),
        ToolInput(tool_name="uppercase", parameters={"text": "python rocks"}),
    ]
    
    batch_results = await executor.execute_batch(batch)
    
    print("\n📋 Batch Results:")
    for i, result in enumerate(batch_results, 1):
        if isinstance(result, ToolResult):
            print(f"  {i}. {result.tool_name}: {result.result}")
        else:
            print(f"  {i}. Error: {result}")
    
    # Example 4: Error handling and retry
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Error Handling (Invalid Expression)")
    print("=" * 70)
    
    try:
        tool_error = ToolInput(
            tool_name="calculator",
            parameters={"expression": "invalid + syntax"}
        )
        result_error = await executor.execute(tool_error)
        print(f"\nResult: {result_error.result}")
    except Exception as e:
        print(f"\n❌ Caught error: {e}")
    
    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
