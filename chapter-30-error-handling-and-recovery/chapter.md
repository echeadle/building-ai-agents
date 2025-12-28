---
chapter: 30
title: "Error Handling and Recovery"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 30: Error Handling and Recovery

## Introduction

Your agent is humming along beautifully. It's planned out a complex research task, called the web search tool, started parsing results—and then the API returns a 500 error. Or the LLM generates invalid JSON. Or the agent gets stuck in a loop, calling the same tool over and over with identical arguments. What happens next determines whether your agent is a prototype or a production system.

In the previous chapters, we've built the core components of our agent: the agentic loop, state management, and planning capabilities. Now we need to make it robust. Errors in agent systems aren't just possible—they're inevitable. APIs go down, LLMs hallucinate invalid tool calls, network connections drop, and rate limits kick in. The difference between a frustrating agent and a reliable one isn't the absence of errors—it's how gracefully it handles them.

In this chapter, we'll build a comprehensive error handling system that catches failures, retries intelligently, falls back gracefully, and even lets the agent correct its own mistakes.

## Learning Objectives

By the end of this chapter, you will be able to:

- Identify and categorize the different types of errors that occur in agent systems
- Implement retry logic with exponential backoff for transient failures
- Design fallback strategies that maintain agent functionality during partial failures
- Build self-correction patterns that let agents recover from their own mistakes
- Create a reusable error handling module for production agents

## Types of Agent Errors

Before we can handle errors, we need to understand them. Agent errors fall into several distinct categories, each requiring different handling strategies.

### API and Network Errors

These are the most common errors you'll encounter. They include:

- **Connection errors**: The network is down or the API endpoint is unreachable
- **Timeout errors**: The API took too long to respond
- **Rate limit errors**: You've exceeded your API quota
- **Authentication errors**: Your API key is invalid or expired
- **Server errors**: The API is experiencing internal issues (5xx errors)

The good news? Most of these are transient. A retry after a brief wait often succeeds.

### LLM Output Errors

The LLM can produce outputs that your code can't process:

- **Malformed JSON**: The LLM was asked for JSON but produced invalid syntax
- **Schema violations**: The JSON is valid but doesn't match your expected structure
- **Invalid tool calls**: The LLM called a tool that doesn't exist or used wrong parameters
- **Incomplete responses**: The response was cut off due to token limits

These often require re-prompting or asking the LLM to fix its output.

### Tool Execution Errors

Your tools can fail for many reasons:

- **External API failures**: The weather API is down, the database is unreachable
- **Invalid input**: The LLM passed bad arguments to your tool
- **Resource errors**: File not found, permission denied, disk full
- **Timeout**: The tool took too long to execute

Tool errors need to be reported back to the LLM so it can adapt its strategy.

### Agent Logic Errors

These are higher-level problems with the agent's behavior:

- **Infinite loops**: The agent keeps repeating the same action
- **Goal abandonment**: The agent gets distracted and forgets its objective
- **Stuck states**: The agent doesn't know how to proceed
- **Resource exhaustion**: Too many tokens consumed, too many API calls made

Logic errors often require intervention—either automatic safeguards or human involvement.

Let's look at how these errors manifest in code:

```python
"""
Demonstrating different types of agent errors.

Chapter 30: Error Handling and Recovery
"""

import os
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

client = anthropic.Anthropic()


def demonstrate_api_errors():
    """Show different API error types."""
    
    # Connection error (simulated with bad base URL)
    try:
        bad_client = anthropic.Anthropic(
            base_url="https://nonexistent.anthropic.com"
        )
        bad_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.APIConnectionError as e:
        print(f"Connection Error: {e}")
    
    # Authentication error (simulated with bad key)
    try:
        bad_client = anthropic.Anthropic(api_key="invalid-key")
        bad_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.AuthenticationError as e:
        print(f"Authentication Error: {e}")


def demonstrate_output_errors():
    """Show LLM output parsing errors."""
    
    # Simulating malformed JSON response
    bad_json = '{"name": "test", "value": }'
    try:
        json.loads(bad_json)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
    
    # Schema violation - valid JSON, wrong structure
    response = '{"wrong_field": "data"}'
    parsed = json.loads(response)
    if "expected_field" not in parsed:
        print("Schema Violation: 'expected_field' missing from response")


def demonstrate_tool_errors():
    """Show tool execution errors."""
    
    # File not found
    try:
        with open("/nonexistent/path/file.txt") as f:
            f.read()
    except FileNotFoundError as e:
        print(f"Tool Error (File): {e}")
    
    # Invalid input to tool
    def calculate_percentage(value: float, total: float) -> float:
        if total == 0:
            raise ValueError("Cannot calculate percentage with zero total")
        return (value / total) * 100
    
    try:
        calculate_percentage(50, 0)
    except ValueError as e:
        print(f"Tool Error (Input): {e}")


if __name__ == "__main__":
    print("=== API Errors ===")
    demonstrate_api_errors()
    
    print("\n=== Output Errors ===")
    demonstrate_output_errors()
    
    print("\n=== Tool Errors ===")
    demonstrate_tool_errors()
```

## Graceful Degradation Strategies

When errors occur, your agent shouldn't just crash. Instead, it should degrade gracefully—maintaining as much functionality as possible while communicating limitations clearly. Here are the key strategies:

### Fallback Responses

When a tool fails, provide a sensible default or alternative:

```python
def get_weather_with_fallback(location: str) -> dict:
    """Get weather with fallback to cached or default data."""
    try:
        return fetch_live_weather(location)
    except WeatherAPIError:
        # Try cached data
        cached = get_cached_weather(location)
        if cached and not is_stale(cached):
            return {**cached, "source": "cache", "note": "Live data unavailable"}
        
        # Return acknowledgment of failure
        return {
            "error": True,
            "message": f"Weather data for {location} is currently unavailable",
            "suggestion": "Try again in a few minutes"
        }
```

### Capability Reduction

If a critical component fails, continue with reduced capabilities:

```python
def create_agent_with_fallbacks(tools: list[dict]) -> dict:
    """Create an agent, disabling tools that fail health checks."""
    available_tools = []
    disabled_tools = []
    
    for tool in tools:
        if check_tool_health(tool):
            available_tools.append(tool)
        else:
            disabled_tools.append(tool["name"])
    
    system_prompt = "You are a helpful assistant."
    if disabled_tools:
        system_prompt += f"\n\nNote: The following tools are currently unavailable: {', '.join(disabled_tools)}. Work with the available tools or inform the user if their request requires unavailable capabilities."
    
    return {
        "tools": available_tools,
        "system_prompt": system_prompt,
        "disabled": disabled_tools
    }
```

### Transparent Communication

Always let users know when something isn't working normally:

```python
def format_degraded_response(result: dict, degradation_info: dict) -> str:
    """Format a response that includes degradation information."""
    response_parts = [result.get("content", "")]
    
    if degradation_info.get("used_cache"):
        response_parts.append(
            "\n\n---\n*Note: Some information may be slightly outdated "
            "due to temporary service issues.*"
        )
    
    if degradation_info.get("disabled_tools"):
        tools = ", ".join(degradation_info["disabled_tools"])
        response_parts.append(
            f"\n\n---\n*Note: Some capabilities ({tools}) are currently "
            "unavailable. Results may be limited.*"
        )
    
    return "".join(response_parts)
```

## Retry Logic with Exponential Backoff

Many errors are transient—they'll succeed if you just try again. But naive retrying (immediate, unlimited retries) can make things worse by overwhelming an already struggling service. The solution is exponential backoff: wait longer between each retry.

Here's a robust retry implementation:

```python
"""
Retry logic with exponential backoff for transient failures.

Chapter 30: Error Handling and Recovery
"""

import os
import time
import random
from typing import TypeVar, Callable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should NOT be retried."""
    pass


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay before next retry using exponential backoff.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds before next retry
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** attempt)
    
    # Cap at maximum delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd problem
    if config.jitter:
        # Add random jitter of ±25%
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)  # Ensure non-negative


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.
    
    Args:
        error: The exception that occurred
        
    Returns:
        True if the error is transient and worth retrying
    """
    # Anthropic-specific retryable errors
    if isinstance(error, anthropic.RateLimitError):
        return True
    if isinstance(error, anthropic.APIConnectionError):
        return True
    if isinstance(error, anthropic.InternalServerError):
        return True
    
    # Explicit retry/non-retry markers
    if isinstance(error, RetryableError):
        return True
    if isinstance(error, NonRetryableError):
        return False
    
    # Non-retryable errors
    if isinstance(error, anthropic.AuthenticationError):
        return False
    if isinstance(error, anthropic.BadRequestError):
        return False
    
    # Default: don't retry unknown errors
    return False


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> T:
    """
    Execute a function with retry logic and exponential backoff.
    
    Args:
        func: The function to execute (should take no arguments)
        config: Retry configuration (uses defaults if None)
        on_retry: Optional callback called before each retry
                  with (error, attempt_number, delay)
    
    Returns:
        The result of the function if successful
        
    Raises:
        The last exception if all retries fail
    """
    config = config or RetryConfig()
    last_error: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            
            # Check if we should retry
            if not is_retryable_error(e):
                raise
            
            # Check if we have retries left
            if attempt >= config.max_retries:
                raise
            
            # Calculate delay and wait
            delay = calculate_delay(attempt, config)
            
            # Call retry callback if provided
            if on_retry:
                on_retry(e, attempt + 1, delay)
            
            time.sleep(delay)
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry loop exited unexpectedly")


def create_api_call_with_retry(client: anthropic.Anthropic):
    """
    Create a retry-wrapped API call function.
    
    Args:
        client: Anthropic client instance
        
    Returns:
        A function that makes API calls with automatic retry
    """
    def call_with_retry(
        messages: list[dict],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> anthropic.types.Message:
        """Make an API call with automatic retry on transient failures."""
        
        def make_call():
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools
            
            return client.messages.create(**kwargs)
        
        def log_retry(error: Exception, attempt: int, delay: float):
            print(f"  Retry {attempt}: {type(error).__name__} - "
                  f"waiting {delay:.1f}s")
        
        return retry_with_backoff(
            make_call,
            config=retry_config,
            on_retry=log_retry
        )
    
    return call_with_retry


# Demonstration
if __name__ == "__main__":
    client = anthropic.Anthropic()
    call_api = create_api_call_with_retry(client)
    
    # Test with a normal call (should succeed immediately)
    print("Making API call with retry wrapper...")
    response = call_api(
        messages=[{"role": "user", "content": "Say 'Hello!' and nothing else."}],
        max_tokens=50
    )
    print(f"Response: {response.content[0].text}")
    
    # Demonstrate backoff calculation
    print("\nBackoff delays for successive attempts:")
    config = RetryConfig(jitter=False)  # Disable jitter for clear demonstration
    for attempt in range(5):
        delay = calculate_delay(attempt, config)
        print(f"  Attempt {attempt + 1}: {delay:.1f}s delay")
```

The key concepts here are:

1. **Exponential growth**: Each retry waits longer than the last (1s, 2s, 4s, 8s, ...)
2. **Maximum cap**: We don't wait forever—there's an upper bound
3. **Jitter**: Random variation prevents all clients from retrying simultaneously
4. **Selective retry**: Only retry errors that are likely to be transient

## Fallback Behaviors

Sometimes retrying isn't enough. The service might be down for an extended period, or the error might be something that retries can't fix. In these cases, we need fallback behaviors.

```python
"""
Fallback strategies for maintaining agent functionality.

Chapter 30: Error Handling and Recovery
"""

import os
import json
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    data: Any
    timestamp: datetime
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_valid(self) -> bool:
        """Check if the cache entry is still valid."""
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=self.ttl_seconds)
    
    def is_stale_but_usable(self, max_stale_seconds: int = 86400) -> bool:
        """Check if entry is stale but could be used as fallback."""
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=max_stale_seconds)


class FallbackCache:
    """Simple in-memory cache for fallback responses."""
    
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry if it exists."""
        return self._cache.get(key)
    
    def set(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Store data in the cache."""
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    def get_valid(self, key: str) -> Optional[Any]:
        """Get data only if the cache entry is still valid."""
        entry = self.get(key)
        if entry and entry.is_valid():
            return entry.data
        return None
    
    def get_stale(self, key: str, max_stale_seconds: int = 86400) -> Optional[Any]:
        """Get data even if stale (for fallback scenarios)."""
        entry = self.get(key)
        if entry and entry.is_stale_but_usable(max_stale_seconds):
            return entry.data
        return None


@dataclass
class FallbackChain:
    """
    A chain of fallback strategies to try in order.
    
    Each strategy is a callable that returns a result or None.
    The chain tries each strategy until one succeeds.
    """
    strategies: list[Callable[[], Optional[Any]]] = field(default_factory=list)
    
    def add(self, strategy: Callable[[], Optional[Any]]) -> "FallbackChain":
        """Add a strategy to the chain."""
        self.strategies.append(strategy)
        return self
    
    def execute(self) -> tuple[Optional[Any], int]:
        """
        Execute strategies in order until one succeeds.
        
        Returns:
            Tuple of (result, strategy_index) or (None, -1) if all failed
        """
        for i, strategy in enumerate(self.strategies):
            try:
                result = strategy()
                if result is not None:
                    return result, i
            except Exception:
                continue  # Try next strategy
        
        return None, -1


class WeatherServiceWithFallback:
    """
    Example service demonstrating fallback patterns.
    
    This simulates a weather service that tries multiple strategies
    when the primary source fails.
    """
    
    def __init__(self):
        self.cache = FallbackCache()
        self.default_weather = {
            "temperature": "Unknown",
            "conditions": "Data unavailable",
            "source": "default"
        }
    
    def _fetch_primary_api(self, location: str) -> Optional[dict]:
        """Simulate primary API call (fails for demonstration)."""
        # In real code, this would call the actual API
        raise ConnectionError("Primary API unavailable")
    
    def _fetch_backup_api(self, location: str) -> Optional[dict]:
        """Simulate backup API call."""
        # Simulating a backup that works for some locations
        if location.lower() in ["new york", "london", "tokyo"]:
            return {
                "temperature": "72°F",
                "conditions": "Partly cloudy",
                "source": "backup_api"
            }
        return None
    
    def _get_cached(self, location: str) -> Optional[dict]:
        """Get cached weather data."""
        cached = self.cache.get_stale(f"weather:{location}")
        if cached:
            return {**cached, "source": "cache", "note": "May be outdated"}
        return None
    
    def _get_default(self, location: str) -> dict:
        """Return default response when all else fails."""
        return {
            **self.default_weather,
            "location": location,
            "note": "Weather data temporarily unavailable"
        }
    
    def get_weather(self, location: str) -> dict:
        """
        Get weather using fallback chain.
        
        Tries: primary API -> backup API -> cache -> default
        """
        chain = FallbackChain()
        chain.add(lambda: self._fetch_primary_api(location))
        chain.add(lambda: self._fetch_backup_api(location))
        chain.add(lambda: self._get_cached(location))
        chain.add(lambda: self._get_default(location))
        
        result, strategy_index = chain.execute()
        strategy_names = ["primary_api", "backup_api", "cache", "default"]
        
        print(f"  Weather retrieved via: {strategy_names[strategy_index]}")
        
        # Cache successful results from APIs
        if strategy_index < 2 and result:  # From primary or backup API
            self.cache.set(f"weather:{location}", result)
        
        return result


@dataclass
class ModelFallbackConfig:
    """Configuration for LLM model fallback."""
    primary_model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = field(default_factory=lambda: [
        "claude-sonnet-4-20250514",  # Same model, retry
        "claude-haiku-4-5-20250929",  # Faster, cheaper fallback
    ])
    reduce_tokens_on_fallback: bool = True
    token_reduction_factor: float = 0.75


def call_with_model_fallback(
    client: anthropic.Anthropic,
    messages: list[dict],
    config: Optional[ModelFallbackConfig] = None,
    **kwargs
) -> anthropic.types.Message:
    """
    Make an API call with model fallback on failure.
    
    Tries the primary model first, then falls back to alternatives.
    """
    config = config or ModelFallbackConfig()
    all_models = [config.primary_model] + config.fallback_models
    
    original_max_tokens = kwargs.get("max_tokens", 1024)
    last_error = None
    
    for i, model in enumerate(all_models):
        try:
            # Optionally reduce tokens for fallback models
            max_tokens = original_max_tokens
            if i > 0 and config.reduce_tokens_on_fallback:
                max_tokens = int(original_max_tokens * config.token_reduction_factor)
            
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            if i > 0:
                print(f"  Succeeded with fallback model: {model}")
            
            return response
            
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last_error = e
            print(f"  Model {model} failed: {type(e).__name__}")
            continue
        except anthropic.APIError as e:
            # Non-retryable errors should not fallback
            raise
    
    # All models failed
    raise last_error or RuntimeError("All fallback models failed")


# Demonstration
if __name__ == "__main__":
    print("=== Weather Service Fallback Demo ===\n")
    
    weather_service = WeatherServiceWithFallback()
    
    # Pre-populate cache for demonstration
    weather_service.cache.set("weather:san francisco", {
        "temperature": "65°F",
        "conditions": "Foggy"
    })
    
    # Test different scenarios
    locations = ["New York", "San Francisco", "Unknown City"]
    
    for location in locations:
        print(f"Getting weather for {location}:")
        result = weather_service.get_weather(location)
        print(f"  Result: {result}\n")
    
    print("=== Model Fallback Demo ===\n")
    
    client = anthropic.Anthropic()
    
    response = call_with_model_fallback(
        client,
        messages=[{"role": "user", "content": "Say 'Fallback test successful!'"}],
        max_tokens=50
    )
    print(f"Response: {response.content[0].text}")
```

## Error Reporting and Logging

When errors occur, you need visibility into what went wrong. Good logging is essential for debugging and monitoring production agents.

```python
"""
Structured logging for agent error tracking.

Chapter 30: Error Handling and Recovery
"""

import os
import json
import logging
import traceback
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ErrorSeverity(Enum):
    """Severity levels for agent errors."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of agent errors."""
    API = "api"
    NETWORK = "network"
    PARSING = "parsing"
    TOOL = "tool"
    LOGIC = "logic"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class AgentError:
    """Structured representation of an agent error."""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    tool_name: Optional[str] = None
    
    # Error details
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Recovery info
    recoverable: bool = True
    retry_count: int = 0
    recovery_action: Optional[str] = None
    
    # Additional context
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class AgentLogger:
    """
    Structured logger for agent operations.
    
    Provides consistent formatting and categorization for agent errors
    and events.
    """
    
    def __init__(
        self,
        name: str = "agent",
        level: int = logging.INFO,
        include_console: bool = True,
        log_file: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if include_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Track errors for summary
        self.error_history: list[AgentError] = []
    
    def _format_context(self, **context) -> str:
        """Format context as key=value pairs."""
        if not context:
            return ""
        pairs = [f"{k}={v}" for k, v in context.items() if v is not None]
        return " | " + " | ".join(pairs) if pairs else ""
    
    def debug(self, message: str, **context):
        """Log debug message."""
        self.logger.debug(f"{message}{self._format_context(**context)}")
    
    def info(self, message: str, **context):
        """Log info message."""
        self.logger.info(f"{message}{self._format_context(**context)}")
    
    def warning(self, message: str, **context):
        """Log warning message."""
        self.logger.warning(f"{message}{self._format_context(**context)}")
    
    def error(self, message: str, **context):
        """Log error message."""
        self.logger.error(f"{message}{self._format_context(**context)}")
    
    def critical(self, message: str, **context):
        """Log critical message."""
        self.logger.critical(f"{message}{self._format_context(**context)}")
    
    def log_agent_error(self, error: AgentError):
        """Log a structured agent error."""
        self.error_history.append(error)
        
        # Map severity to log level
        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        
        level = level_map.get(error.severity, logging.ERROR)
        
        # Format the log message
        context = {
            "category": error.category.value,
            "tool": error.tool_name,
            "recoverable": error.recoverable,
        }
        
        msg = f"{error.message}{self._format_context(**context)}"
        self.logger.log(level, msg)
        
        # Log stack trace for errors and above
        if error.stack_trace and level >= logging.ERROR:
            for line in error.stack_trace.split('\n'):
                self.logger.log(level, f"  {line}")
    
    def log_exception(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[dict] = None,
        recoverable: bool = True
    ) -> AgentError:
        """
        Create and log an AgentError from an exception.
        
        Returns the created AgentError for further processing.
        """
        error = AgentError(
            message=str(exception),
            category=category,
            severity=ErrorSeverity.ERROR,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            recoverable=recoverable,
            metadata=context or {}
        )
        
        self.log_agent_error(error)
        return error
    
    def get_error_summary(self) -> dict:
        """Get a summary of logged errors."""
        summary = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recoverable": 0,
            "non_recoverable": 0,
        }
        
        for error in self.error_history:
            # Count by category
            cat = error.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # Count by severity
            sev = error.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
            # Count recoverable
            if error.recoverable:
                summary["recoverable"] += 1
            else:
                summary["non_recoverable"] += 1
        
        return summary


def categorize_exception(exception: Exception) -> ErrorCategory:
    """
    Automatically categorize an exception.
    
    This helps ensure consistent error categorization across the codebase.
    """
    import anthropic
    
    exception_categories = {
        # API errors
        anthropic.APIConnectionError: ErrorCategory.NETWORK,
        anthropic.RateLimitError: ErrorCategory.API,
        anthropic.AuthenticationError: ErrorCategory.API,
        anthropic.BadRequestError: ErrorCategory.VALIDATION,
        anthropic.InternalServerError: ErrorCategory.API,
        
        # Built-in errors
        json.JSONDecodeError: ErrorCategory.PARSING,
        TimeoutError: ErrorCategory.TIMEOUT,
        ConnectionError: ErrorCategory.NETWORK,
        FileNotFoundError: ErrorCategory.TOOL,
        PermissionError: ErrorCategory.TOOL,
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
    }
    
    for exc_type, category in exception_categories.items():
        if isinstance(exception, exc_type):
            return category
    
    return ErrorCategory.UNKNOWN


# Demonstration
if __name__ == "__main__":
    logger = AgentLogger("demo_agent", level=logging.DEBUG)
    
    print("=== Agent Logging Demo ===\n")
    
    # Log various events
    logger.info("Agent starting", agent_id="agent-001", model="claude-sonnet-4-20250514")
    logger.debug("Loading tools", tools=["calculator", "weather"])
    
    # Log a structured error
    error = AgentError(
        message="Weather API returned invalid JSON",
        category=ErrorCategory.PARSING,
        severity=ErrorSeverity.WARNING,
        tool_name="weather",
        recoverable=True,
        recovery_action="Using cached data"
    )
    logger.log_agent_error(error)
    
    # Log an exception
    try:
        json.loads("invalid json{")
    except json.JSONDecodeError as e:
        logger.log_exception(
            e,
            category=ErrorCategory.PARSING,
            context={"source": "weather_api_response"}
        )
    
    # Log critical error
    critical_error = AgentError(
        message="Agent stuck in infinite loop",
        category=ErrorCategory.LOGIC,
        severity=ErrorSeverity.CRITICAL,
        recoverable=False,
        metadata={"loop_count": 50, "last_tool": "search"}
    )
    logger.log_agent_error(critical_error)
    
    # Show summary
    print("\n=== Error Summary ===")
    summary = logger.get_error_summary()
    print(json.dumps(summary, indent=2))
```

## Self-Correction Patterns

One of the most powerful error handling patterns for agents is self-correction: when the agent detects it made a mistake, it can try to fix it. This is especially useful for LLM output errors like invalid JSON or incorrect tool usage.

```python
"""
Self-correction patterns for agent error recovery.

Chapter 30: Error Handling and Recovery
"""

import os
import json
import re
from typing import Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


@dataclass
class CorrectionAttempt:
    """Record of a self-correction attempt."""
    original_output: str
    error_message: str
    corrected_output: Optional[str]
    success: bool
    attempts: int


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Attempt to extract JSON from a response that may have extra text.
    
    Handles cases where the LLM wraps JSON in markdown code blocks
    or adds explanatory text.
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # Try to find raw JSON (object or array)
    json_patterns = [
        r'(\{[\s\S]*\})',  # Object
        r'(\[[\s\S]*\])',  # Array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    return None


def self_correct_json(
    original_response: str,
    error: json.JSONDecodeError,
    max_attempts: int = 2
) -> CorrectionAttempt:
    """
    Ask the LLM to fix its own invalid JSON output.
    
    Args:
        original_response: The original response containing invalid JSON
        error: The JSON decode error that occurred
        max_attempts: Maximum correction attempts
        
    Returns:
        CorrectionAttempt with results
    """
    # First, try to extract JSON without API call
    extracted = extract_json_from_response(original_response)
    if extracted:
        try:
            json.loads(extracted)
            return CorrectionAttempt(
                original_output=original_response,
                error_message=str(error),
                corrected_output=extracted,
                success=True,
                attempts=0  # No API call needed
            )
        except json.JSONDecodeError:
            pass
    
    # Need to ask the LLM to fix it
    for attempt in range(max_attempts):
        correction_prompt = f"""Your previous response contained invalid JSON that could not be parsed.

Original response:
{original_response}

Error: {error}

Please provide ONLY the corrected, valid JSON with no additional text, explanation, or markdown formatting. Just the raw JSON."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": correction_prompt}]
            )
            
            corrected = response.content[0].text.strip()
            
            # Try to parse the correction
            json.loads(corrected)
            
            return CorrectionAttempt(
                original_output=original_response,
                error_message=str(error),
                corrected_output=corrected,
                success=True,
                attempts=attempt + 1
            )
            
        except json.JSONDecodeError:
            continue  # Try again
        except anthropic.APIError:
            break  # API error, stop trying
    
    return CorrectionAttempt(
        original_output=original_response,
        error_message=str(error),
        corrected_output=None,
        success=False,
        attempts=max_attempts
    )


def self_correct_tool_call(
    tool_name: str,
    tool_input: dict,
    error_message: str,
    available_tools: list[dict],
    conversation_context: list[dict],
    max_attempts: int = 2
) -> Optional[dict]:
    """
    Ask the LLM to fix an invalid tool call.
    
    Args:
        tool_name: The tool that was called
        tool_input: The input that caused the error
        error_message: Description of what went wrong
        available_tools: List of available tool definitions
        conversation_context: Recent conversation for context
        max_attempts: Maximum correction attempts
        
    Returns:
        Corrected tool input dict, or None if correction failed
    """
    # Find the tool definition
    tool_def = next(
        (t for t in available_tools if t["name"] == tool_name),
        None
    )
    
    if not tool_def:
        return None  # Can't correct if we don't know the tool
    
    for attempt in range(max_attempts):
        correction_prompt = f"""Your previous tool call resulted in an error.

Tool: {tool_name}
Input provided: {json.dumps(tool_input, indent=2)}
Error: {error_message}

Tool definition:
{json.dumps(tool_def, indent=2)}

Please provide the corrected tool input as a JSON object. Only output the JSON, no explanation."""

        # Include some conversation context
        messages = conversation_context[-3:] + [
            {"role": "user", "content": correction_prompt}
        ]
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages
            )
            
            corrected_text = response.content[0].text.strip()
            
            # Extract and parse JSON
            extracted = extract_json_from_response(corrected_text)
            if extracted:
                corrected_input = json.loads(extracted)
                return corrected_input
            
            # Try parsing directly
            corrected_input = json.loads(corrected_text)
            return corrected_input
            
        except (json.JSONDecodeError, anthropic.APIError):
            continue
    
    return None


class SelfCorrectingAgent:
    """
    An agent wrapper that attempts self-correction on errors.
    
    This wraps the basic agent loop and intercepts errors,
    attempting to fix them before giving up.
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        tools: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        max_correction_attempts: int = 2
    ):
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_correction_attempts = max_correction_attempts
        self.correction_history: list[CorrectionAttempt] = []
    
    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool and return the result."""
        # This would dispatch to actual tool implementations
        # For demonstration, we'll simulate some responses
        if name == "calculator":
            try:
                expr = input_data.get("expression", "")
                # Very basic and safe evaluation
                result = eval(expr, {"__builtins__": {}}, {})
                return json.dumps({"result": result})
            except Exception as e:
                raise ValueError(f"Calculation error: {e}")
        
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message with self-correction capabilities.
        
        Returns the final response after any needed corrections.
        """
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            # Make API call
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages
            )
            
            # Check for tool use
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        try:
                            result = self._execute_tool(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                        except Exception as e:
                            # Attempt self-correction
                            corrected_input = self_correct_tool_call(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                error_message=str(e),
                                available_tools=self.tools,
                                conversation_context=messages,
                                max_attempts=self.max_correction_attempts
                            )
                            
                            if corrected_input:
                                # Try with corrected input
                                try:
                                    result = self._execute_tool(
                                        tool_name, corrected_input
                                    )
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": result
                                    })
                                    print(f"  Self-correction succeeded for {tool_name}")
                                    continue
                                except Exception:
                                    pass
                            
                            # Correction failed, report error
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({
                                    "error": str(e),
                                    "note": "Tool execution failed"
                                }),
                                "is_error": True
                            })
                
                # Add assistant response and tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
            else:
                # End turn - return text response
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                
                return ""


# Demonstration
if __name__ == "__main__":
    print("=== JSON Self-Correction Demo ===\n")
    
    # Test JSON correction
    invalid_json = """Here's the data you requested:
    
```json
{
    "name": "Test",
    "values": [1, 2, 3,]  // trailing comma - invalid!
}
```

Let me know if you need anything else!"""
    
    try:
        json.loads(invalid_json)
    except json.JSONDecodeError as e:
        print(f"Original error: {e}\n")
        result = self_correct_json(invalid_json, e)
        print(f"Correction successful: {result.success}")
        print(f"Attempts needed: {result.attempts}")
        if result.corrected_output:
            print(f"Corrected JSON:\n{result.corrected_output}")
    
    print("\n=== Self-Correcting Agent Demo ===\n")
    
    # Create agent with calculator tool
    tools = [{
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }]
    
    agent = SelfCorrectingAgent(
        client=client,
        tools=tools,
        system_prompt="You are a helpful assistant with access to a calculator."
    )
    
    response = agent.process_message("What is 15 * 7?")
    print(f"Response: {response}")
```

## The Complete Error Handler

Now let's bring everything together into a comprehensive error handling module that you can use in your agents:

```python
"""
Complete error handling utilities for production agents.

Chapter 30: Error Handling and Recovery
"""

import os
import time
import json
import random
import logging
import traceback
from typing import TypeVar, Callable, Optional, Any, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from dotenv import load_dotenv
import anthropic

load_dotenv()

T = TypeVar('T')


# ============================================================
# Error Types and Classification
# ============================================================

class ErrorSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    API = "api"
    NETWORK = "network"
    PARSING = "parsing"
    TOOL = "tool"
    LOGIC = "logic"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class AgentError(Exception):
    """
    Rich exception class for agent errors.
    
    Carries metadata about the error for logging and recovery decisions.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        original_exception: Optional[Exception] = None,
        context: Optional[dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.context = context or {}
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }


def classify_exception(exc: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
    """
    Classify an exception into category, severity, and recoverability.
    
    Returns:
        Tuple of (category, severity, is_recoverable)
    """
    classifications = {
        # Anthropic API errors
        anthropic.RateLimitError: (ErrorCategory.RATE_LIMIT, ErrorSeverity.WARNING, True),
        anthropic.APIConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING, True),
        anthropic.InternalServerError: (ErrorCategory.API, ErrorSeverity.WARNING, True),
        anthropic.AuthenticationError: (ErrorCategory.API, ErrorSeverity.CRITICAL, False),
        anthropic.BadRequestError: (ErrorCategory.VALIDATION, ErrorSeverity.ERROR, False),
        
        # Standard errors
        json.JSONDecodeError: (ErrorCategory.PARSING, ErrorSeverity.WARNING, True),
        TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.WARNING, True),
        ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.WARNING, True),
        FileNotFoundError: (ErrorCategory.TOOL, ErrorSeverity.ERROR, False),
        PermissionError: (ErrorCategory.TOOL, ErrorSeverity.ERROR, False),
        ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.WARNING, True),
        TypeError: (ErrorCategory.VALIDATION, ErrorSeverity.ERROR, False),
    }
    
    for exc_type, (cat, sev, rec) in classifications.items():
        if isinstance(exc, exc_type):
            return cat, sev, rec
    
    return ErrorCategory.UNKNOWN, ErrorSeverity.ERROR, False


# ============================================================
# Retry Configuration and Logic
# ============================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
        ConnectionError,
        TimeoutError,
    )


def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        delay *= (0.5 + random.random())  # 50-150% of calculated delay
    
    return delay


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator that adds retry logic to a function.
    
    Usage:
        @with_retry(RetryConfig(max_retries=3))
        def my_api_call():
            ...
    """
    config = config or RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt >= config.max_retries:
                        raise AgentError(
                            f"Max retries ({config.max_retries}) exceeded",
                            category=ErrorCategory.API,
                            recoverable=False,
                            original_exception=e
                        )
                    
                    delay = calculate_backoff(attempt, config)
                    logging.warning(
                        f"Retry {attempt + 1}/{config.max_retries}: "
                        f"{type(e).__name__} - waiting {delay:.1f}s"
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================
# Fallback System
# ============================================================

@dataclass
class FallbackResult(Generic[T]):
    """Result from a fallback chain execution."""
    value: T
    source: str
    fallback_used: bool
    attempts: list[str] = field(default_factory=list)


class FallbackChain(Generic[T]):
    """
    Chain of fallback strategies for resilient operations.
    
    Tries each strategy in order until one succeeds.
    """
    
    def __init__(self):
        self._strategies: list[tuple[str, Callable[[], T]]] = []
    
    def add(self, name: str, strategy: Callable[[], T]) -> "FallbackChain[T]":
        """Add a named strategy to the chain."""
        self._strategies.append((name, strategy))
        return self
    
    def execute(self, default: Optional[T] = None) -> FallbackResult[T]:
        """
        Execute strategies until one succeeds.
        
        Args:
            default: Value to return if all strategies fail (None raises exception)
            
        Returns:
            FallbackResult with the successful value and metadata
        """
        attempts = []
        
        for name, strategy in self._strategies:
            try:
                result = strategy()
                if result is not None:
                    return FallbackResult(
                        value=result,
                        source=name,
                        fallback_used=(name != self._strategies[0][0]),
                        attempts=attempts
                    )
                attempts.append(f"{name}: returned None")
            except Exception as e:
                attempts.append(f"{name}: {type(e).__name__}: {e}")
        
        if default is not None:
            return FallbackResult(
                value=default,
                source="default",
                fallback_used=True,
                attempts=attempts
            )
        
        raise AgentError(
            "All fallback strategies failed",
            category=ErrorCategory.UNKNOWN,
            recoverable=False,
            context={"attempts": attempts}
        )


# ============================================================
# Error Handler Class
# ============================================================

class ErrorHandler:
    """
    Comprehensive error handler for agents.
    
    Combines logging, retry logic, fallbacks, and error tracking.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        retry_config: Optional[RetryConfig] = None,
        max_errors_before_abort: int = 10
    ):
        self.logger = logger or logging.getLogger("agent.errors")
        self.retry_config = retry_config or RetryConfig()
        self.max_errors_before_abort = max_errors_before_abort
        
        self.error_count = 0
        self.error_history: list[AgentError] = []
    
    def handle(
        self,
        exception: Exception,
        context: Optional[dict] = None,
        reraise: bool = True
    ) -> Optional[AgentError]:
        """
        Handle an exception with logging and tracking.
        
        Args:
            exception: The exception to handle
            context: Additional context about where/why error occurred
            reraise: Whether to re-raise the exception after handling
            
        Returns:
            AgentError if not reraised, None otherwise
        """
        # Classify the error
        category, severity, recoverable = classify_exception(exception)
        
        # Create structured error
        agent_error = AgentError(
            message=str(exception),
            category=category,
            severity=severity,
            recoverable=recoverable,
            original_exception=exception,
            context=context or {}
        )
        
        # Track error
        self.error_count += 1
        self.error_history.append(agent_error)
        
        # Log the error
        log_msg = f"{category.value.upper()}: {exception}"
        if context:
            log_msg += f" | Context: {context}"
        
        log_level = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.ERROR)
        
        self.logger.log(log_level, log_msg)
        
        # Check for abort condition
        if self.error_count >= self.max_errors_before_abort:
            abort_error = AgentError(
                f"Error limit ({self.max_errors_before_abort}) reached - aborting",
                category=ErrorCategory.LOGIC,
                severity=ErrorSeverity.CRITICAL,
                recoverable=False
            )
            self.logger.critical(abort_error.message)
            raise abort_error
        
        if reraise:
            raise agent_error
        
        return agent_error
    
    def wrap(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to wrap a function with error handling.
        
        Usage:
            @error_handler.wrap
            def my_function():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except AgentError:
                raise  # Already handled
            except Exception as e:
                self.handle(e, context={"function": func.__name__})
        
        return wrapper
    
    def safe_execute(
        self,
        func: Callable[[], T],
        fallback: Optional[T] = None,
        context: Optional[dict] = None
    ) -> tuple[Optional[T], Optional[AgentError]]:
        """
        Execute a function safely, returning fallback on error.
        
        Returns:
            Tuple of (result, error) - one will be None
        """
        try:
            return func(), None
        except Exception as e:
            error = self.handle(e, context=context, reraise=False)
            return fallback, error
    
    def get_summary(self) -> dict:
        """Get a summary of all errors handled."""
        summary = {
            "total_errors": self.error_count,
            "by_category": {},
            "by_severity": {},
            "recoverable_count": 0,
            "recent_errors": []
        }
        
        for error in self.error_history:
            cat = error.category.value
            sev = error.severity.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            if error.recoverable:
                summary["recoverable_count"] += 1
        
        # Include last 5 errors
        for error in self.error_history[-5:]:
            summary["recent_errors"].append({
                "message": error.message,
                "category": error.category.value,
                "timestamp": error.timestamp.isoformat()
            })
        
        return summary
    
    def reset(self):
        """Reset error tracking."""
        self.error_count = 0
        self.error_history.clear()


# ============================================================
# Convenience Functions
# ============================================================

def safe_json_parse(text: str, default: Any = None) -> tuple[Any, Optional[str]]:
    """
    Safely parse JSON with extraction fallback.
    
    Returns:
        Tuple of (parsed_data, error_message)
    """
    # Try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    
    # Try to extract from code blocks
    import re
    patterns = [
        r'```(?:json)?\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})',
        r'(\[[\s\S]*\])',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match), None
            except json.JSONDecodeError:
                continue
    
    return default, f"Could not parse JSON from: {text[:100]}..."


def with_timeout(seconds: float):
    """
    Decorator that adds a timeout to a function.
    
    Note: This is a simple implementation. For production,
    consider using concurrent.futures or asyncio.
    """
    import signal
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def handler(signum, frame):
            raise TimeoutError(f"Function timed out after {seconds}s")
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            
            try:
                return func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


# ============================================================
# Demo and Testing
# ============================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("=== Error Handler Demo ===\n")
    
    # Create error handler
    handler = ErrorHandler(max_errors_before_abort=5)
    
    # Demo retry decorator
    print("1. Retry Decorator Demo:")
    
    call_count = 0
    
    @with_retry(RetryConfig(max_retries=2, base_delay=0.5))
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated network error")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"   Result: {result} (took {call_count} attempts)\n")
    except AgentError as e:
        print(f"   Failed: {e}\n")
    
    # Demo fallback chain
    print("2. Fallback Chain Demo:")
    
    chain: FallbackChain[str] = FallbackChain()
    chain.add("primary", lambda: None)  # Fails (returns None)
    chain.add("secondary", lambda: (_ for _ in ()).throw(ValueError("Nope")))  # Fails
    chain.add("tertiary", lambda: "Fallback succeeded!")
    
    result = chain.execute()
    print(f"   Value: {result.value}")
    print(f"   Source: {result.source}")
    print(f"   Attempts: {result.attempts}\n")
    
    # Demo error handling
    print("3. Error Handler Demo:")
    
    @handler.wrap
    def risky_operation():
        raise ValueError("Something went wrong!")
    
    try:
        risky_operation()
    except AgentError as e:
        print(f"   Caught: {e.message}")
        print(f"   Category: {e.category.value}")
        print(f"   Recoverable: {e.recoverable}\n")
    
    # Demo safe execute
    print("4. Safe Execute Demo:")
    
    result, error = handler.safe_execute(
        lambda: json.loads("invalid"),
        fallback={"default": True}
    )
    print(f"   Result: {result}")
    print(f"   Error: {error.message if error else None}\n")
    
    # Show summary
    print("5. Error Summary:")
    summary = handler.get_summary()
    print(f"   {json.dumps(summary, indent=4)}")
```

## Common Pitfalls

### 1. Catching Too Broadly

Don't catch `Exception` everywhere—you'll hide bugs:

```python
# Bad: Hides all errors including programming bugs
try:
    result = process_data(data)
except Exception:
    result = default_value

# Good: Catch specific, expected errors
try:
    result = process_data(data)
except (ValueError, json.JSONDecodeError) as e:
    logger.warning(f"Data processing failed: {e}")
    result = default_value
except Exception as e:
    # Log and re-raise unexpected errors
    logger.error(f"Unexpected error: {e}")
    raise
```

### 2. Infinite Retry Loops

Always have a maximum retry count and consider the total time:

```python
# Bad: Could retry forever
while True:
    try:
        return make_api_call()
    except RateLimitError:
        time.sleep(1)

# Good: Bounded retries with increasing delays
for attempt in range(max_retries):
    try:
        return make_api_call()
    except RateLimitError:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Silent Failures

Don't swallow errors without logging—you'll never know what's wrong:

```python
# Bad: Error disappears silently
def get_data():
    try:
        return fetch_from_api()
    except Exception:
        return None  # What happened? No one knows.

# Good: Log before falling back
def get_data():
    try:
        return fetch_from_api()
    except APIError as e:
        logger.warning(f"API fetch failed: {e}, using cache")
        return get_from_cache()
```

## Practical Exercise

**Task:** Build an error-resilient tool executor

Create a `ResilientToolExecutor` class that:

1. Executes tools with automatic retry on transient failures
2. Falls back to cached results when available
3. Logs all errors with full context
4. Tracks error statistics for monitoring
5. Supports self-correction for common errors

**Requirements:**
- Retry configuration should be customizable per tool
- The cache should have configurable TTL
- Errors should be categorized and logged with severity levels
- The executor should abort after too many consecutive failures
- Include a method to get error statistics

**Hints:**
- Use the `RetryConfig` and `ErrorHandler` classes from this chapter
- Consider what metadata you need for effective debugging
- Think about how to make cache keys unique per tool call
- The self-correction might involve re-prompting the LLM for invalid inputs

**Solution:** See `code/exercise.py`

## Key Takeaways

- **Errors are inevitable**: Plan for failure from the start, not as an afterthought
- **Categorize errors**: Different error types need different handling strategies
- **Retry intelligently**: Use exponential backoff with jitter for transient failures
- **Degrade gracefully**: When parts fail, maintain as much functionality as possible
- **Log comprehensively**: You can't fix what you can't see—structured logging is essential
- **Enable self-correction**: Let your agent fix its own mistakes when possible
- **Set boundaries**: Maximum retries, error counts, and timeouts prevent runaway failures

## What's Next

Now that our agent can handle errors gracefully, we need to address another critical aspect of production systems: keeping humans in control. In Chapter 31, we'll implement human-in-the-loop patterns that allow for oversight, approval gates, and escalation paths—ensuring that autonomous agents remain supervised and safe.
