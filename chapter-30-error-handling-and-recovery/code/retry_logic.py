"""
Retry logic with exponential backoff for transient failures.

Chapter 30: Error Handling and Recovery

This module provides robust retry functionality for handling transient
errors in API calls and other operations.
"""

import os
import time
import random
from typing import TypeVar, Callable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

T = TypeVar('T')


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        exponential_base: Base for exponential growth (typically 2)
        jitter: Whether to add randomness to prevent thundering herd
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryableError(Exception):
    """
    Base class for errors that should trigger a retry.
    
    Use this to mark custom exceptions as retryable.
    """
    pass


class NonRetryableError(Exception):
    """
    Base class for errors that should NOT be retried.
    
    Use this to explicitly prevent retries.
    """
    pass


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay before next retry using exponential backoff.
    
    The formula is: base_delay * (exponential_base ^ attempt)
    With optional jitter to prevent thundering herd.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds before next retry
        
    Examples:
        >>> config = RetryConfig(base_delay=1.0, jitter=False)
        >>> calculate_delay(0, config)  # First retry
        1.0
        >>> calculate_delay(1, config)  # Second retry
        2.0
        >>> calculate_delay(2, config)  # Third retry
        4.0
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** attempt)
    
    # Cap at maximum delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd problem
    # This is important when many clients are retrying simultaneously
    if config.jitter:
        # Add random jitter of ±25%
        jitter_range = delay * 0.25
        delay = delay + random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)  # Ensure non-negative


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.
    
    This function classifies exceptions into retryable and non-retryable
    categories based on their type.
    
    Args:
        error: The exception that occurred
        
    Returns:
        True if the error is transient and worth retrying
    """
    # Anthropic-specific retryable errors
    if isinstance(error, anthropic.RateLimitError):
        return True  # Rate limits are definitely retryable
    if isinstance(error, anthropic.APIConnectionError):
        return True  # Network issues are usually transient
    if isinstance(error, anthropic.InternalServerError):
        return True  # Server errors (5xx) often recover
    
    # Explicit retry/non-retry markers
    if isinstance(error, RetryableError):
        return True
    if isinstance(error, NonRetryableError):
        return False
    
    # Non-retryable errors - these won't succeed on retry
    if isinstance(error, anthropic.AuthenticationError):
        return False  # Bad credentials won't fix themselves
    if isinstance(error, anthropic.BadRequestError):
        return False  # Invalid request won't become valid
    
    # Standard library retryable errors
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    
    # Default: don't retry unknown errors
    return False


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> T:
    """
    Execute a function with retry logic and exponential backoff.
    
    This function will attempt to execute the provided function,
    retrying on transient failures with increasing delays.
    
    Args:
        func: The function to execute (should take no arguments)
        config: Retry configuration (uses defaults if None)
        on_retry: Optional callback called before each retry
                  with (error, attempt_number, delay)
    
    Returns:
        The result of the function if successful
        
    Raises:
        The last exception if all retries fail
        
    Example:
        >>> def flaky_api_call():
        ...     # Might fail sometimes
        ...     return fetch_data()
        >>> result = retry_with_backoff(flaky_api_call)
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
                raise  # Non-retryable, give up immediately
            
            # Check if we have retries left
            if attempt >= config.max_retries:
                raise  # No more retries
            
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
    
    This factory creates a function that automatically retries
    API calls on transient failures.
    
    Args:
        client: Anthropic client instance
        
    Returns:
        A function that makes API calls with automatic retry
        
    Example:
        >>> client = anthropic.Anthropic()
        >>> call_api = create_api_call_with_retry(client)
        >>> response = call_api(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
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


# Decorator version for convenience
def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator that adds retry logic to any function.
    
    Usage:
        @with_retry(RetryConfig(max_retries=5))
        def my_api_call():
            return client.messages.create(...)
    """
    from functools import wraps
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config
            )
        return wrapper
    return decorator


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("RETRY LOGIC WITH EXPONENTIAL BACKOFF")
    print("=" * 60)
    
    # 1. Show backoff calculation
    print("\n### 1. Backoff Delays ###\n")
    config = RetryConfig(jitter=False)  # Disable jitter for clear demonstration
    
    print("With exponential_base=2.0, base_delay=1.0:")
    for attempt in range(6):
        delay = calculate_delay(attempt, config)
        print(f"  Attempt {attempt + 1}: {delay:.1f}s delay")
    
    # Show with jitter
    print("\nWith jitter enabled (adds randomness):")
    config_with_jitter = RetryConfig(jitter=True)
    for attempt in range(3):
        delays = [calculate_delay(attempt, config_with_jitter) for _ in range(3)]
        delays_str = ", ".join(f"{d:.2f}s" for d in delays)
        print(f"  Attempt {attempt + 1}: {delays_str} (varies each time)")
    
    # 2. Demonstrate retry with simulated failures
    print("\n### 2. Retry in Action ###\n")
    
    call_count = 0
    
    def flaky_function():
        """Simulate a function that fails twice, then succeeds."""
        global call_count
        call_count += 1
        
        if call_count < 3:
            raise ConnectionError(f"Simulated network error (attempt {call_count})")
        return f"Success on attempt {call_count}!"
    
    def log_retry(error, attempt, delay):
        print(f"  Attempt {attempt} failed: {error}")
        print(f"  Waiting {delay:.2f}s before retry...")
    
    print("Calling flaky function (will fail twice, then succeed):")
    result = retry_with_backoff(
        flaky_function,
        config=RetryConfig(base_delay=0.5, jitter=False),
        on_retry=log_retry
    )
    print(f"  Result: {result}")
    
    # 3. Demonstrate real API call with retry
    print("\n### 3. API Call with Retry ###\n")
    
    client = anthropic.Anthropic()
    call_api = create_api_call_with_retry(client)
    
    print("Making API call with retry wrapper:")
    response = call_api(
        messages=[{"role": "user", "content": "Say 'Retry test successful!' and nothing else."}],
        max_tokens=50
    )
    print(f"  Response: {response.content[0].text}")
    
    # 4. Show decorator usage
    print("\n### 4. Decorator Pattern ###\n")
    
    @with_retry(RetryConfig(max_retries=2, base_delay=0.5))
    def decorated_function():
        """This function has automatic retry built in."""
        return "Decorator works!"
    
    print("Using @with_retry decorator:")
    result = decorated_function()
    print(f"  Result: {result}")
    
    # 5. Demonstrate non-retryable error
    print("\n### 5. Non-Retryable Error ###\n")
    
    def authentication_failure():
        """Simulate an auth error - should not retry."""
        raise anthropic.AuthenticationError(
            "Invalid API key",
            response=None,
            body=None
        )
    
    print("Calling function with non-retryable error:")
    try:
        retry_with_backoff(authentication_failure)
    except anthropic.AuthenticationError as e:
        print(f"  Correctly did NOT retry: {type(e).__name__}")
        print(f"  (Auth errors won't succeed on retry)")
    
    print("\n" + "=" * 60)
    print("KEY POINTS:")
    print("- Exponential backoff: 1s, 2s, 4s, 8s, ...")
    print("- Jitter prevents thundering herd")
    print("- Only retry transient errors")
    print("- Always have a max retry limit")
    print("=" * 60)
