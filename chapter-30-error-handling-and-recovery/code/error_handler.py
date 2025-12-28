"""
Complete error handling utilities for production agents.

Chapter 30: Error Handling and Recovery

This module brings together all error handling patterns into a
comprehensive, production-ready error handling system.
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

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

T = TypeVar('T')


# ============================================================
# Error Types and Classification
# ============================================================

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
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class AgentError(Exception):
    """
    Rich exception class for agent errors.
    
    Carries metadata about the error for logging and recovery decisions.
    Can be raised and caught like any exception, but also provides
    structured information for analysis.
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
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "original_exception": str(self.original_exception) if self.original_exception else None,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def classify_exception(exc: Exception) -> tuple[ErrorCategory, ErrorSeverity, bool]:
    """
    Classify an exception into category, severity, and recoverability.
    
    Args:
        exc: The exception to classify
        
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
                    logging.debug(
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
            default: Value to return if all strategies fail
            
        Returns:
            FallbackResult with the successful value and metadata
        """
        attempts = []
        
        for i, (name, strategy) in enumerate(self._strategies):
            try:
                result = strategy()
                if result is not None:
                    return FallbackResult(
                        value=result,
                        source=name,
                        fallback_used=(i > 0),
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
    
    Combines logging, retry logic, fallbacks, and error tracking
    into a single, easy-to-use interface.
    
    Usage:
        handler = ErrorHandler()
        
        @handler.wrap
        def risky_function():
            ...
        
        # Or manually
        try:
            do_something()
        except Exception as e:
            handler.handle(e, context={"operation": "do_something"})
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        retry_config: Optional[RetryConfig] = None,
        max_errors_before_abort: int = 10
    ):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger instance (creates one if None)
            retry_config: Default retry configuration
            max_errors_before_abort: Error count that triggers abort
        """
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
        log_msg = f"[{category.value.upper()}] {exception}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            log_msg += f" | {context_str}"
        
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
        
        This is useful when you want to attempt something but have
        a reasonable fallback if it fails.
        
        Args:
            func: Function to execute
            fallback: Value to return on error
            context: Error context
            
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
                "message": error.message[:50] + "..." if len(error.message) > 50 else error.message,
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
    
    Tries to extract JSON from text that may contain extra content.
    
    Args:
        text: Text that may contain JSON
        default: Value to return if parsing fails
        
    Returns:
        Tuple of (parsed_data, error_message)
    """
    import re
    
    # Try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    
    # Try to extract from code blocks or raw JSON
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
    
    Note: Uses signal, only works on Unix-like systems.
    For cross-platform, use concurrent.futures.
    """
    import signal
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def handler(signum, frame):
            raise TimeoutError(f"Function timed out after {seconds}s")
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
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
# Demonstration
# ============================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 60)
    print("COMPLETE ERROR HANDLER DEMONSTRATION")
    print("=" * 60)
    
    # Create error handler
    handler = ErrorHandler(max_errors_before_abort=10)
    
    # 1. Retry decorator
    print("\n### 1. Retry Decorator ###\n")
    
    call_count = 0
    
    @with_retry(RetryConfig(max_retries=2, base_delay=0.3, jitter=False))
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Simulated error (attempt {call_count})")
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"Result: {result} (took {call_count} attempts)")
    except AgentError as e:
        print(f"Failed: {e}")
    
    # 2. Fallback chain
    print("\n### 2. Fallback Chain ###\n")
    
    chain: FallbackChain[str] = FallbackChain()
    chain.add("primary", lambda: None)  # Returns None
    chain.add("secondary", lambda: (_ for _ in ()).throw(ValueError("Oops!")))
    chain.add("tertiary", lambda: "Fallback succeeded!")
    
    result = chain.execute()
    print(f"Value: {result.value}")
    print(f"Source: {result.source}")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Attempts: {result.attempts}")
    
    # 3. Error handler wrap decorator
    print("\n### 3. Error Handler Wrapper ###\n")
    
    @handler.wrap
    def risky_operation():
        raise ValueError("Something went wrong!")
    
    try:
        risky_operation()
    except AgentError as e:
        print(f"Caught AgentError:")
        print(f"  Message: {e.message}")
        print(f"  Category: {e.category.value}")
        print(f"  Recoverable: {e.recoverable}")
    
    # 4. Safe execute
    print("\n### 4. Safe Execute ###\n")
    
    result, error = handler.safe_execute(
        lambda: json.loads("invalid json{"),
        fallback={"default": True},
        context={"operation": "parse_config"}
    )
    print(f"Result: {result}")
    print(f"Error: {error.message if error else None}")
    
    # 5. Safe JSON parse
    print("\n### 5. Safe JSON Parse ###\n")
    
    test_texts = [
        '{"valid": "json"}',
        'Here is the data: {"wrapped": true}',
        '```json\n{"in_code_block": true}\n```',
        'not json at all',
    ]
    
    for text in test_texts:
        data, error = safe_json_parse(text, default={"fallback": True})
        status = "parsed" if error is None else "fallback"
        print(f"  {text[:30]:30} -> {status}: {data}")
    
    # 6. Multiple errors and summary
    print("\n### 6. Error Summary ###\n")
    
    # Generate some more errors
    test_errors = [
        (ConnectionError("Network down"), {"tool": "api_call"}),
        (json.JSONDecodeError("Expecting value", "", 0), {"tool": "parser"}),
        (ValueError("Invalid input"), {"tool": "validator"}),
    ]
    
    for exc, ctx in test_errors:
        handler.handle(exc, context=ctx, reraise=False)
    
    summary = handler.get_summary()
    print(json.dumps(summary, indent=2))
    
    # 7. Real API call with error handling
    print("\n### 7. Real API Call ###\n")
    
    client = anthropic.Anthropic()
    
    @with_retry(RetryConfig(max_retries=2, base_delay=0.5))
    @handler.wrap
    def make_api_call(prompt: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    result = make_api_call("Say 'Error handling works!' and nothing else.")
    print(f"API Response: {result}")
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING TOOLKIT SUMMARY:")
    print("-" * 60)
    print("Components:")
    print("  - AgentError: Rich exception with metadata")
    print("  - ErrorHandler: Centralized error management")
    print("  - @with_retry: Automatic retry with backoff")
    print("  - FallbackChain: Multiple fallback strategies")
    print("  - safe_execute: Try with fallback")
    print("  - safe_json_parse: Robust JSON parsing")
    print("=" * 60)
