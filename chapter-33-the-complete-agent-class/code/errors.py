"""
Error handling for agents.

Chapter 33: The Complete Agent Class

This module provides comprehensive error handling including
categorization, retry logic, and recovery strategies.
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

T = TypeVar('T')


class ErrorSeverity(Enum):
    """
    Severity levels for errors.
    
    LOW: Can continue with degraded functionality
    MEDIUM: Should retry or use fallback
    HIGH: Should stop current task and report
    CRITICAL: Immediate stop, possible data integrity concerns
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentError:
    """
    Represents an error that occurred during agent execution.
    
    Attributes:
        error_type: Category of the error
        message: Human-readable error message
        severity: How serious the error is
        recoverable: Whether the agent can recover
        context: Additional context about the error
        original_exception: The original Python exception
        timestamp: When the error occurred
    """
    error_type: str
    message: str
    severity: ErrorSeverity
    recoverable: bool
    context: dict | None = None
    original_exception: Exception | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.error_type}: {self.message}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class ErrorHandler:
    """
    Handles errors during agent execution.
    
    Provides:
    - Error categorization
    - Retry logic with exponential backoff
    - Fallback behaviors
    - Error history tracking
    
    Example:
        >>> handler = ErrorHandler(max_retries=3)
        >>> result, error = handler.with_retry(risky_function, arg1, arg2)
        >>> if error:
        ...     print(f"Failed: {error}")
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_enabled: bool = True,
        max_history: int = 100
    ):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (seconds)
            fallback_enabled: Whether to use fallback values
            max_history: Maximum number of errors to keep in history
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_enabled = fallback_enabled
        self.max_history = max_history
        self.error_history: list[AgentError] = []
    
    def categorize_error(self, exception: Exception) -> AgentError:
        """
        Categorize an exception into an AgentError.
        
        Uses the exception type and message to determine severity
        and recoverability.
        
        Args:
            exception: The Python exception
            
        Returns:
            Categorized AgentError
        """
        error_type = type(exception).__name__
        message = str(exception)
        message_lower = message.lower()
        
        # Rate limiting - recoverable
        if any(term in message_lower for term in ["rate", "limit", "quota", "throttl"]):
            return AgentError(
                error_type="RateLimitError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        # Timeout - recoverable
        if any(term in message_lower for term in ["timeout", "timed out", "deadline"]):
            return AgentError(
                error_type="TimeoutError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        # Authentication - not recoverable without intervention
        if any(term in message_lower for term in ["auth", "unauthorized", "forbidden", "api key", "invalid key"]):
            return AgentError(
                error_type="AuthenticationError",
                message=message,
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
                original_exception=exception
            )
        
        # Connection - often recoverable
        if any(term in message_lower for term in ["connection", "network", "dns", "resolve", "unreachable"]):
            return AgentError(
                error_type="ConnectionError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        # Validation - not recoverable without input change
        if any(term in message_lower for term in ["invalid", "validation", "schema", "format"]):
            return AgentError(
                error_type="ValidationError",
                message=message,
                severity=ErrorSeverity.HIGH,
                recoverable=False,
                original_exception=exception
            )
        
        # Resource errors - may need intervention
        if any(term in message_lower for term in ["memory", "disk", "space", "resource"]):
            return AgentError(
                error_type="ResourceError",
                message=message,
                severity=ErrorSeverity.HIGH,
                recoverable=False,
                original_exception=exception
            )
        
        # Parse errors - sometimes recoverable
        if any(term in message_lower for term in ["parse", "json", "decode", "syntax"]):
            return AgentError(
                error_type="ParseError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,  # Often can retry with different approach
                original_exception=exception
            )
        
        # Default - assume not recoverable
        return AgentError(
            error_type=error_type,
            message=message,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            original_exception=exception
        )
    
    def with_retry(
        self,
        func: Callable[..., T],
        *args,
        on_retry: Callable[[int, AgentError], None] | None = None,
        **kwargs
    ) -> tuple[T | None, AgentError | None]:
        """
        Execute a function with retry logic.
        
        Uses exponential backoff between retries.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            on_retry: Optional callback called on each retry
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, error). Error is None on success.
        """
        last_error: AgentError | None = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, None
            
            except Exception as e:
                error = self.categorize_error(e)
                self._add_to_history(error)
                last_error = error
                
                # Don't retry non-recoverable errors
                if not error.recoverable:
                    return None, error
                
                # Call retry callback if provided
                if on_retry and attempt < self.max_retries:
                    on_retry(attempt + 1, error)
                
                # Wait before retrying (exponential backoff)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
        
        return None, last_error
    
    async def with_retry_async(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> tuple[T | None, AgentError | None]:
        """
        Async version of with_retry.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, error). Error is None on success.
        """
        import asyncio
        
        last_error: AgentError | None = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result, None
            
            except Exception as e:
                error = self.categorize_error(e)
                self._add_to_history(error)
                last_error = error
                
                if not error.recoverable:
                    return None, error
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        return None, last_error
    
    def handle_tool_error(
        self,
        tool_name: str,
        exception: Exception,
        fallback_result: Any = None
    ) -> tuple[Any, bool]:
        """
        Handle an error from tool execution.
        
        Args:
            tool_name: Name of the tool that failed
            exception: The exception that occurred
            fallback_result: Value to return if fallback is enabled
            
        Returns:
            Tuple of (result, success). Result may be fallback value.
        """
        error = self.categorize_error(exception)
        error.context = {"tool_name": tool_name}
        self._add_to_history(error)
        
        # Use fallback if enabled and provided
        if self.fallback_enabled and fallback_result is not None:
            return fallback_result, False
        
        # Return helpful error message
        if error.recoverable:
            return f"Tool '{tool_name}' failed temporarily. Error: {error.message}", False
        
        return f"Tool '{tool_name}' failed: {error.message}", False
    
    def _add_to_history(self, error: AgentError) -> None:
        """Add error to history, maintaining max size."""
        self.error_history.append(error)
        
        # Trim old errors
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_error_summary(self) -> str:
        """
        Get a summary of errors encountered.
        
        Returns:
            Human-readable error summary
        """
        if not self.error_history:
            return "No errors recorded."
        
        summary_lines = [f"Total errors: {len(self.error_history)}"]
        
        # Count by severity
        by_severity: dict[str, int] = {}
        for error in self.error_history:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        for severity, count in sorted(by_severity.items()):
            summary_lines.append(f"  {severity}: {count}")
        
        # Count by type
        by_type: dict[str, int] = {}
        for error in self.error_history:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
        
        summary_lines.append("\nBy type:")
        for error_type, count in sorted(by_type.items()):
            summary_lines.append(f"  {error_type}: {count}")
        
        return "\n".join(summary_lines)
    
    def get_recent_errors(self, n: int = 5) -> list[AgentError]:
        """Get the n most recent errors."""
        return self.error_history[-n:]
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors have occurred."""
        return any(
            e.severity == ErrorSeverity.CRITICAL 
            for e in self.error_history
        )
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history = []


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing repeated failures.
    
    When too many failures occur, the circuit "opens" and prevents
    further attempts until a cooldown period passes.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        half_open_max_calls: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            cooldown_seconds: Time before trying again
            half_open_max_calls: Test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if cooldown has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.cooldown_seconds:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    return True
            return False
        
        # HALF_OPEN
        return self.half_open_calls < self.half_open_max_calls
    
    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == "HALF_OPEN":
            # Recovery successful
            self.state = "CLOSED"
        
        self.failure_count = 0
        self.half_open_calls = 0
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            # Recovery failed
            self.state = "OPEN"
            self.half_open_calls = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Raises:
            RuntimeError: If circuit is open
        """
        if not self.can_execute():
            raise RuntimeError(f"Circuit breaker is {self.state}")
        
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


if __name__ == "__main__":
    # Demonstrate error handling
    print("=== ErrorHandler Demonstration ===\n")
    
    handler = ErrorHandler(max_retries=3, retry_delay=0.5)
    
    # Test error categorization
    print("=== Error Categorization ===")
    
    test_exceptions = [
        Exception("Rate limit exceeded"),
        Exception("Connection refused"),
        Exception("Invalid API key"),
        Exception("Request timed out"),
        Exception("JSON parse error"),
        Exception("Unknown error"),
    ]
    
    for exc in test_exceptions:
        error = handler.categorize_error(exc)
        print(f"{error.error_type}: severity={error.severity.value}, recoverable={error.recoverable}")
    
    # Test retry logic
    print("\n=== Retry Logic ===")
    
    call_count = 0
    
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "Success!"
    
    result, error = handler.with_retry(
        flaky_function,
        on_retry=lambda attempt, err: print(f"Retry {attempt}: {err.message}")
    )
    print(f"Result: {result}, Error: {error}")
    
    # Test circuit breaker
    print("\n=== Circuit Breaker ===")
    
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=2)
    
    def always_fails():
        raise Exception("Always fails")
    
    for i in range(5):
        try:
            breaker.execute(always_fails)
        except Exception as e:
            print(f"Call {i+1}: {e} (state={breaker.state})")
    
    # Error summary
    print(f"\n=== Error Summary ===\n{handler.get_error_summary()}")
