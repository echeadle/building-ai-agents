"""
Structured logging for agent error tracking.

Chapter 30: Error Handling and Recovery

This module provides comprehensive logging utilities for tracking
and analyzing agent errors.
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
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# ============================================================
# Error Classification
# ============================================================

class ErrorSeverity(Enum):
    """
    Severity levels for agent errors.
    
    These map to standard logging levels.
    """
    DEBUG = "debug"      # Diagnostic information
    INFO = "info"        # Normal operation markers
    WARNING = "warning"  # Recoverable issues
    ERROR = "error"      # Serious problems
    CRITICAL = "critical"  # System-threatening issues


class ErrorCategory(Enum):
    """
    Categories of agent errors.
    
    Categorizing errors helps with:
    - Filtering and searching logs
    - Automated alerting rules
    - Error analysis and patterns
    """
    API = "api"              # API-related errors
    NETWORK = "network"      # Network/connectivity issues
    PARSING = "parsing"      # JSON/data parsing errors
    TOOL = "tool"            # Tool execution errors
    LOGIC = "logic"          # Agent logic errors (loops, stuck)
    VALIDATION = "validation"  # Input/output validation
    TIMEOUT = "timeout"      # Timeout errors
    UNKNOWN = "unknown"      # Unclassified errors


# ============================================================
# Structured Error Representation
# ============================================================

@dataclass
class AgentError:
    """
    Structured representation of an agent error.
    
    This class captures all relevant information about an error
    for logging, analysis, and debugging.
    """
    # Core error information
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context - where the error occurred
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    tool_name: Optional[str] = None
    step_number: Optional[int] = None
    
    # Exception details
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Recovery information
    recoverable: bool = True
    retry_count: int = 0
    recovery_action: Optional[str] = None
    
    # Additional context
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        **kwargs
    ) -> "AgentError":
        """
        Create an AgentError from an exception.
        
        Automatically extracts exception details.
        """
        # Auto-categorize if not provided
        if category is None:
            category = categorize_exception(exception)
        
        return cls(
            message=str(exception),
            category=category,
            severity=ErrorSeverity.ERROR,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            **kwargs
        )


def categorize_exception(exception: Exception) -> ErrorCategory:
    """
    Automatically categorize an exception.
    
    This helps ensure consistent error categorization across the codebase.
    """
    exception_categories = {
        # Anthropic API errors
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
        KeyError: ErrorCategory.PARSING,
    }
    
    for exc_type, category in exception_categories.items():
        if isinstance(exception, exc_type):
            return category
    
    return ErrorCategory.UNKNOWN


# ============================================================
# Agent Logger
# ============================================================

class AgentLogger:
    """
    Structured logger for agent operations.
    
    Provides consistent formatting and categorization for agent errors
    and events, with support for both human-readable and structured output.
    """
    
    def __init__(
        self,
        name: str = "agent",
        level: int = logging.INFO,
        include_console: bool = True,
        log_file: Optional[str] = None,
        json_format: bool = False
    ):
        """
        Initialize the agent logger.
        
        Args:
            name: Logger name (appears in log messages)
            level: Minimum log level to capture
            include_console: Whether to log to console
            log_file: Optional file path for log output
            json_format: Whether to use JSON format for logs
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        self.json_format = json_format
        
        # Create formatter
        if json_format:
            formatter = logging.Formatter('%(message)s')
        else:
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
    
    def _format_message(self, message: str, **context) -> str:
        """Format a log message with optional context."""
        if self.json_format:
            return json.dumps({
                "message": message,
                "timestamp": datetime.now().isoformat(),
                **context
            })
        return f"{message}{self._format_context(**context)}"
    
    # Standard logging methods
    def debug(self, message: str, **context):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **context))
    
    def info(self, message: str, **context):
        """Log info message."""
        self.logger.info(self._format_message(message, **context))
    
    def warning(self, message: str, **context):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **context))
    
    def error(self, message: str, **context):
        """Log error message."""
        self.logger.error(self._format_message(message, **context))
    
    def critical(self, message: str, **context):
        """Log critical message."""
        self.logger.critical(self._format_message(message, **context))
    
    # Structured error logging
    def log_agent_error(self, error: AgentError):
        """
        Log a structured agent error.
        
        Records the error in history and logs with appropriate level.
        """
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
        
        if self.json_format:
            self.logger.log(level, error.to_json())
        else:
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
                    if line.strip():
                        self.logger.log(level, f"  {line}")
    
    def log_exception(
        self,
        exception: Exception,
        category: Optional[ErrorCategory] = None,
        context: Optional[dict] = None,
        recoverable: bool = True,
        **kwargs
    ) -> AgentError:
        """
        Create and log an AgentError from an exception.
        
        Args:
            exception: The exception to log
            category: Error category (auto-detected if None)
            context: Additional context dictionary
            recoverable: Whether the error is recoverable
            **kwargs: Additional AgentError fields
            
        Returns:
            The created AgentError for further processing
        """
        error = AgentError.from_exception(
            exception,
            category=category,
            recoverable=recoverable,
            metadata=context or {},
            **kwargs
        )
        
        self.log_agent_error(error)
        return error
    
    # Agent lifecycle logging
    def log_agent_start(self, agent_id: str, **context):
        """Log agent startup."""
        self.info(f"Agent starting", agent_id=agent_id, **context)
    
    def log_agent_stop(self, agent_id: str, reason: str = "completed", **context):
        """Log agent shutdown."""
        self.info(f"Agent stopping: {reason}", agent_id=agent_id, **context)
    
    def log_tool_call(self, tool_name: str, input_data: dict, **context):
        """Log a tool invocation."""
        self.debug(
            f"Tool call: {tool_name}",
            tool=tool_name,
            input=json.dumps(input_data)[:100],
            **context
        )
    
    def log_tool_result(self, tool_name: str, success: bool, **context):
        """Log a tool result."""
        status = "success" if success else "failed"
        self.debug(f"Tool {status}: {tool_name}", tool=tool_name, **context)
    
    # Summary and analysis
    def get_error_summary(self) -> dict:
        """
        Get a summary of logged errors.
        
        Returns statistics useful for monitoring and debugging.
        """
        summary = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "recoverable": 0,
            "non_recoverable": 0,
            "unique_tools": set(),
            "time_range": None,
        }
        
        if not self.error_history:
            return summary
        
        timestamps = []
        
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
            
            # Track tools
            if error.tool_name:
                summary["unique_tools"].add(error.tool_name)
            
            timestamps.append(error.timestamp)
        
        # Convert set to list for JSON serialization
        summary["unique_tools"] = list(summary["unique_tools"])
        
        # Time range
        if timestamps:
            summary["time_range"] = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            }
        
        return summary
    
    def get_recent_errors(self, n: int = 5) -> list[dict]:
        """Get the N most recent errors as dictionaries."""
        return [e.to_dict() for e in self.error_history[-n:]]
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGENT ERROR LOGGING DEMONSTRATION")
    print("=" * 60)
    
    # Create logger
    logger = AgentLogger("demo_agent", level=logging.DEBUG)
    
    print("\n### 1. Basic Logging ###\n")
    
    logger.info("Agent starting", agent_id="agent-001", model="claude-sonnet-4-20250514")
    logger.debug("Loading tools", tools=["calculator", "weather", "search"])
    logger.warning("Rate limit approaching", usage="80%")
    
    print("\n### 2. Structured Error Logging ###\n")
    
    # Create and log a structured error
    error = AgentError(
        message="Weather API returned invalid JSON",
        category=ErrorCategory.PARSING,
        severity=ErrorSeverity.WARNING,
        tool_name="weather",
        recoverable=True,
        recovery_action="Using cached data",
        metadata={"response_length": 0, "status_code": 200}
    )
    logger.log_agent_error(error)
    
    print("\n### 3. Exception Logging ###\n")
    
    # Log from exception
    try:
        json.loads("invalid json{")
    except json.JSONDecodeError as e:
        logger.log_exception(
            e,
            context={"source": "weather_api_response", "attempt": 1}
        )
    
    print("\n### 4. Tool Call Logging ###\n")
    
    logger.log_tool_call("calculator", {"expression": "2 + 2"})
    logger.log_tool_result("calculator", success=True)
    
    logger.log_tool_call("weather", {"location": "New York"})
    logger.log_tool_result("weather", success=False)
    
    print("\n### 5. Critical Error ###\n")
    
    critical_error = AgentError(
        message="Agent stuck in infinite loop",
        category=ErrorCategory.LOGIC,
        severity=ErrorSeverity.CRITICAL,
        agent_id="agent-001",
        recoverable=False,
        metadata={"loop_count": 50, "last_tool": "search"}
    )
    logger.log_agent_error(critical_error)
    
    print("\n### 6. Error Summary ###\n")
    
    summary = logger.get_error_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n### 7. Recent Errors ###\n")
    
    recent = logger.get_recent_errors(3)
    for i, error in enumerate(recent, 1):
        print(f"{i}. [{error['category']}] {error['message'][:50]}...")
    
    print("\n### 8. JSON Format Demo ###\n")
    
    json_logger = AgentLogger("json_agent", json_format=True)
    json_logger.info("This is JSON formatted", agent_id="test")
    json_logger.log_agent_error(AgentError(
        message="JSON formatted error",
        category=ErrorCategory.API,
        severity=ErrorSeverity.WARNING
    ))
    
    print("\n" + "=" * 60)
    print("KEY FEATURES:")
    print("- Structured errors with full metadata")
    print("- Automatic exception categorization")
    print("- Error history and summaries")
    print("- Both human-readable and JSON formats")
    print("=" * 60)
