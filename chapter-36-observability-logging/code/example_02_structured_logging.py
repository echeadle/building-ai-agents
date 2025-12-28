"""
Structured logging for AI agents.

Chapter 36: Observability and Logging

This script demonstrates JSON-structured logging, which makes logs
easily parseable by log aggregation tools and allows for sophisticated
querying and analysis.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    A logging formatter that outputs JSON-structured log entries.
    
    This makes logs easily parseable by log aggregation tools like:
    - Elasticsearch/ELK Stack
    - Splunk
    - Datadog
    - AWS CloudWatch
    
    Benefits of structured logging:
    - Machine-readable format
    - Easy filtering and searching
    - Consistent field names
    - Can include arbitrary metadata
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.
        
        Args:
            record: The logging record to format
        
        Returns:
            JSON-formatted string
        """
        # Build the base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add location information for debugging
        log_entry["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add any extra fields that were passed to the logger
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        return json.dumps(log_entry)


class StructuredLogger:
    """
    A wrapper around Python's logging that adds structured data support.
    
    This makes it easy to add arbitrary fields to log messages,
    which is essential for agent observability.
    
    Usage:
        logger = StructuredLogger("my_agent")
        
        # Simple message
        logger.info("Agent started")
        
        # Message with structured data
        logger.info("Tool called", tool_name="weather", duration_ms=150)
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize a structured logger.
        
        Args:
            name: The logger name (usually module or component name)
            level: The minimum log level to record
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Add our JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """
        Internal method to log with extra fields.
        
        Args:
            level: The log level
            message: The log message
            **kwargs: Additional fields to include in the log entry
        """
        # Create the extra dict that StructuredFormatter will look for
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message with optional structured data."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message with optional structured data."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message with optional structured data."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message with optional structured data."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message with optional structured data."""
        self._log(logging.CRITICAL, message, **kwargs)


def demonstrate_structured_logging():
    """
    Demonstrate structured logging with various types of data.
    """
    print("=" * 70)
    print("Structured Logging Demonstration")
    print("=" * 70)
    print()
    print("Each log line is a valid JSON object that can be parsed")
    print("and queried by log aggregation tools.")
    print()
    print("-" * 70)
    print()
    
    logger = StructuredLogger("agent.demo", level=logging.DEBUG)
    
    # Simple message
    logger.info("Agent initialized")
    
    # Message with tool call data
    logger.info(
        "Tool executed",
        tool_name="weather",
        location="San Francisco, CA",
        duration_ms=245,
        success=True
    )
    
    # Message with token usage
    logger.debug(
        "LLM call completed",
        model="claude-sonnet-4-20250514",
        input_tokens=150,
        output_tokens=87,
        total_tokens=237,
        latency_ms=892
    )
    
    # Warning with context
    logger.warning(
        "Rate limit approaching",
        current_requests=95,
        max_requests=100,
        reset_in_seconds=30
    )
    
    # Error with detailed context
    logger.error(
        "Tool execution failed",
        tool_name="calculator",
        input_expression="1/0",
        error_type="ZeroDivisionError",
        retry_count=2
    )
    
    # Message with nested data
    logger.info(
        "Request completed",
        request_id="req_abc123",
        user_id="user_456",
        metrics={
            "total_llm_calls": 3,
            "total_tool_calls": 2,
            "total_tokens": 450
        }
    )


def demonstrate_filtering_structured_logs():
    """
    Show how structured logs can be filtered using command-line tools.
    """
    print()
    print("-" * 70)
    print()
    print("Filtering structured logs with command-line tools:")
    print()
    print("  # Find all errors:")
    print("  cat logs.json | jq 'select(.level == \"ERROR\")'")
    print()
    print("  # Find slow tool calls (>500ms):")
    print("  cat logs.json | jq 'select(.duration_ms > 500)'")
    print()
    print("  # Find all weather tool calls:")
    print("  cat logs.json | jq 'select(.tool_name == \"weather\")'")
    print()
    print("  # Calculate average latency:")
    print("  cat logs.json | jq -s '[.[].latency_ms | select(. != null)] | add/length'")
    print()


if __name__ == "__main__":
    demonstrate_structured_logging()
    demonstrate_filtering_structured_logs()
    
    print("-" * 70)
    print()
    print("Structured logging demonstration complete!")
