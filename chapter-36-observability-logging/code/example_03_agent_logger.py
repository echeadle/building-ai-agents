"""
Complete AgentLogger class for AI agent observability.

Chapter 36: Observability and Logging

This module provides a comprehensive logging solution specifically
designed for AI agents. It includes:
- Structured JSON logging
- Request tracing with unique IDs
- Tool call tracking with timing
- LLM call logging with token counts
- Metrics aggregation
"""

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Generator, Optional


@dataclass
class ToolCallRecord:
    """
    Record of a single tool call.
    
    Captures everything about a tool invocation for later analysis.
    """
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class LLMCallRecord:
    """
    Record of a single LLM API call.
    
    Captures token usage and timing for cost and performance analysis.
    """
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    stop_reason: Optional[str] = None


@dataclass
class RequestTrace:
    """
    Complete trace of a single agent request from start to finish.
    
    This captures everything that happened during one user interaction,
    enabling detailed analysis and debugging.
    """
    trace_id: str
    start_time: str
    end_time: Optional[str] = None
    user_input: Optional[str] = None
    final_output: Optional[str] = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for JSON serialization."""
        return asdict(self)


class _AgentLogFormatter(logging.Formatter):
    """Internal JSON formatter for agent logs."""
    
    def __init__(self, include_timestamps: bool = True):
        super().__init__()
        self.include_timestamps = include_timestamps
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if self.include_timestamps:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            for key, value in record.extra_fields.items():
                if value is not None:  # Skip None values
                    log_entry[key] = value
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class AgentLogger:
    """
    Comprehensive logging and observability for AI agents.
    
    Features:
    - Structured JSON logging
    - Request tracing with unique trace IDs
    - Tool call tracking with timing
    - LLM call tracking with token counts
    - Performance metrics collection
    - Configurable log levels and outputs
    
    Usage:
        logger = AgentLogger("my_agent")
        
        with logger.trace_request("What's the weather?") as trace:
            # Your agent logic here
            with logger.trace_tool_call("weather", {"location": "NYC"}):
                result = call_weather_api()
            
            logger.log_llm_call(
                model="claude-sonnet-4-20250514",
                input_tokens=100,
                output_tokens=50,
                duration_ms=500
            )
        
        # Access metrics
        print(logger.get_metrics_summary())
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_to_file: Optional[str] = None,
        include_timestamps: bool = True
    ):
        """
        Initialize the agent logger.
        
        Args:
            name: Logger name (typically the agent name)
            level: Minimum log level (default: INFO)
            log_to_file: Optional file path for log output
            include_timestamps: Whether to include timestamps in logs
        """
        self.name = name
        self.include_timestamps = include_timestamps
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        self.logger.propagate = False  # Don't propagate to root logger
        
        # Console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._create_formatter())
        self.logger.addHandler(console_handler)
        
        # Optional file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setFormatter(self._create_formatter())
            self.logger.addHandler(file_handler)
        
        # Current active trace (for nested context)
        self._current_trace: Optional[RequestTrace] = None
        
        # Metrics aggregation
        self._all_traces: list[RequestTrace] = []
    
    def _create_formatter(self) -> logging.Formatter:
        """Create the JSON formatter for log output."""
        return _AgentLogFormatter(include_timestamps=self.include_timestamps)
    
    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal logging method with structured data support."""
        extra = {
            "extra_fields": {
                **kwargs,
                "trace_id": self._current_trace.trace_id if self._current_trace else None
            }
        }
        self.logger.log(level, message, extra=extra)
    
    # -------------------------------------------------------------------------
    # Basic Logging Methods
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # Request Tracing
    # -------------------------------------------------------------------------
    
    @contextmanager
    def trace_request(
        self,
        user_input: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> Generator[RequestTrace, None, None]:
        """
        Context manager for tracing a complete agent request.
        
        Args:
            user_input: The user's input message
            metadata: Optional additional metadata to attach to the trace
        
        Yields:
            RequestTrace object that can be updated during execution
        
        Example:
            with logger.trace_request("What's the weather?") as trace:
                # Agent logic here
                trace.metadata["user_id"] = "user123"
        """
        trace_id = str(uuid.uuid4())[:8]  # Short trace ID for readability
        start_time = time.perf_counter()
        
        trace = RequestTrace(
            trace_id=trace_id,
            start_time=self._now(),
            user_input=user_input,
            metadata=metadata or {}
        )
        
        self._current_trace = trace
        
        # Truncate long inputs for logging
        display_input = (
            user_input[:100] + "..." 
            if len(user_input) > 100 
            else user_input
        )
        
        self.info("Request started", user_input=display_input)
        
        try:
            yield trace
            
        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)}"
            self.error("Request failed", error=trace.error)
            raise
            
        finally:
            trace.end_time = self._now()
            trace.total_duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Calculate totals
            trace.total_input_tokens = sum(
                call.input_tokens for call in trace.llm_calls
            )
            trace.total_output_tokens = sum(
                call.output_tokens for call in trace.llm_calls
            )
            
            self.info(
                "Request completed",
                duration_ms=round(trace.total_duration_ms, 2),
                llm_calls=len(trace.llm_calls),
                tool_calls=len(trace.tool_calls),
                total_tokens=trace.total_input_tokens + trace.total_output_tokens
            )
            
            # Store trace for metrics
            self._all_traces.append(trace)
            self._current_trace = None
    
    # -------------------------------------------------------------------------
    # Tool Call Tracing
    # -------------------------------------------------------------------------
    
    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> Generator[ToolCallRecord, None, None]:
        """
        Context manager for tracing a tool call.
        
        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters passed to the tool
        
        Yields:
            ToolCallRecord that can be updated with the result
        
        Example:
            with logger.trace_tool_call("weather", {"city": "NYC"}) as call:
                result = weather_api.get(city="NYC")
                call.tool_output = result
        """
        start_time = time.perf_counter()
        
        record = ToolCallRecord(
            tool_name=tool_name,
            tool_input=tool_input,
            start_time=self._now()
        )
        
        self.debug(
            "Tool call started",
            tool_name=tool_name,
            tool_input=tool_input
        )
        
        try:
            yield record
            
        except Exception as e:
            record.error = f"{type(e).__name__}: {str(e)}"
            self.warning(
                "Tool call failed",
                tool_name=tool_name,
                error=record.error
            )
            raise
            
        finally:
            record.end_time = self._now()
            record.duration_ms = (time.perf_counter() - start_time) * 1000
            
            if self._current_trace:
                self._current_trace.tool_calls.append(record)
            
            self.debug(
                "Tool call completed",
                tool_name=tool_name,
                duration_ms=round(record.duration_ms, 2),
                success=record.error is None
            )
    
    # -------------------------------------------------------------------------
    # LLM Call Logging
    # -------------------------------------------------------------------------
    
    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        stop_reason: Optional[str] = None
    ) -> None:
        """
        Log an LLM API call.
        
        Args:
            model: The model name/ID used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration_ms: How long the call took
            stop_reason: Why the model stopped generating
        """
        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            duration_ms=duration_ms,
            stop_reason=stop_reason
        )
        
        if self._current_trace:
            self._current_trace.llm_calls.append(record)
        
        self.debug(
            "LLM call completed",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration_ms, 2),
            stop_reason=stop_reason
        )
    
    # -------------------------------------------------------------------------
    # Metrics and Reporting
    # -------------------------------------------------------------------------
    
    def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get a summary of all collected metrics.
        
        Returns:
            Dictionary containing aggregated metrics including:
            - Request counts and success rates
            - Latency statistics
            - Token usage
            - Tool usage breakdown
        """
        if not self._all_traces:
            return {"message": "No traces collected yet"}
        
        total_requests = len(self._all_traces)
        successful_requests = sum(
            1 for t in self._all_traces if t.error is None
        )
        
        all_durations = [
            t.total_duration_ms for t in self._all_traces
            if t.total_duration_ms is not None
        ]
        
        total_input_tokens = sum(t.total_input_tokens for t in self._all_traces)
        total_output_tokens = sum(t.total_output_tokens for t in self._all_traces)
        
        total_tool_calls = sum(len(t.tool_calls) for t in self._all_traces)
        total_llm_calls = sum(len(t.llm_calls) for t in self._all_traces)
        
        # Tool call breakdown
        tool_usage: dict[str, int] = {}
        for trace in self._all_traces:
            for tool_call in trace.tool_calls:
                tool_usage[tool_call.tool_name] = (
                    tool_usage.get(tool_call.tool_name, 0) + 1
                )
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": round(successful_requests / total_requests * 100, 1),
            "latency": {
                "mean_ms": round(sum(all_durations) / len(all_durations), 2) if all_durations else 0,
                "min_ms": round(min(all_durations), 2) if all_durations else 0,
                "max_ms": round(max(all_durations), 2) if all_durations else 0,
            },
            "tokens": {
                "total_input": total_input_tokens,
                "total_output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            "calls": {
                "total_llm_calls": total_llm_calls,
                "total_tool_calls": total_tool_calls,
                "avg_llm_calls_per_request": round(total_llm_calls / total_requests, 2),
                "avg_tool_calls_per_request": round(total_tool_calls / total_requests, 2),
            },
            "tool_usage": tool_usage,
        }
    
    def get_recent_traces(self, n: int = 10) -> list[dict[str, Any]]:
        """
        Get the N most recent request traces.
        
        Args:
            n: Number of traces to return
        
        Returns:
            List of trace dictionaries
        """
        return [trace.to_dict() for trace in self._all_traces[-n:]]
    
    def export_traces(self, filepath: str) -> None:
        """
        Export all traces to a JSON file.
        
        Args:
            filepath: Path to write the JSON file
        """
        with open(filepath, "w") as f:
            json.dump(
                [trace.to_dict() for trace in self._all_traces],
                f,
                indent=2
            )
        self.info(f"Exported {len(self._all_traces)} traces", filepath=filepath)
    
    def clear_traces(self) -> None:
        """Clear all stored traces (useful for testing)."""
        self._all_traces.clear()


# Example usage demonstrating all features
if __name__ == "__main__":
    import random
    
    print("=" * 70)
    print("AgentLogger Demonstration")
    print("=" * 70)
    print()
    
    # Create logger with DEBUG level to see all output
    logger = AgentLogger("demo_agent", level=logging.DEBUG)
    
    # Simulate multiple requests
    for i in range(3):
        user_input = f"Test request number {i + 1}"
        
        print(f"\n{'=' * 40}")
        print(f"Request {i + 1}")
        print(f"{'=' * 40}\n")
        
        with logger.trace_request(user_input, metadata={"request_num": i + 1}) as trace:
            # Simulate some tool calls
            with logger.trace_tool_call("calculator", {"expression": "2 + 2"}) as tc:
                time.sleep(0.05)  # Simulate work
                tc.tool_output = "4"
            
            if random.random() > 0.5:
                with logger.trace_tool_call("weather", {"location": "NYC"}) as tc:
                    time.sleep(0.08)
                    tc.tool_output = "Sunny, 72°F"
            
            # Log an LLM call
            logger.log_llm_call(
                model="claude-sonnet-4-20250514",
                input_tokens=random.randint(100, 300),
                output_tokens=random.randint(50, 150),
                duration_ms=random.uniform(300, 800),
                stop_reason="end_turn"
            )
            
            trace.final_output = f"Response to: {user_input}"
    
    # Print metrics summary
    print("\n" + "=" * 70)
    print("Metrics Summary")
    print("=" * 70)
    print(json.dumps(logger.get_metrics_summary(), indent=2))
    
    # Export traces
    logger.export_traces("/tmp/agent_traces.json")
    print(f"\nTraces exported to /tmp/agent_traces.json")
