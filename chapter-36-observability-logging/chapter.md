---
chapter: 36
title: "Observability and Logging"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 36: Observability and Logging

## Introduction

Imagine your agent is deployed in production. A user reports that it "gave a weird answer." How do you investigate? Without proper observability, you're flying blind—guessing at what went wrong, unable to reproduce the issue, and hoping the problem doesn't happen again.

**Observability** is the ability to understand what's happening inside your system by examining its outputs. For AI agents, this means seeing every decision the LLM makes, every tool it calls, how long operations take, and where things go wrong. It's the difference between "something broke" and "at 14:32:07, the agent called the weather tool with invalid coordinates, received an error, and then hallucinated a response instead of retrying."

In the previous chapter, we built a test suite for our agents. Testing tells you if something is broken; observability tells you *why* and *where*. Together, they form the foundation of production-ready agent systems.

In this chapter, you'll build a comprehensive logging and observability system specifically designed for AI agents. This isn't just `print()` statements—it's structured, queryable, actionable insight into your agent's behavior.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement structured logging that captures agent decisions and tool calls
- Build a tracing system to follow the complete lifecycle of agent requests
- Collect and report performance metrics for latency and cost
- Configure log levels to balance verbosity with usefulness
- Design log aggregation patterns for production environments

## Why Agent Observability Is Different

Standard application logging focuses on errors and request/response cycles. Agent observability requires more because agents have unique characteristics:

**Multi-step reasoning**: A single user request might trigger dozens of internal LLM calls and tool invocations. You need to trace the entire chain.

**Non-deterministic behavior**: The same input can produce different outputs. You need enough context to understand *why* the agent made each decision.

**Tool interactions**: Agents call external tools that can fail, timeout, or return unexpected data. You need visibility into each tool call.

**Token economics**: Every API call costs money. You need to track token usage to understand costs.

**Latency budgets**: Users expect fast responses. You need to identify which steps are slow.

Let's build a logging system that addresses all of these needs.

## Python's Logging Module Basics

Before we build agent-specific logging, let's review Python's built-in `logging` module. If you're already familiar with it, feel free to skim this section.

```python
"""
Basic Python logging demonstration.

Chapter 36: Observability and Logging
"""

import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demonstrate_log_levels():
    """Show the different logging levels."""
    logger.debug("Debug: Detailed diagnostic information")
    logger.info("Info: Confirmation that things are working")
    logger.warning("Warning: Something unexpected happened")
    logger.error("Error: A serious problem occurred")
    logger.critical("Critical: The program may not continue")

if __name__ == "__main__":
    demonstrate_log_levels()
```

**Log Levels** (from least to most severe):

| Level | Value | When to Use |
|-------|-------|-------------|
| DEBUG | 10 | Detailed diagnostic info for developers |
| INFO | 20 | Confirmation of normal operation |
| WARNING | 30 | Something unexpected but not critical |
| ERROR | 40 | A problem that prevented an operation |
| CRITICAL | 50 | A serious error that may crash the program |

The key insight: you set a *threshold* level, and only messages at that level or above are logged. In production, you might use WARNING to reduce noise. During debugging, you'd use DEBUG to see everything.

## Structured Logging for Agents

Plain text logs are hard to query and analyze. Instead, we'll use **structured logging**—logs in JSON format that can be parsed, filtered, and aggregated by tools like Elasticsearch, Splunk, or even simple Python scripts.

```python
"""
Structured logging for AI agents.

Chapter 36: Observability and Logging
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    A logging formatter that outputs JSON-structured log entries.
    
    This makes logs easily parseable by log aggregation tools
    and allows for sophisticated querying and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add any extra fields that were passed to the logger
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class StructuredLogger:
    """
    A wrapper around Python's logging that adds structured data support.
    
    Usage:
        logger = StructuredLogger("my_agent")
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
        
        # Avoid adding duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal method to log with extra fields."""
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


# Example usage
if __name__ == "__main__":
    logger = StructuredLogger("agent.tools")
    
    # Simple message
    logger.info("Agent started")
    
    # Message with structured data
    logger.info(
        "Tool executed",
        tool_name="weather",
        location="San Francisco",
        duration_ms=245,
        success=True
    )
    
    # Error with context
    logger.error(
        "Tool failed",
        tool_name="calculator",
        input_expression="1/0",
        error_type="ZeroDivisionError"
    )
```

Running this produces JSON logs:

```json
{"timestamp": "2025-01-15T10:30:00.123456+00:00", "level": "INFO", "logger": "agent.tools", "message": "Agent started"}
{"timestamp": "2025-01-15T10:30:00.124789+00:00", "level": "INFO", "logger": "agent.tools", "message": "Tool executed", "tool_name": "weather", "location": "San Francisco", "duration_ms": 245, "success": true}
{"timestamp": "2025-01-15T10:30:00.125012+00:00", "level": "ERROR", "logger": "agent.tools", "message": "Tool failed", "tool_name": "calculator", "input_expression": "1/0", "error_type": "ZeroDivisionError"}
```

Now you can grep for specific tools, filter by error type, or aggregate durations across all tool calls.

## Building the AgentLogger Class

Let's create a comprehensive logging class specifically designed for agent observability. This will be the foundation of our observability system.

```python
"""
Complete AgentLogger class for AI agent observability.

Chapter 36: Observability and Logging
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
    """Record of a single tool call."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
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
    
    This captures everything that happened during one user interaction.
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
            
            logger.log_llm_call(model="claude-sonnet-4-20250514", input_tokens=100, ...)
        
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
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
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
        
        self.info(
            "Request started",
            user_input=user_input[:100] + "..." if len(user_input) > 100 else user_input
        )
        
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
            Dictionary containing aggregated metrics
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
        """Get the N most recent request traces."""
        return [trace.to_dict() for trace in self._all_traces[-n:]]
    
    def export_traces(self, filepath: str) -> None:
        """Export all traces to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                [trace.to_dict() for trace in self._all_traces],
                f,
                indent=2
            )
        self.info(f"Exported {len(self._all_traces)} traces", filepath=filepath)


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
```

This `AgentLogger` class provides everything you need for comprehensive agent observability:

- **Structured JSON output** for easy parsing
- **Request tracing** with unique IDs to follow a request through the system
- **Tool call tracking** with automatic timing
- **LLM call logging** with token counts
- **Metrics aggregation** for performance analysis

## Tracing Tool Calls and Decisions

One of the most valuable aspects of agent observability is understanding *why* the agent did what it did. Let's build a more detailed tracing system that captures the agent's decision-making process.

```python
"""
Detailed tracing for agent decisions and tool calls.

Chapter 36: Observability and Logging
"""

import os
import time
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Import our AgentLogger (in practice, from your agents package)
# from agents.utils.logging import AgentLogger

# For this example, we'll use a simplified version
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class DecisionTrace:
    """Captures the reasoning behind an agent decision."""
    decision_type: str  # "tool_selection", "response_generation", "planning"
    options_considered: list[str] = field(default_factory=list)
    chosen_option: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


class TracingAgent:
    """
    An agent with detailed decision tracing.
    
    This demonstrates how to capture not just what the agent does,
    but why it makes each decision.
    """
    
    def __init__(self, logger: 'AgentLogger'):
        self.client = anthropic.Anthropic()
        self.logger = logger
        self.model = "claude-sonnet-4-20250514"
        
        # Define available tools
        self.tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name == "get_weather":
            # Simulated weather response
            return f"Weather in {tool_input['location']}: 72°F, sunny"
        elif tool_name == "calculate":
            try:
                # WARNING: eval is dangerous! Use a proper math parser in production
                result = eval(tool_input["expression"])
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
    
    def process_request(self, user_input: str) -> str:
        """
        Process a user request with full tracing.
        
        Args:
            user_input: The user's message
        
        Returns:
            The agent's final response
        """
        with self.logger.trace_request(user_input) as trace:
            messages = [{"role": "user", "content": user_input}]
            
            # Initial LLM call
            self.logger.info("Making initial LLM call")
            start_time = time.perf_counter()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.tools,
                messages=messages
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log the LLM call
            self.logger.log_llm_call(
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=duration_ms,
                stop_reason=response.stop_reason
            )
            
            # Log what the model decided to do
            self.logger.info(
                "LLM decision made",
                stop_reason=response.stop_reason,
                content_blocks=len(response.content),
                has_tool_use=any(
                    block.type == "tool_use" for block in response.content
                )
            )
            
            # Process tool calls if any
            while response.stop_reason == "tool_use":
                # Find all tool use blocks
                tool_uses = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]
                
                tool_results = []
                
                for tool_use in tool_uses:
                    # Trace the tool call
                    with self.logger.trace_tool_call(
                        tool_use.name,
                        tool_use.input
                    ) as tool_trace:
                        # Log what tool was selected and why
                        self.logger.debug(
                            "Tool selected",
                            tool_name=tool_use.name,
                            available_tools=[t["name"] for t in self.tools],
                            input_params=list(tool_use.input.keys())
                        )
                        
                        # Execute the tool
                        result = self._execute_tool(tool_use.name, tool_use.input)
                        tool_trace.tool_output = result
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result
                    })
                
                # Continue the conversation with tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
                # Another LLM call
                start_time = time.perf_counter()
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    tools=self.tools,
                    messages=messages
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                self.logger.log_llm_call(
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    duration_ms=duration_ms,
                    stop_reason=response.stop_reason
                )
            
            # Extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            trace.final_output = final_response
            
            return final_response


# Example usage demonstrating the tracing
if __name__ == "__main__":
    # We'll use the AgentLogger from our earlier example
    # For this demo, we create a simple version
    
    import logging
    import sys
    
    # Setup basic logging to see output
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s',
        stream=sys.stdout
    )
    
    print("=" * 60)
    print("Agent Tracing Demonstration")
    print("=" * 60)
    print()
    print("Note: This example requires a valid ANTHROPIC_API_KEY")
    print("The tracing will show every decision the agent makes.")
    print()
```

## Performance Metrics Collection

Tracking performance metrics helps you understand your agent's behavior over time. Let's build a dedicated metrics collector.

```python
"""
Performance metrics collection for AI agents.

Chapter 36: Observability and Logging
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import statistics


@dataclass
class MetricPoint:
    """A single metric measurement."""
    value: float
    timestamp: str
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics for agents.
    
    Supports:
    - Counters: Values that only increase (e.g., total requests)
    - Gauges: Point-in-time values (e.g., active connections)
    - Histograms: Distribution of values (e.g., latencies)
    
    Usage:
        metrics = MetricsCollector()
        
        # Count things
        metrics.increment("requests_total", labels={"endpoint": "/chat"})
        
        # Track distributions
        metrics.observe("latency_ms", 150.5, labels={"operation": "llm_call"})
        
        # Get summaries
        print(metrics.get_histogram_stats("latency_ms"))
    """
    
    def __init__(self):
        self._counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def _labels_key(self, labels: Optional[dict[str, str]] = None) -> str:
        """Convert labels dict to a hashable string key."""
        if not labels:
            return "__default__"
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    # -------------------------------------------------------------------------
    # Counters
    # -------------------------------------------------------------------------
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Increment a counter.
        
        Args:
            name: Metric name
            value: Amount to increment by (default: 1)
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._counters[name][key] += value
    
    def get_counter(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> float:
        """Get the current value of a counter."""
        key = self._labels_key(labels)
        return self._counters[name][key]
    
    # -------------------------------------------------------------------------
    # Gauges
    # -------------------------------------------------------------------------
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Set a gauge to a specific value.
        
        Args:
            name: Metric name
            value: The value to set
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._gauges[name][key] = value
    
    def get_gauge(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> float:
        """Get the current value of a gauge."""
        key = self._labels_key(labels)
        return self._gauges[name][key]
    
    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------
    
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record an observation in a histogram.
        
        Args:
            name: Metric name
            value: The observed value
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._histograms[name][key].append(value)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> dict[str, float]:
        """
        Get statistics for a histogram.
        
        Returns:
            Dictionary with count, sum, mean, min, max, p50, p95, p99
        """
        key = self._labels_key(labels)
        values = self._histograms[name][key]
        
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        
        sorted_values = sorted(values)
        
        def percentile(p: float) -> float:
            """Calculate the p-th percentile."""
            idx = int(len(sorted_values) * p / 100)
            return sorted_values[min(idx, len(sorted_values) - 1)]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
        }
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        
        # Counters
        for name, label_values in self._counters.items():
            result["counters"][name] = dict(label_values)
        
        # Gauges
        for name, label_values in self._gauges.items():
            result["gauges"][name] = dict(label_values)
        
        # Histograms (with stats)
        for name, label_values in self._histograms.items():
            result["histograms"][name] = {}
            for key, values in label_values.items():
                result["histograms"][name][key] = self.get_histogram_stats(name, None)
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class AgentMetrics:
    """
    Pre-defined metrics for AI agent monitoring.
    
    This wraps MetricsCollector with agent-specific metric names
    and helper methods.
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
    
    def record_request(self, success: bool = True) -> None:
        """Record a request."""
        self.collector.increment("agent_requests_total")
        if success:
            self.collector.increment("agent_requests_success")
        else:
            self.collector.increment("agent_requests_failed")
    
    def record_latency(self, duration_ms: float, operation: str = "total") -> None:
        """Record latency for an operation."""
        self.collector.observe(
            "agent_latency_ms",
            duration_ms,
            labels={"operation": operation}
        )
    
    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage."""
        self.collector.increment("agent_tokens_input", input_tokens)
        self.collector.increment("agent_tokens_output", output_tokens)
        self.collector.increment("agent_tokens_total", input_tokens + output_tokens)
    
    def record_tool_call(self, tool_name: str, success: bool = True) -> None:
        """Record a tool call."""
        self.collector.increment(
            "agent_tool_calls_total",
            labels={"tool": tool_name, "status": "success" if success else "error"}
        )
    
    def record_llm_call(self, model: str, duration_ms: float) -> None:
        """Record an LLM API call."""
        self.collector.increment("agent_llm_calls_total", labels={"model": model})
        self.collector.observe(
            "agent_llm_latency_ms",
            duration_ms,
            labels={"model": model}
        )
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of agent metrics."""
        total = self.collector.get_counter("agent_requests_total")
        success = self.collector.get_counter("agent_requests_success")
        
        return {
            "requests": {
                "total": int(total),
                "success": int(success),
                "failed": int(total - success),
                "success_rate": round(success / total * 100, 1) if total > 0 else 0.0,
            },
            "tokens": {
                "input": int(self.collector.get_counter("agent_tokens_input")),
                "output": int(self.collector.get_counter("agent_tokens_output")),
                "total": int(self.collector.get_counter("agent_tokens_total")),
            },
            "latency": self.collector.get_histogram_stats("agent_latency_ms"),
            "llm_latency": self.collector.get_histogram_stats("agent_llm_latency_ms"),
        }


# Example usage
if __name__ == "__main__":
    metrics = AgentMetrics()
    
    # Simulate some agent activity
    import random
    
    for i in range(100):
        # Record a request
        success = random.random() > 0.1  # 90% success rate
        metrics.record_request(success=success)
        
        # Record latency (simulated)
        latency = random.gauss(500, 150)  # Mean 500ms, stddev 150ms
        metrics.record_latency(max(latency, 50), operation="total")
        
        # Record tokens (simulated)
        metrics.record_tokens(
            input_tokens=random.randint(100, 500),
            output_tokens=random.randint(50, 300)
        )
        
        # Record tool calls (simulated)
        if random.random() > 0.5:
            tool = random.choice(["weather", "calculator", "search"])
            metrics.record_tool_call(tool, success=random.random() > 0.05)
        
        # Record LLM call
        llm_latency = random.gauss(400, 100)
        metrics.record_llm_call("claude-sonnet-4-20250514", max(llm_latency, 100))
    
    # Print summary
    import json
    print("Agent Metrics Summary")
    print("=" * 50)
    print(json.dumps(metrics.get_summary(), indent=2))
```

## Log Levels and Filtering

Choosing the right log level is crucial. Too verbose, and you're drowning in noise. Too quiet, and you miss important information. Here's a guide to log level selection for agents:

| Level | Use For | Example |
|-------|---------|---------|
| DEBUG | Detailed internals, tool inputs/outputs | "Tool input: {location: 'NYC'}" |
| INFO | Request lifecycle, major decisions | "Request started", "Tool selected: weather" |
| WARNING | Recoverable issues, retries | "API rate limited, retrying in 2s" |
| ERROR | Failed operations | "Tool execution failed: connection timeout" |
| CRITICAL | System-wide problems | "API key invalid, all requests failing" |

**Production recommendations:**

- **Default level**: INFO — captures request flow without excessive detail
- **Debugging level**: DEBUG — use temporarily when investigating issues
- **High-volume production**: WARNING — only problems and anomalies

```python
"""
Log level configuration examples.

Chapter 36: Observability and Logging
"""

import logging
import os


def configure_logging_from_environment():
    """
    Configure log level from environment variable.
    
    Set LOG_LEVEL environment variable to: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return level


def configure_per_module_logging():
    """
    Configure different log levels for different components.
    
    This is useful when you want verbose logging for one component
    but quiet logging for others.
    """
    # Root logger at INFO
    logging.basicConfig(level=logging.INFO)
    
    # Verbose logging for our agent
    logging.getLogger("agent").setLevel(logging.DEBUG)
    
    # Quiet logging for HTTP library
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Quiet logging for other libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)


class LogLevelFilter(logging.Filter):
    """
    Filter that allows only specific log levels.
    
    Useful for routing different levels to different outputs.
    """
    
    def __init__(self, levels: list[int]):
        super().__init__()
        self.levels = levels
    
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.levels


def configure_split_output():
    """
    Send INFO to stdout, ERROR and above to stderr.
    
    This is useful in containerized environments where stdout and stderr
    are captured separately.
    """
    import sys
    
    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Stdout handler for INFO and below
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(LogLevelFilter([logging.DEBUG, logging.INFO]))
    
    # Stderr handler for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    
    return logger


if __name__ == "__main__":
    # Demo different configurations
    print("=== Environment-based Configuration ===")
    configure_logging_from_environment()
    logging.info("This message uses environment-configured level")
    
    print("\n=== Per-module Configuration ===")
    configure_per_module_logging()
    
    agent_logger = logging.getLogger("agent")
    http_logger = logging.getLogger("httpx")
    
    agent_logger.debug("Agent debug message (visible)")
    agent_logger.info("Agent info message (visible)")
    http_logger.debug("HTTP debug message (hidden)")
    http_logger.warning("HTTP warning message (visible)")
```

## Log Aggregation Patterns

In production, logs from multiple agent instances need to be aggregated for analysis. Here are patterns for common logging backends.

```python
"""
Log aggregation patterns for production agents.

Chapter 36: Observability and Logging
"""

import json
import logging
import socket
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class ProductionLogFormatter(logging.Formatter):
    """
    Production-ready JSON log formatter.
    
    Includes:
    - Timestamp in ISO format
    - Hostname for multi-server deployments
    - Service name for microservices
    - Trace ID correlation
    """
    
    def __init__(
        self,
        service_name: str,
        environment: str = "production",
        extra_fields: Optional[dict[str, Any]] = None
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "hostname": self.hostname,
        }
        
        # Add static extra fields
        log_entry.update(self.extra_fields)
        
        # Add dynamic extra fields from the log record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry)


def configure_for_elasticsearch():
    """
    Configure logging for Elasticsearch/ELK stack.
    
    The log format is compatible with Filebeat and Logstash.
    Index pattern: logs-{service_name}-{date}
    """
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ProductionLogFormatter(
        service_name="ai-agent",
        environment="production",
        extra_fields={
            "version": "1.0.0",
            "team": "ai-platform"
        }
    ))
    
    logger.addHandler(handler)
    return logger


def configure_for_cloudwatch():
    """
    Configure logging for AWS CloudWatch.
    
    CloudWatch expects JSON logs with specific field names.
    """
    class CloudWatchFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "requestId": getattr(record, "request_id", None),
            })
    
    logger = logging.getLogger("agent")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudWatchFormatter())
    logger.addHandler(handler)
    return logger


def configure_for_datadog():
    """
    Configure logging for Datadog.
    
    Datadog expects specific field names for proper parsing.
    """
    class DatadogFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": record.levelname.lower(),
                "message": record.getMessage(),
                "logger": {"name": record.name},
                "dd": {
                    "service": "ai-agent",
                    "env": "production",
                    "version": "1.0.0",
                    "trace_id": getattr(record, "trace_id", None),
                    "span_id": getattr(record, "span_id", None),
                }
            })
    
    logger = logging.getLogger("agent")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DatadogFormatter())
    logger.addHandler(handler)
    return logger


class CorrelationContext:
    """
    Thread-local storage for request correlation.
    
    This allows you to add trace IDs to all log messages
    within a request context.
    """
    
    _context: dict[str, Any] = {}
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls._context[key] = value
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls._context.get(key, default)
    
    @classmethod
    def clear(cls) -> None:
        cls._context.clear()
    
    @classmethod
    def get_all(cls) -> dict[str, Any]:
        return cls._context.copy()


class CorrelatedLogFormatter(logging.Formatter):
    """
    Formatter that automatically includes correlation context.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add correlation context
        log_entry.update(CorrelationContext.get_all())
        
        return json.dumps(log_entry)


# Example usage
if __name__ == "__main__":
    # Set up correlated logging
    logger = logging.getLogger("agent")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CorrelatedLogFormatter())
    logger.addHandler(handler)
    
    # Simulate a request with correlation
    CorrelationContext.set("trace_id", "abc123")
    CorrelationContext.set("user_id", "user_456")
    
    logger.info("Processing request")
    logger.info("Calling weather tool")
    logger.info("Request completed")
    
    CorrelationContext.clear()
```

## Integrating Logging with Our Agent

Now let's put it all together by integrating our logging system with the Agent class from Chapter 33.

```python
"""
Integrating observability with the Agent class.

Chapter 36: Observability and Logging
"""

import os
import time
from typing import Any, Optional
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Import logging utilities (in practice, from your package)
# from agents.utils.logging import AgentLogger, AgentMetrics

# For this example, we include simplified versions inline
import json
import logging
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class ToolCallRecord:
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class RequestTrace:
    trace_id: str
    start_time: str
    end_time: Optional[str] = None
    user_input: Optional[str] = None
    final_output: Optional[str] = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    llm_calls: int = 0
    error: Optional[str] = None


class ObservableAgent:
    """
    An agent with full observability built in.
    
    This demonstrates how to integrate logging and metrics
    into the agentic loop.
    """
    
    def __init__(
        self,
        name: str = "observable_agent",
        model: str = "claude-sonnet-4-20250514",
        log_level: int = logging.INFO
    ):
        self.name = name
        self.model = model
        self.client = anthropic.Anthropic()
        
        # Set up logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self._create_formatter())
            self.logger.addHandler(handler)
        
        # Metrics storage
        self._traces: list[RequestTrace] = []
        self._current_trace: Optional[RequestTrace] = None
        
        # Tool definitions
        self.tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
        
        # Tool implementations
        self._tool_implementations = {
            "get_weather": self._tool_get_weather,
            "calculate": self._tool_calculate,
        }
    
    def _create_formatter(self) -> logging.Formatter:
        """Create a JSON log formatter."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "agent": record.name,
                    "message": record.getMessage(),
                }
                if hasattr(record, "extra"):
                    log_data.update(record.extra)
                return json.dumps(log_data)
        return JsonFormatter()
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log with extra fields."""
        extra = {"extra": kwargs}
        if self._current_trace:
            extra["extra"]["trace_id"] = self._current_trace.trace_id
        self.logger.log(level, message, extra=extra)
    
    def _tool_get_weather(self, location: str) -> str:
        """Simulated weather tool."""
        time.sleep(0.1)  # Simulate API call
        return f"Weather in {location}: 72°F, sunny with light clouds"
    
    def _tool_calculate(self, expression: str) -> str:
        """Calculator tool."""
        try:
            # WARNING: eval is dangerous! Use a proper math parser in production
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool with logging."""
        start_time = time.perf_counter()
        
        self._log(logging.DEBUG, "Tool execution started",
                  tool_name=tool_name, tool_input=tool_input)
        
        record = ToolCallRecord(tool_name=tool_name, tool_input=tool_input)
        
        try:
            if tool_name in self._tool_implementations:
                result = self._tool_implementations[tool_name](**tool_input)
                record.tool_output = result
            else:
                result = f"Unknown tool: {tool_name}"
                record.error = "Unknown tool"
        except Exception as e:
            result = f"Error: {str(e)}"
            record.error = str(e)
            self._log(logging.WARNING, "Tool execution failed",
                      tool_name=tool_name, error=str(e))
        
        record.duration_ms = (time.perf_counter() - start_time) * 1000
        
        if self._current_trace:
            self._current_trace.tool_calls.append(record)
        
        self._log(logging.DEBUG, "Tool execution completed",
                  tool_name=tool_name, duration_ms=round(record.duration_ms, 2))
        
        return result
    
    def run(self, user_input: str) -> str:
        """
        Process a user request with full observability.
        
        Args:
            user_input: The user's message
        
        Returns:
            The agent's response
        """
        # Start tracing
        trace = RequestTrace(
            trace_id=str(uuid.uuid4())[:8],
            start_time=datetime.now(timezone.utc).isoformat(),
            user_input=user_input
        )
        self._current_trace = trace
        request_start = time.perf_counter()
        
        self._log(logging.INFO, "Request started",
                  user_input=user_input[:100] + "..." if len(user_input) > 100 else user_input)
        
        try:
            messages = [{"role": "user", "content": user_input}]
            
            # Agentic loop
            while True:
                llm_start = time.perf_counter()
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    tools=self.tools,
                    messages=messages
                )
                
                llm_duration = (time.perf_counter() - llm_start) * 1000
                trace.llm_calls += 1
                trace.total_input_tokens += response.usage.input_tokens
                trace.total_output_tokens += response.usage.output_tokens
                
                self._log(logging.DEBUG, "LLM call completed",
                          duration_ms=round(llm_duration, 2),
                          input_tokens=response.usage.input_tokens,
                          output_tokens=response.usage.output_tokens,
                          stop_reason=response.stop_reason)
                
                # Check if we're done
                if response.stop_reason == "end_turn":
                    # Extract text response
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    trace.final_output = final_text
                    break
                
                # Handle tool use
                if response.stop_reason == "tool_use":
                    tool_results = []
                    
                    for block in response.content:
                        if block.type == "tool_use":
                            self._log(logging.INFO, "Tool selected",
                                      tool_name=block.name)
                            
                            result = self._execute_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                    
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # Unexpected stop reason
                    self._log(logging.WARNING, "Unexpected stop reason",
                              stop_reason=response.stop_reason)
                    break
            
            return trace.final_output or ""
            
        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)}"
            self._log(logging.ERROR, "Request failed", error=trace.error)
            raise
            
        finally:
            trace.end_time = datetime.now(timezone.utc).isoformat()
            trace.total_duration_ms = (time.perf_counter() - request_start) * 1000
            
            self._log(logging.INFO, "Request completed",
                      duration_ms=round(trace.total_duration_ms, 2),
                      llm_calls=trace.llm_calls,
                      tool_calls=len(trace.tool_calls),
                      total_tokens=trace.total_input_tokens + trace.total_output_tokens)
            
            self._traces.append(trace)
            self._current_trace = None
    
    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics from all traces."""
        if not self._traces:
            return {"message": "No traces yet"}
        
        total = len(self._traces)
        successful = sum(1 for t in self._traces if t.error is None)
        durations = [t.total_duration_ms for t in self._traces if t.total_duration_ms]
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": total - successful,
            "success_rate_pct": round(successful / total * 100, 1),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
            "total_llm_calls": sum(t.llm_calls for t in self._traces),
            "total_tool_calls": sum(len(t.tool_calls) for t in self._traces),
            "total_tokens": sum(t.total_input_tokens + t.total_output_tokens for t in self._traces),
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Observable Agent Demo")
    print("=" * 60)
    print()
    
    # Create agent with DEBUG level to see all logs
    agent = ObservableAgent(log_level=logging.DEBUG)
    
    # Process a request
    print("\n--- Processing Request ---\n")
    response = agent.run("What's the weather in San Francisco?")
    
    print("\n--- Agent Response ---")
    print(response)
    
    print("\n--- Metrics Summary ---")
    print(json.dumps(agent.get_metrics(), indent=2))
```

## Common Pitfalls

**1. Logging sensitive data**

Never log API keys, user passwords, or personally identifiable information (PII). Create a sanitization layer:

```python
def sanitize_for_logging(data: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields before logging."""
    sensitive_keys = {"password", "api_key", "token", "secret", "ssn", "credit_card"}
    
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if k.lower() in sensitive_keys else _sanitize(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        return obj
    
    return _sanitize(data)
```

**2. Performance impact from excessive logging**

Logging has overhead. In hot paths:

- Use lazy evaluation: `logger.debug("Result: %s", expensive_operation())` instead of `logger.debug(f"Result: {expensive_operation()}")`
- Check log level before expensive operations: `if logger.isEnabledFor(logging.DEBUG):`
- Use async logging for high-throughput systems

**3. Missing correlation IDs**

Without correlation IDs, you can't trace a request across multiple log lines or services. Always include a trace ID and propagate it across all operations within a request.

## Practical Exercise

**Task:** Build a logging dashboard that displays real-time agent metrics

**Requirements:**

1. Create an HTML file that displays metrics from the `AgentMetrics` class
2. Show: total requests, success rate, average latency, token usage
3. Include a table of recent tool calls
4. Auto-refresh every 5 seconds

**Hints:**

- Use the `AgentLogger.get_metrics_summary()` and `get_recent_traces()` methods
- You can use a simple Flask or FastAPI endpoint to serve the data
- Use JavaScript fetch() to poll for updates

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Structured logging** (JSON format) enables powerful querying and analysis of agent behavior
- **Request tracing** with unique IDs lets you follow a single request through multiple LLM calls and tool invocations
- **Metrics collection** (latency, tokens, success rates) reveals patterns and problems
- **Log levels** should be configurable—use DEBUG for development, INFO or WARNING for production
- **Correlation context** ensures all logs from a single request can be linked together
- Production logging systems should integrate with **log aggregation tools** like Elasticsearch, CloudWatch, or Datadog

## What's Next

Now that you can see what your agent is doing, the next chapter covers **Debugging Agents**. You'll learn systematic approaches to finding and fixing problems, from conversation flow issues to infinite loops. The observability tools you built here will be essential for effective debugging.

