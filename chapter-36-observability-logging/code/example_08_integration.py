"""
Integrating observability with a complete Agent class.

Chapter 36: Observability and Logging

This module demonstrates how to integrate all observability components
(logging, tracing, metrics) into a working AI agent.

Note: This example makes actual API calls to Claude.
"""

import os
import time
import json
import logging
import sys
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

import anthropic


# =============================================================================
# Data Classes for Tracing
# =============================================================================

@dataclass
class ToolCallRecord:
    """Record of a single tool call."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    stop_reason: Optional[str] = None


@dataclass
class RequestTrace:
    """Complete trace of a single agent request."""
    trace_id: str
    start_time: str
    end_time: Optional[str] = None
    user_input: Optional[str] = None
    final_output: Optional[str] = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.llm_calls)


# =============================================================================
# JSON Log Formatter
# =============================================================================

class AgentLogFormatter(logging.Formatter):
    """JSON formatter with trace ID support."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                if value is not None:
                    log_entry[key] = value
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


# =============================================================================
# Observable Agent
# =============================================================================

class ObservableAgent:
    """
    An AI agent with comprehensive built-in observability.
    
    Features:
    - Structured JSON logging
    - Request tracing with unique IDs
    - Tool call tracking with timing
    - LLM call metrics
    - Aggregated statistics
    
    Usage:
        agent = ObservableAgent()
        response = agent.run("What's the weather in NYC?")
        print(agent.get_metrics())
    """
    
    def __init__(
        self,
        name: str = "observable_agent",
        model: str = "claude-sonnet-4-20250514",
        log_level: int = logging.INFO,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the observable agent.
        
        Args:
            name: Agent name (used for logging)
            model: Claude model to use
            log_level: Minimum log level
            system_prompt: Optional system prompt for the agent
        """
        self.name = name
        self.model = model
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        
        # Set up logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.propagate = False
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AgentLogFormatter())
        self.logger.addHandler(handler)
        
        # Tracing state
        self._traces: list[RequestTrace] = []
        self._current_trace: Optional[RequestTrace] = None
        
        # Define tools
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
            },
            {
                "name": "get_time",
                "description": "Get the current date and time",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log with structured extra fields."""
        extra = {"extra": kwargs}
        if self._current_trace:
            extra["extra"]["trace_id"] = self._current_trace.trace_id
        self.logger.log(level, message, extra=extra)
    
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name == "get_weather":
            location = tool_input.get("location", "Unknown")
            # Simulated weather (in production, call a real API)
            return f"Weather in {location}: 72°F, sunny with light clouds"
        
        elif tool_name == "calculate":
            expression = tool_input.get("expression", "")
            try:
                # WARNING: eval is dangerous! Use a safe math parser in production
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        elif tool_name == "get_time":
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    @contextmanager
    def _trace_tool(self, tool_name: str, tool_input: dict[str, Any]):
        """Context manager for tracing tool calls."""
        start = time.perf_counter()
        record = ToolCallRecord(tool_name=tool_name, tool_input=tool_input)
        
        self._log(logging.DEBUG, "Tool execution started",
                  tool_name=tool_name, tool_input=tool_input)
        
        try:
            yield record
        except Exception as e:
            record.error = str(e)
            raise
        finally:
            record.duration_ms = (time.perf_counter() - start) * 1000
            if self._current_trace:
                self._current_trace.tool_calls.append(record)
            
            self._log(logging.DEBUG, "Tool execution completed",
                      tool_name=tool_name,
                      duration_ms=round(record.duration_ms, 2),
                      success=record.error is None)
    
    def run(self, user_input: str) -> str:
        """
        Process a user request with full observability.
        
        Args:
            user_input: The user's message
        
        Returns:
            The agent's response
        """
        # Initialize trace
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
            iteration = 0
            max_iterations = 10
            
            while iteration < max_iterations:
                iteration += 1
                
                # Make LLM call
                llm_start = time.perf_counter()
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=messages
                )
                
                llm_duration = (time.perf_counter() - llm_start) * 1000
                
                # Record LLM call
                llm_record = LLMCallRecord(
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    duration_ms=llm_duration,
                    stop_reason=response.stop_reason
                )
                trace.llm_calls.append(llm_record)
                
                self._log(logging.DEBUG, "LLM call completed",
                          iteration=iteration,
                          duration_ms=round(llm_duration, 2),
                          input_tokens=response.usage.input_tokens,
                          output_tokens=response.usage.output_tokens,
                          stop_reason=response.stop_reason)
                
                # Check if done
                if response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    trace.final_output = final_text
                    return final_text
                
                # Handle tool use
                if response.stop_reason == "tool_use":
                    tool_results = []
                    
                    for block in response.content:
                        if block.type == "tool_use":
                            self._log(logging.INFO, "Tool selected",
                                      tool_name=block.name)
                            
                            with self._trace_tool(block.name, block.input) as tool_record:
                                result = self._execute_tool(block.name, block.input)
                                tool_record.tool_output = result
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                    
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                else:
                    self._log(logging.WARNING, "Unexpected stop reason",
                              stop_reason=response.stop_reason)
                    break
            
            return "Max iterations reached"
            
        except Exception as e:
            trace.error = f"{type(e).__name__}: {str(e)}"
            self._log(logging.ERROR, "Request failed", error=trace.error)
            raise
            
        finally:
            trace.end_time = datetime.now(timezone.utc).isoformat()
            trace.total_duration_ms = (time.perf_counter() - request_start) * 1000
            
            self._log(logging.INFO, "Request completed",
                      duration_ms=round(trace.total_duration_ms, 2),
                      llm_calls=len(trace.llm_calls),
                      tool_calls=len(trace.tool_calls),
                      total_tokens=trace.total_tokens)
            
            self._traces.append(trace)
            self._current_trace = None
    
    def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics from all requests."""
        if not self._traces:
            return {"message": "No requests processed yet"}
        
        total = len(self._traces)
        successful = sum(1 for t in self._traces if t.error is None)
        durations = [t.total_duration_ms for t in self._traces if t.total_duration_ms]
        
        # Tool usage breakdown
        tool_usage: dict[str, int] = {}
        for trace in self._traces:
            for tc in trace.tool_calls:
                tool_usage[tc.tool_name] = tool_usage.get(tc.tool_name, 0) + 1
        
        return {
            "requests": {
                "total": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate_pct": round(successful / total * 100, 1) if total else 0
            },
            "latency_ms": {
                "mean": round(sum(durations) / len(durations), 2) if durations else 0,
                "min": round(min(durations), 2) if durations else 0,
                "max": round(max(durations), 2) if durations else 0
            },
            "tokens": {
                "total": sum(t.total_tokens for t in self._traces),
                "per_request_avg": round(sum(t.total_tokens for t in self._traces) / total, 1) if total else 0
            },
            "llm_calls": {
                "total": sum(len(t.llm_calls) for t in self._traces),
                "per_request_avg": round(sum(len(t.llm_calls) for t in self._traces) / total, 2) if total else 0
            },
            "tool_calls": {
                "total": sum(len(t.tool_calls) for t in self._traces),
                "by_tool": tool_usage
            }
        }
    
    def get_recent_traces(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the N most recent request traces."""
        traces = []
        for t in self._traces[-n:]:
            traces.append({
                "trace_id": t.trace_id,
                "start_time": t.start_time,
                "duration_ms": t.total_duration_ms,
                "user_input": t.user_input[:50] + "..." if t.user_input and len(t.user_input) > 50 else t.user_input,
                "llm_calls": len(t.llm_calls),
                "tool_calls": [tc.tool_name for tc in t.tool_calls],
                "tokens": t.total_tokens,
                "error": t.error
            })
        return traces


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Observable Agent Demo")
    print("=" * 70)
    print()
    print("This demo shows a fully observable AI agent in action.")
    print("Watch the structured logs as the agent processes requests.")
    print()
    
    # Create agent with DEBUG level to see all logs
    agent = ObservableAgent(
        name="demo_agent",
        log_level=logging.DEBUG,
        system_prompt="You are a helpful assistant with access to weather, calculator, and time tools."
    )
    
    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What is 42 * 17?",
        "What time is it and what's 100 divided by 4?",
    ]
    
    print("-" * 70)
    print("Processing requests...")
    print("-" * 70)
    print()
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"User: {query}")
        print(f"{'='*50}\n")
        
        try:
            response = agent.run(query)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")
    
    # Show metrics
    print("\n" + "=" * 70)
    print("Aggregated Metrics")
    print("=" * 70)
    print(json.dumps(agent.get_metrics(), indent=2))
    
    # Show recent traces
    print("\n" + "=" * 70)
    print("Recent Traces")
    print("=" * 70)
    print(json.dumps(agent.get_recent_traces(), indent=2))
