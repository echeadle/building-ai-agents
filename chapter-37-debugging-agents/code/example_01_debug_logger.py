"""
Enhanced debug logger for AI agents.

Chapter 37: Debugging Agents

This module provides a comprehensive debug logging system specifically
designed for AI agents. It captures step-by-step events, supports
structured data, and makes it easy to export and analyze agent behavior.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DebugEvent:
    """A single debug event with full context."""
    timestamp: str
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    step_number: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DebugLogger:
    """
    A logger specifically designed for debugging AI agents.
    
    Features:
    - Step-by-step event tracking
    - Full conversation history capture
    - Tool call/result pairing
    - Easy export for analysis
    
    Usage:
        debug = DebugLogger()
        debug.start_trace("user-request-123")
        debug.log_event("user_input", "What's the weather?")
        debug.log_event("tool_call", "Calling weather API", tool="weather", params={...})
        debug.end_trace()
        debug.export("debug_session.json")
    """
    
    def __init__(self, name: str = "agent_debug", verbose: bool = True):
        """
        Initialize the debug logger.
        
        Args:
            name: Logger name
            verbose: If True, print events to console in real-time
        """
        self.name = name
        self.verbose = verbose
        self.events: list[DebugEvent] = []
        self.current_trace_id: Optional[str] = None
        self.step_counter = 0
        
        # Set up Python logger for verbose output
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def _now(self) -> str:
        """Get current timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    def start_trace(self, trace_id: str) -> None:
        """Start a new debug trace."""
        self.current_trace_id = trace_id
        self.step_counter = 0
        self.log_event("trace_start", f"Starting trace: {trace_id}")
    
    def end_trace(self) -> None:
        """End the current trace."""
        self.log_event("trace_end", f"Ending trace: {self.current_trace_id}")
        self.current_trace_id = None
    
    def log_event(
        self,
        event_type: str,
        message: str,
        **data: Any
    ) -> None:
        """
        Log a debug event.
        
        Args:
            event_type: Category of event (e.g., "tool_call", "llm_response")
            message: Human-readable description
            **data: Additional structured data
        """
        self.step_counter += 1
        
        event = DebugEvent(
            timestamp=self._now(),
            event_type=event_type,
            message=message,
            data=data,
            trace_id=self.current_trace_id,
            step_number=self.step_counter
        )
        
        self.events.append(event)
        
        if self.verbose:
            self.logger.debug(
                f"[Step {self.step_counter}] {event_type}: {message}"
            )
            if data:
                for key, value in data.items():
                    value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    self.logger.debug(f"  {key}: {value_str}")
    
    def log_user_input(self, content: str) -> None:
        """Log user input."""
        self.log_event("user_input", "User message received", content=content)
    
    def log_llm_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model: str = ""
    ) -> None:
        """Log an LLM API request."""
        self.log_event(
            "llm_request",
            f"Sending request to {model}",
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
            last_message_role=messages[-1]["role"] if messages else None,
            model=model
        )
    
    def log_llm_response(
        self,
        stop_reason: str,
        content_blocks: int,
        has_tool_use: bool,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Log an LLM response."""
        self.log_event(
            "llm_response",
            f"Received response (stop: {stop_reason})",
            stop_reason=stop_reason,
            content_blocks=content_blocks,
            has_tool_use=has_tool_use,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> None:
        """Log a tool call request."""
        self.log_event(
            "tool_call",
            f"Calling tool: {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id
        )
    
    def log_tool_result(
        self,
        tool_name: str,
        tool_use_id: str,
        result: Any,
        success: bool,
        duration_ms: float
    ) -> None:
        """Log a tool execution result."""
        self.log_event(
            "tool_result",
            f"Tool {tool_name} {'succeeded' if success else 'failed'}",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            result=str(result)[:500],  # Truncate long results
            success=success,
            duration_ms=round(duration_ms, 2)
        )
    
    def log_error(self, error_type: str, message: str, **context: Any) -> None:
        """Log an error."""
        self.log_event(
            "error",
            f"{error_type}: {message}",
            error_type=error_type,
            **context
        )
    
    def log_warning(self, message: str, **context: Any) -> None:
        """Log a warning."""
        self.log_event("warning", message, **context)
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> list[DebugEvent]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type
            trace_id: Filter by trace ID
        
        Returns:
            List of matching events
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if trace_id:
            events = [e for e in events if e.trace_id == trace_id]
        
        return events
    
    def get_trace_summary(self, trace_id: Optional[str] = None) -> dict[str, Any]:
        """Get a summary of a trace."""
        trace_id = trace_id or self.current_trace_id
        events = self.get_events(trace_id=trace_id)
        
        if not events:
            return {"error": "No events found for trace"}
        
        tool_calls = [e for e in events if e.event_type == "tool_call"]
        tool_results = [e for e in events if e.event_type == "tool_result"]
        errors = [e for e in events if e.event_type == "error"]
        llm_responses = [e for e in events if e.event_type == "llm_response"]
        
        return {
            "trace_id": trace_id,
            "total_events": len(events),
            "total_steps": events[-1].step_number if events else 0,
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_responses),
            "errors": len(errors),
            "tools_used": list(set(e.data.get("tool_name", "") for e in tool_calls)),
            "error_types": [e.data.get("error_type", "unknown") for e in errors],
        }
    
    def export(self, filepath: str) -> None:
        """Export all events to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                [e.to_dict() for e in self.events],
                f,
                indent=2
            )
        if self.verbose:
            self.logger.info(f"Exported {len(self.events)} events to {filepath}")
    
    def clear(self) -> None:
        """Clear all events."""
        self.events = []
        self.step_counter = 0
        self.current_trace_id = None
    
    def print_trace(self, trace_id: Optional[str] = None) -> None:
        """Print a human-readable trace summary."""
        events = self.get_events(trace_id=trace_id or self.current_trace_id)
        
        print("\n" + "=" * 60)
        print("DEBUG TRACE")
        print("=" * 60)
        
        for event in events:
            print(f"\n[Step {event.step_number}] {event.event_type.upper()}")
            print(f"  {event.message}")
            
            if event.data:
                for key, value in event.data.items():
                    value_str = str(value)
                    if len(value_str) > 80:
                        value_str = value_str[:80] + "..."
                    print(f"  • {key}: {value_str}")
        
        print("\n" + "=" * 60)


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("DEBUG LOGGER DEMONSTRATION")
    print("=" * 60)
    
    debug = DebugLogger(verbose=True)
    
    # Simulate a debugging session
    debug.start_trace("test-123")
    
    debug.log_user_input("What's the weather in Paris?")
    
    debug.log_llm_request(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{"name": "weather"}],
        model="claude-sonnet-4-20250514"
    )
    
    debug.log_llm_response(
        stop_reason="tool_use",
        content_blocks=2,
        has_tool_use=True,
        input_tokens=150,
        output_tokens=50
    )
    
    debug.log_tool_call(
        tool_name="weather",
        tool_input={"location": "Paris, France"},
        tool_use_id="tool_abc123"
    )
    
    debug.log_tool_result(
        tool_name="weather",
        tool_use_id="tool_abc123",
        result={"temp": 18, "condition": "cloudy"},
        success=True,
        duration_ms=245.5
    )
    
    debug.end_trace()
    
    # Print summary
    print("\n" + "-" * 60)
    print("TRACE SUMMARY")
    print("-" * 60)
    print(json.dumps(debug.get_trace_summary("test-123"), indent=2))
    
    # Export to file
    debug.export("/tmp/debug_events.json")
    print(f"\n✅ Events exported to /tmp/debug_events.json")
