"""
Agent session replay system.

Chapter 37: Debugging Agents

This module provides a complete system for recording agent sessions
and replaying them later for debugging. It captures all LLM requests,
responses, tool calls, and results.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class RecordedEvent:
    """A single recorded event in a session."""
    timestamp: str
    event_type: str  # "llm_request", "llm_response", "tool_call", "tool_result"
    data: dict[str, Any]
    sequence_number: int


@dataclass
class RecordedSession:
    """A complete recorded agent session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    model: str = ""
    system_prompt: Optional[str] = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    events: list[RecordedEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "events": [asdict(e) for e in self.events],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedSession":
        events = [
            RecordedEvent(**e) for e in data.get("events", [])
        ]
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt"),
            tools=data.get("tools", []),
            events=events,
            metadata=data.get("metadata", {})
        )


class SessionRecorder:
    """
    Records agent sessions for later replay.
    
    Captures:
    - All LLM requests and responses
    - Tool calls and results
    - Timing information
    - System prompts and tool definitions
    
    Usage:
        recorder = SessionRecorder()
        recorder.start_session("session-123", model="claude-sonnet-4-20250514")
        
        # Record events as they happen
        recorder.record_llm_request(messages)
        recorder.record_llm_response(response)
        recorder.record_tool_call(tool_name, tool_input)
        recorder.record_tool_result(tool_name, result)
        
        # Save for later
        recorder.end_session()
        recorder.save("session-123.json")
    """
    
    def __init__(self):
        self.current_session: Optional[RecordedSession] = None
        self.sequence_counter = 0
    
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def start_session(
        self,
        session_id: str,
        model: str = "",
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Start recording a new session."""
        self.current_session = RecordedSession(
            session_id=session_id,
            started_at=self._now(),
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
            metadata=metadata or {}
        )
        self.sequence_counter = 0
    
    def end_session(self) -> None:
        """End the current session."""
        if self.current_session:
            self.current_session.ended_at = self._now()
    
    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.sequence_counter += 1
        event = RecordedEvent(
            timestamp=self._now(),
            event_type=event_type,
            data=data,
            sequence_number=self.sequence_counter
        )
        self.current_session.events.append(event)
    
    def record_llm_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> None:
        """Record an LLM request."""
        self._record_event("llm_request", {
            "messages": messages,
            **kwargs
        })
    
    def record_llm_response(
        self,
        content: list[dict[str, Any]],
        stop_reason: str,
        usage: dict[str, int]
    ) -> None:
        """Record an LLM response."""
        # Convert content blocks to serializable format
        serializable_content = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    serializable_content.append({
                        "type": "text",
                        "text": block.text
                    })
                elif block.type == "tool_use":
                    serializable_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
            else:
                serializable_content.append(block)
        
        self._record_event("llm_response", {
            "content": serializable_content,
            "stop_reason": stop_reason,
            "usage": usage
        })
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> None:
        """Record a tool call."""
        self._record_event("tool_call", {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": tool_use_id
        })
    
    def record_tool_result(
        self,
        tool_name: str,
        tool_use_id: str,
        result: Any,
        duration_ms: float
    ) -> None:
        """Record a tool result."""
        self._record_event("tool_result", {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "result": str(result),
            "duration_ms": duration_ms
        })
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        **context: Any
    ) -> None:
        """Record an error event."""
        self._record_event("error", {
            "error_type": error_type,
            "error_message": error_message,
            **context
        })
    
    def save(self, filepath: str) -> None:
        """Save the session to a file."""
        if not self.current_session:
            raise RuntimeError("No session to save")
        
        with open(filepath, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)
    
    def get_session(self) -> Optional[RecordedSession]:
        """Get the current session."""
        return self.current_session


class SessionPlayer:
    """
    Replays recorded agent sessions.
    
    Modes:
    - Step-by-step: Pause after each event
    - Continuous: Play all events
    - Analysis: Extract insights without replay
    
    Usage:
        player = SessionPlayer()
        session = player.load("session-123.json")
        
        # Step through
        for event in player.step():
            print(event)
            input("Press Enter to continue...")
        
        # Or analyze
        analysis = player.analyze()
    """
    
    def __init__(self):
        self.session: Optional[RecordedSession] = None
        self.current_index = 0
    
    def load(self, filepath: str) -> RecordedSession:
        """Load a recorded session."""
        with open(filepath) as f:
            data = json.load(f)
        self.session = RecordedSession.from_dict(data)
        self.current_index = 0
        return self.session
    
    def load_from_dict(self, data: dict[str, Any]) -> RecordedSession:
        """Load a session from a dictionary."""
        self.session = RecordedSession.from_dict(data)
        self.current_index = 0
        return self.session
    
    def reset(self) -> None:
        """Reset to the beginning of the session."""
        self.current_index = 0
    
    def step(self) -> Optional[RecordedEvent]:
        """Get the next event."""
        if not self.session or self.current_index >= len(self.session.events):
            return None
        
        event = self.session.events[self.current_index]
        self.current_index += 1
        return event
    
    def step_all(self) -> list[RecordedEvent]:
        """Get all remaining events."""
        if not self.session:
            return []
        
        events = self.session.events[self.current_index:]
        self.current_index = len(self.session.events)
        return events
    
    def analyze(self) -> dict[str, Any]:
        """Analyze the recorded session."""
        if not self.session:
            return {"error": "No session loaded"}
        
        events = self.session.events
        
        # Count event types
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        # Find tool calls
        tool_calls = [e for e in events if e.event_type == "tool_call"]
        tool_results = [e for e in events if e.event_type == "tool_result"]
        errors = [e for e in events if e.event_type == "error"]
        
        # Calculate timing
        tool_durations = []
        for result in tool_results:
            duration = result.data.get("duration_ms", 0)
            tool_durations.append(duration)
        
        # Find potential issues
        issues = []
        
        # Check for repeated tool calls
        tool_call_hashes = []
        for tc in tool_calls:
            hash_str = f"{tc.data['tool_name']}:{tc.data['tool_input']}"
            if hash_str in tool_call_hashes:
                issues.append(f"Duplicate tool call: {tc.data['tool_name']}")
            tool_call_hashes.append(hash_str)
        
        # Check for missing tool results
        tool_call_ids = {tc.data["tool_use_id"] for tc in tool_calls}
        tool_result_ids = {tr.data["tool_use_id"] for tr in tool_results}
        missing_results = tool_call_ids - tool_result_ids
        if missing_results:
            issues.append(f"Missing tool results for: {missing_results}")
        
        # Check for errors
        if errors:
            for error in errors:
                issues.append(f"Error: {error.data.get('error_type', 'unknown')} - {error.data.get('error_message', 'no message')}")
        
        return {
            "session_id": self.session.session_id,
            "duration_seconds": self._calculate_duration(),
            "event_counts": event_counts,
            "total_events": len(events),
            "tool_stats": {
                "total_calls": len(tool_calls),
                "unique_tools": list(set(tc.data["tool_name"] for tc in tool_calls)),
                "avg_duration_ms": sum(tool_durations) / len(tool_durations) if tool_durations else 0,
                "max_duration_ms": max(tool_durations) if tool_durations else 0,
            },
            "issues_found": issues,
        }
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in seconds."""
        if not self.session or not self.session.events:
            return 0
        
        first = datetime.fromisoformat(self.session.events[0].timestamp)
        last = datetime.fromisoformat(self.session.events[-1].timestamp)
        return (last - first).total_seconds()
    
    def find_events(
        self,
        event_type: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> list[RecordedEvent]:
        """Find events matching criteria."""
        if not self.session:
            return []
        
        events = self.session.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if tool_name:
            events = [
                e for e in events 
                if e.data.get("tool_name") == tool_name
            ]
        
        return events
    
    def print_timeline(self) -> None:
        """Print a visual timeline of the session."""
        if not self.session:
            print("No session loaded")
            return
        
        print("\n" + "=" * 60)
        print(f"SESSION TIMELINE: {self.session.session_id}")
        print("=" * 60)
        print(f"\nModel: {self.session.model}")
        print(f"Started: {self.session.started_at}")
        print(f"Ended: {self.session.ended_at or 'In progress'}")
        
        if self.session.system_prompt:
            print(f"\nSystem Prompt: {self.session.system_prompt[:100]}...")
        
        print("\nEvents:")
        
        for event in self.session.events:
            icon = {
                "llm_request": "📤",
                "llm_response": "📥",
                "tool_call": "🔧",
                "tool_result": "✅",
                "error": "❌",
            }.get(event.event_type, "•")
            
            # Format the event description
            if event.event_type == "llm_request":
                desc = f"Request with {len(event.data.get('messages', []))} messages"
            elif event.event_type == "llm_response":
                desc = f"Response (stop: {event.data.get('stop_reason', 'unknown')})"
            elif event.event_type == "tool_call":
                desc = f"Call {event.data.get('tool_name', 'unknown')}"
            elif event.event_type == "tool_result":
                desc = f"Result from {event.data.get('tool_name', 'unknown')} ({event.data.get('duration_ms', 0):.0f}ms)"
            elif event.event_type == "error":
                desc = f"Error: {event.data.get('error_type', 'unknown')}"
            else:
                desc = event.event_type
            
            print(f"\n{icon} [{event.sequence_number:02d}] {desc}")
            
            # Show relevant details
            if event.event_type == "tool_call":
                input_str = json.dumps(event.data.get('tool_input', {}))
                print(f"   Input: {input_str[:60]}{'...' if len(input_str) > 60 else ''}")
            elif event.event_type == "tool_result":
                result = str(event.data.get('result', ''))[:60]
                print(f"   Result: {result}{'...' if len(str(event.data.get('result', ''))) > 60 else ''}")
            elif event.event_type == "error":
                print(f"   Message: {event.data.get('error_message', 'No message')[:60]}")
        
        print("\n" + "=" * 60)
    
    def export_as_test_case(self, filepath: str) -> None:
        """Export the session as a reproducible test case."""
        if not self.session:
            print("No session loaded")
            return
        
        test_case = {
            "description": f"Reproduction case from session {self.session.session_id}",
            "model": self.session.model,
            "system_prompt": self.session.system_prompt,
            "tools": self.session.tools,
            "initial_messages": [],
            "expected_tool_sequence": [],
        }
        
        # Extract initial user message
        for event in self.session.events:
            if event.event_type == "llm_request":
                messages = event.data.get("messages", [])
                if messages:
                    test_case["initial_messages"] = messages
                    break
        
        # Extract tool sequence
        for event in self.session.events:
            if event.event_type == "tool_call":
                test_case["expected_tool_sequence"].append({
                    "tool": event.data.get("tool_name"),
                    "input": event.data.get("tool_input")
                })
        
        with open(filepath, "w") as f:
            json.dump(test_case, f, indent=2)
        
        print(f"✅ Test case exported to {filepath}")


# Example usage demonstrating recording and replay
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION RECORDING AND REPLAY DEMONSTRATION")
    print("=" * 60)
    
    # Record a session
    print("\n" + "-" * 60)
    print("RECORDING A SESSION")
    print("-" * 60)
    
    recorder = SessionRecorder()
    recorder.start_session(
        session_id="demo-session-001",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant with access to weather and calculator tools.",
        tools=[
            {"name": "weather", "description": "Get weather for a location"},
            {"name": "calculator", "description": "Perform calculations"}
        ],
        metadata={"user_id": "demo-user", "environment": "test"}
    )
    
    # Simulate some events
    print("\nRecording events...")
    
    recorder.record_llm_request(
        messages=[{"role": "user", "content": "What's the weather in Paris and what's 25 * 4?"}]
    )
    
    recorder._record_event("llm_response", {
        "content": [
            {"type": "text", "text": "I'll help you with both questions."},
            {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "Paris"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 30}
    })
    
    recorder.record_tool_call("weather", {"city": "Paris"}, "tool_1")
    time.sleep(0.1)  # Simulate tool execution time
    recorder.record_tool_result("weather", "tool_1", "Paris: 18°C, cloudy", 150.5)
    
    recorder.record_llm_request(
        messages=[
            {"role": "user", "content": "What's the weather in Paris and what's 25 * 4?"},
            {"role": "assistant", "content": "I'll help you with both."},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "Paris: 18°C, cloudy"}]}
        ]
    )
    
    recorder._record_event("llm_response", {
        "content": [
            {"type": "tool_use", "id": "tool_2", "name": "calculator", "input": {"expression": "25 * 4"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 80, "output_tokens": 25}
    })
    
    recorder.record_tool_call("calculator", {"expression": "25 * 4"}, "tool_2")
    time.sleep(0.05)
    recorder.record_tool_result("calculator", "tool_2", "100", 50.0)
    
    recorder._record_event("llm_response", {
        "content": [{"type": "text", "text": "The weather in Paris is 18°C and cloudy, and 25 × 4 = 100."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 30}
    })
    
    recorder.end_session()
    
    # Save the session
    session_file = "/tmp/demo_session.json"
    recorder.save(session_file)
    print(f"✅ Session saved to {session_file}")
    
    # Replay and analyze
    print("\n" + "-" * 60)
    print("REPLAYING AND ANALYZING SESSION")
    print("-" * 60)
    
    player = SessionPlayer()
    player.load(session_file)
    
    # Print timeline
    player.print_timeline()
    
    # Analyze
    print("\n" + "-" * 60)
    print("SESSION ANALYSIS")
    print("-" * 60)
    analysis = player.analyze()
    print(json.dumps(analysis, indent=2))
    
    # Export as test case
    print("\n" + "-" * 60)
    print("EXPORTING AS TEST CASE")
    print("-" * 60)
    player.export_as_test_case("/tmp/test_case.json")
    
    # Step through events
    print("\n" + "-" * 60)
    print("STEPPING THROUGH EVENTS")
    print("-" * 60)
    
    player.reset()
    print("\nEvents step by step:")
    while True:
        event = player.step()
        if event is None:
            break
        print(f"  [{event.sequence_number}] {event.event_type}")
    
    print("\n✅ Session replay demonstration complete!")
