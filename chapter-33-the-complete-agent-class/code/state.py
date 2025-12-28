"""
Agent state management.

Chapter 33: The Complete Agent Class

This module provides comprehensive state management for agents,
including conversation history, working memory, and persistence.
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Message:
    """
    A single message in the conversation.
    
    Attributes:
        role: The role (user, assistant, or system)
        content: Message content (string or list of content blocks)
        timestamp: When the message was created
    """
    role: str  # "user", "assistant", or "system"
    content: Any  # str or list of content blocks
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_api_format(self) -> dict:
        """Convert to format expected by Claude API."""
        return {"role": self.role, "content": self.content}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass  
class ToolCall:
    """
    Record of a tool call.
    
    Attributes:
        tool_name: Name of the tool called
        arguments: Arguments passed to the tool
        result: Result returned by the tool
        timestamp: When the call was made
        success: Whether the call succeeded
        error: Error message if failed
    """
    tool_name: str
    arguments: dict
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": str(self.result),  # Ensure serializable
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            result=data["result"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            success=data["success"],
            error=data["error"]
        )


class AgentState:
    """
    Manages all state for an agent.
    
    Includes:
    - Conversation history (messages)
    - Tool call history
    - Working memory (key-value store)
    - Current plan and progress
    - Persistence (save/load)
    
    Example:
        >>> state = AgentState(max_history_tokens=8000)
        >>> state.add_message("user", "Hello!")
        >>> state.set_memory("user_name", "Alice")
        >>> state.save("state.json")
    """
    
    def __init__(self, max_history_tokens: int = 8000):
        """
        Initialize agent state.
        
        Args:
            max_history_tokens: Maximum tokens to keep in history
        """
        self.max_history_tokens = max_history_tokens
        self.messages: list[Message] = []
        self.tool_calls: list[ToolCall] = []
        self.working_memory: dict[str, Any] = {}
        self.current_plan: list[str] | None = None
        self.completed_steps: list[str] = []
        self.created_at: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()
    
    def add_message(self, role: str, content: Any) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))
        self.last_updated = datetime.now()
        self._trim_history_if_needed()
    
    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        success: bool = True,
        error: str | None = None
    ) -> None:
        """
        Record a tool call.
        
        Args:
            tool_name: Name of the tool
            arguments: Arguments passed
            result: Tool result
            success: Whether it succeeded
            error: Error message if failed
        """
        self.tool_calls.append(ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            error=error
        ))
        self.last_updated = datetime.now()
    
    def get_messages_for_api(self) -> list[dict]:
        """
        Get messages in API format.
        
        Returns:
            List of message dictionaries for the Claude API
        """
        return [msg.to_api_format() for msg in self.messages]
    
    def get_last_user_message(self) -> str | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return None
    
    def get_last_assistant_message(self) -> str | None:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and isinstance(msg.content, str):
                return msg.content
        return None
    
    # Working Memory Operations
    
    def set_memory(self, key: str, value: Any) -> None:
        """
        Store a value in working memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        self.working_memory[key] = value
        self.last_updated = datetime.now()
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from working memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        return self.working_memory.get(key, default)
    
    def has_memory(self, key: str) -> bool:
        """Check if a key exists in working memory."""
        return key in self.working_memory
    
    def delete_memory(self, key: str) -> None:
        """Delete a key from working memory."""
        if key in self.working_memory:
            del self.working_memory[key]
            self.last_updated = datetime.now()
    
    def clear_memory(self) -> None:
        """Clear all working memory."""
        self.working_memory = {}
        self.last_updated = datetime.now()
    
    # Planning Operations
    
    def set_plan(self, steps: list[str]) -> None:
        """
        Set the current plan.
        
        Args:
            steps: List of plan steps
        """
        self.current_plan = steps
        self.completed_steps = []
        self.last_updated = datetime.now()
    
    def complete_step(self, step: str) -> None:
        """
        Mark a step as completed.
        
        Args:
            step: The step to mark complete
        """
        self.completed_steps.append(step)
        self.last_updated = datetime.now()
    
    def get_remaining_steps(self) -> list[str]:
        """
        Get steps that haven't been completed yet.
        
        Returns:
            List of remaining steps
        """
        if not self.current_plan:
            return []
        return [s for s in self.current_plan if s not in self.completed_steps]
    
    def has_plan(self) -> bool:
        """Check if there's an active plan."""
        return self.current_plan is not None and len(self.current_plan) > 0
    
    def clear_plan(self) -> None:
        """Clear the current plan."""
        self.current_plan = None
        self.completed_steps = []
        self.last_updated = datetime.now()
    
    # History Management
    
    def _trim_history_if_needed(self) -> None:
        """Trim old messages if we exceed token limits."""
        # Simple estimation: ~4 characters per token
        estimated_tokens = sum(
            len(str(msg.content)) // 4 for msg in self.messages
        )
        
        while estimated_tokens > self.max_history_tokens and len(self.messages) > 2:
            # Keep at least the first (system) and last message
            # Remove oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg.role != "system" and i < len(self.messages) - 1:
                    removed = self.messages.pop(i)
                    estimated_tokens -= len(str(removed.content)) // 4
                    break
    
    def clear_history(self) -> None:
        """Clear conversation history (keeps working memory)."""
        self.messages = []
        self.tool_calls = []
        self.last_updated = datetime.now()
    
    # Persistence
    
    def save(self, filepath: str) -> None:
        """
        Save state to a JSON file.
        
        Args:
            filepath: Path to save to
        """
        data = {
            "version": "1.0",
            "messages": [m.to_dict() for m in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "working_memory": self._serialize_memory(),
            "current_plan": self.current_plan,
            "completed_steps": self.completed_steps,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "max_history_tokens": self.max_history_tokens
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    def _serialize_memory(self) -> dict:
        """Serialize working memory, handling non-JSON types."""
        serialized = {}
        for key, value in self.working_memory.items():
            try:
                json.dumps(value)  # Test if serializable
                serialized[key] = value
            except (TypeError, ValueError):
                serialized[key] = str(value)
        return serialized
    
    @classmethod
    def load(cls, filepath: str) -> "AgentState":
        """
        Load state from a JSON file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded AgentState
        """
        data = json.loads(Path(filepath).read_text())
        
        state = cls(max_history_tokens=data.get("max_history_tokens", 8000))
        
        # Load messages
        for msg_data in data.get("messages", []):
            state.messages.append(Message.from_dict(msg_data))
        
        # Load tool calls
        for tc_data in data.get("tool_calls", []):
            state.tool_calls.append(ToolCall.from_dict(tc_data))
        
        # Load other fields
        state.working_memory = data.get("working_memory", {})
        state.current_plan = data.get("current_plan")
        state.completed_steps = data.get("completed_steps", [])
        state.created_at = datetime.fromisoformat(data["created_at"])
        state.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return state
    
    # Utilities
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of the current state.
        
        Returns:
            Summary string
        """
        return (
            f"Messages: {len(self.messages)}, "
            f"Tool calls: {len(self.tool_calls)}, "
            f"Memory keys: {list(self.working_memory.keys())}, "
            f"Plan steps remaining: {len(self.get_remaining_steps())}"
        )
    
    def get_tool_call_stats(self) -> dict:
        """
        Get statistics about tool calls.
        
        Returns:
            Dictionary with tool call statistics
        """
        if not self.tool_calls:
            return {"total": 0}
        
        stats = {
            "total": len(self.tool_calls),
            "successful": sum(1 for tc in self.tool_calls if tc.success),
            "failed": sum(1 for tc in self.tool_calls if not tc.success),
            "by_tool": {}
        }
        
        for tc in self.tool_calls:
            if tc.tool_name not in stats["by_tool"]:
                stats["by_tool"][tc.tool_name] = 0
            stats["by_tool"][tc.tool_name] += 1
        
        return stats
    
    def __repr__(self) -> str:
        return f"AgentState({self.get_summary()})"


if __name__ == "__main__":
    # Demonstrate state management
    print("=== AgentState Demonstration ===\n")
    
    # Create state
    state = AgentState(max_history_tokens=8000)
    
    # Add conversation
    state.add_message("user", "What's 2 + 2?")
    state.add_message("assistant", "2 + 2 equals 4.")
    state.add_message("user", "Thanks!")
    
    # Record tool calls
    state.add_tool_call(
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
        result=4,
        success=True
    )
    
    # Use working memory
    state.set_memory("user_preference", "detailed explanations")
    state.set_memory("session_id", "abc123")
    
    # Create a plan
    state.set_plan([
        "1. Analyze the question",
        "2. Use calculator tool",
        "3. Format the response"
    ])
    state.complete_step("1. Analyze the question")
    
    # Show state
    print(f"State summary: {state.get_summary()}")
    print(f"Tool stats: {state.get_tool_call_stats()}")
    print(f"Remaining steps: {state.get_remaining_steps()}")
    print(f"Working memory: {state.working_memory}")
    
    # Save and load
    state.save("/tmp/test_state.json")
    print("\nState saved to /tmp/test_state.json")
    
    loaded = AgentState.load("/tmp/test_state.json")
    print(f"Loaded state: {loaded.get_summary()}")
