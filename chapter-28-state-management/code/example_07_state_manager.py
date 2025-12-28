"""
Complete state management system for agents.

This example combines conversation history, working memory,
and long-term memory into a unified system.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum
import uuid

# Load environment variables from .env file
load_dotenv()


class TaskStatus(Enum):
    """Status of the current task."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """
    Complete state for an agent, combining all three memory layers.
    
    Attributes:
        session_id: Unique identifier for this session
        conversation_history: Messages exchanged in current conversation
        working_memory: Current task context and progress
        agent_id: Identifier for the agent (for multi-agent scenarios)
    """
    
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Layer 1: Conversation History
    conversation_history: list = field(default_factory=list)
    
    # Layer 2: Working Memory
    current_goal: Optional[str] = None
    task_status: str = "not_started"
    steps_completed: list[str] = field(default_factory=list)
    steps_remaining: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    gathered_facts: dict[str, Any] = field(default_factory=dict)
    decisions_made: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    # Metadata
    total_tool_calls: int = 0
    total_tokens_used: int = 0


class StateManager:
    """
    Unified state manager combining conversation history,
    working memory, and long-term memory.
    """
    
    def __init__(
        self,
        agent_id: str = "default",
        storage_dir: str = ".agent_state",
        enable_persistence: bool = True
    ):
        """
        Initialize the state manager.
        
        Args:
            agent_id: Identifier for this agent
            storage_dir: Directory for storing persistent state
            enable_persistence: Whether to persist state to disk
        """
        self.agent_id = agent_id
        self.storage_dir = Path(storage_dir)
        self.enable_persistence = enable_persistence
        
        if enable_persistence:
            self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = AgentState(agent_id=agent_id)
        
        # Long-term memory (separate from session state)
        self._long_term_memory: dict[str, Any] = {}
        self._load_long_term_memory()
    
    # =========================================
    # Layer 1: Conversation History
    # =========================================
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history."""
        self.state.conversation_history.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant text message to conversation history."""
        self.state.conversation_history.append({
            "role": "assistant", 
            "content": content
        })
    
    def add_assistant_response(self, response) -> None:
        """Add a complete assistant response (may include tool use)."""
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
                self.state.total_tool_calls += 1
        
        self.state.conversation_history.append({
            "role": "assistant",
            "content": content
        })
        
        # Track token usage if available
        if hasattr(response, 'usage'):
            self.state.total_tokens_used += (
                response.usage.input_tokens + response.usage.output_tokens
            )
    
    def add_tool_result(
        self, 
        tool_use_id: str, 
        result: str,
        is_error: bool = False
    ) -> None:
        """Add a tool result to conversation history."""
        self.state.conversation_history.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "is_error": is_error
            }]
        })
    
    def get_messages(self) -> list:
        """Get conversation history for API calls."""
        return self.state.conversation_history.copy()
    
    def clear_conversation(self) -> None:
        """Clear conversation history while preserving working memory."""
        self.state.conversation_history = []
    
    # =========================================
    # Layer 2: Working Memory
    # =========================================
    
    def start_task(
        self, 
        goal: str, 
        steps: Optional[list[str]] = None
    ) -> None:
        """Begin a new task."""
        self.state.current_goal = goal
        self.state.task_status = TaskStatus.IN_PROGRESS.value
        self.state.steps_remaining = steps or []
        self.state.steps_completed = []
        self.state.gathered_facts = {}
        self.state.decisions_made = []
        self.state.errors = []
        
        if self.state.steps_remaining:
            self.state.current_step = self.state.steps_remaining[0]
    
    def complete_step(self, step: str) -> None:
        """Mark a step as complete."""
        self.state.steps_completed.append(step)
        if step in self.state.steps_remaining:
            self.state.steps_remaining.remove(step)
        
        if self.state.steps_remaining:
            self.state.current_step = self.state.steps_remaining[0]
        else:
            self.state.current_step = None
    
    def add_fact(self, key: str, value: Any) -> None:
        """Store information gathered during the task."""
        self.state.gathered_facts[key] = value
    
    def get_fact(self, key: str) -> Optional[Any]:
        """Retrieve a gathered fact."""
        return self.state.gathered_facts.get(key)
    
    def record_decision(self, decision: str, reasoning: str) -> None:
        """Record a decision made during the task."""
        self.state.decisions_made.append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_error(self, error: str) -> None:
        """Record an error encountered."""
        self.state.errors.append(error)
    
    def complete_task(self, success: bool = True) -> None:
        """Mark the current task as complete."""
        self.state.task_status = (
            TaskStatus.COMPLETED.value if success 
            else TaskStatus.FAILED.value
        )
        self.state.current_step = None
    
    def get_working_memory_context(self) -> str:
        """Generate working memory context for prompts."""
        lines = []
        
        if self.state.current_goal:
            lines.append(f"CURRENT GOAL: {self.state.current_goal}")
            lines.append(f"STATUS: {self.state.task_status}")
        
        if self.state.current_step:
            lines.append(f"CURRENT STEP: {self.state.current_step}")
        
        if self.state.steps_completed:
            lines.append(f"COMPLETED: {', '.join(self.state.steps_completed)}")
        
        if self.state.steps_remaining:
            lines.append(f"REMAINING: {', '.join(self.state.steps_remaining)}")
        
        if self.state.gathered_facts:
            lines.append("\nGATHERED INFORMATION:")
            for key, value in self.state.gathered_facts.items():
                value_str = str(value)[:200]
                lines.append(f"  - {key}: {value_str}")
        
        if self.state.errors:
            lines.append(f"\nERRORS ENCOUNTERED: {len(self.state.errors)}")
        
        return "\n".join(lines) if lines else ""
    
    # =========================================
    # Layer 3: Long-Term Memory
    # =========================================
    
    def _get_long_term_memory_path(self) -> Path:
        """Get path to long-term memory file."""
        return self.storage_dir / f"{self.agent_id}_long_term.json"
    
    def _load_long_term_memory(self) -> None:
        """Load long-term memories from disk."""
        if not self.enable_persistence:
            return
        
        path = self._get_long_term_memory_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    self._long_term_memory = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._long_term_memory = {}
    
    def _save_long_term_memory(self) -> None:
        """Save long-term memories to disk."""
        if not self.enable_persistence:
            return
        
        path = self._get_long_term_memory_path()
        with open(path, 'w') as f:
            json.dump(self._long_term_memory, f, indent=2)
    
    def remember(self, key: str, value: Any, category: str = "general") -> None:
        """Store a long-term memory."""
        self._long_term_memory[key] = {
            "value": value,
            "category": category,
            "updated_at": datetime.now().isoformat()
        }
        self._save_long_term_memory()
    
    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a long-term memory."""
        if key in self._long_term_memory:
            return self._long_term_memory[key]["value"]
        return None
    
    def recall_category(self, category: str) -> dict[str, Any]:
        """Retrieve all memories in a category."""
        return {
            key: data["value"]
            for key, data in self._long_term_memory.items()
            if data.get("category") == category
        }
    
    def get_long_term_context(self, max_items: int = 10) -> str:
        """Generate long-term memory context for prompts."""
        if not self._long_term_memory:
            return ""
        
        lines = ["\nREMEMBERED FROM PAST SESSIONS:"]
        
        # Get most recent items
        items = sorted(
            self._long_term_memory.items(),
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True
        )[:max_items]
        
        for key, data in items:
            value_str = str(data["value"])[:100]
            lines.append(f"  - {key}: {value_str}")
        
        return "\n".join(lines)
    
    # =========================================
    # Session Persistence
    # =========================================
    
    def _get_session_path(self) -> Path:
        """Get path to session state file."""
        return self.storage_dir / f"session_{self.state.session_id}.json"
    
    def save_session(self) -> str:
        """
        Save current session state to disk.
        
        Returns:
            Path to saved session file
        """
        if not self.enable_persistence:
            raise RuntimeError("Persistence not enabled")
        
        path = self._get_session_path()
        
        # Convert state to dict
        state_dict = asdict(self.state)
        
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        return str(path)
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a previous session from disk.
        
        Args:
            session_id: The session ID to load
            
        Returns:
            True if loaded successfully, False if not found
        """
        if not self.enable_persistence:
            return False
        
        path = self.storage_dir / f"session_{session_id}.json"
        
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                state_dict = json.load(f)
            
            self.state = AgentState(**state_dict)
            return True
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not load session: {e}")
            return False
    
    def list_sessions(self) -> list[dict]:
        """List all saved sessions."""
        if not self.enable_persistence:
            return []
        
        sessions = []
        for path in self.storage_dir.glob("session_*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id"),
                        "created_at": data.get("created_at"),
                        "goal": data.get("current_goal"),
                        "status": data.get("task_status")
                    })
            except (json.JSONDecodeError, IOError):
                continue
        
        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)
    
    # =========================================
    # Unified Context Generation
    # =========================================
    
    def get_full_context(self) -> str:
        """
        Generate complete context from all memory layers.
        
        This is designed to be injected into system prompts.
        """
        sections = []
        
        # Working memory context
        working = self.get_working_memory_context()
        if working:
            sections.append("## Current Task Context\n" + working)
        
        # Long-term memory context
        long_term = self.get_long_term_context()
        if long_term:
            sections.append("## From Previous Sessions" + long_term)
        
        return "\n\n".join(sections) if sections else ""
    
    def get_summary(self) -> dict:
        """Get a summary of current state."""
        return {
            "session_id": self.state.session_id,
            "agent_id": self.state.agent_id,
            "messages": len(self.state.conversation_history),
            "current_goal": self.state.current_goal,
            "task_status": self.state.task_status,
            "facts_gathered": len(self.state.gathered_facts),
            "tool_calls": self.state.total_tool_calls,
            "long_term_memories": len(self._long_term_memory)
        }


def demonstrate_state_manager():
    """Demonstrate the complete state management system."""
    import shutil
    
    print("Demonstrating Complete State Manager")
    print("=" * 50)
    
    # Create state manager
    test_dir = ".test_state"
    manager = StateManager(
        agent_id="research_agent",
        storage_dir=test_dir
    )
    
    print(f"\n1. Session ID: {manager.state.session_id}")
    
    # Start a task
    print("\n2. Starting a task...")
    manager.start_task(
        goal="Research Python web frameworks",
        steps=["List frameworks", "Compare features", "Write summary"]
    )
    
    # Simulate conversation and progress
    print("\n3. Adding conversation and progress...")
    manager.add_user_message("What are the main Python web frameworks?")
    manager.add_fact("frameworks", ["Django", "Flask", "FastAPI", "Pyramid"])
    manager.complete_step("List frameworks")
    
    manager.record_decision(
        decision="Focus on Django, Flask, and FastAPI",
        reasoning="These are the most popular according to surveys"
    )
    
    # Store a long-term memory
    print("\n4. Storing long-term memory...")
    manager.remember("preferred_language", "Python", category="preference")
    manager.remember("skill_level", "intermediate", category="preference")
    
    # Get full context
    print("\n5. Full Context for Prompts:")
    print("-" * 40)
    print(manager.get_full_context())
    print("-" * 40)
    
    # Get summary
    print("\n6. State Summary:")
    summary = manager.get_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Save session
    print("\n7. Saving session...")
    saved_path = manager.save_session()
    print(f"   Saved to: {saved_path}")
    
    # List sessions
    print("\n8. Available sessions:")
    sessions = manager.list_sessions()
    for s in sessions:
        print(f"   - {s['session_id']}: {s['goal']} ({s['status']})")
    
    # Load in new manager
    print("\n9. Loading session in new manager...")
    session_id = manager.state.session_id
    
    new_manager = StateManager(agent_id="research_agent", storage_dir=test_dir)
    if new_manager.load_session(session_id):
        print(f"   Loaded session {session_id}")
        print(f"   Goal: {new_manager.state.current_goal}")
        print(f"   Messages: {len(new_manager.state.conversation_history)}")
        print(f"   Long-term recall: {new_manager.recall('preferred_language')}")
    
    # Clean up
    print("\n10. Cleaning up...")
    shutil.rmtree(test_dir)
    print("   Test directory removed.")


if __name__ == "__main__":
    demonstrate_state_manager()
