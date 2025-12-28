---
chapter: 28
title: "State Management"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 28: State Management

## Introduction

In the previous chapter, you built the core agentic loop—the perceive-think-act cycle that lets an LLM direct its own execution. But there's a critical problem with what we built: the agent has no memory. Each API call is stateless, meaning Claude has no idea what happened in previous iterations unless we explicitly tell it.

Imagine asking your agent to "research the company we discussed earlier and add it to the report you started." Without state management, the agent would have no idea which company you discussed or what report exists. It would be like having a conversation with someone who forgets everything you said after each sentence.

State management is what transforms a stateless API into a coherent, continuous agent experience. In this chapter, you'll learn how to give your agents memory—both for the current task and across sessions.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement conversation history as the foundation of agent state
- Build working memory systems for tracking current task context
- Create persistent state that survives between sessions
- Serialize and deserialize agent state to JSON files
- Design state schemas that capture everything an agent needs to remember

## The Stateless Nature of LLM APIs

Before diving into solutions, let's be crystal clear about the problem. Every time you call the Claude API, it's a completely fresh start. Claude doesn't remember:

- Previous messages in your conversation
- Tools it has called
- Decisions it has made
- Context you've established

**You** are responsible for providing all of this context in every single API call. This is both a challenge and an opportunity—you have complete control over what the agent "remembers."

```python
"""
Demonstrating the stateless nature of API calls.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# First call - introduce a topic
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "My favorite color is blue. Remember that."}
    ]
)
print("Response 1:", response1.content[0].text)

# Second call - completely separate, no memory!
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "What's my favorite color?"}
    ]
)
print("Response 2:", response2.content[0].text)
# Claude won't know - it never saw the first message!
```

If you run this code, Claude will politely explain in the second response that it doesn't know your favorite color. The two API calls are completely independent.

## Layer 1: Conversation History

The most fundamental form of state is **conversation history**—the complete record of messages exchanged between the user and the agent. This is what enables multi-turn conversations.

### The Messages Array

You've seen this pattern since Chapter 5, but let's formalize it as our first state layer:

```python
"""
Conversation history as state.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional

load_dotenv()


@dataclass
class ConversationState:
    """Manages conversation history as state."""
    
    messages: list = field(default_factory=list)
    system_prompt: Optional[str] = None
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def add_tool_use(self, tool_use_block: dict) -> None:
        """Add an assistant message with tool use."""
        self.messages.append({
            "role": "assistant",
            "content": [tool_use_block]
        })
    
    def add_tool_result(self, tool_use_id: str, result: str) -> None:
        """Add a tool result to the history."""
        self.messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result
            }]
        })
    
    def get_messages(self) -> list:
        """Return messages for API call."""
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
    
    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)


def chat_with_history():
    """Demonstrate stateful conversation."""
    client = anthropic.Anthropic()
    state = ConversationState(
        system_prompt="You are a helpful assistant with a good memory."
    )
    
    # First exchange
    state.add_user_message("My favorite color is blue. Remember that.")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=state.system_prompt,
        messages=state.get_messages()
    )
    
    assistant_response = response.content[0].text
    state.add_assistant_message(assistant_response)
    print(f"Assistant: {assistant_response}\n")
    
    # Second exchange - now with history!
    state.add_user_message("What's my favorite color?")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=state.system_prompt,
        messages=state.get_messages()
    )
    
    assistant_response = response.content[0].text
    state.add_assistant_message(assistant_response)
    print(f"Assistant: {assistant_response}")
    
    print(f"\nTotal messages in history: {len(state)}")


if __name__ == "__main__":
    chat_with_history()
```

Now Claude will correctly recall your favorite color because both messages are included in the second API call.

### Handling Tool Use in Conversation History

When agents use tools, the conversation history becomes more complex. You need to track not just text messages, but tool calls and their results:

```python
"""
Conversation history with tool use tracking.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

load_dotenv()


@dataclass
class ToolCall:
    """Record of a tool call and its result."""
    
    tool_name: str
    tool_use_id: str
    arguments: dict
    result: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationStateWithTools:
    """Manages conversation history including tool calls."""
    
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    system_prompt: Optional[str] = None
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_response(self, response) -> None:
        """
        Add an assistant response, handling both text and tool use.
        
        Args:
            response: The API response object
        """
        # Build content list from response
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({
                    "type": "text",
                    "text": block.text
                })
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        self.messages.append({"role": "assistant", "content": content})
    
    def add_tool_result(
        self, 
        tool_use_id: str, 
        tool_name: str,
        arguments: dict,
        result: str,
        is_error: bool = False
    ) -> None:
        """Add a tool result and record the tool call."""
        # Add to messages for API
        self.messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
                "is_error": is_error
            }]
        })
        
        # Record the tool call for our tracking
        self.tool_calls.append(ToolCall(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            arguments=arguments,
            result=result if not is_error else f"ERROR: {result}"
        ))
    
    def get_tool_history(self) -> list[ToolCall]:
        """Get all tool calls made in this conversation."""
        return self.tool_calls.copy()
    
    def get_messages(self) -> list:
        """Return messages for API call."""
        return self.messages.copy()


# Example usage
if __name__ == "__main__":
    state = ConversationStateWithTools(
        system_prompt="You are a helpful assistant."
    )
    
    # Simulate a conversation with tool use
    state.add_user_message("What's 25 * 17?")
    
    # Simulate assistant requesting tool use
    # (In real code, this comes from the API response)
    print("Conversation state tracks both messages and tool calls.")
    print(f"Messages: {len(state.messages)}")
    print(f"Tool calls: {len(state.tool_calls)}")
```

## Layer 2: Working Memory

Conversation history captures what was said, but agents often need to track higher-level context about the current task. This is **working memory**—structured information about what the agent is currently doing.

Think of working memory as the agent's "scratchpad" for the current task:

- What is the current goal?
- What steps have been completed?
- What information has been gathered?
- What decisions have been made?

```python
"""
Working memory for current task context.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

load_dotenv()


class TaskStatus(Enum):
    """Status of the current task."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkingMemory:
    """
    Working memory for tracking current task context.
    
    This is the agent's "scratchpad" for the current task,
    holding structured information about goals, progress, and findings.
    """
    
    # Current task information
    current_goal: Optional[str] = None
    task_status: TaskStatus = TaskStatus.NOT_STARTED
    started_at: Optional[str] = None
    
    # Progress tracking
    steps_completed: list[str] = field(default_factory=list)
    steps_remaining: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    
    # Information gathered during task
    gathered_facts: dict[str, Any] = field(default_factory=dict)
    
    # Decisions and reasoning
    decisions_made: list[dict] = field(default_factory=list)
    
    # Error tracking
    errors_encountered: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def start_task(self, goal: str, planned_steps: list[str] = None) -> None:
        """Begin a new task."""
        self.current_goal = goal
        self.task_status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()
        self.steps_remaining = planned_steps or []
        self.steps_completed = []
        self.gathered_facts = {}
        self.decisions_made = []
        self.errors_encountered = []
        self.retry_count = 0
        
        if self.steps_remaining:
            self.current_step = self.steps_remaining[0]
    
    def complete_step(self, step: str, result: Optional[str] = None) -> None:
        """Mark a step as complete."""
        self.steps_completed.append(step)
        
        if step in self.steps_remaining:
            self.steps_remaining.remove(step)
        
        if result:
            self.gathered_facts[f"step_{len(self.steps_completed)}_result"] = result
        
        # Move to next step
        if self.steps_remaining:
            self.current_step = self.steps_remaining[0]
        else:
            self.current_step = None
    
    def add_fact(self, key: str, value: Any) -> None:
        """Store a piece of information gathered during the task."""
        self.gathered_facts[key] = value
    
    def record_decision(self, decision: str, reasoning: str) -> None:
        """Record a decision made during the task."""
        self.decisions_made.append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_error(self, error: str) -> bool:
        """
        Record an error. Returns True if we should retry.
        """
        self.errors_encountered.append(error)
        self.retry_count += 1
        
        if self.retry_count >= self.max_retries:
            self.task_status = TaskStatus.FAILED
            return False
        
        self.task_status = TaskStatus.BLOCKED
        return True
    
    def complete_task(self, success: bool = True) -> None:
        """Mark the task as complete."""
        self.task_status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.current_step = None
    
    def get_context_summary(self) -> str:
        """
        Generate a summary of current working memory for the LLM.
        
        This is injected into prompts to give the agent context.
        """
        lines = []
        
        if self.current_goal:
            lines.append(f"CURRENT GOAL: {self.current_goal}")
            lines.append(f"STATUS: {self.task_status.value}")
        
        if self.current_step:
            lines.append(f"CURRENT STEP: {self.current_step}")
        
        if self.steps_completed:
            lines.append(f"COMPLETED STEPS: {', '.join(self.steps_completed)}")
        
        if self.steps_remaining:
            lines.append(f"REMAINING STEPS: {', '.join(self.steps_remaining)}")
        
        if self.gathered_facts:
            lines.append("GATHERED INFORMATION:")
            for key, value in self.gathered_facts.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                lines.append(f"  - {key}: {value_str}")
        
        if self.decisions_made:
            lines.append("DECISIONS MADE:")
            for d in self.decisions_made[-3:]:  # Last 3 decisions
                lines.append(f"  - {d['decision']}")
        
        if self.errors_encountered:
            lines.append(f"ERRORS: {len(self.errors_encountered)} encountered")
            lines.append(f"  Last error: {self.errors_encountered[-1]}")
        
        return "\n".join(lines) if lines else "No active task."


# Example usage
if __name__ == "__main__":
    memory = WorkingMemory()
    
    # Start a research task
    memory.start_task(
        goal="Research competitor pricing",
        planned_steps=[
            "Identify top 3 competitors",
            "Find pricing pages",
            "Extract pricing tiers",
            "Summarize findings"
        ]
    )
    
    # Simulate progress
    memory.add_fact("competitors", ["Acme Corp", "Beta Inc", "Gamma LLC"])
    memory.complete_step("Identify top 3 competitors", "Found 3 competitors")
    
    memory.record_decision(
        decision="Focus on Acme Corp first",
        reasoning="They are the market leader with the most similar product"
    )
    
    memory.add_fact("acme_pricing", {"basic": 29, "pro": 99, "enterprise": 299})
    memory.complete_step("Find pricing pages")
    
    # Print current context
    print(memory.get_context_summary())
```

### Injecting Working Memory into Prompts

Working memory is only useful if the agent can access it. Here's how to include working memory context in your prompts:

```python
"""
Injecting working memory into agent prompts.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional

load_dotenv()


def build_system_prompt_with_memory(
    base_prompt: str,
    working_memory: "WorkingMemory"
) -> str:
    """
    Build a system prompt that includes working memory context.
    
    Args:
        base_prompt: The agent's base system prompt
        working_memory: Current working memory state
    
    Returns:
        Complete system prompt with context
    """
    memory_context = working_memory.get_context_summary()
    
    if memory_context == "No active task.":
        return base_prompt
    
    return f"""{base_prompt}

## Current Task Context

{memory_context}

Use this context to inform your responses. Update your approach based on what has already been accomplished and what information has been gathered."""


# Example
if __name__ == "__main__":
    # Assuming WorkingMemory class from previous example
    from dataclasses import dataclass, field
    from typing import Any
    from enum import Enum
    from datetime import datetime
    
    # Simplified WorkingMemory for this example
    @dataclass
    class WorkingMemory:
        current_goal: Optional[str] = None
        gathered_facts: dict = field(default_factory=dict)
        
        def get_context_summary(self) -> str:
            if not self.current_goal:
                return "No active task."
            lines = [f"CURRENT GOAL: {self.current_goal}"]
            if self.gathered_facts:
                lines.append("GATHERED INFORMATION:")
                for k, v in self.gathered_facts.items():
                    lines.append(f"  - {k}: {v}")
            return "\n".join(lines)
    
    # Set up memory with some context
    memory = WorkingMemory()
    memory.current_goal = "Plan a weekend trip to Portland"
    memory.gathered_facts = {
        "budget": "$500",
        "dates": "March 15-17",
        "interests": "food, hiking, coffee"
    }
    
    base_prompt = "You are a helpful travel planning assistant."
    
    full_prompt = build_system_prompt_with_memory(base_prompt, memory)
    print("Full System Prompt:")
    print("-" * 50)
    print(full_prompt)
```

## Layer 3: Long-Term Memory

While conversation history and working memory handle the current session, **long-term memory** persists information across sessions. This enables agents to:

- Remember user preferences discovered in past conversations
- Build on research from previous sessions
- Maintain continuity across multiple interactions

For this chapter, we'll implement a simple file-based long-term memory. In production systems, you might use databases, vector stores, or specialized memory services.

```python
"""
Long-term memory with file-based persistence.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

load_dotenv()


@dataclass
class MemoryEntry:
    """A single long-term memory entry."""
    
    key: str
    value: Any
    category: str  # e.g., "preference", "fact", "context"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.updated_at = datetime.now().isoformat()
        self.access_count += 1


class LongTermMemory:
    """
    File-based long-term memory storage.
    
    Persists memories to a JSON file for retrieval across sessions.
    """
    
    def __init__(self, storage_path: str = "agent_memory.json"):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Path to the JSON file for storing memories
        """
        self.storage_path = Path(storage_path)
        self.memories: dict[str, MemoryEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.memories[key] = MemoryEntry(**entry_data)
                print(f"Loaded {len(self.memories)} memories from {self.storage_path}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load memories: {e}")
                self.memories = {}
        else:
            print(f"No existing memory file. Starting fresh.")
    
    def _save(self) -> None:
        """Save memories to disk."""
        data = {
            key: asdict(entry) 
            for key, entry in self.memories.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember(
        self, 
        key: str, 
        value: Any, 
        category: str = "general"
    ) -> None:
        """
        Store a memory.
        
        Args:
            key: Unique identifier for this memory
            value: The information to remember
            category: Category for organizing memories
        """
        if key in self.memories:
            # Update existing memory
            self.memories[key].value = value
            self.memories[key].updated_at = datetime.now().isoformat()
        else:
            # Create new memory
            self.memories[key] = MemoryEntry(
                key=key,
                value=value,
                category=category
            )
        self._save()
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory by key.
        
        Args:
            key: The memory key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        if key in self.memories:
            self.memories[key].touch()
            self._save()
            return self.memories[key].value
        return None
    
    def recall_by_category(self, category: str) -> dict[str, Any]:
        """
        Retrieve all memories in a category.
        
        Args:
            category: The category to filter by
            
        Returns:
            Dictionary of key-value pairs in the category
        """
        result = {}
        for key, entry in self.memories.items():
            if entry.category == category:
                entry.touch()
                result[key] = entry.value
        if result:
            self._save()
        return result
    
    def forget(self, key: str) -> bool:
        """
        Remove a memory.
        
        Args:
            key: The memory key to remove
            
        Returns:
            True if memory was removed, False if not found
        """
        if key in self.memories:
            del self.memories[key]
            self._save()
            return True
        return False
    
    def get_context_for_prompt(self, max_entries: int = 10) -> str:
        """
        Generate a context string of recent/relevant memories for prompts.
        
        Args:
            max_entries: Maximum number of memories to include
            
        Returns:
            Formatted string of memories
        """
        if not self.memories:
            return "No long-term memories stored."
        
        # Sort by access count and recency
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.access_count, m.updated_at),
            reverse=True
        )[:max_entries]
        
        lines = ["## Remembered Information"]
        
        # Group by category
        by_category: dict[str, list] = {}
        for mem in sorted_memories:
            if mem.category not in by_category:
                by_category[mem.category] = []
            by_category[mem.category].append(mem)
        
        for category, mems in by_category.items():
            lines.append(f"\n### {category.title()}")
            for mem in mems:
                value_str = str(mem.value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"- {mem.key}: {value_str}")
        
        return "\n".join(lines)
    
    def clear_all(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self._save()


# Example usage
if __name__ == "__main__":
    # Create memory store
    memory = LongTermMemory("test_memory.json")
    
    # Store some memories
    memory.remember("user_name", "Alice", category="preference")
    memory.remember("favorite_color", "blue", category="preference")
    memory.remember("timezone", "America/New_York", category="preference")
    memory.remember("last_topic", "machine learning", category="context")
    memory.remember("project_deadline", "2025-03-15", category="fact")
    
    # Retrieve a specific memory
    name = memory.recall("user_name")
    print(f"User's name: {name}")
    
    # Get all preferences
    prefs = memory.recall_by_category("preference")
    print(f"\nPreferences: {prefs}")
    
    # Get context for prompts
    print("\n" + "=" * 50)
    print(memory.get_context_for_prompt())
    
    # Clean up test file
    os.remove("test_memory.json")
```

## Putting It All Together: The Complete State Manager

Now let's combine all three layers into a unified state management system:

```python
"""
Complete state management system for agents.

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


# Example usage
if __name__ == "__main__":
    # Create state manager
    manager = StateManager(
        agent_id="research_agent",
        storage_dir=".test_state"
    )
    
    print(f"Session ID: {manager.state.session_id}")
    
    # Start a task
    manager.start_task(
        goal="Research Python web frameworks",
        steps=["List frameworks", "Compare features", "Write summary"]
    )
    
    # Simulate some progress
    manager.add_user_message("What are the main Python web frameworks?")
    manager.add_fact("frameworks", ["Django", "Flask", "FastAPI", "Pyramid"])
    manager.complete_step("List frameworks")
    
    manager.record_decision(
        decision="Focus on Django, Flask, and FastAPI",
        reasoning="These are the most popular according to surveys"
    )
    
    # Store a long-term memory
    manager.remember("preferred_language", "Python", category="preference")
    
    # Get full context
    print("\n" + "=" * 50)
    print("FULL CONTEXT FOR PROMPTS:")
    print("=" * 50)
    print(manager.get_full_context())
    
    # Save session
    saved_path = manager.save_session()
    print(f"\nSession saved to: {saved_path}")
    
    # Clean up test files
    import shutil
    shutil.rmtree(".test_state")
```

## Building a Stateful Agent

Now let's build a complete agent that uses our state management system:

```python
"""
A complete stateful agent with memory.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import uuid

load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class StatefulAgent:
    """
    An agent with full state management capabilities.
    
    Combines conversation history, working memory, and long-term
    memory to maintain context across interactions.
    """
    
    def __init__(
        self,
        name: str = "Assistant",
        system_prompt: str = "You are a helpful assistant.",
        storage_dir: str = ".agent_state",
        model: str = "claude-sonnet-4-20250514",
        tools: Optional[list] = None,
        tool_handlers: Optional[dict[str, Callable]] = None
    ):
        """
        Initialize the stateful agent.
        
        Args:
            name: Name of the agent
            system_prompt: Base system prompt
            storage_dir: Directory for persistent storage
            model: Model to use
            tools: Tool definitions for the agent
            tool_handlers: Dict mapping tool names to handler functions
        """
        self.name = name
        self.base_system_prompt = system_prompt
        self.model = model
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        
        # Initialize client
        self.client = anthropic.Anthropic()
        
        # Initialize state
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.session_id = str(uuid.uuid4())[:8]
        self.conversation: list = []
        self.working_memory: dict = {
            "current_goal": None,
            "task_status": "idle",
            "gathered_facts": {},
            "steps_completed": [],
            "tool_calls_count": 0
        }
        
        # Load long-term memory
        self.long_term_memory = self._load_long_term_memory()
    
    def _get_memory_path(self) -> Path:
        """Path to long-term memory file."""
        return self.storage_dir / f"{self.name.lower()}_memory.json"
    
    def _load_long_term_memory(self) -> dict:
        """Load long-term memory from disk."""
        path = self._get_memory_path()
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_long_term_memory(self) -> None:
        """Save long-term memory to disk."""
        with open(self._get_memory_path(), 'w') as f:
            json.dump(self.long_term_memory, f, indent=2)
    
    def remember(self, key: str, value: Any) -> None:
        """Store a long-term memory."""
        self.long_term_memory[key] = {
            "value": value,
            "stored_at": datetime.now().isoformat()
        }
        self._save_long_term_memory()
    
    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a long-term memory."""
        if key in self.long_term_memory:
            return self.long_term_memory[key]["value"]
        return None
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with memory context."""
        prompt_parts = [self.base_system_prompt]
        
        # Add working memory context
        if self.working_memory["current_goal"]:
            prompt_parts.append(f"""
## Current Task
Goal: {self.working_memory['current_goal']}
Status: {self.working_memory['task_status']}
""")
            
            if self.working_memory["gathered_facts"]:
                facts = "\n".join(
                    f"- {k}: {v}" 
                    for k, v in self.working_memory["gathered_facts"].items()
                )
                prompt_parts.append(f"Gathered Information:\n{facts}")
            
            if self.working_memory["steps_completed"]:
                steps = ", ".join(self.working_memory["steps_completed"])
                prompt_parts.append(f"Completed Steps: {steps}")
        
        # Add relevant long-term memories
        if self.long_term_memory:
            memories = "\n".join(
                f"- {k}: {v['value']}" 
                for k, v in list(self.long_term_memory.items())[:5]
            )
            prompt_parts.append(f"""
## Remembered Information
{memories}
""")
        
        return "\n".join(prompt_parts)
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name not in self.tool_handlers:
            return f"Error: Unknown tool '{tool_name}'"
        
        try:
            result = self.tool_handlers[tool_name](**tool_input)
            self.working_memory["tool_calls_count"] += 1
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _process_response(self, response) -> tuple[str, bool]:
        """
        Process an API response, handling tool calls if present.
        
        Returns:
            Tuple of (text response, needs_continuation)
        """
        text_parts = []
        tool_uses = []
        
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)
        
        # Store assistant response in conversation
        content = []
        if text_parts:
            content.append({"type": "text", "text": " ".join(text_parts)})
        for tool_use in tool_uses:
            content.append({
                "type": "tool_use",
                "id": tool_use.id,
                "name": tool_use.name,
                "input": tool_use.input
            })
        
        self.conversation.append({"role": "assistant", "content": content})
        
        # Handle tool calls
        if tool_uses:
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                
                # Store fact from tool result
                fact_key = f"{tool_use.name}_result"
                self.working_memory["gathered_facts"][fact_key] = result[:200]
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })
            
            self.conversation.append({"role": "user", "content": tool_results})
            return " ".join(text_parts), True  # Needs continuation
        
        return " ".join(text_parts), False  # Complete
    
    def start_task(self, goal: str) -> None:
        """Start a new task."""
        self.working_memory["current_goal"] = goal
        self.working_memory["task_status"] = "in_progress"
        self.working_memory["gathered_facts"] = {}
        self.working_memory["steps_completed"] = []
    
    def complete_task(self) -> None:
        """Mark current task as complete."""
        self.working_memory["task_status"] = "completed"
        self.working_memory["current_goal"] = None
    
    def chat(self, user_message: str, max_turns: int = 10) -> str:
        """
        Send a message and get a response.
        
        Handles tool use automatically, continuing until the agent
        provides a final text response.
        
        Args:
            user_message: The user's message
            max_turns: Maximum tool use turns to prevent infinite loops
            
        Returns:
            The agent's final text response
        """
        # Add user message to conversation
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self._build_system_prompt(),
            "messages": self.conversation
        }
        
        if self.tools:
            request_params["tools"] = self.tools
        
        turns = 0
        final_response = ""
        
        while turns < max_turns:
            # Make API call
            response = self.client.messages.create(**request_params)
            
            # Process response
            text, needs_continuation = self._process_response(response)
            final_response = text
            
            if not needs_continuation:
                break
            
            # Update request for continuation
            request_params["messages"] = self.conversation
            turns += 1
        
        return final_response
    
    def save_session(self) -> str:
        """Save current session state."""
        session_data = {
            "session_id": self.session_id,
            "conversation": self.conversation,
            "working_memory": self.working_memory,
            "saved_at": datetime.now().isoformat()
        }
        
        path = self.storage_dir / f"session_{self.session_id}.json"
        with open(path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return str(path)
    
    def load_session(self, session_id: str) -> bool:
        """Load a previous session."""
        path = self.storage_dir / f"session_{session_id}.json"
        
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.session_id = data["session_id"]
            self.conversation = data["conversation"]
            self.working_memory = data["working_memory"]
            return True
            
        except (json.JSONDecodeError, KeyError):
            return False
    
    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "messages_count": len(self.conversation),
            "current_goal": self.working_memory.get("current_goal"),
            "task_status": self.working_memory.get("task_status"),
            "tool_calls": self.working_memory.get("tool_calls_count", 0),
            "facts_gathered": len(self.working_memory.get("gathered_facts", {}))
        }


# Define some example tools
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def take_note(note: str) -> str:
    """Store a note for later reference."""
    return f"Note stored: {note}"


# Tool definitions
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations. Pass a math expression like '2 + 2' or '15 * 7'.",
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
        "name": "take_note",
        "description": "Store a note for later reference",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to store"
                }
            },
            "required": ["note"]
        }
    }
]

TOOL_HANDLERS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "take_note": take_note
}


# Example usage
if __name__ == "__main__":
    # Create stateful agent
    agent = StatefulAgent(
        name="ResearchAssistant",
        system_prompt="""You are a helpful research assistant. You help users 
gather information, take notes, and perform calculations. 

When asked to remember something, store it using the take_note tool.
When doing research, break down tasks and track your progress.""",
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        storage_dir=".test_agent_state"
    )
    
    print(f"Agent initialized. Session ID: {agent.session_id}")
    print("-" * 50)
    
    # Have a conversation
    print("User: My name is Alice and my favorite number is 42. Please remember that.")
    response = agent.chat("My name is Alice and my favorite number is 42. Please remember that.")
    print(f"Agent: {response}")
    print()
    
    # Store in long-term memory
    agent.remember("user_name", "Alice")
    agent.remember("favorite_number", 42)
    
    # Ask something that requires tools
    print("User: What's my favorite number multiplied by 10?")
    response = agent.chat("What's my favorite number multiplied by 10?")
    print(f"Agent: {response}")
    print()
    
    # Ask about time
    print("User: What time is it?")
    response = agent.chat("What time is it?")
    print(f"Agent: {response}")
    print()
    
    # Test memory across turns
    print("User: What was my name again?")
    response = agent.chat("What was my name again?")
    print(f"Agent: {response}")
    print()
    
    # Print session summary
    print("-" * 50)
    print("Session Summary:")
    summary = agent.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save session
    saved_path = agent.save_session()
    print(f"\nSession saved to: {saved_path}")
    
    # Demonstrate loading session
    print("\n" + "=" * 50)
    print("Creating new agent and loading previous session...")
    
    new_agent = StatefulAgent(
        name="ResearchAssistant",
        system_prompt="You are a helpful research assistant.",
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        storage_dir=".test_agent_state"
    )
    
    # Load the previous session
    session_id = agent.session_id
    if new_agent.load_session(session_id):
        print(f"Loaded session {session_id}")
        print(f"Conversation has {len(new_agent.conversation)} messages")
        
        # Continue the conversation
        print("\nUser: Based on our conversation, what do you know about me?")
        response = new_agent.chat("Based on our conversation, what do you know about me?")
        print(f"Agent: {response}")
    
    # Clean up test files
    import shutil
    shutil.rmtree(".test_agent_state")
```

## Common Pitfalls

### 1. Token Limits and Conversation Truncation

Long conversations accumulate tokens quickly. If you don't manage this, you'll hit context limits:

```python
"""
Handling conversation truncation.

Chapter 28: State Management
"""

def truncate_conversation(
    messages: list,
    max_messages: int = 50,
    keep_system_context: bool = True
) -> list:
    """
    Truncate conversation to stay within limits.
    
    Keeps the most recent messages while preserving important context.
    
    Args:
        messages: Full conversation history
        max_messages: Maximum messages to keep
        keep_system_context: Whether to summarize and keep early context
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_messages:
        return messages
    
    if keep_system_context:
        # Keep first few messages for context
        context_messages = messages[:2]
        recent_messages = messages[-(max_messages - 2):]
        
        # Add a summary message
        summary = {
            "role": "user",
            "content": "[Earlier conversation truncated for brevity]"
        }
        
        return context_messages + [summary] + recent_messages
    else:
        return messages[-max_messages:]


# More sophisticated approach: summarize old messages
def summarize_and_truncate(
    messages: list,
    client,  # Anthropic client
    threshold: int = 30
) -> list:
    """
    When conversation gets long, summarize older messages.
    
    This preserves context while reducing token count.
    """
    if len(messages) <= threshold:
        return messages
    
    # Messages to summarize (keeping last 10 intact)
    to_summarize = messages[:-10]
    to_keep = messages[-10:]
    
    # Create summary prompt
    summary_messages = [
        {
            "role": "user",
            "content": f"""Summarize this conversation in 2-3 sentences, 
capturing the key points and any important information:

{json.dumps(to_summarize, indent=2)}"""
        }
    ]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=summary_messages
    )
    
    summary_text = response.content[0].text
    
    # Return summary + recent messages
    return [
        {"role": "user", "content": f"[Previous conversation summary: {summary_text}]"},
        {"role": "assistant", "content": "I understand. I'll keep this context in mind."}
    ] + to_keep
```

### 2. State Desynchronization

If you update state in multiple places without proper synchronization, things can get out of sync:

```python
# BAD: State updated in multiple places
conversation.append(message)  # Updated here
state.messages.append(message)  # And here
# These can get out of sync!

# GOOD: Single source of truth
state_manager.add_message(message)  # One place to update
```

### 3. Not Persisting Before Crashes

Always save state at critical points, not just at the end:

```python
def execute_tool_with_save(self, tool_name: str, args: dict) -> str:
    """Execute tool and save state before and after."""
    # Save before (in case tool crashes)
    self.save_session()
    
    try:
        result = self._execute_tool(tool_name, args)
        self.working_memory["last_tool_result"] = result
        # Save after success
        self.save_session()
        return result
    except Exception as e:
        self.working_memory["last_error"] = str(e)
        self.save_session()  # Save error state too
        raise
```

## Practical Exercise

**Task:** Extend the StatefulAgent to include a "memory importance" system.

**Requirements:**

1. Add an importance score (1-10) when storing long-term memories
2. Implement a method that retrieves only high-importance memories (score >= 7)
3. Add decay: memories accessed less frequently should have reduced importance over time
4. Implement a `forget_unimportant()` method that removes memories below a threshold

**Hints:**

- Track `access_count` and `last_accessed` for each memory
- Calculate effective importance as: `base_importance * decay_factor`
- Decay factor could be based on days since last access

**Solution:** See `code/exercise_memory_importance.py`

## Key Takeaways

- **The API is stateless**: You must provide all context in every call. This is both a challenge and an opportunity for control.

- **Three layers of memory**: Conversation history (what was said), working memory (current task context), and long-term memory (persistent facts) each serve different purposes.

- **State enables continuity**: Without state management, agents forget their purpose mid-task and can't build on previous interactions.

- **Persistence is essential**: Save state regularly to survive crashes and enable resuming sessions.

- **Context injection is key**: Working memory and long-term memory only help if they're included in prompts where the agent can use them.

## What's Next

Now that your agents have memory, they can remember what they're doing—but do they know how to approach complex tasks strategically? In Chapter 29, we'll add **Planning and Reasoning** capabilities, teaching your agents to think before they act. You'll implement plan-then-execute patterns that dramatically improve reliability on multi-step tasks.
