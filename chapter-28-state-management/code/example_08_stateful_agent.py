"""
A complete stateful agent with memory.

This example shows a fully functional agent with conversation history,
working memory, and long-term memory capabilities.

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
import shutil

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
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


def main():
    """Demonstrate the stateful agent."""
    test_dir = ".test_agent_state"
    
    print("Demonstrating Stateful Agent")
    print("=" * 50)
    
    # Create stateful agent
    agent = StatefulAgent(
        name="ResearchAssistant",
        system_prompt="""You are a helpful research assistant. You help users 
gather information, take notes, and perform calculations. 

When asked to remember something, store it using the take_note tool.
When doing research, break down tasks and track your progress.""",
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        storage_dir=test_dir
    )
    
    print(f"\nAgent initialized. Session ID: {agent.session_id}")
    print("-" * 50)
    
    # Have a conversation
    print("\n1. Teaching the agent about the user...")
    print("User: My name is Alice and my favorite number is 42. Please remember that.")
    response = agent.chat("My name is Alice and my favorite number is 42. Please remember that.")
    print(f"Agent: {response}")
    
    # Store in long-term memory
    agent.remember("user_name", "Alice")
    agent.remember("favorite_number", 42)
    
    # Ask something that requires tools
    print("\n2. Using calculation tool with context...")
    print("User: What's my favorite number multiplied by 10?")
    response = agent.chat("What's my favorite number multiplied by 10?")
    print(f"Agent: {response}")
    
    # Ask about time
    print("\n3. Using time tool...")
    print("User: What time is it?")
    response = agent.chat("What time is it?")
    print(f"Agent: {response}")
    
    # Test memory across turns
    print("\n4. Testing conversation memory...")
    print("User: What was my name again?")
    response = agent.chat("What was my name again?")
    print(f"Agent: {response}")
    
    # Print session summary
    print("\n" + "-" * 50)
    print("Session Summary:")
    summary = agent.get_session_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save session
    saved_path = agent.save_session()
    print(f"\nSession saved to: {saved_path}")
    
    # Demonstrate loading session
    print("\n" + "=" * 50)
    print("5. Creating new agent and loading previous session...")
    
    new_agent = StatefulAgent(
        name="ResearchAssistant",
        system_prompt="You are a helpful research assistant.",
        tools=TOOLS,
        tool_handlers=TOOL_HANDLERS,
        storage_dir=test_dir
    )
    
    # Load the previous session
    session_id = agent.session_id
    if new_agent.load_session(session_id):
        print(f"   Loaded session {session_id}")
        print(f"   Conversation has {len(new_agent.conversation)} messages")
        
        # Continue the conversation
        print("\nUser: Based on our conversation, what do you know about me?")
        response = new_agent.chat("Based on our conversation, what do you know about me?")
        print(f"Agent: {response}")
    
    # Clean up test files
    print("\n" + "-" * 50)
    print("Cleaning up test files...")
    shutil.rmtree(test_dir)
    print("Done!")


if __name__ == "__main__":
    main()
