"""
Conversation history as state.

This example shows how to maintain conversation history to give
Claude context across multiple messages.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


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
    
    print("Demonstrating Stateful Conversation")
    print("=" * 50)
    
    # First exchange
    print("\n--- First Exchange ---")
    user_message = "My favorite color is blue. Remember that."
    print(f"User: {user_message}")
    
    state.add_user_message(user_message)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=state.system_prompt,
        messages=state.get_messages()
    )
    
    assistant_response = response.content[0].text
    state.add_assistant_message(assistant_response)
    print(f"Claude: {assistant_response}")
    
    # Second exchange - now with history!
    print("\n--- Second Exchange (with history) ---")
    user_message = "What's my favorite color?"
    print(f"User: {user_message}")
    
    state.add_user_message(user_message)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=state.system_prompt,
        messages=state.get_messages()
    )
    
    assistant_response = response.content[0].text
    state.add_assistant_message(assistant_response)
    print(f"Claude: {assistant_response}")
    
    print("\n" + "=" * 50)
    print(f"Total messages in history: {len(state)}")
    print("Now Claude remembers because we included the conversation history!")


if __name__ == "__main__":
    chat_with_history()
