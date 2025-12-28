"""
Conversation history with tool use tracking.

This example shows how to track both messages and tool calls
in your conversation state.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


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
    
    def get_tool_summary(self) -> str:
        """Get a summary of all tool calls made."""
        if not self.tool_calls:
            return "No tool calls made yet."
        
        lines = ["Tool Call History:"]
        for i, tc in enumerate(self.tool_calls, 1):
            lines.append(f"  {i}. {tc.tool_name}({tc.arguments}) -> {tc.result[:50]}...")
        return "\n".join(lines)


def demonstrate_tool_tracking():
    """Show conversation state with tool tracking."""
    client = anthropic.Anthropic()
    
    # Define a simple calculator tool
    tools = [
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
    
    state = ConversationStateWithTools(
        system_prompt="You are a helpful assistant with calculator abilities."
    )
    
    print("Demonstrating Tool Tracking in Conversation State")
    print("=" * 50)
    
    # Add a user message that will trigger tool use
    user_message = "What is 42 * 17?"
    print(f"\nUser: {user_message}")
    state.add_user_message(user_message)
    
    # Make API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=state.system_prompt,
        messages=state.get_messages(),
        tools=tools
    )
    
    # Process response
    state.add_assistant_response(response)
    
    # Check if tool was used
    for block in response.content:
        if block.type == "tool_use":
            print(f"Claude wants to use: {block.name}")
            print(f"Arguments: {block.input}")
            
            # Execute the tool (safely)
            try:
                expression = block.input.get("expression", "")
                # Only allow safe characters
                allowed = set("0123456789+-*/.(). ")
                if all(c in allowed for c in expression):
                    result = str(eval(expression))
                else:
                    result = "Error: Invalid expression"
            except Exception as e:
                result = f"Error: {e}"
            
            print(f"Tool result: {result}")
            
            # Add tool result to state
            state.add_tool_result(
                tool_use_id=block.id,
                tool_name=block.name,
                arguments=block.input,
                result=result
            )
    
    # Get final response after tool use
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=state.system_prompt,
        messages=state.get_messages(),
        tools=tools
    )
    
    state.add_assistant_response(response)
    
    # Print final answer
    for block in response.content:
        if block.type == "text":
            print(f"\nClaude: {block.text}")
    
    # Show tool tracking
    print("\n" + "=" * 50)
    print("\nState Summary:")
    print(f"  Total messages: {len(state.messages)}")
    print(f"  Tool calls made: {len(state.tool_calls)}")
    print(f"\n{state.get_tool_summary()}")


if __name__ == "__main__":
    demonstrate_tool_tracking()
