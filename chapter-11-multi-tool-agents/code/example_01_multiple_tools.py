"""
Demonstrating multiple tools in a single API request.

Chapter 11: Multi-Tool Agents

This example shows how to provide multiple tool definitions to Claude
in a single request. Claude will analyze the user's query and decide
which tool (if any) to use.
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

# Define multiple tools in a single list
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Use this when the user needs to calculate something.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first operand"
                },
                "b": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Gets the current date and time. Use this when the user asks about today's date, the current time, or what day of the week it is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (e.g., 'UTC', 'US/Eastern'). Defaults to local time."
                }
            },
            "required": []
        }
    }
]


def main():
    """Demonstrate providing multiple tools to Claude."""
    
    print("=" * 60)
    print("Multiple Tools Demo")
    print("=" * 60)
    
    # Test with a calculation query
    print("\n--- Test 1: Calculation Query ---")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's 15% of 85?"}
        ]
    )
    
    print(f"Stop reason: {message.stop_reason}")
    for block in message.content:
        if block.type == "tool_use":
            print(f"Tool selected: {block.name}")
            print(f"Arguments: {block.input}")
        elif hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Test with a datetime query
    print("\n--- Test 2: DateTime Query ---")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What day of the week is today?"}
        ]
    )
    
    print(f"Stop reason: {message.stop_reason}")
    for block in message.content:
        if block.type == "tool_use":
            print(f"Tool selected: {block.name}")
            print(f"Arguments: {block.input}")
        elif hasattr(block, "text"):
            print(f"Text: {block.text}")
    
    # Test with a query that doesn't need tools
    print("\n--- Test 3: General Knowledge Query ---")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the capital of France?"}
        ]
    )
    
    print(f"Stop reason: {message.stop_reason}")
    for block in message.content:
        if block.type == "tool_use":
            print(f"Tool selected: {block.name}")
            print(f"Arguments: {block.input}")
        elif hasattr(block, "text"):
            print(f"Text: {block.text}")


if __name__ == "__main__":
    main()
