"""
Example 01: The simplest possible tool definition.

This demonstrates the minimal structure every tool needs:
- name: A unique identifier
- description: What the tool does (Claude reads this!)
- input_schema: Parameter definitions (empty if no parameters)

Chapter 8: Defining Your First Tool
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()

# The simplest possible tool: no parameters
# This tool returns the current time when called
get_time_tool = {
    "name": "get_current_time",
    "description": """Returns the current date and time. 
Use this tool when the user asks about the current time, today's date, 
or needs to know what day it is.""",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


def main():
    """Demonstrate the basic tool definition."""
    
    print("=" * 60)
    print("Example 1: Basic Tool Definition (No Parameters)")
    print("=" * 60)
    print()
    
    # Display the tool definition
    print("Tool Definition:")
    print(json.dumps(get_time_tool, indent=2))
    print()
    
    # Send a message that should trigger tool use
    print("Sending message: 'What time is it right now?'")
    print("-" * 40)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[get_time_tool],
        messages=[
            {
                "role": "user",
                "content": "What time is it right now?"
            }
        ]
    )
    
    # Display the response
    print(f"Stop reason: {response.stop_reason}")
    print()
    
    for block in response.content:
        if block.type == "text":
            print(f"Text response: {block.text}")
        elif block.type == "tool_use":
            print(f"Tool requested: {block.name}")
            print(f"Tool ID: {block.id}")
            print(f"Arguments: {json.dumps(block.input, indent=2)}")
    
    print()
    print("=" * 60)
    print("Note: Claude wants to use the tool, but we haven't executed it.")
    print("In Chapter 9, we'll learn how to handle tool calls and return results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
