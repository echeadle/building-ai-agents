"""
Demonstrating how Claude selects tools based on user queries.

Chapter 11: Multi-Tool Agents

This example runs multiple test queries to show how Claude intelligently
selects the appropriate tool (or no tool) based on the user's intent.
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

# Define our tools
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations (add, subtract, multiply, divide). Use this when the user needs to calculate something.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {"type": "number", "description": "The first operand"},
                "b": {"type": "number", "description": "The second operand"}
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
                    "description": "Optional timezone (e.g., 'UTC', 'America/New_York', 'Asia/Tokyo')"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a location. Use this when the user asks about weather, temperature, or conditions somewhere.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units (default: celsius)"
                }
            },
            "required": ["location"]
        }
    }
]


def test_tool_selection(query: str) -> None:
    """
    Test which tool Claude selects for a given query.
    
    Args:
        query: The user's question to test
    """
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": query}]
    )
    
    if message.stop_reason == "tool_use":
        for block in message.content:
            if block.type == "tool_use":
                print(f"✓ Selected tool: {block.name}")
                print(f"  Arguments: {block.input}")
    else:
        print("✗ No tool selected - Claude will respond directly")
        for block in message.content:
            if hasattr(block, "text"):
                # Truncate long responses for readability
                text = block.text
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"  Response: {text}")


def main():
    """Run tool selection tests with various queries."""
    
    print("=" * 60)
    print("Tool Selection Test Suite")
    print("=" * 60)
    print("\nTools available: calculator, get_current_datetime, get_weather")
    
    # Test queries organized by expected tool
    test_queries = [
        # Calculator queries
        ("Calculator queries:", [
            "What's 234 times 56?",
            "Calculate 15% of 85",
            "What's 100 divided by 7?",
        ]),
        # DateTime queries
        ("DateTime queries:", [
            "What day of the week is it?",
            "What's the current time?",
            "What time is it in Tokyo?",
        ]),
        # Weather queries
        ("Weather queries:", [
            "What's the weather like in London?",
            "Is it hot in Tokyo?",
        ]),
        # No tool needed
        ("No tool needed:", [
            "What's the capital of France?",
            "Who wrote Romeo and Juliet?",
        ]),
    ]
    
    for section_title, queries in test_queries:
        print(f"\n{'=' * 60}")
        print(section_title)
        print("=" * 60)
        for query in queries:
            test_tool_selection(query)


if __name__ == "__main__":
    main()
