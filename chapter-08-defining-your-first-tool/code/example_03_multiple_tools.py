"""
Example 03: Multiple tools for Claude to choose from.

This demonstrates:
- Defining several tools with different purposes
- How Claude selects the appropriate tool based on the user's question
- Tools with varying parameter requirements

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

# Define multiple tools
tools = [
    {
        "name": "calculate",
        "description": """Performs arithmetic calculations on two numbers.
Use this for addition, subtraction, multiplication, or division.
More reliable than mental math, especially for large numbers.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The arithmetic operation",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "get_current_time",
        "description": """Returns the current date and time.
Use when the user asks about the current time, today's date, 
what day of the week it is, or anything time-related.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "convert_temperature",
        "description": """Converts temperatures between Celsius and Fahrenheit.
Use when the user wants to convert a temperature from one unit to another.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The temperature value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "The unit to convert from",
                    "enum": ["celsius", "fahrenheit"]
                },
                "to_unit": {
                    "type": "string",
                    "description": "The unit to convert to",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
    {
        "name": "generate_random_number",
        "description": """Generates a random number within a specified range.
Use when the user needs a random number, wants to pick a random value,
or is playing a game that requires randomness.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_value": {
                    "type": "integer",
                    "description": "The minimum value (inclusive)"
                },
                "max_value": {
                    "type": "integer",
                    "description": "The maximum value (inclusive)"
                }
            },
            "required": ["min_value", "max_value"]
        }
    }
]


def main():
    """Demonstrate how Claude selects the appropriate tool."""
    
    print("=" * 60)
    print("Example 3: Multiple Tools")
    print("=" * 60)
    print()
    
    # Print available tools
    print("Available tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'].split('.')[0]}")
    print()
    
    # Test questions that should trigger different tools
    test_questions = [
        ("What is 847 divided by 3?", "calculate"),
        ("What time is it?", "get_current_time"),
        ("Convert 72 degrees Fahrenheit to Celsius", "convert_temperature"),
        ("Pick a random number between 1 and 100", "generate_random_number"),
        ("What day is today?", "get_current_time"),
        ("If the temperature is 30°C, what is that in Fahrenheit?", "convert_temperature"),
    ]
    
    for question, expected_tool in test_questions:
        print(f"Question: {question}")
        print(f"Expected tool: {expected_tool}")
        print("-" * 40)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        for block in response.content:
            if block.type == "tool_use":
                selected = "✓" if block.name == expected_tool else "✗"
                print(f"Selected tool: {block.name} {selected}")
                print(f"Arguments: {json.dumps(block.input, indent=2)}")
            elif block.type == "text" and block.text.strip():
                print(f"Text: {block.text}")
        
        print()
    
    print("=" * 60)
    print("Claude intelligently selects the right tool for each question!")
    print("=" * 60)


if __name__ == "__main__":
    main()
