"""
Example 02: Calculator tool with parameters.

This demonstrates a practical tool with:
- Multiple parameters (operation, a, b)
- Enum constraints for valid operations
- Clear descriptions that help Claude understand when to use the tool

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

# Calculator tool with well-crafted description and parameter schemas
calculator_tool = {
    "name": "calculate",
    "description": """Performs basic arithmetic calculations on two numbers.

Use this tool whenever you need to:
- Add, subtract, multiply, or divide numbers
- Perform calculations that require precision
- Do math with large numbers or decimals

This tool is more reliable than mental math and should be used for any
non-trivial calculations. It handles edge cases like division by zero.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The arithmetic operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "The first number in the calculation"
            },
            "b": {
                "type": "number",
                "description": "The second number in the calculation"
            }
        },
        "required": ["operation", "a", "b"]
    }
}


def main():
    """Demonstrate the calculator tool with various math questions."""
    
    print("=" * 60)
    print("Example 2: Calculator Tool")
    print("=" * 60)
    print()
    
    # Display the tool definition
    print("Tool Definition:")
    print(json.dumps(calculator_tool, indent=2))
    print()
    
    # Test questions that should trigger the calculator
    test_questions = [
        "What is 1,234 multiplied by 5,678?",
        "If I have $150.75 and spend $43.25, how much do I have left?",
        "What is 999 divided by 37?",
    ]
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 40)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[calculator_tool],
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        for block in response.content:
            if block.type == "text":
                print(f"Text: {block.text}")
            elif block.type == "tool_use":
                print(f"Tool: {block.name}")
                print(f"Arguments: {json.dumps(block.input, indent=2)}")
        
        print()
    
    print("=" * 60)
    print("Notice how Claude correctly identifies the operation and numbers")
    print("from natural language, even with formatting like commas and $.")
    print("=" * 60)


if __name__ == "__main__":
    main()
