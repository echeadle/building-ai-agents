"""
A preview of tool use in Claude.

This script demonstrates the basic structure of tool definitions and
shows how Claude responds when given access to tools. It's a preview
of what you'll build fully in Chapters 8 and 9.

Note: This example shows Claude requesting to use a tool, but does NOT
yet execute the tool. Full tool execution is covered in Chapter 9.

Chapter 7: Introduction to Tool Use
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


def define_calculator_tool() -> list:
    """
    Define a simple calculator tool.
    
    This is PHASE 1 of the tool use cycle: DEFINE.
    We specify what the tool is, what it does, and what parameters it needs.
    
    Returns:
        A list containing the tool definition
    """
    return [
        {
            "name": "calculate",
            "description": (
                "Perform basic arithmetic calculations. Use this tool whenever "
                "the user asks for a mathematical calculation, including addition, "
                "subtraction, multiplication, division, or more complex expressions. "
                "This tool provides accurate results and should be used instead of "
                "attempting mental math."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "The mathematical expression to evaluate. "
                            "Examples: '2 + 2', '15 * 7', '100 / 4', '(10 + 5) * 3'"
                        )
                    }
                },
                "required": ["expression"]
            }
        }
    ]


def demonstrate_tool_request() -> None:
    """
    Show how Claude responds when given access to a tool.
    
    When Claude decides to use a tool, it doesn't execute the tool itself.
    Instead, it responds with a tool_use request that YOUR code must handle.
    """
    tools = define_calculator_tool()
    
    # Questions that should trigger tool use
    math_questions = [
        "What is 1,547 times 892?",
        "If I have 15 apples and give away 7, then buy 23 more, how many do I have?",
        "What's 15% of 847?",
    ]
    
    print("=" * 70)
    print("DEMONSTRATING TOOL USE REQUESTS")
    print("=" * 70)
    print("\nWe've defined a 'calculate' tool. Let's see how Claude uses it.")
    print("-" * 70)
    
    for question in math_questions:
        print(f"\n❓ Question: {question}")
        
        # Send the message with tool definitions
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,  # <-- This is where we provide tool definitions
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        print(f"\n📊 Response Analysis:")
        print(f"   Stop Reason: {response.stop_reason}")
        
        # Examine each content block in the response
        for i, block in enumerate(response.content):
            print(f"\n   Content Block {i + 1}:")
            print(f"   Type: {block.type}")
            
            if block.type == "text":
                print(f"   Text: {block.text}")
            elif block.type == "tool_use":
                print(f"   Tool Name: {block.name}")
                print(f"   Tool ID: {block.id}")
                print(f"   Arguments: {json.dumps(block.input, indent=6)}")
        
        print("\n" + "-" * 70)


def demonstrate_no_tool_needed() -> None:
    """
    Show that Claude doesn't use tools when they're not needed.
    """
    tools = define_calculator_tool()
    
    # Questions that don't need the calculator
    non_math_questions = [
        "What is the capital of France?",
        "Explain what photosynthesis is.",
    ]
    
    print("\n" + "=" * 70)
    print("DEMONSTRATING WHEN TOOLS ARE NOT USED")
    print("=" * 70)
    print("\nClaude only uses tools when they're helpful for the task.")
    print("-" * 70)
    
    for question in non_math_questions:
        print(f"\n❓ Question: {question}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            tools=tools,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        print(f"\n📊 Response Analysis:")
        print(f"   Stop Reason: {response.stop_reason}")
        
        # This should be a text response, not a tool use
        for block in response.content:
            if block.type == "text":
                # Truncate long responses for readability
                text = block.text
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"\n   Response: {text}")
        
        print("\n" + "-" * 70)


def explain_next_steps() -> None:
    """
    Explain what comes next in the tool use cycle.
    """
    print("\n" + "=" * 70)
    print("WHAT HAPPENS NEXT (Covered in Chapter 9)")
    print("=" * 70)
    print("""
When Claude responds with a tool_use request, we need to:

1. DETECT the tool use request (check stop_reason == 'tool_use')

2. PARSE the request:
   - tool name ('calculate')
   - tool arguments ({'expression': '1547 * 892'})

3. EXECUTE the tool:
   - Actually perform the calculation
   - In this case: eval('1547 * 892') = 1,379,924

4. RETURN the result to Claude:
   - Send a new message with the tool result
   - Claude will then generate a natural language response

5. GET the final response:
   - Claude: "1,547 times 892 equals 1,379,924"

This complete cycle is what we call the AGENTIC LOOP, and you'll
implement it step by step in the coming chapters.

For now, the key insight is:
  Claude REQUESTS tool use → YOU execute → Claude RESPONDS

You are always in control of what actually gets executed!
""")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 7: Preview of Tool Use")
    print("=" * 70)
    print("\nThis script previews how Claude interacts with tools.")
    print("Full implementation comes in Chapters 8 and 9.")
    
    # Run the demonstrations
    demonstrate_tool_request()
    demonstrate_no_tool_needed()
    explain_next_steps()
