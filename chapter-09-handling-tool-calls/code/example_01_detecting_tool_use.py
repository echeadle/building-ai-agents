"""
Detecting Tool Use in Claude's Responses

This script demonstrates how to:
1. Send a request to Claude with tools available
2. Detect when Claude wants to use a tool
3. Extract tool use blocks from the response

Chapter 9: Handling Tool Calls
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

# Initialize the Anthropic client
client = anthropic.Anthropic()

# Define our calculator tool (from Chapter 8)
calculator_tool = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations on two numbers. Use this whenever you need to calculate something mathematically.",
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
}


def has_tool_use(response) -> bool:
    """Check if the response indicates Claude wants to use a tool.
    
    Args:
        response: The API response from Claude
        
    Returns:
        True if Claude wants to use a tool, False otherwise
    """
    return response.stop_reason == "tool_use"


def extract_tool_uses(response) -> list:
    """Extract all tool use blocks from a response.
    
    Claude can request multiple tool calls in a single response,
    so we need to iterate through all content blocks.
    
    Args:
        response: The API response from Claude
        
    Returns:
        List of tool use blocks
    """
    tool_uses = []
    for block in response.content:
        if block.type == "tool_use":
            tool_uses.append(block)
    return tool_uses


def extract_text_content(response) -> str:
    """Extract any text content from the response.
    
    Even when using tools, Claude might include explanatory text.
    
    Args:
        response: The API response from Claude
        
    Returns:
        Concatenated text content from the response
    """
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return " ".join(text_parts)


def demonstrate_tool_use_detection():
    """Show how tool use detection works with different prompts."""
    
    print("=" * 60)
    print("DETECTING TOOL USE IN CLAUDE'S RESPONSES")
    print("=" * 60)
    
    # Test 1: A question that SHOULD trigger tool use
    print("\n--- Test 1: Math question (should trigger tool use) ---")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "What is 42 times 17?"}
        ]
    )
    
    print(f"Stop reason: {response.stop_reason}")
    print(f"Has tool use: {has_tool_use(response)}")
    
    if has_tool_use(response):
        tool_uses = extract_tool_uses(response)
        print(f"Number of tool calls: {len(tool_uses)}")
        
        for i, tool_use in enumerate(tool_uses):
            print(f"\nTool call {i + 1}:")
            print(f"  Tool ID: {tool_use.id}")
            print(f"  Tool name: {tool_use.name}")
            print(f"  Arguments: {tool_use.input}")
    
    # Test 2: A question that should NOT trigger tool use
    print("\n--- Test 2: General question (should not trigger tool use) ---")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    print(f"Stop reason: {response.stop_reason}")
    print(f"Has tool use: {has_tool_use(response)}")
    
    text = extract_text_content(response)
    if text:
        print(f"Text response: {text[:100]}...")  # First 100 chars
    
    # Test 3: Multiple operations (might trigger multiple tool calls)
    print("\n--- Test 3: Complex math question ---")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "First add 100 and 50, then tell me what that equals."}
        ]
    )
    
    print(f"Stop reason: {response.stop_reason}")
    print(f"Has tool use: {has_tool_use(response)}")
    
    if has_tool_use(response):
        tool_uses = extract_tool_uses(response)
        print(f"Number of tool calls: {len(tool_uses)}")
        
        for tool_use in tool_uses:
            print(f"  → {tool_use.name}({tool_use.input})")


def demonstrate_response_structure():
    """Show the complete structure of a tool use response."""
    
    print("\n" + "=" * 60)
    print("RESPONSE STRUCTURE DEEP DIVE")
    print("=" * 60)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "Calculate 99 divided by 3"}
        ]
    )
    
    print("\nFull response attributes:")
    print(f"  id: {response.id}")
    print(f"  type: {response.type}")
    print(f"  role: {response.role}")
    print(f"  model: {response.model}")
    print(f"  stop_reason: {response.stop_reason}")
    
    print("\nContent blocks:")
    for i, block in enumerate(response.content):
        print(f"\n  Block {i}:")
        print(f"    type: {block.type}")
        
        if block.type == "tool_use":
            print(f"    id: {block.id}")
            print(f"    name: {block.name}")
            print(f"    input: {block.input}")
        elif hasattr(block, "text"):
            print(f"    text: {block.text[:50]}..." if len(block.text) > 50 else f"    text: {block.text}")
    
    print("\nUsage information:")
    print(f"  input_tokens: {response.usage.input_tokens}")
    print(f"  output_tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    demonstrate_tool_use_detection()
    demonstrate_response_structure()
