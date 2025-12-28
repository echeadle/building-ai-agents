"""
The simplest possible agentic loop.

This demonstrates the core perceive-think-act cycle that powers all agents.
The loop is intentionally minimal to show the fundamental pattern.

Chapter 27: The Agentic Loop
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


# A simple tool for demonstration
SIMPLE_TOOLS = [
    {
        "name": "add_numbers",
        "description": "Adds two numbers together. Use this when you need to perform addition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["a", "b"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return its result.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        
    Returns:
        String result of the tool execution
    """
    if tool_name == "add_numbers":
        result = tool_input["a"] + tool_input["b"]
        return str(result)
    else:
        return f"Unknown tool: {tool_name}"


def simple_agent_loop(user_message: str, max_iterations: int = 10) -> str:
    """
    The simplest agentic loop: perceive, think, act, repeat.
    
    This function demonstrates the core pattern without any extras.
    
    Args:
        user_message: The user's initial request
        max_iterations: Safety limit to prevent infinite loops
        
    Returns:
        The agent's final response
    """
    # PERCEIVE: Initialize with user's message
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # THINK: Let Claude process everything and decide what to do
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=SIMPLE_TOOLS,
            messages=messages
        )
        
        # CHECK: Is Claude done?
        if response.stop_reason == "end_turn":
            # Claude is done - extract and return the final response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""  # No text response
        
        # ACT: Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add assistant's message (including tool calls) to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # PERCEIVE: Add tool results for the next iteration
            messages.append({"role": "user", "content": tool_results})
    
    # Safety fallback
    return "Max iterations reached without completing the task."


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Agentic Loop Demo")
    print("=" * 60)
    
    # Test 1: A simple question (no tool needed)
    print("\nTest 1: Simple question")
    print("-" * 40)
    response = simple_agent_loop("What is the capital of France?")
    print(f"Response: {response}")
    
    # Test 2: A question requiring the tool
    print("\nTest 2: Question requiring tool use")
    print("-" * 40)
    response = simple_agent_loop("What is 42 + 58?")
    print(f"Response: {response}")
    
    # Test 3: A more conversational request
    print("\nTest 3: Conversational math request")
    print("-" * 40)
    response = simple_agent_loop(
        "I have 127 apples and my friend gives me 89 more. How many do I have now?"
    )
    print(f"Response: {response}")
