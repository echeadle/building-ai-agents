"""
Basic agentic loop implementation.

This script demonstrates the fundamental pattern for sequential tool calling:
the agentic loop. It shows how to keep calling Claude until it finishes
using tools and provides a final answer.

Chapter 12: Sequential Tool Calls
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

# Initialize the client
client = anthropic.Anthropic()

# Define a simple calculator tool
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use this for any arithmetic, percentages, or mathematical expressions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '15 * 7', '100 / 4', '2 ** 10')"
                }
            },
            "required": ["expression"]
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool and return the result.
    
    Args:
        name: The tool name
        arguments: The tool arguments
    
    Returns:
        The tool result as a string
    """
    if name == "calculator":
        try:
            expression = arguments.get("expression", "")
            # Safety: only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    else:
        return f"Error: Unknown tool '{name}'"


def run_agent(
    user_message: str,
    max_iterations: int = 10
) -> str:
    """
    Run an agent that can make sequential tool calls.
    
    This is the basic agentic loop pattern:
    1. Send message to Claude with tools
    2. If Claude returns tool_use, execute tools and continue
    3. If Claude returns end_turn, return the final answer
    4. Repeat until done or max_iterations reached
    
    Args:
        user_message: The user's question or request
        max_iterations: Maximum number of tool call iterations (safety limit)
    
    Returns:
        The agent's final text response
    """
    # Initialize conversation with user message
    messages = [{"role": "user", "content": user_message}]
    
    system_prompt = """You are a helpful assistant with access to a calculator.
Use the calculator tool whenever you need to perform calculations.
Always show your work and explain your reasoning."""

    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print('='*50)
        
        # Call Claude with the current conversation
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        # Check if Claude is done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract the final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # IMPORTANT: Add Claude's response to the conversation
            # This includes both text and tool_use blocks
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "text":
                    print(f"Claude's thinking: {block.text}")
                elif block.type == "tool_use":
                    print(f"\nTool call: {block.name}")
                    print(f"Arguments: {json.dumps(block.input, indent=2)}")
                    
                    # Execute the tool
                    result = execute_tool(block.name, block.input)
                    print(f"Result: {result}")
                    
                    # Format the result for Claude
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            # Add tool results as a user message
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # Unexpected stop reason (like max_tokens)
            print(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    return "Maximum iterations reached without a final response."


def main():
    """Run example queries through the agent."""
    
    # Example 1: Simple calculation
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple calculation")
    print("="*60)
    
    question1 = "What is 47 * 83?"
    print(f"\nUser: {question1}")
    answer1 = run_agent(question1)
    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print('='*50)
    print(answer1)
    
    # Example 2: Multi-step calculation
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-step calculation")
    print("="*60)
    
    question2 = """
    I bought 3 items:
    - A book for $24.99
    - A pen for $3.50
    - A notebook for $12.75
    
    What's the total? And if I have a 15% discount, what's the final price?
    """
    print(f"\nUser: {question2}")
    answer2 = run_agent(question2)
    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print('='*50)
    print(answer2)


if __name__ == "__main__":
    main()
