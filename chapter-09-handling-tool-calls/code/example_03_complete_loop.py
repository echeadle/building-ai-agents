"""
The Complete Tool Use Loop

This script demonstrates the full tool use loop:
1. Send a message to Claude with tools available
2. Check if Claude wants to use a tool
3. If yes: execute the tool and send results back
4. Repeat until Claude responds with final text

This is the foundation for all agentic behavior!

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


# =============================================================================
# TOOL DEFINITION
# =============================================================================

calculator_tool = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations on two numbers. Use this whenever you need to calculate something mathematically. Supports addition, subtraction, multiplication, and division.",
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


# =============================================================================
# TOOL IMPLEMENTATION
# =============================================================================

def calculator(operation: str, a: float, b: float) -> float:
    """Execute a calculator operation.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
        
    Returns:
        The result of the operation
        
    Raises:
        ValueError: For division by zero or unknown operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Registry mapping tool names to their implementation functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
}


# =============================================================================
# THE TOOL USE LOOP
# =============================================================================

def process_tool_calls(response, messages: list) -> list:
    """Process all tool calls in a response and update messages.
    
    This function:
    1. Extracts all tool use blocks from Claude's response
    2. Executes each tool
    3. Collects the results
    4. Adds both Claude's response and our results to the message history
    
    Args:
        response: Claude's API response containing tool_use blocks
        messages: The current conversation messages list
        
    Returns:
        Updated messages list with assistant response and tool results
    """
    tool_results = []
    
    # Process each tool use block in the response
    for block in response.content:
        if block.type == "tool_use":
            tool_name = block.name
            tool_args = block.input
            tool_id = block.id
            
            print(f"  🔧 Tool: {tool_name}")
            print(f"     Args: {tool_args}")
            
            # Execute the tool
            try:
                if tool_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[tool_name](**tool_args)
                    
                    # Format result nicely
                    if isinstance(result, float) and result.is_integer():
                        result_str = str(int(result))
                    else:
                        result_str = str(result)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str
                    })
                    print(f"     Result: {result_str}")
                else:
                    # Unknown tool
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Error: Unknown tool '{tool_name}'",
                        "is_error": True
                    })
                    print(f"     Error: Unknown tool")
                    
            except Exception as e:
                # Tool execution failed
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
                print(f"     Error: {e}")
    
    # Add Claude's response to message history
    # This preserves the tool_use blocks so Claude knows what it asked for
    messages.append({
        "role": "assistant",
        "content": response.content
    })
    
    # Add our tool results
    # This goes in a "user" role message (we're responding to Claude)
    messages.append({
        "role": "user",
        "content": tool_results
    })
    
    return messages


def chat_with_tools(user_message: str, verbose: bool = True) -> str:
    """Send a message and handle tool calls until we get a final response.
    
    This is the main entry point. It:
    1. Starts a conversation with the user's message
    2. Loops while Claude keeps requesting tools
    3. Returns Claude's final text response
    
    Args:
        user_message: The user's question or request
        verbose: Whether to print detailed progress
        
    Returns:
        Claude's final text response
    """
    # Initialize conversation with user's message
    messages = [{"role": "user", "content": user_message}]
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"User: {user_message}")
        print("=" * 60)
    
    # Safety limit to prevent infinite loops
    MAX_ITERATIONS = 10
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        if verbose:
            print(f"\n--- API Call {iteration} ---")
        
        # Make API call with tools
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[calculator_tool],
            messages=messages
        )
        
        if verbose:
            print(f"Stop reason: {response.stop_reason}")
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            if verbose:
                print("Claude is using tools...")
            messages = process_tool_calls(response, messages)
        else:
            # No more tool calls - extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            if verbose:
                print("\n" + "=" * 60)
                print(f"Claude: {final_response}")
                print("=" * 60)
            
            return final_response
    
    # If we hit the iteration limit, something went wrong
    return "Error: Maximum iterations reached. The conversation may be stuck in a loop."


def demonstrate_simple_calculation():
    """Show a simple single-tool calculation."""
    print("\n" + "#" * 60)
    print("DEMO 1: Simple Calculation")
    print("#" * 60)
    
    chat_with_tools("What is 42 multiplied by 17?")


def demonstrate_multi_step_calculation():
    """Show a calculation requiring multiple tool calls."""
    print("\n" + "#" * 60)
    print("DEMO 2: Multi-Step Calculation")
    print("#" * 60)
    
    chat_with_tools("What is (15 + 27) multiplied by 4?")


def demonstrate_complex_problem():
    """Show a word problem requiring reasoning and calculation."""
    print("\n" + "#" * 60)
    print("DEMO 3: Word Problem")
    print("#" * 60)
    
    chat_with_tools(
        "If I have 3 boxes with 24 apples each, and I give away 17 apples, "
        "how many apples do I have left?"
    )


def demonstrate_error_handling():
    """Show how errors are handled gracefully."""
    print("\n" + "#" * 60)
    print("DEMO 4: Error Handling (Division by Zero)")
    print("#" * 60)
    
    chat_with_tools("What is 100 divided by 0?")


def demonstrate_no_tool_needed():
    """Show that Claude doesn't use tools when not needed."""
    print("\n" + "#" * 60)
    print("DEMO 5: No Tool Needed")
    print("#" * 60)
    
    chat_with_tools("What is the capital of Japan?")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_simple_calculation()
    demonstrate_multi_step_calculation()
    demonstrate_complex_problem()
    demonstrate_error_handling()
    demonstrate_no_tool_needed()
    
    # Interactive mode
    print("\n" + "#" * 60)
    print("INTERACTIVE MODE")
    print("#" * 60)
    print("Try your own math questions! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_input:
            chat_with_tools(user_input)
