"""
A complete agentic loop with working tools and detailed logging.

This example shows the full loop in action with real tool execution
and logs each phase so you can see what's happening.

Chapter 27: The Agentic Loop
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# Define a toolkit with multiple tools
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Supports +, -, *, /, and parentheses. Use this for any math calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g., '2 + 2' or '(10 * 5) / 2'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets the current date and time. Use this when the user asks about the current time, date, day of week, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format: 'full' (date and time), 'date' (date only), 'time' (time only)",
                    "enum": ["full", "date", "time"]
                }
            },
            "required": []
        }
    },
    {
        "name": "string_length",
        "description": "Counts the number of characters in a string. Use this when the user wants to know how long a text is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to measure"
                }
            },
            "required": ["text"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return its result as a string.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Dictionary of input parameters
        
    Returns:
        String result of the tool execution
    """
    if tool_name == "calculator":
        try:
            expression = tool_input["expression"]
            # Safety check: only allow safe characters
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Expression contains invalid characters. Only numbers and +-*/() are allowed."
            
            # Evaluate the expression
            result = eval(expression)
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except SyntaxError:
            return "Error: Invalid mathematical expression"
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif tool_name == "get_current_time":
        now = datetime.now()
        format_type = tool_input.get("format", "full")
        
        if format_type == "date":
            return now.strftime("%A, %B %d, %Y")
        elif format_type == "time":
            return now.strftime("%I:%M:%S %p")
        else:  # full
            return now.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    
    elif tool_name == "string_length":
        text = tool_input.get("text", "")
        length = len(text)
        return f"{length} characters"
    
    else:
        return f"Error: Unknown tool '{tool_name}'"


def log_phase(phase: str, details: str = "") -> None:
    """Print a formatted log message for a phase."""
    print(f"\n{'─' * 50}")
    print(f"│ {phase}")
    if details:
        print(f"│ {details}")
    print(f"{'─' * 50}")


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """
    Run the agentic loop with detailed logging.
    
    Args:
        user_message: What the user wants to accomplish
        max_iterations: Safety limit for the loop
        
    Returns:
        The agent's final response
    """
    print("\n" + "═" * 60)
    print(f"  USER REQUEST: {user_message}")
    print("═" * 60)
    
    # Initialize conversation
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        print(f"\n{'═' * 60}")
        print(f"  ITERATION {iteration + 1}")
        print("═" * 60)
        
        # PERCEIVE PHASE
        log_phase("PERCEIVE", f"Messages in context: {len(messages)}")
        
        # THINK PHASE
        log_phase("THINK", "Sending to Claude...")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        print(f"│ Stop reason: {response.stop_reason}")
        print(f"│ Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
        
        # Analyze response content
        for block in response.content:
            if hasattr(block, "text"):
                print(f"│ Text: {block.text[:80]}..." if len(block.text) > 80 else f"│ Text: {block.text}")
            elif block.type == "tool_use":
                print(f"│ Tool request: {block.name}({json.dumps(block.input)})")
        
        # CHECK: Is Claude done?
        if response.stop_reason == "end_turn":
            log_phase("ACT", "COMPLETE - Returning final response")
            
            # Extract and return the text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        
        # ACT PHASE: Handle tool calls
        if response.stop_reason == "tool_use":
            log_phase("ACT", "Executing tool calls...")
            
            # Add assistant message to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"│ Executing: {block.name}")
                    print(f"│   Input: {json.dumps(block.input)}")
                    
                    result = execute_tool(block.name, block.input)
                    print(f"│   Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})
    
    log_phase("TERMINATED", f"Max iterations ({max_iterations}) reached")
    return "I wasn't able to complete the task within the allowed iterations."


if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  AGENTIC LOOP DEMONSTRATION")
    print("▓" * 60)
    
    # Test 1: Simple calculation
    print("\n\n" + "▒" * 60)
    print("  TEST 1: Simple Calculation")
    print("▒" * 60)
    result = run_agent("What is 25 multiplied by 4, then add 17?")
    print(f"\n>>> FINAL ANSWER: {result}")
    
    # Test 2: Current time
    print("\n\n" + "▒" * 60)
    print("  TEST 2: Current Time")
    print("▒" * 60)
    result = run_agent("What day of the week is it today?")
    print(f"\n>>> FINAL ANSWER: {result}")
    
    # Test 3: Multiple tools needed
    print("\n\n" + "▒" * 60)
    print("  TEST 3: Multiple Tools")
    print("▒" * 60)
    result = run_agent(
        "First tell me what time it is, then calculate 100 divided by 4, "
        "and finally tell me how many characters are in the word 'artificial intelligence'."
    )
    print(f"\n>>> FINAL ANSWER: {result}")
    
    # Test 4: No tools needed
    print("\n\n" + "▒" * 60)
    print("  TEST 4: No Tools Needed")
    print("▒" * 60)
    result = run_agent("Tell me a one-sentence fun fact about octopuses.")
    print(f"\n>>> FINAL ANSWER: {result}")
