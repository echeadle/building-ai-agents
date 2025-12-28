"""
Agent with infinite loop detection.

This script demonstrates how to detect and prevent infinite loops
in an agentic system. Loops can occur when an agent repeatedly
makes the same tool calls without making progress.

Chapter 12: Sequential Tool Calls
"""

import os
import json
import time
from datetime import datetime
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

# Tool definitions (same as before)
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a specified city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if name == "calculator":
        try:
            expression = arguments.get("expression", "")
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        weather_data = {
            "Tokyo": {"temp": 28, "conditions": "Sunny"},
            "New York": {"temp": 22, "conditions": "Cloudy"},
            "London": {"temp": 18, "conditions": "Rainy"},
        }
        if city in weather_data:
            data = weather_data[city]
            return json.dumps({"city": city, "temperature_celsius": data["temp"]})
        return json.dumps({"city": city, "temperature_celsius": 20, "note": "Default"})
    
    return f"Error: Unknown tool '{name}'"


def detect_loop(tool_history: list[tuple], window: int = 3) -> bool:
    """
    Detect if the agent is stuck in a loop.
    
    A loop is detected when the last N tool calls are identical
    to the N calls before that.
    
    Args:
        tool_history: List of (tool_name, arguments_json) tuples
        window: Number of recent calls to check for repetition
    
    Returns:
        True if a loop is detected
    """
    if len(tool_history) < window * 2:
        return False
    
    recent = tool_history[-window:]
    previous = tool_history[-window*2:-window]
    
    return recent == previous


def detect_repeated_call(tool_history: list[tuple], threshold: int = 3) -> bool:
    """
    Detect if the same tool call has been made too many times.
    
    Args:
        tool_history: List of (tool_name, arguments_json) tuples
        threshold: Number of identical calls to trigger detection
    
    Returns:
        True if repeated calls detected
    """
    if len(tool_history) < threshold:
        return False
    
    # Check if the last N calls are all identical
    last_calls = tool_history[-threshold:]
    return len(set(last_calls)) == 1


def run_agent_with_loop_detection(
    user_message: str,
    max_iterations: int = 10,
    timeout_seconds: float = 60.0,
    loop_window: int = 3
) -> dict:
    """
    Run agent with loop detection and timeout protection.
    
    Args:
        user_message: The user's question or request
        max_iterations: Maximum number of iterations
        timeout_seconds: Maximum time for the entire run
        loop_window: Window size for loop detection
    
    Returns:
        Dictionary with response, status, and diagnostics
    """
    start_time = time.time()
    messages = [{"role": "user", "content": user_message}]
    tool_history = []  # Track tool calls for loop detection
    
    system_prompt = "You are a helpful assistant with access to calculator and weather tools."
    
    for iteration in range(max_iterations):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            return {
                "response": "Agent timed out.",
                "status": "timeout",
                "iterations": iteration + 1,
                "tool_calls": len(tool_history),
                "duration_seconds": elapsed
            }
        
        print(f"\nIteration {iteration + 1} (elapsed: {elapsed:.1f}s)")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        # Check if Claude is done
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
                    break
            
            return {
                "response": final_text or "No response generated.",
                "status": "completed",
                "iterations": iteration + 1,
                "tool_calls": len(tool_history),
                "duration_seconds": time.time() - start_time
            }
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Create a signature for this tool call (for loop detection)
                    call_signature = (
                        block.name,
                        json.dumps(block.input, sort_keys=True)
                    )
                    tool_history.append(call_signature)
                    
                    print(f"  Tool: {block.name}({block.input})")
                    
                    # Check for loops BEFORE executing
                    if detect_loop(tool_history, window=loop_window):
                        return {
                            "response": "Agent detected a repeating pattern and stopped.",
                            "status": "loop_detected",
                            "iterations": iteration + 1,
                            "tool_calls": len(tool_history),
                            "duration_seconds": time.time() - start_time,
                            "loop_pattern": tool_history[-loop_window*2:]
                        }
                    
                    if detect_repeated_call(tool_history, threshold=4):
                        return {
                            "response": "Agent made the same call too many times.",
                            "status": "repeated_call",
                            "iterations": iteration + 1,
                            "tool_calls": len(tool_history),
                            "duration_seconds": time.time() - start_time,
                            "repeated_call": tool_history[-1]
                        }
                    
                    # Execute the tool
                    result = execute_tool(block.name, block.input)
                    print(f"  Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return {
        "response": "Maximum iterations reached.",
        "status": "max_iterations",
        "iterations": max_iterations,
        "tool_calls": len(tool_history),
        "duration_seconds": time.time() - start_time
    }


def main():
    """Demonstrate loop detection with various queries."""
    
    # Example 1: Normal query (should complete successfully)
    print("="*60)
    print("EXAMPLE 1: Normal query")
    print("="*60)
    
    result = run_agent_with_loop_detection(
        "What's 25 * 4 and what's the weather in Tokyo?"
    )
    print(f"\nStatus: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {result['tool_calls']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print(f"\nResponse: {result['response']}")
    
    # Example 2: Multi-city query
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-city weather comparison")
    print("="*60)
    
    result = run_agent_with_loop_detection(
        "Compare the weather in Tokyo, New York, and London. Which is warmest?"
    )
    print(f"\nStatus: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Tool calls: {result['tool_calls']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print(f"\nResponse: {result['response']}")


if __name__ == "__main__":
    main()
