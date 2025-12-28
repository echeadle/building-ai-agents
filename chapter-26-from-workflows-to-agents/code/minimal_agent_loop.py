"""
Minimal Agent Loop Implementation

The fundamental agent pattern: perceive → think → act → repeat.
This is the core building block for all agents.

Chapter 26: From Workflows to Agents
"""

import os
from typing import Callable
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# Define the tools available to our agent
tools = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations. Use this for any math operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time. Use this when asked about the current time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a location. Use this for weather questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for"
                }
            },
            "required": ["location"]
        }
    }
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def execute_calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression"
        
        result = eval(expression)  # Safe here due to character whitelist
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def execute_get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def execute_get_weather(location: str) -> str:
    """Simulate getting weather (in real app, call a weather API)."""
    # This is a simulation - in production, you'd call a real weather API
    import random
    conditions = ["sunny", "cloudy", "partly cloudy", "rainy"]
    temp = random.randint(60, 85)
    condition = random.choice(conditions)
    return f"Weather in {location}: {temp}°F, {condition}"


# Tool execution dispatcher
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name with the given input.
    
    This is the bridge between what the LLM asks for and
    what actually gets executed.
    """
    if tool_name == "calculator":
        return execute_calculator(tool_input["expression"])
    elif tool_name == "get_current_time":
        return execute_get_current_time()
    elif tool_name == "get_weather":
        return execute_get_weather(tool_input["location"])
    else:
        return f"Error: Unknown tool '{tool_name}'"


# =============================================================================
# THE AGENT LOOP
# =============================================================================

def run_agent(
    user_request: str,
    system_prompt: str = None,
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    Run the agent loop until completion or max iterations.
    
    This is the fundamental pattern all agents follow:
    1. PERCEIVE: Gather available information
    2. THINK: Let the LLM reason about what to do
    3. ACT: Execute the chosen action
    4. REPEAT: Continue until task is complete
    
    Args:
        user_request: The initial user request
        system_prompt: Optional system prompt to guide agent behavior
        max_iterations: Maximum number of loop iterations (safety limit)
        verbose: Whether to print detailed progress
        
    Returns:
        The agent's final response
    """
    if system_prompt is None:
        system_prompt = """You are a helpful assistant with access to tools.
        
When you need to perform calculations, get the time, or check weather, use the appropriate tool.
Think step by step about what information you need to answer the user's question.
Only provide your final answer when you have all the information you need."""

    # Initialize conversation with user request
    messages = [{"role": "user", "content": user_request}]
    
    if verbose:
        print("=" * 60)
        print("AGENT LOOP STARTED")
        print("=" * 60)
        print(f"\nUser request: {user_request}\n")
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"--- Iteration {iteration}/{max_iterations} ---")
        
        # =====================================================================
        # PHASE 1: PERCEIVE
        # The agent "sees" the full conversation history including tool results
        # =====================================================================
        
        # (Messages already contain everything the agent needs to perceive)
        
        # =====================================================================
        # PHASE 2: THINK
        # The LLM processes the information and decides what to do
        # =====================================================================
        
        if verbose:
            print("  [THINK] Calling LLM...")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        
        if verbose:
            print(f"  [THINK] Stop reason: {response.stop_reason}")
        
        # =====================================================================
        # CHECK: Is the task complete?
        # =====================================================================
        
        if response.stop_reason == "end_turn":
            # The LLM decided it has enough information to respond
            if verbose:
                print("  [COMPLETE] Agent decided task is done")
            
            # Extract the text response
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            
            return "Agent completed but no text response found."
        
        # =====================================================================
        # PHASE 3: ACT
        # Execute any tool calls the LLM requested
        # =====================================================================
        
        if response.stop_reason == "tool_use":
            # Add the assistant's message (with tool calls) to history
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Execute each tool call
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    if verbose:
                        print(f"  [ACT] Executing tool: {tool_name}")
                        print(f"        Input: {tool_input}")
                    
                    # Execute the tool
                    result = execute_tool(tool_name, tool_input)
                    
                    if verbose:
                        print(f"        Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add tool results to messages for next iteration
            # This is how the agent "perceives" the results
            messages.append({
                "role": "user",
                "content": tool_results
            })
        
        # =====================================================================
        # PHASE 4: REPEAT
        # The loop continues, and the agent will perceive the new information
        # =====================================================================
        
        if verbose:
            print()
    
    # Safety: Max iterations reached
    if verbose:
        print("\n[WARNING] Max iterations reached")
    
    return "I wasn't able to complete the task within the allowed number of steps."


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_agent():
    """Run several examples to show the agent in action."""
    
    examples = [
        # Simple: Single tool call
        "What time is it?",
        
        # Medium: Requires calculation
        "If I have 15 apples and give away 7, then buy 12 more, how many do I have?",
        
        # Complex: Multiple tool calls needed
        "What's the weather in New York, and what's 72 degrees Fahrenheit in Celsius? (The formula is (F - 32) * 5/9)",
        
        # No tools needed
        "What is the capital of France?",
    ]
    
    print("\n" + "=" * 70)
    print("MINIMAL AGENT LOOP DEMONSTRATION")
    print("=" * 70)
    
    for i, example in enumerate(examples, 1):
        print(f"\n\n{'='*70}")
        print(f"EXAMPLE {i}")
        print("=" * 70)
        
        result = run_agent(example, verbose=True)
        
        print(f"\n{'─'*40}")
        print(f"FINAL RESULT:")
        print(f"{'─'*40}")
        print(result)


if __name__ == "__main__":
    demonstrate_agent()
