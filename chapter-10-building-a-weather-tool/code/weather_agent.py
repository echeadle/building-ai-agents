"""
Complete weather agent that can answer questions about current conditions.

This agent integrates the weather tool with Claude, demonstrating the
complete tool use loop. It handles multi-turn conversations and provides
natural language responses about weather.

Chapter 10: Building a Weather Tool
"""

import os
from dotenv import load_dotenv
import anthropic

# Import our weather tool
from weather_tool import get_current_weather, WEATHER_TOOL_DEFINITION

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()

# System prompt for the weather agent
SYSTEM_PROMPT = """You are a helpful weather assistant. Use the get_current_weather tool to answer questions about weather conditions.

Guidelines:
- Always use the weather tool to get current conditions - don't make up weather data
- Be concise but friendly in your responses
- If the user doesn't specify units:
  - Use fahrenheit for US cities
  - Use celsius for cities outside the US
- If the weather tool returns an error, explain the issue helpfully
- You can make reasonable suggestions based on weather (umbrella, jacket, etc.)
- If the user asks about multiple cities, get weather for each one"""

# List of tools available to the agent
TOOLS = [WEATHER_TOOL_DEFINITION]


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return its result.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Dictionary of input parameters
        
    Returns:
        Tool execution result as a string
    """
    if tool_name == "get_current_weather":
        city = tool_input.get("city", "")
        units = tool_input.get("units", "fahrenheit")
        return get_current_weather(city, units)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def chat(user_message: str, verbose: bool = False) -> str:
    """
    Send a message to the weather agent and get a response.
    
    This function handles the complete tool use loop:
    1. Send user message to Claude
    2. If Claude wants to use a tool, execute it
    3. Send tool result back to Claude
    4. Repeat until Claude gives a final response
    
    Args:
        user_message: The user's question or message
        verbose: If True, print intermediate steps
        
    Returns:
        Claude's final response as a string
    """
    # Initialize message history with the user's message
    messages = [{"role": "user", "content": user_message}]
    
    if verbose:
        print(f"\n[User]: {user_message}")
    
    # Initial request to Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )
    
    if verbose:
        print(f"[Claude stop_reason]: {response.stop_reason}")
    
    # Tool use loop - continue while Claude wants to use tools
    while response.stop_reason == "tool_use":
        # Find the tool use block in the response
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
                break
        
        if not tool_use_block:
            break
        
        if verbose:
            print(f"[Tool Call]: {tool_use_block.name}")
            print(f"[Tool Input]: {tool_use_block.input}")
        
        # Execute the requested tool
        tool_result = process_tool_call(
            tool_use_block.name,
            tool_use_block.input
        )
        
        if verbose:
            print(f"[Tool Result]: {tool_result[:100]}...")
        
        # Add Claude's response (with tool use) to message history
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        # Add tool result to message history
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result,
            }]
        })
        
        # Get next response from Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        
        if verbose:
            print(f"[Claude stop_reason]: {response.stop_reason}")
    
    # Extract final text response
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    
    return "I wasn't able to generate a response."


def interactive_chat():
    """
    Run an interactive chat session with the weather agent.
    
    Type 'quit' or 'exit' to end the session.
    Type 'verbose' to toggle verbose mode.
    """
    print("Weather Agent Chat")
    print("=" * 50)
    print("Ask me about the weather in any city!")
    print("Type 'quit' to exit, 'verbose' to toggle debug output")
    print("=" * 50)
    
    verbose = False
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        
        if user_input.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue
        
        response = chat(user_input, verbose=verbose)
        print(f"\nAssistant: {response}")


# Example usage
if __name__ == "__main__":
    # Run some example queries
    print("Weather Agent Demo")
    print("=" * 60)
    
    example_queries = [
        "What's the weather like in Tokyo right now?",
        "Should I bring an umbrella to Seattle today?",
        "What's the temperature in Paris? Use celsius please.",
        "Compare the weather in London and New York.",
    ]
    
    for query in example_queries:
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print("-" * 40)
        response = chat(query)
        print(f"Assistant: {response}")
    
    print(f"\n{'='*60}")
    print("\nStarting interactive mode...")
    print("(Press Enter to continue or Ctrl+C to skip)")
    
    try:
        input()
        interactive_chat()
    except (EOFError, KeyboardInterrupt):
        print("\nDemo complete!")
