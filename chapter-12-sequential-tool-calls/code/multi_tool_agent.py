"""
Multi-tool agent with sequential tool calling.

This script demonstrates an agent that has access to multiple tools
(calculator, weather, datetime) and can chain them together to answer
complex questions.

Chapter 12: Sequential Tool Calls
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

# Initialize the client
client = anthropic.Anthropic()

# Tool definitions
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use this for any arithmetic, percentages, or mathematical expressions. Input should be a valid mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '15 * 7', '100 / 4', '2 ** 10', '(100 - 20) * 0.15')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets the current date and time. Use this when you need to know the current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (e.g., 'UTC', 'US/Eastern'). Defaults to local time."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a specified city. Returns temperature in Celsius, conditions, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name (e.g., 'Tokyo', 'New York', 'London')"
                }
            },
            "required": ["city"]
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
    
    elif name == "get_current_time":
        timezone = arguments.get("timezone", "local")
        now = datetime.now()
        return json.dumps({
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone,
            "day_of_week": now.strftime("%A"),
            "date": now.strftime("%B %d, %Y")
        })
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        # Simulated weather data for demonstration
        # In a real application, this would call a weather API
        weather_data = {
            "Tokyo": {"temp": 28, "conditions": "Sunny", "humidity": 65},
            "New York": {"temp": 22, "conditions": "Partly Cloudy", "humidity": 55},
            "London": {"temp": 18, "conditions": "Rainy", "humidity": 80},
            "Paris": {"temp": 24, "conditions": "Clear", "humidity": 45},
            "Sydney": {"temp": 15, "conditions": "Windy", "humidity": 60},
            "Berlin": {"temp": 20, "conditions": "Overcast", "humidity": 70},
            "Los Angeles": {"temp": 30, "conditions": "Sunny", "humidity": 35},
            "Singapore": {"temp": 32, "conditions": "Humid", "humidity": 85},
        }
        if city in weather_data:
            data = weather_data[city]
            return json.dumps({
                "city": city,
                "temperature_celsius": data["temp"],
                "conditions": data["conditions"],
                "humidity_percent": data["humidity"]
            })
        else:
            # Return reasonable defaults for unknown cities
            return json.dumps({
                "city": city,
                "temperature_celsius": 20,
                "conditions": "Unknown",
                "humidity_percent": 50,
                "note": "Data not available for this city, using defaults"
            })
    
    else:
        return f"Error: Unknown tool '{name}'"


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """
    Run the multi-tool agent.
    
    Args:
        user_message: The user's question or request
        max_iterations: Maximum number of iterations (safety limit)
    
    Returns:
        The agent's final response
    """
    messages = [{"role": "user", "content": user_message}]
    
    system_prompt = """You are a helpful assistant with access to several tools:
- calculator: For mathematical calculations
- get_current_time: To get the current date and time
- get_weather: To get weather information for cities

When answering questions:
1. Think about what information you need
2. Use the appropriate tools to gather that information
3. If a question requires multiple pieces of information, gather them all
4. Combine the results to give a complete, helpful answer

Always explain your reasoning and show your work. Be thorough but concise."""

    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print('='*50)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        # Check if we're done
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "text":
                    # Claude often includes reasoning before tool calls
                    if block.text.strip():
                        print(f"Claude's thinking: {block.text[:200]}...")
                elif block.type == "tool_use":
                    print(f"\nTool: {block.name}")
                    print(f"Input: {json.dumps(block.input, indent=2)}")
                    
                    result = execute_tool(block.name, block.input)
                    print(f"Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
        else:
            print(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    return "Maximum iterations reached without completing the task."


def main():
    """Run example queries through the multi-tool agent."""
    
    # Example 1: Weather comparison with calculation
    print("\n" + "="*70)
    print("EXAMPLE 1: Weather comparison with calculations")
    print("="*70)
    
    question1 = """
    What's the weather like in Tokyo and Paris right now? 
    If Tokyo is warmer than Paris, calculate how much warmer it is 
    and what percentage difference that represents.
    """
    print(f"\nUser: {question1}")
    answer1 = run_agent(question1)
    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print('='*50)
    print(answer1)
    
    # Example 2: Multiple tools working together
    print("\n" + "="*70)
    print("EXAMPLE 2: Time and weather")
    print("="*70)
    
    question2 = "What time is it, and what's the weather like in London right now?"
    print(f"\nUser: {question2}")
    answer2 = run_agent(question2)
    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print('='*50)
    print(answer2)
    
    # Example 3: Complex multi-step question
    print("\n" + "="*70)
    print("EXAMPLE 3: Complex multi-step question")
    print("="*70)
    
    question3 = """
    I'm planning to visit 3 cities: New York, London, and Tokyo.
    
    1. What's the current weather in each city?
    2. What's the average temperature across all three cities?
    3. What's the temperature range (difference between hottest and coldest)?
    """
    print(f"\nUser: {question3}")
    answer3 = run_agent(question3)
    print(f"\n{'='*50}")
    print("FINAL ANSWER:")
    print('='*50)
    print(answer3)


if __name__ == "__main__":
    main()
