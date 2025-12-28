"""
Exercise Solution: Trip Planning Agent

This agent helps plan trips by combining:
- Flight price lookup
- Hotel price lookup
- Currency conversion

It demonstrates sequential tool calls to answer complex travel questions.

Chapter 12: Sequential Tool Calls
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable
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


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    {
        "name": "get_flight_price",
        "description": "Gets the round-trip flight price between two cities. Returns price in USD.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": "Departure city (e.g., 'New York', 'London')"
                },
                "to_city": {
                    "type": "string",
                    "description": "Destination city (e.g., 'Paris', 'Tokyo')"
                }
            },
            "required": ["from_city", "to_city"]
        }
    },
    {
        "name": "get_hotel_price",
        "description": "Gets the average hotel price per night in a city. Returns price in USD.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'Paris', 'Tokyo')"
                },
                "stars": {
                    "type": "integer",
                    "description": "Hotel star rating (1-5). Default is 3.",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "convert_currency",
        "description": "Converts an amount from one currency to another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The amount to convert"
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., 'USD', 'EUR', 'JPY')"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., 'USD', 'EUR', 'JPY')"
                }
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use for totals, percentages, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression (e.g., '100 + 200', '500 * 3')"
                }
            },
            "required": ["expression"]
        }
    }
]


# =============================================================================
# Tool Implementations
# =============================================================================

# Mock data for flight prices (round-trip, in USD)
FLIGHT_PRICES = {
    ("New York", "Paris"): 850,
    ("New York", "London"): 750,
    ("New York", "Tokyo"): 1200,
    ("New York", "Los Angeles"): 350,
    ("Los Angeles", "Tokyo"): 950,
    ("Los Angeles", "Paris"): 1100,
    ("London", "Paris"): 150,
    ("London", "Tokyo"): 1000,
    ("Paris", "Tokyo"): 950,
    ("San Francisco", "Tokyo"): 900,
    ("San Francisco", "Paris"): 1050,
}

# Mock data for hotel prices (per night, in USD)
HOTEL_PRICES = {
    "Paris": {1: 50, 2: 80, 3: 150, 4: 250, 5: 450},
    "London": {1: 60, 2: 90, 3: 180, 4: 300, 5: 500},
    "Tokyo": {1: 40, 2: 70, 3: 130, 4: 220, 5: 400},
    "New York": {1: 80, 2: 120, 3: 200, 4: 350, 5: 600},
    "Los Angeles": {1: 70, 2: 100, 3: 170, 4: 280, 5: 480},
    "San Francisco": {1: 90, 2: 130, 3: 220, 4: 380, 5: 650},
}

# Mock exchange rates (to USD)
EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 1.08,      # 1 EUR = 1.08 USD
    "GBP": 1.27,      # 1 GBP = 1.27 USD
    "JPY": 0.0067,    # 1 JPY = 0.0067 USD
    "CAD": 0.74,      # 1 CAD = 0.74 USD
    "AUD": 0.65,      # 1 AUD = 0.65 USD
}


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool and return the result.
    
    Args:
        name: The tool name
        arguments: The tool arguments
    
    Returns:
        The tool result as a JSON string
    """
    if name == "get_flight_price":
        from_city = arguments.get("from_city", "")
        to_city = arguments.get("to_city", "")
        
        # Try both directions (flights are usually priced similarly)
        key = (from_city, to_city)
        reverse_key = (to_city, from_city)
        
        if key in FLIGHT_PRICES:
            price = FLIGHT_PRICES[key]
        elif reverse_key in FLIGHT_PRICES:
            price = FLIGHT_PRICES[reverse_key]
        else:
            # Generate a reasonable price for unknown routes
            price = 800  # Default price
        
        return json.dumps({
            "from": from_city,
            "to": to_city,
            "price_usd": price,
            "type": "round-trip",
            "note": "Economy class"
        })
    
    elif name == "get_hotel_price":
        city = arguments.get("city", "")
        stars = arguments.get("stars", 3)
        
        if city in HOTEL_PRICES:
            prices = HOTEL_PRICES[city]
            price = prices.get(stars, prices[3])  # Default to 3-star if invalid
        else:
            # Default pricing for unknown cities
            base_prices = {1: 60, 2: 90, 3: 160, 4: 270, 5: 450}
            price = base_prices.get(stars, 160)
        
        return json.dumps({
            "city": city,
            "stars": stars,
            "price_per_night_usd": price,
            "note": "Average price"
        })
    
    elif name == "convert_currency":
        amount = arguments.get("amount", 0)
        from_curr = arguments.get("from_currency", "USD").upper()
        to_curr = arguments.get("to_currency", "USD").upper()
        
        # Get rates
        from_rate = EXCHANGE_RATES.get(from_curr, 1.0)
        to_rate = EXCHANGE_RATES.get(to_curr, 1.0)
        
        if from_rate == 0 or to_rate == 0:
            return json.dumps({"error": "Unknown currency"})
        
        # Convert: from_currency -> USD -> to_currency
        usd_amount = amount * from_rate
        converted = usd_amount / to_rate
        
        return json.dumps({
            "original_amount": amount,
            "from_currency": from_curr,
            "converted_amount": round(converted, 2),
            "to_currency": to_curr,
            "exchange_rate": round(from_rate / to_rate, 6)
        })
    
    elif name == "calculator":
        try:
            expression = arguments.get("expression", "")
            # Safety check
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return json.dumps({"error": "Invalid characters"})
            result = eval(expression)
            return json.dumps({
                "expression": expression,
                "result": round(result, 2) if isinstance(result, float) else result
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# Agent Implementation
# =============================================================================

@dataclass
class ToolCall:
    """Record of a tool call."""
    tool_name: str
    arguments: dict
    result: str
    iteration: int
    duration_ms: float


def run_trip_planner(
    user_message: str,
    max_iterations: int = 15,
    verbose: bool = True
) -> dict:
    """
    Run the trip planning agent.
    
    Args:
        user_message: The user's travel question
        max_iterations: Maximum iterations (higher for complex trips)
        verbose: Print debug information
    
    Returns:
        Dictionary with response and execution details
    """
    start_time = time.time()
    messages = [{"role": "user", "content": user_message}]
    tool_calls: list[ToolCall] = []
    tool_history = []  # For loop detection
    
    system_prompt = """You are a helpful travel planning assistant. You have access to:

1. get_flight_price: Look up round-trip flight prices between cities (returns USD)
2. get_hotel_price: Look up hotel prices per night in a city (returns USD)  
3. convert_currency: Convert amounts between currencies (USD, EUR, GBP, JPY, CAD, AUD)
4. calculator: Perform calculations (totals, percentages, etc.)

When helping with trip planning:
- Always gather all necessary prices before calculating totals
- Show your calculations step by step
- If asked for currency conversion, do the conversion at the end
- Be thorough but concise in your final answer

For hotel costs, multiply the nightly rate by the number of nights.
For total trip cost, add flights + total hotel cost."""

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        if verbose:
            print(f"Stop reason: {response.stop_reason}")
        
        # Check if done
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return {
                        "response": block.text,
                        "status": "completed",
                        "iterations": iteration + 1,
                        "tool_calls": len(tool_calls),
                        "duration_seconds": time.time() - start_time,
                        "tool_history": [
                            {"tool": tc.tool_name, "args": tc.arguments}
                            for tc in tool_calls
                        ]
                    }
            return {
                "response": "No response generated.",
                "status": "error",
                "iterations": iteration + 1,
                "tool_calls": len(tool_calls),
                "duration_seconds": time.time() - start_time,
                "tool_history": []
            }
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Loop detection
                    sig = (block.name, json.dumps(block.input, sort_keys=True))
                    tool_history.append(sig)
                    
                    if len(tool_history) >= 6:
                        recent = tool_history[-3:]
                        prev = tool_history[-6:-3]
                        if recent == prev:
                            return {
                                "response": "Agent detected a loop.",
                                "status": "loop_detected",
                                "iterations": iteration + 1,
                                "tool_calls": len(tool_calls),
                                "duration_seconds": time.time() - start_time,
                                "tool_history": []
                            }
                    
                    # Execute tool
                    if verbose:
                        print(f"Tool: {block.name}")
                        print(f"Args: {json.dumps(block.input)}")
                    
                    call_start = time.time()
                    result = execute_tool(block.name, block.input)
                    duration = (time.time() - call_start) * 1000
                    
                    if verbose:
                        print(f"Result: {result}")
                    
                    tool_calls.append(ToolCall(
                        tool_name=block.name,
                        arguments=block.input,
                        result=result,
                        iteration=iteration + 1,
                        duration_ms=duration
                    ))
                    
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
        "tool_calls": len(tool_calls),
        "duration_seconds": time.time() - start_time,
        "tool_history": []
    }


# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    """Run the trip planner with example queries."""
    
    print("="*70)
    print("TRIP PLANNING AGENT - EXERCISE SOLUTION")
    print("="*70)
    
    # Example 1: Simple trip cost
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic trip cost calculation")
    print("="*70)
    
    query1 = """
    What would a 3-night trip from New York to Paris cost in total?
    I need the flight and a 3-star hotel.
    """
    print(f"\nUser: {query1}")
    result1 = run_trip_planner(query1)
    print(f"\n{'='*50}")
    print(f"Status: {result1['status']}")
    print(f"Iterations: {result1['iterations']}")
    print(f"Tool calls: {result1['tool_calls']}")
    print(f"Duration: {result1['duration_seconds']:.2f}s")
    print(f"\nResponse:\n{result1['response']}")
    
    # Example 2: Trip with currency conversion
    print("\n" + "="*70)
    print("EXAMPLE 2: Trip cost with currency conversion")
    print("="*70)
    
    query2 = """
    What would a 3-night trip from New York to Paris cost in total 
    (flights + hotel), and how much is that in Japanese Yen?
    Use a 3-star hotel.
    """
    print(f"\nUser: {query2}")
    result2 = run_trip_planner(query2)
    print(f"\n{'='*50}")
    print(f"Status: {result2['status']}")
    print(f"Iterations: {result2['iterations']}")
    print(f"Tool calls: {result2['tool_calls']}")
    print(f"Duration: {result2['duration_seconds']:.2f}s")
    print(f"\nResponse:\n{result2['response']}")
    
    # Example 3: Comparing destinations
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing two destinations")
    print("="*70)
    
    query3 = """
    I'm in New York and considering either Paris or Tokyo for a 4-night trip.
    Compare the total costs (flights + 4-star hotel) for both destinations.
    Which is cheaper and by how much?
    """
    print(f"\nUser: {query3}")
    result3 = run_trip_planner(query3)
    print(f"\n{'='*50}")
    print(f"Status: {result3['status']}")
    print(f"Iterations: {result3['iterations']}")
    print(f"Tool calls: {result3['tool_calls']}")
    print(f"Duration: {result3['duration_seconds']:.2f}s")
    print(f"\nResponse:\n{result3['response']}")


if __name__ == "__main__":
    main()
