"""
A complete multi-tool agent with calculator, weather, and datetime tools.

Chapter 11: Multi-Tool Agents

This is the main code deliverable for Chapter 11. It demonstrates:
- The tools registry pattern for organizing multiple tools
- How to build an agent that uses multiple tools
- Proper handling of tool calls and responses
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# --- Tools Registry ---

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    function: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, name: str, description: str, input_schema: dict, function: Callable[..., Any]) -> None:
        self._tools[name] = Tool(name=name, description=description, input_schema=input_schema, function=function)
    
    def get_definitions(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self._tools.values()]
    
    def execute(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        try:
            return self._tools[name].function(**kwargs)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools


# --- Tool Implementations ---

def calculator(operation: str, a: float, b: float) -> dict:
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
    result = operations[operation](a, b)
    if result is None:
        return {"error": "Division by zero"}
    if isinstance(result, float):
        result = round(result, 10)
    return {"result": result}


def get_current_datetime(timezone: str = None) -> dict:
    try:
        if timezone:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
        else:
            now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "timezone": timezone or "local"
        }
    except Exception as e:
        return {"error": f"Invalid timezone: {str(e)}"}


def get_weather(location: str, units: str = "celsius") -> dict:
    mock_weather = {
        "new york": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Overcast", "humidity": 80},
        "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 70},
        "sydney": {"temp": 18, "condition": "Clear", "humidity": 55},
        "paris": {"temp": 19, "condition": "Mostly sunny", "humidity": 60},
    }
    location_lower = location.lower()
    for city, data in mock_weather.items():
        if city in location_lower:
            temp = data["temp"]
            if units == "fahrenheit":
                temp = round((temp * 9/5) + 32, 1)
            return {"location": location, "temperature": temp, "units": units, "condition": data["condition"], "humidity": data["humidity"]}
    return {"location": location, "temperature": 20 if units == "celsius" else 68, "units": units, "condition": "Data unavailable"}


# --- Registry Setup ---

def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    
    registry.register(
        name="calculator",
        description="Performs basic arithmetic operations (add, subtract, multiply, divide). Use this when the user needs to calculate something.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"], "description": "The arithmetic operation"},
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculator
    )
    
    registry.register(
        name="get_current_datetime",
        description="Gets the current date and time. Use this when the user asks about today's date, the current time, or time in a specific timezone.",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Optional timezone (e.g., 'UTC', 'America/New_York', 'Asia/Tokyo')"}
            },
            "required": []
        },
        function=get_current_datetime
    )
    
    registry.register(
        name="get_weather",
        description="Gets the current weather for a location. Use this when the user asks about weather, temperature, or conditions somewhere.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city or location"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units (default: celsius)"}
            },
            "required": ["location"]
        },
        function=get_weather
    )
    
    return registry


# --- Multi-Tool Agent ---

class MultiToolAgent:
    """An agent that can use multiple tools to answer user questions."""
    
    def __init__(self, registry: ToolRegistry, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.registry = registry
        self.model = model
        self.system_prompt = """You are a helpful assistant with access to several tools:

1. **calculator**: For math operations (add, subtract, multiply, divide)
2. **get_weather**: For current weather information
3. **get_current_datetime**: For current date/time in any timezone

Use tools when they would help answer the question accurately. Be concise and helpful."""
    
    def process_query(self, user_message: str, verbose: bool = True) -> str:
        """Process a user query, using tools as needed."""
        messages = [{"role": "user", "content": user_message}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            tools=self.registry.get_definitions(),
            messages=messages
        )
        
        while response.stop_reason == "tool_use":
            tool_use_block = None
            for block in response.content:
                if block.type == "tool_use":
                    tool_use_block = block
                    break
            
            if not tool_use_block:
                break
            
            if verbose:
                print(f"  [Tool: {tool_use_block.name}]")
                print(f"  [Args: {tool_use_block.input}]")
            
            result = self.registry.execute(tool_use_block.name, **tool_use_block.input)
            
            if verbose:
                print(f"  [Result: {result}]")
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": json.dumps(result)}]
            })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.registry.get_definitions(),
                messages=messages
            )
        
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        
        return "I couldn't generate a response."


def main():
    """Run the multi-tool agent demo."""
    registry = create_registry()
    agent = MultiToolAgent(registry)
    
    print("=" * 60)
    print("Multi-Tool Agent Demo")
    print("=" * 60)
    
    test_queries = [
        "What's 15% of 85?",
        "What's the weather like in Tokyo?",
        "What day of the week is it today?",
        "If something costs $45.99 and I have a 20% off coupon, how much will I pay?",
        "What's the capital of Japan?",  # No tool needed
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"User: {query}")
        print("─" * 60)
        response = agent.process_query(query)
        print(f"\nAgent: {response}")
    
    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    main()
