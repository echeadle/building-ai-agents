"""
The Tools Registry Pattern - A clean way to organize multiple tools.

Chapter 11: Multi-Tool Agents

This module provides a reusable ToolRegistry class that:
- Stores tool definitions and implementations together
- Provides a clean interface for registering new tools
- Makes it easy to execute tools by name
- Generates API-compatible tool definitions automatically
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Tool:
    """Represents a tool with its definition and implementation."""
    name: str
    description: str
    input_schema: dict
    function: Callable[..., Any]


class ToolRegistry:
    """
    A registry for managing multiple tools.
    
    Provides a clean interface for:
    - Registering tools with their implementations
    - Getting tool definitions for the API
    - Executing tools by name
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        function: Callable[..., Any]
    ) -> None:
        """Register a tool with its definition and implementation."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            function=function
        )
    
    def get_definitions(self) -> list[dict]:
        """Get all tool definitions in the format expected by the API."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in self._tools.values()
        ]
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with the given arguments."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].function(**kwargs)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"


# --- Tool Implementations ---

def calculator(operation: str, a: float, b: float) -> dict:
    """Perform arithmetic operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
    result = operations[operation](a, b)
    if isinstance(result, str):
        return {"error": result}
    return {"result": result}


def get_current_datetime(timezone: str = None) -> dict:
    """Get the current date and time."""
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
        return {"error": str(e)}


def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location (mock implementation)."""
    mock_weather = {
        "new york": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Overcast", "humidity": 80},
        "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 70},
        "sydney": {"temp": 18, "condition": "Clear", "humidity": 55},
    }
    location_lower = location.lower()
    for city, data in mock_weather.items():
        if city in location_lower:
            temp = data["temp"]
            if units == "fahrenheit":
                temp = (temp * 9/5) + 32
            return {
                "location": location,
                "temperature": temp,
                "units": units,
                "condition": data["condition"],
                "humidity": data["humidity"]
            }
    return {"location": location, "temperature": 20, "units": units, "condition": "Unknown"}


def create_default_registry() -> ToolRegistry:
    """Create a registry with the default set of tools."""
    registry = ToolRegistry()
    
    registry.register(
        name="calculator",
        description="Performs basic arithmetic operations (add, subtract, multiply, divide).",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculator
    )
    
    registry.register(
        name="get_current_datetime",
        description="Gets the current date and time.",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Optional timezone"}
            },
            "required": []
        },
        function=get_current_datetime
    )
    
    registry.register(
        name="get_weather",
        description="Gets the current weather for a location.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        },
        function=get_weather
    )
    
    return registry


def main():
    """Demonstrate the Tools Registry pattern."""
    print("=" * 60)
    print("Tools Registry Pattern Demo")
    print("=" * 60)
    
    registry = create_default_registry()
    
    print(f"\nRegistry: {registry}")
    print(f"Number of tools: {len(registry)}")
    
    print("\n--- Direct Tool Execution ---")
    print(f"calculator(add, 10, 5) = {registry.execute('calculator', operation='add', a=10, b=5)}")
    print(f"get_current_datetime() = {registry.execute('get_current_datetime')}")
    print(f"get_weather(Tokyo) = {registry.execute('get_weather', location='Tokyo')}")


if __name__ == "__main__":
    main()
