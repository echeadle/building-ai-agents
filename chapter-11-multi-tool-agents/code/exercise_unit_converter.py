"""
Exercise Solution: Adding a Unit Converter Tool to the Multi-Tool Agent

Chapter 11: Multi-Tool Agents

The unit_converter tool handles conversions for:
- Length: meters, feet, inches, kilometers, miles
- Weight: kilograms, pounds, ounces
- Temperature: celsius, fahrenheit, kelvin
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


# --- Existing Tools ---

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
    return {"result": round(result, 10) if isinstance(result, float) else result}


def get_current_datetime(timezone: str = None) -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now()
        return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "day_of_week": now.strftime("%A"), "timezone": timezone or "local"}
    except Exception as e:
        return {"error": str(e)}


def get_weather(location: str, units: str = "celsius") -> dict:
    mock_weather = {
        "new york": {"temp": 22, "condition": "Partly cloudy"},
        "london": {"temp": 15, "condition": "Overcast"},
        "tokyo": {"temp": 28, "condition": "Sunny"},
    }
    for city, data in mock_weather.items():
        if city in location.lower():
            temp = data["temp"] if units == "celsius" else round((data["temp"] * 9/5) + 32, 1)
            return {"location": location, "temperature": temp, "units": units, "condition": data["condition"]}
    return {"location": location, "temperature": 20 if units == "celsius" else 68, "condition": "Unknown"}


# --- NEW: Unit Converter Tool ---

def unit_converter(value: float, from_unit: str, to_unit: str, category: str) -> dict:
    """Convert between different units of measurement."""
    
    # Length conversions (base: meters)
    length_to_meters = {
        "meters": 1, "feet": 0.3048, "inches": 0.0254,
        "kilometers": 1000, "miles": 1609.344, "centimeters": 0.01, "yards": 0.9144
    }
    
    # Weight conversions (base: kilograms)
    weight_to_kg = {
        "kilograms": 1, "pounds": 0.453592, "ounces": 0.0283495, "grams": 0.001
    }
    
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    category = category.lower().strip()
    
    try:
        if category == "length":
            if from_unit not in length_to_meters or to_unit not in length_to_meters:
                return {"error": f"Unknown length unit. Valid: {list(length_to_meters.keys())}"}
            meters = value * length_to_meters[from_unit]
            result = meters / length_to_meters[to_unit]
            
        elif category == "weight":
            if from_unit not in weight_to_kg or to_unit not in weight_to_kg:
                return {"error": f"Unknown weight unit. Valid: {list(weight_to_kg.keys())}"}
            kg = value * weight_to_kg[from_unit]
            result = kg / weight_to_kg[to_unit]
            
        elif category == "temperature":
            valid = ["celsius", "fahrenheit", "kelvin"]
            if from_unit not in valid or to_unit not in valid:
                return {"error": f"Unknown temperature unit. Valid: {valid}"}
            
            # Convert to Celsius first
            if from_unit == "celsius":
                c = value
            elif from_unit == "fahrenheit":
                c = (value - 32) * 5/9
            else:  # kelvin
                c = value - 273.15
            
            # Convert from Celsius to target
            if to_unit == "celsius":
                result = c
            elif to_unit == "fahrenheit":
                result = (c * 9/5) + 32
            else:  # kelvin
                result = c + 273.15
        else:
            return {"error": f"Unknown category. Valid: length, weight, temperature"}
        
        return {
            "original_value": value, "original_unit": from_unit,
            "converted_value": round(result, 6), "converted_unit": to_unit
        }
    except Exception as e:
        return {"error": str(e)}


# --- Registry Setup ---

def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    
    registry.register(
        name="calculator",
        description="Performs arithmetic (add, subtract, multiply, divide). NOT for unit conversions.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"}, "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        },
        function=calculator
    )
    
    registry.register(
        name="get_current_datetime",
        description="Gets current date and time.",
        input_schema={
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": []
        },
        function=get_current_datetime
    )
    
    registry.register(
        name="get_weather",
        description="Gets weather for a location.",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        },
        function=get_weather
    )
    
    # NEW: Unit Converter
    registry.register(
        name="unit_converter",
        description="""Converts between units. Categories:
- length: meters, feet, inches, kilometers, miles, centimeters, yards
- weight: kilograms, pounds, ounces, grams
- temperature: celsius, fahrenheit, kelvin

Examples: "5 miles to km", "10 kg in pounds", "100°F to Celsius" """,
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The value to convert"},
                "from_unit": {"type": "string", "description": "Source unit"},
                "to_unit": {"type": "string", "description": "Target unit"},
                "category": {"type": "string", "enum": ["length", "weight", "temperature"]}
            },
            "required": ["value", "from_unit", "to_unit", "category"]
        },
        function=unit_converter
    )
    
    return registry


# --- Agent ---

class MultiToolAgent:
    def __init__(self, registry: ToolRegistry, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.registry = registry
        self.model = model
        self.system_prompt = """You have access to: calculator, get_weather, get_current_datetime, and unit_converter.
Use calculator for math, unit_converter for conversions. Be concise."""
    
    def process_query(self, user_message: str, verbose: bool = True) -> str:
        messages = [{"role": "user", "content": user_message}]
        
        response = self.client.messages.create(
            model=self.model, max_tokens=1024,
            system=self.system_prompt,
            tools=self.registry.get_definitions(),
            messages=messages
        )
        
        while response.stop_reason == "tool_use":
            tool_block = next((b for b in response.content if b.type == "tool_use"), None)
            if not tool_block:
                break
            
            if verbose:
                print(f"  [Tool: {tool_block.name}] [Args: {tool_block.input}]")
            
            result = self.registry.execute(tool_block.name, **tool_block.input)
            if verbose:
                print(f"  [Result: {result}]")
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]})
            
            response = self.client.messages.create(
                model=self.model, max_tokens=1024,
                system=self.system_prompt,
                tools=self.registry.get_definitions(),
                messages=messages
            )
        
        return next((b.text for b in response.content if hasattr(b, "text")), "No response.")


def main():
    registry = create_registry()
    agent = MultiToolAgent(registry)
    
    print("=" * 60)
    print("Multi-Tool Agent with Unit Converter")
    print("=" * 60)
    
    test_queries = [
        "How many kilometers is 5 miles?",
        "Convert 100 pounds to kilograms",
        "What is 98.6°F in Celsius?",
        "How many feet is 100 meters?",
        "What's 50 times 12?",  # Should use calculator
        "What's the weather in Tokyo?",  # Should use weather
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"User: {query}")
        response = agent.process_query(query)
        print(f"Agent: {response}")
    
    print(f"\n{'=' * 60}")
    print("Exercise complete!")


if __name__ == "__main__":
    main()
