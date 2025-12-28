"""
Exercise Solution: Unit Converter Agent

This solution implements an AugmentedLLM with tools for converting between
different units of measurement: temperature, length, and weight.

Requirements:
1. Tools for temperature (Celsius ↔ Fahrenheit ↔ Kelvin)
2. Tools for length (meters ↔ feet ↔ inches)
3. Tools for weight (kilograms ↔ pounds ↔ ounces)
4. Structured output with numeric result and formula used

Chapter 14: Building the Complete Augmented LLM
"""

import os
import json
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def create_temperature_tool(llm: AugmentedLLM) -> None:
    """Register a temperature conversion tool."""
    
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        """
        Convert temperature between Celsius, Fahrenheit, and Kelvin.
        """
        # Normalize unit names
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        # Map common variations
        unit_map = {
            "c": "celsius", "celsius": "celsius",
            "f": "fahrenheit", "fahrenheit": "fahrenheit",
            "k": "kelvin", "kelvin": "kelvin"
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        # First convert to Celsius as intermediate
        if from_unit == "celsius":
            celsius = value
            formula_to_c = f"{value}°C = {value}°C"
        elif from_unit == "fahrenheit":
            celsius = (value - 32) * 5/9
            formula_to_c = f"({value} - 32) × 5/9 = {celsius:.4g}°C"
        elif from_unit == "kelvin":
            if value < 0:
                return {"error": "Kelvin cannot be negative (absolute zero is 0K)"}
            celsius = value - 273.15
            formula_to_c = f"{value} - 273.15 = {celsius:.4g}°C"
        else:
            return {"error": f"Unknown source unit: {from_unit}"}
        
        # Then convert from Celsius to target
        if to_unit == "celsius":
            result = celsius
            formula_from_c = f"{celsius:.4g}°C"
        elif to_unit == "fahrenheit":
            result = celsius * 9/5 + 32
            formula_from_c = f"{celsius:.4g} × 9/5 + 32 = {result:.4g}°F"
        elif to_unit == "kelvin":
            result = celsius + 273.15
            if result < 0:
                return {"error": f"Result would be below absolute zero ({result:.2f}K)"}
            formula_from_c = f"{celsius:.4g} + 273.15 = {result:.4g}K"
        else:
            return {"error": f"Unknown target unit: {to_unit}"}
        
        # Build complete formula
        if from_unit == to_unit:
            formula = f"{value} {from_unit} = {value} {to_unit} (no conversion needed)"
        else:
            formula = f"{value}°{from_unit[0].upper()} → {result:.4g}°{to_unit[0].upper()}"
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "result_value": round(result, 4),
            "result_unit": to_unit,
            "formula": formula
        }
    
    llm.register_tool(
        name="convert_temperature",
        description="""Convert temperature between Celsius (C), Fahrenheit (F), and Kelvin (K).

Examples:
- 100°F to Celsius: convert_temperature(100, "fahrenheit", "celsius")
- 0°C to Kelvin: convert_temperature(0, "celsius", "kelvin")
- 300K to Fahrenheit: convert_temperature(300, "kelvin", "fahrenheit")

Handles absolute zero validation for Kelvin.

Use this for any temperature conversion questions.""",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The temperature value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin", "c", "f", "k"],
                    "description": "The unit to convert from"
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin", "c", "f", "k"],
                    "description": "The unit to convert to"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        },
        function=convert_temperature
    )


def create_length_tool(llm: AugmentedLLM) -> None:
    """Register a length conversion tool."""
    
    def convert_length(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        """
        Convert length between meters, feet, and inches.
        """
        # Normalize unit names
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        # Map variations
        unit_map = {
            "m": "meters", "meter": "meters", "meters": "meters",
            "ft": "feet", "foot": "feet", "feet": "feet",
            "in": "inches", "inch": "inches", "inches": "inches"
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        if value < 0:
            return {"error": "Length cannot be negative"}
        
        # Conversion factors to meters
        to_meters = {
            "meters": 1,
            "feet": 0.3048,
            "inches": 0.0254
        }
        
        # Conversion factors from meters
        from_meters = {
            "meters": 1,
            "feet": 3.28084,
            "inches": 39.3701
        }
        
        if from_unit not in to_meters:
            return {"error": f"Unknown source unit: {from_unit}"}
        if to_unit not in from_meters:
            return {"error": f"Unknown target unit: {to_unit}"}
        
        # Convert to meters, then to target
        meters = value * to_meters[from_unit]
        result = meters * from_meters[to_unit]
        
        # Build formula description
        if from_unit == to_unit:
            formula = f"{value} {from_unit} (no conversion needed)"
        else:
            formula = f"{value} {from_unit} × {to_meters[from_unit]} = {meters:.4g} m × {from_meters[to_unit]} = {result:.4g} {to_unit}"
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "result_value": round(result, 4),
            "result_unit": to_unit,
            "formula": formula
        }
    
    llm.register_tool(
        name="convert_length",
        description="""Convert length between meters (m), feet (ft), and inches (in).

Conversion factors:
- 1 meter = 3.28084 feet
- 1 meter = 39.3701 inches
- 1 foot = 0.3048 meters
- 1 foot = 12 inches

Examples:
- 2.5 meters to feet: convert_length(2.5, "meters", "feet")
- 6 feet to meters: convert_length(6, "feet", "meters")
- 72 inches to meters: convert_length(72, "inches", "meters")

Use this for any length/distance conversion questions.""",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The length value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "enum": ["meters", "feet", "inches", "m", "ft", "in"],
                    "description": "The unit to convert from"
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["meters", "feet", "inches", "m", "ft", "in"],
                    "description": "The unit to convert to"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        },
        function=convert_length
    )


def create_weight_tool(llm: AugmentedLLM) -> None:
    """Register a weight conversion tool."""
    
    def convert_weight(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        """
        Convert weight between kilograms, pounds, and ounces.
        """
        # Normalize unit names
        from_unit = from_unit.lower().strip()
        to_unit = to_unit.lower().strip()
        
        # Map variations
        unit_map = {
            "kg": "kilograms", "kilogram": "kilograms", "kilograms": "kilograms", "kilo": "kilograms",
            "lb": "pounds", "lbs": "pounds", "pound": "pounds", "pounds": "pounds",
            "oz": "ounces", "ounce": "ounces", "ounces": "ounces"
        }
        
        from_unit = unit_map.get(from_unit, from_unit)
        to_unit = unit_map.get(to_unit, to_unit)
        
        if value < 0:
            return {"error": "Weight cannot be negative"}
        
        # Conversion factors to kilograms
        to_kg = {
            "kilograms": 1,
            "pounds": 0.453592,
            "ounces": 0.0283495
        }
        
        # Conversion factors from kilograms
        from_kg = {
            "kilograms": 1,
            "pounds": 2.20462,
            "ounces": 35.274
        }
        
        if from_unit not in to_kg:
            return {"error": f"Unknown source unit: {from_unit}"}
        if to_unit not in from_kg:
            return {"error": f"Unknown target unit: {to_unit}"}
        
        # Convert to kg, then to target
        kg = value * to_kg[from_unit]
        result = kg * from_kg[to_unit]
        
        # Build formula description
        if from_unit == to_unit:
            formula = f"{value} {from_unit} (no conversion needed)"
        else:
            # Simplified formula for user
            direct_factor = to_kg[from_unit] * from_kg[to_unit]
            formula = f"{value} {from_unit} × {direct_factor:.6g} = {result:.4g} {to_unit}"
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "result_value": round(result, 4),
            "result_unit": to_unit,
            "formula": formula
        }
    
    llm.register_tool(
        name="convert_weight",
        description="""Convert weight/mass between kilograms (kg), pounds (lb), and ounces (oz).

Conversion factors:
- 1 kilogram = 2.20462 pounds
- 1 kilogram = 35.274 ounces
- 1 pound = 0.453592 kilograms
- 1 pound = 16 ounces

Examples:
- 150 pounds to kg: convert_weight(150, "pounds", "kilograms")
- 5 kg to pounds: convert_weight(5, "kilograms", "pounds")
- 8 ounces to grams: convert_weight(8, "ounces", "kilograms")

Use this for any weight/mass conversion questions.""",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The weight value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "enum": ["kilograms", "pounds", "ounces", "kg", "lb", "lbs", "oz"],
                    "description": "The unit to convert from"
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["kilograms", "pounds", "ounces", "kg", "lb", "lbs", "oz"],
                    "description": "The unit to convert to"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        },
        function=convert_weight
    )


def main():
    """Run the unit converter agent."""
    
    print("Unit Converter Agent")
    print("=" * 60)
    
    # Create the agent with appropriate system prompt
    config = AugmentedLLMConfig(
        system_prompt="""You are a helpful unit conversion assistant.

You have access to three conversion tools:
1. convert_temperature - For Celsius, Fahrenheit, and Kelvin
2. convert_length - For meters, feet, and inches
3. convert_weight - For kilograms, pounds, and ounces

When answering conversion questions:
1. Use the appropriate conversion tool
2. Report the numeric result clearly
3. Show the conversion formula/method used
4. Be helpful if the user's question is ambiguous

Always use the tools for conversions - don't try to calculate manually.""",
        max_tokens=1024
    )
    
    llm = AugmentedLLM(config=config)
    
    # Register all conversion tools
    create_temperature_tool(llm)
    create_length_tool(llm)
    create_weight_tool(llm)
    
    print(f"\nRegistered tools: {llm.tools.get_tool_names()}")
    
    # Test queries
    test_queries = [
        # Temperature
        "Convert 100 degrees Fahrenheit to Celsius",
        "What is 0 Kelvin in Fahrenheit?",
        
        # Length
        "How many feet is 2.5 meters?",
        "Convert 6 feet to inches",
        
        # Weight
        "What's 150 pounds in kilograms?",
        "How many ounces in 2 pounds?",
        
        # Edge cases
        "What's -40°F in Celsius? (hint: it's a special number!)",
        "Convert 1 meter to feet and also to inches",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print("-" * 60)
        
        response = llm.run(query)
        print(f"A: {response}")
        
        llm.clear_history()  # Reset for next query
    
    # Interactive mode
    print(f"\n{'='*60}")
    print("\nInteractive Mode (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = llm.run(user_input)
            print(f"\nAnswer: {response}")
            llm.clear_history()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
