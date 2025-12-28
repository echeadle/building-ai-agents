"""
Reusable tool collections for the AugmentedLLM.

This module provides pre-built tool collections that can be reused
across different agent projects. Each collection focuses on a specific
domain (math, datetime, text manipulation, etc.).

Chapter 14: Building the Complete Augmented LLM
"""

import os
import math
import json
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from augmented_llm (same directory)
from augmented_llm import ToolRegistry


def create_math_tools() -> ToolRegistry:
    """
    Create a registry with mathematical tools.
    
    Includes:
    - calculate: Evaluate mathematical expressions
    - statistics: Calculate basic statistics on a list of numbers
    
    Returns:
        A ToolRegistry with math tools registered
    """
    registry = ToolRegistry()
    
    # Safe math functions available in the calculator
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    
    def safe_calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            # Format nicely
            if isinstance(result, float):
                # Avoid scientific notation for reasonable numbers
                if abs(result) < 1e10 and abs(result) > 1e-10:
                    return f"{result:.10g}"
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: Invalid expression - {str(e)}"
    
    registry.register(
        name="calculate",
        description="""Evaluate a mathematical expression safely.

Supported operations:
- Basic arithmetic: +, -, *, /, ** (power), // (floor div), % (modulo)
- Functions: sqrt, sin, cos, tan, asin, acos, atan, log, log10, log2, exp, floor, ceil
- Constants: pi, e

Examples:
- "2 + 2" -> 4
- "sqrt(16)" -> 4.0
- "sin(pi/2)" -> 1.0
- "log(e)" -> 1.0
- "pow(2, 10)" -> 1024
- "round(3.14159, 2)" -> 3.14

Use this for any mathematical calculations.""",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        function=safe_calculate
    )
    
    def compute_statistics(numbers: list[float]) -> dict[str, Any]:
        """Compute basic statistics on a list of numbers."""
        if not numbers:
            return {"error": "Empty list provided"}
        
        n = len(numbers)
        sorted_nums = sorted(numbers)
        
        mean = sum(numbers) / n
        
        # Median
        if n % 2 == 0:
            median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
        else:
            median = sorted_nums[n//2]
        
        # Variance and std dev
        variance = sum((x - mean) ** 2 for x in numbers) / n
        std_dev = math.sqrt(variance)
        
        return {
            "count": n,
            "sum": sum(numbers),
            "mean": round(mean, 4),
            "median": median,
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers),
            "variance": round(variance, 4),
            "std_dev": round(std_dev, 4)
        }
    
    registry.register(
        name="statistics",
        description="""Calculate basic statistics on a list of numbers.

Returns: count, sum, mean, median, min, max, range, variance, standard deviation.

Use this when you need to analyze numerical data.""",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers to analyze"
                }
            },
            "required": ["numbers"]
        },
        function=compute_statistics
    )
    
    return registry


def create_datetime_tools() -> ToolRegistry:
    """
    Create a registry with date and time tools.
    
    Includes:
    - get_current_datetime: Get current date/time
    - format_date: Format dates
    - date_difference: Calculate days between dates
    - add_days: Add/subtract days from a date
    
    Returns:
        A ToolRegistry with datetime tools registered
    """
    registry = ToolRegistry()
    
    registry.register(
        name="get_current_datetime",
        description="""Get the current date and time.

Returns datetime in ISO format (YYYY-MM-DDTHH:MM:SS) and also broken down into components.

Use this when users ask about the current time, date, or day of the week.""",
        parameters={
            "type": "object",
            "properties": {}
        },
        function=lambda: {
            "iso": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "day_of_week": datetime.now().strftime("%A"),
            "timezone": "local"
        }
    )
    
    def format_date(date_string: str, output_format: str) -> str:
        """Format a date string into a different format."""
        try:
            # Try parsing as ISO format first
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]:
                    try:
                        dt = datetime.strptime(date_string, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return f"Error: Could not parse date '{date_string}'"
            except Exception:
                return f"Error: Could not parse date '{date_string}'"
        
        try:
            return dt.strftime(output_format)
        except Exception as e:
            return f"Error: Invalid format string - {str(e)}"
    
    registry.register(
        name="format_date",
        description="""Format a date string into a different format.

Common format codes:
- %Y: 4-digit year (2024)
- %m: Month as number (01-12)
- %d: Day of month (01-31)
- %B: Full month name (January)
- %A: Full weekday name (Monday)
- %H: Hour 24h (00-23)
- %M: Minutes (00-59)
- %S: Seconds (00-59)

Examples:
- "%B %d, %Y" -> "January 15, 2024"
- "%m/%d/%Y" -> "01/15/2024"
- "%A, %B %d" -> "Monday, January 15"

Input date can be ISO format (2024-01-15) or common formats.""",
        parameters={
            "type": "object",
            "properties": {
                "date_string": {
                    "type": "string",
                    "description": "The date to format (ISO format preferred: YYYY-MM-DD)"
                },
                "output_format": {
                    "type": "string",
                    "description": "The desired output format using strftime codes"
                }
            },
            "required": ["date_string", "output_format"]
        },
        function=format_date
    )
    
    def date_difference(date1: str, date2: str) -> dict[str, Any]:
        """Calculate the difference between two dates."""
        try:
            d1 = datetime.fromisoformat(date1)
            d2 = datetime.fromisoformat(date2)
        except ValueError as e:
            return {"error": f"Could not parse dates: {str(e)}"}
        
        delta = d2 - d1
        
        return {
            "days": delta.days,
            "weeks": delta.days // 7,
            "remaining_days": delta.days % 7,
            "total_seconds": int(delta.total_seconds()),
            "description": f"{abs(delta.days)} days {'after' if delta.days > 0 else 'before'}"
        }
    
    registry.register(
        name="date_difference",
        description="""Calculate the difference between two dates.

Returns the number of days, weeks, and total seconds between the dates.
Use this to answer questions like "How many days until..." or "How long ago was..."

Dates should be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).""",
        parameters={
            "type": "object",
            "properties": {
                "date1": {
                    "type": "string",
                    "description": "The first date (ISO format)"
                },
                "date2": {
                    "type": "string",
                    "description": "The second date (ISO format)"
                }
            },
            "required": ["date1", "date2"]
        },
        function=date_difference
    )
    
    def add_days(date_string: str, days: int) -> dict[str, str]:
        """Add or subtract days from a date."""
        try:
            dt = datetime.fromisoformat(date_string)
            new_dt = dt + timedelta(days=days)
            return {
                "original": dt.strftime("%Y-%m-%d"),
                "new_date": new_dt.strftime("%Y-%m-%d"),
                "new_day_of_week": new_dt.strftime("%A"),
                "days_added": days
            }
        except ValueError as e:
            return {"error": f"Could not parse date: {str(e)}"}
    
    registry.register(
        name="add_days",
        description="""Add or subtract days from a date.

Use positive numbers to add days, negative to subtract.
Use this for questions like "What's the date 30 days from now?" or "What was the date 2 weeks ago?"

Date should be in ISO format (YYYY-MM-DD).""",
        parameters={
            "type": "object",
            "properties": {
                "date_string": {
                    "type": "string",
                    "description": "The starting date (ISO format: YYYY-MM-DD)"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to add (negative to subtract)"
                }
            },
            "required": ["date_string", "days"]
        },
        function=add_days
    )
    
    return registry


def create_text_tools() -> ToolRegistry:
    """
    Create a registry with text manipulation tools.
    
    Includes:
    - text_transform: Various text transformations
    - word_count: Count words and characters
    - find_replace: Find and replace in text
    
    Returns:
        A ToolRegistry with text tools registered
    """
    registry = ToolRegistry()
    
    def text_transform(text: str, operation: str) -> str:
        """Perform various text transformations."""
        operations = {
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "title": text.title(),
            "capitalize": text.capitalize(),
            "reverse": text[::-1],
            "reverse_words": " ".join(text.split()[::-1]),
            "strip": text.strip(),
            "slug": text.lower().replace(" ", "-"),
        }
        
        if operation not in operations:
            return f"Unknown operation '{operation}'. Available: {list(operations.keys())}"
        
        return operations[operation]
    
    registry.register(
        name="text_transform",
        description="""Transform text in various ways.

Available operations:
- uppercase: Convert to UPPERCASE
- lowercase: Convert to lowercase
- title: Convert To Title Case
- capitalize: Capitalize first letter only
- reverse: Reverse all characters
- reverse_words: Reverse word order
- strip: Remove leading/trailing whitespace
- slug: Convert to url-slug-format

Use this for any text transformation needs.""",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to transform"
                },
                "operation": {
                    "type": "string",
                    "enum": ["uppercase", "lowercase", "title", "capitalize", 
                            "reverse", "reverse_words", "strip", "slug"],
                    "description": "The transformation to apply"
                }
            },
            "required": ["text", "operation"]
        },
        function=text_transform
    )
    
    def word_count(text: str) -> dict[str, Any]:
        """Count words, characters, and more in text."""
        words = text.split()
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        return {
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", "")),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
            "average_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
        }
    
    registry.register(
        name="word_count",
        description="""Count words, characters, sentences, and paragraphs in text.

Returns detailed statistics about the text including:
- Character count (with and without spaces)
- Word count
- Sentence count
- Paragraph count
- Average word length

Use this for text analysis and statistics.""",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                }
            },
            "required": ["text"]
        },
        function=word_count
    )
    
    def find_replace(text: str, find: str, replace: str, case_sensitive: bool = True) -> dict[str, Any]:
        """Find and replace text."""
        if case_sensitive:
            count = text.count(find)
            new_text = text.replace(find, replace)
        else:
            import re
            pattern = re.compile(re.escape(find), re.IGNORECASE)
            count = len(pattern.findall(text))
            new_text = pattern.sub(replace, text)
        
        return {
            "original_length": len(text),
            "new_length": len(new_text),
            "replacements_made": count,
            "result": new_text
        }
    
    registry.register(
        name="find_replace",
        description="""Find and replace text within a string.

Can do case-sensitive or case-insensitive replacement.
Returns the new text and count of replacements made.

Use this for text editing and bulk replacements.""",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to search in"
                },
                "find": {
                    "type": "string",
                    "description": "The text to find"
                },
                "replace": {
                    "type": "string",
                    "description": "The replacement text"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether to match case (default: true)"
                }
            },
            "required": ["text", "find", "replace"]
        },
        function=find_replace
    )
    
    return registry


def merge_registries(*registries: ToolRegistry) -> ToolRegistry:
    """
    Merge multiple tool registries into one.
    
    If tools have the same name in different registries, the later
    registry's version will overwrite the earlier one.
    
    Args:
        *registries: ToolRegistry instances to merge
        
    Returns:
        A new ToolRegistry containing all tools from input registries
    """
    merged = ToolRegistry()
    
    for registry in registries:
        for tool_def in registry.get_definitions():
            name = tool_def["name"]
            # Access internal state to copy both definition and function
            merged._tools[name] = tool_def
            merged._functions[name] = registry._functions[name]
    
    return merged


# Convenience function to create a registry with all standard tools
def create_all_standard_tools() -> ToolRegistry:
    """
    Create a registry with all standard tool collections.
    
    Includes: math, datetime, and text tools.
    
    Returns:
        A ToolRegistry with all standard tools
    """
    return merge_registries(
        create_math_tools(),
        create_datetime_tools(),
        create_text_tools()
    )


if __name__ == "__main__":
    # Demo the tool collections
    print("Tool Collections Demo")
    print("=" * 50)
    
    # Create all tools
    all_tools = create_all_standard_tools()
    
    print(f"\nRegistered {len(all_tools)} tools:")
    for name in all_tools.get_tool_names():
        print(f"  - {name}")
    
    # Test a few tools
    print("\n--- Testing Tools ---")
    
    print(f"\ncalculate('sqrt(16)'): {all_tools.execute('calculate', {'expression': 'sqrt(16)'})}")
    print(f"\nget_current_datetime(): {all_tools.execute('get_current_datetime', {})}")
    print(f"\nword_count('Hello world'): {all_tools.execute('word_count', {'text': 'Hello world'})}")
