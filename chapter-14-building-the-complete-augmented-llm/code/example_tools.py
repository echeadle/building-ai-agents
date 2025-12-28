"""
AugmentedLLM with multiple tools.

This example shows how to register and use multiple tools with the
AugmentedLLM. Tools enable the LLM to take actions in the real world,
like performing calculations, getting the current time, or fetching data.

Chapter 14: Building the Complete Augmented LLM
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def create_calculator_tool(llm: AugmentedLLM) -> None:
    """Register a safe calculator tool."""
    
    # Safe math functions available in the calculator
    safe_functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }
    
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Use eval with restricted builtins for safety
            result = eval(expression, {"__builtins__": {}}, safe_functions)
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    llm.register_tool(
        name="calculator",
        description="""Evaluate mathematical expressions safely.

Supports:
- Basic arithmetic: +, -, *, /, ** (power)
- Functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, pow
- Constants: pi, e

Examples:
- "2 + 2" -> 4
- "sqrt(16)" -> 4.0
- "sin(pi/2)" -> 1.0
- "pow(2, 10)" -> 1024

Use this tool for any mathematical calculations.""",
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
        function=calculate
    )


def create_datetime_tool(llm: AugmentedLLM) -> None:
    """Register a datetime tool."""
    
    def get_datetime() -> str:
        """Get current date and time."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    llm.register_tool(
        name="get_current_datetime",
        description="""Get the current date and time.

Returns the current date and time in YYYY-MM-DD HH:MM:SS format.
Use this when users ask about the current time, date, or day of the week.""",
        parameters={
            "type": "object",
            "properties": {}
        },
        function=get_datetime
    )


def create_string_tool(llm: AugmentedLLM) -> None:
    """Register a string manipulation tool."""
    
    def manipulate_string(text: str, operation: str) -> str:
        """Perform string operations."""
        operations = {
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "title": text.title(),
            "reverse": text[::-1],
            "length": str(len(text)),
            "word_count": str(len(text.split())),
        }
        
        if operation not in operations:
            return f"Unknown operation: {operation}. Available: {list(operations.keys())}"
        
        return operations[operation]
    
    llm.register_tool(
        name="string_tool",
        description="""Perform operations on text strings.

Available operations:
- uppercase: Convert to UPPERCASE
- lowercase: Convert to lowercase  
- title: Convert To Title Case
- reverse: Reverse the string
- length: Count characters
- word_count: Count words

Use this for any text manipulation tasks.""",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to manipulate"
                },
                "operation": {
                    "type": "string",
                    "enum": ["uppercase", "lowercase", "title", "reverse", "length", "word_count"],
                    "description": "The operation to perform"
                }
            },
            "required": ["text", "operation"]
        },
        function=manipulate_string
    )


def main():
    """Demonstrate AugmentedLLM with multiple tools."""
    
    print("AugmentedLLM with Multiple Tools")
    print("=" * 50)
    
    # Create LLM with a helpful system prompt
    config = AugmentedLLMConfig(
        system_prompt="""You are a helpful assistant with access to several tools:

1. calculator - For mathematical calculations
2. get_current_datetime - For current date/time
3. string_tool - For text manipulation

Use these tools whenever they would help answer the user's question accurately.
Always show your work - explain what tool you're using and why."""
    )
    
    llm = AugmentedLLM(config=config)
    
    # Register all tools
    create_calculator_tool(llm)
    create_datetime_tool(llm)
    create_string_tool(llm)
    
    print(f"\nRegistered tools: {llm.tools.get_tool_names()}")
    
    # Test questions that require different tools
    questions = [
        "What's the square root of 144?",
        "What time is it right now?",
        "If I have 15% of 250, how much is that?",
        "How many words are in: 'The quick brown fox jumps over the lazy dog'?",
        "What is sin(pi/4) rounded to 4 decimal places?",
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"Q: {question}")
        print("-" * 50)
        response = llm.run(question)
        print(f"A: {response}")
        llm.clear_history()  # Reset for next question
    
    # Demonstrate multi-tool usage in one query
    print(f"\n{'='*50}")
    print("Multi-tool query:")
    print("-" * 50)
    complex_question = """I need help with a few things:
1. What's 2^10?
2. What's the current date?
3. Reverse the word 'Python'"""
    
    print(f"Q: {complex_question}")
    response = llm.run(complex_question)
    print(f"A: {response}")


if __name__ == "__main__":
    main()
