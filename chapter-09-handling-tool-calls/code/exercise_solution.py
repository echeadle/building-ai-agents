"""
Exercise Solution: Extended Calculator with Power and Modulo

This solution extends the basic calculator with:
1. Power (exponentiation) operation
2. Modulo operation
3. Enhanced error handling

Chapter 9: Handling Tool Calls - Exercise

Try these test queries:
- "What is 2 to the power of 10?"
- "What is 17 modulo 5?"
- "What is 100 divided by 0?"
- "Calculate 3 squared plus 4 squared"
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()


# =============================================================================
# EXTENDED TOOL DEFINITION
# =============================================================================

calculator_tool = {
    "name": "calculator",
    "description": """Perform mathematical operations on two numbers. 

Available operations:
- add: Addition (a + b)
- subtract: Subtraction (a - b)  
- multiply: Multiplication (a × b)
- divide: Division (a ÷ b)
- power: Exponentiation (a raised to the power of b, i.e., a^b)
- modulo: Remainder after division (a mod b)

Use this tool whenever you need to calculate something mathematically.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"],
                "description": "The mathematical operation to perform"
            },
            "a": {
                "type": "number",
                "description": "The first operand (base number for power operation)"
            },
            "b": {
                "type": "number",
                "description": "The second operand (exponent for power operation)"
            }
        },
        "required": ["operation", "a", "b"]
    }
}


# =============================================================================
# EXTENDED CALCULATOR IMPLEMENTATION
# =============================================================================

def calculator(operation: str, a: float, b: float) -> float | str:
    """Perform a mathematical operation on two numbers.
    
    Extended version with power and modulo operations, plus
    comprehensive error handling.
    
    Args:
        operation: The operation to perform
        a: First number
        b: Second number
        
    Returns:
        The result of the operation
        
    Raises:
        ValueError: For invalid operations or mathematical errors
    """
    # Define operations with their implementations
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": _safe_divide,
        "power": _safe_power,
        "modulo": _safe_modulo,
    }
    
    # Check for valid operation
    if operation not in operations:
        raise ValueError(
            f"Unknown operation: '{operation}'. "
            f"Valid operations are: {list(operations.keys())}"
        )
    
    # Execute the operation
    return operations[operation](a, b)


def _safe_divide(a: float, b: float) -> float:
    """Perform division with zero-check.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Result of a / b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError(
            "Division by zero is undefined. "
            "Please provide a non-zero divisor."
        )
    return a / b


def _safe_power(a: float, b: float) -> float:
    """Perform exponentiation with edge case handling.
    
    Args:
        a: Base
        b: Exponent
        
    Returns:
        Result of a raised to the power of b
        
    Raises:
        ValueError: For problematic inputs like 0^negative
    """
    # Handle 0 raised to negative power (undefined)
    if a == 0 and b < 0:
        raise ValueError(
            "Zero raised to a negative power is undefined. "
            "This would result in division by zero."
        )
    
    # Handle negative base with fractional exponent (complex result)
    if a < 0 and not float(b).is_integer():
        raise ValueError(
            f"Cannot raise negative number {a} to fractional power {b}. "
            "This would result in a complex number."
        )
    
    result = a ** b
    
    # Check for overflow
    if result == float('inf'):
        raise ValueError(
            f"Result of {a}^{b} is too large to represent. "
            "Try smaller numbers."
        )
    
    return result


def _safe_modulo(a: float, b: float) -> float:
    """Perform modulo operation with zero-check.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Remainder of a / b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError(
            "Modulo by zero is undefined. "
            "Please provide a non-zero divisor."
        )
    return a % b


# Tool registry
TOOL_FUNCTIONS = {
    "calculator": calculator,
}


# =============================================================================
# THE TOOL USE LOOP (same as example_03, but with extended tool)
# =============================================================================

def process_tool_calls(response, messages: list) -> list:
    """Process all tool calls in a response and update messages."""
    tool_results = []
    
    for block in response.content:
        if block.type == "tool_use":
            tool_name = block.name
            tool_args = block.input
            tool_id = block.id
            
            print(f"  🔧 Tool: {tool_name}")
            print(f"     Args: {tool_args}")
            
            try:
                if tool_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[tool_name](**tool_args)
                    
                    # Format result nicely
                    if isinstance(result, float):
                        if result.is_integer():
                            result_str = str(int(result))
                        else:
                            # Limit decimal places for cleaner output
                            result_str = f"{result:.10g}"
                    else:
                        result_str = str(result)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str
                    })
                    print(f"     ✓ Result: {result_str}")
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Error: Unknown tool '{tool_name}'",
                        "is_error": True
                    })
                    print(f"     ✗ Error: Unknown tool")
                    
            except ValueError as e:
                # Expected errors (division by zero, etc.)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
                print(f"     ✗ Error: {e}")
                
            except Exception as e:
                # Unexpected errors
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Unexpected error: {str(e)}",
                    "is_error": True
                })
                print(f"     ✗ Unexpected error: {e}")
    
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    
    return messages


def chat_with_tools(user_message: str, verbose: bool = True) -> str:
    """Send a message and handle tool calls until we get a final response."""
    messages = [{"role": "user", "content": user_message}]
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"User: {user_message}")
        print("=" * 60)
    
    MAX_ITERATIONS = 10
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        if verbose:
            print(f"\n--- API Call {iteration} ---")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[calculator_tool],
            messages=messages
        )
        
        if verbose:
            print(f"Stop reason: {response.stop_reason}")
        
        if response.stop_reason == "tool_use":
            if verbose:
                print("Claude is using tools...")
            messages = process_tool_calls(response, messages)
        else:
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            if verbose:
                print("\n" + "=" * 60)
                print(f"Claude: {final_response}")
                print("=" * 60)
            
            return final_response
    
    return "Error: Maximum iterations reached."


# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def run_all_demos():
    """Run through all the exercise test cases."""
    
    test_cases = [
        # Power operations
        "What is 2 to the power of 10?",
        "Calculate 5 squared (5 to the power of 2)",
        "What is 10 to the power of 0?",
        
        # Modulo operations  
        "What is 17 modulo 5?",
        "If I divide 100 by 7, what's the remainder?",
        
        # Error handling
        "What is 100 divided by 0?",
        "What is 0 to the power of -1?",
        "What is 17 modulo 0?",
        
        # Complex multi-step
        "Calculate 3 squared plus 4 squared",
        "If 2 to the power of 8 is divided by 16, what do you get?",
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'#' * 60}")
        print(f"TEST CASE {i}")
        print(f"{'#' * 60}")
        chat_with_tools(test)
        
        # Pause between tests for readability
        input("\nPress Enter for next test...")


def interactive_mode():
    """Run in interactive mode for custom queries."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Extended Calculator")
    print("=" * 60)
    print("\nAvailable operations:")
    print("  • add, subtract, multiply, divide")
    print("  • power (exponentiation)")
    print("  • modulo (remainder)")
    print("\nType 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_input:
            chat_with_tools(user_input)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_all_demos()
    else:
        # Run a few quick demos first
        print("Running quick demonstrations...\n")
        
        chat_with_tools("What is 2 to the power of 10?")
        chat_with_tools("What is 17 modulo 5?")
        chat_with_tools("What is 100 divided by 0?")
        
        # Then go interactive
        print("\n" + "-" * 60)
        interactive_mode()
