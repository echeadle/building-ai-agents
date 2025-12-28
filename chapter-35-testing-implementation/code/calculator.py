"""
Calculator tool implementation for testing examples.

Chapter 35: Testing AI Agents - Implementation

This module provides a simple calculator tool that demonstrates
how to build tools that are easy to test.
"""

from typing import Union


def calculator(operation: str, a: float, b: float) -> dict:
    """
    Performs basic arithmetic operations.
    
    This function is designed to be used as a tool by an AI agent.
    It returns a structured response that includes success status,
    result, and helpful error messages.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First operand
        b: Second operand
        
    Returns:
        A dict containing:
        - success (bool): Whether the operation succeeded
        - result (float): The result if successful
        - expression (str): Human-readable expression if successful
        - error (str): Error message if not successful
        
    Examples:
        >>> calculator("add", 5, 3)
        {'success': True, 'result': 8, 'expression': '5 add 3 = 8'}
        
        >>> calculator("divide", 10, 0)
        {'success': False, 'error': 'Cannot divide by zero'}
    """
    # Define available operations
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }
    
    # Validate operation
    if operation not in operations:
        valid_ops = list(operations.keys())
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: {valid_ops}"
        }
    
    # Handle division by zero explicitly
    if operation == "divide" and b == 0:
        return {
            "success": False,
            "error": "Cannot divide by zero"
        }
    
    # Perform the calculation
    result = operations[operation](a, b)
    
    return {
        "success": True,
        "result": result,
        "expression": f"{a} {operation} {b} = {result}"
    }


# Tool definition for the agent
CALCULATOR_TOOL_DEFINITION = {
    "name": "calculator",
    "description": "Performs basic arithmetic operations (add, subtract, multiply, divide). Use this tool when you need to calculate numerical results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {
                "type": "number",
                "description": "The first operand"
            },
            "b": {
                "type": "number",
                "description": "The second operand"
            }
        },
        "required": ["operation", "a", "b"]
    }
}


if __name__ == "__main__":
    # Quick demonstration
    print("Calculator Tool Demonstration")
    print("=" * 40)
    
    test_cases = [
        ("add", 5, 3),
        ("subtract", 10, 4),
        ("multiply", 7, 6),
        ("divide", 20, 4),
        ("divide", 5, 0),
        ("power", 2, 3),
    ]
    
    for op, a, b in test_cases:
        result = calculator(op, a, b)
        if result["success"]:
            print(f"✓ {result['expression']}")
        else:
            print(f"✗ {op}({a}, {b}): {result['error']}")
