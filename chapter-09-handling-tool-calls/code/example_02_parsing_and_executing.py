"""
Parsing and Executing Tool Calls

This script demonstrates how to:
1. Parse tool call arguments from Claude's response
2. Validate the arguments before execution
3. Execute the corresponding Python function
4. Format results for returning to Claude

Chapter 9: Handling Tool Calls
"""

import os
import json
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

# Tool definition
calculator_tool = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations on two numbers.",
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


# =============================================================================
# PARSING TOOL CALLS
# =============================================================================

def parse_tool_call(tool_use_block) -> dict:
    """Parse a tool use block into a convenient dictionary.
    
    The Anthropic SDK already parses the JSON for us, but this function
    provides a consistent interface and adds any additional processing.
    
    Args:
        tool_use_block: A tool_use content block from Claude's response
        
    Returns:
        Dictionary with id, name, and arguments
    """
    return {
        "id": tool_use_block.id,
        "name": tool_use_block.name,
        "arguments": tool_use_block.input  # Already a Python dict
    }


# =============================================================================
# ARGUMENT VALIDATION
# =============================================================================

def validate_calculator_args(arguments: dict) -> tuple[bool, str]:
    """Validate arguments for the calculator tool.
    
    Even though Claude follows schemas well, validation:
    - Catches edge cases
    - Provides clear error messages
    - Makes debugging easier
    
    Args:
        arguments: The arguments dictionary from the tool call
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["operation", "a", "b"]
    valid_operations = ["add", "subtract", "multiply", "divide"]
    
    # Check required fields exist
    for field in required_fields:
        if field not in arguments:
            return False, f"Missing required field: '{field}'"
    
    # Check operation is valid
    if arguments["operation"] not in valid_operations:
        return False, f"Invalid operation: '{arguments['operation']}'. Must be one of: {valid_operations}"
    
    # Check a and b are numbers
    if not isinstance(arguments["a"], (int, float)):
        return False, f"Argument 'a' must be a number, got {type(arguments['a']).__name__}: {arguments['a']}"
    
    if not isinstance(arguments["b"], (int, float)):
        return False, f"Argument 'b' must be a number, got {type(arguments['b']).__name__}: {arguments['b']}"
    
    # Check for division by zero
    if arguments["operation"] == "divide" and arguments["b"] == 0:
        return False, "Cannot divide by zero"
    
    return True, ""


# =============================================================================
# TOOL EXECUTION
# =============================================================================

def calculator(operation: str, a: float, b: float) -> float:
    """Perform a mathematical operation on two numbers.
    
    This is the actual implementation that does the work.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
        
    Returns:
        Result of the operation
        
    Raises:
        ValueError: If operation is invalid
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return operations[operation](a, b)


# Tool registry - maps tool names to functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    # Add more tools here as you build them
}

# Validation registry - maps tool names to validation functions
TOOL_VALIDATORS = {
    "calculator": validate_calculator_args,
}


def execute_tool(name: str, arguments: dict) -> tuple[any, str | None]:
    """Execute a tool by name with the given arguments.
    
    This function:
    1. Looks up the tool function
    2. Optionally validates arguments
    3. Executes the function
    4. Returns result or error
    
    Args:
        name: The tool name
        arguments: Dictionary of arguments to pass
        
    Returns:
        Tuple of (result, error_message)
        - If successful: (result, None)
        - If failed: (None, error_message)
    """
    # Check if tool exists
    if name not in TOOL_FUNCTIONS:
        return None, f"Unknown tool: '{name}'"
    
    # Validate arguments if validator exists
    if name in TOOL_VALIDATORS:
        is_valid, error = TOOL_VALIDATORS[name](arguments)
        if not is_valid:
            return None, error
    
    # Execute the tool
    try:
        func = TOOL_FUNCTIONS[name]
        result = func(**arguments)
        return result, None
    except Exception as e:
        return None, f"Execution error: {str(e)}"


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_tool_result(result: any) -> str:
    """Format a tool result for Claude.
    
    Claude expects results as strings. This function handles
    various Python types appropriately.
    
    Args:
        result: The raw result from tool execution
        
    Returns:
        String representation suitable for Claude
    """
    if result is None:
        return "Operation completed successfully (no return value)"
    elif isinstance(result, bool):
        return "true" if result else "false"
    elif isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    elif isinstance(result, float):
        # Handle floating point nicely
        if result.is_integer():
            return str(int(result))
        return f"{result:.6g}"  # Use general format, max 6 significant digits
    else:
        return str(result)


def create_tool_result(tool_use_id: str, result: any = None, error: str | None = None) -> dict:
    """Create a properly formatted tool result block.
    
    This is what we send back to Claude after executing a tool.
    
    Args:
        tool_use_id: The ID from the original tool_use block (MUST match!)
        result: The successful result (if no error)
        error: The error message (if failed)
        
    Returns:
        A tool result content block ready to send to Claude
    """
    if error:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"Error: {error}",
            "is_error": True
        }
    else:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": format_tool_result(result)
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_parsing():
    """Show how to parse tool calls from a response."""
    
    print("=" * 60)
    print("PARSING TOOL CALLS")
    print("=" * 60)
    
    # Get a response with tool use
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "What is 123 plus 456?"}
        ]
    )
    
    print("\nClaude's response contains tool calls:")
    
    for block in response.content:
        if block.type == "tool_use":
            parsed = parse_tool_call(block)
            print(f"\n  Parsed tool call:")
            print(f"    ID: {parsed['id']}")
            print(f"    Name: {parsed['name']}")
            print(f"    Arguments: {parsed['arguments']}")


def demonstrate_validation():
    """Show how argument validation works."""
    
    print("\n" + "=" * 60)
    print("ARGUMENT VALIDATION")
    print("=" * 60)
    
    test_cases = [
        {"operation": "add", "a": 5, "b": 3},           # Valid
        {"operation": "divide", "a": 10, "b": 0},       # Division by zero
        {"operation": "modulo", "a": 10, "b": 3},       # Invalid operation
        {"operation": "add", "a": "five", "b": 3},      # Invalid type
        {"a": 5, "b": 3},                               # Missing operation
    ]
    
    for args in test_cases:
        is_valid, error = validate_calculator_args(args)
        status = "✓ Valid" if is_valid else f"✗ Invalid: {error}"
        print(f"\n  {args}")
        print(f"    {status}")


def demonstrate_execution():
    """Show the complete parse-validate-execute-format cycle."""
    
    print("\n" + "=" * 60)
    print("COMPLETE EXECUTION CYCLE")
    print("=" * 60)
    
    # Get a response with tool use
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[calculator_tool],
        messages=[
            {"role": "user", "content": "Multiply 99 by 101"}
        ]
    )
    
    for block in response.content:
        if block.type == "tool_use":
            print(f"\n1. Received tool call: {block.name}")
            print(f"   Arguments: {block.input}")
            
            # Parse
            parsed = parse_tool_call(block)
            print(f"\n2. Parsed successfully")
            
            # Validate
            is_valid, error = validate_calculator_args(parsed["arguments"])
            print(f"\n3. Validation: {'passed' if is_valid else f'failed - {error}'}")
            
            if is_valid:
                # Execute
                result, exec_error = execute_tool(parsed["name"], parsed["arguments"])
                
                if exec_error:
                    print(f"\n4. Execution failed: {exec_error}")
                    tool_result = create_tool_result(parsed["id"], error=exec_error)
                else:
                    print(f"\n4. Execution result: {result}")
                    tool_result = create_tool_result(parsed["id"], result=result)
                
                print(f"\n5. Formatted tool result:")
                print(f"   {json.dumps(tool_result, indent=6)}")


def demonstrate_error_handling():
    """Show how errors are handled and reported."""
    
    print("\n" + "=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)
    
    # Simulate various error scenarios
    scenarios = [
        ("Unknown tool", "unknown_tool", {"x": 1}),
        ("Division by zero", "calculator", {"operation": "divide", "a": 10, "b": 0}),
        ("Missing argument", "calculator", {"operation": "add", "a": 5}),
    ]
    
    for scenario_name, tool_name, args in scenarios:
        print(f"\n  Scenario: {scenario_name}")
        print(f"  Tool: {tool_name}, Args: {args}")
        
        result, error = execute_tool(tool_name, args)
        
        if error:
            tool_result = create_tool_result("test_id_123", error=error)
            print(f"  Result: Error returned to Claude")
            print(f"    is_error: {tool_result.get('is_error', False)}")
            print(f"    content: {tool_result['content']}")
        else:
            print(f"  Result: {result}")


if __name__ == "__main__":
    demonstrate_parsing()
    demonstrate_validation()
    demonstrate_execution()
    demonstrate_error_handling()
