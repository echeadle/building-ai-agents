---
chapter: 9
title: "Handling Tool Calls"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 9: Handling Tool Calls

## Introduction

In the previous chapter, you learned how to define tools—giving Claude a menu of capabilities to choose from. But a menu without a kitchen is useless. When Claude decides to use a tool, *someone* has to actually execute it. That someone is you.

This chapter is where the magic happens. You'll learn to detect when Claude wants to use a tool, parse what it's asking for, execute the actual function, and return the results. By the end, you'll have a working calculator that Claude can use to solve math problems—and more importantly, you'll understand the complete tool use loop that powers every AI agent.

Think of yourself as a translator and executor. Claude speaks in structured requests ("please add 5 and 3"), you translate that into real Python code (`5 + 3`), execute it, and report back the answer. This bridge between intent and execution is the foundation of everything we'll build in this book.

## Learning Objectives

By the end of this chapter, you will be able to:

- Detect when Claude's response includes a tool use request
- Parse tool names and arguments from the API response
- Execute Python functions based on Claude's requests
- Return tool results back to Claude in the correct format
- Implement the complete tool use loop from start to finish

## Understanding the Tool Use Response

When you send a message to Claude with tools available, the response structure changes. Instead of (or in addition to) text content, Claude may respond with a **tool use block**. Let's see what this looks like.

### The Response Structure

Here's what a typical tool use response contains:

```python
# When Claude decides to use a tool, the response looks like this:
response = {
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01ABC123",      # Unique ID for this tool call
            "name": "calculator",         # Which tool Claude wants to use
            "input": {                    # The arguments Claude is passing
                "operation": "add",
                "a": 5,
                "b": 3
            }
        }
    ],
    "stop_reason": "tool_use"            # Tells us why Claude stopped
}
```

There are three critical pieces of information here:

1. **`stop_reason`**: When this equals `"tool_use"`, Claude is waiting for you to execute a tool
2. **`content[].type`**: Look for `"tool_use"` blocks in the content array
3. **`content[].id`**: You'll need this ID when returning results to Claude

### Detecting Tool Use

The first step in handling tool calls is detecting when they occur. Here's how:

```python
def has_tool_use(response) -> bool:
    """Check if the response contains any tool use requests."""
    return response.stop_reason == "tool_use"
```

But `stop_reason` alone isn't enough—we need to find and process the actual tool use blocks:

```python
def extract_tool_uses(response) -> list:
    """Extract all tool use blocks from a response."""
    tool_uses = []
    for block in response.content:
        if block.type == "tool_use":
            tool_uses.append(block)
    return tool_uses
```

> **Note:** Claude can request multiple tool calls in a single response. Always iterate through all content blocks, not just the first one.

## Parsing Tool Call Arguments

Once you've detected a tool use request, you need to extract what Claude is asking for. The Anthropic SDK makes this straightforward by parsing the JSON for you.

### Accessing Tool Call Details

Each tool use block contains:

- `id`: A unique identifier for this specific call
- `name`: The name of the tool Claude wants to use
- `input`: A dictionary of arguments

```python
def parse_tool_call(tool_use_block):
    """Parse a tool use block into its components."""
    return {
        "id": tool_use_block.id,
        "name": tool_use_block.name,
        "arguments": tool_use_block.input  # Already a Python dict
    }
```

### Validating Arguments

Before executing a tool, you should validate that the arguments are what you expect:

```python
def validate_calculator_args(arguments: dict) -> tuple[bool, str]:
    """Validate arguments for the calculator tool.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required = ["operation", "a", "b"]
    
    # Check required fields exist
    for field in required:
        if field not in arguments:
            return False, f"Missing required field: {field}"
    
    # Check operation is valid
    valid_operations = ["add", "subtract", "multiply", "divide"]
    if arguments["operation"] not in valid_operations:
        return False, f"Invalid operation: {arguments['operation']}"
    
    # Check a and b are numbers
    if not isinstance(arguments["a"], (int, float)):
        return False, f"Argument 'a' must be a number, got {type(arguments['a'])}"
    if not isinstance(arguments["b"], (int, float)):
        return False, f"Argument 'b' must be a number, got {type(arguments['b'])}"
    
    return True, ""
```

> **💡 Tip:** Always validate tool arguments before execution. Claude is very good at following schemas, but validation catches edge cases and provides clear error messages.

## Executing the Requested Function

Now for the actual execution. You need to map tool names to Python functions and call them with the provided arguments.

### The Calculator Implementation

Let's implement the calculator operations:

```python
def calculator(operation: str, a: float, b: float) -> float:
    """Perform a mathematical operation on two numbers.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
        
    Returns:
        Result of the operation
        
    Raises:
        ValueError: If operation is invalid or division by zero
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

### The Tool Executor Pattern

Rather than using if/else chains for every tool, use a registry pattern:

```python
# Registry mapping tool names to functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    # Add more tools here as we build them
}

def execute_tool(name: str, arguments: dict) -> any:
    """Execute a tool by name with the given arguments.
    
    Args:
        name: The tool name
        arguments: Dictionary of arguments to pass
        
    Returns:
        The result of the tool execution
        
    Raises:
        ValueError: If tool name is not found
    """
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    
    func = TOOL_FUNCTIONS[name]
    return func(**arguments)
```

This pattern scales beautifully. As you add more tools, just register them in the dictionary.

## Returning Results to Claude

After executing a tool, you must send the results back to Claude. This is done by continuing the conversation with a special **tool result** message.

### The Tool Result Format

Here's the structure Claude expects:

```python
tool_result_message = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01ABC123",  # Must match the tool_use id!
            "content": "8"                     # The result as a string
        }
    ]
}
```

Key points:

- The `role` is `"user"` (you're responding to Claude)
- The `tool_use_id` **must match** the `id` from the tool use block
- The `content` should be the result converted to a string

### Formatting Results

Convert your tool results to strings that Claude can understand:

```python
def format_tool_result(result: any) -> str:
    """Format a tool result for Claude.
    
    Args:
        result: The raw result from tool execution
        
    Returns:
        String representation suitable for Claude
    """
    if result is None:
        return "Operation completed successfully (no return value)"
    elif isinstance(result, (dict, list)):
        import json
        return json.dumps(result, indent=2)
    else:
        return str(result)
```

### Handling Errors

When a tool fails, you still need to report back to Claude. Use the `is_error` flag:

```python
def create_tool_result(tool_use_id: str, result: any = None, error: str = None) -> dict:
    """Create a tool result message for Claude.
    
    Args:
        tool_use_id: The ID from the tool use block
        result: The successful result (if no error)
        error: The error message (if failed)
        
    Returns:
        A properly formatted tool result content block
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
```

> **⚠️ Warning:** Always return a tool result for every tool use request, even if the tool failed. If you don't, the conversation will be in an invalid state and the next API call will fail.

## The Complete Tool Use Loop

Now let's put it all together. The complete tool use loop follows this pattern:

1. Send a message to Claude with tools available
2. Check if Claude wants to use a tool
3. If yes: execute the tool and send results back
4. Repeat until Claude responds with text (no more tool calls)

Here's the complete implementation:

```python
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# Tool definition (from Chapter 8)
calculator_tool = {
    "name": "calculator",
    "description": "Perform basic arithmetic operations. Use this when you need to calculate something.",
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

def calculator(operation: str, a: float, b: float) -> float:
    """Execute a calculator operation."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations[operation](a, b)

TOOL_FUNCTIONS = {"calculator": calculator}

def process_tool_calls(response, messages: list) -> list:
    """Process all tool calls in a response and return updated messages."""
    tool_results = []
    
    for block in response.content:
        if block.type == "tool_use":
            tool_name = block.name
            tool_args = block.input
            tool_id = block.id
            
            print(f"  → Executing {tool_name}({tool_args})")
            
            try:
                result = TOOL_FUNCTIONS[tool_name](**tool_args)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(result)
                })
                print(f"  ← Result: {result}")
            except Exception as e:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
                print(f"  ← Error: {e}")
    
    # Add assistant's response and tool results to messages
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    
    return messages

def chat_with_tools(user_message: str) -> str:
    """Send a message and handle any tool calls until we get a final response."""
    messages = [{"role": "user", "content": user_message}]
    
    print(f"User: {user_message}")
    print("-" * 40)
    
    while True:
        # Make API call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[calculator_tool],
            messages=messages
        )
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            print("Claude is using tools...")
            messages = process_tool_calls(response, messages)
        else:
            # No more tool calls - extract final text response
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            print("-" * 40)
            print(f"Claude: {final_response}")
            return final_response

# Example usage
if __name__ == "__main__":
    chat_with_tools("What is 25 multiplied by 17?")
```

When you run this, you'll see:

```
User: What is 25 multiplied by 17?
----------------------------------------
Claude is using tools...
  → Executing calculator({'operation': 'multiply', 'a': 25, 'b': 17})
  ← Result: 425
----------------------------------------
Claude: 25 multiplied by 17 equals 425.
```

## Understanding the Message Flow

Let's trace exactly what happens in the tool use loop:

### Step 1: Initial Request

```python
messages = [
    {"role": "user", "content": "What is 25 multiplied by 17?"}
]
```

### Step 2: Claude Responds with Tool Use

Claude's response:
```python
{
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01ABC123",
            "name": "calculator",
            "input": {"operation": "multiply", "a": 25, "b": 17}
        }
    ],
    "stop_reason": "tool_use"
}
```

### Step 3: We Add to Messages and Execute

```python
messages = [
    {"role": "user", "content": "What is 25 multiplied by 17?"},
    {"role": "assistant", "content": [<tool_use block>]},  # Claude's response
    {"role": "user", "content": [<tool_result block>]}     # Our result
]
```

### Step 4: Claude Provides Final Answer

With the tool result, Claude can now answer:
```python
{
    "content": [
        {"type": "text", "text": "25 multiplied by 17 equals 425."}
    ],
    "stop_reason": "end_turn"
}
```

> **💡 Tip:** Notice that `stop_reason` changes from `"tool_use"` to `"end_turn"` when Claude is done with tools. This is your signal to exit the loop.

## Handling Multiple Tool Calls

Claude might need to use multiple tools to answer a question. Your loop handles this naturally:

```python
# User asks: "What is (15 + 27) * 3?"
# 
# Round 1:
#   Claude calls: calculator(add, 15, 27) → 42
#   
# Round 2:
#   Claude calls: calculator(multiply, 42, 3) → 126
#   
# Round 3:
#   Claude responds: "The result of (15 + 27) * 3 is 126."
```

Each iteration of our `while True` loop handles one round of tool calls. Claude keeps calling tools until it has all the information needed.

## Common Pitfalls

### 1. Forgetting to Return Tool Results

**Problem:** Not sending tool results back causes API errors.

```python
# WRONG - Missing tool result
messages.append({"role": "assistant", "content": response.content})
# Next API call fails because Claude is waiting for results!

# RIGHT - Always include tool results
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": tool_results})
```

### 2. Mismatching Tool Use IDs

**Problem:** The `tool_use_id` in your result must exactly match the `id` from the tool use block.

```python
# WRONG
tool_result = {
    "type": "tool_result",
    "tool_use_id": "some_other_id",  # Wrong ID!
    "content": "42"
}

# RIGHT
tool_result = {
    "type": "tool_result",
    "tool_use_id": block.id,  # Use the exact ID from the tool_use block
    "content": "42"
}
```

### 3. Infinite Loops

**Problem:** If something goes wrong, your loop might never exit.

```python
# Add a safety limit
MAX_ITERATIONS = 10

iteration = 0
while iteration < MAX_ITERATIONS:
    iteration += 1
    response = client.messages.create(...)
    
    if response.stop_reason != "tool_use":
        break
    
    # Process tools...
    
if iteration >= MAX_ITERATIONS:
    print("Warning: Reached maximum iterations")
```

## Practical Exercise

**Task:** Extend the calculator to support more operations and handle edge cases.

**Requirements:**

1. Add support for `power` (exponentiation) and `modulo` operations
2. Add proper error handling for:
   - Division by zero (return a helpful error message)
   - Invalid operations (if somehow an invalid one slips through)
   - Non-numeric inputs
3. Test with questions like:
   - "What is 2 to the power of 10?"
   - "What is 17 modulo 5?"
   - "What is 100 divided by 0?"

**Hints:**

- For power, use Python's `**` operator or `pow()` function
- For modulo, use the `%` operator
- Return errors via the `is_error` field in tool results

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **You are the bridge**: Claude decides what to do, you make it happen
- **Check `stop_reason`**: When it's `"tool_use"`, Claude is waiting for you
- **Match IDs exactly**: The `tool_use_id` in results must match the original request
- **Always respond**: Every tool use request needs a result, even if it's an error
- **Loop until done**: Keep processing until `stop_reason` is not `"tool_use"`
- **Handle errors gracefully**: Report failures back to Claude so it can adapt

## What's Next

You now have a working calculator that Claude can use. But a single tool is limiting—real agents need multiple tools working together. In Chapter 10, we'll build a weather tool that fetches real data from an external API, introducing you to the challenges of working with real-world services. Then in Chapter 11, we'll combine multiple tools into a toolkit, giving your agent real versatility.
