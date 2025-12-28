---
chapter: 8
title: "Defining Your First Tool"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 8: Defining Your First Tool

## Introduction

In Chapter 7, we explored why tools matter—they transform Claude from a text generator into an agent that can take action in the world. But understanding *why* tools matter is only the first step. Now it's time to get our hands dirty and learn *how* to define tools that Claude can actually use.

Here's the thing that surprises many developers: **tool definitions are prompts**. When you define a tool, you're not just creating a technical specification—you're writing instructions that Claude will read to decide when and how to use that tool. A poorly written tool definition leads to a confused agent that uses tools at the wrong times or passes incorrect parameters. A well-written definition leads to an agent that feels almost magical in its ability to understand what you want.

In this chapter, we'll dissect the anatomy of a tool definition, learn the JSON Schema basics you need for parameter definitions, and build a simple calculator tool that Claude can use. By the end, you'll understand not just the structure of tool definitions, but the art of writing them well.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the three required components of a tool definition (name, description, input_schema)
- Write JSON Schema definitions for tool parameters using common types
- Craft tool descriptions that help Claude make good decisions about when to use tools
- Define required vs optional parameters appropriately
- Create a complete, working calculator tool definition

## The Anatomy of a Tool Definition

Every tool you give to Claude needs three things: a **name**, a **description**, and an **input_schema**. Let's examine each one.

### The Basic Structure

Here's the simplest possible tool definition:

```python
tool = {
    "name": "get_current_time",
    "description": "Returns the current date and time.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

This tool has no parameters—it just returns the current time when called. Let's break down each piece:

**name**: A unique identifier for the tool. This is what Claude will reference when it decides to use the tool.

**description**: A human-readable explanation of what the tool does. *This is the most important part*—Claude reads this to decide when to use the tool.

**input_schema**: A JSON Schema object that defines what parameters the tool accepts. Even if your tool takes no parameters, you still need this field.

### Tool Names: Keep Them Clear and Consistent

Tool names should be:

- **Descriptive**: `calculate_sum` is better than `calc` or `math_operation`
- **Action-oriented**: Use verbs like `get_`, `create_`, `search_`, `calculate_`
- **Snake_case**: This is the convention Claude expects (e.g., `get_weather`, not `getWeather`)
- **Unique**: No two tools should have the same name

Here are some good tool name examples:

```python
# Good names - clear and action-oriented
"get_current_weather"
"search_documents"
"calculate_mortgage_payment"
"send_email"
"create_calendar_event"

# Poor names - vague or unclear
"weather"        # Is this getting, setting, or displaying weather?
"do_math"        # What kind of math?
"helper"         # Helper for what?
"process"        # Process what, how?
```

### Tool Descriptions: Write Them for Claude

This is where many developers go wrong. They write descriptions like documentation for humans:

```python
# ❌ Too technical, written for humans
{
    "name": "calculate",
    "description": "Performs arithmetic operations. Params: operation (str), a (float), b (float). Returns: float.",
    ...
}
```

Instead, write descriptions that help Claude understand *when* to use the tool and *what* it can accomplish:

```python
# ✅ Written for Claude - explains when and why to use it
{
    "name": "calculate",
    "description": "Performs basic arithmetic calculations. Use this tool when you need to add, subtract, multiply, or divide numbers. This is more reliable than doing mental math, especially for large numbers or decimal calculations.",
    ...
}
```

> **💡 Tip:** Ask yourself: "If I read only this description, would I know exactly when to use this tool?" If not, revise it.

### The Input Schema: JSON Schema Basics

The `input_schema` field uses **JSON Schema**, a standard for describing the structure of JSON data. Don't worry if you haven't used JSON Schema before—we'll cover everything you need.

Here's the basic structure:

```python
"input_schema": {
    "type": "object",
    "properties": {
        "parameter_name": {
            "type": "string",
            "description": "What this parameter is for"
        }
    },
    "required": ["parameter_name"]
}
```

The input schema always has `"type": "object"` at the top level (because parameters are passed as a JSON object). Inside, you define:

- **properties**: The parameters your tool accepts
- **required**: A list of parameter names that must be provided

## Parameter Types in JSON Schema

JSON Schema supports several types that cover most use cases. Let's explore each one with practical examples.

### String Parameters

Strings are the most common parameter type. Use them for text input of any kind:

```python
"properties": {
    "query": {
        "type": "string",
        "description": "The search query to look for"
    },
    "language": {
        "type": "string",
        "description": "The language code (e.g., 'en', 'es', 'fr')",
        "enum": ["en", "es", "fr", "de", "ja"]  # Limit to specific values
    }
}
```

The `enum` keyword restricts the parameter to specific allowed values. This is incredibly useful—Claude will only choose from these options.

### Number Parameters

Use `number` for any numeric value (integers or decimals) and `integer` for whole numbers only:

```python
"properties": {
    "temperature": {
        "type": "number",
        "description": "The temperature in Celsius"
    },
    "count": {
        "type": "integer",
        "description": "The number of results to return (must be a whole number)"
    },
    "price": {
        "type": "number",
        "description": "The price in dollars",
        "minimum": 0,           # Must be non-negative
        "maximum": 10000        # Upper limit
    }
}
```

### Boolean Parameters

Use booleans for true/false flags:

```python
"properties": {
    "include_metadata": {
        "type": "boolean",
        "description": "Whether to include metadata in the response"
    },
    "case_sensitive": {
        "type": "boolean",
        "description": "If true, the search is case-sensitive"
    }
}
```

### Array Parameters

Arrays let you accept lists of values:

```python
"properties": {
    "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "A list of tags to apply to the item"
    },
    "numbers": {
        "type": "array",
        "items": {"type": "number"},
        "description": "A list of numbers to process"
    }
}
```

The `items` field specifies what type each element in the array should be.

### Object Parameters

For complex nested data, use objects:

```python
"properties": {
    "address": {
        "type": "object",
        "description": "The delivery address",
        "properties": {
            "street": {"type": "string"},
            "city": {"type": "string"},
            "zip_code": {"type": "string"}
        },
        "required": ["street", "city"]
    }
}
```

> **⚠️ Warning:** Deeply nested objects make it harder for Claude to construct correct parameters. Keep your schemas as flat as possible. If you find yourself nesting more than two levels deep, consider redesigning your tool interface.

## Required vs Optional Parameters

The `required` array lists parameters that must be provided. Any parameter not in this list is optional:

```python
{
    "name": "search_products",
    "description": "Search for products in the catalog",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term (required)"
            },
            "category": {
                "type": "string",
                "description": "Filter by category (optional)",
                "enum": ["electronics", "clothing", "home", "toys"]
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price filter (optional)"
            },
            "in_stock_only": {
                "type": "boolean",
                "description": "If true, only show items in stock (optional, defaults to false)"
            }
        },
        "required": ["query"]  # Only query is required
    }
}
```

**Guidelines for required vs optional:**

- **Required**: Parameters the tool literally cannot function without
- **Optional**: Parameters that refine behavior but have sensible defaults

For optional parameters, mention the default behavior in the description (e.g., "defaults to false" or "if not specified, returns all results").

## Building a Calculator Tool

Now let's put everything together and build a practical tool: a calculator that can perform basic arithmetic. This is a classic first tool because it demonstrates all the concepts without external dependencies.

### Designing the Tool Interface

Before writing code, think about how you want the tool to work:

1. What operations should it support? (add, subtract, multiply, divide)
2. What inputs does it need? (two numbers and an operation)
3. What edge cases exist? (division by zero)

### The Complete Calculator Tool Definition

```python
calculator_tool = {
    "name": "calculate",
    "description": """Performs basic arithmetic calculations on two numbers. 
Use this tool whenever you need to:
- Add, subtract, multiply, or divide numbers
- Perform calculations that require precision
- Do math with large numbers or decimals

This tool is more reliable than mental math and should be used for any 
non-trivial calculations.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The arithmetic operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "The first number"
            },
            "b": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
}
```

Notice several things about this definition:

1. **The description is thorough**: It explains what the tool does, when to use it, and why it's valuable
2. **Operations are constrained with enum**: Claude can only choose valid operations
3. **Parameter names are clear**: `a` and `b` are simple but unambiguous in context
4. **All parameters are required**: A calculation needs all three inputs

### A More Sophisticated Version

We can make the calculator more capable by supporting additional operations:

```python
advanced_calculator_tool = {
    "name": "calculate",
    "description": """Performs arithmetic and mathematical calculations.

Use this tool when you need to:
- Perform basic arithmetic (add, subtract, multiply, divide)
- Calculate powers or square roots
- Find remainders (modulo operation)

The tool handles edge cases like division by zero and returns clear error messages.
Always prefer this tool over mental math for accuracy.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The mathematical operation to perform",
                "enum": ["add", "subtract", "multiply", "divide", "power", "sqrt", "modulo"]
            },
            "a": {
                "type": "number",
                "description": "The first number (or the only number for sqrt)"
            },
            "b": {
                "type": "number",
                "description": "The second number (not needed for sqrt)"
            }
        },
        "required": ["operation", "a"]
    }
}
```

Here, we've made `b` optional because square root only needs one number. The description mentions this edge case so Claude knows when `b` isn't needed.

## Putting It All Together: Complete Working Code

Let's see the calculator tool in a complete, runnable script. This code defines the tool and sends it to Claude—we'll handle actually *executing* tool calls in Chapter 9.

```python
"""
Demonstrating tool definition with a calculator tool.

Chapter 8: Defining Your First Tool
"""

import os
from dotenv import load_dotenv
import anthropic
import json

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()

# Define our calculator tool
calculator_tool = {
    "name": "calculate",
    "description": """Performs basic arithmetic calculations on two numbers.
Use this tool whenever you need to:
- Add, subtract, multiply, or divide numbers
- Perform calculations that require precision
- Do math with large numbers or decimals

This tool is more reliable than mental math and should be used for any
non-trivial calculations.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The arithmetic operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "The first number"
            },
            "b": {
                "type": "number",
                "description": "The second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
}

# Make a request to Claude with the tool available
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[calculator_tool],
    messages=[
        {
            "role": "user",
            "content": "What is 1,234 multiplied by 5,678?"
        }
    ]
)

# Display the response
print("Response from Claude:")
print(f"Stop reason: {response.stop_reason}")
print()

for block in response.content:
    if block.type == "text":
        print(f"Text: {block.text}")
    elif block.type == "tool_use":
        print(f"Tool call: {block.name}")
        print(f"Arguments: {json.dumps(block.input, indent=2)}")
```

When you run this code, Claude will recognize that it needs to calculate and will request to use the calculator tool:

```
Response from Claude:
Stop reason: tool_use

Tool call: calculate
Arguments: {
  "operation": "multiply",
  "a": 1234,
  "b": 5678
}
```

Notice that `stop_reason` is `"tool_use"`—this tells us Claude wants to use a tool rather than respond with text. The actual execution of the tool (performing the multiplication) happens in our code, which we'll cover in Chapter 9.

## Multiple Tools Example

You're not limited to one tool. Here's how to define multiple tools for Claude to choose from:

```python
tools = [
    {
        "name": "calculate",
        "description": "Performs arithmetic calculations. Use for any math operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Returns the current date and time. Use when asked about the current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "convert_temperature",
        "description": "Converts temperatures between Celsius and Fahrenheit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The temperature value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "The unit to convert from",
                    "enum": ["celsius", "fahrenheit"]
                },
                "to_unit": {
                    "type": "string",
                    "description": "The unit to convert to",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    }
]
```

Claude will intelligently select the appropriate tool based on the user's question. Ask about time? It picks `get_current_time`. Ask for a calculation? It picks `calculate`.

## Common Pitfalls

### 1. Vague Descriptions

```python
# ❌ Bad - Claude won't know when to use this
{"name": "process_data", "description": "Processes data"}

# ✅ Good - Clear about what it does and when to use it
{"name": "analyze_sentiment", "description": "Analyzes text to determine emotional sentiment (positive, negative, neutral). Use when you need to understand the emotional tone of user feedback, reviews, or messages."}
```

### 2. Missing Parameter Descriptions

```python
# ❌ Bad - What are a and b?
"properties": {
    "a": {"type": "number"},
    "b": {"type": "number"}
}

# ✅ Good - Clear parameter purposes
"properties": {
    "a": {"type": "number", "description": "The first number in the calculation"},
    "b": {"type": "number", "description": "The second number in the calculation"}
}
```

### 3. Overly Complex Schemas

```python
# ❌ Bad - Too deeply nested, confusing
"properties": {
    "data": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "preferences": {
                        "type": "object",
                        "properties": {
                            "settings": {...}
                        }
                    }
                }
            }
        }
    }
}

# ✅ Good - Flat structure, easy to understand
"properties": {
    "user_id": {"type": "string", "description": "The user's ID"},
    "preference_name": {"type": "string", "description": "Which preference to update"},
    "preference_value": {"type": "string", "description": "The new value"}
}
```

## Practical Exercise

**Task:** Define a tool for looking up book information

You're building a reading assistant. Create a tool definition that allows Claude to search for books in a library database.

**Requirements:**

- The tool should be named `search_books`
- It should accept a search query (required)
- It should accept optional filters for: author, genre (from a predefined list), and publication year range
- Write a description that helps Claude understand when to use this tool

**Hints:**

- Think about what genres you want to support (fiction, non-fiction, mystery, sci-fi, etc.)
- For the year range, you might want two parameters: `year_from` and `year_to`
- Remember to describe what happens when optional filters aren't provided

**Solution:** See `code/exercise_book_search.py`

## Key Takeaways

- **Tool definitions have three parts**: name, description, and input_schema—all three are essential
- **Descriptions are prompts**: Write them to help Claude understand *when* and *why* to use the tool, not just what it does technically
- **JSON Schema defines parameters**: Use types like string, number, boolean, array, and object to describe your inputs
- **Use enum for constrained choices**: When a parameter should only accept specific values, list them in an enum
- **Keep schemas flat**: Avoid deep nesting—it makes it harder for Claude to construct correct parameters
- **Required vs optional matters**: Mark as required only what's truly necessary; provide defaults for optional parameters

## What's Next

Now that you can define tools, you need to actually *do something* when Claude calls them. In Chapter 9, we'll implement the other half of tool use: detecting when Claude wants to use a tool, executing the appropriate function, and returning results. We'll bring our calculator tool to life and see the complete tool use loop in action.
