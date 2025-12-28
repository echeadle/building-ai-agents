---
chapter: 11
title: "Multi-Tool Agents"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 11: Multi-Tool Agents

## Introduction

In the previous chapters, you learned how to define a tool, handle tool calls, and build a practical weather tool. But a single tool is like a toolbox with only a hammer—useful for nails, but not much else.

Real agents need multiple tools. A personal assistant agent might need to check your calendar, search the web, send emails, and perform calculations—all in the same conversation. The magic happens when Claude can seamlessly choose the right tool for each part of a request.

In this chapter, you'll learn how to give Claude access to multiple tools at once and how it decides which tool to use. You'll also learn a clean pattern for organizing tools as your toolkit grows—the **tools registry pattern**. By the end, you'll have an agent equipped with calculator, weather, and datetime tools that can handle a wide variety of questions.

## Learning Objectives

By the end of this chapter, you will be able to:

- Provide multiple tools to Claude in a single API request
- Understand how Claude selects which tool to use
- Design tools that complement each other
- Implement the tools registry pattern for clean tool organization
- Build a multi-tool agent that handles diverse requests

## Providing Multiple Tools

Providing multiple tools to Claude is straightforward—you simply include all tool definitions in the `tools` parameter as a list. Claude will examine the user's request and decide which tool (if any) to use.

Let's start by combining the calculator tool from Chapter 8 with a simple datetime tool:

```python
"""
Demonstrating multiple tools in a single request.

Chapter 11: Multi-Tool Agents
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# Define multiple tools
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Use this when the user needs to calculate something.",
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
    },
    {
        "name": "get_current_datetime",
        "description": "Gets the current date and time. Use this when the user asks about today's date, the current time, or what day of the week it is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (e.g., 'UTC', 'US/Eastern'). Defaults to local time."
                }
            },
            "required": []
        }
    }
]

# Make a request with multiple tools available
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's 15% of 85?"}
    ]
)

print(f"Stop reason: {message.stop_reason}")
for block in message.content:
    print(f"Content block: {block}")
```

When you run this, Claude recognizes the calculation request and chooses the calculator tool—even though the datetime tool is also available. The response will include a `tool_use` block requesting the calculator with appropriate arguments.

## How Claude Chooses Between Tools

Claude's tool selection process is driven by three factors:

1. **Tool descriptions**: Claude reads your descriptions to understand what each tool does
2. **User intent**: Claude analyzes what the user is asking for
3. **Parameter matching**: Claude considers whether it has the information needed to call a tool

Let's see this in action with different queries:

```python
"""
Demonstrating how Claude selects tools based on user queries.

Chapter 11: Multi-Tool Agents
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Use this when the user needs to calculate something.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {"type": "number", "description": "The first operand"},
                "b": {"type": "number", "description": "The second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Gets the current date and time. Use this when the user asks about today's date, the current time, or what day of the week it is.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (e.g., 'UTC', 'US/Eastern')"
                }
            },
            "required": []
        }
    }
]


def test_tool_selection(query: str) -> None:
    """Test which tool Claude selects for a given query."""
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": query}]
    )
    
    if message.stop_reason == "tool_use":
        for block in message.content:
            if block.type == "tool_use":
                print(f"Selected tool: {block.name}")
                print(f"Arguments: {block.input}")
    else:
        print("No tool selected - Claude responded directly")
        for block in message.content:
            if hasattr(block, "text"):
                print(f"Response: {block.text[:100]}...")


# Test different queries
test_tool_selection("What's 234 times 56?")
test_tool_selection("What day of the week is it?")
test_tool_selection("What's the capital of France?")  # No tool needed
test_tool_selection("What time is it in Tokyo?")
```

Running this demonstrates:
- **Math queries** → Calculator tool
- **Date/time queries** → Datetime tool
- **General knowledge** → No tool (Claude answers directly)

> **Note:** Claude is remarkably good at understanding intent. "What's 15% of 85?" gets routed to the calculator even though the user didn't say "calculate." Similarly, "What day is it?" routes to datetime even without the word "date."

## Writing Complementary Tool Descriptions

When you have multiple tools, their descriptions should clearly differentiate when each should be used. Ambiguous descriptions lead to inconsistent tool selection.

Here's an example of **poor** tool descriptions:

```python
# BAD: Overlapping, vague descriptions
tools = [
    {
        "name": "search",
        "description": "Searches for information",  # Too vague!
        # ...
    },
    {
        "name": "lookup",
        "description": "Looks up data",  # How is this different from search?
        # ...
    }
]
```

And here's the **improved** version:

```python
# GOOD: Clear, specific, complementary descriptions
tools = [
    {
        "name": "web_search",
        "description": "Searches the internet for current information, news, and web pages. Use this for questions about recent events, current prices, or information that changes frequently.",
        # ...
    },
    {
        "name": "database_lookup",
        "description": "Looks up records in the company database by ID or name. Use this for internal company data like employee records, inventory counts, or order history.",
        # ...
    }
]
```

**Tips for writing complementary descriptions:**

1. **Be specific about what each tool does**: "Performs arithmetic" vs. "Gets weather data"
2. **Specify when to use it**: "Use this when the user needs calculations"
3. **Mention what it doesn't do**: "For weather forecasts, not historical weather data"
4. **Avoid overlapping language**: Don't use "searches" for multiple tools

## The Tools Registry Pattern

As your toolkit grows, managing tool definitions and their implementations becomes unwieldy. The **tools registry pattern** solves this by creating a central registry that:

1. Stores tool definitions and their implementations together
2. Provides a clean interface for registering new tools
3. Makes it easy to execute tools by name

Here's the pattern:

```python
"""
The Tools Registry Pattern - A clean way to organize multiple tools.

Chapter 11: Multi-Tool Agents
"""

from typing import Any, Callable
from dataclasses import dataclass


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
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
```

The registry pattern keeps your code organized and makes it easy to add new tools—just call `registry.register()` with the new tool's details.

## Building the Multi-Tool Agent

Now let's combine everything into a complete multi-tool agent that can handle a variety of requests. The full implementation is in `code/example_04_multi_tool_agent.py`, but here's the core agent class:

```python
class MultiToolAgent:
    """
    An agent that can use multiple tools to answer user questions.
    """
    
    def __init__(self, registry: ToolRegistry, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.registry = registry
        self.model = model
        self.system_prompt = """You are a helpful assistant with access to tools for calculations, weather information, and date/time queries.

When the user asks a question:
1. If it requires a calculation, use the calculator tool
2. If it's about weather, use the get_weather tool
3. If it's about the current date or time, use the get_current_datetime tool
4. If it doesn't require any tools, answer directly from your knowledge

Always be helpful and concise in your responses."""
    
    def process_query(self, user_message: str) -> str:
        """Process a user query, using tools as needed."""
        messages = [{"role": "user", "content": user_message}]
        
        # Make the initial request
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            tools=self.registry.get_definitions(),
            messages=messages
        )
        
        # Handle tool use if needed
        while response.stop_reason == "tool_use":
            # Find and execute the tool
            tool_use_block = next(
                (b for b in response.content if b.type == "tool_use"), 
                None
            )
            
            if not tool_use_block:
                break
            
            result = self.registry.execute(
                tool_use_block.name,
                **tool_use_block.input
            )
            
            # Continue the conversation with the tool result
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": json.dumps(result)
                }]
            })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.registry.get_definitions(),
                messages=messages
            )
        
        # Extract final text response
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        
        return "I couldn't generate a response."
```

When you run this agent, you'll see how it seamlessly selects the appropriate tool for each query:

```
User: What's 15% of 85?
  [Using tool: calculator]
  [Arguments: {'operation': 'multiply', 'a': 85, 'b': 0.15}]
  [Result: {'result': 12.75}]
Agent: 15% of 85 is 12.75.

User: What's the weather like in Tokyo?
  [Using tool: get_weather]
  [Arguments: {'location': 'Tokyo'}]
  [Result: {'location': 'Tokyo', 'temperature': 28, 'units': 'celsius', ...}]
Agent: The current weather in Tokyo is sunny with a temperature of 28°C...

User: What's the capital of Japan?
Agent: The capital of Japan is Tokyo.
```

## Organizing Tools in Separate Modules

As your toolkit grows, you'll want to organize tools into separate files. Here's a recommended structure:

```
project/
├── tools/
│   ├── __init__.py
│   ├── registry.py      # The ToolRegistry class
│   ├── calculator.py    # Calculator tool
│   ├── weather.py       # Weather tool
│   └── datetime_tool.py # DateTime tool
└── agent.py             # Main agent code
```

Each tool module exports its registration function:

```python
# tools/calculator.py
"""Calculator tool for arithmetic operations."""

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
    return {"result": operations[operation](a, b)}


TOOL_DEFINITION = {
    "name": "calculator",
    "description": "Performs basic arithmetic operations (add, subtract, multiply, divide). Use this when the user needs to calculate something.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform"
            },
            "a": {"type": "number", "description": "The first operand"},
            "b": {"type": "number", "description": "The second operand"}
        },
        "required": ["operation", "a", "b"]
    }
}


def register(registry):
    """Register the calculator tool with the given registry."""
    registry.register(
        name=TOOL_DEFINITION["name"],
        description=TOOL_DEFINITION["description"],
        input_schema=TOOL_DEFINITION["input_schema"],
        function=calculator
    )
```

Then in your main agent file:

```python
# agent.py
from tools.registry import ToolRegistry
from tools import calculator, weather, datetime_tool

# Create and populate the registry
registry = ToolRegistry()
calculator.register(registry)
weather.register(registry)
datetime_tool.register(registry)

# Now use the registry with your agent
```

This modular approach keeps your codebase clean and makes it easy to add, remove, or modify tools without touching the core agent logic.

## Common Pitfalls

### 1. Overlapping Tool Descriptions

**Problem:** Two tools have similar descriptions, causing inconsistent tool selection.

```python
# BAD: Which should Claude use for "What time is it in NYC?"
{
    "name": "get_time",
    "description": "Gets the current time"
},
{
    "name": "get_datetime",
    "description": "Gets the current date and time"
}
```

**Solution:** Either combine the tools or make descriptions clearly distinct:

```python
# GOOD: Clear distinction
{
    "name": "get_current_time",
    "description": "Gets ONLY the current time (hours:minutes:seconds). For just the time, not the date."
},
{
    "name": "get_full_datetime",
    "description": "Gets the complete current date AND time. Use when you need both date and time together."
}
```

### 2. Forgetting to Handle Unknown Tools

**Problem:** The agent crashes when it receives an unknown tool name.

```python
# BAD: Will crash if tool not found
result = registry._tools[tool_name].function(**kwargs)
```

**Solution:** Always validate and handle errors:

```python
# GOOD: Graceful error handling
def execute(self, name: str, **kwargs) -> Any:
    if name not in self._tools:
        return {"error": f"Unknown tool: {name}"}
    try:
        return self._tools[name].function(**kwargs)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}
```

### 3. Tool Explosion

**Problem:** Adding too many tools makes it hard for Claude to choose correctly.

**Solution:** 
- Start with fewer, more versatile tools
- Group related functionality into single tools with options
- Test tool selection accuracy as you add tools
- Consider routing to different tool sets for different domains (we'll cover this in Chapter 19)

## Practical Exercise

**Task:** Extend the multi-tool agent with a new `unit_converter` tool.

**Requirements:**
- The tool should convert between common units:
  - Length: meters, feet, inches, kilometers, miles
  - Weight: kilograms, pounds, ounces
  - Temperature: celsius, fahrenheit, kelvin
- Register it with the existing registry
- Test that Claude correctly chooses it for conversion queries

**Hints:**
- Use an enum for the "category" parameter (length, weight, temperature)
- Include clear examples in the description
- Make sure the description doesn't overlap with the calculator tool

**Solution:** See `code/exercise_unit_converter.py`

## Key Takeaways

- **Multiple tools are provided as a list** in the `tools` parameter—Claude examines all of them for each request
- **Tool descriptions drive selection**—write them for Claude to read, with clear use cases
- **Complementary descriptions avoid confusion**—each tool should have a distinct purpose
- **The tools registry pattern** keeps code organized as your toolkit grows
- **More tools = more capability, but also more complexity**—start simple and add tools as needed

## What's Next

Your agent can now use multiple tools, but it only makes one tool call at a time. What happens when answering a question requires multiple steps—like calculating a tip, then converting the currency? In Chapter 12, you'll learn about **sequential tool calls** and how to implement the agentic loop that lets Claude chain multiple tool calls together to accomplish complex tasks.
