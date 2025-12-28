---
chapter: 14
title: "Building the Complete Augmented LLM"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 14: Building the Complete Augmented LLM

## Introduction

You've learned a lot over the past seven chapters. You can define tools, handle tool calls, chain multiple calls together, and parse structured outputs. But right now, all that knowledge exists as scattered code snippets and patterns. Every time you start a new project, you'd have to reassemble these pieces from scratch.

That changes now.

In this chapter, we're taking everything you've learned in Part 2 and assembling it into a single, reusable class: the `AugmentedLLM`. This isn't just cleanup—it's the foundation that everything else in this book builds upon. The workflow patterns in Part 3, the autonomous agents in Part 4, the production systems in Part 5—they all start with this building block.

Think of the `AugmentedLLM` as a souped-up version of a basic API call. It's an LLM that can:
- Follow a system prompt that defines its behavior
- Use tools to take actions in the real world  
- Return structured, validated responses
- Handle the entire tool-use loop automatically

By the end of this chapter, you'll have a clean, well-documented class that you can import into any project and start building agents immediately.

## Learning Objectives

By the end of this chapter, you will be able to:

- Design a class architecture that integrates tools, system prompts, and structured output
- Implement a complete tool-use loop with automatic iteration
- Create a flexible configuration system for customizing agent behavior
- Build a reusable foundation that supports all future agent development
- Write tests that verify your building block works correctly

## The Architecture of an Augmented LLM

Before we write any code, let's design our class. Good architecture makes everything easier later.

### What We're Building

Our `AugmentedLLM` class needs to handle three main responsibilities:

1. **Configuration**: Store settings like the model name, system prompt, and available tools
2. **Execution**: Make API calls and handle the tool-use loop
3. **Response Processing**: Parse results and optionally validate against schemas

Here's a visual representation of how data flows through our class:

```
                    ┌─────────────────────────────────────┐
                    │         AugmentedLLM                │
                    │                                     │
    User Message    │  ┌─────────────┐                   │
    ───────────────►│  │   System    │                   │
                    │  │   Prompt    │                   │
                    │  └──────┬──────┘                   │
                    │         │                          │
                    │         ▼                          │
                    │  ┌─────────────┐    ┌──────────┐  │
                    │  │   Claude    │◄──►│  Tools   │  │
                    │  │    API      │    │ Registry │  │
                    │  └──────┬──────┘    └──────────┘  │
                    │         │                          │
                    │         ▼                          │
                    │  ┌─────────────┐                   │
                    │  │  Response   │                   │    Response
                    │  │  Parser     │──────────────────►│───────────►
                    │  └─────────────┘                   │
                    │                                     │
                    └─────────────────────────────────────┘
```

### Design Decisions

Let's make some deliberate choices about our design:

**1. Tools as a Registry**

Rather than passing tool definitions every time, we'll maintain a registry that maps tool names to both their definitions and their implementation functions. This keeps everything organized and makes it impossible to define a tool without implementing it.

**2. Automatic Tool Loop**

When Claude wants to use a tool, our class handles the entire cycle automatically: execute the tool, send the result back, and continue until Claude is done. The caller doesn't need to manage this complexity.

**3. Optional Structured Output**

Not every use case needs JSON responses. We'll make structured output optional—when you provide a schema, responses are validated; when you don't, you get plain text.

**4. Immutable Configuration**

Once you create an `AugmentedLLM` instance, its configuration is fixed. Want different settings? Create a new instance. This prevents subtle bugs from configuration changes mid-conversation.

**5. Conversation History Management**

The class maintains conversation history internally but also allows you to provide your own. This supports both simple one-shot calls and complex multi-turn interactions.

## The Tool Registry

Let's start building. First, we need a clean way to manage tools.

In earlier chapters, we kept tool definitions and implementations separate. That works for learning, but in production code, it's error-prone. If you add a tool definition but forget the implementation (or vice versa), things break in confusing ways.

Our `ToolRegistry` class solves this by requiring both at registration time:

```python
"""
Tool registry for managing tool definitions and implementations.

Chapter 14: Building the Complete Augmented LLM
"""

from typing import Callable, Any


class ToolRegistry:
    """
    A registry that maps tool names to their definitions and implementations.
    
    This ensures every defined tool has an implementation and every
    implementation has a definition—no mismatches possible.
    """
    
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._functions: dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable
    ) -> None:
        """
        Register a tool with its definition and implementation.
        
        Args:
            name: Unique identifier for the tool
            description: What the tool does (Claude reads this!)
            parameters: JSON Schema defining the tool's parameters
            function: The Python function that implements the tool
        """
        # Store the definition in Claude's expected format
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": parameters
        }
        
        # Store the implementation
        self._functions[name] = function
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions in the format Claude expects."""
        return list(self._tools.values())
    
    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given arguments.
        
        Args:
            name: The tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            The tool's return value
            
        Raises:
            KeyError: If the tool doesn't exist
        """
        if name not in self._functions:
            raise KeyError(f"Unknown tool: {name}")
        
        return self._functions[name](**arguments)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
```

This design has several benefits:

- **Single source of truth**: Tool names, definitions, and implementations all live together
- **Type safety**: The registry enforces that implementations are callable
- **Easy iteration**: `get_definitions()` returns exactly what Claude needs
- **Safe execution**: `execute()` raises a clear error for unknown tools

### Using the Registry

Here's how you'd register the calculator tool from Chapter 9:

```python
registry = ToolRegistry()

registry.register(
    name="calculator",
    description="Perform basic arithmetic operations. Use this for any math calculations.",
    parameters={
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
    },
    function=lambda operation, a, b: {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero"
    }.get(operation, f"Unknown operation: {operation}")
)
```

## Configuration with Dataclasses

Next, we need a clean way to configure our `AugmentedLLM`. Python's `dataclass` decorator is perfect for this—it gives us type hints, default values, and immutability in a concise package.

```python
"""
Configuration for the AugmentedLLM class.

Chapter 14: Building the Complete Augmented LLM
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AugmentedLLMConfig:
    """
    Configuration for an AugmentedLLM instance.
    
    Attributes:
        model: The Claude model to use
        max_tokens: Maximum tokens in the response
        system_prompt: Instructions that define the LLM's behavior
        max_tool_iterations: Safety limit for tool use loops
        temperature: Randomness in responses (0.0 to 1.0)
        response_schema: Optional JSON Schema for structured output
    """
    
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful assistant."
    max_tool_iterations: int = 10
    temperature: float = 1.0
    response_schema: dict[str, Any] | None = None
```

The `frozen=True` parameter makes instances immutable—once created, they can't be modified. This prevents accidental configuration changes that could cause subtle bugs.

> **💡 Tip:** The `max_tool_iterations` setting is crucial. Without it, a buggy tool or confusing prompt could cause Claude to loop forever. Start with a conservative limit (10) and increase only if you have a genuine need.

## The Complete AugmentedLLM Class

Now let's build the main class. This is the heart of everything we've learned in Part 2:

```python
"""
The AugmentedLLM class - a complete building block for agent development.

Chapter 14: Building the Complete Augmented LLM
"""

import os
import json
from typing import Any
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


@dataclass(frozen=True)
class AugmentedLLMConfig:
    """Configuration for an AugmentedLLM instance."""
    
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful assistant."
    max_tool_iterations: int = 10
    temperature: float = 1.0
    response_schema: dict[str, Any] | None = None


class ToolRegistry:
    """Registry for tool definitions and implementations."""
    
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._functions: dict[str, callable] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: callable
    ) -> None:
        """Register a tool with its definition and implementation."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": parameters
        }
        self._functions[name] = function
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions."""
        return list(self._tools.values())
    
    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        if name not in self._functions:
            raise KeyError(f"Unknown tool: {name}")
        return self._functions[name](**arguments)
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)


class AugmentedLLM:
    """
    An LLM enhanced with tools, system prompts, and structured output.
    
    This class is the fundamental building block for all agent development.
    It handles:
    - System prompts for behavior configuration
    - Tool registration and automatic execution
    - Multi-turn tool use loops
    - Optional structured output with validation
    
    Example:
        llm = AugmentedLLM()
        llm.register_tool(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {}},
            function=lambda: datetime.now().isoformat()
        )
        response = llm.run("What time is it?")
    """
    
    def __init__(
        self,
        config: AugmentedLLMConfig | None = None,
        tools: ToolRegistry | None = None
    ):
        """
        Initialize an AugmentedLLM instance.
        
        Args:
            config: Configuration options (uses defaults if not provided)
            tools: Pre-configured tool registry (creates empty one if not provided)
        """
        self.config = config or AugmentedLLMConfig()
        self.tools = tools or ToolRegistry()
        self._client = anthropic.Anthropic()
        self._conversation_history: list[dict[str, Any]] = []
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: callable
    ) -> None:
        """
        Register a tool that the LLM can use.
        
        Args:
            name: Unique identifier for the tool
            description: What the tool does (write this for Claude!)
            parameters: JSON Schema for the tool's parameters
            function: The Python function that implements the tool
        """
        self.tools.register(name, description, parameters, function)
    
    def run(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Send a message and get a response, handling any tool calls automatically.
        
        This is the main entry point for interacting with the LLM. It:
        1. Sends your message to Claude
        2. If Claude wants to use tools, executes them automatically
        3. Continues until Claude provides a final text response
        4. Optionally validates the response against a schema
        
        Args:
            message: The user's message
            conversation_history: Optional history for multi-turn conversations
            
        Returns:
            The LLM's text response (or validated JSON string if schema provided)
            
        Raises:
            RuntimeError: If max tool iterations exceeded
            ValueError: If response doesn't match required schema
        """
        # Initialize conversation
        if conversation_history is not None:
            messages = conversation_history.copy()
        else:
            messages = self._conversation_history.copy()
        
        # Add the new user message
        messages.append({"role": "user", "content": message})
        
        # Build the API request
        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        
        # Add tools if any are registered
        if len(self.tools) > 0:
            request_params["tools"] = self.tools.get_definitions()
        
        # The main loop: call API, handle tools, repeat until done
        iterations = 0
        
        while iterations < self.config.max_tool_iterations:
            iterations += 1
            
            # Make the API call
            response = self._client.messages.create(**request_params)
            
            # Check if we're done (no tool use)
            if response.stop_reason == "end_turn":
                # Extract the text response
                text_response = self._extract_text(response)
                
                # Update internal history
                messages.append({"role": "assistant", "content": response.content})
                self._conversation_history = messages
                
                # Validate against schema if provided
                if self.config.response_schema is not None:
                    return self._validate_response(text_response)
                
                return text_response
            
            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant's response (includes tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})
                
                # Process each tool call
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        result = self._execute_tool(
                            content_block.name,
                            content_block.input,
                            content_block.id
                        )
                        tool_results.append(result)
                
                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})
                
                # Update request for next iteration
                request_params["messages"] = messages
            else:
                # Unexpected stop reason
                text_response = self._extract_text(response)
                self._conversation_history = messages
                return text_response
        
        # If we get here, we've exceeded max iterations
        raise RuntimeError(
            f"Exceeded maximum tool iterations ({self.config.max_tool_iterations}). "
            "The LLM may be stuck in a loop. Check your tool implementations and prompts."
        )
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> dict[str, Any]:
        """
        Execute a tool and format the result for Claude.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Arguments for the tool
            tool_use_id: Unique ID for this tool use
            
        Returns:
            A tool_result block for the messages array
        """
        try:
            result = self.tools.execute(tool_name, tool_input)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                result = json.dumps(result)
            
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result
            }
        except Exception as e:
            # Return error as tool result so Claude can handle it
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error executing tool: {str(e)}",
                "is_error": True
            }
    
    def _extract_text(self, response: anthropic.types.Message) -> str:
        """Extract text content from a response."""
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
    
    def _validate_response(self, response: str) -> str:
        """
        Validate and parse a response against the configured schema.
        
        Args:
            response: The raw text response
            
        Returns:
            The validated JSON string
            
        Raises:
            ValueError: If response doesn't match schema
        """
        try:
            # Parse the JSON
            data = json.loads(response)
            
            # Basic schema validation (checking required fields and types)
            schema = self.config.response_schema
            
            if "required" in schema:
                for field in schema["required"]:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")
            
            if "properties" in schema:
                for field, field_schema in schema["properties"].items():
                    if field in data:
                        self._validate_field(field, data[field], field_schema)
            
            # Return the validated JSON
            return json.dumps(data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}")
    
    def _validate_field(self, name: str, value: Any, schema: dict) -> None:
        """Validate a single field against its schema."""
        expected_type = schema.get("type")
        
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        if expected_type in type_mapping:
            if not isinstance(value, type_mapping[expected_type]):
                raise ValueError(
                    f"Field '{name}' should be {expected_type}, got {type(value).__name__}"
                )
        
        # Check enum values
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(
                f"Field '{name}' must be one of {schema['enum']}, got '{value}'"
            )
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history = []
    
    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of the conversation history."""
        return self._conversation_history.copy()
```

This class encapsulates everything we've built in Part 2:

- **System prompts** (Chapter 6): Configured via `AugmentedLLMConfig.system_prompt`
- **Tool definitions** (Chapter 8): Managed by `ToolRegistry`
- **Tool execution** (Chapter 9-11): Handled automatically in the `run()` loop
- **Sequential tool calls** (Chapter 12): The while loop continues until Claude is done
- **Structured output** (Chapter 13): Optional validation via `response_schema`

## Using the AugmentedLLM Class

Let's see our building block in action. We'll start simple and build up to more complex examples.

### Example 1: Basic Usage (No Tools)

The simplest use case—just a configured LLM:

```python
"""
Basic usage of AugmentedLLM without tools.

Chapter 14: Building the Complete Augmented LLM
"""

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def main():
    # Create with custom system prompt
    config = AugmentedLLMConfig(
        system_prompt="You are a helpful coding assistant. Be concise and practical."
    )
    
    llm = AugmentedLLM(config=config)
    
    # Simple query
    response = llm.run("What's the difference between a list and a tuple in Python?")
    print(response)


if __name__ == "__main__":
    main()
```

### Example 2: With Tools

Now let's add some tools:

```python
"""
AugmentedLLM with multiple tools.

Chapter 14: Building the Complete Augmented LLM
"""

import math
from datetime import datetime

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def main():
    # Create with a helpful assistant persona
    config = AugmentedLLMConfig(
        system_prompt="""You are a helpful assistant with access to tools for 
        math calculations and checking the current time. Use these tools whenever 
        they would help answer the user's question accurately."""
    )
    
    llm = AugmentedLLM(config=config)
    
    # Register a calculator tool
    llm.register_tool(
        name="calculator",
        description="Perform mathematical calculations. Supports basic arithmetic and common functions.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        },
        function=lambda expression: str(eval(expression, {"__builtins__": {}}, {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
        }))
    )
    
    # Register a time tool
    llm.register_tool(
        name="get_current_time",
        description="Get the current date and time. Use this when users ask about the current time or date.",
        parameters={
            "type": "object",
            "properties": {}
        },
        function=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Test with questions that require tools
    questions = [
        "What's the square root of 144?",
        "What time is it right now?",
        "If I have 15% of 250, how much is that?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = llm.run(question)
        print(f"A: {response}")
        llm.clear_history()  # Reset for next question


if __name__ == "__main__":
    main()
```

> **⚠️ Warning:** The calculator example uses `eval()` for simplicity. In production, never use `eval()` with untrusted input! Use a proper expression parser like `numexpr` or `simpleeval` instead.

### Example 3: Structured Output

For cases where you need predictable, parseable responses:

```python
"""
AugmentedLLM with structured output validation.

Chapter 14: Building the Complete Augmented LLM
"""

import json

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def main():
    # Define the response schema
    sentiment_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number"
            },
            "reasoning": {
                "type": "string"
            }
        },
        "required": ["sentiment", "confidence", "reasoning"]
    }
    
    # Create LLM with structured output
    config = AugmentedLLMConfig(
        system_prompt="""You are a sentiment analysis assistant. Analyze the 
        sentiment of user messages and respond with a JSON object containing:
        - sentiment: "positive", "negative", or "neutral"
        - confidence: A number from 0 to 1 indicating your confidence
        - reasoning: A brief explanation of your analysis
        
        Respond ONLY with the JSON object, no other text.""",
        response_schema=sentiment_schema
    )
    
    llm = AugmentedLLM(config=config)
    
    # Analyze some text
    texts = [
        "I absolutely love this product! Best purchase ever!",
        "The service was okay, nothing special.",
        "Terrible experience. Would not recommend to anyone."
    ]
    
    for text in texts:
        print(f"\nText: {text}")
        response = llm.run(f"Analyze this text: {text}")
        result = json.loads(response)
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
        print(f"Reasoning: {result['reasoning']}")
        llm.clear_history()


if __name__ == "__main__":
    main()
```

### Example 4: Multi-Turn Conversation

The `AugmentedLLM` maintains conversation history automatically:

```python
"""
Multi-turn conversation with AugmentedLLM.

Chapter 14: Building the Complete Augmented LLM
"""

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def main():
    config = AugmentedLLMConfig(
        system_prompt="""You are a friendly assistant helping someone learn Python.
        Remember context from earlier in the conversation and build on it.
        Keep explanations clear and provide examples when helpful."""
    )
    
    llm = AugmentedLLM(config=config)
    
    # Multi-turn conversation
    conversation = [
        "I'm trying to learn about Python data structures. What are the main ones?",
        "Can you tell me more about dictionaries?",
        "How would I use one to count word frequencies in a text?",
    ]
    
    for user_message in conversation:
        print(f"\nYou: {user_message}")
        response = llm.run(user_message)
        print(f"\nAssistant: {response}")
    
    # You can inspect the history
    print(f"\n--- Conversation had {len(llm.get_history())} messages ---")


if __name__ == "__main__":
    main()
```

## Creating Reusable Tool Collections

As you build more agents, you'll want to reuse tools across projects. Let's create a pattern for organizing tool collections:

```python
"""
Reusable tool collections for the AugmentedLLM.

Chapter 14: Building the Complete Augmented LLM
"""

from datetime import datetime
import math

from augmented_llm import ToolRegistry


def create_math_tools() -> ToolRegistry:
    """Create a registry with mathematical tools."""
    registry = ToolRegistry()
    
    # Safe math evaluation
    def safe_eval(expression: str) -> str:
        allowed_names = {
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
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    registry.register(
        name="calculate",
        description="Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **) and functions like sqrt, sin, cos, tan, log, exp. Use 'pi' and 'e' for constants.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        function=safe_eval
    )
    
    return registry


def create_datetime_tools() -> ToolRegistry:
    """Create a registry with date and time tools."""
    registry = ToolRegistry()
    
    registry.register(
        name="get_current_datetime",
        description="Get the current date and time in ISO format.",
        parameters={"type": "object", "properties": {}},
        function=lambda: datetime.now().isoformat()
    )
    
    registry.register(
        name="format_date",
        description="Format a date string into a different format.",
        parameters={
            "type": "object",
            "properties": {
                "date_string": {
                    "type": "string",
                    "description": "The date to format (ISO format: YYYY-MM-DD)"
                },
                "format": {
                    "type": "string",
                    "description": "Output format (e.g., '%B %d, %Y' for 'January 01, 2024')"
                }
            },
            "required": ["date_string", "format"]
        },
        function=lambda date_string, format: datetime.fromisoformat(date_string).strftime(format)
    )
    
    return registry


def merge_registries(*registries: ToolRegistry) -> ToolRegistry:
    """Merge multiple tool registries into one."""
    merged = ToolRegistry()
    
    for registry in registries:
        for tool_def in registry.get_definitions():
            name = tool_def["name"]
            merged._tools[name] = tool_def
            merged._functions[name] = registry._functions[name]
    
    return merged
```

Using these collections:

```python
from augmented_llm import AugmentedLLM, AugmentedLLMConfig
from tool_collections import create_math_tools, create_datetime_tools, merge_registries

# Combine tool collections
all_tools = merge_registries(
    create_math_tools(),
    create_datetime_tools()
)

# Create LLM with combined tools
llm = AugmentedLLM(
    config=AugmentedLLMConfig(
        system_prompt="You are a helpful assistant with math and datetime capabilities."
    ),
    tools=all_tools
)
```

## Testing Your Building Block

Reliable agents need reliable foundations. Let's write tests for our `AugmentedLLM` class.

> **💡 Tip:** Testing LLM-based code is tricky because outputs are non-deterministic. Focus on testing structure and behavior, not exact outputs.

```python
"""
Tests for the AugmentedLLM class.

Chapter 14: Building the Complete Augmented LLM
"""

import json
import pytest

from augmented_llm import AugmentedLLM, AugmentedLLMConfig, ToolRegistry


class TestToolRegistry:
    """Tests for the ToolRegistry class."""
    
    def test_register_and_execute(self):
        """Test basic tool registration and execution."""
        registry = ToolRegistry()
        
        registry.register(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            },
            function=lambda a, b: a + b
        )
        
        assert len(registry) == 1
        assert registry.has_tool("add")
        assert registry.execute("add", {"a": 2, "b": 3}) == 5
    
    def test_get_definitions(self):
        """Test that definitions are in Claude's expected format."""
        registry = ToolRegistry()
        
        registry.register(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test"
        )
        
        definitions = registry.get_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
        assert definitions[0]["description"] == "A test tool"
        assert "input_schema" in definitions[0]
    
    def test_unknown_tool_raises_error(self):
        """Test that executing unknown tool raises KeyError."""
        registry = ToolRegistry()
        
        with pytest.raises(KeyError, match="Unknown tool"):
            registry.execute("nonexistent", {})


class TestAugmentedLLMConfig:
    """Tests for the configuration class."""
    
    def test_default_values(self):
        """Test that defaults are sensible."""
        config = AugmentedLLMConfig()
        
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.max_tool_iterations == 10
    
    def test_immutability(self):
        """Test that config is immutable."""
        config = AugmentedLLMConfig()
        
        with pytest.raises(AttributeError):
            config.model = "different-model"


class TestAugmentedLLM:
    """Tests for the main AugmentedLLM class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        llm = AugmentedLLM()
        
        assert llm.config.model == "claude-sonnet-4-20250514"
        assert len(llm.tools) == 0
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = AugmentedLLMConfig(
            system_prompt="Custom prompt",
            max_tokens=1000
        )
        
        llm = AugmentedLLM(config=config)
        
        assert llm.config.system_prompt == "Custom prompt"
        assert llm.config.max_tokens == 1000
    
    def test_register_tool(self):
        """Test tool registration through the class."""
        llm = AugmentedLLM()
        
        llm.register_tool(
            name="test",
            description="Test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "result"
        )
        
        assert len(llm.tools) == 1
        assert llm.tools.has_tool("test")
    
    def test_history_management(self):
        """Test conversation history methods."""
        llm = AugmentedLLM()
        
        # History starts empty
        assert len(llm.get_history()) == 0
        
        # After clearing, still empty
        llm.clear_history()
        assert len(llm.get_history()) == 0
    
    def test_validation_rejects_invalid_json(self):
        """Test that validation catches non-JSON responses."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="not valid JSON"):
            llm._validate_response("not json")
    
    def test_validation_rejects_missing_fields(self):
        """Test that validation catches missing required fields."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {"required_field": {"type": "string"}},
                "required": ["required_field"]
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="Missing required field"):
            llm._validate_response('{"other_field": "value"}')


# Integration tests (require API key)
class TestAugmentedLLMIntegration:
    """Integration tests that make actual API calls."""
    
    @pytest.mark.integration
    def test_simple_query(self):
        """Test a simple query without tools."""
        llm = AugmentedLLM()
        
        response = llm.run("Reply with exactly: 'Hello, World!'")
        
        assert "hello" in response.lower()
    
    @pytest.mark.integration
    def test_tool_usage(self):
        """Test that tools are called when appropriate."""
        llm = AugmentedLLM(
            config=AugmentedLLMConfig(
                system_prompt="Use the add tool to answer math questions."
            )
        )
        
        llm.register_tool(
            name="add",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            },
            function=lambda a, b: a + b
        )
        
        response = llm.run("What is 7 + 5?")
        
        assert "12" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run the tests with:

```bash
# Run unit tests only (no API calls)
pytest tests/test_augmented_llm.py -v -m "not integration"

# Run all tests including integration tests
pytest tests/test_augmented_llm.py -v
```

## Common Pitfalls

### 1. Forgetting to Clear History Between Unrelated Queries

If you're using one `AugmentedLLM` instance for multiple unrelated queries, remember to clear the history:

```python
# Bad: Previous conversation context bleeds into new queries
llm = AugmentedLLM()
llm.run("Tell me about Python lists")
llm.run("What's the weather in Tokyo?")  # May reference lists somehow!

# Good: Clear history between unrelated queries
llm = AugmentedLLM()
llm.run("Tell me about Python lists")
llm.clear_history()
llm.run("What's the weather in Tokyo?")  # Fresh context
```

### 2. Tool Functions That Raise Exceptions

Our class handles tool exceptions gracefully, but it's still better to handle errors in your tool functions:

```python
# Fragile: Exceptions escape
def divide(a: float, b: float) -> float:
    return a / b  # Crashes on b=0

# Robust: Handle errors in the tool
def divide(a: float, b: float) -> str:
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)
```

### 3. Missing Required Fields in Response Schema

When using structured output, make sure your system prompt tells Claude exactly what fields to include:

```python
# Bad: Schema expects fields Claude doesn't know about
config = AugmentedLLMConfig(
    system_prompt="Analyze the sentiment",  # Doesn't mention required fields!
    response_schema={
        "type": "object",
        "required": ["sentiment", "confidence", "keywords"]
    }
)

# Good: System prompt matches schema requirements
config = AugmentedLLMConfig(
    system_prompt="""Analyze sentiment and respond with JSON containing:
    - sentiment: positive/negative/neutral
    - confidence: 0-1
    - keywords: array of relevant words""",
    response_schema={
        "type": "object",
        "required": ["sentiment", "confidence", "keywords"]
    }
)
```

## Practical Exercise

**Task:** Build an `AugmentedLLM` that can help with unit conversions.

**Requirements:**

1. Create tools for converting between:
   - Temperature (Celsius ↔ Fahrenheit ↔ Kelvin)
   - Length (meters ↔ feet ↔ inches)
   - Weight (kilograms ↔ pounds ↔ ounces)

2. Configure a system prompt that explains the available conversions

3. Test with queries like:
   - "Convert 100 degrees Fahrenheit to Celsius"
   - "How many feet is 2.5 meters?"
   - "What's 150 pounds in kilograms?"

4. Add structured output that returns both the numeric result and the formula used

**Hints:**
- Create separate tools for each conversion category
- Include the conversion formula in your tool's return value
- Remember to handle edge cases (like absolute zero for Kelvin)

**Solution:** See `code/exercise_unit_converter.py`

## Key Takeaways

- **The `AugmentedLLM` class is your foundation**—every agent and workflow in this book builds on it. Invest time in understanding it thoroughly.

- **Tools and implementations stay together** in the `ToolRegistry`. This prevents mismatches and makes your code more maintainable.

- **Configuration is immutable** once set. This prevents subtle bugs from mid-conversation changes.

- **The tool loop handles everything automatically**—you send a message, and the class manages all tool calls until Claude provides a final response.

- **Testing LLM code requires different strategies**—focus on behavior and structure, not exact outputs. Use mock responses for unit tests.

- **Reusable tool collections** let you build up a library of capabilities over time. Create domain-specific collections and combine them as needed.

## What's Next

Congratulations! You've completed Part 2 and built a solid foundation for agent development. Your `AugmentedLLM` class is a fully functional building block that can:

- Follow system-defined behavior
- Use tools to interact with the world
- Chain multiple tool calls automatically
- Return validated structured responses

In Part 3, we'll use this building block to implement the five core **workflow patterns**: chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer. These patterns let you compose multiple LLM calls into sophisticated workflows that handle complex tasks.

Up next: **Chapter 15: Introduction to Agentic Workflows**, where we'll survey these patterns and learn when to use each one.
