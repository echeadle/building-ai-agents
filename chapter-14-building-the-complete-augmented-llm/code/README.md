# Chapter 14: Building the Complete Augmented LLM - Code Examples

This directory contains all code examples for Chapter 14.

## Files Overview

### Core Implementation

| File | Description |
|------|-------------|
| `augmented_llm.py` | The main `AugmentedLLM` class, `ToolRegistry`, and `AugmentedLLMConfig`. This is the core building block for all future chapters. |
| `tool_collections.py` | Reusable tool collections (math, datetime, text) that can be combined and used across projects. |

### Examples

| File | Description |
|------|-------------|
| `example_basic.py` | Basic usage without tools - custom system prompts and simple queries |
| `example_tools.py` | Using multiple tools (calculator, datetime, string manipulation) |
| `example_structured.py` | Structured output with JSON Schema validation (sentiment analysis, entity extraction, classification) |
| `example_conversation.py` | Multi-turn conversations with history management |

### Tests

| File | Description |
|------|-------------|
| `test_augmented_llm.py` | Comprehensive test suite with unit tests and integration tests |

### Exercise

| File | Description |
|------|-------------|
| `exercise_unit_converter.py` | Solution to the chapter exercise - a unit conversion agent |

## Setup

1. Make sure you have your `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv pytest
   ```

## Running the Examples

```bash
# Basic usage
python example_basic.py

# With tools
python example_tools.py

# Structured output
python example_structured.py

# Multi-turn conversations
python example_conversation.py

# Exercise solution (interactive)
python exercise_unit_converter.py
```

## Running Tests

```bash
# Unit tests only (no API calls)
pytest test_augmented_llm.py -v -m "not integration"

# All tests including integration tests
pytest test_augmented_llm.py -v

# With coverage
pytest test_augmented_llm.py -v --cov=augmented_llm
```

## Using the AugmentedLLM in Your Projects

The `augmented_llm.py` file is designed to be copied into your own projects. Here's a quick start:

```python
from augmented_llm import AugmentedLLM, AugmentedLLMConfig

# Create with custom settings
config = AugmentedLLMConfig(
    system_prompt="You are a helpful assistant.",
    max_tokens=2048
)

llm = AugmentedLLM(config=config)

# Add tools
llm.register_tool(
    name="my_tool",
    description="Does something useful",
    parameters={"type": "object", "properties": {}},
    function=lambda: "result"
)

# Run queries
response = llm.run("Hello!")
print(response)
```

## Key Concepts Demonstrated

1. **Tool Registry Pattern** - Clean management of tool definitions and implementations
2. **Configuration with Dataclasses** - Immutable, type-safe configuration
3. **Automatic Tool Loop** - Handles sequential tool calls automatically
4. **Response Validation** - Optional JSON Schema validation for structured output
5. **Conversation History** - Built-in management with clear/get methods
6. **Error Handling** - Graceful handling of tool errors and API issues
7. **Reusable Tool Collections** - Modular tool sets that can be combined

## Next Steps

This `AugmentedLLM` class is the foundation for everything in Part 3 (Workflow Patterns) and Part 4 (Building True Agents). Make sure you understand how it works before moving on!
