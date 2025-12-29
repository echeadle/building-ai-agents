# Chapter 8 Code Examples

This directory contains runnable code examples for Chapter 8: Defining Your First Tool.

## Files

### `example_01_basic_tool.py`
The simplest possible tool definition—a tool that returns the current time with no parameters. Demonstrates the minimal structure every tool needs.

### `example_02_calculator_tool.py`
A complete calculator tool definition with proper descriptions and parameter schemas. Shows how to use enums to constrain operations and how to write Claude-friendly descriptions.

### `example_03_multiple_tools.py`
Demonstrates defining and providing multiple tools to Claude. Shows how Claude selects the appropriate tool based on the user's question.

### `example_04_parameter_types.py`
A reference showing all common JSON Schema parameter types: string, number, integer, boolean, array, and object. Includes examples of enums, constraints, and nested structures.

### `exercise_book_search.py`
Solution to the chapter exercise: a book search tool with required and optional parameters.

## Running the Examples

Make sure you have:
1. Created a `.env` file with your `ANTHROPIC_API_KEY`
2. Installed dependencies: `uv add anthropic python-dotenv`

Then run any example:
```bash
uv run python example_01_basic_tool.py
```

## Note

These examples demonstrate tool *definition* only. Claude will request to use these tools, but we don't execute the tools in this chapter. Chapter 9 covers handling tool calls and returning results.
