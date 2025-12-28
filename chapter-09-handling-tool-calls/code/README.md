# Chapter 9: Handling Tool Calls - Code Examples

This directory contains complete, runnable code examples for Chapter 9.

## Prerequisites

Before running these examples, make sure you have:

1. Completed the setup from Chapters 2-3
2. An `.env` file with your `ANTHROPIC_API_KEY`
3. Installed the required packages:
   ```bash
   uv add anthropic python-dotenv
   ```

## Files

### example_01_detecting_tool_use.py
Demonstrates how to detect when Claude wants to use a tool and extract tool use blocks from the response.

```bash
uv run example_01_detecting_tool_use.py
```

### example_02_parsing_and_executing.py
Shows how to parse tool arguments, execute the corresponding Python function, and format results.

```bash
uv run example_02_parsing_and_executing.py
```

### example_03_complete_loop.py
The complete tool use loop implementation - a working calculator that Claude can use to solve math problems.

```bash
uv run example_03_complete_loop.py
```

### exercise_solution.py
Solution to the chapter's practical exercise - an extended calculator with more operations and better error handling.

```bash
uv run exercise_solution.py
```

## Key Concepts

1. **Detecting Tool Use**: Check `response.stop_reason == "tool_use"`
2. **Parsing Arguments**: Access `block.name` and `block.input` from tool_use blocks
3. **Returning Results**: Use the exact `tool_use_id` and include results as strings
4. **The Loop**: Keep calling the API until `stop_reason` is no longer `"tool_use"`

## Common Issues

- **API Key not found**: Make sure your `.env` file exists and contains `ANTHROPIC_API_KEY`
- **Import errors**: Run `uv add anthropic python-dotenv` to install dependencies
- **Invalid tool_use_id**: Always use the exact ID from the tool_use block
