# Chapter 11 Code Examples

This directory contains all runnable code examples for Chapter 11: Multi-Tool Agents.

## Files

| File | Description |
|------|-------------|
| `example_01_multiple_tools.py` | Basic demo of providing multiple tools in one API request |
| `example_02_tool_selection.py` | Tests how Claude selects between different tools |
| `example_03_registry.py` | The Tools Registry Pattern implementation |
| `example_04_multi_tool_agent.py` | **Main deliverable**: Complete multi-tool agent |
| `exercise_unit_converter.py` | Exercise solution: Agent with unit converter tool |

## Running the Examples

1. Make sure you have your `.env` file set up with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

3. Run any example:
   ```bash
   uv run python example_04_multi_tool_agent.py
   ```

## Dependencies

- Python 3.10+
- anthropic
- python-dotenv
