# Chapter 37: Debugging Agents - Code Examples

This directory contains all the code examples for Chapter 37.

## Files

### `example_01_debug_logger.py`
Enhanced debug logger with structured event tracking, step-by-step tracing, and export capabilities.

### `example_02_conversation_debugger.py`
Tools for analyzing conversation structure, detecting flow issues, and finding derailment points.

### `example_03_tool_debugger.py`
Tool selection analyzer that helps diagnose why agents choose (or don't choose) specific tools.

### `example_04_loop_detector.py`
Infinite loop detection and prevention system with pattern recognition for exact repeats, oscillations, and resource exhaustion.

### `example_05_replay_system.py`
Complete session recording and replay system for capturing agent behavior and reproducing bugs.

### `example_06_debugging_helper.py`
Systematic debugging helper that guides you through a structured debugging process.

### `exercise_solution.py`
Solution to the chapter exercise: A comprehensive diagnostic tool that analyzes recorded sessions and generates bug reports.

## Running the Examples

Each example can be run standalone:

```bash
# Set up your environment first
export ANTHROPIC_API_KEY=your-api-key-here

# Or use a .env file (recommended)
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env

# Run any example
uv run python example_01_debug_logger.py
uv run python example_02_conversation_debugger.py
uv run python example_03_tool_debugger.py
uv run python example_04_loop_detector.py
uv run python example_05_replay_system.py
uv run python example_06_debugging_helper.py
uv run python exercise_solution.py
```

## Dependencies

- Python 3.10+
- anthropic
- python-dotenv

Install with:
```bash
pip install anthropic python-dotenv
```

## Key Concepts

1. **Debug Logger**: Captures step-by-step events with structured data for easy analysis
2. **Conversation Debugger**: Analyzes message structure, token usage, and identifies flow issues
3. **Tool Debugger**: Helps understand tool selection decisions and improve tool descriptions
4. **Loop Detector**: Identifies infinite loop patterns before they become problems
5. **Session Replay**: Record everything, replay later for debugging
6. **Systematic Debugging**: Follow a structured process to efficiently find and fix bugs
