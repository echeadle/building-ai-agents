# Chapter 12: Sequential Tool Calls - Code Examples

This directory contains all runnable code examples from Chapter 12.

## Files

| File | Description |
|------|-------------|
| `basic_agentic_loop.py` | Basic implementation of the agentic loop pattern |
| `multi_tool_agent.py` | Multi-tool agent with calculator, weather, and datetime tools |
| `loop_detection.py` | Agent with infinite loop detection |
| `tracking_agent.py` | Agent with comprehensive tool call tracking |
| `sequential_agent.py` | Complete SequentialAgent class (production-ready) |
| `exercise_trip_planner.py` | Solution to the practical exercise |

## Setup

1. Make sure you have your `.env` file with:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

## Running the Examples

Each file is self-contained and can be run directly:

```bash
# Basic agentic loop
uv run python basic_agentic_loop.py

# Multi-tool agent
uv run python multi_tool_agent.py

# Agent with loop detection
uv run python loop_detection.py

# Agent with tracking
uv run python tracking_agent.py

# Complete sequential agent
uv run python sequential_agent.py

# Exercise solution
uv run python exercise_trip_planner.py
```

## Key Concepts Demonstrated

- **The Agentic Loop**: The core while loop that processes tool calls until completion
- **Message Structure**: How to properly format tool_use and tool_result messages
- **Stop Conditions**: Max iterations, timeout, and loop detection
- **Tool Call Tracking**: Recording all tool calls for debugging and observability

## Building On This

This code serves as the foundation for:
- Chapter 13: Structured Outputs (getting predictable response formats)
- Chapter 14: The Complete Augmented LLM (integrating everything)
- Part 4: True Agents (adding planning, memory, and autonomy)
