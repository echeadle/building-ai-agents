# Chapter 27 Code Examples

This directory contains runnable code examples for Chapter 27: The Agentic Loop.

## Files

### example_01_simple_loop.py
The simplest possible agentic loop demonstrating the core perceive-think-act cycle. This is the foundational pattern that all agents build upon.

**Run:** `python example_01_simple_loop.py`

### example_02_complete_agent.py
A complete working agent with calculator and time tools. Shows the full loop in action with real tool execution and logging of each phase.

**Run:** `python example_02_complete_agent.py`

### example_03_termination.py
Demonstrates multiple termination conditions including max iterations, max tool calls, and timeout. Shows how to build safety limits into your agents.

**Run:** `python example_03_termination.py`

### exercise.py
Solution to the chapter exercise: an inventory management agent with three tools (`check_stock`, `get_price`, `calculate_total`) that can answer questions about a fictional inventory system.

**Run:** `python exercise.py`

## Prerequisites

Before running these examples, ensure you have:

1. Python 3.10+ installed
2. The anthropic package: `uv add anthropic python-dotenv`
3. A `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Key Concepts Demonstrated

- The perceive-think-act loop structure
- Handling tool_use vs end_turn stop reasons
- Properly maintaining conversation history
- Multiple termination conditions for safety
- Logging and observability in the loop
