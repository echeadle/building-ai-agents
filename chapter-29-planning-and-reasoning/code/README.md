# Chapter 29: Planning and Reasoning - Code Examples

This directory contains runnable code examples for Chapter 29.

## Files

### `example_01_plan_then_execute.py`
The basic plan-then-execute pattern. Creates a structured plan for a task, then executes each step sequentially while accumulating context.

### `example_02_chain_of_thought.py`
Chain-of-thought reasoning patterns. Demonstrates how to prompt for step-by-step reasoning and compares results with and without CoT.

### `example_03_planning_agent.py`
A complete planning agent with tool use. Includes structured plans with dataclasses, step execution with tools, progress tracking, and plan revision.

### `example_04_plan_revision.py`
Patterns for detecting when plans need revision and creating revised plans based on new information or obstacles.

### `example_05_transparent_planning.py`
Transparent planning with user-visible reasoning. Shows how to make the agent's thinking process visible for trust and debugging.

### `example_06_when_to_plan.py`
Demonstrates adaptive planning—how to decide whether a task needs explicit planning or can be handled directly.

### `exercise.py`
Solution to the practical exercise: a Trip Planning Agent that creates day-by-day itineraries with visible planning and revision capabilities.

## Prerequisites

Before running these examples, ensure you have:

1. Python 3.10 or higher
2. The required packages installed:
   ```bash
   uv add anthropic python-dotenv
   ```
3. A `.env` file in your project root with:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Running the Examples

```bash
# Run any example
python example_01_plan_then_execute.py
python example_02_chain_of_thought.py
python example_03_planning_agent.py
python example_04_plan_revision.py
python example_05_transparent_planning.py
python example_06_when_to_plan.py
python exercise.py
```

## Key Concepts Demonstrated

- **Plan-then-execute**: Separate thinking from doing
- **Chain-of-thought**: Step-by-step reasoning within responses
- **Plan structure**: Goal, steps, success criteria
- **Plan revision**: Adapting when circumstances change
- **Transparency**: Making agent thinking visible
- **Adaptive planning**: Choosing when to plan
