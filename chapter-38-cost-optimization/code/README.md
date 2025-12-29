# Chapter 38: Cost Optimization - Code Files

This directory contains all the code examples for Chapter 38: Cost Optimization.

## Files Overview

### `token_utils.py`
Token estimation and cost calculation utilities.

### `prompt_optimizer.py`
System prompt optimization techniques.

### `conversation_manager.py`
Conversation history management with token budgets.

### `cache.py`
Caching strategies for LLM responses.

### `model_selector.py`
Model selection strategies for cost optimization.

### `cost_monitor.py`
Cost monitoring and alerting system.

### `exercise_solution.py`
Complete solution for the chapter exercise - a cost-aware research agent.

## Running the Examples

Each file can be run directly:

```bash
uv run python token_utils.py
uv run python prompt_optimizer.py
uv run python cache.py
uv run python model_selector.py
uv run python cost_monitor.py
uv run python exercise_solution.py  # Requires API key
```

## Requirements

- Python 3.10+
- anthropic package
- python-dotenv
