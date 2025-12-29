# Chapter 15 Code Examples

This chapter is primarily conceptual, introducing the five workflow patterns.
The code files here provide:

1. **workflow_base.py** - Base classes and interfaces that all patterns will follow
2. **pattern_analyzer.py** - A utility to help analyze tasks and suggest appropriate patterns
3. **simple_vs_workflow.py** - Examples showing when simple prompts are sufficient
4. **exercise_solutions.py** - Solutions to the chapter exercises

## Running the Examples

All examples require your `.env` file with `ANTHROPIC_API_KEY` set.

```bash
# Analyze a task to get pattern recommendations
uv run python pattern_analyzer.py

# See examples of simple prompts vs. workflows
uv run python simple_vs_workflow.py
```

## Key Concepts

This chapter establishes:

- The distinction between workflows (developer-defined control flow) and agents (LLM-directed control flow)
- The five core patterns: Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer
- A decision framework for choosing patterns
- When NOT to use workflows (simple prompts are often enough)
