# Chapter 23 Code Examples

This directory contains complete, runnable code examples for the Orchestrator-Workers Implementation chapter.

## Files

| File | Description |
|------|-------------|
| `orchestrator_basic.py` | Basic orchestrator that creates task plans |
| `worker.py` | Worker class that executes individual subtasks |
| `task_delegator.py` | Coordinates delegation of subtasks to workers |
| `synthesizer.py` | Synthesizes worker results into final responses |
| `research_orchestrator.py` | Complete, production-ready orchestrator class |
| `example_usage.py` | Demonstrates how to use the orchestrator |
| `exercise_parallel_orchestrator.py` | Exercise solution: parallel worker execution |

## Setup

1. Ensure you have your `.env` file with `ANTHROPIC_API_KEY`
2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

## Running Examples

```bash
# Run the complete orchestrator
python research_orchestrator.py

# Try example usage patterns
python example_usage.py

# Run the parallel exercise solution
python exercise_parallel_orchestrator.py
```

## Key Concepts

1. **Orchestrator**: Analyzes queries and breaks them into subtasks
2. **Workers**: Execute individual subtasks (research, analysis, comparison)
3. **Delegator**: Coordinates work distribution and result collection
4. **Synthesizer**: Combines worker outputs into coherent responses

## Building Blocks

Each file builds on the previous:

```
orchestrator_basic.py  →  Creates task plans
         ↓
     worker.py         →  Executes subtasks
         ↓
  task_delegator.py    →  Coordinates execution
         ↓
   synthesizer.py      →  Combines results
         ↓
research_orchestrator.py  →  Complete system
```
