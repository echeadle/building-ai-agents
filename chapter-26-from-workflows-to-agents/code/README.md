# Chapter 26 Code Examples

This directory contains code examples demonstrating the transition from workflows to agents.

## Files

### `workflow_vs_agent_comparison.py`
Demonstrates the conceptual difference between workflows (developer-controlled flow) and agents (LLM-controlled flow). Shows side-by-side examples of how the same task would be approached differently.

### `minimal_agent_loop.py`
The fundamental agent loop pattern: perceive → think → act → repeat. This is a complete, runnable implementation that you can use as a starting point for your own agents.

### `autonomy_patterns.py`
Demonstrates different levels of agent autonomy:
- Confirmation Required (every action needs approval)
- Checkpoint Mode (pauses at key milestones)
- Bounded Actions (limited tool access)
- Fully Autonomous (complete freedom)

### `workflow_analysis_template.py`
A template for analyzing whether a workflow should be upgraded to an agent. Use this for the chapter exercise.

## Running the Examples

1. Make sure you have your `.env` file set up with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

3. Run any example:
   ```bash
   uv run python workflow_vs_agent_comparison.py
   uv run python minimal_agent_loop.py
   uv run python autonomy_patterns.py
   ```

## Key Concepts

- **Workflow**: You (the developer) control the execution flow
- **Agent**: The LLM controls the execution flow
- **Agent Loop**: perceive → think → act → repeat
- **Autonomy Spectrum**: From full human control to full agent control
