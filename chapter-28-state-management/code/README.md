# Chapter 28 Code Examples

This directory contains all runnable code examples from Chapter 28: State Management.

## Files

| File | Description |
|------|-------------|
| `example_01_stateless_demo.py` | Demonstrates the stateless nature of API calls |
| `example_02_conversation_state.py` | Basic conversation history as state |
| `example_03_conversation_with_tools.py` | Conversation state that tracks tool calls |
| `example_04_working_memory.py` | Working memory for current task context |
| `example_05_inject_memory.py` | How to inject working memory into prompts |
| `example_06_long_term_memory.py` | File-based long-term memory persistence |
| `example_07_state_manager.py` | Complete unified state management system |
| `example_08_stateful_agent.py` | Full stateful agent implementation |
| `example_09_truncation.py` | Handling conversation truncation |
| `exercise_memory_importance.py` | Solution to the chapter exercise |

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
   uv run python example_01_stateless_demo.py
   ```

## Notes

- Examples that create state files will clean them up automatically
- The stateful agent example creates a `.test_agent_state` directory which is removed after running
- All examples are self-contained and can be run independently
