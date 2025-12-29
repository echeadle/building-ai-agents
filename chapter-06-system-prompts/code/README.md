# Chapter 6 Code Examples

This directory contains all runnable code examples from Chapter 6: System Prompts and Persona Design.

## Files

| File | Description |
|------|-------------|
| `basic_system_prompt.py` | Demonstrates the basic structure of system prompts |
| `code_review_agent.py` | A code review assistant with a detailed system prompt |
| `persona_example.py` | Example of giving an agent a distinct persona (Byte the tutor) |
| `agent_class.py` | The reusable `Agent` class that loads system prompts from files |
| `test_system_prompt.py` | Framework for testing system prompt effectiveness |
| `exercise_recipe_agent.py` | Solution to the chapter exercise |
| `prompts/` | Directory containing system prompt text files |

## Setup

Before running any examples, ensure you have:

1. Created a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Installed dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

## Running Examples

Each file can be run directly:

```bash
uv run python basic_system_prompt.py
uv run python code_review_agent.py
uv run python persona_example.py
uv run python agent_class.py
uv run python test_system_prompt.py
uv run python exercise_recipe_agent.py
```

## Key Concepts Demonstrated

- **System prompt structure**: Identity, capabilities, behavior, boundaries
- **Persona design**: Giving agents distinct personalities
- **File-based prompts**: Loading prompts from external files for easier iteration
- **Testing prompts**: Systematic testing with defined scenarios
- **Configurable agents**: The `Agent` class pattern for reuse
