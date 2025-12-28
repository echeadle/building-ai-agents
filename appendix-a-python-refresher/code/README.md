# Appendix A Code Examples

This directory contains complete, runnable examples for each Python feature covered in Appendix A.

## Files

### 1. `async_example.py`
Demonstrates async/await patterns for AI agents:
- Synchronous vs asynchronous API calls
- Parallel tool execution with `asyncio.gather()`
- Timeout protection with `asyncio.wait_for()`
- Streaming responses

**Run:** `python async_example.py`

### 2. `type_hints_example.py`
Shows type hints and Pydantic usage:
- Basic type hints for primitives and collections
- Optional and Union types
- Pydantic models for validation
- Type hints for agent components

**Run:** `python type_hints_example.py`

### 3. `context_managers_example.py`
Demonstrates context managers for resource management:
- Basic `with` statement usage
- Custom context managers with `@contextmanager`
- API client management
- Conversation context management

**Run:** `python context_managers_example.py`

### 4. `decorators_example.py`
Shows decorators for cross-cutting concerns:
- Retry with exponential backoff
- Function caching with TTL
- Timing and performance monitoring
- Logging decorator

**Run:** `python decorators_example.py`

### 5. `dataclasses_example.py`
Demonstrates dataclasses for configuration:
- Basic dataclass usage
- Validation with `__post_init__`
- Frozen (immutable) dataclasses
- Dataclasses for agent state

**Run:** `python dataclasses_example.py`

### 6. `complete_example.py`
Combines all five concepts in a production-ready agent:
- Uses async/await for concurrency
- Uses Pydantic for validation
- Uses decorators for resilience
- Uses context managers for resources
- Uses dataclasses for configuration

**Run:** `python complete_example.py`

### 7. `exercise_solution.py`
Complete solution to the practical exercise.

**Run:** `python exercise_solution.py`

## Prerequisites

All examples require:
- Python 3.10+
- `anthropic` package: `uv add anthropic`
- `python-dotenv` package: `uv add python-dotenv`
- `pydantic` package: `uv add pydantic`
- `.env` file with `ANTHROPIC_API_KEY`

## Setup

```bash
# Create project
uv init appendix-a-examples
cd appendix-a-examples

# Add dependencies
uv add anthropic python-dotenv pydantic

# Create .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Run any example
python code/async_example.py
```

## Notes

- All examples are complete and runnable
- Code follows the book's style guidelines
- Examples use `claude-sonnet-4-20250514` as the default model
- Secrets are loaded from `.env` (never hardcoded)
