# Chapter 21: Parallelization - Implementation

Code examples for implementing parallel workflow patterns.

## Files

| File | Description |
|------|-------------|
| `example_01_asyncio_basics.py` | Introduction to asyncio with parallel API calls |
| `example_02_sectioning.py` | Sectioning pattern for parallel subtasks |
| `example_03_voting.py` | Voting pattern for consensus decisions |
| `example_04_error_handling.py` | Error handling patterns for parallel workflows |
| `example_05_code_review.py` | Complete code review system using voting |
| `exercise_translation.py` | Exercise solution: parallel translation system |

## Setup

Make sure you have your `.env` file configured:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Install dependencies:

```bash
uv add anthropic python-dotenv
```

## Running the Examples

Each example can be run directly:

```bash
# Basic asyncio demonstration
python example_01_asyncio_basics.py

# Sectioning pattern
python example_02_sectioning.py

# Voting pattern
python example_03_voting.py

# Error handling
python example_04_error_handling.py

# Full code review system
python example_05_code_review.py

# Exercise solution
python exercise_translation.py
```

## Key Concepts

### Asyncio Essentials

- `async def` - Define a coroutine
- `await` - Pause until operation completes
- `asyncio.gather()` - Run coroutines in parallel
- `asyncio.run()` - Start the event loop

### Patterns Implemented

1. **Sectioning**: Divide work into independent parallel subtasks
2. **Voting**: Get multiple perspectives and aggregate for consensus
3. **Error Handling**: Graceful failure handling with retry support

## Notes

- All examples use `anthropic.AsyncAnthropic()` for async API calls
- Concurrency is limited with `asyncio.Semaphore` to avoid rate limits
- Each example is self-contained and runnable
