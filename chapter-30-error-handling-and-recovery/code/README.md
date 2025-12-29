# Chapter 30: Error Handling and Recovery - Code Examples

This directory contains all runnable code examples from Chapter 30.

## Files

### `error_types.py`
Demonstrates the different types of errors that occur in agent systems:
- API and network errors
- LLM output parsing errors
- Tool execution errors

### `retry_logic.py`
Implements retry logic with exponential backoff:
- `RetryConfig` dataclass for configuration
- `retry_with_backoff()` function for any operation
- `create_api_call_with_retry()` factory for API calls
- Jitter implementation to prevent thundering herd

### `fallback_strategies.py`
Shows fallback patterns for graceful degradation:
- `FallbackCache` for caching responses
- `FallbackChain` for trying multiple strategies
- `WeatherServiceWithFallback` example
- Model fallback for API calls

### `logging_errors.py`
Structured logging for agent error tracking:
- `AgentError` dataclass with full metadata
- `AgentLogger` class with severity-based logging
- Automatic exception categorization
- Error summary generation

### `self_correction.py`
Patterns for agent self-correction:
- `extract_json_from_response()` for parsing help
- `self_correct_json()` for fixing invalid JSON
- `self_correct_tool_call()` for fixing tool inputs
- `SelfCorrectingAgent` wrapper class

### `error_handler.py`
Complete error handling module combining all patterns:
- `ErrorHandler` class with full functionality
- `with_retry` decorator
- `FallbackChain` generic class
- `safe_json_parse()` utility
- Full demo of all features

### `exercise.py`
Solution to the chapter exercise: `ResilientToolExecutor` class

## Running the Examples

Each file can be run independently:

```bash
# Set up environment
cd chapter-30/code
cp .env.example .env  # Add your API key

# Run examples
uv run python error_types.py
uv run python retry_logic.py
uv run python fallback_strategies.py
uv run python logging_errors.py
uv run python self_correction.py
uv run python error_handler.py
uv run python exercise.py
```

## Dependencies

All examples require:
- Python 3.10+
- python-dotenv
- anthropic

Install with:
```bash
pip install python-dotenv anthropic
```
