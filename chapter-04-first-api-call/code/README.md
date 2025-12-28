# Chapter 4 Code Examples

This directory contains all the runnable code examples from Chapter 4: Your First API Call to Claude.

## Prerequisites

Before running these examples, make sure you have:

1. Created a `.env` file in your project root with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Installed the required dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

## Files

### `first_call.py`
Your very first API call to Claude. Run this to verify your setup works.

```bash
uv run python first_call.py
```

### `response_structure.py`
Explores the full structure of Claude's API response, including token counts, stop reasons, and metadata.

```bash
uv run python response_structure.py
```

### `error_handling.py`
Demonstrates how to handle various API errors gracefully (authentication, rate limits, connection issues).

```bash
uv run python error_handling.py
```

### `cost_estimation.py`
Shows how to track token usage and estimate the cost of API calls.

```bash
uv run python cost_estimation.py
```

### `complete_function.py`
A polished, reusable function that wraps API calls with proper typing and structure.

```bash
uv run python complete_function.py
```

### `magic_8_ball.py`
**Exercise Solution**: An interactive Magic 8-Ball that uses Claude to answer yes/no questions mystically.

```bash
uv run python magic_8_ball.py
```

## Expected Output

Each script will make a real API call to Claude, so you'll need a valid API key. The scripts demonstrate progressively more sophisticated ways to interact with the API.

## Notes

- All scripts use `claude-sonnet-4-20250514` as the default model
- Token usage is displayed in several examples to help you understand costs
- Error handling examples show best practices for production code
