# Chapter 10 Code Examples

This directory contains all runnable code examples for Chapter 10: Building a Weather Tool.

## Prerequisites

Before running these examples, ensure you have:

1. Python 3.10+ installed
2. An `.env` file with your `ANTHROPIC_API_KEY`
3. Required packages installed:
   ```bash
   uv add anthropic python-dotenv requests
   ```

## Files

### `explore_api.py`
Explores the Open-Meteo API to understand its structure before building our tool. Run this first to see how the geocoding and weather APIs work.

```bash
uv run python explore_api.py
```

### `weather_tool.py`
The complete weather tool implementation with comprehensive error handling. This module can be imported into other scripts.

```bash
uv run python weather_tool.py
```

### `weather_agent.py`
A complete agent that uses the weather tool to answer questions. This demonstrates the full tool use loop with Claude.

```bash
uv run python weather_agent.py
```

### `test_weather_tool.py`
Tests for the weather tool to verify reliability, edge cases, and performance.

```bash
uv run python test_weather_tool.py
```

### `exercise.py`
Solution to the practical exercise: a weather tool with 3-day forecast support.

```bash
uv run python exercise.py
```

## Key Concepts Demonstrated

- Integrating with external REST APIs
- Comprehensive error handling (timeouts, connection errors, HTTP errors)
- Formatting API responses for LLM consumption
- Tool definition with optional parameters
- Testing external tool integrations

## API Used

This chapter uses [Open-Meteo](https://open-meteo.com/), a free weather API that requires no API key. The same patterns apply to any weather API.

- Geocoding: `https://geocoding-api.open-meteo.com/v1/search`
- Weather: `https://api.open-meteo.com/v1/forecast`
