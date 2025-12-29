# Chapter 36: Observability and Logging - Code Examples

This directory contains all runnable code examples from Chapter 36.

## Files

| File | Description |
|------|-------------|
| `example_01_basic_logging.py` | Introduction to Python's logging module |
| `example_02_structured_logging.py` | JSON-structured logging for agents |
| `example_03_agent_logger.py` | Complete AgentLogger class with tracing |
| `example_04_tracing.py` | Detailed tracing for tool calls and decisions |
| `example_05_metrics.py` | Performance metrics collection |
| `example_06_log_levels.py` | Log level configuration patterns |
| `example_07_aggregation.py` | Log aggregation for production |
| `example_08_integration.py` | Full integration with an observable agent |
| `exercise_solution.py` | Solution to the practical exercise |

## Prerequisites

1. Python 3.10+
2. A `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

3. Required packages:
   ```bash
   uv add anthropic python-dotenv
   ```

## Running the Examples

Most examples can be run directly:

```bash
uv run python example_01_basic_logging.py
uv run python example_02_structured_logging.py
```

Examples 04 and 08 make actual API calls and require a valid API key:

```bash
uv run python example_04_tracing.py
uv run python example_08_integration.py
```

The exercise solution starts a simple web server:

```bash
uv run python exercise_solution.py
# Then open http://localhost:8000 in your browser
```

## Key Concepts

1. **Structured Logging**: JSON output for machine-readable logs
2. **Request Tracing**: Following a request through all operations
3. **Tool Call Tracking**: Recording what tools were called and why
4. **Metrics Collection**: Aggregating performance data
5. **Log Levels**: Filtering logs by severity
6. **Correlation Context**: Linking logs from the same request
