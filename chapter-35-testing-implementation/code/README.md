# Chapter 35: Testing AI Agents - Implementation

## Code Files

This directory contains all code examples for Chapter 35.

### Core Files

| File | Description |
|------|-------------|
| `calculator.py` | Calculator tool implementation for testing |
| `mock_llm.py` | Mock LLM class for deterministic testing |
| `testable_agent.py` | Minimal agent class designed for testing |
| `conftest.py` | Shared pytest fixtures |

### Test Files

| File | Description |
|------|-------------|
| `test_tools.py` | Unit tests for tools in isolation |
| `test_agent_loop.py` | Tests for the agentic loop |
| `test_with_mocks.py` | Demonstrations of mock LLM usage |
| `property_tests.py` | Property-based tests using Hypothesis |
| `test_suite.py` | Complete organized test suite |

### Exercise

| File | Description |
|------|-------------|
| `exercise_solution.py` | Solution for the chapter exercise |

## Running Tests

```bash
# Install dependencies
uv add pytest pytest-asyncio hypothesis pytest-cov

# Run all tests
pytest -v

# Run only unit tests
pytest -m unit -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run property-based tests with more examples
pytest property_tests.py --hypothesis-seed=42
```

## Directory Structure

```
code/
├── README.md              # This file
├── calculator.py          # Tool implementation
├── mock_llm.py           # Mock LLM for testing
├── testable_agent.py     # Agent class for testing
├── conftest.py           # Shared fixtures
├── test_tools.py         # Tool unit tests
├── test_agent_loop.py    # Agent loop tests
├── test_with_mocks.py    # Mock usage examples
├── property_tests.py     # Property-based tests
├── test_suite.py         # Complete test suite
└── exercise_solution.py  # Exercise solution
```
