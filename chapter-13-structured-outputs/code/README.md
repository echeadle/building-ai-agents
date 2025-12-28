# Chapter 13: Structured Outputs and Response Parsing - Code Examples

This directory contains runnable code examples for Chapter 13.

## Prerequisites

Before running these examples, ensure you have:

1. Python 3.10+ installed
2. An Anthropic API key in your `.env` file
3. Required packages installed:

```bash
uv add anthropic python-dotenv pydantic
```

## Code Files

### `example_01_basic_json.py`

Demonstrates the basics of requesting JSON output from Claude:
- Direct JSON requests in prompts
- Using system prompts for consistency
- Simple JSON parsing

Run: `python example_01_basic_json.py`

### `example_02_pydantic_validation.py`

Shows how to use Pydantic for response validation:
- Defining schemas with Pydantic models
- Validating responses against schemas
- Handling validation errors

Run: `python example_02_pydantic_validation.py`

### `example_03_robust_parsing.py`

Demonstrates robust parsing with error handling:
- Cleaning malformed JSON responses
- Retry logic for failed parses
- Error recovery strategies

Run: `python example_03_robust_parsing.py`

### `example_04_response_parser.py`

A complete, reusable response parsing utility:
- The `ResponseParser` class
- Automatic cleanup and validation
- Retry with correction requests

Run: `python example_04_response_parser.py`

### `example_05_document_analyzer.py`

Full example bringing everything together:
- Complex nested schemas
- Real-world document analysis
- Production-ready error handling

Run: `python example_05_document_analyzer.py`

### `exercise_resume_parser.py`

Solution to the chapter exercise:
- Resume parsing with structured output
- Nested Pydantic models for complex data
- Flexible date handling

Run: `python exercise_resume_parser.py`

## Environment Setup

Create a `.env` file in this directory (or parent directory):

```
ANTHROPIC_API_KEY=your-api-key-here
```

## Notes

- All examples use `claude-sonnet-4-20250514` as the default model
- You can substitute other Claude models if desired
- Examples include comprehensive error handling for production use
