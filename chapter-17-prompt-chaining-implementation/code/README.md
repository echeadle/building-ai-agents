# Chapter 17 Code Examples

This directory contains runnable code examples for Chapter 17: Prompt Chaining - Implementation.

## Files

### `example_01_simple_chain.py`
A basic two-step prompt chain demonstrating content generation followed by translation. This is the simplest possible chain implementation.

**Demonstrates:**
- Sequential LLM calls
- Passing output between steps
- Basic chain orchestration

### `example_02_quality_gates.py`
Enhanced chain with quality gates (validation) between steps. Shows both rule-based and LLM-based validation.

**Demonstrates:**
- Rule-based validation (length, content checks)
- LLM-based quality assessment
- Retry logic for failed validations
- Detailed result tracking

### `example_03_chain_class.py`
A reusable `Chain` class that encapsulates the chaining pattern for any sequential workflow.

**Demonstrates:**
- Object-oriented chain design
- Generic step abstraction
- Declarative chain configuration
- Full error handling and retry logic

### `example_04_context_passing.py`
Shows how to pass rich context between steps, allowing later steps to access outputs from any earlier step.

**Demonstrates:**
- ChainContext data structure
- Accessing original input from later steps
- Metadata tracking
- Multi-step research workflow

### `exercise.py`
Solution to the chapter exercise: a content repurposing chain that generates blog posts and social media content.

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
   uv run python example_01_simple_chain.py
   ```

## Key Concepts

- **Prompt Chaining**: Breaking complex tasks into sequential steps
- **Quality Gates**: Validation between steps to catch bad outputs early
- **Error Handling**: Retry logic and graceful failure
- **Context Passing**: Carrying data through the entire chain
