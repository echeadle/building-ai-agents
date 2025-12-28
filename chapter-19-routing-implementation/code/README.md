# Chapter 19 Code Examples

This directory contains all code examples for Chapter 19: Routing - Implementation.

## Files

### `01_simple_classifier.py`
A basic LLM-based classifier that categorizes customer queries into predefined categories (BILLING, TECHNICAL, ACCOUNT, GENERAL). Demonstrates the classification prompt design and validation logic.

### `02_route_handlers.py`
Specialized route handlers with domain-specific system prompts. Shows how to create handlers with different expertise areas using a factory pattern.

### `03_customer_service_router.py`
The complete customer service router that combines classification and handlers. This is the main working example from the chapter.

### `04_router_class.py`
A reusable `Router` class that encapsulates the routing pattern. Includes method chaining, auto-generated classification prompts, and a handler factory.

### `05_testing_classifier.py`
Testing classification accuracy with a labeled test dataset. Includes per-category metrics and a detailed accuracy report.

### `exercise_solution.py`
Solution to the practical exercise: a documentation chatbot router with four specialized categories.

## Running the Examples

1. Make sure you have your `.env` file with `ANTHROPIC_API_KEY` set
2. Install dependencies: `uv add anthropic python-dotenv`
3. Run any example: `python 01_simple_classifier.py`

## Key Concepts

- **Classification prompts**: Clear category definitions with examples for reliable routing
- **Handler factory**: Creating handlers with embedded system prompts
- **Accuracy testing**: Measuring classifier performance with labeled data
- **Router class pattern**: Reusable abstraction for routing workflows
