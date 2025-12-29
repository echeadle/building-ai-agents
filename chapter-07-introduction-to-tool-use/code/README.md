# Chapter 7 Code Examples

This directory contains code examples for Chapter 7: Introduction to Tool Use.

## Files

### `example_01_naked_llm.py`

Demonstrates the limitations of an LLM without tools. Shows how Claude responds when asked about information it cannot access (current stock prices, real-time weather, etc.).

**Run:** `uv run python example_01_naked_llm.py`

### `example_02_tool_preview.py`

A preview of tool use that demonstrates the basic structure. This example shows how to define a tool and send it to Claude, but doesn't yet handle the tool execution (that comes in Chapter 9).

**Run:** `uv run python example_02_tool_preview.py`

### `exercise.py`

Solution to the practical exercise: designing tool definitions for a personal assistant. Contains example tool definitions with detailed descriptions and parameter schemas.

**Run:** `uv run python exercise.py` (prints the tool definitions)

## Prerequisites

Before running these examples, ensure you have:

1. Created a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Installed the required packages:
   ```bash
   uv add anthropic python-dotenv
   ```

## Notes

- Chapter 7 is primarily conceptual, so these examples are simpler than later chapters
- The tool preview shows structure only—full tool handling is covered in Chapter 9
- The exercise solution demonstrates tool design patterns you'll use throughout the book
