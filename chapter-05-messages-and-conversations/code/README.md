# Chapter 5 Code Examples

This directory contains all runnable code examples for Chapter 5: Understanding Messages and Conversations.

## Files

### `example_01_message_structure.py`
Explores the structure of messages and demonstrates valid/invalid message arrays. Run this to understand the anatomy of conversation data.

### `example_02_single_exchange.py`
Shows how a single user-assistant exchange works, highlighting that each API call is independent with no memory.

### `example_03_conversation_history.py`
Demonstrates building conversation history step by step, showing how context accumulates across exchanges.

### `example_04_chat_loop.py`
The main deliverable: a complete interactive chat application with conversation history management and basic truncation. This is the `ChatSession` class you'll reference throughout the book.

### `example_05_token_counting.py`
Demonstrates token counting and shows different truncation strategies for managing long conversations.

### `exercise.py`
Solution to the practical exercise: a `PersistentChat` class that can save and load conversations to/from JSON files.

## Running the Examples

Make sure you have your `.env` file set up with your API key:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Then run any example:

```bash
python example_01_message_structure.py
python example_04_chat_loop.py  # Interactive chat
```

## Dependencies

All examples require:
- `anthropic` SDK
- `python-dotenv`

Install with:
```bash
uv add anthropic python-dotenv
```
