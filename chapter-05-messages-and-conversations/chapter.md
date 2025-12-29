---
chapter: 5
title: "Understanding Messages and Conversations"
part: 1
date: 2025-01-15
draft: false
---

# Chapter 5: Understanding Messages and Conversations

## Introduction

In Chapter 4, you made your first API call to Claude—a single question with a single response. But real conversations don't work that way. When you chat with a friend, you both remember what was said before. You build on previous points, reference earlier topics, and maintain context throughout the discussion.

Here's the critical insight that will shape everything you build: **Claude doesn't remember anything between API calls**. Every single request you make is completely independent. Claude has no memory of what you asked five seconds ago or five days ago. If you want a conversation, _you_ must provide the entire conversation history with every single request.

This might seem like a limitation, but it's actually a superpower. Because you control the conversation history, you control exactly what the model "remembers." You can edit history, summarize it, or curate it precisely. For AI agents, this control is essential—your agent decides what context matters and what to forget.

In this chapter, you'll learn how conversations actually work under the hood and build a chat system that maintains context across multiple exchanges.

## Learning Objectives

By the end of this chapter, you will be able to:

-   Explain the structure of the messages array and the role of each message type
-   Build and maintain conversation history across multiple API calls
-   Implement a working chat loop that preserves context
-   Understand token limits and implement basic truncation strategies
-   Describe why statelessness matters for building reliable agents

## The Messages Array: Anatomy of a Conversation

Every API call to Claude includes a `messages` parameter—an array that contains the entire conversation. Let's examine its structure:

```python
messages = [
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"}
]
```

Each message is a dictionary with two required keys:

-   **`role`**: Who said this—either `"user"` or `"assistant"`
-   **`content`**: What they said—the actual text

When you send this array to Claude, it sees the entire conversation and responds in context. In the example above, Claude understands that "its" refers to Paris because it can see the full conversation history.

### The Three Roles

The Anthropic API supports three distinct roles:

| Role        | Purpose                                  | When It Appears                           |
| ----------- | ---------------------------------------- | ----------------------------------------- |
| `system`    | Sets behavior, persona, and instructions | Separate parameter, not in messages array |
| `user`      | Human messages and inputs                | In messages array                         |
| `assistant` | Claude's responses                       | In messages array                         |

> **Note:** Unlike some other APIs, Claude's system prompt is passed as a separate `system` parameter, not as a message in the array. We'll explore system prompts in depth in Chapter 6.

### Message Ordering Rules

The messages array must follow these rules:

1. **Messages must alternate** between `user` and `assistant` roles
2. **The first message must be from `user`** (after the optional system prompt)
3. **The last message should be from `user`** (this is what Claude responds to)

Here's what a valid conversation looks like:

```python
# Valid conversation structure
messages = [
    {"role": "user", "content": "Hello!"},                    # User starts
    {"role": "assistant", "content": "Hi there!"},           # Assistant responds
    {"role": "user", "content": "How are you?"},             # User continues
    {"role": "assistant", "content": "I'm doing well!"},     # Assistant responds
    {"role": "user", "content": "That's great!"}             # User's turn (Claude will respond)
]
```

If you violate these rules, the API will return an error. This is a common source of bugs when building conversation systems.

## The Stateless Nature of API Calls

This is the most important concept in this chapter, so let's make it crystal clear:

**Every API call to Claude is completely independent. Claude has zero memory between calls.**

When you make a request, Claude processes it, sends a response, and immediately forgets everything. The next request could be about a completely different topic from a completely different user—Claude has no way to know or care.

Consider this scenario:

```python
# First API call
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "My name is Alice."}]
)
# Claude responds: "Nice to meet you, Alice!"

# Second API call - Claude has NO memory of the first call
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "What's my name?"}]
)
# Claude responds: "I don't know your name - you haven't told me!"
```

To make Claude "remember" your name, you must include the previous exchange:

```python
# Third API call - WITH history
response3 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"}
    ]
)
# Claude responds: "Your name is Alice!"
```

### Why Statelessness Matters for Agents

This stateless design has profound implications for building AI agents:

1. **You control memory**: You decide what the agent "remembers" by curating the conversation history
2. **Reproducibility**: Given the same messages array, you'll get consistent behavior
3. **Scalability**: The server doesn't need to track millions of conversation states
4. **Debugging**: You can replay exact conversations by saving the messages array
5. **Flexibility**: You can edit, summarize, or transform history however you need

For agents, this means you're not just building a chat interface—you're building a memory management system. What your agent remembers, forgets, and how it organizes information are all design decisions you control.

## Building Conversation History

Now let's build a system that properly maintains conversation history. The pattern is straightforward:

1. Start with an empty messages list
2. Add the user's message to the list
3. Send the entire list to Claude
4. Add Claude's response to the list
5. Repeat from step 2

Here's the basic implementation:

```python
"""
Building conversation history step by step.
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# Initialize empty conversation history
conversation_history = []

def chat(user_message: str) -> str:
    """
    Send a message and get a response, maintaining conversation history.

    Args:
        user_message: The user's input text

    Returns:
        Claude's response text
    """
    # Step 1: Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Step 2: Send entire history to Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=conversation_history
    )

    # Step 3: Extract the response text
    assistant_message = response.content[0].text

    # Step 4: Add Claude's response to history
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message
```

Each call to `chat()` adds both the user message and Claude's response to the history. The next call includes everything that came before.

### Watching History Build

Let's trace through a conversation to see how history accumulates:

```python
# After first exchange
print(chat("Hi, I'm learning Python."))
# History: [user: "Hi, I'm learning Python.", assistant: "...response..."]

# After second exchange
print(chat("What should I learn first?"))
# History: [user: "Hi...", assistant: "...", user: "What should...", assistant: "..."]

# After third exchange
print(chat("Can you give me an example?"))
# History: [user: "Hi...", assistant: "...", user: "What...", assistant: "...",
#           user: "Can you...", assistant: "..."]
```

Each exchange adds two messages to the history. After 10 exchanges, you'd have 20 messages in the array.

## Token Limits and Conversation Length

There's a catch to maintaining full conversation history: **Claude has a context window limit**. The context window is the maximum amount of text (measured in tokens) that Claude can process in a single request, including both the input messages and the response.

Claude's context window depends on the model version, but it's finite. When your conversation history plus the expected response exceeds this limit, the API returns an error.

### Understanding Tokens

Tokens are how language models process text. They're not exactly words or characters—they're chunks that the model learned during training. As a rough estimate:

-   1 token ≈ 4 characters in English
-   1 token ≈ 0.75 words
-   100 tokens ≈ 75 words

The Anthropic SDK provides a way to count tokens:

```python
import anthropic

client = anthropic.Anthropic()

# Count tokens in a text string
text = "Hello, how are you doing today?"
token_count = client.count_tokens(text)
print(f"Token count: {token_count}")
```

> **Note:** For precise token counts in production, you can use the `usage` field in the API response, which tells you exactly how many input and output tokens were used.

### Basic Truncation Strategies

When conversations get too long, you need a strategy for trimming them. Here are three common approaches:

#### Strategy 1: Keep Recent Messages Only

The simplest approach—keep only the N most recent message pairs:

```python
def truncate_recent(messages: list, max_pairs: int = 10) -> list:
    """
    Keep only the most recent message pairs.

    Args:
        messages: Full conversation history
        max_pairs: Maximum number of user/assistant pairs to keep

    Returns:
        Truncated message list
    """
    # Each pair is 2 messages (user + assistant)
    max_messages = max_pairs * 2

    if len(messages) <= max_messages:
        return messages

    # Keep the most recent messages
    return messages[-max_messages:]
```

**Pros:** Simple, fast, preserves recent context  
**Cons:** Loses important early context (like the user's name or initial instructions)

#### Strategy 2: Keep First + Recent

Preserve the beginning of the conversation (often contains important context) plus recent messages:

```python
def truncate_keep_first(
    messages: list,
    keep_first: int = 2,
    keep_recent: int = 16
) -> list:
    """
    Keep first N messages and most recent M messages.

    Args:
        messages: Full conversation history
        keep_first: Number of initial messages to preserve
        keep_recent: Number of recent messages to preserve

    Returns:
        Truncated message list
    """
    total_keep = keep_first + keep_recent

    if len(messages) <= total_keep:
        return messages

    first_messages = messages[:keep_first]
    recent_messages = messages[-keep_recent:]

    return first_messages + recent_messages
```

**Pros:** Preserves initial context, maintains recent conversation  
**Cons:** Gap in the middle might cause confusion

#### Strategy 3: Summarize Old Messages

Have Claude summarize older parts of the conversation, then use that summary:

```python
def summarize_and_truncate(
    client: anthropic.Anthropic,
    messages: list,
    keep_recent: int = 10
) -> list:
    """
    Summarize old messages and keep recent ones.

    Args:
        client: Anthropic client
        messages: Full conversation history
        keep_recent: Number of recent messages to keep verbatim

    Returns:
        List with summary message + recent messages
    """
    if len(messages) <= keep_recent + 2:
        return messages

    # Messages to summarize
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Create summary request
    summary_prompt = "Summarize this conversation in 2-3 sentences, capturing key facts and context:\n\n"
    for msg in old_messages:
        summary_prompt += f"{msg['role'].upper()}: {msg['content']}\n"

    summary_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": summary_prompt}]
    )

    summary = summary_response.content[0].text

    # Create a new history with summary as context
    summary_message = {
        "role": "user",
        "content": f"[Previous conversation summary: {summary}]"
    }

    # Need to maintain alternating pattern
    placeholder_response = {
        "role": "assistant",
        "content": "I understand. Let's continue our conversation."
    }

    return [summary_message, placeholder_response] + recent_messages
```

**Pros:** Preserves important information in compressed form  
**Cons:** Adds API call overhead, summary might miss details

💡 **Tip:** For agents, Strategy 3 (summarization) is often worth the extra API call because agents may need to reference decisions or facts from much earlier in a task.

## Building a Complete Chat Loop

Now let's put everything together into a working interactive chat application. This will be the foundation for many agent interfaces you'll build later.

**File:** example_04_chat_loop.py

```python
"""
A complete chat loop with conversation history and basic truncation.

Chapter 5: Understanding Messages and Conversations
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class ChatSession:
    """
    Manages a conversation session with Claude.

    Maintains conversation history and handles basic truncation
    to stay within context limits.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_history_pairs: int = 20
    ):
        """
        Initialize a chat session.

        Args:
            model: The Claude model to use
            max_tokens: Maximum tokens in each response
            max_history_pairs: Maximum conversation pairs to retain
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_history_pairs = max_history_pairs
        self.conversation_history: list[dict] = []

    def _truncate_history(self) -> list[dict]:
        """
        Truncate history if it exceeds the maximum pairs.

        Returns:
            Truncated message list for API call
        """
        max_messages = self.max_history_pairs * 2

        if len(self.conversation_history) <= max_messages:
            return self.conversation_history

        # Keep first pair (may contain important context) and recent messages
        first_pair = self.conversation_history[:2]
        recent = self.conversation_history[-(max_messages - 2):]

        return first_pair + recent

    def send_message(self, user_message: str) -> str:
        """
        Send a message to Claude and get a response.

        Args:
            user_message: The user's input

        Returns:
            Claude's response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get (possibly truncated) history for API call
        messages_to_send = self._truncate_history()

        # Make API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages_to_send
        )

        # Extract response text
        assistant_message = response.content[0].text

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def get_history(self) -> list[dict]:
        """Return the full conversation history."""
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def message_count(self) -> int:
        """Return the number of messages in history."""
        return len(self.conversation_history)


def main():
    """Run an interactive chat session."""
    print("Chat with Claude (type 'quit' to exit, 'history' to see conversation)")
    print("-" * 60)

    session = ChatSession()

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        # Handle special commands
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower() == 'history':
            print(f"\n--- Conversation History ({session.message_count()} messages) ---")
            for msg in session.get_history():
                role = msg['role'].upper()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"{role}: {content}")
            continue

        if user_input.lower() == 'clear':
            session.clear_history()
            print("Conversation history cleared.")
            continue

        if not user_input:
            continue

        # Send message and get response
        try:
            response = session.send_message(user_input)
            print(f"\nClaude: {response}")
        except anthropic.APIError as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
```

### Using the Chat Session

Run the script and have a conversation:

```
Chat with Claude (type 'quit' to exit, 'history' to see conversation)
------------------------------------------------------------

You: Hi, my name is Alex and I'm building AI agents.

Claude: Hello Alex! That's exciting - building AI agents is a fascinating area...

You: What should I focus on first?

Claude: Since you mentioned you're building AI agents, I'd suggest starting with...

You: history

--- Conversation History (4 messages) ---
USER: Hi, my name is Alex and I'm building AI agents.
ASSISTANT: Hello Alex! That's exciting - building AI agents is a fascinating...
USER: What should I focus on first?
ASSISTANT: Since you mentioned you're building AI agents, I'd suggest starting...
```

Notice how Claude remembers that your name is Alex and that you're building agents—because that information is in the conversation history you're sending with each request.

## Why This Matters for Agents

Everything you've learned in this chapter is foundational to building AI agents. Here's how these concepts apply:

### Agents Need Working Memory

An agent performing a multi-step task needs to remember what it's already done. The conversation history _is_ its working memory. Each tool call, each decision, each intermediate result gets recorded in the messages array, allowing the agent to reason about its progress.

### Context Is Your Control Mechanism

By controlling what's in the conversation history, you control what the agent knows and considers. This is how you:

-   Keep the agent focused on the current task
-   Provide relevant background information
-   Filter out distracting or irrelevant details
-   Implement different "memory" strategies

### History Shapes Behavior

The conversation history isn't just data—it's behavioral guidance. If your history shows the agent always confirming before taking actions, it will continue that pattern. If it shows the agent being verbose, expect verbosity. This is why curating history matters.

### Debugging Through History

When an agent behaves unexpectedly, the first thing to check is its conversation history. What did it see? What context was it given? The messages array is your debugging log—save it, examine it, and replay it to understand what went wrong.

## Common Pitfalls

### 1. Forgetting That Each Call Is Independent

**The mistake:** Assuming Claude remembers something from a previous API call without including it in the messages array.

**The fix:** Always include all context you want Claude to have in the current request. If in doubt, include it.

### 2. Violating Message Order Rules

**The mistake:** Sending two user messages in a row, or starting with an assistant message.

```python
# This will cause an error
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "user", "content": "Are you there?"}  # Two user messages in a row!
]
```

**The fix:** If you need to combine multiple user inputs, concatenate them into a single message:

```python
messages = [
    {"role": "user", "content": "Hello\n\nAre you there?"}
]
```

### 3. Ignoring Token Limits Until They Break

**The mistake:** Building up conversation history without any truncation, then getting errors when the context window fills up.

**The fix:** Implement truncation from the start, even if you don't think you need it yet. It's much easier to add when you're building than to retrofit later.

## Practical Exercise

**Task:** Build an enhanced chat application with conversation export/import

**Requirements:**

1. Create a `PersistentChat` class that can:

    - Save conversation history to a JSON file
    - Load conversation history from a JSON file
    - Resume a previous conversation seamlessly

2. Add a command `/save <filename>` to save the current conversation

3. Add a command `/load <filename>` to load a previous conversation

4. When loading a conversation, display a summary of what was discussed

**Hints:**

-   Use Python's `json` module for serialization
-   The messages list is already in a JSON-compatible format
-   Consider what happens if someone tries to load a file that doesn't exist
-   Think about how to verify the loaded data is valid

**Solution:** See `code/exercise.py`

## Key Takeaways

-   **The messages array is the entire conversation**: Claude sees only what you send in each request—nothing more, nothing less.

-   **You manage all state**: The API is stateless by design. Your code is responsible for maintaining, curating, and truncating conversation history.

-   **Messages must alternate roles**: User → Assistant → User → Assistant. Violations cause errors.

-   **Token limits require truncation strategies**: Choose an approach (recent only, first + recent, or summarization) based on your use case.

-   **For agents, conversation history is working memory**: What's in the messages array determines what your agent "knows" about its current task.

## What's Next

Now that you understand how conversations work, you're ready to shape Claude's behavior more precisely. In Chapter 6, we'll explore system prompts—the instructions that define your agent's persona, capabilities, and constraints. You'll learn how to craft effective system prompts that make your agents behave consistently and reliably.
