"""
Token counting and conversation truncation strategies.

Chapter 5: Understanding Messages and Conversations

This example demonstrates:
- Understanding token limits
- Different truncation strategies
- When and how to trim conversation history
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    This is a rough estimate: ~4 characters per token for English.
    For precise counts, use the API's usage response.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def show_token_basics():
    """Demonstrate basic token concepts."""
    
    print("=" * 60)
    print("UNDERSTANDING TOKENS")
    print("=" * 60)
    
    examples = [
        "Hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Anthropic's Claude is an AI assistant that uses the Transformer architecture.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    ]
    
    print("\nToken estimates (rough: ~4 chars per token):\n")
    for text in examples:
        chars = len(text)
        est_tokens = estimate_tokens(text)
        words = len(text.split())
        print(f"  '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"    Characters: {chars}, Words: {words}, Est. Tokens: {est_tokens}")
        print()


def demonstrate_usage_response():
    """Show how to get actual token counts from API response."""
    
    print("=" * 60)
    print("ACTUAL TOKEN COUNTS FROM API")
    print("=" * 60)
    
    client = anthropic.Anthropic()
    
    # Make a request and examine the usage
    messages = [
        {"role": "user", "content": "What are three interesting facts about the moon?"}
    ]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=messages
    )
    
    print("\nRequest sent. Usage statistics:")
    print(f"  Input tokens: {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    print(f"  Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")
    
    print(f"\nResponse preview:")
    print(f"  {response.content[0].text[:200]}...")


# ---------------------------------------------------------------------
# TRUNCATION STRATEGIES
# ---------------------------------------------------------------------

def truncate_recent_only(messages: list, max_pairs: int = 10) -> list:
    """
    Strategy 1: Keep only the most recent message pairs.
    
    Simple and fast, but loses early context.
    
    Args:
        messages: Full conversation history
        max_pairs: Maximum number of user/assistant pairs to keep
        
    Returns:
        Truncated message list
    """
    max_messages = max_pairs * 2
    
    if len(messages) <= max_messages:
        return messages
    
    return messages[-max_messages:]


def truncate_keep_first_and_recent(
    messages: list,
    keep_first: int = 2,
    keep_recent: int = 16
) -> list:
    """
    Strategy 2: Keep first N messages and most recent M messages.
    
    Preserves initial context (often important) plus recent conversation.
    
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


def truncate_with_summary(
    client: anthropic.Anthropic,
    messages: list,
    keep_recent: int = 10
) -> list:
    """
    Strategy 3: Summarize old messages and keep recent ones.
    
    Most sophisticated - preserves information in compressed form.
    Adds API call overhead.
    
    Args:
        client: Anthropic client
        messages: Full conversation history
        keep_recent: Number of recent messages to keep verbatim
        
    Returns:
        List with summary message + recent messages
    """
    # Don't summarize if not needed
    if len(messages) <= keep_recent + 2:
        return messages
    
    # Messages to summarize
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    
    # Build summary prompt
    summary_prompt = "Summarize this conversation in 2-3 sentences, capturing the key facts and context that would be important for continuing the conversation:\n\n"
    for msg in old_messages:
        role = msg['role'].upper()
        content = msg['content'][:200]  # Limit each message
        summary_prompt += f"{role}: {content}\n"
    
    # Get summary
    summary_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    summary = summary_response.content[0].text
    
    # Create new history with summary as initial context
    # We need to maintain the alternating pattern
    summary_message = {
        "role": "user",
        "content": f"[Previous conversation summary: {summary}]\n\nLet's continue our conversation."
    }
    
    placeholder_response = {
        "role": "assistant",
        "content": "I understand the context from our previous discussion. Please continue."
    }
    
    return [summary_message, placeholder_response] + recent_messages


def demonstrate_truncation_strategies():
    """Show how each truncation strategy works."""
    
    print("\n" + "=" * 60)
    print("TRUNCATION STRATEGIES COMPARISON")
    print("=" * 60)
    
    # Create a sample long conversation
    sample_conversation = []
    for i in range(1, 21):  # 20 exchanges = 40 messages
        sample_conversation.append({
            "role": "user",
            "content": f"User message {i}: This is message number {i} in our conversation."
        })
        sample_conversation.append({
            "role": "assistant",
            "content": f"Assistant response {i}: I acknowledge message {i}."
        })
    
    print(f"\nOriginal conversation: {len(sample_conversation)} messages")
    
    # Strategy 1: Recent only
    print("\n--- Strategy 1: Recent Only (keep last 5 pairs) ---")
    truncated1 = truncate_recent_only(sample_conversation, max_pairs=5)
    print(f"Result: {len(truncated1)} messages")
    print(f"First message: '{truncated1[0]['content'][:50]}...'")
    print(f"Last message: '{truncated1[-1]['content'][:50]}...'")
    
    # Strategy 2: First + Recent
    print("\n--- Strategy 2: First + Recent (2 first + 8 recent) ---")
    truncated2 = truncate_keep_first_and_recent(
        sample_conversation, 
        keep_first=2, 
        keep_recent=8
    )
    print(f"Result: {len(truncated2)} messages")
    print(f"First message: '{truncated2[0]['content'][:50]}...'")
    print(f"Gap? Messages 2 to {len(sample_conversation) - 8} are skipped")
    print(f"Last message: '{truncated2[-1]['content'][:50]}...'")
    
    # Strategy 3: Summary (requires API call)
    print("\n--- Strategy 3: Summarization ---")
    print("(This strategy requires an API call to create the summary)")
    print("See truncate_with_summary() function for implementation")
    print("Best for agents that need to remember decisions from early in a task")


def demonstrate_live_truncation():
    """Show truncation with a real conversation."""
    
    print("\n" + "=" * 60)
    print("LIVE TRUNCATION DEMONSTRATION")
    print("=" * 60)
    
    client = anthropic.Anthropic()
    
    # Build a conversation where early context matters
    conversation = [
        {"role": "user", "content": "My name is Bob and I'm building a weather app."},
        {"role": "assistant", "content": "Hi Bob! A weather app sounds like a great project. What features are you planning to include?"},
        {"role": "user", "content": "I want to show current temperature and a 5-day forecast."},
        {"role": "assistant", "content": "Those are solid core features. For the temperature, will you be using Celsius or Fahrenheit, or giving users a choice?"},
        {"role": "user", "content": "Let's do Celsius with an option to switch."},
        {"role": "assistant", "content": "Good choice for international users. For the forecast data, have you chosen a weather API yet?"},
        {"role": "user", "content": "I'm thinking OpenWeatherMap."},
        {"role": "assistant", "content": "OpenWeatherMap is a popular choice with a generous free tier. They provide all the data you'd need for your features."},
        {"role": "user", "content": "What about the UI framework?"},
        {"role": "assistant", "content": "For a weather app, you have several good options. React or Vue for web, or Flutter for cross-platform mobile. What's your target platform?"},
        {"role": "user", "content": "Web for now, maybe mobile later."},
        {"role": "assistant", "content": "Starting with web makes sense. React would be a solid choice - it's well-documented and the skills transfer well to React Native for mobile later."},
    ]
    
    # Add a final question that requires early context
    conversation.append({
        "role": "user",
        "content": "What was my name again, and what am I building?"
    })
    
    print(f"\nOriginal conversation: {len(conversation)} messages")
    
    # Without truncation
    print("\n--- Full History ---")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=conversation
    )
    print(f"Claude: {response.content[0].text}")
    
    # With aggressive truncation (loses early context)
    print("\n--- Recent-Only Truncation (loses early context) ---")
    truncated = truncate_recent_only(conversation, max_pairs=2)
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=truncated
        )
        print(f"Claude: {response.content[0].text}")
    except anthropic.APIError as e:
        print(f"Error: {e}")
    
    # With first + recent truncation (preserves early context)
    print("\n--- First + Recent Truncation (preserves context) ---")
    truncated = truncate_keep_first_and_recent(conversation, keep_first=4, keep_recent=4)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=truncated
    )
    print(f"Claude: {response.content[0].text}")
    
    print("\n💡 Notice how keeping the first messages preserves important context!")


if __name__ == "__main__":
    show_token_basics()
    demonstrate_usage_response()
    demonstrate_truncation_strategies()
    demonstrate_live_truncation()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Track token usage via response.usage")
    print("  2. Choose truncation strategy based on your needs")
    print("  3. Early context often contains critical information")
    print("  4. Summarization preserves info but costs an API call")
    print("=" * 60)
