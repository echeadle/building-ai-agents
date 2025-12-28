"""
Handling conversation truncation.

This example shows strategies for managing long conversations
that might exceed token limits.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


def truncate_conversation_simple(
    messages: list,
    max_messages: int = 50
) -> list:
    """
    Simple truncation: keep only the most recent messages.
    
    Args:
        messages: Full conversation history
        max_messages: Maximum messages to keep
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_messages:
        return messages
    
    return messages[-max_messages:]


def truncate_conversation_with_context(
    messages: list,
    max_messages: int = 50,
    context_messages: int = 2
) -> list:
    """
    Truncate while preserving early context.
    
    Keeps the first few messages for context, adds a truncation
    notice, then keeps the most recent messages.
    
    Args:
        messages: Full conversation history
        max_messages: Maximum messages to keep
        context_messages: Number of early messages to preserve
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_messages:
        return messages
    
    # Keep first messages for context
    early_context = messages[:context_messages]
    
    # Calculate how many recent messages to keep
    recent_count = max_messages - context_messages - 1  # -1 for summary
    recent_messages = messages[-recent_count:]
    
    # Add a summary message
    summary = {
        "role": "user",
        "content": "[Note: Earlier parts of this conversation have been truncated for brevity. The key context from the beginning is preserved above.]"
    }
    
    return early_context + [summary] + recent_messages


def summarize_and_truncate(
    messages: list,
    client: anthropic.Anthropic,
    threshold: int = 30,
    keep_recent: int = 10
) -> list:
    """
    Summarize older messages when conversation gets long.
    
    This is more sophisticated - uses Claude to summarize old
    messages, preserving semantic content while reducing tokens.
    
    Args:
        messages: Full conversation history
        client: Anthropic client for summarization
        threshold: When to trigger summarization
        keep_recent: Number of recent messages to keep intact
        
    Returns:
        Conversation with older parts summarized
    """
    if len(messages) <= threshold:
        return messages
    
    # Messages to summarize (keeping recent ones intact)
    to_summarize = messages[:-keep_recent]
    to_keep = messages[-keep_recent:]
    
    # Create summary prompt
    conversation_text = json.dumps(to_summarize, indent=2)
    
    summary_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Please summarize this conversation excerpt in 2-3 sentences, 
capturing the key points, any important facts mentioned, and the main topics discussed:

{conversation_text}

Provide only the summary, no other text."""
        }]
    )
    
    summary_text = summary_response.content[0].text
    
    # Return summary + recent messages
    return [
        {
            "role": "user", 
            "content": f"[Summary of earlier conversation: {summary_text}]"
        },
        {
            "role": "assistant", 
            "content": "I understand. I'll keep this context in mind as we continue."
        }
    ] + to_keep


def sliding_window_truncation(
    messages: list,
    max_tokens_estimate: int = 100000,
    avg_tokens_per_message: int = 200
) -> list:
    """
    Token-aware sliding window truncation.
    
    Estimates token count and removes oldest messages to stay
    under the limit.
    
    Args:
        messages: Full conversation history
        max_tokens_estimate: Approximate max tokens to keep
        avg_tokens_per_message: Estimate of tokens per message
        
    Returns:
        Truncated message list
    """
    max_messages = max_tokens_estimate // avg_tokens_per_message
    
    if len(messages) <= max_messages:
        return messages
    
    return messages[-max_messages:]


def demonstrate_truncation():
    """Demonstrate different truncation strategies."""
    print("Demonstrating Conversation Truncation Strategies")
    print("=" * 50)
    
    # Create a sample long conversation
    messages = []
    for i in range(60):
        messages.append({
            "role": "user",
            "content": f"This is user message {i + 1}. It contains some information."
        })
        messages.append({
            "role": "assistant",
            "content": f"This is assistant response {i + 1}. Acknowledged."
        })
    
    print(f"\nOriginal conversation length: {len(messages)} messages")
    
    # Simple truncation
    print("\n1. Simple Truncation (keep last 20):")
    simple = truncate_conversation_simple(messages, max_messages=20)
    print(f"   Result: {len(simple)} messages")
    print(f"   First message: {simple[0]['content'][:50]}...")
    
    # Context-preserving truncation
    print("\n2. Context-Preserving Truncation:")
    with_context = truncate_conversation_with_context(messages, max_messages=20, context_messages=4)
    print(f"   Result: {len(with_context)} messages")
    print(f"   First message: {with_context[0]['content'][:50]}...")
    print(f"   Has summary: {any('[Note:' in str(m) for m in with_context)}")
    
    # Sliding window
    print("\n3. Token-Aware Sliding Window:")
    sliding = sliding_window_truncation(
        messages, 
        max_tokens_estimate=5000, 
        avg_tokens_per_message=200
    )
    print(f"   Result: {len(sliding)} messages")
    
    # Summarization (requires API call)
    print("\n4. Summarization-based Truncation:")
    print("   (Requires API call - demonstrating with shorter example)")
    
    # Create a shorter conversation for summarization demo
    short_messages = [
        {"role": "user", "content": "Hi, I'm planning a trip to Japan."},
        {"role": "assistant", "content": "That sounds exciting! When are you thinking of going?"},
        {"role": "user", "content": "I'm thinking about cherry blossom season, late March."},
        {"role": "assistant", "content": "Great choice! Late March to early April is beautiful."},
        {"role": "user", "content": "My budget is around $3000 for two weeks."},
        {"role": "assistant", "content": "That's doable! Hostels and trains can help stretch it."},
        {"role": "user", "content": "I definitely want to see Tokyo and Kyoto."},
        {"role": "assistant", "content": "Those are must-sees. Consider the JR Pass for travel."},
        # Recent messages to keep
        {"role": "user", "content": "What about food recommendations?"},
        {"role": "assistant", "content": "Try ramen, sushi, and don't miss convenience store food!"},
        {"role": "user", "content": "Any specific restaurants?"},
        {"role": "assistant", "content": "Ichiran for ramen, Tsukiji outer market for sushi."}
    ]
    
    client = anthropic.Anthropic()
    summarized = summarize_and_truncate(
        short_messages, 
        client, 
        threshold=6, 
        keep_recent=4
    )
    
    print(f"   Original: {len(short_messages)} messages")
    print(f"   After summarization: {len(summarized)} messages")
    print(f"   Summary: {summarized[0]['content'][:100]}...")


class ConversationManager:
    """
    A conversation manager with automatic truncation.
    
    Automatically manages conversation length to prevent
    token limit issues.
    """
    
    def __init__(
        self,
        max_messages: int = 100,
        truncation_strategy: str = "context_preserving"
    ):
        """
        Initialize conversation manager.
        
        Args:
            max_messages: Maximum messages before truncation
            truncation_strategy: One of 'simple', 'context_preserving'
        """
        self.messages: list = []
        self.max_messages = max_messages
        self.truncation_strategy = truncation_strategy
        self.truncation_count = 0
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message, truncating if necessary."""
        self.messages.append({"role": role, "content": content})
        
        if len(self.messages) > self.max_messages:
            self._truncate()
    
    def _truncate(self) -> None:
        """Truncate the conversation."""
        if self.truncation_strategy == "simple":
            self.messages = truncate_conversation_simple(
                self.messages, 
                self.max_messages
            )
        else:
            self.messages = truncate_conversation_with_context(
                self.messages,
                self.max_messages
            )
        self.truncation_count += 1
    
    def get_messages(self) -> list:
        """Get current messages."""
        return self.messages.copy()
    
    def get_stats(self) -> dict:
        """Get conversation statistics."""
        return {
            "current_messages": len(self.messages),
            "max_messages": self.max_messages,
            "truncation_count": self.truncation_count
        }


if __name__ == "__main__":
    demonstrate_truncation()
    
    print("\n" + "=" * 50)
    print("Demonstrating ConversationManager")
    print("=" * 50)
    
    manager = ConversationManager(max_messages=10)
    
    # Add many messages
    for i in range(25):
        manager.add_message("user", f"Message {i + 1}")
        manager.add_message("assistant", f"Response {i + 1}")
    
    stats = manager.get_stats()
    print(f"\nAfter adding 50 messages:")
    print(f"  Current messages: {stats['current_messages']}")
    print(f"  Truncation events: {stats['truncation_count']}")
