"""
Building conversation history step by step.

Chapter 5: Understanding Messages and Conversations

This example shows how to properly accumulate conversation
history across multiple exchanges.
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


# Global conversation history
conversation_history: list[dict] = []

# Create client once
client = anthropic.Anthropic()


def chat(user_message: str) -> str:
    """
    Send a message and get a response, maintaining conversation history.
    
    This function demonstrates the core pattern for building
    multi-turn conversations with Claude.
    
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


def print_history():
    """Print the current conversation history."""
    print("\n--- Current History ---")
    print(f"Total messages: {len(conversation_history)}")
    for i, msg in enumerate(conversation_history, 1):
        role = msg['role'].upper()
        content = msg['content']
        # Truncate long content for display
        if len(content) > 80:
            content = content[:80] + "..."
        print(f"  {i}. [{role}]: {content}")
    print("-" * 40)


def demonstrate_history_building():
    """
    Walk through a conversation showing how history accumulates.
    """
    
    print("=" * 60)
    print("BUILDING CONVERSATION HISTORY")
    print("=" * 60)
    
    # Clear any existing history
    conversation_history.clear()
    
    # Exchange 1
    print("\n>>> Exchange 1")
    print("User: Hi! I'm learning to build AI agents with Python.")
    response1 = chat("Hi! I'm learning to build AI agents with Python.")
    print(f"Claude: {response1[:200]}...")
    print_history()
    
    # Exchange 2
    print("\n>>> Exchange 2")
    print("User: What's the most important concept I should understand first?")
    response2 = chat("What's the most important concept I should understand first?")
    print(f"Claude: {response2[:200]}...")
    print_history()
    
    # Exchange 3 - Reference previous context
    print("\n>>> Exchange 3")
    print("User: Can you give me a simple example of that?")
    response3 = chat("Can you give me a simple example of that?")
    print(f"Claude: {response3[:200]}...")
    print_history()
    
    # Exchange 4 - Test that Claude remembers context
    print("\n>>> Exchange 4 (Testing Memory)")
    print("User: What did I say I was learning at the start of our conversation?")
    response4 = chat("What did I say I was learning at the start of our conversation?")
    print(f"Claude: {response4}")
    print_history()


def demonstrate_history_growth():
    """
    Show how history grows with each exchange.
    """
    
    print("\n" + "=" * 60)
    print("HISTORY GROWTH VISUALIZATION")
    print("=" * 60)
    
    # Clear history
    conversation_history.clear()
    
    messages_to_send = [
        "What is 1+1?",
        "What about 2+2?",
        "And 3+3?",
        "What's the pattern here?",
        "Can you express this mathematically?"
    ]
    
    print("\nSending 5 messages and watching history grow:\n")
    
    for i, msg in enumerate(messages_to_send, 1):
        print(f"Exchange {i}:")
        print(f"  → Sending: '{msg}'")
        print(f"  → History size BEFORE call: {len(conversation_history)} messages")
        
        response = chat(msg)
        
        print(f"  → History size AFTER call: {len(conversation_history)} messages")
        print(f"  → Response: {response[:60]}...")
        print()
    
    print("Final history summary:")
    print(f"  Total messages: {len(conversation_history)}")
    print(f"  User messages: {len([m for m in conversation_history if m['role'] == 'user'])}")
    print(f"  Assistant messages: {len([m for m in conversation_history if m['role'] == 'assistant'])}")


if __name__ == "__main__":
    demonstrate_history_building()
    demonstrate_history_growth()
    
    print("\n" + "=" * 60)
    print("KEY PATTERN:")
    print("  1. Add user message to history")
    print("  2. Send entire history to API")
    print("  3. Add assistant response to history")
    print("  4. Repeat")
    print("=" * 60)
