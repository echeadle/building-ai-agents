"""
Demonstrating the stateless nature of API calls.

Chapter 5: Understanding Messages and Conversations

This example proves that Claude has NO memory between API calls.
Each call is completely independent.
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


def demonstrate_statelessness():
    """
    Prove that Claude doesn't remember between API calls.
    
    This is the most important concept in the chapter!
    """
    
    client = anthropic.Anthropic()
    
    print("=" * 60)
    print("DEMONSTRATING STATELESSNESS")
    print("=" * 60)
    
    # First API call: Tell Claude our name
    print("\n--- First API Call ---")
    print("Sending: 'My name is Alice and I love pizza.'")
    
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "My name is Alice and I love pizza."}
        ]
    )
    
    print(f"Claude says: {response1.content[0].text[:200]}...")
    
    # Second API call: Ask Claude what our name is (WITHOUT history)
    print("\n--- Second API Call (WITHOUT history) ---")
    print("Sending: 'What is my name?'")
    
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "What is my name?"}
        ]
    )
    
    print(f"Claude says: {response2.content[0].text}")
    print("\n⚠️  Notice: Claude doesn't know our name!")
    print("   Each API call is completely independent.")
    
    # Third API call: Ask again WITH history
    print("\n--- Third API Call (WITH history) ---")
    print("Sending the full conversation history...")
    
    response3 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "My name is Alice and I love pizza."},
            {"role": "assistant", "content": response1.content[0].text},
            {"role": "user", "content": "What is my name?"}
        ]
    )
    
    print(f"Claude says: {response3.content[0].text}")
    print("\n✓ Now Claude knows our name because it's in the history!")


def demonstrate_context_matters():
    """
    Show how the same question gets different answers based on context.
    """
    
    client = anthropic.Anthropic()
    
    print("\n" + "=" * 60)
    print("CONTEXT DETERMINES THE RESPONSE")
    print("=" * 60)
    
    question = "What color is it?"
    
    # Context 1: Talking about the sky
    print("\n--- Context 1: Sky ---")
    context1 = [
        {"role": "user", "content": "Let's talk about the sky on a clear day."},
        {"role": "assistant", "content": "Sure! The sky on a clear day is a beautiful sight. What would you like to know about it?"},
        {"role": "user", "content": question}
    ]
    
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=context1
    )
    
    print(f"Q: '{question}'")
    print(f"A: {response1.content[0].text}")
    
    # Context 2: Talking about grass
    print("\n--- Context 2: Grass ---")
    context2 = [
        {"role": "user", "content": "Let's talk about healthy grass in a lawn."},
        {"role": "assistant", "content": "Of course! Healthy lawn grass is important for many homeowners. What would you like to know?"},
        {"role": "user", "content": question}
    ]
    
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=context2
    )
    
    print(f"Q: '{question}'")
    print(f"A: {response2.content[0].text}")
    
    # Context 3: No context at all
    print("\n--- Context 3: No Context ---")
    context3 = [
        {"role": "user", "content": question}
    ]
    
    response3 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=context3
    )
    
    print(f"Q: '{question}'")
    print(f"A: {response3.content[0].text}")
    
    print("\n💡 Key insight: The SAME question produces DIFFERENT answers")
    print("   based entirely on what conversation history you provide!")


if __name__ == "__main__":
    demonstrate_statelessness()
    demonstrate_context_matters()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY: Claude has ZERO memory between API calls.")
    print("YOU must provide all context in the messages array.")
    print("=" * 60)
