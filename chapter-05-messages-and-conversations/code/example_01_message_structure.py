"""
Exploring the structure of messages in the Anthropic API.

Chapter 5: Understanding Messages and Conversations

This example demonstrates:
- The anatomy of a message dictionary
- Valid vs invalid message arrays
- The alternating role requirement
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


def demonstrate_message_structure():
    """Show the structure of messages in detail."""
    
    print("=" * 60)
    print("MESSAGE STRUCTURE DEMONSTRATION")
    print("=" * 60)
    
    # A single message
    single_message = {
        "role": "user",
        "content": "What is Python?"
    }
    
    print("\n1. Single Message Structure:")
    print(f"   Message: {single_message}")
    print(f"   Role: {single_message['role']}")
    print(f"   Content: {single_message['content']}")
    
    # A complete conversation
    conversation = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's its population?"}
    ]
    
    print("\n2. Complete Conversation (3 messages):")
    for i, msg in enumerate(conversation, 1):
        print(f"   Message {i}:")
        print(f"      Role: {msg['role']}")
        print(f"      Content: {msg['content']}")
    
    # Message roles explained
    print("\n3. The Three Roles:")
    print("   - 'user': Human messages and inputs")
    print("   - 'assistant': Claude's responses")
    print("   - 'system': Passed separately, not in messages array")


def demonstrate_valid_invalid_messages():
    """Show examples of valid and invalid message arrays."""
    
    print("\n" + "=" * 60)
    print("VALID VS INVALID MESSAGE ARRAYS")
    print("=" * 60)
    
    # Valid: Alternating roles, starting with user
    valid_conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    print("\n✓ VALID - Alternating roles, starts with user:")
    for msg in valid_conversation:
        print(f"   {msg['role']}: {msg['content']}")
    
    # Valid: Single user message
    valid_single = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    print("\n✓ VALID - Single user message:")
    for msg in valid_single:
        print(f"   {msg['role']}: {msg['content']}")
    
    # Invalid: Two user messages in a row
    invalid_double_user = [
        {"role": "user", "content": "Hello!"},
        {"role": "user", "content": "Are you there?"}  # ERROR!
    ]
    print("\n✗ INVALID - Two user messages in a row:")
    for msg in invalid_double_user:
        print(f"   {msg['role']}: {msg['content']}")
    print("   ^ This will cause an API error!")
    
    # Invalid: Starting with assistant
    invalid_start_assistant = [
        {"role": "assistant", "content": "Hello!"},  # ERROR!
        {"role": "user", "content": "Hi!"}
    ]
    print("\n✗ INVALID - Starting with assistant:")
    for msg in invalid_start_assistant:
        print(f"   {msg['role']}: {msg['content']}")
    print("   ^ This will cause an API error!")
    
    # How to fix: Combine multiple user inputs
    print("\n💡 FIX - Combine multiple user inputs into one message:")
    fixed_message = [
        {"role": "user", "content": "Hello!\n\nAre you there?"}
    ]
    for msg in fixed_message:
        print(f"   {msg['role']}: {msg['content']}")


def make_api_call_with_history():
    """Make a real API call to demonstrate conversation history."""
    
    print("\n" + "=" * 60)
    print("REAL API CALL WITH CONVERSATION HISTORY")
    print("=" * 60)
    
    client = anthropic.Anthropic()
    
    # Conversation about a topic where context matters
    conversation = [
        {"role": "user", "content": "The Eiffel Tower is a famous landmark."},
        {"role": "assistant", "content": "Yes, the Eiffel Tower is indeed one of the most recognizable structures in the world! Built in 1889, it stands in Paris and was designed by Gustave Eiffel's engineering company."},
        {"role": "user", "content": "How tall is it?"}
    ]
    
    print("\nSending conversation with context...")
    print("(Claude should understand 'it' refers to the Eiffel Tower)")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=conversation
    )
    
    print(f"\nConversation sent:")
    for msg in conversation:
        content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"   {msg['role'].upper()}: {content_preview}")
    
    print(f"\nClaude's response:")
    print(f"   {response.content[0].text}")
    
    # Show usage stats
    print(f"\nToken usage:")
    print(f"   Input tokens: {response.usage.input_tokens}")
    print(f"   Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    demonstrate_message_structure()
    demonstrate_valid_invalid_messages()
    make_api_call_with_history()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY: Messages must alternate user/assistant,")
    print("starting with user. Context is built by including history.")
    print("=" * 60)
