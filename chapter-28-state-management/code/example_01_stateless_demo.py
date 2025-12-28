"""
Demonstrating the stateless nature of API calls.

This example shows that each API call is independent - Claude has no
memory of previous calls unless we explicitly provide that context.

Chapter 28: State Management
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


def demonstrate_stateless_calls():
    """Show that API calls have no memory between them."""
    client = anthropic.Anthropic()
    
    print("Demonstrating Stateless API Calls")
    print("=" * 50)
    
    # First call - introduce a topic
    print("\n--- First API Call ---")
    print("User: My favorite color is blue. Remember that.")
    
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "My favorite color is blue. Remember that."}
        ]
    )
    print(f"Claude: {response1.content[0].text}")
    
    # Second call - completely separate, no memory!
    print("\n--- Second API Call (separate, no history) ---")
    print("User: What's my favorite color?")
    
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "What's my favorite color?"}
        ]
    )
    print(f"Claude: {response2.content[0].text}")
    
    print("\n" + "=" * 50)
    print("Notice: Claude doesn't know the answer in the second call")
    print("because each API call is completely independent!")


if __name__ == "__main__":
    demonstrate_stateless_calls()
