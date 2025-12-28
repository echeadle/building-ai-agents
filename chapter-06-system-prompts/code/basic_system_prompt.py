"""
Basic demonstration of how system prompts work.

Shows the difference between conversations with and without system prompts.

Chapter 6: System Prompts and Persona Design
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


def chat_without_system_prompt(user_message: str) -> str:
    """Send a message without any system prompt."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.content[0].text


def chat_with_system_prompt(user_message: str, system_prompt: str) -> str:
    """Send a message with a system prompt."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=system_prompt,  # <-- The system prompt goes here
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.content[0].text


def main():
    """Demonstrate the difference system prompts make."""
    
    user_message = "Tell me about yourself."
    
    # Without system prompt - generic response
    print("=" * 60)
    print("WITHOUT SYSTEM PROMPT")
    print("=" * 60)
    print(f"User: {user_message}")
    print(f"Assistant: {chat_without_system_prompt(user_message)}")
    
    print("\n")
    
    # With system prompt - focused response
    system_prompt = """You are Max, a grumpy but lovable robot mechanic from the 
year 2150. You've been fixing robots for 75 years and you've seen it all. 
You tend to grumble about "kids these days" but you're actually quite helpful.
Keep responses short and in character."""
    
    print("=" * 60)
    print("WITH SYSTEM PROMPT (Max the robot mechanic)")
    print("=" * 60)
    print(f"User: {user_message}")
    print(f"Max: {chat_with_system_prompt(user_message, system_prompt)}")
    
    print("\n")
    
    # Another example showing behavioral guidelines
    professional_prompt = """You are a professional customer service representative 
for a software company. Guidelines:
- Always be polite and professional
- Keep responses concise (2-3 sentences max)
- End each response with a helpful question"""
    
    print("=" * 60)
    print("WITH PROFESSIONAL SYSTEM PROMPT")
    print("=" * 60)
    print(f"User: {user_message}")
    print(f"Assistant: {chat_with_system_prompt(user_message, professional_prompt)}")


if __name__ == "__main__":
    main()
