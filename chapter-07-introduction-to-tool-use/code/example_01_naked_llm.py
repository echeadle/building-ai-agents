"""
Demonstrating the limitations of an LLM without tools.

This script shows what happens when you ask Claude for information
it cannot access without tools—real-time data, current prices,
live information, and actions in the real world.

Chapter 7: Introduction to Tool Use
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

# Initialize the Anthropic client
client = anthropic.Anthropic()


def ask_claude(question: str) -> str:
    """
    Send a question to Claude and return the response.
    
    Args:
        question: The question to ask Claude
        
    Returns:
        Claude's response text
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response.content[0].text


def demonstrate_limitations() -> None:
    """
    Demonstrate various scenarios where Claude cannot help without tools.
    """
    # Questions that require real-time information
    real_time_questions = [
        "What is the current price of Apple stock (AAPL)?",
        "What's the weather like in Tokyo right now?",
        "What are today's top news headlines?",
    ]
    
    # Questions that require taking actions
    action_questions = [
        "Can you send an email to my boss saying I'll be late?",
        "Please add a meeting to my calendar for tomorrow at 2pm.",
        "Can you order me a pizza from the nearest restaurant?",
    ]
    
    # Questions that require accessing external systems
    external_questions = [
        "What's in my Google Drive folder called 'Projects'?",
        "Can you check if the Python package 'requests' has any security vulnerabilities?",
        "What's the current balance in my bank account?",
    ]
    
    print("=" * 70)
    print("DEMONSTRATING LLM LIMITATIONS WITHOUT TOOLS")
    print("=" * 70)
    
    # Test real-time information requests
    print("\n📊 REAL-TIME INFORMATION REQUESTS")
    print("-" * 50)
    for question in real_time_questions:
        print(f"\n❓ Question: {question}")
        response = ask_claude(question)
        print(f"\n💬 Response:\n{response}")
        print("-" * 50)
    
    # Test action requests
    print("\n\n⚡ ACTION REQUESTS")
    print("-" * 50)
    for question in action_questions:
        print(f"\n❓ Question: {question}")
        response = ask_claude(question)
        print(f"\n💬 Response:\n{response}")
        print("-" * 50)
    
    # Test external system requests
    print("\n\n🔌 EXTERNAL SYSTEM REQUESTS")
    print("-" * 50)
    for question in external_questions:
        print(f"\n❓ Question: {question}")
        response = ask_claude(question)
        print(f"\n💬 Response:\n{response}")
        print("-" * 50)


def contrast_with_knowledge() -> None:
    """
    Show what Claude CAN answer from its training knowledge.
    This contrasts with the limitations above.
    """
    knowledge_questions = [
        "What causes thunderstorms?",
        "Explain the concept of compound interest.",
        "What are the basic principles of object-oriented programming?",
    ]
    
    print("\n\n" + "=" * 70)
    print("CONTRAST: WHAT CLAUDE CAN ANSWER FROM TRAINING KNOWLEDGE")
    print("=" * 70)
    
    print("\n📚 KNOWLEDGE-BASED QUESTIONS")
    print("-" * 50)
    for question in knowledge_questions:
        print(f"\n❓ Question: {question}")
        response = ask_claude(question)
        print(f"\n💬 Response:\n{response}")
        print("-" * 50)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Chapter 7: The Limitations of a 'Naked' LLM")
    print("=" * 70)
    print("\nThis script demonstrates what Claude CANNOT do without tools.")
    print("Notice how Claude acknowledges its limitations honestly.")
    
    # Run the demonstrations
    demonstrate_limitations()
    contrast_with_knowledge()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Claude excels at answering questions from its training knowledge,
but cannot:
  • Access real-time information (stocks, weather, news)
  • Take actions in the world (send emails, make purchases)
  • Access external systems (files, databases, APIs)

This is why we need TOOLS - they give Claude the ability to
interact with the world beyond its training data.

In the next chapters, you'll learn to give Claude these capabilities!
""")
