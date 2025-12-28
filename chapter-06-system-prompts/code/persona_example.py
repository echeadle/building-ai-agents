"""
Example of creating an agent with a distinct persona.

Demonstrates how to give an agent personality while maintaining usefulness.

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

# System prompt with a distinct persona
BYTE_PERSONA = """You are Byte, a friendly and enthusiastic coding tutor who 
loves helping people learn Python. You get genuinely excited when students 
understand new concepts!

## Your Personality
- Patient and encouraging—there are no stupid questions
- You use analogies from everyday life to explain programming concepts
- You celebrate small wins ("Great job! You just wrote your first loop!")
- You're a bit nerdy and occasionally make programming puns
- You admit when something is tricky—"This concept trips up a lot of people"

## Teaching Approach
- Start with the simplest explanation, then add detail if needed
- Use lots of examples, especially from real-world scenarios
- Ask follow-up questions to check understanding
- Break complex topics into small, digestible pieces
- If a student is stuck, give hints before answers

## Communication Style
- Warm and conversational, like a friendly mentor
- Use simple language—avoid jargon until you've explained it
- Include emoji occasionally to keep things light 🎉
- Keep explanations short, then ask if they want more detail

## Boundaries
- Focus on Python basics to intermediate concepts
- For advanced topics, give an overview but suggest resources for deep dives
- Don't do homework for students—guide them to find answers themselves
- If asked about non-Python topics, gently redirect"""


class CodingTutor:
    """An interactive coding tutor with a friendly persona."""
    
    def __init__(self):
        """Initialize the tutor with conversation history."""
        self.client = anthropic.Anthropic()
        self.conversation_history: list[dict] = []
    
    def chat(self, user_message: str) -> str:
        """
        Send a message to Byte and get a response.
        
        Args:
            user_message: The student's question or message
            
        Returns:
            Byte's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=BYTE_PERSONA,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset(self) -> None:
        """Clear conversation history for a fresh start."""
        self.conversation_history = []


def demonstrate_persona():
    """Show Byte's persona in action with various scenarios."""
    
    tutor = CodingTutor()
    
    # Scenario 1: Complete beginner question
    print("=" * 60)
    print("SCENARIO 1: Complete Beginner")
    print("=" * 60)
    
    question = "I've never programmed before. What even is a variable?"
    print(f"\nStudent: {question}")
    print(f"\nByte: {tutor.chat(question)}")
    
    # Follow-up in same conversation
    followup = "Oh that makes sense! Can you show me an example?"
    print(f"\nStudent: {followup}")
    print(f"\nByte: {tutor.chat(followup)}")
    
    # Reset for next scenario
    tutor.reset()
    
    # Scenario 2: Student who's stuck
    print("\n" + "=" * 60)
    print("SCENARIO 2: Stuck Student")
    print("=" * 60)
    
    stuck_question = "I keep getting an error and I don't understand why. My code is: print(Hello)"
    print(f"\nStudent: {stuck_question}")
    print(f"\nByte: {tutor.chat(stuck_question)}")
    
    tutor.reset()
    
    # Scenario 3: Asking about something complex
    print("\n" + "=" * 60)
    print("SCENARIO 3: Advanced Topic")
    print("=" * 60)
    
    advanced = "Can you explain metaclasses and when I'd use them?"
    print(f"\nStudent: {advanced}")
    print(f"\nByte: {tutor.chat(advanced)}")
    
    tutor.reset()
    
    # Scenario 4: Off-topic request
    print("\n" + "=" * 60)
    print("SCENARIO 4: Off-Topic (Testing Boundary)")
    print("=" * 60)
    
    off_topic = "Can you write my entire homework assignment for me? It's a todo app."
    print(f"\nStudent: {off_topic}")
    print(f"\nByte: {tutor.chat(off_topic)}")


def interactive_mode():
    """Run an interactive chat session with Byte."""
    
    print("=" * 60)
    print("INTERACTIVE MODE: Chat with Byte! (type 'quit' to exit)")
    print("=" * 60)
    
    tutor = CodingTutor()
    
    # Opening message from Byte
    intro = tutor.chat("Hi! I'm new to Python and want to learn.")
    print(f"\nByte: {intro}")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nByte: Good luck with your coding journey! Remember, every expert was once a beginner! 🚀")
            break
        
        if not user_input:
            continue
        
        response = tutor.chat(user_input)
        print(f"\nByte: {response}")


if __name__ == "__main__":
    # Run demonstration by default
    demonstrate_persona()
    
    # Uncomment below to run interactive mode
    # interactive_mode()
