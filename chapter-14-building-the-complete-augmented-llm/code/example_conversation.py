"""
Multi-turn conversation with AugmentedLLM.

This example demonstrates how the AugmentedLLM maintains conversation
history automatically, enabling natural multi-turn dialogues where
context is preserved across messages.

Chapter 14: Building the Complete Augmented LLM
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def programming_tutor_example():
    """A multi-turn conversation learning Python concepts."""
    
    print("\n--- Programming Tutor Example ---")
    print("(Demonstrating context preservation across turns)")
    
    config = AugmentedLLMConfig(
        system_prompt="""You are a friendly Python tutor helping someone learn programming.

Guidelines:
- Remember context from earlier in the conversation
- Build on previous explanations
- Use examples that connect to what was discussed before
- Keep explanations clear but concise
- Encourage questions and exploration"""
    )
    
    llm = AugmentedLLM(config=config)
    
    # A natural learning progression
    conversation = [
        "I'm trying to learn about Python data structures. What are the main ones I should know?",
        "Can you tell me more about dictionaries? They seem useful.",
        "How would I use one to count word frequencies in a text?",
        "That's helpful! What if I wanted to find the 3 most common words?",
    ]
    
    for message in conversation:
        print(f"\n{'='*50}")
        print(f"You: {message}")
        print("-" * 50)
        response = llm.run(message)
        print(f"Tutor: {response}")
    
    # Show the history
    history = llm.get_history()
    print(f"\n--- Conversation History: {len(history)} messages ---")


def debugging_session_example():
    """A multi-turn debugging conversation."""
    
    print("\n--- Debugging Session Example ---")
    print("(Demonstrating context in a technical discussion)")
    
    config = AugmentedLLMConfig(
        system_prompt="""You are an expert debugger helping solve code problems.

Guidelines:
- Ask clarifying questions when needed
- Remember code snippets shared earlier
- Build your diagnosis incrementally
- Explain your reasoning
- Suggest fixes with explanations"""
    )
    
    llm = AugmentedLLM(config=config)
    
    conversation = [
        """I have a bug in my code. Here's the function:
```python
def process_users(users):
    for user in users:
        print(user['name'])
        user['processed'] = True
```
It crashes sometimes but not always.""",
        "It says: KeyError: 'name'",
        "The users come from an API. Sometimes the response looks different.",
        "That makes sense! How would I handle that gracefully?",
    ]
    
    for message in conversation:
        print(f"\n{'='*50}")
        print(f"You: {message}")
        print("-" * 50)
        response = llm.run(message)
        print(f"Debugger: {response}")


def story_collaboration_example():
    """A creative writing collaboration across turns."""
    
    print("\n--- Story Collaboration Example ---")
    print("(Demonstrating creative context preservation)")
    
    config = AugmentedLLMConfig(
        system_prompt="""You are a creative writing collaborator.

Guidelines:
- Remember all story elements (characters, settings, plot points)
- Build on the narrative established so far
- Offer creative suggestions that fit the story
- Keep responses focused and not too long
- Match the tone set by the human"""
    )
    
    llm = AugmentedLLM(config=config)
    
    conversation = [
        "Let's write a short story together. I'll start: 'Maya discovered the old compass in her grandmother's attic.'",
        "I like that! The compass should be magical somehow.",
        "Perfect. Maya takes it outside. What happens when she first uses it?",
        "Let's wrap up the beginning. Give me a closing line for this first scene that hints at adventure.",
    ]
    
    for message in conversation:
        print(f"\n{'='*50}")
        print(f"You: {message}")
        print("-" * 50)
        response = llm.run(message)
        print(f"Collaborator: {response}")


def history_inspection_example():
    """Demonstrate how to inspect and use conversation history."""
    
    print("\n--- History Inspection Example ---")
    
    config = AugmentedLLMConfig(
        system_prompt="You are a helpful assistant. Be concise."
    )
    
    llm = AugmentedLLM(config=config)
    
    # Have a short conversation
    llm.run("My name is Alex.")
    llm.run("I'm learning Python.")
    llm.run("What should I focus on as a beginner?")
    
    # Inspect history
    history = llm.get_history()
    
    print(f"Total messages in history: {len(history)}")
    print("\nHistory structure:")
    for i, msg in enumerate(history):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Content might be a string or a list of content blocks
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            # It's a list of content blocks
            preview = "[complex content]"
        
        print(f"  [{i}] {role}: {preview}")
    
    # Demonstrate clearing history
    print("\nClearing history...")
    llm.clear_history()
    print(f"Messages after clear: {len(llm.get_history())}")
    
    # Now context is lost
    response = llm.run("What's my name?")
    print(f"\nAfter clearing, asking 'What's my name?':")
    print(f"Response: {response}")


def main():
    """Run all multi-turn conversation examples."""
    
    print("Multi-Turn Conversations with AugmentedLLM")
    print("=" * 50)
    
    programming_tutor_example()
    
    print("\n" + "=" * 60 + "\n")
    
    debugging_session_example()
    
    print("\n" + "=" * 60 + "\n")
    
    story_collaboration_example()
    
    print("\n" + "=" * 60 + "\n")
    
    history_inspection_example()
    
    print("\n" + "=" * 50)
    print("All examples complete!")


if __name__ == "__main__":
    main()
