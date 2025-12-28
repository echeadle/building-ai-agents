"""
Basic usage of AugmentedLLM without tools.

This example demonstrates the simplest use case: an LLM with a custom
system prompt but no tools. Perfect for Q&A, writing assistance, or
any task that doesn't require external actions.

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


def main():
    """Demonstrate basic AugmentedLLM usage."""
    
    print("Basic AugmentedLLM Usage")
    print("=" * 50)
    
    # Create with a custom system prompt
    config = AugmentedLLMConfig(
        system_prompt="""You are a helpful coding assistant. 
        
Your responses should be:
- Concise and practical
- Include code examples when helpful
- Explain concepts clearly for intermediate programmers"""
    )
    
    llm = AugmentedLLM(config=config)
    
    # Example 1: Simple question
    print("\n--- Example 1: Simple Question ---")
    response = llm.run("What's the difference between a list and a tuple in Python?")
    print(f"Response:\n{response}")
    
    # Clear history for next unrelated question
    llm.clear_history()
    
    # Example 2: Request for code
    print("\n--- Example 2: Code Request ---")
    response = llm.run("Show me how to read a JSON file in Python.")
    print(f"Response:\n{response}")
    
    # Example 3: Using default config
    print("\n--- Example 3: Default Configuration ---")
    default_llm = AugmentedLLM()  # Uses all defaults
    response = default_llm.run("Hello! What can you help me with?")
    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
