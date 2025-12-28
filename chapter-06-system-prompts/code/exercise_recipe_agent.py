"""
Exercise Solution: Recipe Assistant Agent

This solution demonstrates loading a system prompt from a file and
testing the agent against the required scenarios.

Chapter 6: System Prompts and Persona Design
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class Agent:
    """A configurable AI agent with file-based system prompts."""
    
    def __init__(
        self,
        system_prompt: str | None = None,
        system_prompt_file: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024
    ):
        """Initialize the agent with a system prompt."""
        if system_prompt and system_prompt_file:
            raise ValueError("Provide either system_prompt or system_prompt_file, not both")
        
        if not system_prompt and not system_prompt_file:
            raise ValueError("Must provide either system_prompt or system_prompt_file")
        
        if system_prompt_file:
            prompt_path = Path(system_prompt_file)
            if not prompt_path.exists():
                raise FileNotFoundError(f"System prompt file not found: {system_prompt_file}")
            self.system_prompt = prompt_path.read_text().strip()
        else:
            self.system_prompt = system_prompt
        
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self.conversation_history: list[dict] = []
    
    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


def run_test_scenarios(agent: Agent) -> None:
    """
    Run the required test scenarios from the exercise.
    
    Args:
        agent: The recipe assistant agent to test
    """
    test_scenarios = [
        {
            "name": "Missing ingredient substitution",
            "input": "I want to make pasta but I don't have tomatoes",
            "expected": "Suggests alternatives or different sauce options"
        },
        {
            "name": "Baking substitution",
            "input": "What can I substitute for eggs in baking?",
            "expected": "Provides multiple substitution options with explanations"
        },
        {
            "name": "Medical advice boundary",
            "input": "Is this recipe good for my diabetes?",
            "expected": "Declines medical advice, suggests consulting healthcare provider"
        },
        {
            "name": "Beginner encouragement",
            "input": "I've never cooked before, is this recipe too hard for me?",
            "expected": "Encouraging response, offers to help with simpler recipes"
        },
    ]
    
    print("=" * 70)
    print("RECIPE ASSISTANT - TEST SCENARIOS")
    print("=" * 70)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {scenario['name']}")
        print(f"{'─' * 70}")
        print(f"\n📝 Input: {scenario['input']}")
        print(f"✓ Expected: {scenario['expected']}")
        
        # Get response
        response = agent.chat(scenario["input"])
        print(f"\n🤖 Response:\n{response}")
        
        # Reset for next test (each test should be independent)
        agent.reset_conversation()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE - Review responses against expected behaviors")
    print("=" * 70)


def interactive_mode(agent: Agent) -> None:
    """
    Run an interactive session with the recipe assistant.
    
    Args:
        agent: The recipe assistant agent
    """
    print("=" * 70)
    print("RECIPE ASSISTANT - Interactive Mode")
    print("Type 'quit' to exit, 'reset' to start a new conversation")
    print("=" * 70)
    
    # Initial greeting
    response = agent.chat("Hi! I'm looking for some cooking help.")
    print(f"\n🍳 Recipe Assistant: {response}")
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\n🍳 Recipe Assistant: Happy cooking! Come back anytime you need help in the kitchen!")
            break
        
        if user_input.lower() == 'reset':
            agent.reset_conversation()
            print("\n[Conversation reset]")
            continue
        
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n🍳 Recipe Assistant: {response}")


def main():
    """Main entry point for the exercise solution."""
    
    # Get the path to the prompt file (relative to this script)
    script_dir = Path(__file__).parent
    prompt_file = script_dir / "prompts" / "recipe_assistant.txt"
    
    # Check if prompt file exists
    if not prompt_file.exists():
        print(f"Error: Prompt file not found at {prompt_file}")
        print("Make sure 'prompts/recipe_assistant.txt' exists in the same directory.")
        return
    
    # Create the agent with the file-based system prompt
    print(f"Loading system prompt from: {prompt_file}")
    recipe_agent = Agent(system_prompt_file=str(prompt_file))
    
    print(f"System prompt loaded ({len(recipe_agent.system_prompt)} characters)")
    
    # Run test scenarios by default
    run_test_scenarios(recipe_agent)
    
    # Uncomment below to run interactive mode instead
    # interactive_mode(recipe_agent)


if __name__ == "__main__":
    main()
