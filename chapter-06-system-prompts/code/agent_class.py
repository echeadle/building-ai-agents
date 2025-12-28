"""
A configurable agent base that loads system prompts from files.

This pattern makes it easy to iterate on prompts without changing code.

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
    """
    A configurable AI agent with file-based system prompts.
    
    This class provides a clean interface for creating agents with different
    personas and behaviors, controlled entirely through system prompts that
    can be loaded from external files.
    
    Attributes:
        system_prompt: The system prompt defining the agent's behavior
        model: The Claude model being used
        max_tokens: Maximum tokens in responses
        conversation_history: List of messages in the current conversation
    """
    
    def __init__(
        self,
        system_prompt: str | None = None,
        system_prompt_file: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024
    ):
        """
        Initialize the agent.
        
        You must provide either a direct system_prompt string OR a path to
        a file containing the system prompt, but not both.
        
        Args:
            system_prompt: Direct system prompt string
            system_prompt_file: Path to file containing system prompt
            model: Claude model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response (default: 1024)
            
        Raises:
            ValueError: If neither or both prompt options provided
            FileNotFoundError: If prompt file doesn't exist
            
        Example:
            # With direct prompt
            agent = Agent(system_prompt="You are a helpful assistant.")
            
            # With file
            agent = Agent(system_prompt_file="prompts/assistant.txt")
        """
        # Validate that exactly one prompt source is provided
        if system_prompt and system_prompt_file:
            raise ValueError(
                "Provide either system_prompt or system_prompt_file, not both"
            )
        
        if not system_prompt and not system_prompt_file:
            raise ValueError(
                "Must provide either system_prompt or system_prompt_file"
            )
        
        # Load system prompt from file if path provided
        if system_prompt_file:
            prompt_path = Path(system_prompt_file)
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"System prompt file not found: {system_prompt_file}"
                )
            self.system_prompt = prompt_path.read_text().strip()
            self._prompt_source = f"file:{system_prompt_file}"
        else:
            self.system_prompt = system_prompt
            self._prompt_source = "direct"
        
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self.conversation_history: list[dict] = []
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        The message is added to conversation history, allowing for
        multi-turn conversations where context is maintained.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The agent's response text
            
        Example:
            response = agent.chat("Hello!")
            print(response)
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Make API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.conversation_history
        )
        
        # Extract response text
        assistant_message = response.content[0].text
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset_conversation(self) -> None:
        """
        Clear conversation history to start fresh.
        
        Call this when you want to start a new conversation without
        creating a new Agent instance.
        """
        self.conversation_history = []
    
    def get_system_prompt(self) -> str:
        """
        Return the current system prompt for inspection.
        
        Returns:
            The system prompt string
        """
        return self.system_prompt
    
    def get_conversation_length(self) -> int:
        """
        Return the number of messages in the conversation.
        
        Returns:
            Number of messages (user + assistant combined)
        """
        return len(self.conversation_history)
    
    def get_prompt_source(self) -> str:
        """
        Return where the system prompt was loaded from.
        
        Returns:
            Either "direct" or "file:<path>"
        """
        return self._prompt_source
    
    def __repr__(self) -> str:
        """Return a string representation of the agent."""
        return (
            f"Agent(model={self.model!r}, "
            f"prompt_source={self._prompt_source!r}, "
            f"messages={len(self.conversation_history)})"
        )


def main():
    """Demonstrate the Agent class with different configurations."""
    
    # Example 1: Agent with direct system prompt
    print("=" * 60)
    print("EXAMPLE 1: Direct System Prompt")
    print("=" * 60)
    
    pirate_agent = Agent(
        system_prompt="""You are a helpful assistant that speaks like a pirate.
Use nautical terms and pirate expressions in your responses.
Keep responses brief and fun. End messages with a pirate phrase."""
    )
    
    print(f"\nAgent info: {pirate_agent}")
    response = pirate_agent.chat("What's the weather like today?")
    print(f"\nUser: What's the weather like today?")
    print(f"Agent: {response}")
    
    # Example 2: Multi-turn conversation
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-turn Conversation")
    print("=" * 60)
    
    assistant = Agent(
        system_prompt="""You are a concise assistant. Keep all responses 
under 50 words. Be helpful but brief."""
    )
    
    exchanges = [
        "What is Python?",
        "What can I build with it?",
        "How do I install it?"
    ]
    
    for msg in exchanges:
        response = assistant.chat(msg)
        print(f"\nUser: {msg}")
        print(f"Agent: {response}")
    
    print(f"\nConversation length: {assistant.get_conversation_length()} messages")
    
    # Example 3: Resetting conversation
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Conversation Reset")
    print("=" * 60)
    
    assistant.reset_conversation()
    print(f"After reset: {assistant.get_conversation_length()} messages")
    
    # New conversation - no context from before
    response = assistant.chat("What were we just talking about?")
    print(f"\nUser: What were we just talking about?")
    print(f"Agent: {response}")


if __name__ == "__main__":
    main()
