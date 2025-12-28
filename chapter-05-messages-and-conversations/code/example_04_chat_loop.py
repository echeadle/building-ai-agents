"""
A complete chat loop with conversation history and basic truncation.

Chapter 5: Understanding Messages and Conversations

This is the main code deliverable for Chapter 5 - a reusable ChatSession
class that properly maintains conversation history.
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


class ChatSession:
    """
    Manages a conversation session with Claude.
    
    This class handles:
    - Maintaining conversation history
    - Basic truncation to stay within context limits
    - Clean interface for sending/receiving messages
    
    This pattern will be used throughout the book as a foundation
    for more sophisticated agent implementations.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_history_pairs: int = 20
    ):
        """
        Initialize a chat session.
        
        Args:
            model: The Claude model to use
            max_tokens: Maximum tokens in each response
            max_history_pairs: Maximum conversation pairs to retain
                              (one pair = one user message + one assistant response)
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_history_pairs = max_history_pairs
        self.conversation_history: list[dict] = []
    
    def _truncate_history(self) -> list[dict]:
        """
        Truncate history if it exceeds the maximum pairs.
        
        Uses a "keep first + keep recent" strategy:
        - Preserves the first pair (often contains important initial context)
        - Keeps the most recent exchanges
        
        Returns:
            Truncated message list suitable for API call
        """
        max_messages = self.max_history_pairs * 2
        
        if len(self.conversation_history) <= max_messages:
            return self.conversation_history
        
        # Keep first pair (may contain important context like user's name, goals)
        first_pair = self.conversation_history[:2]
        
        # Keep the most recent messages
        recent = self.conversation_history[-(max_messages - 2):]
        
        return first_pair + recent
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message to Claude and get a response.
        
        This method:
        1. Adds the user message to history
        2. Sends (possibly truncated) history to Claude
        3. Adds Claude's response to history
        4. Returns the response text
        
        Args:
            user_message: The user's input
            
        Returns:
            Claude's response text
            
        Raises:
            anthropic.APIError: If the API call fails
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get (possibly truncated) history for API call
        messages_to_send = self._truncate_history()
        
        # Make API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages_to_send
        )
        
        # Extract response text
        assistant_message = response.content[0].text
        
        # Add assistant response to full history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def get_history(self) -> list[dict]:
        """
        Return a copy of the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def message_count(self) -> int:
        """
        Return the number of messages in history.
        
        Returns:
            Total message count
        """
        return len(self.conversation_history)
    
    def get_last_exchange(self) -> tuple[str, str] | None:
        """
        Get the most recent user/assistant exchange.
        
        Returns:
            Tuple of (user_message, assistant_message) or None if no exchanges
        """
        if len(self.conversation_history) < 2:
            return None
        
        user_msg = self.conversation_history[-2]["content"]
        assistant_msg = self.conversation_history[-1]["content"]
        return (user_msg, assistant_msg)


def main():
    """Run an interactive chat session."""
    
    print("=" * 60)
    print("INTERACTIVE CHAT WITH CLAUDE")
    print("=" * 60)
    print("\nCommands:")
    print("  'quit' or 'exit' - End the session")
    print("  'history'        - Show conversation history")
    print("  'clear'          - Clear conversation history")
    print("  'count'          - Show message count")
    print("-" * 60)
    
    session = ChatSession()
    
    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        # Handle empty input
        if not user_input:
            continue
        
        # Handle special commands
        command = user_input.lower()
        
        if command in ('quit', 'exit'):
            print("Goodbye!")
            break
        
        if command == 'history':
            history = session.get_history()
            print(f"\n--- Conversation History ({len(history)} messages) ---")
            if not history:
                print("  (empty)")
            for i, msg in enumerate(history, 1):
                role = msg['role'].upper()
                content = msg['content']
                # Truncate long messages for display
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  {i}. [{role}]: {content}")
            print("-" * 40)
            continue
        
        if command == 'clear':
            session.clear_history()
            print("Conversation history cleared.")
            continue
        
        if command == 'count':
            print(f"Messages in history: {session.message_count()}")
            continue
        
        # Send message and get response
        try:
            response = session.send_message(user_input)
            print(f"\nClaude: {response}")
        except anthropic.APIConnectionError:
            print("\nError: Failed to connect to Anthropic API")
        except anthropic.RateLimitError:
            print("\nError: Rate limited. Please wait and try again.")
        except anthropic.APIStatusError as e:
            print(f"\nAPI Error: {e.status_code} - {e.message}")


if __name__ == "__main__":
    main()
