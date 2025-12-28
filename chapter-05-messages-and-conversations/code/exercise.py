"""
Exercise Solution: Persistent Chat with Save/Load Functionality

Chapter 5: Understanding Messages and Conversations

This solution implements a PersistentChat class that can:
- Save conversation history to a JSON file
- Load conversation history from a JSON file
- Resume previous conversations seamlessly
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class PersistentChat:
    """
    A chat session that can save and load conversation history.
    
    This class extends the basic ChatSession concept with persistence,
    allowing users to:
    - Save ongoing conversations to disk
    - Resume previous conversations later
    - Keep a record of important discussions
    
    Attributes:
        model: The Claude model being used
        max_tokens: Maximum tokens per response
        conversation_history: List of message dictionaries
        metadata: Dictionary containing conversation metadata
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024
    ):
        """
        Initialize a persistent chat session.
        
        Args:
            model: The Claude model to use
            max_tokens: Maximum tokens in each response
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.conversation_history: list[dict] = []
        self.metadata: dict = {
            "created_at": datetime.now().isoformat(),
            "model": model,
            "message_count": 0
        }
    
    def send_message(self, user_message: str) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            user_message: The user's input
            
        Returns:
            Claude's response text
        """
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare messages for API (strip timestamps)
        api_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in self.conversation_history
        ]
        
        # Make API call
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=api_messages
        )
        
        # Extract response
        assistant_message = response.content[0].text
        
        # Add to history with timestamp
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update metadata
        self.metadata["message_count"] = len(self.conversation_history)
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        return assistant_message
    
    def save(self, filename: str) -> str:
        """
        Save the conversation to a JSON file.
        
        Args:
            filename: Name of the file (without path)
            
        Returns:
            Full path to the saved file
        """
        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"
        
        # Create saves directory if it doesn't exist
        saves_dir = Path("conversation_saves")
        saves_dir.mkdir(exist_ok=True)
        
        filepath = saves_dir / filename
        
        # Prepare save data
        save_data = {
            "metadata": self.metadata,
            "conversation": self.conversation_history
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load(self, filename: str) -> dict:
        """
        Load a conversation from a JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary with loaded metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        # Handle paths with or without directory
        if not filename.endswith(".json"):
            filename += ".json"
        
        filepath = Path("conversation_saves") / filename
        
        if not filepath.exists():
            # Try as absolute path
            filepath = Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filename}")
        
        # Read file
        with open(filepath, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        
        # Validate structure
        if "conversation" not in save_data:
            raise ValueError("Invalid conversation file: missing 'conversation' key")
        
        # Load data
        self.conversation_history = save_data["conversation"]
        self.metadata = save_data.get("metadata", {})
        self.metadata["loaded_from"] = str(filepath)
        self.metadata["loaded_at"] = datetime.now().isoformat()
        
        return self.metadata
    
    def get_summary(self) -> str:
        """
        Get a summary of the conversation so far.
        
        Uses Claude to summarize the conversation for display
        when loading a previous chat.
        
        Returns:
            Summary string
        """
        if not self.conversation_history:
            return "No conversation history."
        
        # Build summary prompt
        summary_prompt = "Provide a 2-3 sentence summary of what was discussed in this conversation:\n\n"
        for msg in self.conversation_history[:20]:  # Limit for token efficiency
            role = msg["role"].upper()
            content = msg["content"][:200]  # Truncate long messages
            summary_prompt += f"{role}: {content}\n"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        return response.content[0].text
    
    def message_count(self) -> int:
        """Return the number of messages in history."""
        return len(self.conversation_history)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "message_count": 0
        }
    
    def list_saves(self) -> list[str]:
        """
        List all saved conversation files.
        
        Returns:
            List of filenames in the saves directory
        """
        saves_dir = Path("conversation_saves")
        if not saves_dir.exists():
            return []
        
        return [f.name for f in saves_dir.glob("*.json")]


def main():
    """Run the persistent chat application."""
    
    print("=" * 60)
    print("PERSISTENT CHAT WITH CLAUDE")
    print("=" * 60)
    print("\nCommands:")
    print("  /save <name>  - Save conversation to file")
    print("  /load <name>  - Load conversation from file")
    print("  /list         - List saved conversations")
    print("  /summary      - Get summary of current conversation")
    print("  /clear        - Clear current conversation")
    print("  /quit         - Exit the program")
    print("-" * 60)
    
    chat = PersistentChat()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if command == "/quit":
                print("Goodbye!")
                break
            
            elif command == "/save":
                if not arg:
                    print("Usage: /save <filename>")
                    continue
                try:
                    path = chat.save(arg)
                    print(f"✓ Conversation saved to: {path}")
                except Exception as e:
                    print(f"✗ Error saving: {e}")
            
            elif command == "/load":
                if not arg:
                    print("Usage: /load <filename>")
                    continue
                try:
                    metadata = chat.load(arg)
                    print(f"✓ Loaded conversation ({chat.message_count()} messages)")
                    print(f"  Created: {metadata.get('created_at', 'Unknown')}")
                    print("\nGenerating summary...")
                    summary = chat.get_summary()
                    print(f"\nSummary: {summary}")
                except FileNotFoundError as e:
                    print(f"✗ {e}")
                except ValueError as e:
                    print(f"✗ Invalid file: {e}")
            
            elif command == "/list":
                saves = chat.list_saves()
                if saves:
                    print("Saved conversations:")
                    for name in saves:
                        print(f"  - {name}")
                else:
                    print("No saved conversations found.")
            
            elif command == "/summary":
                if chat.message_count() == 0:
                    print("No conversation to summarize.")
                else:
                    print("Generating summary...")
                    summary = chat.get_summary()
                    print(f"\nSummary: {summary}")
            
            elif command == "/clear":
                chat.clear()
                print("Conversation cleared.")
            
            else:
                print(f"Unknown command: {command}")
            
            continue
        
        # Regular message
        try:
            response = chat.send_message(user_input)
            print(f"\nClaude: {response}")
        except anthropic.APIError as e:
            print(f"\nAPI Error: {e}")


if __name__ == "__main__":
    main()
