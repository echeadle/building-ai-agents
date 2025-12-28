"""
Token Management Utility for AI Agents

Appendix E: Troubleshooting Guide
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Any, Optional
from dataclasses import dataclass

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class TokenEstimator:
    """
    Estimate token usage for messages.
    
    Uses a simple heuristic: approximately 4 characters per token.
    This is not exact but good enough for management purposes.
    """
    
    chars_per_token: float = 4.0
    
    def estimate_message_tokens(self, message: dict[str, Any]) -> int:
        """Estimate tokens in a single message."""
        content = message.get("content", "")
        
        # Handle different content types
        if isinstance(content, str):
            return int(len(content) / self.chars_per_token)
        elif isinstance(content, list):
            total = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        total += int(len(block.get("text", "")) / self.chars_per_token)
                    elif block.get("type") == "tool_use":
                        # Tool use blocks are larger due to JSON structure
                        total += 50  # Rough estimate
                    elif block.get("type") == "tool_result":
                        content_str = str(block.get("content", ""))
                        total += int(len(content_str) / self.chars_per_token)
            return total
        else:
            return int(len(str(content)) / self.chars_per_token)
    
    def estimate_conversation_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens in conversation."""
        return sum(self.estimate_message_tokens(msg) for msg in messages)
    
    def estimate_text_tokens(self, text: str) -> int:
        """Estimate tokens in a text string."""
        return int(len(text) / self.chars_per_token)


class TokenAwareAgent:
    """
    Agent that manages conversation length to stay within token limits.
    
    Automatically trims old messages when approaching token limits.
    """
    
    def __init__(
        self,
        tools: Optional[list[dict]] = None,
        max_conversation_tokens: int = 150000,
        model_max_tokens: int = 200000
    ):
        self.tools = tools or []
        self.max_conversation_tokens = max_conversation_tokens
        self.model_max_tokens = model_max_tokens
        self.conversation_history: list[dict] = []
        self.token_estimator = TokenEstimator()
        self.system_prompt = ""
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt (doesn't count toward conversation tokens)."""
        self.system_prompt = prompt
    
    def add_message(self, role: str, content: Any):
        """
        Add a message to conversation with token management.
        
        Args:
            role: "user" or "assistant"
            content: Message content (string or list of content blocks)
        """
        message = {"role": role, "content": content}
        self.conversation_history.append(message)
        
        # Check if we need to trim
        current_tokens = self.token_estimator.estimate_conversation_tokens(
            self.conversation_history
        )
        
        if current_tokens > self.max_conversation_tokens:
            print(f"⚠️  Conversation at {current_tokens} tokens, trimming...")
            self._trim_conversation()
    
    def _trim_conversation(self):
        """
        Trim conversation to stay within token limits.
        
        Keeps the most recent messages and removes older ones.
        """
        if len(self.conversation_history) <= 2:
            print("   Cannot trim further - only 2 messages left")
            return
        
        # Calculate how many messages to keep
        # Start from the end and work backwards
        tokens_to_keep = 0
        messages_to_keep = []
        
        for message in reversed(self.conversation_history):
            msg_tokens = self.token_estimator.estimate_message_tokens(message)
            if tokens_to_keep + msg_tokens < self.max_conversation_tokens:
                messages_to_keep.insert(0, message)
                tokens_to_keep += msg_tokens
            else:
                break
        
        removed_count = len(self.conversation_history) - len(messages_to_keep)
        self.conversation_history = messages_to_keep
        
        new_token_count = self.token_estimator.estimate_conversation_tokens(
            self.conversation_history
        )
        
        print(f"   Removed {removed_count} messages")
        print(f"   New token count: ~{new_token_count}")
    
    def query(self, user_message: str) -> str:
        """
        Send a query to the agent.
        
        Args:
            user_message: The user's message
            
        Returns:
            The agent's response
        """
        # Add user message
        self.add_message("user", user_message)
        
        # Make API call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt if self.system_prompt else [],
            messages=self.conversation_history,
            tools=self.tools if self.tools else []
        )
        
        # Add assistant response
        self.add_message("assistant", response.content)
        
        # Extract and return text
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_blocks)
    
    def get_token_stats(self) -> dict[str, Any]:
        """Get statistics about token usage."""
        current_tokens = self.token_estimator.estimate_conversation_tokens(
            self.conversation_history
        )
        
        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_conversation_tokens,
            "model_limit": self.model_max_tokens,
            "usage_percent": (current_tokens / self.max_conversation_tokens) * 100,
            "remaining_tokens": self.max_conversation_tokens - current_tokens,
            "message_count": len(self.conversation_history),
        }


class SummarizingAgent:
    """
    Agent that summarizes old conversations to save tokens.
    
    Instead of trimming, this agent summarizes old messages into
    a compact context summary.
    """
    
    def __init__(
        self,
        tools: Optional[list[dict]] = None,
        max_conversation_tokens: int = 150000
    ):
        self.tools = tools or []
        self.max_conversation_tokens = max_conversation_tokens
        self.conversation_history: list[dict] = []
        self.context_summary = ""
        self.token_estimator = TokenEstimator()
    
    def add_message(self, role: str, content: Any):
        """Add message with automatic summarization."""
        message = {"role": role, "content": content}
        self.conversation_history.append(message)
        
        # Check if we need to summarize
        current_tokens = self.token_estimator.estimate_conversation_tokens(
            self.conversation_history
        )
        
        if current_tokens > self.max_conversation_tokens:
            print(f"⚠️  Token limit reached, summarizing old messages...")
            self._summarize_and_trim()
    
    def _summarize_and_trim(self):
        """Summarize old messages and keep recent ones."""
        # Keep last 4 messages (2 exchanges)
        recent_messages = self.conversation_history[-4:]
        old_messages = self.conversation_history[:-4]
        
        if not old_messages:
            return
        
        # Create a summary of old messages
        summary_prompt = self._create_summary_prompt(old_messages)
        
        # Get summary from Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,  # Keep summary concise
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary = response.content[0].text
        
        # Update context
        if self.context_summary:
            self.context_summary += "\n\n" + summary
        else:
            self.context_summary = summary
        
        # Replace conversation with recent messages
        self.conversation_history = recent_messages
        
        print(f"   Summarized {len(old_messages)} old messages")
        print(f"   Keeping {len(recent_messages)} recent messages")
    
    def _create_summary_prompt(self, messages: list[dict]) -> str:
        """Create a prompt to summarize messages."""
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])
        
        return f"""Summarize this conversation in 2-3 concise sentences, 
focusing on the key information exchanged:

{conversation_text}

Provide only the summary, no preamble."""
    
    def query(self, user_message: str) -> str:
        """Query with context summary if available."""
        # Prepend context summary to user message if we have one
        if self.context_summary:
            enhanced_message = f"""[Previous conversation context: {self.context_summary}]

{user_message}"""
        else:
            enhanced_message = user_message
        
        self.add_message("user", enhanced_message)
        
        # Make API call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=self.conversation_history,
            tools=self.tools if self.tools else []
        )
        
        self.add_message("assistant", response.content)
        
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_blocks)


# Example usage
if __name__ == "__main__":
    print("Token Management Examples\n")
    print("=" * 50)
    print()
    
    # Example 1: Token-aware agent with trimming
    print("Example 1: Token-aware agent (with trimming)")
    print("-" * 50)
    
    agent = TokenAwareAgent(max_conversation_tokens=1000)  # Very low for demo
    
    # Simulate a long conversation
    for i in range(10):
        response = agent.query(f"Tell me a fact about number {i}")
        print(f"Turn {i + 1}: {response[:50]}...")
        
        # Show stats every few turns
        if i % 3 == 2:
            stats = agent.get_token_stats()
            print(f"   Stats: {stats['current_tokens']} tokens, "
                  f"{stats['message_count']} messages")
            print()
    
    print("\nFinal stats:")
    final_stats = agent.get_token_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Example 2: Summarizing agent
    print("\n\nExample 2: Summarizing agent")
    print("-" * 50)
    
    summarizing_agent = SummarizingAgent(max_conversation_tokens=1000)
    
    # Have a conversation
    questions = [
        "What is Python?",
        "How do I install it?",
        "What are virtual environments?",
        "How do I create one?",
        "What about dependencies?",
    ]
    
    for q in questions:
        response = summarizing_agent.query(q)
        print(f"Q: {q}")
        print(f"A: {response[:100]}...")
        print()
    
    if summarizing_agent.context_summary:
        print(f"\nContext summary created:")
        print(summarizing_agent.context_summary)
