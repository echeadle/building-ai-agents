"""
Conversation history optimization.

Chapter 38: Cost Optimization
"""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class ConversationManager:
    """Manages conversation history with token budgets."""
    
    max_messages: int = 20
    max_tokens: int = 10000
    preserve_first_user: bool = True
    
    def trim_by_count(
        self,
        messages: list[dict[str, Any]],
        max_messages: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Keep only the most recent N messages."""
        max_msg = max_messages or self.max_messages
        
        if len(messages) <= max_msg:
            return messages
        
        if self.preserve_first_user:
            first_user_idx = None
            for i, msg in enumerate(messages):
                if msg.get("role") == "user":
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                first_user = [messages[first_user_idx]]
                recent = messages[-(max_msg - 1):]
                return first_user + recent
        
        return messages[-max_msg:]
    
    def trim_by_tokens(
        self,
        messages: list[dict[str, Any]],
        max_tokens: Optional[int] = None,
        estimate_fn: Optional[callable] = None
    ) -> list[dict[str, Any]]:
        """Keep messages until token budget is reached."""
        max_tok = max_tokens or self.max_tokens
        
        if estimate_fn is None:
            def estimate_fn(msg):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return len(content) // 4
                return len(str(content)) // 4
        
        message_tokens = [(msg, estimate_fn(msg)) for msg in messages]
        
        kept = []
        total_tokens = 0
        
        for msg, tokens in reversed(message_tokens):
            if total_tokens + tokens <= max_tok:
                kept.insert(0, msg)
                total_tokens += tokens
            else:
                break
        
        return kept
    
    def create_summary_message(
        self,
        messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create a summary message from multiple messages."""
        summary_parts = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, str) and content:
                preview = content[:100] + "..." if len(content) > 100 else content
                summary_parts.append(f"[{role}]: {preview}")
        
        summary_text = "[Previous conversation summary]\n" + "\n".join(summary_parts)
        
        return {"role": "user", "content": summary_text}
    
    def trim_with_summary(
        self,
        messages: list[dict[str, Any]],
        keep_recent: int = 10
    ) -> list[dict[str, Any]]:
        """Summarize old messages, keep recent ones."""
        if len(messages) <= keep_recent:
            return messages
        
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]
        
        summary = self.create_summary_message(old_messages)
        
        return [summary] + recent_messages


if __name__ == "__main__":
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"Question {i+1}: " + "x" * 100})
        messages.append({"role": "assistant", "content": f"Answer {i+1}: " + "y" * 200})
    
    print("Conversation Management Demo")
    print("=" * 50)
    print(f"Original: {len(messages)} messages")
    
    manager = ConversationManager()
    
    trimmed = manager.trim_by_count(messages, 10)
    print(f"After trim_by_count(10): {len(trimmed)} messages")
    
    trimmed = manager.trim_by_tokens(messages, 2000)
    print(f"After trim_by_tokens(2000): {len(trimmed)} messages")
    
    trimmed = manager.trim_with_summary(messages, 6)
    print(f"After trim_with_summary(6): {len(trimmed)} messages")
