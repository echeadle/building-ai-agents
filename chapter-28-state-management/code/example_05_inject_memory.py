"""
Injecting working memory into agent prompts.

This example shows how to include working memory context
in system prompts so the agent can use it.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class WorkingMemory:
    """Simplified working memory for this example."""
    
    current_goal: Optional[str] = None
    gathered_facts: dict[str, Any] = field(default_factory=dict)
    steps_completed: list[str] = field(default_factory=list)
    
    def get_context_summary(self) -> str:
        """Generate context summary for prompts."""
        if not self.current_goal:
            return "No active task."
        
        lines = [f"CURRENT GOAL: {self.current_goal}"]
        
        if self.steps_completed:
            lines.append(f"COMPLETED STEPS: {', '.join(self.steps_completed)}")
        
        if self.gathered_facts:
            lines.append("GATHERED INFORMATION:")
            for k, v in self.gathered_facts.items():
                lines.append(f"  - {k}: {v}")
        
        return "\n".join(lines)


def build_system_prompt_with_memory(
    base_prompt: str,
    working_memory: WorkingMemory
) -> str:
    """
    Build a system prompt that includes working memory context.
    
    Args:
        base_prompt: The agent's base system prompt
        working_memory: Current working memory state
    
    Returns:
        Complete system prompt with context
    """
    memory_context = working_memory.get_context_summary()
    
    if memory_context == "No active task.":
        return base_prompt
    
    return f"""{base_prompt}

## Current Task Context

{memory_context}

Use this context to inform your responses. Build on what has already been accomplished and use the information that has been gathered."""


def demonstrate_memory_injection():
    """Show how memory context improves agent responses."""
    client = anthropic.Anthropic()
    
    base_prompt = """You are a helpful travel planning assistant. 
You help users plan trips by gathering information and making recommendations."""
    
    print("Demonstrating Memory Injection")
    print("=" * 50)
    
    # Create working memory with some context
    memory = WorkingMemory()
    memory.current_goal = "Plan a weekend trip to Portland"
    memory.gathered_facts = {
        "budget": "$500",
        "dates": "March 15-17",
        "interests": "food, hiking, coffee",
        "accommodation": "Prefer boutique hotels downtown"
    }
    memory.steps_completed = ["Gather preferences", "Research neighborhoods"]
    
    # Build the enhanced prompt
    enhanced_prompt = build_system_prompt_with_memory(base_prompt, memory)
    
    print("\nEnhanced System Prompt:")
    print("-" * 40)
    print(enhanced_prompt)
    print("-" * 40)
    
    # Now ask a question that benefits from context
    user_message = "What should I prioritize for my trip?"
    print(f"\nUser: {user_message}")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=enhanced_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    print(f"\nAssistant: {response.content[0].text}")
    
    # Compare with response without context
    print("\n" + "=" * 50)
    print("For comparison, without memory context:")
    print("-" * 40)
    
    response_no_context = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=base_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    print(f"Assistant: {response_no_context.content[0].text}")
    
    print("\n" + "=" * 50)
    print("Notice how the context-aware response is more specific and personalized!")


if __name__ == "__main__":
    demonstrate_memory_injection()
