"""
Working memory for current task context.

This example shows how to maintain structured information about
the current task the agent is working on.

Chapter 28: State Management
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from enum import Enum

# Load environment variables from .env file
load_dotenv()


class TaskStatus(Enum):
    """Status of the current task."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkingMemory:
    """
    Working memory for tracking current task context.
    
    This is the agent's "scratchpad" for the current task,
    holding structured information about goals, progress, and findings.
    """
    
    # Current task information
    current_goal: Optional[str] = None
    task_status: TaskStatus = TaskStatus.NOT_STARTED
    started_at: Optional[str] = None
    
    # Progress tracking
    steps_completed: list[str] = field(default_factory=list)
    steps_remaining: list[str] = field(default_factory=list)
    current_step: Optional[str] = None
    
    # Information gathered during task
    gathered_facts: dict[str, Any] = field(default_factory=dict)
    
    # Decisions and reasoning
    decisions_made: list[dict] = field(default_factory=list)
    
    # Error tracking
    errors_encountered: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def start_task(self, goal: str, planned_steps: list[str] = None) -> None:
        """Begin a new task."""
        self.current_goal = goal
        self.task_status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now().isoformat()
        self.steps_remaining = planned_steps or []
        self.steps_completed = []
        self.gathered_facts = {}
        self.decisions_made = []
        self.errors_encountered = []
        self.retry_count = 0
        
        if self.steps_remaining:
            self.current_step = self.steps_remaining[0]
    
    def complete_step(self, step: str, result: Optional[str] = None) -> None:
        """Mark a step as complete."""
        self.steps_completed.append(step)
        
        if step in self.steps_remaining:
            self.steps_remaining.remove(step)
        
        if result:
            self.gathered_facts[f"step_{len(self.steps_completed)}_result"] = result
        
        # Move to next step
        if self.steps_remaining:
            self.current_step = self.steps_remaining[0]
        else:
            self.current_step = None
    
    def add_fact(self, key: str, value: Any) -> None:
        """Store a piece of information gathered during the task."""
        self.gathered_facts[key] = value
    
    def record_decision(self, decision: str, reasoning: str) -> None:
        """Record a decision made during the task."""
        self.decisions_made.append({
            "decision": decision,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_error(self, error: str) -> bool:
        """
        Record an error. Returns True if we should retry.
        """
        self.errors_encountered.append(error)
        self.retry_count += 1
        
        if self.retry_count >= self.max_retries:
            self.task_status = TaskStatus.FAILED
            return False
        
        self.task_status = TaskStatus.BLOCKED
        return True
    
    def complete_task(self, success: bool = True) -> None:
        """Mark the task as complete."""
        self.task_status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.current_step = None
    
    def get_context_summary(self) -> str:
        """
        Generate a summary of current working memory for the LLM.
        
        This is injected into prompts to give the agent context.
        """
        lines = []
        
        if self.current_goal:
            lines.append(f"CURRENT GOAL: {self.current_goal}")
            lines.append(f"STATUS: {self.task_status.value}")
        
        if self.current_step:
            lines.append(f"CURRENT STEP: {self.current_step}")
        
        if self.steps_completed:
            lines.append(f"COMPLETED STEPS: {', '.join(self.steps_completed)}")
        
        if self.steps_remaining:
            lines.append(f"REMAINING STEPS: {', '.join(self.steps_remaining)}")
        
        if self.gathered_facts:
            lines.append("GATHERED INFORMATION:")
            for key, value in self.gathered_facts.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                lines.append(f"  - {key}: {value_str}")
        
        if self.decisions_made:
            lines.append("DECISIONS MADE:")
            for d in self.decisions_made[-3:]:  # Last 3 decisions
                lines.append(f"  - {d['decision']}")
        
        if self.errors_encountered:
            lines.append(f"ERRORS: {len(self.errors_encountered)} encountered")
            lines.append(f"  Last error: {self.errors_encountered[-1]}")
        
        return "\n".join(lines) if lines else "No active task."


def demonstrate_working_memory():
    """Show how working memory tracks task progress."""
    print("Demonstrating Working Memory")
    print("=" * 50)
    
    memory = WorkingMemory()
    
    # Start a research task
    print("\n1. Starting a new task...")
    memory.start_task(
        goal="Research competitor pricing",
        planned_steps=[
            "Identify top 3 competitors",
            "Find pricing pages",
            "Extract pricing tiers",
            "Summarize findings"
        ]
    )
    
    print(f"Goal: {memory.current_goal}")
    print(f"Status: {memory.task_status.value}")
    print(f"Current step: {memory.current_step}")
    
    # Simulate progress
    print("\n2. Making progress...")
    memory.add_fact("competitors", ["Acme Corp", "Beta Inc", "Gamma LLC"])
    memory.complete_step("Identify top 3 competitors", "Found 3 competitors")
    
    memory.record_decision(
        decision="Focus on Acme Corp first",
        reasoning="They are the market leader with the most similar product"
    )
    
    print(f"Steps completed: {memory.steps_completed}")
    print(f"Current step: {memory.current_step}")
    
    # Add more facts
    print("\n3. Gathering information...")
    memory.add_fact("acme_pricing", {"basic": 29, "pro": 99, "enterprise": 299})
    memory.complete_step("Find pricing pages")
    
    # Show the full context
    print("\n4. Full Context Summary for LLM:")
    print("-" * 40)
    print(memory.get_context_summary())
    print("-" * 40)
    
    # Simulate an error
    print("\n5. Handling an error...")
    should_retry = memory.record_error("Beta Inc website returned 403")
    print(f"Should retry: {should_retry}")
    print(f"Status: {memory.task_status.value}")
    
    # Complete the task
    print("\n6. Completing the task...")
    memory.task_status = TaskStatus.IN_PROGRESS  # Resume after error
    memory.complete_step("Extract pricing tiers")
    memory.complete_step("Summarize findings")
    memory.complete_task(success=True)
    
    print(f"Final status: {memory.task_status.value}")
    print(f"Steps completed: {len(memory.steps_completed)}")


if __name__ == "__main__":
    demonstrate_working_memory()
