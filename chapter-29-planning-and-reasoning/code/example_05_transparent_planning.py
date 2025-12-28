"""
Transparent planning with user-visible reasoning.

This example demonstrates how to make an agent's planning and reasoning
visible to users, building trust and enabling debugging.

Chapter 29: Planning and Reasoning
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class ThinkingStep:
    """A visible step in the agent's reasoning."""
    timestamp: datetime
    phase: str  # "planning", "reasoning", "executing", "revising"
    thought: str
    
    def display(self) -> str:
        """Format the thinking step for display."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        phase_icons = {
            "planning": "📋",
            "reasoning": "💭",
            "executing": "⚡",
            "revising": "🔄",
            "complete": "✅"
        }
        icon = phase_icons.get(self.phase, "•")
        return f"[{time_str}] {icon} {self.phase.upper()}: {self.thought}"


class TransparentPlanningAgent:
    """An agent that shows its planning and reasoning to users."""
    
    def __init__(self):
        """Initialize the transparent planning agent."""
        self.thinking_log: list[ThinkingStep] = []
        self.plan: dict | None = None
    
    def think(self, phase: str, thought: str) -> None:
        """
        Record and display a thinking step.
        
        Args:
            phase: The phase of thinking (planning, reasoning, executing, etc.)
            thought: The thought to record
        """
        step = ThinkingStep(
            timestamp=datetime.now(),
            phase=phase,
            thought=thought
        )
        self.thinking_log.append(step)
        print(step.display())
    
    def plan_task(self, task: str) -> dict:
        """
        Create a plan with visible reasoning.
        
        Args:
            task: The task to plan
            
        Returns:
            The generated plan
        """
        self.think("planning", f"Received task: {task[:50]}...")
        self.think("reasoning", "Analyzing what needs to be done...")
        
        # First, reason about the task
        reasoning_prompt = f"""Before creating a plan, think through this task:

Task: {task}

Consider:
1. What is the core objective?
2. What information do I need?
3. What are potential challenges?
4. What's the logical sequence of steps?

Think out loud about your approach."""

        reasoning_response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": reasoning_prompt}]
        )
        
        reasoning = reasoning_response.content[0].text
        
        # Show condensed reasoning to user
        lines = [l for l in reasoning.split('\n') if l.strip()]
        for line in lines[:5]:  # Show first few lines
            self.think("reasoning", line.strip()[:80])
        
        if len(lines) > 5:
            self.think("reasoning", f"...and {len(lines) - 5} more considerations")
        
        # Now create the plan
        self.think("planning", "Formulating structured plan...")
        
        plan_prompt = f"""Based on your analysis, create a concrete plan.

Task: {task}

Your reasoning:
{reasoning}

Create a JSON plan:
{{
    "goal": "Clear statement of the goal",
    "approach": "Brief description of your approach",
    "steps": [
        {{"step": 1, "action": "What to do", "why": "Why this step"}}
    ],
    "risks": ["Potential risk 1", "Potential risk 2"]
}}"""

        plan_response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": plan_prompt}]
        )
        
        text = plan_response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        self.plan = json.loads(text.strip())
        
        # Display the plan
        self.think("planning", f"Goal: {self.plan['goal']}")
        self.think("planning", f"Approach: {self.plan['approach']}")
        
        for step in self.plan['steps']:
            self.think("planning", f"Step {step['step']}: {step['action']}")
        
        if self.plan.get('risks'):
            self.think("reasoning", f"Identified {len(self.plan['risks'])} potential risks")
        
        return self.plan
    
    def execute_with_commentary(self, step: dict) -> str:
        """
        Execute a step with visible reasoning.
        
        Args:
            step: The step to execute
            
        Returns:
            The result of the step
        """
        self.think("executing", f"Starting: {step['action']}")
        self.think("reasoning", f"This step matters because: {step['why']}")
        
        execution_prompt = f"""Execute this step and provide the result.

Action: {step['action']}
Purpose: {step['why']}

Provide:
1. Your execution approach
2. The result
3. Any observations for future steps"""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": execution_prompt}]
        )
        
        result = response.content[0].text
        self.think("executing", f"Completed: {result[:60]}...")
        
        return result
    
    def run(self, task: str) -> str:
        """
        Run the agent with full transparency.
        
        Args:
            task: The task to complete
            
        Returns:
            The final result
        """
        print("=" * 60)
        print("TRANSPARENT PLANNING AGENT")
        print("=" * 60)
        print()
        
        # Plan
        plan = self.plan_task(task)
        print()
        
        # Execute each step
        results = []
        for step in plan['steps']:
            result = self.execute_with_commentary(step)
            results.append({"step": step['step'], "result": result})
            print()
        
        # Synthesize
        self.think("complete", "All steps completed, synthesizing result...")
        
        synthesis_prompt = f"""Synthesize the final result.

Goal: {plan['goal']}

Step results:
{json.dumps(results, indent=2)}

Provide a clear, complete answer to the original task."""

        final_response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        final_result = final_response.content[0].text
        self.think("complete", "Result ready")
        
        return final_result
    
    def get_thinking_log(self) -> str:
        """
        Get the full thinking log as a string.
        
        Returns:
            Formatted thinking log
        """
        return "\n".join(step.display() for step in self.thinking_log)
    
    def get_thinking_summary(self) -> dict:
        """
        Get a summary of the thinking process.
        
        Returns:
            Dictionary with counts by phase
        """
        phases = {}
        for step in self.thinking_log:
            phases[step.phase] = phases.get(step.phase, 0) + 1
        return {
            "total_steps": len(self.thinking_log),
            "by_phase": phases,
            "duration": (
                self.thinking_log[-1].timestamp - self.thinking_log[0].timestamp
            ).total_seconds() if len(self.thinking_log) > 1 else 0
        }


class VerbosityLevels:
    """Different verbosity levels for transparent output."""
    
    QUIET = 0      # Only show final result
    NORMAL = 1     # Show major milestones
    VERBOSE = 2    # Show all reasoning
    DEBUG = 3      # Show everything including raw responses


class ConfigurableTransparentAgent:
    """An agent with configurable transparency levels."""
    
    def __init__(self, verbosity: int = VerbosityLevels.NORMAL):
        """
        Initialize with a verbosity level.
        
        Args:
            verbosity: How much to show (0=quiet, 1=normal, 2=verbose, 3=debug)
        """
        self.verbosity = verbosity
        self.thinking_log: list[ThinkingStep] = []
    
    def think(self, phase: str, thought: str, min_verbosity: int = 1) -> None:
        """
        Record a thinking step, showing it if verbosity is high enough.
        
        Args:
            phase: The thinking phase
            thought: The thought content
            min_verbosity: Minimum verbosity level to display this
        """
        step = ThinkingStep(
            timestamp=datetime.now(),
            phase=phase,
            thought=thought
        )
        self.thinking_log.append(step)
        
        if self.verbosity >= min_verbosity:
            print(step.display())
    
    def run(self, task: str) -> str:
        """Run the agent with configured verbosity."""
        self.think("planning", f"Starting: {task[:40]}...", VerbosityLevels.QUIET)
        
        # Planning
        self.think("reasoning", "Analyzing task requirements...", VerbosityLevels.NORMAL)
        self.think("reasoning", "Identifying key components...", VerbosityLevels.VERBOSE)
        self.think("reasoning", "Checking for edge cases...", VerbosityLevels.DEBUG)
        
        # Create simple plan
        plan_prompt = f"""Create a brief 3-step plan for: {task}
        
Respond in JSON: {{"steps": ["step1", "step2", "step3"]}}"""
        
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=256,
            messages=[{"role": "user", "content": plan_prompt}]
        )
        
        self.think("planning", "Plan created", VerbosityLevels.NORMAL)
        
        # Execute
        self.think("executing", "Running plan...", VerbosityLevels.NORMAL)
        
        execute_prompt = f"""Complete this task: {task}

Provide a helpful response."""

        final_response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": execute_prompt}]
        )
        
        self.think("complete", "Done", VerbosityLevels.QUIET)
        
        return final_response.content[0].text


if __name__ == "__main__":
    # Demo 1: Full transparency
    print("\n" + "=" * 60)
    print("DEMO 1: Full Transparency")
    print("=" * 60 + "\n")
    
    agent = TransparentPlanningAgent()
    
    task = """
    I want to start a small vegetable garden on my apartment balcony.
    Help me figure out what I need and how to get started.
    """
    
    result = agent.run(task)
    
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)
    
    # Show thinking summary
    summary = agent.get_thinking_summary()
    print(f"\n📊 Thinking Summary:")
    print(f"   Total steps: {summary['total_steps']}")
    print(f"   Duration: {summary['duration']:.1f}s")
    print(f"   By phase: {summary['by_phase']}")
    
    # Demo 2: Different verbosity levels
    print("\n" + "=" * 60)
    print("DEMO 2: Verbosity Levels")
    print("=" * 60)
    
    simple_task = "What's a good beginner programming language?"
    
    for level_name, level in [
        ("QUIET", VerbosityLevels.QUIET),
        ("NORMAL", VerbosityLevels.NORMAL),
        ("VERBOSE", VerbosityLevels.VERBOSE)
    ]:
        print(f"\n--- {level_name} MODE ---")
        agent = ConfigurableTransparentAgent(verbosity=level)
        result = agent.run(simple_task)
        print(f"Result: {result[:100]}...")
