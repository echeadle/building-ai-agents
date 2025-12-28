"""
Deciding when to use planning.

This example demonstrates how an agent can decide whether a task
needs explicit planning or can be handled with a direct response.

Chapter 29: Planning and Reasoning
"""

import os
import json
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


def analyze_task_complexity(task: str) -> dict:
    """
    Analyze a task to determine its complexity and planning needs.
    
    Args:
        task: The task description
        
    Returns:
        Dictionary with complexity analysis
    """
    analysis_prompt = f"""Analyze this task's complexity.

Task: {task}

Evaluate:
1. Is this a simple factual question? (yes/no)
2. Does it require multiple distinct steps? (yes/no)
3. Do later steps depend on earlier results? (yes/no)
4. Could getting off track be a risk? (yes/no)
5. How many sub-tasks are involved? (count)

Respond in JSON:
{{
    "simple_question": true/false,
    "multiple_steps": true/false,
    "step_dependencies": true/false,
    "off_track_risk": true/false,
    "subtask_count": number,
    "complexity_score": 1-5,
    "reasoning": "Brief explanation"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    return json.loads(text.strip())


def should_plan(task: str) -> tuple[bool, str, dict]:
    """
    Decide whether a task needs explicit planning.
    
    Args:
        task: The task description
        
    Returns:
        Tuple of (should_plan, reasoning, analysis)
    """
    analysis = analyze_task_complexity(task)
    
    # Decision logic based on analysis
    should_use_plan = (
        analysis.get("complexity_score", 1) >= 3 or
        analysis.get("multiple_steps", False) or
        analysis.get("step_dependencies", False) or
        analysis.get("off_track_risk", False)
    ) and not analysis.get("simple_question", True)
    
    reasoning = analysis.get("reasoning", "")
    
    return should_use_plan, reasoning, analysis


def execute_with_planning(task: str) -> str:
    """
    Execute a task using explicit planning.
    
    Args:
        task: The task to complete
        
    Returns:
        The result
    """
    # Create plan
    plan_prompt = f"""Create and execute a plan for this task:

{task}

First, outline 3-5 specific steps.
Then, execute each step and show your work.
Finally, synthesize the results into a final answer.

Format:
PLAN:
1. [Step 1]
2. [Step 2]
...

EXECUTION:
Step 1: [Result]
Step 2: [Result]
...

FINAL ANSWER:
[Your conclusion]"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": plan_prompt}]
    )
    
    return response.content[0].text


def execute_direct(task: str) -> str:
    """
    Execute a task with a direct response.
    
    Args:
        task: The task to complete
        
    Returns:
        The result
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{"role": "user", "content": task}]
    )
    
    return response.content[0].text


class AdaptiveAgent:
    """An agent that adapts its approach based on task complexity."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the adaptive agent.
        
        Args:
            verbose: Whether to show decision-making
        """
        self.verbose = verbose
        self.decision_history: list[dict] = []
    
    def run(self, task: str) -> str:
        """
        Run the agent, choosing the appropriate approach.
        
        Args:
            task: The task to complete
            
        Returns:
            The result
        """
        if self.verbose:
            print(f"Task: {task[:60]}...")
            print("-" * 40)
        
        # Analyze and decide
        use_planning, reasoning, analysis = should_plan(task)
        
        decision = {
            "task": task,
            "use_planning": use_planning,
            "reasoning": reasoning,
            "complexity_score": analysis.get("complexity_score", 0)
        }
        self.decision_history.append(decision)
        
        if self.verbose:
            approach = "PLANNING" if use_planning else "DIRECT"
            print(f"Approach: {approach}")
            print(f"Reasoning: {reasoning}")
            print(f"Complexity: {analysis.get('complexity_score', '?')}/5")
            print("-" * 40)
        
        # Execute with chosen approach
        if use_planning:
            result = execute_with_planning(task)
        else:
            result = execute_direct(task)
        
        return result
    
    def get_decision_stats(self) -> dict:
        """
        Get statistics about planning decisions.
        
        Returns:
            Dictionary with decision statistics
        """
        total = len(self.decision_history)
        planned = sum(1 for d in self.decision_history if d["use_planning"])
        direct = total - planned
        avg_complexity = sum(
            d["complexity_score"] for d in self.decision_history
        ) / total if total > 0 else 0
        
        return {
            "total_tasks": total,
            "planned": planned,
            "direct": direct,
            "planning_rate": planned / total if total > 0 else 0,
            "avg_complexity": avg_complexity
        }


def demonstrate_adaptive_planning():
    """Demonstrate the adaptive planning approach."""
    
    tasks = [
        # Simple questions - should use direct
        "What's the capital of France?",
        "How many ounces are in a pound?",
        "What does HTTP stand for?",
        
        # Complex tasks - should use planning
        """
        I'm moving to a new city for a job. Help me create a timeline 
        for finding an apartment, setting up utilities, changing my 
        address with various services, and getting settled in over the 
        next 6 weeks.
        """,
        """
        Compare the pros and cons of starting a business as a sole 
        proprietorship vs an LLC. Consider liability, taxes, paperwork,
        and growth potential.
        """,
        """
        Help me plan a week-long road trip from San Francisco to Seattle,
        with stops at interesting places along the way. I'm interested in
        nature, good food, and quirky attractions.
        """
    ]
    
    agent = AdaptiveAgent(verbose=True)
    
    for i, task in enumerate(tasks, 1):
        print("\n" + "=" * 60)
        print(f"TASK {i}")
        print("=" * 60)
        
        result = agent.run(task)
        
        # Show truncated result
        print("\nResult preview:")
        print(result[:300] + "..." if len(result) > 300 else result)
    
    # Show statistics
    stats = agent.get_decision_stats()
    print("\n" + "=" * 60)
    print("DECISION STATISTICS")
    print("=" * 60)
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Used planning: {stats['planned']} ({stats['planning_rate']*100:.0f}%)")
    print(f"Direct response: {stats['direct']}")
    print(f"Average complexity: {stats['avg_complexity']:.1f}/5")


def quick_planning_check(task: str) -> bool:
    """
    A simple heuristic check for whether planning might help.
    
    This is a fast, rule-based check that can be used before
    calling the LLM for a more thorough analysis.
    
    Args:
        task: The task to check
        
    Returns:
        True if planning might be beneficial
    """
    task_lower = task.lower()
    
    # Simple questions usually don't need planning
    simple_patterns = [
        "what is", "what's", "who is", "who's", "when did",
        "where is", "how many", "how much", "define", "explain"
    ]
    
    # Complex indicators suggest planning might help
    complex_patterns = [
        "compare", "analyze", "plan", "help me", "create a",
        "step by step", "timeline", "strategy", "decide",
        "pros and cons", "options", "recommendation"
    ]
    
    # Check for simple patterns first
    for pattern in simple_patterns:
        if task_lower.startswith(pattern):
            return False
    
    # Check for complex patterns
    for pattern in complex_patterns:
        if pattern in task_lower:
            return True
    
    # Default to direct for short tasks, planning for long ones
    return len(task) > 200


if __name__ == "__main__":
    print("ADAPTIVE PLANNING DEMONSTRATION")
    print("=" * 60)
    
    demonstrate_adaptive_planning()
    
    # Also show quick heuristic check
    print("\n" + "=" * 60)
    print("QUICK HEURISTIC CHECK (no LLM call)")
    print("=" * 60)
    
    test_tasks = [
        "What's 2 + 2?",
        "Help me plan my wedding",
        "Define recursion",
        "Compare Python and JavaScript for web development"
    ]
    
    for task in test_tasks:
        needs_planning = quick_planning_check(task)
        indicator = "📋 PLAN" if needs_planning else "⚡ DIRECT"
        print(f"{indicator}: {task[:50]}...")
