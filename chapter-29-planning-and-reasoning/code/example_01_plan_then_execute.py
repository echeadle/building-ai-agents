"""
Basic plan-then-execute pattern.

This example demonstrates the fundamental planning pattern where an agent:
1. Analyzes a task and creates a structured plan
2. Executes each step sequentially
3. Accumulates context from completed steps
4. Synthesizes a final result

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


def create_plan(task: str) -> dict:
    """
    Generate a structured plan for completing a task.
    
    Args:
        task: The task description from the user
        
    Returns:
        A dictionary containing the plan with steps
    """
    planning_prompt = f"""Analyze this task and create a detailed plan to complete it.

Task: {task}

Create a plan with the following structure:
1. Goal: One sentence describing the end goal
2. Steps: A numbered list of specific actions to take (3-7 steps)
3. Success criteria: How to know when the task is complete

Respond in JSON format:
{{
    "goal": "...",
    "steps": [
        {{"step_number": 1, "action": "...", "expected_outcome": "..."}},
        {{"step_number": 2, "action": "...", "expected_outcome": "..."}}
    ],
    "success_criteria": "..."
}}

Be specific and actionable. Each step should be something concrete that can be executed."""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": planning_prompt}]
    )
    
    # Parse the JSON response
    response_text = response.content[0].text
    
    # Handle potential markdown code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    return json.loads(response_text.strip())


def execute_step(step: dict, context: str) -> str:
    """
    Execute a single step from the plan.
    
    Args:
        step: The step dictionary with action and expected_outcome
        context: Accumulated context from previous steps
        
    Returns:
        The result of executing this step
    """
    execution_prompt = f"""You are executing step {step['step_number']} of a plan.

Action to take: {step['action']}
Expected outcome: {step['expected_outcome']}

Previous context:
{context if context else 'This is the first step.'}

Execute this step and provide the result. Be thorough but concise."""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": execution_prompt}]
    )
    
    return response.content[0].text


def plan_and_execute(task: str, verbose: bool = True) -> str:
    """
    Complete a task using the plan-then-execute pattern.
    
    Args:
        task: The task to complete
        verbose: Whether to print progress
        
    Returns:
        The final result
    """
    # Phase 1: Planning
    if verbose:
        print("=" * 50)
        print("PLANNING PHASE")
        print("=" * 50)
    
    plan = create_plan(task)
    
    if verbose:
        print(f"\nGoal: {plan['goal']}")
        print(f"\nSteps:")
        for step in plan['steps']:
            print(f"  {step['step_number']}. {step['action']}")
        print(f"\nSuccess criteria: {plan['success_criteria']}")
        print()
    
    # Phase 2: Execution
    if verbose:
        print("=" * 50)
        print("EXECUTION PHASE")
        print("=" * 50)
    
    context = ""
    results = []
    
    for step in plan['steps']:
        if verbose:
            print(f"\n→ Executing step {step['step_number']}: {step['action']}")
        
        result = execute_step(step, context)
        results.append({
            "step": step['step_number'],
            "action": step['action'],
            "result": result
        })
        
        # Accumulate context for next step
        context += f"\nStep {step['step_number']} ({step['action']}): {result}\n"
        
        if verbose:
            # Show truncated result
            display_result = result[:200] + "..." if len(result) > 200 else result
            print(f"  Result: {display_result}")
    
    # Final synthesis
    synthesis_prompt = f"""You completed a task with the following goal: {plan['goal']}

Here are the results from each step:
{context}

Provide a final summary that addresses the original task. 
Check against success criteria: {plan['success_criteria']}"""

    final_response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    
    return final_response.content[0].text


if __name__ == "__main__":
    # Example: A multi-step analysis task
    task = """
    Help me understand the pros and cons of learning Rust as a second 
    programming language after Python. Consider the learning curve, 
    job market, and practical applications.
    """
    
    print("Task:", task.strip())
    print()
    
    result = plan_and_execute(task)
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(result)
