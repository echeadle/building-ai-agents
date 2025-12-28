---
chapter: 29
title: "Planning and Reasoning"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 29: Planning and Reasoning

## Introduction

Imagine asking someone to organize a surprise birthday party. A hasty person might immediately start calling guests, only to realize they haven't picked a date, a venue, or even checked if the birthday person is available. A thoughtful planner would first sketch out what needs to happen: pick a date, choose a venue, create a guest list, send invitations, order a cake, arrange decorations—all before making a single phone call.

Agents face the same choice. They can dive straight into action, or they can pause to think through their approach first. In Chapter 27, we built the agentic loop that lets agents perceive, think, and act. In Chapter 28, we gave agents memory to maintain context. Now we'll teach our agents to *plan*—to reason explicitly about what they need to do before they start doing it.

Planning transforms agents from reactive tools into strategic problem-solvers. Instead of fumbling through trial and error, a planning agent maps out its approach, anticipates challenges, and executes with purpose. This chapter shows you how to implement planning patterns that make your agents more reliable, transparent, and effective.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why explicit planning improves agent reliability
- Implement plan-then-execute patterns for multi-step tasks
- Use chain-of-thought reasoning to enhance agent decision-making
- Build agents that can revise plans when circumstances change
- Display planning steps to users for transparency and trust

## The Value of Explicit Planning

When we ask an LLM to complete a complex task, it might produce a reasonable result—or it might wander off track. Without explicit planning, the agent makes decisions incrementally, each step influenced only by immediate context. This leads to several problems:

**Problem 1: Getting lost in the weeds.** An agent researching a topic might dive deep into the first interesting tangent, forgetting its original goal. Without a plan to reference, there's no anchor to pull it back on track.

**Problem 2: Inefficient tool use.** An agent might call tools in suboptimal order—fetching data it doesn't need, or missing steps that would have informed better decisions earlier.

**Problem 3: No way to recover.** If something goes wrong midway through a task, an unplanned agent has no roadmap for getting back on course. It doesn't know what it intended to do, so it can't figure out what to try next.

**Problem 4: Opaque behavior.** Users can't understand why an agent is doing what it's doing. This erodes trust and makes debugging nearly impossible.

Explicit planning addresses all of these by making the agent's intentions visible and actionable:

```
Without Planning:
User: "Compare three laptops for me"
Agent: [searches laptop 1] → [gets distracted by review] → [searches laptop 2] 
       → [forgets what to compare] → [gives incomplete answer]

With Planning:
User: "Compare three laptops for me"
Agent: "I'll plan my approach:
        1. Identify the three laptops to compare
        2. Research specs for each (CPU, RAM, storage, display)
        3. Research prices for each
        4. Research user reviews for each
        5. Create comparison table
        6. Provide recommendation"
Agent: [executes each step systematically] → [complete, organized answer]
```

The planning step takes only a moment, but it dramatically improves the quality and reliability of the final result.

## Plan-Then-Execute: The Basic Pattern

The simplest planning pattern separates thinking from doing into two distinct phases:

1. **Planning Phase**: The agent analyzes the task and produces a structured plan
2. **Execution Phase**: The agent follows the plan step by step

Let's implement this pattern:

```python
"""
Basic plan-then-execute pattern.

Chapter 29: Planning and Reasoning
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

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
2. Steps: A numbered list of specific actions to take
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
            print(f"  Result: {result[:200]}..." if len(result) > 200 else f"  Result: {result}")
    
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
    
    result = plan_and_execute(task)
    
    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(result)
```

This basic pattern establishes a clear separation between thinking and doing. The agent first reasons about the entire task, producing a plan that serves as a roadmap. Then it executes each step, accumulating context as it goes.

> **Note:** The plan-then-execute pattern adds latency—you're making extra API calls for planning. This is a worthwhile trade for complex tasks but overkill for simple ones. We'll discuss when to use planning later in this chapter.

## Chain-of-Thought Reasoning

Chain-of-thought (CoT) is a technique that encourages LLMs to reason step by step before producing an answer. Unlike plan-then-execute, which creates an explicit multi-step plan, chain-of-thought elicits reasoning within a single response.

Here's how chain-of-thought works:

```python
"""
Chain-of-thought reasoning patterns.

Chapter 29: Planning and Reasoning
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


def ask_with_chain_of_thought(question: str) -> dict:
    """
    Ask a question using chain-of-thought prompting.
    
    Args:
        question: The question to answer
        
    Returns:
        Dictionary with reasoning and final answer
    """
    cot_prompt = f"""Question: {question}

Please think through this step by step:
1. First, identify what information is needed
2. Then, reason through each relevant consideration  
3. Finally, provide your conclusion

Format your response as:
REASONING:
[Your step-by-step thinking]

ANSWER:
[Your final answer]"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": cot_prompt}]
    )
    
    text = response.content[0].text
    
    # Parse the response
    if "REASONING:" in text and "ANSWER:" in text:
        parts = text.split("ANSWER:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        answer = parts[1].strip()
    else:
        reasoning = text
        answer = text
    
    return {
        "reasoning": reasoning,
        "answer": answer,
        "full_response": text
    }


def compare_with_and_without_cot(question: str) -> None:
    """
    Compare responses with and without chain-of-thought.
    
    Args:
        question: The question to answer both ways
    """
    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    
    # Without CoT
    print("\n--- WITHOUT CHAIN-OF-THOUGHT ---")
    direct_response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{"role": "user", "content": question}]
    )
    print(direct_response.content[0].text)
    
    # With CoT
    print("\n--- WITH CHAIN-OF-THOUGHT ---")
    cot_result = ask_with_chain_of_thought(question)
    print(f"Reasoning:\n{cot_result['reasoning']}")
    print(f"\nFinal Answer:\n{cot_result['answer']}")


if __name__ == "__main__":
    # A question where step-by-step reasoning helps
    question = """
    A farmer has 17 sheep. All but 9 run away. How many sheep does 
    the farmer have left?
    """
    
    compare_with_and_without_cot(question)
    
    print("\n" + "=" * 60)
    
    # A more complex reasoning question
    complex_question = """
    Should a small startup use microservices architecture from day one?
    Consider development speed, operational complexity, and future scaling.
    """
    
    compare_with_and_without_cot(complex_question)
```

Chain-of-thought is particularly valuable when:

- The task requires logical reasoning or calculations
- There are multiple factors to weigh
- The answer isn't immediately obvious
- You want to understand *how* the agent reached its conclusion

💡 **Tip:** You can combine chain-of-thought with plan-then-execute. Use CoT during planning to reason about *which* steps are needed, then execute each step with its own focused reasoning.

## Implementing a Planning Agent

Now let's build a complete agent that incorporates planning into its agentic loop. This agent will:

1. Receive a task
2. Create a plan
3. Execute the plan step by step, using tools as needed
4. Revise the plan if something unexpected happens
5. Report progress transparently

```python
"""
A complete planning agent with tool use.

Chapter 29: Planning and Reasoning
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class PlanStep:
    """A single step in the agent's plan."""
    step_number: int
    action: str
    expected_outcome: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: str = ""
    

@dataclass
class AgentPlan:
    """The agent's plan for completing a task."""
    goal: str
    steps: list[PlanStep]
    success_criteria: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "expected_outcome": s.expected_outcome,
                    "status": s.status,
                    "result": s.result
                }
                for s in self.steps
            ],
            "success_criteria": self.success_criteria
        }
    
    def get_progress_summary(self) -> str:
        completed = sum(1 for s in self.steps if s.status == "completed")
        total = len(self.steps)
        return f"{completed}/{total} steps completed"
    
    def get_next_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "pending":
                return step
        return None


# Define tools the agent can use
TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information on a topic. Use this to find current information, facts, or data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations. Use this for any math operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "take_notes",
        "description": "Save important information for later reference. Use this to track findings and insights.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save"
                },
                "category": {
                    "type": "string",
                    "description": "Category for the note (e.g., 'finding', 'question', 'decision')"
                }
            },
            "required": ["note"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict, notes: list) -> str:
    """
    Execute a tool and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        notes: List to store notes (modified in place)
        
    Returns:
        String result of the tool execution
    """
    if tool_name == "search_web":
        # Simulated search results for demonstration
        query = tool_input.get("query", "")
        return f"[Simulated search results for: {query}]\n" + \
               f"Found information about {query}. Key points include..." + \
               f"\n- Relevant fact 1 about {query}" + \
               f"\n- Relevant fact 2 about {query}" + \
               f"\n- Relevant fact 3 about {query}"
    
    elif tool_name == "calculate":
        expression = tool_input.get("expression", "")
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    elif tool_name == "take_notes":
        note = tool_input.get("note", "")
        category = tool_input.get("category", "general")
        notes.append({"category": category, "note": note})
        return f"Note saved under '{category}': {note}"
    
    else:
        return f"Unknown tool: {tool_name}"


class PlanningAgent:
    """An agent that plans before executing tasks."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.plan: AgentPlan | None = None
        self.notes: list[dict] = []
        self.conversation_history: list[dict] = []
    
    def log(self, message: str, level: str = "info") -> None:
        """Print a log message if verbose mode is on."""
        if self.verbose:
            prefix = {
                "info": "ℹ️ ",
                "plan": "📋",
                "step": "→",
                "success": "✓",
                "error": "✗",
                "think": "💭"
            }.get(level, "  ")
            print(f"{prefix} {message}")
    
    def create_plan(self, task: str) -> AgentPlan:
        """Create a plan for completing a task."""
        self.log("Creating plan...", "plan")
        
        planning_prompt = f"""You are a planning agent. Analyze this task and create a detailed plan.

Task: {task}

Create a plan with 3-7 specific, actionable steps. Consider:
- What information do you need to gather?
- What tools might help (search_web, calculate, take_notes)?
- What's the logical order of operations?

Respond in JSON format:
{{
    "goal": "One sentence describing the end goal",
    "steps": [
        {{"step_number": 1, "action": "Specific action to take", "expected_outcome": "What this achieves"}}
    ],
    "success_criteria": "How to know the task is complete"
}}

Be specific. Each step should be concrete and executable."""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": planning_prompt}]
        )
        
        # Parse response
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        plan_data = json.loads(text.strip())
        
        # Create AgentPlan
        steps = [
            PlanStep(
                step_number=s["step_number"],
                action=s["action"],
                expected_outcome=s["expected_outcome"]
            )
            for s in plan_data["steps"]
        ]
        
        self.plan = AgentPlan(
            goal=plan_data["goal"],
            steps=steps,
            success_criteria=plan_data["success_criteria"]
        )
        
        if self.verbose:
            print(f"\n📋 PLAN: {self.plan.goal}")
            print("-" * 40)
            for step in self.plan.steps:
                print(f"   {step.step_number}. {step.action}")
            print(f"   ✓ Success: {self.plan.success_criteria}")
            print()
        
        return self.plan
    
    def execute_step(self, step: PlanStep) -> bool:
        """
        Execute a single step of the plan.
        
        Returns:
            True if step completed successfully, False otherwise
        """
        step.status = "in_progress"
        self.log(f"Step {step.step_number}: {step.action}", "step")
        
        # Build context from completed steps
        completed_context = "\n".join([
            f"Step {s.step_number} ({s.action}): {s.result}"
            for s in self.plan.steps
            if s.status == "completed"
        ])
        
        # Ask the agent to execute this step
        execution_prompt = f"""You are executing step {step.step_number} of your plan.

CURRENT STEP:
Action: {step.action}
Expected outcome: {step.expected_outcome}

PREVIOUS RESULTS:
{completed_context if completed_context else "This is the first step."}

AVAILABLE NOTES:
{json.dumps(self.notes, indent=2) if self.notes else "No notes yet."}

Execute this step. You have access to these tools: search_web, calculate, take_notes.
Use tools if they would help accomplish this step.
Provide a clear result for this step."""

        messages = [{"role": "user", "content": execution_prompt}]
        
        # Agentic loop for this step (may involve multiple tool calls)
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                tools=TOOLS,
                messages=messages
            )
            
            # Process the response
            if response.stop_reason == "tool_use":
                # Handle tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        self.log(f"  Using tool: {block.name}", "think")
                        result = execute_tool(block.name, block.input, self.notes)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
            else:
                # Step completed
                result_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        result_text += block.text
                
                step.result = result_text
                step.status = "completed"
                self.log(f"  Completed: {result_text[:100]}...", "success")
                return True
        
        # Max iterations reached
        step.status = "failed"
        step.result = "Max iterations reached without completion"
        self.log(f"  Failed: max iterations", "error")
        return False
    
    def should_revise_plan(self, step: PlanStep) -> bool:
        """
        Determine if the plan needs revision after a step.
        
        Returns:
            True if the plan should be revised
        """
        if step.status != "completed":
            return True  # Failed steps may need plan revision
        
        # Ask the agent if revision is needed
        revision_check_prompt = f"""You just completed step {step.step_number} of your plan.

PLAN GOAL: {self.plan.goal}

COMPLETED STEP:
Action: {step.action}
Expected: {step.expected_outcome}
Actual result: {step.result}

REMAINING STEPS:
{chr(10).join([f"{s.step_number}. {s.action}" for s in self.plan.steps if s.status == "pending"])}

Based on the actual result, do the remaining steps still make sense?
Or does the plan need to be revised?

Respond with exactly "CONTINUE" or "REVISE" followed by a brief explanation."""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=256,
            messages=[{"role": "user", "content": revision_check_prompt}]
        )
        
        result = response.content[0].text.strip()
        needs_revision = result.upper().startswith("REVISE")
        
        if needs_revision:
            self.log(f"  Plan revision needed: {result}", "think")
        
        return needs_revision
    
    def revise_plan(self) -> None:
        """Revise the plan based on what's been learned."""
        self.log("Revising plan...", "plan")
        
        completed_results = "\n".join([
            f"Step {s.step_number}: {s.action}\nResult: {s.result}"
            for s in self.plan.steps
            if s.status == "completed"
        ])
        
        revision_prompt = f"""Your plan needs revision based on new information.

ORIGINAL GOAL: {self.plan.goal}

COMPLETED STEPS AND RESULTS:
{completed_results}

NOTES GATHERED:
{json.dumps(self.notes, indent=2) if self.notes else "None"}

Create revised remaining steps to achieve the goal. Consider what you've learned.

Respond in JSON format:
{{
    "revised_steps": [
        {{"step_number": [next number], "action": "...", "expected_outcome": "..."}}
    ],
    "reasoning": "Why these changes were made"
}}"""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": revision_prompt}]
        )
        
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        revision_data = json.loads(text.strip())
        
        # Replace pending steps with revised steps
        completed_steps = [s for s in self.plan.steps if s.status == "completed"]
        next_number = len(completed_steps) + 1
        
        new_steps = [
            PlanStep(
                step_number=next_number + i,
                action=s["action"],
                expected_outcome=s["expected_outcome"]
            )
            for i, s in enumerate(revision_data["revised_steps"])
        ]
        
        self.plan.steps = completed_steps + new_steps
        
        self.log(f"Plan revised: {revision_data.get('reasoning', 'No reason given')}", "plan")
        if self.verbose:
            print("   New remaining steps:")
            for step in new_steps:
                print(f"   {step.step_number}. {step.action}")
    
    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: The task to complete
            
        Returns:
            Final result string
        """
        self.log(f"Starting task: {task}", "info")
        print()
        
        # Create initial plan
        self.create_plan(task)
        
        # Execute steps
        max_steps = 15  # Safety limit
        steps_executed = 0
        
        while steps_executed < max_steps:
            next_step = self.plan.get_next_step()
            
            if next_step is None:
                # All steps completed
                break
            
            success = self.execute_step(next_step)
            steps_executed += 1
            
            if success and self.should_revise_plan(next_step):
                self.revise_plan()
        
        # Generate final result
        self.log("Generating final result...", "info")
        
        all_results = "\n".join([
            f"Step {s.step_number} ({s.action}):\n{s.result}"
            for s in self.plan.steps
            if s.status == "completed"
        ])
        
        synthesis_prompt = f"""You've completed a task. Synthesize the final result.

GOAL: {self.plan.goal}
SUCCESS CRITERIA: {self.plan.success_criteria}

STEP RESULTS:
{all_results}

NOTES:
{json.dumps(self.notes, indent=2) if self.notes else "None"}

Provide a comprehensive final answer that addresses the original goal.
Check that you've met the success criteria."""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return response.content[0].text


if __name__ == "__main__":
    agent = PlanningAgent(verbose=True)
    
    task = """
    I'm deciding between three programming languages for a new web project: 
    Python (Django), JavaScript (Node.js), and Go. I need to consider:
    - Development speed for a small team
    - Performance requirements (moderate traffic expected)
    - Hiring availability in my region
    
    Help me make an informed decision with a clear recommendation.
    """
    
    result = agent.run(task)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)
```

This agent demonstrates several important patterns:

1. **Structured plans** using dataclasses to track goal, steps, and status
2. **Step-by-step execution** with tool access at each step
3. **Progress tracking** so we know where we are in the plan
4. **Revision checks** after each step to adapt when needed
5. **Transparent logging** so users can follow along

## Plan Revision and Adaptation

Plans rarely survive contact with reality unchanged. Good planning agents know when to adapt:

```python
"""
Patterns for plan revision and adaptation.

Chapter 29: Planning and Reasoning
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from enum import Enum

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


class RevisionTrigger(Enum):
    """Reasons a plan might need revision."""
    STEP_FAILED = "step_failed"
    NEW_INFORMATION = "new_information"
    GOAL_CHANGED = "goal_changed"
    OBSTACLE_FOUND = "obstacle_found"
    BETTER_PATH_FOUND = "better_path_found"


def detect_revision_needed(
    original_plan: dict,
    current_step: int,
    step_result: str,
    context: str
) -> tuple[bool, RevisionTrigger | None, str]:
    """
    Analyze whether a plan needs revision.
    
    Args:
        original_plan: The plan being executed
        current_step: Which step just completed
        step_result: Result of the current step
        context: Accumulated context from previous steps
        
    Returns:
        Tuple of (needs_revision, trigger_type, explanation)
    """
    analysis_prompt = f"""Analyze whether this plan needs revision.

ORIGINAL PLAN:
Goal: {original_plan['goal']}
Steps: {json.dumps(original_plan['steps'], indent=2)}

CURRENT PROGRESS:
Just completed step {current_step}
Result: {step_result}

CONTEXT FROM PREVIOUS STEPS:
{context}

Analyze:
1. Did the step succeed as expected?
2. Did we learn something that changes our approach?
3. Are the remaining steps still optimal?
4. Are there any obstacles or better paths now visible?

Respond in JSON:
{{
    "needs_revision": true/false,
    "trigger": "step_failed" | "new_information" | "obstacle_found" | "better_path_found" | null,
    "explanation": "Why revision is or isn't needed",
    "suggested_changes": "What changes to make, if any"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    analysis = json.loads(text.strip())
    
    trigger = None
    if analysis.get("trigger"):
        try:
            trigger = RevisionTrigger(analysis["trigger"])
        except ValueError:
            pass
    
    return (
        analysis.get("needs_revision", False),
        trigger,
        analysis.get("explanation", "")
    )


def revise_plan(
    original_plan: dict,
    completed_steps: list[dict],
    trigger: RevisionTrigger,
    context: str
) -> dict:
    """
    Create a revised plan based on what we've learned.
    
    Args:
        original_plan: The original plan
        completed_steps: Steps already completed with results
        trigger: Why we're revising
        context: What we've learned
        
    Returns:
        Revised plan dictionary
    """
    revision_prompt = f"""Revise this plan based on new circumstances.

ORIGINAL PLAN:
Goal: {original_plan['goal']}

COMPLETED STEPS:
{json.dumps(completed_steps, indent=2)}

REVISION TRIGGER: {trigger.value}

CONTEXT:
{context}

Create a revised plan that:
1. Preserves the original goal (unless it's now impossible)
2. Builds on completed work (don't repeat what's done)
3. Addresses the reason for revision
4. Remains achievable

Respond in JSON:
{{
    "goal": "Same or adjusted goal",
    "completed_steps": {len(completed_steps)},
    "new_steps": [
        {{"step_number": N, "action": "...", "expected_outcome": "...", "rationale": "Why this step"}}
    ],
    "revision_summary": "What changed and why"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": revision_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    return json.loads(text.strip())


def demonstrate_plan_revision():
    """Demonstrate plan revision in action."""
    
    # Simulate a scenario where revision is needed
    original_plan = {
        "goal": "Find the best restaurant for a team dinner in downtown Seattle",
        "steps": [
            {"step_number": 1, "action": "Search for top-rated restaurants downtown", "expected_outcome": "List of candidates"},
            {"step_number": 2, "action": "Check availability for party of 10 on Friday", "expected_outcome": "Available options"},
            {"step_number": 3, "action": "Compare menus for dietary restrictions", "expected_outcome": "Filtered list"},
            {"step_number": 4, "action": "Make reservation at best option", "expected_outcome": "Confirmed booking"}
        ]
    }
    
    # Simulate completing step 1
    step_1_result = "Found 5 top-rated restaurants. However, discovered that 3 of them are permanently closed due to recent economic conditions."
    
    print("ORIGINAL PLAN:")
    print(f"Goal: {original_plan['goal']}")
    for step in original_plan["steps"]:
        print(f"  {step['step_number']}. {step['action']}")
    
    print(f"\nSTEP 1 RESULT: {step_1_result}")
    
    # Check if revision is needed
    needs_revision, trigger, explanation = detect_revision_needed(
        original_plan=original_plan,
        current_step=1,
        step_result=step_1_result,
        context=""
    )
    
    print(f"\nREVISION NEEDED: {needs_revision}")
    print(f"TRIGGER: {trigger.value if trigger else 'None'}")
    print(f"EXPLANATION: {explanation}")
    
    if needs_revision and trigger:
        completed = [{"step_number": 1, "result": step_1_result}]
        revised = revise_plan(original_plan, completed, trigger, step_1_result)
        
        print("\nREVISED PLAN:")
        print(f"Goal: {revised['goal']}")
        print(f"Revision summary: {revised.get('revision_summary', 'N/A')}")
        print("New steps:")
        for step in revised.get("new_steps", []):
            print(f"  {step['step_number']}. {step['action']}")
            if step.get('rationale'):
                print(f"      Rationale: {step['rationale']}")


if __name__ == "__main__":
    demonstrate_plan_revision()
```

Key principles for plan revision:

1. **Don't revise too eagerly.** Minor hiccups don't require new plans—agents should be resilient to small variations.

2. **Don't revise too reluctantly.** When fundamental assumptions break down, sticking to the original plan wastes effort.

3. **Preserve completed work.** Revision means adjusting the remaining path, not starting over.

4. **Document why you revised.** This creates transparency and helps with debugging.

## Showing Planning Steps for Transparency

Users trust agents more when they can see the agent's thinking. Here's how to make planning visible:

```python
"""
Transparent planning with user-visible reasoning.

Chapter 29: Planning and Reasoning
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class ThinkingStep:
    """A visible step in the agent's reasoning."""
    timestamp: datetime
    phase: str  # "planning", "reasoning", "executing", "revising"
    thought: str
    
    def display(self) -> str:
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
        self.thinking_log: list[ThinkingStep] = []
        self.plan: dict | None = None
    
    def think(self, phase: str, thought: str) -> None:
        """Record and display a thinking step."""
        step = ThinkingStep(
            timestamp=datetime.now(),
            phase=phase,
            thought=thought
        )
        self.thinking_log.append(step)
        print(step.display())
    
    def plan_task(self, task: str) -> dict:
        """Create a plan with visible reasoning."""
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
        lines = reasoning.split('\n')
        for line in lines[:5]:  # Show first few lines
            if line.strip():
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
        """Execute a step with visible reasoning."""
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
        """Run the agent with full transparency."""
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
        """Get the full thinking log as a string."""
        return "\n".join(step.display() for step in self.thinking_log)


if __name__ == "__main__":
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
    
    print("\n" + "=" * 60)
    print("COMPLETE THINKING LOG")
    print("=" * 60)
    print(agent.get_thinking_log())
```

Transparency benefits include:

- **Trust**: Users see that the agent has a coherent approach
- **Debugging**: Developers can trace exactly where things went wrong
- **Interruptibility**: Users can stop the agent if they see it going off track
- **Learning**: Users learn how to break down problems by watching the agent

## When to Use Planning

Planning adds value but also adds latency and cost. Here's when to use it:

**Use planning when:**
- The task requires multiple distinct steps
- Steps depend on results from earlier steps
- There are multiple viable approaches to choose from
- The task is complex enough that getting lost is a risk
- Transparency is important for user trust

**Skip planning when:**
- The task is a simple question/answer
- The task is highly routine with a known procedure
- Speed is critical and the task is well-understood
- The "plan" would be just one step anyway

```python
"""
Deciding when to use planning.

Chapter 29: Planning and Reasoning
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


def should_plan(task: str) -> tuple[bool, str]:
    """
    Decide whether a task needs explicit planning.
    
    Args:
        task: The task description
        
    Returns:
        Tuple of (should_plan, reasoning)
    """
    analysis_prompt = f"""Analyze whether this task needs a multi-step plan or can be answered directly.

Task: {task}

Consider:
1. Is this a simple question with a direct answer?
2. Does it require multiple distinct steps?
3. Do later steps depend on earlier results?
4. Is there meaningful risk of getting off track?

Respond with:
VERDICT: PLAN or DIRECT
REASONING: Brief explanation

Be practical - don't over-plan simple tasks."""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    text = response.content[0].text
    should_use_plan = "VERDICT: PLAN" in text.upper()
    
    # Extract reasoning
    reasoning = text
    if "REASONING:" in text:
        reasoning = text.split("REASONING:")[1].strip()
    
    return should_use_plan, reasoning


def adaptive_agent(task: str) -> str:
    """
    An agent that decides whether to plan based on the task.
    """
    # Decide planning approach
    use_planning, reasoning = should_plan(task)
    
    print(f"Task: {task[:60]}...")
    print(f"Planning decision: {'Use plan' if use_planning else 'Direct response'}")
    print(f"Reasoning: {reasoning[:100]}...")
    print()
    
    if use_planning:
        # Use planning approach (simplified for demonstration)
        plan_prompt = f"""Create and execute a plan for: {task}

First outline 3-5 steps, then execute each one.
Show your work clearly."""
        
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{"role": "user", "content": plan_prompt}]
        )
    else:
        # Direct response
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": task}]
        )
    
    return response.content[0].text


if __name__ == "__main__":
    # Simple task - probably doesn't need planning
    simple_task = "What's the capital of France?"
    
    # Complex task - probably needs planning
    complex_task = """
    I'm moving to a new city for a job. Help me create a timeline 
    for finding an apartment, setting up utilities, changing my 
    address with various services, and getting settled in over the 
    next 6 weeks.
    """
    
    print("=" * 60)
    print("SIMPLE TASK")
    print("=" * 60)
    result = adaptive_agent(simple_task)
    print(result[:200])
    
    print("\n" + "=" * 60)
    print("COMPLEX TASK")
    print("=" * 60)
    result = adaptive_agent(complex_task)
    print(result[:500])
```

## Common Pitfalls

⚠️ **Over-planning simple tasks.** A question like "What year was Python released?" doesn't need a five-step plan. Learn to recognize when direct execution is better.

⚠️ **Plans that are too vague.** "Research the topic" is not an actionable step. Each step should be specific enough that success or failure is clear.

⚠️ **Never revising plans.** If an agent stubbornly follows a plan that's clearly not working, it wastes effort and produces poor results. Build in checkpoints for reflection.

⚠️ **Revising too often.** Constantly changing plans creates confusion and prevents progress. Only revise when there's a genuine reason.

## Practical Exercise

**Task:** Build a "Trip Planning Agent" that creates a day-by-day itinerary for a vacation

**Requirements:**
- Accept destination, duration, and traveler preferences as input
- Create a plan for researching and organizing the trip
- Execute the plan step by step
- Allow for plan revision if constraints are discovered (e.g., attraction closed on certain days)
- Display thinking process transparently
- Produce a final day-by-day itinerary

**Hints:**
- Start with a planning phase that identifies what to research
- Use simulated tools for looking up attractions, restaurants, travel times
- Check after each research step if the plan needs adjustment
- Consider how to handle conflicting information

**Solution:** See `code/exercise.py`

## Key Takeaways

- **Explicit planning** makes agents more reliable by providing a roadmap that prevents getting lost in complex tasks

- **Plan-then-execute** separates thinking from doing—create the plan first, then follow it step by step

- **Chain-of-thought** encourages step-by-step reasoning within a single response, improving accuracy for logical and analytical tasks

- **Plans should be revisable**—check after each step whether the remaining plan still makes sense given what you've learned

- **Transparency builds trust**—showing users the agent's thinking helps them understand and verify the agent's approach

- **Not every task needs planning**—match the complexity of your approach to the complexity of the task

## What's Next

Planning helps agents think before they act, but what happens when things go wrong during execution? In Chapter 30, we'll tackle error handling and recovery—how to build agents that gracefully handle failures, retry when appropriate, and recover from unexpected situations. Robust error handling is what separates prototype agents from production-ready systems.
