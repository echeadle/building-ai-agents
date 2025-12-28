"""
A complete planning agent with tool use.

This example demonstrates a full planning agent that:
- Creates structured plans using dataclasses
- Executes steps with access to tools
- Tracks progress through the plan
- Can revise plans when needed
- Synthesizes final results

Chapter 29: Planning and Reasoning
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, field
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
        """Convert plan to dictionary format."""
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
        """Get a summary of plan progress."""
        completed = sum(1 for s in self.steps if s.status == "completed")
        total = len(self.steps)
        return f"{completed}/{total} steps completed"
    
    def get_next_step(self) -> PlanStep | None:
        """Get the next pending step, or None if all complete."""
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
               f"Found relevant information about {query}. Key points:\n" + \
               f"- Relevant fact 1 about {query}\n" + \
               f"- Relevant fact 2 about {query}\n" + \
               f"- Relevant fact 3 about {query}"
    
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
        """
        Initialize the planning agent.
        
        Args:
            verbose: Whether to print progress messages
        """
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
        self.log(f"Starting task: {task[:60]}...", "info")
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
