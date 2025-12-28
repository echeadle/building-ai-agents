"""
Basic orchestrator implementation for creating task plans.

Chapter 23: Orchestrator-Workers - Implementation

This module implements the orchestrator component that analyzes
complex queries and breaks them into manageable subtasks.
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Orchestrator System Prompt
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert task orchestrator. Your role is to analyze complex queries and break them down into focused, independent subtasks that can be researched or analyzed separately.

## Your Responsibilities

1. **Analyze the Query**: Understand what the user is truly asking for
2. **Identify Dimensions**: Find the distinct aspects, perspectives, or components
3. **Create Subtasks**: Break the query into 3-6 focused subtasks
4. **Ensure Coverage**: Make sure subtasks collectively address the full query
5. **Maintain Independence**: Each subtask should be completable on its own

## Subtask Types

- **research**: Gather factual information on a specific topic
- **analysis**: Evaluate implications, trade-offs, and significance
- **comparison**: Compare different options, approaches, or perspectives

## Subtask Guidelines

- Each subtask should be specific and focused
- Subtasks should not overlap significantly
- Include both research tasks (gather information) and analysis tasks (evaluate/compare)
- Consider different perspectives: technical, economic, social, practical
- Aim for 3-6 subtasks (fewer for simple queries, more for complex ones)

## Output Format

You must respond with a JSON object in this exact format:
{
    "query_analysis": "Brief analysis of what the user is asking",
    "subtasks": [
        {
            "id": "task_1",
            "type": "research|analysis|comparison",
            "title": "Short descriptive title",
            "description": "Detailed description of what this subtask should accomplish",
            "focus_areas": ["specific", "areas", "to", "cover"]
        }
    ],
    "synthesis_guidance": "How the subtask results should be combined"
}

Respond ONLY with the JSON object, no other text."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Subtask:
    """
    Represents a subtask created by the orchestrator.
    
    Attributes:
        id: Unique identifier for the subtask
        type: Type of task (research, analysis, comparison)
        title: Short descriptive title
        description: Detailed description of what to accomplish
        focus_areas: Specific areas to focus on
        result: Result from worker execution (filled later)
        status: Current status (pending, in_progress, completed, failed)
        started_at: When execution started
        completed_at: When execution completed
    """
    id: str
    type: str
    title: str
    description: str
    focus_areas: list[str]
    result: Optional[str] = None
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class TaskPlan:
    """
    The orchestrator's plan for handling a complex query.
    
    Attributes:
        original_query: The user's original question
        query_analysis: Orchestrator's understanding of the query
        subtasks: List of subtasks to execute
        synthesis_guidance: How to combine results
        created_at: When the plan was created
    """
    original_query: str
    query_analysis: str
    subtasks: list[Subtask]
    synthesis_guidance: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_pending_subtasks(self) -> list[Subtask]:
        """Return subtasks that haven't been started."""
        return [st for st in self.subtasks if st.status == "pending"]
    
    def get_completed_subtasks(self) -> list[Subtask]:
        """Return subtasks that completed successfully."""
        return [st for st in self.subtasks if st.status == "completed"]
    
    def get_failed_subtasks(self) -> list[Subtask]:
        """Return subtasks that failed."""
        return [st for st in self.subtasks if st.status == "failed"]
    
    def completion_rate(self) -> float:
        """Calculate the completion rate of subtasks."""
        if not self.subtasks:
            return 0.0
        completed = len(self.get_completed_subtasks())
        return completed / len(self.subtasks)


# =============================================================================
# Orchestrator Class
# =============================================================================

class Orchestrator:
    """
    Orchestrates complex tasks by breaking them into subtasks.
    
    The orchestrator analyzes a complex query, identifies its components,
    and creates a plan of subtasks that can be executed independently.
    
    Example:
        orchestrator = Orchestrator()
        plan = orchestrator.create_plan("What are the impacts of AI on education?")
        for subtask in plan.subtasks:
            print(f"- {subtask.title}")
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_subtasks: int = 6
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model: Claude model to use for task analysis
            max_subtasks: Maximum number of subtasks to create
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_subtasks = max_subtasks
    
    def create_plan(self, query: str) -> TaskPlan:
        """
        Analyze a complex query and create a plan of subtasks.
        
        Args:
            query: The complex query to break down
            
        Returns:
            TaskPlan containing subtasks to be executed
            
        Raises:
            ValueError: If the orchestrator returns invalid JSON
        """
        # Ask the orchestrator to decompose the task
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Break down this query into subtasks:\n\n{query}"
                }
            ]
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        try:
            plan_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it contains extra text
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    plan_data = json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Orchestrator returned invalid JSON: {e}")
            else:
                raise ValueError(f"Could not parse orchestrator response as JSON")
        
        # Convert to Subtask objects
        subtasks = []
        for task_data in plan_data.get("subtasks", [])[:self.max_subtasks]:
            subtask = Subtask(
                id=task_data.get("id", f"task_{len(subtasks)+1}"),
                type=task_data.get("type", "research"),
                title=task_data.get("title", "Untitled"),
                description=task_data.get("description", ""),
                focus_areas=task_data.get("focus_areas", [])
            )
            subtasks.append(subtask)
        
        return TaskPlan(
            original_query=query,
            query_analysis=plan_data.get("query_analysis", ""),
            subtasks=subtasks,
            synthesis_guidance=plan_data.get("synthesis_guidance", "")
        )
    
    def print_plan(self, plan: TaskPlan) -> None:
        """
        Pretty-print a task plan.
        
        Args:
            plan: The TaskPlan to display
        """
        print(f"\n{'='*60}")
        print("TASK PLAN")
        print(f"{'='*60}")
        print(f"\nQuery: {plan.original_query}")
        print(f"\nAnalysis: {plan.query_analysis}")
        print(f"\nSubtasks ({len(plan.subtasks)}):")
        
        for i, subtask in enumerate(plan.subtasks, 1):
            print(f"\n  {i}. [{subtask.type.upper()}] {subtask.title}")
            print(f"     ID: {subtask.id}")
            print(f"     Description: {subtask.description}")
            if subtask.focus_areas:
                print(f"     Focus Areas: {', '.join(subtask.focus_areas)}")
        
        print(f"\nSynthesis Guidance: {plan.synthesis_guidance}")
        print(f"\n{'='*60}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Test queries
    test_queries = [
        "What are the environmental and economic impacts of electric vehicle adoption in urban areas?",
        "How is artificial intelligence transforming the healthcare industry?",
        "What are the long-term effects of remote work on employee productivity and wellbeing?",
    ]
    
    # Process first query as example
    query = test_queries[0]
    
    print(f"Testing orchestrator with query:")
    print(f'"{query}"')
    print("\nCreating task plan...")
    
    plan = orchestrator.create_plan(query)
    orchestrator.print_plan(plan)
    
    # Show raw subtask data
    print("\n\nRaw subtask data:")
    for subtask in plan.subtasks:
        print(f"\n{subtask}")
