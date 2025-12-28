"""
Task delegation and result collection system.

Chapter 23: Orchestrator-Workers - Implementation

This module manages the delegation of subtasks to workers
and collects their results.
"""

import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Import from our modules (adjust path as needed for your project)
# For this example, we define everything inline
import anthropic
import time


# =============================================================================
# Data Classes (shared with other modules)
# =============================================================================

@dataclass
class Subtask:
    """Represents a subtask created by the orchestrator."""
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
    """The orchestrator's plan for handling a complex query."""
    original_query: str
    query_analysis: str
    subtasks: list[Subtask]
    synthesis_guidance: str


@dataclass
class WorkerResult:
    """Result from a worker executing a subtask."""
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


# =============================================================================
# Worker Class (simplified for this module)
# =============================================================================

WORKER_PROMPTS = {
    "research": """You are a thorough research assistant. Provide comprehensive, factual information on a specific topic.
Focus on key findings, supporting details, examples, and important caveats.""",
    
    "analysis": """You are an analytical expert. Analyze the topic, evaluating implications, trade-offs, and significance.
Provide balanced analysis with pros/cons and meaningful conclusions.""",
    
    "comparison": """You are a comparison specialist. Compare different options or perspectives fairly.
Use clear criteria, evaluate each option, and provide contextual recommendations."""
}


class Worker:
    """Executes individual subtasks."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def execute(
        self,
        subtask_id: str,
        subtask_type: str,
        title: str,
        description: str,
        focus_areas: list[str],
        context: str = ""
    ) -> WorkerResult:
        """Execute a single subtask."""
        start_time = time.time()
        
        system_prompt = WORKER_PROMPTS.get(subtask_type, WORKER_PROMPTS["research"])
        
        task_message = f"## Task: {title}\n\n{description}\n\n"
        task_message += "## Focus Areas\n"
        task_message += "\n".join(f"- {area}" for area in focus_areas)
        
        if context:
            task_message += f"\n\n## Additional Context\n{context}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": task_message}]
            )
            
            execution_time = time.time() - start_time
            
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content=response.content[0].text,
                success=True,
                execution_time=execution_time
            )
            
        except anthropic.APIError as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=str(e),
                execution_time=execution_time
            )


# =============================================================================
# Task Delegator Class
# =============================================================================

class TaskDelegator:
    """
    Manages the delegation of subtasks to workers and collection of results.
    
    The TaskDelegator coordinates the execution of a TaskPlan by:
    1. Iterating through subtasks in the plan
    2. Dispatching each subtask to a worker
    3. Tracking status and collecting results
    4. Providing progress feedback
    
    Example:
        worker = Worker()
        delegator = TaskDelegator(worker, verbose=True)
        results = delegator.delegate_all(plan)
    """
    
    def __init__(
        self,
        worker: Worker,
        verbose: bool = True
    ):
        """
        Initialize the delegator.
        
        Args:
            worker: Worker instance to execute subtasks
            verbose: Whether to print progress updates
        """
        self.worker = worker
        self.verbose = verbose
        self.results: list[WorkerResult] = []
        self._start_time: Optional[float] = None
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def delegate_single(
        self,
        subtask: Subtask,
        context: str = ""
    ) -> WorkerResult:
        """
        Delegate a single subtask to a worker.
        
        Args:
            subtask: The subtask to execute
            context: Optional additional context
            
        Returns:
            WorkerResult from the execution
        """
        # Update status
        subtask.status = "in_progress"
        subtask.started_at = datetime.now()
        
        # Execute the subtask
        result = self.worker.execute(
            subtask_id=subtask.id,
            subtask_type=subtask.type,
            title=subtask.title,
            description=subtask.description,
            focus_areas=subtask.focus_areas,
            context=context
        )
        
        # Update subtask with result
        subtask.completed_at = datetime.now()
        
        if result.success:
            subtask.result = result.content
            subtask.status = "completed"
        else:
            subtask.status = "failed"
        
        return result
    
    def delegate_all(
        self,
        plan: TaskPlan,
        context: str = ""
    ) -> list[WorkerResult]:
        """
        Delegate all subtasks in a plan to workers.
        
        Args:
            plan: The TaskPlan containing subtasks
            context: Optional additional context for all workers
            
        Returns:
            List of WorkerResults
        """
        self.results = []
        self._start_time = time.time()
        total = len(plan.subtasks)
        
        self._log(f"\n{'='*50}")
        self._log(f"EXECUTING {total} SUBTASKS")
        self._log(f"{'='*50}")
        
        for i, subtask in enumerate(plan.subtasks, 1):
            self._log(f"\n[{i}/{total}] {subtask.title}")
            self._log(f"        Type: {subtask.type}")
            
            result = self.delegate_single(subtask, context)
            self.results.append(result)
            
            if result.success:
                self._log(f"        ✓ Completed in {result.execution_time:.1f}s")
            else:
                self._log(f"        ✗ Failed: {result.error}")
        
        # Summary
        total_time = time.time() - self._start_time
        successful = len(self.get_successful_results())
        failed = len(self.get_failed_results())
        
        self._log(f"\n{'='*50}")
        self._log(f"EXECUTION COMPLETE")
        self._log(f"{'='*50}")
        self._log(f"  Successful: {successful}/{total}")
        self._log(f"  Failed: {failed}/{total}")
        self._log(f"  Total time: {total_time:.1f}s")
        
        return self.results
    
    def get_successful_results(self) -> list[WorkerResult]:
        """Return only successful results."""
        return [r for r in self.results if r.success]
    
    def get_failed_results(self) -> list[WorkerResult]:
        """Return only failed results."""
        return [r for r in self.results if not r.success]
    
    def get_result_by_id(self, subtask_id: str) -> Optional[WorkerResult]:
        """Get a specific result by subtask ID."""
        for result in self.results:
            if result.subtask_id == subtask_id:
                return result
        return None
    
    def get_results_summary(self) -> dict:
        """Get a summary of execution results."""
        total_execution_time = sum(
            r.execution_time or 0 for r in self.results
        )
        
        return {
            "total_subtasks": len(self.results),
            "successful": len(self.get_successful_results()),
            "failed": len(self.get_failed_results()),
            "total_execution_time": total_execution_time,
            "success_rate": len(self.get_successful_results()) / len(self.results) if self.results else 0
        }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create a test plan
    plan = TaskPlan(
        original_query="What are the pros and cons of remote work?",
        query_analysis="User wants to understand benefits and drawbacks of remote work",
        subtasks=[
            Subtask(
                id="task_1",
                type="research",
                title="Benefits of Remote Work for Employees",
                description="Research the key benefits of remote work from the employee perspective",
                focus_areas=[
                    "Flexibility and autonomy",
                    "Work-life balance improvements",
                    "Cost savings (commute, meals, etc.)",
                    "Productivity potential"
                ]
            ),
            Subtask(
                id="task_2",
                type="research",
                title="Challenges of Remote Work",
                description="Research the challenges and drawbacks of remote work",
                focus_areas=[
                    "Social isolation and loneliness",
                    "Communication difficulties",
                    "Work-life boundary blur",
                    "Career advancement concerns"
                ]
            ),
            Subtask(
                id="task_3",
                type="analysis",
                title="Remote Work Trade-offs Analysis",
                description="Analyze the key trade-offs when choosing between remote and office work",
                focus_areas=[
                    "Productivity vs collaboration",
                    "Flexibility vs structure",
                    "Individual vs team dynamics",
                    "Short-term vs long-term impacts"
                ]
            )
        ],
        synthesis_guidance="Combine into a balanced view showing both perspectives with actionable insights"
    )
    
    # Create worker and delegator
    worker = Worker()
    delegator = TaskDelegator(worker, verbose=True)
    
    # Execute all subtasks
    print(f"Query: {plan.original_query}\n")
    results = delegator.delegate_all(plan)
    
    # Show detailed results
    print("\n" + "=" * 50)
    print("DETAILED RESULTS")
    print("=" * 50)
    
    for result in delegator.get_successful_results():
        print(f"\n### {result.subtask_title}")
        print("-" * 40)
        # Show first 500 chars
        content_preview = result.content[:500]
        if len(result.content) > 500:
            content_preview += "..."
        print(content_preview)
    
    # Show summary
    summary = delegator.get_results_summary()
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
