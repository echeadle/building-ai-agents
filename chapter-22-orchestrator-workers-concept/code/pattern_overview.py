"""
Orchestrator-Workers Pattern Overview

This file demonstrates the conceptual structure of the orchestrator-workers
pattern. It shows the interfaces and flow without making actual API calls.

Chapter 22: Orchestrator-Workers - Concept and Design
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# Data Structures
# =============================================================================

class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(Enum):
    """Task priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """
    A subtask identified by the orchestrator.
    
    Attributes:
        id: Unique identifier for the task
        worker_type: Which type of worker should handle this
        description: What the worker should do
        dependencies: Task IDs that must complete first
        priority: How important this task is
        status: Current status of the task
        result: Output from the worker (when completed)
    """
    id: str
    worker_type: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None


@dataclass
class WorkerInput:
    """
    Standard input format for all workers.
    
    Using a consistent input format makes workers interchangeable
    and simplifies the orchestrator's delegation logic.
    """
    task_id: str
    task_description: str
    context: dict = field(default_factory=dict)
    constraints: Optional[dict] = None


@dataclass
class WorkerOutput:
    """
    Standard output format for all workers.
    
    Consistent output format enables reliable result aggregation
    and error handling in the orchestrator.
    """
    task_id: str
    status: str  # "success", "partial", "failed"
    result: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    """
    The orchestrator's plan for handling a request.
    
    Contains analysis of the request and the list of tasks
    needed to fulfill it.
    """
    original_request: str
    analysis: str
    tasks: list[Task]


# =============================================================================
# Abstract Interfaces
# =============================================================================

class BaseWorker:
    """
    Abstract base class for all workers.
    
    Workers are specialists that handle specific types of tasks.
    Each worker should:
    - Focus on a single responsibility
    - Be self-contained (no external state)
    - Return results in a standard format
    - Handle errors gracefully
    """
    
    def __init__(self, name: str, capabilities: str):
        """
        Initialize a worker.
        
        Args:
            name: The worker's identifier
            capabilities: Description of what this worker can do
        """
        self.name = name
        self.capabilities = capabilities
    
    def execute(self, input: WorkerInput) -> WorkerOutput:
        """
        Execute a task and return results.
        
        This is the main entry point for workers. Subclasses
        should override _do_work() rather than this method.
        
        Args:
            input: The task input with description and context
            
        Returns:
            WorkerOutput with results or error information
        """
        try:
            result = self._do_work(input)
            return WorkerOutput(
                task_id=input.task_id,
                status="success",
                result=result,
                metadata={"worker": self.name}
            )
        except Exception as e:
            return WorkerOutput(
                task_id=input.task_id,
                status="failed",
                errors=[str(e)],
                metadata={"worker": self.name}
            )
    
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Perform the actual work. Override in subclasses.
        
        Args:
            input: The task input
            
        Returns:
            Dictionary containing the task results
        """
        raise NotImplementedError("Subclasses must implement _do_work()")


class BaseOrchestrator:
    """
    Abstract base class for orchestrators.
    
    The orchestrator is responsible for:
    1. Planning - Analyzing requests and creating task plans
    2. Delegating - Assigning tasks to appropriate workers
    3. Synthesizing - Combining worker results into final output
    """
    
    def __init__(self, workers: dict[str, BaseWorker]):
        """
        Initialize the orchestrator with available workers.
        
        Args:
            workers: Dictionary mapping worker types to worker instances
        """
        self.workers = workers
    
    def handle_request(self, request: str) -> str:
        """
        Process a request through the full orchestration cycle.
        
        This is the main entry point. It:
        1. Creates a plan for the request
        2. Executes all tasks in the plan
        3. Synthesizes results into final output
        
        Args:
            request: The user's request
            
        Returns:
            The final synthesized response
        """
        # Step 1: Planning
        plan = self.plan(request)
        
        # Step 2: Delegation and Execution
        results = self.execute_plan(plan)
        
        # Step 3: Synthesis
        response = self.synthesize(request, results)
        
        return response
    
    def plan(self, request: str) -> TaskPlan:
        """
        Analyze the request and create a task plan.
        
        Override in subclasses to implement planning logic.
        
        Args:
            request: The user's request
            
        Returns:
            A TaskPlan with the identified subtasks
        """
        raise NotImplementedError("Subclasses must implement plan()")
    
    def execute_plan(self, plan: TaskPlan) -> dict[str, WorkerOutput]:
        """
        Execute all tasks in the plan.
        
        Handles task dependencies and worker delegation.
        
        Args:
            plan: The task plan to execute
            
        Returns:
            Dictionary mapping task IDs to their outputs
        """
        results: dict[str, WorkerOutput] = {}
        
        # Sort tasks by dependencies (topological sort)
        ordered_tasks = self._order_by_dependencies(plan.tasks)
        
        for task in ordered_tasks:
            # Gather dependency results as context
            context = {
                dep_id: results[dep_id].result
                for dep_id in task.dependencies
                if dep_id in results
            }
            
            # Get the appropriate worker
            worker = self.workers.get(task.worker_type)
            if not worker:
                results[task.id] = WorkerOutput(
                    task_id=task.id,
                    status="failed",
                    errors=[f"No worker found for type: {task.worker_type}"]
                )
                continue
            
            # Create input and execute
            worker_input = WorkerInput(
                task_id=task.id,
                task_description=task.description,
                context=context
            )
            
            results[task.id] = worker.execute(worker_input)
        
        return results
    
    def synthesize(
        self, 
        original_request: str, 
        results: dict[str, WorkerOutput]
    ) -> str:
        """
        Combine worker results into a final response.
        
        Override in subclasses to implement synthesis logic.
        
        Args:
            original_request: The original user request
            results: All worker outputs
            
        Returns:
            The final synthesized response
        """
        raise NotImplementedError("Subclasses must implement synthesize()")
    
    def _order_by_dependencies(self, tasks: list[Task]) -> list[Task]:
        """
        Sort tasks so dependencies come before dependents.
        
        Uses a simple topological sort algorithm.
        
        Args:
            tasks: List of tasks to sort
            
        Returns:
            Tasks in dependency order
        """
        # Build a map of task id to task
        task_map = {task.id: task for task in tasks}
        
        # Track visited and ordered tasks
        visited = set()
        ordered = []
        
        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    visit(dep_id)
                ordered.append(task)
        
        for task in tasks:
            visit(task.id)
        
        return ordered


# =============================================================================
# Demonstration
# =============================================================================

def demonstrate_pattern():
    """
    Demonstrates the orchestrator-workers pattern structure.
    
    This creates example data structures to show how the pattern works.
    The actual implementation with LLM calls is in Chapter 23.
    """
    print("=" * 60)
    print("Orchestrator-Workers Pattern Overview")
    print("=" * 60)
    
    # Example: Creating a task plan
    print("\n1. PLANNING PHASE")
    print("-" * 40)
    
    example_plan = TaskPlan(
        original_request="Analyze this codebase for potential issues",
        analysis="This request requires multiple types of analysis to be comprehensive",
        tasks=[
            Task(
                id="task_1",
                worker_type="security",
                description="Scan for security vulnerabilities",
                priority=Priority.HIGH
            ),
            Task(
                id="task_2", 
                worker_type="performance",
                description="Identify performance bottlenecks",
                priority=Priority.MEDIUM
            ),
            Task(
                id="task_3",
                worker_type="style",
                description="Check code style and formatting",
                priority=Priority.LOW
            ),
            Task(
                id="task_4",
                worker_type="synthesis",
                description="Combine all findings into comprehensive report",
                dependencies=["task_1", "task_2", "task_3"],
                priority=Priority.HIGH
            )
        ]
    )
    
    print(f"Request: {example_plan.original_request}")
    print(f"Analysis: {example_plan.analysis}")
    print(f"\nTasks identified:")
    for task in example_plan.tasks:
        deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"  - [{task.priority.value}] {task.id}: {task.description}{deps}")
    
    # Example: Worker input/output
    print("\n2. DELEGATION PHASE")
    print("-" * 40)
    
    example_input = WorkerInput(
        task_id="task_1",
        task_description="Scan for security vulnerabilities",
        context={}
    )
    
    print(f"Worker receives input:")
    print(f"  Task ID: {example_input.task_id}")
    print(f"  Description: {example_input.task_description}")
    
    example_output = WorkerOutput(
        task_id="task_1",
        status="success",
        result={
            "vulnerabilities": [
                {"type": "SQL Injection", "location": "line 45", "severity": "high"},
                {"type": "XSS", "location": "line 123", "severity": "medium"}
            ],
            "summary": "Found 2 potential vulnerabilities"
        },
        metadata={"worker": "security", "scan_time": "2.3s"}
    )
    
    print(f"\nWorker returns output:")
    print(f"  Status: {example_output.status}")
    print(f"  Result: {example_output.result['summary']}")
    
    # Example: Synthesis phase
    print("\n3. SYNTHESIS PHASE")
    print("-" * 40)
    
    print("Orchestrator combines all worker results:")
    print("  - Security: 2 vulnerabilities found")
    print("  - Performance: 3 bottlenecks identified")
    print("  - Style: 15 formatting issues")
    print("\nFinal response integrates all findings into a coherent report.")
    
    print("\n" + "=" * 60)
    print("See Chapter 23 for the full implementation!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_pattern()
