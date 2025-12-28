"""
Base classes and interfaces for agentic workflow patterns.

Chapter 15: Introduction to Agentic Workflows

This module establishes the common interfaces that all workflow patterns
will implement. It provides a consistent structure for building, executing,
and managing workflows throughout Part 3 of this book.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class StepResult:
    """Result from a single step in a workflow."""
    
    step_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.step_name}: {self.duration_ms:.0f}ms"


@dataclass
class WorkflowResult:
    """
    Standard result container for all workflow patterns.
    
    Every workflow execution returns this structure, making it easy
    to handle results consistently regardless of the pattern used.
    """
    
    success: bool
    output: Any
    steps: list[StepResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def total_duration_ms(self) -> float:
        """Total execution time across all steps."""
        return sum(step.duration_ms for step in self.steps)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used across all steps."""
        return sum(step.tokens_used for step in self.steps)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the workflow execution."""
        lines = [
            f"Workflow {'succeeded' if self.success else 'failed'}",
            f"Steps: {len(self.steps)}",
            f"Duration: {self.total_duration_ms:.0f}ms",
            f"Tokens: {self.total_tokens}",
        ]
        
        if self.steps:
            lines.append("\nStep details:")
            for step in self.steps:
                lines.append(f"  {step}")
        
        if self.error:
            lines.append(f"\nError: {self.error}")
        
        return "\n".join(lines)


class WorkflowPattern(ABC):
    """
    Abstract base class for all workflow patterns.
    
    Each pattern (Chaining, Routing, Parallelization, etc.) inherits from
    this class and implements the execute() method according to its logic.
    """
    
    def __init__(self, name: str = "Unnamed Workflow"):
        """
        Initialize the workflow pattern.
        
        Args:
            name: A descriptive name for this workflow instance
        """
        self.name = name
        self._execution_count = 0
        self._created_at = datetime.now()
    
    @abstractmethod
    def execute(self, input_data: Any) -> WorkflowResult:
        """
        Execute the workflow and return results.
        
        This is the main entry point for running a workflow. Each pattern
        implements this differently based on its logic.
        
        Args:
            input_data: The input to process (type depends on the workflow)
            
        Returns:
            WorkflowResult containing success status, output, and metadata
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()


# Preview of pattern classes we'll implement in upcoming chapters
# These are just signatures - full implementations come later

class PromptChain(WorkflowPattern):
    """
    Chain multiple LLM calls in sequence.
    
    Implementation: Chapter 17
    """
    
    def execute(self, input_data: Any) -> WorkflowResult:
        raise NotImplementedError("Full implementation in Chapter 17")


class Router(WorkflowPattern):
    """
    Classify input and route to specialized handlers.
    
    Implementation: Chapter 19
    """
    
    def execute(self, input_data: Any) -> WorkflowResult:
        raise NotImplementedError("Full implementation in Chapter 19")


class ParallelWorkflow(WorkflowPattern):
    """
    Run multiple LLM calls in parallel.
    
    Implementation: Chapter 21
    """
    
    def execute(self, input_data: Any) -> WorkflowResult:
        raise NotImplementedError("Full implementation in Chapter 21")


class Orchestrator(WorkflowPattern):
    """
    Dynamically decompose tasks and delegate to workers.
    
    Implementation: Chapter 23
    """
    
    def execute(self, input_data: Any) -> WorkflowResult:
        raise NotImplementedError("Full implementation in Chapter 23")


class EvaluatorOptimizer(WorkflowPattern):
    """
    Iteratively generate and refine output.
    
    Implementation: Chapter 25
    """
    
    def execute(self, input_data: Any) -> WorkflowResult:
        raise NotImplementedError("Full implementation in Chapter 25")


if __name__ == "__main__":
    # Demonstrate the result structures
    
    # Example of a successful step result
    step1 = StepResult(
        step_name="Generate Draft",
        success=True,
        output="This is the generated draft content...",
        duration_ms=1250.5,
        tokens_used=500
    )
    
    step2 = StepResult(
        step_name="Review Draft",
        success=True,
        output="Draft approved with minor suggestions.",
        duration_ms=850.3,
        tokens_used=350
    )
    
    # Example of a complete workflow result
    result = WorkflowResult(
        success=True,
        output="Final processed content",
        steps=[step1, step2],
        metadata={
            "pattern": "PromptChain",
            "model": "claude-sonnet-4-20250514"
        }
    )
    
    print("Example WorkflowResult:")
    print("-" * 40)
    print(result.summary())
    print("-" * 40)
    print(f"\nRaw output: {result.output}")
