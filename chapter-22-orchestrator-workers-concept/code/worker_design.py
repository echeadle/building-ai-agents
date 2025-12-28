"""
Worker Design Examples

This file demonstrates well-designed worker interfaces following
the four key principles:
1. Single Responsibility
2. Self-Contained Execution
3. Clear Interfaces
4. Graceful Failure

Chapter 22: Orchestrator-Workers - Concept and Design
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
from abc import ABC, abstractmethod


# =============================================================================
# Standard Interfaces (Clear Interfaces Principle)
# =============================================================================

@dataclass
class WorkerInput:
    """
    Standard input format for all workers.
    
    Benefits of standardization:
    - Workers are interchangeable
    - Orchestrator logic is simplified
    - Testing is consistent
    """
    task_id: str
    task_description: str
    context: dict = field(default_factory=dict)
    constraints: Optional[dict] = None


@dataclass
class WorkerOutput:
    """
    Standard output format for all workers.
    
    Benefits of standardization:
    - Reliable result aggregation
    - Consistent error handling
    - Easy logging and debugging
    """
    task_id: str
    status: str  # "success", "partial", "failed"
    result: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Worker(Protocol):
    """
    Protocol defining the worker interface.
    
    Any class implementing this protocol can be used as a worker.
    This enables dependency injection and easy testing.
    """
    name: str
    capabilities: str
    
    def execute(self, input: WorkerInput) -> WorkerOutput:
        """Execute a task and return results."""
        ...


# =============================================================================
# Base Worker Class (Graceful Failure Principle)
# =============================================================================

class BaseWorker(ABC):
    """
    Abstract base class that implements graceful failure.
    
    Subclasses only need to implement _do_work(), and get
    automatic error handling and consistent output formatting.
    """
    
    def __init__(self, name: str, capabilities: str):
        self.name = name
        self.capabilities = capabilities
    
    def execute(self, input: WorkerInput) -> WorkerOutput:
        """
        Execute with automatic error handling.
        
        This wrapper ensures that:
        - Exceptions are caught and reported properly
        - Output format is always consistent
        - Partial results can be preserved on failure
        """
        try:
            result = self._do_work(input)
            return WorkerOutput(
                task_id=input.task_id,
                status="success",
                result=result,
                metadata={
                    "worker": self.name,
                    "task_description": input.task_description
                }
            )
        except PartialResultError as e:
            # Handle partial success (some results available)
            return WorkerOutput(
                task_id=input.task_id,
                status="partial",
                result=e.partial_result,
                errors=[str(e)],
                metadata={"worker": self.name}
            )
        except Exception as e:
            # Handle complete failure
            return WorkerOutput(
                task_id=input.task_id,
                status="failed",
                errors=[f"{type(e).__name__}: {str(e)}"],
                metadata={"worker": self.name}
            )
    
    @abstractmethod
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Perform the actual work. Override in subclasses.
        
        Raises:
            PartialResultError: If partial results are available
            Exception: For complete failures
        """
        pass


class PartialResultError(Exception):
    """
    Exception for when partial results are available.
    
    This allows workers to return whatever they completed
    even if the full task couldn't be finished.
    """
    def __init__(self, message: str, partial_result: dict):
        super().__init__(message)
        self.partial_result = partial_result


# =============================================================================
# Example: Well-Designed Workers (Single Responsibility Principle)
# =============================================================================

class SecurityAnalyzer(BaseWorker):
    """
    Worker specialized in security vulnerability detection.
    
    Single Responsibility: ONLY handles security analysis.
    Does not check performance, style, or other concerns.
    """
    
    def __init__(self):
        super().__init__(
            name="security_analyzer",
            capabilities=(
                "Analyzes code for security vulnerabilities including: "
                "SQL injection, XSS, CSRF, insecure authentication, "
                "sensitive data exposure, and known CVEs."
            )
        )
        # In a real implementation, this would be configured
        self.vulnerability_patterns = [
            "sql_injection",
            "xss",
            "csrf",
            "auth_bypass",
            "data_exposure"
        ]
    
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Scan for security vulnerabilities.
        
        In a real implementation, this would:
        1. Parse the code
        2. Run pattern matching
        3. Check against CVE database
        4. Apply security rules
        """
        # This is a conceptual example - real implementation in Chapter 23
        return {
            "vulnerabilities": [],
            "severity_summary": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "scan_coverage": "100%",
            "recommendations": []
        }


class PerformanceAnalyzer(BaseWorker):
    """
    Worker specialized in performance analysis.
    
    Single Responsibility: ONLY handles performance concerns.
    Does not check security, style, or other issues.
    """
    
    def __init__(self):
        super().__init__(
            name="performance_analyzer",
            capabilities=(
                "Analyzes code for performance issues including: "
                "inefficient algorithms, memory leaks, N+1 queries, "
                "unnecessary computations, and resource bottlenecks."
            )
        )
    
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Analyze for performance issues.
        
        In a real implementation, this would:
        1. Identify algorithmic complexity
        2. Detect memory patterns
        3. Find database query issues
        4. Profile hot paths
        """
        return {
            "issues": [],
            "complexity_analysis": {},
            "memory_concerns": [],
            "optimization_suggestions": []
        }


class StyleReviewer(BaseWorker):
    """
    Worker specialized in code style review.
    
    Single Responsibility: ONLY handles style and formatting.
    Does not check security, performance, or logic.
    """
    
    def __init__(self):
        super().__init__(
            name="style_reviewer",
            capabilities=(
                "Reviews code for style and formatting including: "
                "PEP 8 compliance, naming conventions, documentation, "
                "code organization, and readability."
            )
        )
    
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Review code style.
        
        In a real implementation, this would:
        1. Check against style guide
        2. Verify naming conventions
        3. Analyze documentation coverage
        4. Assess readability metrics
        """
        return {
            "style_issues": [],
            "documentation_coverage": "0%",
            "naming_violations": [],
            "readability_score": 0
        }


# =============================================================================
# Example: Self-Contained Worker (Self-Contained Execution Principle)
# =============================================================================

class ResearchWorker(BaseWorker):
    """
    Worker that researches topics and summarizes findings.
    
    Self-Contained: Receives all necessary input through parameters.
    Does not rely on external state or side effects.
    """
    
    def __init__(self, search_tool=None):
        super().__init__(
            name="researcher",
            capabilities=(
                "Researches topics using web search and summarizes findings. "
                "Can find current information, compare sources, and extract "
                "key facts from multiple documents."
            )
        )
        # Tools are injected, not accessed globally
        self.search_tool = search_tool
    
    def _do_work(self, input: WorkerInput) -> dict:
        """
        Research a topic.
        
        Note how this method:
        - Gets everything it needs from `input`
        - Returns a complete result
        - Doesn't modify any external state
        - Doesn't depend on previous calls
        """
        query = input.task_description
        context = input.context
        
        # All needed information comes from parameters
        # Results are returned, not stored in self
        
        return {
            "query": query,
            "findings": [],
            "sources": [],
            "summary": "",
            "confidence": 0.0
        }


# =============================================================================
# Anti-Patterns (What NOT to Do)
# =============================================================================

class BadWorkerExample:
    """
    Example of a POORLY designed worker. DO NOT USE.
    
    Problems:
    1. No single responsibility (does too many things)
    2. Not self-contained (relies on external state)
    3. Inconsistent interface (no standard input/output)
    4. No error handling (crashes on failures)
    """
    
    # BAD: Global state
    current_task = None
    all_results = []
    
    def analyze(self, code):
        """
        BAD: This method has multiple problems:
        - Does security AND performance AND style (no focus)
        - Stores results in self.all_results (side effect)
        - No error handling (will crash on failure)
        - Inconsistent return type (sometimes None, sometimes dict)
        """
        # BAD: Relies on external state
        if not self.current_task:
            return None
        
        # BAD: Does too many things
        security_issues = self._check_security(code)
        perf_issues = self._check_performance(code)
        style_issues = self._check_style(code)
        
        result = {
            "security": security_issues,
            "performance": perf_issues,
            "style": style_issues
        }
        
        # BAD: Side effect instead of return
        self.all_results.append(result)
        
        # BAD: Inconsistent return
        return result if result else None
    
    def _check_security(self, code):
        return []
    
    def _check_performance(self, code):
        return []
    
    def _check_style(self, code):
        return []


# =============================================================================
# Worker Registry Pattern
# =============================================================================

class WorkerRegistry:
    """
    Registry for managing available workers.
    
    This pattern makes it easy to:
    - Register new workers dynamically
    - Look up workers by type
    - List available capabilities
    """
    
    def __init__(self):
        self._workers: dict[str, BaseWorker] = {}
    
    def register(self, worker_type: str, worker: BaseWorker):
        """Register a worker for a given type."""
        self._workers[worker_type] = worker
    
    def get(self, worker_type: str) -> Optional[BaseWorker]:
        """Get a worker by type."""
        return self._workers.get(worker_type)
    
    def list_workers(self) -> list[dict]:
        """List all registered workers and their capabilities."""
        return [
            {
                "type": worker_type,
                "name": worker.name,
                "capabilities": worker.capabilities
            }
            for worker_type, worker in self._workers.items()
        ]
    
    def capabilities_description(self) -> str:
        """Generate a description of all workers for the orchestrator prompt."""
        lines = []
        for worker_type, worker in self._workers.items():
            lines.append(f"- {worker_type}: {worker.capabilities}")
        return "\n".join(lines)


# =============================================================================
# Demonstration
# =============================================================================

def demonstrate_workers():
    """Demonstrate well-designed worker interfaces."""
    
    print("=" * 60)
    print("Worker Design Examples")
    print("=" * 60)
    
    # Create workers
    security = SecurityAnalyzer()
    performance = PerformanceAnalyzer()
    style = StyleReviewer()
    
    # Create a registry
    registry = WorkerRegistry()
    registry.register("security", security)
    registry.register("performance", performance)
    registry.register("style", style)
    
    print("\n1. Registered Workers")
    print("-" * 40)
    for worker_info in registry.list_workers():
        print(f"\n{worker_info['type']}:")
        print(f"  Name: {worker_info['name']}")
        print(f"  Capabilities: {worker_info['capabilities'][:60]}...")
    
    print("\n\n2. Standard Worker Interface")
    print("-" * 40)
    
    # Demonstrate consistent input/output
    test_input = WorkerInput(
        task_id="test_1",
        task_description="Analyze code for security vulnerabilities",
        context={"file": "example.py"}
    )
    
    print(f"Input: {test_input}")
    
    output = security.execute(test_input)
    print(f"\nOutput: {output}")
    
    print("\n\n3. Graceful Failure Example")
    print("-" * 40)
    
    # Create a worker that will fail
    class FailingWorker(BaseWorker):
        def __init__(self):
            super().__init__("failing", "Always fails for testing")
        
        def _do_work(self, input: WorkerInput) -> dict:
            raise ValueError("Simulated failure!")
    
    failing = FailingWorker()
    failure_output = failing.execute(test_input)
    
    print(f"Status: {failure_output.status}")
    print(f"Errors: {failure_output.errors}")
    print("Note: Worker failed gracefully without crashing!")
    
    print("\n\n4. Worker Capabilities for Orchestrator")
    print("-" * 40)
    print("This text can be included in the orchestrator prompt:\n")
    print(registry.capabilities_description())
    
    print("\n" + "=" * 60)
    print("See Chapter 23 for LLM-powered worker implementations!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_workers()
