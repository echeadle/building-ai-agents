"""
Response time budgets for predictable agent performance.

Chapter 39: Latency Optimization

In production, you need to guarantee response times. A response
time budget enforces limits on how long operations can take.
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetExceededAction(Enum):
    """What to do when budget is exceeded."""
    WARN = "warn"           # Log warning but continue
    ABORT = "abort"         # Raise exception
    FALLBACK = "fallback"   # Use fallback response


@dataclass
class BudgetAllocation:
    """Allocation of time budget across operations."""
    llm_ms: float
    tool_ms: float
    network_ms: float
    buffer_ms: float
    
    @property
    def total_ms(self) -> float:
        """Get total allocated time."""
        return self.llm_ms + self.tool_ms + self.network_ms + self.buffer_ms
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "llm_ms": self.llm_ms,
            "tool_ms": self.tool_ms,
            "network_ms": self.network_ms,
            "buffer_ms": self.buffer_ms,
            "total_ms": self.total_ms
        }


@dataclass
class BudgetStatus:
    """Current status of a time budget."""
    total_budget_ms: float
    elapsed_ms: float
    remaining_ms: float
    allocations_used: dict[str, float] = field(default_factory=dict)
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.remaining_ms <= 0
    
    @property
    def utilization_pct(self) -> float:
        """Get budget utilization percentage."""
        if self.total_budget_ms <= 0:
            return 0
        return (self.elapsed_ms / self.total_budget_ms) * 100
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_budget_ms": round(self.total_budget_ms, 2),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "remaining_ms": round(self.remaining_ms, 2),
            "utilization_pct": round(self.utilization_pct, 1),
            "is_exceeded": self.is_exceeded,
            "allocations_used": {k: round(v, 2) for k, v in self.allocations_used.items()}
        }


class BudgetExceededException(Exception):
    """Raised when a time budget is exceeded."""
    
    def __init__(self, message: str, status: BudgetStatus):
        super().__init__(message)
        self.status = status


class ResponseTimeBudget:
    """
    Manages time budgets for agent operations.
    
    Ensures responses complete within specified time limits by:
    - Tracking time spent in each operation
    - Warning when approaching limits
    - Aborting or falling back when limits exceeded
    
    Usage:
        budget = ResponseTimeBudget(total_ms=5000)
        budget.allocate(llm_ms=3000, tool_ms=1500, buffer_ms=500)
        
        with budget.track("llm_call"):
            response = client.messages.create(...)
        
        if budget.can_continue():
            with budget.track("tool_call"):
                result = execute_tool(...)
    """
    
    def __init__(
        self,
        total_ms: float,
        on_exceeded: BudgetExceededAction = BudgetExceededAction.WARN,
        warning_threshold: float = 0.8
    ):
        """
        Initialize a response time budget.
        
        Args:
            total_ms: Total time budget in milliseconds
            on_exceeded: Action when budget is exceeded
            warning_threshold: Warn when this fraction of budget is used
        """
        self.total_ms = total_ms
        self.on_exceeded = on_exceeded
        self.warning_threshold = warning_threshold
        
        self.start_time: Optional[float] = None
        self.allocations: dict[str, float] = {}
        self.spent: dict[str, float] = {}
        self._warned = False
    
    def allocate(
        self,
        llm_ms: float = 0,
        tool_ms: float = 0,
        network_ms: float = 0,
        buffer_ms: float = 0
    ) -> BudgetAllocation:
        """
        Allocate budget across operation types.
        
        This helps ensure no single operation type consumes
        the entire budget.
        
        Args:
            llm_ms: Time allocated for LLM calls
            tool_ms: Time allocated for tool execution
            network_ms: Time allocated for network operations
            buffer_ms: Buffer time for overhead
        
        Returns:
            BudgetAllocation object
        """
        allocation = BudgetAllocation(
            llm_ms=llm_ms,
            tool_ms=tool_ms,
            network_ms=network_ms,
            buffer_ms=buffer_ms
        )
        
        if allocation.total_ms > self.total_ms:
            logger.warning(
                f"Allocation ({allocation.total_ms}ms) exceeds budget ({self.total_ms}ms)"
            )
        
        self.allocations = {
            "llm": llm_ms,
            "tool": tool_ms,
            "network": network_ms,
            "buffer": buffer_ms
        }
        self.spent = {k: 0.0 for k in self.allocations}
        
        return allocation
    
    def start(self) -> None:
        """Start the budget timer."""
        self.start_time = time.perf_counter()
        self._warned = False
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0
        return (time.perf_counter() - self.start_time) * 1000
    
    def get_remaining_ms(self) -> float:
        """Get remaining time in budget."""
        return max(0, self.total_ms - self.get_elapsed_ms())
    
    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        elapsed = self.get_elapsed_ms()
        return BudgetStatus(
            total_budget_ms=self.total_ms,
            elapsed_ms=elapsed,
            remaining_ms=max(0, self.total_ms - elapsed),
            allocations_used=self.spent.copy()
        )
    
    def can_continue(self, required_ms: float = 0) -> bool:
        """
        Check if there's enough budget to continue.
        
        Args:
            required_ms: Minimum time needed for next operation
        
        Returns:
            True if enough budget remains
        """
        return self.get_remaining_ms() >= required_ms
    
    def check_budget(self) -> None:
        """Check budget and take action if exceeded."""
        status = self.get_status()
        
        # Check warning threshold
        if not self._warned and status.utilization_pct >= self.warning_threshold * 100:
            logger.warning(
                f"Budget {status.utilization_pct:.1f}% used "
                f"({status.elapsed_ms:.0f}ms / {status.total_budget_ms:.0f}ms)"
            )
            self._warned = True
        
        # Check if exceeded
        if status.is_exceeded:
            message = f"Response time budget exceeded: {status.elapsed_ms:.0f}ms > {status.total_budget_ms:.0f}ms"
            
            if self.on_exceeded == BudgetExceededAction.ABORT:
                raise BudgetExceededException(message, status)
            elif self.on_exceeded == BudgetExceededAction.WARN:
                logger.error(message)
    
    @contextmanager
    def track(
        self,
        operation: str,
        category: str = "other"
    ) -> Generator[None, None, None]:
        """
        Track time spent on an operation.
        
        Args:
            operation: Name of the operation
            category: Category for budget allocation (llm, tool, network, other)
        """
        if self.start_time is None:
            self.start()
        
        op_start = time.perf_counter()
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - op_start) * 1000
            
            if category in self.spent:
                self.spent[category] += duration_ms
            else:
                self.spent["other"] = self.spent.get("other", 0) + duration_ms
            
            # Check if category allocation exceeded
            if category in self.allocations and self.allocations[category] > 0:
                if self.spent[category] > self.allocations[category]:
                    logger.warning(
                        f"Category '{category}' exceeded allocation: "
                        f"{self.spent[category]:.0f}ms > {self.allocations[category]:.0f}ms"
                    )
            
            self.check_budget()
    
    def get_allocation_remaining(self, category: str) -> float:
        """
        Get remaining time for a category.
        
        Args:
            category: Category to check
        
        Returns:
            Remaining time in milliseconds
        """
        if category not in self.allocations:
            return self.get_remaining_ms()
        
        return max(0, self.allocations[category] - self.spent.get(category, 0))
    
    def reset(self) -> None:
        """Reset the budget for reuse."""
        self.start_time = None
        self.spent = {k: 0.0 for k in self.allocations} if self.allocations else {}
        self._warned = False


class AdaptiveTimeout:
    """
    Dynamically adjusts timeouts based on remaining budget.
    
    Use this to set tool/API timeouts that respect the overall
    response time budget.
    """
    
    def __init__(
        self,
        budget: ResponseTimeBudget,
        min_timeout_ms: float = 100,
        safety_margin: float = 0.9
    ):
        """
        Initialize adaptive timeout.
        
        Args:
            budget: The response time budget to track
            min_timeout_ms: Minimum timeout to allow
            safety_margin: Use this fraction of remaining time
        """
        self.budget = budget
        self.min_timeout_ms = min_timeout_ms
        self.safety_margin = safety_margin
    
    def get_timeout_seconds(self, for_category: Optional[str] = None) -> float:
        """
        Get appropriate timeout based on remaining budget.
        
        Args:
            for_category: If specified, use category allocation
        
        Returns:
            Timeout in seconds
        """
        if for_category:
            remaining = self.budget.get_allocation_remaining(for_category)
        else:
            remaining = self.budget.get_remaining_ms()
        
        timeout_ms = max(self.min_timeout_ms, remaining * self.safety_margin)
        return timeout_ms / 1000
    
    def get_timeout_ms(self, for_category: Optional[str] = None) -> float:
        """
        Get appropriate timeout in milliseconds.
        
        Args:
            for_category: If specified, use category allocation
        
        Returns:
            Timeout in milliseconds
        """
        return self.get_timeout_seconds(for_category) * 1000


class TimeBudgetedOperation:
    """
    Decorator/context manager for budget-aware operations.
    
    Usage:
        budget = ResponseTimeBudget(total_ms=5000)
        
        @TimeBudgetedOperation(budget, "llm")
        def call_llm():
            return client.messages.create(...)
    """
    
    def __init__(
        self,
        budget: ResponseTimeBudget,
        category: str = "other",
        operation_name: Optional[str] = None
    ):
        """
        Initialize budgeted operation.
        
        Args:
            budget: Budget to track against
            category: Category for this operation
            operation_name: Optional name (uses function name if not provided)
        """
        self.budget = budget
        self.category = category
        self.operation_name = operation_name
    
    def __call__(self, func):
        """Decorator usage."""
        def wrapper(*args, **kwargs):
            op_name = self.operation_name or func.__name__
            with self.budget.track(op_name, self.category):
                return func(*args, **kwargs)
        return wrapper
    
    def __enter__(self):
        """Context manager entry."""
        op_name = self.operation_name or "operation"
        self._cm = self.budget.track(op_name, self.category)
        return self._cm.__enter__()
    
    def __exit__(self, *args):
        """Context manager exit."""
        return self._cm.__exit__(*args)


# Example usage
if __name__ == "__main__":
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("RESPONSE TIME BUDGET DEMO")
    print("=" * 60)
    
    # Create a budget
    budget = ResponseTimeBudget(
        total_ms=2000,  # 2 second budget
        on_exceeded=BudgetExceededAction.WARN,
        warning_threshold=0.8
    )
    
    # Allocate time across categories
    allocation = budget.allocate(
        llm_ms=1200,    # 60% for LLM
        tool_ms=600,    # 30% for tools
        buffer_ms=200   # 10% buffer
    )
    
    print(f"\nBudget: {budget.total_ms}ms")
    print(f"Allocation: {json.dumps(allocation.to_dict(), indent=2)}")
    
    # Start tracking
    budget.start()
    
    print("\n" + "-" * 40)
    print("Executing operations...")
    print("-" * 40)
    
    # Simulate LLM call
    with budget.track("llm_planning", "llm"):
        time.sleep(0.4)
    print(f"After LLM planning: {budget.get_remaining_ms():.0f}ms remaining")
    
    # Check if we can continue
    if budget.can_continue(300):
        with budget.track("tool_execution", "tool"):
            time.sleep(0.2)
        print(f"After tool execution: {budget.get_remaining_ms():.0f}ms remaining")
    
    # Another LLM call
    with budget.track("llm_response", "llm"):
        time.sleep(0.5)
    print(f"After LLM response: {budget.get_remaining_ms():.0f}ms remaining")
    
    # Get final status
    status = budget.get_status()
    print(f"\n" + "-" * 40)
    print("Final Status:")
    print("-" * 40)
    print(json.dumps(status.to_dict(), indent=2))
    
    # Test adaptive timeout
    print("\n" + "-" * 40)
    print("Adaptive Timeout Demo:")
    print("-" * 40)
    
    fresh_budget = ResponseTimeBudget(total_ms=5000)
    fresh_budget.allocate(llm_ms=3000, tool_ms=1500, buffer_ms=500)
    fresh_budget.start()
    
    adaptive = AdaptiveTimeout(fresh_budget, min_timeout_ms=100, safety_margin=0.9)
    
    print(f"Initial timeout: {adaptive.get_timeout_seconds():.2f}s")
    print(f"LLM category timeout: {adaptive.get_timeout_seconds('llm'):.2f}s")
    print(f"Tool category timeout: {adaptive.get_timeout_seconds('tool'):.2f}s")
    
    # Simulate some time passing
    with fresh_budget.track("operation1", "llm"):
        time.sleep(1.0)
    
    print(f"\nAfter 1s operation:")
    print(f"Overall timeout: {adaptive.get_timeout_seconds():.2f}s")
    print(f"LLM category timeout: {adaptive.get_timeout_seconds('llm'):.2f}s")
    
    # Test budget exceeded scenario
    print("\n" + "-" * 40)
    print("Budget Exceeded Demo:")
    print("-" * 40)
    
    tight_budget = ResponseTimeBudget(
        total_ms=500,
        on_exceeded=BudgetExceededAction.WARN
    )
    tight_budget.start()
    
    try:
        with tight_budget.track("slow_operation", "llm"):
            time.sleep(0.6)  # Exceeds budget
    except BudgetExceededException as e:
        print(f"Budget exceeded: {e}")
    
    final_status = tight_budget.get_status()
    print(f"Final utilization: {final_status.utilization_pct:.1f}%")
