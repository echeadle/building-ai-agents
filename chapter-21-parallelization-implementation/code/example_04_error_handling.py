"""
Error handling patterns for parallel workflows.

Chapter 21: Parallelization - Implementation

When running tasks in parallel, individual failures shouldn't 
crash the entire workflow. This module provides robust error
handling patterns with different failure policies.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Coroutine
from enum import Enum
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class FailurePolicy(Enum):
    """
    How to handle failures in parallel execution.
    
    IGNORE: Continue with successful results only
    FAIL_FAST: Fail immediately on first error
    RETRY: Retry failed tasks with exponential backoff
    REQUIRE_ALL: Fail if any task fails (after trying all)
    """
    IGNORE = "ignore"
    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    REQUIRE_ALL = "require_all"


@dataclass
class TaskResult:
    """
    Result of a parallel task execution.
    
    Attributes:
        task_id: Identifier for this task
        success: Whether execution succeeded
        result: The result if successful
        error: Error message if failed
        attempts: Number of execution attempts
    """
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    attempts: int = 1


async def execute_with_retry(
    task_id: str,
    task_func: Callable[[], Coroutine],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    verbose: bool = False
) -> TaskResult:
    """
    Execute a task with automatic retries on failure.
    
    Uses exponential backoff between retries to handle
    transient failures like rate limits.
    
    Args:
        task_id: Identifier for this task
        task_func: Async function to execute
        max_retries: Maximum number of attempts
        retry_delay: Initial delay between retries (seconds)
        verbose: Whether to print retry information
        
    Returns:
        TaskResult with success status and result/error
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await task_func()
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                attempts=attempt
            )
        except Exception as e:
            last_error = str(e)
            
            if verbose:
                print(f"  {task_id}: Attempt {attempt} failed - {last_error}")
            
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, ...
                delay = retry_delay * (2 ** (attempt - 1))
                if verbose:
                    print(f"  {task_id}: Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    return TaskResult(
        task_id=task_id,
        success=False,
        error=last_error,
        attempts=max_retries
    )


async def parallel_with_policy(
    tasks: list[tuple[str, Callable[[], Coroutine]]],
    policy: FailurePolicy = FailurePolicy.IGNORE,
    max_retries: int = 3,
    max_concurrent: int | None = None
) -> list[TaskResult]:
    """
    Execute tasks in parallel with specified failure policy.
    
    Args:
        tasks: List of (task_id, async_function) tuples
        policy: How to handle failures
        max_retries: Retries for RETRY policy
        max_concurrent: Optional concurrency limit
        
    Returns:
        List of TaskResult objects
        
    Raises:
        RuntimeError: If policy is REQUIRE_ALL and any task fails
    """
    if policy == FailurePolicy.FAIL_FAST:
        return await _execute_fail_fast(tasks, max_concurrent)
    elif policy == FailurePolicy.RETRY:
        return await _execute_with_retries(tasks, max_retries, max_concurrent)
    else:
        # IGNORE or REQUIRE_ALL - execute all first
        results = await _execute_all(tasks, max_concurrent)
        
        if policy == FailurePolicy.REQUIRE_ALL:
            failures = [r for r in results if not r.success]
            if failures:
                failed_ids = [f.task_id for f in failures]
                raise RuntimeError(
                    f"{len(failures)} tasks failed: {failed_ids}"
                )
        
        return results


async def _execute_all(
    tasks: list[tuple[str, Callable[[], Coroutine]]],
    max_concurrent: int | None = None
) -> list[TaskResult]:
    """
    Execute all tasks, capturing any errors.
    
    Errors are stored in TaskResult, not raised.
    """
    semaphore = asyncio.Semaphore(max_concurrent or len(tasks))
    
    async def safe_execute(
        task_id: str, 
        func: Callable[[], Coroutine]
    ) -> TaskResult:
        async with semaphore:
            try:
                result = await func()
                return TaskResult(task_id=task_id, success=True, result=result)
            except Exception as e:
                return TaskResult(task_id=task_id, success=False, error=str(e))
    
    return await asyncio.gather(*[
        safe_execute(task_id, func) 
        for task_id, func in tasks
    ])


async def _execute_fail_fast(
    tasks: list[tuple[str, Callable[[], Coroutine]]],
    max_concurrent: int | None = None
) -> list[TaskResult]:
    """
    Execute tasks, failing immediately on first error.
    
    Remaining tasks are cancelled when an error occurs.
    """
    semaphore = asyncio.Semaphore(max_concurrent or len(tasks))
    results: list[TaskResult] = []
    
    async def bounded_execute(
        task_id: str, 
        func: Callable[[], Coroutine]
    ):
        async with semaphore:
            return await func()
    
    # Create tasks with names for identification
    pending = [
        asyncio.create_task(bounded_execute(task_id, func), name=task_id)
        for task_id, func in tasks
    ]
    
    try:
        # Wait for all, propagating first exception
        completed = await asyncio.gather(*pending, return_exceptions=False)
        for task, result in zip(pending, completed):
            results.append(TaskResult(
                task_id=task.get_name(),
                success=True,
                result=result
            ))
    except Exception as e:
        # Cancel remaining tasks
        for task in pending:
            if not task.done():
                task.cancel()
        
        # Wait for cancellations to complete
        await asyncio.gather(*pending, return_exceptions=True)
        
        # Re-raise the original exception
        raise
    
    return results


async def _execute_with_retries(
    tasks: list[tuple[str, Callable[[], Coroutine]]],
    max_retries: int,
    max_concurrent: int | None = None
) -> list[TaskResult]:
    """
    Execute tasks with automatic retries on failure.
    """
    semaphore = asyncio.Semaphore(max_concurrent or len(tasks))
    
    async def bounded_retry(
        task_id: str, 
        func: Callable[[], Coroutine]
    ) -> TaskResult:
        async with semaphore:
            return await execute_with_retry(task_id, func, max_retries)
    
    return await asyncio.gather(*[
        bounded_retry(task_id, func)
        for task_id, func in tasks
    ])


# =============================================================================
# Example: Demonstrating Different Failure Policies
# =============================================================================

async def failure_policy_demo():
    """
    Demonstrate how different failure policies behave.
    """
    import random
    
    # Seed for reproducibility in demo
    random.seed(42)
    
    async def unreliable_task(fail_probability: float = 0.5) -> str:
        """A task that fails with given probability."""
        await asyncio.sleep(0.1)  # Simulate work
        if random.random() < fail_probability:
            raise RuntimeError("Random failure!")
        return "success"
    
    def make_tasks(count: int = 5, fail_prob: float = 0.5):
        """Create a list of unreliable tasks."""
        return [
            (f"task_{i}", lambda p=fail_prob: unreliable_task(p))
            for i in range(count)
        ]
    
    print("=" * 60)
    print("FAILURE POLICY DEMONSTRATION")
    print("=" * 60)
    
    # Test IGNORE policy
    print("\n1. IGNORE Policy (continue with successful results)")
    print("-" * 40)
    random.seed(42)
    tasks = make_tasks(5, 0.4)
    results = await parallel_with_policy(tasks, FailurePolicy.IGNORE)
    success = len([r for r in results if r.success])
    print(f"   Completed: {success}/5 succeeded")
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"   {status} {r.task_id}: {r.result or r.error}")
    
    # Test RETRY policy
    print("\n2. RETRY Policy (retry failed tasks)")
    print("-" * 40)
    random.seed(42)
    tasks = make_tasks(5, 0.6)
    results = await parallel_with_policy(tasks, FailurePolicy.RETRY, max_retries=3)
    for r in results:
        status = "✓" if r.success else "✗"
        attempts = f"({r.attempts} attempts)" if r.attempts > 1 else ""
        print(f"   {status} {r.task_id}: {r.result or r.error} {attempts}")
    
    # Test REQUIRE_ALL policy
    print("\n3. REQUIRE_ALL Policy (fail if any task fails)")
    print("-" * 40)
    random.seed(42)
    tasks = make_tasks(5, 0.4)
    try:
        results = await parallel_with_policy(tasks, FailurePolicy.REQUIRE_ALL)
        print("   All tasks succeeded!")
    except RuntimeError as e:
        print(f"   Failed: {e}")
    
    # Test FAIL_FAST policy
    print("\n4. FAIL_FAST Policy (stop on first failure)")
    print("-" * 40)
    random.seed(42)
    tasks = make_tasks(5, 0.5)
    try:
        results = await parallel_with_policy(tasks, FailurePolicy.FAIL_FAST)
        print("   All tasks succeeded!")
    except Exception as e:
        print(f"   Stopped on first failure: {e}")


# =============================================================================
# Example: Real API Calls with Error Handling
# =============================================================================

async def api_error_handling_example():
    """
    Demonstrate error handling with real API calls.
    """
    async_client = anthropic.AsyncAnthropic()
    
    async def make_api_call(prompt: str) -> str:
        """Make an API call that might fail."""
        response = await async_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    # Create tasks with varying complexity
    prompts = [
        ("What is 2+2?", "simple_math"),
        ("Name three colors.", "colors"),
        ("What day comes after Monday?", "day"),
    ]
    
    tasks = [
        (label, lambda p=prompt: make_api_call(p))
        for prompt, label in prompts
    ]
    
    print("\n" + "=" * 60)
    print("API CALLS WITH ERROR HANDLING")
    print("=" * 60)
    
    print("\nRunning with IGNORE policy...")
    results = await parallel_with_policy(tasks, FailurePolicy.IGNORE)
    
    for r in results:
        status = "✓" if r.success else "✗"
        result_preview = str(r.result)[:50] if r.result else r.error
        print(f"  {status} {r.task_id}: {result_preview}")


# =============================================================================
# Utility: Timeout Wrapper
# =============================================================================

async def with_timeout(
    task_func: Callable[[], Coroutine],
    timeout_seconds: float
) -> Any:
    """
    Execute a task with a timeout.
    
    Args:
        task_func: Async function to execute
        timeout_seconds: Maximum execution time
        
    Returns:
        The task result
        
    Raises:
        asyncio.TimeoutError: If task exceeds timeout
    """
    return await asyncio.wait_for(task_func(), timeout=timeout_seconds)


async def timeout_demo():
    """Demonstrate timeout handling."""
    
    async def slow_task():
        """A task that takes too long."""
        await asyncio.sleep(5)
        return "completed"
    
    async def fast_task():
        """A task that completes quickly."""
        await asyncio.sleep(0.1)
        return "completed"
    
    print("\n" + "=" * 60)
    print("TIMEOUT HANDLING")
    print("=" * 60)
    
    # Fast task should complete
    print("\nFast task with 1s timeout:")
    try:
        result = await with_timeout(fast_task, 1.0)
        print(f"  ✓ Completed: {result}")
    except asyncio.TimeoutError:
        print("  ✗ Timed out")
    
    # Slow task should timeout
    print("\nSlow task with 0.5s timeout:")
    try:
        result = await with_timeout(slow_task, 0.5)
        print(f"  ✓ Completed: {result}")
    except asyncio.TimeoutError:
        print("  ✗ Timed out (as expected)")


# =============================================================================
# Utility: Circuit Breaker Pattern
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern for handling repeated failures.
    
    After a certain number of failures, the circuit "opens" and
    fails fast without attempting the operation. After a cooldown
    period, it allows one test request through.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, no requests pass through
    - HALF_OPEN: Testing with a single request
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"
    
    async def execute(
        self,
        task_func: Callable[[], Coroutine]
    ) -> Any:
        """
        Execute a task through the circuit breaker.
        
        Raises:
            RuntimeError: If circuit is open
        """
        import time
        
        # Check if we should try to recover
        if self.state == "OPEN":
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.cooldown_seconds:
                    self.state = "HALF_OPEN"
                else:
                    raise RuntimeError(
                        f"Circuit breaker open. Retry in "
                        f"{self.cooldown_seconds - elapsed:.0f}s"
                    )
        
        try:
            result = await task_func()
            
            # Success - reset if we were testing
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise


async def circuit_breaker_demo():
    """Demonstrate circuit breaker pattern."""
    import random
    
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=2.0)
    
    async def flaky_service():
        """A service that fails frequently."""
        if random.random() < 0.8:  # 80% failure rate
            raise RuntimeError("Service unavailable")
        return "success"
    
    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER PATTERN")
    print("=" * 60)
    
    random.seed(123)
    
    for i in range(10):
        try:
            result = await breaker.execute(flaky_service)
            print(f"  Request {i+1}: ✓ {result} (state: {breaker.state})")
        except RuntimeError as e:
            print(f"  Request {i+1}: ✗ {e} (state: {breaker.state})")
        
        await asyncio.sleep(0.5)


async def main():
    """Run all error handling examples."""
    await failure_policy_demo()
    await api_error_handling_example()
    await timeout_demo()
    await circuit_breaker_demo()


if __name__ == "__main__":
    asyncio.run(main())
