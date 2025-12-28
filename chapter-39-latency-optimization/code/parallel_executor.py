"""
Parallel tool execution for reduced latency.

Chapter 39: Latency Optimization

When an agent needs to call multiple tools, sequential execution
wastes time. If the tools are independent, run them in parallel.
"""

import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    tool_use_id: str
    result: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "tool_use_id": self.tool_use_id,
            "result": self.result,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
            "error": self.error
        }


class ParallelToolExecutor:
    """
    Executes multiple tool calls in parallel.
    
    When an LLM returns multiple tool_use blocks, they can often
    be executed simultaneously. This class manages parallel execution
    and aggregates results.
    
    Usage:
        executor = ParallelToolExecutor()
        executor.register_tool("weather", get_weather)
        executor.register_tool("stock", get_stock_price)
        
        tool_calls = [
            {"id": "1", "name": "weather", "input": {"city": "NYC"}},
            {"id": "2", "name": "stock", "input": {"symbol": "AAPL"}},
        ]
        
        results = executor.execute_parallel(tool_calls)
    """
    
    def __init__(self, max_workers: int = 5, timeout: float = 30.0):
        """
        Initialize the parallel executor.
        
        Args:
            max_workers: Maximum concurrent tool executions
            timeout: Timeout for each tool call in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.tools: dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable[..., Any]
    ) -> None:
        """
        Register a tool function.
        
        Args:
            name: Tool name
            func: Function to execute
        """
        self.tools[name] = func
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to remove
        
        Returns:
            True if tool was removed, False if not found
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self.tools.keys())
    
    def _execute_single(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool call."""
        start_time = time.perf_counter()
        
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=None,
                duration_ms=0,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
        
        try:
            result = self.tools[tool_name](**tool_input)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=result,
                duration_ms=duration_ms,
                success=True
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
    
    def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]]
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel.
        
        Args:
            tool_calls: List of {"id": str, "name": str, "input": dict}
        
        Returns:
            List of ToolResult in the same order as input
        """
        if not tool_calls:
            return []
        
        # If only one tool, execute directly
        if len(tool_calls) == 1:
            call = tool_calls[0]
            return [self._execute_single(
                call["name"],
                call["id"],
                call["input"]
            )]
        
        # Execute in parallel
        results: dict[str, ToolResult] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_single,
                    call["name"],
                    call["id"],
                    call["input"]
                ): call["id"]
                for call in tool_calls
            }
            
            try:
                for future in as_completed(futures, timeout=self.timeout):
                    tool_use_id = futures[future]
                    try:
                        result = future.result()
                        results[tool_use_id] = result
                    except Exception as e:
                        call = next(c for c in tool_calls if c["id"] == tool_use_id)
                        results[tool_use_id] = ToolResult(
                            tool_name=call["name"],
                            tool_use_id=tool_use_id,
                            result=None,
                            duration_ms=self.timeout * 1000,
                            success=False,
                            error=f"Execution failed: {str(e)}"
                        )
            except TimeoutError:
                # Handle overall timeout
                for call in tool_calls:
                    if call["id"] not in results:
                        results[call["id"]] = ToolResult(
                            tool_name=call["name"],
                            tool_use_id=call["id"],
                            result=None,
                            duration_ms=self.timeout * 1000,
                            success=False,
                            error="Tool execution timed out"
                        )
        
        # Return in original order
        return [results[call["id"]] for call in tool_calls]
    
    def execute_sequential(
        self,
        tool_calls: list[dict[str, Any]]
    ) -> list[ToolResult]:
        """
        Execute tool calls sequentially (for comparison/fallback).
        
        Args:
            tool_calls: List of tool call specifications
        
        Returns:
            List of ToolResult
        """
        return [
            self._execute_single(call["name"], call["id"], call["input"])
            for call in tool_calls
        ]


class AsyncParallelToolExecutor:
    """
    Async version of parallel tool executor.
    
    Use this when your tools are async functions or when
    integrating with async web frameworks.
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize async executor.
        
        Args:
            timeout: Timeout for each tool call in seconds
        """
        self.timeout = timeout
        self.tools: dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable[..., Any]
    ) -> None:
        """Register a tool function (sync or async)."""
        self.tools[name] = func
    
    async def _execute_single(
        self,
        tool_name: str,
        tool_use_id: str,
        tool_input: dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool call."""
        start_time = time.perf_counter()
        
        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=None,
                duration_ms=0,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
        
        try:
            func = self.tools[tool_name]
            
            # Handle both sync and async tools
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**tool_input),
                    timeout=self.timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**tool_input)),
                    timeout=self.timeout
                )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=result,
                duration_ms=duration_ms,
                success=True
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error="Tool execution timed out"
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(e)
            )
    
    async def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]]
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel using asyncio.
        
        Args:
            tool_calls: List of tool call specifications
        
        Returns:
            List of ToolResult
        """
        if not tool_calls:
            return []
        
        tasks = [
            self._execute_single(call["name"], call["id"], call["input"])
            for call in tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)


def compare_execution_methods(
    executor: ParallelToolExecutor,
    tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Compare parallel vs sequential execution times.
    
    Useful for demonstrating the benefit of parallelization.
    
    Args:
        executor: ParallelToolExecutor instance
        tool_calls: Tool calls to execute
    
    Returns:
        Comparison results
    """
    # Sequential
    seq_start = time.perf_counter()
    seq_results = executor.execute_sequential(tool_calls)
    seq_duration = (time.perf_counter() - seq_start) * 1000
    
    # Parallel
    par_start = time.perf_counter()
    par_results = executor.execute_parallel(tool_calls)
    par_duration = (time.perf_counter() - par_start) * 1000
    
    return {
        "sequential_ms": round(seq_duration, 2),
        "parallel_ms": round(par_duration, 2),
        "speedup": round(seq_duration / par_duration, 2) if par_duration > 0 else 0,
        "time_saved_ms": round(seq_duration - par_duration, 2),
        "time_saved_pct": round((seq_duration - par_duration) / seq_duration * 100, 1) if seq_duration > 0 else 0,
        "tool_count": len(tool_calls)
    }


# Example usage
if __name__ == "__main__":
    import random
    
    print("=" * 60)
    print("PARALLEL TOOL EXECUTION DEMO")
    print("=" * 60)
    
    # Create executor
    executor = ParallelToolExecutor(max_workers=5)
    
    # Register some simulated tools
    def slow_tool_1(query: str) -> str:
        """Simulated slow search tool."""
        time.sleep(random.uniform(0.3, 0.5))
        return f"Search result for: {query}"
    
    def slow_tool_2(location: str) -> str:
        """Simulated weather API."""
        time.sleep(random.uniform(0.2, 0.4))
        return f"Weather in {location}: Sunny, 72°F"
    
    def slow_tool_3(symbol: str) -> str:
        """Simulated stock API."""
        time.sleep(random.uniform(0.25, 0.45))
        return f"Stock {symbol}: $150.25 (+2.3%)"
    
    def fast_tool(data: str) -> str:
        """Fast tool for comparison."""
        time.sleep(0.05)
        return f"Processed: {data}"
    
    executor.register_tool("search", slow_tool_1)
    executor.register_tool("weather", slow_tool_2)
    executor.register_tool("stock", slow_tool_3)
    executor.register_tool("process", fast_tool)
    
    # Create tool calls
    tool_calls = [
        {"id": "tool_1", "name": "search", "input": {"query": "Python programming"}},
        {"id": "tool_2", "name": "weather", "input": {"location": "New York"}},
        {"id": "tool_3", "name": "stock", "input": {"symbol": "AAPL"}},
        {"id": "tool_4", "name": "weather", "input": {"location": "London"}},
        {"id": "tool_5", "name": "search", "input": {"query": "AI agents"}},
    ]
    
    print(f"\nExecuting {len(tool_calls)} tool calls...")
    print()
    
    # Compare methods
    comparison = compare_execution_methods(executor, tool_calls)
    
    print("Execution Comparison:")
    print("-" * 40)
    print(f"Sequential execution: {comparison['sequential_ms']}ms")
    print(f"Parallel execution:   {comparison['parallel_ms']}ms")
    print(f"Speedup:              {comparison['speedup']}x")
    print(f"Time saved:           {comparison['time_saved_ms']}ms ({comparison['time_saved_pct']}%)")
    
    print("\n" + "-" * 40)
    print("Individual tool results:")
    print("-" * 40)
    
    results = executor.execute_parallel(tool_calls)
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.tool_name}: {result.duration_ms:.2f}ms")
        if result.success:
            result_preview = str(result.result)[:50]
            print(f"    Result: {result_preview}...")
    
    # Async demo
    print("\n" + "=" * 60)
    print("ASYNC PARALLEL EXECUTION DEMO")
    print("=" * 60)
    
    async def async_slow_tool(param: str) -> str:
        await asyncio.sleep(random.uniform(0.2, 0.4))
        return f"Async result: {param}"
    
    async def run_async_demo():
        async_executor = AsyncParallelToolExecutor()
        async_executor.register_tool("async_tool", async_slow_tool)
        async_executor.register_tool("sync_tool", slow_tool_1)
        
        calls = [
            {"id": "1", "name": "async_tool", "input": {"param": "test1"}},
            {"id": "2", "name": "async_tool", "input": {"param": "test2"}},
            {"id": "3", "name": "sync_tool", "input": {"query": "test3"}},
        ]
        
        start = time.perf_counter()
        results = await async_executor.execute_parallel(calls)
        duration = (time.perf_counter() - start) * 1000
        
        print(f"\nAsync parallel execution: {duration:.2f}ms")
        for r in results:
            print(f"  {r.tool_name}: {r.duration_ms:.2f}ms - {r.success}")
    
    asyncio.run(run_async_demo())
