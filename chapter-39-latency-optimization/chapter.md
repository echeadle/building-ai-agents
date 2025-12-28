---
chapter: 39
title: "Latency Optimization"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 39: Latency Optimization

## Introduction

Your agent is cost-optimized and working correctly. But users are complaining: "It takes forever to respond." They type a question, wait... and wait... and finally get an answer 15 seconds later. By then, they've already opened a new tab.

Speed matters. Research shows that users start abandoning interactions after just 2-3 seconds of waiting. For agents that make multiple LLM calls, achieving sub-second responses might be impossible—but the difference between 3 seconds and 15 seconds is the difference between a product people love and one they tolerate.

In Chapter 38, we optimized for cost. Now we'll optimize for speed. These goals sometimes conflict: the fastest solution might be expensive, and the cheapest might be slow. This chapter teaches you to navigate those tradeoffs and build agents that feel responsive even when doing complex work.

The key insight: **users perceive speed differently than they measure it**. An agent that streams its response feels faster than one that waits and dumps everything at once, even if the total time is the same. We'll exploit this perception throughout the chapter.

## Learning Objectives

By the end of this chapter, you will be able to:

- Identify and measure latency bottlenecks in agent systems
- Implement streaming responses for immediate user feedback
- Execute tool calls in parallel when dependencies allow
- Build precomputation and caching systems for predictable queries
- Select faster models strategically based on task requirements
- Set and enforce response time budgets across agent operations

## Understanding Agent Latency

Before optimizing, you need to understand where time goes. Agent latency has several components:

### The Anatomy of Agent Response Time

```
User sends message
    │
    ├── Network latency (client → server): ~50-200ms
    │
    ├── Server processing: ~10-50ms
    │
    ├── LLM API call #1 (planning): 500-2000ms
    │   ├── Network to Anthropic: ~50ms
    │   ├── Queue time: variable
    │   ├── Inference time: 300-1500ms
    │   └── Network back: ~50ms
    │
    ├── Tool execution #1: variable (100ms - 5000ms)
    │
    ├── LLM API call #2 (process results): 500-2000ms
    │
    ├── Tool execution #2: variable
    │
    ├── LLM API call #3 (final response): 500-2000ms
    │
    └── Network latency (server → client): ~50-200ms

Total: 2-15+ seconds typical
```

### Where the Time Goes

In a typical agent interaction:

| Component | Percentage | Can Optimize? |
|-----------|------------|---------------|
| LLM inference | 60-80% | Partially (model selection, prompt size) |
| Tool execution | 10-30% | Yes (parallelization, caching) |
| Network overhead | 5-15% | Limited (geography, connection reuse) |
| Server processing | 1-5% | Yes (but usually not the bottleneck) |

The LLM dominates. But since we have limited control over Anthropic's infrastructure, we focus on what we *can* control:

1. **Reduce LLM calls** — Do more with fewer round-trips
2. **Make calls faster** — Smaller prompts, faster models
3. **Parallelize** — Execute independent operations simultaneously
4. **Stream** — Show progress immediately
5. **Cache** — Avoid redundant work

Let's build tools for each strategy.

## Measuring Latency

You can't optimize what you don't measure. Let's build a latency profiler specifically designed for agents.

```python
"""
Latency profiling for AI agents.

Chapter 39: Latency Optimization
"""

import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Optional
from collections import defaultdict
import json


@dataclass
class TimingRecord:
    """A single timing measurement."""
    operation: str
    duration_ms: float
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyBreakdown:
    """Breakdown of latency by component."""
    total_ms: float
    llm_ms: float
    tool_ms: float
    network_ms: float
    other_ms: float
    llm_call_count: int
    tool_call_count: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total_ms, 2),
            "llm_ms": round(self.llm_ms, 2),
            "tool_ms": round(self.tool_ms, 2),
            "network_ms": round(self.network_ms, 2),
            "other_ms": round(self.other_ms, 2),
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "llm_percentage": round(self.llm_ms / self.total_ms * 100, 1) if self.total_ms > 0 else 0,
            "tool_percentage": round(self.tool_ms / self.total_ms * 100, 1) if self.total_ms > 0 else 0,
        }


class LatencyProfiler:
    """
    Profiles agent latency to identify bottlenecks.
    
    Usage:
        profiler = LatencyProfiler()
        
        with profiler.measure("total_request"):
            with profiler.measure("llm_call", category="llm"):
                response = client.messages.create(...)
            
            with profiler.measure("weather_api", category="tool"):
                weather = get_weather(...)
        
        print(profiler.get_breakdown())
    """
    
    def __init__(self):
        self.records: list[TimingRecord] = []
        self._active_timers: dict[str, float] = {}
        self._category_totals: dict[str, float] = defaultdict(float)
        self._category_counts: dict[str, int] = defaultdict(int)
    
    def _now(self) -> str:
        """Get current timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    @contextmanager
    def measure(
        self,
        operation: str,
        category: str = "other",
        **metadata: Any
    ) -> Generator[None, None, None]:
        """
        Context manager to measure operation duration.
        
        Args:
            operation: Name of the operation being measured
            category: Category for aggregation (llm, tool, network, other)
            **metadata: Additional data to attach to the timing record
        """
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            record = TimingRecord(
                operation=operation,
                duration_ms=duration_ms,
                timestamp=self._now(),
                metadata={"category": category, **metadata}
            )
            
            self.records.append(record)
            self._category_totals[category] += duration_ms
            self._category_counts[category] += 1
    
    def record(
        self,
        operation: str,
        duration_ms: float,
        category: str = "other",
        **metadata: Any
    ) -> None:
        """
        Record a timing measurement directly.
        
        Use this when you can't use the context manager.
        """
        record = TimingRecord(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=self._now(),
            metadata={"category": category, **metadata}
        )
        
        self.records.append(record)
        self._category_totals[category] += duration_ms
        self._category_counts[category] += 1
    
    def get_breakdown(self) -> LatencyBreakdown:
        """Get a breakdown of latency by category."""
        total = sum(self._category_totals.values())
        
        return LatencyBreakdown(
            total_ms=total,
            llm_ms=self._category_totals.get("llm", 0),
            tool_ms=self._category_totals.get("tool", 0),
            network_ms=self._category_totals.get("network", 0),
            other_ms=self._category_totals.get("other", 0),
            llm_call_count=self._category_counts.get("llm", 0),
            tool_call_count=self._category_counts.get("tool", 0),
        )
    
    def get_operation_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for a specific operation."""
        durations = [
            r.duration_ms for r in self.records
            if r.operation == operation
        ]
        
        if not durations:
            return {"count": 0}
        
        return {
            "count": len(durations),
            "total_ms": sum(durations),
            "mean_ms": statistics.mean(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": statistics.median(durations),
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else max(durations),
        }
    
    def get_slowest_operations(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the N slowest operations."""
        sorted_records = sorted(
            self.records,
            key=lambda r: r.duration_ms,
            reverse=True
        )
        
        return [
            {
                "operation": r.operation,
                "duration_ms": round(r.duration_ms, 2),
                "category": r.metadata.get("category", "other"),
            }
            for r in sorted_records[:n]
        ]
    
    def get_summary(self) -> dict[str, Any]:
        """Get a complete profiling summary."""
        breakdown = self.get_breakdown()
        
        return {
            "breakdown": breakdown.to_dict(),
            "slowest_operations": self.get_slowest_operations(),
            "total_operations": len(self.records),
            "by_operation": {
                op: self.get_operation_stats(op)
                for op in set(r.operation for r in self.records)
            }
        }
    
    def reset(self) -> None:
        """Clear all recorded data."""
        self.records = []
        self._category_totals = defaultdict(float)
        self._category_counts = defaultdict(int)
    
    def print_report(self) -> None:
        """Print a formatted latency report."""
        breakdown = self.get_breakdown()
        
        print("\n" + "=" * 60)
        print("LATENCY PROFILE REPORT")
        print("=" * 60)
        
        print(f"\nTotal time: {breakdown.total_ms:.2f}ms")
        print(f"\nBreakdown by category:")
        print(f"  LLM calls:    {breakdown.llm_ms:>8.2f}ms ({breakdown.llm_ms/breakdown.total_ms*100:.1f}%) - {breakdown.llm_call_count} calls")
        print(f"  Tool calls:   {breakdown.tool_ms:>8.2f}ms ({breakdown.tool_ms/breakdown.total_ms*100:.1f}%) - {breakdown.tool_call_count} calls")
        print(f"  Network:      {breakdown.network_ms:>8.2f}ms ({breakdown.network_ms/breakdown.total_ms*100:.1f}%)")
        print(f"  Other:        {breakdown.other_ms:>8.2f}ms ({breakdown.other_ms/breakdown.total_ms*100:.1f}%)")
        
        print(f"\nSlowest operations:")
        for op in self.get_slowest_operations():
            print(f"  {op['operation']}: {op['duration_ms']}ms ({op['category']})")
        
        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    import random
    
    profiler = LatencyProfiler()
    
    # Simulate an agent request
    print("Simulating agent request with profiling...\n")
    
    with profiler.measure("total_request"):
        # Simulate LLM call
        with profiler.measure("llm_planning", category="llm"):
            time.sleep(random.uniform(0.3, 0.6))
        
        # Simulate tool calls
        with profiler.measure("weather_api", category="tool"):
            time.sleep(random.uniform(0.1, 0.3))
        
        with profiler.measure("database_query", category="tool"):
            time.sleep(random.uniform(0.05, 0.15))
        
        # Simulate another LLM call
        with profiler.measure("llm_response", category="llm"):
            time.sleep(random.uniform(0.4, 0.8))
    
    profiler.print_report()
```

## Streaming Responses

Streaming is the single most impactful technique for perceived performance. Instead of waiting for the complete response, you show text as it's generated.

```python
"""
Streaming responses for improved perceived latency.

Chapter 39: Latency Optimization
"""

import os
import sys
import time
from typing import Any, Callable, Generator, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class StreamMetrics:
    """Metrics from a streaming response."""
    time_to_first_token_ms: float
    total_duration_ms: float
    tokens_generated: int
    tokens_per_second: float


class StreamingAgent:
    """
    An agent that streams responses for better perceived latency.
    
    Streaming provides immediate feedback to users, making the agent
    feel much faster even when total response time is unchanged.
    
    Usage:
        agent = StreamingAgent()
        
        # Stream to console
        for chunk in agent.stream("Tell me about Python"):
            print(chunk, end="", flush=True)
        
        # Or use callbacks
        agent.stream_with_callback(
            "Tell me about Python",
            on_text=lambda text: print(text, end=""),
            on_complete=lambda metrics: print(f"\\n\\nDone in {metrics.total_duration_ms}ms")
        )
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    def stream(
        self,
        user_message: str,
        max_tokens: int = 1024
    ) -> Generator[str, None, StreamMetrics]:
        """
        Stream a response, yielding text chunks as they arrive.
        
        Args:
            user_message: The user's input
            max_tokens: Maximum tokens to generate
        
        Yields:
            Text chunks as they're generated
        
        Returns:
            StreamMetrics with timing information (access via generator.value after exhaustion)
        """
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_generated = 0
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                tokens_generated += 1  # Approximate: 1 chunk ≈ 1 token
                yield text
        
        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000 if first_token_time else total_duration
        
        return StreamMetrics(
            time_to_first_token_ms=ttft,
            total_duration_ms=total_duration,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_generated / (total_duration / 1000) if total_duration > 0 else 0
        )
    
    def stream_with_callback(
        self,
        user_message: str,
        on_text: Callable[[str], None],
        on_complete: Optional[Callable[[StreamMetrics], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        max_tokens: int = 1024
    ) -> Optional[StreamMetrics]:
        """
        Stream a response using callbacks.
        
        This is useful for integrating with async frameworks or UI updates.
        
        Args:
            user_message: The user's input
            on_text: Called for each text chunk
            on_complete: Called when streaming finishes
            on_error: Called if an error occurs
            max_tokens: Maximum tokens to generate
        
        Returns:
            StreamMetrics if successful, None if error
        """
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_generated = 0
        
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}]
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    
                    tokens_generated += 1
                    on_text(text)
            
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000
            ttft = (first_token_time - start_time) * 1000 if first_token_time else total_duration
            
            metrics = StreamMetrics(
                time_to_first_token_ms=ttft,
                total_duration_ms=total_duration,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_generated / (total_duration / 1000) if total_duration > 0 else 0
            )
            
            if on_complete:
                on_complete(metrics)
            
            return metrics
            
        except Exception as e:
            if on_error:
                on_error(e)
            else:
                raise
            return None
    
    def stream_with_tools(
        self,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict], str],
        max_tokens: int = 1024,
        max_iterations: int = 10
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream responses with tool use support.
        
        This is more complex because tool calls interrupt streaming.
        We yield events to indicate what's happening.
        
        Args:
            user_message: The user's input
            tools: Tool definitions
            tool_executor: Function to execute tools (name, input) -> result
            max_tokens: Maximum tokens per response
            max_iterations: Maximum tool use iterations
        
        Yields:
            Events: {"type": "text", "content": "..."} or
                   {"type": "tool_call", "name": "...", "input": {...}} or
                   {"type": "tool_result", "name": "...", "result": "..."} or
                   {"type": "complete", "metrics": {...}}
        """
        messages = [{"role": "user", "content": user_message}]
        iteration = 0
        total_start = time.perf_counter()
        
        while iteration < max_iterations:
            iteration += 1
            
            # Stream the response
            collected_content = []
            tool_uses = []
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                tools=tools,
                messages=messages
            ) as stream:
                current_tool_use = None
                
                for event in stream:
                    # Handle different event types
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                if event.content_block.type == 'tool_use':
                                    current_tool_use = {
                                        'id': event.content_block.id,
                                        'name': event.content_block.name,
                                        'input': ''
                                    }
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                if hasattr(event.delta, 'text'):
                                    yield {"type": "text", "content": event.delta.text}
                                    collected_content.append({
                                        "type": "text",
                                        "text": event.delta.text
                                    })
                                elif hasattr(event.delta, 'partial_json'):
                                    if current_tool_use:
                                        current_tool_use['input'] += event.delta.partial_json
                        elif event.type == 'content_block_stop':
                            if current_tool_use:
                                try:
                                    import json
                                    current_tool_use['input'] = json.loads(current_tool_use['input'])
                                except:
                                    current_tool_use['input'] = {}
                                tool_uses.append(current_tool_use)
                                current_tool_use = None
                
                # Get final message
                final_message = stream.get_final_message()
            
            # Check if we're done
            if final_message.stop_reason == "end_turn":
                total_duration = (time.perf_counter() - total_start) * 1000
                yield {
                    "type": "complete",
                    "metrics": {
                        "total_duration_ms": total_duration,
                        "iterations": iteration
                    }
                }
                return
            
            # Process tool uses
            if final_message.stop_reason == "tool_use" and tool_uses:
                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })
                
                # Execute tools and collect results
                tool_results = []
                for tool_use in tool_uses:
                    yield {
                        "type": "tool_call",
                        "name": tool_use['name'],
                        "input": tool_use['input']
                    }
                    
                    # Execute tool
                    result = tool_executor(tool_use['name'], tool_use['input'])
                    
                    yield {
                        "type": "tool_result", 
                        "name": tool_use['name'],
                        "result": result
                    }
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use['id'],
                        "content": result
                    })
                
                # Add tool results to messages
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # Unexpected stop reason
                break
        
        # Max iterations reached
        total_duration = (time.perf_counter() - total_start) * 1000
        yield {
            "type": "complete",
            "metrics": {
                "total_duration_ms": total_duration,
                "iterations": iteration,
                "max_iterations_reached": True
            }
        }


class ProgressIndicator:
    """
    Shows progress during long operations.
    
    When streaming isn't possible (e.g., during tool execution),
    use progress indicators to maintain user engagement.
    """
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.frame = 0
        self.start_time = time.perf_counter()
        self._running = False
    
    def update(self, status: Optional[str] = None) -> str:
        """Get the next frame of the progress indicator."""
        self.frame = (self.frame + 1) % len(self.SPINNER_FRAMES)
        elapsed = time.perf_counter() - self.start_time
        
        msg = status or self.message
        return f"\r{self.SPINNER_FRAMES[self.frame]} {msg} ({elapsed:.1f}s)"
    
    def print_update(self, status: Optional[str] = None) -> None:
        """Print progress update to console."""
        sys.stdout.write(self.update(status))
        sys.stdout.flush()
    
    def complete(self, final_message: str = "Done") -> str:
        """Mark the operation as complete."""
        elapsed = time.perf_counter() - self.start_time
        return f"\r✓ {final_message} ({elapsed:.1f}s)\n"
    
    def print_complete(self, final_message: str = "Done") -> None:
        """Print completion message to console."""
        sys.stdout.write(self.complete(final_message))
        sys.stdout.flush()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING RESPONSE DEMO")
    print("=" * 60)
    print()
    
    agent = StreamingAgent()
    
    print("Streaming response:\n")
    print("-" * 40)
    
    # Simple streaming
    gen = agent.stream("Write a haiku about programming.")
    for chunk in gen:
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Small delay to see streaming effect
    
    print("\n" + "-" * 40)
    
    # With callbacks and metrics
    print("\n\nWith metrics tracking:\n")
    print("-" * 40)
    
    def on_text(text: str) -> None:
        print(text, end="", flush=True)
    
    def on_complete(metrics: StreamMetrics) -> None:
        print(f"\n\n--- Metrics ---")
        print(f"Time to first token: {metrics.time_to_first_token_ms:.2f}ms")
        print(f"Total duration: {metrics.total_duration_ms:.2f}ms")
        print(f"Tokens/second: {metrics.tokens_per_second:.1f}")
    
    agent.stream_with_callback(
        "Explain async programming in one paragraph.",
        on_text=on_text,
        on_complete=on_complete
    )
    
    print("-" * 40)
```

## Parallel Tool Execution

When an agent needs to call multiple tools, sequential execution wastes time. If the tools are independent, run them in parallel.

```python
"""
Parallel tool execution for reduced latency.

Chapter 39: Latency Optimization
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
        """Register a tool function."""
        self.tools[name] = func
    
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
            
            for future in as_completed(futures, timeout=self.timeout):
                tool_use_id = futures[future]
                try:
                    result = future.result()
                    results[tool_use_id] = result
                except Exception as e:
                    # Handle timeout or other errors
                    call = next(c for c in tool_calls if c["id"] == tool_use_id)
                    results[tool_use_id] = ToolResult(
                        tool_name=call["name"],
                        tool_use_id=tool_use_id,
                        result=None,
                        duration_ms=self.timeout * 1000,
                        success=False,
                        error=f"Execution failed: {str(e)}"
                    )
        
        # Return in original order
        return [results[call["id"]] for call in tool_calls]
    
    def execute_sequential(
        self,
        tool_calls: list[dict[str, Any]]
    ) -> list[ToolResult]:
        """
        Execute tool calls sequentially (for comparison/fallback).
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
        """Execute multiple tool calls in parallel using asyncio."""
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
        time.sleep(random.uniform(0.3, 0.5))  # Simulate API call
        return f"Result for query: {query}"
    
    def slow_tool_2(location: str) -> str:
        time.sleep(random.uniform(0.2, 0.4))
        return f"Weather in {location}: Sunny, 72°F"
    
    def slow_tool_3(symbol: str) -> str:
        time.sleep(random.uniform(0.25, 0.45))
        return f"Stock {symbol}: $150.25"
    
    executor.register_tool("search", slow_tool_1)
    executor.register_tool("weather", slow_tool_2)
    executor.register_tool("stock", slow_tool_3)
    
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
    
    print(f"Sequential execution: {comparison['sequential_ms']}ms")
    print(f"Parallel execution:   {comparison['parallel_ms']}ms")
    print(f"Speedup:              {comparison['speedup']}x")
    print(f"Time saved:           {comparison['time_saved_ms']}ms")
    
    print("\n" + "-" * 60)
    print("Individual tool results:")
    print("-" * 60)
    
    results = executor.execute_parallel(tool_calls)
    for result in results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.tool_name}: {result.duration_ms:.2f}ms")
        if result.success:
            print(f"    Result: {result.result[:50]}...")
```

## Response Time Budgets

In production, you need to guarantee response times. A response time budget enforces limits on how long operations can take.

```python
"""
Response time budgets for predictable agent performance.

Chapter 39: Latency Optimization
"""

import time
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional, TypeVar
from enum import Enum
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


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
        return self.llm_ms + self.tool_ms + self.network_ms + self.buffer_ms


@dataclass
class BudgetStatus:
    """Current status of a time budget."""
    total_budget_ms: float
    elapsed_ms: float
    remaining_ms: float
    allocations_used: dict[str, float] = field(default_factory=dict)
    
    @property
    def is_exceeded(self) -> bool:
        return self.remaining_ms <= 0
    
    @property
    def utilization_pct(self) -> float:
        return (self.elapsed_ms / self.total_budget_ms) * 100 if self.total_budget_ms > 0 else 0


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
        """Check if there's enough budget to continue."""
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
            
            # Check if category allocation exceeded
            if category in self.allocations and self.allocations[category] > 0:
                if self.spent[category] > self.allocations[category]:
                    logger.warning(
                        f"Category '{category}' exceeded allocation: "
                        f"{self.spent[category]:.0f}ms > {self.allocations[category]:.0f}ms"
                    )
            
            self.check_budget()
    
    def get_allocation_remaining(self, category: str) -> float:
        """Get remaining time for a category."""
        if category not in self.allocations:
            return self.get_remaining_ms()
        
        return max(0, self.allocations[category] - self.spent.get(category, 0))


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


class TimeBudgetedAgent:
    """
    Example agent that respects response time budgets.
    """
    
    def __init__(
        self,
        budget_ms: float = 5000,
        on_exceeded: BudgetExceededAction = BudgetExceededAction.WARN
    ):
        self.default_budget_ms = budget_ms
        self.on_exceeded = on_exceeded
    
    def process_with_budget(
        self,
        task: str,
        budget_ms: Optional[float] = None
    ) -> dict[str, Any]:
        """
        Process a task within a time budget.
        
        Returns result along with timing information.
        """
        budget = ResponseTimeBudget(
            total_ms=budget_ms or self.default_budget_ms,
            on_exceeded=self.on_exceeded
        )
        
        # Allocate budget
        budget.allocate(
            llm_ms=budget.total_ms * 0.6,   # 60% for LLM
            tool_ms=budget.total_ms * 0.3,   # 30% for tools
            buffer_ms=budget.total_ms * 0.1  # 10% buffer
        )
        
        budget.start()
        result = {"task": task, "steps": []}
        
        try:
            # Step 1: Planning (simulated LLM call)
            with budget.track("planning", "llm"):
                time.sleep(0.3)  # Simulated
                result["steps"].append("planning")
            
            # Check if we can continue
            if not budget.can_continue(500):
                result["early_exit"] = "insufficient_budget"
                return result
            
            # Step 2: Tool execution
            with budget.track("tool_execution", "tool"):
                time.sleep(0.2)  # Simulated
                result["steps"].append("tool_execution")
            
            # Step 3: Response generation
            with budget.track("response_generation", "llm"):
                time.sleep(0.25)  # Simulated
                result["steps"].append("response_generation")
            
            result["success"] = True
            
        except BudgetExceededException as e:
            result["success"] = False
            result["error"] = str(e)
        
        # Add timing info
        status = budget.get_status()
        result["timing"] = {
            "total_ms": round(status.elapsed_ms, 2),
            "budget_ms": status.total_budget_ms,
            "utilization_pct": round(status.utilization_pct, 1),
            "by_category": {k: round(v, 2) for k, v in status.allocations_used.items()}
        }
        
        return result


# Example usage
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("RESPONSE TIME BUDGET DEMO")
    print("=" * 60)
    
    # Create a budgeted agent
    agent = TimeBudgetedAgent(
        budget_ms=2000,  # 2 second budget
        on_exceeded=BudgetExceededAction.WARN
    )
    
    # Process with comfortable budget
    print("\n1. Processing with comfortable budget (2000ms):")
    result = agent.process_with_budget("Analyze this data")
    print(json.dumps(result, indent=2))
    
    # Process with tight budget
    print("\n2. Processing with tight budget (500ms):")
    result = agent.process_with_budget("Quick task", budget_ms=500)
    print(json.dumps(result, indent=2))
    
    # Demonstrate budget tracking
    print("\n3. Manual budget tracking:")
    print("-" * 40)
    
    budget = ResponseTimeBudget(total_ms=1000)
    budget.allocate(llm_ms=600, tool_ms=300, buffer_ms=100)
    budget.start()
    
    print(f"Starting with {budget.total_ms}ms budget")
    
    with budget.track("operation_1", "llm"):
        time.sleep(0.2)
    print(f"After op 1: {budget.get_remaining_ms():.0f}ms remaining")
    
    with budget.track("operation_2", "tool"):
        time.sleep(0.15)
    print(f"After op 2: {budget.get_remaining_ms():.0f}ms remaining")
    
    status = budget.get_status()
    print(f"\nFinal status:")
    print(f"  Elapsed: {status.elapsed_ms:.0f}ms")
    print(f"  Remaining: {status.remaining_ms:.0f}ms")
    print(f"  Utilization: {status.utilization_pct:.1f}%")
```

## Model Selection for Speed

Different models have different latency characteristics. Choosing the right model can significantly impact response time.

```python
"""
Model selection for optimal latency.

Chapter 39: Latency Optimization
"""

import os
import time
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class TaskComplexity(Enum):
    """Classification of task complexity."""
    SIMPLE = "simple"       # Yes/no, classification, extraction
    MODERATE = "moderate"   # Summarization, simple analysis
    COMPLEX = "complex"     # Multi-step reasoning, creative work


@dataclass
class ModelProfile:
    """Performance profile for a model."""
    name: str
    avg_latency_ms: float      # Average time to first token
    tokens_per_second: float   # Generation speed
    quality_score: float       # Relative quality (0-1)
    cost_per_1k_tokens: float  # Input cost
    best_for: list[str]        # Task types this model excels at


class LatencyAwareModelSelector:
    """
    Selects models based on latency requirements and task complexity.
    
    Balances:
    - Response time requirements
    - Task complexity needs
    - Quality requirements
    - Cost constraints
    
    Usage:
        selector = LatencyAwareModelSelector()
        
        model = selector.select(
            task_complexity=TaskComplexity.SIMPLE,
            max_latency_ms=1000,
            quality_threshold=0.7
        )
    """
    
    # Model profiles (approximate values - actual performance varies)
    MODEL_PROFILES = {
        "claude-3-5-haiku-20241022": ModelProfile(
            name="claude-3-5-haiku-20241022",
            avg_latency_ms=300,
            tokens_per_second=150,
            quality_score=0.75,
            cost_per_1k_tokens=0.0008,
            best_for=["classification", "extraction", "simple_qa", "routing"]
        ),
        "claude-sonnet-4-20250514": ModelProfile(
            name="claude-sonnet-4-20250514",
            avg_latency_ms=500,
            tokens_per_second=100,
            quality_score=0.90,
            cost_per_1k_tokens=0.003,
            best_for=["analysis", "coding", "summarization", "general"]
        ),
        "claude-opus-4-20250514": ModelProfile(
            name="claude-opus-4-20250514",
            avg_latency_ms=800,
            tokens_per_second=60,
            quality_score=0.98,
            cost_per_1k_tokens=0.015,
            best_for=["complex_reasoning", "creative", "research", "difficult_coding"]
        ),
    }
    
    # Task complexity to minimum quality mapping
    COMPLEXITY_QUALITY_MAP = {
        TaskComplexity.SIMPLE: 0.7,
        TaskComplexity.MODERATE: 0.85,
        TaskComplexity.COMPLEX: 0.95,
    }
    
    def __init__(self):
        self.profiles = self.MODEL_PROFILES.copy()
        self._latency_history: dict[str, list[float]] = {}
    
    def select(
        self,
        task_complexity: TaskComplexity = TaskComplexity.MODERATE,
        max_latency_ms: Optional[float] = None,
        quality_threshold: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
        task_type: Optional[str] = None
    ) -> str:
        """
        Select the best model given constraints.
        
        Args:
            task_complexity: How complex is the task
            max_latency_ms: Maximum acceptable latency
            quality_threshold: Minimum quality score required
            max_cost_per_1k: Maximum cost per 1000 tokens
            task_type: Specific task type for optimization
        
        Returns:
            Model name to use
        """
        # Set defaults based on complexity
        if quality_threshold is None:
            quality_threshold = self.COMPLEXITY_QUALITY_MAP[task_complexity]
        
        candidates = []
        
        for name, profile in self.profiles.items():
            # Filter by latency
            if max_latency_ms and profile.avg_latency_ms > max_latency_ms:
                continue
            
            # Filter by quality
            if profile.quality_score < quality_threshold:
                continue
            
            # Filter by cost
            if max_cost_per_1k and profile.cost_per_1k_tokens > max_cost_per_1k:
                continue
            
            # Calculate score
            score = self._calculate_score(
                profile,
                task_type,
                max_latency_ms
            )
            
            candidates.append((name, score, profile))
        
        if not candidates:
            # No model meets all constraints, return best available
            return "claude-sonnet-4-20250514"
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0]
    
    def _calculate_score(
        self,
        profile: ModelProfile,
        task_type: Optional[str],
        max_latency_ms: Optional[float]
    ) -> float:
        """Calculate a selection score for a model."""
        score = 0.0
        
        # Quality contributes most
        score += profile.quality_score * 50
        
        # Latency bonus (lower is better)
        if max_latency_ms:
            latency_ratio = profile.avg_latency_ms / max_latency_ms
            score += (1 - latency_ratio) * 30
        else:
            score += (1 - profile.avg_latency_ms / 1000) * 30
        
        # Task type match bonus
        if task_type and task_type in profile.best_for:
            score += 20
        
        # Speed bonus
        score += min(profile.tokens_per_second / 150, 1) * 10
        
        return score
    
    def record_latency(self, model: str, latency_ms: float) -> None:
        """Record actual latency for a model to improve selection."""
        if model not in self._latency_history:
            self._latency_history[model] = []
        
        self._latency_history[model].append(latency_ms)
        
        # Keep only recent measurements
        if len(self._latency_history[model]) > 100:
            self._latency_history[model] = self._latency_history[model][-100:]
        
        # Update profile with actual measurements
        if model in self.profiles:
            avg = sum(self._latency_history[model]) / len(self._latency_history[model])
            self.profiles[model].avg_latency_ms = avg
    
    def get_recommendation(
        self,
        task_description: str,
        time_budget_ms: float
    ) -> dict[str, Any]:
        """
        Get a detailed model recommendation with explanation.
        """
        # Classify task complexity (simplified)
        complexity = TaskComplexity.MODERATE
        if any(word in task_description.lower() for word in ["simple", "yes/no", "extract", "classify"]):
            complexity = TaskComplexity.SIMPLE
        elif any(word in task_description.lower() for word in ["complex", "analyze", "research", "creative"]):
            complexity = TaskComplexity.COMPLEX
        
        # Determine task type
        task_type = None
        for profile in self.profiles.values():
            for t in profile.best_for:
                if t in task_description.lower():
                    task_type = t
                    break
        
        # Get recommendation
        model = self.select(
            task_complexity=complexity,
            max_latency_ms=time_budget_ms * 0.6,  # Allow 60% for LLM
            task_type=task_type
        )
        
        profile = self.profiles[model]
        
        return {
            "recommended_model": model,
            "task_complexity": complexity.value,
            "detected_task_type": task_type,
            "expected_latency_ms": profile.avg_latency_ms,
            "quality_score": profile.quality_score,
            "fits_budget": profile.avg_latency_ms <= time_budget_ms * 0.6,
            "reasoning": self._generate_reasoning(profile, complexity, time_budget_ms)
        }
    
    def _generate_reasoning(
        self,
        profile: ModelProfile,
        complexity: TaskComplexity,
        budget_ms: float
    ) -> str:
        """Generate explanation for the recommendation."""
        reasons = []
        
        if complexity == TaskComplexity.SIMPLE:
            reasons.append("Task is simple, prioritizing speed")
        elif complexity == TaskComplexity.COMPLEX:
            reasons.append("Task is complex, prioritizing quality")
        
        if profile.avg_latency_ms <= budget_ms * 0.5:
            reasons.append("Model latency well within budget")
        elif profile.avg_latency_ms <= budget_ms * 0.8:
            reasons.append("Model latency fits budget with some margin")
        else:
            reasons.append("Model latency is tight for budget")
        
        return "; ".join(reasons)


class ModelSpeedBenchmark:
    """
    Benchmarks actual model speeds for accurate selection.
    
    Run this periodically to update model profiles with
    real performance data.
    """
    
    TEST_PROMPTS = {
        "simple": "Is Python a programming language? Answer yes or no.",
        "moderate": "Summarize the benefits of cloud computing in 2 sentences.",
        "complex": "Explain the tradeoffs between consistency and availability in distributed systems."
    }
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def benchmark_model(
        self,
        model: str,
        iterations: int = 3
    ) -> dict[str, Any]:
        """
        Benchmark a model's latency.
        
        Args:
            model: Model name to benchmark
            iterations: Number of test iterations
        
        Returns:
            Benchmark results
        """
        results = {
            "model": model,
            "tests": {}
        }
        
        for test_name, prompt in self.TEST_PROMPTS.items():
            latencies = []
            token_counts = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                
                response = self.client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
                token_counts.append(response.usage.output_tokens)
            
            avg_latency = sum(latencies) / len(latencies)
            avg_tokens = sum(token_counts) / len(token_counts)
            
            results["tests"][test_name] = {
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
                "avg_tokens": round(avg_tokens, 1),
                "tokens_per_second": round(avg_tokens / (avg_latency / 1000), 1)
            }
        
        # Calculate overall averages
        all_latencies = [r["avg_latency_ms"] for r in results["tests"].values()]
        results["overall"] = {
            "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 2)
        }
        
        return results


# Example usage
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("LATENCY-AWARE MODEL SELECTION DEMO")
    print("=" * 60)
    
    selector = LatencyAwareModelSelector()
    
    # Test scenarios
    scenarios = [
        {
            "description": "Simple classification task",
            "task_complexity": TaskComplexity.SIMPLE,
            "max_latency_ms": 500,
        },
        {
            "description": "Code analysis task",
            "task_complexity": TaskComplexity.MODERATE,
            "max_latency_ms": 2000,
            "task_type": "coding"
        },
        {
            "description": "Complex research task",
            "task_complexity": TaskComplexity.COMPLEX,
            "max_latency_ms": 5000,
        },
        {
            "description": "Quick extraction with quality requirement",
            "task_complexity": TaskComplexity.SIMPLE,
            "max_latency_ms": 300,
            "quality_threshold": 0.9,  # Need high quality despite simplicity
        },
    ]
    
    print("\nModel Selection Results:")
    print("-" * 60)
    
    for scenario in scenarios:
        model = selector.select(
            task_complexity=scenario["task_complexity"],
            max_latency_ms=scenario.get("max_latency_ms"),
            quality_threshold=scenario.get("quality_threshold"),
            task_type=scenario.get("task_type")
        )
        
        profile = selector.profiles[model]
        
        print(f"\n{scenario['description']}:")
        print(f"  Selected: {model}")
        print(f"  Expected latency: {profile.avg_latency_ms}ms")
        print(f"  Quality score: {profile.quality_score}")
    
    # Get detailed recommendation
    print("\n" + "-" * 60)
    print("Detailed Recommendation:")
    print("-" * 60)
    
    recommendation = selector.get_recommendation(
        task_description="Analyze this code for bugs and suggest improvements",
        time_budget_ms=3000
    )
    
    print(json.dumps(recommendation, indent=2))
```

## Caching for Speed

Caching reduces latency by avoiding redundant work. Here's a speed-optimized cache.

```python
"""
Speed-optimized caching for agent responses.

Chapter 39: Latency Optimization
"""

import hashlib
import time
import json
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    total_saved_ms: float = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class LRUCache:
    """
    Thread-safe LRU cache optimized for low latency.
    
    Features:
    - O(1) get and put operations
    - Automatic expiration
    - Size limits with LRU eviction
    - Hit/miss statistics
    
    Usage:
        cache = LRUCache(max_size=1000, default_ttl=3600)
        
        # Check cache first
        result = cache.get(key)
        if result is None:
            result = expensive_operation()
            cache.put(key, result)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _make_key(self, key: Any) -> str:
        """Convert any key to a string hash."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Returns None if not found or expired.
        """
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[str_key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[str_key]
                self._stats.misses += 1
                return None
            
            # Update access info and move to end (most recently used)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(str_key)
            
            self._stats.hits += 1
            return entry.value
    
    def put(
        self,
        key: Any,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        str_key = self._make_key(key)
        now = time.time()
        
        with self._lock:
            # Remove if exists (to update position)
            if str_key in self._cache:
                del self._cache[str_key]
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
            
            # Add new entry
            entry = CacheEntry(
                key=str_key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl or self.default_ttl
            )
            self._cache[str_key] = entry
            self._stats.size = len(self._cache)
    
    def delete(self, key: Any) -> bool:
        """Delete an entry from the cache."""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key in self._cache:
                del self._cache[str_key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                total_saved_ms=self._stats.total_saved_ms
            )
    
    def record_time_saved(self, ms: float) -> None:
        """Record time saved by a cache hit."""
        with self._lock:
            self._stats.total_saved_ms += ms


class SemanticCache:
    """
    Cache that matches semantically similar queries.
    
    Uses embedding similarity to find cached responses
    for queries that are similar but not identical.
    
    Note: This is a simplified version. In production,
    you'd use a proper embedding model and vector database.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity to return a match
            max_size: Maximum entries to store
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._entries: list[tuple[str, Any, float]] = []  # (query, response, timestamp)
        self._lock = threading.RLock()
    
    def _simple_similarity(self, query1: str, query2: str) -> float:
        """
        Simple word-overlap similarity.
        
        In production, use proper embeddings.
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get(self, query: str) -> Optional[Any]:
        """Get a cached response for a similar query."""
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            for cached_query, response, _ in self._entries:
                similarity = self._simple_similarity(query, cached_query)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = response
            
            if best_similarity >= self.similarity_threshold:
                return best_match
            
            return None
    
    def put(self, query: str, response: Any) -> None:
        """Cache a response for a query."""
        with self._lock:
            # Remove oldest if at capacity
            if len(self._entries) >= self.max_size:
                self._entries.pop(0)
            
            self._entries.append((query, response, time.time()))


class CachedLLMClient:
    """
    LLM client wrapper with automatic caching.
    
    Caches responses to reduce both latency and costs.
    """
    
    def __init__(
        self,
        client: Any,
        cache: Optional[LRUCache] = None,
        cache_ttl: float = 3600
    ):
        """
        Initialize cached client.
        
        Args:
            client: Anthropic client instance
            cache: Cache to use (creates new if None)
            cache_ttl: Default cache TTL in seconds
        """
        self.client = client
        self.cache = cache or LRUCache(max_size=1000, default_ttl=cache_ttl)
        self.cache_ttl = cache_ttl
    
    def _make_cache_key(
        self,
        model: str,
        messages: list[dict],
        **kwargs: Any
    ) -> str:
        """Create a cache key from request parameters."""
        key_data = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k != "stream"}
        }
        return json.dumps(key_data, sort_keys=True)
    
    def create_message(
        self,
        model: str,
        messages: list[dict],
        use_cache: bool = True,
        **kwargs: Any
    ) -> Any:
        """
        Create a message with caching.
        
        Args:
            model: Model to use
            messages: Message history
            use_cache: Whether to use cache
            **kwargs: Additional API parameters
        
        Returns:
            API response (cached or fresh)
        """
        if not use_cache:
            return self.client.messages.create(
                model=model,
                messages=messages,
                **kwargs
            )
        
        cache_key = self._make_cache_key(model, messages, **kwargs)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Make API call
        start_time = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            **kwargs
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Cache the response
        self.cache.put(cache_key, response)
        self.cache.record_time_saved(duration_ms)  # For future hits
        
        return response
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        stats = self.cache.get_stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": f"{stats.hit_rate:.1%}",
            "size": stats.size,
            "total_saved_ms": round(stats.total_saved_ms, 2)
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SPEED-OPTIMIZED CACHE DEMO")
    print("=" * 60)
    
    cache = LRUCache(max_size=100, default_ttl=3600)
    
    # Simulate caching scenario
    print("\nSimulating cache operations:")
    print("-" * 40)
    
    # First request - cache miss
    start = time.perf_counter()
    result = cache.get("query_1")
    print(f"Get 'query_1': {result} (miss)")
    
    # Simulate expensive operation
    time.sleep(0.1)  # 100ms operation
    cache.put("query_1", "expensive result")
    cache.record_time_saved(100)
    
    # Second request - cache hit
    start = time.perf_counter()
    result = cache.get("query_1")
    duration = (time.perf_counter() - start) * 1000
    print(f"Get 'query_1': '{result}' (hit, {duration:.3f}ms)")
    
    # Add more entries
    for i in range(5):
        cache.put(f"query_{i+2}", f"result_{i+2}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats.hits}")
    print(f"  Misses: {stats.misses}")
    print(f"  Hit Rate: {stats.hit_rate:.1%}")
    print(f"  Size: {stats.size}")
    print(f"  Total Saved: {stats.total_saved_ms}ms")
    
    # Demonstrate semantic cache
    print("\n" + "-" * 40)
    print("Semantic Cache Demo:")
    print("-" * 40)
    
    semantic = SemanticCache(similarity_threshold=0.5)
    
    # Cache a response
    semantic.put("What is the weather in New York?", "Sunny, 72°F")
    
    # Try similar queries
    similar_queries = [
        "What's the weather in New York?",
        "Tell me New York weather",
        "How's the weather in Boston?",  # Different city - shouldn't match
    ]
    
    for query in similar_queries:
        result = semantic.get(query)
        print(f"Query: '{query}'")
        print(f"  Result: {result or 'No match'}")
```

## Complete Latency Optimization Module

Let's bring everything together into a comprehensive module.

```python
"""
Complete latency optimization module for AI agents.

Chapter 39: Latency Optimization

This module provides:
- Latency profiling and measurement
- Streaming responses
- Parallel tool execution
- Response time budgets
- Model selection
- Speed-optimized caching
"""

import os
import time
from typing import Any, Callable, Generator, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Import all optimization components
# In practice, these would be in separate files

from latency_profiler import LatencyProfiler, LatencyBreakdown
from streaming_agent import StreamingAgent, StreamMetrics, ProgressIndicator
from parallel_executor import ParallelToolExecutor, ToolResult
from response_budget import ResponseTimeBudget, BudgetExceededAction, AdaptiveTimeout
from model_selector import LatencyAwareModelSelector, TaskComplexity
from speed_cache import LRUCache, CachedLLMClient


@dataclass
class LatencyReport:
    """Complete latency analysis report."""
    total_ms: float
    breakdown: dict[str, float]
    bottleneck: str
    recommendations: list[str]
    cache_stats: dict[str, Any]
    

class LatencyOptimizedAgent:
    """
    An agent with all latency optimizations applied.
    
    Features:
    - Automatic model selection based on task
    - Response caching
    - Parallel tool execution
    - Streaming responses
    - Time budget enforcement
    
    Usage:
        agent = LatencyOptimizedAgent()
        
        # Simple query with automatic optimization
        response = agent.query("What is Python?")
        
        # With specific constraints
        response = agent.query(
            "Analyze this code",
            max_latency_ms=2000,
            stream=True
        )
    """
    
    def __init__(
        self,
        default_model: str = "claude-sonnet-4-20250514",
        enable_cache: bool = True,
        cache_ttl: float = 3600,
        enable_parallel_tools: bool = True
    ):
        self.client = anthropic.Anthropic()
        self.default_model = default_model
        
        # Initialize optimizations
        self.profiler = LatencyProfiler()
        self.model_selector = LatencyAwareModelSelector()
        self.parallel_executor = ParallelToolExecutor() if enable_parallel_tools else None
        
        # Setup caching
        if enable_cache:
            self.cache = LRUCache(max_size=1000, default_ttl=cache_ttl)
            self.cached_client = CachedLLMClient(self.client, self.cache, cache_ttl)
        else:
            self.cache = None
            self.cached_client = None
        
        # Tools registry
        self.tools: list[dict[str, Any]] = []
        self.tool_functions: dict[str, Callable] = {}
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable
    ) -> None:
        """Register a tool with the agent."""
        self.tools.append({
            "name": name,
            "description": description,
            "input_schema": parameters
        })
        self.tool_functions[name] = func
        
        if self.parallel_executor:
            self.parallel_executor.register_tool(name, func)
    
    def query(
        self,
        message: str,
        max_latency_ms: Optional[float] = None,
        stream: bool = False,
        use_cache: bool = True,
        auto_select_model: bool = True,
        task_complexity: Optional[TaskComplexity] = None
    ) -> Any:
        """
        Process a query with latency optimizations.
        
        Args:
            message: User message
            max_latency_ms: Maximum acceptable latency
            stream: Whether to stream the response
            use_cache: Whether to use caching
            auto_select_model: Whether to auto-select model
            task_complexity: Override task complexity detection
        
        Returns:
            Response (generator if streaming, else string)
        """
        self.profiler.reset()
        
        # Select model
        if auto_select_model:
            with self.profiler.measure("model_selection", "other"):
                model = self.model_selector.select(
                    task_complexity=task_complexity or TaskComplexity.MODERATE,
                    max_latency_ms=max_latency_ms
                )
        else:
            model = self.default_model
        
        # Setup budget if specified
        budget = None
        if max_latency_ms:
            budget = ResponseTimeBudget(max_latency_ms, BudgetExceededAction.WARN)
            budget.allocate(
                llm_ms=max_latency_ms * 0.7,
                tool_ms=max_latency_ms * 0.2,
                buffer_ms=max_latency_ms * 0.1
            )
            budget.start()
        
        if stream:
            return self._stream_query(message, model, budget)
        else:
            return self._sync_query(message, model, use_cache, budget)
    
    def _sync_query(
        self,
        message: str,
        model: str,
        use_cache: bool,
        budget: Optional[ResponseTimeBudget]
    ) -> str:
        """Synchronous query processing."""
        # Check cache first
        if use_cache and self.cached_client:
            cache_key = f"{model}:{message}"
            cached = self.cache.get(cache_key)
            if cached:
                self.profiler.record("cache_hit", 1, "cache")
                return cached
        
        messages = [{"role": "user", "content": message}]
        
        # Make LLM call
        with self.profiler.measure("llm_call", "llm"):
            if budget:
                with budget.track("llm_call", "llm"):
                    response = self.client.messages.create(
                        model=model,
                        max_tokens=1024,
                        tools=self.tools if self.tools else None,
                        messages=messages
                    )
            else:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=self.tools if self.tools else None,
                    messages=messages
                )
        
        # Handle tool use
        while response.stop_reason == "tool_use":
            tool_calls = [
                {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                }
                for block in response.content
                if block.type == "tool_use"
            ]
            
            # Execute tools (parallel if available)
            with self.profiler.measure("tool_execution", "tool"):
                if self.parallel_executor and len(tool_calls) > 1:
                    results = self.parallel_executor.execute_parallel(tool_calls)
                else:
                    results = self._execute_tools_sequential(tool_calls)
            
            # Continue conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_use_id,
                        "content": str(r.result) if r.success else f"Error: {r.error}"
                    }
                    for r in results
                ]
            })
            
            with self.profiler.measure("llm_call", "llm"):
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=self.tools,
                    messages=messages
                )
        
        # Extract text
        result = ""
        for block in response.content:
            if hasattr(block, "text"):
                result += block.text
        
        # Cache result
        if use_cache and self.cache:
            cache_key = f"{model}:{message}"
            self.cache.put(cache_key, result)
        
        return result
    
    def _stream_query(
        self,
        message: str,
        model: str,
        budget: Optional[ResponseTimeBudget]
    ) -> Generator[str, None, None]:
        """Streaming query processing."""
        streaming_agent = StreamingAgent(model=model)
        
        for chunk in streaming_agent.stream(message):
            yield chunk
    
    def _execute_tools_sequential(
        self,
        tool_calls: list[dict]
    ) -> list[ToolResult]:
        """Execute tools sequentially (fallback)."""
        results = []
        
        for call in tool_calls:
            start = time.perf_counter()
            
            try:
                if call["name"] in self.tool_functions:
                    result = self.tool_functions[call["name"]](**call["input"])
                    success = True
                    error = None
                else:
                    result = None
                    success = False
                    error = f"Unknown tool: {call['name']}"
            except Exception as e:
                result = None
                success = False
                error = str(e)
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            results.append(ToolResult(
                tool_name=call["name"],
                tool_use_id=call["id"],
                result=result,
                duration_ms=duration_ms,
                success=success,
                error=error
            ))
        
        return results
    
    def get_latency_report(self) -> dict[str, Any]:
        """Get a latency analysis report."""
        breakdown = self.profiler.get_breakdown()
        summary = self.profiler.get_summary()
        
        # Identify bottleneck
        categories = {
            "llm": breakdown.llm_ms,
            "tool": breakdown.tool_ms,
            "network": breakdown.network_ms,
            "other": breakdown.other_ms
        }
        bottleneck = max(categories, key=categories.get)
        
        # Generate recommendations
        recommendations = []
        
        if breakdown.llm_ms > breakdown.total_ms * 0.7:
            recommendations.append("Consider using a faster model for simple tasks")
            recommendations.append("Enable response caching for repeated queries")
        
        if breakdown.tool_ms > breakdown.total_ms * 0.3:
            recommendations.append("Enable parallel tool execution")
            recommendations.append("Add caching for expensive tool calls")
        
        if breakdown.llm_call_count > 3:
            recommendations.append("Consider combining multiple LLM calls")
        
        return {
            "total_ms": round(breakdown.total_ms, 2),
            "breakdown": breakdown.to_dict(),
            "bottleneck": bottleneck,
            "recommendations": recommendations,
            "cache_stats": self.cached_client.get_cache_stats() if self.cached_client else None
        }


# Example usage
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("LATENCY-OPTIMIZED AGENT DEMO")
    print("=" * 60)
    
    # Create optimized agent
    agent = LatencyOptimizedAgent(
        enable_cache=True,
        enable_parallel_tools=True
    )
    
    # Register a sample tool
    def get_time(timezone: str = "UTC") -> str:
        import datetime
        return f"Current time in {timezone}: {datetime.datetime.now()}"
    
    agent.register_tool(
        name="get_time",
        description="Get the current time",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone name"}
            }
        },
        func=get_time
    )
    
    # Make a query
    print("\nMaking optimized query...")
    print("-" * 40)
    
    response = agent.query(
        "What is Python programming language?",
        max_latency_ms=3000,
        auto_select_model=True
    )
    
    print(f"\nResponse: {response[:200]}...")
    
    # Get latency report
    print("\n" + "-" * 40)
    print("Latency Report:")
    print("-" * 40)
    
    report = agent.get_latency_report()
    print(json.dumps(report, indent=2))
    
    # Make same query again (should hit cache)
    print("\n" + "-" * 40)
    print("Making same query again (should hit cache)...")
    
    start = time.perf_counter()
    response2 = agent.query("What is Python programming language?")
    duration = (time.perf_counter() - start) * 1000
    
    print(f"Second query time: {duration:.2f}ms")
```

## Common Pitfalls

**1. Over-optimizing before measuring**

Don't guess where time is spent. Profile first, then optimize the actual bottlenecks. Often the slowest component isn't what you expect.

**2. Caching mutable responses**

Be careful caching responses that should change:
- Time-dependent queries ("What time is it?")
- Personalized responses
- Real-time data

**3. Parallel execution with dependencies**

Not all tool calls can run in parallel. If tool B depends on tool A's output, you must run them sequentially.

**4. Setting budgets too tight**

Unrealistic time budgets cause constant failures. Start generous and tighten based on actual performance data.

**5. Ignoring cold start latency**

The first request is often slower (connection setup, model loading). Don't optimize based only on warm requests.

## Practical Exercise

**Task:** Build a latency dashboard that visualizes agent performance in real-time

**Requirements:**

1. Create an HTML dashboard that displays:
   - Current response time distribution (histogram)
   - Breakdown by category (LLM, tools, network)
   - Cache hit rate over time
   - Slowest operations

2. Use the `LatencyProfiler` to collect data

3. Auto-refresh every 5 seconds

4. Include alerts when latency exceeds thresholds

**Hints:**

- Use the profiler's `get_summary()` method for data
- Store historical data for trend visualization
- Consider using Chart.js for graphs

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Measure before optimizing** — use the `LatencyProfiler` to identify actual bottlenecks
- **Streaming transforms perceived performance** — users see progress immediately even if total time is unchanged
- **Parallel execution can dramatically reduce tool latency** — independent operations should run simultaneously
- **Response time budgets enforce predictability** — set limits and handle exceeded budgets gracefully
- **Model selection matters for speed** — use Haiku for simple tasks, Sonnet for most work, Opus only when quality demands it
- **Caching eliminates redundant work** — cache both LLM responses and tool results
- **Users perceive speed differently than metrics show** — optimize for user experience, not just milliseconds

## What's Next

With your agents optimized for both cost and latency, you're ready to deploy them. The next chapter, **Deployment Patterns**, covers running agents in production: REST APIs with FastAPI, background workers, containerization with Docker, and scaling strategies. You'll learn to make your agents available to real users reliably.
