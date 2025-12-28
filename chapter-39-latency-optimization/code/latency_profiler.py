"""
Latency profiling for AI agents.

Chapter 39: Latency Optimization

This module provides tools for measuring and analyzing latency
in agent operations to identify bottlenecks.
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
        """Convert to dictionary for JSON serialization."""
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
        """Initialize the profiler."""
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
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            category: Category for aggregation
            **metadata: Additional metadata
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
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Operation name to get stats for
        
        Returns:
            Dictionary with count, total, mean, min, max, percentiles
        """
        durations = [
            r.duration_ms for r in self.records
            if r.operation == operation
        ]
        
        if not durations:
            return {"count": 0}
        
        sorted_durations = sorted(durations)
        
        return {
            "count": len(durations),
            "total_ms": round(sum(durations), 2),
            "mean_ms": round(statistics.mean(durations), 2),
            "min_ms": round(min(durations), 2),
            "max_ms": round(max(durations), 2),
            "p50_ms": round(statistics.median(durations), 2),
            "p95_ms": round(sorted_durations[int(len(sorted_durations) * 0.95)] if len(sorted_durations) >= 20 else max(durations), 2),
            "p99_ms": round(sorted_durations[int(len(sorted_durations) * 0.99)] if len(sorted_durations) >= 100 else max(durations), 2),
        }
    
    def get_slowest_operations(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Get the N slowest operations.
        
        Args:
            n: Number of operations to return
        
        Returns:
            List of slowest operations with details
        """
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
                "timestamp": r.timestamp
            }
            for r in sorted_records[:n]
        ]
    
    def get_summary(self) -> dict[str, Any]:
        """Get a complete profiling summary."""
        breakdown = self.get_breakdown()
        
        # Get unique operations
        unique_ops = set(r.operation for r in self.records)
        
        return {
            "breakdown": breakdown.to_dict(),
            "slowest_operations": self.get_slowest_operations(),
            "total_operations": len(self.records),
            "unique_operations": len(unique_ops),
            "by_operation": {
                op: self.get_operation_stats(op)
                for op in unique_ops
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
        
        if breakdown.total_ms == 0:
            print("\nNo data recorded yet.")
            return
        
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
    
    def export_json(self, filepath: str) -> None:
        """Export profiling data to JSON file."""
        data = {
            "summary": self.get_summary(),
            "records": [
                {
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                }
                for r in self.records
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


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
        
        # Simulate network operation
        with profiler.measure("external_service", category="network"):
            time.sleep(random.uniform(0.05, 0.1))
        
        # Simulate another LLM call
        with profiler.measure("llm_response", category="llm"):
            time.sleep(random.uniform(0.4, 0.8))
    
    profiler.print_report()
    
    # Print detailed summary
    print("\nDetailed Summary:")
    print(json.dumps(profiler.get_summary(), indent=2))
