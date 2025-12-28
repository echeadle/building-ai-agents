"""
Performance metrics collection for AI agents.

Chapter 36: Observability and Logging

This module provides a metrics collection system for monitoring
agent performance, including counters, gauges, and histograms.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class MetricPoint:
    """A single metric measurement with timestamp and labels."""
    value: float
    timestamp: str
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    
    Supports three types of metrics:
    
    1. Counters: Values that only increase (e.g., total requests)
    2. Gauges: Point-in-time values (e.g., active connections)
    3. Histograms: Distribution of values (e.g., latencies)
    
    Usage:
        metrics = MetricsCollector()
        
        # Count things
        metrics.increment("requests_total", labels={"endpoint": "/chat"})
        
        # Track distributions
        metrics.observe("latency_ms", 150.5, labels={"operation": "llm_call"})
        
        # Get summaries
        print(metrics.get_histogram_stats("latency_ms"))
    """
    
    def __init__(self):
        # Counters: {metric_name: {labels_key: value}}
        self._counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Gauges: {metric_name: {labels_key: value}}
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Histograms: {metric_name: {labels_key: [values]}}
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
    
    def _labels_key(self, labels: Optional[dict[str, str]] = None) -> str:
        """Convert labels dict to a hashable string key."""
        if not labels:
            return "__default__"
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    # -------------------------------------------------------------------------
    # Counters
    # -------------------------------------------------------------------------
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Increment a counter.
        
        Counters are monotonically increasing values, useful for:
        - Total requests
        - Total errors
        - Total tokens consumed
        
        Args:
            name: Metric name
            value: Amount to increment by (default: 1)
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._counters[name][key] += value
    
    def get_counter(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> float:
        """Get the current value of a counter."""
        key = self._labels_key(labels)
        return self._counters[name][key]
    
    # -------------------------------------------------------------------------
    # Gauges
    # -------------------------------------------------------------------------
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Set a gauge to a specific value.
        
        Gauges are point-in-time values that can go up or down:
        - Active connections
        - Queue depth
        - Memory usage
        
        Args:
            name: Metric name
            value: The value to set
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._gauges[name][key] = value
    
    def increment_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """Increment a gauge by a value."""
        key = self._labels_key(labels)
        self._gauges[name][key] += value
    
    def decrement_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """Decrement a gauge by a value."""
        key = self._labels_key(labels)
        self._gauges[name][key] -= value
    
    def get_gauge(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> float:
        """Get the current value of a gauge."""
        key = self._labels_key(labels)
        return self._gauges[name][key]
    
    # -------------------------------------------------------------------------
    # Histograms
    # -------------------------------------------------------------------------
    
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """
        Record an observation in a histogram.
        
        Histograms track the distribution of values:
        - Request latencies
        - Response sizes
        - Token counts
        
        Args:
            name: Metric name
            value: The observed value
            labels: Optional labels for this metric
        """
        key = self._labels_key(labels)
        self._histograms[name][key].append(value)
    
    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None
    ) -> dict[str, float]:
        """
        Get statistics for a histogram.
        
        Returns:
            Dictionary with count, sum, mean, min, max, p50, p95, p99
        """
        key = self._labels_key(labels)
        values = self._histograms[name][key]
        
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stddev": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        
        sorted_values = sorted(values)
        
        def percentile(p: float) -> float:
            """Calculate the p-th percentile."""
            if not sorted_values:
                return 0.0
            idx = int(len(sorted_values) * p / 100)
            return sorted_values[min(idx, len(sorted_values) - 1)]
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics in a structured format."""
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        
        # Counters
        for name, label_values in self._counters.items():
            result["counters"][name] = dict(label_values)
        
        # Gauges
        for name, label_values in self._gauges.items():
            result["gauges"][name] = dict(label_values)
        
        # Histograms (with stats)
        for name, label_values in self._histograms.items():
            result["histograms"][name] = {}
            for key in label_values:
                # Reconstruct labels from key
                if key == "__default__":
                    result["histograms"][name][key] = self.get_histogram_stats(name)
                else:
                    result["histograms"][name][key] = self.get_histogram_stats(name)
        
        return result
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class AgentMetrics:
    """
    Pre-defined metrics for AI agent monitoring.
    
    This wraps MetricsCollector with agent-specific metric names
    and helper methods for common operations.
    
    Metrics included:
    - agent_requests_total: Total number of requests
    - agent_requests_success: Successful requests
    - agent_requests_failed: Failed requests
    - agent_latency_ms: Request latency distribution
    - agent_tokens_input: Total input tokens
    - agent_tokens_output: Total output tokens
    - agent_tool_calls_total: Total tool calls
    - agent_llm_calls_total: Total LLM API calls
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
    
    def record_request(self, success: bool = True) -> None:
        """
        Record a completed request.
        
        Args:
            success: Whether the request was successful
        """
        self.collector.increment("agent_requests_total")
        if success:
            self.collector.increment("agent_requests_success")
        else:
            self.collector.increment("agent_requests_failed")
    
    def record_latency(self, duration_ms: float, operation: str = "total") -> None:
        """
        Record latency for an operation.
        
        Args:
            duration_ms: Duration in milliseconds
            operation: Type of operation (e.g., "total", "llm_call", "tool_call")
        """
        self.collector.observe(
            "agent_latency_ms",
            duration_ms,
            labels={"operation": operation}
        )
    
    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.collector.increment("agent_tokens_input", input_tokens)
        self.collector.increment("agent_tokens_output", output_tokens)
        self.collector.increment("agent_tokens_total", input_tokens + output_tokens)
    
    def record_tool_call(
        self,
        tool_name: str,
        success: bool = True,
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Record a tool call.
        
        Args:
            tool_name: Name of the tool
            success: Whether the call was successful
            duration_ms: Optional duration of the call
        """
        status = "success" if success else "error"
        self.collector.increment(
            "agent_tool_calls_total",
            labels={"tool": tool_name, "status": status}
        )
        
        if duration_ms is not None:
            self.collector.observe(
                "agent_tool_latency_ms",
                duration_ms,
                labels={"tool": tool_name}
            )
    
    def record_llm_call(
        self,
        model: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """
        Record an LLM API call.
        
        Args:
            model: The model identifier
            duration_ms: How long the call took
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.collector.increment("agent_llm_calls_total", labels={"model": model})
        self.collector.observe(
            "agent_llm_latency_ms",
            duration_ms,
            labels={"model": model}
        )
        
        if input_tokens or output_tokens:
            self.record_tokens(input_tokens, output_tokens)
    
    def set_active_requests(self, count: int) -> None:
        """Set the current number of active requests (gauge)."""
        self.collector.set_gauge("agent_active_requests", count)
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of agent metrics.
        
        Returns:
            Dictionary with key metrics and statistics
        """
        total = self.collector.get_counter("agent_requests_total")
        success = self.collector.get_counter("agent_requests_success")
        
        return {
            "requests": {
                "total": int(total),
                "success": int(success),
                "failed": int(total - success),
                "success_rate_pct": round(success / total * 100, 1) if total > 0 else 0.0,
            },
            "tokens": {
                "input": int(self.collector.get_counter("agent_tokens_input")),
                "output": int(self.collector.get_counter("agent_tokens_output")),
                "total": int(self.collector.get_counter("agent_tokens_total")),
            },
            "latency": self.collector.get_histogram_stats("agent_latency_ms"),
            "llm_latency": self.collector.get_histogram_stats("agent_llm_latency_ms"),
            "llm_calls": int(self.collector.get_counter("agent_llm_calls_total")),
            "tool_calls": self._get_tool_call_summary(),
        }
    
    def _get_tool_call_summary(self) -> dict[str, int]:
        """Get a breakdown of tool calls by tool name."""
        summary = {}
        for key, value in self.collector._counters["agent_tool_calls_total"].items():
            if key != "__default__":
                # Parse the key to extract tool name
                parts = dict(p.split("=") for p in key.split(","))
                tool = parts.get("tool", "unknown")
                summary[tool] = summary.get(tool, 0) + int(value)
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.collector.reset()


# Example usage
if __name__ == "__main__":
    import json
    import random
    
    print("=" * 70)
    print("Agent Metrics Collection Demo")
    print("=" * 70)
    print()
    
    metrics = AgentMetrics()
    
    # Simulate agent activity
    print("Simulating 100 agent requests...")
    print()
    
    for i in range(100):
        # Simulate request success/failure (90% success rate)
        success = random.random() > 0.1
        metrics.record_request(success=success)
        
        # Simulate latency (mean 500ms, stddev 150ms)
        latency = max(50, random.gauss(500, 150))
        metrics.record_latency(latency, operation="total")
        
        # Simulate token usage
        input_tokens = random.randint(100, 500)
        output_tokens = random.randint(50, 300)
        
        # Simulate LLM call
        llm_latency = max(100, random.gauss(400, 100))
        metrics.record_llm_call(
            model="claude-sonnet-4-20250514",
            duration_ms=llm_latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Simulate tool calls (50% of requests use tools)
        if random.random() > 0.5:
            tool = random.choice(["weather", "calculator", "search", "database"])
            tool_success = random.random() > 0.05  # 95% tool success rate
            tool_latency = max(10, random.gauss(100, 30))
            metrics.record_tool_call(
                tool_name=tool,
                success=tool_success,
                duration_ms=tool_latency
            )
    
    # Print summary
    print("Metrics Summary:")
    print("-" * 70)
    print(json.dumps(metrics.get_summary(), indent=2))
    
    print()
    print("-" * 70)
    print("Full Metrics Export:")
    print("-" * 70)
    print(json.dumps(metrics.collector.get_all_metrics(), indent=2))
