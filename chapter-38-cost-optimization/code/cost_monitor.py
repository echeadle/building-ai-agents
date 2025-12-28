"""
Cost monitoring and alerting system.

Chapter 38: Cost Optimization
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Callable


@dataclass
class CostAlert:
    alert_type: str
    message: str
    current_value: float
    threshold: float
    severity: str


class CostMonitor:
    """Comprehensive cost monitoring for AI agents."""
    
    PRICING = {
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    }
    
    def __init__(
        self,
        daily_budget: Optional[float] = None,
        alert_thresholds: Optional[list[float]] = None,
        alert_callback: Optional[Callable[[CostAlert], None]] = None,
    ):
        self.daily_budget = daily_budget
        self.alert_thresholds = alert_thresholds or [50, 75, 90, 100]
        self.alert_callback = alert_callback
        self.total_cost = 0.0
        self.total_calls = 0
        self.usage_by_model: dict[str, dict] = {}
        self.triggered_alerts: set[str] = set()
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(model, self.PRICING["claude-sonnet-4-20250514"])
        return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
    
    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_calls += 1
        
        if model not in self.usage_by_model:
            self.usage_by_model[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        
        self.usage_by_model[model]["calls"] += 1
        self.usage_by_model[model]["input_tokens"] += input_tokens
        self.usage_by_model[model]["output_tokens"] += output_tokens
        self.usage_by_model[model]["cost"] += cost
        
        self._check_alerts()
        return cost
    
    def _check_alerts(self):
        if not self.daily_budget:
            return
        
        pct = (self.total_cost / self.daily_budget) * 100
        
        for threshold in self.alert_thresholds:
            alert_key = f"daily_{threshold}"
            if pct >= threshold and alert_key not in self.triggered_alerts:
                severity = "critical" if threshold >= 100 else "warning"
                alert = CostAlert(
                    alert_type="daily_budget",
                    message=f"Daily spend at {pct:.1f}% of ${self.daily_budget:.2f}",
                    current_value=self.total_cost,
                    threshold=self.daily_budget * (threshold / 100),
                    severity=severity
                )
                self.triggered_alerts.add(alert_key)
                if self.alert_callback:
                    self.alert_callback(alert)
    
    def get_report(self) -> str:
        lines = [
            "=" * 50,
            "COST REPORT",
            "=" * 50,
            f"Total calls: {self.total_calls}",
            f"Total cost: ${self.total_cost:.4f}",
        ]
        
        if self.daily_budget:
            remaining = max(0, self.daily_budget - self.total_cost)
            lines.append(f"Budget remaining: ${remaining:.4f}")
        
        if self.usage_by_model:
            lines.append("\nBy Model:")
            for model, stats in self.usage_by_model.items():
                name = "Haiku" if "haiku" in model else ("Sonnet" if "sonnet" in model else "Opus")
                lines.append(f"  {name}: {stats['calls']} calls, ${stats['cost']:.4f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


if __name__ == "__main__":
    def alert_handler(alert):
        icon = "🚨" if alert.severity == "critical" else "⚠️"
        print(f"{icon} {alert.message}")
    
    monitor = CostMonitor(daily_budget=10.0, alert_callback=alert_handler)
    
    print("Cost Monitor Demo")
    print("=" * 40)
    
    calls = [
        ("claude-sonnet-4-20250514", 500, 200),
        ("claude-3-5-haiku-20241022", 200, 100),
        ("claude-sonnet-4-20250514", 800, 300),
    ]
    
    for model, inp, out in calls:
        cost = monitor.record_usage(model, inp, out)
        print(f"Recorded: ${cost:.4f}")
    
    print(monitor.get_report())
