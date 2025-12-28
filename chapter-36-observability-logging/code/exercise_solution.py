"""
Exercise Solution: Agent Metrics Dashboard

Chapter 36: Observability and Logging

This solution creates a simple web-based dashboard that displays
real-time agent metrics. It uses a basic HTTP server to serve
an HTML page that auto-refreshes metrics every 5 seconds.

To run:
    python exercise_solution.py
    
Then open http://localhost:8000 in your browser.
"""

import json
import random
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any
from datetime import datetime, timezone


# =============================================================================
# Simulated Agent Metrics (same as example_05)
# =============================================================================

class MetricsCollector:
    """Simple metrics collector for the demo."""
    
    def __init__(self):
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
    
    def increment(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0) + value
    
    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0)
    
    def observe(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        # Keep only last 1000 observations
        if len(self._histograms[name]) > 1000:
            self._histograms[name] = self._histograms[name][-1000:]
    
    def get_histogram_stats(self, name: str) -> dict[str, float]:
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0, "mean": 0, "min": 0, "max": 0, "p95": 0}
        
        sorted_vals = sorted(values)
        p95_idx = int(len(sorted_vals) * 0.95)
        
        return {
            "count": len(values),
            "mean": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "p95": round(sorted_vals[min(p95_idx, len(sorted_vals)-1)], 2)
        }


class AgentMetrics:
    """Agent-specific metrics wrapper."""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self._tool_calls: list[dict[str, Any]] = []
    
    def record_request(self, success: bool = True) -> None:
        self.collector.increment("requests_total")
        if success:
            self.collector.increment("requests_success")
        else:
            self.collector.increment("requests_failed")
    
    def record_latency(self, duration_ms: float) -> None:
        self.collector.observe("latency_ms", duration_ms)
    
    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.collector.increment("tokens_input", input_tokens)
        self.collector.increment("tokens_output", output_tokens)
    
    def record_tool_call(self, tool_name: str, duration_ms: float, success: bool) -> None:
        self.collector.increment(f"tool_{tool_name}_total")
        if not success:
            self.collector.increment(f"tool_{tool_name}_errors")
        
        self._tool_calls.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "duration_ms": round(duration_ms, 2),
            "success": success
        })
        # Keep only last 50 tool calls
        if len(self._tool_calls) > 50:
            self._tool_calls = self._tool_calls[-50:]
    
    def get_dashboard_data(self) -> dict[str, Any]:
        total = self.collector.get_counter("requests_total")
        success = self.collector.get_counter("requests_success")
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requests": {
                "total": int(total),
                "success": int(success),
                "failed": int(total - success),
                "success_rate": round(success / total * 100, 1) if total > 0 else 0
            },
            "tokens": {
                "input": int(self.collector.get_counter("tokens_input")),
                "output": int(self.collector.get_counter("tokens_output")),
                "total": int(self.collector.get_counter("tokens_input") + 
                           self.collector.get_counter("tokens_output"))
            },
            "latency": self.collector.get_histogram_stats("latency_ms"),
            "recent_tool_calls": self._tool_calls[-10:][::-1]  # Last 10, newest first
        }


# Global metrics instance
metrics = AgentMetrics()


# =============================================================================
# Background Agent Simulator
# =============================================================================

def simulate_agent_activity():
    """Simulate agent activity in the background."""
    tools = ["weather", "calculator", "search", "database"]
    
    while True:
        # Simulate a request
        success = random.random() > 0.1  # 90% success rate
        metrics.record_request(success=success)
        
        # Simulate latency
        latency = max(50, random.gauss(500, 150))
        metrics.record_latency(latency)
        
        # Simulate token usage
        metrics.record_tokens(
            input_tokens=random.randint(100, 500),
            output_tokens=random.randint(50, 300)
        )
        
        # Simulate tool calls (70% of requests use tools)
        if random.random() > 0.3:
            tool = random.choice(tools)
            tool_success = random.random() > 0.05  # 95% tool success
            tool_duration = max(10, random.gauss(100, 30))
            metrics.record_tool_call(tool, tool_duration, tool_success)
        
        # Wait before next request
        time.sleep(random.uniform(1, 3))


# =============================================================================
# HTTP Server
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Metrics Dashboard</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .card {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label {
            color: #aaa;
        }
        .metric-value {
            font-weight: bold;
            color: #00d4ff;
        }
        .metric-value.success {
            color: #00ff88;
        }
        .metric-value.error {
            color: #ff4757;
        }
        .tool-call {
            background: #0f3460;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .tool-call .tool-name {
            font-weight: bold;
            color: #00d4ff;
        }
        .tool-call .tool-time {
            color: #888;
            font-size: 0.8em;
        }
        .tool-call .tool-status {
            float: right;
        }
        .tool-call .tool-status.success {
            color: #00ff88;
        }
        .tool-call .tool-status.error {
            color: #ff4757;
        }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .big-number {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            padding: 20px 0;
        }
    </style>
</head>
<body>
    <h1>🤖 Agent Metrics Dashboard</h1>
    
    <div class="dashboard">
        <div class="card">
            <h2>📊 Requests</h2>
            <div class="big-number" style="color: #00d4ff;" id="total-requests">-</div>
            <div class="metric">
                <span class="metric-label">Successful</span>
                <span class="metric-value success" id="success-requests">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Failed</span>
                <span class="metric-value error" id="failed-requests">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Success Rate</span>
                <span class="metric-value" id="success-rate">-</span>
            </div>
        </div>
        
        <div class="card">
            <h2>⏱️ Latency (ms)</h2>
            <div class="metric">
                <span class="metric-label">Mean</span>
                <span class="metric-value" id="latency-mean">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Min</span>
                <span class="metric-value" id="latency-min">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Max</span>
                <span class="metric-value" id="latency-max">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">P95</span>
                <span class="metric-value" id="latency-p95">-</span>
            </div>
        </div>
        
        <div class="card">
            <h2>🔤 Token Usage</h2>
            <div class="big-number" style="color: #00ff88;" id="total-tokens">-</div>
            <div class="metric">
                <span class="metric-label">Input Tokens</span>
                <span class="metric-value" id="input-tokens">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">Output Tokens</span>
                <span class="metric-value" id="output-tokens">-</span>
            </div>
        </div>
        
        <div class="card" style="grid-column: span 1;">
            <h2>🔧 Recent Tool Calls</h2>
            <div id="tool-calls">
                <p style="color: #666;">No tool calls yet...</p>
            </div>
        </div>
    </div>
    
    <p class="refresh-info">Auto-refreshing every 5 seconds | Last update: <span id="last-update">-</span></p>
    
    <script>
        function updateDashboard() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    // Requests
                    document.getElementById('total-requests').textContent = data.requests.total;
                    document.getElementById('success-requests').textContent = data.requests.success;
                    document.getElementById('failed-requests').textContent = data.requests.failed;
                    document.getElementById('success-rate').textContent = data.requests.success_rate + '%';
                    
                    // Latency
                    document.getElementById('latency-mean').textContent = data.latency.mean + ' ms';
                    document.getElementById('latency-min').textContent = data.latency.min + ' ms';
                    document.getElementById('latency-max').textContent = data.latency.max + ' ms';
                    document.getElementById('latency-p95').textContent = data.latency.p95 + ' ms';
                    
                    // Tokens
                    document.getElementById('total-tokens').textContent = data.tokens.total.toLocaleString();
                    document.getElementById('input-tokens').textContent = data.tokens.input.toLocaleString();
                    document.getElementById('output-tokens').textContent = data.tokens.output.toLocaleString();
                    
                    // Tool calls
                    const toolCallsDiv = document.getElementById('tool-calls');
                    if (data.recent_tool_calls.length > 0) {
                        toolCallsDiv.innerHTML = data.recent_tool_calls.map(tc => `
                            <div class="tool-call">
                                <span class="tool-name">${tc.tool}</span>
                                <span class="tool-status ${tc.success ? 'success' : 'error'}">
                                    ${tc.success ? '✓' : '✗'}
                                </span>
                                <br>
                                <span class="tool-time">${tc.duration_ms}ms - ${new Date(tc.timestamp).toLocaleTimeString()}</span>
                            </div>
                        `).join('');
                    }
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(err => console.error('Error fetching metrics:', err));
        }
        
        // Initial update
        updateDashboard();
        
        // Refresh every 5 seconds
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        
        elif self.path == '/api/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = metrics.get_dashboard_data()
            self.wfile.write(json.dumps(data).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Agent Metrics Dashboard")
    print("=" * 60)
    print()
    print("Starting dashboard server...")
    print()
    
    # Start background agent simulator
    simulator_thread = threading.Thread(target=simulate_agent_activity, daemon=True)
    simulator_thread.start()
    print("✓ Agent simulator started (generating fake metrics)")
    
    # Start HTTP server
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print("✓ Dashboard server started")
    print()
    print("-" * 60)
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    print()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
