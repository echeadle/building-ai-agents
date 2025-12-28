"""
Exercise Solution: Latency Dashboard

Chapter 39: Latency Optimization

This solution creates a web-based latency dashboard that displays:
- Response time distribution (histogram)
- Breakdown by category (LLM, tools, network)
- Cache hit rate over time
- Slowest operations
- Alerts when latency exceeds thresholds

Run with: python exercise_solution.py
Then open: http://localhost:8080
"""

import os
import json
import time
import random
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any
from dataclasses import dataclass, field, asdict

# Import our latency optimization modules
# In practice, these would be imported from your package
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from latency_profiler import LatencyProfiler


@dataclass
class LatencyRecord:
    """A single latency measurement for the dashboard."""
    timestamp: str
    operation: str
    category: str
    duration_ms: float
    

@dataclass
class DashboardData:
    """Data structure for the latency dashboard."""
    # Current statistics
    total_requests: int = 0
    avg_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    
    # Breakdown by category
    llm_total_ms: float = 0
    tool_total_ms: float = 0
    network_total_ms: float = 0
    other_total_ms: float = 0
    
    # Cache stats
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Recent records for histogram
    recent_latencies: list[float] = field(default_factory=list)
    
    # Slowest operations
    slowest_operations: list[dict] = field(default_factory=list)
    
    # Alerts
    alerts: list[dict] = field(default_factory=list)
    
    # Thresholds
    warning_threshold_ms: float = 2000
    critical_threshold_ms: float = 5000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        cache_total = self.cache_hits + self.cache_misses
        return {
            "total_requests": self.total_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "breakdown": {
                "llm_ms": round(self.llm_total_ms, 2),
                "tool_ms": round(self.tool_total_ms, 2),
                "network_ms": round(self.network_total_ms, 2),
                "other_ms": round(self.other_total_ms, 2),
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hits / cache_total * 100, 1) if cache_total > 0 else 0,
            },
            "recent_latencies": self.recent_latencies[-100:],  # Last 100
            "slowest_operations": self.slowest_operations[:10],  # Top 10
            "alerts": self.alerts[-20:],  # Last 20 alerts
            "thresholds": {
                "warning_ms": self.warning_threshold_ms,
                "critical_ms": self.critical_threshold_ms,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


class LatencyDashboard:
    """
    Collects and serves latency metrics for visualization.
    
    Usage:
        dashboard = LatencyDashboard()
        
        # Record metrics
        dashboard.record_latency("llm_call", "llm", 450.5)
        dashboard.record_latency("tool_execution", "tool", 120.3)
        
        # Get data for display
        data = dashboard.get_data()
    """
    
    def __init__(
        self,
        warning_threshold_ms: float = 2000,
        critical_threshold_ms: float = 5000,
        max_history: int = 1000
    ):
        """
        Initialize the dashboard.
        
        Args:
            warning_threshold_ms: Latency threshold for warnings
            critical_threshold_ms: Latency threshold for critical alerts
            max_history: Maximum number of records to keep
        """
        self.data = DashboardData(
            warning_threshold_ms=warning_threshold_ms,
            critical_threshold_ms=critical_threshold_ms
        )
        self.max_history = max_history
        self._all_latencies: list[float] = []
        self._lock = threading.Lock()
    
    def record_latency(
        self,
        operation: str,
        category: str,
        duration_ms: float
    ) -> None:
        """
        Record a latency measurement.
        
        Args:
            operation: Name of the operation
            category: Category (llm, tool, network, other)
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            self.data.total_requests += 1
            
            # Add to history
            self._all_latencies.append(duration_ms)
            self.data.recent_latencies.append(duration_ms)
            
            # Trim history
            if len(self._all_latencies) > self.max_history:
                self._all_latencies = self._all_latencies[-self.max_history:]
            if len(self.data.recent_latencies) > 100:
                self.data.recent_latencies = self.data.recent_latencies[-100:]
            
            # Update category totals
            if category == "llm":
                self.data.llm_total_ms += duration_ms
            elif category == "tool":
                self.data.tool_total_ms += duration_ms
            elif category == "network":
                self.data.network_total_ms += duration_ms
            else:
                self.data.other_total_ms += duration_ms
            
            # Update statistics
            self._update_stats()
            
            # Track slowest operations
            self._update_slowest(operation, category, duration_ms)
            
            # Check thresholds and create alerts
            self._check_thresholds(operation, duration_ms)
    
    def record_cache_access(self, hit: bool) -> None:
        """Record a cache hit or miss."""
        with self._lock:
            if hit:
                self.data.cache_hits += 1
            else:
                self.data.cache_misses += 1
    
    def _update_stats(self) -> None:
        """Update aggregate statistics."""
        if not self._all_latencies:
            return
        
        sorted_latencies = sorted(self._all_latencies)
        n = len(sorted_latencies)
        
        self.data.avg_latency_ms = sum(sorted_latencies) / n
        self.data.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n >= 20 else max(sorted_latencies)
        self.data.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n >= 100 else max(sorted_latencies)
    
    def _update_slowest(self, operation: str, category: str, duration_ms: float) -> None:
        """Update list of slowest operations."""
        entry = {
            "operation": operation,
            "category": category,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.data.slowest_operations.append(entry)
        self.data.slowest_operations.sort(key=lambda x: x["duration_ms"], reverse=True)
        self.data.slowest_operations = self.data.slowest_operations[:10]
    
    def _check_thresholds(self, operation: str, duration_ms: float) -> None:
        """Check thresholds and create alerts if exceeded."""
        if duration_ms >= self.data.critical_threshold_ms:
            self.data.alerts.append({
                "level": "critical",
                "message": f"Critical latency: {operation} took {duration_ms:.0f}ms",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        elif duration_ms >= self.data.warning_threshold_ms:
            self.data.alerts.append({
                "level": "warning",
                "message": f"High latency: {operation} took {duration_ms:.0f}ms",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Trim alerts
        if len(self.data.alerts) > 50:
            self.data.alerts = self.data.alerts[-50:]
    
    def get_data(self) -> dict[str, Any]:
        """Get dashboard data as dictionary."""
        with self._lock:
            return self.data.to_dict()
    
    def reset(self) -> None:
        """Reset all dashboard data."""
        with self._lock:
            self.data = DashboardData(
                warning_threshold_ms=self.data.warning_threshold_ms,
                critical_threshold_ms=self.data.critical_threshold_ms
            )
            self._all_latencies = []


# Global dashboard instance
dashboard = LatencyDashboard()


# HTML template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Latency Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
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
            font-size: 1.1em;
            border-bottom: 1px solid #0f3460;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
        }
        .metric-label {
            color: #888;
        }
        .metric-value {
            font-weight: bold;
            color: #00d4ff;
        }
        .metric-value.warning {
            color: #ffc107;
        }
        .metric-value.critical {
            color: #ff4444;
        }
        .progress-bar {
            height: 8px;
            background: #0f3460;
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .progress-fill.llm { background: #00d4ff; }
        .progress-fill.tool { background: #00ff88; }
        .progress-fill.network { background: #ff8800; }
        .progress-fill.other { background: #888; }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .alert.warning {
            background: rgba(255, 193, 7, 0.2);
            border-left: 3px solid #ffc107;
        }
        .alert.critical {
            background: rgba(255, 68, 68, 0.2);
            border-left: 3px solid #ff4444;
        }
        .chart-container {
            position: relative;
            height: 200px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #0f3460;
        }
        th {
            color: #888;
        }
        .updated {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>⚡ Agent Latency Dashboard</h1>
    
    <div class="dashboard">
        <!-- Overview Card -->
        <div class="card">
            <h2>📊 Overview</h2>
            <div class="metric">
                <span class="metric-label">Total Requests</span>
                <span class="metric-value" id="total-requests">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Avg Latency</span>
                <span class="metric-value" id="avg-latency">0ms</span>
            </div>
            <div class="metric">
                <span class="metric-label">P95 Latency</span>
                <span class="metric-value" id="p95-latency">0ms</span>
            </div>
            <div class="metric">
                <span class="metric-label">P99 Latency</span>
                <span class="metric-value" id="p99-latency">0ms</span>
            </div>
        </div>
        
        <!-- Breakdown Card -->
        <div class="card">
            <h2>📈 Latency Breakdown</h2>
            <div class="metric">
                <span class="metric-label">LLM Calls</span>
                <span class="metric-value" id="llm-ms">0ms</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill llm" id="llm-bar" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Tool Execution</span>
                <span class="metric-value" id="tool-ms">0ms</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill tool" id="tool-bar" style="width: 0%"></div>
            </div>
            <div class="metric">
                <span class="metric-label">Network</span>
                <span class="metric-value" id="network-ms">0ms</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill network" id="network-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- Cache Card -->
        <div class="card">
            <h2>💾 Cache Performance</h2>
            <div class="metric">
                <span class="metric-label">Cache Hits</span>
                <span class="metric-value" id="cache-hits">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Cache Misses</span>
                <span class="metric-value" id="cache-misses">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">Hit Rate</span>
                <span class="metric-value" id="cache-rate">0%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill llm" id="cache-bar" style="width: 0%"></div>
            </div>
        </div>
        
        <!-- Histogram Card -->
        <div class="card" style="grid-column: span 2;">
            <h2>📉 Latency Distribution</h2>
            <div class="chart-container">
                <canvas id="histogram"></canvas>
            </div>
        </div>
        
        <!-- Slowest Operations Card -->
        <div class="card">
            <h2>🐌 Slowest Operations</h2>
            <table>
                <thead>
                    <tr>
                        <th>Operation</th>
                        <th>Category</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody id="slowest-table">
                </tbody>
            </table>
        </div>
        
        <!-- Alerts Card -->
        <div class="card">
            <h2>🚨 Alerts</h2>
            <div id="alerts-container">
                <p style="color: #666;">No alerts</p>
            </div>
        </div>
    </div>
    
    <p class="updated" id="updated-at">Last updated: --</p>
    
    <script>
        let histogramChart = null;
        
        function initHistogram() {
            const ctx = document.getElementById('histogram').getContext('2d');
            histogramChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Request Count',
                        data: [],
                        backgroundColor: 'rgba(0, 212, 255, 0.6)',
                        borderColor: 'rgba(0, 212, 255, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#0f3460' },
                            ticks: { color: '#888' }
                        },
                        x: {
                            grid: { color: '#0f3460' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
        
        function createHistogramData(latencies) {
            if (!latencies || latencies.length === 0) {
                return { labels: [], data: [] };
            }
            
            const bins = 10;
            const min = Math.min(...latencies);
            const max = Math.max(...latencies);
            const range = max - min || 1;
            const binSize = range / bins;
            
            const counts = new Array(bins).fill(0);
            const labels = [];
            
            for (let i = 0; i < bins; i++) {
                const start = min + i * binSize;
                const end = start + binSize;
                labels.push(`${Math.round(start)}-${Math.round(end)}ms`);
            }
            
            latencies.forEach(lat => {
                const binIndex = Math.min(Math.floor((lat - min) / binSize), bins - 1);
                counts[binIndex]++;
            });
            
            return { labels, data: counts };
        }
        
        function updateDashboard(data) {
            // Overview
            document.getElementById('total-requests').textContent = data.total_requests;
            document.getElementById('avg-latency').textContent = `${data.avg_latency_ms}ms`;
            document.getElementById('p95-latency').textContent = `${data.p95_latency_ms}ms`;
            document.getElementById('p99-latency').textContent = `${data.p99_latency_ms}ms`;
            
            // Apply color coding
            const avgEl = document.getElementById('avg-latency');
            avgEl.className = 'metric-value';
            if (data.avg_latency_ms >= data.thresholds.critical_ms) {
                avgEl.classList.add('critical');
            } else if (data.avg_latency_ms >= data.thresholds.warning_ms) {
                avgEl.classList.add('warning');
            }
            
            // Breakdown
            const total = data.breakdown.llm_ms + data.breakdown.tool_ms + 
                         data.breakdown.network_ms + data.breakdown.other_ms || 1;
            
            document.getElementById('llm-ms').textContent = `${data.breakdown.llm_ms}ms`;
            document.getElementById('tool-ms').textContent = `${data.breakdown.tool_ms}ms`;
            document.getElementById('network-ms').textContent = `${data.breakdown.network_ms}ms`;
            
            document.getElementById('llm-bar').style.width = `${data.breakdown.llm_ms / total * 100}%`;
            document.getElementById('tool-bar').style.width = `${data.breakdown.tool_ms / total * 100}%`;
            document.getElementById('network-bar').style.width = `${data.breakdown.network_ms / total * 100}%`;
            
            // Cache
            document.getElementById('cache-hits').textContent = data.cache.hits;
            document.getElementById('cache-misses').textContent = data.cache.misses;
            document.getElementById('cache-rate').textContent = `${data.cache.hit_rate}%`;
            document.getElementById('cache-bar').style.width = `${data.cache.hit_rate}%`;
            
            // Histogram
            const histData = createHistogramData(data.recent_latencies);
            histogramChart.data.labels = histData.labels;
            histogramChart.data.datasets[0].data = histData.data;
            histogramChart.update();
            
            // Slowest operations table
            const tableBody = document.getElementById('slowest-table');
            tableBody.innerHTML = data.slowest_operations.map(op => `
                <tr>
                    <td>${op.operation}</td>
                    <td>${op.category}</td>
                    <td>${op.duration_ms}ms</td>
                </tr>
            `).join('');
            
            // Alerts
            const alertsContainer = document.getElementById('alerts-container');
            if (data.alerts && data.alerts.length > 0) {
                alertsContainer.innerHTML = data.alerts.slice(-5).reverse().map(alert => `
                    <div class="alert ${alert.level}">
                        ${alert.message}
                    </div>
                `).join('');
            } else {
                alertsContainer.innerHTML = '<p style="color: #666;">No alerts</p>';
            }
            
            // Updated time
            document.getElementById('updated-at').textContent = 
                `Last updated: ${new Date(data.updated_at).toLocaleTimeString()}`;
        }
        
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to fetch data:', error);
            }
        }
        
        // Initialize
        initHistogram();
        fetchData();
        
        // Auto-refresh every 2 seconds
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
'''


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the dashboard."""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = dashboard.get_data()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress logging for cleaner output
        pass


def simulate_agent_traffic():
    """Simulate agent traffic for demonstration."""
    operations = [
        ("llm_planning", "llm"),
        ("llm_response", "llm"),
        ("weather_api", "tool"),
        ("database_query", "tool"),
        ("external_fetch", "network"),
        ("processing", "other"),
    ]
    
    while True:
        # Simulate a request
        op, category = random.choice(operations)
        
        # Generate realistic latencies
        if category == "llm":
            latency = random.gauss(500, 150)
        elif category == "tool":
            latency = random.gauss(200, 80)
        elif category == "network":
            latency = random.gauss(100, 40)
        else:
            latency = random.gauss(50, 20)
        
        # Occasionally have slow requests
        if random.random() < 0.05:
            latency *= 3  # 5% of requests are 3x slower
        
        latency = max(10, latency)  # Minimum 10ms
        
        dashboard.record_latency(op, category, latency)
        
        # Simulate cache access
        if random.random() < 0.7:  # 70% hit rate
            dashboard.record_cache_access(hit=True)
        else:
            dashboard.record_cache_access(hit=False)
        
        time.sleep(random.uniform(0.1, 0.5))


def main():
    """Run the latency dashboard."""
    print("=" * 60)
    print("LATENCY DASHBOARD")
    print("=" * 60)
    print()
    print("Starting dashboard server...")
    print("Open http://localhost:8080 in your browser")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    # Start traffic simulation in background
    simulator = threading.Thread(target=simulate_agent_traffic, daemon=True)
    simulator.start()
    print("✓ Traffic simulator started")
    
    # Start HTTP server
    server = HTTPServer(('localhost', 8080), DashboardHandler)
    print("✓ Dashboard server running on http://localhost:8080")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
