"""
Health checks and monitoring endpoints.

Chapter 40: Deployment Patterns

Health checks are essential for:
- Load balancers to route traffic
- Container orchestration (Kubernetes) to manage pods
- Monitoring systems to alert on issues

Three types of health checks:
- Liveness: Is the process running?
- Readiness: Can it handle requests?
- Detailed: Full system health with dependencies
"""

import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import anthropic

load_dotenv()


# ----- Health Status Types -----

class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"      # Everything working normally
    DEGRADED = "degraded"    # Working but with issues
    UNHEALTHY = "unhealthy"  # Not working properly


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Complete health check response."""
    status: HealthStatus
    timestamp: str
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]


class LivenessResponse(BaseModel):
    """Simple liveness response."""
    alive: bool


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool
    message: str


# ----- Global State -----

_start_time = time.time()
_version = "1.0.0"


# ----- Health Check Functions -----

async def check_anthropic_health() -> ComponentHealth:
    """
    Check Anthropic API connectivity.
    
    Uses a minimal API call to verify the service is reachable.
    This avoids wasting tokens on health checks.
    """
    start = time.perf_counter()
    
    try:
        client = anthropic.Anthropic()
        
        # Use count_tokens - it's fast and doesn't cost tokens
        response = client.messages.count_tokens(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "test"}]
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Token count: {response.input_tokens}"
        )
        
    except anthropic.AuthenticationError:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.UNHEALTHY,
            message="Invalid API key"
        )
    except anthropic.RateLimitError:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.DEGRADED,
            message="Rate limited"
        )
    except anthropic.APIConnectionError:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed"
        )
    except Exception as e:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.UNHEALTHY,
            message=str(e)[:100]
        )


async def check_redis_health(
    host: str = "localhost",
    port: int = 6379
) -> ComponentHealth:
    """
    Check Redis connectivity if configured.
    
    Returns healthy with a note if Redis isn't configured.
    """
    try:
        import redis
        
        start = time.perf_counter()
        r = redis.Redis(host=host, port=port, socket_timeout=5)
        r.ping()
        latency = (time.perf_counter() - start) * 1000
        
        # Get some info
        info = r.info("server")
        version = info.get("redis_version", "unknown")
        
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Redis {version}"
        )
        
    except ImportError:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis not configured (package not installed)"
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=str(e)[:100]
        )


def check_memory_health(
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0
) -> ComponentHealth:
    """
    Check system memory usage.
    
    Args:
        warning_threshold: Percent usage to trigger warning
        critical_threshold: Percent usage to trigger critical
    """
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        if used_percent >= critical_threshold:
            status = HealthStatus.UNHEALTHY
        elif used_percent >= warning_threshold:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="memory",
            status=status,
            message=f"{used_percent:.1f}% used ({memory.used // (1024**2)}MB / {memory.total // (1024**2)}MB)"
        )
        
    except ImportError:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.HEALTHY,
            message="psutil not installed - skipping memory check"
        )


def check_disk_health(
    path: str = "/",
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0
) -> ComponentHealth:
    """Check disk usage."""
    try:
        import psutil
        
        disk = psutil.disk_usage(path)
        used_percent = disk.percent
        
        if used_percent >= critical_threshold:
            status = HealthStatus.UNHEALTHY
        elif used_percent >= warning_threshold:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="disk",
            status=status,
            message=f"{used_percent:.1f}% used"
        )
        
    except ImportError:
        return ComponentHealth(
            name="disk",
            status=HealthStatus.HEALTHY,
            message="psutil not installed - skipping disk check"
        )


# ----- Setup Function -----

def setup_health_routes(app: FastAPI, version: str = "1.0.0"):
    """
    Add health check routes to a FastAPI application.
    
    Routes added:
    - GET /health - Detailed health check (all components)
    - GET /health/live - Liveness probe (is process running?)
    - GET /health/ready - Readiness probe (can handle requests?)
    
    Usage:
        app = FastAPI()
        setup_health_routes(app, version="1.0.0")
    """
    global _version
    _version = version
    
    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        tags=["Health"],
        summary="Detailed health check",
        description="Checks all system components and dependencies"
    )
    async def health_check(response: Response):
        """
        Comprehensive health check including all dependencies.
        
        Returns:
        - 200 if healthy
        - 503 if unhealthy or degraded
        
        Use this for monitoring dashboards and alerts.
        """
        # Run all health checks concurrently
        components = await asyncio.gather(
            check_anthropic_health(),
            check_redis_health(),
            asyncio.to_thread(check_memory_health),
            asyncio.to_thread(check_disk_health)
        )
        
        # Determine overall status
        statuses = [c.status for c in components]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            overall = HealthStatus.HEALTHY
        
        return HealthCheckResponse(
            status=overall,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=version,
            uptime_seconds=round(time.time() - _start_time, 2),
            components=list(components)
        )
    
    @app.get(
        "/health/live",
        response_model=LivenessResponse,
        tags=["Health"],
        summary="Liveness probe",
        description="Simple check that the process is running"
    )
    async def liveness():
        """
        Simple liveness probe.
        
        Returns 200 if the process is running.
        
        Used by Kubernetes to determine when to restart a container.
        This should always return 200 unless the process is deadlocked.
        """
        return LivenessResponse(alive=True)
    
    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["Health"],
        summary="Readiness probe",
        description="Check if service can handle requests"
    )
    async def readiness(response: Response):
        """
        Readiness probe for load balancers.
        
        Returns:
        - 200 if ready to accept requests
        - 503 if not ready
        
        Used by Kubernetes and load balancers to know when to
        send traffic to this instance.
        """
        # Check critical dependencies only
        anthropic_health = await check_anthropic_health()
        
        if anthropic_health.status == HealthStatus.UNHEALTHY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(
                ready=False,
                message=f"Anthropic API unhealthy: {anthropic_health.message}"
            )
        
        return ReadinessResponse(
            ready=True,
            message="Service is ready to accept requests"
        )


# ----- Simple Metrics -----

class SimpleMetrics:
    """
    Simple in-memory metrics collector.
    
    For production, use Prometheus, DataDog, or similar.
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self._lock = asyncio.Lock()
    
    async def record_request(
        self,
        tokens: int = 0,
        latency_ms: float = 0,
        error: bool = False
    ):
        """Record a request."""
        async with self._lock:
            self.request_count += 1
            self.total_tokens += tokens
            self.total_latency_ms += latency_ms
            if error:
                self.error_count += 1
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(error_rate, 4),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "uptime_seconds": round(time.time() - _start_time, 2)
        }


def setup_metrics_route(app: FastAPI, metrics: SimpleMetrics):
    """Add metrics endpoint to FastAPI app."""
    
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics():
        """
        Get application metrics.
        
        Returns basic statistics about requests, errors, and performance.
        """
        return metrics.get_stats()


# ----- Example Application -----

if __name__ == "__main__":
    import uvicorn
    
    app = FastAPI(
        title="Health Check Demo",
        version="1.0.0"
    )
    
    # Setup health routes
    setup_health_routes(app, version="1.0.0")
    
    # Setup metrics
    metrics = SimpleMetrics()
    setup_metrics_route(app, metrics)
    
    @app.get("/")
    async def root():
        return {
            "name": "Health Check Demo",
            "endpoints": {
                "/health": "Detailed health check",
                "/health/live": "Liveness probe",
                "/health/ready": "Readiness probe",
                "/metrics": "Application metrics"
            }
        }
    
    print("=" * 60)
    print("HEALTH CHECK DEMO")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  GET /health      - Detailed health (all components)")
    print("  GET /health/live - Liveness probe")
    print("  GET /health/ready - Readiness probe")
    print("  GET /metrics     - Application metrics")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
