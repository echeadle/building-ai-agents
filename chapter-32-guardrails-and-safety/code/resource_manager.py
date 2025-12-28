"""
Rate limiting and resource management for AI agents.

Chapter 32: Guardrails and Safety

This module provides:
- Resource usage tracking (API calls, tokens, costs)
- Resource limits enforcement
- Rate limiting
- Cost estimation
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Any
from contextlib import contextmanager


@dataclass
class ResourceUsage:
    """Tracks resource usage."""
    api_calls: int = 0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration_seconds(self) -> float:
        """Get the duration since tracking started."""
        return time.time() - self.start_time
    
    @property
    def calls_per_minute(self) -> float:
        """Get the API call rate per minute."""
        if self.duration_seconds < 1:
            return 0
        return (self.api_calls / self.duration_seconds) * 60


@dataclass
class ResourceLimits:
    """Defines resource limits for an agent."""
    # Absolute limits
    max_api_calls: int = 100
    max_tokens: int = 100000
    max_tool_calls: int = 50
    max_errors: int = 5
    max_duration_seconds: int = 300  # 5 minutes
    max_cost_dollars: float = 1.0
    
    # Rate limits (per minute)
    api_calls_per_minute: int = 20
    tool_calls_per_minute: int = 10


class ResourceLimitExceeded(Exception):
    """Raised when a resource limit is exceeded."""
    
    def __init__(self, message: str, resource_type: str, current: Any, limit: Any):
        super().__init__(message)
        self.resource_type = resource_type
        self.current = current
        self.limit = limit


class ResourceManager:
    """
    Manages resource usage and enforces limits.
    
    Thread-safe for use with parallel workflows.
    """
    
    # Approximate costs per token (Claude 3.5 Sonnet pricing)
    INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
    OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens
    
    def __init__(self, limits: ResourceLimits | None = None):
        """
        Initialize the resource manager.
        
        Args:
            limits: Resource limits to enforce (uses defaults if not provided)
        """
        self.limits = limits or ResourceLimits()
        self.usage = ResourceUsage()
        self._lock = threading.Lock()
        
        # Track calls per minute for rate limiting
        self._api_call_times: list[float] = []
        self._tool_call_times: list[float] = []
    
    def reset(self) -> None:
        """Reset all usage counters."""
        with self._lock:
            self.usage = ResourceUsage()
            self._api_call_times.clear()
            self._tool_call_times.clear()
    
    def check_limits(self) -> None:
        """
        Check if any limits have been exceeded.
        
        Raises:
            ResourceLimitExceeded: If any limit is exceeded
        """
        with self._lock:
            if self.usage.api_calls >= self.limits.max_api_calls:
                raise ResourceLimitExceeded(
                    f"API call limit exceeded: {self.usage.api_calls}/{self.limits.max_api_calls}",
                    "api_calls",
                    self.usage.api_calls,
                    self.limits.max_api_calls
                )
            
            if self.usage.tokens_used >= self.limits.max_tokens:
                raise ResourceLimitExceeded(
                    f"Token limit exceeded: {self.usage.tokens_used}/{self.limits.max_tokens}",
                    "tokens",
                    self.usage.tokens_used,
                    self.limits.max_tokens
                )
            
            if self.usage.tool_calls >= self.limits.max_tool_calls:
                raise ResourceLimitExceeded(
                    f"Tool call limit exceeded: {self.usage.tool_calls}/{self.limits.max_tool_calls}",
                    "tool_calls",
                    self.usage.tool_calls,
                    self.limits.max_tool_calls
                )
            
            if self.usage.errors >= self.limits.max_errors:
                raise ResourceLimitExceeded(
                    f"Error limit exceeded: {self.usage.errors}/{self.limits.max_errors}",
                    "errors",
                    self.usage.errors,
                    self.limits.max_errors
                )
            
            if self.usage.duration_seconds >= self.limits.max_duration_seconds:
                raise ResourceLimitExceeded(
                    f"Duration limit exceeded: {self.usage.duration_seconds:.0f}s/{self.limits.max_duration_seconds}s",
                    "duration",
                    self.usage.duration_seconds,
                    self.limits.max_duration_seconds
                )
            
            estimated_cost = self._estimate_cost()
            if estimated_cost >= self.limits.max_cost_dollars:
                raise ResourceLimitExceeded(
                    f"Cost limit exceeded: ${estimated_cost:.2f}/${self.limits.max_cost_dollars:.2f}",
                    "cost",
                    estimated_cost,
                    self.limits.max_cost_dollars
                )
    
    def check_rate_limits(self, call_type: str = "api") -> float:
        """
        Check rate limits and return wait time if needed.
        
        Args:
            call_type: Type of call ("api" or "tool")
            
        Returns:
            Seconds to wait (0 if no wait needed)
        """
        now = time.time()
        window = 60  # 1 minute window
        
        with self._lock:
            if call_type == "api":
                # Clean old entries
                self._api_call_times = [
                    t for t in self._api_call_times 
                    if now - t < window
                ]
                
                if len(self._api_call_times) >= self.limits.api_calls_per_minute:
                    # Calculate wait time
                    oldest = min(self._api_call_times)
                    wait_time = window - (now - oldest)
                    return max(0, wait_time)
                
                return 0
            
            elif call_type == "tool":
                self._tool_call_times = [
                    t for t in self._tool_call_times 
                    if now - t < window
                ]
                
                if len(self._tool_call_times) >= self.limits.tool_calls_per_minute:
                    oldest = min(self._tool_call_times)
                    wait_time = window - (now - oldest)
                    return max(0, wait_time)
                
                return 0
        
        return 0
    
    def wait_for_rate_limit(self, call_type: str = "api") -> None:
        """
        Wait if rate limited, then record the call time.
        
        Args:
            call_type: Type of call ("api" or "tool")
        """
        wait_time = self.check_rate_limits(call_type)
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Record the call time
        with self._lock:
            if call_type == "api":
                self._api_call_times.append(time.time())
            elif call_type == "tool":
                self._tool_call_times.append(time.time())
    
    def record_api_call(
        self, 
        input_tokens: int = 0, 
        output_tokens: int = 0
    ) -> None:
        """
        Record an API call.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        with self._lock:
            self.usage.api_calls += 1
            self.usage.input_tokens += input_tokens
            self.usage.output_tokens += output_tokens
            self.usage.tokens_used += input_tokens + output_tokens
    
    def record_tool_call(self) -> None:
        """Record a tool call."""
        with self._lock:
            self.usage.tool_calls += 1
    
    def record_error(self) -> None:
        """Record an error."""
        with self._lock:
            self.usage.errors += 1
    
    def _estimate_cost(self) -> float:
        """
        Estimate the cost based on token usage.
        
        Returns:
            Estimated cost in dollars
        """
        return (
            self.usage.input_tokens * self.INPUT_COST_PER_TOKEN +
            self.usage.output_tokens * self.OUTPUT_COST_PER_TOKEN
        )
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of resource usage.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            return {
                "api_calls": f"{self.usage.api_calls}/{self.limits.max_api_calls}",
                "tokens": f"{self.usage.tokens_used:,}/{self.limits.max_tokens:,}",
                "input_tokens": f"{self.usage.input_tokens:,}",
                "output_tokens": f"{self.usage.output_tokens:,}",
                "tool_calls": f"{self.usage.tool_calls}/{self.limits.max_tool_calls}",
                "errors": f"{self.usage.errors}/{self.limits.max_errors}",
                "duration": f"{self.usage.duration_seconds:.1f}s/{self.limits.max_duration_seconds}s",
                "estimated_cost": f"${self._estimate_cost():.4f}/${self.limits.max_cost_dollars:.2f}",
                "calls_per_minute": f"{self.usage.calls_per_minute:.1f}",
            }
    
    def get_remaining(self) -> dict[str, Any]:
        """
        Get remaining resources.
        
        Returns:
            Dictionary with remaining resource counts
        """
        with self._lock:
            return {
                "api_calls": self.limits.max_api_calls - self.usage.api_calls,
                "tokens": self.limits.max_tokens - self.usage.tokens_used,
                "tool_calls": self.limits.max_tool_calls - self.usage.tool_calls,
                "errors": self.limits.max_errors - self.usage.errors,
                "duration_seconds": self.limits.max_duration_seconds - self.usage.duration_seconds,
                "cost_dollars": self.limits.max_cost_dollars - self._estimate_cost(),
            }
    
    @contextmanager
    def api_call_context(self):
        """
        Context manager for API calls with automatic tracking.
        
        Usage:
            with resource_manager.api_call_context():
                response = client.messages.create(...)
        """
        self.check_limits()
        self.wait_for_rate_limit("api")
        try:
            yield
        except Exception:
            self.record_error()
            raise
    
    @contextmanager
    def tool_call_context(self):
        """
        Context manager for tool calls with automatic tracking.
        
        Usage:
            with resource_manager.tool_call_context():
                result = execute_tool(...)
        """
        self.check_limits()
        self.wait_for_rate_limit("tool")
        try:
            yield
            self.record_tool_call()
        except Exception:
            self.record_error()
            raise


# Example usage and tests
if __name__ == "__main__":
    print("Testing ResourceManager:")
    
    # Create resource manager with tight limits for testing
    limits = ResourceLimits(
        max_api_calls=5,
        max_tokens=1000,
        max_cost_dollars=0.01,
        max_duration_seconds=60,
        api_calls_per_minute=10,
    )
    resource_manager = ResourceManager(limits)
    
    # Simulate some API calls
    print("\n1. Simulating API calls:")
    for i in range(4):
        try:
            resource_manager.check_limits()
            resource_manager.record_api_call(input_tokens=100, output_tokens=50)
            print(f"   Call {i+1}: Success")
        except ResourceLimitExceeded as e:
            print(f"   Call {i+1}: Blocked - {e}")
    
    # Check current usage
    print("\n2. Current usage summary:")
    for key, value in resource_manager.get_summary().items():
        print(f"   {key}: {value}")
    
    # Check remaining resources
    print("\n3. Remaining resources:")
    for key, value in resource_manager.get_remaining().items():
        print(f"   {key}: {value}")
    
    # Try to exceed limit
    print("\n4. Testing limit exceeded:")
    try:
        for i in range(5):
            resource_manager.check_limits()
            resource_manager.record_api_call(input_tokens=50, output_tokens=25)
            print(f"   Call {i+5}: Success")
    except ResourceLimitExceeded as e:
        print(f"   Limit exceeded: {e}")
        print(f"   Resource type: {e.resource_type}")
        print(f"   Current: {e.current}, Limit: {e.limit}")
    
    # Test context manager
    print("\n5. Testing context manager:")
    resource_manager.reset()
    try:
        for i in range(3):
            with resource_manager.api_call_context():
                # Simulate API call
                print(f"   API call {i+1} within context")
            resource_manager.record_api_call(input_tokens=100, output_tokens=100)
    except ResourceLimitExceeded as e:
        print(f"   Context blocked: {e}")
    
    print(f"\n   Final usage: {resource_manager.get_summary()}")
    
    # Test rate limiting
    print("\n6. Testing rate limiting:")
    fast_limits = ResourceLimits(
        max_api_calls=100,
        api_calls_per_minute=3,  # Only 3 per minute
    )
    fast_manager = ResourceManager(fast_limits)
    
    print("   Making rapid calls (limit: 3/min)...")
    start = time.time()
    for i in range(5):
        wait_time = fast_manager.check_rate_limits("api")
        if wait_time > 0:
            print(f"   Call {i+1}: Would wait {wait_time:.1f}s (skipping for test)")
        else:
            fast_manager._api_call_times.append(time.time())
            print(f"   Call {i+1}: Allowed")
    
    print("\n✅ Resource manager tests complete!")
