"""
Infinite loop detection and prevention.

Chapter 37: Debugging Agents

This module provides a system for detecting and preventing infinite loops
in agent execution. It recognizes patterns like exact repetition, oscillation,
and resource exhaustion.
"""

import hashlib
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class LoopPattern:
    """A detected loop pattern."""
    pattern_type: str  # "exact_repeat", "semantic_repeat", "oscillation"
    tool_sequence: list[str]
    repetitions: int
    first_occurrence: int
    description: str


@dataclass
class LoopDetectorConfig:
    """Configuration for loop detection."""
    max_iterations: int = 25
    max_same_tool_consecutive: int = 3
    max_tool_call_total: int = 50
    pattern_window_size: int = 5
    detect_semantic_loops: bool = True


class LoopDetector:
    """
    Detects and prevents infinite loops in agent execution.
    
    Detection strategies:
    1. Exact repetition: Same tool + same args called multiple times
    2. Semantic repetition: Similar queries producing similar tool calls
    3. Oscillation: Tool A -> Tool B -> Tool A -> Tool B pattern
    4. Resource exhaustion: Too many total calls
    
    Usage:
        detector = LoopDetector()
        
        while not done:
            # Check before each iteration
            if detector.should_stop():
                print(f"Loop detected: {detector.get_stop_reason()}")
                break
            
            # Execute agent step
            tool_name, tool_input = agent.next_action()
            
            # Record the action
            detector.record_tool_call(tool_name, tool_input)
    """
    
    def __init__(self, config: Optional[LoopDetectorConfig] = None):
        """Initialize the loop detector."""
        self.config = config or LoopDetectorConfig()
        
        # Tracking state
        self.tool_calls: list[tuple[str, dict]] = []  # (name, input)
        self.call_hashes: list[str] = []
        self.tool_counts: dict[str, int] = defaultdict(int)
        self.consecutive_same_tool: int = 0
        self.last_tool: Optional[str] = None
        
        # Detection results
        self.detected_patterns: list[LoopPattern] = []
        self.stop_reason: Optional[str] = None
    
    def _hash_call(self, tool_name: str, tool_input: dict) -> str:
        """Create a hash for a tool call for exact match detection."""
        content = f"{tool_name}:{sorted(tool_input.items())}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: Optional[str] = None
    ) -> None:
        """
        Record a tool call for loop detection.
        
        Args:
            tool_name: Name of the tool called
            tool_input: Input parameters
            tool_result: Optional result (for semantic analysis)
        """
        call_hash = self._hash_call(tool_name, tool_input)
        
        self.tool_calls.append((tool_name, tool_input))
        self.call_hashes.append(call_hash)
        self.tool_counts[tool_name] += 1
        
        # Track consecutive same tool
        if tool_name == self.last_tool:
            self.consecutive_same_tool += 1
        else:
            self.consecutive_same_tool = 1
            self.last_tool = tool_name
        
        # Check for patterns after each call
        self._detect_patterns()
    
    def _detect_patterns(self) -> None:
        """Detect loop patterns in the call history."""
        # Check for exact repetition
        self._detect_exact_repetition()
        
        # Check for oscillation patterns
        self._detect_oscillation()
    
    def _detect_exact_repetition(self) -> None:
        """Detect exact same tool calls repeated."""
        if len(self.call_hashes) < 2:
            return
        
        recent_hash = self.call_hashes[-1]
        
        # Count how many times this exact call appears
        count = self.call_hashes.count(recent_hash)
        
        if count >= 3:
            # Find the tool name and input
            idx = self.call_hashes.index(recent_hash)
            tool_name, tool_input = self.tool_calls[idx]
            
            pattern = LoopPattern(
                pattern_type="exact_repeat",
                tool_sequence=[tool_name],
                repetitions=count,
                first_occurrence=idx,
                description=f"Tool '{tool_name}' called {count} times with identical input"
            )
            
            # Avoid duplicate pattern detection
            if not any(p.pattern_type == "exact_repeat" and p.tool_sequence == [tool_name] 
                      for p in self.detected_patterns):
                self.detected_patterns.append(pattern)
    
    def _detect_oscillation(self) -> None:
        """Detect A->B->A->B oscillation patterns."""
        if len(self.tool_calls) < 4:
            return
        
        # Check last 4 calls for A-B-A-B pattern
        recent = [t[0] for t in self.tool_calls[-4:]]
        
        if (recent[0] == recent[2] and 
            recent[1] == recent[3] and 
            recent[0] != recent[1]):
            
            # Check if this pattern continues further back
            pattern_length = 2
            repetitions = 2
            
            for i in range(len(self.tool_calls) - 5, -1, -2):
                if i >= 0 and i + 1 < len(self.tool_calls):
                    if (self.tool_calls[i][0] == recent[0] and 
                        self.tool_calls[i+1][0] == recent[1]):
                        repetitions += 1
                    else:
                        break
            
            if repetitions >= 2:
                pattern = LoopPattern(
                    pattern_type="oscillation",
                    tool_sequence=[recent[0], recent[1]],
                    repetitions=repetitions,
                    first_occurrence=len(self.tool_calls) - repetitions * 2,
                    description=f"Oscillation between '{recent[0]}' and '{recent[1]}' "
                               f"({repetitions} cycles)"
                )
                
                if not any(p.pattern_type == "oscillation" 
                          for p in self.detected_patterns):
                    self.detected_patterns.append(pattern)
    
    def should_stop(self) -> bool:
        """
        Check if the agent should stop due to loop detection.
        
        Returns:
            True if a stopping condition is met
        """
        total_calls = len(self.tool_calls)
        
        # Check max iterations
        if total_calls >= self.config.max_iterations:
            self.stop_reason = (
                f"Max iterations reached ({self.config.max_iterations})"
            )
            return True
        
        # Check consecutive same tool
        if self.consecutive_same_tool >= self.config.max_same_tool_consecutive:
            self.stop_reason = (
                f"Same tool called {self.consecutive_same_tool} times consecutively"
            )
            return True
        
        # Check total calls per tool
        if self.tool_counts:
            max_per_tool = self.config.max_tool_call_total // max(len(self.tool_counts), 1)
            for tool, count in self.tool_counts.items():
                if count >= max_per_tool and count >= 10:
                    self.stop_reason = f"Tool '{tool}' called {count} times (limit reached)"
                    return True
        
        # Check for severe patterns
        for pattern in self.detected_patterns:
            if pattern.pattern_type == "exact_repeat" and pattern.repetitions >= 3:
                self.stop_reason = pattern.description
                return True
            if pattern.pattern_type == "oscillation" and pattern.repetitions >= 3:
                self.stop_reason = pattern.description
                return True
        
        return False
    
    def get_stop_reason(self) -> Optional[str]:
        """Get the reason for stopping."""
        return self.stop_reason
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of loop detection state."""
        return {
            "total_calls": len(self.tool_calls),
            "unique_calls": len(set(self.call_hashes)),
            "tool_counts": dict(self.tool_counts),
            "detected_patterns": [
                {
                    "type": p.pattern_type,
                    "sequence": p.tool_sequence,
                    "repetitions": p.repetitions,
                    "description": p.description
                }
                for p in self.detected_patterns
            ],
            "should_stop": self.should_stop(),
            "stop_reason": self.stop_reason
        }
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.tool_calls = []
        self.call_hashes = []
        self.tool_counts = defaultdict(int)
        self.consecutive_same_tool = 0
        self.last_tool = None
        self.detected_patterns = []
        self.stop_reason = None


class LoopPreventer:
    """
    Wrapper that adds loop prevention to tool execution.
    
    Usage:
        preventer = LoopPreventer(detector=LoopDetector())
        
        @preventer.wrap
        def call_tool(name: str, input: dict) -> str:
            # Your tool execution logic
            return result
        
        # Now call_tool will raise LoopDetectedError if a loop is detected
    """
    
    def __init__(self, detector: Optional[LoopDetector] = None):
        self.detector = detector or LoopDetector()
    
    def wrap(self, func):
        """Decorator to wrap a tool execution function."""
        def wrapper(tool_name: str, tool_input: dict, *args, **kwargs):
            # Check before execution
            if self.detector.should_stop():
                raise LoopDetectedError(
                    self.detector.get_stop_reason() or "Loop detected"
                )
            
            # Execute
            result = func(tool_name, tool_input, *args, **kwargs)
            
            # Record the call
            self.detector.record_tool_call(tool_name, tool_input, str(result))
            
            # Check after execution
            if self.detector.should_stop():
                raise LoopDetectedError(
                    self.detector.get_stop_reason() or "Loop detected"
                )
            
            return result
        
        return wrapper


class LoopDetectedError(Exception):
    """Raised when a loop is detected."""
    pass


# Example usage
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("LOOP DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Demo 1: Exact repetition detection
    print("\n" + "-" * 60)
    print("DEMO 1: Exact Repetition Detection")
    print("-" * 60)
    
    detector = LoopDetector()
    
    print("\nSimulating repeated identical tool calls...")
    
    for i in range(5):
        print(f"\n  Step {i + 1}: Calling search with same query")
        
        if detector.should_stop():
            print(f"  ⛔ STOPPED: {detector.get_stop_reason()}")
            break
        
        detector.record_tool_call("search", {"query": "weather today"})
        
        if detector.detected_patterns:
            for pattern in detector.detected_patterns:
                print(f"  ⚠️  Pattern detected: {pattern.description}")
    
    # Demo 2: Oscillation detection
    print("\n" + "-" * 60)
    print("DEMO 2: Oscillation Detection")
    print("-" * 60)
    
    detector.reset()
    
    print("\nSimulating oscillation between two tools...")
    
    tool_sequence = [
        ("search", {"query": "find info"}),
        ("analyze", {"data": "results"}),
        ("search", {"query": "find info"}),
        ("analyze", {"data": "results"}),
        ("search", {"query": "find info"}),
        ("analyze", {"data": "results"}),
        ("search", {"query": "find info"}),
    ]
    
    for i, (tool, input_data) in enumerate(tool_sequence):
        print(f"\n  Step {i + 1}: Calling {tool}")
        
        if detector.should_stop():
            print(f"  ⛔ STOPPED: {detector.get_stop_reason()}")
            break
        
        detector.record_tool_call(tool, input_data)
        
        if detector.detected_patterns:
            for pattern in detector.detected_patterns:
                if pattern.pattern_type == "oscillation":
                    print(f"  ⚠️  Pattern detected: {pattern.description}")
    
    # Demo 3: Consecutive same tool
    print("\n" + "-" * 60)
    print("DEMO 3: Consecutive Same Tool Detection")
    print("-" * 60)
    
    detector.reset()
    
    print("\nSimulating consecutive calls to the same tool with different inputs...")
    
    for i in range(5):
        print(f"\n  Step {i + 1}: Calling calculator with expression={i}*2")
        
        if detector.should_stop():
            print(f"  ⛔ STOPPED: {detector.get_stop_reason()}")
            break
        
        detector.record_tool_call("calculator", {"expression": f"{i}*2"})
    
    # Demo 4: Full summary
    print("\n" + "-" * 60)
    print("DETECTION SUMMARY")
    print("-" * 60)
    print(json.dumps(detector.get_summary(), indent=2))
    
    # Demo 5: Using the LoopPreventer decorator
    print("\n" + "-" * 60)
    print("DEMO 5: Loop Preventer Decorator")
    print("-" * 60)
    
    preventer = LoopPreventer()
    
    @preventer.wrap
    def execute_tool(tool_name: str, tool_input: dict) -> str:
        return f"Executed {tool_name}"
    
    print("\nUsing wrapped tool execution function...")
    
    try:
        for i in range(30):
            result = execute_tool("search", {"query": "test"})
            print(f"  Step {i + 1}: {result}")
    except LoopDetectedError as e:
        print(f"\n  ⛔ LoopDetectedError: {e}")
    
    print("\n✅ Loop detection demonstration complete!")
