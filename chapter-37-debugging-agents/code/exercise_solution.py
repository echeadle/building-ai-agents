"""
Exercise Solution: Comprehensive Session Diagnostic Tool

Chapter 37: Debugging Agents

This solution combines all the debugging tools from this chapter to analyze
a recorded agent session and produce a comprehensive bug report.

Task: Build a diagnostic tool that analyzes a recorded agent session and 
produces a bug report including:
- Issue category
- Evidence from the trace
- Suggested root cause
- Recommended fixes
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RecordedEvent:
    """A single recorded event in a session."""
    timestamp: str
    event_type: str
    data: dict[str, Any]
    sequence_number: int


@dataclass
class RecordedSession:
    """A complete recorded agent session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    model: str = ""
    system_prompt: Optional[str] = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    events: list[RecordedEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedSession":
        events = [RecordedEvent(**e) for e in data.get("events", [])]
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt"),
            tools=data.get("tools", []),
            events=events,
            metadata=data.get("metadata", {})
        )


@dataclass
class Issue:
    """A detected issue in the session."""
    category: str
    severity: str  # "critical", "warning", "info"
    description: str
    evidence: list[str]
    event_indices: list[int]
    suggested_fixes: list[str]


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a session."""
    session_id: str
    analyzed_at: str
    summary: dict[str, Any]
    issues: list[Issue]
    timeline_highlights: list[dict[str, Any]]
    root_cause_analysis: str
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "analyzed_at": self.analyzed_at,
            "summary": self.summary,
            "issues": [asdict(i) for i in self.issues],
            "timeline_highlights": self.timeline_highlights,
            "root_cause_analysis": self.root_cause_analysis,
            "recommendations": self.recommendations
        }


# =============================================================================
# Session Diagnostic Tool
# =============================================================================

class SessionDiagnosticTool:
    """
    Comprehensive diagnostic tool for analyzing agent sessions.
    
    Analyzes sessions for:
    - Infinite loops (exact repetition, oscillation)
    - Tool selection issues
    - Conversation flow problems
    - Performance issues
    - Errors and failures
    
    Usage:
        tool = SessionDiagnosticTool()
        report = tool.analyze_session(session_data)
        tool.print_report(report)
        tool.export_report(report, "report.json")
    """
    
    # Issue category definitions
    CATEGORIES = {
        "infinite_loop": "Agent stuck in a repetitive loop",
        "tool_selection": "Wrong or missing tool selection",
        "conversation_flow": "Conversation derailment or confusion",
        "malformed_output": "Invalid or unexpected output",
        "performance": "Performance or efficiency issues",
        "error": "Errors or exceptions",
        "missing_data": "Missing or incomplete data",
    }
    
    # Common fixes by category
    FIXES = {
        "infinite_loop": [
            "Add maximum iteration limit to the agent loop",
            "Add 'stop when you have the answer' to system prompt",
            "Implement loop detection with early termination",
            "Check if tool results are being processed correctly",
        ],
        "tool_selection": [
            "Improve tool descriptions with clearer keywords",
            "Add examples of when to use each tool",
            "Remove overlapping functionality between tools",
            "Make tool names more descriptive",
        ],
        "conversation_flow": [
            "Strengthen system prompt with clearer instructions",
            "Summarize long conversations to reduce context",
            "Add task reminders in the conversation",
            "Check for conflicting instructions",
        ],
        "malformed_output": [
            "Add output format examples to the prompt",
            "Use structured output mode if available",
            "Add response validation before processing",
            "Increase max_tokens if responses are truncated",
        ],
        "performance": [
            "Cache repeated tool calls",
            "Use streaming for long responses",
            "Reduce context size by summarizing history",
            "Consider using a faster model for simple tasks",
        ],
        "error": [
            "Add retry logic with exponential backoff",
            "Validate inputs before API calls",
            "Add proper error handling and recovery",
            "Check API key and permissions",
        ],
        "missing_data": [
            "Ensure all tool calls have corresponding results",
            "Validate session recording is complete",
            "Check for network issues during recording",
        ],
    }
    
    def __init__(self):
        self.session: Optional[RecordedSession] = None
    
    def load_session(self, filepath: str) -> RecordedSession:
        """Load a session from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        self.session = RecordedSession.from_dict(data)
        return self.session
    
    def load_session_from_dict(self, data: dict[str, Any]) -> RecordedSession:
        """Load a session from a dictionary."""
        self.session = RecordedSession.from_dict(data)
        return self.session
    
    def analyze_session(
        self, 
        session: Optional[RecordedSession] = None
    ) -> DiagnosticReport:
        """
        Perform comprehensive analysis of an agent session.
        
        Args:
            session: Session to analyze (uses loaded session if None)
        
        Returns:
            DiagnosticReport with all findings
        """
        if session:
            self.session = session
        
        if not self.session:
            raise ValueError("No session loaded. Call load_session() first.")
        
        issues = []
        timeline_highlights = []
        
        # Run all analyzers
        issues.extend(self._analyze_loops())
        issues.extend(self._analyze_tool_selection())
        issues.extend(self._analyze_conversation_flow())
        issues.extend(self._analyze_performance())
        issues.extend(self._analyze_errors())
        issues.extend(self._analyze_completeness())
        
        # Generate timeline highlights
        timeline_highlights = self._generate_timeline_highlights(issues)
        
        # Generate summary
        summary = self._generate_summary(issues)
        
        # Determine root cause
        root_cause = self._determine_root_cause(issues)
        
        # Compile recommendations
        recommendations = self._compile_recommendations(issues)
        
        return DiagnosticReport(
            session_id=self.session.session_id,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            issues=issues,
            timeline_highlights=timeline_highlights,
            root_cause_analysis=root_cause,
            recommendations=recommendations
        )
    
    def _analyze_loops(self) -> list[Issue]:
        """Detect infinite loop patterns."""
        issues = []
        tool_calls = [
            (i, e) for i, e in enumerate(self.session.events)
            if e.event_type == "tool_call"
        ]
        
        if not tool_calls:
            return issues
        
        # Check for exact repetition
        call_hashes = []
        for idx, event in tool_calls:
            tool_name = event.data.get("tool_name", "")
            tool_input = event.data.get("tool_input", {})
            hash_str = f"{tool_name}:{sorted(tool_input.items())}"
            call_hash = hashlib.md5(hash_str.encode()).hexdigest()[:8]
            call_hashes.append((idx, call_hash, tool_name))
        
        # Count repetitions
        hash_counts = defaultdict(list)
        for idx, hash_val, tool_name in call_hashes:
            hash_counts[hash_val].append((idx, tool_name))
        
        for hash_val, occurrences in hash_counts.items():
            if len(occurrences) >= 3:
                tool_name = occurrences[0][1]
                indices = [o[0] for o in occurrences]
                issues.append(Issue(
                    category="infinite_loop",
                    severity="critical",
                    description=f"Tool '{tool_name}' called {len(occurrences)} times with identical input",
                    evidence=[
                        f"First occurrence at event {indices[0]}",
                        f"Repeated at events: {indices[1:]}",
                    ],
                    event_indices=indices,
                    suggested_fixes=self.FIXES["infinite_loop"]
                ))
        
        # Check for oscillation (A->B->A->B pattern)
        if len(tool_calls) >= 4:
            tool_sequence = [e.data.get("tool_name", "") for _, e in tool_calls]
            for i in range(len(tool_sequence) - 3):
                if (tool_sequence[i] == tool_sequence[i+2] and
                    tool_sequence[i+1] == tool_sequence[i+3] and
                    tool_sequence[i] != tool_sequence[i+1]):
                    
                    # Count how long the pattern continues
                    pattern_length = 2
                    for j in range(i+4, len(tool_sequence), 2):
                        if j+1 < len(tool_sequence):
                            if (tool_sequence[j] == tool_sequence[i] and
                                tool_sequence[j+1] == tool_sequence[i+1]):
                                pattern_length += 1
                            else:
                                break
                    
                    if pattern_length >= 2:
                        issues.append(Issue(
                            category="infinite_loop",
                            severity="warning",
                            description=f"Oscillation pattern detected: {tool_sequence[i]} ↔ {tool_sequence[i+1]} ({pattern_length} cycles)",
                            evidence=[
                                f"Pattern starts at tool call {i}",
                                f"Alternates between '{tool_sequence[i]}' and '{tool_sequence[i+1]}'",
                            ],
                            event_indices=[tool_calls[i+j][0] for j in range(min(pattern_length*2, len(tool_calls)-i))],
                            suggested_fixes=self.FIXES["infinite_loop"]
                        ))
                    break
        
        return issues
    
    def _analyze_tool_selection(self) -> list[Issue]:
        """Analyze tool selection patterns."""
        issues = []
        
        tool_calls = [e for e in self.session.events if e.event_type == "tool_call"]
        tool_results = [e for e in self.session.events if e.event_type == "tool_result"]
        
        # Check for failed tool calls (no corresponding result)
        call_ids = {e.data.get("tool_use_id") for e in tool_calls}
        result_ids = {e.data.get("tool_use_id") for e in tool_results}
        missing_results = call_ids - result_ids
        
        if missing_results:
            issues.append(Issue(
                category="tool_selection",
                severity="warning",
                description=f"Tool calls without results: {len(missing_results)} missing",
                evidence=[f"Missing result for tool_use_id: {mid}" for mid in list(missing_results)[:3]],
                event_indices=[],
                suggested_fixes=["Ensure all tool calls receive results", "Check for tool execution errors"]
            ))
        
        # Check for tools that always fail
        tool_success = defaultdict(lambda: {"success": 0, "fail": 0})
        for result in tool_results:
            tool_name = result.data.get("tool_name", "unknown")
            # Check if result indicates failure
            result_content = str(result.data.get("result", ""))
            if "error" in result_content.lower() or "failed" in result_content.lower():
                tool_success[tool_name]["fail"] += 1
            else:
                tool_success[tool_name]["success"] += 1
        
        for tool_name, counts in tool_success.items():
            if counts["fail"] > 0 and counts["fail"] >= counts["success"]:
                issues.append(Issue(
                    category="tool_selection",
                    severity="warning",
                    description=f"Tool '{tool_name}' has high failure rate",
                    evidence=[
                        f"Success: {counts['success']}, Failures: {counts['fail']}",
                    ],
                    event_indices=[],
                    suggested_fixes=["Check tool implementation", "Validate tool inputs"]
                ))
        
        return issues
    
    def _analyze_conversation_flow(self) -> list[Issue]:
        """Analyze conversation structure."""
        issues = []
        
        llm_requests = [e for e in self.session.events if e.event_type == "llm_request"]
        llm_responses = [e for e in self.session.events if e.event_type == "llm_response"]
        
        # Check for very long conversations
        total_messages = sum(
            len(e.data.get("messages", [])) 
            for e in llm_requests
        )
        
        if total_messages > 50:
            issues.append(Issue(
                category="conversation_flow",
                severity="info",
                description=f"Long conversation detected: {total_messages} total messages",
                evidence=["May be approaching context window limits"],
                event_indices=[],
                suggested_fixes=self.FIXES["conversation_flow"]
            ))
        
        # Check for missing system prompt
        if not self.session.system_prompt:
            issues.append(Issue(
                category="conversation_flow",
                severity="warning",
                description="No system prompt recorded",
                evidence=["Agent may lack clear behavioral guidelines"],
                event_indices=[],
                suggested_fixes=["Add a system prompt to define agent behavior"]
            ))
        
        # Check if conversation ended properly
        if llm_responses:
            last_response = llm_responses[-1]
            stop_reason = last_response.data.get("stop_reason", "")
            if stop_reason == "tool_use":
                issues.append(Issue(
                    category="conversation_flow",
                    severity="warning",
                    description="Session ended with pending tool use",
                    evidence=[f"Last stop_reason was '{stop_reason}'"],
                    event_indices=[last_response.sequence_number],
                    suggested_fixes=["Ensure agent completes tool calls", "Check for premature termination"]
                ))
        
        return issues
    
    def _analyze_performance(self) -> list[Issue]:
        """Analyze performance metrics."""
        issues = []
        
        tool_results = [e for e in self.session.events if e.event_type == "tool_result"]
        llm_responses = [e for e in self.session.events if e.event_type == "llm_response"]
        
        # Check tool latencies
        tool_durations = []
        for result in tool_results:
            duration = result.data.get("duration_ms", 0)
            if duration > 0:
                tool_durations.append((result.data.get("tool_name", "unknown"), duration))
        
        slow_tools = [(name, dur) for name, dur in tool_durations if dur > 2000]
        if slow_tools:
            issues.append(Issue(
                category="performance",
                severity="info",
                description=f"Slow tool calls detected: {len(slow_tools)} calls > 2s",
                evidence=[f"{name}: {dur:.0f}ms" for name, dur in slow_tools[:3]],
                event_indices=[],
                suggested_fixes=self.FIXES["performance"]
            ))
        
        # Check token usage
        total_input = sum(e.data.get("usage", {}).get("input_tokens", 0) for e in llm_responses)
        total_output = sum(e.data.get("usage", {}).get("output_tokens", 0) for e in llm_responses)
        
        if total_input + total_output > 50000:
            issues.append(Issue(
                category="performance",
                severity="info",
                description=f"High token usage: {total_input + total_output:,} total tokens",
                evidence=[
                    f"Input tokens: {total_input:,}",
                    f"Output tokens: {total_output:,}",
                ],
                event_indices=[],
                suggested_fixes=["Consider summarizing context", "Use shorter prompts"]
            ))
        
        return issues
    
    def _analyze_errors(self) -> list[Issue]:
        """Analyze error events."""
        issues = []
        
        errors = [e for e in self.session.events if e.event_type == "error"]
        
        for error in errors:
            issues.append(Issue(
                category="error",
                severity="critical",
                description=f"Error: {error.data.get('error_type', 'unknown')}",
                evidence=[error.data.get("error_message", "No message")],
                event_indices=[error.sequence_number],
                suggested_fixes=self.FIXES["error"]
            ))
        
        return issues
    
    def _analyze_completeness(self) -> list[Issue]:
        """Check for missing or incomplete data."""
        issues = []
        
        # Check for orphaned tool results
        tool_calls = {e.data.get("tool_use_id"): e for e in self.session.events if e.event_type == "tool_call"}
        tool_results = [e for e in self.session.events if e.event_type == "tool_result"]
        
        for result in tool_results:
            tool_id = result.data.get("tool_use_id")
            if tool_id not in tool_calls:
                issues.append(Issue(
                    category="missing_data",
                    severity="warning",
                    description=f"Orphaned tool result (no matching call)",
                    evidence=[f"tool_use_id: {tool_id}"],
                    event_indices=[result.sequence_number],
                    suggested_fixes=self.FIXES["missing_data"]
                ))
        
        return issues
    
    def _generate_timeline_highlights(self, issues: list[Issue]) -> list[dict[str, Any]]:
        """Generate highlighted events from issues."""
        highlights = []
        
        # Collect all problematic event indices
        problem_indices = set()
        for issue in issues:
            problem_indices.update(issue.event_indices)
        
        for event in self.session.events:
            if event.sequence_number in problem_indices:
                highlights.append({
                    "sequence": event.sequence_number,
                    "type": event.event_type,
                    "timestamp": event.timestamp,
                    "issue": "Problem detected at this event",
                    "data_preview": str(event.data)[:100]
                })
        
        return highlights
    
    def _generate_summary(self, issues: list[Issue]) -> dict[str, Any]:
        """Generate a summary of the analysis."""
        # Count by severity
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for issue in issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] += 1
        
        # Session stats
        tool_calls = len([e for e in self.session.events if e.event_type == "tool_call"])
        llm_calls = len([e for e in self.session.events if e.event_type == "llm_response"])
        
        return {
            "total_events": len(self.session.events),
            "total_issues": len(issues),
            "issues_by_severity": dict(severity_counts),
            "issues_by_category": dict(category_counts),
            "tool_calls": tool_calls,
            "llm_calls": llm_calls,
            "session_duration": self._calculate_duration(),
            "health_score": self._calculate_health_score(issues),
        }
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in seconds."""
        if not self.session.events:
            return 0
        
        first = datetime.fromisoformat(self.session.events[0].timestamp)
        last = datetime.fromisoformat(self.session.events[-1].timestamp)
        return (last - first).total_seconds()
    
    def _calculate_health_score(self, issues: list[Issue]) -> int:
        """Calculate a health score from 0-100."""
        score = 100
        
        for issue in issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "warning":
                score -= 10
            elif issue.severity == "info":
                score -= 2
        
        return max(0, score)
    
    def _determine_root_cause(self, issues: list[Issue]) -> str:
        """Determine the most likely root cause."""
        if not issues:
            return "No issues detected. Session appears healthy."
        
        # Prioritize by severity
        critical = [i for i in issues if i.severity == "critical"]
        warnings = [i for i in issues if i.severity == "warning"]
        
        if critical:
            main_issue = critical[0]
            return f"Primary issue: {main_issue.description}. " \
                   f"This is a critical issue that likely caused the session to fail. " \
                   f"Recommended action: {main_issue.suggested_fixes[0] if main_issue.suggested_fixes else 'Review the issue'}"
        
        if warnings:
            # Look for patterns
            categories = [i.category for i in warnings]
            most_common = max(set(categories), key=categories.count)
            return f"Multiple {most_common} issues detected. " \
                   f"This pattern suggests a systematic problem with {self.CATEGORIES.get(most_common, most_common)}."
        
        return "Minor issues detected. Session completed but could be optimized."
    
    def _compile_recommendations(self, issues: list[Issue]) -> list[str]:
        """Compile prioritized recommendations."""
        recommendations = []
        seen_fixes = set()
        
        # Sort by severity
        sorted_issues = sorted(
            issues, 
            key=lambda x: {"critical": 0, "warning": 1, "info": 2}.get(x.severity, 3)
        )
        
        for issue in sorted_issues:
            for fix in issue.suggested_fixes:
                if fix not in seen_fixes:
                    recommendations.append(fix)
                    seen_fixes.add(fix)
                if len(recommendations) >= 10:
                    break
            if len(recommendations) >= 10:
                break
        
        return recommendations
    
    def print_report(self, report: DiagnosticReport) -> None:
        """Print a formatted diagnostic report."""
        print("\n" + "=" * 70)
        print("                    DIAGNOSTIC REPORT")
        print("=" * 70)
        
        print(f"\nSession ID: {report.session_id}")
        print(f"Analyzed at: {report.analyzed_at}")
        
        # Summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Total Events: {report.summary['total_events']}")
        print(f"  LLM Calls: {report.summary['llm_calls']}")
        print(f"  Tool Calls: {report.summary['tool_calls']}")
        print(f"  Duration: {report.summary['session_duration']:.2f}s")
        print(f"  Health Score: {report.summary['health_score']}/100")
        print(f"  Total Issues: {report.summary['total_issues']}")
        
        if report.summary['issues_by_severity']:
            print("\n  Issues by Severity:")
            for severity, count in report.summary['issues_by_severity'].items():
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
                print(f"    {icon} {severity}: {count}")
        
        # Issues
        if report.issues:
            print("\n" + "-" * 70)
            print("ISSUES DETECTED")
            print("-" * 70)
            
            for i, issue in enumerate(report.issues, 1):
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
                print(f"\n  {icon} Issue #{i}: {issue.description}")
                print(f"     Category: {issue.category}")
                print(f"     Severity: {issue.severity}")
                
                if issue.evidence:
                    print("     Evidence:")
                    for ev in issue.evidence[:3]:
                        print(f"       • {ev}")
        
        # Root Cause
        print("\n" + "-" * 70)
        print("ROOT CAUSE ANALYSIS")
        print("-" * 70)
        print(f"\n  {report.root_cause_analysis}")
        
        # Recommendations
        if report.recommendations:
            print("\n" + "-" * 70)
            print("RECOMMENDATIONS")
            print("-" * 70)
            
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Timeline highlights
        if report.timeline_highlights:
            print("\n" + "-" * 70)
            print("PROBLEM EVENTS")
            print("-" * 70)
            
            for highlight in report.timeline_highlights[:5]:
                print(f"\n  Event #{highlight['sequence']} ({highlight['type']})")
                print(f"    {highlight['issue']}")
        
        print("\n" + "=" * 70)
    
    def export_report(self, report: DiagnosticReport, filepath: str) -> None:
        """Export the report to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"✅ Report exported to {filepath}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SESSION DIAGNOSTIC TOOL - EXERCISE SOLUTION")
    print("=" * 70)
    
    # Create a sample problematic session for demonstration
    sample_session = {
        "session_id": "problem-session-001",
        "started_at": "2025-01-15T10:00:00+00:00",
        "ended_at": "2025-01-15T10:05:00+00:00",
        "model": "claude-sonnet-4-20250514",
        "system_prompt": None,  # Missing system prompt (issue!)
        "tools": [
            {"name": "search", "description": "Search for information"},
            {"name": "calculate", "description": "Perform calculations"}
        ],
        "events": [
            # Initial request
            {"timestamp": "2025-01-15T10:00:00+00:00", "event_type": "llm_request", 
             "data": {"messages": [{"role": "user", "content": "What is 2+2 and search for cats"}]}, "sequence_number": 1},
            
            {"timestamp": "2025-01-15T10:00:01+00:00", "event_type": "llm_response",
             "data": {"content": [], "stop_reason": "tool_use", "usage": {"input_tokens": 50, "output_tokens": 30}}, "sequence_number": 2},
            
            # First search call
            {"timestamp": "2025-01-15T10:00:02+00:00", "event_type": "tool_call",
             "data": {"tool_name": "search", "tool_input": {"query": "cats"}, "tool_use_id": "tool_1"}, "sequence_number": 3},
            
            {"timestamp": "2025-01-15T10:00:03+00:00", "event_type": "tool_result",
             "data": {"tool_name": "search", "tool_use_id": "tool_1", "result": "Cats are mammals", "duration_ms": 500}, "sequence_number": 4},
            
            # Repeated search call (loop!)
            {"timestamp": "2025-01-15T10:00:04+00:00", "event_type": "tool_call",
             "data": {"tool_name": "search", "tool_input": {"query": "cats"}, "tool_use_id": "tool_2"}, "sequence_number": 5},
            
            {"timestamp": "2025-01-15T10:00:05+00:00", "event_type": "tool_result",
             "data": {"tool_name": "search", "tool_use_id": "tool_2", "result": "Cats are mammals", "duration_ms": 500}, "sequence_number": 6},
            
            # Third repeated call (definite loop!)
            {"timestamp": "2025-01-15T10:00:06+00:00", "event_type": "tool_call",
             "data": {"tool_name": "search", "tool_input": {"query": "cats"}, "tool_use_id": "tool_3"}, "sequence_number": 7},
            
            {"timestamp": "2025-01-15T10:00:07+00:00", "event_type": "tool_result",
             "data": {"tool_name": "search", "tool_use_id": "tool_3", "result": "Cats are mammals", "duration_ms": 500}, "sequence_number": 8},
            
            # LLM response but still wants tools
            {"timestamp": "2025-01-15T10:00:08+00:00", "event_type": "llm_response",
             "data": {"content": [], "stop_reason": "tool_use", "usage": {"input_tokens": 150, "output_tokens": 30}}, "sequence_number": 9},
            
            # Error event
            {"timestamp": "2025-01-15T10:00:09+00:00", "event_type": "error",
             "data": {"error_type": "LoopDetected", "error_message": "Maximum iterations reached"}, "sequence_number": 10},
        ]
    }
    
    # Run the diagnostic tool
    tool = SessionDiagnosticTool()
    tool.load_session_from_dict(sample_session)
    
    print("\nAnalyzing session...")
    report = tool.analyze_session()
    
    # Print the report
    tool.print_report(report)
    
    # Export to file
    tool.export_report(report, "/tmp/diagnostic_report.json")
    
    # Show JSON output
    print("\n" + "-" * 70)
    print("JSON REPORT PREVIEW")
    print("-" * 70)
    print(json.dumps(report.to_dict(), indent=2)[:1500] + "...")
    
    print("\n✅ Diagnostic tool demonstration complete!")
