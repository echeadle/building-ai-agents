"""
Systematic debugging helper.

Chapter 37: Debugging Agents

This module provides a guided, systematic approach to debugging
AI agents. It walks you through categorization, diagnosis, and
resolution of common agent issues.
"""

from typing import Any, Optional
from dataclasses import dataclass
import json


@dataclass
class DebuggingContext:
    """Context collected during debugging."""
    issue_description: str
    category: Optional[str] = None
    trace_file: Optional[str] = None
    findings: list[str] = None
    root_cause: Optional[str] = None
    fix_applied: Optional[str] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []


class DebuggingHelper:
    """
    Interactive helper for systematic agent debugging.
    
    Guides you through a structured debugging process:
    1. Categorize the issue
    2. Run through diagnostic checklist
    3. Identify root cause
    4. Apply and document fix
    5. Generate report
    
    Usage:
        helper = DebuggingHelper()
        helper.start_session("Agent keeps calling the same tool repeatedly")
        helper.categorize()
        helper.set_category("2")  # infinite_loop
        helper.add_finding("Tool called 15 times with same input")
        helper.suggest_fixes()
        helper.record_root_cause("Missing termination condition")
        helper.record_fix("Added max iterations check")
        helper.generate_report()
    """
    
    CATEGORIES = {
        "1": ("tool_selection", "Wrong tool or no tool selected"),
        "2": ("infinite_loop", "Agent stuck in a loop"),
        "3": ("conversation_flow", "Off-topic or confused responses"),
        "4": ("malformed_output", "Invalid or unexpected output format"),
        "5": ("performance", "Slow or expensive execution"),
        "6": ("error", "Exception or API error"),
    }
    
    COMMON_FIXES = {
        "tool_selection": [
            "Improve tool descriptions with more keywords",
            "Add examples to tool descriptions",
            "Remove ambiguity between similar tools",
            "Check if the required tool is actually provided",
            "Add 'Use this tool when...' to descriptions",
        ],
        "infinite_loop": [
            "Add maximum iteration limits",
            "Check for exit conditions in the prompt",
            "Add loop detection",
            "Verify tool results don't trigger the same call",
            "Add 'stop when you have the answer' instruction",
        ],
        "conversation_flow": [
            "Strengthen the system prompt",
            "Summarize long conversations",
            "Add explicit task reminders",
            "Check for context window exhaustion",
            "Remove conflicting instructions",
        ],
        "malformed_output": [
            "Add output format examples to the prompt",
            "Use structured output mode",
            "Add response validation",
            "Simplify the expected format",
            "Provide JSON schema in system prompt",
        ],
        "performance": [
            "Cache repeated operations",
            "Use a faster model for simple tasks",
            "Reduce context size",
            "Parallelize independent operations",
            "Pre-compute common queries",
        ],
        "error": [
            "Add retry logic with backoff",
            "Validate inputs before sending",
            "Check API key and permissions",
            "Handle rate limits gracefully",
            "Add timeout handling",
        ],
    }
    
    DIAGNOSTIC_CHECKLISTS = {
        "tool_selection": [
            "□ Verify the expected tool is in the tools list",
            "□ Check tool description for clarity",
            "□ Look for overlapping tool functionality",
            "□ Check if query matches tool keywords",
            "□ Review the LLM's reasoning (if visible)",
            "□ Test with a more explicit query",
        ],
        "infinite_loop": [
            "□ Check total iteration count",
            "□ Look for repeated identical tool calls",
            "□ Check for oscillation patterns (A→B→A→B)",
            "□ Verify termination conditions exist",
            "□ Check if tool results are being processed",
            "□ Review system prompt for exit instructions",
        ],
        "conversation_flow": [
            "□ Review system prompt completeness",
            "□ Check conversation length (token count)",
            "□ Look for conflicting instructions",
            "□ Verify tool results aren't confusing",
            "□ Check if task context is maintained",
            "□ Look for topic drift in messages",
        ],
        "malformed_output": [
            "□ Check expected vs actual output format",
            "□ Verify JSON/structured output settings",
            "□ Look for truncated responses",
            "□ Check for encoding issues",
            "□ Verify schema definitions",
            "□ Check max_tokens setting",
        ],
        "performance": [
            "□ Measure time per LLM call",
            "□ Count total tokens used",
            "□ Identify repeated operations",
            "□ Check for unnecessary tool calls",
            "□ Review context size over time",
            "□ Check for N+1 query patterns",
        ],
        "error": [
            "□ Check error message and stack trace",
            "□ Verify API key is valid",
            "□ Check for rate limiting",
            "□ Validate input data",
            "□ Check network connectivity",
            "□ Review recent API changes",
        ],
    }
    
    def __init__(self):
        self.context: Optional[DebuggingContext] = None
    
    def start_session(self, issue_description: str) -> None:
        """Start a debugging session."""
        self.context = DebuggingContext(issue_description=issue_description)
        print("\n" + "=" * 60)
        print("DEBUGGING SESSION STARTED")
        print("=" * 60)
        print(f"\nIssue: {issue_description}")
        print("\nNext step: Call categorize() to classify the issue")
    
    def categorize(self) -> str:
        """Help categorize the issue."""
        print("\n" + "-" * 60)
        print("STEP 1: CATEGORIZE THE ISSUE")
        print("-" * 60)
        print("\nSelect the category that best matches your issue:\n")
        
        for key, (_, description) in self.CATEGORIES.items():
            print(f"  {key}. {description}")
        
        print("\nCall set_category(number) with your choice (e.g., set_category('2'))")
        return "Use set_category(number) to select"
    
    def set_category(self, category_num: str) -> None:
        """Set the issue category."""
        if not self.context:
            print("❌ No active session. Call start_session() first.")
            return
            
        if category_num not in self.CATEGORIES:
            print(f"❌ Invalid category. Choose from: {list(self.CATEGORIES.keys())}")
            return
        
        category, description = self.CATEGORIES[category_num]
        self.context.category = category
        print(f"\n✅ Category set: {description}")
        
        # Show relevant diagnostic steps
        self._show_diagnostic_steps(category)
    
    def _show_diagnostic_steps(self, category: str) -> None:
        """Show diagnostic steps for a category."""
        print("\n" + "-" * 60)
        print("STEP 2: DIAGNOSTIC CHECKLIST")
        print("-" * 60)
        print("\nGo through this checklist and use add_finding() to record observations:\n")
        
        checklist = self.DIAGNOSTIC_CHECKLISTS.get(category, [])
        for item in checklist:
            print(f"  {item}")
        
        print("\nExample: helper.add_finding('Tool called 15 times with same input')")
    
    def add_finding(self, finding: str) -> None:
        """Add a debugging finding."""
        if not self.context:
            print("❌ No active session. Call start_session() first.")
            return
            
        self.context.findings.append(finding)
        print(f"📝 Finding recorded: {finding}")
    
    def suggest_fixes(self) -> list[str]:
        """Suggest fixes based on the category."""
        if not self.context:
            print("❌ No active session. Call start_session() first.")
            return []
            
        if not self.context.category:
            print("❌ No category set. Call set_category() first.")
            return []
        
        print("\n" + "-" * 60)
        print("STEP 3: SUGGESTED FIXES")
        print("-" * 60)
        
        fixes = self.COMMON_FIXES.get(self.context.category, [])
        
        print(f"\nCommon fixes for {self.context.category} issues:\n")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
        
        print("\nAfter applying a fix, call record_fix() to document it.")
        return fixes
    
    def record_fix(self, fix_description: str) -> None:
        """Record the fix that was applied."""
        if not self.context:
            print("❌ No active session. Call start_session() first.")
            return
            
        self.context.fix_applied = fix_description
        print(f"\n✅ Fix recorded: {fix_description}")
    
    def record_root_cause(self, root_cause: str) -> None:
        """Record the identified root cause."""
        if not self.context:
            print("❌ No active session. Call start_session() first.")
            return
            
        self.context.root_cause = root_cause
        print(f"\n🎯 Root cause identified: {root_cause}")
    
    def generate_report(self) -> dict[str, Any]:
        """Generate a debugging report."""
        if not self.context:
            return {"error": "No debugging session active"}
        
        report = {
            "issue_description": self.context.issue_description,
            "category": self.context.category,
            "findings": self.context.findings,
            "root_cause": self.context.root_cause,
            "fix_applied": self.context.fix_applied,
            "recommendations": self.COMMON_FIXES.get(self.context.category, []),
            "status": "resolved" if self.context.fix_applied else "in_progress"
        }
        
        print("\n" + "=" * 60)
        print("DEBUGGING REPORT")
        print("=" * 60)
        print(json.dumps(report, indent=2))
        
        return report
    
    def prompt_checklist(self) -> None:
        """Show a checklist for prompt-related issues."""
        print("\n" + "-" * 60)
        print("PROMPT DEBUGGING CHECKLIST")
        print("-" * 60)
        print("""
Before changing code, verify these prompt aspects:

SYSTEM PROMPT:
  □ Is the agent's role clearly defined?
  □ Are there explicit instructions for tool usage?
  □ Are there clear termination conditions?
  □ Are constraints and limitations stated?
  □ Is the expected output format described?

TOOL DESCRIPTIONS:
  □ Does each tool have a clear, specific description?
  □ Are parameter descriptions complete?
  □ Are there keywords users might actually use?
  □ Is there overlap between tool capabilities?
  □ Are edge cases mentioned (what NOT to use it for)?

CONVERSATION CONTEXT:
  □ Is important context near the end (recent messages)?
  □ Are there conflicting instructions in history?
  □ Has the original task been restated recently?
  □ Are tool results being properly attributed?
  
Remember: Agent bugs are often prompt bugs!
""")
    
    def show_debugging_flowchart(self) -> None:
        """Show the systematic debugging flowchart."""
        print("""
┌─────────────────────────────────────────────────────────┐
│                  AGENT NOT WORKING                       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: REPRODUCE THE ISSUE                             │
│ • Enable debug logging                                  │
│ • Capture the exact input that causes the problem       │
│ • Record the session for replay                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: CATEGORIZE THE FAILURE                          │
│ • Wrong tool selected? → Tool Selection Debugging       │
│ • Infinite loop? → Loop Detection                       │
│ • Off-topic response? → Conversation Flow Analysis      │
│ • Malformed output? → Response Validation               │
│ • Performance issue? → Metrics Analysis                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: CHECK THE PROMPT FIRST                          │
│ • Is the system prompt clear and complete?              │
│ • Are tool descriptions unambiguous?                    │
│ • Are there conflicting instructions?                   │
│ • Is context being lost (conversation too long)?        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: ANALYZE THE TRACE                               │
│ • Step through events chronologically                   │
│ • Identify the first point of divergence               │
│ • Check LLM response content for clues                  │
│ • Verify tool inputs and outputs                        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: APPLY THE FIX                                   │
│ • Modify prompts/descriptions if prompt issue           │
│ • Add guardrails if behavior issue                      │
│ • Fix code if implementation issue                      │
│ • Add validation if input/output issue                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: VERIFY AND PREVENT REGRESSION                   │
│ • Replay the original failing case                      │
│ • Add to test suite                                     │
│ • Document the issue and fix                            │
└─────────────────────────────────────────────────────────┘
""")
    
    def export_report(self, filepath: str) -> None:
        """Export the debugging report to a file."""
        if not self.context:
            print("❌ No active session.")
            return
        
        report = self.generate_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING HELPER DEMONSTRATION")
    print("=" * 60)
    
    helper = DebuggingHelper()
    
    # Show the debugging flowchart
    print("\n" + "-" * 60)
    print("SYSTEMATIC DEBUGGING APPROACH")
    print("-" * 60)
    helper.show_debugging_flowchart()
    
    # Start a debugging session
    helper.start_session(
        "Agent keeps calling the search tool repeatedly without giving an answer"
    )
    
    # Categorize
    helper.categorize()
    helper.set_category("2")  # infinite_loop
    
    # Record findings
    helper.add_finding("Search tool called 15 times with same query")
    helper.add_finding("No termination condition in system prompt")
    helper.add_finding("Tool results being ignored in follow-up calls")
    helper.add_finding("Stop reason is always 'tool_use', never 'end_turn'")
    
    # Get suggestions
    helper.suggest_fixes()
    
    # Record resolution
    helper.record_root_cause(
        "System prompt missing instruction to synthesize results and respond"
    )
    helper.record_fix(
        "Added 'After gathering information, synthesize results and provide a "
        "final answer. Do not call the same tool more than 3 times.' to system prompt"
    )
    
    # Generate report
    helper.generate_report()
    
    # Show prompt checklist
    helper.prompt_checklist()
    
    # Export report
    helper.export_report("/tmp/debug_report.json")
    
    print("\n✅ Debugging helper demonstration complete!")
