---
chapter: 37
title: "Debugging Agents"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 37: Debugging Agents

## Introduction

Your agent isn't working. Maybe it's calling the wrong tool. Maybe it's stuck in an infinite loop. Maybe it's giving nonsensical responses. The logs show a flurry of activity, but you can't pinpoint where things went wrong. Sound familiar?

Debugging AI agents is fundamentally different from debugging traditional software. When you debug a Python function, you can trace the exact execution path—every variable, every branch, every return value is deterministic. With agents, the LLM at the core introduces non-determinism: the same input can produce different outputs, reasoning paths can vary, and "bugs" might actually be prompt issues rather than code issues.

In the previous chapter, we built comprehensive logging and observability tools. Those tools let us *see* what the agent is doing. Now we need to interpret what we see and fix what's broken. This chapter gives you a systematic approach to debugging agents—from identifying common failure patterns to building tools that help you reproduce and fix problems.

Here's the key insight that will save you hours of frustration: **agent bugs are often prompt bugs**. Before you dive into code, always check your instructions first.

## Learning Objectives

By the end of this chapter, you will be able to:

- Identify and categorize common agent failure modes
- Debug conversation flow issues using trace analysis
- Diagnose and fix tool selection problems
- Detect and prevent infinite loops in agentic workflows
- Build replay systems to reproduce intermittent bugs
- Apply systematic debugging strategies to agent development

## Common Agent Failure Modes

Before we dive into debugging techniques, let's catalog what can go wrong. Understanding the failure modes helps you quickly diagnose problems.

### 1. Tool Selection Failures

**Symptom:** The agent uses the wrong tool or no tool at all.

**Common causes:**
- Ambiguous tool descriptions
- Overlapping tool functionality
- Missing tools for the task
- Prompt doesn't emphasize tool availability

**Example:** You ask "What's 25 times 47?" but the agent responds with "I believe 25 times 47 is around 1,175" instead of using the calculator tool. The response is wrong (correct answer: 1,175... actually that's right, but the agent guessed instead of calculating).

### 2. Infinite Loops

**Symptom:** The agent keeps calling tools without making progress or reaching a conclusion.

**Common causes:**
- Tool returns results that trigger the same tool call
- Missing termination conditions
- Circular dependencies between tools
- Agent can't recognize task completion

**Example:** Agent calls `search("weather forecast")`, gets results, decides it needs more info, calls `search("weather forecast")` again, and repeats indefinitely.

### 3. Conversation Derailment

**Symptom:** The agent loses track of the original goal or provides irrelevant responses.

**Common causes:**
- Long conversation history dilutes context
- System prompt gets overwhelmed by conversation
- Tool results confuse the agent
- User input contains contradictory instructions

**Example:** User asks for help writing an email, agent provides a draft, user asks to make it shorter, agent responds about a completely different topic from earlier in the conversation.

### 4. Malformed Tool Calls

**Symptom:** Tool execution fails because the agent provided invalid parameters.

**Common causes:**
- Schema mismatch (e.g., string instead of number)
- Missing required parameters
- Invalid parameter values (e.g., negative age)
- Format errors (e.g., wrong date format)

**Example:** Calculator tool expects `{"expression": "2+2"}` but agent provides `{"calculation": "two plus two"}`.

### 5. Hallucinated Tool Results

**Symptom:** Agent claims to have used a tool but actually made up the response.

**Common causes:**
- Agent sees tool in system prompt but wasn't actually given tools
- Previous conversation mentioned tool results
- Agent fills in results when tool call fails silently

**Example:** Agent says "I checked the weather API and it's 72°F in San Francisco" but no actual API call was made.

### 6. Context Window Exhaustion

**Symptom:** Agent behavior degrades as conversation gets longer.

**Common causes:**
- Important information pushed out of context
- System prompt gets truncated
- Tool results accumulate and crowd out instructions

**Example:** Agent follows instructions perfectly for the first 10 exchanges, then starts ignoring rules around exchange 20.

Let's build debugging tools to tackle each of these issues.

## The Debug Logger

First, let's enhance our logging from Chapter 36 with debugging-specific features. This logger captures more detail and makes it easier to analyze agent behavior.

```python
"""
Enhanced debug logger for AI agents.

Chapter 37: Debugging Agents
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DebugEvent:
    """A single debug event with full context."""
    timestamp: str
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    step_number: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DebugLogger:
    """
    A logger specifically designed for debugging AI agents.
    
    Features:
    - Step-by-step event tracking
    - Full conversation history capture
    - Tool call/result pairing
    - Easy export for analysis
    
    Usage:
        debug = DebugLogger()
        debug.start_trace("user-request-123")
        debug.log_event("user_input", "What's the weather?")
        debug.log_event("tool_call", "Calling weather API", tool="weather", params={...})
        debug.end_trace()
        debug.export("debug_session.json")
    """
    
    def __init__(self, name: str = "agent_debug", verbose: bool = True):
        """
        Initialize the debug logger.
        
        Args:
            name: Logger name
            verbose: If True, print events to console in real-time
        """
        self.name = name
        self.verbose = verbose
        self.events: list[DebugEvent] = []
        self.current_trace_id: Optional[str] = None
        self.step_counter = 0
        
        # Set up Python logger for verbose output
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def _now(self) -> str:
        """Get current timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    def start_trace(self, trace_id: str) -> None:
        """Start a new debug trace."""
        self.current_trace_id = trace_id
        self.step_counter = 0
        self.log_event("trace_start", f"Starting trace: {trace_id}")
    
    def end_trace(self) -> None:
        """End the current trace."""
        self.log_event("trace_end", f"Ending trace: {self.current_trace_id}")
        self.current_trace_id = None
    
    def log_event(
        self,
        event_type: str,
        message: str,
        **data: Any
    ) -> None:
        """
        Log a debug event.
        
        Args:
            event_type: Category of event (e.g., "tool_call", "llm_response")
            message: Human-readable description
            **data: Additional structured data
        """
        self.step_counter += 1
        
        event = DebugEvent(
            timestamp=self._now(),
            event_type=event_type,
            message=message,
            data=data,
            trace_id=self.current_trace_id,
            step_number=self.step_counter
        )
        
        self.events.append(event)
        
        if self.verbose:
            self.logger.debug(
                f"[Step {self.step_counter}] {event_type}: {message}"
            )
            if data:
                for key, value in data.items():
                    value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    self.logger.debug(f"  {key}: {value_str}")
    
    def log_user_input(self, content: str) -> None:
        """Log user input."""
        self.log_event("user_input", "User message received", content=content)
    
    def log_llm_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        model: str = ""
    ) -> None:
        """Log an LLM API request."""
        self.log_event(
            "llm_request",
            f"Sending request to {model}",
            message_count=len(messages),
            tool_count=len(tools) if tools else 0,
            last_message_role=messages[-1]["role"] if messages else None,
            model=model
        )
    
    def log_llm_response(
        self,
        stop_reason: str,
        content_blocks: int,
        has_tool_use: bool,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Log an LLM response."""
        self.log_event(
            "llm_response",
            f"Received response (stop: {stop_reason})",
            stop_reason=stop_reason,
            content_blocks=content_blocks,
            has_tool_use=has_tool_use,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> None:
        """Log a tool call request."""
        self.log_event(
            "tool_call",
            f"Calling tool: {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_use_id
        )
    
    def log_tool_result(
        self,
        tool_name: str,
        tool_use_id: str,
        result: Any,
        success: bool,
        duration_ms: float
    ) -> None:
        """Log a tool execution result."""
        self.log_event(
            "tool_result",
            f"Tool {tool_name} {'succeeded' if success else 'failed'}",
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            result=str(result)[:500],  # Truncate long results
            success=success,
            duration_ms=round(duration_ms, 2)
        )
    
    def log_error(self, error_type: str, message: str, **context: Any) -> None:
        """Log an error."""
        self.log_event(
            "error",
            f"{error_type}: {message}",
            error_type=error_type,
            **context
        )
    
    def log_warning(self, message: str, **context: Any) -> None:
        """Log a warning."""
        self.log_event("warning", message, **context)
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> list[DebugEvent]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type
            trace_id: Filter by trace ID
        
        Returns:
            List of matching events
        """
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if trace_id:
            events = [e for e in events if e.trace_id == trace_id]
        
        return events
    
    def get_trace_summary(self, trace_id: Optional[str] = None) -> dict[str, Any]:
        """Get a summary of a trace."""
        trace_id = trace_id or self.current_trace_id
        events = self.get_events(trace_id=trace_id)
        
        if not events:
            return {"error": "No events found for trace"}
        
        tool_calls = [e for e in events if e.event_type == "tool_call"]
        tool_results = [e for e in events if e.event_type == "tool_result"]
        errors = [e for e in events if e.event_type == "error"]
        llm_responses = [e for e in events if e.event_type == "llm_response"]
        
        return {
            "trace_id": trace_id,
            "total_events": len(events),
            "total_steps": events[-1].step_number if events else 0,
            "tool_calls": len(tool_calls),
            "llm_calls": len(llm_responses),
            "errors": len(errors),
            "tools_used": list(set(e.data.get("tool_name", "") for e in tool_calls)),
            "error_types": [e.data.get("error_type", "unknown") for e in errors],
        }
    
    def export(self, filepath: str) -> None:
        """Export all events to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                [e.to_dict() for e in self.events],
                f,
                indent=2
            )
        if self.verbose:
            self.logger.info(f"Exported {len(self.events)} events to {filepath}")
    
    def clear(self) -> None:
        """Clear all events."""
        self.events = []
        self.step_counter = 0
        self.current_trace_id = None
    
    def print_trace(self, trace_id: Optional[str] = None) -> None:
        """Print a human-readable trace summary."""
        events = self.get_events(trace_id=trace_id or self.current_trace_id)
        
        print("\n" + "=" * 60)
        print("DEBUG TRACE")
        print("=" * 60)
        
        for event in events:
            print(f"\n[Step {event.step_number}] {event.event_type.upper()}")
            print(f"  {event.message}")
            
            if event.data:
                for key, value in event.data.items():
                    value_str = str(value)
                    if len(value_str) > 80:
                        value_str = value_str[:80] + "..."
                    print(f"  • {key}: {value_str}")
        
        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    debug = DebugLogger(verbose=True)
    
    # Simulate a debugging session
    debug.start_trace("test-123")
    debug.log_user_input("What's the weather in Paris?")
    debug.log_llm_request(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{"name": "weather"}],
        model="claude-sonnet-4-20250514"
    )
    debug.log_llm_response(
        stop_reason="tool_use",
        content_blocks=2,
        has_tool_use=True,
        input_tokens=150,
        output_tokens=50
    )
    debug.log_tool_call(
        tool_name="weather",
        tool_input={"location": "Paris, France"},
        tool_use_id="tool_abc123"
    )
    debug.log_tool_result(
        tool_name="weather",
        tool_use_id="tool_abc123",
        result={"temp": 18, "condition": "cloudy"},
        success=True,
        duration_ms=245.5
    )
    debug.end_trace()
    
    # Print summary
    print("\nTrace Summary:")
    print(json.dumps(debug.get_trace_summary("test-123"), indent=2))
```

## Debugging Conversation Flow

Conversation flow issues are tricky because they emerge from the interaction between multiple messages. Let's build a tool to analyze conversation structure.

```python
"""
Conversation flow debugger.

Chapter 37: Debugging Agents
"""

import os
from typing import Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ConversationAnalysis:
    """Analysis results for a conversation."""
    total_messages: int
    total_tokens_estimate: int
    user_messages: int
    assistant_messages: int
    tool_uses: int
    tool_results: int
    system_prompt_present: bool
    potential_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ConversationDebugger:
    """
    Analyzes conversation structure to identify flow issues.
    
    Common issues detected:
    - Missing system prompt
    - Unbalanced user/assistant turns
    - Tool calls without results
    - Very long messages that may truncate
    - Context window exhaustion
    """
    
    # Approximate tokens per character (rough estimate)
    TOKENS_PER_CHAR = 0.25
    
    # Warning thresholds
    MAX_RECOMMENDED_TOKENS = 150000  # Before context issues start
    LONG_MESSAGE_THRESHOLD = 10000  # Characters
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)
    
    def analyze_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> ConversationAnalysis:
        """
        Analyze a conversation for potential issues.
        
        Args:
            messages: The messages array
            system_prompt: Optional system prompt
        
        Returns:
            ConversationAnalysis with findings
        """
        analysis = ConversationAnalysis(
            total_messages=len(messages),
            total_tokens_estimate=0,
            user_messages=0,
            assistant_messages=0,
            tool_uses=0,
            tool_results=0,
            system_prompt_present=system_prompt is not None
        )
        
        # Count system prompt tokens
        if system_prompt:
            analysis.total_tokens_estimate += self.estimate_tokens(system_prompt)
        
        pending_tool_calls: dict[str, str] = {}  # tool_use_id -> tool_name
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Count by role
            if role == "user":
                analysis.user_messages += 1
            elif role == "assistant":
                analysis.assistant_messages += 1
            
            # Estimate tokens for this message
            if isinstance(content, str):
                msg_tokens = self.estimate_tokens(content)
                analysis.total_tokens_estimate += msg_tokens
                
                # Check for very long messages
                if len(content) > self.LONG_MESSAGE_THRESHOLD:
                    analysis.potential_issues.append(
                        f"Message {i} is very long ({len(content)} chars). "
                        "Consider summarizing."
                    )
            
            elif isinstance(content, list):
                # Handle content blocks (tool use, tool results, etc.)
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        
                        if block_type == "tool_use":
                            analysis.tool_uses += 1
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")
                            pending_tool_calls[tool_id] = tool_name
                            
                            # Estimate tokens for tool call
                            input_str = str(block.get("input", {}))
                            analysis.total_tokens_estimate += self.estimate_tokens(input_str)
                        
                        elif block_type == "tool_result":
                            analysis.tool_results += 1
                            tool_id = block.get("tool_use_id", "")
                            
                            # Remove from pending
                            if tool_id in pending_tool_calls:
                                del pending_tool_calls[tool_id]
                            else:
                                analysis.potential_issues.append(
                                    f"Tool result for unknown tool_use_id: {tool_id}"
                                )
                            
                            # Estimate tokens for result
                            result_content = block.get("content", "")
                            analysis.total_tokens_estimate += self.estimate_tokens(
                                str(result_content)
                            )
                        
                        elif block_type == "text":
                            text = block.get("text", "")
                            analysis.total_tokens_estimate += self.estimate_tokens(text)
        
        # Check for issues
        if not analysis.system_prompt_present:
            analysis.potential_issues.append(
                "No system prompt detected. Agent may lack clear instructions."
            )
            analysis.recommendations.append(
                "Add a system prompt to define agent behavior and available tools."
            )
        
        if pending_tool_calls:
            analysis.potential_issues.append(
                f"Unanswered tool calls: {list(pending_tool_calls.values())}"
            )
            analysis.recommendations.append(
                "Ensure every tool_use block has a corresponding tool_result."
            )
        
        if analysis.tool_uses != analysis.tool_results:
            analysis.potential_issues.append(
                f"Tool use/result mismatch: {analysis.tool_uses} calls, "
                f"{analysis.tool_results} results"
            )
        
        if analysis.total_tokens_estimate > self.MAX_RECOMMENDED_TOKENS:
            analysis.potential_issues.append(
                f"Estimated {analysis.total_tokens_estimate} tokens. "
                "Context window may be exhausted."
            )
            analysis.recommendations.append(
                "Summarize or truncate older messages to reduce context size."
            )
        
        # Check turn balance
        if abs(analysis.user_messages - analysis.assistant_messages) > 2:
            analysis.potential_issues.append(
                f"Unbalanced turns: {analysis.user_messages} user, "
                f"{analysis.assistant_messages} assistant"
            )
        
        return analysis
    
    def find_derailment_point(
        self,
        messages: list[dict[str, Any]],
        expected_topic: str
    ) -> Optional[int]:
        """
        Find where a conversation went off-topic.
        
        Uses Claude to analyze each turn for relevance to the expected topic.
        
        Args:
            messages: The conversation messages
            expected_topic: What the conversation should be about
        
        Returns:
            Index of first off-topic message, or None if all on-topic
        """
        analysis_prompt = f"""Analyze if this message is relevant to the topic: "{expected_topic}"

Message: {{message}}

Respond with ONLY "relevant" or "off-topic" followed by a brief explanation."""
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    block.get("text", "") 
                    for block in content 
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            
            if not content or len(content) < 20:
                continue
            
            # Analyze this message
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": analysis_prompt.format(message=content[:500])
                }]
            )
            
            result = response.content[0].text.lower()
            if "off-topic" in result:
                return i
        
        return None
    
    def suggest_context_reduction(
        self,
        messages: list[dict[str, Any]],
        target_reduction: float = 0.5
    ) -> list[str]:
        """
        Suggest which messages can be removed or summarized.
        
        Args:
            messages: The conversation messages
            target_reduction: Target reduction ratio (0.5 = reduce by half)
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Find tool result messages (often verbose)
        for i, msg in enumerate(messages):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_content = str(block.get("content", ""))
                        if len(result_content) > 1000:
                            suggestions.append(
                                f"Message {i}: Summarize tool result "
                                f"({len(result_content)} chars)"
                            )
        
        # Find repetitive user messages
        user_contents = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                for j, prev_content in enumerate(user_contents):
                    # Simple similarity check
                    if len(set(content.split()) & set(prev_content.split())) > 10:
                        suggestions.append(
                            f"Message {i}: May be repetitive with earlier message"
                        )
                        break
                user_contents.append(content)
        
        # Suggest removing very old messages
        if len(messages) > 20:
            suggestions.append(
                f"Consider summarizing messages 0-{len(messages)//2} "
                "into a single context message"
            )
        
        return suggestions


def print_analysis(analysis: ConversationAnalysis) -> None:
    """Print a formatted analysis report."""
    print("\n" + "=" * 60)
    print("CONVERSATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal messages: {analysis.total_messages}")
    print(f"Estimated tokens: {analysis.total_tokens_estimate:,}")
    print(f"User messages: {analysis.user_messages}")
    print(f"Assistant messages: {analysis.assistant_messages}")
    print(f"Tool uses: {analysis.tool_uses}")
    print(f"Tool results: {analysis.tool_results}")
    print(f"System prompt: {'Yes' if analysis.system_prompt_present else 'No'}")
    
    if analysis.potential_issues:
        print("\n⚠️  POTENTIAL ISSUES:")
        for issue in analysis.potential_issues:
            print(f"  • {issue}")
    
    if analysis.recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    debugger = ConversationDebugger()
    
    # Example conversation with issues
    problematic_conversation = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll check the weather for you."},
            {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "Tokyo"}}
        ]},
        # Missing tool_result!
        {"role": "user", "content": "Also, what about Paris?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tool_2", "name": "weather", "input": {"city": "Paris"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tool_2", "content": "Paris: 18°C, cloudy"}
        ]},
        # Multiple user messages in a row
        {"role": "user", "content": "Thanks!"},
        {"role": "user", "content": "One more question..."},
    ]
    
    analysis = debugger.analyze_messages(problematic_conversation)
    print_analysis(analysis)
```

## Debugging Tool Selection

When your agent uses the wrong tool—or no tool at all—you need to understand why. This debugger helps analyze tool selection decisions.

```python
"""
Tool selection debugger.

Chapter 37: Debugging Agents
"""

import os
import json
from typing import Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ToolSelectionAnalysis:
    """Analysis of a tool selection decision."""
    query: str
    expected_tool: Optional[str]
    selected_tool: Optional[str]
    correct: bool
    confidence_explanation: str
    tool_scores: dict[str, str]  # tool_name -> reasoning
    suggestions: list[str]


class ToolSelectionDebugger:
    """
    Debugs tool selection issues.
    
    Helps answer:
    - Why did the agent choose this tool?
    - Why didn't the agent choose the expected tool?
    - How can I improve tool descriptions?
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def analyze_selection(
        self,
        query: str,
        tools: list[dict[str, Any]],
        selected_tool: Optional[str],
        expected_tool: Optional[str] = None
    ) -> ToolSelectionAnalysis:
        """
        Analyze why a particular tool was (or wasn't) selected.
        
        Args:
            query: The user's query
            tools: Available tool definitions
            selected_tool: The tool that was actually selected (None if no tool used)
            expected_tool: The tool you expected to be selected
        
        Returns:
            ToolSelectionAnalysis with detailed findings
        """
        tool_names = [t["name"] for t in tools]
        tool_descriptions = {
            t["name"]: t.get("description", "No description") 
            for t in tools
        }
        
        analysis_prompt = f"""Analyze this tool selection scenario:

USER QUERY: "{query}"

AVAILABLE TOOLS:
{json.dumps(tool_descriptions, indent=2)}

SELECTED TOOL: {selected_tool if selected_tool else "None (no tool used)"}
EXPECTED TOOL: {expected_tool if expected_tool else "Not specified"}

Please analyze:
1. For each tool, explain how well it matches the query (score 1-10)
2. Explain why the selected tool was likely chosen
3. If the selection seems wrong, explain why and suggest fixes

Format your response as JSON:
{{
    "tool_scores": {{"tool_name": "score/10 - reasoning", ...}},
    "selection_reasoning": "why the selected tool was chosen",
    "is_correct": true/false,
    "suggestions": ["suggestion 1", "suggestion 2"]
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        # Extract JSON from response
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
            else:
                result = {
                    "tool_scores": {},
                    "selection_reasoning": response_text,
                    "is_correct": selected_tool == expected_tool,
                    "suggestions": []
                }
        except json.JSONDecodeError:
            result = {
                "tool_scores": {},
                "selection_reasoning": response_text,
                "is_correct": selected_tool == expected_tool,
                "suggestions": []
            }
        
        return ToolSelectionAnalysis(
            query=query,
            expected_tool=expected_tool,
            selected_tool=selected_tool,
            correct=result.get("is_correct", selected_tool == expected_tool),
            confidence_explanation=result.get("selection_reasoning", ""),
            tool_scores=result.get("tool_scores", {}),
            suggestions=result.get("suggestions", [])
        )
    
    def suggest_description_improvements(
        self,
        tool: dict[str, Any],
        failed_queries: list[str]
    ) -> list[str]:
        """
        Suggest improvements to a tool description based on queries it failed to match.
        
        Args:
            tool: The tool definition
            failed_queries: Queries where this tool should have been selected but wasn't
        
        Returns:
            List of suggested description improvements
        """
        prompt = f"""A tool with this definition is NOT being selected when it should be:

TOOL NAME: {tool['name']}
CURRENT DESCRIPTION: {tool.get('description', 'No description')}
PARAMETERS: {json.dumps(tool.get('input_schema', {}), indent=2)}

QUERIES WHERE THIS TOOL SHOULD HAVE BEEN USED (BUT WASN'T):
{chr(10).join(f'- "{q}"' for q in failed_queries)}

Suggest 3-5 specific improvements to the tool description that would help the LLM 
recognize when to use this tool. Focus on:
1. Keywords that users might use
2. Clearer explanation of capabilities
3. Examples of when to use it
4. Distinction from similar tools

Format as a numbered list."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse suggestions from response
        response_text = response.content[0].text
        suggestions = []
        
        for line in response_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering/bullets
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    suggestions.append(cleaned)
        
        return suggestions
    
    def test_tool_selection(
        self,
        tools: list[dict[str, Any]],
        test_cases: list[tuple[str, str]]  # (query, expected_tool)
    ) -> dict[str, Any]:
        """
        Test tool selection across multiple queries.
        
        Args:
            tools: Tool definitions
            test_cases: List of (query, expected_tool_name) tuples
        
        Returns:
            Test results with pass/fail for each case
        """
        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "cases": []
        }
        
        for query, expected in test_cases:
            # Make an actual API call to see what gets selected
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                tools=tools,
                messages=[{"role": "user", "content": query}]
            )
            
            # Check if a tool was used
            selected = None
            for block in response.content:
                if block.type == "tool_use":
                    selected = block.name
                    break
            
            passed = selected == expected
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["cases"].append({
                "query": query,
                "expected": expected,
                "selected": selected,
                "passed": passed
            })
        
        return results


def print_tool_analysis(analysis: ToolSelectionAnalysis) -> None:
    """Print formatted tool selection analysis."""
    print("\n" + "=" * 60)
    print("TOOL SELECTION ANALYSIS")
    print("=" * 60)
    
    print(f"\nQuery: \"{analysis.query}\"")
    print(f"Expected tool: {analysis.expected_tool or 'Not specified'}")
    print(f"Selected tool: {analysis.selected_tool or 'None'}")
    print(f"Correct: {'✅ Yes' if analysis.correct else '❌ No'}")
    
    if analysis.tool_scores:
        print("\nTool Scores:")
        for tool, score in analysis.tool_scores.items():
            print(f"  • {tool}: {score}")
    
    if analysis.confidence_explanation:
        print(f"\nReasoning: {analysis.confidence_explanation}")
    
    if analysis.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in analysis.suggestions:
            print(f"  • {suggestion}")
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    debugger = ToolSelectionDebugger()
    
    # Define tools
    tools = [
        {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        },
        {
            "name": "weather",
            "description": "Gets current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "unit_converter",
            "description": "Converts between units of measurement",
            "input_schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    ]
    
    # Test a potentially ambiguous query
    analysis = debugger.analyze_selection(
        query="What's 5 kilometers in miles?",
        tools=tools,
        selected_tool="calculator",  # Simulated wrong selection
        expected_tool="unit_converter"
    )
    
    print_tool_analysis(analysis)
    
    # Get description improvement suggestions
    print("\n" + "=" * 60)
    print("DESCRIPTION IMPROVEMENT SUGGESTIONS")
    print("=" * 60)
    
    suggestions = debugger.suggest_description_improvements(
        tool=tools[2],  # unit_converter
        failed_queries=[
            "What's 5 kilometers in miles?",
            "Convert 100 fahrenheit to celsius",
            "How many pounds is 50 kilograms?"
        ]
    )
    
    print(f"\nFor tool: unit_converter")
    print(f"Current description: {tools[2]['description']}")
    print("\nSuggested improvements:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
```

## Detecting and Preventing Infinite Loops

Infinite loops are one of the most frustrating agent bugs. Let's build a detector and prevention system.

```python
"""
Infinite loop detection and prevention.

Chapter 37: Debugging Agents
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
        for tool, count in self.tool_counts.items():
            if count >= self.config.max_tool_call_total // len(self.tool_counts.keys() or [1]):
                # Allow more if there are more tools
                pass
        
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
    print("LOOP DETECTION DEMO")
    print("=" * 60)
    
    detector = LoopDetector()
    
    # Simulate an agent getting stuck in a loop
    print("\nSimulating agent execution with a loop...")
    
    tool_sequence = [
        ("search", {"query": "weather today"}),
        ("search", {"query": "weather today"}),  # Repeat
        ("search", {"query": "weather today"}),  # Repeat again
        ("calculate", {"expression": "2+2"}),
        ("search", {"query": "weather today"}),  # Back to search
        ("calculate", {"expression": "2+2"}),     # Oscillation start
        ("search", {"query": "weather today"}),
        ("calculate", {"expression": "2+2"}),
        ("search", {"query": "weather today"}),
        ("calculate", {"expression": "2+2"}),
    ]
    
    for i, (tool, input_data) in enumerate(tool_sequence):
        print(f"\nStep {i + 1}: Calling {tool}")
        
        if detector.should_stop():
            print(f"  ⛔ STOPPED: {detector.get_stop_reason()}")
            break
        
        detector.record_tool_call(tool, input_data)
        
        if detector.detected_patterns:
            print(f"  ⚠️  Patterns detected: {len(detector.detected_patterns)}")
    
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(json.dumps(detector.get_summary(), indent=2))
```

## Replay and Reproduction

When bugs are intermittent, you need to capture and replay agent sessions. This system allows you to record everything and replay it later for debugging.

```python
"""
Agent session replay system.

Chapter 37: Debugging Agents
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class RecordedEvent:
    """A single recorded event in a session."""
    timestamp: str
    event_type: str  # "llm_request", "llm_response", "tool_call", "tool_result"
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
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "events": [asdict(e) for e in self.events],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedSession":
        events = [
            RecordedEvent(**e) for e in data.get("events", [])
        ]
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


class SessionRecorder:
    """
    Records agent sessions for later replay.
    
    Captures:
    - All LLM requests and responses
    - Tool calls and results
    - Timing information
    - System prompts and tool definitions
    
    Usage:
        recorder = SessionRecorder()
        recorder.start_session("session-123", model="claude-sonnet-4-20250514")
        
        # Record events as they happen
        recorder.record_llm_request(messages)
        recorder.record_llm_response(response)
        recorder.record_tool_call(tool_name, tool_input)
        recorder.record_tool_result(tool_name, result)
        
        # Save for later
        recorder.end_session()
        recorder.save("session-123.json")
    """
    
    def __init__(self):
        self.current_session: Optional[RecordedSession] = None
        self.sequence_counter = 0
    
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def start_session(
        self,
        session_id: str,
        model: str = "",
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Start recording a new session."""
        self.current_session = RecordedSession(
            session_id=session_id,
            started_at=self._now(),
            model=model,
            system_prompt=system_prompt,
            tools=tools or [],
            metadata=metadata or {}
        )
        self.sequence_counter = 0
    
    def end_session(self) -> None:
        """End the current session."""
        if self.current_session:
            self.current_session.ended_at = self._now()
    
    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event."""
        if not self.current_session:
            raise RuntimeError("No active session. Call start_session() first.")
        
        self.sequence_counter += 1
        event = RecordedEvent(
            timestamp=self._now(),
            event_type=event_type,
            data=data,
            sequence_number=self.sequence_counter
        )
        self.current_session.events.append(event)
    
    def record_llm_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> None:
        """Record an LLM request."""
        self._record_event("llm_request", {
            "messages": messages,
            **kwargs
        })
    
    def record_llm_response(
        self,
        content: list[dict[str, Any]],
        stop_reason: str,
        usage: dict[str, int]
    ) -> None:
        """Record an LLM response."""
        # Convert content blocks to serializable format
        serializable_content = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    serializable_content.append({
                        "type": "text",
                        "text": block.text
                    })
                elif block.type == "tool_use":
                    serializable_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
            else:
                serializable_content.append(block)
        
        self._record_event("llm_response", {
            "content": serializable_content,
            "stop_reason": stop_reason,
            "usage": usage
        })
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> None:
        """Record a tool call."""
        self._record_event("tool_call", {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": tool_use_id
        })
    
    def record_tool_result(
        self,
        tool_name: str,
        tool_use_id: str,
        result: Any,
        duration_ms: float
    ) -> None:
        """Record a tool result."""
        self._record_event("tool_result", {
            "tool_name": tool_name,
            "tool_use_id": tool_use_id,
            "result": str(result),
            "duration_ms": duration_ms
        })
    
    def save(self, filepath: str) -> None:
        """Save the session to a file."""
        if not self.current_session:
            raise RuntimeError("No session to save")
        
        with open(filepath, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)
    
    def get_session(self) -> Optional[RecordedSession]:
        """Get the current session."""
        return self.current_session


class SessionPlayer:
    """
    Replays recorded agent sessions.
    
    Modes:
    - Step-by-step: Pause after each event
    - Continuous: Play all events
    - Analysis: Extract insights without replay
    
    Usage:
        player = SessionPlayer()
        session = player.load("session-123.json")
        
        # Step through
        for event in player.step():
            print(event)
            input("Press Enter to continue...")
        
        # Or analyze
        analysis = player.analyze()
    """
    
    def __init__(self):
        self.session: Optional[RecordedSession] = None
        self.current_index = 0
    
    def load(self, filepath: str) -> RecordedSession:
        """Load a recorded session."""
        with open(filepath) as f:
            data = json.load(f)
        self.session = RecordedSession.from_dict(data)
        self.current_index = 0
        return self.session
    
    def load_from_dict(self, data: dict[str, Any]) -> RecordedSession:
        """Load a session from a dictionary."""
        self.session = RecordedSession.from_dict(data)
        self.current_index = 0
        return self.session
    
    def reset(self) -> None:
        """Reset to the beginning of the session."""
        self.current_index = 0
    
    def step(self) -> Optional[RecordedEvent]:
        """Get the next event."""
        if not self.session or self.current_index >= len(self.session.events):
            return None
        
        event = self.session.events[self.current_index]
        self.current_index += 1
        return event
    
    def step_all(self) -> list[RecordedEvent]:
        """Get all remaining events."""
        if not self.session:
            return []
        
        events = self.session.events[self.current_index:]
        self.current_index = len(self.session.events)
        return events
    
    def analyze(self) -> dict[str, Any]:
        """Analyze the recorded session."""
        if not self.session:
            return {"error": "No session loaded"}
        
        events = self.session.events
        
        # Count event types
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        # Find tool calls
        tool_calls = [e for e in events if e.event_type == "tool_call"]
        tool_results = [e for e in events if e.event_type == "tool_result"]
        
        # Calculate timing
        tool_durations = []
        for result in tool_results:
            duration = result.data.get("duration_ms", 0)
            tool_durations.append(duration)
        
        # Find potential issues
        issues = []
        
        # Check for repeated tool calls
        tool_call_hashes = []
        for tc in tool_calls:
            hash_str = f"{tc.data['tool_name']}:{tc.data['tool_input']}"
            if hash_str in tool_call_hashes:
                issues.append(f"Duplicate tool call: {tc.data['tool_name']}")
            tool_call_hashes.append(hash_str)
        
        # Check for missing tool results
        tool_call_ids = {tc.data["tool_use_id"] for tc in tool_calls}
        tool_result_ids = {tr.data["tool_use_id"] for tr in tool_results}
        missing_results = tool_call_ids - tool_result_ids
        if missing_results:
            issues.append(f"Missing tool results for: {missing_results}")
        
        return {
            "session_id": self.session.session_id,
            "duration_seconds": self._calculate_duration(),
            "event_counts": event_counts,
            "total_events": len(events),
            "tool_stats": {
                "total_calls": len(tool_calls),
                "unique_tools": list(set(tc.data["tool_name"] for tc in tool_calls)),
                "avg_duration_ms": sum(tool_durations) / len(tool_durations) if tool_durations else 0,
            },
            "issues_found": issues,
        }
    
    def _calculate_duration(self) -> float:
        """Calculate session duration in seconds."""
        if not self.session or not self.session.events:
            return 0
        
        first = datetime.fromisoformat(self.session.events[0].timestamp)
        last = datetime.fromisoformat(self.session.events[-1].timestamp)
        return (last - first).total_seconds()
    
    def find_events(
        self,
        event_type: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> list[RecordedEvent]:
        """Find events matching criteria."""
        if not self.session:
            return []
        
        events = self.session.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if tool_name:
            events = [
                e for e in events 
                if e.data.get("tool_name") == tool_name
            ]
        
        return events
    
    def print_timeline(self) -> None:
        """Print a visual timeline of the session."""
        if not self.session:
            print("No session loaded")
            return
        
        print("\n" + "=" * 60)
        print(f"SESSION TIMELINE: {self.session.session_id}")
        print("=" * 60)
        
        for event in self.session.events:
            icon = {
                "llm_request": "📤",
                "llm_response": "📥",
                "tool_call": "🔧",
                "tool_result": "✅",
            }.get(event.event_type, "•")
            
            # Format the event description
            if event.event_type == "llm_request":
                desc = f"Request with {len(event.data.get('messages', []))} messages"
            elif event.event_type == "llm_response":
                desc = f"Response (stop: {event.data.get('stop_reason', 'unknown')})"
            elif event.event_type == "tool_call":
                desc = f"Call {event.data.get('tool_name', 'unknown')}"
            elif event.event_type == "tool_result":
                desc = f"Result from {event.data.get('tool_name', 'unknown')} ({event.data.get('duration_ms', 0):.0f}ms)"
            else:
                desc = event.event_type
            
            print(f"\n{icon} [{event.sequence_number:02d}] {desc}")
            
            # Show relevant details
            if event.event_type == "tool_call":
                print(f"   Input: {json.dumps(event.data.get('tool_input', {}))[:60]}...")
            elif event.event_type == "tool_result":
                result = str(event.data.get('result', ''))[:60]
                print(f"   Result: {result}...")
        
        print("\n" + "=" * 60)


# Example usage demonstrating recording and replay
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION RECORDING AND REPLAY DEMO")
    print("=" * 60)
    
    # Record a session
    recorder = SessionRecorder()
    recorder.start_session(
        session_id="demo-session-001",
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
        tools=[{"name": "weather", "description": "Get weather"}]
    )
    
    # Simulate some events
    recorder.record_llm_request(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}]
    )
    
    # Simulate a response with tool use (using dict format for demo)
    recorder._record_event("llm_response", {
        "content": [
            {"type": "text", "text": "Let me check the weather."},
            {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "Paris"}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 30}
    })
    
    recorder.record_tool_call("weather", {"city": "Paris"}, "tool_1")
    recorder.record_tool_result("weather", "tool_1", "Paris: 18°C, cloudy", 150.5)
    
    recorder.record_llm_request(
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "Paris: 18°C, cloudy"}]}
        ]
    )
    
    recorder._record_event("llm_response", {
        "content": [{"type": "text", "text": "The weather in Paris is 18°C and cloudy."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 80, "output_tokens": 20}
    })
    
    recorder.end_session()
    
    # Save the session
    session_file = "/tmp/demo_session.json"
    recorder.save(session_file)
    print(f"\n✅ Session saved to {session_file}")
    
    # Replay and analyze
    print("\n" + "-" * 60)
    print("REPLAYING SESSION")
    print("-" * 60)
    
    player = SessionPlayer()
    player.load(session_file)
    
    # Print timeline
    player.print_timeline()
    
    # Analyze
    print("\n" + "-" * 60)
    print("SESSION ANALYSIS")
    print("-" * 60)
    analysis = player.analyze()
    print(json.dumps(analysis, indent=2))
```

## A Systematic Debugging Approach

Now let's tie it all together with a systematic approach to debugging agents. Here's a flowchart-style methodology:

```
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
```

Let's implement a debugging helper that walks you through this process:

```python
"""
Systematic debugging helper.

Chapter 37: Debugging Agents
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
    
    Guides you through a structured debugging process.
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
        ],
        "infinite_loop": [
            "Add maximum iteration limits",
            "Check for exit conditions in the prompt",
            "Add loop detection",
            "Verify tool results don't trigger the same call",
        ],
        "conversation_flow": [
            "Strengthen the system prompt",
            "Summarize long conversations",
            "Add explicit task reminders",
            "Check for context window exhaustion",
        ],
        "malformed_output": [
            "Add output format examples to the prompt",
            "Use structured output mode",
            "Add response validation",
            "Simplify the expected format",
        ],
        "performance": [
            "Cache repeated operations",
            "Use a faster model for simple tasks",
            "Reduce context size",
            "Parallelize independent operations",
        ],
        "error": [
            "Add retry logic with backoff",
            "Validate inputs before sending",
            "Check API key and permissions",
            "Handle rate limits gracefully",
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
    
    def categorize(self) -> str:
        """Help categorize the issue."""
        print("\n" + "-" * 60)
        print("STEP 1: CATEGORIZE THE ISSUE")
        print("-" * 60)
        print("\nSelect the category that best matches your issue:\n")
        
        for key, (_, description) in self.CATEGORIES.items():
            print(f"  {key}. {description}")
        
        # In a real interactive session, you'd get user input
        # For this example, we'll return a method to set it
        return "Use set_category(number) to select"
    
    def set_category(self, category_num: str) -> None:
        """Set the issue category."""
        if category_num not in self.CATEGORIES:
            print(f"Invalid category. Choose from: {list(self.CATEGORIES.keys())}")
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
        
        checklists = {
            "tool_selection": [
                "□ Verify the expected tool is in the tools list",
                "□ Check tool description for clarity",
                "□ Look for overlapping tool functionality",
                "□ Check if query matches tool keywords",
                "□ Review the LLM's reasoning (if visible)",
            ],
            "infinite_loop": [
                "□ Check total iteration count",
                "□ Look for repeated identical tool calls",
                "□ Check for oscillation patterns (A→B→A→B)",
                "□ Verify termination conditions exist",
                "□ Check if tool results are being processed",
            ],
            "conversation_flow": [
                "□ Review system prompt completeness",
                "□ Check conversation length (token count)",
                "□ Look for conflicting instructions",
                "□ Verify tool results aren't confusing",
                "□ Check if task context is maintained",
            ],
            "malformed_output": [
                "□ Check expected vs actual output format",
                "□ Verify JSON/structured output settings",
                "□ Look for truncated responses",
                "□ Check for encoding issues",
                "□ Verify schema definitions",
            ],
            "performance": [
                "□ Measure time per LLM call",
                "□ Count total tokens used",
                "□ Identify repeated operations",
                "□ Check for unnecessary tool calls",
                "□ Review context size over time",
            ],
            "error": [
                "□ Check error message and stack trace",
                "□ Verify API key is valid",
                "□ Check for rate limiting",
                "□ Validate input data",
                "□ Check network connectivity",
            ],
        }
        
        for item in checklists.get(category, []):
            print(f"  {item}")
    
    def add_finding(self, finding: str) -> None:
        """Add a debugging finding."""
        if self.context:
            self.context.findings.append(finding)
            print(f"📝 Finding recorded: {finding}")
    
    def suggest_fixes(self) -> list[str]:
        """Suggest fixes based on the category."""
        if not self.context or not self.context.category:
            return ["Set a category first using categorize()"]
        
        print("\n" + "-" * 60)
        print("STEP 3: SUGGESTED FIXES")
        print("-" * 60)
        
        fixes = self.COMMON_FIXES.get(self.context.category, [])
        
        print(f"\nCommon fixes for {self.context.category} issues:\n")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
        
        return fixes
    
    def record_fix(self, fix_description: str) -> None:
        """Record the fix that was applied."""
        if self.context:
            self.context.fix_applied = fix_description
            print(f"\n✅ Fix recorded: {fix_description}")
    
    def record_root_cause(self, root_cause: str) -> None:
        """Record the identified root cause."""
        if self.context:
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
            "recommendations": self.COMMON_FIXES.get(self.context.category, [])
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


# Example usage
if __name__ == "__main__":
    helper = DebuggingHelper()
    
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
    helper.add_finding("Tool results being ignored")
    
    # Get suggestions
    helper.suggest_fixes()
    
    # Record resolution
    helper.record_root_cause(
        "System prompt missing instruction to synthesize results and respond"
    )
    helper.record_fix(
        "Added 'After gathering information, synthesize results and provide a final answer' to system prompt"
    )
    
    # Generate report
    helper.generate_report()
    
    # Show prompt checklist
    helper.prompt_checklist()
```

## Common Pitfalls

**1. Debugging in production without logging**

Never deploy an agent without comprehensive logging. By the time you notice a problem, the evidence is gone. Always capture:
- Full conversation history
- All tool calls and results
- Timing information
- Error details

**2. Assuming code bugs when it's a prompt issue**

Most "bugs" in agent behavior are actually prompt issues. Before diving into code:
- Read the system prompt carefully
- Check tool descriptions for ambiguity
- Verify the agent received the instructions you think it did

**3. Not creating reproducible test cases**

When you find a bug, capture everything needed to reproduce it:
- The exact input
- The system state at the time
- The conversation history
- Random seeds if applicable

Save these as test cases to prevent regressions.

**4. Debugging without understanding expected behavior**

Before debugging, clearly define what the agent *should* do. Write out:
- Expected tool selection for this input
- Expected response format
- Expected number of iterations

Without clear expectations, you can't identify deviations.

## Practical Exercise

**Task:** Build a diagnostic tool that analyzes a recorded agent session and produces a bug report

**Requirements:**

1. Load a recorded session (JSON format from SessionRecorder)
2. Analyze for all issue types (tool selection, loops, conversation flow)
3. Generate a comprehensive bug report including:
   - Issue category
   - Evidence from the trace
   - Suggested root cause
   - Recommended fixes
4. Highlight the specific events where problems occurred

**Hints:**

- Combine the LoopDetector, ConversationDebugger, and ToolSelectionDebugger
- Use the SessionPlayer to iterate through events
- Generate the report in both human-readable and JSON formats

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Agent bugs are often prompt bugs**—always check your system prompt and tool descriptions first
- **Categorize issues** before debugging: tool selection, infinite loops, conversation flow, malformed output, performance, or errors
- **Use structured debugging tools**: DebugLogger for tracing, LoopDetector for infinite loops, ConversationDebugger for flow issues
- **Record and replay sessions** to reproduce intermittent bugs—capture everything
- **Follow a systematic process**: reproduce → categorize → analyze → fix → verify
- **Build test cases from bugs**—every fixed bug should become a regression test

## What's Next

Now that you can find and fix bugs in your agents, the next chapter covers **Cost Optimization**. You'll learn how to track token usage, reduce costs through prompt optimization, and implement caching strategies. Because in production, an agent that works but costs too much is still a problem.

