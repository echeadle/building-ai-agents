---
chapter: 33
title: "The Complete Agent Class"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 33: The Complete Agent Class

## Introduction

You've come a long way. Over the past seven chapters, you've built every piece of a production-ready agent system: the agentic loop that drives autonomous behavior, state management that gives agents memory, planning capabilities that help agents think before acting, error handling that keeps them resilient, human-in-the-loop controls for safety, and guardrails that prevent them from going off the rails.

Now it's time to bring everything together.

In this chapter, we'll assemble all these components into a single, well-architected `Agent` class. This isn't just about combining code—it's about creating a clean, maintainable design that you can extend and customize for any use case. By the end, you'll have a complete agent implementation that represents the culmination of everything we've learned.

## Learning Objectives

By the end of this chapter, you will be able to:

- Architect a complete Agent class that integrates all agentic components
- Configure agents for different use cases through a flexible configuration system
- Create clean interfaces that hide complexity while exposing the right controls
- Document and test an agent implementation properly
- Extend the base Agent class for specialized applications

## The Architecture Overview

Before we dive into code, let's visualize how all our components fit together:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Agent Class                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │    Config    │  │    State     │  │    Tool Registry     │  │
│  │  - model     │  │  - history   │  │  - available tools   │  │
│  │  - limits    │  │  - memory    │  │  - tool handlers     │  │
│  │  - features  │  │  - context   │  │  - descriptions      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Agentic Loop                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │   │
│  │  │ Perceive│→ │  Plan   │→ │   Act   │→ │  Evaluate   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Guardrails  │  │ Human-in-   │  │   Error Handler      │  │
│  │  - input     │  │  the-Loop   │  │  - retry logic       │  │
│  │  - output    │  │  - approvals│  │  - fallbacks         │  │
│  │  - action    │  │  - feedback │  │  - recovery          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The Agent class serves as the orchestrator, coordinating all these components while exposing a simple interface to the outside world. Let's build it piece by piece.

## The Configuration System

Good agents are configurable. Rather than hardcoding values, we'll create a configuration system that makes it easy to customize agent behavior without modifying code.

### The AgentConfig Class

```python
"""
Agent configuration system.

Chapter 33: The Complete Agent Class
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PlanningMode(Enum):
    """How the agent approaches planning."""
    NONE = "none"           # No explicit planning
    SIMPLE = "simple"       # Plan once, then execute
    ADAPTIVE = "adaptive"   # Plan and revise as needed


class HumanApprovalMode(Enum):
    """When to request human approval."""
    NEVER = "never"                 # Fully autonomous
    HIGH_RISK = "high_risk"         # Only for dangerous actions
    ALWAYS = "always"               # Every action needs approval


@dataclass
class AgentConfig:
    """
    Configuration for an Agent instance.
    
    This dataclass holds all configurable parameters for an agent,
    with sensible defaults that work for most use cases.
    """
    
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # System prompt
    system_prompt: str = "You are a helpful AI assistant."
    
    # Loop controls
    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 5
    
    # Planning
    planning_mode: PlanningMode = PlanningMode.SIMPLE
    
    # Human-in-the-loop
    approval_mode: HumanApprovalMode = HumanApprovalMode.HIGH_RISK
    high_risk_actions: list[str] = field(default_factory=lambda: [
        "delete", "send_email", "make_purchase", "execute_code"
    ])
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True
    
    # Guardrails
    input_validation_enabled: bool = True
    output_filtering_enabled: bool = True
    action_constraints_enabled: bool = True
    allowed_tools: Optional[list[str]] = None  # None means all tools allowed
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"password", r"credit.?card", r"ssn", r"social.?security"
    ])
    
    # State management
    max_history_tokens: int = 8000
    persist_state: bool = False
    state_file: Optional[str] = None
    
    # Observability
    verbose: bool = False
    log_tool_calls: bool = True
    log_llm_responses: bool = False  # Can be expensive/verbose
    
    def validate(self) -> list[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        
        if self.max_tokens < 100:
            errors.append("max_tokens should be at least 100")
        
        if self.temperature < 0 or self.temperature > 1:
            errors.append("temperature must be between 0 and 1")
        
        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")
        
        if self.persist_state and not self.state_file:
            errors.append("state_file required when persist_state is True")
        
        return errors
    
    @classmethod
    def for_simple_chat(cls) -> "AgentConfig":
        """Configuration preset for simple chat agents."""
        return cls(
            planning_mode=PlanningMode.NONE,
            approval_mode=HumanApprovalMode.NEVER,
            max_iterations=1,
        )
    
    @classmethod
    def for_autonomous_agent(cls) -> "AgentConfig":
        """Configuration preset for autonomous agents."""
        return cls(
            planning_mode=PlanningMode.ADAPTIVE,
            approval_mode=HumanApprovalMode.HIGH_RISK,
            max_iterations=20,
            persist_state=True,
        )
    
    @classmethod
    def for_safe_agent(cls) -> "AgentConfig":
        """Configuration preset for maximum safety."""
        return cls(
            planning_mode=PlanningMode.SIMPLE,
            approval_mode=HumanApprovalMode.ALWAYS,
            max_iterations=5,
            input_validation_enabled=True,
            output_filtering_enabled=True,
            action_constraints_enabled=True,
        )
```

The configuration system uses Python's `dataclass` for clean, type-hinted settings. Notice how we provide sensible defaults—an agent created with `AgentConfig()` will work reasonably well out of the box.

The class methods like `for_simple_chat()` and `for_autonomous_agent()` are **configuration presets** that make it easy to create agents for common use cases.

## The Tool Registry

We need a clean way to manage tools. The tool registry handles registration, validation, and execution of tools.

```python
"""
Tool registry for managing agent tools.

Chapter 33: The Complete Agent Class
"""

import json
from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """A registered tool with its handler."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]
    requires_approval: bool = False
    

class ToolRegistry:
    """
    Registry for managing agent tools.
    
    Handles tool registration, validation, and execution.
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any],
        requires_approval: bool = False
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Unique tool name
            description: What the tool does (LLM reads this!)
            input_schema: JSON Schema for parameters
            handler: Function to call when tool is invoked
            requires_approval: Whether this tool needs human approval
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            requires_approval=requires_approval
        )
    
    def register_decorator(
        self,
        name: str,
        description: str,
        input_schema: dict,
        requires_approval: bool = False
    ) -> Callable:
        """
        Decorator for registering tools.
        
        Usage:
            @registry.register_decorator("add", "Add two numbers", {...})
            def add(a: int, b: int) -> int:
                return a + b
        """
        def decorator(func: Callable) -> Callable:
            self.register(name, description, input_schema, func, requires_approval)
            return func
        return decorator
    
    def get(self, name: str) -> ToolDefinition:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def list_names(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())
    
    def get_definitions_for_api(
        self,
        allowed_tools: list[str] | None = None
    ) -> list[dict]:
        """
        Get tool definitions in the format expected by the Claude API.
        
        Args:
            allowed_tools: Optional list to filter tools. None means all.
            
        Returns:
            List of tool definitions for the API
        """
        definitions = []
        
        for name, tool in self._tools.items():
            if allowed_tools is not None and name not in allowed_tools:
                continue
                
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
        
        return definitions
    
    def execute(self, name: str, arguments: dict) -> Any:
        """
        Execute a tool with the given arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments as a dictionary
            
        Returns:
            Tool execution result
        """
        tool = self.get(name)
        return tool.handler(**arguments)
    
    def requires_approval(self, name: str) -> bool:
        """Check if a tool requires human approval."""
        return self.get(name).requires_approval
```

The registry provides a clean interface for working with tools. The `register_decorator` method lets you register tools with minimal boilerplate:

```python
registry = ToolRegistry()

@registry.register_decorator(
    "calculate",
    "Perform a mathematical calculation",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
)
def calculate(expression: str) -> float:
    """Safely evaluate a math expression."""
    # Safe evaluation logic here
    pass
```

## The Agent State

State management keeps track of conversation history, working memory, and any context the agent needs to persist.

```python
"""
Agent state management.

Chapter 33: The Complete Agent Class
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", or "system"
    content: Any  # str or list of content blocks
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_api_format(self) -> dict:
        """Convert to format expected by Claude API."""
        return {"role": self.role, "content": self.content}


@dataclass  
class ToolCall:
    """Record of a tool call."""
    tool_name: str
    arguments: dict
    result: Any
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: str | None = None


class AgentState:
    """
    Manages all state for an agent.
    
    Includes conversation history, working memory, and tool call history.
    """
    
    def __init__(self, max_history_tokens: int = 8000):
        self.max_history_tokens = max_history_tokens
        self.messages: list[Message] = []
        self.tool_calls: list[ToolCall] = []
        self.working_memory: dict[str, Any] = {}
        self.current_plan: list[str] | None = None
        self.completed_steps: list[str] = []
        self.created_at: datetime = datetime.now()
        self.last_updated: datetime = datetime.now()
    
    def add_message(self, role: str, content: Any) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))
        self.last_updated = datetime.now()
        self._trim_history_if_needed()
    
    def add_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        success: bool = True,
        error: str | None = None
    ) -> None:
        """Record a tool call."""
        self.tool_calls.append(ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            error=error
        ))
        self.last_updated = datetime.now()
    
    def get_messages_for_api(self) -> list[dict]:
        """Get messages in API format."""
        return [msg.to_api_format() for msg in self.messages]
    
    def set_memory(self, key: str, value: Any) -> None:
        """Store a value in working memory."""
        self.working_memory[key] = value
        self.last_updated = datetime.now()
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from working memory."""
        return self.working_memory.get(key, default)
    
    def clear_memory(self) -> None:
        """Clear working memory."""
        self.working_memory = {}
        self.last_updated = datetime.now()
    
    def set_plan(self, steps: list[str]) -> None:
        """Set the current plan."""
        self.current_plan = steps
        self.completed_steps = []
        self.last_updated = datetime.now()
    
    def complete_step(self, step: str) -> None:
        """Mark a step as completed."""
        self.completed_steps.append(step)
        self.last_updated = datetime.now()
    
    def get_remaining_steps(self) -> list[str]:
        """Get steps that haven't been completed yet."""
        if not self.current_plan:
            return []
        return [s for s in self.current_plan if s not in self.completed_steps]
    
    def _trim_history_if_needed(self) -> None:
        """Trim old messages if we exceed token limits."""
        # Simple estimation: ~4 chars per token
        estimated_tokens = sum(
            len(str(msg.content)) // 4 for msg in self.messages
        )
        
        while estimated_tokens > self.max_history_tokens and len(self.messages) > 2:
            # Keep at least the system message and latest message
            # Remove oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    removed = self.messages.pop(i)
                    estimated_tokens -= len(str(removed.content)) // 4
                    break
    
    def save(self, filepath: str) -> None:
        """Save state to a JSON file."""
        data = {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.messages
            ],
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": str(tc.result),  # Serialize result
                    "timestamp": tc.timestamp.isoformat(),
                    "success": tc.success,
                    "error": tc.error
                }
                for tc in self.tool_calls
            ],
            "working_memory": self.working_memory,
            "current_plan": self.current_plan,
            "completed_steps": self.completed_steps,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: str, max_history_tokens: int = 8000) -> "AgentState":
        """Load state from a JSON file."""
        data = json.loads(Path(filepath).read_text())
        
        state = cls(max_history_tokens=max_history_tokens)
        
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"])
            )
            state.messages.append(msg)
        
        for tc_data in data.get("tool_calls", []):
            tc = ToolCall(
                tool_name=tc_data["tool_name"],
                arguments=tc_data["arguments"],
                result=tc_data["result"],
                timestamp=datetime.fromisoformat(tc_data["timestamp"]),
                success=tc_data["success"],
                error=tc_data["error"]
            )
            state.tool_calls.append(tc)
        
        state.working_memory = data.get("working_memory", {})
        state.current_plan = data.get("current_plan")
        state.completed_steps = data.get("completed_steps", [])
        state.created_at = datetime.fromisoformat(data["created_at"])
        state.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return state
    
    def get_summary(self) -> str:
        """Get a summary of the current state."""
        return (
            f"Messages: {len(self.messages)}, "
            f"Tool calls: {len(self.tool_calls)}, "
            f"Memory keys: {list(self.working_memory.keys())}, "
            f"Plan steps remaining: {len(self.get_remaining_steps())}"
        )
```

## The Guardrails Module

Safety is non-negotiable. The guardrails module provides input validation, output filtering, and action constraints.

```python
"""
Guardrails for agent safety.

Chapter 33: The Complete Agent Class
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: str | None = None
    modified_content: Any = None  # For filtering operations


class Guardrails:
    """
    Safety guardrails for agent operations.
    
    Provides input validation, output filtering, and action constraints.
    """
    
    def __init__(
        self,
        blocked_patterns: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        max_tool_result_length: int = 10000
    ):
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in (blocked_patterns or [])
        ]
        self.allowed_tools = allowed_tools
        self.max_tool_result_length = max_tool_result_length
    
    def validate_input(self, user_input: str) -> GuardrailResult:
        """
        Validate user input before processing.
        
        Checks for blocked patterns and potential injection attacks.
        """
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(user_input):
                return GuardrailResult(
                    passed=False,
                    reason=f"Input contains blocked pattern: {pattern.pattern}"
                )
        
        # Check for potential prompt injection attempts
        injection_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"disregard\s+(your|all)\s+(rules|instructions)",
            r"you\s+are\s+now\s+(a|an)\s+",
            r"new\s+instruction:",
            r"system\s*:",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason="Potential prompt injection detected"
                )
        
        return GuardrailResult(passed=True)
    
    def filter_output(self, output: str) -> GuardrailResult:
        """
        Filter agent output before returning to user.
        
        Removes sensitive information and enforces length limits.
        """
        filtered = output
        
        # Remove any accidentally exposed patterns
        sensitive_patterns = [
            (r"api[_-]?key[:\s]*[a-zA-Z0-9\-_]+", "[API_KEY_REDACTED]"),
            (r"password[:\s]*\S+", "[PASSWORD_REDACTED]"),
            (r"sk-[a-zA-Z0-9]+", "[SECRET_KEY_REDACTED]"),
        ]
        
        for pattern, replacement in sensitive_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)
        
        return GuardrailResult(
            passed=True,
            modified_content=filtered
        )
    
    def check_tool_allowed(self, tool_name: str) -> GuardrailResult:
        """Check if a tool is allowed to be used."""
        if self.allowed_tools is None:
            return GuardrailResult(passed=True)
        
        if tool_name in self.allowed_tools:
            return GuardrailResult(passed=True)
        
        return GuardrailResult(
            passed=False,
            reason=f"Tool '{tool_name}' is not in the allowed list"
        )
    
    def validate_tool_result(self, result: Any) -> GuardrailResult:
        """Validate tool execution result."""
        result_str = str(result)
        
        if len(result_str) > self.max_tool_result_length:
            # Truncate overly long results
            truncated = result_str[:self.max_tool_result_length] + "... [TRUNCATED]"
            return GuardrailResult(
                passed=True,
                modified_content=truncated
            )
        
        return GuardrailResult(passed=True, modified_content=result)
    
    def check_action_safety(
        self,
        action_type: str,
        parameters: dict
    ) -> GuardrailResult:
        """
        Check if an action is safe to perform.
        
        This is a hook for custom safety logic.
        """
        # Prevent file system access outside safe directories
        if "path" in parameters or "file" in parameters:
            path = parameters.get("path") or parameters.get("file", "")
            dangerous_paths = ["/etc", "/usr", "/bin", "/root", ".."]
            
            for dangerous in dangerous_paths:
                if dangerous in str(path):
                    return GuardrailResult(
                        passed=False,
                        reason=f"Access to '{dangerous}' paths is not allowed"
                    )
        
        # Prevent dangerous shell commands
        if action_type == "execute_command":
            command = str(parameters.get("command", ""))
            dangerous_commands = ["rm -rf", "sudo", "chmod 777", "mkfs", "> /dev"]
            
            for dangerous in dangerous_commands:
                if dangerous in command:
                    return GuardrailResult(
                        passed=False,
                        reason=f"Dangerous command pattern detected: {dangerous}"
                    )
        
        return GuardrailResult(passed=True)
```

## The Error Handler

Robust error handling keeps agents running even when things go wrong.

```python
"""
Error handling for agents.

Chapter 33: The Complete Agent Class
"""

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Can continue with degraded functionality
    MEDIUM = "medium"     # Should retry or use fallback
    HIGH = "high"         # Should stop and report
    CRITICAL = "critical" # Immediate stop, possible data loss


@dataclass
class AgentError:
    """Represents an error that occurred during agent execution."""
    error_type: str
    message: str
    severity: ErrorSeverity
    recoverable: bool
    context: dict | None = None
    original_exception: Exception | None = None
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.error_type}: {self.message}"


class ErrorHandler:
    """
    Handles errors during agent execution.
    
    Provides retry logic, fallback behaviors, and error categorization.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        fallback_enabled: bool = True
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_enabled = fallback_enabled
        self.error_history: list[AgentError] = []
    
    def categorize_error(self, exception: Exception) -> AgentError:
        """Categorize an exception into an AgentError."""
        error_type = type(exception).__name__
        message = str(exception)
        
        # Categorize by exception type
        if "rate" in message.lower() or "limit" in message.lower():
            return AgentError(
                error_type="RateLimitError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        if "timeout" in message.lower() or "timed out" in message.lower():
            return AgentError(
                error_type="TimeoutError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        if "auth" in message.lower() or "key" in message.lower():
            return AgentError(
                error_type="AuthenticationError",
                message=message,
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
                original_exception=exception
            )
        
        if "connection" in message.lower() or "network" in message.lower():
            return AgentError(
                error_type="ConnectionError",
                message=message,
                severity=ErrorSeverity.MEDIUM,
                recoverable=True,
                original_exception=exception
            )
        
        # Default categorization
        return AgentError(
            error_type=error_type,
            message=message,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            original_exception=exception
        )
    
    def with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> tuple[Any, AgentError | None]:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Tuple of (result, error). Error is None on success.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, None
            
            except Exception as e:
                error = self.categorize_error(e)
                self.error_history.append(error)
                last_error = error
                
                if not error.recoverable:
                    return None, error
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
        
        return None, last_error
    
    def handle_tool_error(
        self,
        tool_name: str,
        exception: Exception,
        fallback_result: Any = None
    ) -> tuple[Any, bool]:
        """
        Handle an error from tool execution.
        
        Args:
            tool_name: Name of the tool that failed
            exception: The exception that occurred
            fallback_result: Value to return if fallback is enabled
            
        Returns:
            Tuple of (result, success). Result may be fallback value.
        """
        error = self.categorize_error(exception)
        error.context = {"tool_name": tool_name}
        self.error_history.append(error)
        
        if self.fallback_enabled and fallback_result is not None:
            return fallback_result, False
        
        if error.recoverable:
            return f"Tool '{tool_name}' failed temporarily. Please try again.", False
        
        return f"Tool '{tool_name}' failed: {error.message}", False
    
    def get_error_summary(self) -> str:
        """Get a summary of errors encountered."""
        if not self.error_history:
            return "No errors recorded."
        
        summary_lines = [f"Total errors: {len(self.error_history)}"]
        
        # Count by severity
        by_severity = {}
        for error in self.error_history:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        for severity, count in sorted(by_severity.items()):
            summary_lines.append(f"  {severity}: {count}")
        
        return "\n".join(summary_lines)
    
    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history = []
```

## The Complete Agent Class

Now we bring everything together. The `Agent` class is the main interface that coordinates all components.

```python
"""
The Complete Agent Class.

Chapter 33: The Complete Agent Class

This module provides a production-ready Agent class that integrates:
- Configurable behavior through AgentConfig
- Tool management through ToolRegistry
- State management through AgentState
- Safety through Guardrails
- Reliability through ErrorHandler
"""

import os
from typing import Any, Generator
from datetime import datetime
from dotenv import load_dotenv
import anthropic

from config import AgentConfig, PlanningMode, HumanApprovalMode
from tools import ToolRegistry
from state import AgentState
from guardrails import Guardrails
from errors import ErrorHandler, ErrorSeverity

# Load environment variables
load_dotenv()


class Agent:
    """
    A complete, production-ready AI agent.
    
    This class integrates all agentic components into a cohesive system
    that can perceive, plan, act, and learn from its environment.
    
    Example:
        >>> config = AgentConfig.for_autonomous_agent()
        >>> agent = Agent(config)
        >>> agent.register_tool("calculator", ...)
        >>> response = agent.run("What is 25 * 17?")
    """
    
    def __init__(
        self,
        config: AgentConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        state: AgentState | None = None
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration. Uses defaults if not provided.
            tool_registry: Pre-configured tool registry. Creates new if not provided.
            state: Existing state to resume. Creates new if not provided.
        """
        self.config = config or AgentConfig()
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        # Initialize components
        self.tools = tool_registry or ToolRegistry()
        self.state = state or AgentState(
            max_history_tokens=self.config.max_history_tokens
        )
        self.guardrails = Guardrails(
            blocked_patterns=self.config.blocked_patterns,
            allowed_tools=self.config.allowed_tools
        )
        self.error_handler = ErrorHandler(
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            fallback_enabled=self.config.fallback_enabled
        )
        
        # Initialize API client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic()
        
        # Track execution metrics
        self.metrics = {
            "total_iterations": 0,
            "total_tool_calls": 0,
            "total_tokens_used": 0,
            "successful_runs": 0,
            "failed_runs": 0
        }
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: callable,
        requires_approval: bool = False
    ) -> None:
        """
        Register a tool for the agent to use.
        
        Args:
            name: Unique tool name
            description: What the tool does (the LLM reads this!)
            input_schema: JSON Schema for parameters
            handler: Function to execute when tool is called
            requires_approval: Whether tool needs human approval
        """
        self.tools.register(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            requires_approval=requires_approval
        )
    
    def run(self, user_input: str) -> str:
        """
        Run the agent with the given user input.
        
        This is the main entry point for interacting with the agent.
        It handles the complete agentic loop including planning,
        tool use, and response generation.
        
        Args:
            user_input: The user's message or request
            
        Returns:
            The agent's final response
        """
        # Input validation
        if self.config.input_validation_enabled:
            validation = self.guardrails.validate_input(user_input)
            if not validation.passed:
                return f"I can't process that request: {validation.reason}"
        
        # Add user message to state
        self.state.add_message("user", user_input)
        
        try:
            # Planning phase (if enabled)
            if self.config.planning_mode != PlanningMode.NONE:
                self._create_plan(user_input)
            
            # Execute the agentic loop
            response = self._run_agentic_loop()
            
            # Output filtering
            if self.config.output_filtering_enabled:
                filter_result = self.guardrails.filter_output(response)
                response = filter_result.modified_content
            
            self.metrics["successful_runs"] += 1
            return response
            
        except Exception as e:
            self.metrics["failed_runs"] += 1
            error = self.error_handler.categorize_error(e)
            
            if error.severity == ErrorSeverity.CRITICAL:
                raise
            
            return f"I encountered an error: {error.message}"
        
        finally:
            # Save state if persistence is enabled
            if self.config.persist_state and self.config.state_file:
                self.state.save(self.config.state_file)
    
    def run_streaming(self, user_input: str) -> Generator[str, None, None]:
        """
        Run the agent with streaming output.
        
        Yields chunks of the response as they're generated.
        
        Args:
            user_input: The user's message or request
            
        Yields:
            Chunks of the agent's response
        """
        # Input validation
        if self.config.input_validation_enabled:
            validation = self.guardrails.validate_input(user_input)
            if not validation.passed:
                yield f"I can't process that request: {validation.reason}"
                return
        
        self.state.add_message("user", user_input)
        
        # For streaming, we use a simplified loop
        messages = self.state.get_messages_for_api()
        tools = self.tools.get_definitions_for_api(self.config.allowed_tools)
        
        with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                full_response += text
                yield text
            
            self.state.add_message("assistant", full_response)
    
    def _create_plan(self, user_input: str) -> None:
        """Create an execution plan for the task."""
        planning_prompt = f"""Analyze this request and create a step-by-step plan.

Request: {user_input}

Create a numbered list of 3-7 concrete steps to accomplish this task.
Each step should be specific and actionable.
Consider what tools might be needed.

Respond with just the numbered steps, nothing else."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": planning_prompt}]
        )
        
        plan_text = response.content[0].text
        steps = [
            line.strip() 
            for line in plan_text.split("\n") 
            if line.strip() and line.strip()[0].isdigit()
        ]
        
        self.state.set_plan(steps)
        
        if self.config.verbose:
            print(f"📋 Plan created with {len(steps)} steps")
    
    def _run_agentic_loop(self) -> str:
        """
        Execute the main agentic loop.
        
        Returns:
            The final response after all iterations
        """
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            self.metrics["total_iterations"] += 1
            
            if self.config.verbose:
                print(f"🔄 Iteration {iteration}/{self.config.max_iterations}")
            
            # Get current messages and tools
            messages = self.state.get_messages_for_api()
            tools = self.tools.get_definitions_for_api(self.config.allowed_tools)
            
            # Make API call
            response, error = self.error_handler.with_retry(
                self._call_llm,
                messages,
                tools
            )
            
            if error:
                return f"Failed after retries: {error.message}"
            
            # Process response
            assistant_content = response.content
            stop_reason = response.stop_reason
            
            # Track token usage
            self.metrics["total_tokens_used"] += (
                response.usage.input_tokens + response.usage.output_tokens
            )
            
            # Check if we're done (no tool use)
            if stop_reason == "end_turn":
                # Extract text response
                text_parts = [
                    block.text 
                    for block in assistant_content 
                    if hasattr(block, "text")
                ]
                final_response = "\n".join(text_parts)
                self.state.add_message("assistant", final_response)
                return final_response
            
            # Handle tool use
            if stop_reason == "tool_use":
                # Store assistant's response with tool calls
                self.state.add_message("assistant", assistant_content)
                
                # Process each tool call
                tool_results = self._process_tool_calls(assistant_content)
                
                # Add tool results to messages
                self.state.add_message("user", tool_results)
        
        # Max iterations reached
        return "I wasn't able to complete the task within the allowed steps. Here's what I accomplished so far."
    
    def _call_llm(self, messages: list[dict], tools: list[dict]) -> Any:
        """Make an API call to the LLM."""
        return self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN
        )
    
    def _process_tool_calls(self, assistant_content: list) -> list[dict]:
        """
        Process tool calls from the assistant's response.
        
        Returns:
            List of tool results in API format
        """
        tool_results = []
        tool_calls_this_turn = 0
        
        for block in assistant_content:
            if block.type != "tool_use":
                continue
            
            if tool_calls_this_turn >= self.config.max_tool_calls_per_iteration:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Maximum tool calls per turn exceeded. Please continue."
                })
                continue
            
            tool_name = block.name
            tool_input = block.input
            
            if self.config.verbose:
                print(f"🔧 Tool call: {tool_name}")
            
            # Check if tool is allowed
            if self.config.action_constraints_enabled:
                allowed = self.guardrails.check_tool_allowed(tool_name)
                if not allowed.passed:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool not allowed: {allowed.reason}",
                        "is_error": True
                    })
                    continue
            
            # Check if approval is needed
            if self._needs_approval(tool_name):
                approved = self._request_approval(tool_name, tool_input)
                if not approved:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Action was not approved by user.",
                        "is_error": True
                    })
                    continue
            
            # Execute the tool
            try:
                result = self.tools.execute(tool_name, tool_input)
                
                # Validate result
                validation = self.guardrails.validate_tool_result(result)
                result = validation.modified_content
                
                self.state.add_tool_call(
                    tool_name=tool_name,
                    arguments=tool_input,
                    result=result,
                    success=True
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
                
            except Exception as e:
                fallback, success = self.error_handler.handle_tool_error(
                    tool_name, e, fallback_result="Tool execution failed."
                )
                
                self.state.add_tool_call(
                    tool_name=tool_name,
                    arguments=tool_input,
                    result=None,
                    success=False,
                    error=str(e)
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(fallback),
                    "is_error": True
                })
            
            tool_calls_this_turn += 1
            self.metrics["total_tool_calls"] += 1
        
        return tool_results
    
    def _needs_approval(self, tool_name: str) -> bool:
        """Check if a tool call needs human approval."""
        if self.config.approval_mode == HumanApprovalMode.NEVER:
            return False
        
        if self.config.approval_mode == HumanApprovalMode.ALWAYS:
            return True
        
        # HIGH_RISK mode
        if self.tools.requires_approval(tool_name):
            return True
        
        # Check if tool name matches high-risk patterns
        for pattern in self.config.high_risk_actions:
            if pattern.lower() in tool_name.lower():
                return True
        
        return False
    
    def _request_approval(self, tool_name: str, tool_input: dict) -> bool:
        """
        Request human approval for a tool call.
        
        Returns:
            True if approved, False otherwise
        """
        print("\n" + "="*50)
        print("🚨 APPROVAL REQUIRED")
        print("="*50)
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_input}")
        print("="*50)
        
        while True:
            response = input("Approve this action? (yes/no): ").strip().lower()
            if response in ("yes", "y"):
                return True
            elif response in ("no", "n"):
                return False
            print("Please enter 'yes' or 'no'")
    
    def reset(self) -> None:
        """Reset the agent's state for a new conversation."""
        self.state = AgentState(
            max_history_tokens=self.config.max_history_tokens
        )
        self.error_handler.clear_history()
    
    def get_metrics(self) -> dict:
        """Get execution metrics."""
        return self.metrics.copy()
    
    def get_state_summary(self) -> str:
        """Get a summary of the current state."""
        return self.state.get_summary()
    
    def __repr__(self) -> str:
        return (
            f"Agent(model={self.config.model}, "
            f"tools={len(self.tools.list_names())}, "
            f"planning={self.config.planning_mode.value})"
        )
```

## Using the Complete Agent

Let's see how to use our complete Agent class in practice:

```python
"""
Example usage of the complete Agent class.

Chapter 33: The Complete Agent Class
"""

import os
from dotenv import load_dotenv

from agent import Agent
from config import AgentConfig, PlanningMode, HumanApprovalMode

# Load environment variables
load_dotenv()


def main():
    # Create a configuration
    config = AgentConfig(
        system_prompt="""You are a helpful assistant with access to tools.
        Always explain what you're doing before using a tool.
        Be concise but thorough in your responses.""",
        planning_mode=PlanningMode.SIMPLE,
        approval_mode=HumanApprovalMode.HIGH_RISK,
        max_iterations=10,
        verbose=True
    )
    
    # Create the agent
    agent = Agent(config)
    
    # Register some tools
    agent.register_tool(
        name="calculator",
        description="Perform mathematical calculations. Use for any math operations.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        },
        handler=lambda expression: eval(expression)  # Note: Use safe eval in production!
    )
    
    agent.register_tool(
        name="get_current_time",
        description="Get the current date and time.",
        input_schema={
            "type": "object",
            "properties": {}
        },
        handler=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Run the agent
    response = agent.run("What is 42 * 17? Also, what time is it?")
    print("\n" + "="*50)
    print("FINAL RESPONSE:")
    print("="*50)
    print(response)
    
    # Show metrics
    print("\n" + "="*50)
    print("METRICS:")
    print("="*50)
    print(agent.get_metrics())


if __name__ == "__main__":
    from datetime import datetime
    main()
```

## Configuration Presets in Action

The configuration presets make it easy to create agents for specific use cases:

```python
# For simple chat bots
chat_config = AgentConfig.for_simple_chat()
chat_agent = Agent(chat_config)

# For autonomous research agents
research_config = AgentConfig.for_autonomous_agent()
research_config.system_prompt = "You are a research assistant..."
research_agent = Agent(research_config)

# For maximum safety
safe_config = AgentConfig.for_safe_agent()
safe_config.allowed_tools = ["read_file", "search"]  # Limit tools
safe_agent = Agent(safe_config)
```

## Extending the Agent Class

The Agent class is designed to be extended. Here's how to create a specialized agent:

```python
"""
Example of extending the Agent class.

Chapter 33: The Complete Agent Class
"""

from agent import Agent
from config import AgentConfig


class ResearchAgent(Agent):
    """
    A specialized agent for research tasks.
    
    Adds domain-specific functionality on top of the base Agent.
    """
    
    def __init__(self, config: AgentConfig | None = None):
        # Set research-specific defaults
        if config is None:
            config = AgentConfig.for_autonomous_agent()
            config.system_prompt = """You are an expert research assistant.
            Your job is to find, analyze, and synthesize information.
            Always cite your sources and present findings clearly.
            Think step by step when analyzing complex topics."""
            config.max_iterations = 20
        
        super().__init__(config)
        
        # Track research-specific data
        self.sources: list[str] = []
        self.findings: list[dict] = []
    
    def research(self, topic: str) -> dict:
        """
        Conduct research on a topic.
        
        Returns:
            Dictionary with findings, sources, and summary
        """
        prompt = f"""Research the following topic thoroughly: {topic}

        Please:
        1. Search for relevant information
        2. Analyze multiple sources
        3. Synthesize your findings
        4. Provide a clear summary with key points
        5. List all sources consulted"""
        
        response = self.run(prompt)
        
        return {
            "topic": topic,
            "summary": response,
            "sources": self.sources,
            "findings": self.findings,
            "metrics": self.get_metrics()
        }
    
    def add_source(self, url: str, title: str) -> None:
        """Track a source used in research."""
        self.sources.append({"url": url, "title": title})
    
    def add_finding(self, finding: str, confidence: float) -> None:
        """Record a research finding."""
        self.findings.append({
            "finding": finding,
            "confidence": confidence
        })


class CodeAssistantAgent(Agent):
    """
    A specialized agent for code assistance.
    """
    
    def __init__(self, language: str = "python"):
        config = AgentConfig(
            system_prompt=f"""You are an expert {language} developer.
            Help users write, debug, and improve their code.
            Always explain your reasoning.
            Write clean, well-documented code.""",
            planning_mode=PlanningMode.SIMPLE,
            max_iterations=5
        )
        
        super().__init__(config)
        self.language = language
    
    def review_code(self, code: str) -> str:
        """Review code and provide feedback."""
        return self.run(f"""Please review this {self.language} code:

```{self.language}
{code}
```

Provide:
1. Overall assessment
2. Any bugs or issues found
3. Suggestions for improvement
4. A revised version if needed""")
    
    def explain_code(self, code: str) -> str:
        """Explain what code does."""
        return self.run(f"""Explain this {self.language} code in detail:

```{self.language}
{code}
```

Break down what each part does.""")
```

## Common Pitfalls

### 1. Forgetting to Validate Configuration

Always validate your configuration before creating an agent:

```python
# ❌ Bad: No validation
config = AgentConfig(max_iterations=-5)  # Invalid!
agent = Agent(config)  # Will fail later in unexpected ways

# ✅ Good: Validate first
config = AgentConfig(max_iterations=-5)
errors = config.validate()
if errors:
    print(f"Config errors: {errors}")
else:
    agent = Agent(config)
```

### 2. Not Handling State Persistence Errors

If you enable state persistence, handle file I/O errors:

```python
# ❌ Bad: Assumes save always works
config = AgentConfig(persist_state=True, state_file="/invalid/path/state.json")

# ✅ Good: Handle potential errors
config = AgentConfig(persist_state=True, state_file="./agent_state.json")
try:
    agent = Agent(config)
except Exception as e:
    print(f"Failed to initialize agent: {e}")
```

### 3. Registering Tools After Running

Register all tools before running the agent:

```python
# ❌ Bad: Registering tool mid-conversation
agent = Agent(config)
response1 = agent.run("What is 2+2?")  # No calculator tool!
agent.register_tool("calculator", ...)  # Too late for this conversation

# ✅ Good: Register all tools first
agent = Agent(config)
agent.register_tool("calculator", ...)
response1 = agent.run("What is 2+2?")  # Calculator is available
```

## Practical Exercise

**Task:** Create a specialized `CustomerServiceAgent` that extends the base Agent class.

**Requirements:**
1. Extend the base `Agent` class
2. Add a `ticket_history` list to track support tickets
3. Create a custom `handle_ticket()` method that processes customer issues
4. Add a tool for looking up customer information
5. Add a tool for creating support tickets
6. Configure appropriate guardrails for customer data

**Hints:**
- Use `AgentConfig.for_safe_agent()` as your base configuration
- The customer lookup tool should require approval (sensitive data)
- Track ticket IDs and outcomes in your custom class

**Solution:** See `code/exercise_customer_service_agent.py`

## Key Takeaways

- **Good architecture makes agents maintainable**: The Agent class cleanly separates concerns—configuration, state, tools, guardrails, and error handling each have their place.

- **Configuration enables flexibility**: Using a dedicated `AgentConfig` class with validation and presets makes it easy to create agents for different use cases.

- **Composition over inheritance**: The Agent class uses composition to integrate components, but inheritance works well for creating specialized agents.

- **Every component is testable**: Because each component (tools, state, guardrails, errors) is independent, you can test them in isolation.

- **The interface is simple**: Despite all the internal complexity, the public API (`run()`, `register_tool()`, `reset()`) is straightforward and easy to use.

## What's Next

Congratulations—you now have a complete, production-ready Agent class! But building the agent is only half the battle. In Part 5, we'll focus on taking agents from prototype to production. Chapter 34 begins with testing: how do you test something that behaves non-deterministically? We'll explore practical strategies for validating agent behavior and building confidence in your implementations.
