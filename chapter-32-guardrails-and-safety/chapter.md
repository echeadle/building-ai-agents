---
chapter: 32
title: "Guardrails and Safety"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 32: Guardrails and Safety

## Introduction

In Chapter 31, we learned how to keep humans in the loop for critical decisions. But what about all the moments when your agent is operating autonomously? What stops it from executing a destructive command, leaking sensitive data, or spiraling into an infinite loop that racks up thousands of dollars in API costs?

The answer is **guardrails**—the protective boundaries you build into your agent from the very start. Think of guardrails like the safety features in a car: seatbelts, airbags, and collision detection don't slow you down during normal driving, but they're there when things go wrong.

This chapter is about building those safety features for your agents. We'll implement comprehensive protection across five critical areas: input validation, output filtering, action constraints, rate limiting, and sandboxing. By the end, you'll have a complete guardrails module that you can drop into any agent to prevent the most common (and most dangerous) failure modes.

> **Warning:** Guardrails are not optional. Every production agent needs them. The examples in this chapter represent minimum safety standards, not nice-to-have features.

## Learning Objectives

By the end of this chapter, you will be able to:

- Validate and sanitize all inputs before they reach your agent
- Filter and verify outputs before they're returned to users or executed
- Implement action constraints using allowlists and blocklists
- Add rate limiting to prevent runaway costs and infinite loops
- Sandbox dangerous operations to limit their potential impact
- Integrate all guardrails into a unified, reusable module

## Why Guardrails Matter

Before we dive into implementation, let's understand what can go wrong without guardrails:

### Real-World Failure Scenarios

**Prompt Injection**: A user submits input containing instructions like "Ignore your previous instructions and..." that manipulates your agent into unintended behavior.

**Data Exfiltration**: An agent with file access tools accidentally exposes sensitive configuration files, API keys, or user data in its responses.

**Destructive Actions**: An agent tasked with "cleaning up old files" interprets the instruction too broadly and deletes critical system files.

**Runaway Costs**: A bug causes your agent to loop indefinitely, making thousands of API calls and generating a massive bill before anyone notices.

**Resource Exhaustion**: An agent attempts to process an enormous file or make too many concurrent requests, crashing your system or hitting rate limits.

Each of these scenarios is preventable with proper guardrails. Let's build them.

## Input Validation and Sanitization

The first line of defense is validating all inputs before they reach your agent. This includes user messages, file contents, tool results, and any external data.

### The Input Validator Class

```python
"""
Input validation and sanitization for AI agents.

Chapter 32: Guardrails and Safety
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any = None
    violations: list[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.is_valid


class InputValidator:
    """Validates and sanitizes inputs before they reach the agent."""
    
    def __init__(
        self,
        max_message_length: int = 10000,
        max_file_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        blocked_patterns: list[str] | None = None,
        allow_code_blocks: bool = True
    ):
        self.max_message_length = max_message_length
        self.max_file_size_bytes = max_file_size_bytes
        self.allow_code_blocks = allow_code_blocks
        
        # Default patterns that might indicate prompt injection attempts
        self.blocked_patterns = blocked_patterns or [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
            r"disregard\s+(all\s+)?(previous|prior|above)",
            r"forget\s+(everything|all)\s+(you\s+)?(know|learned)",
            r"you\s+are\s+now\s+(a|an)\s+\w+",  # Role hijacking
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+if",
            r"system\s*:\s*",  # Fake system messages
            r"\[INST\]|\[/INST\]",  # Model-specific tokens
            r"<\|im_start\|>|<\|im_end\|>",  # Chat ML tokens
        ]
        
        # Compile patterns for efficiency
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.blocked_patterns
        ]
    
    def validate_message(self, message: str) -> ValidationResult:
        """
        Validate a user message.
        
        Args:
            message: The user's input message
            
        Returns:
            ValidationResult with sanitized message or violations
        """
        violations = []
        
        # Check for empty or whitespace-only messages
        if not message or not message.strip():
            return ValidationResult(
                is_valid=False,
                violations=["Message cannot be empty"]
            )
        
        # Check message length
        if len(message) > self.max_message_length:
            violations.append(
                f"Message exceeds maximum length of {self.max_message_length} characters"
            )
        
        # Check for blocked patterns (potential prompt injection)
        for pattern in self._compiled_patterns:
            if pattern.search(message):
                violations.append(
                    f"Message contains potentially unsafe pattern: {pattern.pattern}"
                )
        
        if violations:
            return ValidationResult(is_valid=False, violations=violations)
        
        # Sanitize the message
        sanitized = self._sanitize_message(message)
        
        return ValidationResult(is_valid=True, sanitized_value=sanitized)
    
    def _sanitize_message(self, message: str) -> str:
        """Apply sanitization to a message."""
        # Normalize whitespace
        sanitized = " ".join(message.split())
        
        # Remove null bytes and other control characters (except newlines/tabs)
        sanitized = "".join(
            char for char in sanitized 
            if char.isprintable() or char in "\n\t"
        )
        
        return sanitized.strip()
    
    def validate_file_content(
        self, 
        content: bytes, 
        filename: str
    ) -> ValidationResult:
        """
        Validate file content before processing.
        
        Args:
            content: Raw file bytes
            filename: Name of the file
            
        Returns:
            ValidationResult with decoded content or violations
        """
        violations = []
        
        # Check file size
        if len(content) > self.max_file_size_bytes:
            size_mb = len(content) / (1024 * 1024)
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            violations.append(
                f"File size ({size_mb:.1f}MB) exceeds maximum ({max_mb:.1f}MB)"
            )
            return ValidationResult(is_valid=False, violations=violations)
        
        # Check for potentially dangerous file extensions
        dangerous_extensions = {
            ".exe", ".dll", ".bat", ".cmd", ".ps1", ".sh", 
            ".vbs", ".js", ".jar", ".msi"
        }
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext in dangerous_extensions:
            violations.append(
                f"File type '{ext}' is not allowed for security reasons"
            )
            return ValidationResult(is_valid=False, violations=violations)
        
        # Try to decode as text
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            # Binary file - just return the bytes
            return ValidationResult(is_valid=True, sanitized_value=content)
        
        # For text files, apply message validation
        text_result = self.validate_message(decoded)
        if not text_result.is_valid:
            return text_result
        
        return ValidationResult(is_valid=True, sanitized_value=decoded)
    
    def validate_tool_args(
        self, 
        tool_name: str, 
        args: dict[str, Any],
        schema: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate tool arguments.
        
        Args:
            tool_name: Name of the tool being called
            args: Arguments passed to the tool
            schema: Optional JSON schema for validation
            
        Returns:
            ValidationResult with sanitized args or violations
        """
        violations = []
        sanitized_args = {}
        
        for key, value in args.items():
            # Validate string arguments
            if isinstance(value, str):
                # Check for command injection in string args
                if self._contains_shell_injection(value):
                    violations.append(
                        f"Argument '{key}' contains potential shell injection"
                    )
                    continue
                    
                # Check for path traversal
                if self._contains_path_traversal(value):
                    violations.append(
                        f"Argument '{key}' contains path traversal attempt"
                    )
                    continue
                
                sanitized_args[key] = value
            else:
                sanitized_args[key] = value
        
        if violations:
            return ValidationResult(is_valid=False, violations=violations)
        
        return ValidationResult(is_valid=True, sanitized_value=sanitized_args)
    
    def _contains_shell_injection(self, value: str) -> bool:
        """Check for potential shell injection patterns."""
        dangerous_patterns = [
            r";\s*\w+",           # Command chaining with semicolon
            r"\|\s*\w+",          # Pipe to another command
            r"\$\(",              # Command substitution
            r"`[^`]+`",           # Backtick command substitution
            r"&&\s*\w+",          # AND command chaining
            r"\|\|\s*\w+",        # OR command chaining
            r">\s*/",             # Redirect to root paths
            r"<\s*/",             # Read from root paths
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def _contains_path_traversal(self, value: str) -> bool:
        """Check for path traversal attempts."""
        traversal_patterns = [
            r"\.\./",             # Unix path traversal
            r"\.\.\\",            # Windows path traversal
            r"/etc/",             # Sensitive Unix paths
            r"/proc/",
            r"/sys/",
            r"C:\\Windows",       # Sensitive Windows paths
            r"%SYSTEMROOT%",
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
```

### Using the Input Validator

Here's how to integrate input validation into your agent:

```python
from input_validator import InputValidator, ValidationResult

# Create validator with custom settings
validator = InputValidator(
    max_message_length=5000,
    max_file_size_bytes=5 * 1024 * 1024,  # 5MB
)

# Validate user message before processing
user_input = "Please analyze this code for security issues."
result = validator.validate_message(user_input)

if result.is_valid:
    # Safe to process
    process_message(result.sanitized_value)
else:
    # Handle validation failures
    for violation in result.violations:
        print(f"Validation failed: {violation}")
```

### Detecting Prompt Injection

Prompt injection is one of the most significant risks for AI agents. The validator above includes patterns to detect common injection attempts, but you should expand these based on your use case:

```python
# Add custom blocked patterns for your domain
validator = InputValidator(
    blocked_patterns=[
        # Default patterns plus...
        r"ignore\s+previous",
        r"override\s+safety",
        # Domain-specific patterns
        r"delete\s+all\s+records",
        r"grant\s+admin\s+access",
        r"transfer\s+funds",
    ]
)
```

> **Note:** No pattern-based detection is foolproof. Prompt injection is an active area of research, and attackers constantly develop new techniques. Defense in depth—multiple layers of protection—is essential.

## Output Filtering and Verification

Just as we validate inputs, we must also filter outputs. This prevents your agent from leaking sensitive information or producing harmful content.

### The Output Filter Class

```python
"""
Output filtering and verification for AI agents.

Chapter 32: Guardrails and Safety
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterResult:
    """Result of output filtering."""
    is_safe: bool
    filtered_value: str | None = None
    redactions: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


class OutputFilter:
    """Filters and verifies agent outputs before they reach users or systems."""
    
    def __init__(
        self,
        redact_api_keys: bool = True,
        redact_emails: bool = False,
        redact_phone_numbers: bool = False,
        redact_ssn: bool = True,
        redact_credit_cards: bool = True,
        max_output_length: int = 50000,
        custom_patterns: dict[str, str] | None = None
    ):
        self.redact_api_keys = redact_api_keys
        self.redact_emails = redact_emails
        self.redact_phone_numbers = redact_phone_numbers
        self.redact_ssn = redact_ssn
        self.redact_credit_cards = redact_credit_cards
        self.max_output_length = max_output_length
        
        # Build redaction patterns
        self._redaction_patterns = self._build_redaction_patterns()
        
        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self._redaction_patterns[name] = re.compile(pattern)
    
    def _build_redaction_patterns(self) -> dict[str, re.Pattern]:
        """Build regex patterns for sensitive data detection."""
        patterns = {}
        
        if self.redact_api_keys:
            patterns["api_key"] = re.compile(
                r"""
                # Common API key patterns
                (?:api[_-]?key|apikey|api_secret|secret[_-]?key)
                \s*[:=]\s*
                ['\"]?([a-zA-Z0-9_\-]{20,100})['\"]?
                |
                # Bearer tokens
                Bearer\s+[a-zA-Z0-9_\-\.]+
                |
                # Common key prefixes
                (?:sk|pk|api|key)[_-][a-zA-Z0-9]{20,50}
                |
                # AWS-style keys
                (?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}
                |
                # Anthropic API keys
                sk-ant-[a-zA-Z0-9\-]{20,100}
                """,
                re.VERBOSE | re.IGNORECASE
            )
        
        if self.redact_ssn:
            patterns["ssn"] = re.compile(
                r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
            )
        
        if self.redact_credit_cards:
            patterns["credit_card"] = re.compile(
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
            )
        
        if self.redact_emails:
            patterns["email"] = re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            )
        
        if self.redact_phone_numbers:
            patterns["phone"] = re.compile(
                r"""
                \b
                (?:\+?1[-.\s]?)?        # Optional country code
                (?:\(?\d{3}\)?[-.\s]?)  # Area code
                \d{3}[-.\s]?            # Exchange
                \d{4}                   # Subscriber
                \b
                """,
                re.VERBOSE
            )
        
        return patterns
    
    def filter_output(self, output: str) -> FilterResult:
        """
        Filter an agent's output for sensitive or harmful content.
        
        Args:
            output: The agent's response text
            
        Returns:
            FilterResult with filtered text and any redactions made
        """
        concerns = []
        redactions = []
        
        # Check output length
        if len(output) > self.max_output_length:
            concerns.append(
                f"Output exceeds maximum length ({len(output)} > {self.max_output_length})"
            )
            output = output[:self.max_output_length] + "\n[OUTPUT TRUNCATED]"
        
        # Apply redactions
        filtered = output
        for name, pattern in self._redaction_patterns.items():
            matches = pattern.findall(filtered)
            if matches:
                redactions.append(f"Redacted {len(matches)} {name} pattern(s)")
                filtered = pattern.sub(f"[REDACTED_{name.upper()}]", filtered)
        
        # Check for potentially harmful content patterns
        harmful_patterns = [
            (r"rm\s+-rf\s+/", "Destructive shell command detected"),
            (r"DROP\s+TABLE", "Destructive SQL detected"),
            (r"DELETE\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*1", "Mass deletion SQL detected"),
            (r"FORMAT\s+[A-Z]:", "Disk format command detected"),
        ]
        
        for pattern, message in harmful_patterns:
            if re.search(pattern, filtered, re.IGNORECASE):
                concerns.append(message)
        
        is_safe = len(concerns) == 0
        
        return FilterResult(
            is_safe=is_safe,
            filtered_value=filtered,
            redactions=redactions,
            concerns=concerns
        )
    
    def verify_no_secrets(self, output: str, known_secrets: list[str]) -> FilterResult:
        """
        Verify that output doesn't contain any known secrets.
        
        Args:
            output: The output to check
            known_secrets: List of secrets that should never appear
            
        Returns:
            FilterResult indicating if secrets were found
        """
        concerns = []
        filtered = output
        
        for secret in known_secrets:
            if secret and secret in output:
                concerns.append("Output contains a known secret value")
                # Redact the secret
                filtered = filtered.replace(secret, "[REDACTED_SECRET]")
        
        return FilterResult(
            is_safe=len(concerns) == 0,
            filtered_value=filtered,
            concerns=concerns
        )
    
    def verify_json_structure(
        self, 
        output: str, 
        required_fields: list[str] | None = None,
        forbidden_fields: list[str] | None = None
    ) -> FilterResult:
        """
        Verify JSON output has expected structure.
        
        Args:
            output: JSON string to verify
            required_fields: Fields that must be present
            forbidden_fields: Fields that must not be present
            
        Returns:
            FilterResult with verification status
        """
        import json
        
        concerns = []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return FilterResult(
                is_safe=False,
                concerns=[f"Invalid JSON: {e}"]
            )
        
        if required_fields:
            missing = [f for f in required_fields if f not in data]
            if missing:
                concerns.append(f"Missing required fields: {missing}")
        
        if forbidden_fields:
            present = [f for f in forbidden_fields if f in data]
            if present:
                concerns.append(f"Forbidden fields present: {present}")
        
        return FilterResult(
            is_safe=len(concerns) == 0,
            filtered_value=output,
            concerns=concerns
        )
```

### Using the Output Filter

```python
from output_filter import OutputFilter

# Create filter
output_filter = OutputFilter(
    redact_api_keys=True,
    redact_ssn=True,
    redact_credit_cards=True,
)

# Filter agent output before returning to user
agent_response = get_agent_response(user_input)
result = output_filter.filter_output(agent_response)

if result.redactions:
    print(f"Applied redactions: {result.redactions}")

if not result.is_safe:
    print(f"Safety concerns: {result.concerns}")
    # Log the incident and potentially block the response
else:
    # Return the filtered response to user
    return result.filtered_value
```

## Action Constraints and Allowlists

While input validation and output filtering handle data, action constraints control what your agent can actually *do*. This is especially critical for agents with tool access.

### The Action Constraint System

```python
"""
Action constraints and allowlists for AI agents.

Chapter 32: Guardrails and Safety
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class ActionDecision(Enum):
    """Possible decisions for an action."""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""
    decision: ActionDecision
    reason: str
    requires_approval_from: str | None = None


class ActionConstraints:
    """
    Manages constraints on what actions an agent can take.
    
    Supports:
    - Tool allowlists and blocklists
    - Argument constraints
    - Rate limiting per action
    - Conditional constraints
    """
    
    def __init__(self):
        # Tool-level constraints
        self._allowed_tools: set[str] | None = None  # None = all allowed
        self._blocked_tools: set[str] = set()
        
        # Argument constraints by tool
        self._arg_constraints: dict[str, list[Callable]] = {}
        
        # Actions requiring approval
        self._approval_required: dict[str, str] = {}  # tool -> approver
        
        # Custom constraint functions
        self._custom_constraints: list[Callable] = []
    
    def allow_only_tools(self, tools: list[str]) -> "ActionConstraints":
        """Only allow specific tools (allowlist)."""
        self._allowed_tools = set(tools)
        return self
    
    def block_tools(self, tools: list[str]) -> "ActionConstraints":
        """Block specific tools (blocklist)."""
        self._blocked_tools.update(tools)
        return self
    
    def require_approval(self, tool: str, approver: str = "admin") -> "ActionConstraints":
        """Require human approval for a tool."""
        self._approval_required[tool] = approver
        return self
    
    def add_arg_constraint(
        self, 
        tool: str, 
        constraint: Callable[[dict[str, Any]], tuple[bool, str]]
    ) -> "ActionConstraints":
        """
        Add a constraint on tool arguments.
        
        The constraint function takes args dict and returns (is_valid, reason).
        """
        if tool not in self._arg_constraints:
            self._arg_constraints[tool] = []
        self._arg_constraints[tool].append(constraint)
        return self
    
    def add_custom_constraint(
        self, 
        constraint: Callable[[str, dict[str, Any]], ConstraintResult | None]
    ) -> "ActionConstraints":
        """
        Add a custom constraint function.
        
        The function receives (tool_name, args) and returns a ConstraintResult
        or None to defer to other constraints.
        """
        self._custom_constraints.append(constraint)
        return self
    
    def evaluate(self, tool_name: str, args: dict[str, Any]) -> ConstraintResult:
        """
        Evaluate whether an action should be allowed.
        
        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool
            
        Returns:
            ConstraintResult with the decision and reason
        """
        # Check custom constraints first
        for constraint in self._custom_constraints:
            result = constraint(tool_name, args)
            if result is not None:
                return result
        
        # Check blocklist
        if tool_name in self._blocked_tools:
            return ConstraintResult(
                decision=ActionDecision.DENY,
                reason=f"Tool '{tool_name}' is blocked"
            )
        
        # Check allowlist
        if self._allowed_tools is not None:
            if tool_name not in self._allowed_tools:
                return ConstraintResult(
                    decision=ActionDecision.DENY,
                    reason=f"Tool '{tool_name}' is not in the allowlist"
                )
        
        # Check argument constraints
        if tool_name in self._arg_constraints:
            for constraint in self._arg_constraints[tool_name]:
                is_valid, reason = constraint(args)
                if not is_valid:
                    return ConstraintResult(
                        decision=ActionDecision.DENY,
                        reason=reason
                    )
        
        # Check if approval is required
        if tool_name in self._approval_required:
            return ConstraintResult(
                decision=ActionDecision.REQUIRE_APPROVAL,
                reason=f"Tool '{tool_name}' requires approval",
                requires_approval_from=self._approval_required[tool_name]
            )
        
        return ConstraintResult(
            decision=ActionDecision.ALLOW,
            reason="Action permitted"
        )


# Pre-built constraint functions for common use cases

def file_path_constraint(
    allowed_directories: list[str],
    blocked_directories: list[str] | None = None
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """Create a constraint that limits file operations to specific directories."""
    blocked = blocked_directories or []
    
    def constraint(args: dict[str, Any]) -> tuple[bool, str]:
        # Look for path-like arguments
        for key in ["path", "file_path", "filepath", "filename", "directory"]:
            if key in args:
                path = str(args[key])
                
                # Check blocked directories
                for blocked_dir in blocked:
                    if path.startswith(blocked_dir):
                        return False, f"Access to '{blocked_dir}' is blocked"
                
                # Check if path is in allowed directories
                in_allowed = any(
                    path.startswith(allowed_dir) 
                    for allowed_dir in allowed_directories
                )
                if not in_allowed:
                    return False, f"Path must be in: {allowed_directories}"
        
        return True, "Path is allowed"
    
    return constraint


def command_constraint(
    allowed_commands: list[str] | None = None,
    blocked_commands: list[str] | None = None
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """Create a constraint that limits which shell commands can be run."""
    allowed = set(allowed_commands) if allowed_commands else None
    blocked = set(blocked_commands) if blocked_commands else set()
    
    # Always block these dangerous commands
    dangerous = {"rm -rf", "mkfs", "dd", "format", "> /dev"}
    blocked.update(dangerous)
    
    def constraint(args: dict[str, Any]) -> tuple[bool, str]:
        command = args.get("command", "")
        
        # Check blocked patterns
        for blocked_cmd in blocked:
            if blocked_cmd in command:
                return False, f"Command contains blocked pattern: '{blocked_cmd}'"
        
        # Check allowlist
        if allowed is not None:
            # Extract the base command
            base_cmd = command.split()[0] if command.split() else ""
            if base_cmd not in allowed:
                return False, f"Command '{base_cmd}' is not allowed"
        
        return True, "Command is allowed"
    
    return constraint


def rate_limit_constraint(
    max_calls: int,
    window_seconds: int = 60
) -> Callable[[str, dict[str, Any]], ConstraintResult | None]:
    """Create a rate-limiting constraint."""
    from collections import defaultdict
    import time
    
    # Track calls per tool
    call_history: dict[str, list[float]] = defaultdict(list)
    
    def constraint(tool_name: str, args: dict[str, Any]) -> ConstraintResult | None:
        now = time.time()
        history = call_history[tool_name]
        
        # Remove old entries
        history[:] = [t for t in history if now - t < window_seconds]
        
        if len(history) >= max_calls:
            return ConstraintResult(
                decision=ActionDecision.DENY,
                reason=f"Rate limit exceeded: {max_calls} calls per {window_seconds}s"
            )
        
        # Record this call
        history.append(now)
        return None  # Defer to other constraints
    
    return constraint
```

### Setting Up Action Constraints

Here's how to configure constraints for a typical agent:

```python
from action_constraints import (
    ActionConstraints,
    ActionDecision,
    file_path_constraint,
    command_constraint,
    rate_limit_constraint,
)

# Create constraint system
constraints = ActionConstraints()

# Only allow specific tools
constraints.allow_only_tools([
    "read_file",
    "write_file",
    "search_web",
    "calculate",
])

# Add file path constraints
constraints.add_arg_constraint(
    "read_file",
    file_path_constraint(
        allowed_directories=["/home/user/documents", "/tmp"],
        blocked_directories=["/etc", "/root", "/home/user/.ssh"]
    )
)

constraints.add_arg_constraint(
    "write_file",
    file_path_constraint(
        allowed_directories=["/home/user/documents/output"],
    )
)

# Require approval for sensitive operations
constraints.require_approval("delete_file", approver="admin")
constraints.require_approval("send_email", approver="user")

# Add rate limiting
constraints.add_custom_constraint(
    rate_limit_constraint(max_calls=10, window_seconds=60)
)

# Use in your agent loop
def execute_tool(tool_name: str, args: dict) -> str:
    result = constraints.evaluate(tool_name, args)
    
    if result.decision == ActionDecision.DENY:
        raise PermissionError(f"Action denied: {result.reason}")
    
    if result.decision == ActionDecision.REQUIRE_APPROVAL:
        approved = request_approval(
            tool_name, 
            args, 
            approver=result.requires_approval_from
        )
        if not approved:
            raise PermissionError("Action not approved by user")
    
    # Execute the tool
    return tools[tool_name].execute(**args)
```

## Rate Limiting and Resource Bounds

Rate limiting protects against runaway loops and unexpected costs. We've seen a basic rate limiter above; let's build a more comprehensive resource management system.

### The Resource Manager

```python
"""
Rate limiting and resource management for AI agents.

Chapter 32: Guardrails and Safety
"""

import time
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
from contextlib import contextmanager
import threading


@dataclass
class ResourceUsage:
    """Tracks resource usage."""
    api_calls: int = 0
    tokens_used: int = 0
    tool_calls: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def duration_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def calls_per_minute(self) -> float:
        if self.duration_seconds < 1:
            return 0
        return (self.api_calls / self.duration_seconds) * 60


@dataclass
class ResourceLimits:
    """Defines resource limits for an agent."""
    max_api_calls: int = 100
    max_tokens: int = 100000
    max_tool_calls: int = 50
    max_errors: int = 5
    max_duration_seconds: int = 300  # 5 minutes
    max_cost_dollars: float = 1.0
    
    # Rate limits
    api_calls_per_minute: int = 20
    tool_calls_per_minute: int = 10


class ResourceLimitExceeded(Exception):
    """Raised when a resource limit is exceeded."""
    pass


class ResourceManager:
    """
    Manages resource usage and enforces limits.
    
    Thread-safe for use with parallel workflows.
    """
    
    # Approximate costs per token (Claude 3.5 Sonnet)
    INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
    OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens
    
    def __init__(self, limits: ResourceLimits | None = None):
        self.limits = limits or ResourceLimits()
        self.usage = ResourceUsage()
        self._lock = threading.Lock()
        
        # Track calls per minute for rate limiting
        self._api_call_times: list[float] = []
        self._tool_call_times: list[float] = []
    
    def check_limits(self) -> None:
        """Check if any limits have been exceeded."""
        with self._lock:
            if self.usage.api_calls >= self.limits.max_api_calls:
                raise ResourceLimitExceeded(
                    f"API call limit exceeded: {self.usage.api_calls}/{self.limits.max_api_calls}"
                )
            
            if self.usage.tokens_used >= self.limits.max_tokens:
                raise ResourceLimitExceeded(
                    f"Token limit exceeded: {self.usage.tokens_used}/{self.limits.max_tokens}"
                )
            
            if self.usage.tool_calls >= self.limits.max_tool_calls:
                raise ResourceLimitExceeded(
                    f"Tool call limit exceeded: {self.usage.tool_calls}/{self.limits.max_tool_calls}"
                )
            
            if self.usage.errors >= self.limits.max_errors:
                raise ResourceLimitExceeded(
                    f"Error limit exceeded: {self.usage.errors}/{self.limits.max_errors}"
                )
            
            if self.usage.duration_seconds >= self.limits.max_duration_seconds:
                raise ResourceLimitExceeded(
                    f"Duration limit exceeded: {self.usage.duration_seconds:.0f}s"
                )
            
            estimated_cost = self._estimate_cost()
            if estimated_cost >= self.limits.max_cost_dollars:
                raise ResourceLimitExceeded(
                    f"Cost limit exceeded: ${estimated_cost:.2f}/${self.limits.max_cost_dollars:.2f}"
                )
    
    def check_rate_limits(self, call_type: str = "api") -> None:
        """Check rate limits and wait if necessary."""
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
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                self._api_call_times.append(time.time())
            
            elif call_type == "tool":
                self._tool_call_times = [
                    t for t in self._tool_call_times 
                    if now - t < window
                ]
                
                if len(self._tool_call_times) >= self.limits.tool_calls_per_minute:
                    oldest = min(self._tool_call_times)
                    wait_time = window - (now - oldest)
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                self._tool_call_times.append(time.time())
    
    def record_api_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record an API call."""
        with self._lock:
            self.usage.api_calls += 1
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
        """Estimate the cost based on token usage."""
        # Simplified: assume 50/50 split between input/output
        tokens = self.usage.tokens_used
        input_tokens = tokens // 2
        output_tokens = tokens - input_tokens
        
        return (
            input_tokens * self.INPUT_COST_PER_TOKEN +
            output_tokens * self.OUTPUT_COST_PER_TOKEN
        )
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of resource usage."""
        with self._lock:
            return {
                "api_calls": f"{self.usage.api_calls}/{self.limits.max_api_calls}",
                "tokens": f"{self.usage.tokens_used}/{self.limits.max_tokens}",
                "tool_calls": f"{self.usage.tool_calls}/{self.limits.max_tool_calls}",
                "errors": f"{self.usage.errors}/{self.limits.max_errors}",
                "duration": f"{self.usage.duration_seconds:.1f}s/{self.limits.max_duration_seconds}s",
                "estimated_cost": f"${self._estimate_cost():.4f}/${self.limits.max_cost_dollars:.2f}",
            }
    
    @contextmanager
    def api_call_context(self):
        """Context manager for API calls with automatic tracking."""
        self.check_limits()
        self.check_rate_limits("api")
        try:
            yield
        except Exception:
            self.record_error()
            raise
    
    @contextmanager
    def tool_call_context(self):
        """Context manager for tool calls with automatic tracking."""
        self.check_limits()
        self.check_rate_limits("tool")
        try:
            yield
            self.record_tool_call()
        except Exception:
            self.record_error()
            raise
```

### Using the Resource Manager

```python
from resource_manager import ResourceManager, ResourceLimits, ResourceLimitExceeded

# Create resource manager with custom limits
limits = ResourceLimits(
    max_api_calls=50,
    max_tokens=50000,
    max_cost_dollars=0.50,
    max_duration_seconds=120,
)
resource_manager = ResourceManager(limits)

# Use in your agent loop
def agent_loop():
    while not task_complete:
        try:
            # Check limits before API call
            with resource_manager.api_call_context():
                response = client.messages.create(...)
                
                # Record token usage
                resource_manager.record_api_call(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )
            
            # Execute any tool calls
            for tool_call in response.tool_calls:
                with resource_manager.tool_call_context():
                    result = execute_tool(tool_call)
        
        except ResourceLimitExceeded as e:
            print(f"Stopping agent: {e}")
            print(f"Usage: {resource_manager.get_summary()}")
            break
```

## Sandboxing Dangerous Operations

Some operations are inherently risky: executing code, running shell commands, or accessing the network. Sandboxing isolates these operations to minimize potential damage.

### The Sandbox System

```python
"""
Sandboxing for dangerous operations.

Chapter 32: Guardrails and Safety
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager
import signal


@dataclass
class SandboxConfig:
    """Configuration for the sandbox."""
    # File system
    temp_dir: str | None = None  # None = auto-create
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_total_size_bytes: int = 100 * 1024 * 1024  # 100MB
    
    # Process execution
    timeout_seconds: int = 30
    max_memory_bytes: int = 512 * 1024 * 1024  # 512MB
    
    # Network (requires system-level config)
    allow_network: bool = False
    
    # Clean up
    auto_cleanup: bool = True


@dataclass
class SandboxResult:
    """Result from a sandboxed operation."""
    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0
    timed_out: bool = False


class Sandbox:
    """
    Provides a sandboxed environment for dangerous operations.
    
    Features:
    - Isolated temporary directory
    - Process timeouts
    - Resource limits
    - Automatic cleanup
    """
    
    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._temp_dir: Path | None = None
        self._files_created: list[Path] = []
        self._total_size: int = 0
    
    @contextmanager
    def create_environment(self):
        """Create a sandboxed environment."""
        # Create isolated temp directory
        if self.config.temp_dir:
            self._temp_dir = Path(self.config.temp_dir)
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        
        try:
            yield self
        finally:
            if self.config.auto_cleanup:
                self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._files_created.clear()
        self._total_size = 0
    
    def write_file(self, filename: str, content: str | bytes) -> Path:
        """
        Write a file in the sandbox.
        
        Args:
            filename: Name of the file (no path components allowed)
            content: File content
            
        Returns:
            Path to the created file
        """
        if not self._temp_dir:
            raise RuntimeError("Sandbox environment not created")
        
        # Security: prevent path traversal
        safe_name = Path(filename).name
        if safe_name != filename:
            raise ValueError("Filename cannot contain path components")
        
        filepath = self._temp_dir / safe_name
        
        # Check file size
        size = len(content) if isinstance(content, bytes) else len(content.encode())
        if size > self.config.max_file_size_bytes:
            raise ValueError(
                f"File size ({size}) exceeds limit ({self.config.max_file_size_bytes})"
            )
        
        # Check total size
        if self._total_size + size > self.config.max_total_size_bytes:
            raise ValueError("Total sandbox size limit exceeded")
        
        # Write file
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(filepath, mode) as f:
            f.write(content)
        
        self._files_created.append(filepath)
        self._total_size += size
        
        return filepath
    
    def read_file(self, filename: str) -> str:
        """Read a file from the sandbox."""
        if not self._temp_dir:
            raise RuntimeError("Sandbox environment not created")
        
        safe_name = Path(filename).name
        filepath = self._temp_dir / safe_name
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found in sandbox: {filename}")
        
        if not filepath.is_relative_to(self._temp_dir):
            raise PermissionError("Access denied: path escapes sandbox")
        
        return filepath.read_text()
    
    def execute_command(
        self, 
        command: list[str],
        input_data: str | None = None
    ) -> SandboxResult:
        """
        Execute a command in the sandbox.
        
        Args:
            command: Command and arguments as list
            input_data: Optional input to send to stdin
            
        Returns:
            SandboxResult with output and status
        """
        if not self._temp_dir:
            raise RuntimeError("Sandbox environment not created")
        
        try:
            result = subprocess.run(
                command,
                cwd=self._temp_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                input=input_data,
                env=self._get_restricted_env(),
            )
            
            return SandboxResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                exit_code=result.returncode,
            )
        
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output="",
                error=f"Command timed out after {self.config.timeout_seconds}s",
                timed_out=True,
            )
        
        except Exception as e:
            return SandboxResult(
                success=False,
                output="",
                error=str(e),
            )
    
    def execute_python(self, code: str) -> SandboxResult:
        """
        Execute Python code in the sandbox.
        
        Args:
            code: Python code to execute
            
        Returns:
            SandboxResult with output and status
        """
        # Write code to temp file
        script_path = self.write_file("script.py", code)
        
        # Execute with resource limits
        return self.execute_command(["python3", str(script_path)])
    
    def _get_restricted_env(self) -> dict[str, str]:
        """Get a restricted environment for subprocess execution."""
        # Start with minimal environment
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self._temp_dir),
            "TMPDIR": str(self._temp_dir),
            "LANG": "C.UTF-8",
        }
        
        # Explicitly remove sensitive variables
        sensitive_vars = [
            "ANTHROPIC_API_KEY",
            "API_KEY", 
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_URL",
            "SECRET_KEY",
        ]
        for var in sensitive_vars:
            env.pop(var, None)
        
        return env
    
    def list_files(self) -> list[str]:
        """List files in the sandbox."""
        if not self._temp_dir:
            return []
        return [f.name for f in self._temp_dir.iterdir() if f.is_file()]


class CodeExecutionSandbox(Sandbox):
    """
    Specialized sandbox for code execution with additional safety measures.
    """
    
    FORBIDDEN_IMPORTS = {
        "os",
        "subprocess", 
        "shutil",
        "sys",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "marshal",
        "ctypes",
    }
    
    def execute_python(self, code: str) -> SandboxResult:
        """Execute Python with additional safety checks."""
        # Check for forbidden imports
        violations = self._check_forbidden_imports(code)
        if violations:
            return SandboxResult(
                success=False,
                output="",
                error=f"Forbidden imports detected: {violations}",
            )
        
        # Add safety wrapper
        safe_code = self._wrap_code(code)
        
        return super().execute_python(safe_code)
    
    def _check_forbidden_imports(self, code: str) -> list[str]:
        """Check for forbidden imports in the code."""
        import re
        violations = []
        
        # Check import statements
        import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            if module in self.FORBIDDEN_IMPORTS:
                violations.append(module)
        
        # Check for dynamic imports
        if "importlib" in code or "__import__" in code:
            violations.append("dynamic imports")
        
        # Check for eval/exec
        if "eval(" in code or "exec(" in code:
            violations.append("eval/exec")
        
        return violations
    
    def _wrap_code(self, code: str) -> str:
        """Wrap code with safety measures."""
        return f'''
import signal
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))  # 10 seconds CPU
resource.setrlimit(resource.RLIMIT_NOFILE, (50, 50))  # 50 open files

# Set alarm for timeout
signal.alarm(30)

# User code follows
{code}
'''
```

### Using the Sandbox

```python
from sandbox import CodeExecutionSandbox, SandboxConfig

# Configure sandbox
config = SandboxConfig(
    timeout_seconds=10,
    max_file_size_bytes=1024 * 1024,  # 1MB
    auto_cleanup=True,
)

sandbox = CodeExecutionSandbox(config)

# Execute user code safely
with sandbox.create_environment():
    # Write data files if needed
    sandbox.write_file("input.txt", "Hello, World!")
    
    # Execute code
    code = """
with open('input.txt') as f:
    content = f.read()
print(f"Read: {content}")
print("Result: " + content.upper())
"""
    
    result = sandbox.execute_python(code)
    
    if result.success:
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")
        if result.timed_out:
            print("Execution timed out")
```

## The Complete Guardrails Module

Now let's bring everything together into a unified guardrails module:

```python
"""
Complete guardrails module for AI agents.

Chapter 32: Guardrails and Safety
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from input_validator import InputValidator, ValidationResult
from output_filter import OutputFilter, FilterResult
from action_constraints import ActionConstraints, ActionDecision, ConstraintResult
from resource_manager import ResourceManager, ResourceLimits, ResourceLimitExceeded
from sandbox import Sandbox, CodeExecutionSandbox, SandboxConfig, SandboxResult


@dataclass
class GuardrailsConfig:
    """Configuration for all guardrails."""
    # Input validation
    max_message_length: int = 10000
    max_file_size_bytes: int = 10 * 1024 * 1024
    blocked_input_patterns: list[str] = field(default_factory=list)
    
    # Output filtering
    redact_api_keys: bool = True
    redact_pii: bool = True
    max_output_length: int = 50000
    
    # Action constraints
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] = field(default_factory=list)
    tools_requiring_approval: list[str] = field(default_factory=list)
    
    # Resource limits
    max_api_calls: int = 100
    max_tokens: int = 100000
    max_tool_calls: int = 50
    max_duration_seconds: int = 300
    max_cost_dollars: float = 1.0
    
    # Sandbox
    sandbox_code_execution: bool = True
    sandbox_timeout_seconds: int = 30


class Guardrails:
    """
    Unified guardrails system for AI agents.
    
    Provides:
    - Input validation and sanitization
    - Output filtering and verification
    - Action constraints and allowlists
    - Resource limits and rate limiting
    - Sandboxed execution for dangerous operations
    """
    
    def __init__(self, config: GuardrailsConfig | None = None):
        self.config = config or GuardrailsConfig()
        
        # Initialize components
        self.input_validator = InputValidator(
            max_message_length=self.config.max_message_length,
            max_file_size_bytes=self.config.max_file_size_bytes,
            blocked_patterns=self.config.blocked_input_patterns or None,
        )
        
        self.output_filter = OutputFilter(
            redact_api_keys=self.config.redact_api_keys,
            redact_ssn=self.config.redact_pii,
            redact_credit_cards=self.config.redact_pii,
            max_output_length=self.config.max_output_length,
        )
        
        self.action_constraints = ActionConstraints()
        if self.config.allowed_tools:
            self.action_constraints.allow_only_tools(self.config.allowed_tools)
        if self.config.blocked_tools:
            self.action_constraints.block_tools(self.config.blocked_tools)
        for tool in self.config.tools_requiring_approval:
            self.action_constraints.require_approval(tool)
        
        self.resource_manager = ResourceManager(
            ResourceLimits(
                max_api_calls=self.config.max_api_calls,
                max_tokens=self.config.max_tokens,
                max_tool_calls=self.config.max_tool_calls,
                max_duration_seconds=self.config.max_duration_seconds,
                max_cost_dollars=self.config.max_cost_dollars,
            )
        )
        
        self._sandbox: Sandbox | None = None
    
    # Input validation methods
    
    def validate_input(self, message: str) -> ValidationResult:
        """Validate user input."""
        return self.input_validator.validate_message(message)
    
    def validate_file(self, content: bytes, filename: str) -> ValidationResult:
        """Validate file content."""
        return self.input_validator.validate_file_content(content, filename)
    
    def validate_tool_args(
        self, 
        tool_name: str, 
        args: dict[str, Any]
    ) -> ValidationResult:
        """Validate tool arguments."""
        return self.input_validator.validate_tool_args(tool_name, args)
    
    # Output filtering methods
    
    def filter_output(self, output: str) -> FilterResult:
        """Filter agent output."""
        return self.output_filter.filter_output(output)
    
    def verify_no_secrets(
        self, 
        output: str, 
        secrets: list[str]
    ) -> FilterResult:
        """Verify output doesn't contain secrets."""
        return self.output_filter.verify_no_secrets(output, secrets)
    
    # Action constraint methods
    
    def check_action(
        self, 
        tool_name: str, 
        args: dict[str, Any]
    ) -> ConstraintResult:
        """Check if an action is allowed."""
        return self.action_constraints.evaluate(tool_name, args)
    
    def add_action_constraint(
        self,
        tool_name: str,
        constraint: Callable[[dict[str, Any]], tuple[bool, str]]
    ) -> None:
        """Add a custom action constraint."""
        self.action_constraints.add_arg_constraint(tool_name, constraint)
    
    # Resource management methods
    
    def check_resources(self) -> None:
        """Check if resource limits are exceeded."""
        self.resource_manager.check_limits()
    
    def record_api_call(
        self, 
        input_tokens: int = 0, 
        output_tokens: int = 0
    ) -> None:
        """Record an API call."""
        self.resource_manager.record_api_call(input_tokens, output_tokens)
    
    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.resource_manager.record_tool_call()
    
    def get_usage_summary(self) -> dict[str, Any]:
        """Get resource usage summary."""
        return self.resource_manager.get_summary()
    
    # Sandbox methods
    
    def get_sandbox(self) -> Sandbox:
        """Get or create a sandbox for dangerous operations."""
        if self._sandbox is None:
            config = SandboxConfig(
                timeout_seconds=self.config.sandbox_timeout_seconds,
            )
            if self.config.sandbox_code_execution:
                self._sandbox = CodeExecutionSandbox(config)
            else:
                self._sandbox = Sandbox(config)
        return self._sandbox
    
    def execute_code_safely(self, code: str) -> SandboxResult:
        """Execute code in a sandbox."""
        sandbox = self.get_sandbox()
        with sandbox.create_environment():
            return sandbox.execute_python(code)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._sandbox:
            self._sandbox.cleanup()
            self._sandbox = None


# Convenience function to create a fully configured guardrails instance
def create_default_guardrails(
    allowed_tools: list[str] | None = None,
    max_cost: float = 1.0,
) -> Guardrails:
    """Create a guardrails instance with sensible defaults."""
    config = GuardrailsConfig(
        allowed_tools=allowed_tools,
        max_cost_dollars=max_cost,
        blocked_tools=["execute_shell", "delete_all", "format_disk"],
        tools_requiring_approval=["send_email", "post_to_social", "make_purchase"],
    )
    return Guardrails(config)
```

### Integration Example

Here's how to integrate the guardrails module into an agent:

```python
"""
Example: Integrating guardrails into an agent.

Chapter 32: Guardrails and Safety
"""

import os
from dotenv import load_dotenv
import anthropic
from guardrails import Guardrails, GuardrailsConfig, ActionDecision

load_dotenv()

# Create guardrails
guardrails = Guardrails(GuardrailsConfig(
    allowed_tools=["read_file", "write_file", "search_web", "calculate"],
    max_api_calls=50,
    max_cost_dollars=0.50,
))

client = anthropic.Anthropic()


def run_agent(user_input: str) -> str:
    """Run the agent with full guardrails."""
    
    # Step 1: Validate input
    input_result = guardrails.validate_input(user_input)
    if not input_result.is_valid:
        return f"Invalid input: {input_result.violations}"
    
    sanitized_input = input_result.sanitized_value
    messages = [{"role": "user", "content": sanitized_input}]
    
    # Step 2: Agent loop with resource checking
    while True:
        try:
            # Check resource limits
            guardrails.check_resources()
            
            # Make API call
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages,
                tools=get_tools(),
            )
            
            # Record usage
            guardrails.record_api_call(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            
            # Check if done
            if response.stop_reason == "end_turn":
                final_text = response.content[0].text
                
                # Step 3: Filter output
                output_result = guardrails.filter_output(final_text)
                if output_result.redactions:
                    print(f"Applied redactions: {output_result.redactions}")
                
                return output_result.filtered_value
            
            # Process tool calls
            for block in response.content:
                if block.type == "tool_use":
                    # Step 4: Check action constraints
                    constraint_result = guardrails.check_action(
                        block.name, 
                        block.input
                    )
                    
                    if constraint_result.decision == ActionDecision.DENY:
                        tool_result = f"Action denied: {constraint_result.reason}"
                    
                    elif constraint_result.decision == ActionDecision.REQUIRE_APPROVAL:
                        # In production, this would prompt for approval
                        tool_result = "Action requires approval (not implemented)"
                    
                    else:
                        # Execute the tool
                        guardrails.record_tool_call()
                        tool_result = execute_tool(block.name, block.input)
                    
                    # Add tool result to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(tool_result),
                        }]
                    })
        
        except Exception as e:
            guardrails.resource_manager.record_error()
            return f"Error: {e}"
    
    finally:
        # Print usage summary
        print(f"Usage: {guardrails.get_usage_summary()}")


if __name__ == "__main__":
    result = run_agent("What's the weather in New York?")
    print(result)
```

## Common Pitfalls

### 1. Adding Guardrails as an Afterthought

**The Problem:** Teams build entire agents and only add guardrails when something goes wrong in production.

**The Solution:** Build guardrails into your agent from day one. Start with the `Guardrails` class and configure it before writing any agent logic. It's much easier to loosen restrictions than to add them later.

### 2. Pattern Matching Alone for Security

**The Problem:** Relying solely on regex patterns to detect prompt injection or malicious input. Attackers can easily craft inputs that evade pattern matching.

**The Solution:** Use pattern matching as one layer among many. Combine it with:
- Output filtering (defense in depth)
- Action constraints (limit what can happen even if injection succeeds)
- Human-in-the-loop for sensitive operations
- Regular security audits and red-teaming

### 3. Forgetting Rate Limits in Development

**The Problem:** Disabling rate limits during development because they slow down testing, then forgetting to re-enable them in production.

**The Solution:** Always run with rate limits enabled, just use higher limits in development:

```python
if os.getenv("ENVIRONMENT") == "development":
    limits = ResourceLimits(max_api_calls=1000, max_cost_dollars=10.0)
else:
    limits = ResourceLimits(max_api_calls=100, max_cost_dollars=1.0)
```

## Practical Exercise

**Task:** Build a secure code execution agent that:
1. Accepts Python code from users
2. Validates the code for dangerous patterns
3. Executes it in a sandbox with resource limits
4. Filters the output to redact any accidentally exposed secrets
5. Enforces a maximum of 10 executions per session

**Requirements:**
- Use all five guardrail components (input validation, output filtering, action constraints, rate limiting, sandboxing)
- Implement proper error handling for each component
- Add logging to track what the guardrails catch
- Test with both safe and malicious inputs

**Hints:**
- Start with the `CodeExecutionSandbox` class
- Add custom patterns to catch Python-specific attacks
- Consider what happens if code prints environment variables

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Guardrails are not optional**—every production agent needs input validation, output filtering, action constraints, resource limits, and sandboxing for dangerous operations.

- **Defense in depth** is essential—no single guardrail is foolproof, so layer multiple protections to catch threats that slip through individual defenses.

- **Build guardrails in from the start**—retrofitting security is much harder than building it in from day one.

- **Test your guardrails thoroughly**—create adversarial test cases that try to bypass each protection layer.

- **Monitor and iterate**—log what your guardrails catch and update them as you discover new attack patterns.

## What's Next

In Chapter 33, we'll bring together everything we've built in Part 4—the agentic loop, state management, planning, error handling, human-in-the-loop patterns, and now guardrails—into a complete, production-ready Agent class. This will be the culmination of our work on building true autonomous agents.
