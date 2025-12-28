"""
Action constraints and allowlists for AI agents.

Chapter 32: Guardrails and Safety

This module provides:
- Tool allowlists and blocklists
- Argument constraints for tools
- Approval requirements for sensitive operations
- Custom constraint functions
"""

from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum
import re


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
    - Actions requiring approval
    - Custom constraint functions
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
        """
        Only allow specific tools (allowlist).
        
        Args:
            tools: List of tool names to allow
            
        Returns:
            Self for method chaining
        """
        self._allowed_tools = set(tools)
        return self
    
    def block_tools(self, tools: list[str]) -> "ActionConstraints":
        """
        Block specific tools (blocklist).
        
        Args:
            tools: List of tool names to block
            
        Returns:
            Self for method chaining
        """
        self._blocked_tools.update(tools)
        return self
    
    def require_approval(
        self, 
        tool: str, 
        approver: str = "admin"
    ) -> "ActionConstraints":
        """
        Require human approval for a tool.
        
        Args:
            tool: Name of the tool requiring approval
            approver: Who must approve (e.g., "admin", "user")
            
        Returns:
            Self for method chaining
        """
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
        
        Args:
            tool: Name of the tool
            constraint: Function that validates arguments
            
        Returns:
            Self for method chaining
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
        
        Args:
            constraint: Custom constraint function
            
        Returns:
            Self for method chaining
        """
        self._custom_constraints.append(constraint)
        return self
    
    def evaluate(
        self, 
        tool_name: str, 
        args: dict[str, Any]
    ) -> ConstraintResult:
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
    """
    Create a constraint that limits file operations to specific directories.
    
    Args:
        allowed_directories: Directories where access is permitted
        blocked_directories: Directories where access is never permitted
        
    Returns:
        Constraint function
    """
    blocked = blocked_directories or []
    
    def constraint(args: dict[str, Any]) -> tuple[bool, str]:
        # Look for path-like arguments
        for key in ["path", "file_path", "filepath", "filename", "directory", "dir"]:
            if key in args:
                path = str(args[key])
                
                # Normalize path
                import os
                path = os.path.normpath(path)
                
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
    blocked_commands: list[str] | None = None,
    blocked_patterns: list[str] | None = None
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Create a constraint that limits which shell commands can be run.
    
    Args:
        allowed_commands: If provided, only these base commands are allowed
        blocked_commands: These commands/patterns are always blocked
        blocked_patterns: Regex patterns to block
        
    Returns:
        Constraint function
    """
    allowed = set(allowed_commands) if allowed_commands else None
    blocked = set(blocked_commands) if blocked_commands else set()
    patterns = blocked_patterns or []
    
    # Always block these dangerous commands
    dangerous = {
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "format",
        "> /dev/sd", "chmod 777 /", ":(){ :|:& };:"
    }
    blocked.update(dangerous)
    
    def constraint(args: dict[str, Any]) -> tuple[bool, str]:
        command = args.get("command", "")
        
        # Check blocked patterns
        for blocked_cmd in blocked:
            if blocked_cmd in command:
                return False, f"Command contains blocked pattern: '{blocked_cmd}'"
        
        # Check regex patterns
        for pattern in patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern"
        
        # Check allowlist
        if allowed is not None:
            # Extract the base command
            base_cmd = command.split()[0] if command.split() else ""
            if base_cmd not in allowed:
                return False, f"Command '{base_cmd}' is not in the allowed list"
        
        return True, "Command is allowed"
    
    return constraint


def rate_limit_constraint(
    max_calls: int,
    window_seconds: int = 60
) -> Callable[[str, dict[str, Any]], ConstraintResult | None]:
    """
    Create a rate-limiting constraint.
    
    Args:
        max_calls: Maximum calls allowed in the time window
        window_seconds: Time window in seconds
        
    Returns:
        Custom constraint function
    """
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


def argument_value_constraint(
    arg_name: str,
    allowed_values: list[Any] | None = None,
    blocked_values: list[Any] | None = None,
    max_length: int | None = None
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Create a constraint on a specific argument's value.
    
    Args:
        arg_name: Name of the argument to constrain
        allowed_values: If provided, only these values are allowed
        blocked_values: These values are always blocked
        max_length: Maximum length for string values
        
    Returns:
        Constraint function
    """
    allowed = set(allowed_values) if allowed_values else None
    blocked = set(blocked_values) if blocked_values else set()
    
    def constraint(args: dict[str, Any]) -> tuple[bool, str]:
        if arg_name not in args:
            return True, f"Argument '{arg_name}' not present"
        
        value = args[arg_name]
        
        # Check blocked values
        if value in blocked:
            return False, f"Value '{value}' is blocked for argument '{arg_name}'"
        
        # Check allowed values
        if allowed is not None and value not in allowed:
            return False, f"Value '{value}' not in allowed list for '{arg_name}'"
        
        # Check length for strings
        if max_length and isinstance(value, str) and len(value) > max_length:
            return False, f"Argument '{arg_name}' exceeds max length {max_length}"
        
        return True, "Argument value is allowed"
    
    return constraint


# Example usage and tests
if __name__ == "__main__":
    # Create constraint system
    constraints = ActionConstraints()
    
    # Set up allowlist
    constraints.allow_only_tools([
        "read_file", 
        "write_file", 
        "search_web", 
        "calculate",
        "send_email"
    ])
    
    # Block dangerous tools
    constraints.block_tools(["execute_shell", "delete_database"])
    
    # Add file path constraints
    constraints.add_arg_constraint(
        "read_file",
        file_path_constraint(
            allowed_directories=["/home/user/documents", "/tmp"],
            blocked_directories=["/etc", "/root", "/home/user/.ssh"]
        )
    )
    
    # Add command constraints
    constraints.add_arg_constraint(
        "execute_command",
        command_constraint(
            allowed_commands=["ls", "cat", "echo", "pwd"],
            blocked_patterns=[r"sudo", r"su\s+"]
        )
    )
    
    # Require approval for email
    constraints.require_approval("send_email", approver="user")
    
    # Add rate limiting
    constraints.add_custom_constraint(
        rate_limit_constraint(max_calls=5, window_seconds=60)
    )
    
    # Test evaluations
    print("Testing action constraints:")
    
    # Test allowed tool
    result = constraints.evaluate("read_file", {"path": "/home/user/documents/file.txt"})
    print(f"\n1. Read allowed file: {result.decision.value}")
    print(f"   Reason: {result.reason}")
    
    # Test blocked directory
    result = constraints.evaluate("read_file", {"path": "/etc/passwd"})
    print(f"\n2. Read blocked directory: {result.decision.value}")
    print(f"   Reason: {result.reason}")
    
    # Test tool not in allowlist
    result = constraints.evaluate("unknown_tool", {})
    print(f"\n3. Unknown tool: {result.decision.value}")
    print(f"   Reason: {result.reason}")
    
    # Test blocked tool
    result = constraints.evaluate("execute_shell", {"command": "ls"})
    print(f"\n4. Blocked tool: {result.decision.value}")
    print(f"   Reason: {result.reason}")
    
    # Test approval required
    result = constraints.evaluate("send_email", {"to": "user@example.com"})
    print(f"\n5. Email (needs approval): {result.decision.value}")
    print(f"   Reason: {result.reason}")
    print(f"   Approver: {result.requires_approval_from}")
    
    # Test rate limiting (make several calls)
    print("\n6. Rate limiting test:")
    for i in range(7):
        result = constraints.evaluate("calculate", {"expression": f"{i}+1"})
        print(f"   Call {i+1}: {result.decision.value}", end="")
        if result.decision == ActionDecision.DENY:
            print(f" - {result.reason}")
        else:
            print()
    
    print("\n✅ Action constraint tests complete!")
