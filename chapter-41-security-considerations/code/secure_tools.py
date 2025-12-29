"""
Principle of least privilege for agent tools.

Chapter 41: Security Considerations
"""

import os
import time
import inspect
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps


class Permission(Enum):
    """Permissions for tool access."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    ADMIN = "admin"


@dataclass
class ToolDefinition:
    """Definition of a secure tool."""
    name: str
    description: str
    func: Callable
    parameters: dict[str, Any]
    required_permissions: set[Permission]
    rate_limit: Optional[int] = None  # Max calls per minute
    allowed_clients: Optional[set[str]] = None  # None = all clients
    
    def to_api_format(self) -> dict[str, Any]:
        """Convert to API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }


class SecureToolRegistry:
    """
    Registry for agent tools with security controls.
    
    Features:
    - Permission-based access control
    - Per-client tool restrictions
    - Rate limiting per tool
    - Audit logging
    
    Usage:
        registry = SecureToolRegistry()
        
        @registry.register(
            name="read_file",
            permissions={Permission.READ, Permission.FILESYSTEM},
            rate_limit=10
        )
        def read_file(path: str) -> str:
            return open(path).read()
        
        # Execute with permission check
        result = registry.execute(
            "read_file",
            {"path": "/tmp/data.txt"},
            client_id="client_123",
            client_permissions={Permission.READ, Permission.FILESYSTEM}
        )
    """
    
    def __init__(self, audit_logger: Optional[Any] = None):
        """
        Initialize the registry.
        
        Args:
            audit_logger: Optional audit logger for security events
        """
        self.tools: dict[str, ToolDefinition] = {}
        self.audit = audit_logger
        self._call_counts: dict[str, list[float]] = {}
    
    def register(
        self,
        name: str,
        description: str = "",
        permissions: Optional[set[Permission]] = None,
        rate_limit: Optional[int] = None,
        allowed_clients: Optional[set[str]] = None
    ) -> Callable:
        """
        Decorator to register a tool.
        
        Args:
            name: Tool name
            description: Tool description
            permissions: Required permissions
            rate_limit: Max calls per minute
            allowed_clients: Allowed client IDs (None = all)
        """
        def decorator(func: Callable) -> Callable:
            # Build parameters schema from function signature
            sig = inspect.signature(func)
            
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                
                parameters["properties"][param_name] = {"type": param_type}
                
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)
            
            # Register the tool
            self.tools[name] = ToolDefinition(
                name=name,
                description=description or func.__doc__ or "",
                func=func,
                parameters=parameters,
                required_permissions=permissions or set(),
                rate_limit=rate_limit,
                allowed_clients=allowed_clients
            )
            
            return func
        
        return decorator
    
    def _check_rate_limit(self, tool_name: str) -> bool:
        """Check if tool call is within rate limit."""
        tool = self.tools.get(tool_name)
        if not tool or not tool.rate_limit:
            return True
        
        now = time.time()
        
        # Clean old calls
        if tool_name in self._call_counts:
            self._call_counts[tool_name] = [
                t for t in self._call_counts[tool_name]
                if t > now - 60
            ]
        else:
            self._call_counts[tool_name] = []
        
        # Check limit
        if len(self._call_counts[tool_name]) >= tool.rate_limit:
            return False
        
        return True
    
    def _record_call(self, tool_name: str) -> None:
        """Record a tool call for rate limiting."""
        if tool_name not in self._call_counts:
            self._call_counts[tool_name] = []
        self._call_counts[tool_name].append(time.time())
    
    def can_execute(
        self,
        tool_name: str,
        client_id: str,
        client_permissions: set[Permission]
    ) -> tuple[bool, str]:
        """
        Check if a client can execute a tool.
        
        Args:
            tool_name: Name of the tool
            client_id: Client identifier
            client_permissions: Permissions the client has
        
        Returns:
            Tuple of (allowed, reason)
        """
        if tool_name not in self.tools:
            return False, f"Tool not found: {tool_name}"
        
        tool = self.tools[tool_name]
        
        # Check client allowlist
        if tool.allowed_clients and client_id not in tool.allowed_clients:
            return False, f"Client not authorized for tool: {tool_name}"
        
        # Check permissions
        missing = tool.required_permissions - client_permissions
        if missing:
            return False, f"Missing permissions: {[p.value for p in missing]}"
        
        # Check rate limit
        if not self._check_rate_limit(tool_name):
            return False, f"Rate limit exceeded for tool: {tool_name}"
        
        return True, "OK"
    
    def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        client_id: str,
        client_permissions: set[Permission]
    ) -> Any:
        """
        Execute a tool with security checks.
        
        Args:
            tool_name: Name of the tool
            tool_input: Input parameters
            client_id: Client identifier
            client_permissions: Permissions the client has
        
        Returns:
            Tool result
        
        Raises:
            PermissionError: If access is denied
        """
        # Check access
        allowed, reason = self.can_execute(tool_name, client_id, client_permissions)
        
        if not allowed:
            if self.audit:
                # Import here to avoid circular dependency
                from audit_logger import SecurityEventType
                self.audit.log(
                    SecurityEventType.ACCESS_DENIED,
                    severity="medium",
                    client_id=client_id,
                    message=f"Tool access denied: {tool_name}",
                    details={"reason": reason}
                )
            raise PermissionError(reason)
        
        # Record call for rate limiting
        self._record_call(tool_name)
        
        # Execute
        tool = self.tools[tool_name]
        
        try:
            result = tool.func(**tool_input)
            
            if self.audit:
                from audit_logger import SecurityEventType
                self.audit.log(
                    SecurityEventType.ACCESS_GRANTED,
                    severity="low",
                    client_id=client_id,
                    message=f"Tool executed: {tool_name}",
                    details={"input_keys": list(tool_input.keys())}
                )
            
            return result
            
        except Exception as e:
            if self.audit:
                from audit_logger import SecurityEventType
                self.audit.log(
                    SecurityEventType.ERROR,
                    severity="medium",
                    client_id=client_id,
                    message=f"Tool execution error: {tool_name}",
                    details={"error": str(e)}
                )
            raise
    
    def get_tools_for_client(
        self,
        client_id: str,
        client_permissions: set[Permission]
    ) -> list[dict[str, Any]]:
        """
        Get available tools for a client based on permissions.
        
        Args:
            client_id: Client identifier
            client_permissions: Client's permissions
        
        Returns:
            List of tool definitions in API format
        """
        available = []
        
        for tool in self.tools.values():
            allowed, _ = self.can_execute(
                tool.name,
                client_id,
                client_permissions
            )
            
            if allowed:
                available.append(tool.to_api_format())
        
        return available


class SandboxedFileAccess:
    """
    Sandboxed file access with path restrictions.
    
    Only allows access to files within specified directories.
    """
    
    def __init__(self, allowed_directories: list[str]):
        """
        Initialize with allowed directories.
        
        Args:
            allowed_directories: List of directory paths that can be accessed
        """
        self.allowed = [os.path.abspath(d) for d in allowed_directories]
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(allowed) for allowed in self.allowed)
    
    def read_file(self, path: str) -> str:
        """
        Read a file within allowed directories.
        
        Args:
            path: File path
        
        Returns:
            File contents
        
        Raises:
            PermissionError: If path is not allowed
        """
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path} is outside allowed directories")
        
        with open(path) as f:
            return f.read()
    
    def write_file(self, path: str, content: str) -> None:
        """
        Write to a file within allowed directories.
        
        Args:
            path: File path
            content: Content to write
        
        Raises:
            PermissionError: If path is not allowed
        """
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path} is outside allowed directories")
        
        with open(path, 'w') as f:
            f.write(content)
    
    def list_directory(self, path: str) -> list[str]:
        """
        List directory contents.
        
        Args:
            path: Directory path
        
        Returns:
            List of filenames
        
        Raises:
            PermissionError: If path is not allowed
        """
        if not self._is_path_allowed(path):
            raise PermissionError(f"Access denied: {path} is outside allowed directories")
        
        return os.listdir(path)


# Example usage
if __name__ == "__main__":
    print("Secure Tool Registry Demo")
    print("=" * 60)
    
    # Create registry
    registry = SecureToolRegistry()
    
    # Create sandboxed file access
    sandbox = SandboxedFileAccess(["/tmp"])
    
    # Register tools with permissions
    @registry.register(
        name="read_file",
        description="Read a file from the allowed directories",
        permissions={Permission.READ, Permission.FILESYSTEM},
        rate_limit=10
    )
    def read_file(path: str) -> str:
        return sandbox.read_file(path)
    
    @registry.register(
        name="write_file",
        description="Write to a file in allowed directories",
        permissions={Permission.WRITE, Permission.FILESYSTEM},
        rate_limit=5
    )
    def write_file(path: str, content: str) -> str:
        sandbox.write_file(path, content)
        return "OK"
    
    @registry.register(
        name="calculator",
        description="Perform a calculation",
        permissions={Permission.EXECUTE},
        rate_limit=100
    )
    def calculator(expression: str) -> str:
        # Safe evaluation (very limited)
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        return str(eval(expression))
    
    # Test access with different permission levels
    print("\nTesting tool access with different permissions...")
    print()
    
    # Client with read-only permissions
    readonly_perms = {Permission.READ, Permission.FILESYSTEM}
    
    # Client with full permissions
    full_perms = {Permission.READ, Permission.WRITE, Permission.FILESYSTEM, Permission.EXECUTE}
    
    # Test cases
    tests = [
        ("read_file", {"path": "/tmp/test.txt"}, "readonly_client", readonly_perms),
        ("write_file", {"path": "/tmp/test.txt", "content": "hello"}, "readonly_client", readonly_perms),
        ("write_file", {"path": "/tmp/test.txt", "content": "hello"}, "admin_client", full_perms),
        ("calculator", {"expression": "2 + 2"}, "readonly_client", readonly_perms),
        ("calculator", {"expression": "2 + 2"}, "admin_client", full_perms),
    ]
    
    for tool_name, tool_input, client_id, perms in tests:
        allowed, reason = registry.can_execute(tool_name, client_id, perms)
        print(f"{tool_name} by {client_id}:")
        print(f"  Allowed: {allowed}")
        if not allowed:
            print(f"  Reason: {reason}")
        print()
    
    # Show available tools for each client type
    print("=" * 60)
    print("Available tools by permission level:")
    print()
    
    print("Read-only client:")
    for tool in registry.get_tools_for_client("readonly", readonly_perms):
        print(f"  - {tool['name']}")
    
    print("\nFull-access client:")
    for tool in registry.get_tools_for_client("admin", full_perms):
        print(f"  - {tool['name']}")