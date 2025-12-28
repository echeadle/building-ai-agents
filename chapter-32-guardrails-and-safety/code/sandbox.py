"""
Sandboxing for dangerous operations.

Chapter 32: Guardrails and Safety

This module provides:
- Isolated temporary directories for file operations
- Process execution with timeouts and resource limits
- Restricted environment variables
- Code execution safety checks
"""

import subprocess
import tempfile
import os
import shutil
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from contextlib import contextmanager


@dataclass
class SandboxConfig:
    """Configuration for the sandbox."""
    # File system
    temp_dir: str | None = None  # None = auto-create
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_total_size_bytes: int = 100 * 1024 * 1024  # 100MB
    
    # Process execution
    timeout_seconds: int = 30
    max_output_size: int = 1024 * 1024  # 1MB
    
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
    files_created: list[str] | None = None


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
        """
        Initialize the sandbox.
        
        Args:
            config: Sandbox configuration (uses defaults if not provided)
        """
        self.config = config or SandboxConfig()
        self._temp_dir: Path | None = None
        self._files_created: list[Path] = []
        self._total_size: int = 0
        self._is_active: bool = False
    
    @property
    def is_active(self) -> bool:
        """Check if the sandbox environment is active."""
        return self._is_active
    
    @property
    def working_directory(self) -> Path | None:
        """Get the sandbox working directory."""
        return self._temp_dir
    
    @contextmanager
    def create_environment(self):
        """
        Create a sandboxed environment.
        
        Usage:
            with sandbox.create_environment():
                sandbox.write_file("test.txt", "content")
                result = sandbox.execute_command(["cat", "test.txt"])
        """
        # Create isolated temp directory
        if self.config.temp_dir:
            self._temp_dir = Path(self.config.temp_dir)
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))
        
        self._is_active = True
        
        try:
            yield self
        finally:
            self._is_active = False
            if self.config.auto_cleanup:
                self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass  # Best effort cleanup
        self._files_created.clear()
        self._total_size = 0
        self._temp_dir = None
    
    def _check_active(self) -> None:
        """Check that sandbox is active."""
        if not self._is_active or not self._temp_dir:
            raise RuntimeError("Sandbox environment not active. Use create_environment() context manager.")
    
    def write_file(self, filename: str, content: str | bytes) -> Path:
        """
        Write a file in the sandbox.
        
        Args:
            filename: Name of the file (no path components allowed)
            content: File content
            
        Returns:
            Path to the created file
            
        Raises:
            ValueError: If filename is invalid or size limits exceeded
            RuntimeError: If sandbox not active
        """
        self._check_active()
        
        # Security: prevent path traversal
        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            raise ValueError("Filename cannot contain path components")
        
        filepath = self._temp_dir / safe_name
        
        # Check file size
        size = len(content) if isinstance(content, bytes) else len(content.encode())
        if size > self.config.max_file_size_bytes:
            raise ValueError(
                f"File size ({size:,} bytes) exceeds limit ({self.config.max_file_size_bytes:,} bytes)"
            )
        
        # Check total size
        if self._total_size + size > self.config.max_total_size_bytes:
            raise ValueError(
                f"Total sandbox size would exceed limit ({self.config.max_total_size_bytes:,} bytes)"
            )
        
        # Write file
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(filepath, mode) as f:
            f.write(content)
        
        self._files_created.append(filepath)
        self._total_size += size
        
        return filepath
    
    def read_file(self, filename: str) -> str:
        """
        Read a file from the sandbox.
        
        Args:
            filename: Name of the file to read
            
        Returns:
            File contents as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If trying to escape sandbox
        """
        self._check_active()
        
        safe_name = Path(filename).name
        filepath = self._temp_dir / safe_name
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found in sandbox: {filename}")
        
        # Security check: ensure path is within sandbox
        try:
            filepath.resolve().relative_to(self._temp_dir.resolve())
        except ValueError:
            raise PermissionError("Access denied: path escapes sandbox")
        
        return filepath.read_text()
    
    def delete_file(self, filename: str) -> None:
        """
        Delete a file from the sandbox.
        
        Args:
            filename: Name of the file to delete
        """
        self._check_active()
        
        safe_name = Path(filename).name
        filepath = self._temp_dir / safe_name
        
        if filepath.exists() and filepath.is_relative_to(self._temp_dir):
            size = filepath.stat().st_size
            filepath.unlink()
            self._total_size -= size
            if filepath in self._files_created:
                self._files_created.remove(filepath)
    
    def list_files(self) -> list[str]:
        """
        List files in the sandbox.
        
        Returns:
            List of filenames in the sandbox
        """
        if not self._temp_dir:
            return []
        return [f.name for f in self._temp_dir.iterdir() if f.is_file()]
    
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
        self._check_active()
        
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
            
            # Truncate output if too large
            stdout = result.stdout[:self.config.max_output_size]
            stderr = result.stderr[:self.config.max_output_size] if result.stderr else None
            
            return SandboxResult(
                success=result.returncode == 0,
                output=stdout,
                error=stderr,
                exit_code=result.returncode,
                files_created=self.list_files(),
            )
        
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output="",
                error=f"Command timed out after {self.config.timeout_seconds}s",
                timed_out=True,
            )
        
        except FileNotFoundError:
            return SandboxResult(
                success=False,
                output="",
                error=f"Command not found: {command[0]}",
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
        script_path = self.write_file("_script.py", code)
        
        # Execute with Python
        return self.execute_command(["python3", str(script_path.name)])
    
    def _get_restricted_env(self) -> dict[str, str]:
        """Get a restricted environment for subprocess execution."""
        # Start with minimal environment
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self._temp_dir),
            "TMPDIR": str(self._temp_dir),
            "TEMP": str(self._temp_dir),
            "TMP": str(self._temp_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        
        # Explicitly exclude sensitive variables
        sensitive_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "DATABASE_URL",
            "SECRET_KEY",
            "PRIVATE_KEY",
            "PASSWORD",
            "TOKEN",
        ]
        for var in sensitive_vars:
            env.pop(var, None)
        
        return env


class CodeExecutionSandbox(Sandbox):
    """
    Specialized sandbox for code execution with additional safety measures.
    
    Adds:
    - Import restrictions
    - Code pattern analysis
    - Resource limit wrappers
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
        "multiprocessing",
        "threading",
        "asyncio",
        "signal",
    }
    
    FORBIDDEN_PATTERNS = [
        (r"__import__\s*\(", "Dynamic import"),
        (r"eval\s*\(", "eval() call"),
        (r"exec\s*\(", "exec() call"),
        (r"compile\s*\(", "compile() call"),
        (r"globals\s*\(\s*\)", "globals() access"),
        (r"locals\s*\(\s*\)", "locals() access"),
        (r"getattr\s*\(", "getattr() call"),
        (r"setattr\s*\(", "setattr() call"),
        (r"delattr\s*\(", "delattr() call"),
        (r"open\s*\([^)]*['\"][wa]", "File write operation"),
        (r"builtins", "builtins access"),
        (r"__class__", "Class access"),
        (r"__bases__", "Base class access"),
        (r"__subclasses__", "Subclass enumeration"),
        (r"__mro__", "MRO access"),
    ]
    
    def analyze_code(self, code: str) -> tuple[bool, list[str]]:
        """
        Analyze code for potentially dangerous patterns.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        violations = []
        
        # Check for forbidden imports
        import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            if module in self.FORBIDDEN_IMPORTS:
                violations.append(f"Forbidden import: {module}")
        
        # Check for forbidden patterns
        for pattern, description in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Forbidden pattern: {description}")
        
        return len(violations) == 0, violations
    
    def execute_python(self, code: str) -> SandboxResult:
        """
        Execute Python with additional safety checks.
        
        Args:
            code: Python code to execute
            
        Returns:
            SandboxResult with output and status
        """
        # Analyze code first
        is_safe, violations = self.analyze_code(code)
        if not is_safe:
            return SandboxResult(
                success=False,
                output="",
                error=f"Code safety violations: {'; '.join(violations)}",
            )
        
        # Wrap code with safety measures
        safe_code = self._wrap_code(code)
        
        return super().execute_python(safe_code)
    
    def _wrap_code(self, code: str) -> str:
        """
        Wrap code with safety measures.
        
        Args:
            code: Original code
            
        Returns:
            Wrapped code with resource limits
        """
        # Note: resource module only works on Unix-like systems
        return f'''
# Safety wrapper
import sys

# Limit recursion
sys.setrecursionlimit(100)

# User code follows
{code}
'''


# Example usage and tests
if __name__ == "__main__":
    print("Testing Sandbox:")
    
    # Basic sandbox test
    print("\n1. Basic sandbox operations:")
    config = SandboxConfig(
        timeout_seconds=5,
        max_file_size_bytes=1024,  # 1KB for testing
    )
    sandbox = Sandbox(config)
    
    with sandbox.create_environment():
        # Write a file
        sandbox.write_file("test.txt", "Hello, Sandbox!")
        print(f"   Created file: test.txt")
        print(f"   Working directory: {sandbox.working_directory}")
        
        # Read it back
        content = sandbox.read_file("test.txt")
        print(f"   Content: {content}")
        
        # List files
        files = sandbox.list_files()
        print(f"   Files: {files}")
        
        # Execute command
        result = sandbox.execute_command(["cat", "test.txt"])
        print(f"   cat output: {result.output.strip()}")
    
    print("   Sandbox cleaned up")
    
    # Test file size limit
    print("\n2. Testing file size limit:")
    with sandbox.create_environment():
        try:
            sandbox.write_file("big.txt", "x" * 2000)  # Over 1KB limit
        except ValueError as e:
            print(f"   Blocked: {e}")
    
    # Test timeout
    print("\n3. Testing timeout:")
    with sandbox.create_environment():
        result = sandbox.execute_command(["sleep", "10"])  # 10s > 5s timeout
        print(f"   Timed out: {result.timed_out}")
        print(f"   Error: {result.error}")
    
    # Test code execution sandbox
    print("\n4. Testing code execution sandbox:")
    code_sandbox = CodeExecutionSandbox(SandboxConfig(timeout_seconds=5))
    
    with code_sandbox.create_environment():
        # Safe code
        result = code_sandbox.execute_python("""
print("Hello from sandbox!")
x = 2 + 2
print(f"2 + 2 = {x}")
""")
        print(f"   Safe code output:\n   {result.output.strip()}")
        
        # Dangerous code - import os
        result = code_sandbox.execute_python("""
import os
print(os.listdir('/'))
""")
        print(f"\n   Dangerous code (import os):")
        print(f"   Success: {result.success}")
        print(f"   Error: {result.error}")
        
        # Dangerous code - eval
        result = code_sandbox.execute_python("""
result = eval("2 + 2")
print(result)
""")
        print(f"\n   Dangerous code (eval):")
        print(f"   Success: {result.success}")
        print(f"   Error: {result.error}")
    
    # Test code analysis
    print("\n5. Testing code analysis:")
    is_safe, violations = code_sandbox.analyze_code("""
import subprocess
result = subprocess.run(['ls'], capture_output=True)
exec("print('hello')")
""")
    print(f"   Is safe: {is_safe}")
    print(f"   Violations: {violations}")
    
    # Test environment isolation
    print("\n6. Testing environment isolation:")
    os.environ["TEST_SECRET"] = "super_secret_value"
    with sandbox.create_environment():
        result = sandbox.execute_command(["env"])
        has_secret = "TEST_SECRET" in result.output
        has_anthropic = "ANTHROPIC" in result.output
        print(f"   TEST_SECRET visible: {has_secret}")
        print(f"   ANTHROPIC vars visible: {has_anthropic}")
    del os.environ["TEST_SECRET"]
    
    print("\n✅ Sandbox tests complete!")
