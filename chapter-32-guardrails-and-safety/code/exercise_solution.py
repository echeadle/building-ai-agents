"""
Exercise Solution: Secure Code Execution Agent

Chapter 32: Guardrails and Safety

This exercise solution demonstrates building a secure code execution agent that:
1. Accepts Python code from users
2. Validates the code for dangerous patterns
3. Executes it in a sandbox with resource limits
4. Filters the output to redact any accidentally exposed secrets
5. Enforces a maximum of 10 executions per session

All five guardrail components are used:
- Input validation
- Output filtering
- Action constraints
- Rate limiting
- Sandboxing
"""

import os
import re
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

from input_validator import InputValidator, ValidationResult
from output_filter import OutputFilter, FilterResult
from action_constraints import ActionConstraints, ActionDecision, ConstraintResult
from resource_manager import ResourceManager, ResourceLimits, ResourceLimitExceeded
from sandbox import CodeExecutionSandbox, SandboxConfig, SandboxResult

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Track execution statistics."""
    total_executions: int = 0
    successful_executions: int = 0
    blocked_by_validation: int = 0
    blocked_by_constraints: int = 0
    blocked_by_rate_limit: int = 0
    sandbox_failures: int = 0
    outputs_redacted: int = 0


class SecureCodeExecutor:
    """
    A secure code execution agent with comprehensive guardrails.
    
    This agent accepts Python code from users and executes it safely
    using all five guardrail components.
    """
    
    # Additional dangerous patterns specific to code execution
    CODE_BLOCKED_PATTERNS = [
        r"import\s+os\b",
        r"import\s+subprocess\b",
        r"import\s+shutil\b",
        r"import\s+socket\b",
        r"import\s+requests\b",
        r"from\s+os\s+import",
        r"from\s+subprocess\s+import",
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"open\s*\([^)]*['\"][wa]",  # File write
        r"globals\s*\(\s*\)",
        r"locals\s*\(\s*\)",
        r"getattr\s*\(",
        r"setattr\s*\(",
        r"delattr\s*\(",
        r"__builtins__",
        r"__class__",
        r"__bases__",
        r"__subclasses__",
    ]
    
    def __init__(self, max_executions: int = 10):
        """
        Initialize the secure code executor.
        
        Args:
            max_executions: Maximum number of code executions per session
        """
        self.max_executions = max_executions
        self.stats = ExecutionStats()
        
        # Get known secrets from environment (to redact if leaked)
        self.known_secrets = self._collect_secrets()
        
        # Initialize guardrail components
        self._setup_guardrails()
        
        logger.info(f"SecureCodeExecutor initialized (max {max_executions} executions)")
    
    def _collect_secrets(self) -> list[str]:
        """Collect known secrets from environment to check for leaks."""
        secrets = []
        secret_vars = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "API_KEY",
            "SECRET_KEY",
            "DATABASE_URL",
            "PASSWORD",
        ]
        for var in secret_vars:
            value = os.getenv(var)
            if value and len(value) > 5:
                secrets.append(value)
        return secrets
    
    def _setup_guardrails(self):
        """Set up all guardrail components."""
        
        # 1. Input Validator - Check code for dangerous patterns
        self.input_validator = InputValidator(
            max_message_length=10000,
            blocked_patterns=self.CODE_BLOCKED_PATTERNS,
        )
        
        # 2. Output Filter - Redact secrets from output
        self.output_filter = OutputFilter(
            redact_api_keys=True,
            redact_ssn=True,
            redact_credit_cards=True,
            max_output_length=10000,
            custom_patterns={
                "env_var_leak": r"(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*['\"][^'\"]+['\"]",
            }
        )
        
        # 3. Action Constraints - Limit what can be executed
        self.action_constraints = ActionConstraints()
        self.action_constraints.allow_only_tools(["execute_python"])
        
        # Add code content constraint
        def code_content_constraint(args: dict) -> tuple[bool, str]:
            code = args.get("code", "")
            
            # Check code length
            if len(code) > 5000:
                return False, "Code exceeds maximum length of 5000 characters"
            
            # Check for forbidden imports
            forbidden_modules = {
                "os", "subprocess", "shutil", "socket", "requests",
                "urllib", "http", "ftplib", "smtplib", "pickle",
                "marshal", "ctypes", "multiprocessing", "threading"
            }
            
            import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
            for match in re.finditer(import_pattern, code):
                module = match.group(1) or match.group(2)
                if module in forbidden_modules:
                    return False, f"Import of '{module}' is not allowed"
            
            # Check for dangerous builtins
            dangerous_builtins = ["eval", "exec", "compile", "__import__"]
            for builtin in dangerous_builtins:
                if re.search(rf"\b{builtin}\s*\(", code):
                    return False, f"Use of '{builtin}()' is not allowed"
            
            return True, "Code is allowed"
        
        self.action_constraints.add_arg_constraint("execute_python", code_content_constraint)
        
        # 4. Resource Manager - Limit executions
        self.resource_manager = ResourceManager(
            ResourceLimits(
                max_tool_calls=self.max_executions,
                max_errors=5,
                max_duration_seconds=300,
            )
        )
        
        # 5. Sandbox - Isolate code execution
        self.sandbox = CodeExecutionSandbox(
            SandboxConfig(
                timeout_seconds=10,
                max_file_size_bytes=1024 * 1024,  # 1MB
                max_total_size_bytes=5 * 1024 * 1024,  # 5MB
                auto_cleanup=True,
            )
        )
    
    def execute(self, code: str) -> dict:
        """
        Execute Python code securely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results and metadata
        """
        self.stats.total_executions += 1
        execution_id = self.stats.total_executions
        
        logger.info(f"Execution #{execution_id} started")
        
        result = {
            "execution_id": execution_id,
            "success": False,
            "output": None,
            "error": None,
            "blocked_reason": None,
            "redactions_applied": [],
            "warnings": [],
        }
        
        # Step 1: Check rate limits (max executions)
        try:
            self.resource_manager.check_limits()
        except ResourceLimitExceeded as e:
            self.stats.blocked_by_rate_limit += 1
            result["blocked_reason"] = f"Rate limit: {e}"
            logger.warning(f"Execution #{execution_id} blocked by rate limit: {e}")
            return result
        
        # Step 2: Validate input (check for dangerous patterns)
        validation_result = self.input_validator.validate_message(code)
        if not validation_result.is_valid:
            self.stats.blocked_by_validation += 1
            result["blocked_reason"] = f"Validation failed: {validation_result.violations}"
            logger.warning(f"Execution #{execution_id} blocked by validation: {validation_result.violations}")
            return result
        
        # Step 3: Check action constraints
        constraint_result = self.action_constraints.evaluate(
            "execute_python", 
            {"code": code}
        )
        if constraint_result.decision == ActionDecision.DENY:
            self.stats.blocked_by_constraints += 1
            result["blocked_reason"] = f"Constraint: {constraint_result.reason}"
            logger.warning(f"Execution #{execution_id} blocked by constraint: {constraint_result.reason}")
            return result
        
        # Step 4: Execute in sandbox
        logger.info(f"Execution #{execution_id} running in sandbox...")
        self.resource_manager.record_tool_call()
        
        with self.sandbox.create_environment():
            sandbox_result = self.sandbox.execute_python(code)
        
        if not sandbox_result.success:
            self.stats.sandbox_failures += 1
            result["error"] = sandbox_result.error
            if sandbox_result.timed_out:
                result["error"] = f"Execution timed out after {self.sandbox.config.timeout_seconds}s"
            logger.warning(f"Execution #{execution_id} sandbox failure: {sandbox_result.error}")
            return result
        
        # Step 5: Filter output (redact secrets)
        raw_output = sandbox_result.output
        
        # First check for known secrets
        secrets_result = self.output_filter.verify_no_secrets(raw_output, self.known_secrets)
        if secrets_result.concerns:
            result["warnings"].append("Potential secret leak detected and redacted")
            raw_output = secrets_result.filtered_value
        
        # Then apply general filtering
        filter_result = self.output_filter.filter_output(raw_output)
        
        if filter_result.redactions:
            self.stats.outputs_redacted += 1
            result["redactions_applied"] = filter_result.redactions
            result["warnings"].append("Some sensitive data was redacted from output")
        
        if not filter_result.is_safe:
            result["warnings"].extend(filter_result.concerns)
        
        # Success!
        self.stats.successful_executions += 1
        result["success"] = True
        result["output"] = filter_result.filtered_value
        
        logger.info(f"Execution #{execution_id} completed successfully")
        return result
    
    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_executions": self.stats.total_executions,
            "successful_executions": self.stats.successful_executions,
            "blocked_by_validation": self.stats.blocked_by_validation,
            "blocked_by_constraints": self.stats.blocked_by_constraints,
            "blocked_by_rate_limit": self.stats.blocked_by_rate_limit,
            "sandbox_failures": self.stats.sandbox_failures,
            "outputs_redacted": self.stats.outputs_redacted,
            "remaining_executions": self.max_executions - self.resource_manager.usage.tool_calls,
        }
    
    def reset(self):
        """Reset the executor for a new session."""
        self.stats = ExecutionStats()
        self.resource_manager.reset()
        logger.info("SecureCodeExecutor reset")


def demo():
    """Demonstrate the secure code executor."""
    print("="*60)
    print("🛡️  SECURE CODE EXECUTOR DEMO")
    print("="*60)
    
    executor = SecureCodeExecutor(max_executions=10)
    
    # Test cases
    test_cases = [
        # Safe code examples
        {
            "name": "Simple calculation",
            "code": """
x = 10
y = 20
print(f"Sum: {x + y}")
print(f"Product: {x * y}")
"""
        },
        {
            "name": "List operations",
            "code": """
numbers = [1, 2, 3, 4, 5]
squared = [n**2 for n in numbers]
print(f"Original: {numbers}")
print(f"Squared: {squared}")
print(f"Sum of squares: {sum(squared)}")
"""
        },
        {
            "name": "String manipulation",
            "code": """
text = "Hello, World!"
print(f"Original: {text}")
print(f"Upper: {text.upper()}")
print(f"Reversed: {text[::-1]}")
"""
        },
        
        # Dangerous code examples (should be blocked)
        {
            "name": "Import os (blocked)",
            "code": """
import os
print(os.listdir('/'))
"""
        },
        {
            "name": "Use eval (blocked)",
            "code": """
result = eval("2 + 2")
print(result)
"""
        },
        {
            "name": "Import subprocess (blocked)",
            "code": """
import subprocess
subprocess.run(['ls', '-la'])
"""
        },
        {
            "name": "Access __builtins__ (blocked)",
            "code": """
print(__builtins__)
"""
        },
        
        # Code that might leak secrets (output filtered)
        {
            "name": "Print environment-like string (filtered)",
            "code": """
# This simulates what might happen if code accidentally prints sensitive data
fake_key = "sk-ant-api03-abcdef123456789"
print(f"Found key: {fake_key}")
print("Normal output continues here")
"""
        },
    ]
    
    # Run test cases
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test['name']}")
        print("="*60)
        print(f"Code:\n{test['code']}")
        print("-"*60)
        
        result = executor.execute(test['code'])
        
        if result['success']:
            print(f"✅ SUCCESS")
            print(f"Output:\n{result['output']}")
        else:
            if result['blocked_reason']:
                print(f"🚫 BLOCKED: {result['blocked_reason']}")
            elif result['error']:
                print(f"❌ ERROR: {result['error']}")
        
        if result['warnings']:
            print(f"⚠️  Warnings: {result['warnings']}")
        if result['redactions_applied']:
            print(f"🔒 Redactions: {result['redactions_applied']}")
    
    # Test rate limiting
    print(f"\n{'='*60}")
    print("RATE LIMIT TEST")
    print("="*60)
    
    remaining = executor.get_stats()['remaining_executions']
    print(f"Remaining executions: {remaining}")
    
    # Try to exceed limit
    for i in range(remaining + 2):
        result = executor.execute("print('test')")
        if result['blocked_reason'] and 'Rate limit' in result['blocked_reason']:
            print(f"Execution {i+1}: 🚫 Rate limited!")
            break
        else:
            print(f"Execution {i+1}: ✅")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("📊 FINAL STATISTICS")
    print("="*60)
    stats = executor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Demo complete!")


def interactive():
    """Run an interactive session."""
    print("="*60)
    print("🛡️  SECURE CODE EXECUTOR - INTERACTIVE MODE")
    print("="*60)
    print("Enter Python code to execute (type 'quit' to exit, 'stats' for statistics)")
    print("Multi-line input: end with a blank line")
    print("="*60)
    
    executor = SecureCodeExecutor(max_executions=10)
    
    while True:
        print("\n>>> ", end="")
        lines = []
        
        try:
            while True:
                line = input()
                if line.strip().lower() == 'quit':
                    print("\n👋 Goodbye!")
                    return
                if line.strip().lower() == 'stats':
                    print("\n📊 Statistics:")
                    for k, v in executor.get_stats().items():
                        print(f"  {k}: {v}")
                    break
                if line.strip().lower() == 'reset':
                    executor.reset()
                    print("🔄 Session reset")
                    break
                if line == "" and lines:
                    break
                lines.append(line)
            
            if not lines:
                continue
            
            code = "\n".join(lines)
            result = executor.execute(code)
            
            if result['success']:
                print(f"\n{result['output']}")
            else:
                if result['blocked_reason']:
                    print(f"\n🚫 Blocked: {result['blocked_reason']}")
                elif result['error']:
                    print(f"\n❌ Error: {result['error']}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"⚠️  {warning}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive()
    else:
        demo()
