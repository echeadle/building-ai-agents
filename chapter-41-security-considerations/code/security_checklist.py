"""
Security hardening checklist for AI agents.

Chapter 41: Security Considerations
"""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class CheckStatus(Enum):
    """Status of a security check."""
    PASS = "✅"
    FAIL = "❌"
    WARN = "⚠️"
    SKIP = "⏭️"


@dataclass
class CheckResult:
    """Result of a security check."""
    name: str
    status: CheckStatus
    message: str
    recommendation: Optional[str] = None


class SecurityChecklist:
    """
    Security hardening checklist for AI agents.
    
    Run this against your deployment to identify security gaps.
    
    Usage:
        checklist = SecurityChecklist()
        checklist.check_api_key_security()
        checklist.check_input_validation(has_validator=True)
        checklist.print_report()
    """
    
    def __init__(self):
        self.results: list[CheckResult] = []
    
    def add_result(
        self,
        name: str,
        status: CheckStatus,
        message: str,
        recommendation: Optional[str] = None
    ) -> None:
        """Add a check result."""
        self.results.append(CheckResult(
            name=name,
            status=status,
            message=message,
            recommendation=recommendation
        ))
    
    def check_api_key_security(self) -> None:
        """Check API key handling."""
        # Check if key is in environment
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self.add_result(
                "API Key Present",
                CheckStatus.FAIL,
                "ANTHROPIC_API_KEY not found in environment",
                "Set the API key as an environment variable, never hardcode it"
            )
            return
        
        self.add_result(
            "API Key Present",
            CheckStatus.PASS,
            "API key loaded from environment"
        )
        
        # Check key format
        if key.startswith("sk-ant-"):
            self.add_result(
                "API Key Format",
                CheckStatus.PASS,
                "API key has expected format"
            )
        else:
            self.add_result(
                "API Key Format",
                CheckStatus.WARN,
                "API key format unexpected",
                "Verify the key is correct"
            )
    
    def check_input_validation(self, validator_exists: bool) -> None:
        """Check if input validation is implemented."""
        if validator_exists:
            self.add_result(
                "Input Validation",
                CheckStatus.PASS,
                "Input validator is configured"
            )
        else:
            self.add_result(
                "Input Validation",
                CheckStatus.FAIL,
                "No input validation configured",
                "Implement InputValidator class to prevent injection attacks"
            )
    
    def check_output_filtering(self, filter_exists: bool) -> None:
        """Check if output filtering is implemented."""
        if filter_exists:
            self.add_result(
                "Output Filtering",
                CheckStatus.PASS,
                "Output security filter is configured"
            )
        else:
            self.add_result(
                "Output Filtering",
                CheckStatus.FAIL,
                "No output filtering configured",
                "Implement OutputSecurityFilter to prevent data leakage"
            )
    
    def check_rate_limiting(self, limiter_exists: bool) -> None:
        """Check if rate limiting is implemented."""
        if limiter_exists:
            self.add_result(
                "Rate Limiting",
                CheckStatus.PASS,
                "Rate limiter is configured"
            )
        else:
            self.add_result(
                "Rate Limiting",
                CheckStatus.WARN,
                "No rate limiting configured",
                "Implement RateLimiter to prevent abuse"
            )
    
    def check_audit_logging(self, logger_exists: bool) -> None:
        """Check if audit logging is implemented."""
        if logger_exists:
            self.add_result(
                "Audit Logging",
                CheckStatus.PASS,
                "Audit logger is configured"
            )
        else:
            self.add_result(
                "Audit Logging",
                CheckStatus.WARN,
                "No audit logging configured",
                "Implement AuditLogger for security monitoring"
            )
    
    def check_tool_permissions(self, registry_exists: bool) -> None:
        """Check if tool permissions are implemented."""
        if registry_exists:
            self.add_result(
                "Tool Permissions",
                CheckStatus.PASS,
                "Secure tool registry is configured"
            )
        else:
            self.add_result(
                "Tool Permissions",
                CheckStatus.WARN,
                "No tool permission system configured",
                "Implement SecureToolRegistry for least privilege"
            )
    
    def check_https(self, using_https: bool) -> None:
        """Check if HTTPS is enforced."""
        if using_https:
            self.add_result(
                "HTTPS",
                CheckStatus.PASS,
                "HTTPS is enforced"
            )
        else:
            self.add_result(
                "HTTPS",
                CheckStatus.FAIL,
                "HTTPS not enforced",
                "Configure TLS/SSL for all API endpoints"
            )
    
    def check_cors(self, cors_restricted: bool) -> None:
        """Check CORS configuration."""
        if cors_restricted:
            self.add_result(
                "CORS Policy",
                CheckStatus.PASS,
                "CORS is properly restricted"
            )
        else:
            self.add_result(
                "CORS Policy",
                CheckStatus.WARN,
                "CORS may be too permissive",
                "Restrict CORS to known origins only"
            )
    
    def check_secrets_management(self, using_secrets_manager: bool) -> None:
        """Check if proper secrets management is in place."""
        if using_secrets_manager:
            self.add_result(
                "Secrets Management",
                CheckStatus.PASS,
                "Using dedicated secrets manager"
            )
        else:
            self.add_result(
                "Secrets Management",
                CheckStatus.WARN,
                "Not using dedicated secrets manager",
                "Consider AWS Secrets Manager, HashiCorp Vault, or similar"
            )
    
    def check_error_handling(self, safe_errors: bool) -> None:
        """Check if error messages are safe (don't leak info)."""
        if safe_errors:
            self.add_result(
                "Error Handling",
                CheckStatus.PASS,
                "Error messages are sanitized"
            )
        else:
            self.add_result(
                "Error Handling",
                CheckStatus.WARN,
                "Error messages may leak sensitive information",
                "Sanitize all error messages before returning to clients"
            )
    
    def print_report(self) -> None:
        """Print the checklist report."""
        print("\n" + "=" * 60)
        print("SECURITY CHECKLIST REPORT")
        print("=" * 60 + "\n")
        
        passed = sum(1 for r in self.results if r.status == CheckStatus.PASS)
        failed = sum(1 for r in self.results if r.status == CheckStatus.FAIL)
        warned = sum(1 for r in self.results if r.status == CheckStatus.WARN)
        
        for result in self.results:
            print(f"{result.status.value} {result.name}")
            print(f"   {result.message}")
            if result.recommendation:
                print(f"   → {result.recommendation}")
            print()
        
        print("=" * 60)
        print(f"Summary: {passed} passed, {failed} failed, {warned} warnings")
        print("=" * 60)
        
        if failed > 0:
            print("\n⚠️  Address FAILED items before deploying to production!")
        elif warned > 0:
            print("\n💡 Consider addressing WARNING items for better security.")
        else:
            print("\n✅ All checks passed!")
    
    def get_score(self) -> float:
        """Get a security score (0-100)."""
        if not self.results:
            return 0
        
        scores = {
            CheckStatus.PASS: 100,
            CheckStatus.WARN: 50,
            CheckStatus.FAIL: 0,
            CheckStatus.SKIP: 50
        }
        
        total = sum(scores[r.status] for r in self.results)
        return total / len(self.results)
    
    def get_failures(self) -> list[CheckResult]:
        """Get all failed checks."""
        return [r for r in self.results if r.status == CheckStatus.FAIL]
    
    def get_warnings(self) -> list[CheckResult]:
        """Get all warning checks."""
        return [r for r in self.results if r.status == CheckStatus.WARN]


def run_security_audit(
    has_input_validator: bool = False,
    has_output_filter: bool = False,
    has_rate_limiter: bool = False,
    has_audit_logger: bool = False,
    has_tool_registry: bool = False,
    using_https: bool = False,
    cors_restricted: bool = False,
    using_secrets_manager: bool = False,
    safe_errors: bool = False
) -> SecurityChecklist:
    """
    Run a security audit against an agent deployment.
    
    Args:
        has_input_validator: Whether input validation is configured
        has_output_filter: Whether output filtering is configured
        has_rate_limiter: Whether rate limiting is configured
        has_audit_logger: Whether audit logging is configured
        has_tool_registry: Whether secure tool registry is configured
        using_https: Whether HTTPS is enforced
        cors_restricted: Whether CORS is properly restricted
        using_secrets_manager: Whether using a secrets manager
        safe_errors: Whether error messages are sanitized
    
    Returns:
        SecurityChecklist with results
    """
    checklist = SecurityChecklist()
    
    checklist.check_api_key_security()
    checklist.check_input_validation(has_input_validator)
    checklist.check_output_filtering(has_output_filter)
    checklist.check_rate_limiting(has_rate_limiter)
    checklist.check_audit_logging(has_audit_logger)
    checklist.check_tool_permissions(has_tool_registry)
    checklist.check_https(using_https)
    checklist.check_cors(cors_restricted)
    checklist.check_secrets_management(using_secrets_manager)
    checklist.check_error_handling(safe_errors)
    
    return checklist


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Running Security Audit...")
    
    # Simulate an agent with partial security
    checklist = run_security_audit(
        has_input_validator=True,
        has_output_filter=True,
        has_rate_limiter=True,
        has_audit_logger=False,
        has_tool_registry=False,
        using_https=False,
        cors_restricted=False,
        using_secrets_manager=False,
        safe_errors=True
    )
    
    checklist.print_report()
    
    print(f"\nSecurity Score: {checklist.get_score():.0f}/100")
    
    # Show what needs attention
    failures = checklist.get_failures()
    if failures:
        print("\n🚨 Critical items to fix:")
        for f in failures:
            print(f"  - {f.name}: {f.recommendation}")
    
    warnings = checklist.get_warnings()
    if warnings:
        print("\n⚠️ Items to consider:")
        for w in warnings:
            print(f"  - {w.name}: {w.recommendation}")