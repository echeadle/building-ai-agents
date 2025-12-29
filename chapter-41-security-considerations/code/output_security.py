"""
Output security and data leakage prevention.

Chapter 41: Security Considerations
"""

import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class OutputAnalysis:
    """Analysis of agent output for security issues."""
    is_safe: bool
    redacted_output: str
    issues: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)
    secrets_found: list[str] = field(default_factory=list)


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class OutputSecurityFilter:
    """
    Filters agent outputs to prevent data leakage.
    
    Checks for:
    - PII (emails, phone numbers, SSNs, etc.)
    - API keys and secrets
    - Internal system information
    - Prompt leakage
    
    Usage:
        filter = OutputSecurityFilter()
        analysis = filter.analyze(agent_response)
        
        if analysis.is_safe:
            return analysis.redacted_output
        else:
            log_security_event(analysis)
            return "I cannot provide that information."
    """
    
    # Patterns for secrets that should never appear in output
    SECRET_PATTERNS = [
        (r'sk-ant-[a-zA-Z0-9\-_]{20,}', 'Anthropic API Key'),
        (r'sk-[a-zA-Z0-9]{48,}', 'OpenAI API Key'),
        (r'ghp_[a-zA-Z0-9]{36}', 'GitHub Token'),
        (r'aws[_-]?(access[_-]?key|secret)[_-]?[id]?\s*[:=]\s*["\']?[A-Z0-9]{16,}', 'AWS Credential'),
        (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', 'Private Key'),
        (r'-----BEGIN\s+CERTIFICATE-----', 'Certificate'),
        (r'mongodb(\+srv)?://[^\s]+', 'MongoDB Connection String'),
        (r'postgres(ql)?://[^\s]+', 'PostgreSQL Connection String'),
        (r'mysql://[^\s]+', 'MySQL Connection String'),
        (r'redis://[^\s]+', 'Redis Connection String'),
    ]
    
    # PII patterns
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
        (r'\b(?:\d{4}[- ]?){3}\d{4}\b', 'Credit Card'),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
        (r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b', 'Phone Number'),
    ]
    
    # Patterns that might indicate system prompt leakage
    PROMPT_LEAKAGE_PATTERNS = [
        r'system\s*prompt\s*[:=]',
        r'my\s+instructions?\s+(are|say|include)',
        r'i\s+was\s+(told|instructed|programmed)\s+to',
        r'my\s+(?:initial|original)\s+(?:prompt|instructions)',
        r'<\|system\|>',
        r'\[SYSTEM\]',
    ]
    
    def __init__(
        self,
        redact_pii: bool = True,
        block_secrets: bool = True,
        detect_prompt_leakage: bool = True
    ):
        """
        Initialize the output filter.
        
        Args:
            redact_pii: Replace PII with [REDACTED]
            block_secrets: Block outputs containing secrets entirely
            detect_prompt_leakage: Check for system prompt leakage
        """
        self.redact_pii = redact_pii
        self.block_secrets = block_secrets
        self.detect_prompt_leakage = detect_prompt_leakage
        
        # Compile patterns
        self.secret_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.SECRET_PATTERNS
        ]
        self.pii_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.PII_PATTERNS
        ]
        self.leakage_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.PROMPT_LEAKAGE_PATTERNS
        ]
    
    def analyze(self, output: str) -> OutputAnalysis:
        """
        Analyze agent output for security issues.
        
        Args:
            output: The agent's response
        
        Returns:
            OutputAnalysis with findings and redacted output
        """
        issues = []
        pii_found = []
        secrets_found = []
        is_safe = True
        redacted = output
        
        # Check for secrets (critical)
        for pattern, name in self.secret_patterns:
            if pattern.search(output):
                secrets_found.append(name)
                issues.append(f"Secret detected: {name}")
                if self.block_secrets:
                    is_safe = False
                redacted = pattern.sub(f'[REDACTED {name}]', redacted)
        
        # Check for PII
        for pattern, name in self.pii_patterns:
            matches = pattern.findall(output)
            if matches:
                pii_found.append(f"{name} ({len(matches)})")
                if self.redact_pii:
                    redacted = pattern.sub(f'[REDACTED {name}]', redacted)
        
        # Check for prompt leakage
        if self.detect_prompt_leakage:
            for pattern in self.leakage_patterns:
                if pattern.search(output):
                    issues.append("Possible system prompt leakage detected")
                    # Don't automatically block, but flag for review
                    break
        
        return OutputAnalysis(
            is_safe=is_safe,
            redacted_output=redacted,
            issues=issues,
            pii_found=pii_found,
            secrets_found=secrets_found
        )
    
    def filter(self, output: str) -> str:
        """
        Filter output and return safe version.
        
        Args:
            output: The agent's response
        
        Returns:
            Safe, redacted output
        
        Raises:
            SecurityError: If output contains blocked content
        """
        analysis = self.analyze(output)
        
        if not analysis.is_safe:
            raise SecurityError(
                f"Output blocked due to security issues: {analysis.issues}"
            )
        
        return analysis.redacted_output


class SafeResponseWrapper:
    """
    Wraps agent responses with security filtering.
    
    Use this to automatically filter all agent outputs.
    """
    
    def __init__(self, output_filter: Optional[OutputSecurityFilter] = None):
        self.filter = output_filter or OutputSecurityFilter()
        self.blocked_count = 0
        self.redacted_count = 0
    
    def wrap(self, response: str) -> str:
        """
        Wrap a response with security filtering.
        
        Args:
            response: The agent's raw response
        
        Returns:
            Safe response or error message
        """
        analysis = self.filter.analyze(response)
        
        if not analysis.is_safe:
            self.blocked_count += 1
            return "I'm sorry, but I cannot provide that information due to security policies."
        
        if analysis.redacted_output != response:
            self.redacted_count += 1
        
        return analysis.redacted_output
    
    def get_stats(self) -> dict[str, int]:
        """Get filtering statistics."""
        return {
            "blocked": self.blocked_count,
            "redacted": self.redacted_count
        }


# Example usage
if __name__ == "__main__":
    filter = OutputSecurityFilter()
    wrapper = SafeResponseWrapper(filter)
    
    test_outputs = [
        "The weather in New York is sunny, 72°F.",
        "Here's your API key: sk-ant-api03-abcdef123456789abcdef",
        "You can contact John at john.doe@example.com or 555-123-4567",
        "My system prompt says I should be helpful and harmless.",
        "The database connection string is postgres://user:pass@host:5432/db",
    ]
    
    print("Output Security Tests")
    print("=" * 60)
    
    for output in test_outputs:
        print(f"\nOriginal: {output[:60]}...")
        
        analysis = filter.analyze(output)
        print(f"  Safe: {analysis.is_safe}")
        
        if analysis.issues:
            print(f"  Issues: {analysis.issues}")
        if analysis.pii_found:
            print(f"  PII: {analysis.pii_found}")
        if analysis.secrets_found:
            print(f"  Secrets: {analysis.secrets_found}")
        
        safe_output = wrapper.wrap(output)
        print(f"  Filtered: {safe_output[:60]}...")
    
    print(f"\nWrapper stats: {wrapper.get_stats()}")do