"""
Output filtering and verification for AI agents.

Chapter 32: Guardrails and Safety

This module provides comprehensive output filtering to:
- Redact sensitive data (API keys, PII, etc.)
- Detect potentially harmful content
- Verify output structure and content
"""

import re
import json
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
        """
        Initialize the output filter.
        
        Args:
            redact_api_keys: Whether to redact API key patterns
            redact_emails: Whether to redact email addresses
            redact_phone_numbers: Whether to redact phone numbers
            redact_ssn: Whether to redact Social Security Numbers
            redact_credit_cards: Whether to redact credit card numbers
            max_output_length: Maximum allowed output length
            custom_patterns: Additional patterns to redact {name: regex_pattern}
        """
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
                |
                # OpenAI API keys
                sk-[a-zA-Z0-9]{40,60}
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
                count = len(matches) if isinstance(matches[0], str) else len(matches)
                redactions.append(f"Redacted {count} {name} pattern(s)")
                filtered = pattern.sub(f"[REDACTED_{name.upper()}]", filtered)
        
        # Check for potentially harmful content patterns
        harmful_patterns = [
            (r"rm\s+-rf\s+/", "Destructive shell command (rm -rf /)"),
            (r"DROP\s+TABLE", "Destructive SQL (DROP TABLE)"),
            (r"DELETE\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*1", "Mass deletion SQL"),
            (r"FORMAT\s+[A-Z]:", "Disk format command"),
            (r":(){ :\|:& };:", "Fork bomb pattern"),
            (r">\s*/dev/sd[a-z]", "Direct disk write"),
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
    
    def verify_no_secrets(
        self, 
        output: str, 
        known_secrets: list[str]
    ) -> FilterResult:
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
            if secret and len(secret) > 5 and secret in output:
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
    
    def verify_no_code_execution(self, output: str) -> FilterResult:
        """
        Check output for code that shouldn't be executed.
        
        Args:
            output: The output to check
            
        Returns:
            FilterResult with verification status
        """
        concerns = []
        
        # Patterns that might indicate executable code in output
        dangerous_patterns = [
            (r"<script\b", "JavaScript script tag"),
            (r"javascript:", "JavaScript URL"),
            (r"eval\s*\(", "eval() call"),
            (r"exec\s*\(", "exec() call"),
            (r"os\.system\s*\(", "os.system() call"),
            (r"subprocess\.", "subprocess usage"),
            (r"__import__\s*\(", "dynamic import"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                concerns.append(f"Potentially dangerous pattern: {description}")
        
        return FilterResult(
            is_safe=len(concerns) == 0,
            filtered_value=output,
            concerns=concerns
        )


# Example usage and tests
if __name__ == "__main__":
    # Create filter with various options
    output_filter = OutputFilter(
        redact_api_keys=True,
        redact_emails=True,
        redact_ssn=True,
        redact_credit_cards=True,
    )
    
    # Test API key redaction
    text_with_api_key = "Use this key: sk-ant-api03-abc123def456ghi789jkl012mno345"
    result = output_filter.filter_output(text_with_api_key)
    print(f"API key redaction:")
    print(f"  Original: {text_with_api_key}")
    print(f"  Filtered: {result.filtered_value}")
    print(f"  Redactions: {result.redactions}")
    
    # Test SSN redaction
    text_with_ssn = "My SSN is 123-45-6789"
    result = output_filter.filter_output(text_with_ssn)
    print(f"\nSSN redaction:")
    print(f"  Filtered: {result.filtered_value}")
    
    # Test credit card redaction
    text_with_cc = "Card: 4111-1111-1111-1111"
    result = output_filter.filter_output(text_with_cc)
    print(f"\nCredit card redaction:")
    print(f"  Filtered: {result.filtered_value}")
    
    # Test harmful content detection
    dangerous_output = "To delete everything: rm -rf /"
    result = output_filter.filter_output(dangerous_output)
    print(f"\nHarmful content detection:")
    print(f"  Is safe: {result.is_safe}")
    print(f"  Concerns: {result.concerns}")
    
    # Test known secrets
    result = output_filter.verify_no_secrets(
        "The password is SuperSecret123!",
        ["SuperSecret123!"]
    )
    print(f"\nKnown secret detection:")
    print(f"  Filtered: {result.filtered_value}")
    print(f"  Concerns: {result.concerns}")
    
    # Test JSON validation
    json_output = '{"name": "John", "api_key": "secret123"}'
    result = output_filter.verify_json_structure(
        json_output,
        required_fields=["name"],
        forbidden_fields=["api_key", "password"]
    )
    print(f"\nJSON structure validation:")
    print(f"  Is safe: {result.is_safe}")
    print(f"  Concerns: {result.concerns}")
    
    print("\n✅ Output filter tests complete!")
