# Chapter 41: Security Considerations - Code Examples

This directory contains all code examples for Chapter 41, which covers security best practices for AI agents in production.

## Files Overview

| File | Description |
|------|-------------|
| `secure_config.py` | Secure API key management using environment variables |
| `secrets_manager.py` | Production secrets management with multiple providers (env, file, AWS) |
| `input_validator.py` | Input validation and prompt injection prevention |
| `output_security.py` | Output filtering to prevent data leakage |
| `rate_limiter.py` | Rate limiting and abuse detection |
| `audit_logger.py` | Security audit logging with tamper-evident hash chains |
| `secure_tools.py` | Least privilege tool access with permission system |
| `security_checklist.py` | Security hardening checklist for deployments |
| `exercise_solution.py` | Complete SecureAgent wrapper combining all security patterns |

## Prerequisites

Install required dependencies:
```bash
uv add anthropic python-dotenv
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Running the Examples

Each file can be run independently to see demonstrations:
```bash
# Secure configuration
uv run python secure_config.py

# Secrets management
uv run python secrets_manager.py

# Input validation
uv run python input_validator.py

# Output security
uv run python output_security.py

# Rate limiting
uv run python rate_limiter.py

# Audit logging
uv run python audit_logger.py

# Secure tools
uv run python secure_tools.py

# Security checklist
uv run python security_checklist.py

# Complete solution
uv run python exercise_solution.py
```

## Key Concepts

### 1. API Key Security

Never hardcode API keys. Use environment variables or secrets managers:
```python
from secure_config import SecureConfig

config = SecureConfig(".env")
api_key = config.anthropic_api_key
```

### 2. Input Validation

Validate all user input before processing:
```python
from input_validator import InputValidator

validator = InputValidator(strict_mode=False)
result = validator.validate(user_input)

if result.is_valid:
    # Safe to process
    process(result.sanitized_input)
else:
    # Block and log
    log_security_event(result)
```

### 3. Output Filtering

Filter agent outputs to prevent data leakage:
```python
from output_security import OutputSecurityFilter

filter = OutputSecurityFilter(redact_pii=True, block_secrets=True)
analysis = filter.analyze(agent_response)

if analysis.is_safe:
    return analysis.redacted_output
else:
    return "Response blocked for security reasons."
```

### 4. Rate Limiting

Protect against abuse with rate limiting:
```python
from rate_limiter import RateLimiter, RateLimitResult

limiter = RateLimiter(requests_per_minute=60)

result = limiter.check(client_id)
if result == RateLimitResult.ALLOWED:
    # Process request
    pass
else:
    # Return 429 Too Many Requests
    pass
```

### 5. Audit Logging

Log all security-relevant events:
```python
from audit_logger import AuditLogger, SecurityEventType

audit = AuditLogger(log_file="audit.log")

audit.log_injection_attempt(
    client_id="user_123",
    input_text=malicious_input,
    patterns_matched=["ignore.*instructions"]
)
```

### 6. Least Privilege Tools

Restrict tool access based on permissions:
```python
from secure_tools import SecureToolRegistry, Permission

registry = SecureToolRegistry()

@registry.register(
    name="read_file",
    permissions={Permission.READ, Permission.FILESYSTEM}
)
def read_file(path: str) -> str:
    return open(path).read()

# Only clients with READ and FILESYSTEM permissions can use this tool
```

### 7. Security Checklist

Run audits before deploying:
```python
from security_checklist import run_security_audit

checklist = run_security_audit(
    has_input_validator=True,
    has_output_filter=True,
    has_rate_limiter=True,
    has_audit_logger=True,
    using_https=True
)

checklist.print_report()
print(f"Security Score: {checklist.get_score()}/100")
```

## The SecureAgent Wrapper

The exercise solution combines all security patterns into a single wrapper:
```python
from exercise_solution import SecureAgent

# Wrap any agent with security
secure = SecureAgent(
    agent=my_agent_function,
    requests_per_minute=60,
    strict_mode=False,
    audit_log_file="security.log"
)

# All requests are now protected
response = secure.run(
    prompt=user_input,
    client_id="user_123",
    ip_address="192.168.1.1"
)

if not response.is_blocked:
    print(response.content)
```

## Security Layers

The SecureAgent implements defense-in-depth:
```
User Input
    │
    ▼
┌─────────────────┐
│  Rate Limiting  │ ──► Block if exceeded
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Abuse Detection │ ──► Block if suspicious patterns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Input Validation │ ──► Block injection attempts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Execution │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Filtering │ ──► Redact PII, block secrets
└────────┬────────┘
         │
         ▼
    Safe Response
```

## Common Pitfalls

1. **Trusting user input** - Always validate, even if it looks safe
2. **Logging sensitive data** - Mask API keys and PII in logs
3. **Overly permissive CORS** - Restrict to known origins
4. **Not rotating secrets** - Use secrets managers that support rotation
5. **Insufficient tool sandboxing** - Use allowlists, not blocklists

## Further Reading

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Anthropic Security Best Practices](https://docs.anthropic.com)
- Chapter 40: Deployment Strategies
- Chapter 42: Research Assistant Agent (next chapter)
