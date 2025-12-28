"""
Code review system using voting for vulnerability detection.

Chapter 21: Parallelization - Implementation

This module implements a comprehensive security code review system
that uses multiple specialized reviewers in parallel, then aggregates
their findings through voting to identify confirmed vulnerabilities.
"""

import asyncio
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class Vulnerability:
    """
    A detected security vulnerability.
    
    Attributes:
        type: Category of vulnerability (e.g., "SQL Injection")
        severity: Risk level (critical, high, medium, low)
        location: Where in the code (line number or function)
        description: What the vulnerability is
        recommendation: How to fix it
    """
    type: str
    severity: str
    location: str
    description: str
    recommendation: str


@dataclass
class ReviewerVote:
    """
    A single reviewer's assessment of the code.
    
    Attributes:
        reviewer_name: Which reviewer provided this assessment
        found_vulnerabilities: List of identified vulnerabilities
        overall_risk: Aggregate risk assessment
        success: Whether the review completed successfully
        error: Error message if failed
        execution_time: How long this review took
    """
    reviewer_name: str
    found_vulnerabilities: list[Vulnerability] = field(default_factory=list)
    overall_risk: str = "unknown"
    success: bool = True
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class CodeReviewResult:
    """
    Aggregated code review results from all reviewers.
    
    Attributes:
        votes: Individual reviewer assessments
        confirmed_vulnerabilities: Issues confirmed by multiple reviewers
        risk_consensus: Agreed-upon risk level
        confidence: Confidence in the consensus
        execution_time: Total review time
    """
    votes: list[ReviewerVote]
    confirmed_vulnerabilities: list[dict]
    risk_consensus: str
    confidence: float
    execution_time: float


class CodeReviewSystem:
    """
    Multi-perspective code review system using voting.
    
    Multiple specialized reviewers analyze code in parallel,
    then results are aggregated to identify confirmed issues.
    
    Reviewers have different specializations:
    - Injection attacks (SQL, command, XSS)
    - Authentication and authorization
    - Cryptography and secrets management
    - Data protection and privacy
    - General security practices
    
    Example usage:
        reviewer = CodeReviewSystem(confirmation_threshold=2)
        result = await reviewer.review(code, "User authentication module")
        print(format_review_report(result))
    """
    
    # Specialized security reviewers
    REVIEWERS = [
        {
            "name": "injection_specialist",
            "system": """You are a security expert specializing in injection attacks.

Your focus areas:
- SQL injection (dynamic queries, string concatenation)
- Command injection (shell commands, subprocess calls)
- XSS (cross-site scripting, HTML injection)
- Template injection (format strings, eval)
- LDAP injection

Be thorough but avoid false positives. Only report issues you're confident about.
Consider context - parameterized queries are safe, string concatenation is not."""
        },
        {
            "name": "auth_specialist",
            "system": """You are a security expert specializing in authentication and authorization.

Your focus areas:
- Broken authentication (weak passwords, no rate limiting)
- Session management (insecure cookies, session fixation)
- Access control issues (missing authorization checks)
- Privilege escalation vulnerabilities
- Insecure direct object references

Be thorough but avoid false positives. Look for missing checks, not just present ones."""
        },
        {
            "name": "crypto_specialist",
            "system": """You are a security expert specializing in cryptography and secrets.

Your focus areas:
- Weak or broken encryption algorithms
- Hardcoded secrets, API keys, passwords
- Insecure random number generation
- Key management issues
- Hash function misuse (MD5, SHA1 for passwords)
- Missing encryption for sensitive data

Be thorough but practical. Note: using secrets for non-security purposes may be ok."""
        },
        {
            "name": "data_specialist",
            "system": """You are a security expert specializing in data protection.

Your focus areas:
- Sensitive data exposure (logging passwords, PII in URLs)
- Insufficient input validation
- Mass assignment vulnerabilities
- Information leakage in error messages
- Insecure data storage
- Missing data sanitization

Focus on how data flows through the code and where it might be exposed."""
        },
        {
            "name": "general_security",
            "system": """You are a general application security expert.

Your role is to catch security issues that specialists might miss:
- Business logic flaws
- Race conditions
- Insecure defaults
- Debug code left in production
- Missing security headers
- Unsafe deserialization
- Dependency issues

Take a holistic view of the code's security posture."""
        }
    ]
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        confirmation_threshold: int = 2
    ):
        """
        Initialize the code review system.
        
        Args:
            model: The Claude model to use for reviews
            max_tokens: Maximum tokens per reviewer response
            confirmation_threshold: Minimum votes to confirm a vulnerability
        """
        self.model = model
        self.max_tokens = max_tokens
        self.confirmation_threshold = confirmation_threshold
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _get_review(
        self,
        reviewer: dict,
        code: str,
        context: str
    ) -> ReviewerVote:
        """
        Get a security review from a single reviewer.
        
        Args:
            reviewer: Reviewer configuration with name and system prompt
            code: The code to review
            context: Description of what the code does
            
        Returns:
            ReviewerVote with findings and risk assessment
        """
        import time
        start = time.time()
        
        prompt = f"""Review the following code for security vulnerabilities.

Context: {context}

Code:
```
{code}
```

For each vulnerability found, provide:
1. TYPE: The category of vulnerability (e.g., "SQL Injection", "XSS", "Hardcoded Secret")
2. SEVERITY: critical/high/medium/low
3. LOCATION: Where in the code (line number or function name)
4. DESCRIPTION: What the vulnerability is and why it's dangerous
5. RECOMMENDATION: How to fix it

Also provide an OVERALL_RISK assessment: critical/high/medium/low/none

Format your response EXACTLY as:

VULNERABILITY 1:
TYPE: [category]
SEVERITY: [level]
LOCATION: [where]
DESCRIPTION: [what and why]
RECOMMENDATION: [how to fix]

VULNERABILITY 2:
TYPE: [category]
SEVERITY: [level]
LOCATION: [where]
DESCRIPTION: [what and why]
RECOMMENDATION: [how to fix]

(continue for additional vulnerabilities...)

OVERALL_RISK: [level]

If no vulnerabilities found, state "NO VULNERABILITIES FOUND" and set OVERALL_RISK: none"""

        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=reviewer["system"],
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            vulnerabilities = self._parse_vulnerabilities(text)
            overall_risk = self._extract_overall_risk(text)
            
            return ReviewerVote(
                reviewer_name=reviewer["name"],
                found_vulnerabilities=vulnerabilities,
                overall_risk=overall_risk,
                execution_time=time.time() - start
            )
            
        except Exception as e:
            return ReviewerVote(
                reviewer_name=reviewer["name"],
                overall_risk="unknown",
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def _parse_vulnerabilities(self, text: str) -> list[Vulnerability]:
        """
        Parse vulnerability entries from response text.
        
        Extracts structured vulnerability data from the reviewer's
        formatted response.
        """
        vulnerabilities = []
        
        # Split by VULNERABILITY markers
        parts = re.split(r'VULNERABILITY\s*\d*:', text, flags=re.IGNORECASE)
        
        for part in parts[1:]:  # Skip first part (before any vulnerability)
            vuln = {}
            for field in ["TYPE", "SEVERITY", "LOCATION", "DESCRIPTION", "RECOMMENDATION"]:
                value = self._extract_field(part, field)
                if value:
                    vuln[field.lower()] = value
            
            # Only add if we have at least type and description
            if vuln.get("type") and vuln.get("description"):
                vulnerabilities.append(Vulnerability(
                    type=vuln.get("type", "Unknown"),
                    severity=vuln.get("severity", "medium").lower(),
                    location=vuln.get("location", "Unknown"),
                    description=vuln.get("description", ""),
                    recommendation=vuln.get("recommendation", "Review and fix")
                ))
        
        return vulnerabilities
    
    def _extract_field(self, text: str, field: str) -> str | None:
        """Extract a field value from formatted text."""
        # Pattern matches FIELD: value (possibly multiline until next field)
        pattern = rf"{field}:\s*(.+?)(?=\n(?:TYPE|SEVERITY|LOCATION|DESCRIPTION|RECOMMENDATION|OVERALL_RISK|VULNERABILITY):|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_overall_risk(self, text: str) -> str:
        """Extract overall risk assessment from response."""
        match = re.search(r"OVERALL_RISK:\s*(\w+)", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "unknown"
    
    def _aggregate_vulnerabilities(
        self, 
        votes: list[ReviewerVote]
    ) -> list[dict]:
        """
        Aggregate vulnerabilities across all reviewers.
        
        Groups similar vulnerabilities and counts how many reviewers
        found each one. Vulnerabilities found by multiple reviewers
        are marked as "confirmed".
        """
        # Collect all vulnerabilities with their sources
        all_vulns = []
        for vote in votes:
            if vote.success:
                for vuln in vote.found_vulnerabilities:
                    all_vulns.append({
                        "vulnerability": vuln,
                        "reporter": vote.reviewer_name
                    })
        
        # Group similar vulnerabilities by type and location
        grouped: dict[str, dict] = {}
        for item in all_vulns:
            vuln = item["vulnerability"]
            # Create a grouping key from type and normalized location
            key = f"{vuln.type.lower().strip()}:{vuln.location.lower().strip()}"
            
            if key not in grouped:
                grouped[key] = {
                    "type": vuln.type,
                    "severity": vuln.severity,
                    "location": vuln.location,
                    "descriptions": [],
                    "recommendations": [],
                    "reporters": []
                }
            
            grouped[key]["descriptions"].append(vuln.description)
            grouped[key]["recommendations"].append(vuln.recommendation)
            grouped[key]["reporters"].append(item["reporter"])
            
            # Upgrade severity if any reporter says higher
            severity_order = ["none", "low", "medium", "high", "critical"]
            current_idx = severity_order.index(grouped[key]["severity"].lower())
            new_idx = severity_order.index(vuln.severity.lower())
            if new_idx > current_idx:
                grouped[key]["severity"] = vuln.severity
        
        # Build result list with vote counts
        result = []
        for key, vuln_data in grouped.items():
            vote_count = len(set(vuln_data["reporters"]))  # Unique reporters
            result.append({
                **vuln_data,
                "vote_count": vote_count,
                "confirmed": vote_count >= self.confirmation_threshold
            })
        
        # Sort: confirmed first, then by severity, then by vote count
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        result.sort(
            key=lambda x: (
                x["confirmed"],
                severity_order.get(x["severity"].lower(), 0),
                x["vote_count"]
            ),
            reverse=True
        )
        
        return result
    
    def _calculate_risk_consensus(
        self, 
        votes: list[ReviewerVote]
    ) -> tuple[str, float]:
        """
        Calculate consensus risk level and confidence.
        
        Returns:
            (risk_level, confidence)
        """
        valid_votes = [
            v.overall_risk.lower() 
            for v in votes 
            if v.success and v.overall_risk not in ("unknown", "")
        ]
        
        if not valid_votes:
            return "unknown", 0.0
        
        counts = Counter(valid_votes)
        winner, count = counts.most_common(1)[0]
        confidence = count / len(valid_votes)
        
        return winner, confidence
    
    async def review(
        self,
        code: str,
        context: str = "General code review"
    ) -> CodeReviewResult:
        """
        Perform a comprehensive security review of the code.
        
        Runs all reviewers in parallel and aggregates their findings.
        
        Args:
            code: The source code to review
            context: Description of what the code does
            
        Returns:
            CodeReviewResult with aggregated findings and consensus
        """
        import time
        start = time.time()
        
        # Get reviews from all specialists in parallel
        votes = await asyncio.gather(*[
            self._get_review(reviewer, code, context)
            for reviewer in self.REVIEWERS
        ])
        
        # Aggregate results
        confirmed_vulns = self._aggregate_vulnerabilities(list(votes))
        risk_consensus, confidence = self._calculate_risk_consensus(list(votes))
        
        return CodeReviewResult(
            votes=list(votes),
            confirmed_vulnerabilities=confirmed_vulns,
            risk_consensus=risk_consensus,
            confidence=confidence,
            execution_time=time.time() - start
        )


def format_review_report(result: CodeReviewResult) -> str:
    """
    Format the review result as a readable security report.
    
    Args:
        result: The CodeReviewResult to format
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("                    SECURITY CODE REVIEW REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Overall Risk Level:    {result.risk_consensus.upper()}")
    lines.append(f"Confidence:            {result.confidence:.0%}")
    lines.append(f"Review Duration:       {result.execution_time:.2f}s")
    confirmed_count = len([v for v in result.confirmed_vulnerabilities if v["confirmed"]])
    total_count = len(result.confirmed_vulnerabilities)
    lines.append(f"Confirmed Issues:      {confirmed_count}")
    lines.append(f"Potential Issues:      {total_count - confirmed_count}")
    lines.append("")
    
    # Reviewer Summary
    lines.append("REVIEWER ASSESSMENTS")
    lines.append("-" * 70)
    for vote in result.votes:
        if vote.success:
            vuln_count = len(vote.found_vulnerabilities)
            lines.append(
                f"  {vote.reviewer_name:25} | "
                f"Risk: {vote.overall_risk:8} | "
                f"Issues: {vuln_count} | "
                f"Time: {vote.execution_time:.1f}s"
            )
        else:
            lines.append(f"  {vote.reviewer_name:25} | ERROR: {vote.error}")
    lines.append("")
    
    # Confirmed Vulnerabilities
    confirmed = [v for v in result.confirmed_vulnerabilities if v["confirmed"]]
    if confirmed:
        lines.append("CONFIRMED VULNERABILITIES")
        lines.append("-" * 70)
        lines.append("(Found by 2+ reviewers - high confidence)")
        lines.append("")
        
        for i, vuln in enumerate(confirmed, 1):
            severity_badge = f"[{vuln['severity'].upper()}]"
            lines.append(f"{i}. {severity_badge} {vuln['type']}")
            lines.append(f"   Location: {vuln['location']}")
            lines.append(f"   Confirmed by: {', '.join(set(vuln['reporters']))}")
            lines.append(f"   Description: {vuln['descriptions'][0][:200]}")
            if len(vuln['descriptions'][0]) > 200:
                lines.append(f"                ...")
            lines.append(f"   Fix: {vuln['recommendations'][0][:200]}")
            if len(vuln['recommendations'][0]) > 200:
                lines.append(f"        ...")
            lines.append("")
    else:
        lines.append("CONFIRMED VULNERABILITIES")
        lines.append("-" * 70)
        lines.append("No issues confirmed by multiple reviewers.")
        lines.append("")
    
    # Potential Issues (not confirmed)
    potential = [v for v in result.confirmed_vulnerabilities if not v["confirmed"]]
    if potential:
        lines.append("POTENTIAL ISSUES (Requires Further Review)")
        lines.append("-" * 70)
        lines.append("(Found by only 1 reviewer - verify manually)")
        lines.append("")
        
        for vuln in potential:
            lines.append(
                f"  • [{vuln['severity'].upper()}] {vuln['type']} "
                f"at {vuln['location']} "
                f"(reported by: {vuln['reporters'][0]})"
            )
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("                         END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Example: Review Vulnerable Code
# =============================================================================

async def main():
    """
    Demonstrate the code review system with intentionally vulnerable code.
    """
    # Sample code with multiple security vulnerabilities
    vulnerable_code = '''
def login(username, password):
    """Authenticate a user and start session."""
    # Build SQL query
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    
    if result:
        # Store password in session for "remember me" feature
        session['password'] = password
        session['user'] = username
        session['role'] = result['role']
        return True
    return False

def get_user_data(user_id):
    """Get user profile data."""
    # Admin bypass for testing (TODO: remove before production)
    if user_id == "admin_debug":
        return get_all_users()
    
    # Get user data
    data = db.query(f"SELECT * FROM profiles WHERE id={user_id}")
    return data

def reset_password(email):
    """Send password reset email."""
    import random
    # Generate reset token
    token = random.randint(1000, 9999)  # 4-digit reset code
    
    # Store token and send email
    cache.set(f"reset_{email}", token, ttl=3600)
    send_email(email, f"Your password reset code is: {token}")
    print(f"DEBUG: Reset token for {email} is {token}")  # For debugging
    
def render_profile(user_data):
    """Render user profile page."""
    template = f"<h1>Welcome {user_data['name']}</h1>"
    template += f"<p>Email: {user_data['email']}</p>"
    template += f"<p>Bio: {user_data['bio']}</p>"
    return template

def change_password(user_id, new_password):
    """Change user's password."""
    # Update password in database
    hashed = md5(new_password.encode()).hexdigest()
    db.execute(f"UPDATE users SET password='{hashed}' WHERE id={user_id}")
    return True

# API endpoint handlers
def handle_admin_action(request):
    """Process admin actions."""
    action = request.get('action')
    eval(action)  # Execute the requested action

# Configuration
DB_PASSWORD = "super_secret_123"
API_KEY = "sk-live-abcd1234efgh5678"
'''
    
    context = """User authentication and profile management module for a web application.
This code handles user login, password reset, profile rendering, and admin operations.
It connects to a database and sends emails for password resets."""
    
    print("Starting parallel security review...")
    print("5 specialized reviewers analyzing code simultaneously...\n")
    
    # Create reviewer and run analysis
    reviewer = CodeReviewSystem(confirmation_threshold=2)
    result = await reviewer.review(vulnerable_code, context)
    
    # Print the formatted report
    report = format_review_report(result)
    print(report)
    
    # Also save to file
    with open("security_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to: security_report.txt")


if __name__ == "__main__":
    asyncio.run(main())
