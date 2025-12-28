# Chapter 31: Human-in-the-Loop - Code Files

This directory contains complete, runnable code examples for implementing human-in-the-loop patterns in AI agents.

## Files

### Core Components

| File | Description |
|------|-------------|
| `approval_gate.py` | Basic approval gate implementation with pending requests, approval history, and configurable auto-approval |
| `approval_aware_agent.py` | Agent that integrates approval gates for sensitive operations |
| `confirmation_patterns.py` | Various confirmation patterns: decorator, tiered, and preview-based |
| `feedback_collector.py` | System for collecting human feedback (ratings, corrections, preferences, flags) |
| `escalation.py` | Escalation manager for situations beyond agent capabilities |
| `hitl_agent.py` | Complete Human-in-the-Loop agent combining all components |

### Exercise

| File | Description |
|------|-------------|
| `exercise_solution.py` | Solution to the file management agent exercise |

## Quick Start

1. Make sure you have your `.env` file set up with your API key:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

2. Install dependencies:
   ```bash
   uv add anthropic python-dotenv
   ```

3. Run the complete HITL agent:
   ```bash
   python hitl_agent.py
   ```

## Component Overview

### Approval Gate
```python
from approval_gate import ApprovalGate, ApprovalStatus

gate = ApprovalGate(auto_approve_low_risk=True)
request = gate.request_approval(
    action_type="send_email",
    description="Send newsletter",
    details={"recipients": 100},
    risk_level="high"
)
result = gate.wait_for_approval(request)

if result.status == ApprovalStatus.APPROVED:
    # Execute the action
    pass
```

### Confirmation Patterns
```python
from confirmation_patterns import requires_confirmation, TieredConfirmation

@requires_confirmation("This will delete all data. Continue?")
def dangerous_operation():
    pass

# Or use tiered confirmation
tiered = TieredConfirmation()
result = tiered.request_confirmation(
    action="Delete database",
    details={"table": "users"},
    risk_level=RiskLevel.CRITICAL
)
```

### Feedback Collection
```python
from feedback_collector import FeedbackCollector

collector = FeedbackCollector(storage_path="feedback.json")
collector.collect_rating(
    context={"task": "summarization"},
    agent_output="Summary of the document..."
)
```

### Escalation
```python
from escalation import EscalationManager, EscalationReason, EscalationPriority

manager = EscalationManager()
escalation = manager.escalate(
    reason=EscalationReason.CUSTOMER_REQUEST,
    summary="User requested human support",
    context={"user_id": "123"},
    conversation_history=[...],
    priority=EscalationPriority.HIGH
)
```

## Testing Without API Calls

Several components can be tested without making API calls:

- `approval_gate.py` - Pure Python, no API needed
- `confirmation_patterns.py` - Pure Python, no API needed  
- `feedback_collector.py` - Pure Python, no API needed
- `escalation.py` - Pure Python, no API needed

Only `approval_aware_agent.py`, `hitl_agent.py`, and `exercise_solution.py` require API access.

## Key Concepts Demonstrated

1. **Risk-based approval** - Different actions require different levels of oversight
2. **Tiered confirmation** - Match confirmation intensity to risk level
3. **Preview patterns** - Show users what will happen before execution
4. **Feedback loops** - Collect and store human feedback for improvement
5. **Escalation triggers** - Automatically detect when human handoff is needed
6. **Audit trails** - Track all decisions and approvals for accountability
