---
chapter: 31
title: "Human-in-the-Loop"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 31: Human-in-the-Loop

## Introduction

In Chapter 30, we learned how to handle errors and help agents recover when things go wrong. But what about preventing catastrophic mistakes in the first place? What happens when your agent is about to send an email to 10,000 customers, delete a database table, or make a financial transaction?

This is where **human-in-the-loop** (HITL) patterns become essential. No matter how sophisticated your agent becomes, there are situations where human judgment is irreplaceable—and required. The goal isn't to limit your agent's capabilities, but to build appropriate checkpoints that keep humans informed and in control.

Think of it like autonomous vehicles: even the most advanced self-driving cars have mechanisms for human override. Your agents should too.

In this chapter, we'll build systems that pause for human approval, request confirmation for sensitive actions, integrate human feedback, and escalate when situations exceed the agent's capabilities.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement approval gates that pause agent execution for human review
- Design confirmation flows for high-stakes actions
- Build feedback loops that improve agent behavior
- Create escalation paths for situations beyond agent capabilities
- Balance automation with appropriate human oversight

## Why Human Oversight Matters

Before diving into implementation, let's understand why human-in-the-loop patterns are critical:

### 1. Irreversible Actions

Some actions cannot be undone:
- Sending emails or messages
- Deleting data
- Financial transactions
- Publishing content
- API calls with side effects

Once executed, these actions have real-world consequences. A human checkpoint before execution can prevent costly mistakes.

### 2. High-Stakes Decisions

Even if reversible, some decisions carry significant weight:
- Decisions affecting multiple users
- Actions with legal implications
- Changes to production systems
- Communications with external parties

### 3. Edge Cases and Ambiguity

LLMs handle typical cases well but can struggle with:
- Unusual situations not covered in training
- Ambiguous instructions
- Conflicting requirements
- Context requiring domain expertise

### 4. Trust Building

Human oversight builds trust gradually:
- Start with high oversight, reduce as confidence grows
- Document agent decisions for review
- Learn from human corrections
- Demonstrate reliability over time

> **Key Principle:** Autonomy should be earned through demonstrated reliability.

## The Approval Gate Pattern

An **approval gate** is a checkpoint where the agent pauses execution and waits for human approval before proceeding. Let's build a flexible approval system.

### Basic Approval Gate

```python
"""
Basic approval gate implementation.

Chapter 31: Human-in-the-Loop
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime


class ApprovalStatus(Enum):
    """Possible outcomes of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"  # Approved with changes
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """Represents a request for human approval."""
    request_id: str
    action_type: str
    description: str
    details: dict[str, Any]
    risk_level: str  # "low", "medium", "high", "critical"
    created_at: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    modified_details: Optional[dict[str, Any]] = None
    reviewed_at: Optional[datetime] = None


class ApprovalGate:
    """
    Manages approval requests for agent actions.
    
    In production, this would integrate with your approval workflow
    (Slack, email, web interface, etc.). For learning, we use
    interactive console input.
    """
    
    def __init__(self, auto_approve_low_risk: bool = False):
        """
        Initialize the approval gate.
        
        Args:
            auto_approve_low_risk: If True, automatically approve
                                   low-risk actions without human input.
        """
        self.auto_approve_low_risk = auto_approve_low_risk
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self.request_history: list[ApprovalRequest] = []
        self._request_counter = 0
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_counter += 1
        return f"REQ-{self._request_counter:04d}"
    
    def request_approval(
        self,
        action_type: str,
        description: str,
        details: dict[str, Any],
        risk_level: str = "medium"
    ) -> ApprovalRequest:
        """
        Create a new approval request.
        
        Args:
            action_type: Category of action (e.g., "email", "delete", "purchase")
            description: Human-readable description of what will happen
            details: Specific details of the action
            risk_level: Risk assessment ("low", "medium", "high", "critical")
        
        Returns:
            ApprovalRequest object with pending status
        """
        request = ApprovalRequest(
            request_id=self._generate_request_id(),
            action_type=action_type,
            description=description,
            details=details,
            risk_level=risk_level
        )
        
        self.pending_requests[request.request_id] = request
        return request
    
    def wait_for_approval(
        self,
        request: ApprovalRequest,
        timeout_seconds: Optional[float] = None
    ) -> ApprovalRequest:
        """
        Wait for human approval of a request.
        
        In production, this might poll an approval service or use webhooks.
        Here we use interactive console input for demonstration.
        
        Args:
            request: The approval request to wait on
            timeout_seconds: Maximum time to wait (None = no timeout)
        
        Returns:
            Updated ApprovalRequest with final status
        """
        # Auto-approve low risk if configured
        if self.auto_approve_low_risk and request.risk_level == "low":
            request.status = ApprovalStatus.APPROVED
            request.review_notes = "Auto-approved (low risk)"
            request.reviewed_at = datetime.now()
            self._finalize_request(request)
            return request
        
        # Display request details
        print("\n" + "=" * 60)
        print("🔔 APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Request ID: {request.request_id}")
        print(f"Action Type: {request.action_type}")
        print(f"Risk Level: {request.risk_level.upper()}")
        print(f"Description: {request.description}")
        print("\nDetails:")
        for key, value in request.details.items():
            print(f"  • {key}: {value}")
        print("=" * 60)
        
        # Get human input
        while True:
            choice = input("\n[A]pprove / [R]eject / [M]odify? ").strip().upper()
            
            if choice == 'A':
                request.status = ApprovalStatus.APPROVED
                request.review_notes = input("Notes (optional): ").strip() or None
                break
            elif choice == 'R':
                request.status = ApprovalStatus.REJECTED
                request.review_notes = input("Reason for rejection: ").strip()
                break
            elif choice == 'M':
                request.status = ApprovalStatus.MODIFIED
                request.review_notes = input("Modification notes: ").strip()
                # In a real system, you'd collect the modified details
                print("(In production, you would specify the modifications here)")
                break
            else:
                print("Please enter A, R, or M")
        
        request.reviewed_at = datetime.now()
        self._finalize_request(request)
        return request
    
    def _finalize_request(self, request: ApprovalRequest) -> None:
        """Move request from pending to history."""
        if request.request_id in self.pending_requests:
            del self.pending_requests[request.request_id]
        self.request_history.append(request)
    
    def get_approval_stats(self) -> dict[str, int]:
        """Get statistics on approval history."""
        stats = {
            "total": len(self.request_history),
            "approved": 0,
            "rejected": 0,
            "modified": 0,
            "timeout": 0
        }
        
        for request in self.request_history:
            if request.status == ApprovalStatus.APPROVED:
                stats["approved"] += 1
            elif request.status == ApprovalStatus.REJECTED:
                stats["rejected"] += 1
            elif request.status == ApprovalStatus.MODIFIED:
                stats["modified"] += 1
            elif request.status == ApprovalStatus.TIMEOUT:
                stats["timeout"] += 1
        
        return stats
```

This basic approval gate provides the foundation. Let's see how to integrate it into an agent.

## Integrating Approval Gates with Agents

Now let's build an agent that uses approval gates for sensitive operations:

```python
"""
Agent with integrated approval gates.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass
from typing import Any, Optional

load_dotenv()

# Import our approval gate (shown above)
# from approval_gate import ApprovalGate, ApprovalStatus


@dataclass
class ActionConfig:
    """Configuration for an action that may require approval."""
    requires_approval: bool = False
    risk_level: str = "low"
    approval_message: str = ""


class ApprovalAwareAgent:
    """
    An agent that requests human approval for sensitive actions.
    """
    
    # Define which actions require approval and their risk levels
    ACTION_CONFIGS = {
        "send_email": ActionConfig(
            requires_approval=True,
            risk_level="high",
            approval_message="Send email to recipients"
        ),
        "delete_file": ActionConfig(
            requires_approval=True,
            risk_level="critical",
            approval_message="Permanently delete file"
        ),
        "create_file": ActionConfig(
            requires_approval=False,
            risk_level="low",
            approval_message="Create new file"
        ),
        "search_web": ActionConfig(
            requires_approval=False,
            risk_level="low",
            approval_message="Search the web"
        ),
        "make_purchase": ActionConfig(
            requires_approval=True,
            risk_level="critical",
            approval_message="Make financial transaction"
        ),
        "update_database": ActionConfig(
            requires_approval=True,
            risk_level="high",
            approval_message="Modify database records"
        ),
    }
    
    def __init__(self, approval_gate: Optional['ApprovalGate'] = None):
        """
        Initialize the agent with an approval gate.
        
        Args:
            approval_gate: The approval gate to use. If None, creates one.
        """
        self.client = anthropic.Anthropic()
        self.approval_gate = approval_gate or ApprovalGate()
        self.conversation_history: list[dict] = []
    
    def _get_tools(self) -> list[dict]:
        """Define tools available to the agent."""
        return [
            {
                "name": "send_email",
                "description": "Send an email to specified recipients. REQUIRES HUMAN APPROVAL.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of recipient email addresses"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file from the system. REQUIRES HUMAN APPROVAL. This action is irreversible.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to delete"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for deletion"
                        }
                    },
                    "required": ["filepath", "reason"]
                }
            },
            {
                "name": "create_file",
                "description": "Create a new file with specified content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path where file should be created"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            },
            {
                "name": "search_web",
                "description": "Search the web for information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def _execute_with_approval(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a tool, requesting approval if necessary.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
        
        Returns:
            Result of the tool execution or rejection info
        """
        config = self.ACTION_CONFIGS.get(
            tool_name,
            ActionConfig()  # Default: no approval needed
        )
        
        if config.requires_approval:
            # Request approval
            request = self.approval_gate.request_approval(
                action_type=tool_name,
                description=config.approval_message,
                details=tool_input,
                risk_level=config.risk_level
            )
            
            # Wait for human decision
            request = self.approval_gate.wait_for_approval(request)
            
            if request.status == ApprovalStatus.REJECTED:
                return {
                    "status": "rejected",
                    "reason": request.review_notes,
                    "message": f"Action '{tool_name}' was rejected by human reviewer."
                }
            elif request.status == ApprovalStatus.MODIFIED:
                # Use modified details if provided
                if request.modified_details:
                    tool_input = request.modified_details
        
        # Execute the tool
        return self._execute_tool(tool_name, tool_input)
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Actually execute the tool (simulation for demonstration).
        
        In a real system, these would perform actual operations.
        """
        if tool_name == "send_email":
            # Simulate sending email
            return {
                "status": "success",
                "message": f"Email sent to {', '.join(tool_input['to'])}",
                "subject": tool_input["subject"]
            }
        
        elif tool_name == "delete_file":
            # Simulate file deletion
            return {
                "status": "success",
                "message": f"File '{tool_input['filepath']}' deleted",
                "reason": tool_input["reason"]
            }
        
        elif tool_name == "create_file":
            # Simulate file creation
            return {
                "status": "success",
                "message": f"File '{tool_input['filepath']}' created",
                "size": len(tool_input["content"])
            }
        
        elif tool_name == "search_web":
            # Simulate web search
            return {
                "status": "success",
                "query": tool_input["query"],
                "results": [
                    "Result 1: Example search result",
                    "Result 2: Another relevant page",
                    "Result 3: More information"
                ]
            }
        
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def run(self, user_message: str) -> str:
        """
        Process a user message with the agent.
        
        Args:
            user_message: The user's input
        
        Returns:
            The agent's final response
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        system_prompt = """You are a helpful assistant with access to various tools.
        
Some tools require human approval before execution (marked in their descriptions).
When an action is rejected, acknowledge this gracefully and suggest alternatives.
When an action is approved, confirm what was done.

Always explain what you're about to do before using a tool."""
        
        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=self._get_tools(),
                messages=self.conversation_history
            )
            
            # Check if we're done (no more tool use)
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_response = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_response += block.text
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                return final_response
            
            # Process tool uses
            assistant_content = response.content
            tool_results = []
            
            for block in assistant_content:
                if block.type == "tool_use":
                    print(f"\n🔧 Agent wants to use tool: {block.name}")
                    
                    # Execute with approval check
                    result = self._execute_with_approval(
                        block.name,
                        block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            # Add assistant message and tool results to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
```

## Confirmation Patterns for High-Stakes Actions

Sometimes you want confirmation inline rather than a full approval workflow. Let's implement several confirmation patterns:

### Pattern 1: Simple Confirmation

```python
"""
Confirmation patterns for high-stakes actions.

Chapter 31: Human-in-the-Loop
"""

from typing import Any, Callable, TypeVar, Optional
from functools import wraps
from dataclasses import dataclass

T = TypeVar('T')


def requires_confirmation(
    message: str = "Are you sure you want to proceed?"
) -> Callable:
    """
    Decorator that requires user confirmation before executing a function.
    
    Args:
        message: The confirmation message to display
    
    Usage:
        @requires_confirmation("This will delete all data. Continue?")
        def delete_all_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            print(f"\n⚠️  {message}")
            response = input("Type 'yes' to confirm: ").strip().lower()
            
            if response == 'yes':
                return func(*args, **kwargs)
            else:
                print("❌ Action cancelled.")
                return None
        
        return wrapper
    return decorator


# Example usage
@requires_confirmation("This will send emails to all subscribers.")
def send_newsletter(subject: str, content: str) -> dict:
    """Send newsletter to all subscribers."""
    return {
        "status": "sent",
        "subject": subject,
        "recipients": 1500
    }
```

### Pattern 2: Tiered Confirmation

Different risk levels require different confirmation intensities:

```python
"""
Tiered confirmation based on risk level.
"""

from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConfirmationResult:
    """Result of a confirmation request."""
    confirmed: bool
    method: str
    notes: Optional[str] = None


class TieredConfirmation:
    """
    Implements different confirmation requirements based on risk level.
    """
    
    def __init__(self):
        self.confirmation_methods = {
            RiskLevel.LOW: self._confirm_low,
            RiskLevel.MEDIUM: self._confirm_medium,
            RiskLevel.HIGH: self._confirm_high,
            RiskLevel.CRITICAL: self._confirm_critical,
        }
    
    def request_confirmation(
        self,
        action: str,
        details: dict[str, Any],
        risk_level: RiskLevel
    ) -> ConfirmationResult:
        """
        Request confirmation appropriate to the risk level.
        
        Args:
            action: Description of the action
            details: Action details
            risk_level: How risky the action is
        
        Returns:
            ConfirmationResult indicating if action was confirmed
        """
        method = self.confirmation_methods.get(
            risk_level,
            self._confirm_medium
        )
        return method(action, details)
    
    def _confirm_low(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """Low risk: Single key press."""
        print(f"\n📋 Action: {action}")
        response = input("Press Enter to continue, or 'n' to cancel: ")
        
        return ConfirmationResult(
            confirmed=(response.lower() != 'n'),
            method="single_key"
        )
    
    def _confirm_medium(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """Medium risk: Type 'yes'."""
        print(f"\n📋 Action: {action}")
        print("Details:", details)
        response = input("Type 'yes' to confirm: ").strip().lower()
        
        return ConfirmationResult(
            confirmed=(response == 'yes'),
            method="type_yes"
        )
    
    def _confirm_high(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """High risk: Type a specific phrase."""
        print(f"\n⚠️  HIGH RISK Action: {action}")
        print("Details:", details)
        
        # Generate a confirmation phrase
        phrase = "I CONFIRM THIS ACTION"
        print(f"\nType exactly: {phrase}")
        response = input("> ").strip()
        
        return ConfirmationResult(
            confirmed=(response == phrase),
            method="type_phrase"
        )
    
    def _confirm_critical(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """Critical risk: Multiple confirmations and reason."""
        print(f"\n🚨 CRITICAL Action: {action}")
        print("Details:", details)
        
        # First confirmation
        print("\n⚠️  This action has serious consequences.")
        response1 = input("Type 'I UNDERSTAND' to proceed: ").strip()
        if response1 != "I UNDERSTAND":
            return ConfirmationResult(confirmed=False, method="critical_multi")
        
        # Require reason
        reason = input("Explain why this action is necessary: ").strip()
        if len(reason) < 10:
            print("Please provide a more detailed reason.")
            return ConfirmationResult(confirmed=False, method="critical_multi")
        
        # Final confirmation
        response2 = input("Final confirmation - type 'EXECUTE': ").strip()
        
        return ConfirmationResult(
            confirmed=(response2 == "EXECUTE"),
            method="critical_multi",
            notes=reason
        )
```

### Pattern 3: Preview Before Execute

Show exactly what will happen before doing it:

```python
"""
Preview pattern - show what will happen before execution.
"""

from abc import ABC, abstractmethod


class PreviewableAction(ABC):
    """Base class for actions that can be previewed."""
    
    @abstractmethod
    def preview(self) -> str:
        """Generate a preview of what this action will do."""
        pass
    
    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Execute the action."""
        pass
    
    def preview_and_confirm(self) -> Optional[dict[str, Any]]:
        """Show preview and get confirmation before executing."""
        print("\n" + "=" * 50)
        print("📝 ACTION PREVIEW")
        print("=" * 50)
        print(self.preview())
        print("=" * 50)
        
        response = input("\nExecute this action? (yes/no): ").strip().lower()
        
        if response == 'yes':
            print("\n⏳ Executing...")
            result = self.execute()
            print("✅ Done!")
            return result
        else:
            print("❌ Cancelled.")
            return None


class EmailAction(PreviewableAction):
    """Email action with preview capability."""
    
    def __init__(
        self,
        recipients: list[str],
        subject: str,
        body: str
    ):
        self.recipients = recipients
        self.subject = subject
        self.body = body
    
    def preview(self) -> str:
        """Preview the email that will be sent."""
        preview_lines = [
            f"TO: {', '.join(self.recipients)}",
            f"SUBJECT: {self.subject}",
            "",
            "BODY:",
            "-" * 40,
            self.body[:500] + ("..." if len(self.body) > 500 else ""),
            "-" * 40,
            "",
            f"📨 This will send to {len(self.recipients)} recipient(s)"
        ]
        return "\n".join(preview_lines)
    
    def execute(self) -> dict[str, Any]:
        """Send the email."""
        # Simulate sending
        return {
            "status": "sent",
            "recipients": len(self.recipients),
            "subject": self.subject
        }


class DatabaseUpdateAction(PreviewableAction):
    """Database update with preview capability."""
    
    def __init__(
        self,
        table: str,
        updates: dict[str, Any],
        where_clause: str,
        affected_rows: int
    ):
        self.table = table
        self.updates = updates
        self.where_clause = where_clause
        self.affected_rows = affected_rows
    
    def preview(self) -> str:
        """Preview the database update."""
        update_str = ", ".join(
            f"{k} = {repr(v)}" for k, v in self.updates.items()
        )
        
        preview_lines = [
            f"TABLE: {self.table}",
            f"UPDATE: {update_str}",
            f"WHERE: {self.where_clause}",
            "",
            f"⚠️  This will modify {self.affected_rows} row(s)"
        ]
        return "\n".join(preview_lines)
    
    def execute(self) -> dict[str, Any]:
        """Execute the update."""
        # Simulate update
        return {
            "status": "updated",
            "table": self.table,
            "rows_affected": self.affected_rows
        }
```

## Human Feedback Integration

Beyond simple approval/rejection, agents can learn from human feedback to improve over time. Let's implement a feedback collection system:

```python
"""
Human feedback integration for agent improvement.

Chapter 31: Human-in-the-Loop
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import json


class FeedbackType(Enum):
    """Types of feedback humans can provide."""
    RATING = "rating"           # 1-5 star rating
    CORRECTION = "correction"   # Corrected output
    PREFERENCE = "preference"   # A vs B preference
    FLAG = "flag"               # Flag problematic output
    COMMENT = "comment"         # Free-form comment


@dataclass
class Feedback:
    """A single piece of human feedback."""
    feedback_id: str
    feedback_type: FeedbackType
    context: dict[str, Any]      # What the agent was doing
    agent_output: str            # What the agent produced
    human_input: Any             # The feedback itself
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackCollector:
    """
    Collects and stores human feedback on agent outputs.
    
    In production, this would persist to a database and potentially
    feed into fine-tuning or prompt improvement pipelines.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the feedback collector.
        
        Args:
            storage_path: Path to store feedback JSON (optional)
        """
        self.feedback_log: list[Feedback] = []
        self.storage_path = storage_path
        self._feedback_counter = 0
    
    def _generate_id(self) -> str:
        """Generate a unique feedback ID."""
        self._feedback_counter += 1
        return f"FB-{self._feedback_counter:04d}"
    
    def collect_rating(
        self,
        context: dict[str, Any],
        agent_output: str,
        prompt: str = "Rate this response (1-5 stars): "
    ) -> Feedback:
        """
        Collect a star rating for an agent output.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            prompt: The prompt to show the user
        
        Returns:
            Feedback object with the rating
        """
        print(f"\n🤖 Agent output:\n{agent_output[:500]}...")
        
        while True:
            try:
                rating = int(input(prompt))
                if 1 <= rating <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.RATING,
            context=context,
            agent_output=agent_output,
            human_input=rating
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def collect_correction(
        self,
        context: dict[str, Any],
        agent_output: str
    ) -> Feedback:
        """
        Collect a corrected version of the agent's output.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
        
        Returns:
            Feedback object with the correction
        """
        print(f"\n🤖 Agent output:\n{agent_output}")
        print("\n" + "-" * 40)
        
        print("Provide the corrected version (or press Enter to skip):")
        correction = input("> ").strip()
        
        if not correction:
            correction = None
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.CORRECTION,
            context=context,
            agent_output=agent_output,
            human_input=correction
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def collect_preference(
        self,
        context: dict[str, Any],
        option_a: str,
        option_b: str
    ) -> Feedback:
        """
        Collect A/B preference between two options.
        
        Args:
            context: The context of the interaction
            option_a: First option
            option_b: Second option
        
        Returns:
            Feedback object with the preference
        """
        print("\n📋 Which response is better?")
        print("\n[A]:")
        print(option_a[:300] + ("..." if len(option_a) > 300 else ""))
        print("\n[B]:")
        print(option_b[:300] + ("..." if len(option_b) > 300 else ""))
        
        while True:
            choice = input("\nPrefer A, B, or Neither? ").strip().upper()
            if choice in ['A', 'B', 'NEITHER', 'N']:
                break
            print("Please enter A, B, or Neither")
        
        preference = choice if choice != 'N' else 'NEITHER'
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.PREFERENCE,
            context=context,
            agent_output=json.dumps({"A": option_a, "B": option_b}),
            human_input=preference
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def collect_flag(
        self,
        context: dict[str, Any],
        agent_output: str,
        flag_categories: list[str] = None
    ) -> Feedback:
        """
        Flag problematic output with categorization.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            flag_categories: Available flag categories
        
        Returns:
            Feedback object with the flag
        """
        if flag_categories is None:
            flag_categories = [
                "incorrect",
                "inappropriate",
                "unhelpful",
                "off_topic",
                "other"
            ]
        
        print(f"\n🤖 Agent output:\n{agent_output[:500]}...")
        print("\n🚩 Flag this output? Categories:")
        
        for i, category in enumerate(flag_categories, 1):
            print(f"  {i}. {category}")
        print(f"  0. Don't flag")
        
        while True:
            try:
                choice = int(input("\nSelect category (0 to skip): "))
                if 0 <= choice <= len(flag_categories):
                    break
                print(f"Please enter 0-{len(flag_categories)}")
            except ValueError:
                print("Please enter a valid number.")
        
        if choice == 0:
            flag_info = None
        else:
            category = flag_categories[choice - 1]
            reason = input("Briefly explain the issue: ").strip()
            flag_info = {"category": category, "reason": reason}
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.FLAG,
            context=context,
            agent_output=agent_output,
            human_input=flag_info
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def _store_feedback(self, feedback: Feedback) -> None:
        """Store feedback in memory and optionally to file."""
        self.feedback_log.append(feedback)
        
        if self.storage_path:
            self._persist_to_file()
    
    def _persist_to_file(self) -> None:
        """Save all feedback to JSON file."""
        data = []
        for fb in self.feedback_log:
            data.append({
                "feedback_id": fb.feedback_id,
                "feedback_type": fb.feedback_type.value,
                "context": fb.context,
                "agent_output": fb.agent_output,
                "human_input": fb.human_input,
                "timestamp": fb.timestamp.isoformat(),
                "metadata": fb.metadata
            })
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected feedback."""
        summary = {
            "total_feedback": len(self.feedback_log),
            "by_type": {},
            "average_rating": None,
            "flag_categories": {}
        }
        
        ratings = []
        
        for fb in self.feedback_log:
            # Count by type
            type_name = fb.feedback_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Collect ratings
            if fb.feedback_type == FeedbackType.RATING:
                ratings.append(fb.human_input)
            
            # Collect flag categories
            if fb.feedback_type == FeedbackType.FLAG and fb.human_input:
                category = fb.human_input.get("category", "unknown")
                summary["flag_categories"][category] = (
                    summary["flag_categories"].get(category, 0) + 1
                )
        
        if ratings:
            summary["average_rating"] = sum(ratings) / len(ratings)
        
        return summary
```

## Escalation Paths

Sometimes an agent encounters situations it cannot or should not handle autonomously. Escalation paths ensure these situations reach the right humans:

```python
"""
Escalation system for situations beyond agent capabilities.

Chapter 31: Human-in-the-Loop
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from enum import Enum


class EscalationReason(Enum):
    """Reasons an agent might escalate to humans."""
    UNCERTAINTY = "uncertainty"         # Agent is unsure
    POLICY_VIOLATION = "policy"         # Potential policy issue
    SENSITIVE_TOPIC = "sensitive"       # Topic requires human handling
    CUSTOMER_REQUEST = "customer_request"  # Customer asked for human
    ERROR_THRESHOLD = "errors"          # Too many errors
    COMPLEXITY = "complexity"           # Task too complex
    AUTHORITY = "authority"             # Requires human authority


class EscalationPriority(Enum):
    """Priority levels for escalations."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class Escalation:
    """Represents an escalation to human handlers."""
    escalation_id: str
    reason: EscalationReason
    priority: EscalationPriority
    summary: str
    context: dict[str, Any]
    conversation_history: list[dict]
    created_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None


class EscalationManager:
    """
    Manages escalations from agent to human handlers.
    
    In production, this would integrate with ticketing systems,
    on-call rotations, Slack/Teams, etc.
    """
    
    def __init__(self):
        self.escalations: list[Escalation] = []
        self.handlers: dict[EscalationReason, list[str]] = {
            EscalationReason.UNCERTAINTY: ["support_team"],
            EscalationReason.POLICY_VIOLATION: ["compliance_team", "legal"],
            EscalationReason.SENSITIVE_TOPIC: ["senior_support"],
            EscalationReason.CUSTOMER_REQUEST: ["support_team"],
            EscalationReason.ERROR_THRESHOLD: ["engineering"],
            EscalationReason.COMPLEXITY: ["senior_support", "specialists"],
            EscalationReason.AUTHORITY: ["management"],
        }
        self._escalation_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique escalation ID."""
        self._escalation_counter += 1
        return f"ESC-{self._escalation_counter:04d}"
    
    def escalate(
        self,
        reason: EscalationReason,
        summary: str,
        context: dict[str, Any],
        conversation_history: list[dict],
        priority: EscalationPriority = EscalationPriority.MEDIUM
    ) -> Escalation:
        """
        Create a new escalation.
        
        Args:
            reason: Why the agent is escalating
            summary: Brief summary of the situation
            context: Relevant context data
            conversation_history: The conversation so far
            priority: How urgent is this escalation
        
        Returns:
            The created Escalation object
        """
        escalation = Escalation(
            escalation_id=self._generate_id(),
            reason=reason,
            priority=priority,
            summary=summary,
            context=context,
            conversation_history=conversation_history
        )
        
        self.escalations.append(escalation)
        
        # Notify appropriate handlers
        self._notify_handlers(escalation)
        
        return escalation
    
    def _notify_handlers(self, escalation: Escalation) -> None:
        """
        Notify appropriate human handlers about the escalation.
        
        In production, this would send notifications via various channels.
        """
        handlers = self.handlers.get(escalation.reason, ["support_team"])
        
        print("\n" + "🚨" * 20)
        print("ESCALATION CREATED")
        print("🚨" * 20)
        print(f"\nID: {escalation.escalation_id}")
        print(f"Priority: {escalation.priority.name}")
        print(f"Reason: {escalation.reason.value}")
        print(f"Summary: {escalation.summary}")
        print(f"\nNotifying: {', '.join(handlers)}")
        print("\nContext:")
        for key, value in escalation.context.items():
            print(f"  • {key}: {value}")
        print("🚨" * 20 + "\n")
    
    def get_pending_escalations(
        self,
        handler: Optional[str] = None,
        min_priority: EscalationPriority = EscalationPriority.LOW
    ) -> list[Escalation]:
        """Get unresolved escalations, optionally filtered."""
        pending = [e for e in self.escalations if not e.resolved]
        
        if min_priority:
            pending = [
                e for e in pending
                if e.priority.value >= min_priority.value
            ]
        
        if handler:
            pending = [
                e for e in pending
                if handler in self.handlers.get(e.reason, [])
            ]
        
        return sorted(pending, key=lambda e: -e.priority.value)
    
    def resolve_escalation(
        self,
        escalation_id: str,
        resolution_notes: str,
        resolved_by: str
    ) -> Optional[Escalation]:
        """Mark an escalation as resolved."""
        for escalation in self.escalations:
            if escalation.escalation_id == escalation_id:
                escalation.resolved = True
                escalation.resolution_notes = resolution_notes
                escalation.assigned_to = resolved_by
                return escalation
        
        return None


class EscalationAwareAgent:
    """
    An agent that knows when to escalate to humans.
    """
    
    # Triggers that indicate escalation might be needed
    ESCALATION_TRIGGERS = {
        "i want to speak to a human": EscalationReason.CUSTOMER_REQUEST,
        "talk to a person": EscalationReason.CUSTOMER_REQUEST,
        "get me a manager": EscalationReason.CUSTOMER_REQUEST,
        "this is urgent": EscalationReason.AUTHORITY,
        "legal matter": EscalationReason.POLICY_VIOLATION,
        "sue": EscalationReason.POLICY_VIOLATION,
        "lawyer": EscalationReason.POLICY_VIOLATION,
    }
    
    # Topics that should always be escalated
    SENSITIVE_TOPICS = [
        "self-harm",
        "suicide",
        "abuse",
        "emergency",
        "medical emergency",
    ]
    
    def __init__(self, escalation_manager: EscalationManager):
        self.escalation_manager = escalation_manager
        self.error_count = 0
        self.error_threshold = 3
    
    def should_escalate(
        self,
        user_message: str,
        agent_confidence: float = 1.0
    ) -> Optional[tuple[EscalationReason, EscalationPriority]]:
        """
        Determine if the current situation requires escalation.
        
        Args:
            user_message: The user's message
            agent_confidence: Agent's confidence in handling (0-1)
        
        Returns:
            Tuple of (reason, priority) if escalation needed, else None
        """
        message_lower = user_message.lower()
        
        # Check for explicit escalation triggers
        for trigger, reason in self.ESCALATION_TRIGGERS.items():
            if trigger in message_lower:
                priority = EscalationPriority.HIGH
                if "urgent" in message_lower or "emergency" in message_lower:
                    priority = EscalationPriority.URGENT
                return (reason, priority)
        
        # Check for sensitive topics
        for topic in self.SENSITIVE_TOPICS:
            if topic in message_lower:
                return (
                    EscalationReason.SENSITIVE_TOPIC,
                    EscalationPriority.URGENT
                )
        
        # Check confidence threshold
        if agent_confidence < 0.5:
            return (
                EscalationReason.UNCERTAINTY,
                EscalationPriority.MEDIUM
            )
        
        # Check error threshold
        if self.error_count >= self.error_threshold:
            return (
                EscalationReason.ERROR_THRESHOLD,
                EscalationPriority.HIGH
            )
        
        return None
    
    def record_error(self) -> None:
        """Record that an error occurred."""
        self.error_count += 1
    
    def reset_errors(self) -> None:
        """Reset the error counter."""
        self.error_count = 0
```

## Complete Human-in-the-Loop Agent

Let's put everything together into a complete agent with all HITL capabilities:

```python
"""
Complete Human-in-the-Loop Agent

Combines approval gates, confirmation patterns, feedback collection,
and escalation handling into a production-ready agent.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

load_dotenv()

# Import all our HITL components
# In practice, these would be in separate files
# from approval_gate import ApprovalGate, ApprovalStatus, ApprovalRequest
# from feedback import FeedbackCollector, FeedbackType
# from escalation import EscalationManager, EscalationReason, EscalationPriority


@dataclass
class HITLConfig:
    """Configuration for human-in-the-loop behavior."""
    
    # Approval settings
    require_approval_for_high_risk: bool = True
    auto_approve_low_risk: bool = True
    
    # Feedback settings
    collect_feedback: bool = True
    feedback_frequency: int = 5  # Collect every N interactions
    
    # Escalation settings
    enable_escalation: bool = True
    confidence_threshold: float = 0.7
    error_threshold: int = 3
    
    # Confirmation settings
    preview_before_execute: bool = True


class HumanInTheLoopAgent:
    """
    A complete agent with human-in-the-loop capabilities.
    
    Features:
    - Approval gates for sensitive actions
    - Confirmation flows with previews
    - Feedback collection
    - Escalation to humans when needed
    """
    
    def __init__(self, config: Optional[HITLConfig] = None):
        """
        Initialize the HITL agent.
        
        Args:
            config: Configuration for HITL behavior
        """
        self.config = config or HITLConfig()
        self.client = anthropic.Anthropic()
        
        # Initialize HITL components
        self.approval_gate = ApprovalGate(
            auto_approve_low_risk=self.config.auto_approve_low_risk
        )
        self.feedback_collector = FeedbackCollector()
        self.escalation_manager = EscalationManager()
        
        # State
        self.conversation_history: list[dict] = []
        self.interaction_count = 0
        self.error_count = 0
        self.is_escalated = False
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt for the agent."""
        return """You are a helpful assistant with human-in-the-loop capabilities.

IMPORTANT GUIDELINES:

1. SENSITIVE ACTIONS: Some actions require human approval. Always inform the
   user when an action needs approval and wait for the result.

2. UNCERTAINTY: If you're unsure about something, say so. It's better to
   ask for clarification or escalate than to guess wrong.

3. ESCALATION TRIGGERS: If the user asks to speak to a human, or if the
   conversation involves sensitive topics, acknowledge this and indicate
   that you'll escalate to a human handler.

4. TRANSPARENCY: Always be clear about what you're doing, what requires
   approval, and when you're waiting for human input.

5. FEEDBACK: Accept feedback gracefully and use it to improve your responses.

You have access to tools that may require approval. The tool descriptions
indicate which ones need human approval before execution."""
    
    def _get_tools(self) -> list[dict]:
        """Define available tools with risk indicators."""
        return [
            {
                "name": "send_message",
                "description": "Send a message to a user or external system. "
                              "RISK: HIGH - Requires approval for external messages.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "message": {"type": "string"},
                        "channel": {
                            "type": "string",
                            "enum": ["email", "sms", "slack"]
                        }
                    },
                    "required": ["recipient", "message", "channel"]
                }
            },
            {
                "name": "search_database",
                "description": "Search the internal database. RISK: LOW - Read-only operation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "table": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "modify_record",
                "description": "Create, update, or delete a database record. "
                              "RISK: CRITICAL - Requires approval for all modifications.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["create", "update", "delete"]
                        },
                        "table": {"type": "string"},
                        "record_id": {"type": "string"},
                        "data": {"type": "object"}
                    },
                    "required": ["operation", "table"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather information. RISK: LOW - Read-only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    
    def _assess_risk(self, tool_name: str, tool_input: dict) -> str:
        """Assess the risk level of a tool call."""
        high_risk_tools = {"send_message"}
        critical_risk_tools = {"modify_record"}
        
        if tool_name in critical_risk_tools:
            return "critical"
        elif tool_name in high_risk_tools:
            return "high"
        else:
            return "low"
    
    def _execute_tool_with_hitl(
        self,
        tool_name: str,
        tool_input: dict
    ) -> dict[str, Any]:
        """Execute a tool with appropriate HITL controls."""
        risk_level = self._assess_risk(tool_name, tool_input)
        
        # Check if approval is needed
        needs_approval = (
            self.config.require_approval_for_high_risk and
            risk_level in ["high", "critical"]
        )
        
        if needs_approval:
            # Request approval
            request = self.approval_gate.request_approval(
                action_type=tool_name,
                description=f"Execute {tool_name}",
                details=tool_input,
                risk_level=risk_level
            )
            
            request = self.approval_gate.wait_for_approval(request)
            
            if request.status == ApprovalStatus.REJECTED:
                return {
                    "status": "rejected",
                    "reason": request.review_notes,
                    "message": "Action was rejected by human reviewer"
                }
            elif request.status == ApprovalStatus.MODIFIED:
                if request.modified_details:
                    tool_input = request.modified_details
        
        # Execute the tool (simulation)
        return self._simulate_tool_execution(tool_name, tool_input)
    
    def _simulate_tool_execution(
        self,
        tool_name: str,
        tool_input: dict
    ) -> dict[str, Any]:
        """Simulate tool execution for demonstration."""
        if tool_name == "send_message":
            return {
                "status": "sent",
                "recipient": tool_input["recipient"],
                "channel": tool_input["channel"],
                "timestamp": datetime.now().isoformat()
            }
        elif tool_name == "search_database":
            return {
                "status": "success",
                "results": [
                    {"id": "1", "name": "Sample Result 1"},
                    {"id": "2", "name": "Sample Result 2"}
                ],
                "count": 2
            }
        elif tool_name == "modify_record":
            return {
                "status": "success",
                "operation": tool_input["operation"],
                "table": tool_input["table"],
                "affected_rows": 1
            }
        elif tool_name == "get_weather":
            return {
                "status": "success",
                "location": tool_input["location"],
                "temperature": "72°F",
                "conditions": "Sunny"
            }
        
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    
    def _check_escalation(self, user_message: str) -> bool:
        """Check if we should escalate to a human."""
        if not self.config.enable_escalation:
            return False
        
        # Check for explicit requests
        escalation_phrases = [
            "speak to a human",
            "talk to a person",
            "get me a manager",
            "human support",
            "real person"
        ]
        
        message_lower = user_message.lower()
        for phrase in escalation_phrases:
            if phrase in message_lower:
                return True
        
        # Check error threshold
        if self.error_count >= self.config.error_threshold:
            return True
        
        return False
    
    def _handle_escalation(self, user_message: str) -> str:
        """Handle escalation to human support."""
        self.is_escalated = True
        
        # Determine reason and priority
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["urgent", "emergency"]):
            priority = EscalationPriority.URGENT
        else:
            priority = EscalationPriority.HIGH
        
        if self.error_count >= self.config.error_threshold:
            reason = EscalationReason.ERROR_THRESHOLD
        else:
            reason = EscalationReason.CUSTOMER_REQUEST
        
        # Create escalation
        escalation = self.escalation_manager.escalate(
            reason=reason,
            summary=f"User requested human support. Last message: {user_message[:100]}",
            context={
                "user_message": user_message,
                "interaction_count": self.interaction_count,
                "error_count": self.error_count
            },
            conversation_history=self.conversation_history,
            priority=priority
        )
        
        return f"""I understand you'd like to speak with a human. I've escalated 
your request to our support team.

**Escalation ID:** {escalation.escalation_id}
**Priority:** {escalation.priority.name}

A team member will be with you shortly. In the meantime, is there anything 
simple I can help you with?"""
    
    def _maybe_collect_feedback(self, response: str) -> None:
        """Optionally collect feedback on the response."""
        if not self.config.collect_feedback:
            return
        
        if self.interaction_count % self.config.feedback_frequency == 0:
            print("\n" + "-" * 40)
            print("📊 Quick feedback (optional)")
            
            self.feedback_collector.collect_rating(
                context={"interaction": self.interaction_count},
                agent_output=response,
                prompt="Rate this response (1-5, or press Enter to skip): "
            )
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_message: The user's input
        
        Returns:
            The agent's response
        """
        self.interaction_count += 1
        
        # Check for escalation
        if self._check_escalation(user_message):
            return self._handle_escalation(user_message)
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Run the agentic loop
            while True:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=self._get_system_prompt(),
                    tools=self._get_tools(),
                    messages=self.conversation_history
                )
                
                # Check if done
                if response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    # Maybe collect feedback
                    self._maybe_collect_feedback(final_text)
                    
                    # Reset error count on success
                    self.error_count = 0
                    
                    return final_text
                
                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"\n🔧 Using tool: {block.name}")
                        result = self._execute_tool_with_hitl(
                            block.name,
                            block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
        
        except Exception as e:
            self.error_count += 1
            return f"I encountered an error: {str(e)}. Let me know if you'd like to try again."
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about agent operation."""
        return {
            "interactions": self.interaction_count,
            "errors": self.error_count,
            "is_escalated": self.is_escalated,
            "approval_stats": self.approval_gate.get_approval_stats(),
            "feedback_summary": self.feedback_collector.get_summary(),
            "pending_escalations": len(
                self.escalation_manager.get_pending_escalations()
            )
        }


def main():
    """Demonstrate the Human-in-the-Loop agent."""
    print("=" * 60)
    print("Human-in-the-Loop Agent Demo")
    print("=" * 60)
    print("\nThis agent demonstrates:")
    print("• Approval gates for sensitive actions")
    print("• Escalation when you ask for a human")
    print("• Periodic feedback collection")
    print("\nTry asking it to:")
    print("• Send an email (requires approval)")
    print("• Check the weather (no approval needed)")
    print("• 'I want to speak to a human' (triggers escalation)")
    print("\nType 'quit' to exit, 'stats' for statistics\n")
    
    config = HITLConfig(
        require_approval_for_high_risk=True,
        auto_approve_low_risk=True,
        collect_feedback=True,
        feedback_frequency=3,
        enable_escalation=True
    )
    
    agent = HumanInTheLoopAgent(config)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        elif user_input.lower() == 'stats':
            stats = agent.get_stats()
            print("\n📊 Agent Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue
        elif not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}")


if __name__ == "__main__":
    main()
```

## Common Pitfalls

### 1. Approval Fatigue

**Problem:** Too many approval requests cause users to approve everything without reading.

**Solution:** 
- Use risk-based tiering—only require approval for genuinely sensitive actions
- Auto-approve low-risk actions
- Batch related approvals when possible
- Track approval patterns and adjust thresholds

### 2. Blocking on Approval Indefinitely

**Problem:** Agent blocks forever waiting for approval that never comes.

**Solution:**
- Implement timeouts for approval requests
- Provide clear status to users about pending approvals
- Allow agents to continue with other tasks while waiting
- Send reminders for pending approvals

### 3. Escalation Black Holes

**Problem:** Escalations go to a queue that no one monitors.

**Solution:**
- Implement priority-based routing
- Set up alerts for high-priority escalations
- Track escalation response times
- Have fallback handlers for different scenarios

## Practical Exercise

**Task:** Build an approval-aware file management agent

Create an agent that can perform file operations with appropriate human oversight:

**Requirements:**

1. **Read operations** (list files, read content): No approval needed
2. **Create operations** (create file, create directory): Low-risk confirmation
3. **Modify operations** (rename, move): Medium-risk approval
4. **Delete operations**: High-risk approval with preview
5. **Bulk operations** (delete multiple files): Critical-risk with detailed confirmation

**Additional Requirements:**
- Track all operations in an audit log
- Allow users to request escalation at any time
- Collect feedback after significant operations

**Hints:**
- Use the `PreviewableAction` pattern for delete operations
- Implement an `AuditLog` class to track all file operations
- Consider implementing an "undo" capability for reversible operations

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Approval gates** pause agent execution for human review of sensitive actions
- **Tiered confirmation** matches the intensity of confirmation to the risk level
- **Preview patterns** show users exactly what will happen before execution
- **Feedback collection** helps improve agent behavior over time
- **Escalation paths** ensure situations beyond agent capabilities reach humans
- **Configuration is key**—make HITL behavior adjustable, not hardcoded
- **Autonomy should be earned** through demonstrated reliability

## What's Next

In Chapter 32, we'll explore **Guardrails and Safety**—building systematic protections that prevent agents from going off the rails. While human-in-the-loop keeps humans informed and in control, guardrails provide automatic protection even when humans aren't actively monitoring.
