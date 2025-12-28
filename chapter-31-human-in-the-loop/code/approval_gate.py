"""
Approval Gate Implementation

Provides a system for requesting and managing human approval
for agent actions before execution.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

load_dotenv()


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
    
    Example:
        gate = ApprovalGate(auto_approve_low_risk=True)
        
        request = gate.request_approval(
            action_type="send_email",
            description="Send newsletter to subscribers",
            details={"recipients": 1500, "subject": "Weekly Update"},
            risk_level="high"
        )
        
        result = gate.wait_for_approval(request)
        
        if result.status == ApprovalStatus.APPROVED:
            # Execute the action
            send_newsletter()
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
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"REQ-{timestamp}-{self._request_counter:04d}"
    
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
        if risk_level not in ["low", "medium", "high", "critical"]:
            raise ValueError(
                f"Invalid risk_level: {risk_level}. "
                "Must be 'low', 'medium', 'high', or 'critical'"
            )
        
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
                            Note: timeout not implemented in this demo
        
        Returns:
            Updated ApprovalRequest with final status
        """
        # Auto-approve low risk if configured
        if self.auto_approve_low_risk and request.risk_level == "low":
            request.status = ApprovalStatus.APPROVED
            request.review_notes = "Auto-approved (low risk)"
            request.reviewed_at = datetime.now()
            self._finalize_request(request)
            print(f"✅ Auto-approved: {request.description} (low risk)")
            return request
        
        # Display request details
        self._display_request(request)
        
        # Get human input
        while True:
            choice = input("\n[A]pprove / [R]eject / [M]odify? ").strip().upper()
            
            if choice == 'A':
                request.status = ApprovalStatus.APPROVED
                notes = input("Notes (optional, press Enter to skip): ").strip()
                request.review_notes = notes if notes else None
                break
            elif choice == 'R':
                request.status = ApprovalStatus.REJECTED
                request.review_notes = input("Reason for rejection: ").strip()
                break
            elif choice == 'M':
                request.status = ApprovalStatus.MODIFIED
                request.review_notes = input("Modification notes: ").strip()
                # In a real system, you'd collect the modified details
                modified = input("Enter modified details as key=value pairs (comma-separated): ")
                if modified.strip():
                    request.modified_details = self._parse_modifications(modified)
                break
            else:
                print("Please enter A, R, or M")
        
        request.reviewed_at = datetime.now()
        self._finalize_request(request)
        
        # Show result
        status_emoji = {
            ApprovalStatus.APPROVED: "✅",
            ApprovalStatus.REJECTED: "❌",
            ApprovalStatus.MODIFIED: "📝"
        }
        print(f"\n{status_emoji.get(request.status, '❓')} "
              f"Request {request.request_id}: {request.status.value}")
        
        return request
    
    def _display_request(self, request: ApprovalRequest) -> None:
        """Display request details for human review."""
        risk_colors = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }
        
        print("\n" + "=" * 60)
        print("🔔 APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Request ID:  {request.request_id}")
        print(f"Action Type: {request.action_type}")
        print(f"Risk Level:  {risk_colors.get(request.risk_level, '⚪')} "
              f"{request.risk_level.upper()}")
        print(f"Description: {request.description}")
        print(f"Created At:  {request.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nDetails:")
        for key, value in request.details.items():
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:100] + "..."
            print(f"  • {key}: {str_value}")
        print("=" * 60)
    
    def _parse_modifications(self, input_str: str) -> dict[str, Any]:
        """Parse modification input string into a dictionary."""
        modifications = {}
        pairs = input_str.split(',')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                modifications[key.strip()] = value.strip()
        return modifications
    
    def _finalize_request(self, request: ApprovalRequest) -> None:
        """Move request from pending to history."""
        if request.request_id in self.pending_requests:
            del self.pending_requests[request.request_id]
        self.request_history.append(request)
    
    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self.pending_requests.values())
    
    def get_request_by_id(self, request_id: str) -> Optional[ApprovalRequest]:
        """Look up a request by ID (pending or historical)."""
        if request_id in self.pending_requests:
            return self.pending_requests[request_id]
        
        for request in self.request_history:
            if request.request_id == request_id:
                return request
        
        return None
    
    def get_approval_stats(self) -> dict[str, Any]:
        """Get statistics on approval history."""
        stats = {
            "total": len(self.request_history),
            "pending": len(self.pending_requests),
            "approved": 0,
            "rejected": 0,
            "modified": 0,
            "timeout": 0,
            "by_risk_level": {
                "low": {"total": 0, "approved": 0, "rejected": 0},
                "medium": {"total": 0, "approved": 0, "rejected": 0},
                "high": {"total": 0, "approved": 0, "rejected": 0},
                "critical": {"total": 0, "approved": 0, "rejected": 0}
            }
        }
        
        for request in self.request_history:
            # Count by status
            if request.status == ApprovalStatus.APPROVED:
                stats["approved"] += 1
            elif request.status == ApprovalStatus.REJECTED:
                stats["rejected"] += 1
            elif request.status == ApprovalStatus.MODIFIED:
                stats["modified"] += 1
            elif request.status == ApprovalStatus.TIMEOUT:
                stats["timeout"] += 1
            
            # Count by risk level
            risk = request.risk_level
            if risk in stats["by_risk_level"]:
                stats["by_risk_level"][risk]["total"] += 1
                if request.status == ApprovalStatus.APPROVED:
                    stats["by_risk_level"][risk]["approved"] += 1
                elif request.status == ApprovalStatus.REJECTED:
                    stats["by_risk_level"][risk]["rejected"] += 1
        
        return stats


def main():
    """Demonstrate the approval gate."""
    print("=" * 60)
    print("Approval Gate Demo")
    print("=" * 60)
    
    # Create gate with auto-approve for low risk
    gate = ApprovalGate(auto_approve_low_risk=True)
    
    # Test low-risk action (auto-approved)
    print("\n1. Testing low-risk action (should auto-approve)...")
    request1 = gate.request_approval(
        action_type="read_file",
        description="Read configuration file",
        details={"filepath": "/config/settings.json"},
        risk_level="low"
    )
    gate.wait_for_approval(request1)
    
    # Test high-risk action (requires approval)
    print("\n2. Testing high-risk action (requires approval)...")
    request2 = gate.request_approval(
        action_type="send_email",
        description="Send promotional email to subscribers",
        details={
            "recipients": 5000,
            "subject": "Special Offer Inside!",
            "template": "promo_template_v2"
        },
        risk_level="high"
    )
    gate.wait_for_approval(request2)
    
    # Test critical action
    print("\n3. Testing critical action (requires approval)...")
    request3 = gate.request_approval(
        action_type="delete_records",
        description="Delete inactive user accounts",
        details={
            "criteria": "last_login < 2 years ago",
            "affected_users": 1250,
            "backup_created": True
        },
        risk_level="critical"
    )
    gate.wait_for_approval(request3)
    
    # Show stats
    print("\n" + "=" * 60)
    print("Approval Statistics")
    print("=" * 60)
    stats = gate.get_approval_stats()
    print(f"Total requests: {stats['total']}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Modified: {stats['modified']}")
    print("\nBy risk level:")
    for level, data in stats["by_risk_level"].items():
        if data["total"] > 0:
            print(f"  {level}: {data['total']} total, "
                  f"{data['approved']} approved, {data['rejected']} rejected")


if __name__ == "__main__":
    main()
