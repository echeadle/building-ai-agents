"""
Escalation System

Manages escalations from agents to human handlers when
situations exceed the agent's capabilities or authority.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum

load_dotenv()


class EscalationReason(Enum):
    """Reasons an agent might escalate to humans."""
    UNCERTAINTY = "uncertainty"           # Agent is unsure how to proceed
    POLICY_VIOLATION = "policy"           # Potential policy or legal issue
    SENSITIVE_TOPIC = "sensitive"         # Topic requires human handling
    CUSTOMER_REQUEST = "customer_request" # Customer explicitly asked for human
    ERROR_THRESHOLD = "errors"            # Too many errors encountered
    COMPLEXITY = "complexity"             # Task too complex for agent
    AUTHORITY = "authority"               # Requires human authority/approval
    EMOTIONAL = "emotional"               # User showing emotional distress
    SECURITY = "security"                 # Potential security concern


class EscalationPriority(Enum):
    """Priority levels for escalations."""
    LOW = 1       # Can wait, non-urgent
    MEDIUM = 2    # Should be handled soon
    HIGH = 3      # Needs attention quickly
    URGENT = 4    # Needs immediate attention
    CRITICAL = 5  # Emergency, highest priority


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
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "escalation_id": self.escalation_id,
            "reason": self.reason.value,
            "priority": self.priority.value,
            "summary": self.summary,
            "context": self.context,
            "conversation_history": self.conversation_history,
            "created_at": self.created_at.isoformat(),
            "assigned_to": self.assigned_to,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class EscalationManager:
    """
    Manages escalations from agent to human handlers.
    
    In production, this would integrate with ticketing systems
    (Zendesk, Jira), communication tools (Slack, Teams), and
    on-call rotation systems (PagerDuty, OpsGenie).
    
    Example:
        manager = EscalationManager()
        
        escalation = manager.escalate(
            reason=EscalationReason.CUSTOMER_REQUEST,
            summary="User requested human support",
            context={"user_id": "12345"},
            conversation_history=[...],
            priority=EscalationPriority.HIGH
        )
        
        # Later, resolve the escalation
        manager.resolve_escalation(
            escalation.escalation_id,
            "Resolved user's billing issue",
            "agent_smith"
        )
    """
    
    # Default handler assignments by reason
    DEFAULT_HANDLERS: dict[EscalationReason, list[str]] = {
        EscalationReason.UNCERTAINTY: ["tier2_support"],
        EscalationReason.POLICY_VIOLATION: ["compliance_team", "legal"],
        EscalationReason.SENSITIVE_TOPIC: ["senior_support", "specialists"],
        EscalationReason.CUSTOMER_REQUEST: ["tier1_support"],
        EscalationReason.ERROR_THRESHOLD: ["engineering", "devops"],
        EscalationReason.COMPLEXITY: ["tier2_support", "specialists"],
        EscalationReason.AUTHORITY: ["management", "approvers"],
        EscalationReason.EMOTIONAL: ["senior_support", "wellness_team"],
        EscalationReason.SECURITY: ["security_team", "devops"],
    }
    
    def __init__(self, handlers: Optional[dict[EscalationReason, list[str]]] = None):
        """
        Initialize the escalation manager.
        
        Args:
            handlers: Custom handler assignments (overrides defaults)
        """
        self.escalations: list[Escalation] = []
        self.handlers = handlers or self.DEFAULT_HANDLERS.copy()
        self._escalation_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique escalation ID."""
        self._escalation_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ESC-{timestamp}-{self._escalation_counter:04d}"
    
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
        
        In production, this would send notifications via:
        - Slack/Teams messages
        - Email alerts
        - PagerDuty/OpsGenie for urgent issues
        - Ticketing system (create ticket)
        """
        handlers = self.handlers.get(escalation.reason, ["general_support"])
        
        # Priority indicators
        priority_indicators = {
            EscalationPriority.LOW: "🟢",
            EscalationPriority.MEDIUM: "🟡",
            EscalationPriority.HIGH: "🟠",
            EscalationPriority.URGENT: "🔴",
            EscalationPriority.CRITICAL: "🚨",
        }
        
        indicator = priority_indicators.get(escalation.priority, "⚪")
        
        print("\n" + "=" * 60)
        print(f"{indicator} ESCALATION CREATED {indicator}")
        print("=" * 60)
        print(f"ID:       {escalation.escalation_id}")
        print(f"Priority: {escalation.priority.name}")
        print(f"Reason:   {escalation.reason.value}")
        print(f"Time:     {escalation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nSummary: {escalation.summary}")
        print(f"\nRouting to: {', '.join(handlers)}")
        
        if escalation.context:
            print("\nContext:")
            for key, value in escalation.context.items():
                print(f"  • {key}: {value}")
        
        # Show conversation snippet
        if escalation.conversation_history:
            print("\nRecent conversation:")
            for msg in escalation.conversation_history[-3:]:
                role = msg.get("role", "unknown")
                content = str(msg.get("content", ""))[:100]
                print(f"  [{role}]: {content}...")
        
        print("=" * 60)
        
        # In production, you would also:
        # - Send Slack message to appropriate channel
        # - Create ticket in ticketing system
        # - Page on-call if critical
        # - Send email notification
    
    def assign_escalation(
        self,
        escalation_id: str,
        assignee: str
    ) -> Optional[Escalation]:
        """
        Assign an escalation to a specific handler.
        
        Args:
            escalation_id: ID of the escalation
            assignee: Username/ID of the assignee
        
        Returns:
            Updated Escalation or None if not found
        """
        for escalation in self.escalations:
            if escalation.escalation_id == escalation_id:
                escalation.assigned_to = assignee
                print(f"📋 Escalation {escalation_id} assigned to {assignee}")
                return escalation
        
        return None
    
    def resolve_escalation(
        self,
        escalation_id: str,
        resolution_notes: str,
        resolved_by: str
    ) -> Optional[Escalation]:
        """
        Mark an escalation as resolved.
        
        Args:
            escalation_id: ID of the escalation
            resolution_notes: Notes about how it was resolved
            resolved_by: Username/ID of resolver
        
        Returns:
            Updated Escalation or None if not found
        """
        for escalation in self.escalations:
            if escalation.escalation_id == escalation_id:
                escalation.resolved = True
                escalation.resolution_notes = resolution_notes
                escalation.assigned_to = resolved_by
                escalation.resolved_at = datetime.now()
                print(f"✅ Escalation {escalation_id} resolved by {resolved_by}")
                return escalation
        
        return None
    
    def get_pending_escalations(
        self,
        handler: Optional[str] = None,
        min_priority: Optional[EscalationPriority] = None,
        reason: Optional[EscalationReason] = None
    ) -> list[Escalation]:
        """
        Get unresolved escalations with optional filters.
        
        Args:
            handler: Filter by assigned handler type
            min_priority: Minimum priority level
            reason: Filter by escalation reason
        
        Returns:
            List of matching escalations, sorted by priority
        """
        pending = [e for e in self.escalations if not e.resolved]
        
        if min_priority:
            pending = [
                e for e in pending
                if e.priority.value >= min_priority.value
            ]
        
        if reason:
            pending = [e for e in pending if e.reason == reason]
        
        if handler:
            pending = [
                e for e in pending
                if handler in self.handlers.get(e.reason, [])
            ]
        
        # Sort by priority (highest first), then by creation time
        return sorted(
            pending,
            key=lambda e: (-e.priority.value, e.created_at)
        )
    
    def get_escalation_by_id(
        self,
        escalation_id: str
    ) -> Optional[Escalation]:
        """Look up an escalation by ID."""
        for escalation in self.escalations:
            if escalation.escalation_id == escalation_id:
                return escalation
        return None
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about escalations."""
        stats = {
            "total": len(self.escalations),
            "pending": len([e for e in self.escalations if not e.resolved]),
            "resolved": len([e for e in self.escalations if e.resolved]),
            "by_priority": {p.name: 0 for p in EscalationPriority},
            "by_reason": {r.value: 0 for r in EscalationReason},
            "average_resolution_time": None
        }
        
        resolution_times = []
        
        for escalation in self.escalations:
            stats["by_priority"][escalation.priority.name] += 1
            stats["by_reason"][escalation.reason.value] += 1
            
            if escalation.resolved and escalation.resolved_at:
                delta = escalation.resolved_at - escalation.created_at
                resolution_times.append(delta.total_seconds())
        
        if resolution_times:
            avg_seconds = sum(resolution_times) / len(resolution_times)
            stats["average_resolution_time"] = f"{avg_seconds / 60:.1f} minutes"
        
        return stats


class EscalationAwareAgent:
    """
    Helper class to determine when escalation is needed.
    
    Provides methods to detect escalation triggers in user messages
    and agent state.
    """
    
    # Phrases that trigger immediate escalation
    EXPLICIT_ESCALATION_PHRASES = [
        "speak to a human",
        "talk to a person",
        "talk to a real person",
        "get me a manager",
        "human support",
        "real person",
        "human agent",
        "transfer to human",
        "live agent",
        "speak to someone",
        "customer service",
    ]
    
    # Topics requiring escalation to specialized handlers
    SENSITIVE_TOPICS = [
        "suicide",
        "self-harm",
        "abuse",
        "violence",
        "emergency",
        "medical emergency",
        "threat",
        "harassment",
    ]
    
    # Keywords suggesting potential policy issues
    POLICY_KEYWORDS = [
        "lawyer",
        "sue",
        "legal action",
        "attorney",
        "lawsuit",
        "court",
        "discrimination",
        "regulatory",
    ]
    
    def __init__(
        self,
        error_threshold: int = 3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize escalation detection.
        
        Args:
            error_threshold: Number of errors before auto-escalating
            confidence_threshold: Minimum confidence before escalating
        """
        self.error_threshold = error_threshold
        self.confidence_threshold = confidence_threshold
        self.error_count = 0
    
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
        
        # Check for explicit escalation requests
        for phrase in self.EXPLICIT_ESCALATION_PHRASES:
            if phrase in message_lower:
                priority = EscalationPriority.HIGH
                if "urgent" in message_lower or "emergency" in message_lower:
                    priority = EscalationPriority.URGENT
                return (EscalationReason.CUSTOMER_REQUEST, priority)
        
        # Check for sensitive topics (highest priority)
        for topic in self.SENSITIVE_TOPICS:
            if topic in message_lower:
                return (
                    EscalationReason.SENSITIVE_TOPIC,
                    EscalationPriority.URGENT
                )
        
        # Check for policy-related keywords
        for keyword in self.POLICY_KEYWORDS:
            if keyword in message_lower:
                return (
                    EscalationReason.POLICY_VIOLATION,
                    EscalationPriority.HIGH
                )
        
        # Check confidence threshold
        if agent_confidence < self.confidence_threshold:
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


def main():
    """Demonstrate the escalation system."""
    print("=" * 60)
    print("Escalation System Demo")
    print("=" * 60)
    
    manager = EscalationManager()
    detector = EscalationAwareAgent(error_threshold=3)
    
    # Test messages
    test_messages = [
        "I want to speak to a human",
        "This is a simple question about your product",
        "I'm going to sue your company!",
        "I need help urgently - it's an emergency",
        "Can you help me with my account?",
    ]
    
    print("\nTesting escalation detection:\n")
    
    for message in test_messages:
        result = detector.should_escalate(message)
        if result:
            reason, priority = result
            print(f"'{message[:40]}...'")
            print(f"  → ESCALATE: {reason.value} ({priority.name})\n")
            
            # Create the escalation
            manager.escalate(
                reason=reason,
                summary=f"User message: {message[:50]}",
                context={"user_id": "demo_user"},
                conversation_history=[{"role": "user", "content": message}],
                priority=priority
            )
        else:
            print(f"'{message[:40]}...'")
            print(f"  → No escalation needed\n")
    
    # Show stats
    print("\n" + "=" * 60)
    print("Escalation Statistics")
    print("=" * 60)
    stats = manager.get_stats()
    print(f"Total escalations: {stats['total']}")
    print(f"Pending: {stats['pending']}")
    print(f"\nBy priority:")
    for priority, count in stats["by_priority"].items():
        if count > 0:
            print(f"  {priority}: {count}")
    print(f"\nBy reason:")
    for reason, count in stats["by_reason"].items():
        if count > 0:
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
