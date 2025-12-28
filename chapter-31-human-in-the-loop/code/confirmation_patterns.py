"""
Confirmation Patterns

Various patterns for requesting confirmation before executing
actions, ranging from simple to complex based on risk level.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
from typing import Any, Callable, TypeVar, Optional
from functools import wraps
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

load_dotenv()

T = TypeVar('T')


# =============================================================================
# Pattern 1: Simple Confirmation Decorator
# =============================================================================

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
                print("✅ Confirmed. Executing...")
                return func(*args, **kwargs)
            else:
                print("❌ Action cancelled.")
                return None
        
        return wrapper
    return decorator


# =============================================================================
# Pattern 2: Tiered Confirmation Based on Risk Level
# =============================================================================

class RiskLevel(Enum):
    """Risk levels for actions requiring confirmation."""
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
    modifications: Optional[dict[str, Any]] = None


class TieredConfirmation:
    """
    Implements different confirmation requirements based on risk level.
    
    - LOW: Single key press to continue
    - MEDIUM: Type 'yes' to confirm
    - HIGH: Type a specific phrase
    - CRITICAL: Multiple confirmations with reason required
    
    Example:
        tiered = TieredConfirmation()
        result = tiered.request_confirmation(
            action="Delete user account",
            details={"user_id": "12345"},
            risk_level=RiskLevel.HIGH
        )
        if result.confirmed:
            # Execute action
            pass
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
        self._print_details(details)
        response = input("\nPress Enter to continue, or 'n' to cancel: ").strip()
        
        confirmed = response.lower() != 'n'
        if confirmed:
            print("✅ Continuing...")
        else:
            print("❌ Cancelled.")
        
        return ConfirmationResult(
            confirmed=confirmed,
            method="single_key"
        )
    
    def _confirm_medium(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """Medium risk: Type 'yes'."""
        print(f"\n⚠️  Action: {action}")
        self._print_details(details)
        response = input("\nType 'yes' to confirm: ").strip().lower()
        
        confirmed = response == 'yes'
        if confirmed:
            print("✅ Confirmed.")
        else:
            print("❌ Not confirmed.")
        
        return ConfirmationResult(
            confirmed=confirmed,
            method="type_yes"
        )
    
    def _confirm_high(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """High risk: Type a specific phrase."""
        print(f"\n🔶 HIGH RISK Action: {action}")
        self._print_details(details)
        
        # Generate a confirmation phrase based on the action
        phrase = "I CONFIRM THIS ACTION"
        print(f"\n⚠️  This is a high-risk operation.")
        print(f"Type exactly: {phrase}")
        response = input("> ").strip()
        
        confirmed = response == phrase
        if confirmed:
            print("✅ Confirmed.")
        else:
            print("❌ Phrase did not match. Action cancelled.")
        
        return ConfirmationResult(
            confirmed=confirmed,
            method="type_phrase"
        )
    
    def _confirm_critical(
        self,
        action: str,
        details: dict[str, Any]
    ) -> ConfirmationResult:
        """Critical risk: Multiple confirmations and reason required."""
        print("\n" + "🚨" * 20)
        print(f"CRITICAL ACTION: {action}")
        print("🚨" * 20)
        self._print_details(details)
        
        # First confirmation
        print("\n⚠️  This action has serious consequences and may be irreversible.")
        response1 = input("Type 'I UNDERSTAND' to proceed: ").strip()
        if response1 != "I UNDERSTAND":
            print("❌ Action cancelled.")
            return ConfirmationResult(confirmed=False, method="critical_multi")
        
        # Require justification
        print("\n📝 Please provide a reason for this action:")
        reason = input("> ").strip()
        if len(reason) < 10:
            print("❌ Please provide a more detailed reason (at least 10 characters).")
            return ConfirmationResult(confirmed=False, method="critical_multi")
        
        # Final confirmation
        print(f"\n🔴 FINAL CONFIRMATION")
        print(f"You are about to: {action}")
        print(f"Reason provided: {reason}")
        response2 = input("\nType 'EXECUTE' to proceed: ").strip()
        
        confirmed = response2 == "EXECUTE"
        if confirmed:
            print("✅ Action confirmed and will be executed.")
        else:
            print("❌ Action cancelled.")
        
        return ConfirmationResult(
            confirmed=confirmed,
            method="critical_multi",
            notes=reason if confirmed else None
        )
    
    def _print_details(self, details: dict[str, Any]) -> None:
        """Print action details in a formatted way."""
        if details:
            print("Details:")
            for key, value in details.items():
                print(f"  • {key}: {value}")


# =============================================================================
# Pattern 3: Preview Before Execute
# =============================================================================

class PreviewableAction(ABC):
    """
    Base class for actions that can be previewed before execution.
    
    Subclasses must implement preview() and execute() methods.
    The preview_and_confirm() method shows the preview and asks
    for confirmation before executing.
    """
    
    @abstractmethod
    def preview(self) -> str:
        """Generate a preview of what this action will do."""
        pass
    
    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Execute the action."""
        pass
    
    def preview_and_confirm(self) -> Optional[dict[str, Any]]:
        """
        Show preview and get confirmation before executing.
        
        Returns:
            Result of execution if confirmed, None if cancelled
        """
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
    """
    Email action with preview capability.
    
    Shows exactly what email will be sent before sending it.
    """
    
    def __init__(
        self,
        recipients: list[str],
        subject: str,
        body: str,
        cc: Optional[list[str]] = None,
        attachments: Optional[list[str]] = None
    ):
        self.recipients = recipients
        self.subject = subject
        self.body = body
        self.cc = cc or []
        self.attachments = attachments or []
    
    def preview(self) -> str:
        """Preview the email that will be sent."""
        lines = [
            f"TO: {', '.join(self.recipients)}",
        ]
        
        if self.cc:
            lines.append(f"CC: {', '.join(self.cc)}")
        
        lines.extend([
            f"SUBJECT: {self.subject}",
            "",
            "BODY:",
            "-" * 40,
        ])
        
        # Truncate long bodies
        body_preview = self.body
        if len(body_preview) > 500:
            body_preview = body_preview[:500] + "\n... (truncated)"
        lines.append(body_preview)
        
        lines.append("-" * 40)
        
        if self.attachments:
            lines.append(f"\nATTACHMENTS: {', '.join(self.attachments)}")
        
        lines.append(f"\n📨 This will send to {len(self.recipients)} recipient(s)")
        
        return "\n".join(lines)
    
    def execute(self) -> dict[str, Any]:
        """Send the email (simulated)."""
        return {
            "status": "sent",
            "recipients": len(self.recipients),
            "subject": self.subject,
            "message_id": "MSG-12345"
        }


class DatabaseUpdateAction(PreviewableAction):
    """
    Database update action with preview capability.
    
    Shows exactly what changes will be made before executing.
    """
    
    def __init__(
        self,
        table: str,
        operation: str,  # "insert", "update", "delete"
        updates: Optional[dict[str, Any]] = None,
        where_clause: Optional[str] = None,
        affected_rows: int = 0
    ):
        self.table = table
        self.operation = operation
        self.updates = updates or {}
        self.where_clause = where_clause or "TRUE"
        self.affected_rows = affected_rows
    
    def preview(self) -> str:
        """Preview the database operation."""
        lines = [
            f"TABLE: {self.table}",
            f"OPERATION: {self.operation.upper()}",
        ]
        
        if self.operation == "update" and self.updates:
            update_str = ", ".join(
                f"{k} = {repr(v)}" for k, v in self.updates.items()
            )
            lines.append(f"SET: {update_str}")
        
        if self.operation in ("update", "delete"):
            lines.append(f"WHERE: {self.where_clause}")
        
        if self.operation == "insert" and self.updates:
            lines.append(f"VALUES: {self.updates}")
        
        # Warning about affected rows
        if self.affected_rows > 0:
            warning = f"\n⚠️  This will affect {self.affected_rows} row(s)"
            if self.affected_rows > 100:
                warning += " - LARGE OPERATION"
            lines.append(warning)
        
        return "\n".join(lines)
    
    def execute(self) -> dict[str, Any]:
        """Execute the database operation (simulated)."""
        return {
            "status": "success",
            "table": self.table,
            "operation": self.operation,
            "rows_affected": self.affected_rows
        }


class FileDeleteAction(PreviewableAction):
    """
    File deletion action with preview capability.
    
    Shows file details before deletion.
    """
    
    def __init__(
        self,
        filepath: str,
        file_size: int = 0,
        reason: str = ""
    ):
        self.filepath = filepath
        self.file_size = file_size
        self.reason = reason
    
    def preview(self) -> str:
        """Preview the file deletion."""
        lines = [
            f"FILE: {self.filepath}",
            f"SIZE: {self._format_size(self.file_size)}",
        ]
        
        if self.reason:
            lines.append(f"REASON: {self.reason}")
        
        lines.append("\n🗑️  This file will be PERMANENTLY DELETED")
        
        return "\n".join(lines)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def execute(self) -> dict[str, Any]:
        """Delete the file (simulated)."""
        return {
            "status": "deleted",
            "filepath": self.filepath,
            "recovered_space": self.file_size
        }


# =============================================================================
# Demonstration
# =============================================================================

def demo_decorator():
    """Demonstrate the confirmation decorator."""
    print("\n" + "=" * 60)
    print("Demo 1: Confirmation Decorator")
    print("=" * 60)
    
    @requires_confirmation("This will reset all user preferences to defaults.")
    def reset_preferences():
        return {"status": "reset", "affected_users": 50}
    
    result = reset_preferences()
    print(f"Result: {result}")


def demo_tiered():
    """Demonstrate tiered confirmation."""
    print("\n" + "=" * 60)
    print("Demo 2: Tiered Confirmation")
    print("=" * 60)
    
    tiered = TieredConfirmation()
    
    # Low risk
    print("\n--- LOW RISK ---")
    tiered.request_confirmation(
        action="View user profile",
        details={"user_id": "123"},
        risk_level=RiskLevel.LOW
    )
    
    # Medium risk
    print("\n--- MEDIUM RISK ---")
    tiered.request_confirmation(
        action="Update user email",
        details={"user_id": "123", "new_email": "new@example.com"},
        risk_level=RiskLevel.MEDIUM
    )
    
    # High risk
    print("\n--- HIGH RISK ---")
    tiered.request_confirmation(
        action="Export all user data",
        details={"format": "CSV", "include_pii": True},
        risk_level=RiskLevel.HIGH
    )


def demo_preview():
    """Demonstrate preview-before-execute pattern."""
    print("\n" + "=" * 60)
    print("Demo 3: Preview Before Execute")
    print("=" * 60)
    
    # Email preview
    print("\n--- EMAIL PREVIEW ---")
    email = EmailAction(
        recipients=["alice@example.com", "bob@example.com"],
        subject="Important Update",
        body="Dear team,\n\nThis is to inform you of an important update...\n\nBest regards",
        cc=["manager@example.com"],
        attachments=["report.pdf"]
    )
    email.preview_and_confirm()
    
    # Database preview
    print("\n--- DATABASE PREVIEW ---")
    db_update = DatabaseUpdateAction(
        table="users",
        operation="update",
        updates={"status": "inactive", "updated_at": "NOW()"},
        where_clause="last_login < '2023-01-01'",
        affected_rows=250
    )
    db_update.preview_and_confirm()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Confirmation Patterns Demo")
    print("=" * 60)
    print("\nThis demo shows three confirmation patterns:")
    print("1. Simple decorator-based confirmation")
    print("2. Tiered confirmation based on risk level")
    print("3. Preview-before-execute pattern")
    
    demo_decorator()
    demo_tiered()
    demo_preview()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
