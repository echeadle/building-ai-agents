"""
Exercise Solution: File Management Agent with Human-in-the-Loop

A complete file management agent that demonstrates:
- Approval gates with risk-based tiers
- Preview-before-execute for deletions
- Audit logging of all operations
- Escalation handling
- Feedback collection after significant operations

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import json

load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Audit Log
# =============================================================================

@dataclass
class AuditEntry:
    """A single entry in the audit log."""
    entry_id: str
    timestamp: datetime
    operation: str
    target: str
    details: dict[str, Any]
    risk_level: str
    approval_status: str  # "auto", "approved", "rejected", "not_required"
    user_id: str
    success: bool
    error_message: Optional[str] = None


class AuditLog:
    """
    Maintains a complete audit trail of all file operations.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.entries: list[AuditEntry] = []
        self.log_file = log_file
        self._counter = 0
    
    def _generate_id(self) -> str:
        self._counter += 1
        return f"AUDIT-{datetime.now().strftime('%Y%m%d')}-{self._counter:05d}"
    
    def log(
        self,
        operation: str,
        target: str,
        details: dict[str, Any],
        risk_level: str,
        approval_status: str,
        user_id: str = "system",
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditEntry:
        """Log an operation to the audit trail."""
        entry = AuditEntry(
            entry_id=self._generate_id(),
            timestamp=datetime.now(),
            operation=operation,
            target=target,
            details=details,
            risk_level=risk_level,
            approval_status=approval_status,
            user_id=user_id,
            success=success,
            error_message=error_message
        )
        
        self.entries.append(entry)
        
        # Print audit entry
        status_icon = "✅" if success else "❌"
        print(f"📋 AUDIT: {status_icon} {operation} on {target} "
              f"[{risk_level}] [{approval_status}]")
        
        if self.log_file:
            self._persist()
        
        return entry
    
    def _persist(self) -> None:
        """Save audit log to file."""
        data = []
        for entry in self.entries:
            data.append({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "operation": entry.operation,
                "target": entry.target,
                "details": entry.details,
                "risk_level": entry.risk_level,
                "approval_status": entry.approval_status,
                "user_id": entry.user_id,
                "success": entry.success,
                "error_message": entry.error_message
            })
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_entries(
        self,
        operation: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> list[AuditEntry]:
        """Get audit entries with optional filters."""
        entries = self.entries
        
        if operation:
            entries = [e for e in entries if e.operation == operation]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        return entries
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the audit log."""
        summary = {
            "total_operations": len(self.entries),
            "successful": len([e for e in self.entries if e.success]),
            "failed": len([e for e in self.entries if not e.success]),
            "by_operation": {},
            "by_risk_level": {},
            "approvals_required": 0,
            "approvals_rejected": 0
        }
        
        for entry in self.entries:
            # By operation
            op = entry.operation
            summary["by_operation"][op] = summary["by_operation"].get(op, 0) + 1
            
            # By risk level
            risk = entry.risk_level
            summary["by_risk_level"][risk] = summary["by_risk_level"].get(risk, 0) + 1
            
            # Approvals
            if entry.approval_status == "approved":
                summary["approvals_required"] += 1
            elif entry.approval_status == "rejected":
                summary["approvals_rejected"] += 1
        
        return summary


# =============================================================================
# Approval System (Simplified for Exercise)
# =============================================================================

class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalResult:
    approved: bool
    modified: bool = False
    notes: Optional[str] = None
    modified_params: Optional[dict] = None


class FileOperationApproval:
    """
    Handles approval for file operations based on risk level.
    """
    
    def __init__(self, auto_approve_none: bool = True):
        self.auto_approve_none = auto_approve_none
    
    def request_approval(
        self,
        operation: str,
        target: str,
        details: dict[str, Any],
        risk_level: RiskLevel,
        preview: Optional[str] = None
    ) -> ApprovalResult:
        """Request approval based on risk level."""
        
        # No approval needed
        if risk_level == RiskLevel.NONE:
            return ApprovalResult(approved=True, notes="No approval required")
        
        # Auto-approve low risk with notification
        if risk_level == RiskLevel.LOW:
            print(f"\n🟢 LOW RISK: {operation} on {target}")
            print("   Auto-approved with notification")
            return ApprovalResult(approved=True, notes="Auto-approved (low risk)")
        
        # Medium risk: Simple confirmation
        if risk_level == RiskLevel.MEDIUM:
            return self._confirm_medium(operation, target, details)
        
        # High risk: Preview and confirm
        if risk_level == RiskLevel.HIGH:
            return self._confirm_high(operation, target, details, preview)
        
        # Critical: Multi-step confirmation
        if risk_level == RiskLevel.CRITICAL:
            return self._confirm_critical(operation, target, details, preview)
        
        return ApprovalResult(approved=False, notes="Unknown risk level")
    
    def _confirm_medium(
        self,
        operation: str,
        target: str,
        details: dict[str, Any]
    ) -> ApprovalResult:
        """Medium risk confirmation."""
        print(f"\n🟡 MEDIUM RISK: {operation}")
        print(f"   Target: {target}")
        for key, value in details.items():
            print(f"   {key}: {value}")
        
        response = input("\nApprove? (yes/no): ").strip().lower()
        
        if response == "yes":
            return ApprovalResult(approved=True)
        else:
            reason = input("Reason for rejection: ").strip()
            return ApprovalResult(approved=False, notes=reason)
    
    def _confirm_high(
        self,
        operation: str,
        target: str,
        details: dict[str, Any],
        preview: Optional[str] = None
    ) -> ApprovalResult:
        """High risk confirmation with preview."""
        print(f"\n🟠 HIGH RISK: {operation}")
        print("=" * 50)
        print(f"Target: {target}")
        
        if preview:
            print("\nPreview:")
            print("-" * 50)
            print(preview)
            print("-" * 50)
        
        for key, value in details.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
        response = input("\nType 'APPROVE' to confirm: ").strip()
        
        if response == "APPROVE":
            return ApprovalResult(approved=True)
        else:
            return ApprovalResult(approved=False, notes="Not approved")
    
    def _confirm_critical(
        self,
        operation: str,
        target: str,
        details: dict[str, Any],
        preview: Optional[str] = None
    ) -> ApprovalResult:
        """Critical risk multi-step confirmation."""
        print("\n" + "🔴" * 20)
        print(f"CRITICAL OPERATION: {operation}")
        print("🔴" * 20)
        print(f"\nTarget: {target}")
        
        if preview:
            print("\nPreview:")
            print("-" * 50)
            print(preview)
            print("-" * 50)
        
        print("\nDetails:")
        for key, value in details.items():
            print(f"  • {key}: {value}")
        
        # Step 1: Acknowledge
        print("\n⚠️  This operation may have serious consequences!")
        ack = input("Type 'I UNDERSTAND' to proceed: ").strip()
        if ack != "I UNDERSTAND":
            return ApprovalResult(approved=False, notes="Did not acknowledge")
        
        # Step 2: Reason
        reason = input("Provide reason for this operation: ").strip()
        if len(reason) < 5:
            return ApprovalResult(approved=False, notes="Insufficient reason")
        
        # Step 3: Final confirm
        confirm = input("Type 'EXECUTE' for final confirmation: ").strip()
        if confirm == "EXECUTE":
            return ApprovalResult(approved=True, notes=f"Reason: {reason}")
        else:
            return ApprovalResult(approved=False, notes="Cancelled at final step")


# =============================================================================
# Feedback Collection (Simplified)
# =============================================================================

class SimpleFeedback:
    """Simple feedback collection after operations."""
    
    def __init__(self):
        self.feedback_log: list[dict] = []
    
    def collect_after_operation(
        self,
        operation: str,
        result: dict[str, Any]
    ) -> Optional[dict]:
        """Optionally collect feedback after an operation."""
        print(f"\n📊 Quick feedback on '{operation}':")
        response = input("Was this helpful? (y/n/skip): ").strip().lower()
        
        if response == 'skip' or response == '':
            return None
        
        feedback = {
            "operation": operation,
            "result": result,
            "helpful": response == 'y',
            "timestamp": datetime.now().isoformat()
        }
        
        if response == 'n':
            feedback["comment"] = input("What could be improved? ").strip()
        
        self.feedback_log.append(feedback)
        print("Thanks for your feedback!")
        return feedback


# =============================================================================
# Escalation (Simplified)
# =============================================================================

class SimpleEscalation:
    """Simple escalation handling."""
    
    ESCALATION_PHRASES = [
        "speak to human",
        "talk to person",
        "human help",
        "real person",
        "manager",
        "escalate"
    ]
    
    def __init__(self):
        self.escalations: list[dict] = []
    
    def check_for_escalation(self, message: str) -> bool:
        """Check if message requests escalation."""
        message_lower = message.lower()
        return any(phrase in message_lower for phrase in self.ESCALATION_PHRASES)
    
    def create_escalation(
        self,
        reason: str,
        context: dict[str, Any]
    ) -> dict:
        """Create an escalation record."""
        escalation = {
            "id": f"ESC-{len(self.escalations) + 1:04d}",
            "reason": reason,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        self.escalations.append(escalation)
        
        print("\n" + "🚨" * 15)
        print("ESCALATION CREATED")
        print(f"ID: {escalation['id']}")
        print(f"Reason: {reason}")
        print("A human operator will assist you shortly.")
        print("🚨" * 15)
        
        return escalation


# =============================================================================
# File Management Agent
# =============================================================================

class FileManagementAgent:
    """
    File management agent with full human-in-the-loop capabilities.
    
    Features:
    - Risk-based approval for operations
    - Preview before delete
    - Complete audit logging
    - Escalation handling
    - Feedback collection
    """
    
    # Risk levels for each operation type
    OPERATION_RISKS = {
        "list_files": RiskLevel.NONE,
        "read_file": RiskLevel.NONE,
        "create_file": RiskLevel.LOW,
        "create_directory": RiskLevel.LOW,
        "rename_file": RiskLevel.MEDIUM,
        "move_file": RiskLevel.MEDIUM,
        "delete_file": RiskLevel.HIGH,
        "bulk_delete": RiskLevel.CRITICAL,
    }
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.approval = FileOperationApproval()
        self.audit = AuditLog()
        self.feedback = SimpleFeedback()
        self.escalation = SimpleEscalation()
        self.conversation_history: list[dict] = []
        
        # Simulated file system
        self.files: dict[str, str] = {
            "/documents/report.txt": "Quarterly report content...",
            "/documents/notes.txt": "Meeting notes from Monday...",
            "/images/logo.png": "[binary image data]",
            "/config/settings.json": '{"theme": "dark", "notifications": true}',
            "/temp/cache.tmp": "Temporary cache data",
            "/temp/old_backup.zip": "[backup archive]",
        }
        self.directories = {"/documents", "/images", "/config", "/temp"}
    
    def _get_tools(self) -> list[dict]:
        """Define available file management tools."""
        return [
            {
                "name": "list_files",
                "description": "List files in a directory. Read-only, no approval needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path to list"
                        }
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "read_file",
                "description": "Read contents of a file. Read-only, no approval needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to file to read"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            {
                "name": "create_file",
                "description": "Create a new file. Low-risk, auto-approved with notification.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path for new file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            },
            {
                "name": "create_directory",
                "description": "Create a new directory. Low-risk, auto-approved.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Path for new directory"
                        }
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "rename_file",
                "description": "Rename a file. Medium-risk, requires confirmation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "old_path": {
                            "type": "string",
                            "description": "Current file path"
                        },
                        "new_path": {
                            "type": "string",
                            "description": "New file path"
                        }
                    },
                    "required": ["old_path", "new_path"]
                }
            },
            {
                "name": "move_file",
                "description": "Move a file to different directory. Medium-risk, requires confirmation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Current file path"
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination path"
                        }
                    },
                    "required": ["source", "destination"]
                }
            },
            {
                "name": "delete_file",
                "description": "Delete a file. HIGH-RISK, requires approval with preview.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to file to delete"
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
                "name": "bulk_delete",
                "description": "Delete multiple files. CRITICAL-RISK, requires multi-step approval.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepaths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths to delete"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for bulk deletion"
                        }
                    },
                    "required": ["filepaths", "reason"]
                }
            }
        ]
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool with approval and audit logging."""
        
        risk_level = self.OPERATION_RISKS.get(tool_name, RiskLevel.MEDIUM)
        
        # Generate preview for delete operations
        preview = None
        if tool_name == "delete_file":
            filepath = tool_input.get("filepath", "")
            if filepath in self.files:
                content = self.files[filepath]
                preview = f"FILE: {filepath}\nSIZE: {len(content)} bytes\nPREVIEW: {content[:100]}..."
        elif tool_name == "bulk_delete":
            filepaths = tool_input.get("filepaths", [])
            preview_lines = ["FILES TO DELETE:"]
            for fp in filepaths:
                if fp in self.files:
                    preview_lines.append(f"  • {fp} ({len(self.files[fp])} bytes)")
                else:
                    preview_lines.append(f"  • {fp} (not found)")
            preview_lines.append(f"\nTOTAL: {len(filepaths)} files")
            preview = "\n".join(preview_lines)
        
        # Request approval if needed
        if risk_level != RiskLevel.NONE:
            approval_result = self.approval.request_approval(
                operation=tool_name,
                target=str(tool_input.get("filepath") or tool_input.get("directory") or tool_input.get("filepaths", [])),
                details=tool_input,
                risk_level=risk_level,
                preview=preview
            )
            
            if not approval_result.approved:
                self.audit.log(
                    operation=tool_name,
                    target=str(tool_input),
                    details=tool_input,
                    risk_level=risk_level.value,
                    approval_status="rejected",
                    success=False,
                    error_message=approval_result.notes
                )
                return {
                    "status": "rejected",
                    "reason": approval_result.notes or "Not approved"
                }
            
            approval_status = "approved"
        else:
            approval_status = "not_required"
        
        # Execute the operation
        try:
            result = self._perform_operation(tool_name, tool_input)
            
            self.audit.log(
                operation=tool_name,
                target=str(tool_input.get("filepath") or tool_input.get("directory") or ""),
                details=tool_input,
                risk_level=risk_level.value,
                approval_status=approval_status,
                success=True
            )
            
            # Collect feedback for significant operations
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.feedback.collect_after_operation(tool_name, result)
            
            return result
            
        except Exception as e:
            self.audit.log(
                operation=tool_name,
                target=str(tool_input),
                details=tool_input,
                risk_level=risk_level.value,
                approval_status=approval_status,
                success=False,
                error_message=str(e)
            )
            return {"status": "error", "message": str(e)}
    
    def _perform_operation(
        self,
        tool_name: str,
        tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform the actual file operation (simulated)."""
        
        if tool_name == "list_files":
            directory = tool_input["directory"]
            files = [f for f in self.files.keys() if f.startswith(directory)]
            return {
                "status": "success",
                "directory": directory,
                "files": files,
                "count": len(files)
            }
        
        elif tool_name == "read_file":
            filepath = tool_input["filepath"]
            if filepath in self.files:
                return {
                    "status": "success",
                    "filepath": filepath,
                    "content": self.files[filepath],
                    "size": len(self.files[filepath])
                }
            return {"status": "error", "message": f"File not found: {filepath}"}
        
        elif tool_name == "create_file":
            filepath = tool_input["filepath"]
            content = tool_input["content"]
            self.files[filepath] = content
            return {
                "status": "success",
                "filepath": filepath,
                "size": len(content),
                "message": f"File created: {filepath}"
            }
        
        elif tool_name == "create_directory":
            directory = tool_input["directory"]
            self.directories.add(directory)
            return {
                "status": "success",
                "directory": directory,
                "message": f"Directory created: {directory}"
            }
        
        elif tool_name == "rename_file":
            old_path = tool_input["old_path"]
            new_path = tool_input["new_path"]
            if old_path in self.files:
                self.files[new_path] = self.files.pop(old_path)
                return {
                    "status": "success",
                    "old_path": old_path,
                    "new_path": new_path,
                    "message": f"Renamed {old_path} to {new_path}"
                }
            return {"status": "error", "message": f"File not found: {old_path}"}
        
        elif tool_name == "move_file":
            source = tool_input["source"]
            destination = tool_input["destination"]
            if source in self.files:
                self.files[destination] = self.files.pop(source)
                return {
                    "status": "success",
                    "source": source,
                    "destination": destination,
                    "message": f"Moved {source} to {destination}"
                }
            return {"status": "error", "message": f"File not found: {source}"}
        
        elif tool_name == "delete_file":
            filepath = tool_input["filepath"]
            if filepath in self.files:
                del self.files[filepath]
                return {
                    "status": "success",
                    "filepath": filepath,
                    "message": f"Deleted: {filepath}"
                }
            return {"status": "error", "message": f"File not found: {filepath}"}
        
        elif tool_name == "bulk_delete":
            filepaths = tool_input["filepaths"]
            deleted = []
            not_found = []
            for fp in filepaths:
                if fp in self.files:
                    del self.files[fp]
                    deleted.append(fp)
                else:
                    not_found.append(fp)
            return {
                "status": "success",
                "deleted": deleted,
                "not_found": not_found,
                "message": f"Deleted {len(deleted)} files"
            }
        
        return {"status": "error", "message": f"Unknown operation: {tool_name}"}
    
    def chat(self, user_message: str) -> str:
        """Process a user message."""
        
        # Check for escalation request
        if self.escalation.check_for_escalation(user_message):
            self.escalation.create_escalation(
                reason="User requested human assistance",
                context={"last_message": user_message}
            )
            return "I've escalated your request to a human operator. They will assist you shortly. Is there anything simple I can help with in the meantime?"
        
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        system_prompt = """You are a file management assistant with access to file operations.

IMPORTANT - Operation Risk Levels:
- READ operations (list, read): No approval needed ✅
- CREATE operations (create file/directory): Low risk, auto-approved 🟢
- MODIFY operations (rename, move): Medium risk, requires confirmation 🟡
- DELETE single file: High risk, requires approval with preview 🟠
- BULK DELETE: Critical risk, requires multi-step approval 🔴

Always:
1. Explain what operation you're about to perform
2. Be clear about the risk level
3. If an operation is rejected, acknowledge and offer alternatives
4. For delete operations, explain what will be deleted before proceeding

Available directories: /documents, /images, /config, /temp

If the user asks to speak to a human or seems frustrated, acknowledge this."""
        
        max_iterations = 10
        
        for _ in range(max_iterations):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=self._get_tools(),
                messages=self.conversation_history
            )
            
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                return final_text
            
            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n🔧 Executing: {block.name}")
                    result = self._execute_tool(block.name, block.input)
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
        
        return "I've reached my operation limit. Please try a simpler request."
    
    def show_audit_summary(self) -> None:
        """Display audit log summary."""
        summary = self.audit.get_summary()
        print("\n" + "=" * 50)
        print("📋 AUDIT LOG SUMMARY")
        print("=" * 50)
        print(f"Total operations: {summary['total_operations']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Approvals required: {summary['approvals_required']}")
        print(f"Approvals rejected: {summary['approvals_rejected']}")
        print("\nBy operation type:")
        for op, count in summary['by_operation'].items():
            print(f"  • {op}: {count}")
        print("=" * 50)


def main():
    """Run the file management agent demo."""
    print("=" * 60)
    print("File Management Agent with Human-in-the-Loop")
    print("=" * 60)
    print("\nThis agent demonstrates:")
    print("  • Risk-based approval tiers")
    print("  • Preview before delete operations")
    print("  • Complete audit logging")
    print("  • Escalation handling")
    print("  • Feedback collection")
    print("\nAvailable directories: /documents, /images, /config, /temp")
    print("\nTry commands like:")
    print("  • 'List all files in /documents'")
    print("  • 'Create a new file called test.txt'")
    print("  • 'Rename report.txt to old_report.txt'")
    print("  • 'Delete the cache file in /temp'")
    print("  • 'Delete all files in /temp' (bulk delete)")
    print("  • 'I want to speak to a human' (escalation)")
    print("\nType 'quit' to exit, 'audit' for audit summary")
    print("=" * 60)
    
    agent = FileManagementAgent()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                agent.show_audit_summary()
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'audit':
                agent.show_audit_summary()
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            agent.show_audit_summary()
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
