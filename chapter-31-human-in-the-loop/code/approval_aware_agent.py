"""
Approval-Aware Agent

An agent that integrates with the approval gate system
to request human approval for sensitive operations.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass
from typing import Any, Optional

from approval_gate import ApprovalGate, ApprovalStatus, ApprovalRequest

load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ActionConfig:
    """Configuration for an action that may require approval."""
    requires_approval: bool = False
    risk_level: str = "low"
    approval_message: str = ""


class ApprovalAwareAgent:
    """
    An agent that requests human approval for sensitive actions.
    
    This agent demonstrates how to integrate approval gates into
    the agentic loop, pausing execution when sensitive actions
    are requested and waiting for human approval.
    
    Example:
        agent = ApprovalAwareAgent()
        response = agent.run("Send an email to all customers about our new product")
        # Agent will pause and request approval before sending
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
        "get_weather": ActionConfig(
            requires_approval=False,
            risk_level="low",
            approval_message="Get weather information"
        ),
    }
    
    def __init__(
        self,
        approval_gate: Optional[ApprovalGate] = None,
        auto_approve_low_risk: bool = True
    ):
        """
        Initialize the agent with an approval gate.
        
        Args:
            approval_gate: The approval gate to use. If None, creates one.
            auto_approve_low_risk: Whether to auto-approve low-risk actions
        """
        self.client = anthropic.Anthropic()
        self.approval_gate = approval_gate or ApprovalGate(
            auto_approve_low_risk=auto_approve_low_risk
        )
        self.conversation_history: list[dict] = []
    
    def _get_tools(self) -> list[dict]:
        """Define tools available to the agent."""
        return [
            {
                "name": "send_email",
                "description": "Send an email to specified recipients. "
                              "⚠️ REQUIRES HUMAN APPROVAL before execution.",
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
                "description": "Delete a file from the system. "
                              "🚨 CRITICAL: Requires human approval. Irreversible action.",
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
                "description": "Create a new file with specified content. "
                              "Low-risk operation.",
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
                "description": "Search the web for information. Read-only, low-risk.",
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
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location. Read-only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "update_database",
                "description": "Update records in the database. "
                              "⚠️ REQUIRES HUMAN APPROVAL for any modifications.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Database table name"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["insert", "update", "delete"],
                            "description": "Type of database operation"
                        },
                        "data": {
                            "type": "object",
                            "description": "Data for the operation"
                        },
                        "where_clause": {
                            "type": "string",
                            "description": "SQL WHERE clause for update/delete"
                        }
                    },
                    "required": ["table", "operation"]
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
                    tool_input.update(request.modified_details)
                    print(f"📝 Using modified parameters: {request.modified_details}")
        
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
            recipients = tool_input.get("to", [])
            return {
                "status": "success",
                "message": f"Email sent to {len(recipients)} recipient(s)",
                "recipients": recipients,
                "subject": tool_input.get("subject", "")
            }
        
        elif tool_name == "delete_file":
            return {
                "status": "success",
                "message": f"File '{tool_input['filepath']}' has been deleted",
                "filepath": tool_input["filepath"],
                "reason": tool_input.get("reason", "Not specified")
            }
        
        elif tool_name == "create_file":
            content = tool_input.get("content", "")
            return {
                "status": "success",
                "message": f"File '{tool_input['filepath']}' created",
                "filepath": tool_input["filepath"],
                "size_bytes": len(content)
            }
        
        elif tool_name == "search_web":
            return {
                "status": "success",
                "query": tool_input["query"],
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1"},
                    {"title": "Result 2", "url": "https://example.com/2"},
                    {"title": "Result 3", "url": "https://example.com/3"}
                ]
            }
        
        elif tool_name == "get_weather":
            return {
                "status": "success",
                "location": tool_input["location"],
                "temperature": "72°F",
                "conditions": "Partly cloudy",
                "humidity": "45%"
            }
        
        elif tool_name == "update_database":
            return {
                "status": "success",
                "table": tool_input["table"],
                "operation": tool_input["operation"],
                "affected_rows": 1
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

IMPORTANT: Some tools require human approval before execution:
- send_email: Requires approval (high risk)
- delete_file: Requires approval (critical risk)
- update_database: Requires approval (high risk)

When you use these tools, the system will pause and ask for human approval.
If an action is rejected, acknowledge this gracefully and offer alternatives.

Always:
1. Explain what you're about to do before using a sensitive tool
2. Be transparent about the approval process
3. If rejected, ask how you can help differently

Tools that DON'T require approval (low risk):
- create_file
- search_web
- get_weather"""
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=self._get_tools(),
                messages=self.conversation_history
            )
            
            # Check if we're done (no more tool use)
            if response.stop_reason == "end_turn":
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
                    print(f"\n🔧 Agent requesting tool: {block.name}")
                    print(f"   Parameters: {block.input}")
                    
                    # Execute with approval check
                    result = self._execute_with_approval(
                        block.name,
                        block.input
                    )
                    
                    print(f"   Result: {result.get('status', 'unknown')}")
                    
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
        
        return "Maximum iterations reached. Please try a simpler request."
    
    def reset_conversation(self) -> None:
        """Clear conversation history for a fresh start."""
        self.conversation_history = []
    
    def get_approval_stats(self) -> dict[str, Any]:
        """Get statistics about approval requests."""
        return self.approval_gate.get_approval_stats()


def main():
    """Demonstrate the approval-aware agent."""
    print("=" * 60)
    print("Approval-Aware Agent Demo")
    print("=" * 60)
    print("\nThis agent will request approval for sensitive operations.")
    print("Try asking it to:")
    print("  • 'Send an email to john@example.com'")
    print("  • 'Check the weather in Paris'")
    print("  • 'Delete the old config file'")
    print("\nType 'quit' to exit, 'stats' for approval statistics")
    print("Type 'reset' to clear conversation history")
    print("=" * 60)
    
    agent = ApprovalAwareAgent(auto_approve_low_risk=True)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_approval_stats()
                print("\n📊 Approval Statistics:")
                print(f"   Total requests: {stats['total']}")
                print(f"   Approved: {stats['approved']}")
                print(f"   Rejected: {stats['rejected']}")
                print(f"   Modified: {stats['modified']}")
                continue
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                print("🔄 Conversation history cleared.")
                continue
            
            response = agent.run(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
