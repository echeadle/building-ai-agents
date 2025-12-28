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

# Import components (in practice, these would be separate files)
from approval_gate import ApprovalGate, ApprovalStatus
from feedback_collector import FeedbackCollector
from escalation import EscalationManager, EscalationReason, EscalationPriority

load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


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
    
    # Preview settings
    preview_before_execute: bool = True


class HumanInTheLoopAgent:
    """
    A complete agent with human-in-the-loop capabilities.
    
    Features:
    - Approval gates for sensitive actions
    - Confirmation flows with previews
    - Feedback collection
    - Escalation to humans when needed
    
    Example:
        config = HITLConfig(
            require_approval_for_high_risk=True,
            collect_feedback=True,
            enable_escalation=True
        )
        
        agent = HumanInTheLoopAgent(config)
        response = agent.chat("Send an email to all customers")
        # Agent will request approval before sending
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
                        "recipient": {
                            "type": "string",
                            "description": "Recipient address or ID"
                        },
                        "message": {
                            "type": "string",
                            "description": "Message content"
                        },
                        "channel": {
                            "type": "string",
                            "enum": ["email", "sms", "slack", "internal"],
                            "description": "Communication channel"
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
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "table": {
                            "type": "string",
                            "description": "Table to search (optional)"
                        }
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
                            "enum": ["create", "update", "delete"],
                            "description": "Type of operation"
                        },
                        "table": {
                            "type": "string",
                            "description": "Target table"
                        },
                        "record_id": {
                            "type": "string",
                            "description": "Record ID (for update/delete)"
                        },
                        "data": {
                            "type": "object",
                            "description": "Record data (for create/update)"
                        }
                    },
                    "required": ["operation", "table"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather for a location. RISK: LOW - Read-only.",
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
                "name": "schedule_task",
                "description": "Schedule a task for future execution. "
                              "RISK: MEDIUM - Requires confirmation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": "Name of the task"
                        },
                        "scheduled_time": {
                            "type": "string",
                            "description": "When to execute (ISO format)"
                        },
                        "task_details": {
                            "type": "object",
                            "description": "Task parameters"
                        }
                    },
                    "required": ["task_name", "scheduled_time"]
                }
            }
        ]
    
    def _assess_risk(self, tool_name: str, tool_input: dict) -> str:
        """Assess the risk level of a tool call."""
        risk_map = {
            "send_message": "high",
            "modify_record": "critical",
            "schedule_task": "medium",
            "search_database": "low",
            "get_weather": "low"
        }
        return risk_map.get(tool_name, "medium")
    
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
                    tool_input.update(request.modified_details)
        
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
                    {"id": "1", "name": "Sample Result 1", "score": 0.95},
                    {"id": "2", "name": "Sample Result 2", "score": 0.87}
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
                "conditions": "Sunny",
                "humidity": "45%"
            }
        
        elif tool_name == "schedule_task":
            return {
                "status": "scheduled",
                "task_name": tool_input["task_name"],
                "scheduled_time": tool_input["scheduled_time"],
                "task_id": "TASK-001"
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
            "real person",
            "live agent"
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
        
        # Determine priority
        message_lower = user_message.lower()
        if any(word in message_lower for word in ["urgent", "emergency"]):
            priority = EscalationPriority.URGENT
        else:
            priority = EscalationPriority.HIGH
        
        # Determine reason
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
            self.feedback_collector.collect_thumbs(
                context={"interaction": self.interaction_count},
                agent_output=response[:200] + "..." if len(response) > 200 else response
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
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Run agentic loop
            max_iterations = 10
            
            for _ in range(max_iterations):
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
                    
                    # Collect feedback
                    self._maybe_collect_feedback(final_text)
                    
                    # Reset errors on success
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
            
            return "Maximum iterations reached. Please try a simpler request."
        
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
    
    def reset(self) -> None:
        """Reset the agent state."""
        self.conversation_history = []
        self.interaction_count = 0
        self.error_count = 0
        self.is_escalated = False


def main():
    """Demonstrate the Human-in-the-Loop agent."""
    print("=" * 60)
    print("Human-in-the-Loop Agent Demo")
    print("=" * 60)
    print("\nThis agent demonstrates:")
    print("  • Approval gates for sensitive actions")
    print("  • Escalation when you ask for a human")
    print("  • Periodic feedback collection")
    print("\nTry asking it to:")
    print("  • 'Send an email to john@example.com' (requires approval)")
    print("  • 'Check the weather in Paris' (no approval needed)")
    print("  • 'Update customer record #123' (requires approval)")
    print("  • 'I want to speak to a human' (triggers escalation)")
    print("\nType 'quit' to exit, 'stats' for statistics, 'reset' to start over")
    print("=" * 60)
    
    config = HITLConfig(
        require_approval_for_high_risk=True,
        auto_approve_low_risk=True,
        collect_feedback=True,
        feedback_frequency=3,
        enable_escalation=True
    )
    
    agent = HumanInTheLoopAgent(config)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n📊 Final Statistics:")
                stats = agent.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 Agent Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower() == 'reset':
                agent.reset()
                print("🔄 Agent reset. Starting fresh.")
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
