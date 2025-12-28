"""
Autonomy Patterns for Agents

Demonstrates different levels of agent autonomy, from fully controlled
to fully autonomous. Choose the right level based on risk assessment.

Chapter 26: From Workflows to Agents
"""

import os
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# =============================================================================
# AUTONOMY LEVELS
# =============================================================================

class AutonomyLevel(Enum):
    """
    The spectrum of agent autonomy.
    
    From low to high:
    - CONFIRMATION_REQUIRED: Every action needs human approval
    - CHECKPOINT_MODE: Agent works freely but pauses at key points
    - BOUNDED_ACTIONS: Agent is autonomous within strict tool limits
    - FULLY_AUTONOMOUS: Agent has complete freedom
    """
    CONFIRMATION_REQUIRED = "confirmation_required"
    CHECKPOINT_MODE = "checkpoint_mode"
    BOUNDED_ACTIONS = "bounded_actions"
    FULLY_AUTONOMOUS = "fully_autonomous"


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# Different risk levels for tools
LOW_RISK_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Safe, read-only operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "search",
        "description": "Search for information. Safe, no side effects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a calculation. Safe, no side effects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

MEDIUM_RISK_TOOLS = [
    {
        "name": "write_file",
        "description": "Write content to a file. Modifies the filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["filepath", "content"]
        }
    },
    {
        "name": "create_task",
        "description": "Create a task in the task management system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title"},
                "description": {"type": "string", "description": "Task description"}
            },
            "required": ["title"]
        }
    }
]

HIGH_RISK_TOOLS = [
    {
        "name": "send_email",
        "description": "Send an email. Cannot be undone once sent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"}
            },
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "delete_file",
        "description": "Delete a file. Cannot be undone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to delete"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command. Potentially dangerous.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"}
            },
            "required": ["command"]
        }
    }
]

ALL_TOOLS = LOW_RISK_TOOLS + MEDIUM_RISK_TOOLS + HIGH_RISK_TOOLS


# =============================================================================
# SIMULATED TOOL EXECUTION
# =============================================================================

def simulate_tool(tool_name: str, tool_input: dict) -> str:
    """Simulate tool execution for demonstration."""
    simulations = {
        "read_file": f"[Contents of {tool_input.get('filepath', 'file')}]: Example file content...",
        "search": f"[Search results for '{tool_input.get('query', '')}']: Found 3 relevant results...",
        "calculate": f"[Result]: {eval(tool_input.get('expression', '0'))}",
        "write_file": f"[Written]: Saved content to {tool_input.get('filepath', 'file')}",
        "create_task": f"[Created]: Task '{tool_input.get('title', '')}' created",
        "send_email": f"[Sent]: Email to {tool_input.get('to', '')} with subject '{tool_input.get('subject', '')}'",
        "delete_file": f"[Deleted]: {tool_input.get('filepath', 'file')}",
        "execute_command": f"[Executed]: {tool_input.get('command', '')}",
    }
    return simulations.get(tool_name, f"[{tool_name}]: Action completed")


# =============================================================================
# HUMAN INPUT SIMULATION
# =============================================================================

def get_user_confirmation(action_description: str) -> bool:
    """Get user confirmation for an action."""
    print(f"\n{'='*50}")
    print("🛑 CONFIRMATION REQUIRED")
    print(f"{'='*50}")
    print(f"The agent wants to: {action_description}")
    
    # In a real application, you'd get actual user input
    # For demonstration, we'll simulate approval
    response = input("Allow this action? [y/n]: ").strip().lower()
    approved = response in ('y', 'yes')
    
    if approved:
        print("✓ Action approved")
    else:
        print("✗ Action denied")
    
    return approved


def get_checkpoint_approval(summary: str, actions_taken: list) -> bool:
    """Get user approval at a checkpoint."""
    print(f"\n{'='*50}")
    print("⏸️  CHECKPOINT")
    print(f"{'='*50}")
    print(f"Progress so far:")
    for i, action in enumerate(actions_taken, 1):
        print(f"  {i}. {action}")
    print(f"\nCurrent status: {summary}")
    
    response = input("Continue? [y/n]: ").strip().lower()
    return response in ('y', 'yes')


# =============================================================================
# PATTERN 1: CONFIRMATION REQUIRED (Lowest Autonomy)
# =============================================================================

def run_agent_confirmation_required(
    user_request: str,
    max_iterations: int = 10
) -> str:
    """
    Agent that requires confirmation for EVERY action.
    
    Use when:
    - Actions are irreversible
    - Stakes are high
    - Building trust with a new agent
    - Dealing with sensitive data
    """
    print("\n" + "=" * 60)
    print("AUTONOMY LEVEL: CONFIRMATION_REQUIRED")
    print("Every action needs human approval")
    print("=" * 60)
    
    messages = [{"role": "user", "content": user_request}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=ALL_TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    # EVERY action requires confirmation
                    action_desc = f"{block.name}({block.input})"
                    
                    if get_user_confirmation(action_desc):
                        result = simulate_tool(block.name, block.input)
                    else:
                        result = "Action was denied by user"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"


# =============================================================================
# PATTERN 2: CHECKPOINT MODE (Medium-Low Autonomy)
# =============================================================================

def run_agent_checkpoint_mode(
    user_request: str,
    checkpoint_interval: int = 3,
    max_iterations: int = 10
) -> str:
    """
    Agent that pauses at checkpoints for approval.
    
    Use when:
    - Tasks have natural breakpoints
    - You want oversight without micromanagement
    - Medium-stakes operations
    """
    print("\n" + "=" * 60)
    print("AUTONOMY LEVEL: CHECKPOINT_MODE")
    print(f"Pauses every {checkpoint_interval} actions for approval")
    print("=" * 60)
    
    messages = [{"role": "user", "content": user_request}]
    actions_since_checkpoint = []
    total_actions = 0
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=ALL_TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → {block.name}({block.input})")
                    
                    # Execute without asking
                    result = simulate_tool(block.name, block.input)
                    actions_since_checkpoint.append(f"{block.name}: {result}")
                    total_actions += 1
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
            
            # Check if we've hit a checkpoint
            if len(actions_since_checkpoint) >= checkpoint_interval:
                if not get_checkpoint_approval(
                    f"{total_actions} actions completed",
                    actions_since_checkpoint
                ):
                    return "User stopped agent at checkpoint"
                actions_since_checkpoint = []
    
    return "Max iterations reached"


# =============================================================================
# PATTERN 3: BOUNDED ACTIONS (Medium-High Autonomy)
# =============================================================================

def run_agent_bounded_actions(
    user_request: str,
    allowed_tools: list = None,
    max_iterations: int = 10
) -> str:
    """
    Agent that is autonomous but limited to specific tools.
    
    Use when:
    - You want efficiency but with guardrails
    - Some actions are safe, others are not
    - The agent needs to work quickly on routine tasks
    """
    if allowed_tools is None:
        # Default: only low-risk tools
        allowed_tools = [t["name"] for t in LOW_RISK_TOOLS]
    
    print("\n" + "=" * 60)
    print("AUTONOMY LEVEL: BOUNDED_ACTIONS")
    print(f"Allowed tools: {allowed_tools}")
    print("=" * 60)
    
    # Only provide the allowed tools to the model
    available_tools = [t for t in ALL_TOOLS if t["name"] in allowed_tools]
    
    messages = [{"role": "user", "content": user_request}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=available_tools,  # Limited tool set
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    # Double-check the tool is allowed (defense in depth)
                    if block.name not in allowed_tools:
                        print(f"  ✗ Blocked: {block.name} (not in allowed list)")
                        result = "Error: This tool is not available"
                    else:
                        print(f"  ✓ {block.name}({block.input})")
                        result = simulate_tool(block.name, block.input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"


# =============================================================================
# PATTERN 4: FULLY AUTONOMOUS (Highest Autonomy)
# =============================================================================

def run_agent_fully_autonomous(
    user_request: str,
    max_iterations: int = 10
) -> str:
    """
    Agent with complete autonomy over all available tools.
    
    Use when:
    - Stakes are low
    - All actions are reversible
    - You have high trust in the agent
    - Speed is critical
    
    WARNING: Use with caution in production!
    """
    print("\n" + "=" * 60)
    print("AUTONOMY LEVEL: FULLY_AUTONOMOUS")
    print("⚠️  Agent has access to ALL tools without restrictions")
    print("=" * 60)
    
    messages = [{"role": "user", "content": user_request}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=ALL_TOOLS,  # All tools available
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            return response.content[0].text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    # Execute immediately without any checks
                    print(f"  → {block.name}({block.input})")
                    result = simulate_tool(block.name, block.input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached"


# =============================================================================
# RISK ASSESSMENT HELPER
# =============================================================================

@dataclass
class RiskAssessment:
    """Assessment of risk factors for an agent deployment."""
    data_access: str  # "read_only", "read_write", "full"
    external_actions: bool  # Does it send emails, make purchases, etc.?
    scope: str  # "narrow", "medium", "broad"
    reversibility: str  # "fully", "partially", "not_reversible"
    data_sensitivity: str  # "public", "internal", "confidential", "restricted"
    
    def recommend_autonomy(self) -> AutonomyLevel:
        """Recommend an autonomy level based on risk factors."""
        risk_score = 0
        
        # Data access
        if self.data_access == "full":
            risk_score += 3
        elif self.data_access == "read_write":
            risk_score += 2
        else:
            risk_score += 1
        
        # External actions
        if self.external_actions:
            risk_score += 3
        
        # Scope
        if self.scope == "broad":
            risk_score += 2
        elif self.scope == "medium":
            risk_score += 1
        
        # Reversibility
        if self.reversibility == "not_reversible":
            risk_score += 3
        elif self.reversibility == "partially":
            risk_score += 1
        
        # Data sensitivity
        sensitivity_scores = {
            "restricted": 4,
            "confidential": 3,
            "internal": 2,
            "public": 1
        }
        risk_score += sensitivity_scores.get(self.data_sensitivity, 2)
        
        # Recommend based on total risk
        if risk_score >= 12:
            return AutonomyLevel.CONFIRMATION_REQUIRED
        elif risk_score >= 8:
            return AutonomyLevel.CHECKPOINT_MODE
        elif risk_score >= 5:
            return AutonomyLevel.BOUNDED_ACTIONS
        else:
            return AutonomyLevel.FULLY_AUTONOMOUS


def demonstrate_risk_assessment():
    """Show how to assess risk and choose autonomy level."""
    print("\n" + "=" * 60)
    print("RISK ASSESSMENT EXAMPLES")
    print("=" * 60)
    
    scenarios = [
        ("Research Assistant", RiskAssessment(
            data_access="read_only",
            external_actions=False,
            scope="narrow",
            reversibility="fully",
            data_sensitivity="public"
        )),
        ("Task Manager", RiskAssessment(
            data_access="read_write",
            external_actions=False,
            scope="medium",
            reversibility="fully",
            data_sensitivity="internal"
        )),
        ("Email Assistant", RiskAssessment(
            data_access="read_write",
            external_actions=True,
            scope="medium",
            reversibility="not_reversible",
            data_sensitivity="confidential"
        )),
        ("Financial Trading Bot", RiskAssessment(
            data_access="full",
            external_actions=True,
            scope="broad",
            reversibility="not_reversible",
            data_sensitivity="restricted"
        ))
    ]
    
    for name, assessment in scenarios:
        recommendation = assessment.recommend_autonomy()
        print(f"\n{name}:")
        print(f"  Data access: {assessment.data_access}")
        print(f"  External actions: {assessment.external_actions}")
        print(f"  Scope: {assessment.scope}")
        print(f"  Reversibility: {assessment.reversibility}")
        print(f"  Data sensitivity: {assessment.data_sensitivity}")
        print(f"  → Recommended: {recommendation.value}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AUTONOMY PATTERNS DEMONSTRATION")
    print("=" * 70)
    
    # Show risk assessment
    demonstrate_risk_assessment()
    
    # Demonstrate each autonomy level
    test_request = "Search for information about Python best practices and summarize what you find."
    
    print("\n\n" + "=" * 70)
    print("Testing different autonomy levels with the same request:")
    print(f"Request: {test_request}")
    print("=" * 70)
    
    # Uncomment the level you want to test:
    
    # Level 1: Every action confirmed
    # result = run_agent_confirmation_required(test_request)
    
    # Level 2: Checkpoint every 3 actions
    # result = run_agent_checkpoint_mode(test_request, checkpoint_interval=3)
    
    # Level 3: Only safe tools allowed
    result = run_agent_bounded_actions(
        test_request,
        allowed_tools=["search", "read_file", "calculate"]
    )
    
    # Level 4: Full autonomy (use carefully!)
    # result = run_agent_fully_autonomous(test_request)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print("=" * 60)
    print(result)
