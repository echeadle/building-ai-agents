"""
Agent configuration system.

Chapter 33: The Complete Agent Class

This module provides a comprehensive configuration system for agents,
including validation, presets, and type-safe settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PlanningMode(Enum):
    """
    How the agent approaches planning.
    
    NONE: No explicit planning - agent responds directly
    SIMPLE: Plan once at the start, then execute steps
    ADAPTIVE: Plan and revise as new information comes in
    """
    NONE = "none"
    SIMPLE = "simple"
    ADAPTIVE = "adaptive"


class HumanApprovalMode(Enum):
    """
    When to request human approval for actions.
    
    NEVER: Fully autonomous - no approval needed
    HIGH_RISK: Only ask for dangerous/sensitive actions
    ALWAYS: Every action requires explicit approval
    """
    NEVER = "never"
    HIGH_RISK = "high_risk"
    ALWAYS = "always"


@dataclass
class AgentConfig:
    """
    Configuration for an Agent instance.
    
    This dataclass holds all configurable parameters for an agent,
    with sensible defaults that work for most use cases.
    
    Example:
        >>> config = AgentConfig(
        ...     system_prompt="You are a helpful assistant.",
        ...     max_iterations=10,
        ...     verbose=True
        ... )
        >>> errors = config.validate()
        >>> if not errors:
        ...     agent = Agent(config)
    """
    
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # System prompt - the agent's "constitution"
    system_prompt: str = "You are a helpful AI assistant."
    
    # Loop controls - prevent runaway execution
    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 5
    
    # Planning configuration
    planning_mode: PlanningMode = PlanningMode.SIMPLE
    
    # Human-in-the-loop settings
    approval_mode: HumanApprovalMode = HumanApprovalMode.HIGH_RISK
    high_risk_actions: list[str] = field(default_factory=lambda: [
        "delete", "send_email", "make_purchase", "execute_code",
        "modify_file", "run_command", "transfer_funds"
    ])
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True
    
    # Guardrails - safety settings
    input_validation_enabled: bool = True
    output_filtering_enabled: bool = True
    action_constraints_enabled: bool = True
    allowed_tools: Optional[list[str]] = None  # None means all tools allowed
    blocked_patterns: list[str] = field(default_factory=lambda: [
        r"password", r"credit.?card", r"ssn", r"social.?security",
        r"api.?key", r"secret", r"private.?key"
    ])
    
    # State management
    max_history_tokens: int = 8000
    persist_state: bool = False
    state_file: Optional[str] = None
    
    # Observability - logging and debugging
    verbose: bool = False
    log_tool_calls: bool = True
    log_llm_responses: bool = False  # Can be expensive/verbose
    
    def validate(self) -> list[str]:
        """
        Validate configuration settings.
        
        Checks all settings for valid values and consistency.
        
        Returns:
            List of validation error messages. Empty list if valid.
            
        Example:
            >>> config = AgentConfig(max_iterations=-1)
            >>> errors = config.validate()
            >>> print(errors)
            ['max_iterations must be at least 1']
        """
        errors = []
        
        # Validate numeric bounds
        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        
        if self.max_iterations > 100:
            errors.append("max_iterations should not exceed 100 for safety")
        
        if self.max_tokens < 100:
            errors.append("max_tokens should be at least 100")
        
        if self.max_tokens > 100000:
            errors.append("max_tokens exceeds maximum allowed value")
        
        if self.temperature < 0 or self.temperature > 1:
            errors.append("temperature must be between 0 and 1")
        
        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")
        
        if self.retry_delay < 0:
            errors.append("retry_delay cannot be negative")
        
        if self.max_tool_calls_per_iteration < 1:
            errors.append("max_tool_calls_per_iteration must be at least 1")
        
        if self.max_history_tokens < 1000:
            errors.append("max_history_tokens should be at least 1000")
        
        # Validate consistency
        if self.persist_state and not self.state_file:
            errors.append("state_file required when persist_state is True")
        
        if not self.system_prompt or not self.system_prompt.strip():
            errors.append("system_prompt cannot be empty")
        
        return errors
    
    @classmethod
    def for_simple_chat(cls) -> "AgentConfig":
        """
        Configuration preset for simple chat agents.
        
        Best for: Basic Q&A, simple conversations, no tools needed.
        
        Returns:
            AgentConfig configured for simple chat
        """
        return cls(
            planning_mode=PlanningMode.NONE,
            approval_mode=HumanApprovalMode.NEVER,
            max_iterations=1,
            max_tool_calls_per_iteration=1,
            verbose=False
        )
    
    @classmethod
    def for_autonomous_agent(cls) -> "AgentConfig":
        """
        Configuration preset for autonomous agents.
        
        Best for: Complex tasks, research, multi-step workflows.
        Includes persistence and adaptive planning.
        
        Returns:
            AgentConfig configured for autonomous operation
        """
        return cls(
            planning_mode=PlanningMode.ADAPTIVE,
            approval_mode=HumanApprovalMode.HIGH_RISK,
            max_iterations=20,
            max_tool_calls_per_iteration=10,
            persist_state=True,
            state_file="agent_state.json",
            verbose=True
        )
    
    @classmethod
    def for_safe_agent(cls) -> "AgentConfig":
        """
        Configuration preset for maximum safety.
        
        Best for: Sensitive operations, production, customer-facing.
        All guardrails enabled, human approval required.
        
        Returns:
            AgentConfig configured for maximum safety
        """
        return cls(
            planning_mode=PlanningMode.SIMPLE,
            approval_mode=HumanApprovalMode.ALWAYS,
            max_iterations=5,
            max_tool_calls_per_iteration=3,
            input_validation_enabled=True,
            output_filtering_enabled=True,
            action_constraints_enabled=True,
            verbose=True
        )
    
    @classmethod
    def for_development(cls) -> "AgentConfig":
        """
        Configuration preset for development and testing.
        
        Best for: Debugging, testing, development environments.
        Maximum verbosity, no safety restrictions.
        
        Returns:
            AgentConfig configured for development
        """
        return cls(
            planning_mode=PlanningMode.SIMPLE,
            approval_mode=HumanApprovalMode.NEVER,
            max_iterations=5,
            verbose=True,
            log_tool_calls=True,
            log_llm_responses=True,
            input_validation_enabled=False,
            output_filtering_enabled=False,
            action_constraints_enabled=False
        )
    
    def __post_init__(self):
        """Ensure enum types are correct after initialization."""
        if isinstance(self.planning_mode, str):
            self.planning_mode = PlanningMode(self.planning_mode)
        if isinstance(self.approval_mode, str):
            self.approval_mode = HumanApprovalMode(self.approval_mode)


if __name__ == "__main__":
    # Demonstrate configuration usage
    print("=== AgentConfig Demonstration ===\n")
    
    # Default configuration
    default_config = AgentConfig()
    print(f"Default config: {default_config.model}, iterations={default_config.max_iterations}")
    
    # Validate a bad configuration
    bad_config = AgentConfig(max_iterations=-5, temperature=2.0)
    errors = bad_config.validate()
    print(f"\nBad config errors: {errors}")
    
    # Configuration presets
    print("\n=== Configuration Presets ===")
    
    chat = AgentConfig.for_simple_chat()
    print(f"Simple chat: planning={chat.planning_mode.value}, iterations={chat.max_iterations}")
    
    auto = AgentConfig.for_autonomous_agent()
    print(f"Autonomous: planning={auto.planning_mode.value}, iterations={auto.max_iterations}")
    
    safe = AgentConfig.for_safe_agent()
    print(f"Safe: approval={safe.approval_mode.value}, iterations={safe.max_iterations}")
    
    dev = AgentConfig.for_development()
    print(f"Development: verbose={dev.verbose}, validation={dev.input_validation_enabled}")
