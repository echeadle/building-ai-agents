"""
The Complete Agent Class.

Chapter 33: The Complete Agent Class

This module provides a production-ready Agent class that integrates:
- Configurable behavior through AgentConfig
- Tool management through ToolRegistry
- State management through AgentState
- Safety through Guardrails
- Reliability through ErrorHandler

This is the culmination of Part 4: Building True Agents.
"""

import os
from typing import Any, Generator
from datetime import datetime
from dotenv import load_dotenv
import anthropic

from config import AgentConfig, PlanningMode, HumanApprovalMode
from tools import ToolRegistry
from state import AgentState
from guardrails import Guardrails
from errors import ErrorHandler, ErrorSeverity

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class Agent:
    """
    A complete, production-ready AI agent.
    
    This class integrates all agentic components into a cohesive system
    that can perceive, plan, act, and learn from its environment.
    
    Features:
    - Configurable behavior through AgentConfig
    - Multiple tool support through ToolRegistry  
    - Persistent state management through AgentState
    - Safety guardrails through Guardrails
    - Robust error handling through ErrorHandler
    - Human-in-the-loop approval for sensitive actions
    - Planning capabilities for complex tasks
    - Streaming support for real-time responses
    
    Example:
        >>> config = AgentConfig.for_autonomous_agent()
        >>> agent = Agent(config)
        >>> agent.register_tool("calculator", ...)
        >>> response = agent.run("What is 25 * 17?")
        >>> print(response)
    """
    
    def __init__(
        self,
        config: AgentConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        state: AgentState | None = None
    ):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration. Uses defaults if not provided.
            tool_registry: Pre-configured tool registry. Creates new if not provided.
            state: Existing state to resume. Creates new if not provided.
            
        Raises:
            ValueError: If configuration is invalid or API key is missing
        """
        self.config = config or AgentConfig()
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
        
        # Initialize components
        self.tools = tool_registry or ToolRegistry()
        self.state = state or AgentState(
            max_history_tokens=self.config.max_history_tokens
        )
        self.guardrails = Guardrails(
            blocked_patterns=self.config.blocked_patterns,
            allowed_tools=self.config.allowed_tools
        )
        self.error_handler = ErrorHandler(
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            fallback_enabled=self.config.fallback_enabled
        )
        
        # Initialize API client
        self.client = anthropic.Anthropic()
        
        # Track execution metrics
        self.metrics = {
            "total_iterations": 0,
            "total_tool_calls": 0,
            "total_tokens_used": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_planning_calls": 0
        }
        
        self._log("Agent initialized", {
            "model": self.config.model,
            "planning_mode": self.config.planning_mode.value,
            "approval_mode": self.config.approval_mode.value
        })
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: callable,
        requires_approval: bool = False
    ) -> None:
        """
        Register a tool for the agent to use.
        
        Args:
            name: Unique tool name
            description: What the tool does (the LLM reads this!)
            input_schema: JSON Schema for parameters
            handler: Function to execute when tool is called
            requires_approval: Whether tool needs human approval
            
        Example:
            >>> agent.register_tool(
            ...     name="search",
            ...     description="Search the web for information",
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "query": {"type": "string", "description": "Search query"}
            ...         },
            ...         "required": ["query"]
            ...     },
            ...     handler=search_function,
            ...     requires_approval=False
            ... )
        """
        self.tools.register(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            requires_approval=requires_approval
        )
        self._log(f"Tool registered: {name}")
    
    def run(self, user_input: str) -> str:
        """
        Run the agent with the given user input.
        
        This is the main entry point for interacting with the agent.
        It handles the complete agentic loop including planning,
        tool use, and response generation.
        
        Args:
            user_input: The user's message or request
            
        Returns:
            The agent's final response
            
        Example:
            >>> response = agent.run("What's the weather in New York?")
            >>> print(response)
        """
        # Input validation
        if self.config.input_validation_enabled:
            validation = self.guardrails.validate_input(user_input)
            if not validation.passed:
                self._log(f"Input validation failed: {validation.reason}")
                return f"I can't process that request: {validation.reason}"
        
        # Add user message to state
        self.state.add_message("user", user_input)
        self._log(f"User input received: {user_input[:100]}...")
        
        try:
            # Planning phase (if enabled)
            if self.config.planning_mode != PlanningMode.NONE:
                self._create_plan(user_input)
            
            # Execute the agentic loop
            response = self._run_agentic_loop()
            
            # Output filtering
            if self.config.output_filtering_enabled:
                filter_result = self.guardrails.filter_output(response)
                response = filter_result.modified_content
            
            self.metrics["successful_runs"] += 1
            self._log(f"Run completed successfully")
            return response
            
        except Exception as e:
            self.metrics["failed_runs"] += 1
            error = self.error_handler.categorize_error(e)
            self._log(f"Run failed: {error}", level="error")
            
            if error.severity == ErrorSeverity.CRITICAL:
                raise
            
            return f"I encountered an error: {error.message}"
        
        finally:
            # Save state if persistence is enabled
            if self.config.persist_state and self.config.state_file:
                try:
                    self.state.save(self.config.state_file)
                    self._log(f"State saved to {self.config.state_file}")
                except Exception as e:
                    self._log(f"Failed to save state: {e}", level="error")
    
    def run_streaming(self, user_input: str) -> Generator[str, None, None]:
        """
        Run the agent with streaming output.
        
        Yields chunks of the response as they're generated.
        Note: Tool use is handled non-streaming, only final response streams.
        
        Args:
            user_input: The user's message or request
            
        Yields:
            Chunks of the agent's response
            
        Example:
            >>> for chunk in agent.run_streaming("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        # Input validation
        if self.config.input_validation_enabled:
            validation = self.guardrails.validate_input(user_input)
            if not validation.passed:
                yield f"I can't process that request: {validation.reason}"
                return
        
        self.state.add_message("user", user_input)
        
        # For streaming, we use a simplified single-turn approach
        messages = self.state.get_messages_for_api()
        tools = self.tools.get_definitions_for_api(self.config.allowed_tools)
        
        try:
            with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                messages=messages,
                tools=tools if tools else anthropic.NOT_GIVEN
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text
                
                self.state.add_message("assistant", full_response)
                self.metrics["successful_runs"] += 1
                
        except Exception as e:
            self.metrics["failed_runs"] += 1
            error = self.error_handler.categorize_error(e)
            yield f"\n\nError: {error.message}"
    
    def _create_plan(self, user_input: str) -> None:
        """
        Create an execution plan for the task.
        
        Args:
            user_input: The user's request to plan for
        """
        self._log("Creating plan...")
        
        # Get available tools for context
        available_tools = self.tools.list_names()
        tools_context = f"Available tools: {', '.join(available_tools)}" if available_tools else "No tools available"
        
        planning_prompt = f"""Analyze this request and create a step-by-step plan.

Request: {user_input}

{tools_context}

Create a numbered list of 3-7 concrete steps to accomplish this task.
Each step should be specific and actionable.
Consider what tools might be needed.

Respond with just the numbered steps, nothing else."""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": planning_prompt}]
            )
            
            self.metrics["total_planning_calls"] += 1
            self.metrics["total_tokens_used"] += (
                response.usage.input_tokens + response.usage.output_tokens
            )
            
            plan_text = response.content[0].text
            
            # Parse steps from response
            steps = []
            for line in plan_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    steps.append(line)
            
            if steps:
                self.state.set_plan(steps)
                self._log(f"Plan created with {len(steps)} steps")
            else:
                self._log("Could not parse plan from response", level="warning")
                
        except Exception as e:
            self._log(f"Planning failed: {e}", level="error")
            # Continue without plan
    
    def _run_agentic_loop(self) -> str:
        """
        Execute the main agentic loop.
        
        Iteratively:
        1. Get LLM response
        2. If tool calls, execute them and continue
        3. If no tool calls, return the response
        
        Returns:
            The final response after all iterations
        """
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            self.metrics["total_iterations"] += 1
            
            self._log(f"Iteration {iteration}/{self.config.max_iterations}")
            
            # Get current messages and tools
            messages = self.state.get_messages_for_api()
            tools = self.tools.get_definitions_for_api(self.config.allowed_tools)
            
            # Make API call with retry
            response, error = self.error_handler.with_retry(
                self._call_llm,
                messages,
                tools
            )
            
            if error:
                return f"Failed after retries: {error.message}"
            
            # Process response
            assistant_content = response.content
            stop_reason = response.stop_reason
            
            # Track token usage
            self.metrics["total_tokens_used"] += (
                response.usage.input_tokens + response.usage.output_tokens
            )
            
            # Check if we're done (no tool use)
            if stop_reason == "end_turn":
                # Extract text response
                text_parts = [
                    block.text 
                    for block in assistant_content 
                    if hasattr(block, "text")
                ]
                final_response = "\n".join(text_parts)
                self.state.add_message("assistant", final_response)
                self._log("Agentic loop completed")
                return final_response
            
            # Handle tool use
            if stop_reason == "tool_use":
                # Store assistant's response with tool calls
                self.state.add_message("assistant", assistant_content)
                
                # Process each tool call
                tool_results = self._process_tool_calls(assistant_content)
                
                # Add tool results to messages
                self.state.add_message("user", tool_results)
        
        # Max iterations reached
        self._log("Max iterations reached", level="warning")
        return "I wasn't able to complete the task within the allowed steps. Here's what I accomplished so far."
    
    def _call_llm(self, messages: list[dict], tools: list[dict]) -> Any:
        """
        Make an API call to the LLM.
        
        Args:
            messages: Conversation messages
            tools: Tool definitions
            
        Returns:
            API response
        """
        return self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            messages=messages,
            tools=tools if tools else anthropic.NOT_GIVEN
        )
    
    def _process_tool_calls(self, assistant_content: list) -> list[dict]:
        """
        Process tool calls from the assistant's response.
        
        Args:
            assistant_content: Content blocks from assistant response
            
        Returns:
            List of tool results in API format
        """
        tool_results = []
        tool_calls_this_turn = 0
        
        for block in assistant_content:
            if block.type != "tool_use":
                continue
            
            # Check tool call limit
            if tool_calls_this_turn >= self.config.max_tool_calls_per_iteration:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Maximum tool calls per turn exceeded. Please continue with available results."
                })
                continue
            
            tool_name = block.name
            tool_input = block.input
            
            self._log(f"Tool call: {tool_name}", {"arguments": tool_input})
            
            # Check if tool is allowed
            if self.config.action_constraints_enabled:
                allowed = self.guardrails.check_tool_allowed(tool_name)
                if not allowed.passed:
                    self._log(f"Tool blocked: {allowed.reason}", level="warning")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool not allowed: {allowed.reason}",
                        "is_error": True
                    })
                    continue
            
            # Check action safety
            if self.config.action_constraints_enabled:
                safety = self.guardrails.check_action_safety(tool_name, tool_input)
                if not safety.passed:
                    self._log(f"Action blocked: {safety.reason}", level="warning")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Action blocked for safety: {safety.reason}",
                        "is_error": True
                    })
                    continue
            
            # Check if approval is needed
            if self._needs_approval(tool_name):
                approved = self._request_approval(tool_name, tool_input)
                if not approved:
                    self._log(f"Tool call rejected by user: {tool_name}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Action was not approved by user.",
                        "is_error": True
                    })
                    continue
            
            # Execute the tool
            try:
                result = self.tools.execute(tool_name, tool_input)
                
                # Validate result
                validation = self.guardrails.validate_tool_result(result)
                result = validation.modified_content
                
                self.state.add_tool_call(
                    tool_name=tool_name,
                    arguments=tool_input,
                    result=result,
                    success=True
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
                
                self._log(f"Tool completed: {tool_name}")
                
            except Exception as e:
                fallback, success = self.error_handler.handle_tool_error(
                    tool_name, e, fallback_result="Tool execution failed."
                )
                
                self.state.add_tool_call(
                    tool_name=tool_name,
                    arguments=tool_input,
                    result=None,
                    success=False,
                    error=str(e)
                )
                
                self._log(f"Tool failed: {tool_name} - {e}", level="error")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(fallback),
                    "is_error": True
                })
            
            tool_calls_this_turn += 1
            self.metrics["total_tool_calls"] += 1
        
        return tool_results
    
    def _needs_approval(self, tool_name: str) -> bool:
        """
        Check if a tool call needs human approval.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if approval is needed
        """
        if self.config.approval_mode == HumanApprovalMode.NEVER:
            return False
        
        if self.config.approval_mode == HumanApprovalMode.ALWAYS:
            return True
        
        # HIGH_RISK mode - check tool and patterns
        if self.tools.has(tool_name) and self.tools.requires_approval(tool_name):
            return True
        
        # Check if tool name matches high-risk patterns
        for pattern in self.config.high_risk_actions:
            if pattern.lower() in tool_name.lower():
                return True
        
        return False
    
    def _request_approval(self, tool_name: str, tool_input: dict) -> bool:
        """
        Request human approval for a tool call.
        
        Args:
            tool_name: Name of the tool
            tool_input: Tool arguments
            
        Returns:
            True if approved, False otherwise
        """
        print("\n" + "=" * 60)
        print("🚨 APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Tool: {tool_name}")
        print(f"Arguments: {tool_input}")
        print("=" * 60)
        
        while True:
            try:
                response = input("Approve this action? (yes/no): ").strip().lower()
                if response in ("yes", "y"):
                    self._log(f"User approved: {tool_name}")
                    return True
                elif response in ("no", "n"):
                    self._log(f"User rejected: {tool_name}")
                    return False
                print("Please enter 'yes' or 'no'")
            except (EOFError, KeyboardInterrupt):
                return False
    
    def _log(self, message: str, context: dict | None = None, level: str = "info") -> None:
        """
        Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log
            context: Additional context
            level: Log level (info, warning, error)
        """
        if not self.config.verbose:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌"
        }.get(level, "•")
        
        log_line = f"[{timestamp}] {prefix} {message}"
        if context:
            log_line += f" | {context}"
        
        print(log_line)
    
    # Public utilities
    
    def reset(self) -> None:
        """
        Reset the agent's state for a new conversation.
        
        Clears conversation history, tool calls, and working memory.
        """
        self.state = AgentState(
            max_history_tokens=self.config.max_history_tokens
        )
        self.error_handler.clear_history()
        self._log("Agent state reset")
    
    def get_metrics(self) -> dict:
        """
        Get execution metrics.
        
        Returns:
            Dictionary with metrics including iterations, tool calls,
            tokens used, and success/failure counts
        """
        return self.metrics.copy()
    
    def get_state_summary(self) -> str:
        """
        Get a summary of the current state.
        
        Returns:
            Human-readable state summary
        """
        return self.state.get_summary()
    
    def get_error_summary(self) -> str:
        """
        Get a summary of errors encountered.
        
        Returns:
            Human-readable error summary
        """
        return self.error_handler.get_error_summary()
    
    def get_conversation_history(self) -> list[dict]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.state.get_messages_for_api()
    
    def add_to_memory(self, key: str, value: Any) -> None:
        """
        Add a value to the agent's working memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        self.state.set_memory(key, value)
    
    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the agent's working memory.
        
        Args:
            key: Memory key
            default: Default value if not found
            
        Returns:
            Stored value or default
        """
        return self.state.get_memory(key, default)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Agent(model={self.config.model}, "
            f"tools={len(self.tools.list_names())}, "
            f"planning={self.config.planning_mode.value})"
        )


if __name__ == "__main__":
    # Quick demonstration
    print("=== Agent Class Demonstration ===\n")
    
    # Create agent with verbose output
    config = AgentConfig(
        system_prompt="You are a helpful assistant with access to tools. Be concise.",
        planning_mode=PlanningMode.SIMPLE,
        max_iterations=5,
        verbose=True
    )
    
    agent = Agent(config)
    
    # Register a simple tool
    agent.register_tool(
        name="get_time",
        description="Get the current date and time",
        input_schema={"type": "object", "properties": {}},
        handler=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print(f"\nAgent: {agent}")
    print(f"Available tools: {agent.tools.list_names()}")
    
    # Run a simple query
    print("\n" + "=" * 60)
    response = agent.run("What time is it right now?")
    print("\n" + "=" * 60)
    print(f"Final Response:\n{response}")
    
    # Show metrics
    print("\n=== Metrics ===")
    for key, value in agent.get_metrics().items():
        print(f"  {key}: {value}")
