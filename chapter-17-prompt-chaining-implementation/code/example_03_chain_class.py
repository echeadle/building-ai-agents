"""
Reusable Chain class for building prompt chains.

This module provides a flexible, reusable pattern for building
sequential prompt chains with built-in error handling and validation.

Features:
- Generic ChainStep base class for custom steps
- LLMStep for common LLM-based operations
- Full error handling with retries
- Detailed result tracking
- Declarative chain configuration

Chapter 17: Prompt Chaining - Implementation
"""

import os
import time
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Generic, TypeVar
from abc import ABC, abstractmethod
from enum import Enum

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Type variables for generic step types
T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type


class StepStatus(Enum):
    """Status of a chain step execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result from executing a chain step."""
    name: str
    status: StepStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    duration_seconds: float = 0.0


@dataclass 
class ChainResult:
    """Result from executing a complete chain."""
    success: bool
    steps: list[StepResult]
    final_output: Optional[Any] = None
    total_duration_seconds: float = 0.0
    
    def get_step(self, name: str) -> Optional[StepResult]:
        """Get a step result by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_output(self, step_name: str) -> Optional[Any]:
        """Get the output from a specific step."""
        step = self.get_step(step_name)
        return step.output if step else None


class ChainStep(ABC, Generic[T, U]):
    """
    Abstract base class for chain steps.
    
    Inherit from this to create custom steps with specific logic.
    Each step receives input of type T and produces output of type U.
    
    Example:
        class MyCustomStep(ChainStep[str, dict]):
            def execute(self, input_data: str) -> dict:
                return {"processed": input_data.upper()}
    """
    
    def __init__(self, name: str, max_retries: int = 2):
        """
        Initialize the step.
        
        Args:
            name: Human-readable name for this step
            max_retries: Number of retry attempts for validation failures
        """
        self.name = name
        self.max_retries = max_retries
    
    @abstractmethod
    def execute(self, input_data: T) -> U:
        """
        Execute the step logic.
        
        Override this method in subclasses to implement step behavior.
        
        Args:
            input_data: Input from the previous step (or initial input)
        
        Returns:
            Output to pass to the next step
        """
        pass
    
    def validate(self, output: U) -> bool:
        """
        Validate the step output.
        
        Override this method to add custom validation logic.
        Return False to trigger a retry.
        
        Args:
            output: The output to validate
        
        Returns:
            True if output is valid, False otherwise
        """
        return True
    
    def on_retry(self, attempt: int, error: Optional[str] = None) -> None:
        """
        Called before each retry attempt.
        
        Override to add custom retry behavior (e.g., adjust parameters).
        
        Args:
            attempt: The upcoming attempt number (1-indexed)
            error: Description of why the previous attempt failed
        """
        pass


class LLMStep(ChainStep[str, str]):
    """
    A chain step that makes an LLM call.
    
    This is the most common type of step in prompt chains.
    It takes a string input, formats it into a prompt, and returns
    the LLM's response.
    
    Example:
        step = LLMStep(
            name="Summarize",
            prompt_template="Summarize this text in 2 sentences: {input}",
            client=anthropic.Anthropic()
        )
        result = step.execute("Long text here...")
    """
    
    def __init__(
        self,
        name: str,
        prompt_template: str,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
        max_retries: int = 2
    ):
        """
        Initialize an LLM step.
        
        Args:
            name: Human-readable step name
            prompt_template: Template with {input} placeholder
            client: Anthropic client instance
            model: Model to use
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt
            validator: Optional function to validate output
            max_retries: Retry attempts for validation failures
        """
        super().__init__(name, max_retries)
        self.prompt_template = prompt_template
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._validator = validator
    
    def execute(self, input_data: str) -> str:
        """Execute the LLM call with the input data."""
        # Format the prompt with input data
        prompt = self.prompt_template.format(input=input_data)
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        
        if self.system_prompt:
            kwargs["system"] = self.system_prompt
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def validate(self, output: str) -> bool:
        """Validate using the provided validator function."""
        if self._validator:
            return self._validator(output)
        return True


class Chain:
    """
    A composable chain of steps that execute sequentially.
    
    The Chain class manages execution flow, error handling, retries,
    and result tracking for a sequence of steps.
    
    Example:
        chain = Chain("Content Pipeline")
        chain.add_step(GenerateStep(...))
        chain.add_step(TranslateStep(...))
        chain.add_step(FormatStep(...))
        
        result = chain.run("initial input")
        if result.success:
            print(result.final_output)
    """
    
    def __init__(self, name: str = "Chain"):
        """
        Initialize a new chain.
        
        Args:
            name: Human-readable name for this chain
        """
        self.name = name
        self.steps: list[ChainStep] = []
    
    def add_step(self, step: ChainStep) -> "Chain":
        """
        Add a step to the chain.
        
        Args:
            step: The step to add
        
        Returns:
            Self, for method chaining
        """
        self.steps.append(step)
        return self
    
    def run(
        self, 
        initial_input: Any,
        stop_on_failure: bool = True,
        verbose: bool = True
    ) -> ChainResult:
        """
        Execute all steps in sequence.
        
        Args:
            initial_input: The input to the first step
            stop_on_failure: If True, stop chain on first failure
            verbose: If True, print progress messages
        
        Returns:
            ChainResult with all step results and final output
        """
        start_time = time.time()
        step_results = []
        current_input = initial_input
        success = True
        
        for i, step in enumerate(self.steps):
            if verbose:
                print(f"\n[{i+1}/{len(self.steps)}] {step.name}")
            
            result = self._execute_step(step, current_input, verbose)
            step_results.append(result)
            
            if result.status == StepStatus.SUCCESS:
                if verbose:
                    print(f"  ✓ Completed in {result.duration_seconds:.2f}s")
                current_input = result.output
            else:
                if verbose:
                    print(f"  ✗ Failed: {result.error}")
                success = False
                
                if stop_on_failure:
                    # Mark remaining steps as skipped
                    for remaining_step in self.steps[i+1:]:
                        step_results.append(StepResult(
                            name=remaining_step.name,
                            status=StepStatus.SKIPPED,
                            error="Previous step failed"
                        ))
                    break
        
        return ChainResult(
            success=success,
            steps=step_results,
            final_output=current_input if success else None,
            total_duration_seconds=time.time() - start_time
        )
    
    def _execute_step(
        self, 
        step: ChainStep, 
        input_data: Any,
        verbose: bool = True
    ) -> StepResult:
        """Execute a single step with retry logic."""
        start_time = time.time()
        result = StepResult(name=step.name, status=StepStatus.RUNNING)
        
        for attempt in range(step.max_retries + 1):
            result.attempts = attempt + 1
            
            try:
                output = step.execute(input_data)
                
                if not step.validate(output):
                    if attempt < step.max_retries:
                        step.on_retry(attempt + 1, "Validation failed")
                        if verbose:
                            print(f"  Validation failed, retrying ({attempt + 1}/{step.max_retries})...")
                        continue
                    result.status = StepStatus.FAILED
                    result.error = "Output failed validation after all retries"
                    result.output = output
                    break
                
                result.status = StepStatus.SUCCESS
                result.output = output
                break
                
            except anthropic.RateLimitError:
                if attempt < step.max_retries:
                    delay = 2 ** attempt
                    if verbose:
                        print(f"  Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    step.on_retry(attempt + 1, "Rate limited")
                    continue
                result.status = StepStatus.FAILED
                result.error = "Rate limit exceeded after retries"
                
            except anthropic.APIConnectionError as e:
                if attempt < step.max_retries:
                    delay = 2 ** attempt
                    if verbose:
                        print(f"  Connection error, waiting {delay}s...")
                    time.sleep(delay)
                    step.on_retry(attempt + 1, f"Connection error: {e}")
                    continue
                result.status = StepStatus.FAILED
                result.error = f"Connection error: {str(e)}"
                
            except anthropic.APIError as e:
                result.status = StepStatus.FAILED
                result.error = f"API error: {str(e)}"
                break
                
            except Exception as e:
                result.status = StepStatus.FAILED
                result.error = f"Unexpected error: {type(e).__name__}: {str(e)}"
                break
        
        result.duration_seconds = time.time() - start_time
        return result


def create_llm_chain(
    steps: list[dict],
    client: Optional[anthropic.Anthropic] = None,
    name: str = "LLM Chain"
) -> Chain:
    """
    Convenience function to create a chain from configuration dictionaries.
    
    This allows declarative chain definition without manually creating
    step objects.
    
    Args:
        steps: List of dicts with keys:
            - name: Step name (required)
            - prompt_template: Template with {input} placeholder (required)
            - system_prompt: Optional system prompt
            - validator: Optional validation function
            - max_retries: Retry count (default: 2)
        client: Anthropic client (creates one if not provided)
        name: Name for the chain
    
    Returns:
        Configured Chain ready to run
    
    Example:
        chain = create_llm_chain([
            {"name": "Step 1", "prompt_template": "Process: {input}"},
            {"name": "Step 2", "prompt_template": "Refine: {input}"}
        ])
        result = chain.run("initial input")
    """
    if client is None:
        client = anthropic.Anthropic()
    
    chain = Chain(name)
    
    for step_config in steps:
        step = LLMStep(
            name=step_config["name"],
            prompt_template=step_config["prompt_template"],
            client=client,
            system_prompt=step_config.get("system_prompt"),
            validator=step_config.get("validator"),
            max_retries=step_config.get("max_retries", 2)
        )
        chain.add_step(step)
    
    return chain


# Example usage
if __name__ == "__main__":
    client = anthropic.Anthropic()
    
    # Define a three-step content chain
    chain = create_llm_chain([
        {
            "name": "Generate Marketing Copy",
            "prompt_template": """Write a compelling 3-4 sentence marketing paragraph about: {input}

Include clear benefits and a call to action.""",
            "validator": lambda x: len(x) >= 50  # At least 50 chars
        },
        {
            "name": "Translate to Spanish",
            "prompt_template": """Translate the following marketing copy to Spanish.
Maintain the same persuasive tone and adapt idioms naturally.

Text to translate:
{input}""",
            "validator": lambda x: len(x) >= 50
        },
        {
            "name": "Create Social Media Version",
            "prompt_template": """Create a short social media post (under 280 characters) based on this content.
Keep the key message but make it punchy and engaging. Use the same language as the input.

Original content:
{input}""",
            "validator": lambda x: len(x) <= 300 and len(x) >= 20
        }
    ], client, name="Content Pipeline")
    
    # Run the chain
    print("="*50)
    print("EXECUTING CONTENT CHAIN")
    print("="*50)
    
    result = chain.run("eco-friendly reusable coffee cups made from recycled ocean plastic")
    
    # Display results
    print("\n" + "="*50)
    print("CHAIN RESULTS")
    print("="*50)
    print(f"\nChain: {chain.name}")
    print(f"Success: {result.success}")
    print(f"Total duration: {result.total_duration_seconds:.2f}s")
    
    for step_result in result.steps:
        status_icon = {
            StepStatus.SUCCESS: "✓",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "○"
        }.get(step_result.status, "?")
        
        print(f"\n{status_icon} {step_result.name}")
        print(f"  Status: {step_result.status.value}")
        print(f"  Attempts: {step_result.attempts}")
        print(f"  Duration: {step_result.duration_seconds:.2f}s")
        
        if step_result.output:
            # Truncate long outputs for display
            output_preview = step_result.output
            if len(output_preview) > 200:
                output_preview = output_preview[:200] + "..."
            print(f"  Output: {output_preview}")
        if step_result.error:
            print(f"  Error: {step_result.error}")
    
    # Show final output
    if result.final_output:
        print("\n" + "="*50)
        print("FINAL OUTPUT (Social Media Post)")
        print("="*50)
        print(result.final_output)
