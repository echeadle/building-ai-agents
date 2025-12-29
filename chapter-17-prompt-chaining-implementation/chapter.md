---
chapter: 17
title: "Prompt Chaining - Implementation"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 17: Prompt Chaining - Implementation

## Introduction

In the previous chapter, we explored the concept of prompt chaining—breaking complex tasks into a series of simpler, focused steps. We discussed when to use chaining, how to identify natural task boundaries, and the importance of quality gates between steps. Now it's time to build it.

This chapter transforms those concepts into working code. We'll start with a simple two-step chain, add quality gates for validation between steps, and ultimately create a reusable `Chain` class that you can adapt for any sequential workflow. By the end, you'll have a pattern you can apply to content pipelines, data processing, multi-stage analysis, and countless other use cases.

This chapter builds on the `AugmentedLLM` concepts from Part 2 and serves as our first hands-on implementation of an agentic workflow pattern. The patterns we establish here will reappear throughout the remaining workflow chapters.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement a multi-step prompt chain that passes data between steps
- Add quality gates that validate outputs before proceeding to the next step
- Handle errors gracefully at each stage of the chain
- Build a reusable `Chain` class for any sequential workflow
- Debug chains by inspecting intermediate outputs

## A Simple Two-Step Chain

Let's start with the most basic chain possible: content generation followed by translation. This example demonstrates the core mechanics without any additional complexity.

### The Scenario

Imagine you need to create marketing copy and then translate it to Spanish. You could ask the LLM to do both in a single prompt, but splitting it into two steps gives you:

1. **Better quality** — Each step can focus on doing one thing well
2. **Inspectability** — You can review the English copy before translation
3. **Flexibility** — You can swap out the translator or add more languages later

### Basic Implementation

Here's the simplest possible chain:

```python
"""
Simple two-step prompt chain: Generate content, then translate it.

Chapter 17: Prompt Chaining - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


def generate_content(topic: str, style: str = "professional") -> str:
    """
    Step 1: Generate marketing copy about a topic.
    
    Args:
        topic: The subject to write about
        style: The tone of the content (professional, casual, enthusiastic)
    
    Returns:
        Generated marketing copy as a string
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Write a short marketing paragraph (3-4 sentences) about: {topic}
                
Style: {style}

Focus on benefits and include a call to action. Keep it concise."""
            }
        ]
    )
    return response.content[0].text


def translate_content(content: str, target_language: str) -> str:
    """
    Step 2: Translate content to another language.
    
    Args:
        content: The text to translate
        target_language: The language to translate into
    
    Returns:
        Translated content as a string
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Translate the following marketing copy to {target_language}.

Maintain the same tone, style, and persuasive intent. Adapt idioms naturally.

Content to translate:
{content}"""
            }
        ]
    )
    return response.content[0].text


def content_chain(topic: str, target_language: str, style: str = "professional") -> dict:
    """
    Execute the full content generation and translation chain.
    
    Args:
        topic: The subject to write about
        target_language: The language to translate into
        style: The tone of the content
    
    Returns:
        Dictionary containing original and translated content
    """
    # Step 1: Generate content
    print(f"Step 1: Generating {style} content about '{topic}'...")
    original_content = generate_content(topic, style)
    print(f"Generated: {original_content[:100]}...")
    
    # Step 2: Translate content
    print(f"\nStep 2: Translating to {target_language}...")
    translated_content = translate_content(original_content, target_language)
    print(f"Translated: {translated_content[:100]}...")
    
    return {
        "topic": topic,
        "style": style,
        "target_language": target_language,
        "original": original_content,
        "translated": translated_content
    }


if __name__ == "__main__":
    result = content_chain(
        topic="cloud-based project management software",
        target_language="Spanish",
        style="enthusiastic"
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"\nOriginal ({result['style']}):\n{result['original']}")
    print(f"\nTranslated ({result['target_language']}):\n{result['translated']}")
```

### Understanding the Flow

Notice the structure:

1. **Each step is a focused function** — `generate_content()` only generates, `translate_content()` only translates
2. **Output becomes input** — The result of step 1 (`original_content`) feeds directly into step 2
3. **The chain function orchestrates** — `content_chain()` coordinates the steps and manages the data flow
4. **Everything is inspectable** — Print statements show what's happening at each stage

This is the essence of prompt chaining: simple functions composed into a pipeline.

## Adding Quality Gates

The basic chain works, but what if step 1 produces poor content? The translation step will dutifully translate garbage into garbage in another language. **Quality gates** prevent bad outputs from propagating through the chain.

### What Quality Gates Check

Quality gates can validate:

- **Length** — Is the output within expected bounds?
- **Format** — Does it match the expected structure?
- **Content** — Does it contain required elements?
- **Quality** — Does it meet a minimum standard (often judged by another LLM call)?

### Implementation with Validation

Let's add quality gates to our chain:

```python
"""
Prompt chain with quality gates between steps.

Chapter 17: Prompt Chaining - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass
from typing import Optional

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class QualityCheckResult:
    """Result of a quality gate check."""
    passed: bool
    message: str
    score: Optional[float] = None


class ChainError(Exception):
    """Raised when a chain step fails validation."""
    def __init__(self, step: str, message: str, output: str):
        self.step = step
        self.output = output
        super().__init__(f"Chain failed at '{step}': {message}")


def check_content_quality(content: str, min_length: int = 50) -> QualityCheckResult:
    """
    Quality gate: Check if generated content meets basic requirements.
    
    This is a simple rule-based check. For more sophisticated validation,
    you could use another LLM call to evaluate quality.
    """
    # Check minimum length
    if len(content) < min_length:
        return QualityCheckResult(
            passed=False,
            message=f"Content too short: {len(content)} chars (minimum: {min_length})"
        )
    
    # Check for call to action (simple heuristic)
    cta_indicators = ["try", "start", "get", "discover", "learn", "join", "contact", "visit"]
    has_cta = any(word in content.lower() for word in cta_indicators)
    
    if not has_cta:
        return QualityCheckResult(
            passed=False,
            message="Content missing call to action"
        )
    
    return QualityCheckResult(
        passed=True,
        message="Content meets quality requirements"
    )


def check_translation_quality(
    original: str, 
    translated: str, 
    target_language: str
) -> QualityCheckResult:
    """
    Quality gate: Use LLM to verify translation quality.
    
    This demonstrates using an LLM as a quality checker—a powerful pattern
    for validating complex outputs.
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"""Evaluate this translation. Rate it 1-10 and identify any issues.

Original (English):
{original}

Translation ({target_language}):
{translated}

Respond in this exact format:
SCORE: [1-10]
PASSED: [YES/NO]
ISSUES: [Brief description or "None"]"""
            }
        ]
    )
    
    result_text = response.content[0].text
    
    # Parse the response
    try:
        lines = result_text.strip().split("\n")
        score_line = [l for l in lines if l.startswith("SCORE:")][0]
        passed_line = [l for l in lines if l.startswith("PASSED:")][0]
        issues_line = [l for l in lines if l.startswith("ISSUES:")][0]
        
        score = float(score_line.split(":")[1].strip())
        passed = "YES" in passed_line.upper()
        issues = issues_line.split(":", 1)[1].strip()
        
        return QualityCheckResult(
            passed=passed,
            message=issues if issues != "None" else "Translation approved",
            score=score
        )
    except (IndexError, ValueError) as e:
        # If parsing fails, be conservative and pass
        return QualityCheckResult(
            passed=True,
            message=f"Could not parse quality check (assuming pass): {result_text[:100]}"
        )


def generate_content(topic: str, style: str = "professional") -> str:
    """Step 1: Generate marketing copy."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Write a short marketing paragraph (3-4 sentences) about: {topic}
                
Style: {style}

Requirements:
- Focus on benefits
- Include a clear call to action
- Keep it concise but compelling"""
            }
        ]
    )
    return response.content[0].text


def translate_content(content: str, target_language: str) -> str:
    """Step 2: Translate content to another language."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Translate the following marketing copy to {target_language}.

Maintain the same tone, style, and persuasive intent. Adapt idioms naturally.

Content to translate:
{content}"""
            }
        ]
    )
    return response.content[0].text


def content_chain_with_gates(
    topic: str, 
    target_language: str, 
    style: str = "professional",
    max_retries: int = 2
) -> dict:
    """
    Execute content chain with quality gates and retry logic.
    
    Args:
        topic: The subject to write about
        target_language: The language to translate into  
        style: The tone of the content
        max_retries: Number of times to retry a failed step
    
    Returns:
        Dictionary containing original and translated content with quality scores
    
    Raises:
        ChainError: If a step fails validation after all retries
    """
    results = {
        "topic": topic,
        "style": style,
        "target_language": target_language,
        "steps": []
    }
    
    # Step 1: Generate content (with retries)
    print(f"Step 1: Generating {style} content about '{topic}'...")
    
    for attempt in range(max_retries + 1):
        original_content = generate_content(topic, style)
        quality_check = check_content_quality(original_content)
        
        results["steps"].append({
            "step": "generate",
            "attempt": attempt + 1,
            "output_preview": original_content[:100] + "...",
            "quality_check": {
                "passed": quality_check.passed,
                "message": quality_check.message
            }
        })
        
        if quality_check.passed:
            print(f"  ✓ Quality gate passed: {quality_check.message}")
            break
        else:
            print(f"  ✗ Quality gate failed (attempt {attempt + 1}): {quality_check.message}")
            if attempt == max_retries:
                raise ChainError("generate", quality_check.message, original_content)
    
    results["original"] = original_content
    
    # Step 2: Translate content (with retries)
    print(f"\nStep 2: Translating to {target_language}...")
    
    for attempt in range(max_retries + 1):
        translated_content = translate_content(original_content, target_language)
        quality_check = check_translation_quality(
            original_content, 
            translated_content, 
            target_language
        )
        
        results["steps"].append({
            "step": "translate",
            "attempt": attempt + 1,
            "output_preview": translated_content[:100] + "...",
            "quality_check": {
                "passed": quality_check.passed,
                "message": quality_check.message,
                "score": quality_check.score
            }
        })
        
        if quality_check.passed:
            print(f"  ✓ Quality gate passed (score: {quality_check.score}): {quality_check.message}")
            break
        else:
            print(f"  ✗ Quality gate failed (attempt {attempt + 1}): {quality_check.message}")
            if attempt == max_retries:
                raise ChainError("translate", quality_check.message, translated_content)
    
    results["translated"] = translated_content
    results["translation_score"] = quality_check.score
    
    return results


if __name__ == "__main__":
    try:
        result = content_chain_with_gates(
            topic="AI-powered customer support chatbots",
            target_language="French",
            style="professional"
        )
        
        print("\n" + "="*50)
        print("CHAIN COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"\nOriginal:\n{result['original']}")
        print(f"\nTranslated (score: {result['translation_score']}):\n{result['translated']}")
        
    except ChainError as e:
        print(f"\n❌ Chain failed at step '{e.step}'")
        print(f"Reason: {e}")
        print(f"Last output: {e.output}")
```

### Key Quality Gate Patterns

Several important patterns emerge from this implementation:

**1. Rule-Based vs. LLM-Based Checks**

The content quality check uses simple rules (length, keyword presence). The translation quality check uses an LLM to evaluate. Choose based on:

- **Rule-based**: Fast, deterministic, good for format/length checks
- **LLM-based**: Slower, more expensive, but can evaluate nuanced quality

**2. Retry Logic**

When a quality gate fails, we retry the step instead of immediately failing. This handles the inherent variability in LLM outputs.

**3. Detailed Tracking**

The `results` dictionary tracks every attempt and quality check. This is invaluable for debugging and understanding chain behavior.

**4. Fail Fast with Context**

The `ChainError` exception includes the step name and the output that failed. This makes debugging much easier than a generic error.

## Error Handling in Chains

Real-world chains face more than just quality issues. API calls can fail, rate limits can trigger, and unexpected outputs can cause parsing errors. Let's build robust error handling:

```python
"""
Robust error handling patterns for prompt chains.

Chapter 17: Prompt Chaining - Implementation
"""

import os
import time
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum

load_dotenv()

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


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
    metadata: dict = field(default_factory=dict)


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0
) -> Any:
    """
    Execute a function with exponential backoff retry logic.
    
    Handles common API errors gracefully.
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        
        except anthropic.RateLimitError as e:
            last_exception = e
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  Rate limited, waiting {delay:.1f}s before retry...")
            time.sleep(delay)
            
        except anthropic.APIConnectionError as e:
            last_exception = e
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  Connection error, waiting {delay:.1f}s before retry...")
            time.sleep(delay)
            
        except anthropic.APIStatusError as e:
            # Don't retry client errors (4xx) except rate limits
            if e.status_code < 500:
                raise
            last_exception = e
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  Server error ({e.status_code}), waiting {delay:.1f}s before retry...")
            time.sleep(delay)
    
    raise last_exception


def execute_step(
    name: str,
    func: Callable[[], str],
    validator: Optional[Callable[[str], bool]] = None,
    max_retries: int = 2
) -> StepResult:
    """
    Execute a single chain step with full error handling.
    
    Args:
        name: Human-readable step name
        func: The function to execute (should return a string)
        validator: Optional function to validate the output
        max_retries: Maximum retry attempts for validation failures
    
    Returns:
        StepResult with status and output/error
    """
    start_time = time.time()
    result = StepResult(name=name, status=StepStatus.RUNNING)
    
    for attempt in range(max_retries + 1):
        result.attempts = attempt + 1
        
        try:
            # Execute with retry logic for API errors
            output = retry_with_backoff(func)
            
            # Validate if validator provided
            if validator and not validator(output):
                if attempt < max_retries:
                    print(f"  Validation failed for '{name}', retrying...")
                    continue
                else:
                    result.status = StepStatus.FAILED
                    result.error = "Output failed validation after all retries"
                    result.output = output
                    break
            
            # Success!
            result.status = StepStatus.SUCCESS
            result.output = output
            break
            
        except anthropic.APIError as e:
            result.status = StepStatus.FAILED
            result.error = f"API error: {type(e).__name__}: {str(e)}"
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = f"Unexpected error: {type(e).__name__}: {str(e)}"
            break
    
    result.duration_seconds = time.time() - start_time
    return result


def run_chain(steps: list[tuple[str, Callable, Optional[Callable]]]) -> list[StepResult]:
    """
    Run a sequence of steps, stopping on first failure.
    
    Args:
        steps: List of (name, function, validator) tuples
               The function receives the previous step's output as its first argument
               (except for the first step which receives no arguments)
    
    Returns:
        List of StepResult objects
    """
    results = []
    previous_output = None
    
    for i, (name, func, validator) in enumerate(steps):
        print(f"\n[{i+1}/{len(steps)}] Executing: {name}")
        
        # Wrap the function to pass previous output
        if i == 0:
            step_func = func
        else:
            step_func = lambda f=func, po=previous_output: f(po)
        
        result = execute_step(name, step_func, validator)
        results.append(result)
        
        if result.status == StepStatus.SUCCESS:
            print(f"  ✓ Completed in {result.duration_seconds:.2f}s")
            previous_output = result.output
        else:
            print(f"  ✗ Failed: {result.error}")
            # Mark remaining steps as skipped
            for remaining_name, _, _ in steps[i+1:]:
                results.append(StepResult(
                    name=remaining_name,
                    status=StepStatus.SKIPPED,
                    error="Previous step failed"
                ))
            break
    
    return results


# Example usage with our content chain
def create_content_chain(topic: str, target_language: str):
    """Create a content generation and translation chain."""
    
    def generate_step():
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Write a compelling 3-4 sentence marketing paragraph about: {topic}"
            }]
        )
        return response.content[0].text
    
    def translate_step(content: str):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Translate to {target_language}, maintaining tone:\n\n{content}"
            }]
        )
        return response.content[0].text
    
    def validate_length(content: str) -> bool:
        return len(content) >= 50
    
    return [
        ("Generate marketing content", generate_step, validate_length),
        ("Translate content", translate_step, validate_length),
    ]


if __name__ == "__main__":
    chain = create_content_chain(
        topic="sustainable bamboo water bottles",
        target_language="German"
    )
    
    results = run_chain(chain)
    
    print("\n" + "="*50)
    print("CHAIN EXECUTION SUMMARY")
    print("="*50)
    
    for result in results:
        status_icon = {
            StepStatus.SUCCESS: "✓",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "○"
        }.get(result.status, "?")
        
        print(f"\n{status_icon} {result.name}")
        print(f"  Status: {result.status.value}")
        print(f"  Attempts: {result.attempts}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        
        if result.output:
            print(f"  Output preview: {result.output[:80]}...")
        if result.error:
            print(f"  Error: {result.error}")
```

### Error Handling Best Practices

This implementation demonstrates several important patterns:

**1. Categorize Errors**

Not all errors are created equal:
- **Rate limits**: Always retry with backoff
- **Connection errors**: Usually retry 
- **Server errors (5xx)**: Retry with backoff
- **Client errors (4xx)**: Usually don't retry (except rate limits)
- **Validation failures**: Retry the step, not the API call

**2. Exponential Backoff**

When retrying, increase the delay exponentially: 1s, 2s, 4s, 8s, etc. This prevents hammering a struggling service and respects rate limits.

**3. Fail Fast, Fail Informatively**

When a step fails, stop the chain (don't waste API calls on doomed steps) but provide detailed information about what went wrong and where.

**4. Track Everything**

Record attempts, durations, and errors. This data is gold when debugging production issues.

## The Chain Class Pattern

Now let's consolidate everything into a reusable `Chain` class that you can use for any sequential workflow:

```python
"""
Reusable Chain class for building prompt chains.

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

load_dotenv()

T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type


class StepStatus(Enum):
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


class ChainStep(ABC, Generic[T, U]):
    """
    Abstract base class for chain steps.
    
    Inherit from this to create custom steps with validation logic.
    """
    
    def __init__(self, name: str, max_retries: int = 2):
        self.name = name
        self.max_retries = max_retries
    
    @abstractmethod
    def execute(self, input_data: T) -> U:
        """Execute the step logic. Override this in subclasses."""
        pass
    
    def validate(self, output: U) -> bool:
        """Validate the output. Override for custom validation."""
        return True
    
    def on_retry(self, attempt: int, error: Optional[str] = None):
        """Called before each retry. Override to customize retry behavior."""
        pass


class LLMStep(ChainStep[str, str]):
    """
    A chain step that makes an LLM call.
    
    This is the most common type of step in prompt chains.
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
    
    Example usage:
        chain = Chain()
        chain.add_step(GenerateStep(...))
        chain.add_step(TranslateStep(...))
        result = chain.run("initial input")
    """
    
    def __init__(self, name: str = "Chain"):
        self.name = name
        self.steps: list[ChainStep] = []
    
    def add_step(self, step: ChainStep) -> "Chain":
        """Add a step to the chain. Returns self for chaining."""
        self.steps.append(step)
        return self
    
    def run(
        self, 
        initial_input: Any,
        stop_on_failure: bool = True
    ) -> ChainResult:
        """
        Execute all steps in sequence.
        
        Args:
            initial_input: The input to the first step
            stop_on_failure: If True, stop chain on first failure
        
        Returns:
            ChainResult with all step results and final output
        """
        start_time = time.time()
        step_results = []
        current_input = initial_input
        success = True
        
        for i, step in enumerate(self.steps):
            print(f"\n[{i+1}/{len(self.steps)}] {step.name}")
            
            result = self._execute_step(step, current_input)
            step_results.append(result)
            
            if result.status == StepStatus.SUCCESS:
                print(f"  ✓ Completed in {result.duration_seconds:.2f}s")
                current_input = result.output
            else:
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
    
    def _execute_step(self, step: ChainStep, input_data: Any) -> StepResult:
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
                        print(f"  Validation failed, retrying ({attempt + 1}/{step.max_retries})...")
                        continue
                    result.status = StepStatus.FAILED
                    result.error = "Output failed validation"
                    result.output = output
                    break
                
                result.status = StepStatus.SUCCESS
                result.output = output
                break
                
            except anthropic.RateLimitError:
                if attempt < step.max_retries:
                    delay = 2 ** attempt
                    print(f"  Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    step.on_retry(attempt + 1, "Rate limited")
                    continue
                result.status = StepStatus.FAILED
                result.error = "Rate limit exceeded after retries"
                
            except anthropic.APIError as e:
                result.status = StepStatus.FAILED
                result.error = f"API error: {str(e)}"
                break
                
            except Exception as e:
                result.status = StepStatus.FAILED
                result.error = f"Unexpected error: {str(e)}"
                break
        
        result.duration_seconds = time.time() - start_time
        return result


# Convenience function for simple LLM chains
def create_llm_chain(
    steps: list[dict],
    client: Optional[anthropic.Anthropic] = None
) -> Chain:
    """
    Create a chain from a list of step configurations.
    
    Args:
        steps: List of dicts with keys: name, prompt_template, 
               and optionally: system_prompt, validator, max_retries
        client: Anthropic client (creates one if not provided)
    
    Returns:
        Configured Chain ready to run
    """
    if client is None:
        client = anthropic.Anthropic()
    
    chain = Chain()
    
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
    
    # Define the chain using the convenience function
    chain = create_llm_chain([
        {
            "name": "Generate Marketing Copy",
            "prompt_template": """Write a compelling 3-4 sentence marketing paragraph about: {input}

Include clear benefits and a call to action.""",
            "validator": lambda x: len(x) >= 50
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
Keep the key message but make it punchy and engaging.

Original content:
{input}""",
            "validator": lambda x: len(x) <= 300
        }
    ], client)
    
    # Run the chain
    print("="*50)
    print("EXECUTING CONTENT CHAIN")
    print("="*50)
    
    result = chain.run("eco-friendly reusable coffee cups made from recycled ocean plastic")
    
    print("\n" + "="*50)
    print("CHAIN RESULTS")
    print("="*50)
    print(f"\nSuccess: {result.success}")
    print(f"Total duration: {result.total_duration_seconds:.2f}s")
    
    for step_result in result.steps:
        print(f"\n--- {step_result.name} ---")
        print(f"Status: {step_result.status.value}")
        if step_result.output:
            print(f"Output:\n{step_result.output}")
```

### Using the Chain Class

The `Chain` class provides several benefits:

**1. Declarative Configuration**

Define chains as data structures, not imperative code:

```python
chain = create_llm_chain([
    {"name": "Step 1", "prompt_template": "..."},
    {"name": "Step 2", "prompt_template": "..."},
])
```

**2. Extensibility**

Create custom step types by inheriting from `ChainStep`:

```python
class DatabaseStep(ChainStep[str, dict]):
    def execute(self, query: str) -> dict:
        # Fetch from database
        return {"data": "..."}
```

**3. Consistent Error Handling**

Every step gets the same retry logic and error tracking without duplicating code.

**4. Inspectable Results**

Access any step's output for debugging:

```python
result = chain.run(input_data)
translation = result.get_step("Translate to Spanish").output
```

## Passing Context Between Steps

Sometimes steps need more than just the previous step's output. They need context from earlier in the chain or from the original input. Here's how to handle that:

```python
"""
Passing rich context between chain steps.

Chapter 17: Prompt Chaining - Implementation
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChainContext:
    """
    Carries data through a chain, accumulating outputs from each step.
    
    This allows later steps to access outputs from any earlier step,
    not just the immediately preceding one.
    """
    original_input: Any
    step_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_output(self, step_name: str, output: Any):
        """Record output from a step."""
        self.step_outputs[step_name] = output
    
    def get_output(self, step_name: str) -> Any:
        """Get output from a specific step."""
        return self.step_outputs.get(step_name)
    
    @property
    def latest_output(self) -> Any:
        """Get the most recent step output."""
        if not self.step_outputs:
            return self.original_input
        return list(self.step_outputs.values())[-1]


# Example: A chain where step 3 needs outputs from both steps 1 and 2

def research_chain_example():
    """
    A research chain where the summary step needs access to
    both the research findings AND the original question.
    """
    import anthropic
    from dotenv import load_dotenv
    
    load_dotenv()
    client = anthropic.Anthropic()
    MODEL = "claude-sonnet-4-20250514"
    
    # Initialize context with the research question
    context = ChainContext(
        original_input="What are the environmental impacts of cryptocurrency mining?",
        metadata={"requested_by": "user123", "timestamp": "2025-01-15"}
    )
    
    # Step 1: Generate research points
    print("Step 1: Generating research points...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""List 5 key research points to investigate for this question:
            
{context.original_input}

Format as a numbered list."""
        }]
    )
    research_points = response.content[0].text
    context.add_output("research_points", research_points)
    print(f"Generated {len(research_points.split(chr(10)))} points")
    
    # Step 2: Expand on each point
    print("\nStep 2: Expanding research points...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Expand on each of these research points with 2-3 sentences of detail:

{research_points}"""
        }]
    )
    expanded_research = response.content[0].text
    context.add_output("expanded_research", expanded_research)
    print(f"Expanded to {len(expanded_research)} characters")
    
    # Step 3: Synthesize - needs BOTH original question AND expanded research
    print("\nStep 3: Synthesizing final answer...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Based on this research, write a comprehensive answer to the original question.

ORIGINAL QUESTION:
{context.original_input}

RESEARCH FINDINGS:
{context.get_output("expanded_research")}

Write a well-structured response that directly addresses the question."""
        }]
    )
    final_answer = response.content[0].text
    context.add_output("final_answer", final_answer)
    
    return context


if __name__ == "__main__":
    context = research_chain_example()
    
    print("\n" + "="*50)
    print("CHAIN COMPLETE")
    print("="*50)
    print(f"\nOriginal question: {context.original_input}")
    print(f"\nFinal answer:\n{context.get_output('final_answer')}")
    print(f"\nMetadata: {context.metadata}")
```

### Context Patterns

The `ChainContext` class enables several useful patterns:

**1. Reference Original Input**

Later steps often need the original request, not just intermediate outputs:

```python
# Step 3 can reference both
prompt = f"""
Original request: {context.original_input}
Processed data: {context.get_output("step2")}
"""
```

**2. Skip Steps Conditionally**

Check earlier outputs to decide whether to run a step:

```python
if "urgent" in context.get_output("classify").lower():
    # Run fast path
else:
    # Run thorough path
```

**3. Accumulate Metadata**

Track information that isn't part of the main data flow:

```python
context.metadata["tokens_used"] = 1500
context.metadata["cache_hits"] = 2
```

## Common Pitfalls

### 1. Overly Long Chains

**Problem**: Creating chains with 10+ steps where each step does very little.

**Why it's bad**: Each LLM call adds latency and cost. Long chains amplify errors—if each step has 95% accuracy, a 10-step chain is only 60% accurate overall.

**Solution**: Combine related operations into single steps. A chain of 3-5 well-designed steps usually outperforms 10+ tiny steps.

### 2. Insufficient Context Passing

**Problem**: Each step only receives the previous step's output, losing important context.

**Why it's bad**: Step 5 might need information from step 1 that was lost along the way.

**Solution**: Use a `ChainContext` object that accumulates all outputs, or explicitly include relevant context in prompts.

### 3. No Quality Gates

**Problem**: Blindly passing outputs through without validation.

**Why it's bad**: Garbage propagates and compounds. A bad step 2 output makes step 3, 4, and 5 all produce garbage.

**Solution**: Add validation after critical steps. You don't need gates after every step—focus on points where quality matters most.

## Practical Exercise

**Task**: Build a content repurposing chain that takes a blog post topic and produces content for multiple platforms.

**Requirements**:

1. **Step 1**: Generate a detailed blog post outline (5-7 sections)
2. **Step 2**: Write the full blog post based on the outline
3. **Step 3**: Create a LinkedIn post summarizing the blog (under 300 characters)
4. **Step 4**: Generate 3 tweet variations based on the blog post

**Quality Gates**:
- Blog outline must have at least 5 sections
- Full blog post must be at least 500 words
- LinkedIn post must be under 300 characters
- Each tweet must be under 280 characters

**Bonus**: Add a validation step that uses Claude to score the overall content quality (1-10) and reject outputs scoring below 7.

**Hints**:
- Use the `Chain` class from this chapter
- Consider what context each step needs
- Think about how to validate tweet character counts

**Solution**: See `code/exercise.py`

## Key Takeaways

- **Prompt chains break complex tasks into focused steps**, where each step does one thing well and passes its output to the next step.

- **Quality gates prevent bad outputs from propagating** through the chain. Use rule-based validation for simple checks and LLM-based validation for complex quality assessment.

- **Error handling should include retries with exponential backoff** for transient failures, but fail fast for permanent errors.

- **The Chain class pattern provides a reusable, composable structure** that handles error handling, retry logic, and result tracking consistently.

- **Context objects allow later steps to access earlier outputs**, not just the immediately preceding step's output.

- **Each link in the chain should do one thing well** — this makes chains easier to debug, test, and modify.

## What's Next

In Chapter 18, we'll explore the **Routing** pattern—directing inputs to specialized handlers based on classification. While chaining processes every input through the same sequence of steps, routing chooses different paths for different types of inputs. This is essential for building agents that handle diverse requests efficiently.
