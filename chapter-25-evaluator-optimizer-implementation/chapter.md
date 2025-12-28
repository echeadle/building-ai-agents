---
chapter: 25
title: "Evaluator-Optimizer - Implementation"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 25: Evaluator-Optimizer - Implementation

## Introduction

In Chapter 24, we explored the concept of the evaluator-optimizer pattern—a powerful approach where one LLM generates content while another evaluates and provides feedback, creating an iterative refinement loop. We discussed when this pattern shines and how to design effective evaluation criteria.

Now it's time to build it.

In this chapter, we'll implement a complete evaluator-optimizer system from scratch. We'll create a writing assistant that takes a rough draft and iteratively improves it based on specific quality criteria. By the end, you'll have a reusable pattern for any task where iterative refinement produces better results than a single attempt.

This pattern is particularly powerful because it mirrors how humans actually work—we write, review, revise, and repeat until we're satisfied. By giving our agent this same capability, we unlock a level of quality that single-shot generation simply cannot match.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement a generator LLM that produces and revises content
- Implement an evaluator LLM with structured feedback output
- Build a feedback loop that passes evaluation results back to the generator
- Detect convergence to prevent unnecessary iterations
- Add maximum iteration safeguards to prevent runaway loops
- Create a complete writing assistant that iteratively improves drafts

## The Core Components

Before diving into code, let's review the three key components we need to build:

1. **Generator**: Creates initial content and produces revisions based on feedback
2. **Evaluator**: Assesses content quality and provides specific, actionable feedback
3. **Loop Controller**: Manages the iteration cycle, detects convergence, and enforces limits

Let's implement each one, starting with the simplest version and building up to a complete, production-ready system.

## Implementing the Generator

The generator has two jobs: create initial content from a prompt, and revise existing content based on feedback. We'll implement these as separate methods since they require different prompting strategies.

```python
"""
Generator component for the evaluator-optimizer pattern.

This module implements the content generation and revision functionality.
"""

import anthropic
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for the generator."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.7  # Slightly creative for writing tasks


class Generator:
    """
    Generates and revises content based on feedback.
    
    The generator handles two distinct tasks:
    1. Initial generation from a user prompt
    2. Revision based on evaluator feedback
    """
    
    def __init__(self, client: anthropic.Anthropic, config: GeneratorConfig | None = None):
        """
        Initialize the generator.
        
        Args:
            client: Anthropic API client
            config: Generator configuration (uses defaults if not provided)
        """
        self.client = client
        self.config = config or GeneratorConfig()
        
        self.system_prompt = """You are a skilled writer focused on producing clear, 
engaging, and well-structured content. Your writing is concise yet comprehensive,
avoiding unnecessary filler while ensuring all important points are covered.

When revising content based on feedback:
1. Address each piece of feedback specifically
2. Maintain the original intent and key messages
3. Improve clarity and flow
4. Keep what already works well

Always output only the revised content, not explanations of your changes."""

    def generate(self, prompt: str, context: str = "") -> str:
        """
        Generate initial content from a prompt.
        
        Args:
            prompt: The user's writing request
            context: Optional additional context
            
        Returns:
            The generated content
        """
        user_message = prompt
        if context:
            user_message = f"{context}\n\n{prompt}"
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text

    def revise(self, content: str, feedback: str, original_prompt: str) -> str:
        """
        Revise content based on evaluator feedback.
        
        Args:
            content: The current version of the content
            feedback: Specific feedback from the evaluator
            original_prompt: The original user request (for context)
            
        Returns:
            The revised content
        """
        revision_prompt = f"""Original request: {original_prompt}

Current content:
---
{content}
---

Feedback to address:
{feedback}

Please revise the content to address all feedback points while maintaining 
the original intent. Output only the revised content."""

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": revision_prompt}
            ]
        )
        
        return response.content[0].text
```

Notice how the `revise` method includes both the original prompt and the current content. This context is crucial—without it, the generator might drift from the original intent while addressing feedback.

## Implementing the Evaluator

The evaluator is the heart of this pattern. It must assess content against specific criteria and produce structured, actionable feedback. Let's implement an evaluator that returns both a quality score and specific improvement suggestions.

```python
"""
Evaluator component for the evaluator-optimizer pattern.

This module implements content evaluation with structured feedback.
"""

import json
import anthropic
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Structured result from content evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool  # Whether content meets quality threshold
    feedback: list[str]  # Specific improvement suggestions
    strengths: list[str]  # What's working well
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "NEEDS REVISION"
        feedback_str = "\n".join(f"  - {f}" for f in self.feedback)
        strengths_str = "\n".join(f"  - {s}" for s in self.strengths)
        return f"""Evaluation: {status} (Score: {self.score:.2f})

Strengths:
{strengths_str}

Areas for Improvement:
{feedback_str}"""


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.3  # Lower temperature for consistent evaluation
    quality_threshold: float = 0.8  # Score needed to pass


class Evaluator:
    """
    Evaluates content quality and provides structured feedback.
    
    The evaluator assesses content against defined criteria and returns
    both a quality score and specific, actionable feedback for improvement.
    """
    
    def __init__(
        self, 
        client: anthropic.Anthropic, 
        criteria: list[str],
        config: EvaluatorConfig | None = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            client: Anthropic API client
            criteria: List of evaluation criteria
            config: Evaluator configuration
        """
        self.client = client
        self.criteria = criteria
        self.config = config or EvaluatorConfig()
        
        # Build criteria list for prompt
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        
        self.system_prompt = f"""You are a content evaluator. Your job is to assess 
writing quality against specific criteria and provide actionable feedback.

Evaluation Criteria:
{criteria_text}

You must evaluate content fairly and consistently. Provide specific, actionable 
feedback that a writer can use to improve their work.

Always respond with valid JSON in this exact format:
{{
    "score": <float between 0 and 1>,
    "strengths": ["strength 1", "strength 2"],
    "improvements": ["specific improvement 1", "specific improvement 2"],
    "passed": <true if score >= {self.config.quality_threshold}, false otherwise>
}}

Be specific in your feedback. Instead of "improve clarity", say "The second 
paragraph's main point is unclear - consider stating the key takeaway in the 
first sentence."
"""

    def evaluate(self, content: str, context: str = "") -> EvaluationResult:
        """
        Evaluate content and return structured feedback.
        
        Args:
            content: The content to evaluate
            context: Optional context about the content's purpose
            
        Returns:
            EvaluationResult with score, pass/fail, and feedback
        """
        eval_prompt = f"""Please evaluate the following content:

---
{content}
---"""
        
        if context:
            eval_prompt = f"Context: {context}\n\n{eval_prompt}"
        
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        # Parse the JSON response
        response_text = response.content[0].text
        
        try:
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            
            return EvaluationResult(
                score=float(result.get("score", 0)),
                passed=bool(result.get("passed", False)),
                feedback=result.get("improvements", []),
                strengths=result.get("strengths", [])
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback for parsing errors
            print(f"Warning: Failed to parse evaluation response: {e}")
            return EvaluationResult(
                score=0.5,
                passed=False,
                feedback=["Evaluation parsing failed - please try again"],
                strengths=[]
            )
```

The evaluator uses a lower temperature (0.3) for consistency—we want similar content to receive similar scores across evaluations. The structured JSON output makes it easy to programmatically process the feedback.

> **💡 Tip:** The quality threshold (0.8 by default) determines when content is "good enough." Set this based on your specific needs—a quick draft might use 0.7, while published content might require 0.9.

## The Feedback Loop

Now let's combine the generator and evaluator into a complete feedback loop. This is where the magic happens—the loop controller manages iterations, tracks progress, and knows when to stop.

```python
"""
Evaluator-Optimizer loop controller.

This module implements the iterative refinement loop that combines
the generator and evaluator into a complete optimization system.
"""

import anthropic
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IterationRecord:
    """Record of a single iteration in the optimization loop."""
    iteration: int
    content: str
    score: float
    passed: bool
    feedback: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """Final result of the optimization process."""
    final_content: str
    final_score: float
    iterations_used: int
    converged: bool
    history: list[IterationRecord]
    
    def summary(self) -> str:
        """Generate a summary of the optimization process."""
        status = "converged" if self.converged else "hit max iterations"
        scores = [r.score for r in self.history]
        improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        
        return f"""Optimization Complete
---
Final Score: {self.final_score:.2f}
Iterations: {self.iterations_used}
Status: {status}
Score Improvement: {improvement:+.2f}
Score History: {' → '.join(f'{s:.2f}' for s in scores)}"""


@dataclass
class LoopConfig:
    """Configuration for the optimization loop."""
    max_iterations: int = 5
    convergence_threshold: float = 0.02  # Stop if improvement < this
    min_iterations: int = 1  # Always run at least this many
    verbose: bool = True  # Print progress


class EvaluatorOptimizer:
    """
    Orchestrates the evaluator-optimizer loop.
    
    This class manages the iterative refinement process, tracking progress,
    detecting convergence, and enforcing iteration limits.
    """
    
    def __init__(
        self,
        generator: 'Generator',
        evaluator: 'Evaluator',
        config: LoopConfig | None = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            generator: Generator instance for content creation/revision
            evaluator: Evaluator instance for quality assessment
            config: Loop configuration
        """
        self.generator = generator
        self.evaluator = evaluator
        self.config = config or LoopConfig()
    
    def optimize(self, prompt: str, context: str = "") -> OptimizationResult:
        """
        Run the optimization loop until convergence or max iterations.
        
        Args:
            prompt: The original content request
            context: Optional context for evaluation
            
        Returns:
            OptimizationResult with final content and history
        """
        history: list[IterationRecord] = []
        current_content = ""
        previous_score = 0.0
        
        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {iteration}/{self.config.max_iterations}")
                print('='*50)
            
            # Generate or revise content
            if iteration == 1:
                if self.config.verbose:
                    print("Generating initial content...")
                current_content = self.generator.generate(prompt, context)
            else:
                # Format feedback for revision
                feedback_text = "\n".join(f"- {f}" for f in history[-1].feedback)
                if self.config.verbose:
                    print("Revising based on feedback...")
                current_content = self.generator.revise(
                    current_content, 
                    feedback_text,
                    prompt
                )
            
            # Evaluate the content
            if self.config.verbose:
                print("Evaluating content...")
            evaluation = self.evaluator.evaluate(current_content, context)
            
            # Record this iteration
            record = IterationRecord(
                iteration=iteration,
                content=current_content,
                score=evaluation.score,
                passed=evaluation.passed,
                feedback=evaluation.feedback
            )
            history.append(record)
            
            if self.config.verbose:
                print(f"Score: {evaluation.score:.2f}")
                if evaluation.strengths:
                    print(f"Strengths: {', '.join(evaluation.strengths[:2])}")
                if evaluation.feedback and not evaluation.passed:
                    print(f"Improvements needed: {len(evaluation.feedback)}")
            
            # Check if we've passed
            if evaluation.passed:
                if self.config.verbose:
                    print(f"\n✓ Content passed evaluation!")
                return OptimizationResult(
                    final_content=current_content,
                    final_score=evaluation.score,
                    iterations_used=iteration,
                    converged=True,
                    history=history
                )
            
            # Check for convergence (improvement too small to continue)
            if iteration >= self.config.min_iterations:
                improvement = evaluation.score - previous_score
                if improvement < self.config.convergence_threshold and iteration > 1:
                    if self.config.verbose:
                        print(f"\n⚠ Converged (improvement {improvement:.3f} < threshold)")
                    return OptimizationResult(
                        final_content=current_content,
                        final_score=evaluation.score,
                        iterations_used=iteration,
                        converged=True,
                        history=history
                    )
            
            previous_score = evaluation.score
        
        # Max iterations reached
        if self.config.verbose:
            print(f"\n⚠ Max iterations ({self.config.max_iterations}) reached")
        
        return OptimizationResult(
            final_content=current_content,
            final_score=history[-1].score,
            iterations_used=self.config.max_iterations,
            converged=False,
            history=history
        )
```

The loop controller implements two important safeguards:

1. **Maximum iterations**: Prevents runaway loops that could burn through API credits
2. **Convergence detection**: Stops early if improvements become negligible

## Convergence Detection

Convergence detection deserves special attention. We don't want to waste iterations when the content has stopped improving meaningfully. Our implementation uses a simple but effective approach: compare the current score to the previous score.

```python
# Check for convergence (improvement too small to continue)
if iteration >= self.config.min_iterations:
    improvement = evaluation.score - previous_score
    if improvement < self.config.convergence_threshold and iteration > 1:
        # Stop - we're not making meaningful progress
        return result
```

However, this simple approach has limitations. Consider these scenarios:

**Scenario 1: Oscillating scores**
- Iteration 1: 0.65
- Iteration 2: 0.72
- Iteration 3: 0.68
- Iteration 4: 0.74

The scores oscillate, but the content is improving. Our simple check might miss this.

**Scenario 2: Plateau then improvement**
- Iteration 1: 0.65
- Iteration 2: 0.66
- Iteration 3: 0.67
- Iteration 4: 0.82

Early iterations show minimal improvement, but a breakthrough happens later.

For more sophisticated applications, you might implement a sliding window average or track the best score seen:

```python
def check_convergence_advanced(
    history: list[IterationRecord],
    window_size: int = 3,
    threshold: float = 0.02
) -> bool:
    """
    Advanced convergence check using sliding window.
    
    Args:
        history: List of iteration records
        window_size: Number of recent iterations to consider
        threshold: Minimum average improvement to continue
        
    Returns:
        True if converged (should stop), False otherwise
    """
    if len(history) < window_size + 1:
        return False
    
    recent_scores = [r.score for r in history[-window_size:]]
    older_scores = [r.score for r in history[-(window_size+1):-1]]
    
    recent_avg = sum(recent_scores) / len(recent_scores)
    older_avg = sum(older_scores) / len(older_scores)
    
    return (recent_avg - older_avg) < threshold
```

## Building the Writing Assistant

Now let's put everything together into a complete writing assistant. This example demonstrates the full pattern with sensible defaults for writing tasks.

```python
"""
Writing Assistant using the Evaluator-Optimizer pattern.

A complete example that iteratively improves written content
based on structured feedback.
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Default evaluation criteria for writing
WRITING_CRITERIA = [
    "Clarity: Is the writing easy to understand? Are ideas expressed clearly?",
    "Structure: Is the content well-organized with logical flow between sections?",
    "Conciseness: Is the writing free of unnecessary words and redundancy?",
    "Engagement: Does the writing hold the reader's attention?",
    "Completeness: Does the content fully address the topic or request?",
    "Grammar and Style: Is the writing grammatically correct and stylistically consistent?",
]


class WritingAssistant:
    """
    A writing assistant that iteratively improves content.
    
    This class provides a high-level interface to the evaluator-optimizer
    pattern, specifically configured for writing tasks.
    """
    
    def __init__(
        self,
        criteria: list[str] | None = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        verbose: bool = True
    ):
        """
        Initialize the writing assistant.
        
        Args:
            criteria: Evaluation criteria (uses defaults if not provided)
            max_iterations: Maximum revision cycles
            quality_threshold: Score needed to pass (0.0 to 1.0)
            verbose: Print progress during optimization
        """
        self.client = anthropic.Anthropic()
        
        # Set up components with configurations
        generator_config = GeneratorConfig(
            temperature=0.7,
            max_tokens=2048
        )
        self.generator = Generator(self.client, generator_config)
        
        evaluator_config = EvaluatorConfig(
            quality_threshold=quality_threshold,
            temperature=0.3
        )
        self.evaluator = Evaluator(
            self.client,
            criteria or WRITING_CRITERIA,
            evaluator_config
        )
        
        loop_config = LoopConfig(
            max_iterations=max_iterations,
            verbose=verbose,
            convergence_threshold=0.02
        )
        self.optimizer = EvaluatorOptimizer(
            self.generator,
            self.evaluator,
            loop_config
        )
    
    def write(self, request: str, context: str = "") -> OptimizationResult:
        """
        Generate and iteratively improve content based on a request.
        
        Args:
            request: What to write (e.g., "Write an introduction to machine learning")
            context: Optional context (e.g., "For a technical blog aimed at beginners")
            
        Returns:
            OptimizationResult with final content and improvement history
        """
        return self.optimizer.optimize(request, context)
    
    def improve(self, existing_content: str, context: str = "") -> OptimizationResult:
        """
        Improve existing content through evaluation and revision.
        
        Args:
            existing_content: Content to improve
            context: Optional context about the content's purpose
            
        Returns:
            OptimizationResult with improved content and history
        """
        # Create a prompt that includes the existing content
        prompt = f"""Improve the following content while maintaining its core message 
and intent:

{existing_content}"""
        
        return self.optimizer.optimize(prompt, context)


def main():
    """Demonstrate the writing assistant."""
    
    # Create the assistant
    assistant = WritingAssistant(
        max_iterations=4,
        quality_threshold=0.8,
        verbose=True
    )
    
    # Example: Write a product description
    print("\n" + "="*60)
    print("WRITING ASSISTANT DEMO")
    print("="*60)
    
    request = """Write a compelling product description for a new smart water bottle 
that tracks hydration, syncs with fitness apps, and reminds users to drink water. 
The target audience is health-conscious professionals aged 25-45."""
    
    context = "For an e-commerce product page. Should be persuasive but not pushy."
    
    print(f"\nRequest: {request[:100]}...")
    print(f"Context: {context}")
    
    # Run the optimization
    result = assistant.write(request, context)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result.summary())
    print("\n--- Final Content ---")
    print(result.final_content)
    
    # Show iteration history
    print("\n--- Iteration History ---")
    for record in result.history:
        print(f"\nIteration {record.iteration}: Score {record.score:.2f}")
        if record.feedback:
            print(f"  Feedback: {record.feedback[0][:80]}...")


if __name__ == "__main__":
    main()
```

When you run this, you'll see the assistant work through multiple iterations, with scores gradually improving as it addresses feedback.

## Customizing for Different Use Cases

The evaluator-optimizer pattern is highly adaptable. Here are configurations for different use cases:

### Technical Documentation

```python
TECH_DOC_CRITERIA = [
    "Accuracy: Are all technical details correct and precise?",
    "Completeness: Are all necessary steps and concepts covered?",
    "Clarity: Can a developer follow this without ambiguity?",
    "Code Examples: Are code samples correct, runnable, and well-commented?",
    "Prerequisites: Are dependencies and requirements clearly stated?",
]

tech_assistant = WritingAssistant(
    criteria=TECH_DOC_CRITERIA,
    quality_threshold=0.85,  # Higher bar for documentation
    max_iterations=6
)
```

### Marketing Copy

```python
MARKETING_CRITERIA = [
    "Hook: Does the opening grab attention immediately?",
    "Benefits: Are benefits clearly stated (not just features)?",
    "Call to Action: Is there a clear, compelling CTA?",
    "Tone: Is the tone appropriate for the brand and audience?",
    "Brevity: Is every word earning its place?",
]

marketing_assistant = WritingAssistant(
    criteria=MARKETING_CRITERIA,
    quality_threshold=0.75,  # Marketing can be more subjective
    max_iterations=4
)
```

### Academic Writing

```python
ACADEMIC_CRITERIA = [
    "Thesis: Is the main argument clear and well-supported?",
    "Evidence: Are claims backed by appropriate evidence?",
    "Structure: Does the paper follow logical academic structure?",
    "Citations: Are sources properly attributed?",
    "Objectivity: Is the tone appropriately academic and unbiased?",
    "Originality: Does the work contribute new insights?",
]

academic_assistant = WritingAssistant(
    criteria=ACADEMIC_CRITERIA,
    quality_threshold=0.85,
    max_iterations=5
)
```

## Handling Edge Cases

Real-world usage requires handling several edge cases gracefully.

### Empty or Minimal Feedback

Sometimes the evaluator returns high scores with little feedback. Handle this by stopping the loop even if the threshold isn't quite met:

```python
def should_continue(evaluation: EvaluationResult, iteration: int) -> bool:
    """Determine if we should continue iterating."""
    
    # Stop if passed
    if evaluation.passed:
        return False
    
    # Stop if no actionable feedback
    if not evaluation.feedback:
        return False
    
    # Stop if feedback is too vague to act on
    vague_indicators = ["overall", "generally", "consider", "might"]
    if all(any(v in f.lower() for v in vague_indicators) for f in evaluation.feedback):
        return False
    
    return True
```

### Degrading Scores

Sometimes revisions make content worse. Implement a rollback mechanism:

```python
def optimize_with_rollback(self, prompt: str) -> OptimizationResult:
    """Optimize with rollback to best version if scores degrade."""
    
    best_content = ""
    best_score = 0.0
    
    for iteration in range(self.config.max_iterations):
        # ... generate/revise content ...
        
        evaluation = self.evaluator.evaluate(current_content)
        
        # Track best version
        if evaluation.score > best_score:
            best_content = current_content
            best_score = evaluation.score
        
        # Rollback if we've degraded significantly
        if evaluation.score < best_score - 0.1:
            print(f"Score degraded, rolling back to best version")
            current_content = best_content
    
    return OptimizationResult(
        final_content=best_content,
        final_score=best_score,
        # ...
    )
```

### Infinite Loops and Safety

Always enforce hard limits, even if your convergence detection fails:

```python
import time

def optimize_with_safety(self, prompt: str) -> OptimizationResult:
    """Optimize with additional safety measures."""
    
    start_time = time.time()
    max_time_seconds = 300  # 5 minute hard limit
    
    for iteration in range(self.config.max_iterations):
        # Time-based circuit breaker
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            print(f"Time limit reached ({elapsed:.0f}s)")
            break
        
        # ... rest of loop ...
```

## Common Pitfalls

### 1. Vague Evaluation Criteria

Criteria like "make it better" or "improve quality" give the evaluator nothing concrete to assess. Always be specific:

```python
# ❌ Bad
criteria = ["Good writing quality", "Make it engaging"]

# ✅ Good
criteria = [
    "Every paragraph should have a clear topic sentence",
    "Use concrete examples to illustrate abstract concepts",
    "Sentences should average 15-20 words for readability"
]
```

### 2. Setting the Threshold Too High

A quality threshold of 0.95 sounds great but may be impossible to achieve consistently. This leads to wasted iterations and frustrated users. Start at 0.75-0.80 and adjust based on observed results.

### 3. Not Passing Context to the Generator

When revising, the generator needs to know the original intent. Without the original prompt, revisions can drift:

```python
# ❌ Bad - Generator loses context
current_content = generator.revise(content, feedback)

# ✅ Good - Generator remembers the goal
current_content = generator.revise(content, feedback, original_prompt)
```

## Practical Exercise

**Task:** Build a code review assistant using the evaluator-optimizer pattern

Your assistant should:

1. Take a code snippet as input
2. Evaluate it against code quality criteria (readability, efficiency, best practices)
3. Suggest improvements
4. Apply improvements and re-evaluate
5. Return the improved code along with the improvement history

**Requirements:**
- Use at least 5 code quality criteria
- Implement convergence detection
- Handle the case where code is already good (minimal changes needed)
- Maximum 4 iterations

**Hints:**
- The "generator" becomes a code improver
- The "evaluator" assesses code quality
- Consider criteria like: naming conventions, error handling, documentation, efficiency, and idiomatic patterns

**Solution:** See `code/exercise.py`

## Key Takeaways

- **The evaluator-optimizer pattern creates a feedback loop** where one LLM generates content and another evaluates it, enabling iterative refinement

- **Clear evaluation criteria are essential**—vague criteria produce vague feedback, which produces vague improvements

- **Convergence detection prevents wasted iterations** by stopping when improvements become negligible

- **Maximum iteration safeguards are non-negotiable**—always enforce hard limits to prevent runaway loops

- **The pattern is highly adaptable**—customize criteria and thresholds for different use cases (writing, code, documentation)

- **Track iteration history for transparency**—knowing how content evolved helps with debugging and user trust

## What's Next

In Chapter 26, we'll make a significant leap: from orchestrated workflows to true autonomous agents. We'll explore what makes an agent different from a workflow—specifically, the agent's ability to direct its own control flow. You'll learn how to build agents that can decide their own next steps, a powerful capability that requires careful design to use safely.

The evaluator-optimizer pattern you've learned here will prove valuable in agent development, particularly for self-improvement and quality assurance within agent loops.
