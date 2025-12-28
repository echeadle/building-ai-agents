"""
Evaluator-Optimizer Pattern Implementation

This module provides the core components for the evaluator-optimizer pattern:
- Generator: Creates and revises content based on feedback
- Evaluator: Assesses content quality with structured feedback
- EvaluatorOptimizer: Orchestrates the refinement loop

Chapter 25: Evaluator-Optimizer - Implementation
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for the Generator component."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.7  # Slightly creative for writing tasks


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator component."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.3  # Lower temperature for consistent evaluation
    quality_threshold: float = 0.8  # Score needed to pass


@dataclass
class LoopConfig:
    """Configuration for the optimization loop."""
    max_iterations: int = 5
    convergence_threshold: float = 0.02  # Stop if improvement < this
    min_iterations: int = 1  # Always run at least this many
    verbose: bool = True  # Print progress


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class EvaluationResult:
    """Structured result from content evaluation."""
    score: float  # 0.0 to 1.0
    passed: bool  # Whether content meets quality threshold
    feedback: list[str]  # Specific improvement suggestions
    strengths: list[str]  # What's working well
    
    def __str__(self) -> str:
        """Format evaluation result for display."""
        status = "PASSED" if self.passed else "NEEDS REVISION"
        feedback_str = "\n".join(f"  - {f}" for f in self.feedback) if self.feedback else "  (none)"
        strengths_str = "\n".join(f"  - {s}" for s in self.strengths) if self.strengths else "  (none)"
        return f"""Evaluation: {status} (Score: {self.score:.2f})

Strengths:
{strengths_str}

Areas for Improvement:
{feedback_str}"""


@dataclass
class IterationRecord:
    """Record of a single iteration in the optimization loop."""
    iteration: int
    content: str
    score: float
    passed: bool
    feedback: list[str]
    strengths: list[str] = field(default_factory=list)
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


# =============================================================================
# Generator Component
# =============================================================================

class Generator:
    """
    Generates and revises content based on feedback.
    
    The generator handles two distinct tasks:
    1. Initial generation from a user prompt
    2. Revision based on evaluator feedback
    
    Example:
        client = anthropic.Anthropic()
        generator = Generator(client)
        
        # Generate initial content
        content = generator.generate("Write a product description for headphones")
        
        # Revise based on feedback
        revised = generator.revise(content, "Add more emotional appeal", original_prompt)
    """
    
    def __init__(
        self, 
        client: anthropic.Anthropic, 
        config: GeneratorConfig | None = None,
        system_prompt: str | None = None
    ):
        """
        Initialize the generator.
        
        Args:
            client: Anthropic API client
            config: Generator configuration (uses defaults if not provided)
            system_prompt: Custom system prompt (uses default if not provided)
        """
        self.client = client
        self.config = config or GeneratorConfig()
        
        self.system_prompt = system_prompt or """You are a skilled writer focused on 
producing clear, engaging, and well-structured content. Your writing is concise yet 
comprehensive, avoiding unnecessary filler while ensuring all important points are covered.

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
            prompt: The user's content request
            context: Optional additional context
            
        Returns:
            The generated content
        """
        user_message = prompt
        if context:
            user_message = f"Context: {context}\n\n{prompt}"
        
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
the original intent. Output only the revised content, without any preamble 
or explanation of changes."""

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


# =============================================================================
# Evaluator Component
# =============================================================================

class Evaluator:
    """
    Evaluates content quality and provides structured feedback.
    
    The evaluator assesses content against defined criteria and returns
    both a quality score and specific, actionable feedback for improvement.
    
    Example:
        client = anthropic.Anthropic()
        criteria = [
            "Clarity: Is the writing easy to understand?",
            "Engagement: Does it hold the reader's attention?"
        ]
        evaluator = Evaluator(client, criteria)
        
        result = evaluator.evaluate("Your content here...")
        print(f"Score: {result.score}, Passed: {result.passed}")
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
            criteria: List of evaluation criteria (be specific!)
            config: Evaluator configuration
        """
        self.client = client
        self.criteria = criteria
        self.config = config or EvaluatorConfig()
        
        # Build criteria list for prompt
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        
        self.system_prompt = f"""You are a content evaluator. Your job is to assess 
content quality against specific criteria and provide actionable feedback.

Evaluation Criteria:
{criteria_text}

You must evaluate content fairly and consistently. Provide specific, actionable 
feedback that can be used to improve the content.

IMPORTANT: Always respond with valid JSON in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "strengths": ["strength 1", "strength 2"],
    "improvements": ["specific improvement 1", "specific improvement 2"],
    "passed": <true if score >= {self.config.quality_threshold}, false otherwise>
}}

Be specific in your feedback. Instead of "improve clarity", say "The second 
paragraph's main point is unclear - consider stating the key takeaway in the 
first sentence."

Respond ONLY with the JSON object, no other text."""

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
            
            # Validate and extract fields
            score = float(result.get("score", 0))
            score = max(0.0, min(1.0, score))  # Clamp to valid range
            
            return EvaluationResult(
                score=score,
                passed=bool(result.get("passed", score >= self.config.quality_threshold)),
                feedback=result.get("improvements", []),
                strengths=result.get("strengths", [])
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback for parsing errors
            print(f"Warning: Failed to parse evaluation response: {e}")
            print(f"Response was: {response_text[:200]}...")
            return EvaluationResult(
                score=0.5,
                passed=False,
                feedback=["Evaluation parsing failed - please try again"],
                strengths=[]
            )


# =============================================================================
# Evaluator-Optimizer Loop Controller
# =============================================================================

class EvaluatorOptimizer:
    """
    Orchestrates the evaluator-optimizer loop.
    
    This class manages the iterative refinement process, tracking progress,
    detecting convergence, and enforcing iteration limits.
    
    Example:
        client = anthropic.Anthropic()
        generator = Generator(client)
        evaluator = Evaluator(client, criteria)
        
        optimizer = EvaluatorOptimizer(generator, evaluator)
        result = optimizer.optimize("Write a blog post about Python")
        
        print(result.final_content)
        print(result.summary())
    """
    
    def __init__(
        self,
        generator: Generator,
        evaluator: Evaluator,
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
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)
    
    def _check_convergence(
        self, 
        current_score: float, 
        previous_score: float, 
        iteration: int
    ) -> bool:
        """
        Check if the optimization has converged.
        
        Args:
            current_score: Score from current iteration
            previous_score: Score from previous iteration
            iteration: Current iteration number
            
        Returns:
            True if converged and should stop, False otherwise
        """
        if iteration < self.config.min_iterations:
            return False
        
        if iteration == 1:
            return False
        
        improvement = current_score - previous_score
        return improvement < self.config.convergence_threshold
    
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
        best_content = ""
        best_score = 0.0
        
        for iteration in range(1, self.config.max_iterations + 1):
            self._log(f"\n{'='*50}")
            self._log(f"Iteration {iteration}/{self.config.max_iterations}")
            self._log('='*50)
            
            # Generate or revise content
            if iteration == 1:
                self._log("Generating initial content...")
                current_content = self.generator.generate(prompt, context)
            else:
                # Format feedback for revision
                feedback_text = "\n".join(f"- {f}" for f in history[-1].feedback)
                self._log("Revising based on feedback...")
                current_content = self.generator.revise(
                    current_content, 
                    feedback_text,
                    prompt
                )
            
            # Evaluate the content
            self._log("Evaluating content...")
            evaluation = self.evaluator.evaluate(current_content, context)
            
            # Track best version (for rollback if needed)
            if evaluation.score > best_score:
                best_content = current_content
                best_score = evaluation.score
            
            # Record this iteration
            record = IterationRecord(
                iteration=iteration,
                content=current_content,
                score=evaluation.score,
                passed=evaluation.passed,
                feedback=evaluation.feedback,
                strengths=evaluation.strengths
            )
            history.append(record)
            
            self._log(f"Score: {evaluation.score:.2f}")
            if evaluation.strengths:
                self._log(f"Strengths: {', '.join(evaluation.strengths[:2])}")
            if evaluation.feedback and not evaluation.passed:
                self._log(f"Improvements needed: {len(evaluation.feedback)}")
            
            # Check if we've passed
            if evaluation.passed:
                self._log(f"\n✓ Content passed evaluation!")
                return OptimizationResult(
                    final_content=current_content,
                    final_score=evaluation.score,
                    iterations_used=iteration,
                    converged=True,
                    history=history
                )
            
            # Check for convergence
            if self._check_convergence(evaluation.score, previous_score, iteration):
                self._log(f"\n⚠ Converged (minimal improvement detected)")
                # Return best version if current is worse
                final_content = best_content if best_score > evaluation.score else current_content
                return OptimizationResult(
                    final_content=final_content,
                    final_score=max(best_score, evaluation.score),
                    iterations_used=iteration,
                    converged=True,
                    history=history
                )
            
            previous_score = evaluation.score
        
        # Max iterations reached - return best version
        self._log(f"\n⚠ Max iterations ({self.config.max_iterations}) reached")
        
        final_content = best_content if best_score > history[-1].score else history[-1].content
        final_score = max(best_score, history[-1].score)
        
        return OptimizationResult(
            final_content=final_content,
            final_score=final_score,
            iterations_used=self.config.max_iterations,
            converged=False,
            history=history
        )


# =============================================================================
# Demonstration
# =============================================================================

if __name__ == "__main__":
    # Create client
    client = anthropic.Anthropic()
    
    # Define evaluation criteria
    criteria = [
        "Clarity: Is the message easy to understand on first read?",
        "Persuasiveness: Does it make a compelling case?",
        "Conciseness: Is it free of unnecessary words?",
        "Call to Action: Is there a clear next step for the reader?",
    ]
    
    # Create components
    generator = Generator(client)
    evaluator = Evaluator(client, criteria)
    optimizer = EvaluatorOptimizer(
        generator, 
        evaluator,
        LoopConfig(max_iterations=3, verbose=True)
    )
    
    # Run optimization
    prompt = "Write a short email inviting colleagues to a team lunch next Friday at noon."
    result = optimizer.optimize(prompt)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result.summary())
    print("\n--- Final Content ---")
    print(result.final_content)
