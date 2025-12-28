"""
Code Review Assistant - Exercise Solution

This exercise implements a code review assistant using the evaluator-optimizer 
pattern. The assistant takes code, evaluates it against quality criteria,
suggests improvements, applies them, and re-evaluates.

Chapter 25: Evaluator-Optimizer - Implementation

Exercise Requirements:
1. Take a code snippet as input
2. Evaluate it against code quality criteria (readability, efficiency, best practices)
3. Suggest improvements
4. Apply improvements and re-evaluate
5. Return the improved code along with the improvement history
6. Use at least 5 code quality criteria
7. Implement convergence detection
8. Handle the case where code is already good
9. Maximum 4 iterations
"""

import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Configuration
# =============================================================================

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 4
QUALITY_THRESHOLD = 0.8
CONVERGENCE_THRESHOLD = 0.03  # Stop if improvement < 3%

# Code quality criteria (requirement: at least 5)
CODE_CRITERIA = [
    "Readability: Is the code easy to read and understand? Are variable/function names descriptive?",
    "Best Practices: Does the code follow Python best practices (PEP 8, idioms)?",
    "Error Handling: Does the code handle potential errors appropriately?",
    "Efficiency: Is the code reasonably efficient? Are there obvious performance issues?",
    "Documentation: Are there appropriate docstrings and comments where needed?",
    "Modularity: Is the code well-organized? Are functions focused on single tasks?",
    "Type Safety: Are type hints used appropriately?",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReviewResult:
    """Result from a single code review iteration."""
    iteration: int
    code: str
    score: float
    passed: bool
    issues: list[str]
    strengths: list[str]


@dataclass
class OptimizationResult:
    """Final result from the code optimization process."""
    original_code: str
    improved_code: str
    original_score: float
    final_score: float
    iterations: int
    history: list[ReviewResult]
    
    def improvement_summary(self) -> str:
        """Generate a summary of improvements made."""
        improvement = self.final_score - self.original_score
        return f"""Code Review Complete
{'='*40}
Original Score: {self.original_score:.2f}
Final Score:    {self.final_score:.2f}
Improvement:    {improvement:+.2f}
Iterations:     {self.iterations}
"""


# =============================================================================
# Code Review Assistant
# =============================================================================

class CodeReviewAssistant:
    """
    A code review assistant that iteratively improves code quality.
    
    Example:
        assistant = CodeReviewAssistant()
        result = assistant.review_and_improve('''
        def calc(x,y):
            return x+y
        ''')
        print(result.improved_code)
    """
    
    def __init__(
        self,
        criteria: list[str] | None = None,
        max_iterations: int = MAX_ITERATIONS,
        quality_threshold: float = QUALITY_THRESHOLD,
        verbose: bool = True
    ):
        """
        Initialize the code review assistant.
        
        Args:
            criteria: Custom code quality criteria (uses defaults if None)
            max_iterations: Maximum improvement iterations
            quality_threshold: Score needed to pass (0.0 to 1.0)
            verbose: Print progress during review
        """
        self.client = anthropic.Anthropic()
        self.criteria = criteria or CODE_CRITERIA
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.verbose = verbose
        
        # Build criteria text for prompts
        self.criteria_text = "\n".join(f"- {c}" for c in self.criteria)
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _evaluate_code(self, code: str, context: str = "") -> dict:
        """
        Evaluate code against quality criteria.
        
        Returns dict with: score, passed, issues, strengths
        """
        system_prompt = f"""You are a code reviewer evaluating Python code quality.

Evaluation Criteria:
{self.criteria_text}

Score the code 0.0 to 1.0 based on how well it meets these criteria.
A score of {self.quality_threshold} or higher passes the review.

IMPORTANT: Respond with ONLY this JSON format:
{{
    "score": 0.75,
    "passed": false,
    "issues": ["specific issue 1", "specific issue 2"],
    "strengths": ["what's done well 1", "what's done well 2"]
}}

Be specific about issues - reference exact lines or patterns.
If the code is already good quality, acknowledge it with high score and few/no issues."""

        user_prompt = f"""Review this Python code:

```python
{code}
```"""
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"
        
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=800,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            
            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            
            return {
                "score": score,
                "passed": score >= self.quality_threshold,
                "issues": result.get("issues", []),
                "strengths": result.get("strengths", [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            self._log(f"Warning: Failed to parse evaluation: {e}")
            return {
                "score": 0.5,
                "passed": False,
                "issues": ["Evaluation parsing failed"],
                "strengths": []
            }
    
    def _improve_code(self, code: str, issues: list[str], context: str = "") -> str:
        """
        Improve code based on identified issues.
        
        Returns the improved code.
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        system_prompt = """You are a Python code improver. Your job is to fix 
code issues while preserving the original functionality.

Rules:
1. Address each issue specifically
2. Keep the code's original purpose and logic
3. Follow Python best practices (PEP 8)
4. Add docstrings and type hints where appropriate
5. Only change what needs to be changed

Output ONLY the improved Python code, no explanations."""

        user_prompt = f"""Improve this code to address the following issues:

Issues to fix:
{issues_text}

Original code:
```python
{code}
```

Output the improved code:"""
        
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"
        
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        improved_code = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if improved_code.startswith("```python"):
            improved_code = improved_code[9:]
        if improved_code.startswith("```"):
            improved_code = improved_code[3:]
        if improved_code.endswith("```"):
            improved_code = improved_code[:-3]
        
        return improved_code.strip()
    
    def review_and_improve(self, code: str, context: str = "") -> OptimizationResult:
        """
        Review code and iteratively improve it until it passes.
        
        Args:
            code: The Python code to review and improve
            context: Optional context about the code's purpose
            
        Returns:
            OptimizationResult with improved code and history
        """
        self._log(f"\n{'='*50}")
        self._log("CODE REVIEW ASSISTANT")
        self._log(f"{'='*50}")
        
        original_code = code
        current_code = code
        history: list[ReviewResult] = []
        previous_score = 0.0
        best_code = code
        best_score = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Evaluate current code
            self._log("Evaluating code...")
            evaluation = self._evaluate_code(current_code, context)
            
            self._log(f"Score: {evaluation['score']:.2f}")
            if evaluation["strengths"]:
                self._log(f"Strengths: {len(evaluation['strengths'])} identified")
            if evaluation["issues"]:
                self._log(f"Issues: {len(evaluation['issues'])} found")
            
            # Track best version
            if evaluation["score"] > best_score:
                best_code = current_code
                best_score = evaluation["score"]
            
            # Record history
            history.append(ReviewResult(
                iteration=iteration,
                code=current_code,
                score=evaluation["score"],
                passed=evaluation["passed"],
                issues=evaluation["issues"],
                strengths=evaluation["strengths"]
            ))
            
            # Check if passed (requirement: handle good code)
            if evaluation["passed"]:
                self._log(f"\n✓ Code passed review! (score >= {self.quality_threshold})")
                break
            
            # Check for convergence (requirement: implement convergence detection)
            if iteration > 1:
                improvement = evaluation["score"] - previous_score
                if improvement < CONVERGENCE_THRESHOLD:
                    self._log(f"\n⚠ Converged (improvement {improvement:.3f} < {CONVERGENCE_THRESHOLD})")
                    break
            
            # If issues found, improve the code
            if evaluation["issues"]:
                self._log("Improving code...")
                current_code = self._improve_code(
                    current_code, 
                    evaluation["issues"],
                    context
                )
            else:
                # No specific issues but didn't pass - unusual case
                self._log("No specific issues identified but score below threshold")
                break
            
            previous_score = evaluation["score"]
        
        # Return best version
        original_eval = history[0] if history else None
        original_score = original_eval.score if original_eval else 0.0
        
        return OptimizationResult(
            original_code=original_code,
            improved_code=best_code,
            original_score=original_score,
            final_score=best_score,
            iterations=len(history),
            history=history
        )
    
    def review_only(self, code: str, context: str = "") -> dict:
        """
        Review code without making improvements.
        
        Useful for assessment without modification.
        
        Returns:
            dict with score, passed, issues, strengths
        """
        self._log(f"\n{'='*50}")
        self._log("CODE REVIEW (Assessment Only)")
        self._log(f"{'='*50}")
        
        evaluation = self._evaluate_code(code, context)
        
        self._log(f"\nScore: {evaluation['score']:.2f}")
        self._log(f"Status: {'PASSED' if evaluation['passed'] else 'NEEDS IMPROVEMENT'}")
        
        if evaluation["strengths"]:
            self._log("\nStrengths:")
            for s in evaluation["strengths"]:
                self._log(f"  ✓ {s}")
        
        if evaluation["issues"]:
            self._log("\nIssues:")
            for i in evaluation["issues"]:
                self._log(f"  ✗ {i}")
        
        return evaluation


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_poor_code():
    """Demonstrate improving poor quality code."""
    
    # Deliberately poor code for demonstration
    poor_code = '''
def calc(x,y,op):
    if op=="add":
        return x+y
    if op=="sub":
        return x-y
    if op=="mul":
        return x*y
    if op=="div":
        return x/y

def proc(data):
    r=[]
    for i in range(len(data)):
        if data[i]>0:
            r.append(data[i]*2)
    return r
'''
    
    assistant = CodeReviewAssistant(verbose=True)
    result = assistant.review_and_improve(
        poor_code,
        context="Basic calculator and data processing utilities"
    )
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result.improvement_summary())
    
    print("--- Original Code ---")
    print(result.original_code)
    
    print("\n--- Improved Code ---")
    print(result.improved_code)
    
    return result


def demo_good_code():
    """Demonstrate that good code requires minimal changes."""
    
    # Already decent code
    good_code = '''
def calculate(a: float, b: float, operation: str) -> float:
    """
    Perform a basic arithmetic operation.
    
    Args:
        a: First operand
        b: Second operand
        operation: One of 'add', 'subtract', 'multiply', 'divide'
    
    Returns:
        Result of the operation
    
    Raises:
        ValueError: If operation is unknown or division by zero
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x / y if y != 0 else None,
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    result = operations[operation](a, b)
    if result is None:
        raise ValueError("Cannot divide by zero")
    
    return result
'''
    
    assistant = CodeReviewAssistant(verbose=True)
    result = assistant.review_and_improve(
        good_code,
        context="Calculator utility function"
    )
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result.improvement_summary())
    
    # For good code, improvements should be minimal
    print(f"Code was {'already good!' if result.iterations == 1 else 'improved'}")
    
    return result


def demo_review_only():
    """Demonstrate review-only mode."""
    
    code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
    
    assistant = CodeReviewAssistant(verbose=True)
    evaluation = assistant.review_only(
        code,
        context="Fibonacci sequence generator"
    )
    
    return evaluation


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    demos = {
        "poor": ("Improve poor quality code", demo_poor_code),
        "good": ("Review already good code", demo_good_code),
        "review": ("Review only (no improvements)", demo_review_only),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, func = demos[sys.argv[1]]
        print(f"Running: {name}")
        func()
    else:
        print("Code Review Assistant - Exercise Solution")
        print("="*50)
        print("\nAvailable demos:")
        for key, (description, _) in demos.items():
            print(f"  python exercise.py {key}  - {description}")
        
        print("\nRunning default demo (poor code improvement)...")
        demo_poor_code()
