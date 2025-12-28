"""
Conceptual overview of the Evaluator-Optimizer pattern.

Chapter 24: Evaluator-Optimizer - Concept and Design

NOTE: This file is pseudocode/conceptual—it shows the structure
of the pattern but is NOT meant to be run directly. We'll implement
the full working version in Chapter 25.

The purpose is to help you understand the architecture before diving
into implementation details.
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# Data Structures
# ============================================================

@dataclass
class EvaluationResult:
    """The evaluator's assessment of generated content."""
    
    status: str  # "approved" or "needs_revision"
    scores: dict[str, float]  # criterion -> score (1-5)
    feedback: str  # Specific, actionable feedback
    overall_score: float  # Average of all criteria


@dataclass
class IterationRecord:
    """Record of a single iteration for debugging/analysis."""
    
    iteration_number: int
    generated_content: str
    evaluation: EvaluationResult
    feedback_given: str


@dataclass
class OptimizationResult:
    """The final result of the optimization process."""
    
    final_content: str
    status: str  # "approved", "best_effort", "failed"
    iterations_used: int
    final_scores: dict[str, float]
    history: list[IterationRecord]


# ============================================================
# Conceptual Generator
# ============================================================

class ConceptualGenerator:
    """
    The Generator's responsibility: Create or improve content.
    
    On iteration 1: Creates initial content from task description
    On iteration 2+: Revises content based on evaluator feedback
    """
    
    def __init__(self, system_prompt: str):
        """
        Initialize with a system prompt defining the generator's role.
        
        Example system prompt:
        "You are an expert copywriter specializing in product descriptions.
         Write compelling, benefit-focused descriptions that drive conversions."
        """
        self.system_prompt = system_prompt
    
    def generate(
        self, 
        task: str, 
        previous_output: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> str:
        """
        Generate or revise content.
        
        Args:
            task: The original task description
            previous_output: The last generated version (None on first iteration)
            feedback: The evaluator's feedback to incorporate (None on first iteration)
        
        Returns:
            Generated or revised content
        
        Implementation (Chapter 25):
        - First iteration: Just the task
        - Later iterations: Task + previous output + feedback
        - Uses LLM to generate response
        """
        if previous_output is None:
            # First iteration: generate from scratch
            prompt = f"Task: {task}"
        else:
            # Later iterations: revise based on feedback
            prompt = f"""
            Original task: {task}
            
            Previous version:
            {previous_output}
            
            Feedback to address:
            {feedback}
            
            Please revise the content to address the feedback.
            """
        
        # In Chapter 25, this calls the Anthropic API
        # response = client.messages.create(...)
        # return response.content[0].text
        
        return "[Generated content would appear here]"


# ============================================================
# Conceptual Evaluator
# ============================================================

class ConceptualEvaluator:
    """
    The Evaluator's responsibility: Assess content against criteria.
    
    Provides:
    1. Scores for each criterion
    2. Specific, actionable feedback
    3. Approve/revise decision
    """
    
    def __init__(self, criteria: list[dict], threshold: float = 4.0):
        """
        Initialize with evaluation criteria.
        
        Args:
            criteria: List of criteria with names, descriptions, and rubrics
            threshold: Minimum score to approve (default 4.0)
        
        Example criteria:
        [
            {
                "name": "clarity",
                "description": "How clear and understandable is the content?",
                "rubric": "5=crystal clear, 4=mostly clear, 3=adequate, 2=confusing, 1=incomprehensible"
            },
            ...
        ]
        """
        self.criteria = criteria
        self.threshold = threshold
    
    def evaluate(self, content: str) -> EvaluationResult:
        """
        Evaluate content against all criteria.
        
        Args:
            content: The generated content to evaluate
        
        Returns:
            EvaluationResult with scores, feedback, and decision
        
        Implementation (Chapter 25):
        - Constructs prompt with criteria and rubrics
        - Asks LLM to score each criterion
        - Extracts structured response (JSON)
        - Returns parsed evaluation
        """
        # In Chapter 25, this calls the Anthropic API with structured output
        # response = client.messages.create(
        #     messages=[{"role": "user", "content": evaluation_prompt}],
        #     ...
        # )
        # parsed = json.loads(response.content[0].text)
        
        # Placeholder return
        return EvaluationResult(
            status="needs_revision",
            scores={"criterion1": 3.5, "criterion2": 4.0},
            feedback="[Specific feedback would appear here]",
            overall_score=3.75
        )


# ============================================================
# Conceptual Evaluator-Optimizer Loop
# ============================================================

class ConceptualEvaluatorOptimizer:
    """
    Orchestrates the generator-evaluator feedback loop.
    
    This is the main class users interact with.
    """
    
    def __init__(
        self,
        generator: ConceptualGenerator,
        evaluator: ConceptualEvaluator,
        max_iterations: int = 5,
        min_improvement: float = 0.1
    ):
        """
        Initialize the optimizer.
        
        Args:
            generator: The content generator
            evaluator: The content evaluator
            max_iterations: Safety limit on iterations
            min_improvement: Minimum score improvement to continue
        """
        self.generator = generator
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
    
    def optimize(self, task: str) -> OptimizationResult:
        """
        Run the optimization loop until criteria are met or limits reached.
        
        Args:
            task: The task description
        
        Returns:
            OptimizationResult with final content and metadata
        """
        history: list[IterationRecord] = []
        previous_output: Optional[str] = None
        feedback: Optional[str] = None
        best_output: str = ""
        best_score: float = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            # Step 1: Generate (or revise) content
            content = self.generator.generate(
                task=task,
                previous_output=previous_output,
                feedback=feedback
            )
            
            # Step 2: Evaluate the content
            evaluation = self.evaluator.evaluate(content)
            
            # Step 3: Record this iteration
            record = IterationRecord(
                iteration_number=iteration,
                generated_content=content,
                evaluation=evaluation,
                feedback_given=feedback or "(initial generation)"
            )
            history.append(record)
            
            # Step 4: Track best version
            if evaluation.overall_score > best_score:
                best_output = content
                best_score = evaluation.overall_score
            
            # Step 5: Check stopping conditions
            
            # Condition A: Quality threshold met
            if evaluation.status == "approved":
                return OptimizationResult(
                    final_content=content,
                    status="approved",
                    iterations_used=iteration,
                    final_scores=evaluation.scores,
                    history=history
                )
            
            # Condition B: Diminishing returns
            if iteration > 1:
                prev_score = history[-2].evaluation.overall_score
                improvement = evaluation.overall_score - prev_score
                if improvement < self.min_improvement:
                    return OptimizationResult(
                        final_content=best_output,
                        status="best_effort",
                        iterations_used=iteration,
                        final_scores=evaluation.scores,
                        history=history
                    )
            
            # Condition C: Convergent feedback
            if iteration > 1 and feedback == evaluation.feedback:
                return OptimizationResult(
                    final_content=best_output,
                    status="best_effort",
                    iterations_used=iteration,
                    final_scores=evaluation.scores,
                    history=history
                )
            
            # Step 6: Prepare for next iteration
            previous_output = content
            feedback = evaluation.feedback
        
        # Max iterations reached without approval
        return OptimizationResult(
            final_content=best_output,
            status="best_effort",
            iterations_used=self.max_iterations,
            final_scores=history[-1].evaluation.scores,
            history=history
        )


# ============================================================
# Conceptual Usage Example
# ============================================================

def conceptual_usage_example():
    """
    Shows how the pattern would be used (not runnable).
    
    This is what we'll implement in Chapter 25.
    """
    
    # Define the generator
    generator = ConceptualGenerator(
        system_prompt="""
        You are an expert product copywriter. Write compelling,
        benefit-focused product descriptions that convert browsers
        into buyers.
        """
    )
    
    # Define evaluation criteria
    criteria = [
        {
            "name": "hook_quality",
            "description": "How compelling is the opening?",
            "rubric": "5=irresistible, 4=engaging, 3=adequate, 2=weak, 1=poor"
        },
        {
            "name": "benefit_clarity", 
            "description": "Are benefits clear and customer-focused?",
            "rubric": "5=crystal clear benefits, 4=clear, 3=mixed, 2=feature-heavy, 1=features only"
        },
        {
            "name": "persuasion",
            "description": "Does it create desire and motivate action?",
            "rubric": "5=must-have feeling, 4=want it, 3=neutral, 2=uninterested, 1=turned off"
        }
    ]
    
    # Define the evaluator
    evaluator = ConceptualEvaluator(
        criteria=criteria,
        threshold=4.0  # All criteria must score 4+ to approve
    )
    
    # Create the optimizer
    optimizer = ConceptualEvaluatorOptimizer(
        generator=generator,
        evaluator=evaluator,
        max_iterations=5,
        min_improvement=0.1
    )
    
    # Run optimization
    task = """
    Write a product description for wireless noise-canceling headphones.
    Key features: 40-hour battery, premium 40mm drivers, active noise cancellation.
    Target audience: Remote workers and commuters.
    Tone: Professional but warm.
    Length: 100-150 words.
    """
    
    result = optimizer.optimize(task)
    
    # Process result
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iterations_used}")
    print(f"Final Scores: {result.final_scores}")
    print(f"\nFinal Content:\n{result.final_content}")
    
    # Review history if needed
    if result.status != "approved":
        print("\n--- Iteration History ---")
        for record in result.history:
            print(f"\nIteration {record.iteration_number}:")
            print(f"  Score: {record.evaluation.overall_score:.1f}")
            print(f"  Feedback: {record.evaluation.feedback[:100]}...")


# ============================================================
# Key Concepts Summary
# ============================================================

"""
THE EVALUATOR-OPTIMIZER PATTERN

1. GENERATOR creates content
   - First pass: generates from task
   - Later passes: revises based on feedback

2. EVALUATOR assesses content
   - Scores against specific criteria
   - Provides actionable feedback
   - Decides: approve or revise

3. CONTROL LOOP manages the process
   - Tracks iterations
   - Detects convergence
   - Enforces limits
   - Returns best result

4. STOPPING CONDITIONS prevent waste
   - Quality threshold met → approved
   - Max iterations reached → best effort
   - Diminishing returns → best effort
   - Feedback converged → best effort

5. OUTPUT includes metadata
   - Final content
   - Status (approved/best_effort/failed)
   - Iteration count
   - Score history
   - Full iteration history (for debugging)

NEXT STEP: Implement this fully in Chapter 25!
"""
