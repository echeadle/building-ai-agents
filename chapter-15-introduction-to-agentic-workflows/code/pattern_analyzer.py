"""
Pattern Analyzer: A utility to help choose the right workflow pattern.

Chapter 15: Introduction to Agentic Workflows

This interactive tool asks questions about your task and recommends
the most appropriate workflow pattern (or suggests a simple prompt
if no workflow is needed).
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class PatternRecommendation:
    """Recommendation for which pattern to use."""
    
    pattern: str
    confidence: str  # "high", "medium", "low"
    reasoning: str
    considerations: list[str]
    alternative: Optional[str] = None


class PatternAnalyzer:
    """
    Analyzes task requirements and recommends workflow patterns.
    
    Uses a combination of rule-based analysis and LLM-assisted
    evaluation to determine the best approach.
    """
    
    PATTERNS = {
        "simple_prompt": {
            "name": "Simple Prompt",
            "description": "A single, well-crafted LLM call",
            "best_for": "Straightforward tasks with clear objectives"
        },
        "prompt_chaining": {
            "name": "Prompt Chaining",
            "description": "Sequential steps where each output feeds the next",
            "best_for": "Multi-step tasks with clear sequential dependencies"
        },
        "routing": {
            "name": "Routing",
            "description": "Classify input and direct to specialized handlers",
            "best_for": "Tasks where different input types need different handling"
        },
        "parallelization": {
            "name": "Parallelization",
            "description": "Run multiple independent tasks simultaneously",
            "best_for": "Independent subtasks or multiple perspectives needed"
        },
        "orchestrator_workers": {
            "name": "Orchestrator-Workers",
            "description": "Dynamic task decomposition with delegated execution",
            "best_for": "Complex tasks where subtasks can't be predicted"
        },
        "evaluator_optimizer": {
            "name": "Evaluator-Optimizer",
            "description": "Iterative generation with feedback-driven refinement",
            "best_for": "Tasks requiring quality iteration with clear criteria"
        }
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self.client = anthropic.Anthropic()
    
    def analyze_interactive(self) -> PatternRecommendation:
        """
        Run an interactive analysis session.
        
        Asks the user questions about their task and provides
        a recommendation based on their answers.
        """
        print("\n" + "=" * 60)
        print("WORKFLOW PATTERN ANALYZER")
        print("=" * 60)
        print("\nAnswer a few questions about your task to get a recommendation.")
        print("(Press Enter for default answer)\n")
        
        # Gather information about the task
        task_desc = input("Describe your task briefly: ").strip()
        if not task_desc:
            task_desc = "General task"
        
        # Question 1: Single call sufficiency
        print("\n1. Can a single LLM call with a good prompt handle this task?")
        print("   [y] Yes, probably")
        print("   [n] No, it's too complex")
        print("   [u] Unsure")
        single_call = input("   Answer [y/n/u]: ").strip().lower() or "u"
        
        if single_call == "y":
            return PatternRecommendation(
                pattern="simple_prompt",
                confidence="high",
                reasoning="You indicated a single LLM call should work.",
                considerations=[
                    "Start with a well-crafted prompt",
                    "Add a workflow only if quality is insufficient",
                    "Measure performance before adding complexity"
                ]
            )
        
        # Question 2: Sequential steps
        print("\n2. Does your task have clear sequential steps?")
        print("   [y] Yes, step A must complete before step B")
        print("   [n] No, steps are independent or order doesn't matter")
        sequential = input("   Answer [y/n]: ").strip().lower() or "n"
        
        # Question 3: Different input types
        print("\n3. Do different inputs need fundamentally different handling?")
        print("   [y] Yes, input type determines processing path")
        print("   [n] No, all inputs are processed similarly")
        different_types = input("   Answer [y/n]: ").strip().lower() or "n"
        
        # Question 4: Independent subtasks
        print("\n4. Are there independent subtasks that could run in parallel?")
        print("   [y] Yes, several subtasks don't depend on each other")
        print("   [n] No, everything depends on previous steps")
        parallel_possible = input("   Answer [y/n]: ").strip().lower() or "n"
        
        # Question 5: Predictable subtasks
        print("\n5. Can you predict the subtasks at design time?")
        print("   [y] Yes, I know exactly what steps are needed")
        print("   [n] No, it depends on the specific input")
        predictable = input("   Answer [y/n]: ").strip().lower() or "y"
        
        # Question 6: Iteration helpful
        print("\n6. Would iterative refinement improve the output?")
        print("   [y] Yes, multiple revision passes would help")
        print("   [n] No, first draft is usually good enough")
        iteration_helps = input("   Answer [y/n]: ").strip().lower() or "n"
        
        # Analyze answers and make recommendation
        return self._analyze_answers(
            task_desc=task_desc,
            single_call=single_call,
            sequential=sequential == "y",
            different_types=different_types == "y",
            parallel_possible=parallel_possible == "y",
            predictable=predictable == "y",
            iteration_helps=iteration_helps == "y"
        )
    
    def _analyze_answers(
        self,
        task_desc: str,
        single_call: str,
        sequential: bool,
        different_types: bool,
        parallel_possible: bool,
        predictable: bool,
        iteration_helps: bool
    ) -> PatternRecommendation:
        """Analyze answers and return a recommendation."""
        
        # Decision logic based on the chapter's framework
        
        # Different types need routing
        if different_types:
            return PatternRecommendation(
                pattern="routing",
                confidence="high",
                reasoning="Different input types requiring different handling is a clear routing case.",
                considerations=[
                    "Design a robust classifier for your input types",
                    "Create specialized handlers for each type",
                    "Include a default/fallback handler",
                    "Test classification accuracy thoroughly"
                ],
                alternative="prompt_chaining" if sequential else None
            )
        
        # Unpredictable subtasks need orchestration
        if not predictable:
            return PatternRecommendation(
                pattern="orchestrator_workers",
                confidence="high",
                reasoning="Unpredictable subtasks require dynamic task decomposition.",
                considerations=[
                    "Design a clear orchestrator prompt that understands task breakdown",
                    "Create flexible worker tasks",
                    "Plan how to synthesize worker outputs",
                    "Set limits on worker count to control costs"
                ],
                alternative="parallelization" if parallel_possible else None
            )
        
        # Independent subtasks can parallelize
        if parallel_possible and not sequential:
            return PatternRecommendation(
                pattern="parallelization",
                confidence="high",
                reasoning="Independent subtasks are ideal for parallel execution.",
                considerations=[
                    "Consider 'sectioning' (different subtasks) vs 'voting' (same task, multiple perspectives)",
                    "Plan how to aggregate parallel results",
                    "Handle partial failures gracefully",
                    "Monitor costs (parallel = more calls)"
                ],
                alternative="prompt_chaining" if sequential else None
            )
        
        # Iteration helps: evaluator-optimizer
        if iteration_helps:
            return PatternRecommendation(
                pattern="evaluator_optimizer",
                confidence="medium",
                reasoning="Clear evaluation criteria enable productive iteration.",
                considerations=[
                    "Define specific, measurable evaluation criteria",
                    "Set a maximum iteration count",
                    "Decide what 'good enough' means",
                    "Balance quality improvement vs. latency/cost"
                ],
                alternative="prompt_chaining"
            )
        
        # Sequential steps: chaining
        if sequential:
            return PatternRecommendation(
                pattern="prompt_chaining",
                confidence="high",
                reasoning="Sequential steps with clear dependencies fit the chaining pattern.",
                considerations=[
                    "Add quality gates between steps",
                    "Design clear input/output contracts for each step",
                    "Consider which steps could be parallelized",
                    "Plan error handling and rollback"
                ],
                alternative=None
            )
        
        # Default: try a simple prompt first
        return PatternRecommendation(
            pattern="simple_prompt",
            confidence="medium",
            reasoning="No clear pattern indicators. Start simple and add complexity if needed.",
            considerations=[
                "Craft a detailed, well-structured prompt",
                "Test with various inputs",
                "Identify where quality falls short",
                "Add workflow patterns only to address specific issues"
            ],
            alternative="prompt_chaining"
        )
    
    def analyze_with_llm(self, task_description: str) -> PatternRecommendation:
        """
        Use Claude to analyze a task and recommend a pattern.
        
        This provides a more nuanced analysis than the rule-based
        approach, especially for complex or ambiguous tasks.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            PatternRecommendation with LLM-generated analysis
        """
        prompt = f"""Analyze this task and recommend the best workflow pattern.

Task: {task_description}

Available patterns:
1. Simple Prompt - A single LLM call with good instructions
2. Prompt Chaining - Sequential steps, each feeding the next
3. Routing - Classify input and direct to specialized handlers
4. Parallelization - Run independent subtasks simultaneously
5. Orchestrator-Workers - Dynamic task breakdown with delegation
6. Evaluator-Optimizer - Iterative refinement with feedback

Respond with:
1. RECOMMENDED PATTERN: (name)
2. CONFIDENCE: (high/medium/low)
3. REASONING: (2-3 sentences)
4. CONSIDERATIONS: (3-4 bullet points)
5. ALTERNATIVE: (pattern name or "none")

Be practical and favor simplicity when possible."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response (in a real implementation, you'd use
        # structured output - we'll cover that in Chapter 13)
        response_text = response.content[0].text
        
        # Basic parsing (simplified for this example)
        lines = response_text.strip().split("\n")
        pattern = "simple_prompt"
        confidence = "medium"
        reasoning = ""
        considerations = []
        alternative = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("1. RECOMMENDED PATTERN:"):
                pattern_name = line.split(":", 1)[1].strip().lower()
                pattern = self._normalize_pattern_name(pattern_name)
            elif line.startswith("2. CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().lower()
            elif line.startswith("3. REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("-") or line.startswith("•"):
                considerations.append(line[1:].strip())
            elif line.startswith("5. ALTERNATIVE:"):
                alt = line.split(":", 1)[1].strip().lower()
                if alt != "none":
                    alternative = self._normalize_pattern_name(alt)
        
        return PatternRecommendation(
            pattern=pattern,
            confidence=confidence,
            reasoning=reasoning,
            considerations=considerations,
            alternative=alternative
        )
    
    def _normalize_pattern_name(self, name: str) -> str:
        """Convert various pattern name formats to internal keys."""
        name = name.lower().replace("-", "_").replace(" ", "_")
        
        mappings = {
            "simple": "simple_prompt",
            "simple_prompt": "simple_prompt",
            "chaining": "prompt_chaining",
            "prompt_chaining": "prompt_chaining",
            "routing": "routing",
            "router": "routing",
            "parallel": "parallelization",
            "parallelization": "parallelization",
            "orchestrator": "orchestrator_workers",
            "orchestrator_workers": "orchestrator_workers",
            "evaluator": "evaluator_optimizer",
            "evaluator_optimizer": "evaluator_optimizer"
        }
        
        return mappings.get(name, "simple_prompt")
    
    def print_recommendation(self, rec: PatternRecommendation) -> None:
        """Print a formatted recommendation."""
        pattern_info = self.PATTERNS.get(rec.pattern, {})
        
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        
        print(f"\n📋 Pattern: {pattern_info.get('name', rec.pattern)}")
        print(f"   {pattern_info.get('description', '')}")
        print(f"\n🎯 Confidence: {rec.confidence.upper()}")
        print(f"\n💭 Reasoning:\n   {rec.reasoning}")
        
        print("\n📝 Considerations:")
        for c in rec.considerations:
            print(f"   • {c}")
        
        if rec.alternative:
            alt_info = self.PATTERNS.get(rec.alternative, {})
            print(f"\n🔄 Alternative: {alt_info.get('name', rec.alternative)}")
        
        print("\n" + "=" * 60)


def main():
    """Run the pattern analyzer."""
    analyzer = PatternAnalyzer()
    
    print("\nWelcome to the Workflow Pattern Analyzer!")
    print("\nChoose analysis mode:")
    print("[1] Interactive (answer questions)")
    print("[2] LLM-assisted (describe your task)")
    
    choice = input("\nChoice [1/2]: ").strip() or "1"
    
    if choice == "1":
        recommendation = analyzer.analyze_interactive()
    else:
        print("\nDescribe your task in detail:")
        task = input("> ").strip()
        if not task:
            task = "Process user feedback and generate appropriate responses"
        print("\nAnalyzing with Claude...")
        recommendation = analyzer.analyze_with_llm(task)
    
    analyzer.print_recommendation(recommendation)


if __name__ == "__main__":
    main()
