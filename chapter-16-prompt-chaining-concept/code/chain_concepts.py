"""
Prompt Chaining Concepts and Patterns

Chapter 16: Prompt Chaining - Concept and Design

This file illustrates the structural patterns and concepts discussed in Chapter 16.
These are conceptual examples showing how to think about chain design.
Full implementations are provided in Chapter 17.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
from enum import Enum


# =============================================================================
# STEP DEFINITION PATTERNS
# =============================================================================

@dataclass
class ChainStep:
    """
    Represents a single step in a prompt chain.
    
    Each step has a clear purpose, defined inputs/outputs, and optionally
    a quality gate to validate its output.
    """
    name: str
    description: str
    prompt_template: str
    output_format: str  # e.g., "json", "text", "markdown"
    validator: Optional[Callable[[Any], bool]] = None
    model: str = "claude-sonnet-4-20250514"
    
    def __repr__(self) -> str:
        return f"ChainStep(name='{self.name}')"


# Example step definitions for a blog post chain
BLOG_POST_STEPS = [
    ChainStep(
        name="ideation",
        description="Generate multiple angles for the blog topic",
        prompt_template="""Generate 5 unique angles for a blog post about: {topic}
Target audience: {audience}

For each angle, provide:
1. A one-sentence description of the angle
2. A compelling hook (opening line)

Output as JSON: {{"angles": [{{"angle": "...", "hook": "..."}}, ...]}}""",
        output_format="json",
        validator=lambda x: isinstance(x, dict) and len(x.get("angles", [])) == 5
    ),
    ChainStep(
        name="selection_and_outline",
        description="Choose best angle and create detailed outline",
        prompt_template="""Given these angles:
{angles}

Select the most compelling angle for the target audience: {audience}

Create a detailed outline with:
- Title
- Introduction summary
- 3-5 main sections with key points
- Conclusion summary

Output as JSON with structure: {{"chosen_angle": "...", "outline": {{...}}}}""",
        output_format="json",
        validator=lambda x: "outline" in x and "sections" in x.get("outline", {})
    ),
    ChainStep(
        name="draft_writing",
        description="Write the full blog post from outline",
        prompt_template="""Write a complete blog post following this outline:
{outline}

Requirements:
- Engaging, conversational tone
- Target length: {word_count} words
- Include relevant examples
- Follow the outline structure exactly

Output the complete blog post as markdown.""",
        output_format="markdown",
        validator=lambda x: len(x.split()) >= 200  # Minimum word count
    ),
    ChainStep(
        name="polish_and_seo",
        description="Polish the draft and optimize for SEO",
        prompt_template="""Review and polish this blog post draft:
{draft}

Improvements to make:
1. Strengthen the opening hook
2. Improve transitions between sections
3. Make headings SEO-friendly for the keyword: {keyword}
4. Ensure the conclusion has a clear call-to-action

Output the final polished blog post as markdown.""",
        output_format="markdown",
        validator=None  # Final step, validated by overall chain output
    ),
]


# =============================================================================
# CHAIN CONFIGURATION PATTERNS
# =============================================================================

class QualityLevel(Enum):
    """Configuration presets for chain quality vs speed tradeoff."""
    FAST = "fast"
    BALANCED = "balanced"
    CAREFUL = "careful"


@dataclass
class ChainConfig:
    """
    Configuration for chain execution behavior.
    
    Allows tuning the tradeoff between speed and reliability.
    """
    max_retries: int
    validate_every_step: bool
    use_semantic_validation: bool
    fast_model_for_simple_steps: bool
    timeout_per_step: int  # seconds
    
    @classmethod
    def for_quality_level(cls, level: QualityLevel) -> "ChainConfig":
        """Factory method to create config from quality level preset."""
        configs = {
            QualityLevel.FAST: cls(
                max_retries=1,
                validate_every_step=False,
                use_semantic_validation=False,
                fast_model_for_simple_steps=True,
                timeout_per_step=30
            ),
            QualityLevel.BALANCED: cls(
                max_retries=2,
                validate_every_step=True,
                use_semantic_validation=False,
                fast_model_for_simple_steps=False,
                timeout_per_step=60
            ),
            QualityLevel.CAREFUL: cls(
                max_retries=3,
                validate_every_step=True,
                use_semantic_validation=True,
                fast_model_for_simple_steps=False,
                timeout_per_step=120
            ),
        }
        return configs[level]


# =============================================================================
# CHAIN CONTEXT PATTERN
# =============================================================================

@dataclass
class ChainContext:
    """
    Carries context through the chain.
    
    As the chain executes, context accumulates decisions and artifacts
    that later steps may need. This ensures consistency across steps.
    """
    # Original inputs
    initial_input: dict
    
    # Accumulated results from each step
    step_results: dict
    
    # Decisions made during execution (e.g., "chosen_tone": "formal")
    decisions: dict
    
    # Metadata about execution
    execution_log: list
    
    def add_step_result(self, step_name: str, result: Any) -> None:
        """Record the result of a completed step."""
        self.step_results[step_name] = result
        self.execution_log.append({
            "step": step_name,
            "status": "completed",
            "result_type": type(result).__name__
        })
    
    def add_decision(self, key: str, value: Any) -> None:
        """Record a decision that affects later steps."""
        self.decisions[key] = value
        self.execution_log.append({
            "decision": key,
            "value": value
        })
    
    def get_for_step(self, step_name: str) -> dict:
        """Get relevant context for a specific step."""
        return {
            "initial_input": self.initial_input,
            "previous_results": self.step_results,
            "decisions": self.decisions
        }


# =============================================================================
# ARCHITECTURE VISUALIZATION
# =============================================================================

def print_chain_architecture(steps: list[ChainStep], chain_name: str) -> None:
    """
    Print a text visualization of a chain's architecture.
    
    Useful for documentation and debugging.
    """
    print("=" * 70)
    print(f"CHAIN: {chain_name}")
    print("=" * 70)
    print()
    
    for i, step in enumerate(steps, 1):
        print(f"┌{'─' * 68}┐")
        print(f"│ STEP {i}: {step.name:<58} │")
        print(f"├{'─' * 68}┤")
        print(f"│ {step.description:<66} │")
        print(f"│ Output format: {step.output_format:<51} │")
        print(f"│ Model: {step.model:<59} │")
        has_gate = "Yes" if step.validator else "No"
        print(f"│ Has quality gate: {has_gate:<48} │")
        print(f"└{'─' * 68}┘")
        
        if i < len(steps):
            if step.validator:
                print("              │")
                print("              ▼")
                print("     ┌─────────────────┐")
                print(f"     │   GATE {i}        │")
                print("     │   [Validation]  │")
                print("     └─────────────────┘")
            print("              │")
            print("              ▼")
    
    print()
    print("     ┌─────────────────┐")
    print("     │   FINAL OUTPUT  │")
    print("     └─────────────────┘")
    print()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demonstrate the chain architecture visualization
    print("\n" + "=" * 70)
    print("PROMPT CHAINING CONCEPTS - CHAPTER 16")
    print("=" * 70 + "\n")
    
    # Show the blog post chain architecture
    print_chain_architecture(BLOG_POST_STEPS, "Blog Post Generator")
    
    # Show configuration options
    print("\n" + "=" * 70)
    print("CHAIN CONFIGURATION OPTIONS")
    print("=" * 70 + "\n")
    
    for level in QualityLevel:
        config = ChainConfig.for_quality_level(level)
        print(f"{level.value.upper()} mode:")
        print(f"  - Max retries: {config.max_retries}")
        print(f"  - Validate every step: {config.validate_every_step}")
        print(f"  - Semantic validation: {config.use_semantic_validation}")
        print(f"  - Fast model for simple steps: {config.fast_model_for_simple_steps}")
        print(f"  - Timeout per step: {config.timeout_per_step}s")
        print()
    
    # Demonstrate chain context
    print("=" * 70)
    print("CHAIN CONTEXT EXAMPLE")
    print("=" * 70 + "\n")
    
    context = ChainContext(
        initial_input={"topic": "renewable energy", "audience": "homeowners"},
        step_results={},
        decisions={},
        execution_log=[]
    )
    
    # Simulate chain execution
    context.add_step_result("ideation", {"angles": ["cost savings", "environmental impact"]})
    context.add_decision("chosen_angle", "cost savings")
    context.add_step_result("outline", {"title": "Save Money with Solar", "sections": [...]})
    
    print("Context after partial execution:")
    print(f"  Step results: {list(context.step_results.keys())}")
    print(f"  Decisions: {context.decisions}")
    print(f"  Execution log entries: {len(context.execution_log)}")
