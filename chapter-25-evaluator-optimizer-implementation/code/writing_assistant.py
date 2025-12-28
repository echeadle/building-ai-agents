"""
Writing Assistant using the Evaluator-Optimizer Pattern

A complete example that iteratively improves written content
based on structured feedback against customizable criteria.

Chapter 25: Evaluator-Optimizer - Implementation

Usage:
    python writing_assistant.py
    
    Or import and use programmatically:
    
    from writing_assistant import WritingAssistant
    
    assistant = WritingAssistant()
    result = assistant.write("Write a blog post about Python decorators")
    print(result.final_content)
"""

import os
from dotenv import load_dotenv
import anthropic

# Import components from the evaluator module
from evaluator import (
    Generator, 
    GeneratorConfig,
    Evaluator, 
    EvaluatorConfig,
    EvaluatorOptimizer, 
    LoopConfig,
    OptimizationResult
)

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Predefined Criteria Sets
# =============================================================================

# Default criteria for general writing tasks
GENERAL_WRITING_CRITERIA = [
    "Clarity: Is the writing easy to understand? Are ideas expressed clearly without ambiguity?",
    "Structure: Is the content well-organized with logical flow between paragraphs?",
    "Conciseness: Is the writing free of unnecessary words, redundancy, and filler?",
    "Engagement: Does the writing hold the reader's attention throughout?",
    "Completeness: Does the content fully address the topic or request?",
    "Grammar and Style: Is the writing grammatically correct and stylistically consistent?",
]

# Criteria for technical documentation
TECHNICAL_DOC_CRITERIA = [
    "Accuracy: Are all technical details correct and precise?",
    "Completeness: Are all necessary steps, parameters, and concepts covered?",
    "Clarity: Can a developer follow this without confusion or ambiguity?",
    "Examples: Are code samples correct, runnable, and well-commented?",
    "Prerequisites: Are dependencies and requirements clearly stated upfront?",
    "Structure: Is information organized logically with clear sections?",
]

# Criteria for marketing and sales copy
MARKETING_COPY_CRITERIA = [
    "Hook: Does the opening grab attention immediately?",
    "Benefits: Are benefits clearly stated, not just features?",
    "Call to Action: Is there a clear, compelling CTA?",
    "Tone: Is the tone appropriate for the brand and audience?",
    "Brevity: Is every word earning its place?",
    "Credibility: Does it build trust without overpromising?",
]

# Criteria for academic/professional writing
ACADEMIC_WRITING_CRITERIA = [
    "Thesis: Is the main argument clear and stated early?",
    "Evidence: Are claims backed by appropriate evidence or reasoning?",
    "Structure: Does the piece follow logical structure with clear transitions?",
    "Objectivity: Is the tone appropriately balanced and professional?",
    "Depth: Does the analysis go beyond surface-level observations?",
    "Originality: Does the work contribute insights or perspectives?",
]

# Criteria for email communication
EMAIL_CRITERIA = [
    "Purpose: Is the email's purpose clear within the first two sentences?",
    "Action Items: Are any required actions explicit and easy to identify?",
    "Brevity: Is the email as short as it can be while remaining clear?",
    "Tone: Is the tone appropriate for the recipient and context?",
    "Subject Line: Would the subject line make sense in an inbox full of emails?",
]


# =============================================================================
# Writing Assistant Class
# =============================================================================

class WritingAssistant:
    """
    A writing assistant that iteratively improves content using the
    evaluator-optimizer pattern.
    
    This class provides a high-level interface for writing tasks,
    with pre-configured criteria sets for different content types.
    
    Example:
        assistant = WritingAssistant()
        
        # Write new content
        result = assistant.write(
            "Write an introduction to machine learning for beginners",
            context="For a technical blog aimed at developers"
        )
        print(result.final_content)
        
        # Improve existing content
        result = assistant.improve(
            existing_content="Your draft text here...",
            context="Product description for an e-commerce site"
        )
    """
    
    # Available criteria presets
    PRESETS = {
        "general": GENERAL_WRITING_CRITERIA,
        "technical": TECHNICAL_DOC_CRITERIA,
        "marketing": MARKETING_COPY_CRITERIA,
        "academic": ACADEMIC_WRITING_CRITERIA,
        "email": EMAIL_CRITERIA,
    }
    
    def __init__(
        self,
        criteria: list[str] | str | None = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        verbose: bool = True
    ):
        """
        Initialize the writing assistant.
        
        Args:
            criteria: Evaluation criteria. Can be:
                - A list of custom criteria strings
                - A preset name: "general", "technical", "marketing", "academic", "email"
                - None to use the default general criteria
            max_iterations: Maximum revision cycles (default: 5)
            quality_threshold: Score needed to pass, 0.0 to 1.0 (default: 0.8)
            verbose: Print progress during optimization (default: True)
        """
        self.client = anthropic.Anthropic()
        self.verbose = verbose
        
        # Resolve criteria
        if criteria is None:
            resolved_criteria = GENERAL_WRITING_CRITERIA
        elif isinstance(criteria, str):
            if criteria not in self.PRESETS:
                raise ValueError(
                    f"Unknown preset '{criteria}'. "
                    f"Available: {list(self.PRESETS.keys())}"
                )
            resolved_criteria = self.PRESETS[criteria]
        else:
            resolved_criteria = criteria
        
        # Set up generator
        generator_config = GeneratorConfig(
            temperature=0.7,
            max_tokens=2048
        )
        self.generator = Generator(self.client, generator_config)
        
        # Set up evaluator
        evaluator_config = EvaluatorConfig(
            quality_threshold=quality_threshold,
            temperature=0.3
        )
        self.evaluator = Evaluator(
            self.client,
            resolved_criteria,
            evaluator_config
        )
        
        # Set up optimizer
        loop_config = LoopConfig(
            max_iterations=max_iterations,
            verbose=verbose,
            convergence_threshold=0.02,
            min_iterations=1
        )
        self.optimizer = EvaluatorOptimizer(
            self.generator,
            self.evaluator,
            loop_config
        )
        
        # Store configuration for reference
        self.criteria = resolved_criteria
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def write(self, request: str, context: str = "") -> OptimizationResult:
        """
        Generate and iteratively improve content based on a request.
        
        Args:
            request: What to write (e.g., "Write an introduction to machine learning")
            context: Optional context (e.g., "For a technical blog aimed at beginners")
            
        Returns:
            OptimizationResult with final content and improvement history
        """
        if self.verbose:
            print("\n" + "="*60)
            print("WRITING ASSISTANT - NEW CONTENT")
            print("="*60)
            print(f"Request: {request[:80]}{'...' if len(request) > 80 else ''}")
            if context:
                print(f"Context: {context[:60]}{'...' if len(context) > 60 else ''}")
        
        return self.optimizer.optimize(request, context)
    
    def improve(self, existing_content: str, context: str = "") -> OptimizationResult:
        """
        Improve existing content through evaluation and revision.
        
        Use this when you already have a draft and want to refine it.
        
        Args:
            existing_content: Content to improve
            context: Optional context about the content's purpose
            
        Returns:
            OptimizationResult with improved content and history
        """
        if self.verbose:
            print("\n" + "="*60)
            print("WRITING ASSISTANT - IMPROVE EXISTING")
            print("="*60)
            print(f"Content length: {len(existing_content)} characters")
            if context:
                print(f"Context: {context[:60]}{'...' if len(context) > 60 else ''}")
        
        # Create a prompt that frames improvement of existing content
        prompt = f"""Improve the following content while maintaining its core message 
and intent. Make it clearer, more engaging, and better structured.

Existing content:
---
{existing_content}
---

Provide the improved version."""
        
        return self.optimizer.optimize(prompt, context)
    
    def evaluate_only(self, content: str, context: str = "") -> None:
        """
        Evaluate content without revision (useful for assessment).
        
        Args:
            content: Content to evaluate
            context: Optional context about the content's purpose
        """
        print("\n" + "="*60)
        print("CONTENT EVALUATION")
        print("="*60)
        
        result = self.evaluator.evaluate(content, context)
        print(result)


# =============================================================================
# Example Usage
# =============================================================================

def demo_product_description():
    """Demonstrate writing a product description."""
    assistant = WritingAssistant(
        criteria="marketing",
        max_iterations=4,
        quality_threshold=0.75,
        verbose=True
    )
    
    request = """Write a compelling product description for a new smart water bottle 
that tracks hydration levels, syncs with fitness apps, reminds users to drink water,
and has a built-in UV sterilization system. The target audience is health-conscious
professionals aged 25-45."""
    
    context = "For an e-commerce product page. Should be persuasive but authentic."
    
    result = assistant.write(request, context)
    
    print("\n" + "="*60)
    print("FINAL PRODUCT DESCRIPTION")
    print("="*60)
    print(result.summary())
    print("\n" + result.final_content)
    
    return result


def demo_technical_documentation():
    """Demonstrate writing technical documentation."""
    assistant = WritingAssistant(
        criteria="technical",
        max_iterations=3,
        quality_threshold=0.8,
        verbose=True
    )
    
    request = """Write documentation explaining how to use Python's context managers
(the 'with' statement). Include what they are, why they're useful, and provide
practical examples including creating custom context managers."""
    
    context = "For intermediate Python developers who understand classes but haven't used context managers."
    
    result = assistant.write(request, context)
    
    print("\n" + "="*60)
    print("FINAL DOCUMENTATION")
    print("="*60)
    print(result.summary())
    print("\n" + result.final_content)
    
    return result


def demo_improve_existing():
    """Demonstrate improving existing content."""
    assistant = WritingAssistant(
        criteria="general",
        max_iterations=3,
        quality_threshold=0.8,
        verbose=True
    )
    
    # A deliberately rough draft
    rough_draft = """Python is a programming language. It was made by Guido van Rossum.
Python is used for lots of things like web development and data science and AI.
Python is easy to learn because it has simple syntax that is readable.
Many companies use Python. You should learn Python if you want to be a developer.
Python has lots of libraries that help you do things."""
    
    context = "Opening paragraph for a 'Why Learn Python' blog post."
    
    result = assistant.improve(rough_draft, context)
    
    print("\n" + "="*60)
    print("IMPROVED CONTENT")
    print("="*60)
    print(result.summary())
    print("\n--- Original ---")
    print(rough_draft)
    print("\n--- Improved ---")
    print(result.final_content)
    
    return result


def demo_email():
    """Demonstrate writing a professional email."""
    assistant = WritingAssistant(
        criteria="email",
        max_iterations=3,
        quality_threshold=0.8,
        verbose=True
    )
    
    request = """Write a professional email to a client explaining that their 
project will be delivered one week later than originally planned due to 
unexpected technical challenges. Be apologetic but professional, and offer
a small discount as compensation."""
    
    context = "B2B software development context. Client is a returning customer."
    
    result = assistant.write(request, context)
    
    print("\n" + "="*60)
    print("FINAL EMAIL")
    print("="*60)
    print(result.summary())
    print("\n" + result.final_content)
    
    return result


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    demos = {
        "product": demo_product_description,
        "technical": demo_technical_documentation,
        "improve": demo_improve_existing,
        "email": demo_email,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        print("Writing Assistant Demo")
        print("="*60)
        print("\nAvailable demos:")
        for name, func in demos.items():
            print(f"  python writing_assistant.py {name}")
        
        print("\nRunning default demo (product description)...")
        demo_product_description()
