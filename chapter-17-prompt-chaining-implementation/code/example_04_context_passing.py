"""
Passing rich context between chain steps.

This example demonstrates how to carry data through a chain so that
later steps can access outputs from any earlier step, not just the
immediately preceding one.

Features:
- ChainContext data structure
- Accessing original input from later steps
- Metadata tracking
- Multi-step research workflow

Chapter 17: Prompt Chaining - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class ChainContext:
    """
    Carries data through a chain, accumulating outputs from each step.
    
    This allows later steps to access outputs from any earlier step,
    not just the immediately preceding one. It also tracks metadata
    about the chain execution.
    
    Attributes:
        original_input: The initial input to the chain
        step_outputs: Dictionary mapping step names to their outputs
        metadata: Additional tracking information
    """
    original_input: Any
    step_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_output(self, step_name: str, output: Any) -> None:
        """
        Record output from a step.
        
        Args:
            step_name: Name of the step that produced this output
            output: The output to store
        """
        self.step_outputs[step_name] = output
        self.metadata[f"{step_name}_timestamp"] = datetime.now().isoformat()
    
    def get_output(self, step_name: str) -> Any:
        """
        Get output from a specific step.
        
        Args:
            step_name: Name of the step
        
        Returns:
            The step's output, or None if not found
        """
        return self.step_outputs.get(step_name)
    
    def has_output(self, step_name: str) -> bool:
        """Check if a step has produced output."""
        return step_name in self.step_outputs
    
    @property
    def latest_output(self) -> Any:
        """Get the most recent step output."""
        if not self.step_outputs:
            return self.original_input
        return list(self.step_outputs.values())[-1]
    
    @property
    def all_outputs(self) -> list[tuple[str, Any]]:
        """Get all outputs as (step_name, output) pairs."""
        return list(self.step_outputs.items())


def run_research_chain(question: str) -> ChainContext:
    """
    Execute a research chain where later steps need access to earlier outputs.
    
    This demonstrates a scenario where:
    - Step 1: Generates research points from the question
    - Step 2: Expands on the research points
    - Step 3: Synthesizes a final answer using BOTH the original question
              AND the expanded research (not just the previous step's output)
    
    Args:
        question: The research question to investigate
    
    Returns:
        ChainContext with all outputs and metadata
    """
    # Initialize context with the research question
    context = ChainContext(
        original_input=question,
        metadata={
            "started_at": datetime.now().isoformat(),
            "model": MODEL_NAME
        }
    )
    
    # Step 1: Generate research points
    print("Step 1: Generating research points...")
    print(f"  Question: {question[:80]}...")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""List 5 key research points to investigate for this question:
            
{context.original_input}

Format as a numbered list with brief descriptions of what each point covers."""
        }]
    )
    research_points = response.content[0].text
    context.add_output("research_points", research_points)
    
    point_count = len([l for l in research_points.split('\n') if l.strip() and l.strip()[0].isdigit()])
    print(f"  ✓ Generated {point_count} research points")
    
    # Step 2: Expand on each point
    print("\nStep 2: Expanding research points...")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Expand on each of these research points with 2-3 sentences of factual detail:

{research_points}

Provide concrete information, statistics where relevant, and specific examples."""
        }]
    )
    expanded_research = response.content[0].text
    context.add_output("expanded_research", expanded_research)
    print(f"  ✓ Expanded to {len(expanded_research)} characters")
    
    # Step 3: Synthesize final answer
    # This step needs BOTH the original question AND the expanded research
    print("\nStep 3: Synthesizing final answer...")
    print("  (Using original question + expanded research)")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Based on this research, write a comprehensive answer to the original question.

ORIGINAL QUESTION:
{context.original_input}

RESEARCH FINDINGS:
{context.get_output("expanded_research")}

Write a well-structured response that:
1. Directly addresses the original question
2. Incorporates key findings from the research
3. Provides a balanced perspective
4. Is approximately 3-4 paragraphs"""
        }]
    )
    final_answer = response.content[0].text
    context.add_output("final_answer", final_answer)
    print(f"  ✓ Synthesized answer ({len(final_answer)} characters)")
    
    # Step 4: Generate a one-sentence summary
    # This needs the final answer but also references the original question
    print("\nStep 4: Creating executive summary...")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Write a single-sentence executive summary (under 100 words) that answers this question:

Question: {context.original_input}

Based on this analysis:
{context.get_output("final_answer")[:500]}...

The summary should be suitable for a busy executive who needs the key takeaway."""
        }]
    )
    summary = response.content[0].text
    context.add_output("executive_summary", summary)
    print(f"  ✓ Created summary ({len(summary.split())} words)")
    
    # Update metadata
    context.metadata["completed_at"] = datetime.now().isoformat()
    context.metadata["total_steps"] = len(context.step_outputs)
    
    return context


def run_content_localization_chain(
    content: str, 
    source_language: str,
    target_languages: list[str]
) -> ChainContext:
    """
    A content localization chain that adapts content for multiple markets.
    
    This demonstrates accessing multiple earlier outputs:
    - Step 1: Analyze the source content
    - Step 2: Adapt for each target language (needs both original AND analysis)
    - Step 3: Create a summary comparison (needs original AND all translations)
    
    Args:
        content: Original content to localize
        source_language: Source language
        target_languages: List of target languages
    
    Returns:
        ChainContext with all localized versions
    """
    context = ChainContext(
        original_input=content,
        metadata={
            "source_language": source_language,
            "target_languages": target_languages
        }
    )
    
    # Step 1: Analyze the source content
    print("Step 1: Analyzing source content...")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Analyze this {source_language} marketing content:

{content}

Identify:
1. Main message and tone
2. Cultural references that may need adaptation
3. Key terms that must be preserved
4. Call-to-action elements

Be concise."""
        }]
    )
    analysis = response.content[0].text
    context.add_output("content_analysis", analysis)
    print(f"  ✓ Analysis complete")
    
    # Step 2: Localize for each target language
    print("\nStep 2: Localizing content...")
    
    for lang in target_languages:
        print(f"  Translating to {lang}...")
        
        # Each translation uses BOTH original content AND analysis
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""Localize this content to {lang}.

ORIGINAL CONTENT ({source_language}):
{context.original_input}

CONTENT ANALYSIS:
{context.get_output("content_analysis")}

Guidelines:
- Adapt cultural references appropriately
- Maintain the core message and tone
- Preserve key terms identified in analysis
- Make the call-to-action natural in {lang}"""
            }]
        )
        localized = response.content[0].text
        context.add_output(f"localized_{lang.lower()}", localized)
        print(f"    ✓ {lang} version complete")
    
    # Step 3: Create comparison summary
    # This needs the original AND all translations
    print("\nStep 3: Creating comparison summary...")
    
    all_versions = [f"ORIGINAL ({source_language}):\n{context.original_input}"]
    for lang in target_languages:
        version = context.get_output(f"localized_{lang.lower()}")
        all_versions.append(f"{lang.upper()} VERSION:\n{version}")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Create a brief comparison summary of these localized versions.

{chr(10).join(all_versions)}

Note any significant adaptations made for each language and confirm the core message is preserved."""
        }]
    )
    comparison = response.content[0].text
    context.add_output("comparison_summary", comparison)
    print(f"  ✓ Comparison complete")
    
    return context


if __name__ == "__main__":
    print("="*60)
    print("EXAMPLE 1: Research Chain with Context")
    print("="*60)
    
    research_context = run_research_chain(
        "What are the environmental impacts of cryptocurrency mining, "
        "and what solutions are being developed?"
    )
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"\nOriginal Question:\n{research_context.original_input}")
    print(f"\nExecutive Summary:\n{research_context.get_output('executive_summary')}")
    print(f"\nMetadata: {research_context.metadata}")
    
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Content Localization Chain")
    print("="*60)
    
    localization_context = run_content_localization_chain(
        content="""Unlock your productivity potential with TaskMaster Pro! 
Our AI-powered task manager learns your work style and helps you crush your goals. 
Start your free trial today and join millions of high achievers.""",
        source_language="English",
        target_languages=["Spanish", "French", "German"]
    )
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"\nOriginal:\n{localization_context.original_input}")
    print(f"\nSpanish Version:\n{localization_context.get_output('localized_spanish')}")
    print(f"\nComparison:\n{localization_context.get_output('comparison_summary')}")
