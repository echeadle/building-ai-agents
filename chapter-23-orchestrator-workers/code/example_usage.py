"""
Example usage of the Research Orchestrator.

Chapter 23: Orchestrator-Workers - Implementation

This file demonstrates various ways to use the ResearchOrchestrator
for different research scenarios.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Import our orchestrator
from research_orchestrator import ResearchOrchestrator


def example_basic_usage():
    """Basic usage - just pass a query and get results."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=True)
    
    result = orchestrator.research(
        "How is artificial intelligence transforming the healthcare industry?"
    )
    
    print("\n--- Final Answer ---")
    print(result.synthesis)
    
    return result


def example_with_context():
    """Provide additional context to guide the research."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: With Additional Context")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=True)
    
    result = orchestrator.research(
        query="What are the best practices for implementing microservices?",
        context="""The user is working on a medium-sized e-commerce platform 
        with about 50 developers. They're currently using a monolithic 
        architecture and considering migration to microservices."""
    )
    
    print("\n--- Final Answer ---")
    print(result.synthesis)
    
    return result


def example_preview_plan():
    """Preview the task plan before executing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Preview Plan Before Execution")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=True)
    
    # First, just get the plan
    plan = orchestrator.get_plan_only(
        "What are the long-term effects of social media on mental health?"
    )
    
    print("\n--- Task Plan ---")
    print(f"Analysis: {plan.query_analysis}")
    print(f"\nSubtasks ({len(plan.subtasks)}):")
    
    for i, subtask in enumerate(plan.subtasks, 1):
        print(f"\n  {i}. [{subtask.type.upper()}] {subtask.title}")
        print(f"     {subtask.description}")
        print(f"     Focus: {', '.join(subtask.focus_areas)}")
    
    print(f"\nSynthesis guidance: {plan.synthesis_guidance}")
    
    # Ask user if they want to proceed (simulated)
    proceed = True  # In real usage, you'd prompt the user
    
    if proceed:
        print("\n--- Executing Plan ---")
        result = orchestrator.execute_plan(plan)
        print("\n--- Final Answer ---")
        print(result.synthesis)
        return result
    
    return plan


def example_access_worker_results():
    """Access individual worker results for detailed analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Accessing Worker Results")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=False)  # Quiet mode
    
    result = orchestrator.research(
        "What are the pros and cons of remote work for companies?"
    )
    
    print("\n--- Individual Worker Results ---")
    
    for wr in result.worker_results:
        if wr.success:
            print(f"\n### {wr.subtask_title}")
            print(f"Execution time: {wr.execution_time:.1f}s")
            print("-" * 40)
            # Show first 300 characters
            preview = wr.content[:300]
            if len(wr.content) > 300:
                preview += "..."
            print(preview)
        else:
            print(f"\n### {wr.subtask_title} - FAILED")
            print(f"Error: {wr.error}")
    
    print("\n--- Final Synthesis ---")
    print(result.synthesis)
    
    return result


def example_quiet_mode():
    """Run without verbose output for production use."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Quiet Mode (Production)")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=False)
    
    result = orchestrator.research(
        "What are emerging trends in renewable energy?"
    )
    
    # Just show the final result
    print("\n--- Result ---")
    print(f"Success: {result.success}")
    print(f"Subtasks: {result.subtasks_completed}/{len(result.worker_results)}")
    print(f"Time: {result.total_time:.1f}s")
    print(f"\nAnswer:\n{result.synthesis[:500]}...")
    
    return result


def example_custom_configuration():
    """Customize orchestrator behavior."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom Configuration")
    print("=" * 60)
    
    # Use fewer subtasks for simpler queries
    orchestrator = ResearchOrchestrator(
        model="claude-sonnet-4-20250514",
        max_subtasks=3,  # Limit to 3 subtasks
        verbose=True
    )
    
    result = orchestrator.research(
        "What is quantum computing?"
    )
    
    print(f"\n--- Created {len(result.plan.subtasks)} subtasks ---")
    print("\n--- Final Answer ---")
    print(result.synthesis)
    
    return result


def example_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Error Handling")
    print("=" * 60)
    
    orchestrator = ResearchOrchestrator(verbose=True)
    
    result = orchestrator.research(
        "Explain the latest developments in fusion energy research."
    )
    
    # Check for failures
    if result.subtasks_failed > 0:
        print(f"\n⚠️  Warning: {result.subtasks_failed} subtask(s) failed")
        for wr in result.worker_results:
            if not wr.success:
                print(f"   - {wr.subtask_title}: {wr.error}")
    
    # The synthesis still works with partial results
    if result.success:
        print("\n✓ Synthesis succeeded despite failures")
        print(result.synthesis[:500] + "...")
    else:
        print("\n✗ Complete failure - no results available")
    
    return result


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESEARCH ORCHESTRATOR EXAMPLES")
    print("=" * 60)
    print("""
Select an example to run:
1. Basic usage
2. With additional context
3. Preview plan before execution
4. Access worker results
5. Quiet mode (production)
6. Custom configuration
7. Error handling
0. Run all examples
""")
    
    choice = input("Enter choice (1-7, or 0 for all): ").strip()
    
    examples = {
        "1": example_basic_usage,
        "2": example_with_context,
        "3": example_preview_plan,
        "4": example_access_worker_results,
        "5": example_quiet_mode,
        "6": example_custom_configuration,
        "7": example_error_handling,
    }
    
    if choice == "0":
        # Run all examples
        for key in sorted(examples.keys()):
            try:
                examples[key]()
            except Exception as e:
                print(f"Example {key} failed: {e}")
            print("\n" + "-" * 60 + "\n")
    elif choice in examples:
        examples[choice]()
    else:
        # Default to basic usage
        print("Invalid choice, running basic example...")
        example_basic_usage()
