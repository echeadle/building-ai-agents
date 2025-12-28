"""
Example: Using the Research Assistant Agent

Chapter 42: Project - Research Assistant Agent

This script demonstrates various research tasks using the ResearchAgent.
"""

import os
from dotenv import load_dotenv
from research_agent import ResearchAgent

load_dotenv()

# Verify API keys
api_key = os.getenv("ANTHROPIC_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

if not serpapi_key:
    print("⚠️ Warning: SERPAPI_API_KEY not found.")
    print("Get a free key at https://serpapi.com/")
    print("The agent cannot search without this key.\n")
    exit(1)


def research_and_save(
    agent: ResearchAgent,
    question: str,
    filename: str,
    max_searches: int = 8,
    max_reads: int = 12
):
    """
    Conduct research and save the report
    
    Args:
        agent: The research agent instance
        question: Research question
        filename: Output filename for the report
        max_searches: Maximum number of searches to perform
        max_reads: Maximum number of pages to read
    """
    print(f"\n{'='*80}")
    print(f"Research Question: {question}")
    print(f"{'='*80}\n")
    
    # Conduct research
    report = agent.research(
        question=question,
        max_searches=max_searches,
        max_reads=max_reads,
        verbose=True
    )
    
    # Save report
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"---\n\n")
        f.write(report)
    
    print(f"\n✅ Report saved to {filename}")
    print(f"Report length: {len(report)} characters\n")
    
    return report


def main():
    """Run example research tasks"""
    
    # Create the research agent
    agent = ResearchAgent(api_key=api_key)
    
    print("\n" + "="*80)
    print("RESEARCH ASSISTANT AGENT - EXAMPLES")
    print("="*80)
    
    # Example 1: Technology research
    print("\n📚 Example 1: Technology Research")
    research_and_save(
        agent=agent,
        question="What are the main applications of large language models in 2024?",
        filename="llm_applications_report.md",
        max_searches=6,
        max_reads=10
    )
    
    # Example 2: Scientific research
    print("\n📚 Example 2: Scientific Research")
    research_and_save(
        agent=agent,
        question="What progress has been made in nuclear fusion energy in the past year?",
        filename="fusion_energy_report.md",
        max_searches=6,
        max_reads=10
    )
    
    # Example 3: Business research
    print("\n📚 Example 3: Business/Social Research")
    research_and_save(
        agent=agent,
        question="What are the emerging trends in remote work and digital nomadism?",
        filename="remote_work_report.md",
        max_searches=6,
        max_reads=10
    )
    
    # Example 4: Comparative research
    print("\n📚 Example 4: Comparative Analysis")
    research_and_save(
        agent=agent,
        question="How do renewable energy costs compare to fossil fuels in 2024?",
        filename="energy_costs_report.md",
        max_searches=8,
        max_reads=12
    )
    
    print("\n" + "="*80)
    print("All research tasks complete!")
    print("="*80)
    print("\nGenerated reports:")
    print("  - llm_applications_report.md")
    print("  - fusion_energy_report.md")
    print("  - remote_work_report.md")
    print("  - energy_costs_report.md")
    print("\n")


def quick_research(question: str):
    """
    Run a single quick research task
    
    Args:
        question: The research question
    """
    agent = ResearchAgent(api_key=api_key)
    
    print("\n" + "="*80)
    print(f"Quick Research: {question}")
    print("="*80 + "\n")
    
    report = agent.research(
        question=question,
        max_searches=5,
        max_reads=8,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("RESEARCH REPORT")
    print("="*80)
    print(report)
    print("="*80 + "\n")
    
    return report


if __name__ == "__main__":
    import sys
    
    # Check if a custom question was provided as command line argument
    if len(sys.argv) > 1:
        # Run quick research with the provided question
        custom_question = " ".join(sys.argv[1:])
        quick_research(custom_question)
    else:
        # Run all example research tasks
        main()
