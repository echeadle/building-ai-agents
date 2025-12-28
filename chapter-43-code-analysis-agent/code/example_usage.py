"""
Example usage of the Code Analysis Agent

This demonstrates how to use the code analysis agent to analyze
different types of codebases with various goals.

Chapter 43: Project - Code Analysis Agent
"""

from code_analysis_agent import run_agent, get_summary, get_findings


def example_1_analyze_current_project():
    """Example 1: Analyze the current project for overall quality."""
    print("="*70)
    print("EXAMPLE 1: Analyzing current project")
    print("="*70)
    
    report = run_agent(
        codebase_path=".",
        analysis_goal="Analyze code quality and structure",
        max_iterations=12,
        verbose=True
    )
    
    print("\n" + report)


def example_2_security_focused():
    """Example 2: Focus on security concerns."""
    print("="*70)
    print("EXAMPLE 2: Security-focused analysis")
    print("="*70)
    
    report = run_agent(
        codebase_path="/path/to/web/app",
        analysis_goal="Focus on security: input validation, authentication, sensitive data handling",
        max_iterations=10,
        verbose=True
    )
    
    print("\n" + report)
    
    # Show just security findings
    security_findings = get_findings("security")
    print("\n" + "="*70)
    print("SECURITY FINDINGS DETAIL")
    print("="*70)
    for finding in security_findings['findings']:
        print(f"\n⚠️  {finding['finding']}")
        if finding['file_path']:
            location = f"{finding['file_path']}"
            if finding['line_number']:
                location += f":{finding['line_number']}"
            print(f"    Location: {location}")


def example_3_architecture_review():
    """Example 3: Review architectural decisions."""
    print("="*70)
    print("EXAMPLE 3: Architecture review")
    print("="*70)
    
    report = run_agent(
        codebase_path="/path/to/project",
        analysis_goal="Review architecture: module organization, separation of concerns, design patterns",
        max_iterations=15,
        verbose=True
    )
    
    print("\n" + report)
    
    # Show structure and pattern findings
    structure = get_findings("structure")
    patterns = get_findings("patterns")
    
    print("\n" + "="*70)
    print("ARCHITECTURAL INSIGHTS")
    print("="*70)
    
    if structure['findings']:
        print("\nStructure:")
        for finding in structure['findings']:
            print(f"  • {finding['finding']}")
    
    if patterns['findings']:
        print("\nPatterns:")
        for finding in patterns['findings']:
            print(f"  • {finding['finding']}")


def example_4_quick_assessment():
    """Example 4: Quick assessment with limited iterations."""
    print("="*70)
    print("EXAMPLE 4: Quick assessment")
    print("="*70)
    
    report = run_agent(
        codebase_path="/path/to/small/project",
        analysis_goal="Quick overview: what is this project and is it well-structured?",
        max_iterations=6,  # Fewer iterations for quick assessment
        verbose=False  # Less verbose output
    )
    
    print(report)


def example_5_dependency_analysis():
    """Example 5: Focus on dependencies and coupling."""
    print("="*70)
    print("EXAMPLE 5: Dependency analysis")
    print("="*70)
    
    report = run_agent(
        codebase_path="/path/to/project",
        analysis_goal="Analyze dependencies: external packages, internal coupling, circular dependencies",
        max_iterations=10,
        verbose=True
    )
    
    print("\n" + report)
    
    # Show dependency findings
    deps = get_findings("dependencies")
    print("\n" + "="*70)
    print("DEPENDENCY DETAILS")
    print("="*70)
    for finding in deps['findings']:
        print(f"\n📦 {finding['finding']}")


def example_6_compare_summary():
    """Example 6: Run analysis and get structured summary."""
    print("="*70)
    print("EXAMPLE 6: Analysis with structured summary")
    print("="*70)
    
    # Run analysis
    report = run_agent(
        codebase_path=".",
        analysis_goal="Comprehensive analysis",
        max_iterations=12,
        verbose=False
    )
    
    # Get structured data about findings
    summary = get_summary()
    
    print("\nFINDINGS SUMMARY")
    print("-" * 70)
    print(f"Total findings: {summary['total_findings']}")
    print(f"\nBy category:")
    for category, count in sorted(summary['by_category'].items()):
        print(f"  {category:20s}: {count:2d}")
    
    print(f"\nBy severity:")
    for severity in ['error', 'warning', 'info']:
        count = summary['by_severity'].get(severity, 0)
        if count > 0:
            icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[severity]
            print(f"  {icon} {severity:10s}: {count:2d}")
    
    print("\n" + "="*70)
    print("FULL REPORT")
    print("="*70)
    print(report)


if __name__ == "__main__":
    # Run the first example
    # Uncomment others to try different analysis approaches
    
    example_1_analyze_current_project()
    
    # example_2_security_focused()
    # example_3_architecture_review()
    # example_4_quick_assessment()
    # example_5_dependency_analysis()
    # example_6_compare_summary()
