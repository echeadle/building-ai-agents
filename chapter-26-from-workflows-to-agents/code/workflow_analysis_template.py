"""
Workflow Analysis Template

Use this template to analyze whether a workflow should be upgraded to an agent.

Chapter 26: From Workflows to Agents - Exercise Solution
"""

from dataclasses import dataclass, field
from enum import Enum


class RecommendationLevel(Enum):
    """Recommendation for whether to use an agent."""
    KEEP_WORKFLOW = "keep_workflow"
    CONSIDER_AGENT = "consider_agent"
    UPGRADE_TO_AGENT = "upgrade_to_agent"


@dataclass
class WorkflowAnalysis:
    """
    Template for analyzing whether to upgrade a workflow to an agent.
    
    Fill in each section to make an informed decision.
    """
    
    # Basic Information
    name: str = ""
    description: str = ""
    current_pattern: str = ""  # chaining, routing, parallel, orchestrator, evaluator
    
    # Scenarios where workflow handles well
    handles_well: list = field(default_factory=list)
    
    # Scenarios where workflow struggles
    struggles_with: list = field(default_factory=list)
    
    # Why an agent would help
    agent_advantages: list = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = ""  # low, medium, high
    reversibility: str = ""  # fully, partially, not_reversible
    
    # Final recommendation
    recommendation: RecommendationLevel = None
    reasoning: str = ""
    
    def analyze(self) -> str:
        """Generate a formatted analysis report."""
        report = []
        report.append("=" * 60)
        report.append(f"WORKFLOW ANALYSIS: {self.name}")
        report.append("=" * 60)
        
        report.append(f"\n📋 DESCRIPTION:")
        report.append(f"   {self.description}")
        report.append(f"\n   Current pattern: {self.current_pattern}")
        
        report.append(f"\n\n✅ HANDLES WELL:")
        for item in self.handles_well:
            report.append(f"   • {item}")
        
        report.append(f"\n\n❌ STRUGGLES WITH:")
        for item in self.struggles_with:
            report.append(f"   • {item}")
        
        report.append(f"\n\n🤖 AGENT ADVANTAGES:")
        for item in self.agent_advantages:
            report.append(f"   • {item}")
        
        report.append(f"\n\n⚠️ RISK ASSESSMENT:")
        report.append(f"   Risk level: {self.risk_level}")
        report.append(f"   Reversibility: {self.reversibility}")
        
        report.append(f"\n\n📊 RECOMMENDATION: {self.recommendation.value if self.recommendation else 'Not determined'}")
        report.append(f"   {self.reasoning}")
        
        return "\n".join(report)


# =============================================================================
# EXAMPLE ANALYSES
# =============================================================================

def example_customer_service_router():
    """Example: Customer Service Router (from Chapter 19)."""
    
    analysis = WorkflowAnalysis(
        name="Customer Service Router",
        description="Routes customer queries to specialized handlers based on category (billing, technical, general)",
        current_pattern="routing",
        
        handles_well=[
            "Simple billing questions → routes to billing handler correctly",
            "Clear technical support requests → routes to tech handler",
            "General inquiries → handled by general handler",
            "High volume of straightforward queries",
            "Consistent, predictable responses for common issues"
        ],
        
        struggles_with=[
            "User starts with billing question, but root cause is technical",
            "Complex issues spanning multiple categories",
            "Multi-step troubleshooting that requires back-and-forth",
            "Situations where initial classification is wrong",
            "Users who don't know how to categorize their own problem"
        ],
        
        agent_advantages=[
            "Can switch strategies when initial approach fails",
            "Can combine knowledge from multiple domains",
            "Can engage in multi-step troubleshooting",
            "Can ask clarifying questions to understand the real issue",
            "Can escalate or involve multiple 'specialists' as needed"
        ],
        
        risk_level="low",
        reversibility="fully",
        
        recommendation=RecommendationLevel.CONSIDER_AGENT,
        reasoning="Keep the workflow for high-volume, simple queries (cost-effective). "
                  "Add an agent path for complex cases that the router struggles with. "
                  "The workflow can act as a 'tier 1' filter, with agents handling 'tier 2' complexity."
    )
    
    return analysis


def example_content_translation_chain():
    """Example: Content Translation Chain (from Chapter 17)."""
    
    analysis = WorkflowAnalysis(
        name="Content Translation Chain",
        description="Generates content, then translates it through a fixed chain of steps",
        current_pattern="chaining",
        
        handles_well=[
            "Straightforward content generation tasks",
            "When the content type is well-defined",
            "When translation requirements are clear",
            "Consistent output format needed",
            "High-volume, repetitive tasks"
        ],
        
        struggles_with=[
            "When generated content doesn't meet quality standards",
            "When translation fails and needs retry with different approach",
            "When user requirements are ambiguous",
            "When intermediate steps need human review",
            "When the content needs iterative refinement"
        ],
        
        agent_advantages=[
            "Can regenerate content if quality check fails",
            "Can try alternative translation strategies",
            "Can ask for clarification on ambiguous requirements",
            "Can iterate until quality standards are met",
            "Can adapt the process based on content complexity"
        ],
        
        risk_level="low",
        reversibility="fully",
        
        recommendation=RecommendationLevel.KEEP_WORKFLOW,
        reasoning="The workflow is sufficient for most cases. "
                  "Consider adding a simple retry mechanism or quality gate instead of full agent. "
                  "If quality issues are frequent, consider the Evaluator-Optimizer pattern first. "
                  "Full agent autonomy is overkill for this linear task."
    )
    
    return analysis


def example_code_review_voting():
    """Example: Code Review Voting System (from Chapter 21)."""
    
    analysis = WorkflowAnalysis(
        name="Code Review Voting",
        description="Multiple LLM perspectives analyze code for vulnerabilities, results aggregated by voting",
        current_pattern="parallelization (voting)",
        
        handles_well=[
            "Catching common vulnerability patterns",
            "Providing multiple perspectives quickly",
            "High confidence on issues where reviewers agree",
            "Consistent coverage of standard security concerns",
            "Fast turnaround for code review"
        ],
        
        struggles_with=[
            "Deep investigation of suspicious patterns",
            "Following a chain of dependencies to find root cause",
            "Context-dependent vulnerabilities",
            "Issues that require understanding the broader codebase",
            "When reviewers disagree and need deeper analysis"
        ],
        
        agent_advantages=[
            "Can investigate suspicious findings in depth",
            "Can follow dependency chains across files",
            "Can run tests or execute code to verify concerns",
            "Can ask for more context when needed",
            "Can adapt investigation strategy based on findings"
        ],
        
        risk_level="medium",
        reversibility="fully",
        
        recommendation=RecommendationLevel.CONSIDER_AGENT,
        reasoning="Use the voting workflow as a first pass (fast, comprehensive coverage). "
                  "When voting results are split or findings are unclear, trigger an agent "
                  "for deeper investigation. This hybrid approach gets the best of both: "
                  "speed for routine reviews, depth for complex issues."
    )
    
    return analysis


def example_research_orchestrator():
    """Example: Research Orchestrator (from Chapter 23)."""
    
    analysis = WorkflowAnalysis(
        name="Research Orchestrator",
        description="Orchestrator breaks down research tasks and delegates to specialized workers",
        current_pattern="orchestrator-workers",
        
        handles_well=[
            "Research tasks that can be decomposed upfront",
            "When subtasks are independent",
            "When the scope is well-defined",
            "Parallel research across multiple domains",
            "Synthesizing findings from multiple sources"
        ],
        
        struggles_with=[
            "When initial task decomposition is wrong",
            "When findings from one worker should redirect others",
            "When research reveals the original question was wrong",
            "Iterative research where each finding leads to new questions",
            "When the scope is unclear or needs refinement"
        ],
        
        agent_advantages=[
            "Can replan when initial approach doesn't work",
            "Can redirect research based on emerging findings",
            "Can refine the original question as understanding grows",
            "Can follow unexpected leads dynamically",
            "Can ask clarifying questions mid-research"
        ],
        
        risk_level="low",
        reversibility="fully",
        
        recommendation=RecommendationLevel.UPGRADE_TO_AGENT,
        reasoning="Research is inherently exploratory. The orchestrator pattern works for "
                  "well-defined tasks, but real research often requires dynamic adaptation. "
                  "An agent-based researcher can follow leads, adjust strategy, and refine "
                  "understanding iteratively. The orchestrator-workers pattern can still be "
                  "used internally by the agent when parallel investigation is beneficial."
    )
    
    return analysis


# =============================================================================
# BLANK TEMPLATE FOR YOUR ANALYSIS
# =============================================================================

def your_analysis():
    """
    Fill in this template with your own workflow analysis.
    
    Instructions:
    1. Think of a workflow you've built or could build
    2. Fill in each field honestly
    3. Run the analysis to see the formatted report
    4. Make your recommendation with reasoning
    """
    
    analysis = WorkflowAnalysis(
        name="Your Workflow Name",
        description="What does this workflow do?",
        current_pattern="chaining / routing / parallel / orchestrator / evaluator",
        
        handles_well=[
            "Scenario 1 where workflow works well",
            "Scenario 2 where workflow works well",
            "Scenario 3 where workflow works well",
        ],
        
        struggles_with=[
            "Scenario 1 where workflow struggles",
            "Scenario 2 where workflow struggles",
            "Scenario 3 where workflow struggles",
        ],
        
        agent_advantages=[
            "Why an agent would handle scenario 1 better",
            "Why an agent would handle scenario 2 better",
            "Why an agent would handle scenario 3 better",
        ],
        
        risk_level="low / medium / high",
        reversibility="fully / partially / not_reversible",
        
        recommendation=RecommendationLevel.KEEP_WORKFLOW,  # or CONSIDER_AGENT or UPGRADE_TO_AGENT
        reasoning="Your reasoning for the recommendation..."
    )
    
    return analysis


# =============================================================================
# DECISION HELPER
# =============================================================================

def should_use_agent(analysis: WorkflowAnalysis) -> str:
    """
    Generate a decision summary based on the analysis.
    """
    scores = {
        "agent_needed": 0,
        "workflow_sufficient": 0
    }
    
    # More struggles than successes suggests agent might help
    if len(analysis.struggles_with) > len(analysis.handles_well):
        scores["agent_needed"] += 2
    else:
        scores["workflow_sufficient"] += 1
    
    # Many agent advantages suggests upgrade
    if len(analysis.agent_advantages) >= 3:
        scores["agent_needed"] += 1
    
    # Risk assessment
    if analysis.risk_level == "high":
        scores["workflow_sufficient"] += 2  # High risk = prefer predictable workflow
    elif analysis.risk_level == "low":
        scores["agent_needed"] += 1  # Low risk = can experiment with agent
    
    # Reversibility
    if analysis.reversibility == "not_reversible":
        scores["workflow_sufficient"] += 2  # Irreversible = prefer controlled workflow
    
    # Generate summary
    if scores["agent_needed"] > scores["workflow_sufficient"]:
        decision = "Consider upgrading to an agent"
    elif scores["agent_needed"] == scores["workflow_sufficient"]:
        decision = "Hybrid approach recommended (workflow + agent for complex cases)"
    else:
        decision = "Keep the workflow, add targeted improvements"
    
    return decision


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WORKFLOW → AGENT ANALYSIS TOOL")
    print("=" * 70)
    
    # Run example analyses
    examples = [
        example_customer_service_router(),
        example_content_translation_chain(),
        example_code_review_voting(),
        example_research_orchestrator(),
    ]
    
    for analysis in examples:
        print("\n\n")
        print(analysis.analyze())
        decision = should_use_agent(analysis)
        print(f"\n🎯 Decision Helper Suggests: {decision}")
    
    print("\n\n" + "=" * 70)
    print("YOUR TURN!")
    print("=" * 70)
    print("""
To analyze your own workflow:

1. Edit the `your_analysis()` function in this file
2. Fill in all the fields with your workflow's details
3. Run this script again

Or create a new analysis in Python:

    from workflow_analysis_template import WorkflowAnalysis, RecommendationLevel
    
    my_analysis = WorkflowAnalysis(
        name="My Workflow",
        # ... fill in fields ...
    )
    
    print(my_analysis.analyze())
""")
