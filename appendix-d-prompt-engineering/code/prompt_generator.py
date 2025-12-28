"""
Prompt template generator for common agent patterns.

Appendix D: Prompt Engineering for Agents

This module provides ready-to-use prompt templates that you can customize
for your specific use case.
"""

from typing import Any
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Types of agents with different prompt patterns."""
    BASIC = "basic"
    TOOL_USER = "tool_user"
    MULTI_STEP = "multi_step"
    DOMAIN_EXPERT = "domain_expert"
    CUSTOMER_SERVICE = "customer_service"
    RESEARCH = "research"
    CODE_ASSISTANT = "code_assistant"


@dataclass
class PromptTemplate:
    """A prompt template with placeholders."""
    template: str
    required_fields: list[str]
    optional_fields: list[str]
    description: str


class PromptGenerator:
    """
    Generate customized prompts from templates.
    
    Usage:
        generator = PromptGenerator()
        prompt = generator.generate(
            AgentType.RESEARCH,
            domain="AI research",
            capabilities=["search papers", "summarize findings"],
            constraints=["cite sources", "be objective"]
        )
    """
    
    TEMPLATES = {
        AgentType.BASIC: PromptTemplate(
            template="""You are a {role} that helps users {purpose}.

Your capabilities:
{capabilities}

Important rules:
{constraints}

When responding:
{guidelines}

Output format:
{output_format}""",
            required_fields=["role", "purpose"],
            optional_fields=["capabilities", "constraints", "guidelines", "output_format"],
            description="Basic agent with clear role and guidelines"
        ),
        
        AgentType.TOOL_USER: PromptTemplate(
            template="""You are a {role}.

Available tools:
{tool_descriptions}

Tool selection guidelines:
- Always use the most specific tool available
- Verify tool results before responding
- If a tool fails, try an alternative approach
- Never call the same tool repeatedly without changing parameters

After gathering information with tools:
1. Synthesize results into a clear answer
2. Cite which tools provided which information
3. Note any limitations or uncertainties

{additional_guidelines}""",
            required_fields=["role", "tool_descriptions"],
            optional_fields=["additional_guidelines"],
            description="Agent that uses tools to accomplish tasks"
        ),
        
        AgentType.MULTI_STEP: PromptTemplate(
            template="""You are a {role}.

For complex tasks, follow this process:

PHASE 1 - PLANNING:
- Understand the user's goal
- Break it into clear steps
- Identify required tools
- Present plan to user

PHASE 2 - EXECUTION:
- Execute each step in order
- Show progress as you go
- Verify each result before proceeding
- Stop if you encounter blockers

PHASE 3 - COMPLETION:
- Summarize what was accomplished
- Note any partial completions or issues
- Suggest next steps if applicable

Maximum iterations: {max_iterations}
If you reach this limit, summarize progress and ask how to proceed.

{additional_guidelines}""",
            required_fields=["role"],
            optional_fields=["max_iterations", "additional_guidelines"],
            description="Agent that breaks complex tasks into steps"
        ),
        
        AgentType.DOMAIN_EXPERT: PromptTemplate(
            template="""You are a {domain} expert specializing in {specialization}.

Your expertise includes:
{expertise_areas}

When users ask about {domain} topics:
1. Assess their knowledge level from their question
2. Provide explanations appropriate to that level
3. Use domain-specific terminology but explain new terms
4. Give practical examples from real-world scenarios
5. Reference authoritative sources when available

You should NOT:
- Give advice outside your domain of {domain}
- Make up information if you're unsure
- Provide advice that requires professional certification

When unsure, say: "This question is outside my area of expertise in {domain}. I recommend consulting with a {fallback_expert}."

{additional_guidelines}""",
            required_fields=["domain", "specialization"],
            optional_fields=["expertise_areas", "fallback_expert", "additional_guidelines"],
            description="Domain expert with specialized knowledge"
        ),
        
        AgentType.CUSTOMER_SERVICE: PromptTemplate(
            template="""You are a {company} customer service agent helping customers with {primary_services}.

Your goal is to:
- Resolve customer issues efficiently
- Maintain a friendly, professional tone
- Follow company policies strictly
- Escalate when appropriate

Available actions:
{available_actions}

Escalation criteria:
{escalation_criteria}

Important policies:
{policies}

When responding:
- Acknowledge the customer's concern first
- Provide clear, step-by-step solutions
- Verify understanding before closing
- Thank them for their patience

Tone guidelines:
- Professional but warm
- Patient and understanding
- Clear and concise
- Apologize when appropriate but don't over-apologize""",
            required_fields=["company", "primary_services"],
            optional_fields=["available_actions", "escalation_criteria", "policies"],
            description="Customer service agent with escalation paths"
        ),
        
        AgentType.RESEARCH: PromptTemplate(
            template="""You are a research assistant specializing in {research_domain}.

Your research process:
1. Understand the research question
2. Identify relevant sources to search
3. Gather information from multiple sources
4. Analyze and synthesize findings
5. Present results with proper citations

Research guidelines:
- Always cite sources with URLs when available
- Distinguish facts from interpretations
- Present multiple perspectives when they exist
- Note limitations and gaps in the research
- Be explicit about confidence levels

Citation format:
{citation_format}

When information is limited or conflicting:
- State this explicitly
- Present what you did find
- Suggest alternative search strategies

Output structure:
## Summary
[1-2 sentence overview]

## Findings
[Detailed information with citations]

## Analysis
[Your synthesis and interpretation]

## Limitations
[What's missing or uncertain]

## Sources
[Numbered references]""",
            required_fields=["research_domain"],
            optional_fields=["citation_format"],
            description="Research assistant with citation requirements"
        ),
        
        AgentType.CODE_ASSISTANT: PromptTemplate(
            template="""You are a {language} programming assistant.

Your capabilities:
- Write clean, well-documented code
- Debug and fix errors
- Explain code concepts
- Suggest best practices
- Review code for improvements

When writing code:
- Include docstrings and comments
- Follow {language} style guidelines ({style_guide})
- Handle errors appropriately
- Write tests when requested
- Explain your implementation choices

When debugging:
1. Understand the error or unexpected behavior
2. Identify the root cause
3. Propose a fix with explanation
4. Suggest how to prevent similar issues

When explaining concepts:
- Start with simple examples
- Build up to complex cases
- Provide runnable code when possible
- Link to official documentation

Code formatting:
- Use {indentation}
- Maximum line length: {max_line_length}
- Type hints: {use_type_hints}

{additional_guidelines}""",
            required_fields=["language"],
            optional_fields=[
                "style_guide", 
                "indentation", 
                "max_line_length",
                "use_type_hints",
                "additional_guidelines"
            ],
            description="Programming assistant with coding standards"
        ),
    }
    
    def generate(
        self, 
        agent_type: AgentType,
        **kwargs: Any
    ) -> str:
        """
        Generate a prompt from a template.
        
        Args:
            agent_type: Type of agent to generate prompt for
            **kwargs: Template field values
            
        Returns:
            Completed prompt string
            
        Raises:
            ValueError: If required fields are missing
        """
        template = self.TEMPLATES[agent_type]
        
        # Check required fields
        missing = [f for f in template.required_fields if f not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )
        
        # Set defaults for optional fields
        for field in template.optional_fields:
            if field not in kwargs:
                kwargs[field] = self._get_default(field, agent_type)
        
        # Format lists as bullet points
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = "\n".join(f"- {item}" for item in value)
        
        # Generate the prompt
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Invalid template field: {e}")
    
    def _get_default(self, field: str, agent_type: AgentType) -> str:
        """Get default value for optional field."""
        defaults = {
            "capabilities": "- [Specify capabilities]",
            "constraints": "- [Specify constraints]",
            "guidelines": "- [Specify guidelines]",
            "output_format": "Provide clear, structured responses",
            "additional_guidelines": "",
            "max_iterations": "10",
            "expertise_areas": "- [Specify areas of expertise]",
            "fallback_expert": "qualified professional",
            "available_actions": "- [Specify available actions]",
            "escalation_criteria": "- [Specify when to escalate]",
            "policies": "- [Specify company policies]",
            "citation_format": "Use inline citations [1] with numbered references at the end",
            "style_guide": "PEP 8",
            "indentation": "4 spaces",
            "max_line_length": "88",
            "use_type_hints": "Required for all function parameters and returns",
        }
        return defaults.get(field, "")
    
    def list_templates(self) -> None:
        """Print available templates and their descriptions."""
        print("Available Prompt Templates:")
        print("=" * 70)
        
        for agent_type, template in self.TEMPLATES.items():
            print(f"\n{agent_type.value.upper()}")
            print("-" * 70)
            print(f"Description: {template.description}")
            print(f"Required fields: {', '.join(template.required_fields)}")
            if template.optional_fields:
                print(f"Optional fields: {', '.join(template.optional_fields)}")
    
    def get_example(self, agent_type: AgentType) -> str:
        """
        Get an example prompt for a template.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Example prompt
        """
        examples = {
            AgentType.BASIC: {
                "role": "productivity assistant",
                "purpose": "organize and prioritize tasks",
                "capabilities": [
                    "Create task lists",
                    "Set priorities",
                    "Suggest time management strategies"
                ],
                "constraints": [
                    "Don't delete user's tasks without confirmation",
                    "Respect user's working hours preferences"
                ],
                "guidelines": [
                    "Always confirm before making changes",
                    "Provide reasoning for priority suggestions"
                ],
                "output_format": "Use markdown formatting with clear sections"
            },
            AgentType.TOOL_USER: {
                "role": "data analyst",
                "tool_descriptions": [
                    "query_database: Retrieve data from SQL database",
                    "calculate_stats: Compute statistical measures",
                    "create_chart: Generate visualizations"
                ]
            },
            AgentType.MULTI_STEP: {
                "role": "project planner",
                "max_iterations": "15"
            },
            AgentType.DOMAIN_EXPERT: {
                "domain": "nutrition",
                "specialization": "sports nutrition and meal planning",
                "expertise_areas": [
                    "Macronutrient planning",
                    "Supplement recommendations",
                    "Performance nutrition timing"
                ],
                "fallback_expert": "registered dietitian"
            },
            AgentType.CUSTOMER_SERVICE: {
                "company": "TechCorp",
                "primary_services": "software subscriptions and technical support",
                "available_actions": [
                    "Look up account details",
                    "Process refunds",
                    "Reset passwords",
                    "Create support tickets"
                ],
                "escalation_criteria": [
                    "Billing disputes over $100",
                    "Data security concerns",
                    "Legal or compliance questions"
                ],
                "policies": [
                    "Refunds within 30 days of purchase",
                    "One password reset per 24 hours",
                    "No account sharing allowed"
                ]
            },
            AgentType.RESEARCH: {
                "research_domain": "artificial intelligence and machine learning",
                "citation_format": "Use APA format with inline citations and full references"
            },
            AgentType.CODE_ASSISTANT: {
                "language": "Python",
                "style_guide": "PEP 8",
                "indentation": "4 spaces",
                "max_line_length": "88",
                "use_type_hints": "Required"
            }
        }
        
        return self.generate(agent_type, **examples.get(agent_type, {}))


# Example usage
if __name__ == "__main__":
    generator = PromptGenerator()
    
    # List available templates
    print("=" * 70)
    print("PROMPT TEMPLATE GENERATOR")
    print("=" * 70)
    generator.list_templates()
    
    # Generate some example prompts
    print("\n\n" + "=" * 70)
    print("EXAMPLE: RESEARCH ASSISTANT")
    print("=" * 70)
    
    research_prompt = generator.generate(
        AgentType.RESEARCH,
        research_domain="climate science and environmental policy",
        citation_format="Use inline citations [1] and include DOI when available"
    )
    print(research_prompt)
    
    print("\n\n" + "=" * 70)
    print("EXAMPLE: CUSTOMER SERVICE AGENT")
    print("=" * 70)
    
    cs_prompt = generator.generate(
        AgentType.CUSTOMER_SERVICE,
        company="CloudSoft",
        primary_services="cloud storage and backup solutions",
        available_actions=[
            "Check storage quota",
            "Restore deleted files",
            "Upgrade/downgrade plans",
            "Reset account passwords"
        ],
        escalation_criteria=[
            "Data loss claims",
            "Billing disputes over $50",
            "Security breach reports",
            "Legal requests"
        ],
        policies=[
            "30-day free trial with no credit card",
            "Files kept in trash for 30 days",
            "Refunds issued within 5 business days",
            "No file size limits on paid plans"
        ]
    )
    print(cs_prompt)
    
    print("\n\n" + "=" * 70)
    print("EXAMPLE: CODE ASSISTANT")
    print("=" * 70)
    
    code_prompt = generator.generate(
        AgentType.CODE_ASSISTANT,
        language="Python",
        style_guide="PEP 8 with Black formatting",
        use_type_hints="Required for all functions",
        additional_guidelines="""
Special focus areas:
- Async/await best practices
- Error handling patterns
- Performance optimization
- Security considerations"""
    )
    print(code_prompt)
    
    print("\n\n" + "=" * 70)
    print("\nTip: Customize these templates for your specific use case!")
    print("Test your prompts with prompt_tester.py to find weaknesses.")
