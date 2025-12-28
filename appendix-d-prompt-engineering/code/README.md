# Appendix D: Code Examples

This directory contains practical tools for prompt engineering.

## Files

### `prompt_generator.py`
Template-based prompt generator for common agent patterns.

**Features:**
- 7 pre-built templates (basic, tool-user, multi-step, etc.)
- Customizable with your specific requirements
- Examples for each template type

**Usage:**
```python
from prompt_generator import PromptGenerator, AgentType

generator = PromptGenerator()

# Generate a research assistant prompt
prompt = generator.generate(
    AgentType.RESEARCH,
    research_domain="machine learning",
    citation_format="APA format with DOI"
)
```

### `prompt_tester.py`
Framework for testing prompts against diverse scenarios.

**Features:**
- 10 test categories (happy path, edge cases, etc.)
- Automated test suite execution
- Detailed reporting of failures

**Usage:**
```python
from prompt_tester import PromptTester, TestCase, TestCategory

# Define your system prompt and tools
system_prompt = "..."
tools = [...]

# Create and run tests
tester = PromptTester(system_prompt, tools)
test_cases = [...]  # Your test cases
results = tester.run_test_suite(test_cases)
tester.print_report(results)
```

## Quick Start

1. **Generate a prompt template:**
   ```bash
   python prompt_generator.py
   ```
   This shows all available templates with examples.

2. **Test your prompt:**
   ```bash
   python prompt_tester.py
   ```
   This runs a sample test suite against a weather assistant.

3. **Iterate:**
   - Review failed tests
   - Adjust your prompt
   - Re-run tests
   - Repeat until all tests pass

## Testing Strategy

Use the 10-test rule from the appendix:

1. **Happy path** - Ideal case
2. **Edge cases** - Unusual but valid
3. **Ambiguous** - Multiple interpretations
4. **Missing info** - Incomplete input
5. **Contradictory** - Conflicting instructions
6. **Off-topic** - Outside scope
7. **Malformed** - Typos and errors
8. **Complex** - Multi-part questions
9. **Follow-up** - Needs context
10. **Stress test** - Maximum complexity

## Key Principles

1. **Start with templates** - Don't write prompts from scratch
2. **Test early** - Catch issues before deployment
3. **Iterate based on failures** - Let real failures guide improvements
4. **Be specific** - Vague prompts produce vague behavior
5. **Show examples** - More effective than descriptions

## Common Patterns

### Tool Descriptions
```python
{
    "name": "action_verb",  # search_web, not just "search"
    "description": """What it does (one sentence).
    
Use when:
- Specific scenario 1
- Specific scenario 2

DO NOT use for:
- What it's NOT for
""",
    "input_schema": {
        # Detailed parameter descriptions with examples
    }
}
```

### System Prompts
```python
system_prompt = """You are a [SPECIFIC ROLE].

Capabilities:
- Concrete capability 1
- Concrete capability 2

Rules:
- Specific constraint 1
- Specific constraint 2

When responding:
- Behavioral guideline 1
- Behavioral guideline 2

[Include termination conditions for agents]
"""
```

## Further Reading

- **Appendix D**: Full guide to prompt engineering patterns
- **Chapter 6**: System Prompts and Instructions
- **Chapter 37**: Debugging Agents
- **Anthropic Docs**: https://docs.anthropic.com/claude/docs/prompt-engineering

## Tips

💡 **Test your prompts** against edge cases before production

💡 **Include "what NOT to do"** in tool descriptions to reduce confusion

💡 **Add termination conditions** for agentic loops

💡 **Show examples** for complex tool use patterns

⚠️ **Agent bugs are usually prompt bugs** - check prompts first when debugging

⚠️ **Vague prompts produce vague behavior** - be specific about roles and constraints
