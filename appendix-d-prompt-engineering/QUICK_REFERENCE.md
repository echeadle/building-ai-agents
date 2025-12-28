# Prompt Engineering Quick Reference

## The 5-Part System Prompt

```python
system_prompt = """
1. ROLE: You are a [specific role]...

2. CAPABILITIES: What you can do
   - Capability 1
   - Capability 2

3. CONSTRAINTS: What you must NOT do
   - Constraint 1
   - Constraint 2

4. GUIDELINES: How to behave
   - Guideline 1
   - Guideline 2

5. OUTPUT FORMAT: Expected structure
   [Description of format]
"""
```

## Tool Description Template

```python
{
    "name": "action_verb_target",  # search_web, not "search"
    "description": """[One sentence: what it does]
    
Use this tool when:
- [Specific scenario 1]
- [Specific scenario 2]
- [Specific scenario 3]

DO NOT use this tool for:
- [What it's NOT for 1]
- [What it's NOT for 2]
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "What it is. Format: [format]. Example: [example]"
            }
        },
        "required": ["param_name"]
    }
}
```

## Common Mistakes to Avoid

| ❌ Bad | ✅ Good |
|--------|---------|
| "You are a helpful assistant" | "You are a technical documentation assistant for API developers" |
| "Search for things" | "Searches internal docs for API endpoints and code examples" |
| "Be concise but detailed" | "1-2 sentences for facts, detailed for explanations" |
| No termination conditions | "Stop after 5 tool calls or when you have complete information" |
| "Use the right tool" | "web_search for facts, calculator for math, email for messages" |

## Patterns at a Glance

### Chain-of-Thought
```python
"For complex requests, think step-by-step:
1. Understand the goal
2. Plan the approach
3. Execute with tools
4. Verify the result
5. Respond with details"
```

### Plan-Then-Execute
```python
"PHASE 1: Create a numbered plan
PHASE 2: Execute each step, showing progress
PHASE 3: Summarize what was accomplished"
```

### Self-Correction
```python
"After using a tool:
1. Check if the result makes sense
2. Look for obvious errors
3. If wrong, try a different approach"
```

### Confidence Levels
```python
"Express confidence based on sources:
- HIGH: Multiple authoritative sources agree
- MEDIUM: Single source or missing details
- LOW: Sparse or conflicting information
- UNKNOWN: No reliable information found"
```

## Testing Checklist

Before deploying, test with:

- [ ] 1. Happy path (ideal case)
- [ ] 2. Edge case (unusual but valid)
- [ ] 3. Ambiguous input
- [ ] 4. Missing information
- [ ] 5. Contradictory instructions
- [ ] 6. Off-topic request
- [ ] 7. Malformed input (typos)
- [ ] 8. Complex multi-part request
- [ ] 9. Follow-up without context
- [ ] 10. Maximum complexity stress test

## Debugging Decision Tree

```
Agent not working?
│
├─ Wrong tool selected?
│  └─ Check tool descriptions for clarity
│
├─ Infinite loop?
│  └─ Add explicit termination conditions
│
├─ Ignoring instructions?
│  └─ Check for conflicting guidance
│
├─ Inconsistent behavior?
│  └─ Add few-shot examples
│
└─ Off-topic responses?
   └─ Strengthen role definition
```

## Quick Wins

1. **Add "DO NOT use for"** to tool descriptions → Reduces wrong tool selection by ~40%

2. **Put important info at the end** → LLMs have recency bias

3. **Include examples for tricky behaviors** → More effective than descriptions

4. **Specify termination conditions** → Prevents infinite loops

5. **Test with the 10-test rule** → Catches 90% of issues before production

## Useful Prompting Phrases

### For clarity:
- "Use [tool_name] when the user asks about [specific scenario]"
- "If [condition], then [action]. Otherwise, [alternative action]"
- "Always [behavior] before [other behavior]"

### For constraints:
- "Never [action] without [condition]"
- "Only [action] if [requirement] is met"
- "Stop when [termination condition]"

### For format:
- "Respond in this format: [template]"
- "For [situation], use [format]. For [other situation], use [other format]"
- "Always include: [required elements]"

## Remember

🎯 **80% of agent bugs are prompt bugs**

🎯 **Test prompts early and often**

🎯 **Iterate based on real failures**

🎯 **Be specific > Be vague**

🎯 **Show examples > Describe behavior**
