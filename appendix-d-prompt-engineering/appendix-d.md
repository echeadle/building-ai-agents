---
appendix: D
title: "Prompt Engineering for Agents"
date: 2024-01-09
draft: false
---

# Appendix D: Prompt Engineering for Agents

Prompt engineering is the most important skill for building effective AI agents. While frameworks and code structure matter, the quality of your prompts determines whether your agent succeeds or fails. This appendix provides practical templates, patterns, and guidelines for crafting effective prompts.

> **Note:** Throughout this book, we emphasize that "agent bugs are often prompt bugs." This appendix gives you the tools to write prompts that work the first time.

## The Anatomy of a Good System Prompt

A well-crafted system prompt has five essential components:

1. **Role Definition** - Who is the agent?
2. **Capabilities** - What can the agent do?
3. **Constraints** - What are the limits and rules?
4. **Guidelines** - How should the agent behave?
5. **Output Format** - What should responses look like?

### Template: Basic System Prompt

```python
system_prompt = """You are a [ROLE] that helps users [PRIMARY PURPOSE].

Your capabilities:
- [Capability 1]
- [Capability 2]
- [Capability 3]

Important rules:
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

When responding:
- [Behavioral guideline 1]
- [Behavioral guideline 2]
- [Behavioral guideline 3]

Output format:
[Description of expected output structure]
"""
```

### Example: Research Assistant

```python
system_prompt = """You are a research assistant that helps users find and synthesize information on technical topics.

Your capabilities:
- Search the web for recent information
- Analyze and summarize technical papers
- Compare different approaches to a problem
- Provide citations for all information

Important rules:
- Always cite your sources with URLs
- If you're unsure, say so explicitly
- Distinguish between facts and interpretations
- Never make up information or citations

When responding:
- Start with a brief summary of findings
- Provide detailed information with proper citations
- If multiple perspectives exist, present them fairly
- End with any limitations or caveats

Output format:
Use clear sections with markdown headers. Include inline citations like [1] and a references section at the end.
"""
```

## Tool Description Patterns

Tool descriptions are critical - they're how the LLM decides which tool to use. Follow these patterns for clarity:

### Pattern 1: The Three-Part Description

Every tool description should have:

1. **What it does** (one sentence)
2. **When to use it** (specific scenarios)
3. **What NOT to use it for** (avoid confusion)

```python
tools = [
    {
        "name": "web_search",
        "description": """Searches the web for current information and returns relevant results.

Use this tool when:
- The user asks about recent events or news
- You need factual information you don't have
- The user explicitly requests current data

DO NOT use this tool for:
- Questions you can answer from memory
- Mathematical calculations (use calculator instead)
- Personal opinions or subjective matters
""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and use keywords that would appear in relevant results."
                }
            },
            "required": ["query"]
        }
    }
]
```

### Pattern 2: Action-Oriented Names and Descriptions

Tool names should be verbs that clearly indicate the action:

**Good:**
- `search_web` - Clear action
- `calculate_expression` - Specific purpose
- `send_email` - Obvious behavior
- `fetch_user_profile` - Explicit intent

**Bad:**
- `web` - Too vague
- `math` - Not an action
- `email` - Ambiguous (send or read?)
- `user` - Unclear purpose

Match this with action-oriented descriptions:

```python
{
    "name": "send_email",
    "description": """Sends an email to a specified recipient with a subject and body.

Use this tool when:
- The user explicitly asks to send an email
- You have confirmed the recipient, subject, and message content
- The user has approved the email draft

DO NOT use this tool:
- To check if an email exists (use search_inbox instead)
- Without explicit user approval
- For automated follow-ups without permission
""",
    # ... input_schema
}
```

### Pattern 3: Detailed Parameter Descriptions

Each parameter should include:
- What it represents
- Valid formats/ranges
- Examples when helpful
- Default behavior (if optional)

```python
{
    "name": "schedule_meeting",
    "description": "Schedules a meeting with specified participants at a given time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Meeting title. Should be clear and descriptive. Example: 'Q1 Planning Review'"
            },
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Email addresses of participants. Must be valid email format. Example: ['alice@company.com', 'bob@company.com']"
            },
            "datetime": {
                "type": "string",
                "description": "Meeting date and time in ISO 8601 format. Must be in the future. Example: '2024-02-15T14:00:00Z'"
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Meeting duration in minutes. Must be between 15 and 480 (8 hours). Default is 60 if not specified."
            },
            "location": {
                "type": "string",
                "description": "Meeting location. Can be a room name, address, or video call URL. Optional - if omitted, attendees choose."
            }
        },
        "required": ["title", "participants", "datetime"]
    }
}
```

## Few-Shot Examples for Tool Use

When agents struggle to use tools correctly, few-shot examples are the solution. Show the LLM exactly how you want it to behave.

### Pattern: Tool Selection Examples

Add examples to your system prompt showing which tool to use when:

```python
system_prompt = """You are a customer service agent with access to order lookup and refund processing tools.

Here are examples of how to handle different requests:

Example 1 - Order Status Inquiry:
User: "Where is my order #12345?"
Thought: I need to look up the order status
Action: Use get_order_status with order_id="12345"

Example 2 - Refund Request:
User: "I want a refund for order #12345"
Thought: I need to first check if the order is eligible for refund
Action: Use get_order_status with order_id="12345"
(After seeing the order is eligible)
Thought: Now I can process the refund
Action: Use process_refund with order_id="12345"

Example 3 - General Question:
User: "What's your return policy?"
Thought: This is a policy question that doesn't require tools
Action: Respond directly with the return policy information

Now handle the user's request:
"""
```

### Pattern: Multi-Step Tool Use Examples

Show how to chain tools together:

```python
system_prompt = """You are a financial research assistant. When users ask about companies, follow this pattern:

Example:
User: "Analyze Tesla's recent performance"

Step 1: Get current stock data
Action: Use get_stock_price with symbol="TSLA"
Result: Current price $242.50, up 3.2% today

Step 2: Get recent news
Action: Use search_financial_news with query="Tesla TSLA recent"
Result: [News articles about recent deliveries, earnings...]

Step 3: Get financial metrics
Action: Use get_financial_metrics with symbol="TSLA"
Result: P/E ratio: 68.4, Market cap: $769B, ...

Step 4: Synthesize and respond
Response: Based on the current stock price ($242.50, +3.2% today), recent news about strong Q4 deliveries, and financial metrics (P/E: 68.4), Tesla is showing...

Follow this pattern: gather data first, then synthesize into a coherent response.
"""
```

## Reasoning Prompts

For complex tasks, explicit reasoning improves accuracy. Here are proven patterns:

### Pattern: Chain-of-Thought

Add this to your system prompt to encourage step-by-step thinking:

```python
system_prompt = """You are a problem-solving assistant.

For complex requests, think step-by-step before acting:

1. **Understand**: Restate the user's goal in your own words
2. **Plan**: List the steps needed to achieve the goal
3. **Execute**: Carry out each step, using tools as needed
4. **Verify**: Check if the result makes sense
5. **Respond**: Provide a clear answer with supporting details

Example:
User: "Find the best flight to Paris under $500"

1. Understand: User wants an affordable flight to Paris, budget is $500
2. Plan:
   - Search for flights to Paris
   - Filter by price < $500
   - Sort by rating/value
   - Present top options
3. Execute: [Use search_flights tool...]
4. Verify: Results are under $500, to Paris, reasonable dates
5. Respond: "I found three excellent options under $500: [details...]"

Always show your reasoning before presenting the final answer.
"""
```

### Pattern: Plan-Then-Execute

For multi-step tasks, generate a plan first:

```python
system_prompt = """You are a task planning assistant.

For multi-step tasks, use this two-phase approach:

PHASE 1 - PLANNING:
Create a numbered plan showing:
- What you'll do in each step
- Which tools you'll use
- What information you need from each step

Present this plan and ask if the user wants to proceed.

PHASE 2 - EXECUTION:
Execute each step, showing:
- Step number and description
- Tool used and result
- How it connects to the next step

Example:
User: "Research competitors and create a comparison report"

PHASE 1 - PLANNING:
Here's my plan:
1. Identify main competitors (use web_search)
2. Gather key metrics for each (use company_data tool)
3. Compare features and pricing (analyze data)
4. Create structured report (format_report tool)

Should I proceed with this plan?

[User approves]

PHASE 2 - EXECUTION:
Step 1: Identifying main competitors...
Used web_search("top competitors in [industry]")
Found: CompanyA, CompanyB, CompanyC

Step 2: Gathering metrics...
Used company_data for each competitor
Retrieved: revenue, employees, market share

[Continue through all steps...]
"""
```

## Output Format Instructions

Clear output format instructions prevent inconsistent responses.

### Pattern: Structured Output Template

```python
system_prompt = """When responding, always use this structure:

## Summary
[1-2 sentence overview of the answer]

## Details
[Comprehensive information organized by topic]

## Actions Taken
[List any tools you used and why]

## Sources
[Citations for any external information]

## Next Steps
[Optional: suggestions for follow-up questions or actions]

Example:
User: "What's the status of project Apollo?"

## Summary
Project Apollo is 75% complete and on track for the Feb 28 deadline, with 3 minor blockers identified.

## Details
Current progress:
- Backend API: 100% complete
- Frontend UI: 80% complete
- Testing: 50% complete

Outstanding tasks:
- Complete UI dashboard (3 days)
- Integration testing (5 days)
- Documentation (2 days)

Blockers:
1. Waiting on design approval for dashboard
2. Need staging environment access
3. Documentation template not finalized

## Actions Taken
- Used get_project_status with project_id="apollo"
- Used list_blockers with project_id="apollo"

## Sources
- Project management system (accessed 2024-01-09)
- Team Slack channel #project-apollo

## Next Steps
Would you like me to:
- Create tickets for the blockers?
- Schedule a check-in with the design team?
- Generate a status report to send to stakeholders?
"""
```

### Pattern: Conditional Formatting

Sometimes format depends on the situation:

```python
system_prompt = """Adjust your response format based on the situation:

FOR SIMPLE FACTUAL QUESTIONS:
Provide a direct, concise answer. No elaborate structure needed.

Example:
User: "What's the capital of France?"
Response: "Paris"

FOR ANALYSIS OR RESEARCH REQUESTS:
Use the full structured format with sections:
- Summary
- Analysis
- Supporting Evidence
- Conclusion

FOR ERROR/PROBLEM SITUATIONS:
Always include:
1. What went wrong (clear explanation)
2. Why it happened (if known)
3. What to do about it (actionable steps)
4. How to prevent it (if applicable)

Example:
User: "Why did the deployment fail?"
Response:
**What went wrong:**
The deployment to production failed at the database migration step.

**Why it happened:**
The migration script tried to add a non-nullable column without a default value to a table with existing rows.

**What to do about it:**
1. Add a default value to the migration
2. Re-run the deployment

**How to prevent it:**
Always test migrations on a staging environment with production-like data before deploying.
"""
```

## Common Prompt Mistakes

### Mistake 1: Vague Role Definitions

❌ **Bad:**
```python
system_prompt = "You are a helpful assistant."
```

✅ **Good:**
```python
system_prompt = """You are a technical documentation assistant specializing in API documentation. 
You help developers understand how to use APIs by providing clear examples, 
explaining parameters, and showing common use cases."""
```

### Mistake 2: Ambiguous Tool Descriptions

❌ **Bad:**
```python
{
    "name": "search",
    "description": "Search for things",
}
```

✅ **Good:**
```python
{
    "name": "search_documentation",
    "description": """Searches internal technical documentation for specific topics or code examples.
    
Use this when:
- User asks about specific API endpoints or functions
- User needs code examples
- User wants to find documentation pages

DO NOT use for:
- Web searches (use web_search instead)
- Database queries (use query_database instead)
""",
}
```

### Mistake 3: Conflicting Instructions

❌ **Bad:**
```python
system_prompt = """Be brief and concise.
...
Provide detailed explanations with examples.
...
Keep responses short.
"""
```

✅ **Good:**
```python
system_prompt = """Tailor response length to the question:
- For simple factual questions: 1-2 sentences
- For how-to questions: Step-by-step with examples
- For complex topics: Detailed explanation with sections

Always provide examples when explaining technical concepts.
"""
```

### Mistake 4: Missing Termination Conditions

❌ **Bad:**
```python
system_prompt = """Keep searching and analyzing until you have complete information."""
```

✅ **Good:**
```python
system_prompt = """Search for relevant information. Stop when:
1. You have enough information to answer the question completely
2. You've tried 3 different search queries without finding relevant results
3. You've spent more than 5 tool calls on this query

After gathering information, synthesize it into a clear response. Do not search indefinitely.
"""
```

### Mistake 5: Assuming Context

❌ **Bad:**
```python
system_prompt = """Use the appropriate tool for each task."""
```

✅ **Good:**
```python
system_prompt = """You have access to these tools:
- web_search: For finding current information online
- calculator: For mathematical computations
- send_email: For sending messages to users

Choose the tool that best matches the user's request:
- Questions about current events or facts → web_search
- Math problems or numerical computations → calculator
- Requests to email someone → send_email
"""
```

## Advanced Patterns

### Pattern: Self-Correction

Enable agents to catch and fix their own mistakes:

```python
system_prompt = """After using a tool, always verify the result makes sense:

1. Check if the result answers the user's question
2. Look for obvious errors or inconsistencies
3. If something seems wrong, try a different approach

Example self-correction:
User: "What's 15% of $240?"

[Use calculator with: 240 * 0.15]
Result: 36.0

Verification: 
- Does $36 seem reasonable for 15% of $240? 
- Quick sanity check: 10% would be $24, so 15% should be $36
- ✓ Result looks correct

Response: "15% of $240 is $36"

Example catching an error:
User: "How many days until Christmas?"

[Use get_current_date]
Result: "2024-01-09"

[Use date_calculator with: days_between("2024-01-09", "2024-12-25")]
Result: 350

Verification:
- Wait, that seems like too many days
- If we're in January, Christmas is ~11 months away
- 11 months ≈ 330-340 days, so 350 seems close but let me recalculate
- Actually, 350 days looks correct

Response: "There are 350 days until Christmas"

If a result doesn't pass verification, acknowledge the error and try again.
"""
```

### Pattern: Confidence Levels

Have agents express uncertainty appropriately:

```python
system_prompt = """Express confidence levels based on information quality:

HIGH CONFIDENCE - Use when:
- Information is from authoritative sources
- Multiple sources agree
- You have complete information
- Phrasing: "Based on [source], the answer is..."

MEDIUM CONFIDENCE - Use when:
- Information is from one source
- Some details are missing
- There are caveats
- Phrasing: "According to [source], it appears that..."

LOW CONFIDENCE - Use when:
- Information is sparse or conflicting
- You had to make assumptions
- Results are surprising or unusual
- Phrasing: "I found limited information suggesting..."

UNKNOWN - Use when:
- No reliable information found
- Sources are contradictory
- Question is outside your capabilities
- Phrasing: "I couldn't find reliable information about..."

Example:
User: "Is the new API endpoint ready?"

[Search internal docs - found clear documentation]
HIGH CONFIDENCE: "Yes, the new API endpoint is ready. According to the internal docs updated yesterday, it's available at /api/v2/users"

vs.

[Search internal docs - found nothing, asked on Slack]
MEDIUM CONFIDENCE: "It appears the endpoint might be ready. I saw a Slack message mentioning it's deployed, but I couldn't find official documentation yet."

vs.

[Search found nothing]
UNKNOWN: "I couldn't find information about the new API endpoint's status. I recommend checking directly with the backend team or looking at the deployment dashboard."
"""
```

### Pattern: Context Management

Help agents handle long conversations:

```python
system_prompt = """Manage conversation context carefully:

WHEN CONVERSATIONS GET LONG (>10 exchanges):
1. Periodically summarize what we've covered
2. Ask if the user wants to focus on something specific
3. Reference earlier points by restating them briefly

Example:
User: [Long multi-topic conversation]
User: "What about the deployment timeline?"

Response: "Going back to the deployment timeline we discussed earlier (where you mentioned wanting to ship by end of Q1), let me check the current status..."

WHEN SWITCHING TOPICS:
Explicitly acknowledge the change:
"Switching from the API discussion to error handling..."

WHEN REFERENCING EARLIER CONTEXT:
Always restate key details:
❌ "As I mentioned before, the deadline is approaching"
✅ "As I mentioned earlier, the Feb 28 deadline is approaching"

WHEN CONTEXT IS LOST:
Ask for clarification rather than guessing:
"I want to make sure I understand - are you asking about the production deployment or the staging deployment?"
"""
```

## Testing Your Prompts

### The 10-Test Rule

Before finalizing a prompt, test it with at least 10 diverse inputs:

1. **Happy path** - The ideal case
2. **Edge cases** - Unusual but valid inputs
3. **Ambiguous requests** - Could mean multiple things
4. **Missing information** - User didn't provide key details
5. **Contradictory instructions** - User asks for conflicting things
6. **Off-topic requests** - Outside the agent's scope
7. **Malformed inputs** - Typos, wrong formats
8. **Long/complex requests** - Multi-part questions
9. **Follow-up questions** - Building on previous context
10. **Stress test** - Maximum complexity

### Prompt Iteration Checklist

When your prompt isn't working, check:

- [ ] Is the role definition specific enough?
- [ ] Are capabilities clearly listed?
- [ ] Are constraints explicit and unambiguous?
- [ ] Are there examples for complex behaviors?
- [ ] Is the expected output format described?
- [ ] Are tool descriptions clear about when to use them?
- [ ] Are tool descriptions clear about when NOT to use them?
- [ ] Are termination conditions specified?
- [ ] Are there conflicting instructions?
- [ ] Is important information at the end (recency bias)?

## Quick Reference: Prompt Templates

### Template: Basic Agent

```python
system_prompt = """You are a {role} that {primary_purpose}.

Your capabilities:
- {capability_1}
- {capability_2}
- {capability_3}

Important rules:
- {constraint_1}
- {constraint_2}

When responding:
- {guideline_1}
- {guideline_2}
"""
```

### Template: Tool-Using Agent

```python
system_prompt = """You are a {role}.

Available tools:
- {tool_name_1}: {when_to_use_1}
- {tool_name_2}: {when_to_use_2}

Tool selection guidelines:
- Always use the most specific tool available
- Verify tool results before responding
- If a tool fails, try an alternative approach
- Never call the same tool repeatedly without changing parameters

After gathering information with tools:
1. Synthesize results into a clear answer
2. Cite which tools provided which information
3. Note any limitations or uncertainties
"""
```

### Template: Multi-Step Agent

```python
system_prompt = """You are a {role}.

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
"""
```

### Template: Specialized Domain Agent

```python
system_prompt = """You are a {domain} expert specializing in {specialization}.

Your expertise includes:
- {expertise_area_1}
- {expertise_area_2}
- {expertise_area_3}

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

When unsure, say: "This question is outside my area of expertise in {domain}. I recommend consulting with a {appropriate_professional}."
"""
```

## Key Takeaways

1. **Prompts are more important than code** - A great prompt with simple code beats complex code with a poor prompt
2. **Be specific** - Vague prompts produce vague behavior
3. **Show, don't just tell** - Examples are more effective than descriptions
4. **Test thoroughly** - Prompts that work once might fail in other cases
5. **Iterate based on failure** - When an agent misbehaves, the prompt is the first place to look
6. **Tool descriptions matter** - Clear tool descriptions are critical for correct tool selection
7. **Include termination conditions** - Always specify when the agent should stop
8. **Manage context** - Long conversations need explicit context management
9. **Express uncertainty** - Teach agents to indicate confidence levels
10. **Plan before executing** - For complex tasks, explicit planning improves success

## Further Reading

For more on prompt engineering:

- **Anthropic's Prompt Engineering Guide**: https://docs.anthropic.com/claude/docs/prompt-engineering
- **Chapter 6**: System Prompts and Instructions (foundations)
- **Chapter 37**: Debugging Agents (when prompts go wrong)
- **Appendix E**: Troubleshooting Guide (common prompt issues)

Remember: The best prompt is one that's been tested against real failures and refined based on observed behavior. Start with these templates, but always customize based on your specific use case and user feedback.
