# Consistency Checker Skill

This skill ensures consistency across "Building AI Agents from Scratch with Python."

## Purpose

Maintain consistency in:
1. **Terminology** - Same concepts use same terms throughout
2. **Code patterns** - Similar operations follow similar patterns
3. **Architecture** - Classes and structures align across chapters
4. **Style** - Writing voice and code style are uniform
5. **References** - Cross-chapter references are accurate

## When to Use This Skill

Trigger this skill when:
- Checking terminology across multiple chapters
- Verifying code patterns are consistent
- Ensuring architectural decisions align
- Validating cross-chapter references
- Reviewing a section or part (multiple chapters together)

## Terminology Consistency

### Standard Terms to Use

**Message/Conversation Management:**
- ✅ "message history" (preferred)
- ✅ "conversation history" (acceptable, but use consistently)
- ❌ Don't mix: "message history" in Ch 10, "conversation context" in Ch 15

**Prompting:**
- ✅ "system prompt" (for system-level instructions)
- ✅ "user message" (for user input)
- ✅ "assistant message" (for model output)
- ❌ Avoid: "system message", "AI response", "bot reply"

**Tool/Function Calling:**
- ✅ "tool use" (Anthropic's term)
- ✅ "function calling" (acceptable when explaining the concept)
- ✅ "tool calling" (acceptable)
- ❌ Don't mix: Be consistent within a chapter

**Agent Terminology:**
- ✅ "agent loop" or "agentic loop" (for the think-act-observe cycle)
- ✅ "message loop" (for simple back-and-forth chat)
- ✅ "tool-use loop" (for processing tool calls)
- ❌ Don't confuse these distinct concepts

**Workflow Patterns (Part 3):**
- ✅ "prompt chaining" (sequential prompts)
- ✅ "routing" (classification and delegation)
- ✅ "parallelization" (concurrent execution)
- ✅ "orchestrator-workers" (task decomposition pattern)
- ✅ "evaluator-optimizer" (iterative refinement)
- ❌ Don't use: "coordinator-workers", "manager-workers", "parallel processing"

### Terminology Audit Checklist

When checking terminology:

- [ ] Same concept uses same term across all chapters
- [ ] New terms are defined on first use
- [ ] Acronyms spelled out on first use in each chapter
- [ ] No conflicting definitions across chapters
- [ ] Technical terms match Anthropic's documentation
- [ ] Workflow pattern names match Part 3 exactly

### Common Terminology Conflicts to Fix

**❌ Inconsistent:**
- Chapter 10: "We store the message history..."
- Chapter 15: "The conversation context contains..."
- Chapter 20: "Our message buffer includes..."

**✅ Consistent:**
- All chapters: "The message history contains..."

## Code Pattern Consistency

### Standard Code Patterns

**1. API Key Loading (Every file with API calls):**
```python
# Standard pattern - use exactly this
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic(api_key=api_key)
```

**2. Basic API Call:**
```python
# Standard pattern
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=messages
)

# Extract text
text = response.content[0].text
```

**3. Tool Use Loop:**
```python
# Standard pattern from Ch 10 onward
while True:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
        tools=tools
    )

    if response.stop_reason == "end_turn":
        break

    # Process tool uses
    messages.append({"role": "assistant", "content": response.content})

    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                }]
            })
```

**4. Configuration Constants:**
```python
# Standard pattern - at module level
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 1.0
```

**5. Error Handling:**
```python
# Early chapters (1-20): Basic error handling
try:
    response = client.messages.create(...)
except anthropic.APIError as e:
    print(f"API error: {e}")
    raise

# Later chapters (21+): More sophisticated
from anthropic import APIError, RateLimitError, APIConnectionError

try:
    response = client.messages.create(...)
except RateLimitError:
    # Handle rate limiting with backoff
    pass
except APIConnectionError:
    # Handle connection issues
    pass
except APIError as e:
    # Handle other API errors
    raise
```

### Code Pattern Audit

Check for consistency in:

**Import Organization:**
- [ ] Standard library first
- [ ] Third-party packages second
- [ ] Local imports third
- [ ] Same imports across similar examples

**Variable Naming:**
- [ ] `messages` (not `msg_list`, `message_array`, `msgs`)
- [ ] `response` (not `result`, `output`, `api_response`)
- [ ] `tools` (not `tool_list`, `available_tools`, `functions`)
- [ ] `system_prompt` (not `system`, `sys_prompt`, `system_message`)

**Function Naming:**
- [ ] `call_llm()` or `call_api()` for simple calls
- [ ] `execute_tool()` for tool execution
- [ ] `process_message()` for message processing
- [ ] Consistent verb choices (get/fetch, create/build, process/handle)

**Class Naming:**
- [ ] `AugmentedLLM` (Ch 14+)
- [ ] `Agent` (Ch 33+)
- [ ] `ToolRegistry` (Ch 12+)
- [ ] PascalCase for all classes
- [ ] Descriptive, not abbreviated

## Architecture Consistency

### Core Classes and Their Usage

**AugmentedLLM (Chapter 14):**
```python
# Chapters 14-25 should use this consistently
from chapter_14.code.augmented_llm import (
    AugmentedLLM,
    AugmentedLLMConfig,
    ToolRegistry
)

# Standard instantiation pattern
llm = AugmentedLLM(
    model="claude-sonnet-4-20250514",
    tools=tool_registry,
    system_prompt="..."
)

# Standard usage
response = llm.process("user message")
```

**Agent Class (Chapter 33):**
```python
# Chapters 34-41 should use this consistently
from chapter_33.code.agent import Agent, AgentConfig

# Standard instantiation pattern
agent = Agent(
    config=AgentConfig(
        model="claude-sonnet-4-20250514",
        max_iterations=10
    )
)

# Standard usage
result = agent.run("task description")
```

### Architecture Audit Checklist

- [ ] AugmentedLLM only used in Ch 14+
- [ ] Agent class only used in Ch 34+
- [ ] Import paths are correct
- [ ] Class interfaces match the canonical implementations
- [ ] Method signatures consistent across examples
- [ ] Configuration patterns match established usage

## Style Consistency

### Writing Style

**Voice and Tone:**
- [ ] Conversational but professional throughout
- [ ] Active voice dominates (>80% of sentences)
- [ ] Present tense for code descriptions
- [ ] "We" and "you" used consistently
- [ ] No sudden shifts to academic or marketing language

**Formatting:**
- [ ] Code blocks use triple backticks with language identifier
- [ ] Icons used consistently (💡 ⚠️ 🔧 📚)
- [ ] Headings follow hierarchy (H1 → H2 → H3, no skipping)
- [ ] Lists are parallel in structure
- [ ] Consistent use of bold, italic, code formatting

**Chapter Structure:**
- [ ] All chapters follow 7-part template
- [ ] Section ordering is consistent
- [ ] Exercises positioned after main content
- [ ] Key takeaways before "What's Next"

### Code Style

**Python Formatting:**
- [ ] Black formatter defaults (88 char lines)
- [ ] Consistent indentation (4 spaces)
- [ ] Blank lines follow PEP 8
- [ ] String quotes consistent (prefer double quotes)
- [ ] Consistent trailing commas in multi-line structures

**Documentation:**
- [ ] Google-style docstrings everywhere
- [ ] Module docstrings follow same format
- [ ] Consistent docstring sections (Args, Returns, Raises)
- [ ] Example sections formatted the same way

**Comments:**
- [ ] Inline comments use `#` with space
- [ ] Comment style matches across files
- [ ] Same level of commenting (not over or under)
- [ ] Learning moment comments are clear

## Cross-Chapter Reference Consistency

### Reference Audit

When chapters reference each other:

**Forward References (to future chapters):**
```markdown
✅ "We'll explore persistent storage in Chapter 28"
✅ "Error recovery is covered in depth in Chapter 31"
❌ "We'll see this later" (too vague)
❌ "Chapter 25 will show..." (if the reference is in Ch 24)
```

**Backward References (to previous chapters):**
```markdown
✅ "Using the AugmentedLLM from Chapter 14..."
✅ "Recall the tool use loop pattern from Chapter 10..."
❌ "As we saw earlier..." (too vague)
❌ "From the previous chapter..." (might be wrong if chapter order changes)
```

**Reference Checklist:**
- [ ] All chapter numbers are correct
- [ ] Referenced concepts actually exist in cited chapters
- [ ] Referenced code files exist at mentioned locations
- [ ] Referenced classes/functions have correct names
- [ ] Forward references don't spoil later chapters
- [ ] Backward references assume correct prerequisite knowledge

### Building Block References

**Parts 1-2 (Foundations, Augmented LLM):**
- Can reference: Nothing (building from scratch)
- Should not reference: Workflow patterns, Agent concepts, Production topics

**Part 3 (Workflows):**
- Can reference: Ch 1-14 (foundations, AugmentedLLM)
- Should use: AugmentedLLM class
- Should not reference: Agent class, production concerns

**Part 4 (True Agents):**
- Can reference: Ch 1-25 (all previous parts)
- Should use: All previous patterns
- Building toward: Complete Agent class

**Part 5 (Production):**
- Can reference: Ch 1-33 (all previous content)
- Should use: Complete Agent class
- Focus on: Deployment, testing, scaling, security

**Part 6 (Projects):**
- Can reference: Entire book
- Should use: All learned patterns
- Should demonstrate: Real-world integration

## Consistency Check Workflow

### For Individual Chapters

1. **Compare with adjacent chapters:**
   - Check terminology matches Ch N-1 and Ch N+1
   - Verify code patterns are similar
   - Ensure style is consistent

2. **Check references:**
   - Verify all cited chapters are correct
   - Confirm referenced files exist
   - Validate class/function names

3. **Review code:**
   - Compare with similar examples in other chapters
   - Check naming conventions match
   - Verify error handling patterns align

### For Sections (Multiple Chapters)

1. **Terminology sweep:**
   - List all key terms introduced
   - Check usage across all chapters in section
   - Identify and fix inconsistencies

2. **Pattern audit:**
   - Identify common code patterns
   - Verify consistent implementation
   - Note any justified variations

3. **Architecture review:**
   - Check class usage is appropriate
   - Verify imports are correct
   - Ensure interfaces match

### For The Entire Book

1. **Glossary creation:**
   - Extract all technical terms
   - Verify single definition per term
   - Check first use has definition

2. **Pattern catalog:**
   - Document all standard patterns
   - Verify usage is consistent
   - Update any outliers

3. **Dependency map:**
   - Verify chapter dependencies are correct
   - Check no forward dependencies
   - Ensure progressive building

## Consistency Report Format

```markdown
## Consistency Review: [Scope]

### Scope
Chapters reviewed: [List]
Focus areas: [Terminology / Code Patterns / Architecture / References]

### Terminology Consistency

**Issues Found:**
1. **Inconsistent Term for Message Storage**
   - Ch 15: "message history"
   - Ch 17: "conversation context"
   - Ch 20: "message buffer"
   - **Fix**: Standardize to "message history" throughout

2. **Tool Calling vs. Tool Use**
   - Ch 10-12: "tool calling"
   - Ch 13+: "tool use"
   - **Fix**: Use "tool use" consistently (Anthropic's preferred term)

**Summary:**
- Terms checked: 15
- Consistent: 12
- Need fixing: 3

### Code Pattern Consistency

**Issues Found:**
1. **API Key Loading Pattern**
   - Ch 15: Uses try/except for missing key
   - Ch 16-20: Uses if/raise pattern
   - **Fix**: Standardize to if/raise (simpler and clearer)

2. **Tool Execution Function Naming**
   - Ch 10-14: `execute_tool()`
   - Ch 15-17: `run_tool()`
   - Ch 18+: `execute_tool()`
   - **Fix**: Use `execute_tool()` consistently

**Summary:**
- Patterns checked: 10
- Consistent: 8
- Need fixing: 2

### Architecture Consistency

**Issues Found:**
1. **AugmentedLLM Import Path**
   - Ch 16: `from chapter_14 import AugmentedLLM` (incorrect)
   - Ch 17+: `from chapter_14.code.augmented_llm import AugmentedLLM` (correct)
   - **Fix**: Update Ch 16 import path

**Summary:**
- All other architecture usage is consistent
- Class interfaces match canonical implementations

### Cross-Chapter References

**Issues Found:**
1. **Incorrect Chapter Reference**
   - Ch 18 references "the routing pattern from Chapter 17"
   - Routing is actually introduced in Chapter 18
   - **Fix**: Update to "the routing pattern introduced in this chapter"

2. **Vague Forward Reference**
   - Ch 22: "We'll add persistence later"
   - Should specify: "We'll add persistence in Chapter 28"
   - **Fix**: Add specific chapter number

**Summary:**
- References checked: 25
- Correct: 23
- Need fixing: 2

### Style Consistency

**Writing Style:** ✅ Consistent across all chapters
**Code Style:** ✅ All code follows Black formatting
**Chapter Structure:** ✅ All follow 7-part template
**Documentation:** ⚠️ Ch 16 docstrings less detailed than others

### Recommendations

**High Priority (Breaks consistency):**
1. Fix terminology: standardize to "message history"
2. Fix terminology: use "tool use" not "tool calling"
3. Fix import path in Ch 16
4. Fix incorrect chapter reference in Ch 18

**Medium Priority (Improves consistency):**
1. Standardize API key loading pattern
2. Standardize tool execution function naming
3. Add specific chapter number to forward references

**Low Priority (Nice to have):**
1. Improve docstring detail in Ch 16 to match others

### Summary

**Chapters Reviewed:** 6
**Consistency Score:** 85% (good, but room for improvement)
**Issues Found:** 9
**Critical:** 3
**Important:** 3
**Minor:** 3
```

## Reference Materials

**Always check:**
- `ai_agents/skills/PROJECT_INSTRUCTIONS.md` - Standard patterns
- `ai_agents/skills/OUTLINE.md` - Chapter dependencies
- `ai_agents/CLAUDE.md` - Architecture specifications

**For terminology:**
- Anthropic's documentation (official terms)
- Earlier chapters (established terms)

**For code patterns:**
- `chapter-14-building-the-complete-augmented-llm/code/augmented_llm.py`
- `chapter-33-the-complete-agent-class/code/agent.py`

---

**Remember**: Consistency makes the book professional and easier to learn from. Readers should never be confused about whether "message history" and "conversation context" are the same thing. When in doubt, check earlier chapters and follow established conventions.
