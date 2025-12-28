# Technical Reviewer Skill

This skill provides comprehensive technical review standards for "Building AI Agents from Scratch with Python."

## Purpose

Review chapters and code to ensure:
1. **Technical accuracy** - All information is correct and current
2. **Code quality** - Examples work and teach effectively
3. **Architectural consistency** - Patterns align across chapters
4. **Progressive learning** - Complexity builds appropriately
5. **Production readiness** - Code follows best practices

## When to Use This Skill

Trigger this skill when:
- Reviewing completed chapters before marking them done
- Verifying code examples work correctly
- Checking technical accuracy of explanations
- Ensuring consistency with previous chapters
- Validating architectural decisions

## Review Process

### Phase 1: Chapter Structure Review

**Check the chapter follows the 7-part template:**

- [ ] Introduction with hook, context, and preview
- [ ] Learning objectives (specific and measurable)
- [ ] Main content with clear sections
- [ ] Common pitfalls section (2-3 items)
- [ ] Practical exercise with requirements
- [ ] Key takeaways (bulleted summary)
- [ ] What's next (preview of next chapter)

**Verify transitions and flow:**
- [ ] Smooth transitions between sections
- [ ] Clear narrative arc from intro to summary
- [ ] Each section builds on the previous
- [ ] References to other chapters are accurate

### Phase 2: Technical Accuracy Review

**Core Concepts:**
- [ ] All technical concepts are explained correctly
- [ ] Terminology is used consistently and accurately
- [ ] No outdated information or deprecated patterns
- [ ] External references (papers, docs) are correct

**API Usage:**
- [ ] Anthropic API calls use current SDK patterns
- [ ] Model names are correct (`claude-sonnet-4-20250514`)
- [ ] API parameters are used correctly
- [ ] Response handling matches current API structure

**Architecture Patterns:**
- [ ] Patterns match established conventions
- [ ] AugmentedLLM usage (Ch 14+) is correct
- [ ] Agent class usage (Ch 34+) is correct
- [ ] Workflow patterns follow Anthropic's best practices

### Phase 3: Code Quality Review

**Run every code example:**
```bash
cd ai_agents/chapter-XX-title/code/
python example_01.py
python example_02.py
python example_03.py
python exercise_solution.py
```

**Verify for each file:**

#### Functionality
- [ ] Code runs without errors
- [ ] Produces expected output
- [ ] Handles edge cases appropriately
- [ ] Error messages are helpful

#### Teaching Quality
- [ ] Demonstrates the chapter concept clearly
- [ ] Appropriate complexity for chapter position
- [ ] Shows progression from previous chapters
- [ ] Comments explain learning moments
- [ ] Doesn't over-comment obvious code

#### Code Standards
- [ ] Type hints on all function signatures
- [ ] Docstrings (Google style) on all public functions/classes
- [ ] Meaningful variable names
- [ ] Functions are focused and single-purpose
- [ ] Proper error handling (no bare excepts)
- [ ] No unnecessary duplication

#### Security
- [ ] No hardcoded credentials
- [ ] Uses `python-dotenv` for environment variables
- [ ] `.env.example` file provided
- [ ] Input validation where appropriate
- [ ] Error messages don't leak sensitive information

#### Dependencies
- [ ] Uses `uv` for package management (if needed)
- [ ] No framework dependencies (no LangChain, etc.)
- [ ] Only necessary packages included
- [ ] `pyproject.toml` is correct (if present)

### Phase 4: Chapter-Code Alignment

**Verify consistency between text and code:**

- [ ] Code file names mentioned in text match actual files
- [ ] Example numbers in text match file names
- [ ] Code snippets in chapter match actual code files
- [ ] Explanations accurately describe what code does
- [ ] All code files referenced in text exist
- [ ] All code files are referenced in text or README

**Check code references:**
```markdown
# In chapter text, should see:
"Let's look at this implementation (from `example_01_simple.py`):"

# And the file should exist at:
ai_agents/chapter-XX-title/code/example_01_simple.py
```

### Phase 5: Progressive Complexity Review

**Verify the chapter builds appropriately:**

**For early chapters (1-13):**
- [ ] Doesn't assume knowledge of AugmentedLLM
- [ ] Uses only direct API calls
- [ ] Introduces concepts before using them
- [ ] Doesn't reference patterns from later chapters

**For workflow chapters (15-25):**
- [ ] Can use/import AugmentedLLM from Chapter 14
- [ ] Builds on previous workflow patterns
- [ ] Doesn't jump to Agent concepts from Ch 26+
- [ ] Shows evolution of ideas

**For agent chapters (26-33):**
- [ ] Can use all previous patterns
- [ ] Builds toward the complete Agent class
- [ ] Doesn't reference production topics prematurely
- [ ] Shows why each component is needed

**For production chapters (34-41):**
- [ ] Can use complete Agent class
- [ ] Assumes reader understands agent architecture
- [ ] Introduces production concerns appropriately
- [ ] Builds on testing/deployment fundamentals

**For capstone chapters (42-45):**
- [ ] Can use all previous patterns and classes
- [ ] Combines concepts from multiple chapters
- [ ] Shows real-world application
- [ ] Reinforces key learnings

### Phase 6: Cross-Chapter Consistency

**Check terminology consistency:**
- [ ] Same terms used as previous chapters
- [ ] New terms are defined clearly
- [ ] Acronyms spelled out on first use
- [ ] No conflicting definitions

**Common terminology to verify:**
- Message loop / Agent loop / Agentic loop
- Message history / Conversation context / History
- System prompt / System message
- Tool use / Function calling / Tool calling
- Tool result / Function result

**Check pattern consistency:**
- [ ] Error handling follows established patterns
- [ ] API calls use consistent structure
- [ ] Configuration follows previous conventions
- [ ] Class design matches architectural patterns

**Verify code style consistency:**
- [ ] Variable naming matches book conventions
- [ ] Function structure is consistent
- [ ] Import patterns match previous chapters
- [ ] Comment style is uniform

## Common Issues to Flag

### Critical Issues (Must Fix)

**Security Vulnerabilities:**
```python
# ❌ CRITICAL: Hardcoded API key
client = anthropic.Anthropic(api_key="sk-ant-api03-...")

# ❌ CRITICAL: No input validation on user input
def dangerous_eval(user_code: str):
    return eval(user_code)  # Never do this!

# ❌ CRITICAL: Logging sensitive data
logger.info(f"API key: {api_key}")
```

**Broken Code:**
- Code doesn't run
- Import errors
- Missing dependencies
- Syntax errors
- Logic errors that break functionality

**Incorrect Technical Information:**
- Wrong API usage
- Deprecated methods
- Inaccurate explanations
- Misleading comments

### Important Issues (Should Fix)

**Code Quality:**
```python
# ❌ No type hints
def process(msg):
    return client.call(msg)

# ❌ No docstring
def complex_function(a, b, c):
    # Complex logic here
    pass

# ❌ Bare except
try:
    risky_operation()
except:
    pass
```

**Teaching Issues:**
- Code too complex for chapter position
- Missing explanations for non-obvious decisions
- Not building on previous chapters
- Jumping ahead to future concepts

**Consistency Issues:**
- Different terminology than previous chapters
- Different coding patterns
- Conflicting explanations
- Inconsistent error handling

### Nice-to-Have Improvements

**Code improvements:**
- More descriptive variable names
- Additional comments for complex logic
- Better examples in docstrings
- Extracted helper functions

**Documentation improvements:**
- More detailed explanations
- Additional diagrams
- Better analogies
- Extended examples

## Review Output Format

Provide feedback using this structure:

```markdown
## Technical Review: Chapter XX - [Title]

### Overall Assessment
[2-3 sentence summary of chapter quality and readiness]

### Critical Issues ⛔
**Must be fixed before publication**

1. **Security: Hardcoded API Key** (line 45 of example_01.py)
   - Found: `api_key = "sk-ant-api03-..."`
   - Fix: Use dotenv pattern
   - Impact: Security vulnerability

2. **Broken Code: Import Error** (example_02.py)
   - Found: `from chapter_20 import NonexistentClass`
   - Fix: Should import from chapter_14
   - Impact: Code doesn't run

### Important Issues ⚠️
**Should be fixed for quality**

1. **Missing Type Hints** (example_03.py, lines 12-25)
   - Functions `process_message` and `handle_tools` lack type hints
   - Add proper type annotations per code standards

2. **Inconsistent Terminology**
   - Chapter uses "conversation context" but Ch 15 uses "message history"
   - Standardize to "message history" per established convention

3. **Code-Text Mismatch**
   - Text references `example_04.py` but file doesn't exist
   - Either create the file or update the text

### Nice-to-Have Improvements 💡

1. **Add Diagram**: The tool use loop would benefit from a diagram
2. **Expand Example**: example_02 could show error recovery
3. **Better Variable Names**: `x` on line 67 should be `tool_result`

### Code Testing Results

**example_01_simple.py**: ✅ Runs correctly
**example_02_with_history.py**: ✅ Runs correctly
**example_03_complete.py**: ⛔ Import error (see Critical Issues #2)
**exercise_solution.py**: ✅ Runs correctly

### Technical Accuracy

**API Usage**: ✅ Correct
**Architecture Patterns**: ✅ Follows established patterns
**Concepts Explained**: ⚠️ Some inconsistency with Ch 15 terminology
**External References**: ✅ All links valid

### Progressive Complexity

**Prerequisites**: ✅ Correctly assumes Ch 14 knowledge
**New Concepts**: ✅ Introduces one focused concept
**Future References**: ⚠️ Mentions Ch 30 concept prematurely
**Build-up**: ✅ Good progression from simple to complex

### Cross-Chapter Consistency

**Terminology**: ⚠️ See Important Issues #2
**Code Patterns**: ✅ Consistent with previous chapters
**Architecture**: ✅ Correct use of AugmentedLLM
**Style**: ✅ Matches book conventions

### Chapter-Code Alignment

**File References**: ⚠️ Missing example_04.py (see Important Issues #3)
**Code Snippets**: ✅ Match actual files
**Explanations**: ✅ Accurately describe code
**README**: ✅ Complete and accurate

### Recommendations

**Before Publication:**
1. Fix critical security issue in example_01.py
2. Fix import error in example_03.py
3. Resolve terminology inconsistency
4. Fix missing file reference

**Optional Improvements:**
1. Add diagram for tool use loop
2. Improve variable names in example_03.py
3. Consider expanding example_02.py

### Sign-off

- [ ] Ready for publication (after critical issues fixed)
- [ ] Needs another review after changes
- [ ] Significant rework required

**Reviewer Notes:**
[Any additional context, suggestions, or observations]
```

## Special Review Considerations

### For Early Chapters (1-6)
- Check that concepts are introduced gently
- Verify no assumptions about prior knowledge
- Ensure setup instructions are clear
- Confirm security is taught from the start

### For Workflow Chapters (15-25)
- Verify each pattern is distinct and clear
- Check concept/implementation pairs align
- Ensure patterns build on each other
- Confirm examples show clear use cases

### For Agent Chapters (26-33)
- Verify agent components are explained thoroughly
- Check state management is correct
- Ensure planning logic is clear
- Confirm error handling is robust

### For Production Chapters (34-41)
- Verify production practices are current
- Check testing approaches are sound
- Ensure deployment strategies are secure
- Confirm monitoring/observability is practical

### For Capstone Chapters (42-45)
- Verify projects integrate multiple concepts
- Check complexity is appropriate for advanced readers
- Ensure projects are realistic and useful
- Confirm all concepts are correctly applied

## Reference Materials

**Always consult:**
- `ai_agents/skills/PROJECT_INSTRUCTIONS.md` - Writing and coding standards
- `ai_agents/skills/OUTLINE.md` - Book structure and dependencies
- `ai_agents/CLAUDE.md` - Technical specifications

**For architecture verification:**
- `chapter-14-building-the-complete-augmented-llm/code/augmented_llm.py`
- `chapter-33-the-complete-agent-class/code/agent.py`

**For API verification:**
- Anthropic API documentation (current version)
- Python SDK documentation

## Final Checklist

Before approving a chapter:

### Must Have
- [ ] All code runs without errors
- [ ] No security vulnerabilities
- [ ] Technical information is accurate
- [ ] Follows chapter template structure
- [ ] Builds appropriately on previous chapters
- [ ] Terminology is consistent
- [ ] All references are correct

### Should Have
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Good error handling
- [ ] Clear, helpful comments
- [ ] README.md is complete
- [ ] .env.example provided

### Quality Indicators
- [ ] Examples progress from simple to complex
- [ ] Code teaches effectively
- [ ] Explanations are clear and accurate
- [ ] Exercises reinforce learning
- [ ] Chapter achieves stated learning objectives

---

**Remember**: Technical review is about ensuring quality, correctness, and consistency. Be thorough, be specific, and provide actionable feedback. The goal is to help make the chapter the best it can be.
