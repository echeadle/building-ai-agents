# Building AI Agents from Scratch with Python

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a technical book titled **"Building AI Agents from Scratch with Python"** that teaches intermediate Python programmers how to build AI agents without frameworks, using simple, composable patterns.

The book follows Anthropic's "Building Effective Agents" philosophy:
> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."

## Quick Links

- **Project Instructions**: `skills/PROJECT_INSTRUCTIONS.md` - Detailed writing and coding standards
- **Book Outline**: `skills/OUTLINE.md` - Complete table of contents and chapter structure
- **Skills**: `.claude/skills/` - Specialized skills for writing, coding, reviewing, and consistency
- **Subagents**: `.claude/agents/` - Subagents for complex tasks like chapter authoring and code architecture

## Repository Structure

```
advanced-agents-no-frameworks/
├── CLAUDE.md                           # This file - comprehensive project guide
├── .claude/                            # Claude Code configuration
│   ├── skills/                         # Specialized skills for book creation
│   │   ├── chapter-writer.md          # Writing new chapters
│   │   ├── code-developer.md          # Creating code examples
│   │   ├── technical-reviewer.md      # Reviewing technical content
│   │   └── consistency-checker.md     # Ensuring book-wide consistency
│   └── agents/                         # Subagents for specialized tasks
│       ├── chapter-author.json        # Chapter drafting and writing
│       ├── code-architect.json        # Designing code examples
│       ├── technical-editor.json      # Technical review and editing
│       └── integration-specialist.json # Cross-chapter consistency
├── chapter-XX-title/                   # Individual chapters (45 total)
│   ├── chapter.md                      # Chapter content (markdown)
│   └── code/                           # Code examples
│       ├── README.md                   # Explains each code file
│       ├── example_01.py               # Numbered examples
│       ├── example_02.py
│       └── exercise*.py                # Exercise solutions
├── appendix_A/ through appendix_F/     # Reference material
└── skills/                             # Book-level documentation
    ├── PROJECT_INSTRUCTIONS.md         # Comprehensive writing guidelines
    └── OUTLINE.md                      # Complete book structure
```

## Book Structure

**45 Chapters in 6 Parts:**

- **Part 1 (Ch 1-6)**: Foundations - Setup, API calls, conversations, system prompts
- **Part 2 (Ch 7-14)**: Augmented LLM - Tools, function calling, structured outputs
- **Part 3 (Ch 15-25)**: Workflows - Chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer
- **Part 4 (Ch 26-33)**: True Agents - Agentic loop, state management, planning, error handling
- **Part 5 (Ch 34-41)**: Production - Testing, observability, deployment, security
- **Part 6 (Ch 42-45)**: Projects - Capstone applications

## Tech Stack & Dependencies

- **Python 3.10+** (modern type hints required)
- **Package Manager**: `uv` (NOT pip or poetry)
- **Key Dependencies**:
  - `anthropic` SDK - Claude API access
  - `python-dotenv` - Secure secrets management
  - `pydantic` - Data validation (in structured output chapters)
  - `pytest` - Testing (in testing chapters)
  - `fastapi`, `uvicorn` - Deployment (in deployment chapters)

**NO FRAMEWORKS** - This is fundamental to the book's philosophy. No LangChain, no LlamaIndex, no agent frameworks.

## Environment Setup

All Python code uses the standard pattern:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
```

**Required `.env` file** (not in repo, user must create):
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Running Code Examples

All Python files are **complete and runnable**. Each file can be executed standalone:

```bash
# Navigate to a chapter's code directory
cd chapter-XX-title/code/

# Run any example
python example_01.py

# Or use uv
uv run example_01.py
```

**Testing** (Chapter 35):
```bash
cd chapter-35-testing-implementation/code/
pytest test_tools.py -v
pytest test_suite.py -v
```

## Core Architecture Patterns

### 1. AugmentedLLM Class (Chapter 14)

The foundational building block located in `chapter-14-building-the-complete-augmented-llm/code/augmented_llm.py`:

- Wraps the Anthropic API
- Manages system prompts, tools, and conversation history
- Handles the tool-use loop automatically
- Supports structured output with JSON Schema validation

**Key classes:**
- `AugmentedLLMConfig`: Immutable configuration dataclass
- `ToolRegistry`: Maps tool names to definitions and implementations
- `AugmentedLLM`: Main class that orchestrates everything

**Used in chapters 14-25 for workflow patterns and serves as base class for the Agent class.**

### 2. Agent Class (Chapter 33)

The complete agent implementation in `chapter-33-the-complete-agent-class/code/agent.py`:

- Built on top of AugmentedLLM
- Adds state management, guardrails, error recovery
- Implements the full agentic loop with planning and reasoning
- Production-ready with human-in-the-loop support

**Related modules:**
- `config.py`: Agent configuration
- `state.py`: State management
- `tools.py`: Tool registry patterns
- `guardrails.py`: Safety constraints
- `errors.py`: Error handling utilities

**Used in chapters 34-41 for production topics.**

### 3. Workflow Patterns

**Five core patterns covered:**
1. **Prompt Chaining** (Ch 16-17): Sequential, specialized prompts
2. **Routing** (Ch 18-19): Classification and delegation
3. **Parallelization** (Ch 20-21): Concurrent processing with asyncio
4. **Orchestrator-Workers** (Ch 22-23): Task decomposition and synthesis
5. **Evaluator-Optimizer** (Ch 24-25): Iterative refinement

## Code Standards

### Required in Every Python File

1. **Docstring** at the top explaining what the file demonstrates
2. **Chapter reference** in the docstring
3. **Complete imports** - nothing assumed
4. **Type hints** on all function signatures
5. **Error handling** appropriate to the concept being taught
6. **Runnable example** in `if __name__ == "__main__":` block
7. **dotenv pattern** for any API keys

### Model Reference

Default model in examples: `claude-sonnet-4-20250514`

### Code Style

- `snake_case` for functions and variables
- `PascalCase` for classes
- Max line length: 88 characters (Black formatter)
- f-strings for formatting
- Explicit over implicit

## Chapter Structure

Every chapter follows this 7-part template:
1. **Introduction** - Hook, context, preview
2. **Learning Objectives** - Specific, measurable goals
3. **Main Content** - Sections with code and explanation
4. **Common Pitfalls** - 2-3 mistakes to avoid
5. **Practical Exercise** - Hands-on task with solution
6. **Key Takeaways** - Summary bullets
7. **What's Next** - Preview of next chapter

## Progressive Complexity

Code dependencies build progressively:
- **Ch 1-3**: No code dependencies, setup only
- **Ch 4-6**: Basic API call patterns established
- **Ch 7-13**: Tool use patterns
- **Ch 14+**: Can use `AugmentedLLM` class
- **Ch 15-25**: Can use workflow patterns
- **Ch 26-33**: Can use `Agent` class
- **Ch 34+**: Can use testing/production patterns

When working on later chapters, you can import and use classes from earlier chapters.

## Writing Style

- **Friendly and practical** - like a knowledgeable colleague
- **Direct** - no fluff or marketing speak
- **Use "you" and "we"** to engage readers
- **Explain the "why"** not just the "how"
- **One focused concept per chapter**
- **Icons in prose** (not in code):
  - 💡 Tips and best practices
  - ⚠️ Warnings and pitfalls
  - 🔧 Practical exercises
  - 📚 Further reading

## Key Principles

1. **No frameworks** - Teaching fundamentals, not abstractions
2. **Complete, runnable code** - Copy, paste, run
3. **Security first** - Always use dotenv, never hardcode secrets
4. **Progressive complexity** - Each chapter builds on the last
5. **Practical focus** - Theory supports practice
6. **One concept per chapter** - Focused and referenceable

## When Modifying Code

✅ **Do:**
- Ensure all code remains complete and runnable
- Maintain the dotenv pattern for API keys
- Keep type hints and docstrings
- Test that code actually works
- Preserve the simple, composable patterns
- Stay consistent with established patterns from earlier chapters

❌ **Don't:**
- Add framework dependencies (LangChain, etc.)
- Remove error handling or security measures
- Make code snippets that won't run standalone
- Hardcode API keys or secrets
- Add complexity beyond what the chapter teaches
- Break backward compatibility with earlier chapter patterns

## Available Skills & Subagents

### Skills (Always Available)
Use these skills for specific tasks:

1. **chapter-writer** - Write new chapter content following book standards
2. **code-developer** - Create code examples that teach effectively
3. **technical-reviewer** - Review chapters for accuracy and clarity
4. **consistency-checker** - Ensure terminology and patterns are consistent

### Subagents (Delegate Complex Tasks)
Invoke these for specialized work:

1. **chapter-author** - Draft complete chapters with structure and flow
2. **code-architect** - Design and implement code example progressions
3. **technical-editor** - Deep technical review and editing
4. **integration-specialist** - Ensure chapters work together cohesively

### When to Use Skills vs Subagents

**Skills** are always active and provide guidance:
- Quick reviews and checks
- Standard operations (writing, coding, reviewing)
- Ensuring consistency
- Following established patterns

**Subagents** handle complex, multi-step tasks:
- Drafting entire chapters
- Designing multi-file code architectures
- Deep technical editing
- Cross-chapter integration work

### Example Usage
```
"Use the chapter-writer skill to help me draft the introduction for Chapter 20"
"Invoke the code-architect subagent to design the code examples for the parallelization chapter"
"Use the consistency-checker skill to verify terminology across Part 3"
"Invoke the integration-specialist subagent to check consistency across Chapters 15-25"
```

## Common Workflows

### Writing a New Chapter
1. Use **chapter-author** subagent to draft complete chapter
2. Use **code-architect** subagent for code examples
3. Use **technical-reviewer** skill for review
4. Use **consistency-checker** skill for final check

### Updating Code Examples
1. Read existing code in chapter
2. Use **code-developer** skill for modifications
3. Test code runs correctly
4. Update chapter text if needed
5. Use **technical-reviewer** skill

### Technical Review
1. Use **technical-editor** subagent for comprehensive review
2. Check against `skills/PROJECT_INSTRUCTIONS.md` standards
3. Verify code follows patterns from earlier chapters
4. Ensure progressive complexity

### Consistency Check
1. Use **consistency-checker** skill across multiple chapters
2. Verify terminology is standardized
3. Check that examples build on each other
4. Ensure coding patterns are consistent

## Reference Files

**For detailed guidance, consult:**
- `skills/PROJECT_INSTRUCTIONS.md` - Comprehensive writing and coding standards
- `skills/OUTLINE.md` - Complete book outline with all chapters
- `README.md` - Book overview and structure (if exists)

**For architecture patterns, see:**
- `chapter-14-building-the-complete-augmented-llm/code/augmented_llm.py`
- `chapter-33-the-complete-agent-class/code/agent.py`

**For specialized guidance, see:**
- `.claude/skills/chapter-writer.md` - Chapter writing standards
- `.claude/skills/code-developer.md` - Code example standards
- `.claude/skills/technical-reviewer.md` - Review standards
- `.claude/skills/consistency-checker.md` - Consistency standards

## Git Workflow

The repository uses standard git practices. The `.gitignore` is comprehensive and excludes:
- `.env` files (secrets)
- `__pycache__/` and `*.pyc`
- Virtual environments (`.venv/`, `env/`)
- IDE configs (`.idea/`, `.vscode/`)

Always verify secrets are not committed before creating commits.

## Current Status

This is an advanced, comprehensive technical book with 45 chapters covering:
- Foundations through production deployment
- Simple patterns to complex agents
- Theory backed by working code
- Progressive skill building

**Focus Areas:**
- Maintaining code quality across all examples
- Ensuring progressive complexity
- Teaching understanding, not just functionality
- Building a complete agent framework by the end

## Getting Started

1. **Read the outline**: Check `skills/OUTLINE.md` for complete book structure
2. **Understand standards**: Read `skills/PROJECT_INSTRUCTIONS.md` for writing and coding guidelines
3. **Review skills**: Explore `.claude/skills/` for specialized guidance
4. **Use the tools**: Leverage skills and subagents for your work
5. **Maintain quality**: Follow all coding and writing standards

---

**Remember**: This book teaches readers to build their own framework. Every chapter should illuminate concepts clearly and build toward that goal.
