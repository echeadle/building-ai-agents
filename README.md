# Building AI Agents from Scratch with Python

> A practical guide for intermediate Python programmers who want to understand and build AI agents from first principles—without frameworks.

## About This Book

This book teaches you to build AI agents the way Anthropic recommends: with **simple, composable patterns** instead of complex frameworks. You'll go from your first API call to deploying production-ready agents, understanding every piece of code along the way.

By the end, you'll have built a complete agent framework yourself—approximately 2,000 lines of well-structured Python code that you fully understand and can adapt to any use case.

## What You'll Learn

### Part 1: Foundations (Chapters 1-6)
- Setting up your development environment with `uv`
- Making your first Claude API call
- Managing conversations and system prompts
- Secure secrets management with dotenv

### Part 2: Augmented LLM (Chapters 7-14)
- Understanding tool use and function calling
- Defining and implementing custom tools
- Building the complete `AugmentedLLM` class
- Structured outputs with JSON Schema validation

### Part 3: Workflows (Chapters 15-25)
- **Prompt Chaining**: Sequential, specialized prompts
- **Routing**: Classification and delegation patterns
- **Parallelization**: Concurrent processing with asyncio
- **Orchestrator-Workers**: Task decomposition and synthesis
- **Evaluator-Optimizer**: Iterative refinement loops

### Part 4: True Agents (Chapters 26-33)
- The agentic loop: plan, act, observe, reflect
- State management and persistence
- Planning and reasoning capabilities
- Error handling and recovery strategies
- Building the complete `Agent` class

### Part 5: Production (Chapters 34-41)
- Testing strategies for non-deterministic systems
- Observability, logging, and debugging
- Cost optimization techniques
- Deployment patterns (CLI, API, async workers)
- Security considerations and best practices

### Part 6: Capstone Projects (Chapters 42-45)
- **Research Assistant Agent**: Literature review and synthesis
- **Code Analysis Agent**: Codebase exploration and review
- **Personal Productivity Agent**: Task and calendar management
- **Designing Your Own Agent**: Framework and templates

## Philosophy

> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."
>
> — Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

**This book embodies that principle:**

- ❌ **No LangChain** - Build understanding, not dependency
- ❌ **No LlamaIndex** - Learn the patterns, not the abstractions
- ❌ **No agent frameworks** - Create your own lightweight system
- ✅ **Just Python** - Clean, readable, maintainable code
- ✅ **Simple patterns** - Easy to understand and adapt
- ✅ **Complete examples** - Every file runs standalone

## Prerequisites

- **Python 3.10+** with modern type hints
- **Intermediate Python knowledge**: classes, decorators, async/await
- **Basic API understanding**: HTTP requests, JSON
- **Claude API account** from [Anthropic](https://console.anthropic.com)

No prior experience with AI agents or LLMs required—we build everything from scratch.

## Repository Structure

```
building-ai-agents/
├── README.md                              # This file
├── chapter-01-what-are-ai-agents/
│   ├── chapter.md                         # Chapter content (markdown)
│   └── code/                              # Runnable Python examples
│       ├── README.md                      # Explains each code file
│       └── *.py                           # Complete, runnable scripts
├── chapter-02-environment-setup/
│   ├── chapter.md
│   └── code/
├── ...                                    # Chapters 3-45
├── appendix_A/                            # API Reference
├── appendix_B/                            # Tool Catalog
├── appendix_C/                            # Workflow Patterns
├── appendix_D/                            # Error Handling Strategies
├── appendix_E/                            # Deployment Checklist
└── appendix_F/                            # Security Checklist
```

**Total: 45 chapters + 6 appendices**

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd building-ai-agents
```

### 2. Install `uv` Package Manager

This book uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Get Your Claude API Key

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. Create an API key
3. Keep it secure—never commit it to version control

### 4. Set Up Your Environment

Navigate to any chapter's code directory and create a `.env` file:

```bash
cd chapter-04-your-first-api-call/code/

# Create .env file
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

### 5. Install Dependencies

```bash
# Install required packages for the chapter
uv add anthropic python-dotenv
```

### 6. Run Your First Example

```bash
uv run python first_call.py
```

**Every Python file in this book is complete and runnable.** Copy, paste, execute—no modifications needed.

## Tech Stack

| Tool | Purpose | Why |
|------|---------|-----|
| **Python 3.10+** | Core language | Modern type hints, async support |
| **uv** | Package manager | Fast, reliable, better than pip |
| **Anthropic SDK** | Claude API | Official Python client |
| **python-dotenv** | Secrets | Secure environment variable management |
| **Pydantic** | Validation | Data models for structured outputs |
| **pytest** | Testing | Industry-standard testing framework |
| **FastAPI** | Deployment | Production API deployment |

## Code Standards

All code in this book follows these principles:

✅ **Complete and runnable** - No pseudocode or placeholders
✅ **Type hints** - Clear function signatures
✅ **Docstrings** - Every file explains what it demonstrates
✅ **Security first** - Environment variables, never hardcoded secrets
✅ **Error handling** - Appropriate to the concept being taught
✅ **Progressive complexity** - Each chapter builds on previous patterns

## Key Architecture Patterns

### The AugmentedLLM Class (Chapter 14)

The foundation for all workflow patterns:

```python
from augmented_llm import AugmentedLLM, AugmentedLLMConfig, ToolRegistry

# Register tools
registry = ToolRegistry()
registry.register("get_weather", weather_definition, weather_impl)

# Configure
config = AugmentedLLMConfig(
    model="claude-sonnet-4-20250514",
    system_prompt="You are a helpful assistant."
)

# Create and use
llm = AugmentedLLM(config, registry)
response = llm.generate("What's the weather in San Francisco?")
```

### The Agent Class (Chapter 33)

The complete agent with state, planning, and error recovery:

```python
from agent import Agent, AgentConfig

config = AgentConfig(
    model="claude-sonnet-4-20250514",
    max_iterations=10,
    enable_planning=True,
    enable_human_in_loop=True
)

agent = Agent(config)
result = agent.run("Analyze this codebase and suggest improvements")
```

## Chapter Format

Every chapter follows this consistent 7-part structure:

1. **Introduction** - Hook, context, and preview of concepts
2. **Learning Objectives** - Specific, measurable goals
3. **Main Content** - Detailed explanations with inline code examples
4. **Common Pitfalls** - Mistakes to avoid (2-3 key errors)
5. **Practical Exercise** - Hands-on task with complete solution
6. **Key Takeaways** - Summary bullets for quick reference
7. **What's Next** - Preview of the next chapter

This structure makes chapters easy to reference and review later.

## Working with the Code

### Run Any Example

```bash
cd chapter-XX-title/code/
uv run python example_name.py
```

### Run Tests (Production Chapters)

```bash
cd chapter-35-testing-implementation/code/
pytest test_tools.py -v
pytest test_suite.py -v
```

### Add Dependencies

```bash
# Add a new package to current chapter
uv add package-name

# Install from requirements if present
uv sync
```

## Learning Path

### For Beginners to AI Agents

**Week 1-2**: Chapters 1-14
- Focus on fundamentals
- Get comfortable with tool use
- Build the `AugmentedLLM` class

**Week 3-4**: Chapters 15-25
- Learn all five workflow patterns
- Practice each pattern independently
- Combine patterns for complex tasks

**Week 5-6**: Chapters 26-33
- Understand true agentic behavior
- Build the complete `Agent` class
- Implement state management

**Week 7-8**: Chapters 34-45
- Production readiness
- Testing and deployment
- Complete capstone projects

### For Experienced Developers

**Fast track** (1-2 weeks):
- Skim Chapters 1-6 (review if needed)
- Deep dive Chapters 7-14 (tool use fundamentals)
- Focus on Chapters 26-33 (agent architecture)
- Study Chapters 34-41 (production patterns)
- Build one capstone project (Chapters 42-44)

## Testing the Code

Each chapter's examples can be tested independently:

```bash
# Navigate to chapter
cd chapter-XX-title/code/

# Check what's available
ls *.py

# Run any example
uv run python example.py
```

For comprehensive testing across the entire book, create a testing plan:

1. Start with Chapters 1-3 (setup validation)
2. Test each workflow pattern (Chapters 15-25)
3. Validate agent behavior (Chapters 26-33)
4. Run production examples (Chapters 34-41)
5. Execute capstone projects (Chapters 42-44)

## Common Commands Reference

```bash
# Install dependencies
uv add anthropic python-dotenv

# Run any Python file
uv run python script.py

# Run tests
pytest -v

# Format code (if modifying examples)
uv add --dev black
uv run black *.py

# Check types
uv add --dev mypy
uv run mypy script.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'anthropic'"
```bash
uv add anthropic python-dotenv
```

### "ValueError: ANTHROPIC_API_KEY not found"
```bash
# Create .env file in chapter's code/ directory
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### "API key is invalid"
- Verify your key at [console.anthropic.com](https://console.anthropic.com)
- Check for leading/trailing spaces in `.env`
- Ensure no quotes around the API key value

### Import errors from earlier chapters
```bash
# Some chapters import from earlier ones
# Make sure you're in the repository root when running
cd /path/to/building-ai-agents
uv run python chapter-XX-title/code/example.py
```

## Resources

- **Claude API Docs**: [docs.anthropic.com](https://docs.anthropic.com)
- **Building Effective Agents**: [Anthropic's Guide](https://www.anthropic.com/engineering/building-effective-agents)
- **uv Documentation**: [astral.sh/uv](https://astral.sh/uv)
- **Python Type Hints**: [PEP 484](https://peps.python.org/pep-0484/)

## Contributing

This is a published book repository. Code examples are maintained to match the book content.

**Found an error?**
1. Open an issue with the chapter number and file name
2. Describe expected vs actual behavior
3. Include any error messages

**Want to suggest improvements?**
- Clarity issues in explanations
- Code that doesn't run as expected
- Missing edge cases in exercises

## License

[License information to be determined]

## Acknowledgments

This book is inspired by Anthropic's research on building effective agents and their commitment to simple, composable patterns over complex frameworks.

## About the Author

[Author information to be added]

---

**Ready to start?** Head to [Chapter 1: What Are AI Agents](chapter-01-what-are-ai-agents/chapter.md) and begin your journey into building AI agents from scratch.

**Questions?** Each chapter includes exercises and solutions. Work through them to solidify your understanding before moving forward.

**Stuck?** Check the appendices for quick references on common patterns, tools, and deployment strategies.
