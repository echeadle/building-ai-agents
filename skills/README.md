# Building AI Agents from Scratch with Python

> A practical guide for intermediate Python programmers who want to understand and build AI agents from first principles—without frameworks.

## About This Book

This book teaches you to build AI agents the way Anthropic recommends: with simple, composable patterns instead of complex frameworks. You'll go from your first API call to deploying production-ready agents, understanding every piece of code along the way.

### What You'll Learn

- **Foundations**: Environment setup, API calls, conversations, and system prompts
- **The Augmented LLM**: Tool use, function calling, and structured outputs
- **Workflow Patterns**: Chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer
- **True Agents**: The agentic loop, state management, planning, and error handling
- **Production**: Testing, observability, optimization, deployment, and security
- **Projects**: Build real agents (research assistant, code analyzer, productivity agent)

### Philosophy

Every chapter:
- Focuses on **one concept** — easy to reference later
- Provides **complete, runnable code** — copy, paste, run
- Includes **detailed explanations** — understand the "why"
- Uses **python-dotenv** from day one — never commit secrets

## Target Audience

- Intermediate Python programmers
- Comfortable with classes and OOP
- No prior AI/ML experience required
- Want to learn by building, not memorizing

## Structure

```
building-ai-agents-from-scratch-with-python/
├── README.md                           # This file
├── OUTLINE.md                          # Complete book outline
├── chapter-01-what-are-ai-agents/
│   ├── chapter.md                      # Chapter content
│   └── code/                           # Runnable examples
├── chapter-02-environment-setup/
│   ├── chapter.md
│   └── code/
├── ...
└── appendix-a-python-refresher/
    └── appendix-a.md
```

## Book at a Glance

| Part | Chapters | Focus |
|------|----------|-------|
| 1: Foundations | 1-6 | Setup, API calls, conversations |
| 2: Augmented LLM | 7-14 | Tools, function calling, building blocks |
| 3: Workflows | 15-25 | Five agentic patterns |
| 4: True Agents | 26-33 | Autonomous agents |
| 5: Production | 34-41 | Testing, deployment, security |
| 6: Projects | 42-45 | Capstone applications |
| Appendices | A-G | References and resources |

**Total: 45 chapters + 7 appendices**

## Tech Stack

- **Python 3.10+** — Modern Python features
- **uv** — Fast, reliable package manager
- **python-dotenv** — Secure secrets management
- **Anthropic SDK** — Claude API access
- **No frameworks** — Just Python and patterns

## Getting Started

See [Chapter 2: Setting Up Your Development Environment](chapter-02-environment-setup/chapter.md) to begin.

## Code Repository

All code examples are available in each chapter's `code/` directory. A complete working repository structure is documented in the outline.

## Based On

This book is inspired by and builds upon Anthropic's excellent article [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), which emphasizes:

> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."

---

*License: [To be determined]*
