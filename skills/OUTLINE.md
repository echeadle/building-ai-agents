# Building AI Agents from Scratch with Python - Outline

## Target Audience

Intermediate Python programmers who:

-   Are comfortable with Python syntax, functions, and classes
-   Want to understand AI agents deeply, not just use frameworks
-   Prefer learning through complete, runnable code examples
-   Want to build production-ready agents from first principles

## Prerequisites

-   Python 3.10+ proficiency
-   Understanding of classes and object-oriented programming
-   Basic familiarity with HTTP/REST APIs (helpful but not required)
-   Command line comfort
-   No prior AI/ML experience required

## Learning Outcomes

After completing this book, readers will be able to:

1. Build AI agents from scratch without relying on frameworks
2. Implement all major agentic patterns (chaining, routing, parallelization, etc.)
3. Design effective tools and function interfaces for agents
4. Handle errors, manage state, and build reliable agent systems
5. Test, debug, and deploy agents to production
6. Make informed decisions about when to use agents vs simpler solutions

## Philosophy

This book follows the principles outlined in Anthropic's "Building Effective Agents":

> "The most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns."

Every chapter provides:

-   **One focused concept** - easy to reference later
-   **Complete, runnable code** - copy, paste, run
-   **Detailed explanations** - understand the "why"
-   **Practical exercises** - reinforce learning

---

## Book Structure

### Part 1: Foundations (Chapters 1-6)

Getting your environment ready and making your first API calls.

---

#### Chapter 1: What Are AI Agents?

**Focus:** Understanding the landscape before building

-   The spectrum from simple prompts to autonomous agents
-   Workflows vs Agents: A critical distinction
-   When agents are overkill (and when they shine)
-   The building blocks we'll construct throughout this book
-   Real-world agent examples and use cases

**Key Takeaway:** Agents are LLMs that dynamically direct their own processes and tool usage.

---

#### Chapter 2: Setting Up Your Development Environment

**Focus:** One-time setup that all future code depends on

-   Installing Python 3.10+ and verifying your version
-   Installing uv (the modern Python package manager)
-   Why uv over pip/poetry (speed, reliability, lockfiles)
-   Creating your first project with `uv init`
-   Understanding pyproject.toml

**Code Deliverable:** A configured project directory ready for development

**Key Takeaway:** A solid foundation prevents countless headaches later.

---

#### Chapter 3: Managing Secrets with python-dotenv

**Focus:** Never commit API keys to version control

-   Why secrets management matters (horror stories)
-   Installing python-dotenv with uv
-   Creating and structuring your .env file
-   Loading environment variables in Python
-   Setting up .gitignore properly
-   Validating that secrets are loaded correctly

**Code Deliverable:** A reusable secrets loading pattern used throughout the book

**Key Takeaway:** Security is not optional—build good habits from day one.

---

#### Chapter 4: Your First API Call to Claude

**Focus:** The minimal code to talk to an LLM

-   Installing the Anthropic SDK
-   Anatomy of an API request (model, max_tokens, messages)
-   Making a simple completion request
-   Understanding the response structure
-   Handling basic errors (rate limits, auth failures)
-   Cost awareness: tokens and pricing

**Code Deliverable:** A working script that sends a prompt and prints the response

**Key Takeaway:** At its core, an LLM API call is just: send messages, get text back.

---

#### Chapter 5: Understanding Messages and Conversations

**Focus:** How multi-turn conversations work

-   The messages array structure
-   Roles: system, user, assistant
-   Building conversation history
-   Why conversation history matters for agents
-   Token limits and conversation truncation strategies
-   The stateless nature of API calls

**Code Deliverable:** A simple chat loop that maintains conversation history

**Key Takeaway:** You manage all state—the API remembers nothing between calls.

---

#### Chapter 6: System Prompts and Persona Design

**Focus:** Shaping agent behavior through instructions

-   What system prompts are and why they matter
-   Crafting effective system prompts
-   Giving your agent a persona and purpose
-   Setting boundaries and constraints
-   System prompt best practices and anti-patterns
-   Testing system prompt effectiveness

**Code Deliverable:** A configurable agent base with system prompt loading

**Key Takeaway:** The system prompt is your agent's constitution—design it carefully.

---

### Part 2: The Augmented LLM (Chapters 7-14)

Building the fundamental building block: an LLM enhanced with tools.

---

#### Chapter 7: Introduction to Tool Use

**Focus:** Why agents need tools and how they work

-   The limitations of a "naked" LLM
-   What tool use enables (actions in the real world)
-   The tool use cycle: define → call → respond
-   How Claude decides when to use tools
-   Overview of what we'll build in this section

**Key Takeaway:** Tools transform LLMs from text generators into agents that can act.

---

#### Chapter 8: Defining Your First Tool

**Focus:** The anatomy of a tool definition

-   Tool definition structure (name, description, parameters)
-   JSON Schema basics for parameter definitions
-   Writing clear tool descriptions (LLMs read these!)
-   Required vs optional parameters
-   Parameter types: string, number, boolean, array, object

**Code Deliverable:** A simple calculator tool definition

**Key Takeaway:** Tool descriptions are prompts—write them for the LLM, not just humans.

---

#### Chapter 9: Handling Tool Calls

**Focus:** Processing what the LLM asks you to do

-   Detecting tool use in API responses
-   Parsing tool call arguments
-   Executing the requested function
-   Returning results to the LLM
-   The complete tool use loop

**Code Deliverable:** A working calculator that Claude can use

**Key Takeaway:** You are the bridge between Claude's intent and real-world execution.

---

#### Chapter 10: Building a Weather Tool

**Focus:** A practical tool that fetches real data

-   Designing the tool interface
-   Integrating with a weather API (free tier)
-   Handling API errors gracefully
-   Formatting results for the LLM
-   Testing tool reliability

**Code Deliverable:** A weather tool Claude can use to answer weather questions

**Key Takeaway:** Real tools need real error handling.

---

#### Chapter 11: Multi-Tool Agents

**Focus:** Giving your agent a toolkit

-   Providing multiple tools in one request
-   How Claude chooses between tools
-   Tools that complement each other
-   Organizing your tool definitions
-   The tools registry pattern

**Code Deliverable:** An agent with calculator, weather, and datetime tools

**Key Takeaway:** More tools = more capability, but also more complexity to manage.

---

#### Chapter 12: Sequential Tool Calls

**Focus:** When one tool call isn't enough

-   Understanding multi-turn tool use
-   The agentic loop: call → respond → call again
-   When to stop the loop
-   Preventing infinite loops
-   Tracking tool call history

**Code Deliverable:** An agent that can chain multiple tool calls to answer complex questions

**Key Takeaway:** Real agent tasks often require multiple tool calls in sequence.

---

#### Chapter 13: Structured Outputs and Response Parsing

**Focus:** Getting predictable, parseable responses

-   Requesting JSON output
-   Defining response schemas
-   Validating LLM responses
-   Handling malformed responses gracefully
-   When to use structured vs freeform output

**Code Deliverable:** A response parsing utility with validation

**Key Takeaway:** Structured output makes agents programmable; validation makes them reliable.

---

#### Chapter 14: Building the Complete Augmented LLM

**Focus:** Assembling the building block

-   The AugmentedLLM class architecture
-   Integrating tools, system prompts, and structured output
-   A clean, reusable interface
-   Configuration and customization
-   Testing your building block

**Code Deliverable:** A complete `AugmentedLLM` class that serves as the foundation for all future work

**Key Takeaway:** This building block is the foundation—everything else builds on it.

---

### Part 3: Workflow Patterns (Chapters 15-25)

Implementing the five core agentic workflow patterns.

---

#### Chapter 15: Introduction to Agentic Workflows

**Focus:** Understanding workflow patterns and when to use them

-   Workflows vs Agents recap
-   The five workflow patterns we'll implement
-   Choosing the right pattern for your use case
-   Combining patterns effectively
-   When simple prompts are enough

**Key Takeaway:** Match the pattern to the problem—don't over-engineer.

---

#### Chapter 16: Prompt Chaining - Concept and Design

**Focus:** Breaking complex tasks into steps

-   What prompt chaining is and when to use it
-   Designing a chain: identifying subtasks
-   Quality gates between steps
-   Trading latency for accuracy
-   Prompt chaining architecture diagram

**Key Takeaway:** Chaining makes hard tasks easy by making each step simple.

---

#### Chapter 17: Prompt Chaining - Implementation

**Focus:** Building a working prompt chain

-   Implementing a content generation → translation chain
-   Adding quality gates (validation between steps)
-   Error handling in chains
-   Passing context between steps
-   The Chain class pattern

**Code Deliverable:** A working prompt chain with quality gates

**Key Takeaway:** Each link in the chain should do one thing well.

---

#### Chapter 18: Routing - Concept and Design

**Focus:** Directing inputs to specialized handlers

-   What routing is and when to use it
-   Classification strategies (LLM vs rule-based)
-   Designing route handlers
-   Default/fallback routes
-   Routing architecture diagram

**Key Takeaway:** Routing lets you optimize for specific use cases without compromising others.

---

#### Chapter 19: Routing - Implementation

**Focus:** Building a working router

-   Implementing an LLM-based classifier
-   Creating specialized route handlers
-   Building a customer service router example
-   Testing classification accuracy
-   The Router class pattern

**Code Deliverable:** A customer service query router with specialized handlers

**Key Takeaway:** Good routing depends on good classification—test it thoroughly.

---

#### Chapter 20: Parallelization - Concept and Design

**Focus:** Running LLM calls simultaneously

-   Sectioning: independent subtasks in parallel
-   Voting: multiple perspectives on the same task
-   When parallelization helps (and when it doesn't)
-   Aggregating parallel results
-   Parallelization architecture diagrams

**Key Takeaway:** Parallelization trades cost for speed and/or confidence.

---

#### Chapter 21: Parallelization - Implementation

**Focus:** Building parallel workflows

-   Python asyncio basics for parallel calls
-   Implementing sectioning (parallel subtasks)
-   Implementing voting (parallel perspectives)
-   Aggregation strategies (majority vote, consensus, merge)
-   Error handling in parallel workflows

**Code Deliverable:** A code review system using voting for vulnerability detection

**Key Takeaway:** Parallel execution requires careful result aggregation.

---

#### Chapter 22: Orchestrator-Workers - Concept and Design

**Focus:** Dynamic task decomposition

-   What orchestrator-workers is and when to use it
-   The orchestrator's role (planning, delegating, synthesizing)
-   Worker design principles
-   Dynamic vs static task breakdown
-   Orchestrator-workers architecture diagram

**Key Takeaway:** Use orchestrator-workers when you can't predict subtasks in advance.

---

#### Chapter 23: Orchestrator-Workers - Implementation

**Focus:** Building a dynamic orchestrator

-   Implementing the orchestrator LLM
-   Creating flexible worker tasks
-   Task delegation and result collection
-   Synthesizing worker outputs
-   The Orchestrator class pattern

**Code Deliverable:** A research orchestrator that delegates to specialized workers

**Key Takeaway:** The orchestrator prompt is critical—it must understand how to break down tasks.

---

#### Chapter 24: Evaluator-Optimizer - Concept and Design

**Focus:** Iterative refinement through feedback

-   What evaluator-optimizer is and when to use it
-   The generator-evaluator loop
-   Designing effective evaluation criteria
-   Knowing when to stop iterating
-   Evaluator-optimizer architecture diagram

**Key Takeaway:** If humans can give useful feedback, LLMs can too.

---

#### Chapter 25: Evaluator-Optimizer - Implementation

**Focus:** Building a refinement loop

-   Implementing the generator LLM
-   Implementing the evaluator LLM
-   The feedback loop mechanism
-   Convergence detection
-   Maximum iteration safeguards

**Code Deliverable:** A writing assistant that iteratively improves drafts

**Key Takeaway:** Clear evaluation criteria are essential—vague feedback produces vague improvements.

---

### Part 4: Building True Agents (Chapters 26-33)

Moving from workflows to autonomous agents.

---

#### Chapter 26: From Workflows to Agents

**Focus:** Understanding the leap to autonomy

-   The key difference: LLM-directed control flow
-   When workflows aren't enough
-   The agent loop: perceive → think → act → repeat
-   Trust and autonomy considerations
-   Agent architecture overview

**Key Takeaway:** Agents decide their own next steps—this is powerful and risky.

---

#### Chapter 27: The Agentic Loop

**Focus:** The core execution cycle

-   Implementing the basic agent loop
-   Perceiving: gathering input and tool results
-   Thinking: letting the LLM reason and plan
-   Acting: executing chosen actions
-   Termination conditions

**Code Deliverable:** A minimal agentic loop implementation

**Key Takeaway:** The loop is simple; the complexity is in the details.

---

#### Chapter 28: State Management

**Focus:** What agents need to remember

-   Conversation history as state
-   Working memory: current task context
-   Long-term memory patterns (basic)
-   State persistence between sessions
-   State serialization and loading

**Code Deliverable:** A stateful agent with persistent memory

**Key Takeaway:** Agents without memory forget their purpose mid-task.

---

#### Chapter 29: Planning and Reasoning

**Focus:** Helping agents think before they act

-   The value of explicit planning
-   Plan-then-execute patterns
-   Step-by-step reasoning (chain-of-thought)
-   Plan revision and adaptation
-   Showing planning steps for transparency

**Code Deliverable:** An agent that plans before executing multi-step tasks

**Key Takeaway:** Planning improves reliability but adds latency—find the right balance.

---

#### Chapter 30: Error Handling and Recovery

**Focus:** When things go wrong (they will)

-   Types of agent errors (tool failures, bad outputs, loops)
-   Graceful degradation strategies
-   Retry logic with backoff
-   Fallback behaviors
-   Error reporting and logging
-   Self-correction patterns

**Code Deliverable:** Robust error handling utilities for agents

**Key Takeaway:** Errors are inevitable; how you handle them defines reliability.

---

#### Chapter 31: Human-in-the-Loop

**Focus:** Keeping humans in control

-   Why human oversight matters
-   Checkpoint patterns (pause for approval)
-   Confirmation for high-stakes actions
-   Human feedback integration
-   Escalation paths

**Code Deliverable:** An agent with approval gates for sensitive operations

**Key Takeaway:** Autonomy should be earned through demonstrated reliability.

---

#### Chapter 32: Guardrails and Safety

**Focus:** Preventing agents from going off the rails

-   Input validation and sanitization
-   Output filtering and verification
-   Action constraints and allowlists
-   Rate limiting and resource bounds
-   Sandboxing dangerous operations

**Code Deliverable:** A guardrails module for agent safety

**Key Takeaway:** Guardrails are not optional—build them in from the start.

---

#### Chapter 33: The Complete Agent Class

**Focus:** Assembling everything into a production-ready agent

-   The Agent class architecture
-   Integrating all components
-   Configuration and customization
-   Clean interfaces for different use cases
-   Documentation and usage examples

**Code Deliverable:** A complete, well-documented Agent class

**Key Takeaway:** Good architecture makes agents maintainable and extensible.

---

### Part 5: Production Readiness (Chapters 34-41)

Taking agents from prototype to production.

---

#### Chapter 34: Testing AI Agents - Philosophy

**Focus:** How testing agents differs from testing regular code

-   The challenge: non-deterministic outputs
-   Test types: unit, integration, end-to-end
-   What to test: behavior, not exact outputs
-   Building test datasets
-   Evaluation metrics for agents

**Key Takeaway:** You can't unit test randomness, but you can test behavior patterns.

---

#### Chapter 35: Testing AI Agents - Implementation

**Focus:** Writing practical agent tests

-   Testing tools in isolation
-   Testing the agentic loop
-   Mock LLM responses for deterministic tests
-   Property-based testing for agents
-   Continuous evaluation in CI/CD

**Code Deliverable:** A test suite for the Agent class

**Key Takeaway:** Test infrastructure is as important as the agent itself.

---

#### Chapter 36: Observability and Logging

**Focus:** Seeing what your agent is doing

-   Structured logging for agents
-   Tracing tool calls and decisions
-   Performance metrics collection
-   Log levels and filtering
-   Log aggregation patterns

**Code Deliverable:** A logging module for agent observability

**Key Takeaway:** You can't debug what you can't see.

---

#### Chapter 37: Debugging Agents

**Focus:** Finding and fixing agent problems

-   Common agent failure modes
-   Debugging conversation flow
-   Debugging tool selection
-   Debugging infinite loops
-   Replay and reproduction techniques

**Code Deliverable:** Debugging utilities and techniques

**Key Takeaway:** Agent bugs are often prompt bugs—check your instructions first.

---

#### Chapter 38: Cost Optimization

**Focus:** Managing API costs at scale

-   Understanding token costs
-   Prompt optimization techniques
-   Response length management
-   Caching strategies
-   Model selection (when to use cheaper models)
-   Cost monitoring and alerts

**Code Deliverable:** A cost tracking and optimization module

**Key Takeaway:** Unmonitored agents can generate surprising bills—always track costs.

---

#### Chapter 39: Latency Optimization

**Focus:** Making agents faster

-   Identifying latency bottlenecks
-   Streaming responses
-   Parallel tool execution
-   Caching and precomputation
-   Choosing faster models for simple tasks
-   Response time budgets

**Code Deliverable:** Latency optimization utilities

**Key Takeaway:** User experience depends on speed—optimize the critical path.

---

#### Chapter 40: Deployment Patterns

**Focus:** Running agents in production

-   Agents as REST APIs (FastAPI)
-   Agents as background workers
-   Containerization with Docker
-   Environment configuration
-   Health checks and monitoring
-   Scaling considerations

**Code Deliverable:** A FastAPI wrapper for serving agents

**Key Takeaway:** Production agents need production infrastructure.

---

#### Chapter 41: Security Considerations

**Focus:** Keeping your agents secure

-   API key management in production
-   Input injection attacks
-   Output security (preventing data leaks)
-   Rate limiting and abuse prevention
-   Audit logging
-   Principle of least privilege for tools

**Code Deliverable:** Security hardening checklist and utilities

**Key Takeaway:** Agents are attack surfaces—secure them accordingly.

---

### Part 6: Capstone Projects (Chapters 42-45)

Applying everything to build real agents.

---

#### Chapter 42: Project - Research Assistant Agent

**Focus:** An agent that searches, reads, and summarizes

-   Project requirements and design
-   Tool suite: web search, page reader, note-taking
-   Multi-step research workflow
-   Synthesis and report generation
-   Complete implementation walkthrough

**Code Deliverable:** A fully functional research assistant agent

---

#### Chapter 43: Project - Code Analysis Agent

**Focus:** An agent that understands and analyzes code

-   Project requirements and design
-   Tool suite: file reader, code parser, pattern matcher
-   Analysis strategies
-   Report generation
-   Complete implementation walkthrough

**Code Deliverable:** A fully functional code analysis agent

---

#### Chapter 44: Project - Personal Productivity Agent

**Focus:** An agent that helps manage tasks and information

-   Project requirements and design
-   Tool suite: calendar, notes, reminders
-   Context-aware responses
-   Personalization
-   Complete implementation walkthrough

**Code Deliverable:** A fully functional productivity agent

---

#### Chapter 45: Designing Your Own Agent

**Focus:** Framework for building custom agents

-   Identifying good agent use cases
-   Requirements gathering process
-   Tool design methodology
-   Incremental development approach
-   Testing strategy
-   Deployment planning

**Key Takeaway:** You now have all the tools—go build something amazing.

---

## Appendices

### Appendix A: Python Refresher for Agent Development

-   Async/await essentials
-   Type hints and Pydantic
-   Context managers
-   Decorators for agents
-   Dataclasses for configuration

### Appendix B: API Reference Quick Guide

-   Anthropic API parameters reference
-   Common error codes and solutions
-   Rate limit handling
-   Token counting

### Appendix C: Tool Design Patterns

-   Tool naming conventions
-   Description writing guide
-   Parameter design patterns
-   Error return conventions
-   Tool composition patterns

### Appendix D: Prompt Engineering for Agents

-   System prompt templates
-   Few-shot examples for tools
-   Reasoning prompts
-   Output format instructions
-   Common prompt mistakes

### Appendix E: Troubleshooting Guide

-   Agent won't use tools
-   Agent uses wrong tool
-   Agent loops forever
-   Agent gives inconsistent results
-   Token limit errors
-   Rate limit errors

### Appendix F: Glossary

-   Key terms and definitions

### Appendix G: Resources and Further Reading

-   Official documentation links
-   Recommended papers
-   Community resources
-   Related projects

---

## Code Repository Structure

```
agents-from-scratch/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── src/
│   └── agents/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── augmented_llm.py
│       │   ├── agent.py
│       │   └── config.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── calculator.py
│       │   ├── weather.py
│       │   └── ...
│       ├── workflows/
│       │   ├── __init__.py
│       │   ├── chain.py
│       │   ├── router.py
│       │   ├── parallel.py
│       │   ├── orchestrator.py
│       │   └── evaluator.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── errors.py
│           └── costs.py
├── tests/
│   └── ...
├── examples/
│   ├── ch04_first_call.py
│   ├── ch09_calculator.py
│   └── ...
└── projects/
    ├── research_assistant/
    ├── code_analyzer/
    └── productivity_agent/
```

---

## Conventions Used in This Book

### Code Examples

-   All code is complete and runnable
-   Every example loads environment variables from `.env`
-   Imports are always shown explicitly
-   Type hints are used consistently
-   Docstrings explain function purpose

### Formatting

-   **Bold** for key terms on first use
-   `monospace` for code, commands, and filenames
-   > Blockquotes for important notes and warnings

### Icons

-   💡 Tips and best practices
-   ⚠️ Warnings and common pitfalls
-   🔧 Practical exercises
-   📚 Further reading references
