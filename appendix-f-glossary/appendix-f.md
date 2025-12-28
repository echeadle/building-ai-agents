---
appendix: F
title: "Glossary"
date: 2025-01-15
draft: false
---

# Appendix F: Glossary

This glossary defines key terms used throughout the book. Terms are organized alphabetically with cross-references where helpful.

---

## A

**Agent**
A software system that uses a large language model to make decisions, take actions via tools, and work autonomously toward goals. Unlike a simple chatbot, an agent can call functions, maintain state, and execute multi-step workflows. See also: *Agentic Loop*, *Workflow*.

**Agentic Loop**
The core pattern of agent execution: (1) Send messages to the LLM, (2) Check if LLM wants to use tools, (3) Execute those tools, (4) Send results back to LLM, (5) Repeat until complete. Also called the "tool use loop" or "agent loop." Introduced in Chapter 12.

**Anthropic API**
The REST API provided by Anthropic for accessing Claude models. Requires an API key and accepts HTTP requests with JSON payloads.

**API Key**
A secret token that authenticates your requests to the Anthropic API. Should never be hardcoded or committed to version control. Always load from environment variables using `python-dotenv`.

**Assistant Role**
One of the three message roles in the Anthropic API. Messages with `role: "assistant"` represent responses from Claude. See also: *User Role*, *System Role*.

**Augmented LLM**
The fundamental building block of agents: a large language model enhanced with tools, a system prompt, and structured configuration. Defined and implemented in Chapter 14. Sometimes called "function-calling LLM" or "tool-using LLM."

---

## B

**Backoff Strategy**
A technique for handling transient failures by waiting progressively longer between retry attempts. Example: wait 1 second, then 2, then 4, then 8. Often used with exponential backoff. See *Retry Logic*.

**Batch Processing**
Running agent tasks on multiple inputs simultaneously or in sequence. Useful for processing large datasets or handling multiple requests efficiently. Covered in Chapter 38.

---

## C

**Chain** (Prompt Chain)
A workflow pattern where tasks are broken into sequential steps, with each step's output feeding into the next. Quality gates can validate outputs between steps. One of the five core agentic workflows. Implemented in Chapters 16-17.

**Claude**
Anthropic's family of large language models. Examples: `claude-sonnet-4-20250514`, `claude-opus-4-20250514`. Used throughout this book as the LLM powering agents.

**Context Window**
The maximum amount of text (measured in tokens) that an LLM can process in a single request. Includes the system prompt, conversation history, tool definitions, and user message. Claude models typically have context windows of 200,000+ tokens.

**Conversation History**
The array of previous messages exchanged between the user and assistant. Must be explicitly maintained and passed with each API call since the API is stateless. See Chapter 5.

---

## D

**Delegation**
A workflow pattern where a main agent delegates specific subtasks to specialized sub-agents. Example: a research agent delegating fact-checking to a verification agent. Covered in Chapter 32.

**Dotenv** (python-dotenv)
A Python library for loading environment variables from a `.env` file. Used to manage API keys and other secrets securely. Core pattern established in Chapter 3.

---

## E

**Environment Variable**
A configuration value stored in the operating system's environment, not in code. Used to store sensitive data like API keys. Loaded in Python using `os.getenv()`.

**Error Handling**
Code that anticipates and manages failures gracefully. Essential for production agents since API calls can fail, tools can error, and LLMs can produce unexpected outputs. See Chapter 37.

**Evaluator-Optimizer**
A workflow pattern that iteratively evaluates outputs against criteria and optimizes them until they meet quality standards. One of the five core agentic workflows. Implemented in Chapters 23-25.

---

## F

**Few-Shot Examples**
Example input-output pairs included in prompts to demonstrate desired behavior. More examples ("shots") generally improve performance. Used extensively in system prompts for tools.

**Function Calling**
Anthropic's term for tool use. The LLM "calls" functions by returning structured JSON that specifies which tool to use and with what parameters. You execute the function and return results.

**Function Definition**
The JSON schema that describes a tool to the LLM, including its name, description, and parameters. Must be carefully crafted since this is how the LLM learns what tools are available.

---

## G

**Grounding**
Connecting an LLM's outputs to verifiable external sources. Example: web search grounds responses in current web content. Tools provide grounding by giving LLMs access to real data.

---

## H

**Hallucination**
When an LLM generates plausible-sounding but incorrect or fabricated information. Tools and structured workflows help reduce hallucinations by anchoring responses in real data.

**Human-in-the-Loop**
A pattern where humans review or approve agent decisions before they're executed. Essential for high-stakes applications. Introduced in Chapter 35.

---

## I

**Input Schema**
The JSON schema defining the parameters a tool accepts. Uses JSON Schema format. Each parameter needs a type, description, and indication of whether it's required.

**Iteration**
One pass through the agentic loop: LLM generates a response, tools are executed (if needed), and control returns to the LLM. Complex tasks may require many iterations.

---

## J

**JSON Mode**
A feature where you instruct the LLM to return only valid JSON, making responses easier to parse programmatically. Useful for structured outputs. See Chapter 13.

---

## L

**Latency**
The time between sending a request and receiving a response. Agents often trade increased latency for improved accuracy by using multiple LLM calls or tool executions.

**LLM** (Large Language Model)
A neural network trained on vast amounts of text data to understand and generate human-like language. Examples: Claude, GPT-4, Llama. The "intelligence" behind agents.

**Logging**
Recording agent actions, decisions, and tool calls for debugging and monitoring. Essential for understanding agent behavior and troubleshooting issues. See Chapter 29.

---

## M

**Max Tokens**
The maximum number of tokens the LLM can generate in a response. Set via the `max_tokens` parameter. Prevents runaway generation and controls costs.

**Memory** (Agent Memory)
Information an agent retains across interactions. Can be short-term (conversation history) or long-term (persisted state). See Chapter 36.

**Message**
A single unit in a conversation, containing a role (`system`, `user`, or `assistant`) and content. Messages are the fundamental way to communicate with the Anthropic API.

**Model String**
The identifier for a specific Claude model. Example: `claude-sonnet-4-20250514`. Specified in the `model` parameter of API requests.

---

## O

**Observability**
The ability to understand what an agent is doing and why. Achieved through logging, metrics, and tracing. Critical for debugging and monitoring production agents. See Chapter 29.

**Orchestrator-Workers**
A workflow pattern where a central orchestrator agent distributes tasks to multiple specialized worker agents. Used for complex tasks requiring different expertise. Implemented in Chapters 20-22.

---

## P

**Parallel Execution**
Running multiple agent tasks simultaneously to reduce total latency. Example: querying multiple APIs at once. One of the five core agentic workflows. Implemented in Chapters 18-19.

**Preamble**
Text the LLM generates before deciding to use a tool. Example: "Let me search for that information." Often contains reasoning about what the agent plans to do.

**Prompt**
The input text sent to an LLM. Can be a simple question or a complex instruction with context, examples, and constraints.

**Prompt Chain**
See *Chain*.

**Prompt Engineering**
The art and science of crafting prompts that elicit desired behaviors from LLMs. Includes techniques like few-shot examples, chain-of-thought reasoning, and structured instructions.

---

## Q

**Quality Gate**
A validation step between stages of a workflow that ensures outputs meet standards before proceeding. Example: checking that a translation is accurate before using it in the next step.

---

## R

**Rate Limit**
A restriction on how many API requests you can make in a time period. Anthropic enforces rate limits to ensure fair usage. Requires retry logic with backoff.

**Retry Logic**
Code that automatically retries failed operations, typically with exponential backoff. Essential for handling transient failures like network issues or rate limits.

**Role**
The speaker in a message. Three roles: `system` (instructions), `user` (human input), `assistant` (LLM output). See *System Role*, *User Role*, *Assistant Role*.

**Router**
A workflow pattern that examines an input and routes it to the appropriate handler or specialized agent. One of the five core agentic workflows. Implemented in Chapters 20-21.

---

## S

**SDK** (Software Development Kit)
A library that simplifies API usage. The `anthropic` Python package is the SDK for the Anthropic API. Handles authentication, request formatting, and response parsing.

**Stateless**
Property of the Anthropic API: it retains no memory between requests. You must send complete conversation history with every call.

**Stop Reason**
Why the LLM stopped generating. Examples: `end_turn` (naturally finished), `max_tokens` (hit limit), `tool_use` (wants to call a tool). Returned in the API response.

**Stop Sequence**
A string that, when generated by the LLM, immediately ends generation. Useful for structured outputs with clear delimiters.

**Structured Output**
LLM responses formatted according to a specific schema, usually JSON. Makes parsing and validation easier than freeform text. See Chapter 13.

**System Prompt**
Instructions that define an agent's behavior, capabilities, and constraints. Set once at the beginning of a conversation with `role: "system"`. The agent's "constitution." See Chapter 6.

**System Role**
The message role used for system prompts. Only appears once, at the start of the conversation. Sets global instructions for the agent.

---

## T

**Token**
The basic unit of text for LLMs. Roughly 4 characters or 0.75 words. Used to measure input/output size and calculate costs. Example: "Hello world" ≈ 2 tokens.

**Tool**
A function that an agent can call to interact with the external world. Examples: web search, calculator, database query. Defined using JSON schemas and executed by your code.

**Tool Choice**
A parameter that controls whether the LLM must use tools. Options: `auto` (LLM decides), `any` (must use some tool), `tool` (must use specific tool).

**Tool Definition**
See *Function Definition*.

**Tool Result**
The output from executing a tool, formatted as a message and sent back to the LLM. Must include the `tool_use_id` to match the request.

**Tool Use**
When the LLM decides to call a tool. Returns a `tool_use` content block specifying the tool name and parameters as JSON.

**Tool Use Loop**
See *Agentic Loop*.

**Type Hints**
Python syntax for specifying expected types of variables and function parameters. Example: `def add(x: int, y: int) -> int`. Improves code clarity and enables better IDE support.

---

## U

**User Role**
The message role representing human input. Messages from the user have `role: "user"`.

**UV**
A modern Python package manager that's faster and more reliable than pip. Used throughout this book for dependency management. See Chapter 2.

---

## V

**Validation**
Checking that data meets expected criteria. Essential for tool outputs, LLM responses, and user inputs. Prevents errors from propagating through the system.

---

## W

**Workflow**
A structured pattern for executing multi-step agent tasks. The five core workflows: Prompt Chaining, Routing, Parallel Execution, Orchestrator-Workers, and Evaluator-Optimizer. Covered in Part 3.

**Workflow Pattern**
See *Workflow*.

---

## Cross-Reference: Related Terms

**If you're looking for information about:**
- API authentication → see *API Key*, *Environment Variable*
- Breaking tasks into steps → see *Chain*, *Workflow*
- Calling external functions → see *Tool*, *Function Calling*
- Controlling agent behavior → see *System Prompt*, *Prompt Engineering*
- Handling failures → see *Error Handling*, *Retry Logic*, *Backoff Strategy*
- Making agents smarter → see *Augmented LLM*, *Tool*
- Managing conversation → see *Conversation History*, *Message*, *Role*
- Multi-agent systems → see *Orchestrator-Workers*, *Delegation*
- Performance → see *Latency*, *Parallel Execution*, *Batch Processing*
- Production deployment → see *Observability*, *Logging*, *Monitoring*
- Understanding costs → see *Token*, *Max Tokens*

---

## Acronyms and Abbreviations

- **API**: Application Programming Interface
- **JSON**: JavaScript Object Notation
- **LLM**: Large Language Model
- **REST**: Representational State Transfer
- **SDK**: Software Development Kit
- **URL**: Uniform Resource Locator

---

## Common Patterns and Conventions

**Naming Conventions in This Book:**
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPERCASE` for constants
- `tool_name_tool()` for tool functions

**File Naming:**
- `chapter_XX_name.py` for chapter examples
- `tool_name_tool.py` for tool implementations
- `test_name.py` for test files
- `.env` for environment variables (never commit!)

**Import Conventions:**
```python
import os
from dotenv import load_dotenv
import anthropic
from typing import List, Dict, Optional
```

---

## Further Reading

For more detailed information on specific terms:
- **Chapters 1-6**: Basic concepts and setup
- **Chapters 7-14**: Augmented LLM and tools
- **Chapters 15-25**: Workflow patterns
- **Chapters 26-35**: Advanced agent capabilities
- **Chapters 36-41**: Production considerations
- **Chapters 42-45**: Complete projects
- **Appendix D**: Prompt engineering guide
- **Appendix E**: Troubleshooting guide

---

**Note:** This glossary reflects terminology and concepts as used in this book. Some terms may have different definitions in other contexts or frameworks. We've chosen definitions that emphasize the simple, composable patterns at the heart of building effective agents.
