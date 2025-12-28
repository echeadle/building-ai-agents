---
chapter: 7
title: "Introduction to Tool Use"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 7: Introduction to Tool Use

## Introduction

Imagine asking a brilliant friend to check tomorrow's weather. They can eloquently describe what weather is, explain meteorology in fascinating detail, and even discuss historical weather patterns—but they can't actually tell you if it will rain tomorrow. Without access to current data, their knowledge hits a wall.

This is exactly where we find ourselves with the LLM we've built so far. Claude can write poetry, explain quantum physics, and debug code. But ask it to check your calendar, send an email, or look up today's stock prices? It's stuck. The model's knowledge is frozen at its training cutoff, and it has no way to interact with the outside world.

**Tools change everything.**

In Part 1, you learned to communicate with Claude—sending messages, maintaining conversations, and crafting system prompts. Now, in Part 2, we'll transform that conversational LLM into something far more powerful: an **Augmented LLM** that can take actions in the real world.

This chapter introduces the concept of tool use and explains why it's the foundation of every useful AI agent. You won't write much code here—instead, you'll understand the "why" and "how" that makes everything in the next seven chapters click into place.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain why LLMs need tools to be useful agents
- Describe the three-phase tool use cycle (define → call → respond)
- Understand how Claude decides when and which tools to use
- Identify the types of capabilities tools can provide
- Recognize the architectural pattern we'll build throughout Part 2

## The Limitations of a "Naked" LLM

Let's be concrete about what Claude *cannot* do on its own.

### No Access to Current Information

Claude's training data has a cutoff date. Ask about events after that date, and it can only say "I don't have information about that." Even for events before the cutoff, Claude can't verify if things have changed.

```python
# What happens when you ask about current information
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "What's the current price of Apple stock?"}
    ]
)
```

Claude will respond honestly that it can't provide real-time stock prices. It's not being coy—it genuinely has no mechanism to fetch that data.

### No Ability to Take Actions

Claude processes text and generates text. That's it. On its own, it cannot:

- Send emails or messages
- Create, read, or modify files
- Make purchases or reservations
- Control smart home devices
- Post to social media
- Execute code
- Query databases

### No Persistent Memory Beyond the Conversation

As you learned in Chapter 5, Claude doesn't remember previous conversations. But it goes deeper than that—within a single API call, Claude has no way to store information for later retrieval. It can't write notes to itself, save intermediate calculations, or build up a knowledge base.

### No Way to Verify Its Own Claims

Perhaps most critically, Claude has no way to fact-check itself. It generates responses based on patterns in training data, but it cannot:

- Search the web to verify a claim
- Look up a source to confirm a citation
- Check current documentation for a programming library
- Verify that a business is still in operation

> **The Core Problem:** An LLM without tools is like a brilliant brain in a jar—full of knowledge and reasoning ability, but unable to perceive or affect the outside world.

## What Tool Use Enables

Tools are Claude's hands, eyes, and ears. They let the model reach beyond its training data to interact with the real world.

### Categories of Tool Capabilities

Tools generally fall into four categories:

**1. Information Retrieval**
Tools that fetch current data from external sources:
- Web search
- Database queries
- API calls (weather, stocks, news)
- File reading
- Calendar/email access

**2. Actions and Side Effects**
Tools that change something in the world:
- Sending messages (email, SMS, Slack)
- Creating or modifying files
- Making API calls that change state
- Controlling external systems

**3. Computation**
Tools that perform calculations the LLM shouldn't do on its own:
- Mathematical calculations (LLMs are unreliable at math)
- Code execution
- Data analysis
- Format conversions

**4. Verification and Validation**
Tools that check or validate information:
- Fact-checking against authoritative sources
- Input validation
- Spell/grammar checking
- Code linting

### The Transformation

With tools, Claude transforms from a text generator into an **agent** that can:

| Without Tools | With Tools |
|---------------|------------|
| "I don't have access to real-time stock data" | Fetches current stock price and provides analysis |
| "I can't send emails for you" | Drafts and sends the email after confirmation |
| "My training data might be outdated" | Searches for current documentation |
| "I can't perform complex calculations reliably" | Uses a calculator tool for precise math |

This is the difference between a chatbot and an assistant that actually gets things done.

## The Tool Use Cycle

Tool use follows a three-phase cycle. Understanding this cycle is essential—every tool interaction you build will follow this pattern.

### Phase 1: Define

Before Claude can use a tool, you must define it. A tool definition tells Claude:

- **What the tool is called** (a unique name)
- **What the tool does** (a description Claude reads to decide when to use it)
- **What inputs the tool accepts** (parameters with their types and descriptions)

Here's a conceptual example:

```
Tool: get_weather
Description: Get the current weather for a specified city.
             Use this when the user asks about weather conditions.
Parameters:
  - city (string, required): The city name, e.g., "London" or "New York"
  - units (string, optional): Temperature units, either "celsius" or "fahrenheit"
```

Claude reads this definition and understands: "If someone asks about weather, I can use `get_weather` with a city name to find out."

> **Key Insight:** Tool descriptions are prompts. Claude uses natural language understanding to decide when a tool is appropriate. Write descriptions for Claude, not just for human developers.

### Phase 2: Call

When Claude determines a tool would help answer a query, it doesn't execute the tool itself. Instead, it responds with a **tool use request**—a structured message saying "I'd like to call this tool with these arguments."

The response might look like:

```
I'll check the weather in London for you.

[Tool Call: get_weather]
  city: "London"
  units: "celsius"
```

At this point, Claude stops and waits. It's your code's job to:
1. Detect that Claude wants to use a tool
2. Parse the tool name and arguments
3. Actually execute the tool (call the weather API, query the database, etc.)
4. Return the results to Claude

**Claude never executes tools directly.** You are always in control. This is a safety feature—you decide which tools exist and how they work.

### Phase 3: Respond

After you execute the tool and return the results, Claude incorporates that information into its final response:

```
The current weather in London is 15°C with partly cloudy skies.
Humidity is at 72%, and there's a light breeze from the west.
It looks like a good day for a walk, though you might want
a light jacket!
```

Claude takes the raw tool output (probably JSON with temperature, conditions, etc.) and transforms it into a natural, helpful response.

### The Complete Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE TOOL USE CYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. DEFINE                                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  You define tools with names, descriptions,             │  │
│   │  and parameter schemas                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│   2. CALL                                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Claude receives user message + tool definitions        │  │
│   │  Claude decides to use a tool                           │  │
│   │  Claude returns tool_use request with arguments         │  │
│   │  YOUR CODE executes the tool                            │  │
│   │  Your code returns results to Claude                    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│   3. RESPOND                                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Claude incorporates tool results                       │  │
│   │  Claude generates final response to user                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## How Claude Decides When to Use Tools

Claude doesn't randomly use tools—it makes intelligent decisions based on several factors.

### The Decision Process

When Claude receives a message along with tool definitions, it considers:

1. **Does the user's request require information I don't have?**
   "What's the weather?" → Yes, I need current data → Consider weather tool

2. **Does any available tool match what's needed?**
   Looking at tool descriptions: `get_weather` can help → Select this tool

3. **Can I answer without tools?**
   "What causes rain?" → I know this from training → No tools needed

4. **Do I have the required parameters?**
   User said "London" → I have the city name → Ready to call tool
   User said "the weather" → Which city? → Ask for clarification instead

### Tool Selection with Multiple Tools

When multiple tools are available, Claude reads all their descriptions and selects the most appropriate one:

```
Available tools:
- get_weather: Get current weather conditions
- get_forecast: Get 5-day weather forecast  
- get_news: Search recent news articles

User: "Will I need an umbrella for my trip to Paris next week?"
```

Claude's reasoning (simplified):
- User is asking about future weather, not current weather
- `get_forecast` matches better than `get_weather`
- This isn't a news query, so `get_news` isn't relevant
- Decision: Use `get_forecast` with city="Paris"

### When Claude Won't Use Tools

Claude might choose not to use available tools when:

- The question can be answered from training knowledge
- No available tool matches the request
- Required parameters are missing (Claude will ask for clarification)
- The tool description doesn't seem relevant

> **Important:** The quality of your tool descriptions directly affects how well Claude uses your tools. We'll dive deep into writing effective descriptions in Chapter 8.

## A Peek at What's Coming

Let's look at a simple example that previews what you'll build in the coming chapters. Don't worry about understanding every detail—the goal is to see the big picture.

```python
"""
Preview of tool use - you'll understand every line by Chapter 9.
"""

import anthropic

client = anthropic.Anthropic()

# Phase 1: DEFINE the tool
tools = [
    {
        "name": "calculate",
        "description": "Perform basic arithmetic. Use this for any math calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g., '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        }
    }
]

# Send message with tool definitions
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,  # <-- Tool definitions go here
    messages=[
        {"role": "user", "content": "What's 1,547 times 892?"}
    ]
)

# Phase 2: Claude will CALL the tool
# (We'll handle this properly in Chapter 9)
print("Claude's response type:", response.stop_reason)
# Output: Claude's response type: tool_use
```

Notice:
- Tools are passed to the API in a `tools` parameter
- Each tool has a name, description, and schema for its parameters
- When Claude wants to use a tool, it stops with `stop_reason: tool_use`

In Chapters 8 and 9, you'll learn to define tools properly and handle tool calls. By Chapter 14, you'll have a complete `AugmentedLLM` class that manages all of this elegantly.

## The Building Block We're Constructing

Throughout Part 2, we're building what Anthropic calls the **Augmented LLM**—an LLM enhanced with tools that serves as the fundamental building block for all agent architectures.

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE AUGMENTED LLM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    LLM (Claude)                         │  │
│   │   • Understands natural language                        │  │
│   │   • Reasons about tasks                                 │  │
│   │   • Generates responses                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            │ Enhanced with                      │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                      TOOLS                              │  │
│   │   • Retrieval: Search, databases, APIs                  │  │
│   │   • Actions: Email, files, external systems             │  │
│   │   • Computation: Math, code execution                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            │ Producing                          │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │               STRUCTURED OUTPUTS                        │  │
│   │   • Validated responses                                 │  │
│   │   • Predictable formats                                 │  │
│   │   • Parseable data                                      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Every workflow pattern in Part 3 and every agent in Part 4 will use this Augmented LLM as its core component. By building it properly now, you'll have a solid foundation for everything that follows.

### What We'll Build in Part 2

Here's the roadmap for the next seven chapters:

| Chapter | Focus | What You'll Build |
|---------|-------|-------------------|
| 8 | Defining Tools | Calculator tool definition |
| 9 | Handling Tool Calls | Working tool execution loop |
| 10 | Real-World Tools | Weather API integration |
| 11 | Multiple Tools | Multi-tool agent |
| 12 | Sequential Calls | Agentic tool loop |
| 13 | Structured Output | Response validation |
| 14 | Complete Package | The AugmentedLLM class |

Each chapter builds directly on the previous one. By Chapter 14, you'll have a production-ready building block that you'll use throughout the rest of the book.

## Common Pitfalls

Even at this conceptual stage, it's helpful to know what trips people up:

### 1. Thinking Claude Executes Tools Directly

Claude never runs code or calls APIs itself. It only *requests* that you execute tools on its behalf. You are always in control of what actually happens.

**Why this matters:** This is a security feature. You can validate inputs, rate-limit calls, and prevent dangerous operations before they happen.

### 2. Writing Tool Descriptions for Humans, Not Claude

Tool descriptions are prompts that Claude reads to understand when to use a tool. Writing "Gets weather data" is less helpful than "Get the current weather conditions for a city. Use this when the user asks about weather, temperature, or if they need to know what to wear."

**Why this matters:** Claude makes better tool choices when descriptions are detailed and include usage guidance.

### 3. Providing Too Many Tools at Once

If you give Claude 50 tools, it has to parse all their descriptions and reason about which to use. This increases latency, cost, and the chance of selecting the wrong tool.

**Why this matters:** Start with the minimum tools needed. You can always add more based on specific workflows.

## Practical Exercise

This chapter is conceptual, so the exercise focuses on designing rather than coding.

**Task:** Design tool definitions for a personal assistant

Imagine you're building a personal assistant that can:
- Check the weather in any city
- Look up events on a calendar
- Send email messages
- Set reminders

For each capability, write:
1. A descriptive tool name
2. A clear description that tells Claude when to use it
3. The parameters it would need (name, type, required/optional)

**Don't write code yet**—just write out the tool definitions in plain text or as comments.

**Example format:**
```
Tool: tool_name
Description: What this tool does and when Claude should use it.
Parameters:
  - param1 (type, required): What this parameter is
  - param2 (type, optional): What this parameter is
```

**Hints:**
- Think about what information Claude needs to call each tool
- Consider what makes each tool distinct from the others
- Write descriptions as if you're explaining to a smart colleague

**Solution:** See `code/exercise.py` for example tool designs.

## Key Takeaways

- **LLMs alone are limited**: Without tools, Claude can't access current information, take actions, or verify its outputs.

- **Tools transform capabilities**: Tools let Claude interact with the real world—fetching data, performing actions, and executing computations.

- **The tool use cycle has three phases**: Define (specify what tools exist), Call (Claude requests, you execute), Respond (Claude uses results).

- **Claude chooses tools intelligently**: Based on user intent, tool descriptions, and available parameters, Claude decides whether and which tools to use.

- **You're always in control**: Claude never executes tools directly—your code handles all actual execution, giving you full control over safety and validation.

- **This is the foundation**: The Augmented LLM we build in Part 2 is the core building block for every agent architecture.

## What's Next

In Chapter 8, we'll roll up our sleeves and write our first real tool definition. You'll learn the exact structure Claude expects, how to write JSON schemas for parameters, and the art of crafting descriptions that help Claude make smart decisions. We'll build a simple calculator tool that demonstrates all these concepts in working code.

Let's make Claude capable of doing math it can actually trust.
