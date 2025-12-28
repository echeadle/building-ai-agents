---
chapter: 38
title: "Cost Optimization"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 38: Cost Optimization

## Introduction

Your agent works beautifully. It passes all tests, handles errors gracefully, and users love it. Then you check your API bill at the end of the month: $2,847.53. For a side project.

This isn't hypothetical. Unmonitored agents are notorious for generating surprising bills. Every LLM call costs money, and agents make *many* LLM calls. A single user request might trigger 5, 10, or even 20 API calls as the agent reasons through a complex task. Multiply that by thousands of users, and costs escalate quickly.

In Chapter 37, we learned to debug agent behavior. Now we'll debug agent *economics*. This chapter teaches you to understand, track, and optimize the costs of running AI agents in production. You'll learn where the money goes, how to spend less of it without sacrificing quality, and how to set up monitoring so you're never surprised by a bill again.

The goal isn't to minimize costs at all costs—it's to maximize *value per dollar*. Sometimes spending more on a better model is worth it. Sometimes a cached response is just as good as a fresh one. The key is making these tradeoffs deliberately, with data.

## Learning Objectives

By the end of this chapter, you will be able to:

- Calculate the true cost of agent operations using token-based pricing
- Implement prompt optimization techniques that reduce input tokens without losing effectiveness
- Control response length to manage output token costs
- Build caching systems that eliminate redundant API calls
- Select appropriate models based on task complexity and cost requirements
- Set up cost monitoring with alerts for budget protection

## Understanding Token Costs

Before we can optimize costs, we need to understand how they're calculated. Claude's API charges based on **tokens**—the fundamental units that LLMs use to process text.

### What Are Tokens?

Tokens are pieces of words. A rough rule of thumb:
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

But tokenization isn't perfectly predictable. Common words might be single tokens, while unusual words get split into multiple tokens. Code and non-English text often use more tokens per character.

### Claude's Pricing Model

Claude charges separately for **input tokens** (what you send) and **output tokens** (what Claude generates). As of early 2025, here are the approximate prices:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |

> **Note:** Prices change over time. Always check Anthropic's current pricing at https://www.anthropic.com/pricing

The key insight: **output tokens cost 5x more than input tokens**. This means controlling response length has a bigger impact on costs than reducing prompt length.

### Where Agent Costs Come From

Agents are expensive because they make multiple API calls per request. Let's trace the costs:

```
User Request: "Research the top 3 electric vehicles and compare them"

Call 1: Planning (analyze request)
  - Input: 500 tokens (system prompt + user request)
  - Output: 200 tokens (plan)
  - Cost: $0.0045

Call 2: Search for EV #1
  - Input: 800 tokens (context + tool definitions)
  - Output: 100 tokens (tool call)
  - Cost: $0.0039

Call 3: Process search results
  - Input: 2000 tokens (context + search results)
  - Output: 300 tokens (analysis)
  - Cost: $0.0105

... (repeat for EVs #2 and #3)

Call 8: Final synthesis
  - Input: 4000 tokens (all gathered information)
  - Output: 1000 tokens (comprehensive comparison)
  - Cost: $0.027

Total: 8 API calls, ~$0.15 per user request
```

At $0.15 per request, 10,000 daily users would cost $1,500/day or $45,000/month. This is why cost optimization matters.

## Prompt Optimization Techniques

The first line of defense against high costs is writing efficient prompts. Here's how to reduce input tokens without sacrificing effectiveness.

### 1. Eliminate Redundancy

Many prompts repeat information or include unnecessary context. See `code/prompt_optimizer.py` for implementation.

### 2. Optimize Tool Definitions

Tool definitions are sent with every request. Bloated descriptions waste tokens on every single call. See `code/tool_optimizer.py` for techniques.

### 3. Context Window Management

As conversations grow, so do costs. Implement smart truncation with the `ContextManager` class in `code/context_manager.py`.

## Response Length Management

Since output tokens cost 5x more than input tokens, controlling response length has the biggest impact on costs. See `code/response_manager.py` for the `ResponseLengthManager` class.

Key strategies:
- Set appropriate `max_tokens` for each response type
- Use explicit length instructions in prompts
- Request structured/concise formats

## Caching Strategies

The best API call is the one you don't make. Caching can dramatically reduce costs for repeated or similar queries. See `code/response_cache.py` for the complete `ResponseCache` implementation.

Caching strategies:
- **Exact match**: Cache identical prompts
- **TTL-based expiration**: Invalidate stale entries
- **Size-based eviction**: Remove old entries when full

## Model Selection Strategy

Not every task needs the most powerful (and expensive) model. The `ModelSelector` class in `code/model_selector.py` helps you choose the right model.

Guidelines:
- **Haiku**: Simple tasks, classification, yes/no questions
- **Sonnet**: Most tasks, coding, analysis
- **Opus**: Complex reasoning, creative work, high-stakes decisions

## Cost Monitoring and Alerts

The final piece: tracking costs in real-time and alerting when budgets are exceeded. See `code/cost_tracker.py` for the complete `CostTracker` implementation.

Features:
- Per-request cost tracking
- Daily/weekly/monthly aggregation
- Budget enforcement
- Alert generation

## The Complete Cost Optimization Module

All components work together in `code/cost_optimization.py`. This module provides:
- Token estimation
- Cost calculation
- Prompt optimization
- Response caching
- Model selection
- Cost tracking and alerts

## Common Pitfalls

**1. Not tracking costs from day one**

Many developers add cost tracking after getting an unexpected bill. By then, you've lost valuable usage data and may have already overspent. Add tracking before your first production API call.

**2. Over-aggressive caching**

Caching can cause issues when:
- Responses should vary (creative tasks, personalized content)
- Information changes frequently (real-time data)
- Cache keys are too broad (different contexts, same prompt)

Always consider whether a cached response is appropriate for the use case.

**3. Wrong model selection logic**

Don't assume the cheapest model is always best. Consider:
- Quality requirements (some tasks need better models)
- Retry costs (a cheaper model that fails requires re-running)
- User experience (faster models may justify higher costs)

**4. Ignoring context growth**

Agents accumulate context over time. A conversation that starts at 500 tokens can grow to 50,000 tokens. Always implement context management before you need it.

## Practical Exercise

**Task:** Build a cost dashboard that tracks agent spending

**Requirements:**

1. Create a simple web dashboard (HTML + JavaScript) that displays:
   - Current daily, weekly, monthly spending
   - Budget utilization as progress bars
   - Recent requests with costs
   - Alerts and warnings

2. The dashboard should read from a JSON file updated by `CostTracker`

3. Include a "Cost Projection" section that estimates end-of-month costs based on current usage

**Hints:**

- Use the `CostTracker.get_cost_summary()` method for aggregate data
- Store usage records with `persist_path` for the dashboard to read
- Calculate projections by averaging daily costs and multiplying by remaining days

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Output tokens cost 5x more than input tokens**—controlling response length has the biggest impact on costs
- **Track costs from day one**—use the `CostTracker` class to monitor every API call
- **Cache aggressively but wisely**—caching can eliminate most redundant API calls
- **Use the right model for the task**—Haiku for simple tasks, Sonnet for most work, Opus for complex reasoning
- **Optimize prompts**—remove redundancy, compress instructions, manage context window
- **Set budgets and alerts**—never be surprised by a bill again
- **Monitor token accumulation**—conversations grow over time, implement context management early

## What's Next

Cost optimization keeps your agents affordable. But users also care about speed. In the next chapter, **Latency Optimization**, you'll learn to make your agents faster through streaming, parallel execution, caching, and smart architecture choices. Because in production, every millisecond matters.
