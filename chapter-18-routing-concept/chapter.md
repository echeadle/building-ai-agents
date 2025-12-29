---
chapter: 18
title: "Routing - Concept and Design"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 18: Routing - Concept and Design

## Introduction

Imagine you're building a customer support system. Some users ask billing questions, others have technical issues, and some just want to know your business hours. You could try to create one super-prompt that handles everything, but you've probably noticed that prompts optimized for one task often perform poorly on others. A prompt that's great at technical troubleshooting might be terrible at handling emotional customers who want refunds.

This is where **routing** comes in—the second workflow pattern in our toolkit.

In Chapter 17, you learned prompt chaining, where tasks flow through a predetermined sequence of steps. Routing is different: instead of a fixed path, you analyze the input first and then direct it to the most appropriate handler. Think of it like a phone tree that actually works—each caller gets connected to the specialist best equipped to help them.

In this chapter, we'll explore the concept and design of routing systems. You'll learn when routing is the right pattern, how to classify inputs effectively, and how to design route handlers that excel at their specific tasks. In Chapter 19, we'll implement a complete routing system.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain what routing is and identify use cases where it excels
- Compare LLM-based classification with rule-based approaches
- Design specialized route handlers for different input types
- Create effective fallback strategies for unmatched inputs
- Sketch routing architectures for real-world problems

## What Is Routing?

**Routing** is a workflow pattern where you classify an input and direct it to a specialized handler based on that classification. It's the agentic equivalent of a switchboard operator or a triage nurse.

Here's the core idea:

```
Input --> Classifier --> Route Selection --> Specialized Handler --> Output
```

The classifier examines the input and assigns it to a category. Each category has its own specialized handler—a prompt, a tool set, or even a different model entirely—optimized for that specific type of request.

### Why Not Just Use One Prompt?

You might wonder: "Can't I just write a really good prompt that handles everything?"

You can try, but you'll hit several problems:

1. **Prompt bloat**: As you add instructions for every possible scenario, your prompt becomes enormous and unwieldy.

2. **Conflicting instructions**: What's appropriate for a technical query ("be precise, use jargon") conflicts with emotional support ("be warm, avoid technical terms").

3. **Diminishing returns**: The more tasks you cram into one prompt, the worse the model performs at each individual task.

4. **Cost inefficiency**: You're paying tokens to include instructions for scenarios that don't apply to the current input.

Routing solves these problems by letting you **optimize each handler for its specific use case** without compromising the others.

### The Routing Mental Model

Think of routing like a hospital emergency room:

1. **Triage (Classifier)**: A nurse quickly assesses each patient and assigns a priority level and department.

2. **Departments (Routes)**: Cardiac patients go to cardiology, trauma patients go to surgery, minor injuries go to urgent care.

3. **Specialists (Handlers)**: Each department has doctors specialized in that area.

4. **Default Path (Fallback)**: Unclear cases go to general medicine for further evaluation.

This structure ensures that experts handle what they're best at, while still providing care for edge cases.

## When to Use Routing

Routing shines in specific scenarios. Here's when to reach for this pattern:

### Good Use Cases for Routing

**Distinct input categories with different optimal approaches:**

- Customer support (billing vs. technical vs. complaints)
- Content moderation (spam vs. hate speech vs. safe content)
- Document processing (invoices vs. contracts vs. emails)
- Multi-language support (route to language-specific handlers)

**Different resource requirements by category:**

- Simple questions → fast, cheap model
- Complex analysis → capable, expensive model
- Sensitive topics → model with safety specialization

**Specialized tools per category:**

- Math questions → route to calculator-equipped handler
- Code questions → route to code-execution handler
- Research questions → route to web-search handler

### When Routing May Be Overkill

Not every problem needs routing:

- **Single-purpose applications**: If your agent only does one thing, routing adds complexity without benefit.
- **Highly overlapping categories**: If most inputs could fit multiple categories equally well, you'll spend more time classifying than helping.
- **Simple variations**: If the handling only differs slightly between categories, a single prompt with conditional logic might suffice.

> **💡 Tip:** Start without routing. Add it when you notice that optimizing for one type of input hurts performance on another.


## Classification Strategies

The classifier is the brain of your routing system. It examines each input and decides where it should go. You have two main approaches: LLM-based classification and rule-based classification.

### LLM-Based Classification

Use the LLM itself to categorize inputs. You provide the categories and let the model decide.

**How it works:**

```
System: You are a classifier. Categorize the user message into exactly 
one of these categories: BILLING, TECHNICAL, GENERAL, COMPLAINT.

Respond with only the category name.

User: I have been charged twice for my subscription this month!
```

**Expected response:** `BILLING`

**Advantages of LLM classification:**

- **Handles nuance**: Can understand context, sarcasm, and complex phrasing
- **No maintenance**: Adapts to new phrasings without code changes
- **Multi-label capable**: Can detect multiple categories if needed
- **Explanation available**: Can provide reasoning for its choice

**Disadvantages:**

- **Latency**: Adds an API call before the main task
- **Cost**: Every classification costs tokens
- **Unpredictability**: May occasionally misclassify

### Rule-Based Classification

Use pattern matching, keywords, or heuristics to classify inputs without calling the LLM.

**How it works:**

```python
def classify_with_rules(message: str) -> str:
    message_lower = message.lower()
    
    # Check for billing keywords
    billing_keywords = ["charge", "bill", "payment", "invoice", "refund", "price"]
    if any(word in message_lower for word in billing_keywords):
        return "BILLING"
    
    # Check for technical keywords
    technical_keywords = ["error", "bug", "crash", "not working", "broken"]
    if any(word in message_lower for word in technical_keywords):
        return "TECHNICAL"
    
    # Check for complaint signals
    if any(phrase in message_lower for phrase in ["terrible", "worst", "angry", "furious"]):
        return "COMPLAINT"
    
    # Default fallback
    return "GENERAL"
```

**Advantages of rule-based classification:**

- **Speed**: No API call, instant classification
- **Cost**: Zero token cost
- **Predictability**: Same input always produces same output
- **Transparency**: Easy to debug and explain

**Disadvantages:**

- **Brittleness**: Misses variations and edge cases
- **Maintenance burden**: Must manually add new patterns
- **No context understanding**: Cannot grasp nuance or intent


### Hybrid Classification

In practice, you often want the best of both worlds. A **hybrid approach** uses rules first and falls back to the LLM for uncertain cases.

```
Input --> Rule-Based Check --> If confident: Route directly
                          --> If uncertain: LLM Classification --> Route
```

**When to use each:**

| Scenario | Approach |
|----------|----------|
| High volume, tight latency budget | Rules first, LLM fallback |
| Complex, nuanced inputs | LLM only |
| Well-defined, keyword-heavy inputs | Rules only |
| Critical accuracy requirements | LLM with confidence threshold |

### Classification Design Tips

1. **Keep categories mutually exclusive**: Overlapping categories confuse the classifier and create routing ambiguity.

2. **Use clear category names**: Names like `BILLING` and `TECHNICAL` are better than `TYPE_A` and `TYPE_B`.

3. **Provide examples in the prompt**: Few-shot examples dramatically improve LLM classification accuracy.

4. **Include confidence scores**: Ask the LLM to rate its confidence so you can handle uncertain cases differently.

5. **Test with edge cases**: Build a test set of ambiguous inputs to measure classification accuracy.


## Designing Route Handlers

Each route needs a **handler**—the specialized component that actually processes the input once it's been classified. Good handler design is what makes routing worthwhile.

### What Makes a Good Handler?

A handler should be **optimized for its specific category**. This means:

**1. Specialized System Prompt**

Each handler gets a system prompt tailored to its task:

```python
HANDLER_PROMPTS = {
    "BILLING": """You are a billing support specialist. You help customers with:
- Understanding charges and invoices
- Processing refunds and credits
- Explaining pricing and plans
- Resolving payment issues

Be empathetic about billing concerns. Always verify the customer's account 
before making changes. If you cannot resolve the issue, escalate to a human.""",

    "TECHNICAL": """You are a technical support engineer. You help customers with:
- Troubleshooting errors and bugs
- Explaining how features work
- Guiding through configuration
- Identifying workarounds

Be precise and technical. Ask clarifying questions about error messages, 
browser versions, and steps to reproduce. Include code examples when helpful.""",

    "COMPLAINT": """You are a customer experience specialist handling complaints.
Your priorities:
1. Acknowledge the customer's frustration
2. Apologize sincerely for their experience
3. Understand the full scope of the issue
4. Offer concrete resolution or escalation

Never be defensive. The customer's feelings are valid even if the 
complaint seems minor.""",

    "GENERAL": """You are a helpful customer service representative. 
Answer general questions about our products and services. 
If a question requires specialized knowledge (billing, technical, etc.), 
let the customer know you'll connect them with the right team."""
}
```

**2. Appropriate Tools**

Different handlers may need different tools:

- **BILLING handler**: Account lookup, refund processing, plan comparison
- **TECHNICAL handler**: Log viewer, diagnostics runner, documentation search
- **COMPLAINT handler**: Ticket creation, escalation trigger, compensation calculator
- **GENERAL handler**: FAQ search, business hours lookup

**3. Different Models (Optional)**

You can even route to different models based on category:

- Simple FAQs → Claude Haiku (fast, cheap)
- Technical troubleshooting → Claude Sonnet (balanced)
- Complex complaints → Claude Opus (highest capability)


### The Handler Interface Pattern

To keep your code organized, define a consistent interface for all handlers:

```python
from abc import ABC, abstractmethod
from typing import Any

class RouteHandler(ABC):
    """Base class for all route handlers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this handler."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this handler's LLM calls."""
        pass
    
    @property
    def tools(self) -> list[dict]:
        """Tools available to this handler. Override to add tools."""
        return []
    
    @property
    def model(self) -> str:
        """Model to use. Override for different models per handler."""
        return "claude-sonnet-4-20250514"
    
    @abstractmethod
    def handle(self, message: str, context: dict[str, Any] = None) -> str:
        """Process the message and return a response."""
        pass
```

This pattern gives you:

- **Consistency**: All handlers work the same way
- **Flexibility**: Each handler can customize prompts, tools, and models
- **Testability**: Easy to test handlers in isolation
- **Extensibility**: Adding new handlers is straightforward


## Default and Fallback Routes

Not every input fits neatly into predefined categories. You need strategies for handling the uncertain cases.

### The Default Route

A **default route** handles inputs that don't match any specific category. Think of it as your "catch-all" handler.

**Good default route characteristics:**

- **Generalist capabilities**: Can handle a wide range of topics
- **Escalation awareness**: Knows when to hand off to a specialist
- **Information gathering**: Asks clarifying questions to determine the right route

Example default handler behavior:

```
User: I need help with something

Default Handler: I'd be happy to help! Could you tell me a bit more about 
what you need assistance with? For example:
- Are you having issues with a charge or payment?
- Is something not working correctly?
- Do you have a general question about our service?
```

### The Fallback Route

A **fallback route** activates when something goes wrong—the classifier fails, a handler throws an error, or the system reaches an unexpected state.

**Fallback route responsibilities:**

1. **Apologize gracefully**: Don't expose technical errors to users
2. **Attempt recovery**: Try a simpler approach or the default handler
3. **Log the issue**: Record what went wrong for debugging
4. **Escalate if needed**: Connect to a human if recovery fails

```python
def fallback_handler(message: str, error: Exception = None) -> str:
    """Handle routing failures gracefully."""
    
    # Log the error for debugging
    if error:
        logger.error(f"Routing failed: {error}", exc_info=True)
    
    return """I apologize, but I'm having trouble processing your request 
right now. Let me connect you with someone who can help directly. 
In the meantime, could you briefly describe what you need assistance with?"""
```

### Confidence-Based Routing

For LLM classification, you can add a confidence threshold to determine when to use the default route:

```python
def classify_with_confidence(message: str) -> tuple[str, float]:
    """Return both category and confidence score."""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Classify this message and rate your confidence.

Message: {message}

Categories: BILLING, TECHNICAL, COMPLAINT, GENERAL

Respond in this exact format:
CATEGORY: <category>
CONFIDENCE: <0.0-1.0>"""
        }]
    )
    
    # Parse response...
    return category, confidence


def route_with_threshold(message: str, threshold: float = 0.7) -> str:
    """Route based on confidence threshold."""
    
    category, confidence = classify_with_confidence(message)
    
    if confidence >= threshold:
        return handlers[category].handle(message)
    else:
        # Low confidence—use default handler
        return handlers["GENERAL"].handle(message)
```

This approach prevents misrouting when the classifier is uncertain.


## Routing Architecture Diagram

Here's how all the pieces fit together:

```
                              ┌─────────────────────┐
                              │    User Message     │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │     Classifier      │
                              │  (LLM or Rules)     │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
           │    BILLING    │    │   TECHNICAL   │    │   COMPLAINT   │
           │    Handler    │    │    Handler    │    │    Handler    │
           │               │    │               │    │               │
           │ • Billing     │    │ • Tech        │    │ • Empathy     │
           │   prompt      │    │   prompt      │    │   prompt      │
           │ • Account     │    │ • Debug       │    │ • Escalation  │
           │   tools       │    │   tools       │    │   tools       │
           │ • Fast model  │    │ • Smart model │    │ • Best model  │
           └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                   │                    │                    │
                   └────────────────────┼────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │     Response        │
                              └─────────────────────┘
                              
                              
                  Fallback Path (on classifier failure or low confidence):
                  
                              ┌─────────────────────┐
                              │    User Message     │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │     Classifier      │──── Error/Low Confidence
                              └──────────┬──────────┘           │
                                         │                      │
                                         ▼                      ▼
                              ┌─────────────────────┐  ┌─────────────────────┐
                              │   Normal Routing    │  │   DEFAULT Handler   │
                              └─────────────────────┘  └─────────────────────┘
```

### Data Flow Summary

1. **Input arrives**: User sends a message
2. **Classification**: Classifier determines the category
3. **Routing**: Message goes to the appropriate handler
4. **Processing**: Handler uses its specialized prompt/tools/model
5. **Response**: Handler returns the result

If classification fails or confidence is low, the default handler takes over.


## Common Pitfalls

### 1. Too Many Categories

**The Problem:** You create 15 different categories, and the classifier struggles to distinguish between them. Accuracy drops, and many inputs get misrouted.

**The Solution:** Start with 3-5 broad categories. You can always split them later if needed. A message going to a slightly-wrong handler is usually better than frequent misclassification.

### 2. Overlapping Categories

**The Problem:** Your categories aren't mutually exclusive. "I'm frustrated that my payment failed" could be BILLING, TECHNICAL, or COMPLAINT. The classifier picks randomly.

**The Solution:** Define clear boundaries. Ask yourself: "If this message matches multiple categories, which one takes priority?" Create a hierarchy:

```
COMPLAINT > BILLING > TECHNICAL > GENERAL
```

Or redefine categories to be mutually exclusive based on the primary need.

### 3. Ignoring the Default Route

**The Problem:** You build specialized handlers but forget the default case. When unexpected inputs arrive, the system crashes or gives nonsensical responses.

**The Solution:** Always implement a robust default handler first. It's your safety net. Make it capable of handling general queries and smart enough to request clarification.

## Practical Exercise

**Task:** Design a routing system for a software documentation assistant

You're building an assistant that helps developers with documentation. Design the routing architecture without writing implementation code yet.

**Requirements:**

1. Identify 4-5 distinct categories of documentation questions
2. For each category, define:
   - What kinds of questions it handles
   - Key characteristics of its system prompt
   - What tools it might need (conceptually)
3. Design the classification approach (LLM, rules, or hybrid)
4. Define your default/fallback strategy

**Hints:**

- Think about different types of documentation: API reference, tutorials, troubleshooting, conceptual explanations
- Consider whether questions might need code examples, links, or just explanations
- What would make each handler uniquely good at its category?

**Deliverable:** Write up your design in a markdown document with:

- Category definitions
- Handler descriptions
- Classification strategy
- Default/fallback plan
- A simple architecture diagram

This exercise prepares you for Chapter 19, where we'll implement a complete router.


## Key Takeaways

- **Routing directs inputs to specialized handlers** based on classification, allowing you to optimize each handler for its specific use case without compromising others.

- **Choose your classification strategy wisely**: LLM classification handles nuance but costs tokens and time; rule-based classification is fast and cheap but brittle. Hybrid approaches often work best.

- **Design handlers around their specialty**: Each handler should have a tailored system prompt, appropriate tools, and potentially a different model suited to its task complexity.

- **Never forget the fallback**: Default and fallback routes are your safety net. Build them first, make them robust, and test them thoroughly.

- **Start simple**: Begin with a few broad categories. It's easier to split categories later than to merge them.

## What's Next

In Chapter 19, we'll implement everything we designed in this chapter. You'll build a complete customer service router with LLM-based classification, specialized handlers for different query types, and robust fallback handling. By the end, you'll have a working routing system you can adapt for your own projects.

