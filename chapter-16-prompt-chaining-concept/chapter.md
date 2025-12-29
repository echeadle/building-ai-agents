---
chapter: 16
title: "Prompt Chaining - Concept and Design"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 16: Prompt Chaining - Concept and Design

## Introduction

Imagine you're asked to write a comprehensive market analysis report. If you tried to do everything at once—research the market, analyze competitors, identify trends, write the executive summary, format the document—you'd likely feel overwhelmed. The quality would suffer because you're juggling too many things simultaneously.

Now imagine breaking that task into steps: first gather the data, then analyze it, then write each section, then review and polish. Each step becomes manageable, and you can verify quality at each stage before moving on. This is exactly what prompt chaining does for LLM tasks.

In Chapter 15, we introduced the five workflow patterns that form the backbone of agentic systems. Now we dive deep into the first and most fundamental pattern: **prompt chaining**. This pattern transforms complex, error-prone tasks into sequences of simple, reliable steps.

In this chapter, you'll learn how to think about prompt chaining—when to use it, how to design chains, and what makes them effective. In Chapter 17, we'll implement these concepts in working code.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain what prompt chaining is and identify tasks that benefit from it
- Decompose complex tasks into appropriate subtasks for chaining
- Design quality gates that catch errors between chain steps
- Make informed decisions about the latency-accuracy tradeoff
- Sketch the architecture of a prompt chain before implementing it

## What Is Prompt Chaining?

**Prompt chaining** is a workflow pattern where you break a complex task into a sequence of simpler steps, with each step's output becoming the next step's input. Instead of asking an LLM to do everything at once, you guide it through a pipeline of focused subtasks.

Here's the key insight: LLMs perform better on focused tasks than on sprawling ones. When you ask an LLM to "write a blog post about renewable energy, make sure it's SEO optimized, includes statistics, has a catchy title, and appeals to both experts and beginners," you're asking it to hold many constraints in mind simultaneously. Some will slip.

But if you ask it to:
1. Generate five potential angles for a renewable energy blog post
2. Choose the best angle and outline the structure
3. Write each section with a focus on clarity
4. Add relevant statistics to support key claims
5. Optimize the title and headings for SEO
6. Review for audience appropriateness

...each step becomes tractable. The LLM can focus its full "attention" on one aspect at a time.

### The Chain Metaphor

Think of prompt chaining like an assembly line in a factory. Raw materials (your initial input) enter at one end. At each station, a specialized worker performs one operation. Quality inspectors check the work between stations. A finished product (your final output) emerges at the other end.

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Input  │────▶│ Step 1  │────▶│ Step 2  │────▶│ Step 3  │────▶ Output
└─────────┘     └─────────┘     └─────────┘     └─────────┘
                     │               │               │
                     ▼               ▼               ▼
                ┌─────────┐    ┌─────────┐    ┌─────────┐
                │  Gate   │    │  Gate   │    │  Gate   │
                └─────────┘    └─────────┘    └─────────┘
```

Each step is an LLM call with a specific purpose. Each gate is a checkpoint that validates the output before passing it along.

### When to Use Prompt Chaining

Prompt chaining shines in these scenarios:

**Tasks with clear sequential phases.** If you can naturally say "first do X, then do Y, then do Z," chaining is a good fit. Examples: research → analysis → writing, or parsing → transformation → formatting.

**Tasks where quality compounds.** When errors in early steps propagate and magnify in later steps, catching them early through chaining saves significant rework. Translation workflows benefit greatly—a mistranslation in step 1 corrupts everything downstream.

**Tasks requiring different "modes" of thinking.** Creative brainstorming, analytical evaluation, and precise formatting require different prompting styles. Chaining lets you optimize each step's prompt for its specific purpose.

**Tasks where you want to inspect intermediate results.** Chaining gives you visibility into the process. If something goes wrong, you can identify exactly which step failed and why.

### When NOT to Use Prompt Chaining

Don't reach for chaining when simpler approaches work:

**Simple, atomic tasks.** If a single well-crafted prompt reliably produces good results, don't add complexity. "Summarize this paragraph" doesn't need three steps.

**Highly interactive tasks.** If the task requires frequent human input or real-time adjustment, a conversational approach may be better than a rigid chain.

**Tasks with no natural decomposition.** Some tasks are genuinely holistic. Creative writing that requires a unified voice throughout might suffer from being chopped into pieces.

> **Note:** Start with the simplest approach that might work. Add chaining only when single prompts prove unreliable or when you need the benefits chaining provides (visibility, quality gates, modularity).

## Designing a Chain: Identifying Subtasks

Good chain design starts with task decomposition. Here's a systematic approach to breaking down complex tasks.

### Step 1: Write Out the Full Task

Start by describing the complete task as you'd explain it to a human. Don't think about LLMs yet—just capture what needs to happen.

**Example: Document Translation Pipeline**

"Take a technical document in English, translate it to Spanish, ensure technical terminology is handled correctly, maintain the original formatting, and produce a final document that a Spanish-speaking engineer would find natural to read."

### Step 2: Identify Natural Break Points

Look for moments where the nature of the work changes. Ask yourself:
- Where would a human pause to check their work?
- Where does one type of expertise hand off to another?
- What intermediate artifacts would be valuable on their own?

For our translation example:
- Extract and identify technical terms (analysis)
- Create a terminology glossary with approved translations (research/decision)
- Translate the document using the glossary (transformation)
- Review for naturalness and flow (quality assurance)
- Format the final document (presentation)

### Step 3: Define Clear Inputs and Outputs

Each step needs a well-defined contract: what does it receive, and what does it produce?

| Step | Input | Output |
|------|-------|--------|
| 1. Extract Terms | Source document | List of technical terms with context |
| 2. Create Glossary | Term list | Term → Translation mapping |
| 3. Translate | Document + Glossary | Translated document (draft) |
| 4. Review | Draft translation | Revised translation with notes |
| 5. Format | Revised translation | Final formatted document |

### Step 4: Ensure Each Step Is Self-Contained

Each step should be understandable and testable in isolation. If you need to explain three previous steps to understand what step 4 is supposed to do, you've coupled things too tightly.

Good step definition:
> "Given a draft translation and a terminology glossary, review the translation for natural phrasing. Output the revised text and a list of changes made with explanations."

Poor step definition:
> "Continue improving the translation based on what we discussed."

### Step 5: Consider Step Granularity

How fine should your steps be? This is a design judgment with tradeoffs:

**More granular steps:**
- ✅ Easier to debug
- ✅ More opportunities for quality gates
- ✅ Each step is more reliable
- ❌ More API calls (cost and latency)
- ❌ More "coordination overhead" in passing context

**Fewer, larger steps:**
- ✅ Faster execution
- ✅ Less context repetition
- ✅ Simpler architecture
- ❌ Harder to pinpoint failures
- ❌ Each step is more complex and error-prone

A good rule of thumb: **each step should have one clear purpose that can be described in a single sentence**. If you need "and" to describe what a step does, consider splitting it.

## Quality Gates Between Steps

Quality gates are checkpoints between chain steps that validate outputs before passing them to the next step. They're your safety net against error propagation.

### Why Quality Gates Matter

Without quality gates, errors cascade. If step 1 produces malformed output, step 2 tries to work with garbage, produces worse garbage, and step 3 produces complete nonsense. By the end, you've wasted API calls and gotten unusable results.

Quality gates catch problems early, when they're cheapest to fix. They also provide clear signals about where your chain needs improvement.

### Types of Quality Gates

**Structural Validation**

Check that the output has the expected format:
- Is it valid JSON/XML/Markdown?
- Does it have the required fields?
- Are values within expected ranges?

```python
# Example: Validating a glossary step output
def validate_glossary(output: dict) -> bool:
    """Check that glossary has required structure."""
    if not isinstance(output, dict):
        return False
    if "terms" not in output:
        return False
    for term in output["terms"]:
        if "original" not in term or "translation" not in term:
            return False
    return True
```

**Semantic Validation**

Check that the output makes sense for the task:
- Does a summary actually capture the key points?
- Is a translation in the target language?
- Does a classification result match one of the allowed categories?

Semantic validation often requires another LLM call—using the LLM to check its own work or using a different prompt to verify.

```python
# Example: Semantic validation prompt
VALIDATION_PROMPT = """
Review this translation and verify:
1. The translation is in Spanish (not English or another language)
2. Technical terms are translated consistently
3. No sentences were accidentally omitted

Original: {original}
Translation: {translation}

Respond with JSON:
{{"is_valid": true/false, "issues": ["list of any issues found"]}}
"""
```

**Completeness Checks**

Ensure nothing was dropped or truncated:
- Does the output cover all items in the input?
- Are there any obvious gaps or missing sections?
- Did the LLM stop mid-sentence due to token limits?

```python
# Example: Completeness check
def check_completeness(input_items: list, output_items: list) -> bool:
    """Verify all input items appear in output."""
    input_set = set(item.lower() for item in input_items)
    output_mentions = " ".join(output_items).lower()
    
    missing = [item for item in input_set if item not in output_mentions]
    return len(missing) == 0
```

### Handling Gate Failures

When a quality gate fails, you have several options:

**Retry the step.** Sometimes LLMs produce inconsistent results. A simple retry (perhaps with a slightly modified prompt) often succeeds.

**Retry with feedback.** Include the validation error in a new prompt: "Your previous output was invalid because [reason]. Please try again, ensuring [specific requirement]."

**Fallback to a different approach.** If retries fail, try a different prompt strategy or break the step into smaller pieces.

**Fail gracefully.** Sometimes you need to abort the chain and report what went wrong. Better to return a partial result with clear error information than to produce garbage confidently.

```python
# Example: Gate failure handling strategy
MAX_RETRIES = 3

def execute_with_gate(step_func, validator, input_data):
    """Execute a step with validation and retry logic."""
    for attempt in range(MAX_RETRIES):
        output = step_func(input_data)
        
        if validator(output):
            return output
        
        # Retry with feedback on non-final attempts
        if attempt < MAX_RETRIES - 1:
            input_data = add_feedback(input_data, validator.last_error)
    
    raise ChainStepError(f"Step failed after {MAX_RETRIES} attempts")
```

### Gate Placement Strategy

You don't need a gate after every step. Place gates strategically:

**After steps that transform data structure.** If step 1 extracts entities into a list, validate the list before step 2 uses it.

**Before expensive steps.** If step 4 is slow or costly, gate before it to avoid wasting resources on bad input.

**At points where errors would be hard to detect later.** If a translation error in step 2 would be invisible by step 5, catch it at step 2.

**At trust boundaries.** If a step's output will be shown to users or sent to external systems, validate it.

## Trading Latency for Accuracy

Prompt chaining inherently trades speed for reliability. Each step adds:
- Network round-trip time to the API
- LLM processing time
- Any gate validation time

For a 5-step chain where each step takes 2 seconds, you're looking at 10+ seconds of execution time. A single prompt might complete in 3 seconds.

### When Latency Matters

**Interactive applications.** Users waiting for a response notice every second. A chatbot can't take 30 seconds to reply.

**High-volume processing.** If you're processing thousands of documents, 10 seconds versus 3 seconds per document is the difference between 8 hours and 28 hours.

**Time-sensitive tasks.** Real-time alerts, trading signals, or urgent notifications can't wait for multi-step processing.

### When Accuracy Matters More

**High-stakes outputs.** Legal documents, medical information, financial reports—errors are costly.

**Tasks where errors propagate.** Code generation, data transformation, anything that feeds into automated systems.

**Situations where human review is expensive.** If every output needs manual verification anyway, better for the chain to catch errors than humans.

### Finding the Right Balance

Here are strategies for managing the latency-accuracy tradeoff:

**Parallelize where possible.** If steps don't depend on each other, run them simultaneously. (We'll cover this in Chapter 20 on Parallelization.)

**Use faster models for simpler steps.** Not every step needs your most capable model. Simple extraction or formatting steps might work fine with faster, cheaper models.

**Cache repeated operations.** If you translate documents with similar terminology, cache the glossary step.

**Make chains configurable.** Allow users to choose between "fast" mode (fewer gates, less retries) and "careful" mode (all gates, multiple retries).

**Measure and optimize.** Track timing for each step. Often one or two steps dominate latency and are worth optimizing.

```python
# Example: Configurable chain quality levels
class ChainConfig:
    FAST = {
        "max_retries": 1,
        "skip_semantic_validation": True,
        "use_fast_model": True
    }
    
    BALANCED = {
        "max_retries": 2,
        "skip_semantic_validation": False,
        "use_fast_model": False
    }
    
    CAREFUL = {
        "max_retries": 3,
        "skip_semantic_validation": False,
        "use_fast_model": False,
        "additional_review_step": True
    }
```

## Prompt Chaining Architecture

Before implementing a chain, it helps to visualize its architecture. Here's a template for designing chains.

### Architecture Diagram Template

```
┌──────────────────────────────────────────────────────────────────┐
│                        CHAIN: [Name]                              │
│                                                                   │
│  Input: [What the chain receives]                                │
│  Output: [What the chain produces]                               │
│  Purpose: [One sentence describing the goal]                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: [Name]                                                  │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   [What this step receives]                              │
│  Prompt:  [Summary of the prompt's instruction]                  │
│  Output:  [What this step produces]                              │
│  Model:   [Which model to use]                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 1        │
                    │   [Validation]  │──── Fail ────▶ [Retry/Abort]
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: [Name]                                                  │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   [Output from Step 1]                                   │
│  Prompt:  [Summary of the prompt's instruction]                  │
│  Output:  [What this step produces]                              │
│  Model:   [Which model to use]                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                           [...]
                              │
                              ▼
                    ┌─────────────────┐
                    │   FINAL OUTPUT  │
                    └─────────────────┘
```

### Example: Blog Post Generation Chain

Let's design a chain for generating high-quality blog posts:

```
┌──────────────────────────────────────────────────────────────────┐
│                    CHAIN: Blog Post Generator                     │
│                                                                   │
│  Input: Topic and target audience description                    │
│  Output: Complete, polished blog post                            │
│  Purpose: Generate engaging, well-structured blog content        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Ideation                                                │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Topic + audience                                       │
│  Prompt:  Generate 5 unique angles for this topic that would     │
│           appeal to the target audience. Include a hook for each.│
│  Output:  List of 5 angles with hooks                            │
│  Model:   claude-sonnet-4-20250514                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 1        │
                    │   - Has 5 items │
                    │   - Each has    │──── Fail ────▶ Retry Step 1
                    │     hook        │
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Selection & Outline                                     │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   5 angles from Step 1                                   │
│  Prompt:  Choose the most compelling angle. Create a detailed    │
│           outline with intro, 3-5 main sections, conclusion.     │
│  Output:  Chosen angle + structured outline                      │
│  Model:   claude-sonnet-4-20250514                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 2        │
                    │   - Has 3-5     │
                    │     sections    │──── Fail ────▶ Retry Step 2
                    │   - Logical     │
                    │     flow        │
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Draft Writing                                           │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Outline from Step 2                                    │
│  Prompt:  Write the full blog post following this outline.       │
│           Use engaging prose, include examples, aim for          │
│           [word count] words.                                    │
│  Output:  Complete draft                                         │
│  Model:   claude-sonnet-4-20250514                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 3        │
                    │   - Word count  │
                    │     in range    │──── Fail ────▶ Retry Step 3
                    │   - All sections│
                    │     present     │
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: Polish & SEO                                            │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Draft from Step 3                                      │
│  Prompt:  Review and polish this draft. Improve transitions,     │
│           optimize headings for SEO, ensure engaging intro.      │
│  Output:  Final polished blog post                               │
│  Model:   claude-sonnet-4-20250514                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   FINAL OUTPUT  │
                    │   Polished Post │
                    └─────────────────┘
```

This architecture diagram tells you everything you need to implement the chain. You can see the flow, understand each step's responsibility, and know what to validate between steps.

## Designing Effective Step Prompts

Each step in a chain needs a well-crafted prompt. Here are principles for writing effective chain step prompts.

### Be Explicit About Input and Output

The LLM should know exactly what it's receiving and what it should produce.

**Weak:**
```
Improve this text.
```

**Strong:**
```
You will receive a draft blog post.

Your task is to improve the draft by:
1. Strengthening the opening hook
2. Improving transitions between sections
3. Making the conclusion more actionable

Output the improved blog post in full. Preserve the overall structure.
```

### Provide Context About the Chain

Sometimes it helps the LLM to understand where this step fits in the larger process.

```
This is step 3 of a 4-step blog post creation process.

Previous steps have already:
- Generated the topic angle (Step 1)
- Created a detailed outline (Step 2)

Your job is to write the full draft based on the outline below.
The next step will handle polishing and SEO optimization, so focus
on getting the core content right.
```

### Constrain the Output Format

If subsequent steps expect a specific format, enforce it.

```
Output your response as JSON with this exact structure:
{
    "chosen_angle": "The selected angle",
    "outline": {
        "title": "Proposed title",
        "introduction": "Brief description of the intro",
        "sections": [
            {"heading": "Section 1 heading", "key_points": ["point 1", "point 2"]},
            ...
        ],
        "conclusion": "Brief description of the conclusion"
    }
}
```

### Don't Overload a Single Step

If your prompt has many bullet points, subpoints, and exceptions, the step is doing too much. Split it.

## Common Pitfalls

### 1. Chains That Are Too Long

**Problem:** Every step adds latency and opportunities for error. A 10-step chain is slow, expensive, and fragile.

**Solution:** Consolidate steps that naturally belong together. If you have "generate title" and "generate subtitle" as separate steps, combine them. Aim for 3-5 steps for most chains.

### 2. Poor Context Passing Between Steps

**Problem:** Each step operates in isolation, leading to inconsistency. Step 3 uses a different tone than Step 1 because it doesn't know what Step 1 decided.

**Solution:** Pass relevant context forward explicitly. If Step 1 decides on a formal tone, include "Use a formal tone" in Step 3's prompt. Consider a "chain context" object that accumulates decisions.

### 3. Missing or Weak Quality Gates

**Problem:** Errors propagate through the chain, and by the final output, you can't tell what went wrong.

**Solution:** Add gates at critical points. At minimum, validate structure (is the output parseable?) and completeness (is anything obviously missing?). Log intermediate outputs for debugging.

## Practical Exercise

**Task:** Design a prompt chain for generating product descriptions for an e-commerce site.

**Requirements:**
- Input: Product name, category, key features, and target customer
- Output: A compelling product description (150-200 words) with a catchy headline
- The chain should ensure accuracy (features match the input) and appeal (language suits the target customer)

**Your design should include:**
1. A list of 3-5 steps with clear purposes
2. Input/output definitions for each step
3. At least two quality gates with specific validation criteria
4. An architecture diagram following the template shown in this chapter

**Hints:**
- Consider what aspects of a good product description might need different "thinking modes"
- Think about where errors would be most costly and place gates there
- Don't forget to validate that the final description actually mentions the product's key features

**Solution:** See `code/exercise_solution.md`

## Key Takeaways

- **Prompt chaining breaks complex tasks into sequences of simpler steps**, where each step's output feeds into the next step's input.

- **Use chaining when tasks have natural phases**, when quality compounds, when you need visibility into the process, or when different steps need different prompting strategies.

- **Design chains by decomposing the full task**, identifying clear inputs and outputs for each step, and ensuring each step has a single, clear purpose.

- **Quality gates between steps catch errors early**, preventing cascading failures. Validate structure, semantics, and completeness at strategic points.

- **Chaining trades latency for accuracy**—more steps mean slower execution but more reliable results. Configure chains to balance speed and quality based on your use case.

- **Sketch your chain architecture before implementing it.** A clear diagram of steps, gates, and data flow makes implementation straightforward.

## What's Next

You now understand the concepts and design principles behind prompt chaining. In Chapter 17, we'll turn these ideas into working code. You'll implement a complete prompt chain with quality gates, error handling, and a clean, reusable architecture. We'll build a content generation chain that creates and translates marketing copy—a practical example that demonstrates all the principles you've learned here.
