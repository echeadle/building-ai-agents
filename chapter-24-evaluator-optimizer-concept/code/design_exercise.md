# Design Exercise Solution: Product Description Evaluator-Optimizer

This document provides a sample solution for designing an evaluator-optimizer system for improving product descriptions.

---

## 1. Task Definition: What Makes a Good Product Description?

A good product description should:

- **Capture attention** in the first sentence
- **Communicate key benefits** (not just features)
- **Address the target customer's needs** directly
- **Provide enough detail** to make an informed decision
- **Include a clear call-to-action** (implicit or explicit)
- **Be scannable** for quick reading
- **Build trust** through specificity and honesty

**Example Task Prompt for the Generator:**

```
Write a compelling product description for the following product:

Product: {product_name}
Category: {category}
Key Features: {features}
Target Audience: {audience}
Tone: {tone}

The description should be 100-200 words and convince potential 
customers to consider purchasing this product.
```

---

## 2. Evaluation Criteria

I've identified five specific, measurable criteria:

### Criterion 1: HOOK QUALITY
Does the opening immediately grab attention and communicate value?

### Criterion 2: BENEFIT CLARITY
Are the top benefits clear and customer-focused (not just features)?

### Criterion 3: SPECIFICITY
Does the description include concrete details that build credibility?

### Criterion 4: READABILITY
Is the text easy to scan and understand quickly?

### Criterion 5: PERSUASION
Does it create desire and motivate action?

---

## 3. Scoring Rubrics

### HOOK QUALITY (1-5)

```
5 - Exceptional:
    - First sentence creates immediate curiosity or emotional response
    - Directly addresses a pain point or desire
    - Reader is compelled to continue reading
    
4 - Good:
    - Opening is interesting and relevant
    - Hints at value proposition
    - Reader is engaged
    
3 - Adequate:
    - Opening is clear but generic
    - Value proposition takes time to emerge
    - Reader may or may not continue
    
2 - Weak:
    - Opening is bland or unclear
    - Starts with product name or features instead of benefit
    - Reader loses interest
    
1 - Poor:
    - Opening is confusing, irrelevant, or off-putting
    - No hook whatsoever
    - Reader abandons immediately
```

### BENEFIT CLARITY (1-5)

```
5 - Exceptional:
    - Top 3 benefits crystal clear within first 50 words
    - Benefits tied directly to customer outcomes
    - Features explained in terms of what they DO for the customer
    
4 - Good:
    - Benefits are clear, though may take 50-100 words
    - Customer value is evident
    - Most features connected to benefits
    
3 - Adequate:
    - Benefits present but somewhat buried
    - Mix of features and benefits
    - Reader has to infer some value
    
2 - Weak:
    - Features dominate; benefits are secondary
    - Customer has to figure out value themselves
    - "So what?" feeling
    
1 - Poor:
    - Only features listed; no benefits
    - No connection to customer needs
    - Reads like a spec sheet
```

### SPECIFICITY (1-5)

```
5 - Exceptional:
    - Concrete numbers, materials, or results mentioned
    - Specific use cases illustrated
    - Claims feel credible and verifiable
    
4 - Good:
    - Several specific details included
    - Examples help understand the product
    - Most claims feel substantiated
    
3 - Adequate:
    - Some specific details
    - Mixed with generic claims
    - Partially credible
    
2 - Weak:
    - Mostly generic language
    - Vague claims ("high quality," "amazing")
    - Few concrete details
    
1 - Poor:
    - Entirely generic
    - No specific details whatsoever
    - Could apply to any product
```

### READABILITY (1-5)

```
5 - Exceptional:
    - Scannable in under 10 seconds
    - Short sentences and paragraphs
    - Key points jump out immediately
    
4 - Good:
    - Easy to read quickly
    - Good use of structure
    - Most key points visible
    
3 - Adequate:
    - Readable but requires attention
    - Some long sentences or dense paragraphs
    - Structure could be improved
    
2 - Weak:
    - Dense text that's hard to scan
    - Long sentences with multiple clauses
    - Key points buried
    
1 - Poor:
    - Wall of text
    - Confusing sentence structure
    - Reader struggles to extract meaning
```

### PERSUASION (1-5)

```
5 - Exceptional:
    - Creates genuine desire for the product
    - Addresses objections implicitly
    - Clear path to action
    - Emotional resonance
    
4 - Good:
    - Builds interest effectively
    - Makes a compelling case
    - Reader considers purchasing
    
3 - Adequate:
    - Informative but not compelling
    - "Nice to know" rather than "must have"
    - Neutral emotional response
    
2 - Weak:
    - Fails to build interest
    - Missing emotional connection
    - Reader is unmoved
    
1 - Poor:
    - Actively off-putting
    - Creates doubt or confusion
    - Reader less likely to buy after reading
```

---

## 4. Stopping Conditions

### Quality Threshold

**Approved if:** All criteria score 4 or higher

**Rationale:** A score of 4 ("Good") on all dimensions produces a professional-quality description that will perform well. Requiring all 5s would be unrealistic and waste tokens.

### Maximum Iterations

**Maximum:** 5 iterations

**Rationale:** 
- Most descriptions improve significantly in iterations 1-3
- After 3-4 iterations, improvements become marginal
- 5 iterations provides a buffer for complex products

### Diminishing Returns Detection

**Stop if:** Score improvement < 0.2 points average across all criteria

**Rationale:** If we're only improving by fractions of a point, further iteration isn't worth the cost.

### Convergence Detection

**Stop if:** Evaluator provides substantially similar feedback twice in a row

**Rationale:** If the same issues keep appearing, the generator may not be able to address them, or there may be conflicting requirements.

---

## 5. Failure Handling

### Scenario: Max Iterations Without Approval

**Response:**
```python
{
    "output": best_version_so_far,
    "status": "best_effort",
    "message": "Could not fully meet criteria after 5 iterations",
    "final_scores": {
        "hook_quality": 3.5,
        "benefit_clarity": 4.0,
        "specificity": 3.0,  # Struggled here
        "readability": 4.0,
        "persuasion": 3.5
    },
    "recommendation": "Consider human review for specificity improvements"
}
```

### Scenario: Criteria Impossible to Meet

**Detection:** Same criterion scores below 3 for 3+ iterations

**Response:**
- Log the problematic criterion
- Return best effort with specific warning
- Suggest criteria adjustment for future runs

### Scenario: Generator Degrades Quality

**Detection:** Score decreases after a revision

**Response:**
- Revert to previous version
- Try revision with more explicit feedback
- If it happens twice, stop and return best version

---

## 6. Example Evaluation Flow

**Product:** Wireless Noise-Canceling Headphones

### Iteration 1

**Generated Description:**
> "Our new headphones feature advanced noise cancellation technology, 40-hour battery life, and premium drivers. Available in black, silver, and blue. Compatible with all Bluetooth devices. Includes carrying case and charging cable."

**Evaluation:**
| Criterion | Score | Feedback |
|-----------|-------|----------|
| Hook Quality | 2 | Opens with "Our new headphones" - generic and feature-focused. Lead with the benefit. |
| Benefit Clarity | 2 | All features, no benefits. Why does 40-hour battery matter to ME? |
| Specificity | 4 | Good concrete details (40 hours, colors, included items) |
| Readability | 4 | Short and scannable, but list-like |
| Persuasion | 2 | No emotional connection, just facts |

**Average:** 2.8 | **Status:** needs_revision

### Iteration 2

**Generated Description:**
> "Escape into your music with studio-quality sound wherever you go. Our noise-canceling headphones block out distractions so you can focus, relax, or just enjoy the moment. With 40 hours of battery life, they'll last your entire work week on a single charge. Premium 40mm drivers deliver rich bass and crystal-clear highs. Available in black, silver, and blue with a protective carrying case included."

**Evaluation:**
| Criterion | Score | Feedback |
|-----------|-------|----------|
| Hook Quality | 4 | "Escape into your music" creates emotional connection and desire |
| Benefit Clarity | 4 | Benefits clear: focus, relax, long-lasting, great sound |
| Specificity | 4 | Kept concrete details, added "40mm drivers" |
| Readability | 4 | Good flow, easy to scan |
| Persuasion | 3 | Better but could be stronger. No urgency or social proof |

**Average:** 3.8 | **Status:** needs_revision

### Iteration 3

**Generated Description:**
> "Escape into your music with studio-quality sound wherever you go. Our noise-canceling headphones block out distractions—whether you're in a busy café, on a crowded train, or working from home—so you can focus, relax, or simply enjoy the moment. With 40 hours of battery life, they'll power through your entire work week on a single charge. Premium 40mm drivers deliver the rich bass and crystal-clear highs that audiophiles love. Choose your style in midnight black, brushed silver, or ocean blue. Every pair comes with a protective carrying case so they're ready whenever you are."

**Evaluation:**
| Criterion | Score | Feedback |
|-----------|-------|----------|
| Hook Quality | 4 | Strong emotional opening maintained |
| Benefit Clarity | 5 | Excellent—benefits tied to specific scenarios |
| Specificity | 4 | Good details, specific use cases added |
| Readability | 4 | Flows well, easy to scan |
| Persuasion | 4 | "Audiophiles love" adds credibility, "ready whenever you are" creates desire |

**Average:** 4.2 | **Status:** approved ✓

---

## 7. Additional Design Considerations

### Handling Different Product Types

Different products may need adjusted criteria weights:

| Product Type | Priority Criteria |
|--------------|-------------------|
| Tech gadgets | Specificity, Benefit Clarity |
| Fashion items | Hook Quality, Persuasion |
| Software/SaaS | Benefit Clarity, Readability |
| Luxury goods | Persuasion, Hook Quality |

### A/B Testing Integration

In production, connect evaluator scores to actual conversion data:
- Track which evaluation scores correlate with higher conversions
- Adjust rubrics based on real-world performance
- Fine-tune thresholds for different product categories

### Human Feedback Loop

For high-value products, add human review:
1. Generator creates description
2. LLM Evaluator scores it
3. If approved by LLM, route to human reviewer
4. Human can approve, edit, or request another LLM iteration

---

## Summary

This design provides:

- **Clear task definition** that guides the generator
- **Five specific criteria** with detailed rubrics
- **Multiple stopping conditions** to prevent waste
- **Failure handling** for edge cases
- **Example flow** showing the system in action

With this design complete, we're ready to implement the system in Chapter 25!
