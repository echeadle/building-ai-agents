# Exercise Solution: Product Description Chain Design

## The Task

Design a prompt chain for generating product descriptions for an e-commerce site.

**Input:** Product name, category, key features, and target customer
**Output:** A compelling product description (150-200 words) with a catchy headline

## Solution

### Chain Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                CHAIN: Product Description Generator               │
│                                                                   │
│  Input: Product name, category, features list, target customer   │
│  Output: Headline + description (150-200 words)                  │
│  Purpose: Generate compelling e-commerce product descriptions    │
└──────────────────────────────────────────────────────────────────┘
```

### Step Breakdown

#### Step 1: Feature Analysis & Benefit Extraction

**Purpose:** Transform raw features into customer-focused benefits

**Why this step?** Raw features ("waterproof rating: IPX7") don't sell products. Benefits ("Never worry about rain ruining your adventure") do. This step converts technical specs into emotional appeals.

| Aspect | Definition |
|--------|------------|
| Input | Product name, category, features list, target customer profile |
| Output | List of benefits with emotional hooks, prioritized by target customer appeal |
| Model | claude-sonnet-4-20250514 |

**Prompt template:**
```
You are a product marketing expert.

Product: {product_name}
Category: {category}
Features: {features}
Target Customer: {target_customer}

For each feature, identify:
1. The core benefit to the customer
2. An emotional hook that resonates with the target customer
3. Priority ranking (1-5) based on what matters most to this customer type

Output as JSON:
{
    "benefits": [
        {
            "feature": "original feature",
            "benefit": "customer benefit",
            "emotional_hook": "emotional appeal",
            "priority": 1-5
        }
    ],
    "key_selling_point": "The single most compelling benefit"
}
```

---

#### Step 2: Headline Generation

**Purpose:** Create attention-grabbing headlines based on the key selling point

**Why this step?** Headlines are critical for e-commerce. They determine whether a customer reads further. This step focuses purely on crafting compelling headlines, using the prioritized benefits from Step 1.

| Aspect | Definition |
|--------|------------|
| Input | Product name, key selling point, top 3 benefits from Step 1 |
| Output | 3-5 headline options with different approaches |
| Model | claude-sonnet-4-20250514 |

**Prompt template:**
```
You are an e-commerce copywriter specializing in product headlines.

Product: {product_name}
Key Selling Point: {key_selling_point}
Top Benefits: {top_benefits}
Target Customer: {target_customer}

Generate 5 headline options using different approaches:
1. Benefit-focused (lead with the main benefit)
2. Problem-solution (address a pain point)
3. Aspirational (paint a desirable outcome)
4. Curiosity-driven (make them want to learn more)
5. Social proof angle (imply popularity or validation)

Each headline should be under 10 words.

Output as JSON:
{
    "headlines": [
        {"type": "benefit", "headline": "..."},
        {"type": "problem-solution", "headline": "..."},
        ...
    ],
    "recommended": "headline type",
    "recommendation_reason": "why this works best for target customer"
}
```

---

#### Step 3: Description Drafting

**Purpose:** Write the full product description using the best headline and benefits

**Why this step?** This is the main writing step. It synthesizes everything from previous steps into a cohesive, compelling description. By this point, we know what to emphasize (Step 1) and how to hook them (Step 2).

| Aspect | Definition |
|--------|------------|
| Input | Recommended headline, prioritized benefits, product details, target customer |
| Output | Complete product description (150-200 words) |
| Model | claude-sonnet-4-20250514 |

**Prompt template:**
```
Write a product description for e-commerce.

Headline: {recommended_headline}
Product: {product_name}
Category: {category}
Benefits (in priority order): {prioritized_benefits}
Original Features: {features}
Target Customer: {target_customer}

Requirements:
- Start with the headline
- Open with the key benefit in the first sentence
- Address 3-4 benefits in the body
- Include specific features as proof points for benefits
- End with a subtle call-to-action
- Total length: 150-200 words
- Tone: {appropriate_tone_for_target}

Output the complete product description.
```

---

#### Step 4: Accuracy & Polish Review

**Purpose:** Verify all features are accurately represented and polish the copy

**Why this step?** E-commerce descriptions must be accurate (legal/trust issues) and compelling. This step catches errors where benefits don't match features, and improves readability.

| Aspect | Definition |
|--------|------------|
| Input | Draft description, original features list, original requirements |
| Output | Final polished description with accuracy confirmation |
| Model | claude-sonnet-4-20250514 |

**Prompt template:**
```
Review this product description for accuracy and polish.

Draft:
{draft_description}

Original Features:
{original_features}

Requirements to verify:
1. Every claim in the description is supported by the features
2. No features are misrepresented or exaggerated
3. Word count is 150-200 words
4. All key features are mentioned

Review tasks:
1. Flag any inaccuracies or unsupported claims
2. Improve sentence flow and readability
3. Strengthen weak phrases
4. Ensure the call-to-action is clear but not pushy

Output as JSON:
{
    "accuracy_check": {
        "all_claims_supported": true/false,
        "issues": ["list of any issues"]
    },
    "word_count": number,
    "final_description": "The polished description",
    "changes_made": ["list of changes"]
}
```

---

### Quality Gates

#### Gate 1: After Feature Analysis (Step 1)

**Position:** Between Step 1 and Step 2

**Validation criteria:**
- Output is valid JSON with required structure
- At least 3 benefits are identified
- Each benefit has all required fields (feature, benefit, emotional_hook, priority)
- Priorities are numbers 1-5
- A key_selling_point is identified

**Why here?** If benefit extraction fails, the entire chain fails. Headlines and descriptions depend on having good benefits to work with. This is the foundation.

```python
def validate_benefit_analysis(output: dict) -> tuple[bool, str]:
    """Validate Step 1 output."""
    if "benefits" not in output:
        return False, "Missing 'benefits' field"
    
    if len(output["benefits"]) < 3:
        return False, f"Too few benefits: {len(output['benefits'])}, need at least 3"
    
    for i, benefit in enumerate(output["benefits"]):
        required = ["feature", "benefit", "emotional_hook", "priority"]
        missing = [f for f in required if f not in benefit]
        if missing:
            return False, f"Benefit {i+1} missing fields: {missing}"
        
        if not isinstance(benefit["priority"], int) or not 1 <= benefit["priority"] <= 5:
            return False, f"Benefit {i+1} has invalid priority: {benefit['priority']}"
    
    if "key_selling_point" not in output:
        return False, "Missing 'key_selling_point'"
    
    return True, ""
```

---

#### Gate 2: After Description Drafting (Step 3)

**Position:** Between Step 3 and Step 4

**Validation criteria:**
- Word count is between 120-250 words (allow some flexibility before final polish)
- Description starts with the provided headline
- No obvious truncation (doesn't end mid-sentence)
- All original features appear in the text (mentioned directly or via benefits)

**Why here?** Before investing in the review step, ensure the draft is structurally sound. Catching truncation or missing features here saves a wasted review step.

```python
def validate_draft_description(
    output: str, 
    headline: str, 
    features: list[str]
) -> tuple[bool, str]:
    """Validate Step 3 output."""
    # Check word count (with flexibility)
    word_count = len(output.split())
    if word_count < 120:
        return False, f"Too short: {word_count} words, need at least 120"
    if word_count > 250:
        return False, f"Too long: {word_count} words, max is 250"
    
    # Check headline is included
    if not output.strip().startswith(headline.strip()):
        return False, "Description doesn't start with the provided headline"
    
    # Check for truncation
    last_char = output.strip()[-1]
    if last_char not in '.!?"\'':
        return False, "Description appears truncated (doesn't end with sentence-ending punctuation)"
    
    # Check features are addressed
    output_lower = output.lower()
    missing_features = []
    for feature in features:
        # Check if feature or key words from feature appear
        feature_words = feature.lower().split()
        key_words = [w for w in feature_words if len(w) > 4]  # Skip small words
        
        if not any(word in output_lower for word in key_words):
            missing_features.append(feature)
    
    if len(missing_features) > len(features) // 2:  # Allow some flexibility
        return False, f"Too many features missing: {missing_features}"
    
    return True, ""
```

---

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Feature Analysis & Benefit Extraction                   │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Product name, category, features, target customer      │
│  Output:  Prioritized benefits + key selling point (JSON)        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 1        │
                    │ - Valid JSON    │
                    │ - ≥3 benefits   │──── Fail ────▶ Retry with feedback
                    │ - Key selling   │
                    │   point exists  │
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Headline Generation                                     │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Key selling point + top benefits                       │
│  Output:  5 headlines + recommendation (JSON)                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       [No gate here - low risk, 
                        validated in Step 3]
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Description Drafting                                    │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Recommended headline + benefits + features             │
│  Output:  Complete product description (150-200 words)           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   GATE 2        │
                    │ - Word count    │
                    │   120-250       │──── Fail ────▶ Retry Step 3
                    │ - Starts with   │
                    │   headline      │
                    │ - Not truncated │
                    │ - Features      │
                    │   addressed     │
                    └─────────────────┘
                              │ Pass
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4: Accuracy & Polish Review                                │
│  ─────────────────────────────────────────────────────────────── │
│  Input:   Draft + original features + requirements               │
│  Output:  Final description + accuracy confirmation              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   FINAL OUTPUT  │
                    │  - Headline     │
                    │  - Description  │
                    │  - Word count   │
                    └─────────────────┘
```

---

### Design Decisions Explained

**Why 4 steps instead of 2 or 6?**

- 2 steps (write + review) wouldn't separate the distinct thinking modes needed: analytical (features→benefits), creative (headlines), synthesis (drafting), and critical (review).
- 6+ steps would add unnecessary latency without proportional quality gains.
- 4 steps maps to natural phases: analyze → hook → write → verify.

**Why no gate after Step 2 (Headlines)?**

- Headlines are validated implicitly when used in Step 3.
- If Step 3 fails to incorporate the headline properly, Gate 2 catches it.
- Adding a gate would slow the chain without catching errors that matter.

**Why validate feature coverage in Gate 2?**

- Feature accuracy is critical for e-commerce (legal, trust).
- Catching missing features before the review step allows targeted retry.
- The review step assumes a complete draft; missing features would waste it.

---

### Extension Ideas

1. **Add A/B headline testing:** Generate multiple descriptions with different headlines, then use a voting step to select the best.

2. **Add competitor comparison:** Insert a step that compares the description to competitor products and highlights unique differentiators.

3. **Add SEO optimization:** Add a step specifically for keyword optimization if descriptions need to rank in search.

4. **Parallelize Steps 1 and 2:** If you have a standard product template, headline generation could happen in parallel with detailed benefit analysis.
