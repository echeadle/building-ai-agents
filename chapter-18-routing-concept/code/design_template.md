# Documentation Assistant Router - Design Template

Use this template to complete the Chapter 18 practical exercise.

## Overview

**Purpose:** Design a routing system for a software documentation assistant that helps developers find and understand documentation.

---

## 1. Category Definitions

Define 4-5 distinct categories for documentation questions.

### Category 1: [NAME]

- **Description:** What types of questions does this handle?
- **Example questions:**
  - 
  - 
  - 
- **Why it's distinct:** What makes this category different from others?

### Category 2: [NAME]

- **Description:** 
- **Example questions:**
  - 
  - 
  - 
- **Why it's distinct:** 

### Category 3: [NAME]

- **Description:** 
- **Example questions:**
  - 
  - 
  - 
- **Why it's distinct:** 

### Category 4: [NAME]

- **Description:** 
- **Example questions:**
  - 
  - 
  - 
- **Why it's distinct:** 

### Category 5 (Optional): [NAME]

- **Description:** 
- **Example questions:**
  - 
  - 
  - 
- **Why it's distinct:** 

---

## 2. Handler Designs

For each category, describe the specialized handler.

### Handler: [CATEGORY 1 NAME]

**System Prompt Key Points:**
- 
- 
- 

**Tools Needed:**
- 
- 

**Model Choice:** (Haiku/Sonnet/Opus and why)


### Handler: [CATEGORY 2 NAME]

**System Prompt Key Points:**
- 
- 
- 

**Tools Needed:**
- 
- 

**Model Choice:**


### Handler: [CATEGORY 3 NAME]

**System Prompt Key Points:**
- 
- 
- 

**Tools Needed:**
- 
- 

**Model Choice:**


### Handler: [CATEGORY 4 NAME]

**System Prompt Key Points:**
- 
- 
- 

**Tools Needed:**
- 
- 

**Model Choice:**

---

## 3. Classification Strategy

### Approach: [ ] LLM-only  [ ] Rules-only  [ ] Hybrid

**Rationale:** Why did you choose this approach?



### If using rules, what patterns/keywords would you check?

| Category | Keywords/Patterns |
|----------|-------------------|
|          |                   |
|          |                   |
|          |                   |

### If using LLM, write your classifier prompt:

```
System: 

```

### If using hybrid, describe the flow:

1. First, check...
2. If no match, then...
3. Fall back to...

---

## 4. Default and Fallback Strategy

### Default Handler

**When activated:** (Low confidence? No category match?)

**Behavior:**
- 
- 
- 

**System Prompt Key Points:**
- 
- 

### Fallback Handler

**When activated:** (Classifier error? Handler error?)

**Behavior:**
- 
- 

---

## 5. Architecture Diagram

Draw a simple diagram showing the flow from input to output.

```
[Input] --> [?] --> [?] --> [?] --> [Output]
            |
            +--> [Fallback path]
```

---

## 6. Edge Cases

List 3-5 tricky inputs and how your system would handle them.

| Input | Expected Category | Why? |
|-------|-------------------|------|
|       |                   |      |
|       |                   |      |
|       |                   |      |

---

## 7. Evaluation Plan

How would you measure if your routing system is working well?

- **Accuracy metric:**
- **Test cases to include:**
- **Success criteria:**

---

## Notes

Add any additional thoughts or considerations here.
