# Exercise Solution: Content Moderation System Design

## The Challenge

Design a parallel workflow for content moderation that checks posts for:
- Hate speech (with high confidence via voting)
- Spam/advertising
- Personal information exposure
- Copyright violations

## Architecture Diagram

```
                          ┌─────────────────┐
                          │  User Post      │
                          │  (input text)   │
                          └────────┬────────┘
                                   │
                                   ▼
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │ Hate Speech  │         │    Spam      │         │   Personal   │
   │   Checker    │         │   Checker    │         │     Info     │
   │  (3 voters)  │         │  (single)    │         │   Checker    │
   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
          │                        │                        │
     ┌────┼────┐                   │                        │
     │    │    │                   │                        │
     ▼    ▼    ▼                   │                        │
    V1   V2   V3                   │                        │
     │    │    │                   │                        │
     └────┼────┘                   │                        │
          │                        │                        │
          ▼                        │                        │
   ┌──────────────┐                │                        │
   │ Majority     │                │                        │
   │ Vote (2/3)   │                │                        │
   └──────┬───────┘                │                        │
          │                        │                        │
          ├────────────────────────┼────────────────────────┤
          │                        │                        │
          │                        ▼                        │
          │                 ┌──────────────┐                │
          │                 │  Copyright   │                │
          │                 │   Checker    │                │
          │                 └──────┬───────┘                │
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   Aggregator    │
                          │  (union of all  │
                          │   violations)   │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Moderation      │
                          │ Decision        │
                          └─────────────────┘
```

## Components

### 1. Hate Speech Checker (with Voting)

**Why voting?** Hate speech detection is nuanced and high-stakes. False negatives let harmful content through; false positives censor legitimate speech. Multiple perspectives increase reliability.

**Configuration:**
- 3 parallel voters with slightly varied prompts
- Majority vote (2/3) required to flag as hate speech
- If 2+ voters detect hate speech → flagged
- If 1 or 0 detect → passes (but log the dissenting opinion)

**Prompts (varied for diversity):**
```
Voter 1: "Analyze this text for hate speech. Consider language targeting 
         protected groups. Respond with HATE_SPEECH or SAFE."

Voter 2: "You are a content moderation expert. Does this text contain 
         hateful or discriminatory language? Respond HATE_SPEECH or SAFE."

Voter 3: "Review this user post. Flag any content that attacks, demeans, 
         or incites violence against groups based on protected 
         characteristics. Respond HATE_SPEECH or SAFE."
```

### 2. Spam Checker (Single Call)

**Why single call?** Spam detection is more clear-cut (promotional language, links, repetition). A single call is usually sufficient.

**Output:**
```json
{
  "is_spam": true/false,
  "spam_indicators": ["promotional language", "suspicious links", ...]
}
```

### 3. Personal Information Checker (Single Call)

**Why single call?** PII detection is pattern-based (emails, phone numbers, addresses). High accuracy with a single well-prompted call.

**Output:**
```json
{
  "has_pii": true/false,
  "pii_types": ["email", "phone_number", "address", ...],
  "redaction_suggestions": ["Replace XXX-XXX-XXXX with [PHONE]", ...]
}
```

### 4. Copyright Checker (Single Call)

**Why single call?** Copyright detection typically looks for quoted content, lyrics, specific passages. Single call with clear instructions.

**Output:**
```json
{
  "has_copyright_issues": true/false,
  "potential_violations": ["Song lyrics detected: 'Never gonna give...'", ...]
}
```

## Aggregation Strategy

### Per-Component Aggregation

**Hate Speech (Voting):**
```python
def aggregate_hate_speech_votes(votes: list[str]) -> dict:
    """Majority vote aggregation for hate speech detection."""
    hate_count = sum(1 for v in votes if v == "HATE_SPEECH")
    
    return {
        "flagged": hate_count >= 2,  # 2 out of 3 required
        "confidence": hate_count / 3,
        "votes": votes
    }
```

### Final Aggregation (Union)

Since ANY violation should flag the post, we use a union strategy:

```python
def aggregate_moderation_results(
    hate_speech: dict,
    spam: dict,
    pii: dict,
    copyright: dict
) -> dict:
    """Combine all moderation results into final decision."""
    
    violations = []
    
    if hate_speech["flagged"]:
        violations.append({
            "type": "hate_speech",
            "confidence": hate_speech["confidence"],
            "action": "reject"
        })
    
    if spam["is_spam"]:
        violations.append({
            "type": "spam",
            "indicators": spam["spam_indicators"],
            "action": "reject"
        })
    
    if pii["has_pii"]:
        violations.append({
            "type": "pii_exposure",
            "pii_types": pii["pii_types"],
            "action": "require_redaction"  # Could allow with edits
        })
    
    if copyright["has_copyright_issues"]:
        violations.append({
            "type": "copyright",
            "issues": copyright["potential_violations"],
            "action": "reject"
        })
    
    return {
        "approved": len(violations) == 0,
        "violations": violations,
        "action": "reject" if any(v["action"] == "reject" for v in violations)
                  else "require_edits" if violations 
                  else "approve"
    }
```

## Final Output Structure

```json
{
  "post_id": "abc123",
  "approved": false,
  "action": "reject",
  "violations": [
    {
      "type": "hate_speech",
      "confidence": 0.67,
      "action": "reject"
    },
    {
      "type": "pii_exposure",
      "pii_types": ["phone_number"],
      "action": "require_redaction"
    }
  ],
  "processing_time_ms": 1250,
  "checks_completed": {
    "hate_speech": true,
    "spam": true,
    "pii": true,
    "copyright": true
  }
}
```

## Failure Handling

### Scenario 1: One Voter Fails (Hate Speech)

**Strategy**: Continue with remaining votes if at least 2 complete.

```python
def handle_voting_with_failures(votes: list[str | None]) -> dict:
    """Handle partial voting results."""
    valid_votes = [v for v in votes if v is not None]
    
    if len(valid_votes) < 2:
        # Not enough votes for confidence
        return {
            "flagged": True,  # Conservative: flag for human review
            "confidence": 0,
            "status": "insufficient_votes",
            "requires_human_review": True
        }
    
    hate_count = sum(1 for v in valid_votes if v == "HATE_SPEECH")
    threshold = len(valid_votes) / 2  # Majority of available votes
    
    return {
        "flagged": hate_count > threshold,
        "confidence": hate_count / len(valid_votes),
        "votes_counted": len(valid_votes)
    }
```

### Scenario 2: One Checker Fails Entirely

**Strategy**: Mark that check as incomplete, continue with others, flag for review.

```python
def aggregate_with_failures(results: dict) -> dict:
    """Handle partial results when some checkers fail."""
    
    failed_checks = [k for k, v in results.items() if v is None]
    
    if failed_checks:
        return {
            "approved": False,  # Conservative: don't approve incomplete checks
            "action": "human_review",
            "reason": f"Checks failed: {', '.join(failed_checks)}",
            "partial_results": {k: v for k, v in results.items() if v is not None}
        }
    
    # Normal aggregation if all checks completed
    return aggregate_moderation_results(**results)
```

### Scenario 3: Timeout on Parallel Calls

**Strategy**: Set a timeout, process whatever completes, flag incomplete.

```python
# Pseudocode for timeout handling
async def run_with_timeout(checks: list, timeout_seconds: float = 5.0):
    """Run checks with timeout, return partial results."""
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*checks, return_exceptions=True),
            timeout=timeout_seconds
        )
        return results
    except asyncio.TimeoutError:
        # Return whatever completed
        return partial_results
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Parallel calls | 6 total | 3 hate speech + 3 other checks |
| Expected latency | ~1-2 seconds | Limited by slowest check |
| API cost | 6 calls per post | 3x for hate speech voting |
| Throughput | Limited by rate limits | Use batching for scale |

## Comparison: Sequential vs. Parallel

| Approach | Latency | Calls | Complexity |
|----------|---------|-------|------------|
| Sequential | ~6 seconds | 6 | Low |
| Parallel (this design) | ~1-2 seconds | 6 | Medium |

Parallel execution reduces latency by ~3-6x with the same API cost.

## Extension Ideas

1. **Add severity levels**: Instead of binary flags, return severity (low/medium/high)
2. **User reputation factor**: Weight results based on user history
3. **Language detection**: Route to language-specific models
4. **Appeal handling**: Store votes for review on appeals
5. **Learning loop**: Track false positives/negatives to improve prompts
