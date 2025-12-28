# Chapter 20: Parallelization - Concept and Design

## Code Files

This is a concept chapter, so the "code" here consists of design documents and pseudocode rather than runnable Python implementations. The actual implementations come in Chapter 21.

### Files

| File | Description |
|------|-------------|
| `exercise_design.md` | Solution to the practical exercise: content moderation system design |
| `architecture_examples.py` | Pseudocode examples showing workflow structures |

## Key Concepts in This Chapter

1. **Sectioning Pattern**: Divide work into independent subtasks, run in parallel, combine results
2. **Voting Pattern**: Run same task multiple times, aggregate for confidence
3. **Aggregation Strategies**: Majority vote, unanimous, weighted, threshold-based, union

## Next Chapter

Chapter 21 implements these concepts with working Python code using `asyncio`.
