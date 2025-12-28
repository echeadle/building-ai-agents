"""
Exercise Solution: Python Docstring Evaluator-Optimizer Design

This file contains the complete design for an evaluator-optimizer system
that generates and refines Python function documentation (docstrings).

Chapter 24: Evaluator-Optimizer - Concept and Design

Exercise Task:
1. Define at least 5 evaluation criteria for Python docstrings
2. Write the generator system prompt (for initial generation)
3. Write the evaluator system prompt
4. Define clear stopping conditions
5. Consider at least 2 edge cases and how to handle them
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# PART 1: EVALUATION CRITERIA FOR PYTHON DOCSTRINGS
# =============================================================================

@dataclass
class DocstringCriterion:
    """A criterion for evaluating Python docstrings."""
    name: str
    description: str
    priority: str
    pass_condition: str
    fail_condition: str


DOCSTRING_CRITERIA = [
    DocstringCriterion(
        name="summary_line",
        description="First line is a concise, complete sentence describing the function's purpose",
        priority="critical",
        pass_condition="First line is under 80 chars, describes what the function does (not how), ends with a period",
        fail_condition="Missing, too long, incomplete sentence, or describes implementation details"
    ),
    DocstringCriterion(
        name="parameter_documentation",
        description="All parameters are documented with name, type, and description",
        priority="critical",
        pass_condition="Every parameter has: name, type annotation, description of what it represents and any constraints",
        fail_condition="Any parameter missing documentation, or type/description unclear"
    ),
    DocstringCriterion(
        name="return_documentation",
        description="Return value is documented with type and description",
        priority="critical",
        pass_condition="Return section specifies type and explains what the returned value represents",
        fail_condition="Missing return documentation, or return type/meaning unclear"
    ),
    DocstringCriterion(
        name="exception_documentation",
        description="All raised exceptions are documented with conditions",
        priority="important",
        pass_condition="Each exception that can be raised lists: exception type and condition that triggers it",
        fail_condition="Exceptions are raised but not documented, or conditions are vague"
    ),
    DocstringCriterion(
        name="usage_example",
        description="Includes at least one working code example",
        priority="important",
        pass_condition="Example shows typical usage, is syntactically correct, and demonstrates expected output",
        fail_condition="No example, example has syntax errors, or doesn't show output"
    ),
    DocstringCriterion(
        name="format_consistency",
        description="Follows Google-style docstring format consistently",
        priority="important",
        pass_condition="Uses Args:, Returns:, Raises:, Example: sections with correct indentation",
        fail_condition="Mixed formats, wrong section names, or inconsistent indentation"
    ),
    DocstringCriterion(
        name="clarity_and_completeness",
        description="A developer unfamiliar with the code can use the function correctly",
        priority="critical",
        pass_condition="No need to read implementation to understand usage; edge cases and defaults noted",
        fail_condition="Requires reading code to understand behavior; missing important usage notes"
    ),
]


# =============================================================================
# PART 2: GENERATOR SYSTEM PROMPT
# =============================================================================

GENERATOR_SYSTEM_PROMPT = """You are an expert Python documentation writer specializing in 
clear, comprehensive docstrings. Your goal is to write documentation that helps 
developers use functions correctly without needing to read the implementation.

DOCSTRING STYLE: Google-style Python docstrings

FORMAT:
```
\"\"\"One-line summary of the function (under 80 characters).

Longer description if needed, explaining behavior, use cases, and 
important notes. This section is optional for simple functions.

Args:
    param_name (type): Description of the parameter. Include any
        constraints (e.g., "Must be positive") and defaults.
    another_param (type, optional): Description. Defaults to X.

Returns:
    type: Description of what is returned.

Raises:
    ExceptionType: Condition that causes this exception.

Example:
    >>> function_name(arg1, arg2)
    expected_output
    
    # More complex example with context
    >>> result = function_name(special_case)
    >>> print(result)
    special_output
\"\"\"
```

GUIDELINES:
1. Summary line: Active voice, describes WHAT not HOW, complete sentence
2. Parameters: Document ALL parameters including *args and **kwargs
3. Types: Use Python type annotation style (list[str], dict[str, int], etc.)
4. Examples: Must be syntactically correct and show realistic usage
5. Be specific about edge cases, defaults, and constraints
6. Don't document obvious things ("x: The x value")"""


GENERATOR_INITIAL_PROMPT_TEMPLATE = """Write a Google-style docstring for the following Python function:

```python
{function_code}
```

Analyze the function to understand:
1. What it does and why someone would use it
2. All parameters and their expected types/constraints
3. What it returns and when
4. What exceptions it might raise
5. Common usage patterns

Generate a complete, well-formatted docstring."""


GENERATOR_REFINEMENT_PROMPT_TEMPLATE = """Your previous docstring:

```
{previous_docstring}
```

Feedback to address:
{feedback}

Revise the docstring to address this feedback while maintaining correct format 
and accuracy. Focus specifically on the issues identified."""


# =============================================================================
# PART 3: EVALUATOR SYSTEM PROMPT
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = """You are a senior Python developer reviewing docstrings for 
quality and completeness. Your role is to evaluate documentation against specific 
criteria and provide actionable feedback for improvement.

Be constructive and specific. Don't just say something is wrong—explain what's 
missing and how to fix it.

OUTPUT FORMAT:
For each criterion, provide:
- Criterion name
- Verdict: PASS or NEEDS_IMPROVEMENT
- If NEEDS_IMPROVEMENT: Specific issue and suggested fix

Then provide:
- Overall verdict: APPROVED or REVISE
- If REVISE: Prioritized list of changes (most important first)"""


EVALUATOR_PROMPT_TEMPLATE = """Evaluate the following docstring for the given function.

FUNCTION:
```python
{function_code}
```

DOCSTRING TO EVALUATE:
```
{docstring}
```

EVALUATION CRITERIA:

1. SUMMARY_LINE (Critical)
   First line is a concise, complete sentence describing the function's purpose.
   Pass: Under 80 chars, describes what (not how), ends with period
   Fail: Missing, too long, incomplete, or describes implementation

2. PARAMETER_DOCUMENTATION (Critical)
   All parameters documented with name, type, and description.
   Pass: Every parameter has name, type, description, and constraints
   Fail: Any parameter missing or unclear

3. RETURN_DOCUMENTATION (Critical)
   Return value documented with type and description.
   Pass: Type and meaning of return value are clear
   Fail: Missing or unclear return documentation

4. EXCEPTION_DOCUMENTATION (Important)
   Raised exceptions documented with conditions.
   Pass: Each exception lists type and trigger condition
   Fail: Exceptions raised but not documented

5. USAGE_EXAMPLE (Important)
   At least one working code example.
   Pass: Shows typical usage, syntactically correct, shows output
   Fail: No example, syntax errors, or no output shown

6. FORMAT_CONSISTENCY (Important)
   Follows Google-style format consistently.
   Pass: Correct section names (Args:, Returns:, etc.) and indentation
   Fail: Mixed formats or wrong section names

7. CLARITY_AND_COMPLETENESS (Critical)
   Developer can use function without reading implementation.
   Pass: No code reading needed, edge cases noted
   Fail: Must read code to understand behavior

---

Evaluate each criterion, then provide overall verdict and prioritized feedback."""


# =============================================================================
# PART 4: STOPPING CONDITIONS
# =============================================================================

@dataclass
class StoppingConditions:
    """Configuration for when to stop the refinement loop."""
    
    # Primary condition: evaluator approves
    stop_on_approval: bool = True
    
    # Safety limit: maximum iterations
    max_iterations: int = 3
    
    # Diminishing returns: stop if same issues persist
    stop_on_repeated_feedback: bool = True
    
    # Quality floor: if all critical criteria pass, allow stopping
    stop_if_critical_pass: bool = True


def should_stop(
    iteration: int,
    evaluation_result: dict,
    previous_feedback: Optional[str],
    current_feedback: str,
    config: StoppingConditions
) -> tuple[bool, str]:
    """
    Determine if the refinement loop should stop.
    
    Args:
        iteration: Current iteration number (1-indexed)
        evaluation_result: Result from evaluator with 'verdict' and 'criteria_results'
        previous_feedback: Feedback from previous iteration (None if first)
        current_feedback: Feedback from current iteration
        config: Stopping condition configuration
        
    Returns:
        Tuple of (should_stop, reason)
    """
    # Check 1: Approved by evaluator
    if config.stop_on_approval and evaluation_result.get("verdict") == "APPROVED":
        return True, "approved"
    
    # Check 2: Maximum iterations reached
    if iteration >= config.max_iterations:
        return True, "max_iterations"
    
    # Check 3: Repeated feedback (plateau detection)
    if config.stop_on_repeated_feedback and previous_feedback:
        # Simple check: if feedback is very similar, we've plateaued
        if _feedback_similarity(previous_feedback, current_feedback) > 0.8:
            return True, "diminishing_returns"
    
    # Check 4: All critical criteria pass (even if not fully approved)
    if config.stop_if_critical_pass:
        critical_criteria = ["summary_line", "parameter_documentation", 
                           "return_documentation", "clarity_and_completeness"]
        criteria_results = evaluation_result.get("criteria_results", {})
        all_critical_pass = all(
            criteria_results.get(c, {}).get("verdict") == "PASS"
            for c in critical_criteria
        )
        if all_critical_pass and iteration >= 2:  # At least try once to improve
            return True, "critical_criteria_met"
    
    # Don't stop yet
    return False, ""


def _feedback_similarity(feedback1: str, feedback2: str) -> float:
    """
    Calculate similarity between two feedback strings.
    Simple implementation - production would use better comparison.
    
    Returns:
        Float between 0 and 1 indicating similarity
    """
    # Simple word overlap calculation
    words1 = set(feedback1.lower().split())
    words2 = set(feedback2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


# =============================================================================
# PART 5: EDGE CASES AND HANDLING
# =============================================================================

"""
EDGE CASE 1: Function with no parameters or return value

Problem: A function like `def clear_cache():` has no Args or Returns sections.
The docstring might look incomplete but is actually correct.

Handling:
- Generator: Detect when function has no params/returns and skip those sections
- Evaluator: Mark param/return criteria as PASS (not applicable) for such functions
- Include in evaluator prompt: "Mark criteria as PASS if not applicable to this function"

Example:
```python
def clear_cache():
    \"\"\"Clear all cached data from the application.
    
    This removes all entries from the in-memory cache. Call this when
    memory usage is high or when cached data may be stale.
    
    Note:
        This operation cannot be undone. Any data not persisted will be lost.
    \"\"\"
```


EDGE CASE 2: Complex return types (generators, context managers, etc.)

Problem: Functions returning generators, async iterators, or context managers
need special documentation that standard templates don't cover.

Handling:
- Generator prompt: Include examples for common complex types
- Evaluator: Check for appropriate documentation of iteration behavior, 
  context manager protocol, or async patterns
- Add criterion: "Documents special behavior (generator yields, context 
  manager enter/exit, async patterns) when applicable"

Example:
```python
def read_large_file(path: str) -> Iterator[str]:
    \"\"\"Read a large file line by line without loading into memory.
    
    This generator yields lines one at a time, making it suitable for
    files too large to fit in memory.
    
    Args:
        path (str): Path to the file to read.
        
    Yields:
        str: Each line from the file, with trailing newline stripped.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        
    Example:
        >>> for line in read_large_file('data.txt'):
        ...     process(line)
    \"\"\"
```


EDGE CASE 3: Private methods and internal functions

Problem: Internal functions (prefixed with _) may not need full documentation
for external users, but still need docs for maintainers.

Handling:
- Detect private functions by name prefix
- Adjust criteria: Skip "usage example" for private functions
- Evaluator prompt: "For private functions (starting with _), examples are optional
  but implementation notes are more important"


EDGE CASE 4: Decorated functions and class methods

Problem: Decorators like @property, @staticmethod, @classmethod change
how functions should be documented and called.

Handling:
- Generator: Detect decorators and adjust documentation style
- For @property: Don't document as function call, describe the attribute
- For @classmethod/@staticmethod: Note in docstring, show correct calling syntax
- Evaluator: Verify examples use correct calling syntax for decorated methods

Example for @property:
```python
@property
def is_valid(self) -> bool:
    \"\"\"Whether the configuration is valid.
    
    Returns True if all required fields are set and values are within
    acceptable ranges.
    
    Returns:
        bool: True if valid, False otherwise.
    \"\"\"
```


EDGE CASE 5: Function that modifies arguments in place

Problem: Functions that modify mutable arguments (like lists) need clear
documentation about side effects.

Handling:
- Generator prompt: "If function modifies arguments in place, document this
  clearly with a Note or Warning section"
- Evaluator: Check for side effect documentation when function modifies
  mutable parameters
- Add criterion consideration: "Side effects and argument mutations are documented"

Example:
```python
def shuffle_in_place(items: list) -> None:
    \"\"\"Randomly shuffle a list in place.
    
    Args:
        items (list): The list to shuffle. **Modified in place.**
        
    Returns:
        None: The function modifies the input list directly.
        
    Warning:
        The original list order is lost. Make a copy first if you need
        to preserve the original: `shuffled = shuffle_in_place(items.copy())`
    \"\"\"
```
"""


# =============================================================================
# COMPLETE CONFIGURATION
# =============================================================================

@dataclass
class DocstringEvaluatorOptimizerConfig:
    """Complete configuration for the docstring evaluator-optimizer."""
    
    # Criteria
    criteria: list[DocstringCriterion]
    
    # Prompts
    generator_system_prompt: str
    generator_initial_template: str
    generator_refinement_template: str
    evaluator_system_prompt: str
    evaluator_template: str
    
    # Stopping conditions
    stopping: StoppingConditions
    
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 0.3  # Lower for more consistent documentation


def get_default_config() -> DocstringEvaluatorOptimizerConfig:
    """Get the default configuration for docstring generation."""
    return DocstringEvaluatorOptimizerConfig(
        criteria=DOCSTRING_CRITERIA,
        generator_system_prompt=GENERATOR_SYSTEM_PROMPT,
        generator_initial_template=GENERATOR_INITIAL_PROMPT_TEMPLATE,
        generator_refinement_template=GENERATOR_REFINEMENT_PROMPT_TEMPLATE,
        evaluator_system_prompt=EVALUATOR_SYSTEM_PROMPT,
        evaluator_template=EVALUATOR_PROMPT_TEMPLATE,
        stopping=StoppingConditions(
            stop_on_approval=True,
            max_iterations=3,
            stop_on_repeated_feedback=True,
            stop_if_critical_pass=True
        ),
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.3
    )


# =============================================================================
# MAIN: DISPLAY THE DESIGN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXERCISE SOLUTION: PYTHON DOCSTRING EVALUATOR-OPTIMIZER DESIGN")
    print("=" * 70)
    
    config = get_default_config()
    
    print("\n" + "-" * 70)
    print("PART 1: EVALUATION CRITERIA")
    print("-" * 70)
    print(f"\nTotal criteria: {len(config.criteria)}")
    for c in config.criteria:
        print(f"\n  [{c.priority.upper()}] {c.name}")
        print(f"    {c.description}")
        print(f"    Pass: {c.pass_condition[:60]}...")
    
    print("\n" + "-" * 70)
    print("PART 2: GENERATOR SYSTEM PROMPT")
    print("-" * 70)
    print(f"\n{config.generator_system_prompt[:500]}...")
    
    print("\n" + "-" * 70)
    print("PART 3: EVALUATOR SYSTEM PROMPT")
    print("-" * 70)
    print(f"\n{config.evaluator_system_prompt[:500]}...")
    
    print("\n" + "-" * 70)
    print("PART 4: STOPPING CONDITIONS")
    print("-" * 70)
    print(f"\n  stop_on_approval: {config.stopping.stop_on_approval}")
    print(f"  max_iterations: {config.stopping.max_iterations}")
    print(f"  stop_on_repeated_feedback: {config.stopping.stop_on_repeated_feedback}")
    print(f"  stop_if_critical_pass: {config.stopping.stop_if_critical_pass}")
    
    print("\n" + "-" * 70)
    print("PART 5: EDGE CASES")
    print("-" * 70)
    print("""
    1. Functions with no parameters or return value
       → Mark inapplicable criteria as PASS
       
    2. Complex return types (generators, context managers)
       → Document yields/enter/exit behavior
       
    3. Private methods (_prefix)
       → Skip examples, focus on implementation notes
       
    4. Decorated functions (@property, @classmethod)
       → Adjust calling syntax in examples
       
    5. Functions that modify arguments in place
       → Document side effects clearly with Warning section
    """)
    
    print("\n" + "-" * 70)
    print("MODEL CONFIGURATION")
    print("-" * 70)
    print(f"\n  model: {config.model}")
    print(f"  max_tokens: {config.max_tokens}")
    print(f"  temperature: {config.temperature}")
    
    print("\n" + "=" * 70)
    print("Design complete! See Chapter 25 for implementation.")
    print("=" * 70)
