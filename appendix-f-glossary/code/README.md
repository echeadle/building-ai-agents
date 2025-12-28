# Appendix F Code Examples

This directory contains utilities for working with the glossary.

## Files

### `glossary_lookup.py`
A command-line utility for quickly looking up term definitions from the glossary.

**Features:**
- Look up individual terms
- Search for terms by keyword
- Find related terms
- See which chapter introduced each concept
- Interactive mode for exploring the glossary

**Usage:**
```bash
python glossary_lookup.py
```

This will run demonstrations and then enter interactive mode where you can look up any term.

**Example queries:**
- `agent` - Core concept definition
- `agentic loop` - How the agent execution cycle works
- `tool` - What tools are and how they work
- `system prompt` - Controlling agent behavior
- `workflow` - Multi-step task patterns

**Programmatic usage:**
```python
from glossary_lookup import GlossaryLookup

glossary = GlossaryLookup()

# Look up a term
data = glossary.lookup("agent")
print(data["definition"])

# Search for terms
matches = glossary.search("tool")
print(f"Found: {matches}")

# Get related terms
related = glossary.get_related("augmented_llm")
print(f"Related: {related}")
```

## When to Use This

Use the glossary lookup utility when:
- You encounter an unfamiliar term while reading
- You need a quick reminder of what something means
- You want to explore related concepts
- You're trying to remember which chapter covered a topic

## Notes

The glossary in this utility contains the most commonly referenced terms from the book. The complete glossary in `appendix-f.md` includes:
- All terms (100+ definitions)
- Cross-references between concepts
- Common patterns and conventions
- Naming conventions used throughout the book
- Acronyms and abbreviations

For the full glossary with all terms and detailed cross-references, refer to the main `appendix-f.md` file.
