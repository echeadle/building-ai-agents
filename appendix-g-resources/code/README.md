# Appendix G: Code Examples

This directory contains utility scripts for working with the resources listed in Appendix G.

## Files

### `resource_validator.py`

A utility for managing and validating the resource links from Appendix G.

**Features:**
- Validates URL formats
- Prints organized resource summaries
- Searches resources by topic
- Exports resources to markdown

**Usage:**

```bash
# Run the full demonstration
python resource_validator.py

# Or use programmatically
```

```python
from resource_validator import ResourceValidator

validator = ResourceValidator()

# Search for resources
results = validator.get_resources_by_topic("agent")
for name, url in results:
    print(f"{name}: {url}")

# Export to markdown
validator.export_to_markdown("my_resources.md")
```

**Example Output:**

```
==================================================================
APPENDIX G: RESOURCES AND FURTHER READING
==================================================================

Official Docs (5 resources)
------------------------------------------------------------------
  ✓ Claude API Documentation
     https://docs.anthropic.com

  ✓ Building Effective Agents
     https://www.anthropic.com/engineering/building-effective-agents
     
  ...
```

## Notes

- The `resource_validator.py` script uses only standard library functions
- For actual HTTP validation, add the `requests` library: `uv add requests`
- URLs are current as of January 2025 - check for updates periodically

## Extending the Code

To add URL accessibility checking:

```python
import requests

def check_url_accessibility(url: str, timeout: int = 5) -> Tuple[bool, str]:
    """Check if a URL is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"Error {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)
```

Then add this to the `ResourceValidator` class to validate links periodically.

## Use Cases

1. **Maintaining the book's resources**: Run the validator to check for broken links
2. **Creating study guides**: Export filtered resources for specific topics
3. **Building learning paths**: Organize resources for different learning goals
4. **Sharing with others**: Generate markdown summaries of key resources
