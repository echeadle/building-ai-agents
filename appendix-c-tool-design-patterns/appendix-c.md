---
appendix: C
title: "Tool Design Patterns"
date: 2024-12-09
draft: false
---

# Appendix C: Tool Design Patterns

This appendix provides practical patterns and conventions for designing tools that work well with Claude. Good tool design is critical—tools are how your agent interacts with the world, and poorly designed tools lead to confusion, errors, and unpredictable behavior.

## Tool Naming Conventions

### General Rules

**Use clear, verb-based names** that describe what the tool does:

```python
# Good
get_weather
search_documents
calculate_distance
send_email

# Bad (vague or noun-based)
weather
documents
math
email
```

**Use snake_case** for tool names:

```python
# Good
get_current_weather
calculate_mortgage_payment

# Bad
getCurrentWeather
calculate-mortgage-payment
CalculateMortgagePayment
```

**Be specific when needed**:

```python
# If you have multiple search tools
search_local_documents
search_web
search_database

# Not
search_1
search_2
search_3
```

### Naming Patterns by Category

**Data Retrieval**:
```python
get_*        # Retrieve a single item
list_*       # Retrieve multiple items
search_*     # Query-based retrieval
fetch_*      # Remote retrieval
```

**Data Modification**:
```python
create_*     # Create new resource
update_*     # Modify existing resource
delete_*     # Remove resource
set_*        # Set a value
```

**Actions**:
```python
send_*       # Send something (email, message, etc.)
calculate_*  # Perform calculation
analyze_*    # Analyze data
generate_*   # Generate new content
```

**Validation**:
```python
validate_*   # Check if something is valid
check_*      # Verify a condition
is_*         # Boolean check (returns true/false)
```

### Examples from Real Systems

```python
# File system operations
read_file
write_file
list_directory
delete_file
file_exists

# API operations
get_user_profile
update_user_settings
search_products
create_order

# Calculation tools
calculate_distance
calculate_tax
convert_currency
compute_statistics
```

## Description Writing Guide

The tool description is your chance to teach Claude when and how to use the tool. Write descriptions that Claude can understand and act on.

### Anatomy of a Good Description

```python
tools = [
    {
        "name": "search_documents",
        "description": """Search through the company's document database using keywords.
        
Use this tool when the user asks about:
- Company policies, procedures, or guidelines
- Historical decisions or meeting notes
- Technical documentation

This tool searches titles, content, and metadata. Use specific keywords for best results.
Returns up to 10 most relevant documents with excerpts.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for (e.g., 'vacation policy', 'Q3 budget')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-20, default: 10)"
                }
            },
            "required": ["query"]
        }
    }
]
```

### Description Template

```
[One-sentence summary of what the tool does]

Use this tool when [specific conditions/scenarios]:
- [Scenario 1]
- [Scenario 2]
- [Scenario 3]

[Important details about behavior, limitations, or caveats]

Returns [description of output format and what it contains]
```

### Key Principles

**1. Start with a clear summary**:
```python
# Good
"Search through the company's document database using keywords."

# Bad
"This tool allows you to search."
```

**2. Tell Claude WHEN to use the tool**:
```python
# Good
"Use this tool when the user asks about current weather conditions."

# Bad
"Gets weather data."
```

**3. Explain important behaviors**:
```python
# Good
"Returns the 10 most recent matching documents. If no matches found, returns empty list."

# Bad
"Returns documents."
```

**4. Be specific about limitations**:
```python
# Good
"Only searches documents created in the last 5 years. For historical data, use search_archive."

# Bad
"Searches documents."
```

**5. Avoid ambiguity**:
```python
# Good
"Returns weather in Celsius. Temperature range: -50 to 50. Updates every 15 minutes."

# Bad
"Returns current temperature."
```

### Common Description Mistakes

**❌ Too vague**:
```python
"description": "Gets information"
# Claude doesn't know what information or when to use it
```

**❌ Missing use cases**:
```python
"description": "Search the database"
# Claude doesn't know what's in the database or when to search
```

**❌ Unclear outputs**:
```python
"description": "Returns results"
# What kind of results? In what format?
```

**❌ Hidden requirements**:
```python
"description": "Send email"
# Needs user to be authenticated? Has rate limits? Claude won't know
```

**✅ Good complete description**:
```python
"description": """Send an email through the company email system.

Use this tool when the user explicitly asks to send an email or when it's the natural next step after drafting content.

Requirements:
- User must be authenticated (check with is_authenticated tool first)
- Rate limit: 10 emails per hour per user
- Subject and body are required; recipients can be comma-separated

Returns confirmation with message ID or error details."""
```

## Parameter Design Patterns

### Required vs Optional Parameters

**Make parameters required only if they're truly necessary**:

```python
# Good - city is required, units are optional
"input_schema": {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name (e.g., 'London', 'Tokyo')"
        },
        "units": {
            "type": "string",
            "description": "Temperature units: 'celsius' or 'fahrenheit' (default: celsius)",
            "enum": ["celsius", "fahrenheit"]
        }
    },
    "required": ["city"]
}

# Bad - forcing unnecessary parameters
"required": ["city", "units", "country_code", "include_forecast"]
```

### Parameter Types

**Use the most specific type possible**:

```python
# String with enum for fixed choices
{
    "type": "string",
    "enum": ["high", "medium", "low"],
    "description": "Priority level"
}

# Integer with constraints
{
    "type": "integer",
    "minimum": 1,
    "maximum": 100,
    "description": "Number of results (1-100)"
}

# Boolean for yes/no decisions
{
    "type": "boolean",
    "description": "Include archived items"
}

# Array for lists
{
    "type": "array",
    "items": {"type": "string"},
    "description": "List of tags to filter by"
}

# Object for complex structures
{
    "type": "object",
    "properties": {
        "start_date": {"type": "string"},
        "end_date": {"type": "string"}
    }
}
```

### Parameter Descriptions

**Every parameter needs a clear description**:

```python
# Good
{
    "query": {
        "type": "string",
        "description": "Search query. Use specific keywords for better results. Example: 'Python asyncio tutorial'"
    },
    "max_results": {
        "type": "integer",
        "description": "Maximum number of results to return. Higher values may slow down response. Range: 1-50",
        "minimum": 1,
        "maximum": 50
    }
}

# Bad
{
    "query": {
        "type": "string",
        "description": "The query"
    },
    "max_results": {
        "type": "integer",
        "description": "Results"
    }
}
```

### Default Values

**Document default values in the description**:

```python
{
    "sort_order": {
        "type": "string",
        "enum": ["asc", "desc"],
        "description": "Sort order for results (default: desc - newest first)"
    },
    "page_size": {
        "type": "integer",
        "description": "Items per page (default: 20, max: 100)",
        "minimum": 1,
        "maximum": 100
    }
}
```

> **Note:** Claude doesn't automatically know default values. You must either make the parameter optional and handle the default in your code, or document it clearly in the description.

### Complex Parameters

**For nested structures, provide examples**:

```python
{
    "filters": {
        "type": "object",
        "description": """Filter criteria as an object. Example: 
        {
            "category": "electronics",
            "price_range": {"min": 100, "max": 500},
            "in_stock": true
        }""",
        "properties": {
            "category": {"type": "string"},
            "price_range": {
                "type": "object",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"}
                }
            },
            "in_stock": {"type": "boolean"}
        }
    }
}
```

### Parameter Validation

**Use schema constraints where possible**:

```python
{
    "email": {
        "type": "string",
        "description": "Email address",
        "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    },
    "date": {
        "type": "string",
        "description": "Date in YYYY-MM-DD format",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
    },
    "priority": {
        "type": "integer",
        "description": "Priority level (1=highest, 5=lowest)",
        "minimum": 1,
        "maximum": 5
    }
}
```

## Error Return Conventions

How you return errors affects Claude's ability to handle them. Be consistent and informative.

### Error Response Structure

**Return errors as structured data, not exceptions**:

```python
def my_tool(param: str) -> dict[str, Any]:
    """Tool that returns structured errors."""
    
    # Validate input
    if not param:
        return {
            "success": False,
            "error": "parameter_missing",
            "message": "The 'param' parameter is required",
            "details": None
        }
    
    # Attempt operation
    try:
        result = some_operation(param)
        return {
            "success": True,
            "data": result,
            "error": None
        }
    except ValueError as e:
        return {
            "success": False,
            "error": "invalid_value",
            "message": str(e),
            "details": {"param": param}
        }
    except Exception as e:
        return {
            "success": False,
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(e).__name__}
        }
```

### Error Categories

**Use consistent error types**:

```python
ERROR_TYPES = {
    "invalid_input": "Input validation failed",
    "not_found": "Requested resource not found",
    "permission_denied": "Insufficient permissions",
    "rate_limited": "Too many requests",
    "timeout": "Operation timed out",
    "service_unavailable": "External service unavailable",
    "internal_error": "Internal processing error"
}
```

### Error Messages

**Make error messages actionable**:

```python
# Good - tells Claude what to do
return {
    "success": False,
    "error": "not_found",
    "message": "Document '12345' not found. Verify the document ID or try searching by title.",
    "suggestion": "Use search_documents tool to find the document by title"
}

# Bad - vague and unhelpful
return {
    "success": False,
    "error": "error",
    "message": "Failed"
}
```

### Partial Success

**Handle partial failures explicitly**:

```python
def bulk_operation(items: list[str]) -> dict[str, Any]:
    """Process multiple items, some may fail."""
    
    successful = []
    failed = []
    
    for item in items:
        try:
            result = process_item(item)
            successful.append({"item": item, "result": result})
        except Exception as e:
            failed.append({"item": item, "error": str(e)})
    
    return {
        "success": len(failed) == 0,
        "processed": len(items),
        "successful": len(successful),
        "failed": len(failed),
        "results": successful,
        "errors": failed
    }
```

## Tool Composition Patterns

### The Wrapper Pattern

**Wrap external APIs with your own tool interface**:

```python
def search_web(query: str, max_results: int = 10) -> dict[str, Any]:
    """
    Wrapper for external search API.
    
    Standardizes the interface and adds error handling.
    """
    try:
        # Call external API
        external_results = some_search_api.search(
            q=query,
            limit=max_results,
            api_key=os.getenv("SEARCH_API_KEY")
        )
        
        # Transform to consistent format
        results = [
            {
                "title": item["heading"],
                "url": item["link"],
                "snippet": item["description"]
            }
            for item in external_results
        ]
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "search_failed",
            "message": f"Search failed: {str(e)}"
        }
```

### The Facade Pattern

**Combine multiple operations into one tool**:

```python
def get_user_summary(user_id: str) -> dict[str, Any]:
    """
    Get comprehensive user information.
    
    Combines profile, activity, and preferences into one call.
    """
    try:
        profile = get_profile(user_id)
        activity = get_recent_activity(user_id)
        preferences = get_preferences(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "profile": profile,
            "activity": activity,
            "preferences": preferences
        }
    except Exception as e:
        return {
            "success": False,
            "error": "fetch_failed",
            "message": f"Could not retrieve user summary: {str(e)}"
        }
```

### The Builder Pattern

**Tools that configure complex operations**:

```python
def create_report(
    report_type: str,
    data_source: str,
    filters: dict[str, Any],
    format: str = "pdf"
) -> dict[str, Any]:
    """
    Build and generate a custom report.
    
    This tool orchestrates multiple steps:
    1. Validate configuration
    2. Fetch data
    3. Apply filters
    4. Generate report
    5. Save to file
    """
    
    # Validate
    if report_type not in ["sales", "inventory", "analytics"]:
        return {"success": False, "error": "invalid_report_type"}
    
    # Build report
    builder = ReportBuilder(report_type)
    builder.set_data_source(data_source)
    builder.apply_filters(filters)
    builder.set_format(format)
    
    # Generate
    try:
        report_path = builder.generate()
        return {
            "success": True,
            "report_type": report_type,
            "file_path": report_path,
            "format": format
        }
    except Exception as e:
        return {
            "success": False,
            "error": "generation_failed",
            "message": str(e)
        }
```

### The Validator Pattern

**Tools that check before executing**:

```python
def validate_and_send_email(
    to: str,
    subject: str,
    body: str
) -> dict[str, Any]:
    """
    Validate email parameters before sending.
    
    Checks:
    - Email format
    - Content safety
    - User permissions
    """
    
    # Validate email format
    if not is_valid_email(to):
        return {
            "success": False,
            "error": "invalid_email",
            "message": f"'{to}' is not a valid email address"
        }
    
    # Check content
    if contains_sensitive_data(body):
        return {
            "success": False,
            "error": "sensitive_content",
            "message": "Email contains sensitive information. Please review."
        }
    
    # Check permissions
    if not user_can_send_email():
        return {
            "success": False,
            "error": "permission_denied",
            "message": "User does not have email sending permissions"
        }
    
    # Send
    try:
        message_id = send_email(to, subject, body)
        return {
            "success": True,
            "message_id": message_id,
            "message": f"Email sent to {to}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": "send_failed",
            "message": str(e)
        }
```

## Real-World Tool Examples

### Example 1: Database Query Tool

```python
tools = [
    {
        "name": "query_database",
        "description": """Execute SQL queries on the product database.

Use this tool when the user asks about:
- Product inventory, prices, or details
- Sales data or statistics
- Customer orders or history

IMPORTANT: Only SELECT queries are allowed. No INSERT, UPDATE, or DELETE.

The database has these tables:
- products (id, name, price, category, stock)
- orders (id, customer_id, product_id, quantity, date)
- customers (id, name, email, created_at)

Returns query results as a list of dictionaries.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query. Example: 'SELECT name, price FROM products WHERE category = \"electronics\" LIMIT 10'"
                }
            },
            "required": ["query"]
        }
    }
]
```

### Example 2: File System Tool

```python
tools = [
    {
        "name": "read_file",
        "description": """Read contents of a file from the workspace.

Use this tool when the user asks to:
- Read or view a file
- Analyze file contents
- Extract information from a document

Supports text files up to 1MB. Binary files will return base64 encoding.
File paths are relative to the workspace root.

Returns file contents as a string.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (e.g., 'documents/report.txt', 'data/config.json')"
                },
                "encoding": {
                    "type": "string",
                    "description": "Text encoding (default: utf-8)",
                    "enum": ["utf-8", "ascii", "latin-1"]
                }
            },
            "required": ["file_path"]
        }
    }
]
```

### Example 3: API Integration Tool

```python
tools = [
    {
        "name": "get_stock_price",
        "description": """Get current stock price and basic information.

Use this tool when the user asks about:
- Current stock prices
- Stock symbols or tickers
- Basic market information

Data updates every 15 minutes during market hours.
Only works for US stock exchanges (NYSE, NASDAQ).

Returns price, change, percent change, and last update time.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')"
                }
            },
            "required": ["symbol"]
        }
    }
]
```

## Tool Design Checklist

Before deploying a tool, verify:

**Naming**:
- [ ] Name is clear and verb-based
- [ ] Name follows snake_case convention
- [ ] Name is specific enough if similar tools exist

**Description**:
- [ ] One-sentence summary is clear
- [ ] Explains when to use the tool
- [ ] Documents important behaviors
- [ ] Specifies limitations
- [ ] Describes output format

**Parameters**:
- [ ] All parameters have clear descriptions
- [ ] Required vs optional is correct
- [ ] Types are as specific as possible
- [ ] Examples provided for complex parameters
- [ ] Default values are documented

**Error Handling**:
- [ ] Returns structured error objects
- [ ] Error messages are actionable
- [ ] Different error types are distinguished
- [ ] Partial success is handled appropriately

**Testing**:
- [ ] Works with expected inputs
- [ ] Handles edge cases gracefully
- [ ] Returns consistent output format
- [ ] Error messages help Claude recover

## Common Tool Design Mistakes

### Mistake 1: Overloaded Tools

**❌ Bad - one tool does too much**:
```python
{
    "name": "manage_documents",
    "description": "Create, read, update, delete, search, or analyze documents",
    # Claude won't know which operation to use when
}
```

**✅ Good - separate tools for separate operations**:
```python
{
    "name": "create_document",
    "description": "Create a new document with specified content"
},
{
    "name": "search_documents",
    "description": "Search existing documents by keywords"
},
{
    "name": "update_document",
    "description": "Update the content of an existing document"
}
```

### Mistake 2: Vague Parameters

**❌ Bad**:
```python
{
    "data": {
        "type": "string",
        "description": "The data"
    }
}
```

**✅ Good**:
```python
{
    "search_query": {
        "type": "string",
        "description": "Keywords to search for. Use specific terms for better results. Example: 'machine learning python tutorials'"
    }
}
```

### Mistake 3: Hidden Dependencies

**❌ Bad - doesn't mention authentication requirement**:
```python
{
    "name": "send_email",
    "description": "Send an email",
    # Claude tries to use it, gets "Not authenticated" error
}
```

**✅ Good - documents requirements**:
```python
{
    "name": "send_email",
    "description": """Send an email through the user's account.

REQUIRES: User must be authenticated. Check with is_authenticated tool first.

Use this tool when the user explicitly requests to send an email."""
}
```

### Mistake 4: Ambiguous Outputs

**❌ Bad**:
```python
{
    "name": "search",
    "description": "Search for things",
    # Returns what? In what format? How many?
}
```

**✅ Good**:
```python
{
    "name": "search_articles",
    "description": """Search article database by keywords.

Returns up to 20 articles as a list of objects, each containing:
- title: Article title
- author: Author name
- date: Publication date (YYYY-MM-DD)
- excerpt: First 200 characters
- url: Full article URL

Results sorted by relevance."""
}
```

## Summary

Good tool design makes your agent more capable and reliable. Remember:

1. **Names** should be clear and verb-based
2. **Descriptions** should teach Claude when and how to use the tool
3. **Parameters** should be well-documented with examples
4. **Errors** should be structured and actionable
5. **Composition** patterns help manage complexity

When in doubt, ask yourself: "If I only had this tool description, would I know when to use this tool and how to use it correctly?" If the answer is no, improve the description.

## Further Reading

- See **Appendix D: Prompt Engineering for Agents** for how prompts interact with tools
- See **Appendix E: Troubleshooting Guide** for debugging tool issues
- See **Chapter 8: Your First Tool** for basic tool implementation
- See **Chapter 16: The Router Pattern** for advanced tool selection strategies
