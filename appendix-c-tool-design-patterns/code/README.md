# Code Examples for Appendix C: Tool Design Patterns

This directory contains practical examples demonstrating tool design patterns and best practices.

## Files

### `tool_examples.py`
Comprehensive examples of well-designed tools covering:
- **Basic tool with validation** (`get_weather`) - Shows proper parameter handling, input validation, and structured error returns
- **Security validation** (`query_database`) - Demonstrates security checks and query sanitization
- **File operations** (`read_file`) - Path validation, encoding handling, and security constraints
- **Bulk operations** (`bulk_send_emails`) - Handling multiple items with partial success tracking
- **Complex parameters** (`create_report`) - Nested parameter validation and comprehensive error messages

Each example follows the design patterns in Appendix C and includes:
- Clear naming conventions
- Comprehensive descriptions
- Proper input validation
- Structured error returns
- Helpful error messages

**Run it:**
```bash
python tool_examples.py
```

### `tool_validator.py`
A validation utility that checks tool definitions against best practices.

Features:
- Validates tool structure (required fields)
- Checks naming conventions (snake_case, verb-based)
- Analyzes description quality (use cases, return values, limitations)
- Validates parameter schemas
- Provides actionable fixes for issues

Categories checked:
- **Structure**: Required fields present
- **Naming**: Conventions and clarity
- **Description**: Completeness and helpfulness
- **Schema**: Parameter definitions
- **Parameters**: Individual parameter quality

**Run it:**
```bash
python tool_validator.py
```

### `exercise_solution.py`
Complete solution to the tool design exercise: a task management system with three well-designed tools.

**Tools implemented:**
1. `create_task` - Create new tasks with validation
2. `update_task` - Update existing tasks with partial updates
3. `search_tasks` - Search and filter tasks flexibly

**Features demonstrated:**
- Verb-based naming
- Comprehensive descriptions with use cases
- Required vs optional parameters
- Enum constraints for fixed choices
- Clear error messages with suggestions
- Structured success/error responses
- Proper validation at multiple levels

**Run it:**
```bash
python exercise_solution.py
```

## Running All Examples

To see all examples in action:

```bash
# Individual examples
python tool_examples.py
python tool_validator.py
python exercise_solution.py

# Or run them all
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo ""
done
```

## Key Takeaways

After studying these examples, you should understand:

1. **Naming**: Use clear, verb-based names in snake_case
2. **Descriptions**: Include when to use, what it returns, and limitations
3. **Parameters**: Document thoroughly with examples and constraints
4. **Validation**: Check inputs early and provide helpful error messages
5. **Errors**: Return structured error objects with actionable suggestions
6. **Consistency**: Follow patterns across all your tools

## Using the Validator

The `tool_validator.py` is a practical tool you can use in your own projects:

```python
from tool_validator import ToolValidator

validator = ToolValidator()
issues = validator.validate(your_tool_definition)
validator.print_report()
```

Use it to check your tools before deploying them!

## Further Reading

- **Appendix C** - Complete guide to tool design patterns
- **Chapter 8** - Your first tool (basic implementation)
- **Chapter 16** - The router pattern (tool selection strategies)
- **Appendix D** - Prompt engineering for agents
