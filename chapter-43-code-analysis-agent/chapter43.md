---
chapter: 43
title: "Project - Code Analysis Agent"
part: 6
date: 2025-01-15
draft: false
---

# Chapter 43: Project - Code Analysis Agent

## Introduction

In Chapter 42, you built a research assistant that searches the web and synthesizes information. Now, you'll build a completely different type of agent: one that reads, understands, and analyzes codebases.

Code analysis agents are valuable for several reasons. They can audit code quality, identify patterns and anti-patterns, assess technical debt, document undocumented codebases, and help new developers understand unfamiliar projects. What takes a developer hours of manual exploration—reading files, tracing dependencies, understanding architecture—an agent can do in minutes.

Here's what makes this interesting: **the patterns are exactly the same as Chapter 42, but the domain is different**. Instead of searching the web, you're exploring a file system. Instead of reading web pages, you're reading source files. Instead of synthesizing research, you're generating technical analysis.

This demonstrates a crucial insight: once you master the agent patterns, you can apply them to any domain. The tools change, but the architecture remains consistent.

By the end of this chapter, you'll have a fully functional code analysis agent that can examine a Python codebase and produce a comprehensive technical report covering structure, dependencies, code quality, patterns, and recommendations.

## Learning Objectives

By the end of this chapter, you will be able to:

- Adapt agent patterns to work with structured data (file systems and code)
- Build a tool suite for code exploration and analysis
- Implement code parsing and pattern detection programmatically
- Generate technical documentation from automated analysis
- Apply the same agentic patterns across different domains

## Project Requirements

Let's define what our code analysis agent needs to do.

### Functional Requirements

**The code analysis agent must:**

1. **Accept a codebase path and analysis goals**
   - "Analyze the structure of this Flask application"
   - "Identify code quality issues in this project"
   - "Document the main components and their relationships"

2. **Explore the codebase structure**
   - List directories and files
   - Identify entry points and key modules
   - Understand the project layout

3. **Read and parse source files**
   - Extract code from Python files
   - Identify functions, classes, and their purposes
   - Handle different file types appropriately

4. **Analyze code patterns and quality**
   - Detect common patterns (factory, singleton, etc.)
   - Identify potential issues (long functions, deep nesting, etc.)
   - Assess code organization and structure

5. **Track dependencies and relationships**
   - Extract import statements
   - Map module dependencies
   - Identify external vs internal dependencies

6. **Organize findings progressively**
   - Save discoveries as exploration continues
   - Build understanding incrementally
   - Track what has been analyzed

7. **Generate comprehensive analysis reports**
   - Structure findings logically
   - Provide actionable recommendations
   - Include code examples where relevant
   - Present information in a developer-friendly format

### Non-Functional Requirements

**The code analysis agent must also:**

- **Be safe**: Never execute code, only read and analyze
- **Be efficient**: Don't read every file—focus on relevant ones
- **Be thorough**: Cover key aspects without getting lost in minutiae
- **Handle errors**: Deal with unparseable files, permission issues, etc.
- **Be observable**: Log what it's analyzing and why

## Design Overview

The code analysis agent follows the same architecture as the research assistant, adapted for file system exploration.

### The Analysis Loop

Code analysis is an iterative discovery process:

```
1. Understand the analysis request
2. Explore the codebase structure
3. Read relevant source files
4. Analyze code patterns and quality
5. Track findings and relationships
6. Assess if more analysis is needed
   ↓ Yes? Go to step 2
   ↓ No? Go to step 7
7. Synthesize findings into a report
```

Just like research, the LLM decides what to explore next based on what it finds.

### Tool Suite

Our agent needs five core tools:

**1. `list_directory`**
- Lists files and subdirectories in a path
- Returns file names, types, and sizes
- The agent uses this to explore structure

**2. `read_file`**
- Reads the contents of a source file
- Returns the full code with line numbers
- The agent uses this to examine code

**3. `analyze_imports`**
- Extracts import statements from a Python file
- Returns dependencies and their types (stdlib, third-party, local)
- The agent uses this to map relationships

**4. `find_pattern`**
- Searches code for patterns using regex
- Returns matches with context
- The agent uses this for targeted searches

**5. `save_finding`**
- Saves an analysis finding
- Organizes insights as analysis progresses
- The agent uses this to remember discoveries

### State Management

The agent maintains three pieces of state:

```python
{
    "analysis_goal": "The original request",
    "findings": [
        {"category": "structure", "insight": "what we discovered"},
        {"category": "quality", "insight": "issues found"},
        ...
    ],
    "files_analyzed": ["path1.py", "path2.py", ...]
}
```

This allows the agent to track what it has analyzed and build knowledge progressively.

### Workflow Logic

The key difference from web research: **files are local and safe**. We don't need rate limiting or external API keys. But we do need to:

- Handle file permissions and errors
- Parse Python code correctly
- Avoid analyzing irrelevant files (tests, migrations, etc.)
- Focus on the most important parts of the codebase

## Building the Tool Suite

Let's implement each tool. These are real, working functions that interact with the file system and parse Python code.

### Tool 1: Directory Listing

See `code/list_directory_tool.py` for the implementation.

This tool explores the file system, filtering out irrelevant directories and identifying important files.

**Key features:**
- Recursively lists directories up to a configurable depth
- Filters out `.git`, `__pycache__`, `node_modules`, etc.
- Returns file sizes to help the agent prioritize
- Includes file type information

### Tool 2: File Reading

See `code/read_file_tool.py` for the implementation.

This tool reads source files safely with proper error handling.

**Key features:**
- Reads text files with UTF-8 encoding
- Adds line numbers for reference
- Handles encoding errors gracefully
- Limits file size to prevent memory issues
- Sanitizes paths to prevent directory traversal

### Tool 3: Import Analysis

See `code/analyze_imports_tool.py` for the implementation.

This tool extracts and categorizes import statements from Python files.

**Key features:**
- Parses both `import` and `from ... import` statements
- Distinguishes stdlib, third-party, and local imports
- Uses the AST module for reliable parsing
- Handles import aliases
- Returns structured dependency information

### Tool 4: Pattern Finding

See `code/find_pattern_tool.py` for the implementation.

This tool searches for code patterns using regular expressions.

**Key features:**
- Searches specific files or entire directories
- Returns matches with surrounding context
- Supports standard regex patterns
- Useful for finding TODOs, deprecated patterns, etc.
- Configurable context lines for better understanding

### Tool 5: Finding Storage

See `code/save_finding_tool.py` for the implementation.

This tool manages the agent's discoveries.

**Key features:**
- Categorizes findings (structure, quality, patterns, etc.)
- Stores findings in memory during analysis
- Retrieves findings by category
- Returns findings for report generation

## The Complete Code Analysis Agent

Now let's build the full agent that brings everything together.

See `code/code_analysis_agent.py` for the complete implementation.

### System Prompt Design

The system prompt is critical. It must guide the agent to:

1. **Explore strategically**: Start with project structure, then dive into key files
2. **Analyze thoroughly**: Look at code quality, patterns, dependencies
3. **Be selective**: Don't read every file—focus on the most important ones
4. **Save findings progressively**: Use `save_finding` to track discoveries
5. **Know when to stop**: Generate a report after sufficient analysis

Here's an excerpt from the system prompt:

```python
SYSTEM_PROMPT = """You are a code analysis agent. Your job is to examine Python codebases and provide comprehensive technical analysis.

ANALYSIS STRATEGY:

1. START BROAD: Use list_directory to understand the project structure
   - Identify the project type (web app, library, CLI tool, etc.)
   - Locate key files (main.py, app.py, __init__.py, etc.)
   - Note the organization (monolithic, modular, etc.)

2. ANALYZE STRUCTURE: Read key files to understand the architecture
   - Entry points and main components
   - Directory organization and purpose
   - Configuration files

3. EXAMINE DEPENDENCIES: Use analyze_imports to map relationships
   - External dependencies
   - Internal module structure
   - Circular dependencies or coupling issues

4. ASSESS QUALITY: Read representative files for code quality
   - Function and class design
   - Code complexity and readability
   - Error handling and edge cases
   - Documentation quality

5. IDENTIFY PATTERNS: Look for design patterns and anti-patterns
   - Common patterns used (factory, singleton, etc.)
   - Architectural patterns (MVC, layered, etc.)
   - Anti-patterns or code smells

6. SAVE FINDINGS: Use save_finding throughout the analysis
   - Save insights in appropriate categories
   - Be specific with examples and line numbers
   - Note both strengths and weaknesses

7. GENERATE REPORT: When you have sufficient findings, create a comprehensive report

IMPORTANT GUIDELINES:

- Don't read every file. Focus on the most important ones.
- Skip test files, migrations, and configuration unless specifically relevant
- Save findings as you go—don't try to remember everything
- Be specific: cite files, line numbers, and code examples
- Balance depth and breadth: cover major aspects without getting lost in details
- Maximum 15 iterations. If you hit this limit, generate a report with what you have.
"""
```

### The Agent Loop

The agent loop is nearly identical to Chapter 42's research assistant:

```python
def run_agent(codebase_path: str, analysis_goal: str, max_iterations: int = 15):
    """Run the code analysis agent."""
    
    # Initialize state
    state = {
        "analysis_goal": analysis_goal,
        "codebase_path": codebase_path,
        "findings": [],
        "files_analyzed": []
    }
    
    # Build message history
    messages = [
        {
            "role": "user",
            "content": f"Analyze this codebase: {codebase_path}\n\nGoal: {analysis_goal}"
        }
    ]
    
    # Agentic loop
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{max_iterations}")
        print(f"{'='*60}")
        
        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )
        
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})
        
        # Check if Claude is done (no tool calls)
        if response.stop_reason == "end_turn":
            # Extract final report from response
            final_text = extract_text_from_response(response)
            return final_text
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        # Add tool results to messages
        messages.append({"role": "user", "content": tool_results})
    
    # If we hit max iterations, ask for a report
    messages.append({
        "role": "user",
        "content": "You've reached the maximum iterations. Please generate your analysis report now based on the findings you've collected."
    })
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=messages
    )
    
    return extract_text_from_response(response)
```

The pattern is the same: loop, call Claude, execute tools, repeat until done.

## How It Works: A Walkthrough

Let's trace through what happens when you ask: "Analyze the structure and code quality of this Flask application"

**Iteration 1:**
- Claude receives the request and the codebase path
- Calls `list_directory(path)` to see the top-level structure
- Gets back: `app.py`, `models.py`, `routes/`, `templates/`, `requirements.txt`, etc.
- Recognizes this as a Flask web application

**Iteration 2:**
- Claude identifies `app.py` as the likely entry point
- Calls `read_file("app.py")`
- Gets back the full code with Flask app initialization
- Sees routes are organized in a `routes/` directory

**Iteration 3:**
- Claude wants to understand the app structure better
- Calls `list_directory("routes/")`
- Sees `auth.py`, `api.py`, `main.py` - clearly separated by concern

**Iteration 4:**
- Claude calls `save_finding()`:
  - Category: "structure"
  - Insight: "Well-organized Flask app with routes separated by concern into individual modules (auth, api, main). Clear separation of concerns."

**Iteration 5:**
- Claude examines dependencies
- Calls `analyze_imports("app.py")`
- Gets back: Flask, SQLAlchemy, various local modules
- Sees standard Flask architecture

**Iteration 6:**
- Claude reads `models.py` to check data model design
- Calls `read_file("models.py")`
- Examines class definitions and relationships

**Iteration 7:**
- Claude spots potential quality issues in `models.py`
- Calls `save_finding()`:
  - Category: "quality"
  - Insight: "Large User class (150 lines) in models.py (lines 45-195). Consider splitting authentication logic into a separate UserAuth mixin."

**Iterations 8-12:**
- Claude examines route handlers for quality
- Checks for error handling patterns
- Looks for SQL injection risks
- Identifies good and bad practices
- Saves findings in appropriate categories

**Iteration 13:**
- Claude calls `find_pattern()` to search for "TODO" comments
- Finds 3 instances in different files
- Saves as a finding about incomplete work

**Iteration 14:**
- Claude decides it has enough information
- Doesn't call any tools
- Generates comprehensive analysis report

**Result:**
A structured technical analysis with:
- Project overview and structure
- Architecture assessment
- Code quality findings (good and bad)
- Dependency analysis
- Security considerations
- Recommendations for improvement
- Specific examples with file names and line numbers

## Running the Agent

Here's how to use the code analysis agent:

```python
# Analyze a codebase
result = run_agent(
    codebase_path="/path/to/project",
    analysis_goal="Analyze structure and identify code quality issues"
)

print(result)
```

The agent will explore the codebase, analyze the code, and produce a detailed report.

## Example Analysis Report

Here's what the agent might produce for a Flask application:

```markdown
# Code Analysis Report

## Project Overview

**Type:** Flask Web Application
**Structure:** Modular organization with clear separation of concerns
**Size:** ~2,500 lines of Python code across 15 modules

## Architecture Assessment

### Strengths

1. **Well-Organized Routes**
   - Routes separated by concern (auth, api, main)
   - Blueprint pattern used correctly
   - Clear file: routes/auth.py, routes/api.py, routes/main.py

2. **Clean Separation**
   - Models isolated in models.py
   - Business logic separated from routes
   - Configuration in separate config.py

3. **Standard Flask Patterns**
   - Factory pattern for app creation
   - Extension initialization done correctly
   - Blueprints registered in app/__init__.py

### Areas for Improvement

1. **Large Model Classes**
   - User class is 150 lines (models.py lines 45-195)
   - Recommendation: Extract authentication logic to UserAuth mixin
   - Recommendation: Split user preferences into UserProfile class

2. **Inconsistent Error Handling**
   - Some routes use try/except, others don't
   - Example: auth/login.py (line 34) handles errors, but api/users.py (line 67) doesn't
   - Recommendation: Implement consistent error handling decorator

3. **Missing Input Validation**
   - Direct use of request.form without validation in several routes
   - Example: routes/api.py lines 112-125
   - Recommendation: Add form validation using Flask-WTF

## Code Quality Findings

### Good Practices

- ✅ Type hints used consistently
- ✅ Docstrings on most functions
- ✅ Environment variables for configuration
- ✅ Password hashing with werkzeug.security

### Code Smells

1. **Long Functions**
   - process_payment() in routes/api.py is 85 lines (lines 200-285)
   - Recommendation: Break into smaller functions

2. **Deeply Nested Logic**
   - 4-level nested if statements in auth/register.py (lines 45-78)
   - Recommendation: Use early returns to reduce nesting

3. **Duplicate Code**
   - Similar validation logic in 3 route files
   - Recommendation: Create shared validation utilities

## Dependency Analysis

### External Dependencies

- Flask 2.0.1
- SQLAlchemy 1.4.25
- Werkzeug 2.0.2
- 12 other packages in requirements.txt

### Observations

- All dependencies are pinned (good for reproducibility)
- No dev dependencies separated from production
- Recommendation: Use requirements-dev.txt for testing tools

### Internal Structure

```
app/
├── __init__.py (app factory)
├── models.py (database models)
├── routes/ (organized by concern)
│   ├── auth.py
│   ├── api.py
│   └── main.py
└── utils.py (helper functions)
```

- Clear module boundaries
- No circular imports detected
- Good use of relative imports

## Security Considerations

### Strengths

- ✅ Password hashing implemented
- ✅ CSRF protection enabled
- ✅ SQL injection protected by SQLAlchemy ORM

### Potential Issues

- ⚠️ No rate limiting on login endpoint (auth/login.py line 28)
- ⚠️ Passwords requirements not enforced programmatically
- ⚠️ Session secret loaded from environment but no validation

## Recommendations

### Immediate (High Priority)

1. Add rate limiting to authentication endpoints
2. Implement consistent error handling across all routes
3. Add input validation to all form handlers

### Short Term (Medium Priority)

1. Refactor large model classes
2. Break up long functions into smaller, testable pieces
3. Extract duplicate validation logic
4. Add comprehensive error logging

### Long Term (Lower Priority)

1. Consider adding automated tests (none found)
2. Implement API versioning
3. Add OpenAPI/Swagger documentation
4. Consider splitting into microservices as complexity grows

## Summary

This is a well-structured Flask application with clear organization and good separation of concerns. The codebase follows Flask best practices in most areas. Main areas for improvement are code quality (long functions, duplication) and security hardening (rate limiting, validation). The modular structure provides a solid foundation for future growth.

**Overall Assessment:** B+ (Good, with clear path to excellent)
```

## Common Pitfalls

**1. Reading Too Many Files**

**Problem:** The agent tries to read every single file in the codebase.

**Why it happens:** No guidance on file selection strategy.

**Solution:** Instruct the agent to prioritize:
```python
"Focus on key files first:
- Entry points (main.py, app.py, __init__.py)
- Core business logic
- Large or complex files
Skip: tests, migrations, generated code, vendor directories"
```

**2. Shallow Analysis**

**Problem:** The agent just describes what it sees without analyzing quality.

**Why it happens:** Unclear analysis criteria in the prompt.

**Solution:** Provide specific analysis dimensions:
```python
"For each file you read, assess:
- Code quality: readability, complexity, structure
- Best practices: follows conventions, proper error handling
- Potential issues: smells, anti-patterns, security risks
- Documentation: comments, docstrings, clarity"
```

**3. Generic Findings**

**Problem:** Findings are vague: "The code could be improved."

**Why it happens:** No examples of specific findings.

**Solution:** Demonstrate specificity:
```python
"Save specific, actionable findings with examples:
✓ 'User class in models.py (lines 45-195) is 150 lines. Extract authentication to UserAuth mixin.'
✓ 'No input validation in routes/api.py (lines 112-125). Add Flask-WTF validation.'
✗ 'Code quality could be better' (too vague)"
```

**4. Inconsistent Report Structure**

**Problem:** Reports vary wildly in organization and content.

**Why it happens:** No report template provided.

**Solution:** Include a report structure in the system prompt:
```python
"Generate your report using this structure:
1. Project Overview (type, size, structure)
2. Architecture Assessment (strengths and weaknesses)
3. Code Quality Findings (good practices and smells)
4. Dependency Analysis (external and internal)
5. Security Considerations
6. Recommendations (prioritized)
7. Summary and overall assessment"
```

## Practical Exercise

**Task:** Add a refactoring suggestion feature to the code analysis agent

The agent should identify specific refactoring opportunities and suggest concrete improvements with code examples.

**Requirements:**
1. Add a new tool: `suggest_refactoring(file_path, issue_description)`
2. When the agent finds code smells, it should generate refactoring suggestions
3. Suggestions should include before/after code examples
4. Add a "Refactoring Opportunities" section to the report

**Hints:**
- Use the existing pattern detection and file reading tools
- Create a prompt that guides Claude to generate specific refactoring advice
- Store refactoring suggestions in a new findings category
- You might want to use Claude to generate the refactored code

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Same patterns, different domain**: The agentic loop, tools, and state management patterns from Chapter 42 work perfectly for code analysis—only the tools change

- **File system exploration is safer than web scraping**: No rate limits, no external dependencies, predictable structure—but you still need error handling

- **Parsing beats regex for structured code**: Using Python's `ast` module is more reliable than regex for extracting code structure

- **Strategic reading is essential**: Don't read every file—guide the agent to prioritize important files and skip irrelevant ones

- **Specific findings beat generic observations**: "User class is 150 lines (lines 45-195)" is more valuable than "some classes are too large"

- **Domain expertise in prompts matters**: The system prompt encodes code analysis best practices that guide the agent's assessment

- **Reports need structure**: A consistent report template ensures useful, actionable output

## What's Next

You've now built two complete capstone projects: a research assistant and a code analysis agent. Both use the same fundamental patterns—agentic loops, tools, state management—but solve completely different problems.

This demonstrates the power of the patterns you've learned. They're not specific to one domain; they're general-purpose building blocks for any agentic system.

In Chapter 44, we'll build one more capstone project: a personal productivity agent that helps manage tasks, calendar events, and notes. This will show how agents can work with personal data and provide context-aware assistance.