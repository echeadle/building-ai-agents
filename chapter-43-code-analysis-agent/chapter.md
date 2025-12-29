---
chapter: 43
title: "Project - Code Analysis Agent"
part: 6
date: 2025-01-15
draft: false
---

# Chapter 43: Project - Code Analysis Agent

## Introduction

In Chapter 42, you built a research assistant that investigates topics and synthesizes information from the web. Now you'll build something different: a code analysis agent that explores codebases, understands their structure, and generates comprehensive analysis reports—all without human guidance.

Code review and analysis is perfect for AI agents. Reading thousands of lines of code, identifying patterns, tracking dependencies, and assessing quality are tasks that take humans hours or days. An agent can do this in minutes, systematically exploring every corner of a codebase while maintaining focus and consistency that humans struggle to sustain.

Here's what makes a code analysis agent compelling: it performs the tedious, mechanical work of code review—checking structure, finding patterns, analyzing imports—while freeing human developers to focus on the creative aspects of improving code quality. Ask it to "analyze this codebase for security issues," and it will explore the directory structure, read relevant files, identify patterns, and produce a detailed report with specific findings and recommendations.

This chapter walks through building a complete code analysis agent: requirements, tool design, the agentic workflow, and the full implementation. By the end, you'll have an agent that can analyze any Python codebase and understand how to adapt these patterns to other code analysis tasks.

## Learning Objectives

By the end of this chapter, you will be able to:

- Design a specialized tool suite for code analysis tasks
- Implement strategic exploration patterns for large codebases
- Build an agent that understands code structure progressively
- Generate structured analysis reports with categorized findings
- Deploy an agent that provides real value for code review tasks

## Project Requirements

Let's define what our code analysis agent needs to do. Unlike the research assistant which queries external APIs, this agent works entirely with local filesystems.

### Functional Requirements

**The code analysis agent must:**

1. **Accept analysis requests with specific goals**
   - "Analyze this codebase for security vulnerabilities"
   - "Review code quality and identify technical debt"
   - "Generate an architecture overview for new developers"
   - "Analyze dependencies and identify coupling issues"

2. **Explore directory structure intelligently**
   - List files and directories
   - Filter out irrelevant paths (`.git`, `__pycache__`, `node_modules`)
   - Identify entry points and key modules
   - Understand project type (web app, CLI tool, library)

3. **Read and analyze source code**
   - Read files with line numbers for reference
   - Support reading specific line ranges
   - Handle encoding issues gracefully
   - Reject binary files and enforce size limits

4. **Analyze code structure and patterns**
   - Extract import statements
   - Categorize dependencies (stdlib, third-party, local)
   - Find specific patterns using regex
   - Track code quality indicators

5. **Maintain organized findings**
   - Save discoveries by category (structure, quality, security, dependencies)
   - Support severity levels (info, warning, error)
   - Retrieve findings for report generation
   - Provide actionable recommendations

6. **Generate comprehensive reports**
   - Synthesize all findings into coherent analysis
   - Include specific file references and line numbers
   - Prioritize findings by importance
   - Suggest concrete improvements

### Non-Functional Requirements

**The agent should:**

- **Be efficient**: Don't read every file—focus on representative samples
- **Be adaptable**: Handle different project structures and types
- **Be safe**: Never modify files, only read and analyze
- **Be informative**: Provide context and examples for findings
- **Be respectful**: Honor `.gitignore` patterns and exclude generated code

### What the Agent Should NOT Do

**Important boundaries:**

- ❌ **Don't execute code**: Static analysis only, never run user code
- ❌ **Don't modify files**: Read-only access to prevent accidents
- ❌ **Don't read secrets**: Skip `.env`, `credentials.json`, etc.
- ❌ **Don't read everything**: Use strategic sampling, not exhaustive reading
- ❌ **Don't hallucinate**: Only report findings actually observed in the code

## Tool Design

The code analysis agent needs five specialized tools. Each has a focused purpose.

### Tool 1: `list_directory` - Explore Structure

**Purpose**: Understand directory structure without reading files.

**Parameters**:
- `path` (string, required): Directory to list
- `max_depth` (integer, optional): How deep to recurse (default: 1)

**Returns**:
```python
{
    "path": "./myproject",
    "directories": ["src/", "tests/", "docs/"],
    "code_files": ["src/main.py", "src/utils.py"],
    "other_files": ["README.md", "setup.py"],
    "total_size": "245KB",
    "file_count": 45
}
```

**Why this tool?**
Agents need to understand project structure before diving into files. This tool filters out noise (`.git`, `__pycache__`) and categorizes what's important.

### Tool 2: `read_file` - Read Source Code

**Purpose**: Read file contents with line numbers for reference.

**Parameters**:
- `path` (string, required): File to read
- `start_line` (integer, optional): First line to read
- `end_line` (integer, optional): Last line to read

**Returns**:
```python
{
    "path": "src/main.py",
    "content": "1: import os\n2: import sys\n...",
    "total_lines": 150,
    "size": "4.2KB",
    "encoding": "utf-8"
}
```

**Why this tool?**
Line numbers are critical for code analysis—findings need specific references. The optional range parameters prevent reading huge files entirely.

### Tool 3: `analyze_imports` - Extract Dependencies

**Purpose**: Parse Python files to extract import statements.

**Parameters**:
- `path` (string, required): Python file to analyze

**Returns**:
```python
{
    "stdlib_imports": [
        {"module": "os", "line": 1},
        {"module": "sys", "line": 2}
    ],
    "third_party_imports": [
        {"module": "anthropic", "line": 4},
        {"module": "flask", "line": 5}
    ],
    "local_imports": [
        {"module": "utils", "line": 7},
        {"module": "config", "line": 8}
    ]
}
```

**Why this tool?**
Understanding dependencies is crucial for architecture analysis. Categorizing imports (stdlib vs third-party vs local) reveals coupling and external dependencies.

### Tool 4: `find_pattern` - Search for Patterns

**Purpose**: Search for regex patterns in code files.

**Parameters**:
- `pattern` (string, required): Regex pattern to find
- `path` (string, required): File or directory to search
- `context_lines` (integer, optional): Lines of context around matches

**Returns**:
```python
{
    "pattern": "TODO|FIXME",
    "matches": [
        {
            "file": "src/main.py",
            "line": 45,
            "match": "# TODO: Add error handling",
            "context": "Lines 43-47 showing surrounding code"
        }
    ],
    "total_matches": 12
}
```

**Why this tool?**
Patterns reveal code quality issues: TODOs, hardcoded secrets, deprecated patterns, security anti-patterns. Regex gives flexibility for any pattern.

### Tool 5: `save_finding` - Store Discoveries

**Purpose**: Categorize and save analysis findings.

**Parameters**:
- `category` (string, required): One of: structure, quality, patterns, dependencies, security, documentation, recommendations
- `finding` (string, required): The discovery to save
- `severity` (string, optional): info, warning, or error
- `file_reference` (string, optional): File and line number

**Returns**:
```python
{
    "category": "security",
    "finding": "Hardcoded API key found in config.py:23",
    "severity": "error",
    "saved": true
}
```

**Why this tool?**
Findings need organization for report generation. Categories and severity levels help prioritize and structure the final analysis.

## The Agentic Workflow

Unlike the research assistant which follows a linear search → read → synthesize flow, code analysis requires strategic exploration. The agent must decide what to read based on what it's learned.

### Workflow Phases

**Phase 1: Initial Exploration (Iterations 1-3)**

1. Start with directory listing at the root
2. Identify project type (web app, CLI tool, library, etc.)
3. Locate entry points (main.py, app.py, __init__.py)
4. Understand high-level structure

**Phase 2: Targeted Reading (Iterations 4-10)**

1. Read key files identified in Phase 1
2. Analyze imports to understand dependencies
3. Identify important modules and classes
4. Look for configuration and setup files

**Phase 3: Pattern Analysis (Iterations 11-15)**

1. Search for quality indicators (TODO comments, duplicated code)
2. Look for security patterns (hardcoded secrets, SQL injection risks)
3. Analyze code organization patterns
4. Identify areas needing improvement

**Phase 4: Synthesis and Reporting (Final Iterations)**

1. Retrieve all saved findings
2. Prioritize by category and severity
3. Generate structured report with recommendations
4. Provide specific file references and line numbers

### Key Workflow Patterns

**Strategic Sampling, Not Exhaustive Reading**

The agent doesn't read every file. It:
- Reads entry points and main modules first
- Samples representative files from each directory
- Skips tests, generated code, and vendored dependencies
- Focuses on files that reveal architecture and patterns

**Progressive Understanding**

Early iterations build context for later analysis:
- First: "What kind of project is this?"
- Then: "What are the main components?"
- Later: "What patterns and issues exist?"
- Finally: "What recommendations can I make?"

**Organized Information Storage**

Findings are categorized as discovered:
```python
# During exploration
save_finding(
    category="structure",
    finding="Flask web application with blueprints architecture",
    severity="info"
)

# During quality analysis
save_finding(
    category="quality",
    finding="Circular import between auth.py and models.py",
    severity="warning",
    file_reference="src/auth.py:15"
)

# During security review
save_finding(
    category="security",
    finding="Hardcoded database password",
    severity="error",
    file_reference="config.py:12"
)
```

## Implementation

Let's build the complete code analysis agent. We'll implement each tool, then the main agent loop.

### Tool Implementation Example: `analyze_imports`

Here's how we implement the imports analysis tool using Python's AST:

```python
"""
Tool: analyze_imports
Extracts and categorizes import statements from Python files.

Chapter 43: Code Analysis Agent
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List

STDLIB_MODULES = set(sys.stdlib_module_names)


def analyze_imports(path: str) -> Dict:
    """
    Analyze imports in a Python file using AST parsing.

    Returns imports categorized as:
    - stdlib: Python standard library
    - third_party: External packages (pip-installed)
    - local: Project-local imports
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {"error": f"File not found: {path}"}

        if not file_path.suffix == '.py':
            return {"error": "Not a Python file"}

        # Parse the file's AST
        source = file_path.read_text()
        tree = ast.parse(source)

        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        # Walk the AST looking for import nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    import_info = {
                        "module": alias.name,
                        "line": node.lineno
                    }

                    if module_name in STDLIB_MODULES:
                        stdlib_imports.append(import_info)
                    elif module_name.startswith('.'):
                        local_imports.append(import_info)
                    else:
                        third_party_imports.append(import_info)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    import_info = {
                        "module": node.module,
                        "line": node.lineno,
                        "names": [alias.name for alias in node.names]
                    }

                    if module_name in STDLIB_MODULES:
                        stdlib_imports.append(import_info)
                    else:
                        third_party_imports.append(import_info)
                else:
                    # Relative import (from . import foo)
                    local_imports.append({
                        "module": ".",
                        "line": node.lineno,
                        "names": [alias.name for alias in node.names]
                    })

        return {
            "path": path,
            "stdlib_imports": stdlib_imports,
            "third_party_imports": third_party_imports,
            "local_imports": local_imports,
            "total_imports": (
                len(stdlib_imports) +
                len(third_party_imports) +
                len(local_imports)
            )
        }

    except SyntaxError as e:
        return {
            "error": f"Syntax error in {path}: {e}",
            "line": e.lineno
        }
    except Exception as e:
        return {"error": f"Failed to analyze {path}: {e}"}


# Tool definition for Claude
analyze_imports_tool = {
    "name": "analyze_imports",
    "description": (
        "Analyzes Python source files to extract and categorize import statements. "
        "Returns imports grouped as stdlib (Python standard library), third_party "
        "(external packages), or local (project imports). Useful for understanding "
        "dependencies, architecture, and coupling. Each import includes the module "
        "name and line number where it appears."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the Python file to analyze"
            }
        },
        "required": ["path"]
    }
}
```

### The Main Agent Loop

The agent follows the agentic loop pattern from Chapter 27, customized for code analysis:

```python
"""
Complete code analysis agent with strategic exploration.

Chapter 43: Code Analysis Agent
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Dict, List

# Import all tool functions
from list_directory_tool import list_directory, list_directory_tool
from read_file_tool import read_file, read_file_tool
from analyze_imports_tool import analyze_imports, analyze_imports_tool
from find_pattern_tool import find_pattern, find_pattern_tool
from save_finding_tool import save_finding, save_finding_tool, get_findings


load_dotenv()
client = anthropic.Anthropic()

# Tool registry
TOOLS = [
    list_directory_tool,
    read_file_tool,
    analyze_imports_tool,
    find_pattern_tool,
    save_finding_tool
]

TOOL_FUNCTIONS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "analyze_imports": analyze_imports,
    "find_pattern": find_pattern,
    "save_finding": save_finding
}


def run_agent(codebase_path: str, analysis_goal: str, max_iterations: int = 15) -> str:
    """
    Run the code analysis agent on a codebase.

    Args:
        codebase_path: Path to the codebase to analyze
        analysis_goal: What to focus on in the analysis
        max_iterations: Maximum tool use iterations

    Returns:
        Comprehensive analysis report
    """
    system_prompt = f"""You are an expert code analyzer. Your task is to analyze the codebase at '{codebase_path}' with this goal: {analysis_goal}

You have access to tools for exploring directories, reading files, analyzing imports, finding patterns, and saving findings.

Follow this strategy:

1. START: Explore the directory structure to understand the project
2. IDENTIFY: Locate entry points and key modules
3. READ: Read important files to understand architecture
4. ANALYZE: Look for patterns, dependencies, and issues
5. CATEGORIZE: Save findings by category (structure, quality, security, etc.)
6. REPORT: Synthesize findings into a comprehensive analysis

Be strategic:
- Don't read every file—sample representative files
- Focus on files that reveal architecture and patterns
- Skip tests, generated code, and vendored dependencies
- Save findings as you discover them
- Provide specific file references and line numbers

After sufficient exploration (10-12 tool calls), retrieve all findings and generate a comprehensive report with:
- Project overview and structure
- Key findings by category
- Security and quality issues with severity levels
- Specific recommendations with file references
- Prioritized action items

Be thorough but efficient. Quality over quantity."""

    messages = [
        {
            "role": "user",
            "content": f"Analyze the codebase at '{codebase_path}'. Goal: {analysis_goal}"
        }
    ]

    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Call Claude with tools
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        # Add assistant response to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Check if done
        if response.stop_reason == "end_turn":
            # Extract final response text
            final_report = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_report += block.text

            return final_report

        # Process tool calls
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    print(f"[Iteration {iteration}] Using tool: {tool_name}")

                    # Execute the tool
                    if tool_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[tool_name](**tool_input)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    # Add result to list
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })

    # Max iterations reached
    return "Analysis incomplete: reached maximum iterations. Try increasing max_iterations or making the analysis goal more specific."


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        codebase = sys.argv[1]
    else:
        codebase = "."

    print(f"Analyzing codebase: {codebase}")
    print("=" * 60)

    report = run_agent(
        codebase_path=codebase,
        analysis_goal="Provide a comprehensive analysis covering structure, quality, security, and recommendations",
        max_iterations=15
    )

    print(report)
    print("=" * 60)

    # Show all findings by category
    print("\nFindings by Category:")
    print("-" * 60)

    for category in ["structure", "quality", "security", "dependencies", "recommendations"]:
        findings = get_findings(category)
        if findings['findings']:
            print(f"\n{category.upper()}:")
            for finding in findings['findings']:
                severity = finding.get('severity', 'info')
                print(f"  [{severity}] {finding['finding']}")
```

## Analysis in Action

Let's see how the agent analyzes a Flask web application:

**Iteration 1**: Lists root directory
```
Tool: list_directory(".")
Result: Finds app/, tests/, requirements.txt, config.py
Conclusion: Flask web application
```

**Iteration 2**: Explores app/ directory
```
Tool: list_directory("app")
Result: __init__.py, models.py, routes.py, auth.py
Conclusion: MVC-style structure with authentication
```

**Iteration 3**: Reads main entry point
```
Tool: read_file("app/__init__.py")
Discovers: Flask app factory pattern, blueprints registration
Save: "Uses Flask app factory pattern (good practice)"
```

**Iteration 4**: Analyzes dependencies
```
Tool: analyze_imports("app/__init__.py")
Discovers: flask, flask_sqlalchemy, flask_login
Save: "Dependencies: Flask, SQLAlchemy, Flask-Login"
```

**Iteration 5**: Reads authentication module
```
Tool: read_file("app/auth.py")
Discovers: Password hashing, login decorators
```

**Iteration 6**: Checks for security patterns
```
Tool: find_pattern(pattern="password|secret|api_key", path=".")
Discovers: Hardcoded secret in config.py
Save: "Hardcoded SECRET_KEY in config.py:8 (security risk)"
```

**Iterations 7-12**: Continue reading key files, analyzing patterns

**Iteration 13**: Generate report
```
Tool: save_finding("recommendations", "Move secrets to environment variables")
Then: Retrieve all findings and synthesize report
```

The final report includes:
- Project structure overview
- Architecture patterns identified
- Security issues with specific locations
- Quality recommendations
- Prioritized action items

## Common Pitfalls

### 1. Reading Too Many Files

**Problem**: Agent reads every file, wasting tokens and time.

**Solution**: Use strategic sampling:
```python
# ❌ Bad - reads everything
list_directory(".", max_depth=10)  # Recursively lists everything

# ✅ Good - samples strategically
# First: Get overview
list_directory(".", max_depth=1)

# Then: Read 2-3 representative files per directory
# Focus on entry points, main modules
```

### 2. Not Saving Findings Incrementally

**Problem**: Agent tries to remember everything until the end, then loses details.

**Solution**: Save findings as you discover them:
```python
# ✅ Good - save immediately
save_finding(
    category="security",
    finding="SQL injection risk in user_query() function",
    severity="error",
    file_reference="db.py:45"
)
```

### 3. Vague Analysis Goals

**Problem**: "Analyze this codebase" is too broad.

**Solution**: Be specific:
```python
# ❌ Bad - too vague
run_agent(codebase=".", analysis_goal="Analyze this")

# ✅ Good - specific focus
run_agent(
    codebase=".",
    analysis_goal="Security audit: check for hardcoded secrets, SQL injection, and insecure authentication"
)
```

### 4. No File Reference in Findings

**Problem**: Findings lack specific locations.

**Solution**: Always include file and line numbers:
```python
# ❌ Bad - no location
save_finding(category="quality", finding="Too many parameters")

# ✅ Good - specific reference
save_finding(
    category="quality",
    finding="Function has 8 parameters (consider refactoring)",
    file_reference="utils.py:45-67"
)
```

### 5. Reading Binary or Generated Files

**Problem**: Agent wastes tokens on compiled files, images, etc.

**Solution**: Filter in `list_directory`:
```python
# Exclude patterns
EXCLUDED_DIRS = {'.git', '__pycache__', 'node_modules', '.venv'}
EXCLUDED_EXTENSIONS = {'.pyc', '.pyo', '.so', '.dylib', '.jpg', '.png'}
```

## Practical Exercise

**Task**: Enhance the code analysis agent with refactoring suggestions.

**Requirements**:

1. Add a new tool: `suggest_refactoring`
   - Takes a file path and function/class name
   - Analyzes the code for refactoring opportunities
   - Returns specific suggestions with before/after examples

2. Integration:
   - Use this tool when code quality issues are found
   - Generate concrete refactoring recommendations
   - Include severity assessment (optional vs recommended vs critical)

3. Test with a real codebase that has:
   - Long functions (>50 lines)
   - Duplicate code
   - Complex conditionals
   - Poor naming

**Hints**:
- Use AST to analyze function length and complexity
- Look for duplicated code blocks
- Identify deeply nested conditionals
- Suggest extracting methods, using guard clauses, etc.

**Solution**: See `code/exercise_solution.py`

## Key Takeaways

- **Strategic exploration beats exhaustive reading**: Sample representative files, don't read everything
- **Progressive understanding enables better analysis**: Build context early, analyze patterns later
- **Organized findings create better reports**: Categorize discoveries as you make them
- **Specific references add credibility**: Always include file paths and line numbers
- **Tool design matters**: Each tool should have a focused, clear purpose
- **Flexibility enables adaptation**: The same agent handles different project types and analysis goals
- **Safety first**: Read-only access prevents accidents; validate all file operations

## What's Next

You've now built three complete capstone projects: a research assistant, a code analysis agent, and in the next chapter, a personal productivity agent. Each demonstrates different patterns: external API integration, filesystem operations, and persistent state management.

In Chapter 44, we'll build our final project: a personal productivity agent that maintains long-term memory, manages tasks and notes, and provides context-aware assistance across multiple sessions. This brings together everything you've learned into an agent that's genuinely useful in daily work.
