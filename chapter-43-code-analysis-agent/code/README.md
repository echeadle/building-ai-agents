# Code Analysis Agent - Code Files

This directory contains all the code for Chapter 43: Code Analysis Agent.

## Files Overview

### Core Agent

**`code_analysis_agent.py`** - The main agent implementation
- Complete agentic loop for code analysis
- Integrates all tools
- Generates comprehensive analysis reports
- Run with: `python code_analysis_agent.py [path/to/codebase]`

### Tool Implementations

**`list_directory_tool.py`** - Directory exploration
- Lists files and subdirectories
- Filters out irrelevant directories (`.git`, `__pycache__`, etc.)
- Identifies code files vs other files
- Returns file sizes and counts

**`read_file_tool.py`** - Source code reading
- Reads text files with line numbers
- Supports reading specific line ranges
- Handles encoding issues gracefully
- Maximum 5MB file size limit
- Rejects binary files

**`analyze_imports_tool.py`** - Import analysis
- Extracts import statements using Python's AST
- Categorizes as stdlib, third-party, or local
- Returns line numbers for each import
- Handles both `import` and `from ... import` syntax

**`find_pattern_tool.py`** - Pattern searching
- Searches for regex patterns in code
- Returns matches with context lines
- Includes helper function for finding TODOs
- Can search specific files or entire directories

**`save_finding_tool.py`** - Finding storage
- Saves analysis findings by category
- Categories: structure, quality, patterns, dependencies, security, documentation, recommendations
- Supports severity levels: info, warning, error
- Retrieves findings for report generation

### Examples and Exercises

**`example_usage.py`** - Usage examples
- Six different example scenarios
- Shows various analysis goals
- Demonstrates how to customize max iterations
- Shows how to retrieve findings by category

**`exercise_solution.py`** - Exercise solution
- Enhanced agent with refactoring suggestions
- New `suggest_refactoring` tool
- Generates before/after code examples
- Includes detailed refactoring reports

## Getting Started

### Prerequisites

1. Python 3.10 or higher
2. Required packages:
   ```bash
   pip install anthropic python-dotenv --break-system-packages
   ```
3. Environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your-api-key-here
   ```
   Or create a `.env` file with:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

### Basic Usage

Analyze the current directory:
```bash
python code_analysis_agent.py .
```

Analyze a specific project:
```bash
python code_analysis_agent.py /path/to/project
```

Run the examples:
```bash
python example_usage.py
```

Run the enhanced agent with refactoring:
```bash
python exercise_solution.py
```

### Testing Individual Tools

Each tool file can be run independently to see how it works:

```bash
# Test directory listing
python list_directory_tool.py

# Test file reading
python read_file_tool.py

# Test import analysis
python analyze_imports_tool.py

# Test pattern finding
python find_pattern_tool.py

# Test finding storage
python save_finding_tool.py
```

## How It Works

The agent follows this workflow:

1. **Initial Request**: User provides codebase path and analysis goal
2. **Agentic Loop**: Agent decides what to do each iteration:
   - Explore directory structure
   - Read important files
   - Analyze imports and dependencies
   - Search for patterns
   - Save findings
3. **Tool Execution**: Each tool call returns results
4. **Iteration**: Agent processes results and decides next action
5. **Report Generation**: After sufficient analysis, generate comprehensive report

## Customization

### Adjusting Iterations

Change `max_iterations` to control analysis depth:
```python
# Quick analysis (6-8 iterations)
report = run_agent(codebase_path=".", analysis_goal="Quick overview", max_iterations=8)

# Deep analysis (15-20 iterations)
report = run_agent(codebase_path=".", analysis_goal="Comprehensive", max_iterations=20)
```

### Focusing Analysis

Guide the agent with specific goals:
```python
# Security focus
run_agent(codebase_path=".", analysis_goal="Focus on security vulnerabilities")

# Architecture focus
run_agent(codebase_path=".", analysis_goal="Review architecture and design patterns")

# Quality focus
run_agent(codebase_path=".", analysis_goal="Assess code quality and maintainability")
```

### Retrieving Specific Findings

```python
from save_finding_tool import get_findings

# Get all security findings
security = get_findings("security")
for finding in security['findings']:
    print(finding['finding'])

# Get quality issues
quality = get_findings("quality")
```

## Architecture

```
code_analysis_agent.py
    ├── Agentic Loop
    │   ├── Call Claude with tools
    │   ├── Process tool use requests
    │   └── Continue until done
    │
    └── Tools
        ├── list_directory()    → Explore structure
        ├── read_file()         → Read source code
        ├── analyze_imports()   → Extract dependencies
        ├── find_pattern()      → Search patterns
        └── save_finding()      → Store discoveries
```

## Key Patterns

### 1. Strategic Exploration
The agent doesn't read every file. It:
- Starts with directory structure
- Identifies key files (entry points, main modules)
- Focuses on representative samples
- Skips test files and generated code

### 2. Progressive Understanding
The agent builds knowledge incrementally:
- Early: Understanding project type and structure
- Middle: Analyzing code quality and patterns
- Late: Identifying issues and forming recommendations

### 3. Organized Findings
Findings are categorized for structured reports:
- **structure**: Architecture and organization
- **quality**: Code quality issues
- **patterns**: Design patterns found
- **dependencies**: Import and coupling analysis
- **security**: Security concerns
- **documentation**: Documentation quality
- **recommendations**: Actionable improvements

### 4. Flexible Analysis
The same agent handles different project types:
- Web applications (Flask, Django)
- CLI tools
- Libraries
- APIs
- Data processing scripts

## Common Use Cases

### 1. Onboarding
"Help me understand this codebase I'm joining"
```python
run_agent(
    codebase_path="/new/project",
    analysis_goal="Provide overview for a new developer: structure, key components, and getting started"
)
```

### 2. Code Review
"Review this PR for quality issues"
```python
run_agent(
    codebase_path="/pr/branch",
    analysis_goal="Code review: quality, best practices, potential issues"
)
```

### 3. Technical Debt Assessment
"How much technical debt do we have?"
```python
run_agent(
    codebase_path="/legacy/project",
    analysis_goal="Assess technical debt: code smells, outdated patterns, improvement opportunities"
)
```

### 4. Security Audit
"Are there security issues?"
```python
run_agent(
    codebase_path="/web/app",
    analysis_goal="Security audit: input validation, authentication, sensitive data"
)
```

### 5. Dependency Analysis
"What are our dependencies?"
```python
run_agent(
    codebase_path="/project",
    analysis_goal="Analyze all dependencies: third-party packages, internal coupling, update recommendations"
)
```

## Limitations

- **Python-specific**: Currently only analyzes Python code
- **No execution**: Reads and analyzes code but doesn't run it
- **Context window**: Very large files may be truncated
- **Static analysis**: Can't detect runtime issues
- **Heuristic-based**: Uses patterns and best practices, not formal verification

## Extension Ideas

1. **Multi-language support**: Add tools for JavaScript, Java, etc.
2. **Metrics calculation**: Cyclomatic complexity, code coverage
3. **Visualization**: Generate dependency graphs
4. **Interactive mode**: Ask follow-up questions about findings
5. **Continuous monitoring**: Run on every commit
6. **Custom rules**: Define project-specific quality rules
7. **Integration**: Connect to GitHub, GitLab for PR reviews

## Troubleshooting

**"Permission denied" errors**
- Check file permissions in the target directory
- Run with appropriate user permissions

**"File too large" errors**
- Files over 5MB are rejected
- Increase `MAX_FILE_SIZE` in `read_file_tool.py` if needed

**Agent reads too many files**
- Reduce `max_iterations`
- Make analysis goal more specific
- Add exclusion patterns to `list_directory_tool.py`

**Reports are too generic**
- Increase `max_iterations` for deeper analysis
- Use more specific analysis goals
- Enhance the system prompt with domain expertise

## Learn More

See Chapter 43 in the book for:
- Complete implementation walkthrough
- Design decisions and tradeoffs
- Common pitfalls and solutions
- How to adapt the patterns to other domains
