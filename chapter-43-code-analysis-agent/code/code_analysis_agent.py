"""
Code Analysis Agent - Complete Implementation

This agent analyzes Python codebases to assess structure, quality,
patterns, and dependencies. It uses an agentic loop to explore code
strategically and generate comprehensive analysis reports.

Chapter 43: Project - Code Analysis Agent
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Import our tools
from list_directory_tool import list_directory, TOOL_DEFINITION as LIST_DIR_TOOL
from read_file_tool import read_file, TOOL_DEFINITION as READ_FILE_TOOL
from analyze_imports_tool import analyze_imports, TOOL_DEFINITION as ANALYZE_IMPORTS_TOOL
from find_pattern_tool import find_pattern, find_todos, TOOL_DEFINITION as FIND_PATTERN_TOOL
from save_finding_tool import (
    save_finding, get_findings, get_summary, clear_findings,
    TOOL_DEFINITION as SAVE_FINDING_TOOL
)

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# System prompt for the code analysis agent
SYSTEM_PROMPT = """You are a code analysis agent. Your job is to examine Python codebases and provide comprehensive technical analysis.

ANALYSIS STRATEGY:

1. START BROAD: Use list_directory to understand the project structure
   - Identify the project type (web app, library, CLI tool, etc.)
   - Locate key files (main.py, app.py, __init__.py, setup.py, etc.)
   - Note the organization (monolithic, modular, packages, etc.)

2. ANALYZE STRUCTURE: Read key files to understand the architecture
   - Entry points and main components
   - Directory organization and purpose
   - Configuration files and dependencies

3. EXAMINE DEPENDENCIES: Use analyze_imports to map relationships
   - External dependencies (third-party packages)
   - Internal module structure
   - Look for circular dependencies or tight coupling

4. ASSESS QUALITY: Read representative files for code quality
   - Function and class design
   - Code complexity and readability
   - Error handling and edge cases
   - Documentation quality (docstrings, comments)
   - Type hints usage

5. IDENTIFY PATTERNS: Look for design patterns and anti-patterns
   - Common patterns used (factory, singleton, strategy, etc.)
   - Architectural patterns (MVC, layered architecture, etc.)
   - Anti-patterns or code smells (god objects, long functions, etc.)

6. SECURITY CHECK: Look for potential security issues
   - Input validation
   - Authentication and authorization patterns
   - Sensitive data handling
   - SQL injection risks (if applicable)

7. SAVE FINDINGS: Use save_finding throughout the analysis
   - Save insights in appropriate categories:
     * structure: Architecture and organization findings
     * quality: Code quality, readability, maintainability
     * patterns: Design patterns and anti-patterns
     * dependencies: Dependency analysis and coupling
     * security: Security concerns and best practices
     * documentation: Documentation quality
     * recommendations: Actionable improvement suggestions
   - Be specific with examples, file paths, and line numbers
   - Note both strengths and weaknesses

8. GENERATE REPORT: When you have sufficient findings (typically 8-15 iterations)
   - Create a comprehensive markdown report
   - Structure it clearly with sections
   - Include specific examples with file references
   - Provide actionable recommendations
   - Give an overall assessment

IMPORTANT GUIDELINES:

- Don't read every file. Focus on the most important ones:
  * Entry points (main.py, app.py, __main__.py)
  * Core modules (models, views, controllers, services)
  * Configuration files
  * A few representative files from each major component

- Skip unless specifically relevant:
  * Test files (test_*.py, *_test.py)
  * Migration files
  * Auto-generated code
  * Example/demo files
  * Documentation files

- Save findings as you go—don't try to remember everything
  * After reading each important file, save 1-3 findings
  * Be specific: cite files, line numbers, and code examples
  * Use appropriate severity: info for observations, warning for issues, error for serious problems

- Balance depth and breadth:
  * Don't spend 5 iterations on one file
  * Cover major aspects without getting lost in details
  * Aim for a representative sample, not exhaustive analysis

- Maximum 15 iterations. If you hit this limit, generate a report with what you have.

- When ready to generate the report, simply write it—don't use any tools.

REPORT STRUCTURE:

# Code Analysis Report

## Project Overview
[Type, structure, size]

## Architecture Assessment
### Strengths
[What's well done]
### Areas for Improvement
[What could be better]

## Code Quality Findings
### Good Practices
[Things done well]
### Code Smells
[Issues found with specific examples]

## Dependency Analysis
[External and internal dependencies]

## Security Considerations
[Security findings if any]

## Recommendations
### Immediate (High Priority)
### Short Term (Medium Priority)
### Long Term (Lower Priority)

## Summary
[Overall assessment and key takeaways]
"""

# Tool definitions
tools = [
    LIST_DIR_TOOL,
    READ_FILE_TOOL,
    ANALYZE_IMPORTS_TOOL,
    FIND_PATTERN_TOOL,
    SAVE_FINDING_TOOL
]


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Execute a tool by name with the given input.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
    
    Returns:
        Tool execution result
    """
    try:
        if tool_name == "list_directory":
            path = tool_input.get("path", ".")
            max_depth = tool_input.get("max_depth", 2)
            result = list_directory(path, max_depth)
            
        elif tool_name == "read_file":
            file_path = tool_input["file_path"]
            start_line = tool_input.get("start_line")
            end_line = tool_input.get("end_line")
            result = read_file(file_path, start_line=start_line, end_line=end_line)
            
        elif tool_name == "analyze_imports":
            file_path = tool_input["file_path"]
            result = analyze_imports(file_path)
            
        elif tool_name == "find_pattern":
            pattern = tool_input["pattern"]
            search_path = tool_input["search_path"]
            file_extension = tool_input.get("file_extension", ".py")
            context_lines = tool_input.get("context_lines", 2)
            result = find_pattern(pattern, search_path, file_extension, context_lines)
            
        elif tool_name == "save_finding":
            category = tool_input["category"]
            finding = tool_input["finding"]
            file_path = tool_input.get("file_path")
            line_number = tool_input.get("line_number")
            severity = tool_input.get("severity", "info")
            result = save_finding(category, finding, file_path, line_number, severity)
            
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return result
        
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def extract_text_from_response(response) -> str:
    """Extract text content from Claude's response."""
    text_parts = []
    for block in response.content:
        if hasattr(block, 'text'):
            text_parts.append(block.text)
    return '\n'.join(text_parts)


def run_agent(
    codebase_path: str,
    analysis_goal: str,
    max_iterations: int = 15,
    verbose: bool = True
) -> str:
    """
    Run the code analysis agent on a codebase.
    
    Args:
        codebase_path: Path to the codebase to analyze
        analysis_goal: What aspect to focus on
        max_iterations: Maximum agentic loop iterations (default: 15)
        verbose: Whether to print progress (default: True)
    
    Returns:
        The final analysis report as a string
    """
    # Clear any previous findings
    clear_findings()
    
    # Build initial message
    initial_message = f"""Analyze this codebase: {codebase_path}

Analysis Goal: {analysis_goal}

Please explore the codebase, analyze it according to your strategy, save your findings as you go, and generate a comprehensive analysis report."""
    
    messages = [{"role": "user", "content": initial_message}]
    
    if verbose:
        print("="*70)
        print("CODE ANALYSIS AGENT")
        print("="*70)
        print(f"Codebase: {codebase_path}")
        print(f"Goal: {analysis_goal}")
        print(f"Max iterations: {max_iterations}")
        print("="*70)
    
    # Agentic loop
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
        
        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )
        
        # Add assistant response to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })
        
        # Check if Claude is done (no tool calls)
        if response.stop_reason == "end_turn":
            if verbose:
                print("\n✓ Analysis complete. Generating report...")
            final_text = extract_text_from_response(response)
            return final_text
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    if verbose:
                        print(f"\n🔧 Tool: {tool_name}")
                        if tool_name == "read_file":
                            print(f"   Reading: {tool_input.get('file_path', 'unknown')}")
                        elif tool_name == "list_directory":
                            print(f"   Listing: {tool_input.get('path', '.')}")
                        elif tool_name == "analyze_imports":
                            print(f"   Analyzing imports: {tool_input.get('file_path', 'unknown')}")
                        elif tool_name == "find_pattern":
                            print(f"   Searching: {tool_input.get('pattern', 'unknown')}")
                        elif tool_name == "save_finding":
                            print(f"   Saving {tool_input.get('category', 'unknown')} finding")
                    
                    # Execute tool
                    result = execute_tool(tool_name, tool_input)
                    
                    # Add to tool results
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })
    
    # If we hit max iterations, request final report
    if verbose:
        print(f"\n⚠️  Reached maximum iterations ({max_iterations})")
        print("Requesting final report...\n")
    
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


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Default to current directory if no path provided
    codebase_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Run analysis
    report = run_agent(
        codebase_path=codebase_path,
        analysis_goal="Analyze structure, code quality, and identify improvement opportunities",
        max_iterations=15,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("ANALYSIS REPORT")
    print("="*70)
    print(report)
    
    # Show findings summary
    summary = get_summary()
    print("\n" + "="*70)
    print("FINDINGS SUMMARY")
    print("="*70)
    print(f"Total findings: {summary['total_findings']}")
    print("\nBy category:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    print("\nBy severity:")
    for severity, count in summary['by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
