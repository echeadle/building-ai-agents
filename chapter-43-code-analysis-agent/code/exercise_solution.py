"""
Exercise Solution: Code Analysis Agent with Refactoring Suggestions

This enhanced version adds the ability to generate specific refactoring
recommendations with before/after code examples.

Chapter 43: Project - Code Analysis Agent
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Import existing tools
from list_directory_tool import list_directory, TOOL_DEFINITION as LIST_DIR_TOOL
from read_file_tool import read_file, TOOL_DEFINITION as READ_FILE_TOOL
from analyze_imports_tool import analyze_imports, TOOL_DEFINITION as ANALYZE_IMPORTS_TOOL
from find_pattern_tool import find_pattern, TOOL_DEFINITION as FIND_PATTERN_TOOL
from save_finding_tool import (
    save_finding, get_findings, get_summary, clear_findings,
    TOOL_DEFINITION as SAVE_FINDING_TOOL
)

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic(api_key=api_key)


# NEW: Refactoring suggestion storage
_refactoring_suggestions = []


def suggest_refactoring(
    file_path: str,
    issue_description: str,
    start_line: int,
    end_line: int,
    original_code: str
) -> dict:
    """
    Generate a refactoring suggestion with before/after code.
    
    Args:
        file_path: File containing the code to refactor
        issue_description: Description of the issue
        start_line: Starting line of code to refactor
        end_line: Ending line of code to refactor
        original_code: The original code snippet
    
    Returns:
        Refactoring suggestion with improved code
    """
    # Call Claude to generate refactored code
    refactoring_prompt = f"""You are a code refactoring expert. Given this code issue, provide a specific refactoring solution.

File: {file_path}
Lines: {start_line}-{end_line}
Issue: {issue_description}

Original Code:
```python
{original_code}
```

Please provide:
1. A brief explanation of the problem (2-3 sentences)
2. The refactored code that fixes the issue
3. A brief explanation of why the refactoring is better (2-3 sentences)

Format your response as:
PROBLEM:
[Your explanation]

REFACTORED CODE:
```python
[Your refactored code]
```

IMPROVEMENT:
[Your explanation]
"""
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": refactoring_prompt}]
        )
        
        # Extract text from response
        refactoring_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                refactoring_text += block.text
        
        # Parse the response
        parts = refactoring_text.split("REFACTORED CODE:")
        problem_part = parts[0].replace("PROBLEM:", "").strip()
        
        if len(parts) > 1:
            code_and_improvement = parts[1].split("IMPROVEMENT:")
            refactored_code = code_and_improvement[0].strip()
            # Remove code fences if present
            refactored_code = refactored_code.replace("```python", "").replace("```", "").strip()
            improvement = code_and_improvement[1].strip() if len(code_and_improvement) > 1 else ""
        else:
            refactored_code = "Unable to generate refactoring"
            improvement = ""
        
        suggestion = {
            "file_path": file_path,
            "lines": f"{start_line}-{end_line}",
            "issue": issue_description,
            "problem_explanation": problem_part,
            "original_code": original_code,
            "refactored_code": refactored_code,
            "improvement_explanation": improvement
        }
        
        _refactoring_suggestions.append(suggestion)
        
        return {
            "status": "success",
            "suggestion_number": len(_refactoring_suggestions),
            "summary": f"Generated refactoring suggestion for {file_path} lines {start_line}-{end_line}"
        }
        
    except Exception as e:
        return {"error": f"Failed to generate refactoring: {str(e)}"}


def get_refactoring_suggestions() -> list:
    """Get all refactoring suggestions."""
    return _refactoring_suggestions


def clear_refactoring_suggestions():
    """Clear all refactoring suggestions."""
    global _refactoring_suggestions
    _refactoring_suggestions = []


# Tool definition for the new refactoring tool
REFACTORING_TOOL = {
    "name": "suggest_refactoring",
    "description": (
        "Generates specific refactoring suggestions with before/after code examples. "
        "Use this when you identify code smells or quality issues that could be improved through refactoring. "
        "Provide the problematic code and a description of the issue, and this tool will generate "
        "a concrete refactoring solution with improved code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file containing the code"
            },
            "issue_description": {
                "type": "string",
                "description": "Description of the code quality issue or smell"
            },
            "start_line": {
                "type": "integer",
                "description": "Starting line number of the code to refactor"
            },
            "end_line": {
                "type": "integer",
                "description": "Ending line number of the code to refactor"
            },
            "original_code": {
                "type": "string",
                "description": "The original code snippet that needs refactoring"
            }
        },
        "required": ["file_path", "issue_description", "start_line", "end_line", "original_code"]
    }
}


# Enhanced system prompt
ENHANCED_SYSTEM_PROMPT = """You are a code analysis agent with refactoring capabilities. Your job is to examine Python codebases, identify issues, and provide specific refactoring suggestions.

ANALYSIS STRATEGY:

1. START BROAD: Use list_directory to understand the project structure
2. ANALYZE STRUCTURE: Read key files to understand the architecture
3. EXAMINE DEPENDENCIES: Use analyze_imports to map relationships
4. ASSESS QUALITY: Read representative files for code quality

When you identify code quality issues (long functions, code smells, etc.):
5. GENERATE REFACTORING: Use suggest_refactoring to provide concrete solutions
   - Extract the problematic code section
   - Describe the specific issue
   - The tool will generate refactored code with explanations

6. SAVE FINDINGS: Use save_finding for observations
   - Use category "refactoring" for refactoring suggestions
   - Reference the suggestion number in your finding

7. GENERATE REPORT: Include a "Refactoring Opportunities" section

REFACTORING GUIDELINES:

- Look for these common refactoring opportunities:
  * Long functions (>50 lines) → Break into smaller functions
  * Duplicate code → Extract to shared function
  * Long parameter lists → Use config objects or builder pattern
  * Deep nesting → Early returns or guard clauses
  * Large classes → Split responsibilities
  * Complex conditionals → Simplify or extract methods

- When you find an issue worth refactoring:
  1. Read the relevant code section
  2. Call suggest_refactoring with the code and issue
  3. Save a finding in the "refactoring" category

- Maximum 15 iterations total (including refactoring calls)

ENHANCED REPORT STRUCTURE:

Include a new section after Code Quality Findings:

## Refactoring Opportunities

[List each refactoring suggestion with before/after code]

For each suggestion:
### [Issue description]
**Location:** [file:lines]
**Problem:** [Explanation]
**Solution:** [Code example]
"""


def execute_tool_enhanced(tool_name: str, tool_input: dict) -> dict:
    """Execute tools including the new refactoring tool."""
    if tool_name == "suggest_refactoring":
        return suggest_refactoring(
            file_path=tool_input["file_path"],
            issue_description=tool_input["issue_description"],
            start_line=tool_input["start_line"],
            end_line=tool_input["end_line"],
            original_code=tool_input["original_code"]
        )
    elif tool_name == "list_directory":
        return list_directory(tool_input.get("path", "."), tool_input.get("max_depth", 2))
    elif tool_name == "read_file":
        return read_file(tool_input["file_path"], 
                        start_line=tool_input.get("start_line"),
                        end_line=tool_input.get("end_line"))
    elif tool_name == "analyze_imports":
        return analyze_imports(tool_input["file_path"])
    elif tool_name == "find_pattern":
        return find_pattern(tool_input["pattern"], tool_input["search_path"],
                          tool_input.get("file_extension", ".py"),
                          tool_input.get("context_lines", 2))
    elif tool_name == "save_finding":
        return save_finding(tool_input["category"], tool_input["finding"],
                          tool_input.get("file_path"), tool_input.get("line_number"),
                          tool_input.get("severity", "info"))
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def run_enhanced_agent(
    codebase_path: str,
    analysis_goal: str,
    max_iterations: int = 15,
    verbose: bool = True
) -> str:
    """
    Run the enhanced code analysis agent with refactoring suggestions.
    """
    clear_findings()
    clear_refactoring_suggestions()
    
    # All tools including the new refactoring tool
    tools = [
        LIST_DIR_TOOL,
        READ_FILE_TOOL,
        ANALYZE_IMPORTS_TOOL,
        FIND_PATTERN_TOOL,
        SAVE_FINDING_TOOL,
        REFACTORING_TOOL  # NEW!
    ]
    
    initial_message = f"""Analyze this codebase: {codebase_path}

Analysis Goal: {analysis_goal}

When you find code quality issues, use the suggest_refactoring tool to generate specific improvement suggestions with before/after code examples."""
    
    messages = [{"role": "user", "content": initial_message}]
    
    if verbose:
        print("="*70)
        print("ENHANCED CODE ANALYSIS AGENT (WITH REFACTORING)")
        print("="*70)
        print(f"Codebase: {codebase_path}")
        print(f"Goal: {analysis_goal}")
        print("="*70)
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=ENHANCED_SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )
        
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "end_turn":
            if verbose:
                print("\n✓ Analysis complete")
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text
            return final_text
        
        if response.stop_reason == "tool_use":
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    if verbose:
                        print(f"\n🔧 Tool: {tool_name}")
                        if tool_name == "suggest_refactoring":
                            print(f"   Generating refactoring for: {tool_input.get('file_path')}")
                    
                    result = execute_tool_enhanced(tool_name, tool_input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    # Max iterations reached
    messages.append({
        "role": "user",
        "content": "Generate your analysis report now, including the refactoring opportunities section."
    })
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=ENHANCED_SYSTEM_PROMPT,
        messages=messages
    )
    
    final_text = ""
    for block in response.content:
        if hasattr(block, 'text'):
            final_text += block.text
    return final_text


if __name__ == "__main__":
    # Example: Analyze with refactoring suggestions
    report = run_enhanced_agent(
        codebase_path=".",
        analysis_goal="Analyze code quality and generate specific refactoring suggestions",
        max_iterations=15,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("ANALYSIS REPORT WITH REFACTORING SUGGESTIONS")
    print("="*70)
    print(report)
    
    # Show detailed refactoring suggestions
    suggestions = get_refactoring_suggestions()
    if suggestions:
        print("\n" + "="*70)
        print(f"DETAILED REFACTORING SUGGESTIONS ({len(suggestions)})")
        print("="*70)
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{'='*70}")
            print(f"SUGGESTION {i}: {suggestion['issue']}")
            print(f"{'='*70}")
            print(f"File: {suggestion['file_path']}")
            print(f"Lines: {suggestion['lines']}")
            print(f"\nProblem:")
            print(suggestion['problem_explanation'])
            print(f"\nOriginal Code:")
            print("```python")
            print(suggestion['original_code'])
            print("```")
            print(f"\nRefactored Code:")
            print("```python")
            print(suggestion['refactored_code'])
            print("```")
            print(f"\nWhy This Is Better:")
            print(suggestion['improvement_explanation'])
