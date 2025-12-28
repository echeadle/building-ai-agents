"""
Pattern finding tool for code analysis agent.
Searches for code patterns using regular expressions.

Chapter 43: Project - Code Analysis Agent
"""

import re
from pathlib import Path
from typing import Dict, List, Optional


def find_pattern(
    pattern: str,
    search_path: str,
    file_extension: str = ".py",
    context_lines: int = 2,
    case_sensitive: bool = True
) -> Dict[str, any]:
    """
    Search for a regex pattern in code files.
    
    Args:
        pattern: Regular expression pattern to search for
        search_path: File or directory path to search in
        file_extension: File extension to search (default: .py)
        context_lines: Number of lines before/after to include (default: 2)
        case_sensitive: Whether search is case-sensitive (default: True)
    
    Returns:
        Dictionary with matches and their locations
    """
    try:
        # Compile the regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}
        
        path_obj = Path(search_path).resolve()
        
        if not path_obj.exists():
            return {"error": f"Path does not exist: {search_path}"}
        
        # Determine if we're searching a file or directory
        if path_obj.is_file():
            files_to_search = [path_obj]
        elif path_obj.is_dir():
            # Find all files with the specified extension
            files_to_search = list(path_obj.rglob(f"*{file_extension}"))
            # Filter out common directories to ignore
            ignored_dirs = {'__pycache__', '.git', 'venv', 'node_modules'}
            files_to_search = [
                f for f in files_to_search 
                if not any(ignored in f.parts for ignored in ignored_dirs)
            ]
        else:
            return {"error": "Path must be a file or directory"}
        
        # Search each file
        matches = []
        for file_path in files_to_search:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Search for pattern in each line
                for line_num, line in enumerate(lines, start=1):
                    if regex.search(line):
                        # Extract context
                        start_line = max(0, line_num - context_lines - 1)
                        end_line = min(len(lines), line_num + context_lines)
                        context = lines[start_line:end_line]
                        
                        # Format context with line numbers
                        formatted_context = []
                        for i, ctx_line in enumerate(context, start=start_line + 1):
                            marker = ">>>" if i == line_num else "   "
                            formatted_context.append(
                                f"{marker} {i:4d} | {ctx_line.rstrip()}"
                            )
                        
                        matches.append({
                            "file": str(file_path.relative_to(path_obj.parent) 
                                      if path_obj.is_dir() else file_path.name),
                            "line_number": line_num,
                            "line_content": line.rstrip(),
                            "match": regex.search(line).group(0),
                            "context": "\n".join(formatted_context)
                        })
            
            except (PermissionError, UnicodeDecodeError):
                # Skip files we can't read
                continue
            except Exception as e:
                # Log but continue with other files
                continue
        
        # Build result
        result = {
            "pattern": pattern,
            "search_path": str(path_obj),
            "files_searched": len(files_to_search),
            "total_matches": len(matches),
            "matches": matches
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to search pattern: {str(e)}"}


def find_todos(search_path: str) -> Dict[str, any]:
    """
    Convenience function to find TODO comments in code.
    
    Args:
        search_path: File or directory path to search
    
    Returns:
        Dictionary with TODO matches
    """
    # Pattern matches: TODO, FIXME, HACK, XXX, NOTE
    pattern = r'#\s*(TODO|FIXME|HACK|XXX|NOTE):?\s*(.+)'
    return find_pattern(pattern, search_path, context_lines=1)


def find_long_functions(file_path: str, threshold: int = 50) -> Dict[str, List]:
    """
    Find functions longer than a threshold number of lines.
    
    Args:
        file_path: Python file to analyze
        threshold: Minimum number of lines (default: 50)
    
    Returns:
        Dictionary with long functions found
    """
    try:
        path_obj = Path(file_path)
        
        if not path_obj.exists() or path_obj.suffix != '.py':
            return {"error": "Invalid Python file"}
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Simple heuristic: find function definitions and count lines until next def/class
        long_functions = []
        in_function = False
        function_name = ""
        function_start = 0
        function_lines = 0
        
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            
            # New function/class definition
            if stripped.startswith('def ') or stripped.startswith('class '):
                # Save previous function if it was long
                if in_function and function_lines >= threshold:
                    long_functions.append({
                        "name": function_name,
                        "start_line": function_start,
                        "lines": function_lines
                    })
                
                # Start new function
                in_function = stripped.startswith('def ')
                if in_function:
                    function_name = stripped.split('(')[0].replace('def ', '')
                    function_start = i
                    function_lines = 1
            elif in_function:
                # Count non-empty lines
                if stripped and not stripped.startswith('#'):
                    function_lines += 1
        
        # Check last function
        if in_function and function_lines >= threshold:
            long_functions.append({
                "name": function_name,
                "start_line": function_start,
                "lines": function_lines
            })
        
        return {
            "file": str(path_obj.name),
            "threshold": threshold,
            "long_functions": long_functions
        }
        
    except Exception as e:
        return {"error": f"Failed to analyze functions: {str(e)}"}


# Tool definition for Claude
TOOL_DEFINITION = {
    "name": "find_pattern",
    "description": (
        "Searches for a regex pattern in code files. "
        "Can search a single file or recursively through a directory. "
        "Returns matches with line numbers and surrounding context. "
        "Useful for finding TODO comments, deprecated patterns, specific function calls, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for"
            },
            "search_path": {
                "type": "string",
                "description": "File or directory path to search in"
            },
            "file_extension": {
                "type": "string",
                "description": "File extension to search (default: .py)",
                "default": ".py"
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines to show (default: 2)",
                "default": 2
            }
        },
        "required": ["pattern", "search_path"]
    }
}


if __name__ == "__main__":
    # Example usage
    print("Pattern Finding Tool - Example Usage")
    print("="*60)
    
    # Find TODO comments in current directory
    print("Searching for TODO comments...")
    result = find_todos(".")
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Files searched: {result['files_searched']}")
        print(f"Matches found: {result['total_matches']}\n")
        
        for match in result['matches'][:5]:  # Show first 5
            print(f"📍 {match['file']}:{match['line_number']}")
            print(f"   Match: {match['match']}")
            print(f"\n{match['context']}\n")
    
    print("\n" + "="*60)
    print("Searching for function definitions...")
    
    # Find all function definitions
    result = find_pattern(
        pattern=r'^def\s+\w+',
        search_path=".",
        context_lines=0
    )
    
    if "error" not in result and result['total_matches'] > 0:
        print(f"Found {result['total_matches']} functions\n")
        for match in result['matches'][:3]:  # Show first 3
            print(f"  {match['file']}:{match['line_number']} - {match['match']}")
