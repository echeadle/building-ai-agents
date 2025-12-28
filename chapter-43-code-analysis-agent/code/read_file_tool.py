"""
File reading tool for code analysis agent.
Reads source files with line numbers and error handling.

Chapter 43: Project - Code Analysis Agent
"""

import os
from pathlib import Path
from typing import Dict, Optional


# Maximum file size to read (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Binary file extensions to reject
BINARY_EXTENSIONS = {
    '.pyc', '.so', '.dll', '.exe', '.bin',
    '.jpg', '.jpeg', '.png', '.gif', '.pdf',
    '.zip', '.tar', '.gz', '.mp3', '.mp4'
}


def read_file(
    file_path: str,
    show_line_numbers: bool = True,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> Dict[str, str]:
    """
    Read a source file and return its contents.
    
    Args:
        file_path: Path to the file to read
        show_line_numbers: Whether to add line numbers (default: True)
        start_line: Optional line to start from (1-indexed)
        end_line: Optional line to end at (1-indexed, inclusive)
    
    Returns:
        Dictionary with file contents and metadata
    """
    try:
        # Resolve and validate path
        path_obj = Path(file_path).resolve()
        
        # Security: Prevent directory traversal
        # In production, you'd check against allowed base paths
        if not path_obj.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        if not path_obj.is_file():
            return {"error": f"Path is not a file: {file_path}"}
        
        # Check file extension
        if path_obj.suffix.lower() in BINARY_EXTENSIONS:
            return {"error": f"Binary file type not supported: {path_obj.suffix}"}
        
        # Check file size
        file_size = path_obj.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return {
                "error": f"File too large: {file_size / 1024 / 1024:.1f} MB "
                         f"(max: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB)"
            }
        
        # Read file
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Try with latin-1 encoding as fallback
            try:
                with open(path_obj, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except Exception as e:
                return {"error": f"Failed to decode file: {str(e)}"}
        
        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            lines = lines[start:end]
        
        # Format with line numbers if requested
        if show_line_numbers:
            line_offset = start_line if start_line else 1
            formatted_lines = []
            for i, line in enumerate(lines, start=line_offset):
                # Remove trailing newline for cleaner output
                line_content = line.rstrip('\n')
                formatted_lines.append(f"{i:4d} | {line_content}")
            content = '\n'.join(formatted_lines)
        else:
            content = ''.join(lines)
        
        # Build result
        result = {
            "file_path": str(path_obj),
            "file_name": path_obj.name,
            "extension": path_obj.suffix,
            "size_bytes": file_size,
            "total_lines": len(lines) if not (start_line or end_line) else None,
            "lines_shown": len(lines),
            "content": content
        }
        
        return result
        
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}


def read_file_snippet(
    file_path: str,
    line_number: int,
    context_lines: int = 3
) -> Dict[str, str]:
    """
    Read a snippet of a file around a specific line number.
    
    Args:
        file_path: Path to the file
        line_number: Target line number (1-indexed)
        context_lines: Number of lines before/after to include
    
    Returns:
        Dictionary with the code snippet
    """
    start = max(1, line_number - context_lines)
    end = line_number + context_lines
    
    result = read_file(file_path, show_line_numbers=True, start_line=start, end_line=end)
    
    if "error" not in result:
        result["target_line"] = line_number
        result["context_lines"] = context_lines
    
    return result


# Tool definition for Claude
TOOL_DEFINITION = {
    "name": "read_file",
    "description": (
        "Reads the contents of a source code file. "
        "Returns the full file with line numbers for easy reference. "
        "Use this to examine specific files after identifying them with list_directory. "
        "Automatically handles text encoding and rejects binary files. "
        "Maximum file size: 5MB."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (absolute or relative)"
            },
            "start_line": {
                "type": "integer",
                "description": "Optional: Line number to start reading from (1-indexed)"
            },
            "end_line": {
                "type": "integer",
                "description": "Optional: Line number to stop reading at (1-indexed, inclusive)"
            }
        },
        "required": ["file_path"]
    }
}


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("File Reading Tool - Example Usage")
    print("="*60)
    
    # Read this file as an example
    result = read_file(__file__)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"File: {result['file_name']}")
        print(f"Path: {result['file_path']}")
        print(f"Size: {result['size_bytes']} bytes")
        print(f"Lines: {result['total_lines']}")
        print(f"\nFirst 20 lines:\n")
        
        # Show first 20 lines
        lines = result['content'].split('\n')
        for line in lines[:20]:
            print(line)
        
        print(f"\n... ({result['total_lines'] - 20} more lines)")
    
    # Example: Read with line range
    print("\n" + "="*60)
    print("Reading lines 40-50 of this file:")
    print("="*60)
    
    result = read_file(__file__, start_line=40, end_line=50)
    if "error" not in result:
        print(result['content'])
