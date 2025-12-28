"""
Directory listing tool for code analysis agent.
Lists files and directories with filtering for relevant code files.

Chapter 43: Project - Code Analysis Agent
"""

import os
from pathlib import Path
from typing import Dict, List, Any


# Directories to ignore during code analysis
IGNORED_DIRS = {
    '.git', '.svn', '.hg',  # Version control
    '__pycache__', '.pytest_cache', '.mypy_cache',  # Python caches
    'node_modules', 'bower_components',  # JavaScript
    'venv', 'env', '.venv', '.env',  # Virtual environments
    'build', 'dist', '.egg-info',  # Build artifacts
    '.idea', '.vscode', '.vs',  # IDEs
    'htmlcov', 'coverage',  # Coverage reports
}

# File extensions to highlight as important
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
    '.rb', '.go', '.rs', '.php', '.swift', '.kt'
}


def list_directory(
    path: str, 
    max_depth: int = 2,
    show_hidden: bool = False
) -> Dict[str, Any]:
    """
    List files and directories at the given path.
    
    Args:
        path: Directory path to list
        max_depth: Maximum depth to recurse (default: 2)
        show_hidden: Whether to show hidden files (default: False)
    
    Returns:
        Dictionary with directory structure and metadata
    """
    try:
        path_obj = Path(path).resolve()
        
        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}
        
        if not path_obj.is_dir():
            return {"error": f"Path is not a directory: {path}"}
        
        result = {
            "path": str(path_obj),
            "files": [],
            "directories": [],
            "summary": {}
        }
        
        # List immediate contents
        items = []
        try:
            items = list(path_obj.iterdir())
        except PermissionError:
            return {"error": f"Permission denied: {path}"}
        
        # Process each item
        for item in sorted(items):
            # Skip hidden files unless requested
            if not show_hidden and item.name.startswith('.'):
                continue
            
            # Skip ignored directories
            if item.is_dir() and item.name in IGNORED_DIRS:
                continue
            
            if item.is_file():
                file_info = {
                    "name": item.name,
                    "path": str(item.relative_to(path_obj)),
                    "size": item.stat().st_size,
                    "extension": item.suffix,
                    "is_code": item.suffix in CODE_EXTENSIONS
                }
                result["files"].append(file_info)
            
            elif item.is_dir():
                dir_info = {
                    "name": item.name,
                    "path": str(item.relative_to(path_obj))
                }
                
                # Count files in directory (non-recursive)
                try:
                    file_count = sum(1 for _ in item.iterdir() if _.is_file())
                    dir_info["file_count"] = file_count
                except PermissionError:
                    dir_info["file_count"] = "Permission denied"
                
                result["directories"].append(dir_info)
        
        # Generate summary
        result["summary"] = {
            "total_files": len(result["files"]),
            "total_directories": len(result["directories"]),
            "code_files": sum(1 for f in result["files"] if f["is_code"]),
            "total_size_bytes": sum(f["size"] for f in result["files"])
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to list directory: {str(e)}"}


def list_directory_recursive(
    path: str,
    max_depth: int = 3,
    current_depth: int = 0
) -> Dict[str, Any]:
    """
    Recursively list directory structure up to max_depth.
    
    Args:
        path: Directory path to list
        max_depth: Maximum recursion depth
        current_depth: Current recursion level (internal)
    
    Returns:
        Nested dictionary with full directory tree
    """
    if current_depth >= max_depth:
        return {"truncated": "Max depth reached"}
    
    try:
        path_obj = Path(path).resolve()
        
        if not path_obj.exists() or not path_obj.is_dir():
            return {"error": "Invalid path"}
        
        result = {
            "name": path_obj.name,
            "path": str(path_obj),
            "files": [],
            "subdirectories": []
        }
        
        # List contents
        try:
            items = list(path_obj.iterdir())
        except PermissionError:
            return {"error": "Permission denied"}
        
        for item in sorted(items):
            # Skip hidden and ignored
            if item.name.startswith('.') or item.name in IGNORED_DIRS:
                continue
            
            if item.is_file():
                result["files"].append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "extension": item.suffix
                })
            
            elif item.is_dir():
                # Recurse into subdirectory
                subdir_result = list_directory_recursive(
                    str(item),
                    max_depth=max_depth,
                    current_depth=current_depth + 1
                )
                result["subdirectories"].append(subdir_result)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


# Tool definition for Claude
TOOL_DEFINITION = {
    "name": "list_directory",
    "description": (
        "Lists files and directories at the specified path. "
        "Use this to explore codebase structure and identify important files. "
        "Returns file names, sizes, and types. "
        "Automatically filters out irrelevant directories like .git, __pycache__, node_modules, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The directory path to list (absolute or relative)"
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for listing (default: 2, max: 5)",
                "default": 2
            }
        },
        "required": ["path"]
    }
}


if __name__ == "__main__":
    # Example usage
    print("Listing current directory:")
    print("="*60)
    
    result = list_directory(".", max_depth=1)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Path: {result['path']}\n")
        
        print(f"Files ({result['summary']['total_files']}):")
        for f in result['files'][:10]:  # Show first 10
            size_kb = f['size'] / 1024
            code_indicator = "📄" if f['is_code'] else "  "
            print(f"  {code_indicator} {f['name']:<30} {size_kb:>8.1f} KB")
        
        if len(result['files']) > 10:
            print(f"  ... and {len(result['files']) - 10} more files")
        
        print(f"\nDirectories ({result['summary']['total_directories']}):")
        for d in result['directories'][:10]:  # Show first 10
            print(f"  📁 {d['name']:<30} ({d.get('file_count', '?')} files)")
        
        if len(result['directories']) > 10:
            print(f"  ... and {len(result['directories']) - 10} more directories")
        
        print(f"\nSummary:")
        print(f"  Code files: {result['summary']['code_files']}")
        print(f"  Total size: {result['summary']['total_size_bytes'] / 1024:.1f} KB")
