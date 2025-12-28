"""
Import analysis tool for code analysis agent.
Extracts and categorizes import statements from Python files.

Chapter 43: Project - Code Analysis Agent
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set


# Standard library modules (common ones - not exhaustive)
STDLIB_MODULES = {
    'os', 'sys', 're', 'json', 'math', 'random', 'datetime', 'time',
    'collections', 'itertools', 'functools', 'typing', 'pathlib',
    'logging', 'argparse', 'subprocess', 'threading', 'multiprocessing',
    'asyncio', 'unittest', 'pytest', 'http', 'urllib', 'socket',
    'abc', 'dataclasses', 'enum', 'copy', 'pickle', 'csv', 'sqlite3',
}


def analyze_imports(file_path: str) -> Dict[str, any]:
    """
    Analyze import statements in a Python file.
    
    Args:
        file_path: Path to the Python file to analyze
    
    Returns:
        Dictionary with categorized imports and metadata
    """
    try:
        path_obj = Path(file_path).resolve()
        
        if not path_obj.exists():
            return {"error": f"File does not exist: {file_path}"}
        
        if path_obj.suffix != '.py':
            return {"error": f"Not a Python file: {file_path}"}
        
        # Read and parse the file
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(path_obj))
        except SyntaxError as e:
            return {"error": f"Syntax error in file: {e}"}
        except Exception as e:
            return {"error": f"Failed to parse file: {str(e)}"}
        
        # Extract imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import module
                for alias in node.names:
                    module_name = alias.name
                    import_info = {
                        "module": module_name,
                        "alias": alias.asname if alias.asname else None,
                        "type": "import",
                        "line": node.lineno
                    }
                    categorize_import(module_name, import_info, 
                                    stdlib_imports, third_party_imports, local_imports)
            
            elif isinstance(node, ast.ImportFrom):
                # Handle: from module import name
                module_name = node.module if node.module else ""
                
                # Handle relative imports (from . import x)
                if node.level > 0:
                    prefix = "." * node.level
                    module_name = f"{prefix}{module_name}"
                
                for alias in node.names:
                    import_info = {
                        "module": module_name,
                        "name": alias.name,
                        "alias": alias.asname if alias.asname else None,
                        "type": "from_import",
                        "line": node.lineno
                    }
                    categorize_import(module_name, import_info,
                                    stdlib_imports, third_party_imports, local_imports)
        
        # Build result
        result = {
            "file_path": str(path_obj),
            "file_name": path_obj.name,
            "stdlib_imports": stdlib_imports,
            "third_party_imports": third_party_imports,
            "local_imports": local_imports,
            "summary": {
                "total_imports": len(stdlib_imports) + len(third_party_imports) + len(local_imports),
                "stdlib_count": len(stdlib_imports),
                "third_party_count": len(third_party_imports),
                "local_count": len(local_imports)
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to analyze imports: {str(e)}"}


def categorize_import(
    module_name: str,
    import_info: Dict,
    stdlib_imports: List,
    third_party_imports: List,
    local_imports: List
) -> None:
    """
    Categorize an import as stdlib, third-party, or local.
    
    Args:
        module_name: The module being imported
        import_info: Import information dictionary
        stdlib_imports: List to append stdlib imports to
        third_party_imports: List to append third-party imports to
        local_imports: List to append local imports to
    """
    # Relative imports are always local
    if module_name.startswith('.'):
        local_imports.append(import_info)
        return
    
    # Get the top-level module name
    top_level = module_name.split('.')[0]
    
    # Check if it's a standard library module
    if top_level in STDLIB_MODULES or top_level in sys.builtin_module_names:
        stdlib_imports.append(import_info)
    # If it starts with common local patterns, it's probably local
    elif top_level in {'app', 'src', 'lib', 'utils', 'core', 'models', 'views', 'controllers'}:
        local_imports.append(import_info)
    # Otherwise, assume third-party
    else:
        third_party_imports.append(import_info)


def get_import_graph(directory_path: str) -> Dict[str, List[str]]:
    """
    Build a dependency graph for all Python files in a directory.
    
    Args:
        directory_path: Path to the directory to analyze
    
    Returns:
        Dictionary mapping file paths to lists of imported modules
    """
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        return {"error": "Invalid directory path"}
    
    import_graph = {}
    
    # Find all Python files
    for py_file in directory.rglob("*.py"):
        # Skip __pycache__ and other ignored directories
        if "__pycache__" in py_file.parts or ".git" in py_file.parts:
            continue
        
        result = analyze_imports(str(py_file))
        
        if "error" not in result:
            # Extract just the module names
            all_imports = []
            for imp in result['stdlib_imports']:
                all_imports.append(imp['module'])
            for imp in result['third_party_imports']:
                all_imports.append(imp['module'])
            for imp in result['local_imports']:
                all_imports.append(imp['module'])
            
            import_graph[str(py_file.relative_to(directory))] = list(set(all_imports))
    
    return import_graph


# Tool definition for Claude
TOOL_DEFINITION = {
    "name": "analyze_imports",
    "description": (
        "Analyzes import statements in a Python file. "
        "Extracts and categorizes imports as standard library, third-party, or local modules. "
        "Returns detailed information about each import including line numbers. "
        "Use this to understand dependencies and module relationships."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the Python file to analyze"
            }
        },
        "required": ["file_path"]
    }
}


if __name__ == "__main__":
    # Example usage
    print("Import Analysis Tool - Example Usage")
    print("="*60)
    
    # Analyze this file itself
    result = analyze_imports(__file__)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"File: {result['file_name']}")
        print(f"Total imports: {result['summary']['total_imports']}\n")
        
        if result['stdlib_imports']:
            print(f"Standard Library ({result['summary']['stdlib_count']}):")
            for imp in result['stdlib_imports']:
                alias_str = f" as {imp['alias']}" if imp['alias'] else ""
                if imp['type'] == 'import':
                    print(f"  Line {imp['line']}: import {imp['module']}{alias_str}")
                else:
                    print(f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{alias_str}")
        
        if result['third_party_imports']:
            print(f"\nThird-Party ({result['summary']['third_party_count']}):")
            for imp in result['third_party_imports']:
                alias_str = f" as {imp['alias']}" if imp['alias'] else ""
                if imp['type'] == 'import':
                    print(f"  Line {imp['line']}: import {imp['module']}{alias_str}")
                else:
                    print(f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{alias_str}")
        
        if result['local_imports']:
            print(f"\nLocal ({result['summary']['local_count']}):")
            for imp in result['local_imports']:
                alias_str = f" as {imp['alias']}" if imp['alias'] else ""
                if imp['type'] == 'import':
                    print(f"  Line {imp['line']}: import {imp['module']}{alias_str}")
                else:
                    print(f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{alias_str}")
