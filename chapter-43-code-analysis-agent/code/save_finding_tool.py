"""
Finding storage tool for code analysis agent.
Saves and retrieves analysis findings during agent execution.

Chapter 43: Project - Code Analysis Agent
"""

from typing import Dict, List, Optional


class FindingStore:
    """
    In-memory storage for analysis findings.
    Findings are categorized for organized report generation.
    """
    
    def __init__(self):
        self.findings: List[Dict] = []
        self.categories = {
            "structure": [],
            "quality": [],
            "patterns": [],
            "dependencies": [],
            "security": [],
            "documentation": [],
            "recommendations": [],
            "other": []
        }
    
    def save_finding(
        self,
        category: str,
        finding: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        severity: str = "info"
    ) -> Dict[str, str]:
        """
        Save an analysis finding.
        
        Args:
            category: Finding category (structure, quality, patterns, etc.)
            finding: The finding description
            file_path: Optional file path where finding was discovered
            line_number: Optional line number
            severity: Severity level (info, warning, error)
        
        Returns:
            Success message
        """
        # Validate category
        if category not in self.categories:
            category = "other"
        
        finding_entry = {
            "category": category,
            "finding": finding,
            "file_path": file_path,
            "line_number": line_number,
            "severity": severity
        }
        
        self.findings.append(finding_entry)
        self.categories[category].append(finding_entry)
        
        return {
            "status": "saved",
            "category": category,
            "finding_number": len(self.findings)
        }
    
    def get_findings(
        self,
        category: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Retrieve findings, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            Dictionary with findings
        """
        if category:
            if category not in self.categories:
                return {"error": f"Invalid category: {category}"}
            
            return {
                "category": category,
                "count": len(self.categories[category]),
                "findings": self.categories[category]
            }
        else:
            return {
                "total_findings": len(self.findings),
                "by_category": {
                    cat: len(items) 
                    for cat, items in self.categories.items() 
                    if items
                },
                "all_findings": self.findings
            }
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of all findings.
        
        Returns:
            Summary statistics
        """
        severity_counts = {"info": 0, "warning": 0, "error": 0}
        for finding in self.findings:
            severity = finding.get("severity", "info")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_findings": len(self.findings),
            "by_category": {
                cat: len(items) 
                for cat, items in self.categories.items() 
                if items
            },
            "by_severity": severity_counts
        }
    
    def clear(self) -> None:
        """Clear all findings."""
        self.findings = []
        for category in self.categories:
            self.categories[category] = []


# Global finding store instance
_finding_store = FindingStore()


def save_finding(
    category: str,
    finding: str,
    file_path: Optional[str] = None,
    line_number: Optional[int] = None,
    severity: str = "info"
) -> Dict[str, str]:
    """
    Save an analysis finding to the global store.
    
    This is the function exposed to the agent as a tool.
    """
    return _finding_store.save_finding(category, finding, file_path, line_number, severity)


def get_findings(category: Optional[str] = None) -> Dict[str, any]:
    """
    Retrieve findings from the global store.
    """
    return _finding_store.get_findings(category)


def get_summary() -> Dict[str, any]:
    """
    Get a summary of all findings.
    """
    return _finding_store.get_summary()


def clear_findings() -> None:
    """
    Clear all findings from the global store.
    """
    _finding_store.clear()


# Tool definition for Claude
TOOL_DEFINITION = {
    "name": "save_finding",
    "description": (
        "Saves an analysis finding to memory for later report generation. "
        "Use this throughout your analysis to track discoveries, issues, and insights. "
        "Findings are categorized to organize the final report. "
        "Available categories: structure, quality, patterns, dependencies, security, documentation, recommendations, other."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Finding category: structure, quality, patterns, dependencies, security, documentation, recommendations, or other",
                "enum": ["structure", "quality", "patterns", "dependencies", "security", "documentation", "recommendations", "other"]
            },
            "finding": {
                "type": "string",
                "description": "The finding description. Be specific and include examples."
            },
            "file_path": {
                "type": "string",
                "description": "Optional: File path where this finding was discovered"
            },
            "line_number": {
                "type": "integer",
                "description": "Optional: Line number where this finding was discovered"
            },
            "severity": {
                "type": "string",
                "description": "Severity level: info, warning, or error (default: info)",
                "enum": ["info", "warning", "error"],
                "default": "info"
            }
        },
        "required": ["category", "finding"]
    }
}


if __name__ == "__main__":
    # Example usage
    print("Finding Storage Tool - Example Usage")
    print("="*60)
    
    # Save some example findings
    save_finding(
        category="structure",
        finding="Well-organized Flask app with routes separated by concern",
        file_path="app/__init__.py"
    )
    
    save_finding(
        category="quality",
        finding="User class is 150 lines long. Consider splitting into smaller classes.",
        file_path="models.py",
        line_number=45,
        severity="warning"
    )
    
    save_finding(
        category="security",
        finding="No rate limiting on login endpoint",
        file_path="routes/auth.py",
        line_number=28,
        severity="error"
    )
    
    save_finding(
        category="patterns",
        finding="Factory pattern used correctly for app creation",
        file_path="app/__init__.py",
        severity="info"
    )
    
    save_finding(
        category="recommendations",
        finding="Add comprehensive tests. No test files found in the codebase.",
        severity="warning"
    )
    
    # Display summary
    summary = get_summary()
    print("\nFindings Summary:")
    print(f"Total findings: {summary['total_findings']}\n")
    
    print("By Category:")
    for category, count in summary['by_category'].items():
        print(f"  {category}: {count}")
    
    print("\nBy Severity:")
    for severity, count in summary['by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    
    # Display findings by category
    print("\n" + "="*60)
    print("Quality Findings:")
    print("="*60)
    
    quality = get_findings("quality")
    for finding in quality['findings']:
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
        icon = severity_icon.get(finding['severity'], "")
        location = f"{finding['file_path']}:{finding['line_number']}" if finding['file_path'] else "General"
        print(f"\n{icon} {finding['finding']}")
        print(f"   Location: {location}")
    
    # Clear findings
    print("\n" + "="*60)
    clear_findings()
    summary = get_summary()
    print(f"After clearing: {summary['total_findings']} findings")
