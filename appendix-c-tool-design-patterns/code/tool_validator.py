"""
Tool definition validator - checks tool design quality.

Appendix C: Tool Design Patterns
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a tool definition."""
    severity: str  # "error", "warning", "suggestion"
    category: str
    message: str
    fix: str


class ToolValidator:
    """
    Validates tool definitions against best practices.
    
    Helps ensure tools follow the design patterns in Appendix C.
    """
    
    def __init__(self):
        self.issues: list[ValidationIssue] = []
    
    def validate(self, tool: dict[str, Any]) -> list[ValidationIssue]:
        """
        Validate a tool definition.
        
        Args:
            tool: Tool definition dictionary
            
        Returns:
            List of validation issues found
        """
        self.issues = []
        
        # Check required fields
        self._check_required_fields(tool)
        
        # Validate name
        if "name" in tool:
            self._validate_name(tool["name"])
        
        # Validate description
        if "description" in tool:
            self._validate_description(tool["description"])
        
        # Validate input schema
        if "input_schema" in tool:
            self._validate_input_schema(tool["input_schema"])
        
        return self.issues
    
    def _check_required_fields(self, tool: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        required = ["name", "description", "input_schema"]
        
        for field in required:
            if field not in tool:
                self.issues.append(ValidationIssue(
                    severity="error",
                    category="structure",
                    message=f"Missing required field: '{field}'",
                    fix=f"Add '{field}' to tool definition"
                ))
    
    def _validate_name(self, name: str) -> None:
        """Validate tool name follows conventions."""
        # Check snake_case
        if not name.islower() or " " in name:
            self.issues.append(ValidationIssue(
                severity="error",
                category="naming",
                message=f"Tool name '{name}' should be lowercase snake_case",
                fix=f"Change to: {name.lower().replace(' ', '_').replace('-', '_')}"
            ))
        
        # Check if verb-based
        verb_prefixes = [
            "get", "set", "create", "update", "delete", "list", "search",
            "fetch", "send", "calculate", "analyze", "generate", "validate",
            "check", "is", "has"
        ]
        
        if not any(name.startswith(prefix) for prefix in verb_prefixes):
            self.issues.append(ValidationIssue(
                severity="warning",
                category="naming",
                message=f"Tool name '{name}' doesn't start with a clear action verb",
                fix="Consider using verb-based names like 'get_weather', 'search_docs', etc."
            ))
        
        # Check length
        if len(name) > 50:
            self.issues.append(ValidationIssue(
                severity="warning",
                category="naming",
                message=f"Tool name is very long ({len(name)} chars)",
                fix="Consider a shorter, more concise name"
            ))
    
    def _validate_description(self, description: str) -> None:
        """Validate tool description follows best practices."""
        # Check minimum length
        if len(description) < 50:
            self.issues.append(ValidationIssue(
                severity="warning",
                category="description",
                message="Description is very short",
                fix="Add more details: when to use, what it returns, limitations"
            ))
        
        # Check for key phrases
        has_use_cases = any(phrase in description.lower() for phrase in [
            "use this tool when",
            "use this when",
            "use when"
        ])
        
        if not has_use_cases:
            self.issues.append(ValidationIssue(
                severity="suggestion",
                category="description",
                message="Description doesn't specify when to use the tool",
                fix="Add 'Use this tool when...' section with specific scenarios"
            ))
        
        # Check for return value description
        has_return = any(phrase in description.lower() for phrase in [
            "returns",
            "return",
            "output"
        ])
        
        if not has_return:
            self.issues.append(ValidationIssue(
                severity="suggestion",
                category="description",
                message="Description doesn't explain what the tool returns",
                fix="Add 'Returns...' section describing output format"
            ))
        
        # Check for limitations
        has_limitations = any(phrase in description.lower() for phrase in [
            "limit",
            "important",
            "note",
            "warning",
            "only",
            "cannot",
            "requires"
        ])
        
        if not has_limitations and len(description) > 100:
            self.issues.append(ValidationIssue(
                severity="suggestion",
                category="description",
                message="Consider documenting limitations or requirements",
                fix="Add notes about rate limits, requirements, or constraints"
            ))
    
    def _validate_input_schema(self, schema: dict[str, Any]) -> None:
        """Validate input schema structure and parameters."""
        # Check schema type
        if schema.get("type") != "object":
            self.issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message="Input schema type must be 'object'",
                fix="Set 'type': 'object' in input_schema"
            ))
            return
        
        # Check properties exist
        if "properties" not in schema:
            self.issues.append(ValidationIssue(
                severity="error",
                category="schema",
                message="Input schema missing 'properties'",
                fix="Add 'properties' object defining parameters"
            ))
            return
        
        properties = schema["properties"]
        required = schema.get("required", [])
        
        # Validate each property
        for prop_name, prop_schema in properties.items():
            self._validate_parameter(prop_name, prop_schema, prop_name in required)
        
        # Check if too many required parameters
        if len(required) > 5:
            self.issues.append(ValidationIssue(
                severity="warning",
                category="schema",
                message=f"Many required parameters ({len(required)})",
                fix="Consider if all parameters are truly required"
            ))
    
    def _validate_parameter(
        self,
        name: str,
        schema: dict[str, Any],
        is_required: bool
    ) -> None:
        """Validate a single parameter."""
        # Check description exists
        if "description" not in schema:
            self.issues.append(ValidationIssue(
                severity="error",
                category="parameter",
                message=f"Parameter '{name}' missing description",
                fix=f"Add 'description' to '{name}' explaining its purpose"
            ))
        else:
            # Check description quality
            desc = schema["description"]
            if len(desc) < 20:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    category="parameter",
                    message=f"Parameter '{name}' has short description",
                    fix="Add examples or more detail about expected values"
                ))
        
        # Check type is specified
        if "type" not in schema:
            self.issues.append(ValidationIssue(
                severity="error",
                category="parameter",
                message=f"Parameter '{name}' missing type",
                fix=f"Add 'type' (string, integer, boolean, array, object)"
            ))
        
        # Check for enum on string types when appropriate
        param_type = schema.get("type")
        if param_type == "string" and name.endswith(("_type", "_mode", "_status")):
            if "enum" not in schema:
                self.issues.append(ValidationIssue(
                    severity="suggestion",
                    category="parameter",
                    message=f"Parameter '{name}' might benefit from enum",
                    fix="If there are fixed choices, add 'enum' list"
                ))
        
        # Check for constraints on numbers
        if param_type in ["integer", "number"]:
            has_constraints = any(k in schema for k in ["minimum", "maximum", "enum"])
            if not has_constraints:
                self.issues.append(ValidationIssue(
                    severity="suggestion",
                    category="parameter",
                    message=f"Numeric parameter '{name}' has no constraints",
                    fix="Consider adding 'minimum' and 'maximum' values"
                ))
    
    def print_report(self) -> None:
        """Print a formatted validation report."""
        if not self.issues:
            print("✅ Tool definition looks good!")
            return
        
        # Group by severity
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        suggestions = [i for i in self.issues if i.severity == "suggestion"]
        
        print("\n" + "=" * 60)
        print("TOOL VALIDATION REPORT")
        print("=" * 60)
        
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                print(f"\n  [{issue.category}] {issue.message}")
                print(f"  Fix: {issue.fix}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"\n  [{issue.category}] {issue.message}")
                print(f"  Fix: {issue.fix}")
        
        if suggestions:
            print(f"\n💡 SUGGESTIONS ({len(suggestions)}):")
            for issue in suggestions:
                print(f"\n  [{issue.category}] {issue.message}")
                print(f"  Fix: {issue.fix}")
        
        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    validator = ToolValidator()
    
    # Example 1: Good tool
    print("Example 1: Well-designed tool")
    print("-" * 60)
    
    good_tool = {
        "name": "get_weather",
        "description": """Get current weather conditions for any city.
        
Use this tool when the user asks about current weather, temperature, or conditions.

Returns temperature, condition, humidity, and wind speed.
Data updates every 15 minutes.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'Tokyo')"
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units: 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["city"]
        }
    }
    
    validator.validate(good_tool)
    validator.print_report()
    
    # Example 2: Poor tool
    print("\n\nExample 2: Tool with issues")
    print("-" * 60)
    
    bad_tool = {
        "name": "DoWeather",  # Bad naming
        "description": "Gets weather",  # Too short
        "input_schema": {
            "type": "object",
            "properties": {
                "c": {  # Unclear name
                    "type": "string"
                    # Missing description
                },
                "temp_type": {
                    "type": "string"
                    # Should use enum
                }
            },
            "required": ["c", "temp_type"]
        }
    }
    
    validator.validate(bad_tool)
    validator.print_report()
    
    # Example 3: Missing fields
    print("\n\nExample 3: Incomplete tool")
    print("-" * 60)
    
    incomplete_tool = {
        "name": "search"
        # Missing description and input_schema
    }
    
    validator.validate(incomplete_tool)
    validator.print_report()
