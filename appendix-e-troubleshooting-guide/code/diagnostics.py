"""
Diagnostic Utilities for Troubleshooting AI Agents

Appendix E: Troubleshooting Guide
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Any, Optional
import json

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class AgentDiagnostics:
    """
    Collection of diagnostic tools for troubleshooting agents.
    
    Use these functions to understand what's going wrong with your agent.
    """
    
    @staticmethod
    def check_tools_configuration(tools: list[dict]) -> dict[str, Any]:
        """
        Verify tool configurations are valid.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []
        
        for i, tool in enumerate(tools):
            tool_name = tool.get("name", f"Tool #{i}")
            
            # Check required fields
            if "name" not in tool:
                issues.append(f"{tool_name}: Missing 'name' field")
            
            if "description" not in tool:
                issues.append(f"{tool_name}: Missing 'description' field")
            elif len(tool["description"]) < 10:
                warnings.append(f"{tool_name}: Description is very short (< 10 chars)")
            
            if "input_schema" not in tool:
                issues.append(f"{tool_name}: Missing 'input_schema' field")
            else:
                schema = tool["input_schema"]
                
                # Check schema structure
                if "type" not in schema:
                    issues.append(f"{tool_name}: Schema missing 'type' field")
                
                if "properties" not in schema:
                    warnings.append(f"{tool_name}: Schema has no properties")
                else:
                    # Check each property has a description
                    for prop_name, prop_def in schema["properties"].items():
                        if "description" not in prop_def:
                            warnings.append(
                                f"{tool_name}: Property '{prop_name}' missing description"
                            )
            
            # Check for similar names
            for j, other_tool in enumerate(tools[i+1:], start=i+1):
                other_name = other_tool.get("name", f"Tool #{j}")
                if tool_name and other_name:
                    # Check for confusingly similar names
                    if tool_name.lower().replace("_", "") == other_name.lower().replace("_", ""):
                        warnings.append(
                            f"Tools '{tool_name}' and '{other_name}' have similar names"
                        )
        
        return {
            "valid": len(issues) == 0,
            "tool_count": len(tools),
            "issues": issues,
            "warnings": warnings,
        }
    
    @staticmethod
    def diagnose_tool_selection(
        conversation: list[dict],
        expected_tool: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Analyze which tools were called and why.
        
        Args:
            conversation: Conversation history
            expected_tool: Tool that should have been called
            
        Returns:
            Analysis of tool selection
        """
        tool_calls = []
        
        for message in conversation:
            if message.get("role") == "assistant":
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls.append({
                                "name": block.get("name"),
                                "input": block.get("input"),
                            })
        
        analysis = {
            "total_tool_calls": len(tool_calls),
            "tools_called": [tc["name"] for tc in tool_calls],
            "unique_tools": list(set(tc["name"] for tc in tool_calls)),
        }
        
        if expected_tool:
            analysis["expected_tool_called"] = expected_tool in analysis["tools_called"]
            if not analysis["expected_tool_called"]:
                analysis["issue"] = f"Expected '{expected_tool}' but it was not called"
        
        return analysis
    
    @staticmethod
    def check_api_connectivity() -> dict[str, Any]:
        """
        Test connection to Anthropic API.
        
        Returns:
            Connection test results
        """
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            return {
                "connected": True,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "message": "Successfully connected to Anthropic API",
            }
        
        except anthropic.APIConnectionError as e:
            return {
                "connected": False,
                "error": "connection_failed",
                "message": "Could not connect to Anthropic API",
                "details": str(e),
            }
        
        except anthropic.AuthenticationError as e:
            return {
                "connected": False,
                "error": "authentication_failed",
                "message": "API key is invalid",
                "details": str(e),
            }
        
        except Exception as e:
            return {
                "connected": False,
                "error": "unknown",
                "message": str(e),
            }
    
    @staticmethod
    def analyze_conversation_flow(conversation: list[dict]) -> dict[str, Any]:
        """
        Analyze the flow of a conversation to identify issues.
        
        Args:
            conversation: Conversation history
            
        Returns:
            Flow analysis
        """
        analysis = {
            "total_turns": len(conversation),
            "user_messages": 0,
            "assistant_messages": 0,
            "tool_uses": 0,
            "tool_results": 0,
            "issues": [],
        }
        
        for i, message in enumerate(conversation):
            role = message.get("role")
            content = message.get("content", [])
            
            if role == "user":
                analysis["user_messages"] += 1
            elif role == "assistant":
                analysis["assistant_messages"] += 1
                
                # Count tool uses
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            analysis["tool_uses"] += 1
            
            # Count tool results
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        analysis["tool_results"] += 1
        
        # Check for issues
        if analysis["tool_uses"] != analysis["tool_results"]:
            analysis["issues"].append(
                f"Mismatch: {analysis['tool_uses']} tool uses but "
                f"{analysis['tool_results']} tool results"
            )
        
        if analysis["user_messages"] == 0:
            analysis["issues"].append("No user messages in conversation")
        
        if analysis["assistant_messages"] == 0:
            analysis["issues"].append("No assistant messages in conversation")
        
        return analysis
    
    @staticmethod
    def test_tool_definitions(tools: list[dict]) -> dict[str, Any]:
        """
        Test if tool definitions work with the API.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Test results
        """
        results = {
            "tools_tested": len(tools),
            "successful": [],
            "failed": [],
        }
        
        for tool in tools:
            tool_name = tool.get("name", "unknown")
            
            try:
                # Try to make an API call with this tool
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": f"Use the {tool_name} tool to help me."
                    }],
                    tools=[tool]
                )
                
                results["successful"].append({
                    "name": tool_name,
                    "status": "ok",
                    "stop_reason": response.stop_reason,
                })
            
            except Exception as e:
                results["failed"].append({
                    "name": tool_name,
                    "error": str(e),
                })
        
        return results


def run_full_diagnostics(
    tools: Optional[list[dict]] = None,
    conversation: Optional[list[dict]] = None
) -> dict[str, Any]:
    """
    Run a complete diagnostic check.
    
    Args:
        tools: Tool definitions to check
        conversation: Conversation history to analyze
        
    Returns:
        Complete diagnostic report
    """
    print("Running diagnostics...\n")
    
    diagnostics = AgentDiagnostics()
    report = {}
    
    # Check API connectivity
    print("1. Checking API connectivity...")
    api_check = diagnostics.check_api_connectivity()
    report["api_connectivity"] = api_check
    
    if api_check["connected"]:
        print("   ✅ Connected to API")
    else:
        print(f"   ❌ Connection failed: {api_check.get('message')}")
    
    # Check tools if provided
    if tools:
        print("\n2. Checking tool configurations...")
        tool_check = diagnostics.check_tools_configuration(tools)
        report["tool_configuration"] = tool_check
        
        if tool_check["valid"]:
            print(f"   ✅ All {tool_check['tool_count']} tools valid")
        else:
            print(f"   ❌ Issues found: {len(tool_check['issues'])}")
            for issue in tool_check["issues"]:
                print(f"      - {issue}")
        
        if tool_check["warnings"]:
            print(f"   ⚠️  Warnings: {len(tool_check['warnings'])}")
            for warning in tool_check["warnings"]:
                print(f"      - {warning}")
    
    # Analyze conversation if provided
    if conversation:
        print("\n3. Analyzing conversation flow...")
        flow_analysis = diagnostics.analyze_conversation_flow(conversation)
        report["conversation_flow"] = flow_analysis
        
        print(f"   Total turns: {flow_analysis['total_turns']}")
        print(f"   Tool uses: {flow_analysis['tool_uses']}")
        print(f"   Tool results: {flow_analysis['tool_results']}")
        
        if flow_analysis["issues"]:
            print(f"   ⚠️  Issues found:")
            for issue in flow_analysis["issues"]:
                print(f"      - {issue}")
        else:
            print("   ✅ No issues detected")
    
    return report


# Example usage
if __name__ == "__main__":
    print("Agent Diagnostic Tools\n")
    print("=" * 50)
    print()
    
    # Example 1: Check tool configuration
    print("Example 1: Checking tool configuration")
    print("-" * 50)
    
    example_tools = [
        {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "search",
            # Missing description!
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}  # Missing description!
                },
                "required": ["query"]
            }
        }
    ]
    
    diagnostics = AgentDiagnostics()
    result = diagnostics.check_tools_configuration(example_tools)
    
    print(f"Valid: {result['valid']}")
    print(f"Issues: {result['issues']}")
    print(f"Warnings: {result['warnings']}")
    
    # Example 2: Run full diagnostics
    print("\n\nExample 2: Full diagnostic report")
    print("-" * 50)
    
    report = run_full_diagnostics(tools=example_tools)
    
    print("\n\nFull report:")
    print(json.dumps(report, indent=2))
