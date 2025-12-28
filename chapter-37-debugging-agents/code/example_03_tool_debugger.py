"""
Tool selection debugger.

Chapter 37: Debugging Agents

This module provides tools for analyzing and debugging tool selection
decisions. It helps understand why agents choose certain tools and
suggests improvements to tool descriptions.
"""

import os
import json
from typing import Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ToolSelectionAnalysis:
    """Analysis of a tool selection decision."""
    query: str
    expected_tool: Optional[str]
    selected_tool: Optional[str]
    correct: bool
    confidence_explanation: str
    tool_scores: dict[str, str]  # tool_name -> reasoning
    suggestions: list[str]


class ToolSelectionDebugger:
    """
    Debugs tool selection issues.
    
    Helps answer:
    - Why did the agent choose this tool?
    - Why didn't the agent choose the expected tool?
    - How can I improve tool descriptions?
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def analyze_selection(
        self,
        query: str,
        tools: list[dict[str, Any]],
        selected_tool: Optional[str],
        expected_tool: Optional[str] = None
    ) -> ToolSelectionAnalysis:
        """
        Analyze why a particular tool was (or wasn't) selected.
        
        Args:
            query: The user's query
            tools: Available tool definitions
            selected_tool: The tool that was actually selected (None if no tool used)
            expected_tool: The tool you expected to be selected
        
        Returns:
            ToolSelectionAnalysis with detailed findings
        """
        tool_names = [t["name"] for t in tools]
        tool_descriptions = {
            t["name"]: t.get("description", "No description") 
            for t in tools
        }
        
        analysis_prompt = f"""Analyze this tool selection scenario:

USER QUERY: "{query}"

AVAILABLE TOOLS:
{json.dumps(tool_descriptions, indent=2)}

SELECTED TOOL: {selected_tool if selected_tool else "None (no tool used)"}
EXPECTED TOOL: {expected_tool if expected_tool else "Not specified"}

Please analyze:
1. For each tool, explain how well it matches the query (score 1-10)
2. Explain why the selected tool was likely chosen
3. If the selection seems wrong, explain why and suggest fixes

Format your response as JSON:
{{
    "tool_scores": {{"tool_name": "score/10 - reasoning", ...}},
    "selection_reasoning": "why the selected tool was chosen",
    "is_correct": true/false,
    "suggestions": ["suggestion 1", "suggestion 2"]
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Parse the response
        response_text = response.content[0].text
        
        # Extract JSON from response
        try:
            # Find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
            else:
                result = {
                    "tool_scores": {},
                    "selection_reasoning": response_text,
                    "is_correct": selected_tool == expected_tool,
                    "suggestions": []
                }
        except json.JSONDecodeError:
            result = {
                "tool_scores": {},
                "selection_reasoning": response_text,
                "is_correct": selected_tool == expected_tool,
                "suggestions": []
            }
        
        return ToolSelectionAnalysis(
            query=query,
            expected_tool=expected_tool,
            selected_tool=selected_tool,
            correct=result.get("is_correct", selected_tool == expected_tool),
            confidence_explanation=result.get("selection_reasoning", ""),
            tool_scores=result.get("tool_scores", {}),
            suggestions=result.get("suggestions", [])
        )
    
    def suggest_description_improvements(
        self,
        tool: dict[str, Any],
        failed_queries: list[str]
    ) -> list[str]:
        """
        Suggest improvements to a tool description based on queries it failed to match.
        
        Args:
            tool: The tool definition
            failed_queries: Queries where this tool should have been selected but wasn't
        
        Returns:
            List of suggested description improvements
        """
        prompt = f"""A tool with this definition is NOT being selected when it should be:

TOOL NAME: {tool['name']}
CURRENT DESCRIPTION: {tool.get('description', 'No description')}
PARAMETERS: {json.dumps(tool.get('input_schema', {}), indent=2)}

QUERIES WHERE THIS TOOL SHOULD HAVE BEEN USED (BUT WASN'T):
{chr(10).join(f'- "{q}"' for q in failed_queries)}

Suggest 3-5 specific improvements to the tool description that would help the LLM 
recognize when to use this tool. Focus on:
1. Keywords that users might use
2. Clearer explanation of capabilities
3. Examples of when to use it
4. Distinction from similar tools

Format as a numbered list."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse suggestions from response
        response_text = response.content[0].text
        suggestions = []
        
        for line in response_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering/bullets
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    suggestions.append(cleaned)
        
        return suggestions
    
    def test_tool_selection(
        self,
        tools: list[dict[str, Any]],
        test_cases: list[tuple[str, str]]  # (query, expected_tool)
    ) -> dict[str, Any]:
        """
        Test tool selection across multiple queries.
        
        Args:
            tools: Tool definitions
            test_cases: List of (query, expected_tool_name) tuples
        
        Returns:
            Test results with pass/fail for each case
        """
        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "cases": []
        }
        
        for query, expected in test_cases:
            # Make an actual API call to see what gets selected
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                tools=tools,
                messages=[{"role": "user", "content": query}]
            )
            
            # Check if a tool was used
            selected = None
            for block in response.content:
                if block.type == "tool_use":
                    selected = block.name
                    break
            
            passed = selected == expected
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["cases"].append({
                "query": query,
                "expected": expected,
                "selected": selected,
                "passed": passed
            })
        
        return results
    
    def compare_tool_overlap(
        self,
        tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Analyze tools for potential overlap in functionality.
        
        Args:
            tools: Tool definitions
        
        Returns:
            List of potential overlap issues
        """
        overlaps = []
        
        for i, tool1 in enumerate(tools):
            for tool2 in tools[i + 1:]:
                desc1 = tool1.get("description", "").lower()
                desc2 = tool2.get("description", "").lower()
                
                # Find common words (excluding common words)
                common_words = {"the", "a", "an", "to", "for", "and", "or", "is", "are", "in", "on", "with"}
                words1 = set(desc1.split()) - common_words
                words2 = set(desc2.split()) - common_words
                
                common = words1 & words2
                
                if len(common) >= 3:
                    overlaps.append({
                        "tool1": tool1["name"],
                        "tool2": tool2["name"],
                        "common_words": list(common),
                        "suggestion": f"Consider clarifying the difference between {tool1['name']} and {tool2['name']}"
                    })
        
        return overlaps


def print_tool_analysis(analysis: ToolSelectionAnalysis) -> None:
    """Print formatted tool selection analysis."""
    print("\n" + "=" * 60)
    print("TOOL SELECTION ANALYSIS")
    print("=" * 60)
    
    print(f"\nQuery: \"{analysis.query}\"")
    print(f"Expected tool: {analysis.expected_tool or 'Not specified'}")
    print(f"Selected tool: {analysis.selected_tool or 'None'}")
    print(f"Correct: {'✅ Yes' if analysis.correct else '❌ No'}")
    
    if analysis.tool_scores:
        print("\nTool Scores:")
        for tool, score in analysis.tool_scores.items():
            print(f"  • {tool}: {score}")
    
    if analysis.confidence_explanation:
        print(f"\nReasoning: {analysis.confidence_explanation}")
    
    if analysis.suggestions:
        print("\n💡 Suggestions:")
        for suggestion in analysis.suggestions:
            print(f"  • {suggestion}")
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("TOOL SELECTION DEBUGGER DEMONSTRATION")
    print("=" * 60)
    
    debugger = ToolSelectionDebugger()
    
    # Define tools
    tools = [
        {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        },
        {
            "name": "weather",
            "description": "Gets current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "unit_converter",
            "description": "Converts between units of measurement",
            "input_schema": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    ]
    
    # Test a potentially ambiguous query
    print("\nAnalyzing a potentially ambiguous query...")
    analysis = debugger.analyze_selection(
        query="What's 5 kilometers in miles?",
        tools=tools,
        selected_tool="calculator",  # Simulated wrong selection
        expected_tool="unit_converter"
    )
    
    print_tool_analysis(analysis)
    
    # Check for tool overlap
    print("\n" + "-" * 60)
    print("CHECKING FOR TOOL OVERLAP")
    print("-" * 60)
    
    overlaps = debugger.compare_tool_overlap(tools)
    if overlaps:
        print("\nPotential overlaps found:")
        for overlap in overlaps:
            print(f"\n  {overlap['tool1']} vs {overlap['tool2']}")
            print(f"    Common words: {', '.join(overlap['common_words'])}")
            print(f"    {overlap['suggestion']}")
    else:
        print("\n✅ No significant tool overlap detected")
    
    # Get description improvement suggestions
    print("\n" + "-" * 60)
    print("DESCRIPTION IMPROVEMENT SUGGESTIONS")
    print("-" * 60)
    
    print("\nGetting suggestions for improving unit_converter description...")
    suggestions = debugger.suggest_description_improvements(
        tool=tools[2],  # unit_converter
        failed_queries=[
            "What's 5 kilometers in miles?",
            "Convert 100 fahrenheit to celsius",
            "How many pounds is 50 kilograms?"
        ]
    )
    
    print(f"\nFor tool: unit_converter")
    print(f"Current description: {tools[2]['description']}")
    print("\nSuggested improvements:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    # Run tool selection tests (commented out to avoid API costs)
    # print("\n" + "-" * 60)
    # print("RUNNING TOOL SELECTION TESTS")
    # print("-" * 60)
    # 
    # test_cases = [
    #     ("What's 2 + 2?", "calculator"),
    #     ("Weather in Paris?", "weather"),
    #     ("Convert 10 miles to km", "unit_converter"),
    # ]
    # 
    # results = debugger.test_tool_selection(tools, test_cases)
    # print(f"\nResults: {results['passed']}/{results['total']} passed")
