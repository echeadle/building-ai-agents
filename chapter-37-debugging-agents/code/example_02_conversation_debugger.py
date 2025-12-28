"""
Conversation flow debugger.

Chapter 37: Debugging Agents

This module provides tools for analyzing conversation structure,
detecting flow issues, finding derailment points, and suggesting
context reduction strategies.
"""

import os
from typing import Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ConversationAnalysis:
    """Analysis results for a conversation."""
    total_messages: int
    total_tokens_estimate: int
    user_messages: int
    assistant_messages: int
    tool_uses: int
    tool_results: int
    system_prompt_present: bool
    potential_issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class ConversationDebugger:
    """
    Analyzes conversation structure to identify flow issues.
    
    Common issues detected:
    - Missing system prompt
    - Unbalanced user/assistant turns
    - Tool calls without results
    - Very long messages that may truncate
    - Context window exhaustion
    """
    
    # Approximate tokens per character (rough estimate)
    TOKENS_PER_CHAR = 0.25
    
    # Warning thresholds
    MAX_RECOMMENDED_TOKENS = 150000  # Before context issues start
    LONG_MESSAGE_THRESHOLD = 10000  # Characters
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.TOKENS_PER_CHAR)
    
    def analyze_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> ConversationAnalysis:
        """
        Analyze a conversation for potential issues.
        
        Args:
            messages: The messages array
            system_prompt: Optional system prompt
        
        Returns:
            ConversationAnalysis with findings
        """
        analysis = ConversationAnalysis(
            total_messages=len(messages),
            total_tokens_estimate=0,
            user_messages=0,
            assistant_messages=0,
            tool_uses=0,
            tool_results=0,
            system_prompt_present=system_prompt is not None
        )
        
        # Count system prompt tokens
        if system_prompt:
            analysis.total_tokens_estimate += self.estimate_tokens(system_prompt)
        
        pending_tool_calls: dict[str, str] = {}  # tool_use_id -> tool_name
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Count by role
            if role == "user":
                analysis.user_messages += 1
            elif role == "assistant":
                analysis.assistant_messages += 1
            
            # Estimate tokens for this message
            if isinstance(content, str):
                msg_tokens = self.estimate_tokens(content)
                analysis.total_tokens_estimate += msg_tokens
                
                # Check for very long messages
                if len(content) > self.LONG_MESSAGE_THRESHOLD:
                    analysis.potential_issues.append(
                        f"Message {i} is very long ({len(content)} chars). "
                        "Consider summarizing."
                    )
            
            elif isinstance(content, list):
                # Handle content blocks (tool use, tool results, etc.)
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        
                        if block_type == "tool_use":
                            analysis.tool_uses += 1
                            tool_id = block.get("id", "")
                            tool_name = block.get("name", "unknown")
                            pending_tool_calls[tool_id] = tool_name
                            
                            # Estimate tokens for tool call
                            input_str = str(block.get("input", {}))
                            analysis.total_tokens_estimate += self.estimate_tokens(input_str)
                        
                        elif block_type == "tool_result":
                            analysis.tool_results += 1
                            tool_id = block.get("tool_use_id", "")
                            
                            # Remove from pending
                            if tool_id in pending_tool_calls:
                                del pending_tool_calls[tool_id]
                            else:
                                analysis.potential_issues.append(
                                    f"Tool result for unknown tool_use_id: {tool_id}"
                                )
                            
                            # Estimate tokens for result
                            result_content = block.get("content", "")
                            analysis.total_tokens_estimate += self.estimate_tokens(
                                str(result_content)
                            )
                        
                        elif block_type == "text":
                            text = block.get("text", "")
                            analysis.total_tokens_estimate += self.estimate_tokens(text)
        
        # Check for issues
        if not analysis.system_prompt_present:
            analysis.potential_issues.append(
                "No system prompt detected. Agent may lack clear instructions."
            )
            analysis.recommendations.append(
                "Add a system prompt to define agent behavior and available tools."
            )
        
        if pending_tool_calls:
            analysis.potential_issues.append(
                f"Unanswered tool calls: {list(pending_tool_calls.values())}"
            )
            analysis.recommendations.append(
                "Ensure every tool_use block has a corresponding tool_result."
            )
        
        if analysis.tool_uses != analysis.tool_results:
            analysis.potential_issues.append(
                f"Tool use/result mismatch: {analysis.tool_uses} calls, "
                f"{analysis.tool_results} results"
            )
        
        if analysis.total_tokens_estimate > self.MAX_RECOMMENDED_TOKENS:
            analysis.potential_issues.append(
                f"Estimated {analysis.total_tokens_estimate} tokens. "
                "Context window may be exhausted."
            )
            analysis.recommendations.append(
                "Summarize or truncate older messages to reduce context size."
            )
        
        # Check turn balance
        if abs(analysis.user_messages - analysis.assistant_messages) > 2:
            analysis.potential_issues.append(
                f"Unbalanced turns: {analysis.user_messages} user, "
                f"{analysis.assistant_messages} assistant"
            )
        
        return analysis
    
    def find_derailment_point(
        self,
        messages: list[dict[str, Any]],
        expected_topic: str
    ) -> Optional[int]:
        """
        Find where a conversation went off-topic.
        
        Uses Claude to analyze each turn for relevance to the expected topic.
        
        Args:
            messages: The conversation messages
            expected_topic: What the conversation should be about
        
        Returns:
            Index of first off-topic message, or None if all on-topic
        """
        analysis_prompt = f"""Analyze if this message is relevant to the topic: "{expected_topic}"

Message: {{message}}

Respond with ONLY "relevant" or "off-topic" followed by a brief explanation."""
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    block.get("text", "") 
                    for block in content 
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            
            if not content or len(content) < 20:
                continue
            
            # Analyze this message
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": analysis_prompt.format(message=content[:500])
                }]
            )
            
            result = response.content[0].text.lower()
            if "off-topic" in result:
                return i
        
        return None
    
    def suggest_context_reduction(
        self,
        messages: list[dict[str, Any]],
        target_reduction: float = 0.5
    ) -> list[str]:
        """
        Suggest which messages can be removed or summarized.
        
        Args:
            messages: The conversation messages
            target_reduction: Target reduction ratio (0.5 = reduce by half)
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Find tool result messages (often verbose)
        for i, msg in enumerate(messages):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_content = str(block.get("content", ""))
                        if len(result_content) > 1000:
                            suggestions.append(
                                f"Message {i}: Summarize tool result "
                                f"({len(result_content)} chars)"
                            )
        
        # Find repetitive user messages
        user_contents = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                for j, prev_content in enumerate(user_contents):
                    # Simple similarity check
                    if len(set(content.split()) & set(prev_content.split())) > 10:
                        suggestions.append(
                            f"Message {i}: May be repetitive with earlier message"
                        )
                        break
                user_contents.append(content)
        
        # Suggest removing very old messages
        if len(messages) > 20:
            suggestions.append(
                f"Consider summarizing messages 0-{len(messages)//2} "
                "into a single context message"
            )
        
        return suggestions


def print_analysis(analysis: ConversationAnalysis) -> None:
    """Print a formatted analysis report."""
    print("\n" + "=" * 60)
    print("CONVERSATION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal messages: {analysis.total_messages}")
    print(f"Estimated tokens: {analysis.total_tokens_estimate:,}")
    print(f"User messages: {analysis.user_messages}")
    print(f"Assistant messages: {analysis.assistant_messages}")
    print(f"Tool uses: {analysis.tool_uses}")
    print(f"Tool results: {analysis.tool_results}")
    print(f"System prompt: {'Yes' if analysis.system_prompt_present else 'No'}")
    
    if analysis.potential_issues:
        print("\n⚠️  POTENTIAL ISSUES:")
        for issue in analysis.potential_issues:
            print(f"  • {issue}")
    
    if analysis.recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in analysis.recommendations:
            print(f"  • {rec}")
    
    print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION DEBUGGER DEMONSTRATION")
    print("=" * 60)
    
    debugger = ConversationDebugger()
    
    # Example conversation with issues
    problematic_conversation = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll check the weather for you."},
            {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "Tokyo"}}
        ]},
        # Missing tool_result!
        {"role": "user", "content": "Also, what about Paris?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tool_2", "name": "weather", "input": {"city": "Paris"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tool_2", "content": "Paris: 18°C, cloudy"}
        ]},
        # Multiple user messages in a row
        {"role": "user", "content": "Thanks!"},
        {"role": "user", "content": "One more question..."},
    ]
    
    print("\nAnalyzing problematic conversation...")
    analysis = debugger.analyze_messages(problematic_conversation)
    print_analysis(analysis)
    
    # Example of a healthier conversation
    print("\n" + "-" * 60)
    print("Analyzing a healthier conversation...")
    print("-" * 60)
    
    healthy_conversation = [
        {"role": "user", "content": "What's the weather in London?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check that for you."},
            {"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "London"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tool_1", "content": "London: 12°C, rainy"}
        ]},
        {"role": "assistant", "content": "The weather in London is currently 12°C and rainy."},
    ]
    
    healthy_analysis = debugger.analyze_messages(
        healthy_conversation,
        system_prompt="You are a helpful weather assistant."
    )
    print_analysis(healthy_analysis)
    
    # Suggest context reduction
    print("\n" + "-" * 60)
    print("Context Reduction Suggestions")
    print("-" * 60)
    
    long_conversation = problematic_conversation * 10  # Simulate long conversation
    suggestions = debugger.suggest_context_reduction(long_conversation)
    
    print("\nSuggestions for reducing context:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")
