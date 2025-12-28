"""
Resource Link Validator

This script validates URLs from Appendix G and checks their accessibility.
Useful for maintaining the resource list as links change over time.

Appendix G: Resources and Further Reading
"""

import os
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import time

# Note: requests library needed for actual validation
# This is a demonstration of the pattern - add 'import requests' and
# 'uv add requests' for production use


class ResourceValidator:
    """
    Validates resource links and checks their status.
    
    This class demonstrates how to programmatically maintain
    a list of learning resources and verify they're still accessible.
    """
    
    def __init__(self):
        self.resources = self._load_resources()
    
    def _load_resources(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load all resources from Appendix G.
        
        Returns:
            Dictionary mapping categories to lists of (name, url) tuples
        """
        return {
            "official_docs": [
                ("Claude API Documentation", "https://docs.anthropic.com"),
                ("Building Effective Agents", "https://www.anthropic.com/engineering/building-effective-agents"),
                ("Prompt Engineering Guide", "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering"),
                ("Anthropic Cookbook", "https://github.com/anthropics/anthropic-cookbook"),
                ("Tool Use Guide", "https://docs.anthropic.com/en/docs/build-with-claude/tool-use"),
            ],
            "python_resources": [
                ("Python Documentation", "https://docs.python.org/3/"),
                ("Type Hints PEP 484", "https://www.python.org/dev/peps/pep-0484/"),
                ("asyncio Documentation", "https://docs.python.org/3/library/asyncio.html"),
            ],
            "research_papers": [
                ("Chain-of-Thought", "https://arxiv.org/abs/2201.11903"),
                ("ReAct", "https://arxiv.org/abs/2210.03629"),
                ("Reflexion", "https://arxiv.org/abs/2303.11366"),
                ("Toolformer", "https://arxiv.org/abs/2302.04761"),
                ("Zero-Shot Reasoners", "https://arxiv.org/abs/2205.11916"),
                ("The Prompt Report", "https://arxiv.org/abs/2406.06608"),
                ("Generative Agents", "https://arxiv.org/abs/2304.03442"),
            ],
            "community": [
                ("Anthropic Discord", "https://discord.gg/anthropic"),
                ("r/ClaudeAI", "https://reddit.com/r/ClaudeAI"),
                ("Hacker News", "https://news.ycombinator.com"),
            ],
            "open_source": [
                ("AutoGPT", "https://github.com/Significant-Gravitas/AutoGPT"),
                ("LangChain", "https://github.com/langchain-ai/langchain"),
                ("LlamaIndex", "https://github.com/run-llama/llama_index"),
                ("Semantic Kernel", "https://github.com/microsoft/semantic-kernel"),
            ],
            "tools": [
                ("python-dotenv", "https://github.com/theskumar/python-dotenv"),
                ("Pydantic", "https://github.com/pydantic/pydantic"),
                ("Tenacity", "https://github.com/jd/tenacity"),
                ("Rich", "https://github.com/Textualize/rich"),
                ("Loguru", "https://github.com/Delgan/loguru"),
                ("uv", "https://github.com/astral-sh/uv"),
                ("Ruff", "https://github.com/astral-sh/ruff"),
                ("mypy", "https://github.com/python/mypy"),
                ("pytest", "https://github.com/pytest-dev/pytest"),
            ],
        }
    
    def check_url_format(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL format.
        
        Args:
            url: The URL to validate
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]):
                return True, "Valid URL format"
            else:
                return False, "Missing scheme or netloc"
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def print_resource_summary(self) -> None:
        """Print a summary of all resources."""
        print("=" * 70)
        print("APPENDIX G: RESOURCES AND FURTHER READING")
        print("=" * 70)
        print()
        
        total_resources = 0
        
        for category, resources in self.resources.items():
            category_name = category.replace("_", " ").title()
            print(f"{category_name} ({len(resources)} resources)")
            print("-" * 70)
            
            for name, url in resources:
                is_valid, msg = self.check_url_format(url)
                status = "✓" if is_valid else "✗"
                print(f"  {status} {name}")
                print(f"     {url}")
                if not is_valid:
                    print(f"     Error: {msg}")
                print()
            
            total_resources += len(resources)
        
        print("=" * 70)
        print(f"Total resources: {total_resources}")
        print("=" * 70)
    
    def export_to_markdown(self, filepath: str) -> None:
        """
        Export resources to a markdown file.
        
        Args:
            filepath: Where to save the markdown file
        """
        with open(filepath, 'w') as f:
            f.write("# Quick Reference: Resources from Appendix G\n\n")
            
            for category, resources in self.resources.items():
                category_name = category.replace("_", " ").title()
                f.write(f"## {category_name}\n\n")
                
                for name, url in resources:
                    f.write(f"- **{name}**: [{url}]({url})\n")
                
                f.write("\n")
        
        print(f"✓ Exported to {filepath}")
    
    def get_resources_by_topic(self, topic: str) -> List[Tuple[str, str]]:
        """
        Get resources related to a specific topic.
        
        Args:
            topic: Topic keyword to search for
        
        Returns:
            List of (name, url) tuples matching the topic
        """
        topic_lower = topic.lower()
        results = []
        
        for category, resources in self.resources.items():
            for name, url in resources:
                if (topic_lower in name.lower() or 
                    topic_lower in url.lower() or
                    topic_lower in category.lower()):
                    results.append((name, url))
        
        return results


def search_resources(validator: ResourceValidator, query: str) -> None:
    """
    Search for resources matching a query.
    
    Args:
        validator: The ResourceValidator instance
        query: Search query
    """
    print(f"\nSearching for: '{query}'")
    print("-" * 70)
    
    results = validator.get_resources_by_topic(query)
    
    if results:
        for name, url in results:
            print(f"• {name}")
            print(f"  {url}")
            print()
    else:
        print("No resources found matching your query.")


if __name__ == "__main__":
    # Create validator instance
    validator = ResourceValidator()
    
    # Print full summary
    print("\n")
    validator.print_resource_summary()
    
    # Example: Search for specific topics
    print("\n")
    print("=" * 70)
    print("EXAMPLE SEARCHES")
    print("=" * 70)
    
    search_resources(validator, "agent")
    search_resources(validator, "python")
    search_resources(validator, "tool")
    
    # Export to markdown
    print("\n")
    output_path = "/tmp/resources_quick_reference.md"
    validator.export_to_markdown(output_path)
    print(f"\n💡 Tip: Bookmark {output_path} for quick reference!")
    
    print("\n")
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Start with official documentation:
   • Claude API: https://docs.anthropic.com
   • Building Effective Agents guide

2. Read key research papers:
   • ReAct: https://arxiv.org/abs/2210.03629
   • Chain-of-Thought: https://arxiv.org/abs/2201.11903

3. Join the community:
   • Anthropic Discord: https://discord.gg/anthropic
   • Share what you build!

4. Keep learning:
   • Try the code examples from this book
   • Build your own agents from scratch
   • Contribute to the ecosystem

Happy building! 🚀
    """)
