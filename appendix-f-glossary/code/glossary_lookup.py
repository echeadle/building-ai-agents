"""
Glossary Term Lookup Utility

A simple tool for quickly finding definitions from the glossary.
Useful for reference while working through examples or building agents.

Appendix F: Glossary
"""

from typing import Dict, List, Optional


class GlossaryLookup:
    """
    A glossary lookup utility for agent development terms.
    
    This provides quick access to definitions without having to
    search through the full glossary document.
    """
    
    def __init__(self):
        """Initialize the glossary with all terms and definitions."""
        self.terms: Dict[str, Dict[str, str]] = {
            # Core Concepts
            "agent": {
                "definition": "A software system that uses a large language model to make decisions, take actions via tools, and work autonomously toward goals.",
                "related": ["agentic_loop", "workflow", "augmented_llm"],
                "chapter": "1"
            },
            "agentic_loop": {
                "definition": "The core pattern of agent execution: (1) Send messages to LLM, (2) Check if LLM wants tools, (3) Execute tools, (4) Send results back, (5) Repeat until complete.",
                "related": ["tool", "iteration"],
                "chapter": "12"
            },
            "augmented_llm": {
                "definition": "A large language model enhanced with tools, a system prompt, and structured configuration. The fundamental building block of agents.",
                "related": ["tool", "system_prompt"],
                "chapter": "14"
            },
            
            # API and Communication
            "api_key": {
                "definition": "A secret token that authenticates requests to the Anthropic API. Should never be hardcoded.",
                "related": ["environment_variable", "dotenv"],
                "chapter": "3"
            },
            "message": {
                "definition": "A single unit in a conversation, containing a role (system/user/assistant) and content.",
                "related": ["role", "conversation_history"],
                "chapter": "5"
            },
            "role": {
                "definition": "The speaker in a message: 'system' (instructions), 'user' (human), 'assistant' (LLM).",
                "related": ["message", "system_prompt"],
                "chapter": "5"
            },
            "system_prompt": {
                "definition": "Instructions that define an agent's behavior, capabilities, and constraints. The agent's 'constitution'.",
                "related": ["role", "prompt_engineering"],
                "chapter": "6"
            },
            
            # Tools
            "tool": {
                "definition": "A function that an agent can call to interact with the external world. Examples: web search, calculator, database query.",
                "related": ["function_calling", "tool_definition"],
                "chapter": "7"
            },
            "tool_definition": {
                "definition": "The JSON schema describing a tool to the LLM, including name, description, and parameters.",
                "related": ["input_schema", "tool"],
                "chapter": "8"
            },
            "tool_use": {
                "definition": "When the LLM decides to call a tool, returning structured JSON specifying the tool name and parameters.",
                "related": ["tool_result", "agentic_loop"],
                "chapter": "9"
            },
            
            # Workflows
            "workflow": {
                "definition": "A structured pattern for executing multi-step agent tasks. Five core workflows: Chaining, Routing, Parallel, Orchestrator-Workers, Evaluator-Optimizer.",
                "related": ["chain", "router", "orchestrator"],
                "chapter": "15"
            },
            "chain": {
                "definition": "A workflow pattern where tasks are broken into sequential steps, each step's output feeding the next.",
                "related": ["quality_gate", "workflow"],
                "chapter": "16-17"
            },
            "router": {
                "definition": "A workflow pattern that examines input and routes it to the appropriate handler or specialized agent.",
                "related": ["workflow"],
                "chapter": "20-21"
            },
            
            # Technical Concepts
            "token": {
                "definition": "The basic unit of text for LLMs. Roughly 4 characters or 0.75 words. Used to measure size and costs.",
                "related": ["max_tokens", "context_window"],
                "chapter": "4"
            },
            "context_window": {
                "definition": "The maximum amount of text (in tokens) that an LLM can process in a single request.",
                "related": ["token", "conversation_history"],
                "chapter": "5"
            },
            "hallucination": {
                "definition": "When an LLM generates plausible-sounding but incorrect or fabricated information.",
                "related": ["grounding", "tool"],
                "chapter": "1"
            },
            
            # Production Concepts
            "observability": {
                "definition": "The ability to understand what an agent is doing and why. Achieved through logging, metrics, and tracing.",
                "related": ["logging", "monitoring"],
                "chapter": "29"
            },
            "rate_limit": {
                "definition": "A restriction on how many API requests you can make in a time period.",
                "related": ["retry_logic", "backoff_strategy"],
                "chapter": "37"
            },
            "retry_logic": {
                "definition": "Code that automatically retries failed operations, typically with exponential backoff.",
                "related": ["backoff_strategy", "error_handling"],
                "chapter": "37"
            },
        }
        
        # Build reverse lookup for related terms
        self._build_reverse_index()
    
    def _build_reverse_index(self) -> None:
        """Build an index of which terms reference each term."""
        self.referenced_by: Dict[str, List[str]] = {}
        
        for term, data in self.terms.items():
            for related_term in data.get("related", []):
                if related_term not in self.referenced_by:
                    self.referenced_by[related_term] = []
                self.referenced_by[related_term].append(term)
    
    def lookup(self, term: str) -> Optional[Dict[str, str]]:
        """
        Look up a term in the glossary.
        
        Args:
            term: The term to look up (case-insensitive, underscores or spaces OK)
            
        Returns:
            Dictionary with definition and metadata, or None if not found
        """
        # Normalize the term
        normalized = term.lower().replace(" ", "_")
        
        return self.terms.get(normalized)
    
    def search(self, query: str) -> List[str]:
        """
        Search for terms matching a query.
        
        Args:
            query: Search string (partial matches OK)
            
        Returns:
            List of matching term names
        """
        query_lower = query.lower()
        matches = []
        
        for term, data in self.terms.items():
            # Search in term name
            if query_lower in term.lower():
                matches.append(term)
                continue
            
            # Search in definition
            if query_lower in data["definition"].lower():
                matches.append(term)
        
        return matches
    
    def get_related(self, term: str) -> List[str]:
        """
        Get terms related to the given term.
        
        Args:
            term: The term to find relations for
            
        Returns:
            List of related term names
        """
        normalized = term.lower().replace(" ", "_")
        data = self.terms.get(normalized)
        
        if not data:
            return []
        
        return data.get("related", [])
    
    def get_chapter(self, term: str) -> Optional[str]:
        """
        Get the chapter where a term is introduced or explained.
        
        Args:
            term: The term to look up
            
        Returns:
            Chapter reference or None
        """
        normalized = term.lower().replace(" ", "_")
        data = self.terms.get(normalized)
        
        if not data:
            return None
        
        return data.get("chapter")
    
    def list_all_terms(self) -> List[str]:
        """Get a list of all available terms."""
        return sorted(self.terms.keys())
    
    def print_term(self, term: str) -> None:
        """
        Print a formatted definition for a term.
        
        Args:
            term: The term to print
        """
        data = self.lookup(term)
        
        if not data:
            print(f"Term '{term}' not found in glossary.")
            print(f"\nDid you mean one of these?")
            matches = self.search(term)
            for match in matches[:5]:
                print(f"  - {match}")
            return
        
        print(f"\n{'='*60}")
        print(f"TERM: {term.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(f"\nDefinition:")
        print(f"  {data['definition']}")
        
        if data.get('related'):
            print(f"\nRelated Terms:")
            for related in data['related']:
                print(f"  - {related.replace('_', ' ')}")
        
        if data.get('chapter'):
            print(f"\nIntroduced in: Chapter {data['chapter']}")
        
        print(f"{'='*60}\n")


def main():
    """Demonstrate the glossary lookup utility."""
    glossary = GlossaryLookup()
    
    print("Agent Development Glossary Lookup")
    print("=" * 60)
    
    # Example lookups
    print("\n1. Looking up 'agent':")
    glossary.print_term("agent")
    
    print("\n2. Looking up 'agentic loop':")
    glossary.print_term("agentic loop")
    
    print("\n3. Searching for terms related to 'tool':")
    matches = glossary.search("tool")
    print(f"Found {len(matches)} terms: {', '.join(matches)}")
    
    print("\n4. Finding related terms for 'augmented_llm':")
    related = glossary.get_related("augmented_llm")
    print(f"Related terms: {', '.join(related)}")
    
    print("\n5. All available terms:")
    all_terms = glossary.list_all_terms()
    print(f"Total terms: {len(all_terms)}")
    print("First 10:", ", ".join(all_terms[:10]))
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Try it yourself!")
    print("=" * 60)
    
    while True:
        term = input("\nEnter a term to look up (or 'quit' to exit): ").strip()
        
        if term.lower() in ['quit', 'exit', 'q']:
            break
        
        if not term:
            continue
        
        glossary.print_term(term)


if __name__ == "__main__":
    main()
