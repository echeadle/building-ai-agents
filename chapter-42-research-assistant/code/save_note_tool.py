"""
Note-taking tool for the research assistant.

Chapter 42: Project - Research Assistant Agent
"""

from typing import Dict, List, Any


class ResearchNotes:
    """
    Manages research notes for the agent.
    
    This is a stateful tool that keeps track of findings
    as research progresses.
    """
    
    def __init__(self):
        """Initialize empty notes."""
        self.notes: List[Dict[str, str]] = []
        self.sources_read: List[str] = []
    
    def save_note(self, finding: str, source: str = "") -> str:
        """
        Save a research finding.
        
        Args:
            finding: The key finding or insight to save
            source: Optional URL or source identifier
            
        Returns:
            Confirmation message
        """
        note = {
            "finding": finding,
            "source": source
        }
        self.notes.append(note)
        
        if source and source not in self.sources_read:
            self.sources_read.append(source)
        
        return f"Note saved. Total notes: {len(self.notes)}"
    
    def get_all_notes(self) -> List[Dict[str, str]]:
        """Get all saved notes."""
        return self.notes.copy()
    
    def get_summary(self) -> str:
        """Get a summary of research progress."""
        return (
            f"Research progress:\n"
            f"- Notes taken: {len(self.notes)}\n"
            f"- Sources consulted: {len(self.sources_read)}\n"
        )
    
    def clear(self):
        """Clear all notes (start fresh research)."""
        self.notes = []
        self.sources_read = []


# Tool definition for the agent
SAVE_NOTE_TOOL = {
    "name": "save_note",
    "description": (
        "Save an important finding or insight from your research. "
        "Use this to remember key information as you research. "
        "Each note should capture one main idea or fact. "
        "Include the source URL if applicable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "finding": {
                "type": "string",
                "description": "The key finding or insight to save"
            },
            "source": {
                "type": "string",
                "description": "The URL or source of this information (optional)",
                "default": ""
            }
        },
        "required": ["finding"]
    }
}


if __name__ == "__main__":
    # Test the notes system
    print("Testing note-taking system...\n")
    
    notes = ResearchNotes()
    
    # Save some test notes
    print(notes.save_note(
        "Python asyncio uses an event loop for concurrent execution",
        "https://docs.python.org/3/library/asyncio.html"
    ))
    
    print(notes.save_note(
        "AsyncIO is single-threaded but can handle multiple I/O operations",
        "https://realpython.com/async-io-python/"
    ))
    
    print(notes.save_note(
        "Use asyncio.gather() to run multiple coroutines concurrently"
    ))
    
    print("\n" + notes.get_summary())
    
    print("\nAll notes:")
    for i, note in enumerate(notes.get_all_notes(), 1):
        print(f"{i}. {note['finding']}")
        if note['source']:
            print(f"   Source: {note['source']}")
