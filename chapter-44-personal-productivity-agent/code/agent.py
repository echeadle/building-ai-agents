"""
Personal productivity agent implementation.

Chapter 44: Project - Personal Productivity Agent
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Dict, List, Any, Optional
import json

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class ProductivityAgent:
    """An agent that helps manage tasks, notes, and personal productivity."""
    
    def __init__(self, storage: 'Storage'):
        """
        Initialize the productivity agent.
        
        Args:
            storage: Storage instance for persistent data
        """
        self.storage = storage
        self.client = anthropic.Anthropic()
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Import tool functions
        from tools import (
            create_task, list_tasks, update_task,
            save_note, search_notes,
            get_context, update_context
        )
        
        self.tool_functions = {
            "create_task": create_task,
            "list_tasks": list_tasks,
            "update_task": update_task,
            "save_note": save_note,
            "search_notes": search_notes,
            "get_context": get_context,
            "update_context": update_context
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a helpful personal productivity assistant. You help users manage their tasks, organize notes, and stay productive.

Your capabilities:
- Create, list, and update tasks
- Save and search notes
- Track projects and context
- Remember user preferences
- Provide context-aware assistance

Guidelines:
1. Be proactive - suggest tasks or notes when relevant
2. Be concise - users are busy
3. Be context-aware - remember what the user is working on
4. Be helpful - offer to do things, don't just answer questions
5. Be organized - help users stay on top of their work

When users ask to "remind me" or "follow up", create a task.
When users share information worth remembering, offer to save a note.
When appropriate, retrieve context to understand what they're working on.

Always prioritize user productivity and organization."""
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for Claude."""
        return [
            {
                "name": "create_task",
                "description": "Create a new task. Use this when the user wants to add something to their todo list, set a reminder, or track something to do.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short, clear task title"
                        },
                        "description": {
                            "type": "string",
                            "description": "Additional details about the task"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Task priority"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in YYYY-MM-DD format"
                        },
                        "project": {
                            "type": "string",
                            "description": "Project this task belongs to"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorizing the task"
                        }
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "list_tasks",
                "description": "List tasks with optional filters. Use this to show the user their tasks, or to check what's on their plate.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done", "archived"],
                            "description": "Filter by task status"
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project name"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Filter by priority"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag"
                        }
                    }
                }
            },
            {
                "name": "update_task",
                "description": "Update an existing task. Use this when the user wants to mark something as done, change priority, or update details.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to update"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["todo", "in_progress", "done", "archived"],
                            "description": "New status"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "New priority"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "New due date in YYYY-MM-DD format"
                        },
                        "description": {
                            "type": "string",
                            "description": "Updated description"
                        },
                        "project": {
                            "type": "string",
                            "description": "New project"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Updated tags"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "save_note",
                "description": "Save a new note. Use this when the user shares information worth remembering, has an idea, or wants to document something.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Note title or summary"
                        },
                        "content": {
                            "type": "string",
                            "description": "Full note content"
                        },
                        "project": {
                            "type": "string",
                            "description": "Project this note relates to"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorizing the note"
                        },
                        "linked_tasks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Task IDs this note relates to"
                        }
                    },
                    "required": ["title", "content"]
                }
            },
            {
                "name": "search_notes",
                "description": "Search for notes by keyword or tag. Use this when the user wants to find information they saved previously.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (matches title and content)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project"
                        }
                    }
                }
            },
            {
                "name": "get_context",
                "description": "Get the user's current context, including active project, preferences, and recent activity. Use this when you need to understand what the user is working on.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "update_context",
                "description": "Update user context or preferences. Use this when the user mentions what they're working on or shares preferences.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_project": {
                            "type": "string",
                            "description": "Set the current active project"
                        },
                        "preferences": {
                            "type": "object",
                            "description": "Update user preferences"
                        }
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool and return the result as a string.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
        
        Returns:
            JSON string of tool result
        """
        try:
            tool_func = self.tool_functions[tool_name]
            
            # All our tools need the storage instance
            result = tool_func(self.storage, **tool_input)
            
            return json.dumps(result, indent=2)
        
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            })
    
    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.
        
        Args:
            user_message: User's input message
        
        Returns:
            Agent's text response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Agentic loop
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call Claude with tools
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self._get_system_prompt(),
                messages=self.conversation_history,
                tools=self._get_tools()
            )
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract text response
                text_response = ""
                for block in response.content:
                    if block.type == "text":
                        text_response += block.text
                return text_response
            
            # Execute tools if requested
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                # Add tool results to history
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Continue the loop to process tool results
                continue
        
        return "I apologize, but I've reached my processing limit. Could you rephrase your request?"
    
    def reset_conversation(self) -> None:
        """Reset the conversation history (but keep persistent data)."""
        self.conversation_history = []


def main():
    """Demo the productivity agent."""
    from storage import Storage
    
    # Initialize storage
    storage = Storage()
    
    # Create agent
    agent = ProductivityAgent(storage)
    
    print("Personal Productivity Agent")
    print("==========================")
    print("Type 'quit' to exit, 'reset' to start a new conversation\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'reset':
            agent.reset_conversation()
            print("\nConversation reset (data preserved)\n")
            continue
        
        # Get response
        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
