"""
Exercise Solution: Designing Tool Definitions for a Personal Assistant

This file contains example tool definitions for a personal assistant
that can check weather, manage a calendar, send emails, and set reminders.

The focus is on writing clear, detailed tool definitions that help
Claude understand WHEN and HOW to use each tool.

Chapter 7: Introduction to Tool Use
"""

import json


def get_personal_assistant_tools() -> list:
    """
    Define tools for a personal assistant agent.
    
    Each tool definition includes:
    - name: A clear, descriptive identifier
    - description: Detailed explanation for Claude about when/how to use it
    - input_schema: JSON Schema defining required and optional parameters
    
    Returns:
        A list of tool definitions
    """
    tools = [
        # Tool 1: Weather
        {
            "name": "get_weather",
            "description": (
                "Get current weather conditions and forecast for a specified city. "
                "Use this tool when the user asks about weather, temperature, "
                "precipitation, or wants to know what to wear or whether to bring "
                "an umbrella. This tool provides current conditions and can give "
                "forecasts for up to 7 days ahead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "The city name to get weather for. Can include country "
                            "for clarity, e.g., 'Paris, France' or 'Portland, Oregon'"
                        )
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": (
                            "Temperature units. Use 'celsius' for most of the world, "
                            "'fahrenheit' for US users. Default is 'celsius' if not specified."
                        )
                    },
                    "days": {
                        "type": "integer",
                        "description": (
                            "Number of days for forecast (1-7). Use 1 for just today's "
                            "weather, higher numbers for multi-day forecasts. Default is 1."
                        ),
                        "minimum": 1,
                        "maximum": 7
                    }
                },
                "required": ["city"]
            }
        },
        
        # Tool 2: Calendar
        {
            "name": "check_calendar",
            "description": (
                "Look up events on the user's calendar. Use this tool when the user "
                "asks about their schedule, appointments, meetings, or availability. "
                "Can query specific dates, date ranges, or find specific events by name. "
                "Returns event details including time, title, location, and attendees."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": (
                            "The date to check in YYYY-MM-DD format. For queries like "
                            "'tomorrow' or 'next Monday', convert to the actual date."
                        )
                    },
                    "end_date": {
                        "type": "string",
                        "description": (
                            "Optional end date for range queries (YYYY-MM-DD). "
                            "If not provided, only the single date is checked."
                        )
                    },
                    "search_term": {
                        "type": "string",
                        "description": (
                            "Optional text to search for in event titles or descriptions. "
                            "Use this to find specific events like 'dentist' or 'team meeting'."
                        )
                    }
                },
                "required": ["date"]
            }
        },
        
        # Tool 3: Add Calendar Event
        {
            "name": "add_calendar_event",
            "description": (
                "Add a new event to the user's calendar. Use this when the user wants "
                "to schedule a meeting, appointment, or reminder. Always confirm the "
                "details with the user before creating the event if any information "
                "is ambiguous or missing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The name/title of the event"
                    },
                    "date": {
                        "type": "string",
                        "description": "Event date in YYYY-MM-DD format"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in HH:MM format (24-hour)"
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "End time in HH:MM format (24-hour). "
                            "If not provided, default to 1 hour after start."
                        )
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location or meeting link"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional notes or description for the event"
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of attendee email addresses"
                    }
                },
                "required": ["title", "date", "start_time"]
            }
        },
        
        # Tool 4: Send Email
        {
            "name": "send_email",
            "description": (
                "Send an email on behalf of the user. Use this when the user explicitly "
                "asks to send, write, or compose an email. IMPORTANT: Always show the "
                "user the email content and ask for confirmation before actually sending. "
                "Never send emails automatically without user approval."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of recipient email addresses. Can include names in "
                            "format 'Name <email@example.com>' or just email addresses."
                        )
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "The main content of the email. Can include basic formatting "
                            "like line breaks. Should be professional unless user specifies "
                            "otherwise."
                        )
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of CC recipients"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Email priority. Default is 'normal'."
                    },
                    "draft_only": {
                        "type": "boolean",
                        "description": (
                            "If true, save as draft instead of sending immediately. "
                            "Use this when user wants to review before sending."
                        )
                    }
                },
                "required": ["to", "subject", "body"]
            }
        },
        
        # Tool 5: Set Reminder
        {
            "name": "set_reminder",
            "description": (
                "Set a reminder for the user at a specific time. Use this when the user "
                "asks to be reminded about something, wants to set an alarm or alert, "
                "or needs a notification at a future time. Reminders are different from "
                "calendar events - they're simple alerts without duration or attendees."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": (
                            "The reminder message to show the user. Should be clear "
                            "and actionable, e.g., 'Take medication' or 'Call mom'"
                        )
                    },
                    "datetime": {
                        "type": "string",
                        "description": (
                            "When to trigger the reminder in ISO 8601 format "
                            "(YYYY-MM-DDTHH:MM:SS). Convert relative times like "
                            "'in 30 minutes' or 'tomorrow at 9am' to absolute times."
                        )
                    },
                    "recurring": {
                        "type": "string",
                        "enum": ["none", "daily", "weekly", "monthly"],
                        "description": (
                            "Whether the reminder should repeat. "
                            "Default is 'none' for one-time reminders."
                        )
                    }
                },
                "required": ["message", "datetime"]
            }
        }
    ]
    
    return tools


def print_tool_definitions() -> None:
    """
    Print the tool definitions in a readable format.
    """
    tools = get_personal_assistant_tools()
    
    print("=" * 70)
    print("PERSONAL ASSISTANT TOOL DEFINITIONS")
    print("=" * 70)
    
    for i, tool in enumerate(tools, 1):
        print(f"\n{'─' * 70}")
        print(f"TOOL {i}: {tool['name']}")
        print("─" * 70)
        
        print(f"\n📝 Description:")
        # Wrap description for readability
        desc = tool['description']
        words = desc.split()
        line = "   "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "   " + word
            else:
                line += " " + word if line.strip() else word
        print(line)
        
        print(f"\n📋 Parameters:")
        schema = tool['input_schema']
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        for param_name, param_info in properties.items():
            req_status = "required" if param_name in required else "optional"
            param_type = param_info.get('type', 'unknown')
            
            # Handle array types
            if param_type == 'array':
                items_type = param_info.get('items', {}).get('type', 'unknown')
                param_type = f"array of {items_type}s"
            
            # Handle enums
            if 'enum' in param_info:
                param_type = f"string, one of: {param_info['enum']}"
            
            print(f"\n   • {param_name} ({param_type}, {req_status})")
            print(f"     {param_info.get('description', 'No description')}")
    
    print("\n" + "=" * 70)


def print_json_format() -> None:
    """
    Print the tool definitions as JSON (the format Claude's API expects).
    """
    tools = get_personal_assistant_tools()
    
    print("\n" + "=" * 70)
    print("TOOL DEFINITIONS AS JSON (for Claude API)")
    print("=" * 70)
    print("\nThis is the exact format you'd pass to the API's 'tools' parameter:\n")
    print(json.dumps(tools, indent=2))


def main() -> None:
    """
    Main function to run the exercise solution.
    """
    print("\n" + "=" * 70)
    print("Chapter 7 Exercise Solution: Personal Assistant Tools")
    print("=" * 70)
    print("""
This exercise demonstrates how to design tool definitions that:

1. Have clear, descriptive names
2. Include detailed descriptions Claude can understand
3. Define parameters with proper types and descriptions
4. Distinguish between required and optional parameters
5. Include guidance on when to use each tool

Key Design Principles Demonstrated:
─────────────────────────────────────
• Tool descriptions explain WHEN to use the tool, not just WHAT it does
• Parameters include examples and format specifications
• Safety considerations are noted (e.g., confirming before sending email)
• Related tools are distinguished (calendar events vs reminders)
""")
    
    # Print human-readable format
    print_tool_definitions()
    
    # Ask if user wants to see JSON format
    print("\n" + "─" * 70)
    print("The above shows tool definitions in a readable format.")
    print("The JSON format (what the API expects) is also available.")
    print("Run with --json flag to see: python exercise.py --json")
    

if __name__ == "__main__":
    import sys
    
    if "--json" in sys.argv:
        print_json_format()
    else:
        main()
