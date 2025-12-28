"""
Comprehensive tool design examples.

Appendix C: Tool Design Patterns
"""

import os
import re
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Example 1: Well-designed basic tool
# ====================================

def get_weather(city: str, units: str = "celsius") -> dict[str, Any]:
    """
    Get current weather for a city.
    
    This demonstrates:
    - Clear naming (verb-based)
    - Required + optional parameters
    - Structured error returns
    - Consistent output format
    """
    # Validate input
    if not city or not city.strip():
        return {
            "success": False,
            "error": "invalid_input",
            "message": "City name cannot be empty",
            "suggestion": "Provide a valid city name like 'London' or 'Tokyo'"
        }
    
    if units not in ["celsius", "fahrenheit"]:
        return {
            "success": False,
            "error": "invalid_input",
            "message": f"Invalid units: {units}",
            "suggestion": "Use 'celsius' or 'fahrenheit'"
        }
    
    # Simulate API call
    # In real implementation, this would call a weather API
    mock_data = {
        "temperature": 18 if units == "celsius" else 64,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 12,
        "last_updated": "2024-12-09T10:30:00Z"
    }
    
    return {
        "success": True,
        "city": city.strip().title(),
        "units": units,
        "data": mock_data
    }


# Tool definition for get_weather
WEATHER_TOOL = {
    "name": "get_weather",
    "description": """Get current weather conditions for any city worldwide.

Use this tool when the user asks about:
- Current weather, temperature, or conditions
- Whether to bring an umbrella or jacket
- Weather for travel planning

Data updates every 15 minutes.
Temperature returned in requested units (default: celsius).

Returns temperature, condition, humidity, wind speed, and last update time.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g., 'London', 'Tokyo', 'New York'). Can include country for disambiguation."
            },
            "units": {
                "type": "string",
                "description": "Temperature units: 'celsius' or 'fahrenheit' (default: celsius)",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["city"]
    }
}


# Example 2: Database query tool with validation
# ==============================================

def query_database(query: str) -> dict[str, Any]:
    """
    Execute SQL query with safety checks.
    
    This demonstrates:
    - Input validation
    - Security constraints
    - Clear error messages
    - Helpful suggestions
    """
    # Security: Only allow SELECT
    query_upper = query.strip().upper()
    
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            return {
                "success": False,
                "error": "forbidden_operation",
                "message": f"Operation '{keyword}' is not allowed",
                "suggestion": "Only SELECT queries are permitted. Use search tools for data retrieval."
            }
    
    # Validate it's a SELECT
    if not query_upper.startswith("SELECT"):
        return {
            "success": False,
            "error": "invalid_query",
            "message": "Query must start with SELECT",
            "suggestion": "Example: SELECT name, price FROM products WHERE category = 'electronics'"
        }
    
    # Mock execution
    # In real implementation, this would execute against actual database
    mock_results = [
        {"name": "Laptop", "price": 999.99, "category": "electronics"},
        {"name": "Mouse", "price": 29.99, "category": "electronics"}
    ]
    
    return {
        "success": True,
        "query": query,
        "row_count": len(mock_results),
        "results": mock_results
    }


DATABASE_TOOL = {
    "name": "query_database",
    "description": """Execute SQL queries on the product database.

Use this tool when the user asks about:
- Product inventory, prices, or details
- Sales data or statistics
- Customer orders or history

IMPORTANT: Only SELECT queries allowed. No INSERT, UPDATE, DELETE, or DROP.

Database schema:
- products: id, name, price, category, stock
- orders: id, customer_id, product_id, quantity, date
- customers: id, name, email, created_at

Returns query results as a list of dictionaries.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query. Example: 'SELECT name, price FROM products WHERE category = \"electronics\" LIMIT 10'"
            }
        },
        "required": ["query"]
    }
}


# Example 3: File operations with comprehensive validation
# ========================================================

def read_file(file_path: str, encoding: str = "utf-8") -> dict[str, Any]:
    """
    Read file contents with proper error handling.
    
    This demonstrates:
    - Path validation
    - Encoding handling
    - Size limits
    - Detailed error messages
    """
    # Validate path
    if not file_path:
        return {
            "success": False,
            "error": "invalid_path",
            "message": "File path cannot be empty"
        }
    
    # Security: Prevent directory traversal
    if ".." in file_path or file_path.startswith("/"):
        return {
            "success": False,
            "error": "invalid_path",
            "message": "Invalid file path. Paths must be relative to workspace root.",
            "suggestion": "Use paths like 'documents/file.txt', not '../file.txt'"
        }
    
    # Validate encoding
    valid_encodings = ["utf-8", "ascii", "latin-1"]
    if encoding not in valid_encodings:
        return {
            "success": False,
            "error": "invalid_encoding",
            "message": f"Unsupported encoding: {encoding}",
            "suggestion": f"Use one of: {', '.join(valid_encodings)}"
        }
    
    # Mock file reading
    # In real implementation, this would read actual file
    try:
        mock_content = "This is the file content.\nLine 2 of the file."
        
        return {
            "success": True,
            "file_path": file_path,
            "encoding": encoding,
            "size_bytes": len(mock_content),
            "content": mock_content
        }
        
    except UnicodeDecodeError:
        return {
            "success": False,
            "error": "encoding_error",
            "message": f"File cannot be decoded with {encoding} encoding",
            "suggestion": "Try 'latin-1' encoding or check if file is binary"
        }
    except PermissionError:
        return {
            "success": False,
            "error": "permission_denied",
            "message": f"No permission to read {file_path}",
            "suggestion": "Check file permissions or use a different file"
        }


FILE_READ_TOOL = {
    "name": "read_file",
    "description": """Read contents of a text file from the workspace.

Use this tool when the user asks to:
- Read or view a file
- Analyze file contents
- Extract information from a document

Supports text files up to 1MB.
File paths are relative to workspace root (e.g., 'documents/report.txt').

Returns file contents as a string with size and encoding information.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to file relative to workspace root (e.g., 'documents/report.txt', 'data/config.json')"
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding (default: utf-8). Use 'latin-1' for files with special characters.",
                "enum": ["utf-8", "ascii", "latin-1"]
            }
        },
        "required": ["file_path"]
    }
}


# Example 4: Bulk operation with partial success
# ==============================================

def bulk_send_emails(recipients: list[dict[str, str]]) -> dict[str, Any]:
    """
    Send emails to multiple recipients, tracking success and failures.
    
    This demonstrates:
    - Handling multiple operations
    - Partial success reporting
    - Detailed error tracking
    - Actionable results
    """
    if not recipients:
        return {
            "success": False,
            "error": "invalid_input",
            "message": "Recipients list cannot be empty"
        }
    
    successful = []
    failed = []
    
    for idx, recipient in enumerate(recipients):
        # Validate recipient structure
        if "email" not in recipient or "subject" not in recipient:
            failed.append({
                "index": idx,
                "recipient": recipient.get("email", "unknown"),
                "error": "Missing required fields (email, subject)"
            })
            continue
        
        # Validate email format
        email = recipient["email"]
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
            failed.append({
                "index": idx,
                "recipient": email,
                "error": f"Invalid email format: {email}"
            })
            continue
        
        # Simulate sending
        try:
            # In real implementation, would actually send email
            message_id = f"msg_{idx}_{hash(email)}"
            successful.append({
                "index": idx,
                "recipient": email,
                "message_id": message_id,
                "subject": recipient["subject"]
            })
        except Exception as e:
            failed.append({
                "index": idx,
                "recipient": email,
                "error": str(e)
            })
    
    return {
        "success": len(failed) == 0,
        "total": len(recipients),
        "successful": len(successful),
        "failed": len(failed),
        "results": successful,
        "errors": failed,
        "message": f"Sent {len(successful)} of {len(recipients)} emails"
    }


BULK_EMAIL_TOOL = {
    "name": "bulk_send_emails",
    "description": """Send emails to multiple recipients in one operation.

Use this tool when the user wants to:
- Send the same email to multiple people
- Send personalized emails to a list
- Notify a group of users

Each recipient must have 'email' and 'subject' fields.
Optional 'body' field for message content.

Returns detailed results showing which emails succeeded and which failed.
Rate limit: 50 emails per batch.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "recipients": {
                "type": "array",
                "description": """List of recipient objects. Each must have:
                {
                    "email": "user@example.com",
                    "subject": "Email subject",
                    "body": "Optional email body"
                }""",
                "items": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"}
                    },
                    "required": ["email", "subject"]
                }
            }
        },
        "required": ["recipients"]
    }
}


# Example 5: Complex parameter validation
# =======================================

def create_report(
    report_type: str,
    date_range: dict[str, str],
    filters: dict[str, Any],
    format: str = "pdf"
) -> dict[str, Any]:
    """
    Generate a custom report with complex parameters.
    
    This demonstrates:
    - Nested parameter validation
    - Multiple validation stages
    - Helpful error messages for complex inputs
    - Default value handling
    """
    # Validate report type
    valid_types = ["sales", "inventory", "analytics", "financial"]
    if report_type not in valid_types:
        return {
            "success": False,
            "error": "invalid_report_type",
            "message": f"Unknown report type: {report_type}",
            "suggestion": f"Valid types: {', '.join(valid_types)}"
        }
    
    # Validate date range
    if "start_date" not in date_range or "end_date" not in date_range:
        return {
            "success": False,
            "error": "invalid_date_range",
            "message": "Date range must include 'start_date' and 'end_date'",
            "suggestion": "Example: {'start_date': '2024-01-01', 'end_date': '2024-12-31'}"
        }
    
    # Validate date format
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    for key in ["start_date", "end_date"]:
        if not re.match(date_pattern, date_range[key]):
            return {
                "success": False,
                "error": "invalid_date_format",
                "message": f"Invalid {key}: {date_range[key]}",
                "suggestion": "Use YYYY-MM-DD format (e.g., '2024-12-09')"
            }
    
    # Validate format
    valid_formats = ["pdf", "excel", "csv", "html"]
    if format not in valid_formats:
        return {
            "success": False,
            "error": "invalid_format",
            "message": f"Unknown format: {format}",
            "suggestion": f"Valid formats: {', '.join(valid_formats)}"
        }
    
    # Generate report (mock)
    report_id = f"report_{hash(report_type)}_{hash(date_range['start_date'])}"
    
    return {
        "success": True,
        "report_id": report_id,
        "report_type": report_type,
        "date_range": date_range,
        "filters_applied": filters,
        "format": format,
        "file_path": f"/reports/{report_id}.{format}",
        "message": f"{report_type.title()} report generated successfully"
    }


REPORT_TOOL = {
    "name": "create_report",
    "description": """Generate custom business reports with specified parameters.

Use this tool when the user asks for:
- Sales reports or analytics
- Inventory summaries
- Financial statements
- Custom data analysis

Report types available:
- sales: Revenue and transaction data
- inventory: Stock levels and movements
- analytics: User behavior and metrics
- financial: Profit/loss and expenses

Generated reports can be exported in multiple formats.

Returns report ID and file path for download.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "report_type": {
                "type": "string",
                "description": "Type of report to generate",
                "enum": ["sales", "inventory", "analytics", "financial"]
            },
            "date_range": {
                "type": "object",
                "description": "Date range for the report in YYYY-MM-DD format. Example: {'start_date': '2024-01-01', 'end_date': '2024-12-31'}",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    }
                },
                "required": ["start_date", "end_date"]
            },
            "filters": {
                "type": "object",
                "description": """Optional filters to apply. Example:
                {
                    "category": "electronics",
                    "region": "North America",
                    "min_value": 1000
                }"""
            },
            "format": {
                "type": "string",
                "description": "Output format (default: pdf)",
                "enum": ["pdf", "excel", "csv", "html"]
            }
        },
        "required": ["report_type", "date_range"]
    }
}


# Demonstration
# =============

if __name__ == "__main__":
    print("=" * 60)
    print("TOOL DESIGN PATTERNS - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Weather tool
    print("\n1. BASIC TOOL WITH VALIDATION")
    print("-" * 60)
    result = get_weather("London")
    print(f"✅ Success: {result}")
    
    result = get_weather("")
    print(f"❌ Error: {result}")
    
    # Example 2: Database tool
    print("\n2. SECURITY VALIDATION")
    print("-" * 60)
    result = query_database("SELECT * FROM products LIMIT 5")
    print(f"✅ Valid query: {result['success']}")
    
    result = query_database("DROP TABLE products")
    print(f"❌ Blocked: {result['error']}: {result['message']}")
    
    # Example 3: File reading
    print("\n3. FILE OPERATIONS")
    print("-" * 60)
    result = read_file("documents/report.txt")
    print(f"✅ File read: {result['success']}")
    
    result = read_file("../etc/passwd")
    print(f"❌ Security: {result['error']}: {result['message']}")
    
    # Example 4: Bulk operations
    print("\n4. BULK OPERATIONS WITH PARTIAL SUCCESS")
    print("-" * 60)
    recipients = [
        {"email": "user1@example.com", "subject": "Hello"},
        {"email": "invalid-email", "subject": "Test"},
        {"email": "user2@example.com", "subject": "Hi"}
    ]
    result = bulk_send_emails(recipients)
    print(f"Results: {result['successful']}/{result['total']} succeeded")
    print(f"Errors: {result['errors']}")
    
    # Example 5: Complex parameters
    print("\n5. COMPLEX PARAMETER VALIDATION")
    print("-" * 60)
    result = create_report(
        report_type="sales",
        date_range={"start_date": "2024-01-01", "end_date": "2024-12-31"},
        filters={"region": "North America"},
        format="pdf"
    )
    print(f"✅ Report created: {result['report_id']}")
    
    result = create_report(
        report_type="unknown",
        date_range={"start_date": "2024-01-01", "end_date": "2024-12-31"},
        filters={}
    )
    print(f"❌ Invalid type: {result['message']}")
    
    print("\n" + "=" * 60)
    print("All patterns demonstrated successfully!")
    print("=" * 60)
