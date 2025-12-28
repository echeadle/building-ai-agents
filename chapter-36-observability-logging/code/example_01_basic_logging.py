"""
Basic Python logging demonstration.

Chapter 36: Observability and Logging

This script introduces Python's built-in logging module,
showing the different log levels and basic configuration.
"""

import logging

# Create a logger for this module
# __name__ is the module name, which creates a hierarchy
logger = logging.getLogger(__name__)

# Configure the root logger with basic settings
# This affects all loggers unless they have specific configuration
logging.basicConfig(
    level=logging.DEBUG,  # Minimum level to capture
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def demonstrate_log_levels():
    """
    Demonstrate all five standard logging levels.
    
    Log levels from least to most severe:
    - DEBUG (10): Detailed diagnostic information
    - INFO (20): Confirmation that things work as expected
    - WARNING (30): Something unexpected happened, but program continues
    - ERROR (40): A more serious problem occurred
    - CRITICAL (50): A serious error, program may not be able to continue
    """
    print("=" * 60)
    print("Demonstrating Python Logging Levels")
    print("=" * 60)
    print()
    
    logger.debug("DEBUG: Detailed diagnostic information for developers")
    logger.info("INFO: General operational information")
    logger.warning("WARNING: Something unexpected but not critical")
    logger.error("ERROR: A serious problem prevented an operation")
    logger.critical("CRITICAL: The program may not be able to continue")


def demonstrate_log_filtering():
    """
    Show how changing the log level filters messages.
    """
    print()
    print("=" * 60)
    print("Demonstrating Log Level Filtering")
    print("=" * 60)
    
    # Create a new logger for this demo
    filtered_logger = logging.getLogger("filtered_demo")
    
    # Test different thresholds
    for level_name in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        level = getattr(logging, level_name)
        filtered_logger.setLevel(level)
        
        print(f"\n--- Logger level set to {level_name} ---")
        filtered_logger.debug("debug message")
        filtered_logger.info("info message")
        filtered_logger.warning("warning message")
        filtered_logger.error("error message")


def demonstrate_extra_data():
    """
    Show how to include additional data in log messages.
    """
    print()
    print("=" * 60)
    print("Demonstrating Extra Data in Logs")
    print("=" * 60)
    print()
    
    # Method 1: String formatting (preferred for performance)
    user_id = "user_123"
    action = "login"
    logger.info("User %s performed action: %s", user_id, action)
    
    # Method 2: f-strings (convenient but always evaluated)
    duration_ms = 150.5
    logger.info(f"Operation completed in {duration_ms}ms")
    
    # Method 3: Using extra parameter (for custom formatters)
    logger.info(
        "Request processed",
        extra={"request_id": "req_456", "status": "success"}
    )


def demonstrate_exception_logging():
    """
    Show how to log exceptions with full tracebacks.
    """
    print()
    print("=" * 60)
    print("Demonstrating Exception Logging")
    print("=" * 60)
    print()
    
    try:
        result = 1 / 0
    except ZeroDivisionError:
        # exc_info=True includes the full traceback
        logger.error("An error occurred during calculation", exc_info=True)
    
    # Alternative: logger.exception() always includes traceback
    try:
        data = {"key": "value"}
        _ = data["missing_key"]
    except KeyError:
        logger.exception("Failed to access dictionary key")


if __name__ == "__main__":
    demonstrate_log_levels()
    demonstrate_log_filtering()
    demonstrate_extra_data()
    demonstrate_exception_logging()
    
    print()
    print("=" * 60)
    print("Logging demonstration complete!")
    print("=" * 60)
