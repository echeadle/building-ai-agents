"""
Log level configuration patterns for AI agents.

Chapter 36: Observability and Logging

This script demonstrates various ways to configure log levels
for different environments and use cases.
"""

import logging
import os
import sys
from typing import Optional


def configure_from_environment() -> int:
    """
    Configure log level from LOG_LEVEL environment variable.
    
    This is the most common pattern for production deployments,
    allowing operators to adjust verbosity without code changes.
    
    Usage:
        LOG_LEVEL=DEBUG python my_agent.py
        LOG_LEVEL=WARNING python my_agent.py
    
    Returns:
        The configured log level
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"Log level configured from environment: {level_name}")
    return level


def configure_per_module():
    """
    Configure different log levels for different modules.
    
    This is useful when you want verbose logging for your agent
    but quiet logging for noisy dependencies.
    """
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verbose logging for our agent code
    logging.getLogger("agent").setLevel(logging.DEBUG)
    logging.getLogger("agent.tools").setLevel(logging.DEBUG)
    
    # Quiet logging for HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Quiet logging for the Anthropic SDK
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    
    # Even quieter for very noisy libraries
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    
    print("Per-module logging configured:")
    print("  - agent.*: DEBUG")
    print("  - httpx/httpcore: WARNING")
    print("  - anthropic: WARNING")
    print("  - urllib3: ERROR")


class LevelFilter(logging.Filter):
    """
    Filter that allows only specific log levels.
    
    This is useful for routing different levels to different outputs,
    like INFO to stdout and ERROR to stderr.
    """
    
    def __init__(self, levels: list[int]):
        """
        Initialize the filter.
        
        Args:
            levels: List of log levels to allow
        """
        super().__init__()
        self.levels = levels
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be logged."""
        return record.levelno in self.levels


class MaxLevelFilter(logging.Filter):
    """
    Filter that allows levels up to a maximum.
    
    Useful for sending only low-severity logs to stdout.
    """
    
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level
    
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def configure_split_output():
    """
    Send INFO/DEBUG to stdout, WARNING+ to stderr.
    
    This is useful in containerized environments where stdout and stderr
    are captured and processed separately.
    """
    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Stdout handler for DEBUG and INFO only
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    
    # Stderr handler for WARNING and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    
    return logger


def configure_file_and_console(
    log_file: str = "agent.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
):
    """
    Log to both console and file with different levels.
    
    Common pattern: verbose logging to file for debugging,
    summary logging to console for operators.
    
    Args:
        log_file: Path to the log file
        console_level: Minimum level for console output
        file_level: Minimum level for file output
    """
    logger = logging.getLogger("agent")
    logger.setLevel(min(console_level, file_level))
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to console (level={logging.getLevelName(console_level)}) "
                f"and file (level={logging.getLevelName(file_level)})")
    
    return logger


class DynamicLevelHandler(logging.Handler):
    """
    A handler that allows runtime log level changes.
    
    This is useful for temporarily increasing verbosity
    during debugging without restarting the application.
    """
    
    def __init__(self, initial_level: int = logging.INFO):
        super().__init__(initial_level)
        self._dynamic_level = initial_level
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def emit(self, record: logging.LogRecord):
        if record.levelno >= self._dynamic_level:
            msg = self.format(record)
            print(msg)
    
    def set_level_runtime(self, level: int):
        """Change the log level at runtime."""
        self._dynamic_level = level
        print(f"Log level changed to: {logging.getLevelName(level)}")


# Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Log Level Configuration Patterns")
    print("=" * 70)
    
    # Demo 1: Environment-based configuration
    print("\n1. Environment-based Configuration")
    print("-" * 40)
    configure_from_environment()
    logging.debug("Debug message (may be hidden)")
    logging.info("Info message")
    logging.warning("Warning message")
    
    # Reset
    logging.root.handlers.clear()
    
    # Demo 2: Per-module configuration
    print("\n2. Per-module Configuration")
    print("-" * 40)
    configure_per_module()
    
    agent_logger = logging.getLogger("agent")
    http_logger = logging.getLogger("httpx")
    
    agent_logger.debug("Agent debug - visible")
    agent_logger.info("Agent info - visible")
    http_logger.debug("HTTP debug - hidden")
    http_logger.info("HTTP info - hidden")
    http_logger.warning("HTTP warning - visible")
    
    # Reset
    logging.root.handlers.clear()
    
    # Demo 3: Split output
    print("\n3. Split Output (stdout/stderr)")
    print("-" * 40)
    split_logger = configure_split_output()
    
    print("The following messages go to different streams:")
    split_logger.debug("Debug -> stdout")
    split_logger.info("Info -> stdout")
    split_logger.warning("Warning -> stderr")
    split_logger.error("Error -> stderr")
    
    # Demo 4: Dynamic level changes
    print("\n4. Dynamic Level Changes")
    print("-" * 40)
    
    dynamic_handler = DynamicLevelHandler(logging.INFO)
    dynamic_logger = logging.getLogger("dynamic")
    dynamic_logger.handlers.clear()
    dynamic_logger.addHandler(dynamic_handler)
    dynamic_logger.setLevel(logging.DEBUG)
    
    print("Initial level: INFO")
    dynamic_logger.debug("Debug message (hidden)")
    dynamic_logger.info("Info message (visible)")
    
    print("\nChanging to DEBUG...")
    dynamic_handler.set_level_runtime(logging.DEBUG)
    dynamic_logger.debug("Debug message (now visible)")
    dynamic_logger.info("Info message (still visible)")
    
    print("\nChanging to WARNING...")
    dynamic_handler.set_level_runtime(logging.WARNING)
    dynamic_logger.info("Info message (now hidden)")
    dynamic_logger.warning("Warning message (visible)")
    
    print("\n" + "=" * 70)
    print("Log Level Guidelines for Agents")
    print("=" * 70)
    print("""
    DEBUG: Detailed internal information
           - Tool inputs and outputs
           - LLM response parsing details
           - State transitions
    
    INFO:  Operational information
           - Request started/completed
           - Tool selected
           - Key milestones
    
    WARNING: Recoverable issues
             - Retry attempts
             - Rate limit approaching
             - Unexpected but handled conditions
    
    ERROR: Failed operations
           - Tool execution failures
           - API errors
           - Validation failures
    
    CRITICAL: System-wide problems
              - Configuration errors
              - Unrecoverable state
              - Security issues
    
    Recommendations:
    - Development: DEBUG
    - Staging: DEBUG or INFO
    - Production: INFO or WARNING
    - Debugging production: Temporarily enable DEBUG
    """)
