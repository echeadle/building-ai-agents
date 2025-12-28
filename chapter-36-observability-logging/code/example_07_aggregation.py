"""
Log aggregation patterns for production agents.

Chapter 36: Observability and Logging

This module provides formatters and utilities for integrating
agent logs with popular log aggregation systems:
- Elasticsearch / ELK Stack
- AWS CloudWatch
- Datadog
- Generic JSON logging
"""

import json
import logging
import socket
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional


class ProductionLogFormatter(logging.Formatter):
    """
    Production-ready JSON log formatter with rich metadata.
    
    Includes:
    - ISO 8601 timestamp
    - Hostname for multi-server deployments
    - Service name for microservices
    - Environment identifier
    - Custom static fields
    
    Compatible with most log aggregation systems.
    """
    
    def __init__(
        self,
        service_name: str,
        environment: str = "production",
        extra_fields: Optional[dict[str, Any]] = None
    ):
        """
        Initialize the formatter.
        
        Args:
            service_name: Name of the service (e.g., "ai-agent")
            environment: Environment name (e.g., "production", "staging")
            extra_fields: Additional static fields to include in every log
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "hostname": self.hostname,
        }
        
        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Add static extra fields
        log_entry.update(self.extra_fields)
        
        # Add dynamic extra fields from the log record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class ElasticsearchFormatter(logging.Formatter):
    """
    Formatter optimized for Elasticsearch / ELK Stack.
    
    Uses Elastic Common Schema (ECS) field names for better
    integration with Kibana and other Elastic tools.
    """
    
    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        super().__init__()
        self.service_name = service_name
        self.service_version = service_version
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
        # ECS-compatible log format
        log_entry = {
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "log": {
                "level": record.levelname.lower(),
                "logger": record.name,
            },
            "message": record.getMessage(),
            "service": {
                "name": self.service_name,
                "version": self.service_version,
            },
            "host": {
                "hostname": self.hostname,
            },
            "ecs": {
                "version": "1.6.0"
            }
        }
        
        # Add trace context if available
        if hasattr(record, "trace_id"):
            log_entry["trace"] = {"id": record.trace_id}
        if hasattr(record, "span_id"):
            log_entry["span"] = {"id": record.span_id}
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry["labels"] = record.extra_fields
        
        # Add error details
        if record.exc_info:
            log_entry["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry)


class CloudWatchFormatter(logging.Formatter):
    """
    Formatter optimized for AWS CloudWatch Logs.
    
    CloudWatch expects specific field names and formats
    for optimal integration with CloudWatch Insights.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        # Add request ID if available (useful for Lambda)
        if hasattr(record, "request_id"):
            log_entry["requestId"] = record.request_id
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class DatadogFormatter(logging.Formatter):
    """
    Formatter optimized for Datadog Log Management.
    
    Uses Datadog's standard attributes for proper parsing
    and integration with APM traces.
    """
    
    def __init__(
        self,
        service: str,
        env: str = "production",
        version: str = "1.0.0"
    ):
        super().__init__()
        self.service = service
        self.env = env
        self.version = version
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": {"name": record.name},
            "dd": {
                "service": self.service,
                "env": self.env,
                "version": self.version,
            }
        }
        
        # Add trace correlation if available
        if hasattr(record, "trace_id"):
            log_entry["dd"]["trace_id"] = record.trace_id
        if hasattr(record, "span_id"):
            log_entry["dd"]["span_id"] = record.span_id
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add error info
        if record.exc_info:
            log_entry["error"] = {
                "kind": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack": self.formatException(record.exc_info)
            }
        
        return json.dumps(log_entry)


class CorrelationContext:
    """
    Thread-local storage for request correlation.
    
    This allows you to add trace IDs and other context
    to all log messages within a request, even across
    function calls.
    
    Usage:
        CorrelationContext.set("trace_id", "abc123")
        CorrelationContext.set("user_id", "user456")
        
        # All logs will now include these fields
        logger.info("Processing request")
        
        # Clear when done
        CorrelationContext.clear()
    """
    
    _local = threading.local()
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a correlation context value."""
        if not hasattr(cls._local, "context"):
            cls._local.context = {}
        cls._local.context[key] = value
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a correlation context value."""
        if not hasattr(cls._local, "context"):
            return default
        return cls._local.context.get(key, default)
    
    @classmethod
    def get_all(cls) -> dict[str, Any]:
        """Get all correlation context values."""
        if not hasattr(cls._local, "context"):
            return {}
        return cls._local.context.copy()
    
    @classmethod
    def clear(cls) -> None:
        """Clear all correlation context values."""
        if hasattr(cls._local, "context"):
            cls._local.context.clear()


class CorrelatedFormatter(logging.Formatter):
    """
    Formatter that automatically includes correlation context.
    
    Works with CorrelationContext to add trace IDs and other
    context to all log messages.
    """
    
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "message": record.getMessage(),
        }
        
        # Add all correlation context
        context = CorrelationContext.get_all()
        if context:
            log_entry.update(context)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_elasticsearch_logging(
    service_name: str,
    service_version: str = "1.0.0",
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logging for Elasticsearch/ELK Stack."""
    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ElasticsearchFormatter(service_name, service_version))
    logger.addHandler(handler)
    
    return logger


def setup_cloudwatch_logging(
    logger_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logging for AWS CloudWatch."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudWatchFormatter())
    logger.addHandler(handler)
    
    return logger


def setup_datadog_logging(
    service: str,
    env: str = "production",
    version: str = "1.0.0",
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logging for Datadog."""
    logger = logging.getLogger(service)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DatadogFormatter(service, env, version))
    logger.addHandler(handler)
    
    return logger


# Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Log Aggregation Patterns Demo")
    print("=" * 70)
    
    # Demo 1: Production formatter
    print("\n1. Production Formatter (Generic)")
    print("-" * 40)
    
    prod_logger = logging.getLogger("production")
    prod_logger.setLevel(logging.INFO)
    prod_logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ProductionLogFormatter(
        service_name="ai-agent",
        environment="production",
        extra_fields={"version": "1.0.0", "team": "ai-platform"}
    ))
    prod_logger.addHandler(handler)
    
    prod_logger.info("Agent started")
    
    # Demo 2: Elasticsearch formatter
    print("\n2. Elasticsearch / ELK Stack")
    print("-" * 40)
    
    es_logger = setup_elasticsearch_logging("ai-agent", "1.0.0")
    es_logger.info("Processing request")
    
    # Demo 3: CloudWatch formatter
    print("\n3. AWS CloudWatch")
    print("-" * 40)
    
    cw_logger = setup_cloudwatch_logging("ai-agent")
    cw_logger.info("Lambda function invoked")
    
    # Demo 4: Datadog formatter
    print("\n4. Datadog")
    print("-" * 40)
    
    dd_logger = setup_datadog_logging("ai-agent", "production", "1.0.0")
    dd_logger.info("Request processed")
    
    # Demo 5: Correlation context
    print("\n5. Correlation Context")
    print("-" * 40)
    
    corr_logger = logging.getLogger("correlated")
    corr_logger.setLevel(logging.INFO)
    corr_logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CorrelatedFormatter("ai-agent"))
    corr_logger.addHandler(handler)
    
    # Set correlation context
    CorrelationContext.set("trace_id", "abc123")
    CorrelationContext.set("user_id", "user_456")
    CorrelationContext.set("request_id", "req_789")
    
    corr_logger.info("Starting request processing")
    corr_logger.info("Calling weather tool")
    corr_logger.info("Request completed")
    
    CorrelationContext.clear()
    
    print("\n" + "=" * 70)
    print("Log Aggregation Best Practices")
    print("=" * 70)
    print("""
    1. Always include a trace/correlation ID
       - Links all logs from a single request
       - Essential for distributed tracing
    
    2. Use consistent field names
       - Follow the conventions of your log system
       - ECS for Elasticsearch, standard attributes for Datadog
    
    3. Include service metadata
       - Service name, version, environment
       - Helps filter logs in multi-service deployments
    
    4. Structure your data
       - JSON format for machine parsing
       - Avoid nested structures more than 2 levels deep
    
    5. Mind your costs
       - Log aggregation can be expensive at scale
       - Use appropriate log levels
       - Consider sampling for high-volume debug logs
    """)
