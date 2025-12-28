"""
Structured JSON logging with correlation IDs for Phase 6.

Features:
- JSON-formatted log entries
- Request ID correlation
- Trace ID correlation (when OpenTelemetry is enabled)
- Timestamp in ISO format
- Sensitive field redaction (passwords, tokens)
"""
import logging
import sys
import re
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional, Any

from pythonjsonlogger.json import JsonFormatter as jsonlogger

# Context variables for request correlation
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Sensitive field patterns to redact
SENSITIVE_FIELDS = {
    "password", "passwd", "pwd", "secret", "token",
    "access_token", "refresh_token", "api_key", "apikey",
    "authorization", "auth", "credential", "private_key",
}

REDACTED_VALUE = "[REDACTED]"


def set_request_context(request_id: Optional[str] = None) -> None:
    """Set the request context for logging correlation."""
    if request_id:
        _request_id.set(request_id)


def clear_request_context() -> None:
    """Clear the request context."""
    _request_id.set(None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return _request_id.get()


def _should_redact(field_name: str) -> bool:
    """Check if a field name should be redacted."""
    field_lower = field_name.lower()
    return any(sensitive in field_lower for sensitive in SENSITIVE_FIELDS)


def _redact_sensitive_fields(record_dict: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields from log record."""
    result = {}
    for key, value in record_dict.items():
        if _should_redact(key):
            result[key] = REDACTED_VALUE
        elif isinstance(value, dict):
            result[key] = _redact_sensitive_fields(value)
        else:
            result[key] = value
    return result


class CorrelatedJsonFormatter(jsonlogger):
    """
    JSON log formatter with correlation IDs and sensitive field redaction.

    Adds:
    - timestamp: ISO format timestamp
    - level: Log level name
    - logger: Logger name
    - request_id: Request correlation ID (when in context)
    - trace_id: OpenTelemetry trace ID (when available)
    - span_id: OpenTelemetry span ID (when available)
    """

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add log level
        log_record["level"] = record.levelname

        # Add logger name
        log_record["logger"] = record.name

        # Add message
        if "message" not in log_record:
            log_record["message"] = record.getMessage()

        # Add request ID from context
        request_id = get_request_id()
        if request_id:
            log_record["request_id"] = request_id
        else:
            log_record["request_id"] = None

        # Add trace context if OpenTelemetry is available
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.is_recording():
                ctx = span.get_span_context()
                log_record["trace_id"] = format(ctx.trace_id, "032x")
                log_record["span_id"] = format(ctx.span_id, "016x")
        except ImportError:
            pass  # OpenTelemetry not installed
        except Exception:
            pass  # Span context not available

        # Add exception info if present
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
            log_record["traceback"] = "".join(traceback.format_exception(*record.exc_info))

        # Redact sensitive fields
        for key in list(log_record.keys()):
            if _should_redact(key) and log_record[key] not in (None, REDACTED_VALUE):
                log_record[key] = REDACTED_VALUE

    def format(self, record: logging.LogRecord) -> str:
        """Format log record, applying redaction to extra fields."""
        # Process extra fields from the record
        message_dict = {}

        # Check for extra fields that might contain sensitive data
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                if _should_redact(key):
                    message_dict[key] = REDACTED_VALUE
                else:
                    message_dict[key] = value

        return super().format(record)


def get_json_logger(name: str) -> logging.Logger:
    """
    Get a logger configured for JSON output.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    format_json: bool = True,
) -> None:
    """
    Configure root logger with JSON formatting.

    Args:
        level: Log level (default: INFO)
        format_json: Use JSON formatting (default: True)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if format_json:
        formatter = CorrelatedJsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
