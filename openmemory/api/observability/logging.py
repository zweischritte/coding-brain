"""Structured logging with trace correlation.

Phase 0c implementation per IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md section 13.

Features:
- Structured logs include: trace_id, user_id, org_id, repo_id, tool_name, latency_ms
- Trace correlation with OpenTelemetry spans
- JSON formatted output for log aggregation
- Context binding for request-scoped metadata
- Log processors for customization
"""

from __future__ import annotations

import contextvars
import json
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from io import StringIO
from typing import (
    Any,
    Callable,
    Generator,
    TextIO,
    Tuple,
    Type,
)

from openmemory.api.observability.tracing import get_current_span


class LogLevel(IntEnum):
    """Log levels with numeric ordering for filtering."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Type alias for exception info tuple
ExcInfo = Tuple[Type[BaseException], BaseException, Any] | None

# Context variable for global bound context
_global_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_global_context", default={}
)


def bind_context(**kwargs: Any) -> None:
    """Bind context globally for all loggers.

    Args:
        **kwargs: Key-value pairs to bind.
    """
    current = _global_context.get().copy()
    current.update(kwargs)
    _global_context.set(current)


def unbind_context(*keys: str) -> None:
    """Unbind context keys globally.

    Args:
        *keys: Keys to unbind.
    """
    current = _global_context.get().copy()
    for key in keys:
        current.pop(key, None)
    _global_context.set(current)


def get_trace_context() -> dict[str, str | None]:
    """Get current trace context for log correlation.

    Returns:
        Dict with trace_id and span_id (None if no active span).
    """
    span = get_current_span()
    if span is None:
        return {"trace_id": None, "span_id": None}

    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
    }


@dataclass
class LogRecord:
    """A structured log record.

    Attributes:
        level: Log level.
        message: Log message.
        timestamp: When the log was created.
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
        extra: Additional structured fields.
        exc_info: Optional exception information.
    """

    level: LogLevel
    message: str
    timestamp: datetime
    trace_id: str | None = None
    span_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    exc_info: ExcInfo = None


# Type alias for log processors
LogProcessor = Callable[[LogRecord], LogRecord]


@dataclass
class LogConfig:
    """Configuration for structured logging.

    Attributes:
        service_name: Name of the service.
        level: Minimum log level to output.
        json_format: Whether to output JSON (default True).
        include_timestamp: Whether to include timestamps.
        output: Output stream (default stderr).
        processors: List of log processors to apply.
    """

    service_name: str
    level: LogLevel = LogLevel.INFO
    json_format: bool = True
    include_timestamp: bool = True
    output: TextIO = field(default_factory=lambda: sys.stderr)
    processors: list[LogProcessor] = field(default_factory=list)


class JSONFormatter:
    """Format log records as JSON."""

    def format(self, record: LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation.
        """
        data: dict[str, Any] = {
            "level": record.level.name,
            "message": record.message,
        }

        if record.timestamp:
            data["timestamp"] = record.timestamp.isoformat()

        if record.trace_id:
            data["trace_id"] = record.trace_id

        if record.span_id:
            data["span_id"] = record.span_id

        # Merge extra fields
        data.update(record.extra)

        # Handle exception info
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            data["exception"] = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )

        return json.dumps(data, default=str)


class StructuredLogger:
    """Structured logger with trace correlation.

    Provides structured logging with automatic trace context,
    context binding, and JSON formatting.
    """

    def __init__(
        self,
        config: LogConfig,
        name: str = "",
        bound_context: dict[str, Any] | None = None,
    ):
        """Initialize the logger.

        Args:
            config: Logging configuration.
            name: Logger name (e.g., module name).
            bound_context: Pre-bound context (from parent logger).
        """
        self._config = config
        self._name = name or config.service_name
        self._bound_context = bound_context.copy() if bound_context else {}
        self._formatter = JSONFormatter()

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._config.service_name

    @property
    def name(self) -> str:
        """Get the logger name."""
        return self._name

    def bind(self, **kwargs: Any) -> None:
        """Bind context to this logger instance.

        Args:
            **kwargs: Key-value pairs to bind.
        """
        self._bound_context.update(kwargs)

    def unbind(self, *keys: str) -> None:
        """Unbind context keys from this logger.

        Args:
            *keys: Keys to unbind.
        """
        for key in keys:
            self._bound_context.pop(key, None)

    def get_bound_value(self, key: str) -> Any:
        """Get a bound context value.

        Args:
            key: The context key.

        Returns:
            The value or None if not bound.
        """
        return self._bound_context.get(key)

    @contextmanager
    def context(self, **kwargs: Any) -> Generator[None, None, None]:
        """Temporarily bind context.

        Args:
            **kwargs: Key-value pairs to temporarily bind.

        Yields:
            None
        """
        old_context = self._bound_context.copy()
        self._bound_context.update(kwargs)
        try:
            yield
        finally:
            self._bound_context = old_context

    def child(self, name: str) -> StructuredLogger:
        """Create a child logger with inherited context.

        Args:
            name: Name for the child logger.

        Returns:
            A new logger with inherited context.
        """
        return StructuredLogger(
            config=self._config,
            name=name,
            bound_context=self._bound_context,
        )

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message at this level should be logged."""
        return level >= self._config.level

    def _create_record(
        self,
        level: LogLevel,
        message: str,
        exc_info: ExcInfo = None,
        **kwargs: Any,
    ) -> LogRecord:
        """Create a log record with all context.

        Args:
            level: Log level.
            message: Log message.
            exc_info: Optional exception info.
            **kwargs: Additional fields.

        Returns:
            Populated LogRecord.
        """
        # Get trace context
        trace_ctx = get_trace_context()

        # Merge contexts: global < bound < local
        extra = {}
        extra.update(_global_context.get())
        extra.update(self._bound_context)
        extra.update(kwargs)

        return LogRecord(
            level=level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            trace_id=trace_ctx["trace_id"],
            span_id=trace_ctx["span_id"],
            extra=extra,
            exc_info=exc_info,
        )

    def _write(self, record: LogRecord) -> None:
        """Write a log record (internal, for mocking in tests).

        Args:
            record: The log record to write.
        """
        # Apply processors
        processed = record
        for processor in self._config.processors:
            processed = processor(processed)

        self._output(processed)

    def _output(self, record: LogRecord) -> None:
        """Output a processed log record.

        Args:
            record: The processed log record.
        """
        if self._config.json_format:
            output = self._formatter.format(record)
        else:
            output = f"[{record.level.name}] {record.message}"

        print(output, file=self._config.output)

    def _log(
        self,
        level: LogLevel,
        message: str,
        exc_info: ExcInfo = None,
        **kwargs: Any,
    ) -> None:
        """Internal log method.

        Args:
            level: Log level.
            message: Log message.
            exc_info: Optional exception info.
            **kwargs: Additional fields.
        """
        if not self._should_log(level):
            return

        record = self._create_record(level, message, exc_info=exc_info, **kwargs)
        self._write(record)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(
        self,
        message: str,
        exc_info: ExcInfo = None,
        **kwargs: Any,
    ) -> None:
        """Log at ERROR level.

        Args:
            message: Log message.
            exc_info: Optional exception info.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level with current exception info.

        Args:
            message: Log message.
            **kwargs: Additional fields.
        """
        exc_info = sys.exc_info()
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)


# Global logger registry
_logger_registry: dict[str, StructuredLogger] = {}

# Default config for convenience functions
_default_config: LogConfig | None = None


def create_logger(
    config: LogConfig,
    name: str = "",
) -> StructuredLogger:
    """Create a structured logger.

    Args:
        config: Logging configuration.
        name: Optional logger name.

    Returns:
        StructuredLogger instance.
    """
    global _default_config
    if _default_config is None:
        _default_config = config

    return StructuredLogger(config, name=name)


def get_logger(name: str) -> StructuredLogger:
    """Get or create a logger by name.

    Uses a default config if one has been set, otherwise creates minimal config.

    Args:
        name: Logger name.

    Returns:
        StructuredLogger instance.
    """
    if name in _logger_registry:
        return _logger_registry[name]

    config = _default_config or LogConfig(service_name="openmemory")
    logger = StructuredLogger(config, name=name)
    _logger_registry[name] = logger
    return logger
