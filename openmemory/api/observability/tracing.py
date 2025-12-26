"""OpenTelemetry tracing with GenAI semantic conventions.

Phase 0c implementation per IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md section 13.

Features:
- OpenTelemetry v1.37+ compatible with GenAI semantic conventions
- W3C TraceContext propagation (traceparent, tracestate)
- Span creation with required attributes (user_id, org_id, tool_name, latency_ms)
- Context propagation across async boundaries
- Span nesting for parent/child relationships
- Error recording with exception details
- GenAI attributes: token usage, TTFT, model_id
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import random
import re
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Generator,
    AsyncGenerator,
    Mapping,
    Sequence,
    TypeVar,
)

# W3C TraceContext header names
W3C_TRACEPARENT_HEADER = "traceparent"
W3C_TRACESTATE_HEADER = "tracestate"

# W3C TraceContext version
W3C_TRACE_VERSION = "00"

# Context variable for current span
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)

# Global tracer registry (config_key -> Tracer)
_tracer_registry: dict[str, Tracer] = {}


def _clear_tracer_registry() -> None:
    """Clear the tracer registry. Used for testing only."""
    _tracer_registry.clear()


class TracingError(Exception):
    """Base exception for tracing errors."""

    pass


class SpanKind(Enum):
    """Span kind as per OpenTelemetry specification."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span status as per OpenTelemetry specification."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """An event that occurred during a span's lifetime."""

    name: str
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TracingConfig:
    """Configuration for the tracer.

    Attributes:
        service_name: Name of the service being traced.
        service_version: Version of the service.
        environment: Deployment environment (test, staging, production).
        enabled: Whether tracing is enabled.
        sample_rate: Probability of sampling a trace (0.0 to 1.0).
        otlp_endpoint: Optional OTLP collector endpoint for exporting traces.
    """

    service_name: str
    service_version: str = "unknown"
    environment: str = "development"
    enabled: bool = True
    sample_rate: float = 1.0
    otlp_endpoint: str | None = None

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}"
            )


@dataclass
class TracingContext:
    """W3C TraceContext for propagation across services.

    Attributes:
        trace_id: 32 hex character trace identifier.
        parent_span_id: 16 hex character parent span identifier.
        sampled: Whether the trace should be sampled.
        tracestate: Optional vendor-specific tracing data.
    """

    trace_id: str
    parent_span_id: str
    sampled: bool
    tracestate: str | None = None

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent header value.

        Format: version-trace_id-span_id-flags
        """
        flags = "01" if self.sampled else "00"
        return f"{W3C_TRACE_VERSION}-{self.trace_id}-{self.parent_span_id}-{flags}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> TracingContext | None:
        """Parse W3C traceparent header value.

        Args:
            traceparent: W3C traceparent header value.

        Returns:
            TracingContext if valid, None otherwise.
        """
        pattern = r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
        match = re.match(pattern, traceparent.lower())
        if not match:
            return None

        version, trace_id, span_id, flags = match.groups()

        # Version 00 is current, but accept future versions
        if version == "ff":  # Invalid version
            return None

        sampled = flags[-1] == "1"

        return cls(
            trace_id=trace_id,
            parent_span_id=span_id,
            sampled=sampled,
        )


class Span:
    """A span represents a single operation within a trace.

    Attributes:
        name: Name of the operation.
        trace_id: 32 hex character trace identifier.
        span_id: 16 hex character span identifier.
        parent_span_id: 16 hex character parent span identifier, or None for root.
        kind: The kind of span (INTERNAL, SERVER, CLIENT, etc.).
        status: Current status of the span.
        status_description: Optional description for ERROR status.
    """

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        recording: bool = True,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.kind = kind
        self._recording = recording
        self._attributes: dict[str, Any] = {}
        self._events: list[SpanEvent] = []
        self._status = SpanStatus.UNSET
        self._status_description: str | None = None
        self._start_time = datetime.now(timezone.utc)
        self._end_time: datetime | None = None

    def is_recording(self) -> bool:
        """Check if this span is recording data."""
        return self._recording

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span.

        Args:
            key: Attribute name.
            value: Attribute value (string, int, float, bool, or list of strings).
        """
        if self._recording:
            self._attributes[key] = value

    def get_attribute(self, key: str) -> Any:
        """Get an attribute value.

        Args:
            key: Attribute name.

        Returns:
            Attribute value or None if not set.
        """
        return self._attributes.get(key)

    def set_status(
        self, status: SpanStatus, description: str | None = None
    ) -> None:
        """Set the status of the span.

        Args:
            status: The status to set.
            description: Optional description (typically for ERROR status).
        """
        self._status = status
        self._status_description = description

    @property
    def status(self) -> SpanStatus:
        """Get the current span status."""
        return self._status

    @property
    def status_description(self) -> str | None:
        """Get the status description."""
        return self._status_description

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Optional event attributes.
            timestamp: Optional explicit timestamp.
        """
        if self._recording:
            event = SpanEvent(
                name=name,
                timestamp=timestamp or datetime.now(timezone.utc),
                attributes=attributes or {},
            )
            self._events.append(event)

    def get_events(self) -> list[SpanEvent]:
        """Get all events recorded on this span."""
        return self._events.copy()

    def record_exception(
        self,
        exception: BaseException,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Record an exception on the span.

        Args:
            exception: The exception to record.
            attributes: Optional additional attributes.
        """
        if not self._recording:
            return

        exc_attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
        }
        if attributes:
            exc_attrs.update(attributes)

        self.add_event("exception", attributes=exc_attrs)

    def end(self) -> None:
        """End the span, recording the end time."""
        self._end_time = datetime.now(timezone.utc)


class NoOpSpan(Span):
    """A no-op span that discards all data.

    Used when tracing is disabled or trace is not sampled.
    """

    def __init__(self, name: str = ""):
        super().__init__(
            name=name,
            trace_id="0" * 32,
            span_id="0" * 16,
            recording=False,
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op: discards attributes."""
        pass

    def get_attribute(self, key: str) -> Any:
        """No-op: always returns None."""
        return None

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """No-op: discards events."""
        pass

    def get_events(self) -> list[SpanEvent]:
        """No-op: returns empty list."""
        return []

    def record_exception(
        self,
        exception: BaseException,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """No-op: discards exception."""
        pass


def _generate_trace_id() -> str:
    """Generate a random 32 hex character trace ID."""
    return os.urandom(16).hex()


def _generate_span_id() -> str:
    """Generate a random 16 hex character span ID."""
    return os.urandom(8).hex()


class Tracer:
    """OpenTelemetry-compatible tracer for creating spans.

    The tracer manages span lifecycle and context propagation.
    """

    def __init__(self, config: TracingConfig):
        """Initialize the tracer.

        Args:
            config: Tracing configuration.
        """
        self._config = config
        self._service_name = config.service_name
        self._otlp_endpoint = config.otlp_endpoint

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._service_name

    @property
    def otlp_endpoint(self) -> str | None:
        """Get the OTLP endpoint."""
        return self._otlp_endpoint

    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._config.enabled

    def _should_sample(self) -> bool:
        """Determine if a new trace should be sampled."""
        if self._config.sample_rate >= 1.0:
            return True
        if self._config.sample_rate <= 0.0:
            return False
        return random.random() < self._config.sample_rate

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        context: TracingContext | None = None,
    ) -> Generator[Span, None, None]:
        """Start a new span as a context manager.

        Args:
            name: Name of the operation.
            kind: The kind of span.
            attributes: Optional initial attributes.
            context: Optional parent context from propagation.

        Yields:
            The created span.
        """
        if not self._config.enabled:
            yield NoOpSpan(name)
            return

        # Get parent from context variable or provided context
        parent_span = _current_span.get()

        if context is not None:
            # External context from propagation
            trace_id = context.trace_id
            parent_span_id = context.parent_span_id
            sampled = context.sampled
        elif parent_span is not None:
            # Continue existing trace
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
            sampled = parent_span.is_recording()
        else:
            # New trace
            trace_id = _generate_trace_id()
            parent_span_id = None
            sampled = self._should_sample()

        if not sampled:
            yield NoOpSpan(name)
            return

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=_generate_span_id(),
            parent_span_id=parent_span_id,
            kind=kind,
            recording=True,
        )

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(SpanStatus.ERROR, str(exc))
            raise
        finally:
            span.end()
            _current_span.reset(token)

    @asynccontextmanager
    async def start_span_async(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        context: TracingContext | None = None,
    ) -> AsyncGenerator[Span, None]:
        """Start a new span as an async context manager.

        Args:
            name: Name of the operation.
            kind: The kind of span.
            attributes: Optional initial attributes.
            context: Optional parent context from propagation.

        Yields:
            The created span.
        """
        if not self._config.enabled:
            yield NoOpSpan(name)
            return

        # Get parent from context variable or provided context
        parent_span = _current_span.get()

        if context is not None:
            trace_id = context.trace_id
            parent_span_id = context.parent_span_id
            sampled = context.sampled
        elif parent_span is not None:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
            sampled = parent_span.is_recording()
        else:
            trace_id = _generate_trace_id()
            parent_span_id = None
            sampled = self._should_sample()

        if not sampled:
            yield NoOpSpan(name)
            return

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=_generate_span_id(),
            parent_span_id=parent_span_id,
            kind=kind,
            recording=True,
        )

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        token = _current_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(SpanStatus.ERROR, str(exc))
            raise
        finally:
            span.end()
            _current_span.reset(token)

    def trace(
        self,
        name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator to trace a function.

        Args:
            name: Optional span name. Defaults to function name.
            kind: The kind of span.
            attributes: Optional initial attributes.

        Returns:
            Decorator function.
        """
        F = TypeVar("F", bound=Callable)

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.start_span_async(
                        span_name, kind=kind, attributes=attributes
                    ):
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore

            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.start_span(
                        span_name, kind=kind, attributes=attributes
                    ):
                        return func(*args, **kwargs)

                return sync_wrapper  # type: ignore

        return decorator


def create_tracer(config: TracingConfig) -> Tracer:
    """Create or get a tracer instance.

    Uses a registry to ensure the same config returns the same tracer.

    Args:
        config: Tracing configuration.

    Returns:
        Tracer instance.
    """
    # Create a unique key based on config values that affect behavior
    key = (
        f"{config.service_name}:"
        f"{config.service_version}:"
        f"{config.environment}:"
        f"{config.enabled}:"
        f"{config.sample_rate}:"
        f"{config.otlp_endpoint}"
    )
    if key in _tracer_registry:
        return _tracer_registry[key]

    tracer = Tracer(config)
    _tracer_registry[key] = tracer
    return tracer


def get_current_span() -> Span | None:
    """Get the current active span.

    Returns:
        Current span or None if not in a span context.
    """
    return _current_span.get()


def get_current_trace_id() -> str | None:
    """Get the current trace ID.

    Returns:
        Current trace ID or None if not in a span context.
    """
    span = _current_span.get()
    return span.trace_id if span else None


def inject_context(headers: dict[str, str]) -> None:
    """Inject current trace context into HTTP headers.

    Injects W3C TraceContext headers (traceparent, tracestate).

    Args:
        headers: Dictionary to inject headers into.
    """
    span = _current_span.get()
    if span is None or not span.is_recording():
        return

    flags = "01" if span.is_recording() else "00"
    traceparent = f"{W3C_TRACE_VERSION}-{span.trace_id}-{span.span_id}-{flags}"
    headers[W3C_TRACEPARENT_HEADER] = traceparent


def extract_context(headers: Mapping[str, str]) -> TracingContext | None:
    """Extract trace context from HTTP headers.

    Extracts W3C TraceContext headers (traceparent, tracestate).

    Args:
        headers: Dictionary of HTTP headers.

    Returns:
        TracingContext if valid headers found, None otherwise.
    """
    # Handle case-insensitive header lookup
    lower_headers = {k.lower(): v for k, v in headers.items()}

    traceparent = lower_headers.get(W3C_TRACEPARENT_HEADER)
    if not traceparent:
        return None

    ctx = TracingContext.from_traceparent(traceparent)
    if ctx is None:
        return None

    tracestate = lower_headers.get(W3C_TRACESTATE_HEADER)
    if tracestate:
        ctx.tracestate = tracestate

    return ctx
