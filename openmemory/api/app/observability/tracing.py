"""
OpenTelemetry Distributed Tracing for Phase 6.

Features:
- TracerProvider configuration with service name
- OTLP exporter support (when endpoint configured)
- FastAPI auto-instrumentation
- Trace ID extraction for logging correlation
"""
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Default service name
DEFAULT_SERVICE_NAME = "codingbrain-api"


def setup_tracing(
    service_name: str = DEFAULT_SERVICE_NAME,
    exporter_endpoint: Optional[str] = None,
) -> TracerProvider:
    """
    Configure OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification
        exporter_endpoint: OTLP exporter endpoint (optional)

    Returns:
        Configured TracerProvider
    """
    # Create resource with service name
    resource = Resource.create({
        SERVICE_NAME: service_name,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure exporter if endpoint provided
    endpoint = exporter_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # OTLP exporter not available

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    return provider


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer for the given module name.

    Args:
        name: Module or component name

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hex string.

    Returns:
        32-character hex trace ID, or None if not in a span
    """
    span = trace.get_current_span()
    if span is None:
        return None

    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None

    trace_id = ctx.trace_id
    if trace_id == 0:
        return None

    return format(trace_id, "032x")


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID as a hex string.

    Returns:
        16-character hex span ID, or None if not in a span
    """
    span = trace.get_current_span()
    if span is None:
        return None

    ctx = span.get_span_context()
    if ctx is None or not ctx.is_valid:
        return None

    span_id = ctx.span_id
    if span_id == 0:
        return None

    return format(span_id, "016x")


def instrument_app(app) -> None:
    """
    Instrument a FastAPI application with OpenTelemetry.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)


def uninstrument_app(app) -> None:
    """
    Remove OpenTelemetry instrumentation from a FastAPI application.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.uninstrument_app(app)
