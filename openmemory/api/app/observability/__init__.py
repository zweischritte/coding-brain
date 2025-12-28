"""
Observability module for Phase 6: Operability and Resilience.

Provides:
- Structured JSON logging with correlation IDs
- OpenTelemetry tracing instrumentation
- Prometheus metrics collection
"""
from app.observability.logging import (
    CorrelatedJsonFormatter,
    get_json_logger,
    setup_logging,
    set_request_context,
    clear_request_context,
)
from app.observability.tracing import (
    setup_tracing,
    get_tracer,
    get_current_trace_id,
    get_current_span_id,
    instrument_app,
    uninstrument_app,
)

__all__ = [
    # Logging
    "CorrelatedJsonFormatter",
    "get_json_logger",
    "setup_logging",
    "set_request_context",
    "clear_request_context",
    # Tracing
    "setup_tracing",
    "get_tracer",
    "get_current_trace_id",
    "get_current_span_id",
    "instrument_app",
    "uninstrument_app",
]
