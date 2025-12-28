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

__all__ = [
    "CorrelatedJsonFormatter",
    "get_json_logger",
    "setup_logging",
    "set_request_context",
    "clear_request_context",
]
