"""Observability module for OpenMemory API.

Phase 0c implementation per IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md section 13.

Components:
- tracing: OpenTelemetry tracing with GenAI semantic conventions
- logging: Structured logging with trace correlation
- audit: Audit hooks for security events
- slo: SLO definitions and tracking
"""

from openmemory.api.observability.tracing import (
    Tracer,
    TracingContext,
    SpanKind,
    SpanStatus,
    create_tracer,
    get_current_span,
    get_current_trace_id,
    inject_context,
    extract_context,
)
from openmemory.api.observability.logging import (
    StructuredLogger,
    LogLevel,
    create_logger,
    get_trace_context,
)
from openmemory.api.observability.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    create_audit_logger,
)
from openmemory.api.observability.slo import (
    SLOTracker,
    SLODefinition,
    SLOBudget,
    BurnRate,
    create_slo_tracker,
)

__all__ = [
    # Tracing
    "Tracer",
    "TracingContext",
    "SpanKind",
    "SpanStatus",
    "create_tracer",
    "get_current_span",
    "get_current_trace_id",
    "inject_context",
    "extract_context",
    # Logging
    "StructuredLogger",
    "LogLevel",
    "create_logger",
    "get_trace_context",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "create_audit_logger",
    # SLO
    "SLOTracker",
    "SLODefinition",
    "SLOBudget",
    "BurnRate",
    "create_slo_tracker",
]
