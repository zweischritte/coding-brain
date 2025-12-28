"""
MCP Session Binding Metrics for Phase 2.

Prometheus metrics for session binding operations:
- Session creation, validation, expiration counts
- Store operation errors
- Active session gauge (memory store only)
- Validation latency histogram
"""
import time
from functools import wraps
from typing import Callable, Optional

from prometheus_client import Counter, Gauge, Histogram

# Session binding operation counters
mcp_session_bindings_created_total = Counter(
    "mcp_session_bindings_created_total",
    "Total MCP session bindings created",
    ["store_type", "dpop_bound"]
)

mcp_session_bindings_validated_total = Counter(
    "mcp_session_bindings_validated_total",
    "Total MCP session binding validation attempts",
    ["result"]  # success, user_mismatch, org_mismatch, dpop_mismatch, expired, not_found
)

mcp_session_bindings_expired_total = Counter(
    "mcp_session_bindings_expired_total",
    "Total MCP session bindings expired/cleaned up",
    ["store_type"]
)

mcp_session_bindings_deleted_total = Counter(
    "mcp_session_bindings_deleted_total",
    "Total MCP session bindings explicitly deleted",
    ["store_type"]
)

mcp_session_binding_store_errors_total = Counter(
    "mcp_session_binding_store_errors_total",
    "Total MCP session binding store errors",
    ["store_type", "operation"]  # create, get, validate, delete, cleanup
)

# Active sessions gauge (memory store only)
mcp_session_bindings_active = Gauge(
    "mcp_session_bindings_active",
    "Current active MCP session bindings (memory store only)",
    ["store_type"]
)

# Latency histogram for validation operations
mcp_session_binding_validation_duration_seconds = Histogram(
    "mcp_session_binding_validation_duration_seconds",
    "MCP session binding validation latency in seconds",
    ["store_type", "result"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# DPoP validation metrics
mcp_dpop_validations_total = Counter(
    "mcp_dpop_validations_total",
    "Total DPoP proof validation attempts",
    ["result"]  # valid, invalid, missing, error
)

# Cleanup scheduler metrics
mcp_session_cleanup_runs_total = Counter(
    "mcp_session_cleanup_runs_total",
    "Total session cleanup scheduler runs",
    ["status"]  # success, error
)

mcp_session_cleanup_last_run_timestamp = Gauge(
    "mcp_session_cleanup_last_run_timestamp",
    "Timestamp of last successful cleanup run (unix epoch)"
)


def record_session_created(store_type: str, dpop_bound: bool) -> None:
    """Record a session binding creation."""
    mcp_session_bindings_created_total.labels(
        store_type=store_type,
        dpop_bound=str(dpop_bound).lower()
    ).inc()


def record_session_validated(result: str) -> None:
    """Record a session binding validation result.

    Args:
        result: One of success, user_mismatch, org_mismatch, dpop_mismatch, expired, not_found
    """
    mcp_session_bindings_validated_total.labels(result=result).inc()


def record_sessions_expired(store_type: str, count: int) -> None:
    """Record expired session bindings cleaned up."""
    mcp_session_bindings_expired_total.labels(store_type=store_type).inc(count)


def record_session_deleted(store_type: str) -> None:
    """Record an explicit session binding deletion."""
    mcp_session_bindings_deleted_total.labels(store_type=store_type).inc()


def record_store_error(store_type: str, operation: str) -> None:
    """Record a session binding store error.

    Args:
        store_type: memory or valkey
        operation: create, get, validate, delete, cleanup, health_check
    """
    mcp_session_binding_store_errors_total.labels(
        store_type=store_type,
        operation=operation
    ).inc()


def set_active_sessions(store_type: str, count: int) -> None:
    """Set the current number of active sessions (memory store only)."""
    mcp_session_bindings_active.labels(store_type=store_type).set(count)


def record_validation_duration(store_type: str, result: str, duration_seconds: float) -> None:
    """Record session binding validation latency."""
    mcp_session_binding_validation_duration_seconds.labels(
        store_type=store_type,
        result=result
    ).observe(duration_seconds)


def record_dpop_validation(result: str) -> None:
    """Record a DPoP validation result.

    Args:
        result: One of valid, invalid, missing, error
    """
    mcp_dpop_validations_total.labels(result=result).inc()


def record_cleanup_run(success: bool) -> None:
    """Record a session cleanup scheduler run."""
    status = "success" if success else "error"
    mcp_session_cleanup_runs_total.labels(status=status).inc()
    if success:
        mcp_session_cleanup_last_run_timestamp.set(time.time())


def timed_validation(store_type: str):
    """Decorator to time session binding validation operations.

    Usage:
        @timed_validation("memory")
        def validate(...) -> tuple[bool, str]:
            # Returns (success, result_label)
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tuple[bool, str]:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                # Assume result is tuple of (success, result_label) or bool
                if isinstance(result, tuple):
                    success, result_label = result
                else:
                    success = result
                    result_label = "success" if success else "failure"
                record_validation_duration(store_type, result_label, duration)
                return result
            except Exception:
                duration = time.perf_counter() - start
                record_validation_duration(store_type, "error", duration)
                raise
        return wrapper
    return decorator
