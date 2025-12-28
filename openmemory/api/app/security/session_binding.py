"""
Session binding store for MCP SSE authentication.

Binds SSE session_id to authenticated principal to prevent session hijacking.
Supports both memory-based (local/dev) and Valkey-based (multi-worker) stores.

Phase 2 enhancements:
- Prometheus metrics for all operations
- OpenTelemetry tracing spans
- Structured security audit logging
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Union
from uuid import UUID

from opentelemetry import trace

from ..observability.session_metrics import (
    record_session_created,
    record_session_validated,
    record_sessions_expired,
    record_session_deleted,
    record_store_error,
    set_active_sessions,
    record_validation_duration,
)
from .audit_log import security_audit

logger = logging.getLogger(__name__)

# Get tracer for session binding operations
tracer = trace.get_tracer(__name__)


@dataclass
class SessionBinding:
    """Binding between SSE session and authenticated principal."""

    session_id: UUID
    user_id: str
    org_id: str
    issued_at: datetime
    expires_at: datetime
    dpop_thumbprint: Optional[str] = None


class MemorySessionBindingStore:
    """Thread-safe in-memory session binding store with TTL.

    Used for local development and single-process deployments.
    For multi-worker production deployments, use ValkeySessionBindingStore (Phase 1).

    Phase 2: Includes metrics, tracing, and audit logging.
    """

    STORE_TYPE = "memory"

    def __init__(self, default_ttl_seconds: int = 1800):
        self._bindings: Dict[UUID, SessionBinding] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def create(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> SessionBinding:
        """Create a new session binding with metrics and tracing."""
        with tracer.start_as_current_span("mcp.session.create") as span:
            span.set_attribute("session.user_id", user_id[:8] if len(user_id) > 8 else user_id)
            span.set_attribute("session.org_id", org_id)
            span.set_attribute("session.dpop_bound", dpop_thumbprint is not None)
            span.set_attribute("session.store_type", self.STORE_TYPE)

            try:
                ttl = ttl_seconds or self._default_ttl
                now = datetime.now(timezone.utc)
                binding = SessionBinding(
                    session_id=session_id,
                    user_id=user_id,
                    org_id=org_id,
                    issued_at=now,
                    expires_at=now + timedelta(seconds=ttl),
                    dpop_thumbprint=dpop_thumbprint,
                )
                with self._lock:
                    self._bindings[session_id] = binding
                    # Update active sessions gauge
                    set_active_sessions(self.STORE_TYPE, len(self._bindings))

                # Record metrics
                record_session_created(self.STORE_TYPE, dpop_thumbprint is not None)

                # Audit log
                security_audit.log_session_created(
                    session_id=str(session_id),
                    user_id=user_id,
                    org_id=org_id,
                    dpop_bound=dpop_thumbprint is not None,
                    store_type=self.STORE_TYPE,
                    ttl_seconds=ttl,
                )

                return binding
            except Exception as e:
                record_store_error(self.STORE_TYPE, "create")
                span.record_exception(e)
                raise

    def get(self, session_id: UUID) -> Optional[SessionBinding]:
        """Get binding if exists and not expired."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding and binding.expires_at > datetime.now(timezone.utc):
                return binding
            elif binding:
                del self._bindings[session_id]  # Cleanup expired
                set_active_sessions(self.STORE_TYPE, len(self._bindings))
            return None

    def validate(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
    ) -> bool:
        """Validate that session binding matches principal with metrics and tracing."""
        with tracer.start_as_current_span("mcp.session.validate") as span:
            span.set_attribute("session.user_id", user_id[:8] if len(user_id) > 8 else user_id)
            span.set_attribute("session.org_id", org_id)
            span.set_attribute("session.dpop_bound", dpop_thumbprint is not None)
            span.set_attribute("session.store_type", self.STORE_TYPE)

            start_time = time.perf_counter()
            result = self._validate_internal(session_id, user_id, org_id, dpop_thumbprint)
            duration = time.perf_counter() - start_time

            # Determine result label
            success, result_label = result
            span.set_attribute("session.validation_result", result_label)

            # Record metrics
            record_session_validated(result_label)
            record_validation_duration(self.STORE_TYPE, result_label, duration)

            # Audit log
            security_audit.log_session_validated(
                session_id=str(session_id),
                user_id=user_id,
                result=result_label,
                dpop_checked=dpop_thumbprint is not None,
            )

            return success

    def _validate_internal(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Internal validation logic returning (success, result_label)."""
        binding = self.get(session_id)
        if not binding:
            return (False, "not_found")
        if binding.user_id != user_id:
            # Log hijack attempt
            security_audit.log_session_hijack_attempt(
                session_id=str(session_id),
                expected_user_id=binding.user_id,
                actual_user_id=user_id,
                expected_org_id=binding.org_id,
                actual_org_id=org_id,
            )
            return (False, "user_mismatch")
        if binding.org_id != org_id:
            return (False, "org_mismatch")
        if binding.dpop_thumbprint and dpop_thumbprint != binding.dpop_thumbprint:
            return (False, "dpop_mismatch")
        return (True, "success")

    def delete(self, session_id: UUID) -> bool:
        """Delete a session binding with metrics."""
        with self._lock:
            if session_id in self._bindings:
                del self._bindings[session_id]
                set_active_sessions(self.STORE_TYPE, len(self._bindings))
                record_session_deleted(self.STORE_TYPE)
                security_audit.log_session_deleted(str(session_id))
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired bindings with metrics. Returns count of removed bindings."""
        with tracer.start_as_current_span("mcp.session.cleanup") as span:
            span.set_attribute("session.store_type", self.STORE_TYPE)

            start_time = time.perf_counter()
            now = datetime.now(timezone.utc)
            removed = 0
            with self._lock:
                expired = [
                    sid
                    for sid, b in self._bindings.items()
                    if b.expires_at <= now
                ]
                for sid in expired:
                    del self._bindings[sid]
                    removed += 1
                set_active_sessions(self.STORE_TYPE, len(self._bindings))

            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("session.cleanup_count", removed)

            # Record metrics
            if removed > 0:
                record_sessions_expired(self.STORE_TYPE, removed)

            # Audit log
            security_audit.log_cleanup_cycle(
                store_type=self.STORE_TYPE,
                sessions_removed=removed,
                duration_ms=duration_ms,
            )

            return removed

    def active_count(self) -> int:
        """Return the current count of active (non-expired) sessions."""
        with self._lock:
            return len(self._bindings)


# Singleton instances
_session_binding_store: Optional[Union["MemorySessionBindingStore", "ValkeySessionBindingStore"]] = None

# Type alias for store types
SessionBindingStoreType = Union["MemorySessionBindingStore", "ValkeySessionBindingStore"]


def get_valkey_session_binding_store(
    default_ttl_seconds: Optional[int] = None,
) -> Optional["ValkeySessionBindingStore"]:
    """Factory function to create a ValkeySessionBindingStore.

    Attempts to connect to Valkey using environment configuration.
    Returns None if Valkey is not available.

    Args:
        default_ttl_seconds: Optional TTL override

    Returns:
        ValkeySessionBindingStore instance or None if unavailable
    """
    try:
        import redis
        from .valkey_session_binding import ValkeySessionBindingStore

        host = os.getenv("VALKEY_HOST", "valkey")
        port = int(os.getenv("VALKEY_PORT", "6379"))
        ttl = default_ttl_seconds or int(os.getenv("MCP_SESSION_TTL_SECONDS", "1800"))

        client = redis.Redis(host=host, port=port, socket_timeout=5)
        client.ping()

        return ValkeySessionBindingStore(client=client, default_ttl_seconds=ttl)
    except Exception as e:
        logger.warning(f"Failed to connect to Valkey for session binding: {e}")
        return None


def get_session_binding_store() -> SessionBindingStoreType:
    """Get the session binding store singleton.

    Store type is selected based on MCP_SESSION_STORE environment variable:
    - "memory" (default): Use in-memory store for local dev/single worker
    - "valkey": Use Valkey-backed store for multi-worker production

    If valkey is configured but unavailable, falls back to memory store.
    """
    global _session_binding_store
    if _session_binding_store is None:
        ttl = int(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
        store_type = os.environ.get("MCP_SESSION_STORE", "memory").lower()

        if store_type == "valkey":
            _session_binding_store = get_valkey_session_binding_store(ttl)
            if _session_binding_store is not None:
                logger.info("Using Valkey session binding store")
            else:
                logger.warning("Valkey unavailable, falling back to memory store")
                _session_binding_store = MemorySessionBindingStore(default_ttl_seconds=ttl)
        else:
            _session_binding_store = MemorySessionBindingStore(default_ttl_seconds=ttl)
            logger.info("Using memory session binding store")

    return _session_binding_store


def reset_session_binding_store() -> None:
    """Reset the singleton (for testing only)."""
    global _session_binding_store
    _session_binding_store = None


# Import ValkeySessionBindingStore for type hints (lazy import to avoid circular deps)
if False:  # TYPE_CHECKING equivalent without import
    from .valkey_session_binding import ValkeySessionBindingStore
