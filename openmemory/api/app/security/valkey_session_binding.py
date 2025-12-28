"""
Valkey-backed session binding store for MCP SSE authentication.

Provides distributed session binding storage for multi-worker deployments.
Uses Valkey (Redis-compatible) with automatic TTL expiration.

Phase 2 enhancements:
- Prometheus metrics for all operations
- OpenTelemetry tracing spans
- Structured security audit logging
"""
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, Tuple
from uuid import UUID

from opentelemetry import trace

from ..observability.session_metrics import (
    record_session_created,
    record_session_validated,
    record_sessions_expired,
    record_session_deleted,
    record_store_error,
    record_validation_duration,
)
from .audit_log import security_audit
from .session_binding import SessionBinding


logger = logging.getLogger(__name__)

# Get tracer for session binding operations
tracer = trace.get_tracer(__name__)


class ValkeyClientProtocol(Protocol):
    """Protocol for Valkey/Redis client operations."""

    def get(self, key: str) -> Optional[bytes]: ...
    def setex(self, key: str, seconds: int, value: str) -> bool: ...
    def delete(self, *keys: str) -> int: ...
    def keys(self, pattern: str) -> list[bytes]: ...
    def ping(self) -> bool: ...


class ValkeySessionBindingStore:
    """Valkey-backed session binding store with automatic TTL.

    Used for multi-worker production deployments where session bindings
    need to be shared across processes.

    Key format: mcp:session:{session_id}

    Phase 2: Includes metrics, tracing, and audit logging.
    """

    KEY_PREFIX = "mcp:session:"
    STORE_TYPE = "valkey"

    def __init__(
        self,
        client: ValkeyClientProtocol,
        default_ttl_seconds: int = 1800,
    ):
        """Initialize the Valkey session binding store.

        Args:
            client: Valkey/Redis client instance
            default_ttl_seconds: Default TTL for session bindings (30 minutes)
        """
        self.client = client
        self._default_ttl = default_ttl_seconds

    def _key(self, session_id: UUID) -> str:
        """Generate the Valkey key for a session binding."""
        return f"{self.KEY_PREFIX}{session_id}"

    def _serialize(self, binding: SessionBinding) -> str:
        """Serialize a SessionBinding to JSON."""
        data = {
            "session_id": str(binding.session_id),
            "user_id": binding.user_id,
            "org_id": binding.org_id,
            "issued_at": binding.issued_at.isoformat(),
            "expires_at": binding.expires_at.isoformat(),
            "dpop_thumbprint": binding.dpop_thumbprint,
        }
        return json.dumps(data)

    def _deserialize(self, data: bytes) -> SessionBinding:
        """Deserialize JSON to a SessionBinding."""
        obj = json.loads(data.decode("utf-8"))
        return SessionBinding(
            session_id=UUID(obj["session_id"]),
            user_id=obj["user_id"],
            org_id=obj["org_id"],
            issued_at=datetime.fromisoformat(obj["issued_at"]),
            expires_at=datetime.fromisoformat(obj["expires_at"]),
            dpop_thumbprint=obj.get("dpop_thumbprint"),
        )

    def create(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> SessionBinding:
        """Create a new session binding with metrics and tracing.

        Stores the binding in Valkey with automatic TTL expiration.

        Args:
            session_id: The SSE session UUID
            user_id: The authenticated user's ID
            org_id: The authenticated user's organization ID
            dpop_thumbprint: Optional DPoP key thumbprint for binding
            ttl_seconds: Custom TTL (uses default if not provided)

        Returns:
            The created SessionBinding

        Raises:
            ConnectionError: If Valkey connection fails (fail closed)
        """
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

                key = self._key(session_id)
                serialized = self._serialize(binding)
                self.client.setex(key, ttl, serialized)

                logger.debug(f"Created session binding for {session_id}")

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
                logger.error(f"Failed to create session binding: {e}")
                raise

    def get(self, session_id: UUID) -> Optional[SessionBinding]:
        """Get a session binding if it exists.

        Valkey handles TTL expiration automatically, so we don't need
        to check expires_at manually.

        Args:
            session_id: The SSE session UUID to look up

        Returns:
            The SessionBinding if found and valid, None otherwise
        """
        key = self._key(session_id)

        try:
            data = self.client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            record_store_error(self.STORE_TYPE, "get")
            logger.error(f"Failed to get session binding: {e}")
            # Fail closed: return None on error
            return None

    def validate(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
    ) -> bool:
        """Validate that a session binding matches the given principal with metrics and tracing.

        Args:
            session_id: The SSE session UUID
            user_id: The user ID to validate
            org_id: The organization ID to validate
            dpop_thumbprint: Optional DPoP thumbprint to validate

        Returns:
            True if binding exists and matches, False otherwise
        """
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
        """Delete a session binding with metrics.

        Args:
            session_id: The SSE session UUID to delete

        Returns:
            True if binding was deleted, False if not found or error
        """
        key = self._key(session_id)

        try:
            deleted = self.client.delete(key)
            if deleted > 0:
                record_session_deleted(self.STORE_TYPE)
                security_audit.log_session_deleted(str(session_id))
                return True
            return False
        except Exception as e:
            record_store_error(self.STORE_TYPE, "delete")
            logger.error(f"Failed to delete session binding: {e}")
            return False

    def cleanup_expired(self) -> int:
        """Cleanup expired bindings.

        For Valkey, TTL handles expiration automatically, so this is a no-op.
        We still emit metrics for monitoring purposes.

        Returns:
            Always 0 for Valkey (no manual cleanup needed)
        """
        with tracer.start_as_current_span("mcp.session.cleanup") as span:
            span.set_attribute("session.store_type", self.STORE_TYPE)
            span.set_attribute("session.cleanup_count", 0)

            # Audit log for monitoring
            security_audit.log_cleanup_cycle(
                store_type=self.STORE_TYPE,
                sessions_removed=0,
                duration_ms=0.0,
            )

            # Valkey handles TTL-based expiration automatically
            return 0

    def health_check(self) -> bool:
        """Check if Valkey connection is healthy with metrics.

        Returns:
            True if connected, False otherwise
        """
        try:
            result = self.client.ping()
            security_audit.log_store_health_change(
                store_type=self.STORE_TYPE,
                healthy=result,
            )
            return result
        except Exception as e:
            record_store_error(self.STORE_TYPE, "health_check")
            security_audit.log_store_health_change(
                store_type=self.STORE_TYPE,
                healthy=False,
                error=str(e),
            )
            logger.error(f"Valkey health check failed: {e}")
            return False


def get_valkey_session_binding_store(
    default_ttl_seconds: Optional[int] = None,
) -> Optional[ValkeySessionBindingStore]:
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

        host = os.getenv("VALKEY_HOST", "valkey")
        port = int(os.getenv("VALKEY_PORT", "6379"))
        ttl = default_ttl_seconds or int(os.getenv("MCP_SESSION_TTL_SECONDS", "1800"))

        client = redis.Redis(host=host, port=port, socket_timeout=5)
        client.ping()

        return ValkeySessionBindingStore(client=client, default_ttl_seconds=ttl)
    except Exception as e:
        logger.warning(f"Failed to connect to Valkey for session binding: {e}")
        return None
