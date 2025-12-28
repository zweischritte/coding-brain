"""
Security Event Audit Logging for Phase 2.

Provides structured JSON logging for security-relevant events:
- Session binding lifecycle events
- Session hijack detection
- DPoP validation events
- Authentication failures

All log entries include:
- Timestamp (ISO format)
- Event type
- Correlation IDs (request_id, trace_id, span_id)
- Relevant context (sanitized for privacy)
"""
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..observability.logging import get_json_logger

# Security event logger
logger = get_json_logger("security.audit")


class SecurityEventLogger:
    """Structured logger for security-relevant events.

    All methods emit structured JSON logs suitable for SIEM ingestion.
    User identifiers are partially masked for privacy while maintaining
    enough information for debugging.
    """

    @staticmethod
    def _mask_id(id_value: str, visible_chars: int = 8) -> str:
        """Mask an identifier for privacy, keeping first N chars visible.

        Args:
            id_value: The identifier to mask
            visible_chars: Number of characters to keep visible (default 8)

        Returns:
            Masked identifier like "abc12345..."
        """
        if not id_value or len(id_value) <= visible_chars:
            return id_value
        return id_value[:visible_chars] + "..."

    @classmethod
    def log_session_created(
        cls,
        session_id: str,
        user_id: str,
        org_id: str,
        dpop_bound: bool,
        store_type: str = "memory",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Log a session binding creation event.

        Args:
            session_id: The SSE session UUID
            user_id: The authenticated user ID
            org_id: The organization ID
            dpop_bound: Whether session is bound to DPoP proof
            store_type: The session store type (memory/valkey)
            ttl_seconds: Session TTL in seconds
        """
        logger.info(
            "Session binding created",
            extra={
                "event_type": "session.created",
                "session_id": cls._mask_id(session_id),
                "user_id": cls._mask_id(user_id),
                "org_id": org_id,
                "dpop_bound": dpop_bound,
                "store_type": store_type,
                "ttl_seconds": ttl_seconds,
            }
        )

    @classmethod
    def log_session_validated(
        cls,
        session_id: str,
        user_id: str,
        result: str,
        dpop_checked: bool = False,
    ) -> None:
        """Log a session binding validation event.

        Args:
            session_id: The SSE session UUID
            user_id: The user ID that was validated
            result: Validation result (success, user_mismatch, org_mismatch, dpop_mismatch, expired, not_found)
            dpop_checked: Whether DPoP was checked during validation
        """
        log_level = logging.INFO if result == "success" else logging.WARNING
        logger.log(
            log_level,
            f"Session binding validation: {result}",
            extra={
                "event_type": "session.validated",
                "session_id": cls._mask_id(session_id),
                "user_id": cls._mask_id(user_id),
                "result": result,
                "dpop_checked": dpop_checked,
            }
        )

    @classmethod
    def log_session_hijack_attempt(
        cls,
        session_id: str,
        expected_user_id: str,
        actual_user_id: str,
        expected_org_id: Optional[str] = None,
        actual_org_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log a potential session hijack attempt.

        This is a high-severity security event logged at ERROR level.

        Args:
            session_id: The SSE session UUID
            expected_user_id: The user ID bound to the session
            actual_user_id: The user ID that attempted to use the session
            expected_org_id: The org ID bound to the session
            actual_org_id: The org ID that attempted to use the session
            ip_address: The IP address of the request (if available)
        """
        logger.error(
            "Session hijack attempt detected",
            extra={
                "event_type": "session.hijack_attempt",
                "severity": "critical",
                "session_id": cls._mask_id(session_id),
                "expected_user_id": cls._mask_id(expected_user_id),
                "actual_user_id": cls._mask_id(actual_user_id),
                "expected_org_id": expected_org_id,
                "actual_org_id": actual_org_id,
                "ip_address": ip_address,
            }
        )

    @classmethod
    def log_dpop_validation_failed(
        cls,
        session_id: Optional[str],
        reason: str,
        method: Optional[str] = None,
        uri: Optional[str] = None,
    ) -> None:
        """Log a DPoP validation failure.

        Args:
            session_id: The SSE session UUID (may be None for SSE connect)
            reason: The reason for failure (invalid_signature, expired, replay, missing_claims, etc.)
            method: The HTTP method being validated
            uri: The URI being validated
        """
        logger.warning(
            f"DPoP validation failed: {reason}",
            extra={
                "event_type": "dpop.validation_failed",
                "session_id": cls._mask_id(session_id) if session_id else None,
                "reason": reason,
                "method": method,
                "uri": uri,
            }
        )

    @classmethod
    def log_dpop_validation_success(
        cls,
        session_id: Optional[str],
        thumbprint: str,
        method: Optional[str] = None,
    ) -> None:
        """Log a successful DPoP validation.

        Args:
            session_id: The SSE session UUID
            thumbprint: The DPoP key thumbprint (masked)
            method: The HTTP method that was validated
        """
        logger.debug(
            "DPoP validation successful",
            extra={
                "event_type": "dpop.validation_success",
                "session_id": cls._mask_id(session_id) if session_id else None,
                "thumbprint": cls._mask_id(thumbprint),
                "method": method,
            }
        )

    @classmethod
    def log_session_expired(
        cls,
        session_id: str,
        user_id: str,
        org_id: str,
        age_seconds: Optional[float] = None,
    ) -> None:
        """Log a session binding expiration event.

        Args:
            session_id: The SSE session UUID
            user_id: The user ID of the expired session
            org_id: The organization ID
            age_seconds: How old the session was when it expired
        """
        logger.info(
            "Session binding expired",
            extra={
                "event_type": "session.expired",
                "session_id": cls._mask_id(session_id),
                "user_id": cls._mask_id(user_id),
                "org_id": org_id,
                "age_seconds": age_seconds,
            }
        )

    @classmethod
    def log_session_deleted(
        cls,
        session_id: str,
        reason: str = "explicit_delete",
    ) -> None:
        """Log an explicit session binding deletion.

        Args:
            session_id: The SSE session UUID
            reason: The reason for deletion (explicit_delete, cleanup, logout, etc.)
        """
        logger.info(
            "Session binding deleted",
            extra={
                "event_type": "session.deleted",
                "session_id": cls._mask_id(session_id),
                "reason": reason,
            }
        )

    @classmethod
    def log_cleanup_cycle(
        cls,
        store_type: str,
        sessions_removed: int,
        duration_ms: float,
    ) -> None:
        """Log a session cleanup cycle completion.

        Args:
            store_type: The session store type
            sessions_removed: Number of sessions removed in this cycle
            duration_ms: How long the cleanup took in milliseconds
        """
        logger.info(
            "Session cleanup cycle completed",
            extra={
                "event_type": "session.cleanup_cycle",
                "store_type": store_type,
                "sessions_removed": sessions_removed,
                "duration_ms": round(duration_ms, 2),
            }
        )

    @classmethod
    def log_store_health_change(
        cls,
        store_type: str,
        healthy: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log a session store health status change.

        Args:
            store_type: The session store type
            healthy: Whether the store is now healthy
            error: Error message if unhealthy
        """
        log_level = logging.INFO if healthy else logging.ERROR
        status = "healthy" if healthy else "unhealthy"
        logger.log(
            log_level,
            f"Session store health: {status}",
            extra={
                "event_type": "session.store_health",
                "store_type": store_type,
                "healthy": healthy,
                "error": error,
            }
        )


# Convenience singleton instance
security_audit = SecurityEventLogger()
