"""Audit hooks for security events.

Phase 0c implementation per IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md section 5.1.

Features:
- Audit logging for all security-relevant events
- Event types: auth, access, admin, SCIM, AI, security
- Trace correlation in audit events
- Append-only storage with hash chaining
- Failed operations and denied access attempts
- GDPR compliance with tombstone deletion
"""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from openmemory.api.observability.tracing import get_current_span


class AuditError(Exception):
    """Base exception for audit errors."""

    pass


class AuditIntegrityError(AuditError):
    """Raised when audit chain integrity is violated."""

    pass


class AuditEventType(Enum):
    """Audit event types per implementation plan section 5.1."""

    # Authentication events
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_MFA_CHALLENGE = "auth.mfa_challenge"

    # Access events
    ACCESS_QUERY = "access.query"
    ACCESS_CONTEXT_LOAD = "access.context_load"
    ACCESS_DENIED = "access.denied"
    ACCESS_DATA_EXPORT = "access.data_export"
    ACCESS_TOOL_INVOKED = "access.tool_invoked"

    # Admin events
    ADMIN_POLICY_CHANGE = "admin.policy_change"
    ADMIN_ROLE_CHANGE = "admin.role_change"
    ADMIN_USER_PROVISION = "admin.user_provision"
    ADMIN_USER_DEPROVISION = "admin.user_deprovision"

    # SCIM events
    SCIM_USER_PROVISION = "scim.user_provision"
    SCIM_USER_DEPROVISION = "scim.user_deprovision"

    # AI events
    AI_SUGGESTION_GENERATED = "ai.suggestion_generated"
    AI_MODEL_SELECTED = "ai.model_selected"

    # Security events
    SECURITY_SECRET_DETECTED = "security.secret_detected"
    SECURITY_CONTENT_EXCLUDED = "security.content_excluded"
    SECURITY_BREAK_GLASS = "security.break_glass"
    SECURITY_MCP_REVOKED = "security.mcp_revoked"


@dataclass
class AuditEvent:
    """An audit event for security logging.

    Attributes:
        event_type: Type of audit event.
        timestamp: When the event occurred.
        event_id: Unique identifier for this event.
        user_id: Optional user ID.
        org_id: Optional organization ID.
        enterprise_id: Optional enterprise ID.
        trace_id: Optional trace ID for correlation.
        details: Structured event details.
        outcome: success, failure, or None.
        hash: SHA-256 hash of event content.
        previous_hash: Hash of previous event in chain.
        retention_until: When this event can be deleted.
    """

    event_type: AuditEventType
    timestamp: datetime
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    org_id: str | None = None
    enterprise_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None
    hash: str | None = None
    previous_hash: str | None = None
    retention_until: datetime | None = None
    _is_tombstone: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "org_id": self.org_id,
            "enterprise_id": self.enterprise_id,
            "trace_id": self.trace_id,
            "details": self.details,
            "outcome": self.outcome,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event content."""
        # Hash includes immutable content only (not hash fields)
        content = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "org_id": self.org_id,
            "enterprise_id": self.enterprise_id,
            "trace_id": self.trace_id,
            "details": self.details,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        service_name: Name of the service.
        enabled: Whether audit logging is enabled.
        hash_chain: Whether to use hash chaining for integrity.
        retention_days: Number of days to retain events.
    """

    service_name: str
    enabled: bool = True
    hash_chain: bool = True
    retention_days: int = 365


class AuditStore(ABC):
    """Abstract base for audit storage backends."""

    @abstractmethod
    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        pass

    @abstractmethod
    def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        pass

    @abstractmethod
    def get_last_hash(self) -> str | None:
        """Get the hash of the last event in the chain."""
        pass


class MemoryAuditStore(AuditStore):
    """In-memory audit store for testing."""

    def __init__(self):
        self.events: list[AuditEvent] = []
        self._tombstones: list[AuditEvent] = []

    def store(self, event: AuditEvent) -> None:
        """Store an audit event."""
        self.events.append(event)

    def query(
        self,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with filters."""
        results = []
        for event in self.events:
            # Skip tombstones
            if event._is_tombstone:
                continue

            # Apply filters
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if org_id and event.org_id != org_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results

    def get_last_hash(self) -> str | None:
        """Get the hash of the last event in the chain."""
        if not self.events:
            return None
        return self.events[-1].hash

    def export_user_events(self, user_id: str) -> list[AuditEvent]:
        """Export all events for a user (SAR compliance)."""
        return [e for e in self.events if e.user_id == user_id and not e._is_tombstone]

    def gdpr_delete(self, user_id: str) -> None:
        """Perform GDPR delete - replace content with tombstones."""
        for i, event in enumerate(self.events):
            if event.user_id == user_id and not event._is_tombstone:
                # Create tombstone preserving event type for audit trail
                tombstone = AuditEvent(
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    event_id=event.event_id,
                    user_id=None,  # Remove PII
                    org_id=event.org_id,
                    enterprise_id=event.enterprise_id,
                    trace_id=event.trace_id,
                    details={},  # Remove content
                    outcome=event.outcome,
                    hash=event.hash,
                    previous_hash=event.previous_hash,
                    _is_tombstone=True,
                )
                self.events[i] = tombstone
                self._tombstones.append(tombstone)

    def get_tombstones(self) -> list[AuditEvent]:
        """Get all tombstone events."""
        return self._tombstones.copy()


class AuditChainVerifier:
    """Verifies integrity of audit event chain."""

    def verify(self, events: list[AuditEvent]) -> tuple[bool, list[str]]:
        """Verify hash chain integrity.

        Args:
            events: List of audit events in order.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        for i, event in enumerate(events):
            # Verify hash
            expected_hash = event.compute_hash()
            if event.hash != expected_hash:
                errors.append(
                    f"Event {event.event_id}: hash mismatch "
                    f"(expected {expected_hash[:8]}..., got {event.hash[:8] if event.hash else 'None'}...)"
                )

            # Verify chain
            if i == 0:
                if event.previous_hash is not None:
                    errors.append(
                        f"Event {event.event_id}: first event should have no previous_hash"
                    )
            else:
                if event.previous_hash != events[i - 1].hash:
                    errors.append(
                        f"Event {event.event_id}: previous_hash does not match previous event"
                    )

        return len(errors) == 0, errors


class AuditEventBuilder:
    """Fluent builder for audit events."""

    def __init__(self):
        self._event_type: AuditEventType | None = None
        self._user_id: str | None = None
        self._org_id: str | None = None
        self._enterprise_id: str | None = None
        self._details: dict[str, Any] = {}
        self._outcome: str | None = None
        self._timestamp: datetime | None = None

    def type(self, event_type: AuditEventType) -> AuditEventBuilder:
        """Set event type."""
        self._event_type = event_type
        return self

    def user(self, user_id: str) -> AuditEventBuilder:
        """Set user ID."""
        self._user_id = user_id
        return self

    def org(self, org_id: str) -> AuditEventBuilder:
        """Set organization ID."""
        self._org_id = org_id
        return self

    def enterprise(self, enterprise_id: str) -> AuditEventBuilder:
        """Set enterprise ID."""
        self._enterprise_id = enterprise_id
        return self

    def detail(self, key: str, value: Any) -> AuditEventBuilder:
        """Add a detail field."""
        self._details[key] = value
        return self

    def success(self) -> AuditEventBuilder:
        """Mark as successful outcome."""
        self._outcome = "success"
        return self

    def failure(self, reason: str | None = None) -> AuditEventBuilder:
        """Mark as failed outcome."""
        self._outcome = "failure"
        if reason:
            self._details["reason"] = reason
        return self

    def timestamp(self, ts: datetime) -> AuditEventBuilder:
        """Set explicit timestamp."""
        self._timestamp = ts
        return self

    def build(self) -> AuditEvent:
        """Build the audit event."""
        if self._event_type is None:
            raise AuditError("Event type is required")

        # Get trace context
        span = get_current_span()
        trace_id = span.trace_id if span else None

        return AuditEvent(
            event_type=self._event_type,
            timestamp=self._timestamp or datetime.now(timezone.utc),
            user_id=self._user_id,
            org_id=self._org_id,
            enterprise_id=self._enterprise_id,
            trace_id=trace_id,
            details=self._details,
            outcome=self._outcome,
        )


class AuditLogger:
    """Audit logger for security events.

    Provides structured audit logging with hash chaining,
    trace correlation, and GDPR compliance features.
    """

    def __init__(self, config: AuditConfig, store: AuditStore):
        """Initialize the audit logger.

        Args:
            config: Audit configuration.
            store: Storage backend for events.
        """
        self._config = config
        self._store = store

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._config.service_name

    def _add_hash_chain(self, event: AuditEvent) -> None:
        """Add hash chain to event."""
        if not self._config.hash_chain:
            return

        # Get previous hash
        event.previous_hash = self._store.get_last_hash()

        # Compute hash
        event.hash = event.compute_hash()

    def _add_trace_context(self, event: AuditEvent) -> None:
        """Add trace context to event."""
        if event.trace_id is None:
            span = get_current_span()
            if span:
                event.trace_id = span.trace_id

    def _add_retention(self, event: AuditEvent) -> None:
        """Add retention metadata to event."""
        if event.retention_until is None:
            event.retention_until = datetime.now(timezone.utc) + timedelta(
                days=self._config.retention_days
            )

    def log(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: The audit event to log.
        """
        if not self._config.enabled:
            return

        # Enrich event
        self._add_trace_context(event)
        self._add_retention(event)
        self._add_hash_chain(event)

        # Store event
        self._store.store(event)

    def log_auth_login(
        self,
        user_id: str,
        org_id: str | None = None,
        method: str | None = None,
        outcome: str = "success",
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log authentication login event."""
        details = {**kwargs}
        if method:
            details["method"] = method
        if reason:
            details["reason"] = reason

        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            org_id=org_id,
            details=details,
            outcome=outcome,
        )
        self.log(event)

    def log_auth_logout(
        self,
        user_id: str,
        org_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log authentication logout event."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGOUT,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            org_id=org_id,
            details=kwargs,
            outcome="success",
        )
        self.log(event)

    def log_mfa_challenge(
        self,
        user_id: str,
        challenge_type: str,
        outcome: str = "success",
        **kwargs: Any,
    ) -> None:
        """Log MFA challenge event."""
        details = {"challenge_type": challenge_type, **kwargs}
        event = AuditEvent(
            event_type=AuditEventType.AUTH_MFA_CHALLENGE,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            details=details,
            outcome=outcome,
        )
        self.log(event)

    def log_access_query(
        self,
        user_id: str,
        query: str,
        outcome: str = "success",
        error: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log access query event."""
        details = {"query": query, **kwargs}
        if error:
            details["error"] = error

        event = AuditEvent(
            event_type=AuditEventType.ACCESS_QUERY,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            details=details,
            outcome=outcome,
        )
        self.log(event)

    def log_access_denied(
        self,
        user_id: str,
        resource: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Log access denied event."""
        details = {"resource": resource, "reason": reason, **kwargs}
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            details=details,
            outcome="failure",
        )
        self.log(event)

    def log_tool_invoked(
        self,
        user_id: str,
        org_id: str | None = None,
        tool_name: str | None = None,
        latency_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log tool invocation event."""
        details = {**kwargs}
        if tool_name:
            details["tool_name"] = tool_name
        if latency_ms is not None:
            details["latency_ms"] = latency_ms

        event = AuditEvent(
            event_type=AuditEventType.ACCESS_TOOL_INVOKED,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            org_id=org_id,
            details=details,
            outcome="success",
        )
        self.log(event)

    def log_secret_detected(
        self,
        file_path: str,
        secret_type: str,
        action: str,
        **kwargs: Any,
    ) -> None:
        """Log secret detection event."""
        details = {
            "file_path": file_path,
            "secret_type": secret_type,
            "action": action,
            **kwargs,
        }
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_SECRET_DETECTED,
            timestamp=datetime.now(timezone.utc),
            details=details,
        )
        self.log(event)


def create_audit_logger(
    config: AuditConfig,
    store: AuditStore | None = None,
) -> AuditLogger:
    """Create an audit logger.

    Args:
        config: Audit configuration.
        store: Optional storage backend. Uses MemoryAuditStore if not provided.

    Returns:
        AuditLogger instance.
    """
    if store is None:
        store = MemoryAuditStore()
    return AuditLogger(config, store)
