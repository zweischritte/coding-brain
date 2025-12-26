"""Tests for audit hooks and security event logging.

Tests cover Phase 0c requirements per implementation plan section 5.1:
- Audit logging for all security-relevant events
- Event types: auth, access, admin, SCIM, AI, security
- Trace correlation in audit events
- Append-only storage with hash chaining
- Failed operations and denied access attempts
"""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from openmemory.api.observability.audit import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditConfig,
    AuditStore,
    MemoryAuditStore,
    AuditChainVerifier,
    AuditEventBuilder,
    create_audit_logger,
    AuditError,
    AuditIntegrityError,
)
from openmemory.api.observability.tracing import (
    TracingConfig,
    create_tracer,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def audit_config() -> AuditConfig:
    """Default audit configuration."""
    return AuditConfig(
        service_name="openmemory-api",
        enabled=True,
        hash_chain=True,
        retention_days=365,
    )


@pytest.fixture
def audit_logger(audit_config) -> AuditLogger:
    """Create an audit logger for testing."""
    store = MemoryAuditStore()
    return create_audit_logger(audit_config, store=store)


@pytest.fixture
def memory_store() -> MemoryAuditStore:
    """Create an in-memory audit store."""
    return MemoryAuditStore()


@pytest.fixture
def tracer():
    """Create a tracer for correlation tests."""
    config = TracingConfig(
        service_name="test-audit-service",
        enabled=True,
        sample_rate=1.0,
    )
    return create_tracer(config)


# ============================================================================
# AuditEventType Tests
# ============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_auth_event_types(self):
        """Authentication event types exist."""
        assert AuditEventType.AUTH_LOGIN.value == "auth.login"
        assert AuditEventType.AUTH_LOGOUT.value == "auth.logout"
        assert AuditEventType.AUTH_MFA_CHALLENGE.value == "auth.mfa_challenge"

    def test_access_event_types(self):
        """Access event types exist."""
        assert AuditEventType.ACCESS_QUERY.value == "access.query"
        assert AuditEventType.ACCESS_CONTEXT_LOAD.value == "access.context_load"
        assert AuditEventType.ACCESS_DENIED.value == "access.denied"
        assert AuditEventType.ACCESS_DATA_EXPORT.value == "access.data_export"
        assert AuditEventType.ACCESS_TOOL_INVOKED.value == "access.tool_invoked"

    def test_admin_event_types(self):
        """Admin event types exist."""
        assert AuditEventType.ADMIN_POLICY_CHANGE.value == "admin.policy_change"
        assert AuditEventType.ADMIN_ROLE_CHANGE.value == "admin.role_change"
        assert AuditEventType.ADMIN_USER_PROVISION.value == "admin.user_provision"
        assert AuditEventType.ADMIN_USER_DEPROVISION.value == "admin.user_deprovision"

    def test_scim_event_types(self):
        """SCIM event types exist."""
        assert AuditEventType.SCIM_USER_PROVISION.value == "scim.user_provision"
        assert AuditEventType.SCIM_USER_DEPROVISION.value == "scim.user_deprovision"

    def test_ai_event_types(self):
        """AI event types exist."""
        assert AuditEventType.AI_SUGGESTION_GENERATED.value == "ai.suggestion_generated"
        assert AuditEventType.AI_MODEL_SELECTED.value == "ai.model_selected"

    def test_security_event_types(self):
        """Security event types exist."""
        assert AuditEventType.SECURITY_SECRET_DETECTED.value == "security.secret_detected"
        assert AuditEventType.SECURITY_CONTENT_EXCLUDED.value == "security.content_excluded"
        assert AuditEventType.SECURITY_BREAK_GLASS.value == "security.break_glass"
        assert AuditEventType.SECURITY_MCP_REVOKED.value == "security.mcp_revoked"


# ============================================================================
# AuditEvent Tests
# ============================================================================


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_basic_event(self):
        """AuditEvent can be created with required fields."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        assert event.event_type == AuditEventType.AUTH_LOGIN
        assert event.timestamp is not None

    def test_create_event_with_context(self):
        """AuditEvent includes user and org context."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_QUERY,
            timestamp=datetime.now(timezone.utc),
            user_id="user-123",
            org_id="org-456",
            enterprise_id="ent-789",
        )
        assert event.user_id == "user-123"
        assert event.org_id == "org-456"
        assert event.enterprise_id == "ent-789"

    def test_create_event_with_trace(self):
        """AuditEvent includes trace context."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_QUERY,
            timestamp=datetime.now(timezone.utc),
            trace_id="abc123def456",
        )
        assert event.trace_id == "abc123def456"

    def test_create_event_with_details(self):
        """AuditEvent includes structured details."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_TOOL_INVOKED,
            timestamp=datetime.now(timezone.utc),
            details={
                "tool_name": "search_code_semantic",
                "latency_ms": 45.2,
                "result_count": 10,
            },
        )
        assert event.details["tool_name"] == "search_code_semantic"
        assert event.details["latency_ms"] == 45.2

    def test_event_id_generation(self):
        """AuditEvent has unique ID."""
        event1 = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        event2 = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        assert event1.event_id != event2.event_id

    def test_event_outcome_success(self):
        """AuditEvent can record success outcome."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
            outcome="success",
        )
        assert event.outcome == "success"

    def test_event_outcome_failure(self):
        """AuditEvent can record failure outcome."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            timestamp=datetime.now(timezone.utc),
            outcome="failure",
            details={"reason": "Insufficient permissions"},
        )
        assert event.outcome == "failure"

    def test_event_to_dict(self):
        """AuditEvent can be converted to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            user_id="user-123",
        )
        d = event.to_dict()
        assert d["event_type"] == "auth.login"
        assert d["user_id"] == "user-123"
        assert "timestamp" in d


# ============================================================================
# AuditConfig Tests
# ============================================================================


class TestAuditConfig:
    """Tests for AuditConfig dataclass."""

    def test_default_values(self):
        """AuditConfig has sensible defaults."""
        config = AuditConfig(service_name="test")
        assert config.enabled is True
        assert config.hash_chain is True
        assert config.retention_days == 365

    def test_custom_values(self):
        """AuditConfig accepts custom values."""
        config = AuditConfig(
            service_name="custom",
            enabled=True,
            hash_chain=True,
            retention_days=2555,  # 7 years
        )
        assert config.retention_days == 2555


# ============================================================================
# AuditLogger Creation Tests
# ============================================================================


class TestAuditLoggerCreation:
    """Tests for audit logger creation."""

    def test_create_audit_logger_basic(self, audit_config):
        """create_audit_logger creates a properly configured logger."""
        store = MemoryAuditStore()
        logger = create_audit_logger(audit_config, store=store)
        assert logger is not None
        assert logger.service_name == "openmemory-api"

    def test_create_audit_logger_with_store(self, audit_config):
        """Audit logger uses provided store."""
        store = MemoryAuditStore()
        logger = create_audit_logger(audit_config, store=store)
        assert logger._store is store

    def test_create_audit_logger_disabled(self):
        """Disabled audit logger doesn't store events."""
        config = AuditConfig(service_name="test", enabled=False)
        store = MemoryAuditStore()
        logger = create_audit_logger(config, store=store)

        logger.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
            )
        )
        assert len(store.events) == 0


# ============================================================================
# AuditLogger Logging Tests
# ============================================================================


class TestAuditLogging:
    """Tests for audit logging operations."""

    def test_log_event(self, audit_logger, memory_store):
        """Audit logger stores events."""
        audit_logger._store = memory_store
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
            user_id="user-123",
        )
        audit_logger.log(event)

        assert len(memory_store.events) == 1
        assert memory_store.events[0].user_id == "user-123"

    def test_log_multiple_events(self, audit_logger, memory_store):
        """Multiple events are stored in order."""
        audit_logger._store = memory_store
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.ACCESS_QUERY,
                timestamp=datetime.now(timezone.utc),
                details={"query_num": i},
            )
            audit_logger.log(event)

        assert len(memory_store.events) == 5

    def test_log_auth_login(self, audit_logger, memory_store):
        """Log authentication login event."""
        audit_logger._store = memory_store
        audit_logger.log_auth_login(
            user_id="user-123",
            org_id="org-456",
            method="oauth",
        )

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.AUTH_LOGIN
        assert event.details["method"] == "oauth"

    def test_log_auth_logout(self, audit_logger, memory_store):
        """Log authentication logout event."""
        audit_logger._store = memory_store
        audit_logger.log_auth_logout(user_id="user-123")

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.AUTH_LOGOUT

    def test_log_access_denied(self, audit_logger, memory_store):
        """Log access denied event."""
        audit_logger._store = memory_store
        audit_logger.log_access_denied(
            user_id="user-123",
            resource="/api/admin",
            reason="Insufficient permissions",
        )

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.ACCESS_DENIED
        assert event.outcome == "failure"
        assert event.details["reason"] == "Insufficient permissions"

    def test_log_tool_invoked(self, audit_logger, memory_store):
        """Log tool invocation event."""
        audit_logger._store = memory_store
        audit_logger.log_tool_invoked(
            user_id="user-123",
            org_id="org-456",
            tool_name="search_code_semantic",
            latency_ms=45.2,
        )

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.ACCESS_TOOL_INVOKED
        assert event.details["tool_name"] == "search_code_semantic"
        assert event.details["latency_ms"] == 45.2

    def test_log_secret_detected(self, audit_logger, memory_store):
        """Log secret detection event."""
        audit_logger._store = memory_store
        audit_logger.log_secret_detected(
            file_path="/repo/config.py",
            secret_type="api_key",
            action="blocked",
        )

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.SECURITY_SECRET_DETECTED
        assert event.details["action"] == "blocked"


# ============================================================================
# Trace Correlation Tests
# ============================================================================


class TestAuditTraceCorrelation:
    """Tests for trace correlation in audit events."""

    def test_audit_event_includes_trace_id(self, audit_logger, memory_store, tracer):
        """Audit events include trace_id from active span."""
        audit_logger._store = memory_store
        with tracer.start_span("operation") as span:
            audit_logger.log_auth_login(
                user_id="user-123",
                org_id="org-456",
            )

            event = memory_store.events[0]
            assert event.trace_id == span.trace_id

    def test_audit_without_span_no_trace(self, audit_logger, memory_store):
        """Audit events without span have no trace context."""
        audit_logger._store = memory_store
        audit_logger.log_auth_login(user_id="user-123")

        event = memory_store.events[0]
        assert event.trace_id is None


# ============================================================================
# Hash Chain Tests
# ============================================================================


class TestAuditHashChain:
    """Tests for append-only storage with hash chaining."""

    def test_first_event_hash(self, audit_logger, memory_store):
        """First event has hash based on content."""
        audit_logger._store = memory_store
        audit_logger.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
            )
        )

        event = memory_store.events[0]
        assert event.hash is not None
        assert len(event.hash) == 64  # SHA-256 hex

    def test_chain_includes_previous_hash(self, audit_logger, memory_store):
        """Subsequent events include previous hash in chain."""
        audit_logger._store = memory_store
        audit_logger.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
            )
        )
        audit_logger.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGOUT,
                timestamp=datetime.now(timezone.utc),
            )
        )

        event1 = memory_store.events[0]
        event2 = memory_store.events[1]

        assert event2.previous_hash == event1.hash

    def test_chain_is_verifiable(self, audit_logger, memory_store):
        """Hash chain can be verified for integrity."""
        audit_logger._store = memory_store
        for i in range(10):
            audit_logger.log(
                AuditEvent(
                    event_type=AuditEventType.ACCESS_QUERY,
                    timestamp=datetime.now(timezone.utc),
                    details={"query": i},
                )
            )

        verifier = AuditChainVerifier()
        is_valid, errors = verifier.verify(memory_store.events)
        assert is_valid is True
        assert len(errors) == 0

    def test_chain_detects_tampering(self, audit_logger, memory_store):
        """Hash chain detects content tampering."""
        audit_logger._store = memory_store
        for i in range(5):
            audit_logger.log(
                AuditEvent(
                    event_type=AuditEventType.ACCESS_QUERY,
                    timestamp=datetime.now(timezone.utc),
                    details={"query": i},
                )
            )

        # Tamper with an event
        memory_store.events[2].details["query"] = "tampered"

        verifier = AuditChainVerifier()
        is_valid, errors = verifier.verify(memory_store.events)
        assert is_valid is False
        assert len(errors) > 0

    def test_chain_detects_deletion(self, audit_logger, memory_store):
        """Hash chain detects event deletion."""
        audit_logger._store = memory_store
        for i in range(5):
            audit_logger.log(
                AuditEvent(
                    event_type=AuditEventType.ACCESS_QUERY,
                    timestamp=datetime.now(timezone.utc),
                    details={"query": i},
                )
            )

        # Delete an event from middle
        del memory_store.events[2]

        verifier = AuditChainVerifier()
        is_valid, errors = verifier.verify(memory_store.events)
        assert is_valid is False


# ============================================================================
# AuditEventBuilder Tests
# ============================================================================


class TestAuditEventBuilder:
    """Tests for fluent event builder."""

    def test_builder_basic(self):
        """Builder creates events with fluent API."""
        event = (
            AuditEventBuilder()
            .type(AuditEventType.AUTH_LOGIN)
            .user("user-123")
            .org("org-456")
            .build()
        )
        assert event.event_type == AuditEventType.AUTH_LOGIN
        assert event.user_id == "user-123"
        assert event.org_id == "org-456"

    def test_builder_with_details(self):
        """Builder adds structured details."""
        event = (
            AuditEventBuilder()
            .type(AuditEventType.ACCESS_TOOL_INVOKED)
            .user("user-123")
            .detail("tool_name", "search")
            .detail("latency_ms", 45.2)
            .build()
        )
        assert event.details["tool_name"] == "search"
        assert event.details["latency_ms"] == 45.2

    def test_builder_with_outcome(self):
        """Builder sets outcome."""
        event = (
            AuditEventBuilder()
            .type(AuditEventType.ACCESS_DENIED)
            .user("user-123")
            .failure("Permission denied")
            .build()
        )
        assert event.outcome == "failure"
        assert event.details["reason"] == "Permission denied"

    def test_builder_success_outcome(self):
        """Builder marks success outcome."""
        event = (
            AuditEventBuilder()
            .type(AuditEventType.AUTH_LOGIN)
            .user("user-123")
            .success()
            .build()
        )
        assert event.outcome == "success"


# ============================================================================
# Memory Audit Store Tests
# ============================================================================


class TestMemoryAuditStore:
    """Tests for in-memory audit store."""

    def test_store_event(self, memory_store):
        """Store accepts events."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
        )
        memory_store.store(event)
        assert len(memory_store.events) == 1

    def test_query_by_type(self, memory_store):
        """Store can query by event type."""
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
            )
        )
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGOUT,
                timestamp=datetime.now(timezone.utc),
            )
        )
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
            )
        )

        results = memory_store.query(event_type=AuditEventType.AUTH_LOGIN)
        assert len(results) == 2

    def test_query_by_user(self, memory_store):
        """Store can query by user_id."""
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
                user_id="user-1",
            )
        )
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=datetime.now(timezone.utc),
                user_id="user-2",
            )
        )

        results = memory_store.query(user_id="user-1")
        assert len(results) == 1
        assert results[0].user_id == "user-1"

    def test_query_by_time_range(self, memory_store):
        """Store can query by time range."""
        now = datetime.now(timezone.utc)

        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=now - timedelta(hours=2),
            )
        )
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=now - timedelta(minutes=30),
            )
        )
        memory_store.store(
            AuditEvent(
                event_type=AuditEventType.AUTH_LOGIN,
                timestamp=now,
            )
        )

        results = memory_store.query(
            start_time=now - timedelta(hours=1),
            end_time=now,
        )
        assert len(results) == 2

    def test_query_limit(self, memory_store):
        """Store respects query limit."""
        for i in range(10):
            memory_store.store(
                AuditEvent(
                    event_type=AuditEventType.AUTH_LOGIN,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        results = memory_store.query(limit=5)
        assert len(results) == 5


# ============================================================================
# Failed Operations Tests
# ============================================================================


class TestFailedOperations:
    """Tests for logging failed operations."""

    def test_log_failed_query(self, audit_logger, memory_store):
        """Failed queries are logged."""
        audit_logger._store = memory_store
        audit_logger.log_access_query(
            user_id="user-123",
            query="SELECT * FROM secrets",
            outcome="failure",
            error="Access denied",
        )

        event = memory_store.events[0]
        assert event.outcome == "failure"
        assert event.details["error"] == "Access denied"

    def test_log_failed_auth(self, audit_logger, memory_store):
        """Failed authentication is logged."""
        audit_logger._store = memory_store
        audit_logger.log_auth_login(
            user_id="user-123",
            outcome="failure",
            reason="Invalid credentials",
        )

        event = memory_store.events[0]
        assert event.outcome == "failure"

    def test_log_mfa_challenge(self, audit_logger, memory_store):
        """MFA challenge is logged."""
        audit_logger._store = memory_store
        audit_logger.log_mfa_challenge(
            user_id="user-123",
            challenge_type="totp",
            outcome="success",
        )

        event = memory_store.events[0]
        assert event.event_type == AuditEventType.AUTH_MFA_CHALLENGE


# ============================================================================
# GDPR/Compliance Tests
# ============================================================================


class TestAuditCompliance:
    """Tests for GDPR and compliance features."""

    def test_event_has_timestamp(self, audit_logger, memory_store):
        """Events always have timestamps."""
        audit_logger._store = memory_store
        audit_logger.log_auth_login(user_id="user-123")

        event = memory_store.events[0]
        assert event.timestamp is not None
        assert event.timestamp.tzinfo is not None  # Must be timezone-aware

    def test_event_retention(self, memory_store):
        """Events have retention metadata."""
        now = datetime.now(timezone.utc)
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=now,
            retention_until=now + timedelta(days=365),
        )
        memory_store.store(event)

        stored = memory_store.events[0]
        assert stored.retention_until > now

    def test_query_for_export(self, memory_store):
        """Store supports querying for data export (SAR)."""
        for i in range(5):
            memory_store.store(
                AuditEvent(
                    event_type=AuditEventType.ACCESS_QUERY,
                    timestamp=datetime.now(timezone.utc),
                    user_id="user-export",
                )
            )

        # Export all events for a user
        export_events = memory_store.export_user_events("user-export")
        assert len(export_events) == 5

    def test_deletion_leaves_tombstone(self, memory_store):
        """GDPR delete leaves audit trail without content."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN,
            timestamp=datetime.now(timezone.utc),
            user_id="user-delete",
            details={"ip": "192.168.1.1"},
        )
        memory_store.store(event)

        # Perform GDPR delete
        memory_store.gdpr_delete("user-delete")

        # Original event should be replaced with tombstone
        results = memory_store.query(user_id="user-delete")
        assert len(results) == 0

        # But tombstone exists
        tombstones = memory_store.get_tombstones()
        assert len(tombstones) == 1
        assert tombstones[0].event_type == AuditEventType.AUTH_LOGIN
        assert tombstones[0].details == {}  # Content removed


# ============================================================================
# Integration Tests
# ============================================================================


class TestAuditIntegration:
    """Integration tests for audit logging."""

    def test_full_request_audit(self, audit_logger, memory_store, tracer):
        """Simulate full request audit trail."""
        audit_logger._store = memory_store

        with tracer.start_span("http_request") as span:
            # Login
            audit_logger.log_auth_login(
                user_id="user-123",
                org_id="org-456",
                method="oauth",
            )

            # Access queries
            audit_logger.log_access_query(
                user_id="user-123",
                query="search_code_semantic",
                outcome="success",
            )

            # Tool invocation
            audit_logger.log_tool_invoked(
                user_id="user-123",
                org_id="org-456",
                tool_name="search_code_semantic",
                latency_ms=45.2,
            )

            # Logout
            audit_logger.log_auth_logout(user_id="user-123")

        # All events should have same trace_id
        assert len(memory_store.events) == 4
        for event in memory_store.events:
            assert event.trace_id == span.trace_id

    def test_security_incident_audit(self, audit_logger, memory_store):
        """Simulate security incident audit trail."""
        audit_logger._store = memory_store

        # Secret detected
        audit_logger.log_secret_detected(
            file_path="/repo/config.py",
            secret_type="api_key",
            action="blocked",
        )

        # Access denied
        audit_logger.log_access_denied(
            user_id="user-suspect",
            resource="/api/admin/secrets",
            reason="Insufficient permissions",
        )

        # Break glass access
        audit_logger.log(
            AuditEventBuilder()
            .type(AuditEventType.SECURITY_BREAK_GLASS)
            .user("admin-123")
            .detail("reason", "Production incident response")
            .detail("ticket", "INC-12345")
            .detail("expires_at", "2024-01-15T11:30:00Z")
            .build()
        )

        assert len(memory_store.events) == 3

        # Verify chain integrity
        verifier = AuditChainVerifier()
        is_valid, _ = verifier.verify(memory_store.events)
        assert is_valid is True
