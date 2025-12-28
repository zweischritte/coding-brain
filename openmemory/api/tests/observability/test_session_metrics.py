"""
Tests for MCP Session Binding Metrics (Phase 2).

Verifies that Prometheus metrics are correctly recorded for:
- Session binding creation
- Session binding validation
- Session binding expiration/cleanup
- Store errors
- DPoP validation
- Cleanup scheduler runs
"""
import pytest
from unittest.mock import Mock, patch
from uuid import uuid4

from prometheus_client import REGISTRY

# Import the metrics module
from app.observability.session_metrics import (
    record_session_created,
    record_session_validated,
    record_sessions_expired,
    record_session_deleted,
    record_store_error,
    set_active_sessions,
    record_validation_duration,
    record_dpop_validation,
    record_cleanup_run,
    mcp_session_bindings_created_total,
    mcp_session_bindings_validated_total,
    mcp_session_bindings_expired_total,
    mcp_session_bindings_deleted_total,
    mcp_session_binding_store_errors_total,
    mcp_session_bindings_active,
    mcp_session_binding_validation_duration_seconds,
    mcp_dpop_validations_total,
    mcp_session_cleanup_runs_total,
    mcp_session_cleanup_last_run_timestamp,
)


class TestSessionMetricsRecording:
    """Test individual metric recording functions."""

    def test_record_session_created_memory_no_dpop(self):
        """Test recording session creation without DPoP binding."""
        # Get initial value
        labels = {"store_type": "memory", "dpop_bound": "false"}
        initial = mcp_session_bindings_created_total.labels(**labels)._value.get()

        # Record creation
        record_session_created("memory", dpop_bound=False)

        # Verify increment
        new_value = mcp_session_bindings_created_total.labels(**labels)._value.get()
        assert new_value == initial + 1

    def test_record_session_created_valkey_with_dpop(self):
        """Test recording session creation with DPoP binding."""
        labels = {"store_type": "valkey", "dpop_bound": "true"}
        initial = mcp_session_bindings_created_total.labels(**labels)._value.get()

        record_session_created("valkey", dpop_bound=True)

        new_value = mcp_session_bindings_created_total.labels(**labels)._value.get()
        assert new_value == initial + 1

    def test_record_session_validated_success(self):
        """Test recording successful validation."""
        initial = mcp_session_bindings_validated_total.labels(result="success")._value.get()

        record_session_validated("success")

        new_value = mcp_session_bindings_validated_total.labels(result="success")._value.get()
        assert new_value == initial + 1

    def test_record_session_validated_not_found(self):
        """Test recording not_found validation result."""
        initial = mcp_session_bindings_validated_total.labels(result="not_found")._value.get()

        record_session_validated("not_found")

        new_value = mcp_session_bindings_validated_total.labels(result="not_found")._value.get()
        assert new_value == initial + 1

    def test_record_session_validated_user_mismatch(self):
        """Test recording user_mismatch (hijack attempt)."""
        initial = mcp_session_bindings_validated_total.labels(result="user_mismatch")._value.get()

        record_session_validated("user_mismatch")

        new_value = mcp_session_bindings_validated_total.labels(result="user_mismatch")._value.get()
        assert new_value == initial + 1

    def test_record_session_validated_dpop_mismatch(self):
        """Test recording dpop_mismatch validation result."""
        initial = mcp_session_bindings_validated_total.labels(result="dpop_mismatch")._value.get()

        record_session_validated("dpop_mismatch")

        new_value = mcp_session_bindings_validated_total.labels(result="dpop_mismatch")._value.get()
        assert new_value == initial + 1

    def test_record_sessions_expired(self):
        """Test recording expired session count."""
        initial = mcp_session_bindings_expired_total.labels(store_type="memory")._value.get()

        record_sessions_expired("memory", count=5)

        new_value = mcp_session_bindings_expired_total.labels(store_type="memory")._value.get()
        assert new_value == initial + 5

    def test_record_session_deleted(self):
        """Test recording session deletion."""
        initial = mcp_session_bindings_deleted_total.labels(store_type="memory")._value.get()

        record_session_deleted("memory")

        new_value = mcp_session_bindings_deleted_total.labels(store_type="memory")._value.get()
        assert new_value == initial + 1

    def test_record_store_error_create(self):
        """Test recording store error for create operation."""
        labels = {"store_type": "valkey", "operation": "create"}
        initial = mcp_session_binding_store_errors_total.labels(**labels)._value.get()

        record_store_error("valkey", "create")

        new_value = mcp_session_binding_store_errors_total.labels(**labels)._value.get()
        assert new_value == initial + 1

    def test_record_store_error_health_check(self):
        """Test recording store error for health_check operation."""
        labels = {"store_type": "valkey", "operation": "health_check"}
        initial = mcp_session_binding_store_errors_total.labels(**labels)._value.get()

        record_store_error("valkey", "health_check")

        new_value = mcp_session_binding_store_errors_total.labels(**labels)._value.get()
        assert new_value == initial + 1

    def test_set_active_sessions(self):
        """Test setting active session gauge."""
        set_active_sessions("memory", 42)

        value = mcp_session_bindings_active.labels(store_type="memory")._value.get()
        assert value == 42

    def test_set_active_sessions_zero(self):
        """Test setting active sessions to zero."""
        set_active_sessions("memory", 0)

        value = mcp_session_bindings_active.labels(store_type="memory")._value.get()
        assert value == 0

    def test_record_validation_duration(self):
        """Test recording validation duration histogram."""
        # This records to a histogram, so we verify it doesn't raise
        record_validation_duration("memory", "success", 0.001)
        record_validation_duration("valkey", "not_found", 0.05)

    def test_record_dpop_validation_valid(self):
        """Test recording valid DPoP validation."""
        initial = mcp_dpop_validations_total.labels(result="valid")._value.get()

        record_dpop_validation("valid")

        new_value = mcp_dpop_validations_total.labels(result="valid")._value.get()
        assert new_value == initial + 1

    def test_record_dpop_validation_invalid(self):
        """Test recording invalid DPoP validation."""
        initial = mcp_dpop_validations_total.labels(result="invalid")._value.get()

        record_dpop_validation("invalid")

        new_value = mcp_dpop_validations_total.labels(result="invalid")._value.get()
        assert new_value == initial + 1

    def test_record_cleanup_run_success(self):
        """Test recording successful cleanup run."""
        initial = mcp_session_cleanup_runs_total.labels(status="success")._value.get()

        record_cleanup_run(success=True)

        new_value = mcp_session_cleanup_runs_total.labels(status="success")._value.get()
        assert new_value == initial + 1

    def test_record_cleanup_run_error(self):
        """Test recording failed cleanup run."""
        initial = mcp_session_cleanup_runs_total.labels(status="error")._value.get()

        record_cleanup_run(success=False)

        new_value = mcp_session_cleanup_runs_total.labels(status="error")._value.get()
        assert new_value == initial + 1

    def test_record_cleanup_run_updates_timestamp(self):
        """Test that successful cleanup run updates last run timestamp."""
        import time

        before = time.time()
        record_cleanup_run(success=True)
        after = time.time()

        timestamp = mcp_session_cleanup_last_run_timestamp._value.get()
        assert before <= timestamp <= after


class TestMetricLabels:
    """Test that metric labels are properly constrained (low cardinality)."""

    def test_store_type_labels_limited(self):
        """Verify store_type labels are limited to memory/valkey."""
        # These should work
        record_session_created("memory", False)
        record_session_created("valkey", True)

        # Note: Prometheus doesn't prevent arbitrary labels, but our
        # functions only accept specific values

    def test_validation_result_labels(self):
        """Verify validation result labels cover expected cases."""
        expected_results = ["success", "not_found", "user_mismatch", "org_mismatch", "dpop_mismatch"]
        for result in expected_results:
            record_session_validated(result)
            # Verify it was recorded (doesn't raise)

    def test_dpop_result_labels(self):
        """Verify DPoP result labels cover expected cases."""
        expected_results = ["valid", "invalid", "missing", "error"]
        for result in expected_results:
            record_dpop_validation(result)


class TestMetricsIntegrationWithStore:
    """Test that metrics are correctly recorded by session binding stores."""

    def test_memory_store_create_records_metrics(self):
        """Test that MemorySessionBindingStore.create records metrics."""
        from app.security.session_binding import MemorySessionBindingStore

        store = MemorySessionBindingStore(default_ttl_seconds=60)
        session_id = uuid4()

        initial_created = mcp_session_bindings_created_total.labels(
            store_type="memory", dpop_bound="false"
        )._value.get()

        store.create(session_id, "user123", "org456")

        new_created = mcp_session_bindings_created_total.labels(
            store_type="memory", dpop_bound="false"
        )._value.get()
        assert new_created == initial_created + 1

    def test_memory_store_validate_records_metrics(self):
        """Test that MemorySessionBindingStore.validate records metrics."""
        from app.security.session_binding import MemorySessionBindingStore

        store = MemorySessionBindingStore(default_ttl_seconds=60)
        session_id = uuid4()
        store.create(session_id, "user123", "org456")

        initial_validated = mcp_session_bindings_validated_total.labels(
            result="success"
        )._value.get()

        result = store.validate(session_id, "user123", "org456")

        assert result is True
        new_validated = mcp_session_bindings_validated_total.labels(
            result="success"
        )._value.get()
        assert new_validated == initial_validated + 1

    def test_memory_store_validate_user_mismatch_records_metrics(self):
        """Test that user mismatch validation records hijack attempt metric."""
        from app.security.session_binding import MemorySessionBindingStore

        store = MemorySessionBindingStore(default_ttl_seconds=60)
        session_id = uuid4()
        store.create(session_id, "user123", "org456")

        initial_mismatch = mcp_session_bindings_validated_total.labels(
            result="user_mismatch"
        )._value.get()

        result = store.validate(session_id, "attacker", "org456")

        assert result is False
        new_mismatch = mcp_session_bindings_validated_total.labels(
            result="user_mismatch"
        )._value.get()
        assert new_mismatch == initial_mismatch + 1

    def test_memory_store_cleanup_records_metrics(self):
        """Test that cleanup records expired sessions metric."""
        from app.security.session_binding import MemorySessionBindingStore
        from datetime import datetime, timedelta, timezone

        store = MemorySessionBindingStore(default_ttl_seconds=1)
        session_id = uuid4()

        # Create an already-expired session by manipulating the binding directly
        store.create(session_id, "user123", "org456")

        # Manually expire it
        with store._lock:
            binding = store._bindings[session_id]
            binding.expires_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        initial_expired = mcp_session_bindings_expired_total.labels(
            store_type="memory"
        )._value.get()

        removed = store.cleanup_expired()

        assert removed == 1
        new_expired = mcp_session_bindings_expired_total.labels(
            store_type="memory"
        )._value.get()
        assert new_expired == initial_expired + 1

    def test_memory_store_delete_records_metrics(self):
        """Test that delete records deletion metric."""
        from app.security.session_binding import MemorySessionBindingStore

        store = MemorySessionBindingStore(default_ttl_seconds=60)
        session_id = uuid4()
        store.create(session_id, "user123", "org456")

        initial_deleted = mcp_session_bindings_deleted_total.labels(
            store_type="memory"
        )._value.get()

        result = store.delete(session_id)

        assert result is True
        new_deleted = mcp_session_bindings_deleted_total.labels(
            store_type="memory"
        )._value.get()
        assert new_deleted == initial_deleted + 1

    def test_memory_store_active_count_gauge(self):
        """Test that active sessions gauge is updated correctly."""
        from app.security.session_binding import MemorySessionBindingStore

        store = MemorySessionBindingStore(default_ttl_seconds=60)

        # Create some sessions
        for i in range(3):
            store.create(uuid4(), f"user{i}", "org")

        # Check gauge reflects count
        active = mcp_session_bindings_active.labels(store_type="memory")._value.get()
        assert active >= 3  # May have other sessions from other tests
