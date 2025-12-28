"""
Tests for Security Audit Logging (Phase 2).

Verifies that security events are logged correctly as structured JSON.
"""
import json
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from app.security.audit_log import SecurityEventLogger, security_audit


class TestSecurityEventLoggerMasking:
    """Test ID masking for privacy."""

    def test_mask_id_short_string(self):
        """Test that short strings are not masked."""
        result = SecurityEventLogger._mask_id("abc")
        assert result == "abc"

    def test_mask_id_exact_length(self):
        """Test string exactly at visible length."""
        result = SecurityEventLogger._mask_id("12345678")
        assert result == "12345678"

    def test_mask_id_long_string(self):
        """Test that long strings are properly masked."""
        result = SecurityEventLogger._mask_id("1234567890abcdef")
        assert result == "12345678..."

    def test_mask_id_empty_string(self):
        """Test empty string handling."""
        result = SecurityEventLogger._mask_id("")
        assert result == ""

    def test_mask_id_custom_visible_chars(self):
        """Test custom visible character count."""
        result = SecurityEventLogger._mask_id("1234567890", visible_chars=4)
        assert result == "1234..."


class TestSecurityEventLogging:
    """Test security event logging methods."""

    @pytest.fixture
    def capture_logs(self):
        """Fixture to capture log output."""
        logger = logging.getLogger("security.audit")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        yield log_capture

        logger.removeHandler(handler)
        logger.setLevel(original_level)

    def test_log_session_created(self, capture_logs):
        """Test session creation logging."""
        SecurityEventLogger.log_session_created(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            user_id="user_1234567890",
            org_id="org_abc",
            dpop_bound=True,
            store_type="memory",
            ttl_seconds=1800,
        )

        log_output = capture_logs.getvalue()
        assert "Session binding created" in log_output

    def test_log_session_validated_success(self, capture_logs):
        """Test successful validation logging."""
        SecurityEventLogger.log_session_validated(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            user_id="user_1234567890",
            result="success",
            dpop_checked=True,
        )

        log_output = capture_logs.getvalue()
        assert "validation: success" in log_output

    def test_log_session_validated_failure(self, capture_logs):
        """Test failed validation logging."""
        SecurityEventLogger.log_session_validated(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            user_id="user_1234567890",
            result="not_found",
            dpop_checked=False,
        )

        log_output = capture_logs.getvalue()
        assert "not_found" in log_output

    def test_log_session_hijack_attempt(self, capture_logs):
        """Test hijack attempt logging (critical severity)."""
        SecurityEventLogger.log_session_hijack_attempt(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            expected_user_id="legitimate_user",
            actual_user_id="attacker_user",
            expected_org_id="org_abc",
            actual_org_id="org_abc",
            ip_address="192.168.1.100",
        )

        log_output = capture_logs.getvalue()
        assert "hijack attempt" in log_output.lower()

    def test_log_dpop_validation_failed(self, capture_logs):
        """Test DPoP validation failure logging."""
        SecurityEventLogger.log_dpop_validation_failed(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            reason="invalid_signature",
            method="POST",
            uri="/mcp/messages/",
        )

        log_output = capture_logs.getvalue()
        assert "DPoP validation failed" in log_output
        assert "invalid_signature" in log_output

    def test_log_dpop_validation_success(self, capture_logs):
        """Test DPoP validation success logging (debug level)."""
        SecurityEventLogger.log_dpop_validation_success(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            thumbprint="thumbprint_1234567890",
            method="POST",
        )

        log_output = capture_logs.getvalue()
        # Debug level should be captured with our fixture
        assert "DPoP validation successful" in log_output

    def test_log_session_expired(self, capture_logs):
        """Test session expiration logging."""
        SecurityEventLogger.log_session_expired(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            user_id="user_1234567890",
            org_id="org_abc",
            age_seconds=1800.5,
        )

        log_output = capture_logs.getvalue()
        assert "expired" in log_output.lower()

    def test_log_session_deleted(self, capture_logs):
        """Test session deletion logging."""
        SecurityEventLogger.log_session_deleted(
            session_id="abc12345-def6-7890-ghij-klmnopqrstuv",
            reason="explicit_delete",
        )

        log_output = capture_logs.getvalue()
        assert "deleted" in log_output.lower()

    def test_log_cleanup_cycle(self, capture_logs):
        """Test cleanup cycle logging."""
        SecurityEventLogger.log_cleanup_cycle(
            store_type="memory",
            sessions_removed=5,
            duration_ms=12.34,
        )

        log_output = capture_logs.getvalue()
        assert "cleanup" in log_output.lower()

    def test_log_store_health_change_healthy(self, capture_logs):
        """Test store health change logging (healthy)."""
        SecurityEventLogger.log_store_health_change(
            store_type="valkey",
            healthy=True,
        )

        log_output = capture_logs.getvalue()
        assert "health" in log_output.lower()
        assert "healthy" in log_output.lower()

    def test_log_store_health_change_unhealthy(self, capture_logs):
        """Test store health change logging (unhealthy)."""
        SecurityEventLogger.log_store_health_change(
            store_type="valkey",
            healthy=False,
            error="Connection refused",
        )

        log_output = capture_logs.getvalue()
        assert "unhealthy" in log_output.lower()


class TestSecurityAuditSingleton:
    """Test the security_audit singleton."""

    def test_singleton_exists(self):
        """Test that security_audit singleton is properly instantiated."""
        assert security_audit is not None
        assert isinstance(security_audit, SecurityEventLogger)

    def test_singleton_methods_callable(self):
        """Test that singleton methods are callable."""
        # These should not raise
        security_audit.log_session_created(
            session_id="test",
            user_id="test",
            org_id="test",
            dpop_bound=False,
        )


class TestAuditLogEventTypes:
    """Test that event types are correctly set in log entries."""

    @pytest.fixture
    def mock_logger(self):
        """Mock the audit logger to capture log calls."""
        with patch("app.security.audit_log.logger") as mock:
            yield mock

    def test_session_created_event_type(self, mock_logger):
        """Verify session.created event type."""
        SecurityEventLogger.log_session_created(
            session_id="test",
            user_id="test",
            org_id="test",
            dpop_bound=False,
        )

        # Check that info was called with the right extra data
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args
        assert call_kwargs[1]["extra"]["event_type"] == "session.created"

    def test_session_validated_event_type(self, mock_logger):
        """Verify session.validated event type."""
        SecurityEventLogger.log_session_validated(
            session_id="test",
            user_id="test",
            result="success",
        )

        call_kwargs = mock_logger.log.call_args
        assert call_kwargs[1]["extra"]["event_type"] == "session.validated"

    def test_session_hijack_attempt_event_type(self, mock_logger):
        """Verify session.hijack_attempt event type."""
        SecurityEventLogger.log_session_hijack_attempt(
            session_id="test",
            expected_user_id="user1",
            actual_user_id="user2",
        )

        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args
        assert call_kwargs[1]["extra"]["event_type"] == "session.hijack_attempt"
        assert call_kwargs[1]["extra"]["severity"] == "critical"

    def test_dpop_validation_failed_event_type(self, mock_logger):
        """Verify dpop.validation_failed event type."""
        SecurityEventLogger.log_dpop_validation_failed(
            session_id="test",
            reason="expired",
        )

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        assert call_kwargs[1]["extra"]["event_type"] == "dpop.validation_failed"

    def test_cleanup_cycle_event_type(self, mock_logger):
        """Verify session.cleanup_cycle event type."""
        SecurityEventLogger.log_cleanup_cycle(
            store_type="memory",
            sessions_removed=0,
            duration_ms=1.0,
        )

        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args
        assert call_kwargs[1]["extra"]["event_type"] == "session.cleanup_cycle"
