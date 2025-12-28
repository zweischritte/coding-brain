"""
Tests for Phase 6: Structured Logging with JSON formatting and correlation IDs.

TDD: These tests are written first and should fail until implementation is complete.
"""
import json
import logging
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock


class TestJSONLogFormatting:
    """Test that log entries are formatted as valid JSON."""

    def test_log_entry_is_valid_json(self):
        """Log entries must be valid JSON."""
        from app.observability.logging import get_json_logger, setup_logging

        # Setup logging with JSON formatter
        setup_logging()
        logger = get_json_logger("test_json")

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        from app.observability.logging import CorrelatedJsonFormatter
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log a message
        logger.info("Test message")

        # Verify output is valid JSON
        log_output = stream.getvalue().strip()
        try:
            log_entry = json.loads(log_output)
            assert isinstance(log_entry, dict)
        except json.JSONDecodeError as e:
            pytest.fail(f"Log entry is not valid JSON: {log_output}\nError: {e}")

    def test_log_entry_includes_timestamp(self):
        """Log entries must include a timestamp in ISO format."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter
        import datetime

        logger = get_json_logger("test_timestamp")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        assert "timestamp" in log_entry, "Log entry must include 'timestamp'"
        # Verify ISO format
        try:
            datetime.datetime.fromisoformat(log_entry["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Timestamp is not valid ISO format: {log_entry['timestamp']}")

    def test_log_entry_includes_level(self):
        """Log entries must include the log level."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter

        logger = get_json_logger("test_level")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        test_cases = [
            (logger.debug, "DEBUG"),
            (logger.info, "INFO"),
            (logger.warning, "WARNING"),
            (logger.error, "ERROR"),
        ]

        for log_method, expected_level in test_cases:
            stream.truncate(0)
            stream.seek(0)
            log_method(f"Test {expected_level}")

            log_output = stream.getvalue().strip()
            log_entry = json.loads(log_output)

            assert "level" in log_entry, f"Log entry must include 'level'"
            assert log_entry["level"] == expected_level, f"Expected {expected_level}, got {log_entry['level']}"

    def test_log_entry_includes_logger_name(self):
        """Log entries must include the logger name."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter

        logger = get_json_logger("my_custom_logger")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        assert "logger" in log_entry, "Log entry must include 'logger'"
        assert log_entry["logger"] == "my_custom_logger"


class TestCorrelationIds:
    """Test request_id and trace_id correlation in log entries."""

    def test_log_includes_request_id_when_in_context(self):
        """Log entry must include request_id when in request context."""
        from app.observability.logging import (
            get_json_logger, CorrelatedJsonFormatter,
            set_request_context, clear_request_context
        )

        logger = get_json_logger("test_request_id")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Set request context
        request_id = "req-12345-abcde"
        set_request_context(request_id=request_id)

        try:
            logger.info("Test with request context")

            log_output = stream.getvalue().strip()
            log_entry = json.loads(log_output)

            assert "request_id" in log_entry, "Log entry must include 'request_id' when in context"
            assert log_entry["request_id"] == request_id
        finally:
            clear_request_context()

    def test_log_omits_request_id_when_not_in_context(self):
        """Log entry should not have request_id when not in request context."""
        from app.observability.logging import (
            get_json_logger, CorrelatedJsonFormatter, clear_request_context
        )

        # Ensure no context
        clear_request_context()

        logger = get_json_logger("test_no_request_id")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test without request context")

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        # request_id should either be absent or None
        if "request_id" in log_entry:
            assert log_entry["request_id"] is None


class TestExceptionLogging:
    """Test structured logging of exceptions."""

    def test_exception_includes_stack_trace(self):
        """Exception log entries must include stack trace."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter

        logger = get_json_logger("test_exception")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        # Should have exception info
        assert "exc_info" in log_entry or "exception" in log_entry or "traceback" in log_entry, (
            "Exception log must include exception info"
        )


class TestSensitiveFieldRedaction:
    """Test that sensitive fields are redacted from logs."""

    def test_password_field_is_redacted(self):
        """Password fields should be redacted in log entries."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter

        logger = get_json_logger("test_redaction")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log with extra fields that contain sensitive data
        logger.info("User login attempt", extra={"password": "secret123", "username": "testuser"})

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        # Password should be redacted
        if "password" in log_entry:
            assert log_entry["password"] != "secret123", "Password must be redacted"
            assert "[REDACTED]" in str(log_entry.get("password", "")) or "password" not in log_entry

    def test_token_field_is_redacted(self):
        """Token fields should be redacted in log entries."""
        from app.observability.logging import get_json_logger, CorrelatedJsonFormatter

        logger = get_json_logger("test_token_redaction")
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(CorrelatedJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Log with token field
        logger.info("API call", extra={"access_token": "eyJ0eXAiOiJKV1Q...", "endpoint": "/api/v1/data"})

        log_output = stream.getvalue().strip()
        log_entry = json.loads(log_output)

        # Token should be redacted
        if "access_token" in log_entry:
            assert "eyJ" not in str(log_entry.get("access_token", "")), "Token must be redacted"
