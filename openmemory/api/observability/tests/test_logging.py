"""Tests for structured logging with trace correlation.

Tests cover Phase 0c requirements per implementation plan section 13:
- Structured logs include: trace_id, user_id, org_id, repo_id, tool_name, latency_ms
- Trace correlation with OpenTelemetry spans
- Log levels and formatting
- Context propagation in logs
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openmemory.api.observability.logging import (
    StructuredLogger,
    LogLevel,
    LogConfig,
    LogRecord,
    LogProcessor,
    JSONFormatter,
    create_logger,
    get_trace_context,
    bind_context,
    unbind_context,
    get_logger,
)
from openmemory.api.observability.tracing import (
    TracingConfig,
    create_tracer,
    get_current_span,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def log_config() -> LogConfig:
    """Default log configuration."""
    return LogConfig(
        service_name="openmemory-api",
        level=LogLevel.DEBUG,
        json_format=True,
        include_timestamp=True,
    )


@pytest.fixture
def logger(log_config) -> StructuredLogger:
    """Create a logger instance for testing."""
    return create_logger(log_config)


@pytest.fixture
def tracer():
    """Create a tracer for correlation tests."""
    config = TracingConfig(
        service_name="test-logging-service",
        enabled=True,
        sample_rate=1.0,
    )
    return create_tracer(config)


@pytest.fixture
def capture_output() -> StringIO:
    """Capture log output."""
    return StringIO()


# ============================================================================
# LogConfig Tests
# ============================================================================


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self):
        """LogConfig has sensible defaults."""
        config = LogConfig(service_name="test")
        assert config.service_name == "test"
        assert config.level == LogLevel.INFO
        assert config.json_format is True
        assert config.include_timestamp is True

    def test_custom_values(self):
        """LogConfig accepts custom values."""
        config = LogConfig(
            service_name="custom",
            level=LogLevel.DEBUG,
            json_format=False,
            include_timestamp=False,
        )
        assert config.level == LogLevel.DEBUG
        assert config.json_format is False
        assert config.include_timestamp is False


# ============================================================================
# LogLevel Tests
# ============================================================================


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels_exist(self):
        """All expected log levels exist."""
        assert LogLevel.DEBUG
        assert LogLevel.INFO
        assert LogLevel.WARNING
        assert LogLevel.ERROR
        assert LogLevel.CRITICAL

    def test_log_level_ordering(self):
        """Log levels have correct ordering."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value


# ============================================================================
# Logger Creation Tests
# ============================================================================


class TestLoggerCreation:
    """Tests for logger creation."""

    def test_create_logger_with_config(self, log_config):
        """create_logger creates a properly configured logger."""
        logger = create_logger(log_config)
        assert logger is not None
        assert logger.service_name == "openmemory-api"

    def test_create_logger_with_name(self):
        """create_logger can create named loggers."""
        config = LogConfig(service_name="test")
        logger = create_logger(config, name="my.module")
        assert logger.name == "my.module"

    def test_get_logger_convenience(self):
        """get_logger provides a simpler API."""
        logger = get_logger("my.module")
        assert logger is not None
        assert logger.name == "my.module"

    def test_logger_inherits_context(self, log_config):
        """Child loggers inherit parent context."""
        parent = create_logger(log_config)
        parent.bind(request_id="req-123")

        child = parent.child("child.module")
        # Child should have access to parent's bound context
        assert child.get_bound_value("request_id") == "req-123"


# ============================================================================
# Basic Logging Tests
# ============================================================================


class TestBasicLogging:
    """Tests for basic logging operations."""

    def test_debug_log(self, logger, capture_output):
        """Logger can log at DEBUG level."""
        with patch.object(logger, "_write") as mock_write:
            logger.debug("Debug message")
            mock_write.assert_called_once()
            record = mock_write.call_args[0][0]
            assert record.level == LogLevel.DEBUG
            assert record.message == "Debug message"

    def test_info_log(self, logger, capture_output):
        """Logger can log at INFO level."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Info message")
            mock_write.assert_called_once()
            record = mock_write.call_args[0][0]
            assert record.level == LogLevel.INFO

    def test_warning_log(self, logger, capture_output):
        """Logger can log at WARNING level."""
        with patch.object(logger, "_write") as mock_write:
            logger.warning("Warning message")
            mock_write.assert_called_once()
            record = mock_write.call_args[0][0]
            assert record.level == LogLevel.WARNING

    def test_error_log(self, logger, capture_output):
        """Logger can log at ERROR level."""
        with patch.object(logger, "_write") as mock_write:
            logger.error("Error message")
            mock_write.assert_called_once()
            record = mock_write.call_args[0][0]
            assert record.level == LogLevel.ERROR

    def test_critical_log(self, logger, capture_output):
        """Logger can log at CRITICAL level."""
        with patch.object(logger, "_write") as mock_write:
            logger.critical("Critical message")
            mock_write.assert_called_once()
            record = mock_write.call_args[0][0]
            assert record.level == LogLevel.CRITICAL

    def test_log_with_extra_fields(self, logger):
        """Logger accepts extra fields."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Message with extra", user_id="user-123", count=42)
            record = mock_write.call_args[0][0]
            assert record.extra["user_id"] == "user-123"
            assert record.extra["count"] == 42


# ============================================================================
# LogRecord Tests
# ============================================================================


class TestLogRecord:
    """Tests for LogRecord dataclass."""

    def test_log_record_creation(self):
        """LogRecord can be created with required fields."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            timestamp=datetime.now(timezone.utc),
        )
        assert record.level == LogLevel.INFO
        assert record.message == "Test message"

    def test_log_record_with_extra(self):
        """LogRecord includes extra fields."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            timestamp=datetime.now(timezone.utc),
            extra={"key": "value"},
        )
        assert record.extra["key"] == "value"

    def test_log_record_with_trace_context(self):
        """LogRecord includes trace context."""
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            timestamp=datetime.now(timezone.utc),
            trace_id="abc123",
            span_id="def456",
        )
        assert record.trace_id == "abc123"
        assert record.span_id == "def456"


# ============================================================================
# Trace Correlation Tests
# ============================================================================


class TestTraceCorrelation:
    """Tests for trace correlation in logs."""

    def test_get_trace_context_with_active_span(self, tracer):
        """get_trace_context returns trace info when span is active."""
        with tracer.start_span("test-operation"):
            ctx = get_trace_context()
            assert ctx["trace_id"] is not None
            assert ctx["span_id"] is not None
            assert len(ctx["trace_id"]) == 32
            assert len(ctx["span_id"]) == 16

    def test_get_trace_context_no_span(self):
        """get_trace_context returns None values when no span."""
        ctx = get_trace_context()
        assert ctx["trace_id"] is None
        assert ctx["span_id"] is None

    def test_log_includes_trace_id(self, logger, tracer):
        """Log records include trace_id from active span."""
        with patch.object(logger, "_write") as mock_write:
            with tracer.start_span("test-operation") as span:
                logger.info("Message in span")

                record = mock_write.call_args[0][0]
                assert record.trace_id == span.trace_id
                assert record.span_id == span.span_id

    def test_log_without_span_no_trace(self, logger):
        """Log records without span have no trace context."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Message without span")

            record = mock_write.call_args[0][0]
            assert record.trace_id is None
            assert record.span_id is None

    def test_nested_span_trace_correlation(self, logger, tracer):
        """Nested spans correlate correctly in logs."""
        records = []
        with patch.object(logger, "_write", side_effect=lambda r: records.append(r)):
            with tracer.start_span("parent") as parent:
                logger.info("In parent")
                with tracer.start_span("child") as child:
                    logger.info("In child")

        assert len(records) == 2
        # Both should have same trace_id
        assert records[0].trace_id == records[1].trace_id
        # But different span_ids
        assert records[0].span_id == parent.span_id
        assert records[1].span_id == child.span_id


# ============================================================================
# Required Fields Tests
# ============================================================================


class TestRequiredFields:
    """Tests for required log fields per implementation plan."""

    def test_log_includes_user_id(self, logger):
        """Logs can include user_id."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("User action", user_id="user-123")
            record = mock_write.call_args[0][0]
            assert record.extra["user_id"] == "user-123"

    def test_log_includes_org_id(self, logger):
        """Logs can include org_id."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Org action", org_id="org-456")
            record = mock_write.call_args[0][0]
            assert record.extra["org_id"] == "org-456"

    def test_log_includes_repo_id(self, logger):
        """Logs can include repo_id."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Repo action", repo_id="repo-789")
            record = mock_write.call_args[0][0]
            assert record.extra["repo_id"] == "repo-789"

    def test_log_includes_tool_name(self, logger):
        """Logs can include tool_name."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Tool invoked", tool_name="search_code_semantic")
            record = mock_write.call_args[0][0]
            assert record.extra["tool_name"] == "search_code_semantic"

    def test_log_includes_latency_ms(self, logger):
        """Logs can include latency_ms."""
        with patch.object(logger, "_write") as mock_write:
            logger.info("Request completed", latency_ms=45.2)
            record = mock_write.call_args[0][0]
            assert record.extra["latency_ms"] == 45.2

    def test_log_includes_all_required_fields(self, logger, tracer):
        """Logs can include all required fields together."""
        with patch.object(logger, "_write") as mock_write:
            with tracer.start_span("operation"):
                logger.info(
                    "Full context log",
                    user_id="user-123",
                    org_id="org-456",
                    repo_id="repo-789",
                    tool_name="search_code_semantic",
                    latency_ms=45.2,
                )

                record = mock_write.call_args[0][0]
                assert record.trace_id is not None
                assert record.extra["user_id"] == "user-123"
                assert record.extra["org_id"] == "org-456"
                assert record.extra["repo_id"] == "repo-789"
                assert record.extra["tool_name"] == "search_code_semantic"
                assert record.extra["latency_ms"] == 45.2


# ============================================================================
# Context Binding Tests
# ============================================================================


class TestContextBinding:
    """Tests for binding context to logs."""

    def test_bind_context(self, logger):
        """bind adds context to all subsequent logs."""
        logger.bind(request_id="req-abc")

        with patch.object(logger, "_write") as mock_write:
            logger.info("Message 1")
            logger.info("Message 2")

            # Both logs should have request_id
            for call in mock_write.call_args_list:
                record = call[0][0]
                assert record.extra.get("request_id") == "req-abc"

    def test_bind_multiple_values(self, logger):
        """bind can add multiple context values."""
        logger.bind(user_id="user-1", org_id="org-2", session_id="sess-3")

        with patch.object(logger, "_write") as mock_write:
            logger.info("Message")

            record = mock_write.call_args[0][0]
            assert record.extra["user_id"] == "user-1"
            assert record.extra["org_id"] == "org-2"
            assert record.extra["session_id"] == "sess-3"

    def test_unbind_removes_context(self, logger):
        """unbind removes context from subsequent logs."""
        logger.bind(request_id="req-abc")
        logger.unbind("request_id")

        with patch.object(logger, "_write") as mock_write:
            logger.info("Message")

            record = mock_write.call_args[0][0]
            assert "request_id" not in record.extra

    def test_local_context_overrides_bound(self, logger):
        """Local log fields override bound context."""
        logger.bind(user_id="bound-user")

        with patch.object(logger, "_write") as mock_write:
            logger.info("Message", user_id="local-user")

            record = mock_write.call_args[0][0]
            assert record.extra["user_id"] == "local-user"

    def test_context_manager_bind(self, logger):
        """Context manager temporarily binds context."""
        with patch.object(logger, "_write") as mock_write:
            with logger.context(temp_key="temp_value"):
                logger.info("Inside context")
                record1 = mock_write.call_args[0][0]
                assert record1.extra["temp_key"] == "temp_value"

            mock_write.reset_mock()
            logger.info("Outside context")
            record2 = mock_write.call_args[0][0]
            assert "temp_key" not in record2.extra


# ============================================================================
# Global Context Binding Tests
# ============================================================================


class TestGlobalContextBinding:
    """Tests for global context binding functions."""

    def test_global_bind_context(self, logger):
        """bind_context adds global context."""
        bind_context(global_key="global_value")
        try:
            with patch.object(logger, "_write") as mock_write:
                logger.info("Message")
                record = mock_write.call_args[0][0]
                assert record.extra.get("global_key") == "global_value"
        finally:
            unbind_context("global_key")

    def test_global_unbind_context(self, logger):
        """unbind_context removes global context."""
        bind_context(temp_key="temp")
        unbind_context("temp_key")

        with patch.object(logger, "_write") as mock_write:
            logger.info("Message")
            record = mock_write.call_args[0][0]
            assert "temp_key" not in record.extra


# ============================================================================
# JSON Formatter Tests
# ============================================================================


class TestJSONFormatter:
    """Tests for JSON log formatting."""

    def test_json_formatter_basic(self):
        """JSONFormatter produces valid JSON."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test message",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_json_formatter_with_trace(self):
        """JSONFormatter includes trace context."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            timestamp=datetime.now(timezone.utc),
            trace_id="abc123def456",
            span_id="12345678",
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["trace_id"] == "abc123def456"
        assert parsed["span_id"] == "12345678"

    def test_json_formatter_with_extra(self):
        """JSONFormatter includes extra fields."""
        formatter = JSONFormatter()
        record = LogRecord(
            level=LogLevel.INFO,
            message="Test",
            timestamp=datetime.now(timezone.utc),
            extra={"user_id": "u-123", "latency_ms": 45.5},
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["user_id"] == "u-123"
        assert parsed["latency_ms"] == 45.5

    def test_json_formatter_exception(self):
        """JSONFormatter handles exceptions."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = LogRecord(
            level=LogLevel.ERROR,
            message="Error occurred",
            timestamp=datetime.now(timezone.utc),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


# ============================================================================
# Log Processor Tests
# ============================================================================


class TestLogProcessor:
    """Tests for log processors."""

    def test_processor_chain(self, log_config):
        """Multiple processors can be chained."""

        def add_env(record: LogRecord) -> LogRecord:
            record.extra["environment"] = "test"
            return record

        def add_version(record: LogRecord) -> LogRecord:
            record.extra["version"] = "1.0.0"
            return record

        config = LogConfig(
            service_name="test",
            processors=[add_env, add_version],
        )
        logger = create_logger(config)

        with patch.object(logger, "_output") as mock_output:
            logger.info("Message")
            record = mock_output.call_args[0][0]
            assert record.extra["environment"] == "test"
            assert record.extra["version"] == "1.0.0"

    def test_processor_can_modify_message(self, log_config):
        """Processors can modify log messages."""

        def uppercase_message(record: LogRecord) -> LogRecord:
            record.message = record.message.upper()
            return record

        config = LogConfig(
            service_name="test",
            processors=[uppercase_message],
        )
        logger = create_logger(config)

        with patch.object(logger, "_output") as mock_output:
            logger.info("hello world")
            record = mock_output.call_args[0][0]
            assert record.message == "HELLO WORLD"


# ============================================================================
# Log Level Filtering Tests
# ============================================================================


class TestLogLevelFiltering:
    """Tests for log level filtering."""

    def test_below_level_not_logged(self):
        """Messages below configured level are not logged."""
        config = LogConfig(service_name="test", level=LogLevel.WARNING)
        logger = create_logger(config)

        with patch.object(logger, "_write") as mock_write:
            logger.debug("Debug message")
            logger.info("Info message")
            mock_write.assert_not_called()

    def test_at_level_logged(self):
        """Messages at configured level are logged."""
        config = LogConfig(service_name="test", level=LogLevel.WARNING)
        logger = create_logger(config)

        with patch.object(logger, "_write") as mock_write:
            logger.warning("Warning message")
            mock_write.assert_called_once()

    def test_above_level_logged(self):
        """Messages above configured level are logged."""
        config = LogConfig(service_name="test", level=LogLevel.WARNING)
        logger = create_logger(config)

        with patch.object(logger, "_write") as mock_write:
            logger.error("Error message")
            logger.critical("Critical message")
            assert mock_write.call_count == 2


# ============================================================================
# Async Logging Tests
# ============================================================================


class TestAsyncLogging:
    """Tests for async logging support."""

    @pytest.mark.asyncio
    async def test_async_context_preserved(self, logger, tracer):
        """Trace context is preserved in async operations."""
        records = []

        with patch.object(logger, "_write", side_effect=lambda r: records.append(r)):
            async with tracer.start_span_async("async-op") as span:
                await asyncio.sleep(0.001)
                logger.info("In async context")

        assert len(records) == 1
        assert records[0].trace_id == span.trace_id

    @pytest.mark.asyncio
    async def test_concurrent_async_contexts(self, logger, tracer):
        """Concurrent async operations have correct trace contexts."""
        records = []

        with patch.object(logger, "_write", side_effect=lambda r: records.append(r)):

            async def task(name: str):
                async with tracer.start_span_async(name) as span:
                    await asyncio.sleep(0.001)
                    logger.info(f"In {name}", task_name=name)
                    return span.span_id

            async with tracer.start_span_async("parent"):
                await asyncio.gather(task("task1"), task("task2"))

        assert len(records) == 2
        # Each log should have its own span_id
        span_ids = [r.span_id for r in records]
        assert len(set(span_ids)) == 2


# ============================================================================
# Exception Logging Tests
# ============================================================================


class TestExceptionLogging:
    """Tests for exception logging."""

    def test_exception_method(self, logger):
        """logger.exception logs with exception info."""
        try:
            raise ValueError("Test error")
        except ValueError:
            with patch.object(logger, "_write") as mock_write:
                logger.exception("Something went wrong")

                record = mock_write.call_args[0][0]
                assert record.level == LogLevel.ERROR
                assert record.exc_info is not None

    def test_error_with_exc_info(self, logger):
        """logger.error can include exc_info."""
        try:
            raise RuntimeError("Test")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()

            with patch.object(logger, "_write") as mock_write:
                logger.error("Error with exc", exc_info=exc_info)

                record = mock_write.call_args[0][0]
                assert record.exc_info is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestLoggingIntegration:
    """Integration tests for logging with tracing."""

    def test_full_request_logging(self, tracer):
        """Simulate full request logging with trace correlation."""
        config = LogConfig(
            service_name="openmemory-api",
            level=LogLevel.DEBUG,
        )
        logger = create_logger(config)
        records = []

        with patch.object(logger, "_write", side_effect=lambda r: records.append(r)):
            with tracer.start_span("http_request") as request_span:
                logger.bind(
                    user_id="user-123",
                    org_id="org-456",
                )

                logger.info("Request started", path="/api/search")

                with tracer.start_span("database_query"):
                    logger.debug("Executing query", query_type="vector_search")
                    time.sleep(0.01)
                    logger.debug("Query completed", latency_ms=10.5)

                logger.info("Request completed", status_code=200, latency_ms=50.0)

        # Verify all logs have trace correlation
        assert len(records) == 4
        for record in records:
            assert record.trace_id == request_span.trace_id
            assert record.extra.get("user_id") == "user-123"
            assert record.extra.get("org_id") == "org-456"
