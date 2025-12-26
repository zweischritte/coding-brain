"""Tests for OpenTelemetry tracing with GenAI semantic conventions.

Tests cover Phase 0c requirements per implementation plan section 13:
- OpenTelemetry v1.37+ with GenAI semantic conventions
- W3C TraceContext propagation across services
- Span creation with required attributes
- Context propagation across async boundaries
- Span nesting for parent/child relationships
- Error recording with exception details
- GenAI attributes: token usage, TTFT, model_id
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from openmemory.api.observability.tracing import (
    Tracer,
    TracingContext,
    SpanKind,
    SpanStatus,
    Span,
    TracingConfig,
    TracingError,
    create_tracer,
    get_current_span,
    get_current_trace_id,
    inject_context,
    extract_context,
    W3C_TRACEPARENT_HEADER,
    W3C_TRACESTATE_HEADER,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tracing_config() -> TracingConfig:
    """Default tracing configuration."""
    return TracingConfig(
        service_name="openmemory-api",
        service_version="1.0.0",
        environment="test",
        enabled=True,
        sample_rate=1.0,  # Sample everything in tests
        otlp_endpoint=None,  # No export in unit tests
    )


@pytest.fixture
def tracer(tracing_config) -> Tracer:
    """Create a tracer instance for testing."""
    return create_tracer(tracing_config)


@pytest.fixture
def mock_span() -> MagicMock:
    """Create a mock span for testing."""
    span = MagicMock(spec=Span)
    span.trace_id = "0af7651916cd43dd8448eb211c80319c"
    span.span_id = "b7ad6b7169203331"
    span.is_recording.return_value = True
    return span


# ============================================================================
# TracingConfig Tests
# ============================================================================


class TestTracingConfig:
    """Tests for TracingConfig dataclass."""

    def test_default_values(self):
        """TracingConfig has sensible defaults."""
        config = TracingConfig(service_name="test-service")
        assert config.service_name == "test-service"
        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.otlp_endpoint is None

    def test_custom_values(self):
        """TracingConfig accepts custom values."""
        config = TracingConfig(
            service_name="custom-service",
            service_version="2.0.0",
            environment="production",
            enabled=True,
            sample_rate=0.1,
            otlp_endpoint="http://otel-collector:4317",
        )
        assert config.service_name == "custom-service"
        assert config.service_version == "2.0.0"
        assert config.environment == "production"
        assert config.sample_rate == 0.1
        assert config.otlp_endpoint == "http://otel-collector:4317"

    def test_sample_rate_validation(self):
        """TracingConfig validates sample_rate bounds."""
        with pytest.raises(ValueError, match="sample_rate"):
            TracingConfig(service_name="test", sample_rate=-0.1)
        with pytest.raises(ValueError, match="sample_rate"):
            TracingConfig(service_name="test", sample_rate=1.5)


# ============================================================================
# Tracer Creation Tests
# ============================================================================


class TestTracerCreation:
    """Tests for tracer creation and configuration."""

    def test_create_tracer_with_config(self, tracing_config):
        """create_tracer creates a properly configured tracer."""
        tracer = create_tracer(tracing_config)
        assert tracer is not None
        assert tracer.service_name == "openmemory-api"
        assert tracer.is_enabled() is True

    def test_create_tracer_disabled(self):
        """Disabled tracer creates no-op spans."""
        config = TracingConfig(service_name="test", enabled=False)
        tracer = create_tracer(config)
        assert tracer.is_enabled() is False

    def test_create_tracer_with_otlp_endpoint(self):
        """Tracer can be configured with OTLP endpoint."""
        config = TracingConfig(
            service_name="test",
            otlp_endpoint="http://localhost:4317",
        )
        tracer = create_tracer(config)
        assert tracer.otlp_endpoint == "http://localhost:4317"

    def test_tracer_singleton_per_service(self, tracing_config):
        """Same config returns same tracer instance."""
        tracer1 = create_tracer(tracing_config)
        tracer2 = create_tracer(tracing_config)
        assert tracer1 is tracer2


# ============================================================================
# Span Creation Tests
# ============================================================================


class TestSpanCreation:
    """Tests for span creation with required attributes."""

    def test_create_span_basic(self, tracer):
        """Create a basic span with name."""
        with tracer.start_span("test-operation") as span:
            assert span is not None
            assert span.name == "test-operation"
            assert span.trace_id is not None
            assert span.span_id is not None

    def test_span_has_valid_trace_id_format(self, tracer):
        """Span trace_id follows W3C format (32 hex chars)."""
        with tracer.start_span("test") as span:
            assert len(span.trace_id) == 32
            assert all(c in "0123456789abcdef" for c in span.trace_id)

    def test_span_has_valid_span_id_format(self, tracer):
        """Span span_id follows W3C format (16 hex chars)."""
        with tracer.start_span("test") as span:
            assert len(span.span_id) == 16
            assert all(c in "0123456789abcdef" for c in span.span_id)

    def test_span_kinds(self, tracer):
        """Span can be created with different SpanKinds."""
        kinds = [
            SpanKind.INTERNAL,
            SpanKind.SERVER,
            SpanKind.CLIENT,
            SpanKind.PRODUCER,
            SpanKind.CONSUMER,
        ]
        for kind in kinds:
            with tracer.start_span("test", kind=kind) as span:
                assert span.kind == kind

    def test_span_default_kind_is_internal(self, tracer):
        """Default SpanKind is INTERNAL."""
        with tracer.start_span("test") as span:
            assert span.kind == SpanKind.INTERNAL

    def test_span_with_attributes(self, tracer):
        """Span can be created with initial attributes."""
        attrs = {
            "user_id": "user-123",
            "org_id": "org-456",
            "repo_id": "repo-789",
        }
        with tracer.start_span("test", attributes=attrs) as span:
            assert span.get_attribute("user_id") == "user-123"
            assert span.get_attribute("org_id") == "org-456"
            assert span.get_attribute("repo_id") == "repo-789"


# ============================================================================
# Required Attribute Tests
# ============================================================================


class TestRequiredAttributes:
    """Tests for required span attributes per implementation plan."""

    def test_required_attributes_user_context(self, tracer):
        """Span accepts user context attributes."""
        with tracer.start_span("test") as span:
            span.set_attribute("user_id", "user-123")
            span.set_attribute("org_id", "org-456")
            span.set_attribute("enterprise_id", "ent-789")
            span.set_attribute("session_id", "sess-abc")

            assert span.get_attribute("user_id") == "user-123"
            assert span.get_attribute("org_id") == "org-456"
            assert span.get_attribute("enterprise_id") == "ent-789"
            assert span.get_attribute("session_id") == "sess-abc"

    def test_required_attributes_tool_context(self, tracer):
        """Span accepts tool context attributes."""
        with tracer.start_span("mcp_tool_call") as span:
            span.set_attribute("tool_name", "search_code_semantic")
            span.set_attribute("tool_version", "1.0.0")
            span.set_attribute("repo_id", "repo-123")

            assert span.get_attribute("tool_name") == "search_code_semantic"
            assert span.get_attribute("tool_version") == "1.0.0"

    def test_required_attributes_latency(self, tracer):
        """Span tracks latency_ms attribute."""
        with tracer.start_span("test") as span:
            time.sleep(0.01)  # 10ms
            span.set_attribute("latency_ms", 10.5)

            assert span.get_attribute("latency_ms") == 10.5

    def test_attribute_types(self, tracer):
        """Span accepts various attribute types."""
        with tracer.start_span("test") as span:
            # String
            span.set_attribute("string_attr", "value")
            # Int
            span.set_attribute("int_attr", 42)
            # Float
            span.set_attribute("float_attr", 3.14)
            # Bool
            span.set_attribute("bool_attr", True)
            # List of strings
            span.set_attribute("list_attr", ["a", "b", "c"])

            assert span.get_attribute("string_attr") == "value"
            assert span.get_attribute("int_attr") == 42
            assert span.get_attribute("float_attr") == 3.14
            assert span.get_attribute("bool_attr") is True
            assert span.get_attribute("list_attr") == ["a", "b", "c"]


# ============================================================================
# GenAI Semantic Conventions Tests
# ============================================================================


class TestGenAISemanticConventions:
    """Tests for GenAI semantic conventions (OpenTelemetry v1.37+)."""

    def test_genai_token_usage(self, tracer):
        """Span tracks GenAI token usage attributes."""
        with tracer.start_span("llm_call") as span:
            span.set_attribute("gen_ai.usage.input_tokens", 150)
            span.set_attribute("gen_ai.usage.output_tokens", 250)
            span.set_attribute("gen_ai.usage.total_tokens", 400)

            assert span.get_attribute("gen_ai.usage.input_tokens") == 150
            assert span.get_attribute("gen_ai.usage.output_tokens") == 250
            assert span.get_attribute("gen_ai.usage.total_tokens") == 400

    def test_genai_model_attributes(self, tracer):
        """Span tracks GenAI model attributes."""
        with tracer.start_span("llm_call") as span:
            span.set_attribute("gen_ai.system", "openai")
            span.set_attribute("gen_ai.request.model", "gpt-4")
            span.set_attribute("gen_ai.response.model", "gpt-4-0613")

            assert span.get_attribute("gen_ai.system") == "openai"
            assert span.get_attribute("gen_ai.request.model") == "gpt-4"
            assert span.get_attribute("gen_ai.response.model") == "gpt-4-0613"

    def test_genai_ttft(self, tracer):
        """Span tracks time-to-first-token (TTFT)."""
        with tracer.start_span("llm_call") as span:
            span.set_attribute("gen_ai.response.ttft_ms", 250.5)
            span.set_attribute("gen_ai.response.latency_ms", 1500.0)

            assert span.get_attribute("gen_ai.response.ttft_ms") == 250.5
            assert span.get_attribute("gen_ai.response.latency_ms") == 1500.0

    def test_genai_request_attributes(self, tracer):
        """Span tracks GenAI request attributes."""
        with tracer.start_span("llm_call") as span:
            span.set_attribute("gen_ai.request.temperature", 0.7)
            span.set_attribute("gen_ai.request.max_tokens", 1000)
            span.set_attribute("gen_ai.request.top_p", 0.9)

            assert span.get_attribute("gen_ai.request.temperature") == 0.7
            assert span.get_attribute("gen_ai.request.max_tokens") == 1000

    def test_genai_embedding_attributes(self, tracer):
        """Span tracks embedding-specific attributes."""
        with tracer.start_span("embedding_call") as span:
            span.set_attribute("gen_ai.system", "qwen3")
            span.set_attribute("gen_ai.request.model", "qwen3-embedding-8b")
            span.set_attribute("gen_ai.request.embedding_dimensions", 1024)
            span.set_attribute("gen_ai.usage.input_tokens", 512)

            assert span.get_attribute("gen_ai.request.embedding_dimensions") == 1024


# ============================================================================
# W3C TraceContext Propagation Tests
# ============================================================================


class TestW3CTraceContextPropagation:
    """Tests for W3C TraceContext propagation."""

    def test_traceparent_header_name(self):
        """W3C traceparent header constant is correct."""
        assert W3C_TRACEPARENT_HEADER == "traceparent"

    def test_tracestate_header_name(self):
        """W3C tracestate header constant is correct."""
        assert W3C_TRACESTATE_HEADER == "tracestate"

    def test_inject_context_to_headers(self, tracer):
        """Context can be injected into HTTP headers."""
        with tracer.start_span("test") as span:
            headers = {}
            inject_context(headers)

            assert "traceparent" in headers
            # Format: version-trace_id-span_id-flags
            parts = headers["traceparent"].split("-")
            assert len(parts) == 4
            assert parts[0] == "00"  # Version
            assert len(parts[1]) == 32  # trace_id
            assert len(parts[2]) == 16  # span_id
            assert parts[3] in ("00", "01")  # flags

    def test_extract_context_from_headers(self, tracer):
        """Context can be extracted from HTTP headers."""
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        parent_span_id = "b7ad6b7169203331"
        headers = {
            "traceparent": f"00-{trace_id}-{parent_span_id}-01",
        }

        ctx = extract_context(headers)
        assert ctx is not None
        assert ctx.trace_id == trace_id
        assert ctx.parent_span_id == parent_span_id
        assert ctx.sampled is True

    def test_extract_context_with_tracestate(self, tracer):
        """Context extraction includes tracestate."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "tracestate": "vendor1=value1,vendor2=value2",
        }

        ctx = extract_context(headers)
        assert ctx.tracestate == "vendor1=value1,vendor2=value2"

    def test_extract_context_invalid_traceparent(self, tracer):
        """Invalid traceparent returns None context."""
        headers = {"traceparent": "invalid"}
        ctx = extract_context(headers)
        assert ctx is None

    def test_extract_context_missing_header(self, tracer):
        """Missing traceparent header returns None context."""
        headers = {}
        ctx = extract_context(headers)
        assert ctx is None

    def test_context_propagation_creates_child_span(self, tracer):
        """Extracted context creates child span with same trace_id."""
        trace_id = "0af7651916cd43dd8448eb211c80319c"
        parent_span_id = "b7ad6b7169203331"
        headers = {
            "traceparent": f"00-{trace_id}-{parent_span_id}-01",
        }

        ctx = extract_context(headers)
        with tracer.start_span("child", context=ctx) as span:
            assert span.trace_id == trace_id
            assert span.parent_span_id == parent_span_id
            assert span.span_id != parent_span_id  # New span ID

    def test_inject_and_extract_roundtrip(self, tracer):
        """Context survives injection and extraction roundtrip."""
        with tracer.start_span("parent") as parent_span:
            headers = {}
            inject_context(headers)

            ctx = extract_context(headers)
            assert ctx.trace_id == parent_span.trace_id


# ============================================================================
# Span Nesting and Parent/Child Tests
# ============================================================================


class TestSpanNesting:
    """Tests for span nesting and parent/child relationships."""

    def test_nested_spans_share_trace_id(self, tracer):
        """Nested spans share the same trace_id."""
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.trace_id == parent.trace_id

    def test_nested_span_has_parent_span_id(self, tracer):
        """Child span has parent's span_id as parent_span_id."""
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id

    def test_deeply_nested_spans(self, tracer):
        """Deeply nested spans maintain correct hierarchy."""
        with tracer.start_span("grandparent") as gp:
            with tracer.start_span("parent") as p:
                with tracer.start_span("child") as c:
                    assert c.trace_id == gp.trace_id
                    assert c.parent_span_id == p.span_id
                    assert p.parent_span_id == gp.span_id

    def test_sibling_spans_different_span_ids(self, tracer):
        """Sibling spans have different span_ids but same parent."""
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child1") as child1:
                pass
            with tracer.start_span("child2") as child2:
                pass

            assert child1.span_id != child2.span_id
            assert child1.parent_span_id == parent.span_id
            assert child2.parent_span_id == parent.span_id
            assert child1.trace_id == child2.trace_id


# ============================================================================
# Async Context Propagation Tests
# ============================================================================


class TestAsyncContextPropagation:
    """Tests for context propagation across async boundaries."""

    @pytest.mark.asyncio
    async def test_async_span_creation(self, tracer):
        """Spans can be created in async context."""
        async with tracer.start_span_async("async-operation") as span:
            assert span is not None
            assert span.name == "async-operation"

    @pytest.mark.asyncio
    async def test_async_nested_spans(self, tracer):
        """Async nested spans maintain parent/child relationship."""
        async with tracer.start_span_async("parent") as parent:
            async with tracer.start_span_async("child") as child:
                assert child.trace_id == parent.trace_id
                assert child.parent_span_id == parent.span_id

    @pytest.mark.asyncio
    async def test_async_context_propagation_across_await(self, tracer):
        """Context propagates correctly across await points."""
        trace_ids = []

        async def inner_task():
            async with tracer.start_span_async("inner") as span:
                trace_ids.append(span.trace_id)

        async with tracer.start_span_async("outer") as outer:
            trace_ids.append(outer.trace_id)
            await inner_task()

        # Both spans should have same trace_id
        assert len(trace_ids) == 2
        assert trace_ids[0] == trace_ids[1]

    @pytest.mark.asyncio
    async def test_concurrent_async_spans(self, tracer):
        """Concurrent async operations maintain separate contexts."""
        results = []

        async def task(name: str):
            async with tracer.start_span_async(name) as span:
                await asyncio.sleep(0.01)
                results.append((name, span.trace_id, span.span_id))

        async with tracer.start_span_async("parent"):
            await asyncio.gather(task("task1"), task("task2"), task("task3"))

        # All tasks should have same trace_id but different span_ids
        trace_ids = [r[1] for r in results]
        span_ids = [r[2] for r in results]

        assert len(set(trace_ids)) == 1  # Same trace
        assert len(set(span_ids)) == 3  # Different spans


# ============================================================================
# Span Status and Error Recording Tests
# ============================================================================


class TestSpanStatusAndErrors:
    """Tests for span status and error recording."""

    def test_span_status_ok(self, tracer):
        """Span can be marked with OK status."""
        with tracer.start_span("test") as span:
            span.set_status(SpanStatus.OK)
            assert span.status == SpanStatus.OK

    def test_span_status_error(self, tracer):
        """Span can be marked with ERROR status."""
        with tracer.start_span("test") as span:
            span.set_status(SpanStatus.ERROR, "Something went wrong")
            assert span.status == SpanStatus.ERROR
            assert span.status_description == "Something went wrong"

    def test_span_records_exception(self, tracer):
        """Span records exception details."""
        exc = ValueError("Test error")
        with tracer.start_span("test") as span:
            span.record_exception(exc)

            events = span.get_events()
            assert len(events) == 1
            assert events[0].name == "exception"
            assert events[0].attributes["exception.type"] == "ValueError"
            assert events[0].attributes["exception.message"] == "Test error"

    def test_span_exception_includes_stacktrace(self, tracer):
        """Recorded exception includes stacktrace."""
        try:
            raise RuntimeError("Test error with trace")
        except RuntimeError as exc:
            with tracer.start_span("test") as span:
                span.record_exception(exc)

                events = span.get_events()
                assert "exception.stacktrace" in events[0].attributes
                assert "RuntimeError" in events[0].attributes["exception.stacktrace"]

    def test_span_context_manager_records_exception(self, tracer):
        """Context manager automatically records exceptions."""
        with pytest.raises(ValueError):
            with tracer.start_span("test") as span:
                raise ValueError("Auto-recorded error")

        events = span.get_events()
        assert len(events) >= 1
        assert any(e.name == "exception" for e in events)

    def test_span_exception_sets_error_status(self, tracer):
        """Exception in span sets ERROR status."""
        with pytest.raises(ValueError):
            with tracer.start_span("test") as span:
                raise ValueError("Error status test")

        assert span.status == SpanStatus.ERROR


# ============================================================================
# Span Events Tests
# ============================================================================


class TestSpanEvents:
    """Tests for span events."""

    def test_add_event_basic(self, tracer):
        """Events can be added to spans."""
        with tracer.start_span("test") as span:
            span.add_event("checkpoint")

            events = span.get_events()
            assert len(events) == 1
            assert events[0].name == "checkpoint"

    def test_add_event_with_attributes(self, tracer):
        """Events can include attributes."""
        with tracer.start_span("test") as span:
            span.add_event(
                "query_executed",
                attributes={"db.statement": "SELECT *", "db.rows_affected": 10},
            )

            events = span.get_events()
            assert events[0].attributes["db.statement"] == "SELECT *"
            assert events[0].attributes["db.rows_affected"] == 10

    def test_add_event_with_timestamp(self, tracer):
        """Events can have explicit timestamps."""
        timestamp = datetime.now(timezone.utc)
        with tracer.start_span("test") as span:
            span.add_event("timed_event", timestamp=timestamp)

            events = span.get_events()
            assert events[0].timestamp == timestamp

    def test_multiple_events(self, tracer):
        """Multiple events are recorded in order."""
        with tracer.start_span("test") as span:
            span.add_event("start")
            span.add_event("middle")
            span.add_event("end")

            events = span.get_events()
            assert len(events) == 3
            assert [e.name for e in events] == ["start", "middle", "end"]


# ============================================================================
# Current Context Tests
# ============================================================================


class TestCurrentContext:
    """Tests for accessing current span context."""

    def test_get_current_span(self, tracer):
        """get_current_span returns active span."""
        with tracer.start_span("test"):
            current = get_current_span()
            assert current is not None
            assert current.name == "test"

    def test_get_current_span_outside_context(self, tracer):
        """get_current_span returns None outside span context."""
        current = get_current_span()
        assert current is None

    def test_get_current_trace_id(self, tracer):
        """get_current_trace_id returns active trace ID."""
        with tracer.start_span("test") as span:
            trace_id = get_current_trace_id()
            assert trace_id == span.trace_id

    def test_get_current_trace_id_outside_context(self, tracer):
        """get_current_trace_id returns None outside span context."""
        trace_id = get_current_trace_id()
        assert trace_id is None

    def test_nested_get_current_span(self, tracer):
        """get_current_span returns innermost span."""
        with tracer.start_span("outer"):
            with tracer.start_span("inner"):
                current = get_current_span()
                assert current.name == "inner"


# ============================================================================
# Disabled Tracer Tests
# ============================================================================


class TestDisabledTracer:
    """Tests for disabled tracer (no-op behavior)."""

    @pytest.fixture
    def disabled_tracer(self) -> Tracer:
        """Create a disabled tracer."""
        config = TracingConfig(service_name="test", enabled=False)
        return create_tracer(config)

    def test_disabled_tracer_creates_noop_spans(self, disabled_tracer):
        """Disabled tracer creates no-op spans."""
        with disabled_tracer.start_span("test") as span:
            assert span is not None
            assert span.is_recording() is False

    def test_disabled_tracer_noop_span_accepts_attributes(self, disabled_tracer):
        """No-op span accepts but discards attributes."""
        with disabled_tracer.start_span("test") as span:
            span.set_attribute("key", "value")
            # Should not raise, attribute is just ignored
            assert span.get_attribute("key") is None

    def test_disabled_tracer_noop_span_accepts_events(self, disabled_tracer):
        """No-op span accepts but discards events."""
        with disabled_tracer.start_span("test") as span:
            span.add_event("event")
            events = span.get_events()
            assert events == []


# ============================================================================
# Sampling Tests
# ============================================================================


class TestSampling:
    """Tests for trace sampling."""

    def test_sample_rate_zero(self):
        """Sample rate 0 creates non-recording spans."""
        config = TracingConfig(service_name="test", sample_rate=0.0)
        tracer = create_tracer(config)

        with tracer.start_span("test") as span:
            assert span.is_recording() is False

    def test_sample_rate_one(self):
        """Sample rate 1.0 always records."""
        config = TracingConfig(service_name="test", sample_rate=1.0)
        tracer = create_tracer(config)

        with tracer.start_span("test") as span:
            assert span.is_recording() is True

    def test_sampled_flag_in_traceparent(self, tracer):
        """Sampled flag is correctly set in traceparent."""
        with tracer.start_span("test"):
            headers = {}
            inject_context(headers)

            parts = headers["traceparent"].split("-")
            # flags = "01" means sampled
            assert parts[3] == "01"


# ============================================================================
# TracingContext Tests
# ============================================================================


class TestTracingContext:
    """Tests for TracingContext dataclass."""

    def test_tracing_context_creation(self):
        """TracingContext can be created with required fields."""
        ctx = TracingContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            parent_span_id="b7ad6b7169203331",
            sampled=True,
        )
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.parent_span_id == "b7ad6b7169203331"
        assert ctx.sampled is True

    def test_tracing_context_with_tracestate(self):
        """TracingContext can include tracestate."""
        ctx = TracingContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            parent_span_id="b7ad6b7169203331",
            sampled=True,
            tracestate="vendor=value",
        )
        assert ctx.tracestate == "vendor=value"

    def test_tracing_context_to_traceparent(self):
        """TracingContext can generate traceparent header."""
        ctx = TracingContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            parent_span_id="b7ad6b7169203331",
            sampled=True,
        )
        traceparent = ctx.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_tracing_context_from_traceparent(self):
        """TracingContext can be parsed from traceparent header."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = TracingContext.from_traceparent(traceparent)
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.parent_span_id == "b7ad6b7169203331"
        assert ctx.sampled is True


# ============================================================================
# Decorator Tests
# ============================================================================


class TestTracingDecorators:
    """Tests for tracing decorators."""

    def test_trace_decorator(self, tracer):
        """@trace decorator creates span for function."""
        @tracer.trace("decorated_function")
        def my_function(x: int) -> int:
            current = get_current_span()
            assert current is not None
            assert current.name == "decorated_function"
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_trace_decorator_with_attributes(self, tracer):
        """@trace decorator can set initial attributes."""
        @tracer.trace("decorated", attributes={"static_attr": "value"})
        def my_function():
            current = get_current_span()
            assert current.get_attribute("static_attr") == "value"

        my_function()

    @pytest.mark.asyncio
    async def test_async_trace_decorator(self, tracer):
        """@trace decorator works with async functions."""
        @tracer.trace("async_decorated")
        async def async_function(x: int) -> int:
            current = get_current_span()
            assert current is not None
            assert current.name == "async_decorated"
            await asyncio.sleep(0.001)
            return x * 2

        result = await async_function(5)
        assert result == 10

    def test_trace_decorator_records_exception(self, tracer):
        """@trace decorator records exceptions."""
        @tracer.trace("failing_function")
        def failing_function():
            raise ValueError("Decorated failure")

        with pytest.raises(ValueError):
            failing_function()

        # Exception should have been recorded


# ============================================================================
# Integration Tests
# ============================================================================


class TestTracingIntegration:
    """Integration tests for tracing components."""

    def test_full_request_trace(self, tracer):
        """Simulate a full request trace with nested operations."""
        with tracer.start_span("http_request", kind=SpanKind.SERVER) as request_span:
            request_span.set_attribute("http.method", "POST")
            request_span.set_attribute("http.url", "/api/search")
            request_span.set_attribute("user_id", "user-123")
            request_span.set_attribute("org_id", "org-456")

            with tracer.start_span("authenticate", kind=SpanKind.INTERNAL) as auth_span:
                auth_span.set_attribute("auth.method", "jwt")
                auth_span.set_status(SpanStatus.OK)

            with tracer.start_span("search_semantic", kind=SpanKind.CLIENT) as search_span:
                search_span.set_attribute("tool_name", "search_code_semantic")
                search_span.set_attribute("latency_ms", 45.2)

                with tracer.start_span("qdrant_query", kind=SpanKind.CLIENT) as db_span:
                    db_span.set_attribute("db.system", "qdrant")
                    db_span.set_attribute("db.operation", "search")

            request_span.set_attribute("http.status_code", 200)
            request_span.set_status(SpanStatus.OK)

        # Verify trace structure
        assert search_span.trace_id == request_span.trace_id
        assert search_span.parent_span_id == request_span.span_id

    def test_cross_service_propagation(self, tracer):
        """Simulate cross-service trace propagation."""
        # Service A starts trace
        with tracer.start_span("service_a_request") as service_a_span:
            headers = {}
            inject_context(headers)

        # Service B receives trace context
        ctx = extract_context(headers)
        with tracer.start_span("service_b_request", context=ctx) as service_b_span:
            # Should continue same trace
            assert service_b_span.trace_id == service_a_span.trace_id
            assert service_b_span.parent_span_id == service_a_span.span_id
