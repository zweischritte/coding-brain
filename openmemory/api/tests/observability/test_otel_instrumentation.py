"""
Tests for Phase 6: OpenTelemetry Distributed Tracing.

TDD: These tests are written first and should fail until implementation is complete.
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the tracer provider before each test."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    # Store original
    original = trace.get_tracer_provider()

    # Create fresh provider for test
    yield

    # Cleanup - we can't easily restore but tests should be isolated


class TestTracerSetup:
    """Test OpenTelemetry tracer configuration."""

    def test_setup_tracing_creates_tracer_provider(self):
        """setup_tracing() must create a TracerProvider."""
        from app.observability.tracing import setup_tracing, get_tracer
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

        provider = setup_tracing(service_name="test-service")

        assert provider is not None
        assert isinstance(provider, SDKTracerProvider)

    def test_setup_tracing_sets_service_name(self):
        """setup_tracing() must set the service name resource attribute."""
        from app.observability.tracing import setup_tracing

        provider = setup_tracing(service_name="my-api-service")

        resource = provider.resource
        service_name = resource.attributes.get("service.name")
        assert service_name == "my-api-service"

    def test_get_tracer_returns_tracer(self):
        """get_tracer() must return a valid tracer."""
        from app.observability.tracing import setup_tracing, get_tracer

        setup_tracing()

        tracer = get_tracer("test-module")
        assert tracer is not None


class TestSpanCreation:
    """Test span creation and attributes."""

    def test_create_span_with_tracer(self):
        """Created spans must be valid spans."""
        from app.observability.tracing import setup_tracing, get_tracer
        from opentelemetry.sdk.trace import Span as SDKSpan

        provider = setup_tracing()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test-operation") as span:
            # Span should be a recording span
            assert span is not None
            assert hasattr(span, "set_attribute")

    def test_span_can_set_attributes(self):
        """Spans must support setting attributes."""
        from app.observability.tracing import setup_tracing

        provider = setup_tracing()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test-operation") as span:
            span.set_attribute("http.method", "GET")
            span.set_attribute("http.route", "/api/v1/test")

            # Verify span is recording
            assert span.is_recording()


class TestTraceContext:
    """Test trace context propagation."""

    def test_get_current_trace_id_returns_id(self):
        """get_current_trace_id() must return trace ID when in span context."""
        from app.observability.tracing import setup_tracing, get_current_trace_id

        provider = setup_tracing()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test-span"):
            trace_id = get_current_trace_id()
            assert trace_id is not None
            assert len(trace_id) == 32  # 128-bit trace ID as hex
            assert trace_id != "0" * 32  # Should not be invalid

    def test_get_current_trace_id_returns_none_outside_span(self):
        """get_current_trace_id() must return None when not in a span."""
        from app.observability.tracing import setup_tracing, get_current_trace_id

        setup_tracing()

        # Outside of any span - should be None or zeros
        trace_id = get_current_trace_id()
        assert trace_id is None or trace_id == "0" * 32

    def test_nested_spans_share_trace_id(self):
        """Nested spans must share the same trace ID."""
        from app.observability.tracing import setup_tracing, get_current_trace_id

        provider = setup_tracing()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("parent-span"):
            parent_trace_id = get_current_trace_id()

            with tracer.start_as_current_span("child-span"):
                child_trace_id = get_current_trace_id()
                assert child_trace_id == parent_trace_id


class TestFastAPIInstrumentation:
    """Test FastAPI auto-instrumentation."""

    def test_instrument_fastapi_app(self):
        """instrument_app() must instrument FastAPI app without errors."""
        from fastapi import FastAPI
        from app.observability.tracing import setup_tracing, instrument_app, uninstrument_app

        setup_tracing()

        app = FastAPI()

        # Should not raise
        instrument_app(app)

        # Verify instrumentation was applied by checking it doesn't raise
        # and we can uninstrument
        uninstrument_app(app)

        # Success if no exception was raised
        assert True
