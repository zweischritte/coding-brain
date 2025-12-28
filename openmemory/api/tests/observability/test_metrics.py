"""
Tests for Phase 6: Prometheus Metrics.

TDD: These tests are written first and should fail until implementation is complete.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestMetricsEndpoint:
    """Test /metrics endpoint for Prometheus scraping."""

    def test_metrics_endpoint_exists(self):
        """The /metrics endpoint must exist."""
        from app.observability.metrics import create_metrics_app

        app = create_metrics_app()
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code != 404, "/metrics endpoint not found"

    def test_metrics_returns_prometheus_format(self):
        """The /metrics endpoint must return Prometheus format."""
        from app.observability.metrics import create_metrics_app

        app = create_metrics_app()
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200

        # Prometheus format uses text/plain with charset
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "text/plain; charset=utf-8" in content_type

    def test_metrics_includes_default_metrics(self):
        """The /metrics endpoint must include default Python metrics."""
        from app.observability.metrics import create_metrics_app

        app = create_metrics_app()
        client = TestClient(app)

        response = client.get("/metrics")
        content = response.text

        # Should include standard Python process metrics
        assert "python_info" in content or "process_" in content


class TestRequestMetrics:
    """Test HTTP request metrics collection."""

    def test_http_requests_total_counter_exists(self):
        """http_requests_total counter must be defined."""
        from app.observability.metrics import http_requests_total

        assert http_requests_total is not None

    def test_http_request_duration_histogram_exists(self):
        """http_request_duration_seconds histogram must be defined."""
        from app.observability.metrics import http_request_duration_seconds

        assert http_request_duration_seconds is not None

    def test_request_metrics_have_labels(self):
        """Request metrics must have method, endpoint, and status_code labels."""
        from app.observability.metrics import http_requests_total

        # Labels should be defined
        label_names = http_requests_total._labelnames
        assert "method" in label_names
        assert "endpoint" in label_names
        assert "status_code" in label_names


class TestMetricsMiddleware:
    """Test metrics collection middleware."""

    def test_middleware_records_request_count(self):
        """Middleware must increment request count."""
        from app.observability.metrics import (
            create_metrics_app, http_requests_total, MetricsMiddleware
        )
        from prometheus_client import REGISTRY

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Get initial count
        initial = http_requests_total.labels(
            method="GET", endpoint="/test", status_code="200"
        )._value.get() or 0

        # Make request
        response = client.get("/test")
        assert response.status_code == 200

        # Count should increase
        after = http_requests_total.labels(
            method="GET", endpoint="/test", status_code="200"
        )._value.get()
        assert after > initial

    def test_middleware_records_request_duration(self):
        """Middleware must record request duration."""
        from app.observability.metrics import (
            http_request_duration_seconds, MetricsMiddleware
        )

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Make request
        response = client.get("/test")
        assert response.status_code == 200

        # Duration should be recorded (sum > 0 after request)
        duration = http_request_duration_seconds.labels(
            method="GET", endpoint="/test"
        )._sum.get()
        assert duration > 0


class TestCustomMetrics:
    """Test custom business metrics."""

    def test_custom_counter_can_be_registered(self):
        """Custom counters can be registered and used."""
        from app.observability.metrics import register_counter

        counter = register_counter(
            "test_events_total",
            "Total test events",
            ["event_type"]
        )

        assert counter is not None
        counter.labels(event_type="test").inc()

        # Should be able to get the value
        value = counter.labels(event_type="test")._value.get()
        assert value >= 1
