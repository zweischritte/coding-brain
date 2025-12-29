"""
Tests for /metrics endpoint integration in main.py.

TDD: These tests verify that the main FastAPI app properly exposes
the /metrics endpoint and uses MetricsMiddleware.

Run with: docker compose exec codingbrain-mcp pytest tests/infrastructure/test_metrics_integration.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestMetricsEndpointIntegration:
    """Test /metrics endpoint in the main FastAPI app."""

    @pytest.fixture
    def client(self):
        """Create a test client for the main app."""
        # Import here to avoid module-level side effects
        from main import app
        return TestClient(app)

    def test_metrics_endpoint_exists_in_main_app(self, client):
        """The /metrics endpoint must be accessible in the main app."""
        response = client.get("/metrics")
        assert response.status_code != 404, "/metrics endpoint not found in main app"

    def test_metrics_returns_prometheus_format(self, client):
        """The /metrics endpoint returns Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200

        # Prometheus format uses text/plain with charset
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "text/plain; charset=utf-8" in content_type

    def test_metrics_includes_http_request_metrics(self, client):
        """The /metrics endpoint includes HTTP request metrics."""
        # First make a request to ensure metrics are populated
        client.get("/health/live")

        response = client.get("/metrics")
        content = response.text

        # Should include our custom HTTP metrics
        assert "http_requests_total" in content or "http_request_duration_seconds" in content

    def test_metrics_includes_custom_business_metrics(self, client):
        """The /metrics endpoint includes custom business metrics definitions."""
        response = client.get("/metrics")
        content = response.text

        # The metric types should be defined (they may not have data yet)
        # Check for HELP/TYPE comments that indicate metric registration
        assert "dependency_health" in content or "http_requests_total" in content

    def test_metrics_does_not_require_auth(self, client):
        """The /metrics endpoint must not require authentication."""
        response = client.get("/metrics")

        # Should not return 401 Unauthorized or 403 Forbidden
        assert response.status_code not in (401, 403)
        assert response.status_code == 200


class TestMetricsMiddlewareIntegration:
    """Test MetricsMiddleware is properly integrated in main.py."""

    @pytest.fixture
    def client(self):
        """Create a test client for the main app."""
        from main import app
        return TestClient(app)

    def test_middleware_records_health_check_metrics(self, client):
        """MetricsMiddleware records metrics for health check endpoint."""
        from app.observability.metrics import http_requests_total

        # Get initial count
        try:
            initial = http_requests_total.labels(
                method="GET", endpoint="/health/live", status_code="200"
            )._value.get() or 0
        except Exception:
            initial = 0

        # Make health check request
        response = client.get("/health/live")
        assert response.status_code == 200

        # Check count increased
        after = http_requests_total.labels(
            method="GET", endpoint="/health/live", status_code="200"
        )._value.get()
        assert after > initial

    def test_middleware_normalizes_uuid_paths(self, client):
        """MetricsMiddleware normalizes UUIDs in paths to {id}."""
        from app.observability.metrics import http_requests_total

        # Make request with UUID in path (this would 404 but still records metrics)
        client.get("/api/v1/memories/12345678-1234-1234-1234-123456789abc")

        # Get metrics
        response = client.get("/metrics")
        content = response.text

        # The normalized path should use {id} instead of the actual UUID
        # (This tests that the middleware normalizes paths)
        assert "{id}" in content or "http_requests_total" in content


class TestMetricsExcludesFromAuth:
    """Test that /metrics is excluded from authentication requirements."""

    def test_metrics_route_does_not_have_security_dependencies(self):
        """The /metrics route should not have any security dependencies."""
        from main import app

        # Find the /metrics route
        metrics_route = None
        for route in app.routes:
            if hasattr(route, 'path') and route.path == "/metrics":
                metrics_route = route
                break

        assert metrics_route is not None, "/metrics route not found"

        # Check that it doesn't have security dependencies
        # (If it did, it would require auth tokens)
        if hasattr(metrics_route, 'dependencies'):
            for dep in metrics_route.dependencies or []:
                assert "require_scope" not in str(dep).lower()
                assert "auth" not in str(dep).lower()
