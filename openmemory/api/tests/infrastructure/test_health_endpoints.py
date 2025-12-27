"""
Tests for Phase 0.5: Infrastructure Prerequisites - API Health Endpoints.

TDD: These tests are written first and should fail until implementation is complete.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from main import app
    return TestClient(app)


class TestLivenessEndpoint:
    """Test /health/live endpoint for container liveness probes."""

    def test_live_endpoint_exists(self, client):
        """The /health/live endpoint must exist."""
        response = client.get("/health/live")
        assert response.status_code != 404, "/health/live endpoint not found"

    def test_live_returns_200(self, client):
        """The /health/live endpoint must return 200 when the app is running."""
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_live_returns_status_ok(self, client):
        """The /health/live endpoint must return status: ok."""
        response = client.get("/health/live")
        data = response.json()
        assert "status" in data, "Response must include 'status' field"
        assert data["status"] == "ok", "Liveness status must be 'ok'"


class TestReadinessEndpoint:
    """Test /health/ready endpoint for container readiness probes."""

    def test_ready_endpoint_exists(self, client):
        """The /health/ready endpoint must exist."""
        response = client.get("/health/ready")
        assert response.status_code != 404, "/health/ready endpoint not found"

    def test_ready_returns_valid_status(self, client):
        """The /health/ready endpoint must return 200 or 503."""
        response = client.get("/health/ready")
        assert response.status_code in [200, 503], (
            f"Readiness must return 200 (ready) or 503 (not ready), got {response.status_code}"
        )

    def test_ready_returns_status_field(self, client):
        """The /health/ready endpoint must include a status field."""
        response = client.get("/health/ready")
        data = response.json()
        assert "status" in data, "Response must include 'status' field"
        assert data["status"] in ["ok", "degraded", "unavailable"]


class TestDependencyHealthEndpoint:
    """Test /health/deps endpoint for dependency status."""

    def test_deps_endpoint_exists(self, client):
        """The /health/deps endpoint must exist."""
        response = client.get("/health/deps")
        assert response.status_code != 404, "/health/deps endpoint not found"

    def test_deps_returns_valid_status(self, client):
        """The /health/deps endpoint must return 200 or 503."""
        response = client.get("/health/deps")
        assert response.status_code in [200, 503]

    def test_deps_returns_dependency_list(self, client):
        """The /health/deps endpoint must return a list of dependency statuses."""
        response = client.get("/health/deps")
        data = response.json()

        assert "dependencies" in data, "Response must include 'dependencies' field"
        deps = data["dependencies"]
        assert isinstance(deps, dict), "Dependencies must be a dictionary"

    def test_deps_includes_required_services(self, client):
        """The /health/deps endpoint must include status for all required services."""
        response = client.get("/health/deps")
        data = response.json()
        deps = data.get("dependencies", {})

        required_services = ["postgres", "neo4j", "opensearch", "qdrant", "valkey"]
        for service in required_services:
            # Check if service is present (may use different casing)
            found = any(service.lower() in key.lower() for key in deps.keys())
            assert found, f"Dependency status for {service} not found in /health/deps"

    def test_deps_shows_connection_status(self, client):
        """Each dependency in /health/deps must show connection status."""
        response = client.get("/health/deps")
        data = response.json()
        deps = data.get("dependencies", {})

        for service_name, service_status in deps.items():
            assert "status" in service_status, (
                f"Dependency {service_name} must have 'status' field"
            )
            assert service_status["status"] in ["healthy", "unhealthy", "unknown"], (
                f"Dependency {service_name} has invalid status: {service_status['status']}"
            )


class TestHealthEndpointFormat:
    """Test health endpoint response format and headers."""

    def test_health_endpoints_return_json(self, client):
        """All health endpoints must return JSON content type."""
        endpoints = ["/health/live", "/health/ready", "/health/deps"]
        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code != 404:
                assert "application/json" in response.headers.get("content-type", ""), (
                    f"{endpoint} must return JSON content type"
                )

    def test_health_endpoints_fast_response(self, client):
        """Health endpoints should respond quickly (for probe timeouts)."""
        import time

        endpoints = ["/health/live", "/health/ready"]
        for endpoint in endpoints:
            start = time.time()
            response = client.get(endpoint)
            elapsed = time.time() - start

            if response.status_code != 404:
                # Health endpoints should respond within 5 seconds
                assert elapsed < 5.0, (
                    f"{endpoint} took {elapsed:.2f}s which is too slow for health probes"
                )
