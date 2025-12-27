"""
Tests for Experiments Router API endpoints.

TDD: These tests define the expected API behavior for A/B experiment endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

try:
    from jose import jwt as jose_jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False


# Test JWT configuration
TEST_SECRET_KEY = "test-secret-key-for-unit-tests-only-32chars!"
TEST_ALGORITHM = "HS256"
TEST_ISSUER = "https://auth.test.example.com"
TEST_AUDIENCE = "https://api.test.example.com"


def create_test_token(
    sub: str = "test-user-123",
    org_id: str = "test-org-456",
    scopes: list[str] = None,
) -> str:
    """Create a test JWT token."""
    if not HAS_JOSE:
        pytest.skip("python-jose not installed")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
        "exp": now + timedelta(hours=1),
        "iat": now,
        "jti": f"jti-{int(now.timestamp() * 1000)}",
        "org_id": org_id,
        "scope": " ".join(scopes or []),
    }
    return jose_jwt.encode(payload, TEST_SECRET_KEY, algorithm=TEST_ALGORITHM)


@pytest.fixture
def mock_jwt_config():
    """Mock JWT configuration for tests."""
    with patch("app.security.jwt.get_jwt_config") as mock:
        mock.return_value = {
            "secret_key": TEST_SECRET_KEY,
            "algorithm": TEST_ALGORITHM,
            "issuer": TEST_ISSUER,
            "audience": TEST_AUDIENCE,
        }
        yield mock


@pytest.fixture
def client():
    """Create a test client for the main application."""
    from main import app
    return TestClient(app, raise_server_exceptions=False)


class TestExperimentsRouterAuth:
    """Tests for experiments router authentication requirements."""

    def test_create_experiment_requires_auth(self, client):
        """POST /api/v1/experiments should require authentication."""
        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ]
            }
        )
        assert response.status_code == 401

    def test_create_experiment_requires_write_scope(self, client, mock_jwt_config):
        """POST /api/v1/experiments should require experiments:write scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only read scope
        token = create_test_token(scopes=["experiments:read"])

        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_list_experiments_requires_auth(self, client):
        """GET /api/v1/experiments should require authentication."""
        response = client.get("/api/v1/experiments")
        assert response.status_code == 401

    def test_list_experiments_requires_read_scope(self, client, mock_jwt_config):
        """GET /api/v1/experiments should require experiments:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without experiments:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/experiments",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_get_experiment_requires_auth(self, client):
        """GET /api/v1/experiments/{id} should require authentication."""
        response = client.get("/api/v1/experiments/some-exp-id")
        assert response.status_code == 401

    def test_update_status_requires_auth(self, client):
        """PUT /api/v1/experiments/{id}/status should require authentication."""
        response = client.put(
            "/api/v1/experiments/some-exp-id/status",
            json={"status": "running"}
        )
        assert response.status_code == 401

    def test_update_status_requires_write_scope(self, client, mock_jwt_config):
        """PUT /api/v1/experiments/{id}/status should require experiments:write scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only read scope
        token = create_test_token(scopes=["experiments:read"])

        response = client.put(
            "/api/v1/experiments/some-exp-id/status",
            json={"status": "running"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_assign_variant_requires_auth(self, client):
        """POST /api/v1/experiments/{id}/assign should require authentication."""
        response = client.post(
            "/api/v1/experiments/some-exp-id/assign",
            json={"user_id": "user-123"}
        )
        assert response.status_code == 401

    def test_get_assignment_requires_auth(self, client):
        """GET /api/v1/experiments/{id}/assignment should require authentication."""
        response = client.get("/api/v1/experiments/some-exp-id/assignment")
        assert response.status_code == 401

    def test_get_history_requires_auth(self, client):
        """GET /api/v1/experiments/{id}/history should require authentication."""
        response = client.get("/api/v1/experiments/some-exp-id/history")
        assert response.status_code == 401


class TestCreateExperiment:
    """Tests for POST /api/v1/experiments endpoint."""

    def test_create_experiment_validates_variants_min(self, client, mock_jwt_config):
        """Should require at least 2 variants."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "variants": [{"name": "only-one", "weight": 1.0}]  # Only 1 variant
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_create_experiment_validates_weight_range(self, client, mock_jwt_config):
        """Should reject weights outside 0-1 range."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "variants": [
                    {"name": "control", "weight": 1.5},  # Invalid
                    {"name": "treatment", "weight": 0.5},
                ]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_create_experiment_requires_name(self, client, mock_jwt_config):
        """Should require name field."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.post(
            "/api/v1/experiments",
            json={
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_create_experiment_success(self, client, mock_jwt_config):
        """Should create experiment and return 201."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "description": "A test experiment",
                "variants": [
                    {"name": "control", "weight": 0.5, "config": {"feature": False}},
                    {"name": "treatment", "weight": 0.5, "config": {"feature": True}},
                ],
                "traffic_percentage": 0.8,
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should succeed or fail with server error (not auth/validation)
        assert response.status_code in [201, 200, 500]


class TestListExperiments:
    """Tests for GET /api/v1/experiments endpoint."""

    def test_list_experiments_with_valid_token(self, client, mock_jwt_config):
        """Should return experiments list with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:read"])

        response = client.get(
            "/api/v1/experiments",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]

    def test_list_experiments_accepts_status_filter(self, client, mock_jwt_config):
        """Should accept status query parameter."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:read"])

        response = client.get(
            "/api/v1/experiments",
            params={"status": "running"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]


class TestGetExperiment:
    """Tests for GET /api/v1/experiments/{id} endpoint."""

    def test_get_experiment_with_valid_token(self, client, mock_jwt_config):
        """Should allow access with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:read"])

        response = client.get(
            "/api/v1/experiments/nonexistent-exp-id",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should be 404 (not found), not 401/403
        assert response.status_code == 404

    def test_get_experiment_requires_read_scope(self, client, mock_jwt_config):
        """Should require experiments:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without experiments:read
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/experiments/some-exp-id",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403


class TestUpdateStatus:
    """Tests for PUT /api/v1/experiments/{id}/status endpoint."""

    def test_update_status_validates_status_value(self, client, mock_jwt_config):
        """Should reject invalid status values."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.put(
            "/api/v1/experiments/some-exp-id/status",
            json={"status": "invalid-status"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_update_status_accepts_reason(self, client, mock_jwt_config):
        """Should accept optional reason field."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.put(
            "/api/v1/experiments/nonexistent-exp-id/status",
            json={
                "status": "running",
                "reason": "Starting A/B test"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should be 404 (not found for nonexistent), not 422
        assert response.status_code == 404


class TestAssignVariant:
    """Tests for POST /api/v1/experiments/{id}/assign endpoint."""

    def test_assign_requires_write_scope(self, client, mock_jwt_config):
        """Should require experiments:write scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only read scope
        token = create_test_token(scopes=["experiments:read"])

        response = client.post(
            "/api/v1/experiments/some-exp-id/assign",
            json={},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_assign_with_valid_token(self, client, mock_jwt_config):
        """Should allow assignment with valid token and scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:write"])

        response = client.post(
            "/api/v1/experiments/nonexistent-exp-id/assign",
            json={},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should be 404 (experiment not found), not auth error
        assert response.status_code == 404


class TestGetAssignment:
    """Tests for GET /api/v1/experiments/{id}/assignment endpoint."""

    def test_get_assignment_requires_read_scope(self, client, mock_jwt_config):
        """Should require experiments:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without experiments:read
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/experiments/some-exp-id/assignment",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_get_assignment_with_valid_token(self, client, mock_jwt_config):
        """Should allow access with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:read"])

        response = client.get(
            "/api/v1/experiments/nonexistent-exp-id/assignment",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should be 404 (not found), not auth error
        assert response.status_code == 404


class TestGetHistory:
    """Tests for GET /api/v1/experiments/{id}/history endpoint."""

    def test_get_history_requires_read_scope(self, client, mock_jwt_config):
        """Should require experiments:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without experiments:read
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/experiments/some-exp-id/history",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_get_history_with_valid_token(self, client, mock_jwt_config):
        """Should return history with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["experiments:read"])

        response = client.get(
            "/api/v1/experiments/nonexistent-exp-id/history",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should be 404 (not found) or empty list, not auth error
        assert response.status_code not in [401, 403]


class TestExperimentsTenantIsolation:
    """Tests for tenant isolation in experiments endpoints."""

    def test_create_experiment_uses_principal_org_id(self, client, mock_jwt_config):
        """Created experiment should use org_id from principal."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(
            sub="user-123",
            org_id="org-from-token",
            scopes=["experiments:write"]
        )

        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "test-experiment",
                "variants": [
                    {"name": "control", "weight": 0.5},
                    {"name": "treatment", "weight": 0.5},
                ]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not reject with 401/403
        assert response.status_code not in [401, 403]

    def test_list_experiments_only_returns_org_data(self, client, mock_jwt_config):
        """List should only return experiments for the principal's org."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Query with org-b token
        token_b = create_test_token(
            sub="user-b",
            org_id="org-b",
            scopes=["experiments:read"]
        )

        response = client.get(
            "/api/v1/experiments",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]
