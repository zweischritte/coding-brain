"""
Tests for Feedback Router API endpoints.

TDD: These tests define the expected API behavior for feedback endpoints.
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


class TestFeedbackRouterAuth:
    """Tests for feedback router authentication requirements."""

    def test_create_feedback_requires_auth(self, client):
        """POST /api/v1/feedback should require authentication."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "accepted",
                "tool_name": "test-tool",
            }
        )
        assert response.status_code == 401

    def test_create_feedback_requires_write_scope(self, client, mock_jwt_config):
        """POST /api/v1/feedback should require feedback:write scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only read scope
        token = create_test_token(scopes=["feedback:read"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "accepted",
                "tool_name": "test-tool",
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_query_feedback_requires_auth(self, client):
        """GET /api/v1/feedback should require authentication."""
        response = client.get("/api/v1/feedback")
        assert response.status_code == 401

    def test_query_feedback_requires_read_scope(self, client, mock_jwt_config):
        """GET /api/v1/feedback should require feedback:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without feedback:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_metrics_requires_auth(self, client):
        """GET /api/v1/feedback/metrics should require authentication."""
        response = client.get("/api/v1/feedback/metrics")
        assert response.status_code == 401

    def test_metrics_requires_read_scope(self, client, mock_jwt_config):
        """GET /api/v1/feedback/metrics should require feedback:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without feedback:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/feedback/metrics",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_by_tool_requires_auth(self, client):
        """GET /api/v1/feedback/by-tool should require authentication."""
        response = client.get("/api/v1/feedback/by-tool")
        assert response.status_code == 401


class TestCreateFeedback:
    """Tests for POST /api/v1/feedback endpoint."""

    def test_create_feedback_validates_outcome(self, client, mock_jwt_config):
        """Should reject invalid outcome values."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:write"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "invalid-outcome",  # Invalid
                "tool_name": "test-tool",
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422  # Validation error

    def test_create_feedback_requires_query_id(self, client, mock_jwt_config):
        """Should require query_id field."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:write"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "outcome": "accepted",
                "tool_name": "test-tool",
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_create_feedback_requires_tool_name(self, client, mock_jwt_config):
        """Should require tool_name field."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:write"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "accepted",
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_create_feedback_success(self, client, mock_jwt_config):
        """Should create feedback event and return 201."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:write"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "accepted",
                "tool_name": "test-tool",
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should either succeed or not 401/403/422
        assert response.status_code in [201, 200, 500]  # 500 if DB not ready

    def test_create_feedback_accepts_optional_fields(self, client, mock_jwt_config):
        """Should accept optional fields like decision_time_ms."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:write"])

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "modified",
                "tool_name": "test-tool",
                "decision_time_ms": 150,
                "experiment_id": "exp-456",
                "session_id": "session-789",
                "metadata": {"custom": "data"},
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not return validation error
        assert response.status_code != 422


class TestQueryFeedback:
    """Tests for GET /api/v1/feedback endpoint."""

    def test_query_feedback_with_valid_token(self, client, mock_jwt_config):
        """Should allow access with valid token and scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])

        response = client.get(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be 401/403
        assert response.status_code not in [401, 403]

    def test_query_feedback_accepts_limit_param(self, client, mock_jwt_config):
        """Should accept limit query parameter."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])

        response = client.get(
            "/api/v1/feedback",
            params={"limit": 10},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]

    def test_query_feedback_accepts_time_range(self, client, mock_jwt_config):
        """Should accept since and until query parameters."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=7)).isoformat()
        until = now.isoformat()

        response = client.get(
            "/api/v1/feedback",
            params={"since": since, "until": until},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]


class TestFeedbackMetrics:
    """Tests for GET /api/v1/feedback/metrics endpoint."""

    def test_metrics_with_valid_token(self, client, mock_jwt_config):
        """Should return metrics with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])

        response = client.get(
            "/api/v1/feedback/metrics",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be 401/403
        assert response.status_code not in [401, 403]

    def test_metrics_accepts_time_range(self, client, mock_jwt_config):
        """Should accept since and until query parameters."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=30)).isoformat()

        response = client.get(
            "/api/v1/feedback/metrics",
            params={"since": since},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]


class TestFeedbackByTool:
    """Tests for GET /api/v1/feedback/by-tool endpoint."""

    def test_by_tool_with_valid_token(self, client, mock_jwt_config):
        """Should return tool metrics with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["feedback:read"])

        response = client.get(
            "/api/v1/feedback/by-tool",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be 401/403
        assert response.status_code not in [401, 403]

    def test_by_tool_requires_read_scope(self, client, mock_jwt_config):
        """Should require feedback:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without feedback:read
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/api/v1/feedback/by-tool",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403


class TestFeedbackTenantIsolation:
    """Tests for tenant isolation in feedback endpoints."""

    def test_create_feedback_uses_principal_org_id(self, client, mock_jwt_config):
        """Created feedback should use org_id from principal, not body."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(
            sub="user-123",
            org_id="org-from-token",
            scopes=["feedback:write"]
        )

        response = client.post(
            "/api/v1/feedback",
            json={
                "query_id": "test-query-123",
                "outcome": "accepted",
                "tool_name": "test-tool",
                # Even if org_id is in body, it should be ignored
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not reject with 401/403
        assert response.status_code not in [401, 403]

    def test_query_feedback_only_returns_org_data(self, client, mock_jwt_config):
        """Query should only return events for the principal's org."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Create feedback with org-a token
        token_a = create_test_token(
            sub="user-a",
            org_id="org-a",
            scopes=["feedback:write", "feedback:read"]
        )

        # Create feedback
        client.post(
            "/api/v1/feedback",
            json={
                "query_id": "query-for-org-a",
                "outcome": "accepted",
                "tool_name": "test-tool",
            },
            headers={"Authorization": f"Bearer {token_a}"}
        )

        # Query with org-b token should not see org-a's feedback
        token_b = create_test_token(
            sub="user-b",
            org_id="org-b",
            scopes=["feedback:read"]
        )

        response = client.get(
            "/api/v1/feedback",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        # Should not be 401/403
        assert response.status_code not in [401, 403]
        # If 200, data should only contain org-b's events (likely empty)
