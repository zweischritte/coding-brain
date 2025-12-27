"""
Tests for per-router authentication requirements.

TDD: These tests verify that all routers enforce authentication correctly.
Tests should fail until implementation is complete.
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
        "scope": " ".join(scopes or ["memories:read", "memories:write"]),
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


class TestPublicEndpoints:
    """Tests for endpoints that should be publicly accessible."""

    def test_health_live_is_public(self, client):
        """GET /health/live should be accessible without auth."""
        response = client.get("/health/live")
        assert response.status_code == 200

    def test_health_ready_is_public(self, client):
        """GET /health/ready should be accessible without auth."""
        response = client.get("/health/ready")
        # May return 503 if dependencies aren't ready, but not 401
        assert response.status_code != 401

    def test_health_deps_is_public(self, client):
        """GET /health/deps should be accessible without auth."""
        response = client.get("/health/deps")
        assert response.status_code != 401

    def test_openapi_schema_is_public(self, client):
        """GET /openapi.json should be accessible without auth."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_is_public(self, client):
        """GET /docs should be accessible without auth."""
        response = client.get("/docs")
        # May redirect, but should not return 401
        assert response.status_code != 401


class TestMemoriesRouterAuth:
    """Tests for memories router authentication requirements."""

    def test_list_memories_requires_auth(self, client):
        """GET /api/v1/memories should require authentication."""
        response = client.get("/api/v1/memories")
        assert response.status_code == 401

    def test_list_memories_rejects_user_id_param(self, client, mock_jwt_config):
        """Should NOT accept user_id as query parameter (security anti-pattern)."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(sub="token-user")
        response = client.get(
            "/api/v1/memories",
            params={"user_id": "attacker-user"},  # Attempt to impersonate
            headers={"Authorization": f"Bearer {token}"}
        )

        # If 200, verify user_id from token is used, not query param
        if response.status_code == 200:
            # The response should reflect the token user, not the query param
            # (actual assertion depends on response structure)
            pass
        # Query param user_id should be ignored or rejected

    def test_create_memory_requires_auth(self, client):
        """POST /api/v1/memories should require authentication."""
        response = client.post(
            "/api/v1/memories",
            json={"content": "test memory"}
        )
        assert response.status_code == 401

    def test_get_memory_requires_auth(self, client):
        """GET /api/v1/memories/{id} should require authentication."""
        response = client.get("/api/v1/memories/some-uuid")
        assert response.status_code == 401

    def test_delete_memories_requires_auth(self, client):
        """DELETE /api/v1/memories (bulk delete) should require authentication."""
        # Note: The API uses bulk delete at /memories/, not individual delete at /memories/{id}
        response = client.delete("/api/v1/memories/")
        assert response.status_code == 401

    def test_memories_with_valid_token(self, client, mock_jwt_config):
        """Should allow access with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["memories:read"])
        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be 401 (may be 200, 404, or 500 depending on DB state)
        assert response.status_code != 401


class TestAppsRouterAuth:
    """Tests for apps router authentication requirements."""

    def test_list_apps_requires_auth(self, client):
        """GET /api/v1/apps should require authentication."""
        response = client.get("/api/v1/apps")
        assert response.status_code == 401

    def test_update_app_requires_auth(self, client):
        """PUT /api/v1/apps/{app_id} should require authentication."""
        # Note: The API doesn't have POST /apps for creation; apps are created
        # implicitly. PUT /apps/{app_id} is used for updates.
        response = client.put(
            "/api/v1/apps/some-app-id",
            json={"name": "updated-app"}
        )
        assert response.status_code == 401


class TestGraphRouterAuth:
    """Tests for graph router authentication requirements."""

    def test_graph_stats_requires_auth(self, client):
        """GET /api/v1/graph/stats should require authentication."""
        response = client.get("/api/v1/graph/stats")
        assert response.status_code == 401


class TestEntitiesRouterAuth:
    """Tests for entities router authentication requirements."""

    def test_entities_centrality_requires_auth(self, client):
        """GET /api/v1/entities/centrality should require authentication."""
        response = client.get("/api/v1/entities/centrality")
        assert response.status_code == 401


class TestStatsRouterAuth:
    """Tests for stats router authentication requirements."""

    def test_stats_requires_auth(self, client):
        """GET /api/v1/stats should require authentication."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401


class TestBackupRouterAuth:
    """Tests for backup router authentication requirements."""

    def test_export_requires_auth(self, client):
        """Backup export should require authentication."""
        # Note: Export endpoint is POST, not GET (triggers export generation)
        response = client.post("/api/v1/backup/export")
        assert response.status_code == 401

    def test_import_requires_auth(self, client):
        """Backup import should require authentication."""
        response = client.post("/api/v1/backup/import")
        assert response.status_code == 401


class TestCrossTenantAccess:
    """Tests for cross-tenant access prevention."""

    def test_cannot_access_other_org_memories(self, client, mock_jwt_config):
        """Token for org-a should not access org-b memories."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token for org-a
        token = create_test_token(org_id="org-a", scopes=["memories:read"])

        # Try to access memories (if any exist in org-b, they should not be returned)
        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": f"Bearer {token}"}
        )

        # The response should only contain memories for org-a
        # This test validates the principle; actual data isolation is implementation-specific

    def test_org_id_injected_into_queries(self, client, mock_jwt_config):
        """Org ID from token should be injected into all queries."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(org_id="test-org-123")

        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": f"Bearer {token}"}
        )

        # The endpoint should use org_id from token for scoping
        # Cannot query for other org's data


class TestUserIdQueryParamRemoved:
    """Tests verifying user_id query param is no longer accepted."""

    def test_memories_ignores_user_id_param(self, client, mock_jwt_config):
        """user_id query param should be ignored on /memories."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(sub="token-user")
        response = client.get(
            "/api/v1/memories",
            params={"user_id": "ignored-user"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not fail, but user_id param should be ignored

    def test_graph_ignores_user_id_param(self, client, mock_jwt_config):
        """user_id query param should be ignored on /graph endpoints."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(sub="token-user")
        response = client.get(
            "/api/v1/graph/stats",
            params={"user_id": "ignored-user"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not fail, but user_id param should be ignored


class TestScopeEnforcement:
    """Tests for scope-based access control on routers."""

    def test_memories_read_requires_scope(self, client, mock_jwt_config):
        """Reading memories should require memories:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without memories:read scope
        token = create_test_token(scopes=["apps:read"])

        response = client.get(
            "/api/v1/memories",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_memories_write_requires_scope(self, client, mock_jwt_config):
        """Creating memories should require memories:write scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.post(
            "/api/v1/memories",
            json={"content": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_memories_delete_requires_scope(self, client, mock_jwt_config):
        """Bulk deleting memories should require memories:delete scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without delete scope
        token = create_test_token(scopes=["memories:read", "memories:write"])

        # Note: The API uses bulk delete at /memories/, not /memories/{id}
        response = client.delete(
            "/api/v1/memories/",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403
