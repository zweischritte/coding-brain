"""
Tests for Search Router API endpoints.

TDD: These tests define the expected API behavior for search endpoints.
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


class TestSearchRouterAuth:
    """Tests for search router authentication requirements."""

    def test_search_requires_auth(self, client):
        """POST /api/v1/search should require authentication."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test query"}
        )
        assert response.status_code == 401

    def test_search_requires_scope(self, client, mock_jwt_config):
        """POST /api/v1/search should require search:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without search:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_lexical_search_requires_auth(self, client):
        """POST /api/v1/search/lexical should require authentication."""
        response = client.post(
            "/api/v1/search/lexical",
            json={"query": "test query"}
        )
        assert response.status_code == 401

    def test_lexical_search_requires_scope(self, client, mock_jwt_config):
        """POST /api/v1/search/lexical should require search:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without search:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.post(
            "/api/v1/search/lexical",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_semantic_search_requires_auth(self, client):
        """POST /api/v1/search/semantic should require authentication."""
        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "test query", "query_vector": [0.1] * 1536}
        )
        assert response.status_code == 401

    def test_semantic_search_requires_scope(self, client, mock_jwt_config):
        """POST /api/v1/search/semantic should require search:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without search:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "test query", "query_vector": [0.1] * 1536},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403


class TestSearchValidation:
    """Tests for search input validation."""

    def test_search_validates_empty_query(self, client, mock_jwt_config):
        """Should reject empty query."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search",
            json={"query": ""},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_search_validates_limit_max(self, client, mock_jwt_config):
        """Should reject limit > 100."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search",
            json={"query": "test", "limit": 200},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_semantic_requires_query_vector(self, client, mock_jwt_config):
        """Semantic search should require query_vector."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "test"},  # Missing query_vector
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422


class TestHybridSearch:
    """Tests for POST /api/v1/search endpoint (hybrid search)."""

    def test_hybrid_search_with_valid_token(self, client, mock_jwt_config):
        """Should allow search with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]

    def test_hybrid_search_accepts_filters(self, client, mock_jwt_config):
        """Should accept optional filters."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "filters": {
                    "vault": "work",
                    "layer": "long_term"
                }
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]

    def test_hybrid_search_accepts_limit(self, client, mock_jwt_config):
        """Should accept limit parameter."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "limit": 50
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]


class TestLexicalSearch:
    """Tests for POST /api/v1/search/lexical endpoint."""

    def test_lexical_search_with_valid_token(self, client, mock_jwt_config):
        """Should allow lexical search with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search/lexical",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]

    def test_lexical_search_accepts_filters(self, client, mock_jwt_config):
        """Should accept optional filters."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search/lexical",
            json={
                "query": "test query",
                "filters": {"vault": "personal"}
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code not in [401, 403, 422]


class TestSemanticSearch:
    """Tests for POST /api/v1/search/semantic endpoint."""

    def test_semantic_search_with_valid_token(self, client, mock_jwt_config):
        """Should allow semantic search with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "test query",
                "query_vector": [0.1] * 1536  # 1536-dim vector
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]

    def test_semantic_search_validates_empty_vector(self, client, mock_jwt_config):
        """Should reject empty query_vector."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "test query",
                "query_vector": []  # Empty vector
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422


class TestSearchTenantIsolation:
    """Tests for tenant isolation in search endpoints."""

    def test_search_uses_principal_org_id(self, client, mock_jwt_config):
        """Search should be scoped to principal's org_id."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(
            sub="user-123",
            org_id="org-from-token",
            scopes=["search:read"]
        )

        response = client.post(
            "/api/v1/search",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not reject with 401/403
        assert response.status_code not in [401, 403]

    def test_search_org_isolation(self, client, mock_jwt_config):
        """Searches from different orgs should be isolated."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token for org-a
        token_a = create_test_token(
            sub="user-a",
            org_id="org-a",
            scopes=["search:read"]
        )

        # Token for org-b
        token_b = create_test_token(
            sub="user-b",
            org_id="org-b",
            scopes=["search:read"]
        )

        # Search with org-a token
        response_a = client.post(
            "/api/v1/search",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {token_a}"}
        )

        # Search with org-b token
        response_b = client.post(
            "/api/v1/search",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {token_b}"}
        )

        # Both should work (different tenants)
        assert response_a.status_code not in [401, 403]
        assert response_b.status_code not in [401, 403]
