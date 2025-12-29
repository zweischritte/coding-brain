"""
Tests for Code Intelligence REST Router.

TDD: These tests define the expected API behavior for code-intel endpoints.
All endpoints are under /api/v1/code/*

Scope requirements (from PRD):
- search_code_hybrid: search:read
- find_callers/callees: graph:read
- impact_analysis: graph:read
- explain_code: search:read + graph:read
- adr_automation: search:read + graph:read
- test_generation: search:read + graph:read
- pr_analysis: search:read + graph:read
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


# =============================================================================
# Test: Endpoints Exist (Basic smoke test)
# =============================================================================

class TestCodeRouterEndpointsExist:
    """Verify that all code-intel endpoints exist."""

    def test_code_search_endpoint_exists(self, client):
        """POST /api/v1/code/search should exist (not 404)."""
        response = client.post("/api/v1/code/search", json={"query": "test"})
        assert response.status_code != 404, "Code search endpoint should exist"

    def test_code_explain_endpoint_exists(self, client):
        """POST /api/v1/code/explain should exist (not 404)."""
        response = client.post("/api/v1/code/explain", json={"symbol_id": "test"})
        assert response.status_code != 404, "Code explain endpoint should exist"

    def test_code_callers_endpoint_exists(self, client):
        """POST /api/v1/code/callers should exist (not 404)."""
        response = client.post("/api/v1/code/callers", json={
            "repo_id": "test",
            "symbol_name": "main"
        })
        assert response.status_code != 404, "Code callers endpoint should exist"

    def test_code_callees_endpoint_exists(self, client):
        """POST /api/v1/code/callees should exist (not 404)."""
        response = client.post("/api/v1/code/callees", json={
            "repo_id": "test",
            "symbol_name": "main"
        })
        assert response.status_code != 404, "Code callees endpoint should exist"

    def test_code_impact_endpoint_exists(self, client):
        """POST /api/v1/code/impact should exist (not 404)."""
        response = client.post("/api/v1/code/impact", json={"repo_id": "test"})
        assert response.status_code != 404, "Code impact endpoint should exist"

    def test_code_adr_endpoint_exists(self, client):
        """POST /api/v1/code/adr should exist (not 404)."""
        response = client.post("/api/v1/code/adr", json={"diff": "test"})
        assert response.status_code != 404, "Code ADR endpoint should exist"

    def test_code_test_generation_endpoint_exists(self, client):
        """POST /api/v1/code/test-generation should exist (not 404)."""
        response = client.post("/api/v1/code/test-generation", json={
            "symbol_id": "test"
        })
        assert response.status_code != 404, "Code test-generation endpoint should exist"

    def test_code_pr_analysis_endpoint_exists(self, client):
        """POST /api/v1/code/pr-analysis should exist (not 404)."""
        response = client.post("/api/v1/code/pr-analysis", json={
            "repo_id": "test"
        })
        assert response.status_code != 404, "Code pr-analysis endpoint should exist"


# =============================================================================
# Test: Authentication Requirements
# =============================================================================

class TestCodeRouterAuthentication:
    """All code-intel endpoints require authentication."""

    def test_code_search_requires_auth(self, client):
        """POST /api/v1/code/search should require authentication."""
        response = client.post(
            "/api/v1/code/search",
            json={"query": "test query"}
        )
        assert response.status_code == 401

    def test_code_explain_requires_auth(self, client):
        """POST /api/v1/code/explain should require authentication."""
        response = client.post(
            "/api/v1/code/explain",
            json={"symbol_id": "test"}
        )
        assert response.status_code == 401

    def test_code_callers_requires_auth(self, client):
        """POST /api/v1/code/callers should require authentication."""
        response = client.post(
            "/api/v1/code/callers",
            json={"repo_id": "test", "symbol_name": "main"}
        )
        assert response.status_code == 401

    def test_code_callees_requires_auth(self, client):
        """POST /api/v1/code/callees should require authentication."""
        response = client.post(
            "/api/v1/code/callees",
            json={"repo_id": "test", "symbol_name": "main"}
        )
        assert response.status_code == 401

    def test_code_impact_requires_auth(self, client):
        """POST /api/v1/code/impact should require authentication."""
        response = client.post(
            "/api/v1/code/impact",
            json={"repo_id": "test"}
        )
        assert response.status_code == 401

    def test_code_adr_requires_auth(self, client):
        """POST /api/v1/code/adr should require authentication."""
        response = client.post(
            "/api/v1/code/adr",
            json={"diff": "test"}
        )
        assert response.status_code == 401

    def test_code_test_generation_requires_auth(self, client):
        """POST /api/v1/code/test-generation should require authentication."""
        response = client.post(
            "/api/v1/code/test-generation",
            json={"symbol_id": "test"}
        )
        assert response.status_code == 401

    def test_code_pr_analysis_requires_auth(self, client):
        """POST /api/v1/code/pr-analysis should require authentication."""
        response = client.post(
            "/api/v1/code/pr-analysis",
            json={"repo_id": "test"}
        )
        assert response.status_code == 401


# =============================================================================
# Test: Scope Requirements
# =============================================================================

class TestCodeRouterScopeRequirements:
    """Test that endpoints enforce correct scopes."""

    def test_code_search_requires_search_read(self, client, mock_jwt_config):
        """POST /api/v1/code/search requires search:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without search:read scope
        token = create_test_token(scopes=["memories:read"])

        response = client.post(
            "/api/v1/code/search",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_code_search_allows_search_read(self, client, mock_jwt_config):
        """POST /api/v1/code/search allows with search:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/search",
            json={"query": "test query"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error (may be 200 or 503 depending on deps)
        assert response.status_code not in [401, 403]

    def test_code_callers_requires_graph_read(self, client, mock_jwt_config):
        """POST /api/v1/code/callers requires graph:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token without graph:read scope
        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/callers",
            json={"repo_id": "test", "symbol_name": "main"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_code_callers_allows_graph_read(self, client, mock_jwt_config):
        """POST /api/v1/code/callers allows with graph:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        response = client.post(
            "/api/v1/code/callers",
            json={"repo_id": "test", "symbol_name": "main"},
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should not be auth error
        assert response.status_code not in [401, 403]

    def test_code_callees_requires_graph_read(self, client, mock_jwt_config):
        """POST /api/v1/code/callees requires graph:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/callees",
            json={"repo_id": "test", "symbol_name": "main"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_code_impact_requires_graph_read(self, client, mock_jwt_config):
        """POST /api/v1/code/impact requires graph:read scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/impact",
            json={"repo_id": "test"},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403

    def test_code_explain_requires_both_scopes(self, client, mock_jwt_config):
        """POST /api/v1/code/explain requires search:read AND graph:read."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only search:read
        token_search = create_test_token(scopes=["search:read"])
        response_search = client.post(
            "/api/v1/code/explain",
            json={"symbol_id": "test"},
            headers={"Authorization": f"Bearer {token_search}"}
        )
        assert response_search.status_code == 403

        # Token with only graph:read
        token_graph = create_test_token(scopes=["graph:read"])
        response_graph = client.post(
            "/api/v1/code/explain",
            json={"symbol_id": "test"},
            headers={"Authorization": f"Bearer {token_graph}"}
        )
        assert response_graph.status_code == 403

        # Token with both
        token_both = create_test_token(scopes=["search:read", "graph:read"])
        response_both = client.post(
            "/api/v1/code/explain",
            json={"symbol_id": "test"},
            headers={"Authorization": f"Bearer {token_both}"}
        )
        assert response_both.status_code not in [401, 403]

    def test_code_adr_requires_both_scopes(self, client, mock_jwt_config):
        """POST /api/v1/code/adr requires search:read AND graph:read."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only search:read
        token_search = create_test_token(scopes=["search:read"])
        response = client.post(
            "/api/v1/code/adr",
            json={"diff": "test diff"},
            headers={"Authorization": f"Bearer {token_search}"}
        )
        assert response.status_code == 403

        # Token with both
        token_both = create_test_token(scopes=["search:read", "graph:read"])
        response_both = client.post(
            "/api/v1/code/adr",
            json={"diff": "test diff"},
            headers={"Authorization": f"Bearer {token_both}"}
        )
        assert response_both.status_code not in [401, 403]

    def test_code_test_generation_requires_both_scopes(self, client, mock_jwt_config):
        """POST /api/v1/code/test-generation requires search:read AND graph:read."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only graph:read
        token_graph = create_test_token(scopes=["graph:read"])
        response = client.post(
            "/api/v1/code/test-generation",
            json={"symbol_id": "test"},
            headers={"Authorization": f"Bearer {token_graph}"}
        )
        assert response.status_code == 403

        # Token with both
        token_both = create_test_token(scopes=["search:read", "graph:read"])
        response_both = client.post(
            "/api/v1/code/test-generation",
            json={"symbol_id": "test"},
            headers={"Authorization": f"Bearer {token_both}"}
        )
        assert response_both.status_code not in [401, 403]

    def test_code_pr_analysis_requires_both_scopes(self, client, mock_jwt_config):
        """POST /api/v1/code/pr-analysis requires search:read AND graph:read."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token with only search:read
        token_search = create_test_token(scopes=["search:read"])
        response = client.post(
            "/api/v1/code/pr-analysis",
            json={"repo_id": "test"},
            headers={"Authorization": f"Bearer {token_search}"}
        )
        assert response.status_code == 403

        # Token with both
        token_both = create_test_token(scopes=["search:read", "graph:read"])
        response_both = client.post(
            "/api/v1/code/pr-analysis",
            json={"repo_id": "test"},
            headers={"Authorization": f"Bearer {token_both}"}
        )
        assert response_both.status_code not in [401, 403]


# =============================================================================
# Test: Input Validation
# =============================================================================

class TestCodeRouterInputValidation:
    """Test input validation for code-intel endpoints."""

    def test_code_search_requires_query(self, client, mock_jwt_config):
        """POST /api/v1/code/search requires query parameter."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/search",
            json={},  # Missing query
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_code_search_rejects_empty_query(self, client, mock_jwt_config):
        """POST /api/v1/code/search rejects empty query."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        response = client.post(
            "/api/v1/code/search",
            json={"query": ""},  # Empty query
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_code_search_validates_limit_range(self, client, mock_jwt_config):
        """POST /api/v1/code/search validates limit range (1-100)."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        # Too high
        response_high = client.post(
            "/api/v1/code/search",
            json={"query": "test", "limit": 200},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response_high.status_code == 422

        # Too low
        response_low = client.post(
            "/api/v1/code/search",
            json={"query": "test", "limit": 0},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response_low.status_code == 422

    def test_code_callers_requires_repo_id(self, client, mock_jwt_config):
        """POST /api/v1/code/callers requires repo_id."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        response = client.post(
            "/api/v1/code/callers",
            json={"symbol_name": "main"},  # Missing repo_id
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_code_callers_requires_symbol_identifier(self, client, mock_jwt_config):
        """POST /api/v1/code/callers requires symbol_id or symbol_name."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        response = client.post(
            "/api/v1/code/callers",
            json={"repo_id": "test"},  # Missing symbol identifier
            headers={"Authorization": f"Bearer {token}"}
        )
        # Should fail validation - need symbol_id or symbol_name
        assert response.status_code == 422

    def test_code_explain_requires_symbol_id(self, client, mock_jwt_config):
        """POST /api/v1/code/explain requires symbol_id."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read", "graph:read"])

        response = client.post(
            "/api/v1/code/explain",
            json={},  # Missing symbol_id
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_code_impact_validates_depth_range(self, client, mock_jwt_config):
        """POST /api/v1/code/impact validates max_depth range (1-10)."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        response = client.post(
            "/api/v1/code/impact",
            json={"repo_id": "test", "max_depth": 20},  # Too high
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422

    def test_code_impact_validates_confidence_threshold(self, client, mock_jwt_config):
        """POST /api/v1/code/impact validates confidence_threshold (0-1)."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        response = client.post(
            "/api/v1/code/impact",
            json={"repo_id": "test", "confidence_threshold": 1.5},  # > 1
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 422


# =============================================================================
# Test: Graceful Degradation
# =============================================================================

class TestCodeRouterGracefulDegradation:
    """Test graceful degradation when dependencies are unavailable."""

    def test_search_returns_degraded_when_opensearch_unavailable(
        self, client, mock_jwt_config
    ):
        """Should return 200 with degraded_mode=true when OpenSearch unavailable."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        # Mock toolkit to simulate OpenSearch unavailable
        with patch("app.routers.code.get_code_toolkit") as mock_toolkit:
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = False
            mock_toolkit.return_value = mock_instance

            response = client.post(
                "/api/v1/code/search",
                json={"query": "test query"},
                headers={"Authorization": f"Bearer {token}"}
            )

            # Should return 200 with degraded meta, not crash
            assert response.status_code == 200
            data = response.json()
            assert "meta" in data
            assert data["meta"]["degraded_mode"] is True

    def test_callers_returns_degraded_when_neo4j_unavailable(
        self, client, mock_jwt_config
    ):
        """Should return 200 with degraded_mode=true when Neo4j unavailable."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        with patch("app.routers.code.get_code_toolkit") as mock_toolkit:
            mock_instance = MagicMock()
            mock_instance.is_available.side_effect = lambda x: x != "neo4j"
            mock_toolkit.return_value = mock_instance

            response = client.post(
                "/api/v1/code/callers",
                json={"repo_id": "test", "symbol_name": "main"},
                headers={"Authorization": f"Bearer {token}"}
            )

            # Should not crash - return 200 with degraded response
            assert response.status_code in [200, 503]
            if response.status_code == 200:
                data = response.json()
                assert "meta" in data
                assert data["meta"]["degraded_mode"] is True


# =============================================================================
# Test: Response Format
# =============================================================================

class TestCodeRouterResponseFormat:
    """Test response format for code-intel endpoints."""

    def test_code_search_returns_results_and_meta(self, client, mock_jwt_config):
        """POST /api/v1/code/search returns results and meta."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["search:read"])

        with patch("app.routers.code.get_code_toolkit") as mock_toolkit:
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = True
            mock_instance.search_tool = MagicMock()
            mock_instance.search_tool.search.return_value = MagicMock(
                results=[],
                meta=MagicMock(
                    request_id="test-123",
                    degraded_mode=False,
                    missing_sources=[]
                ),
                next_cursor=None
            )
            mock_toolkit.return_value = mock_instance

            response = client.post(
                "/api/v1/code/search",
                json={"query": "test query"},
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 200:
                data = response.json()
                assert "results" in data or "hits" in data
                assert "meta" in data

    def test_code_callers_returns_nodes_and_edges(self, client, mock_jwt_config):
        """POST /api/v1/code/callers returns nodes and edges."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["graph:read"])

        with patch("app.routers.code.get_code_toolkit") as mock_toolkit:
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = True
            mock_instance.callers_tool = MagicMock()
            mock_instance.callers_tool.find.return_value = MagicMock(
                nodes=[],
                edges=[],
                meta=MagicMock(
                    request_id="test-123",
                    degraded_mode=False,
                    missing_sources=[]
                ),
                next_cursor=None
            )
            mock_toolkit.return_value = mock_instance

            response = client.post(
                "/api/v1/code/callers",
                json={"repo_id": "test", "symbol_name": "main"},
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 200:
                data = response.json()
                assert "nodes" in data
                assert "edges" in data
                assert "meta" in data
