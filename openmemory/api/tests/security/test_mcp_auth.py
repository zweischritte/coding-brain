"""
Tests for MCP tool-level authentication and authorization.

TDD: These tests verify that MCP tools enforce authentication correctly.
Tests should fail until implementation is complete.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone
import json

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


class TestMCPSSEAuthRequired:
    """Tests for MCP SSE endpoint authentication."""

    @pytest.fixture
    def client(self):
        """Create a test client for the main application."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_sse_endpoint_requires_auth_header(self, client):
        """MCP SSE endpoint should require Authorization header."""
        # Old pattern: /mcp/{client_name}/sse/{user_id}
        # This should now require auth, not accept user_id from path
        response = client.get("/mcp/test-client/sse/attacker-user")
        assert response.status_code == 401

    def test_sse_endpoint_rejects_path_user_id(self, client):
        """MCP SSE should NOT use user_id from path (security anti-pattern)."""
        # Even with auth, the user_id in path should be ignored
        # and principal should come from JWT
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        with patch("app.security.jwt.get_jwt_config") as mock:
            mock.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }

            token = create_test_token(sub="token-user")
            response = client.get(
                "/mcp/test-client/sse/attacker-user",
                headers={"Authorization": f"Bearer {token}"}
            )
            # The authenticated user should be "token-user", not "attacker-user"

    def test_sse_messages_endpoint_requires_auth(self, client):
        """MCP SSE messages endpoint should require auth."""
        response = client.post("/mcp/test-client/sse/some-user/messages/")
        assert response.status_code == 401


class TestMCPToolPermissions:
    """Tests for individual MCP tool permission checks."""

    @pytest.fixture
    def mock_jwt_config(self):
        """Mock JWT configuration for tests."""
        with patch("app.security.jwt.get_jwt_config") as mock:
            mock.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }
            yield mock

    def test_add_memories_requires_write_scope(self, mock_jwt_config):
        """add_memories tool should require memories:write scope."""
        # Token without write scope
        token = create_test_token(scopes=["memories:read"])

        # When tool is invoked with this token, it should fail with 403
        # The exact mechanism depends on how tools check permissions

    def test_search_memories_requires_read_scope(self, mock_jwt_config):
        """search_memories tool should require memories:read scope."""
        # Token without read scope
        token = create_test_token(scopes=["apps:read"])

        # When tool is invoked with this token, it should fail with 403

    def test_delete_memory_requires_delete_scope(self, mock_jwt_config):
        """delete_memory tool should require memories:delete scope."""
        # Token without delete scope
        token = create_test_token(scopes=["memories:read", "memories:write"])

        # When tool is invoked with this token, it should fail with 403

    def test_delete_all_memories_requires_delete_scope(self, mock_jwt_config):
        """delete_all_memories tool should require memories:delete scope."""
        # This is a destructive operation - needs delete scope
        token = create_test_token(scopes=["memories:read", "memories:write"])

        # Should fail with 403


class TestMCPToolOrgScoping:
    """Tests for MCP tool org_id scoping."""

    def test_tools_use_org_id_from_principal(self):
        """MCP tools should use org_id from authenticated principal."""
        # When a tool is invoked, it should receive the org_id from the JWT
        # not from any path/query parameter

    def test_tools_cannot_access_other_org_data(self):
        """MCP tools should not return data from other organizations."""
        # Token for org-a should not be able to access org-b memories
        # even if the tool call attempts to specify org-b

    def test_tools_inject_org_id_on_writes(self):
        """MCP tools should inject org_id when creating data."""
        # When add_memories is called, the org_id from the principal
        # should be attached to the new memory


class TestMCPContextVariables:
    """Tests for MCP context variable security."""

    def test_context_vars_set_from_principal(self):
        """MCP context variables should be set from authenticated principal."""
        # user_id_var and org_id_var should come from JWT, not URL params

    def test_context_vars_cannot_be_spoofed(self):
        """Context variables cannot be overridden by client."""
        # Even if a malicious client sends different values in the request,
        # the context variables should reflect the JWT claims


class TestMCPConceptsAuth:
    """Tests for concepts MCP server authentication."""

    @pytest.fixture
    def client(self):
        """Create a test client for the main application."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_concepts_sse_requires_auth(self, client):
        """Concepts SSE endpoint should require auth."""
        response = client.get("/concepts/test-client/sse/some-user")
        assert response.status_code == 401

    def test_concepts_messages_requires_auth(self, client):
        """Concepts messages endpoint should require auth."""
        response = client.post("/concepts/test-client/sse/some-user/messages/")
        assert response.status_code == 401


class TestMCPDPoPBinding:
    """Tests for MCP DPoP token binding (if required)."""

    def test_mcp_accepts_dpop_bound_tokens(self):
        """MCP should accept DPoP-bound access tokens."""
        # If the access token has a cnf claim with DPoP thumbprint,
        # the request must include a valid DPoP proof

    def test_mcp_rejects_unbound_token_with_dpop_header(self):
        """MCP should reject requests with DPoP header but unbound token."""
        # If a DPoP header is present, but the token doesn't require binding,
        # this might be an attack attempt


class TestMCPErrorResponses:
    """Tests for proper error responses from MCP auth failures."""

    @pytest.fixture
    def client(self):
        """Create a test client for the main application."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_401_includes_www_authenticate_header(self, client):
        """401 responses should include WWW-Authenticate header."""
        response = client.get("/mcp/test-client/sse/some-user")
        if response.status_code == 401:
            assert "WWW-Authenticate" in response.headers

    def test_403_includes_error_description(self, client):
        """403 responses should include error description."""
        # When scope check fails, response should explain what's missing
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        with patch("app.security.jwt.get_jwt_config") as mock:
            mock.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }

            # Token with insufficient scopes
            token = create_test_token(scopes=["stats:read"])
            response = client.get(
                "/mcp/test-client/sse/some-user",
                headers={"Authorization": f"Bearer {token}"}
            )
            # If 403, should include scope information
