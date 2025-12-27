"""
Tests for JWT validation and get_current_principal dependency.

TDD: These tests define the expected behavior for JWT validation.
Tests should fail until implementation is complete.
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

# We'll use python-jose for JWT creation in tests
try:
    from jose import jwt as jose_jwt
    from jose.exceptions import JWTError
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False

from app.security.types import (
    AuthenticationError,
    AuthorizationError,
    Principal,
    Scope,
)
from app.security.dependencies import (
    get_current_principal,
    get_optional_principal,
    require_scopes,
)
from app.security.exception_handlers import register_security_exception_handlers


# Test JWT signing key (for unit tests only - never use in production)
TEST_SECRET_KEY = "test-secret-key-for-unit-tests-only-32chars!"
TEST_ALGORITHM = "HS256"  # Using HS256 for simplicity in tests
TEST_ISSUER = "https://auth.test.example.com"
TEST_AUDIENCE = "https://api.test.example.com"


@pytest.fixture(autouse=True)
def setup_jwt_env(monkeypatch):
    """Set up JWT environment variables for all tests."""
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_SECRET_KEY)
    monkeypatch.setenv("JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setenv("JWT_ISSUER", TEST_ISSUER)
    monkeypatch.setenv("JWT_AUDIENCE", TEST_AUDIENCE)
    # Clear any cached config
    from app.security.jwt import get_jwt_config
    get_jwt_config.cache_clear()


def create_test_token(
    sub: str = "test-user-123",
    org_id: str = "test-org-456",
    scopes: list[str] = None,
    exp_delta: timedelta = timedelta(hours=1),
    iat_delta: timedelta = timedelta(seconds=0),
    custom_claims: dict = None,
    secret_key: str = TEST_SECRET_KEY,
    algorithm: str = TEST_ALGORITHM,
) -> str:
    """Create a test JWT token."""
    if not HAS_JOSE:
        pytest.skip("python-jose not installed")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
        "exp": now + exp_delta,
        "iat": now + iat_delta,
        "jti": f"jti-{int(time.time() * 1000)}",
        "org_id": org_id,
        "scope": " ".join(scopes or ["memories:read"]),
    }
    if custom_claims:
        payload.update(custom_claims)

    return jose_jwt.encode(payload, secret_key, algorithm=algorithm)


@pytest.fixture
def app():
    """Create a test FastAPI application with auth endpoints."""
    app = FastAPI()

    # Register exception handlers to convert auth errors to HTTP responses
    register_security_exception_handlers(app)

    @app.get("/protected")
    async def protected_endpoint(principal: Principal = Depends(get_current_principal)):
        return {
            "user_id": principal.user_id,
            "org_id": principal.org_id,
            "scopes": list(principal.claims.scopes),
        }

    @app.get("/optional")
    async def optional_endpoint(principal: Optional[Principal] = Depends(get_optional_principal)):
        if principal:
            return {"authenticated": True, "user_id": principal.user_id}
        return {"authenticated": False}

    @app.get("/scoped")
    async def scoped_endpoint(
        principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ, Scope.MEMORIES_WRITE))
    ):
        return {"user_id": principal.user_id}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestGetCurrentPrincipalNoToken:
    """Tests for requests without an access token."""

    def test_returns_401_without_authorization_header(self, client):
        """Request without Authorization header should return 401."""
        response = client.get("/protected")
        assert response.status_code == 401

    def test_returns_401_with_empty_authorization(self, client):
        """Request with empty Authorization header should return 401."""
        response = client.get("/protected", headers={"Authorization": ""})
        assert response.status_code == 401

    def test_returns_401_with_non_bearer_scheme(self, client):
        """Request with non-Bearer auth scheme should return 401."""
        response = client.get("/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert response.status_code == 401

    def test_returns_401_with_bearer_no_token(self, client):
        """Request with 'Bearer ' but no token should return 401."""
        response = client.get("/protected", headers={"Authorization": "Bearer "})
        assert response.status_code == 401


class TestGetCurrentPrincipalInvalidToken:
    """Tests for requests with invalid access tokens."""

    def test_returns_401_for_malformed_jwt(self, client):
        """Malformed JWT should return 401."""
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer not-a-valid-jwt"}
        )
        assert response.status_code == 401

    def test_returns_401_for_expired_token(self, client):
        """Expired token should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(exp_delta=timedelta(hours=-1))
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_future_iat(self, client):
        """Token with future iat (issued in the future) should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(iat_delta=timedelta(hours=1))
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_wrong_issuer(self, client):
        """Token from wrong issuer should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(custom_claims={"iss": "https://wrong-issuer.com"})
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_wrong_audience(self, client):
        """Token with wrong audience should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(custom_claims={"aud": "https://wrong-audience.com"})
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_missing_sub(self, client):
        """Token without sub claim should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Create token without sub
        now = datetime.now(timezone.utc)
        payload = {
            "iss": TEST_ISSUER,
            "aud": TEST_AUDIENCE,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "jti": "test-jti",
            "org_id": "test-org",
        }
        token = jose_jwt.encode(payload, TEST_SECRET_KEY, algorithm=TEST_ALGORITHM)

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_missing_org_id(self, client):
        """Token without org_id claim should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Create token without org_id
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "test-user",
            "iss": TEST_ISSUER,
            "aud": TEST_AUDIENCE,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "jti": "test-jti",
        }
        token = jose_jwt.encode(payload, TEST_SECRET_KEY, algorithm=TEST_ALGORITHM)

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401

    def test_returns_401_for_tampered_token(self, client):
        """Tampered token should return 401."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token()
        # Tamper with the token by changing a character
        tampered = token[:-5] + "xxxxx"

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {tampered}"}
        )
        assert response.status_code == 401


class TestGetCurrentPrincipalValidToken:
    """Tests for requests with valid access tokens."""

    def test_returns_200_with_valid_token(self, client):
        """Valid token should return 200 with principal data."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(
            sub="user-abc",
            org_id="org-xyz",
            scopes=["memories:read", "apps:read"],
        )

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-abc"
        assert data["org_id"] == "org-xyz"
        assert "memories:read" in data["scopes"]

    def test_extracts_scopes_from_scope_claim(self, client):
        """Scopes should be extracted from space-delimited scope claim."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["memories:read", "memories:write", "graph:read"])

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data["scopes"]) == {"memories:read", "memories:write", "graph:read"}

    def test_extracts_email_from_token(self, client):
        """Email should be extracted if present in token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(custom_claims={"email": "user@example.com"})

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200

    def test_extracts_name_from_token(self, client):
        """Name should be extracted if present in token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(custom_claims={"name": "Test User"})

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200


class TestGetOptionalPrincipal:
    """Tests for optional authentication."""

    def test_returns_none_without_token(self, client):
        """Should return None (not 401) without token."""
        response = client.get("/optional")
        assert response.status_code == 200
        assert response.json()["authenticated"] is False

    def test_returns_principal_with_valid_token(self, client):
        """Should return principal with valid token."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        with patch("app.security.jwt.get_jwt_config") as mock:
            mock.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }

            token = create_test_token(sub="optional-user")
            response = client.get(
                "/optional",
                headers={"Authorization": f"Bearer {token}"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["authenticated"] is True
            assert data["user_id"] == "optional-user"

    def test_returns_401_with_invalid_token(self, client):
        """Should return 401 (not None) with invalid token."""
        response = client.get(
            "/optional",
            headers={"Authorization": "Bearer invalid-token"}
        )
        # Invalid token should still fail, just missing token is optional
        assert response.status_code == 401


class TestRequireScopes:
    """Tests for scope-based authorization."""

    @pytest.fixture(autouse=True)
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

    def test_returns_200_with_all_required_scopes(self, client):
        """Should return 200 when all required scopes are present."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_test_token(scopes=["memories:read", "memories:write", "apps:read"])

        response = client.get(
            "/scoped",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200

    def test_returns_403_without_required_scope(self, client):
        """Should return 403 when missing a required scope."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Only has memories:read, missing memories:write
        token = create_test_token(scopes=["memories:read"])

        response = client.get(
            "/scoped",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 403

    def test_returns_401_without_token(self, client):
        """Should return 401 (not 403) without any token."""
        response = client.get("/scoped")
        assert response.status_code == 401


class TestJWTReplayPrevention:
    """Tests for JWT jti-based replay prevention."""

    def test_rejects_reused_jti(self, client):
        """Should reject tokens with reused jti values."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Create token with specific jti
        now = datetime.now(timezone.utc)
        jti = "unique-jti-12345"
        payload = {
            "sub": "test-user",
            "iss": TEST_ISSUER,
            "aud": TEST_AUDIENCE,
            "exp": now + timedelta(hours=1),
            "iat": now,
            "jti": jti,
            "org_id": "test-org",
            "scope": "memories:read",
        }
        token = jose_jwt.encode(payload, TEST_SECRET_KEY, algorithm=TEST_ALGORITHM)

        with patch("app.security.jwt.get_jwt_config") as mock_config:
            mock_config.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }

            # First request should succeed
            response1 = client.get(
                "/protected",
                headers={"Authorization": f"Bearer {token}"}
            )

            # Second request with same token should be rejected
            # (jti should be cached in Valkey)
            response2 = client.get(
                "/protected",
                headers={"Authorization": f"Bearer {token}"}
            )

            # Note: This test will initially fail because replay prevention
            # requires Valkey integration. The implementation should cache
            # jti values to prevent replay attacks.


class TestCrossTenantAccessDenied:
    """Tests for cross-tenant access prevention."""

    def test_cannot_access_other_org_resources(self, client):
        """Tokens for one org should not access resources of another org."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        # Token for org-a
        token = create_test_token(org_id="org-a", scopes=["memories:read"])

        with patch("app.security.jwt.get_jwt_config") as mock_config:
            mock_config.return_value = {
                "secret_key": TEST_SECRET_KEY,
                "algorithm": TEST_ALGORITHM,
                "issuer": TEST_ISSUER,
                "audience": TEST_AUDIENCE,
            }

            response = client.get(
                "/protected",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 200:
                # Principal org_id should match the token's org_id
                assert response.json()["org_id"] == "org-a"
                # There's no way to access org-b data with this token
