"""
Tests for JWT grants parsing into TokenClaims and Principal.

TDD: These tests define the expected behavior for grants parsing.
Tests should fail until implementation is complete.

Grant format: list of access_entity strings that the user has access to.
Examples:
- "user:grischa" (always implied from sub)
- "team:cloudfactory/acme/billing/backend"
- "project:cloudfactory/acme/billing-api"
- "org:cloudfactory"
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

try:
    from jose import jwt as jose_jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False

from app.security.types import (
    AuthenticationError,
    Principal,
    TokenClaims,
    Scope,
)
from app.security.dependencies import get_current_principal
from app.security.exception_handlers import register_security_exception_handlers


# Test JWT signing key
TEST_SECRET_KEY = "test-secret-key-for-unit-tests-only-32chars!"
TEST_ALGORITHM = "HS256"
TEST_ISSUER = "https://auth.test.example.com"
TEST_AUDIENCE = "https://api.test.example.com"


@pytest.fixture(autouse=True)
def setup_jwt_env(monkeypatch):
    """Set up JWT environment variables for all tests."""
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_SECRET_KEY)
    monkeypatch.setenv("JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setenv("JWT_ISSUER", TEST_ISSUER)
    monkeypatch.setenv("JWT_AUDIENCE", TEST_AUDIENCE)
    from app.security.jwt import get_jwt_config
    get_jwt_config.cache_clear()


def create_token_with_grants(
    sub: str = "test-user-123",
    org_id: str = "test-org-456",
    scopes: list[str] = None,
    grants: list[str] = None,
    exp_delta: timedelta = timedelta(hours=1),
) -> str:
    """Create a test JWT token with grants claim."""
    if not HAS_JOSE:
        pytest.skip("python-jose not installed")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
        "exp": now + exp_delta,
        "iat": now,
        "jti": f"jti-{int(time.time() * 1000)}",
        "org_id": org_id,
        "scope": " ".join(scopes or ["memories:read"]),
    }

    if grants is not None:
        payload["grants"] = grants

    return jose_jwt.encode(payload, TEST_SECRET_KEY, algorithm=TEST_ALGORITHM)


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    register_security_exception_handlers(app)

    @app.get("/grants")
    async def grants_endpoint(principal: Principal = Depends(get_current_principal)):
        return {
            "user_id": principal.user_id,
            "org_id": principal.org_id,
            "grants": list(principal.claims.grants) if hasattr(principal.claims, 'grants') else [],
        }

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app, raise_server_exceptions=False)


class TestTokenClaimsGrants:
    """Tests for grants field in TokenClaims."""

    def test_token_claims_has_grants_field(self):
        """TokenClaims should have a grants field."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            scopes={"memories:read"},
            grants={"user:user-123", "team:cloudfactory/billing"},
        )
        assert hasattr(claims, 'grants')
        assert "user:user-123" in claims.grants
        assert "team:cloudfactory/billing" in claims.grants

    def test_token_claims_grants_defaults_empty(self):
        """TokenClaims grants should default to empty set."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
        )
        assert hasattr(claims, 'grants')
        assert claims.grants == set() or claims.grants is not None

    def test_token_claims_has_grant_method(self):
        """TokenClaims should have has_grant method."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            grants={"user:user-123", "team:cloudfactory/billing"},
        )
        assert claims.has_grant("user:user-123") is True
        assert claims.has_grant("team:cloudfactory/billing") is True
        assert claims.has_grant("team:other/team") is False

    def test_token_claims_has_any_grant_method(self):
        """TokenClaims should have has_any_grant method."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            grants={"user:user-123", "team:cloudfactory/billing"},
        )
        # Should return True if any grant matches
        assert claims.has_any_grant({"team:cloudfactory/billing", "team:other"}) is True
        assert claims.has_any_grant({"team:nope", "project:nope"}) is False


class TestPrincipalGrants:
    """Tests for grants on Principal."""

    def test_principal_has_grant_method(self):
        """Principal should have has_grant method that delegates to claims."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            grants={"user:user-123", "team:cloudfactory/billing"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="org-456",
            claims=claims,
        )
        assert principal.has_grant("user:user-123") is True
        assert principal.has_grant("team:cloudfactory/billing") is True
        assert principal.has_grant("team:other") is False

    def test_principal_has_any_grant_method(self):
        """Principal should have has_any_grant method."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            grants={"user:user-123", "org:cloudfactory"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="org-456",
            claims=claims,
        )
        assert principal.has_any_grant({"org:cloudfactory", "team:nope"}) is True
        assert principal.has_any_grant({"team:nope", "project:nope"}) is False

    def test_principal_get_allowed_access_entities(self):
        """Principal should have method to get all allowed access_entity values."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="org-456",
            grants={"user:user-123", "team:cloudfactory/billing", "org:cloudfactory"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="org-456",
            claims=claims,
        )
        allowed = principal.get_allowed_access_entities()
        assert "user:user-123" in allowed
        assert "team:cloudfactory/billing" in allowed
        assert "org:cloudfactory" in allowed


class TestJWTGrantsParsing:
    """Tests for parsing grants from JWT tokens."""

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

    def test_jwt_with_grants_parsed_to_claims(self, client):
        """JWT with grants claim should be parsed into TokenClaims.grants."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_token_with_grants(
            sub="user-abc",
            org_id="org-xyz",
            grants=["user:user-abc", "team:cloudfactory/billing/backend"],
        )

        response = client.get(
            "/grants",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "user:user-abc" in data["grants"]
        assert "team:cloudfactory/billing/backend" in data["grants"]

    def test_jwt_without_grants_defaults_to_user_grant(self, client):
        """JWT without grants claim should default to user:<sub> grant."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_token_with_grants(
            sub="user-abc",
            org_id="org-xyz",
            grants=None,  # No grants in JWT
        )

        response = client.get(
            "/grants",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        # Should have implied user grant from sub
        assert "user:user-abc" in data["grants"]

    def test_jwt_grants_as_list_parsed(self, client):
        """JWT grants as list should be parsed correctly."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_token_with_grants(
            sub="grischa",
            grants=[
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
                "project:cloudfactory/acme/billing-api",
                "org:cloudfactory",
            ],
        )

        response = client.get(
            "/grants",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        grants = data["grants"]
        assert "user:grischa" in grants
        assert "team:cloudfactory/acme/billing/backend" in grants
        assert "project:cloudfactory/acme/billing-api" in grants
        assert "org:cloudfactory" in grants

    def test_jwt_empty_grants_still_includes_user_grant(self, client):
        """JWT with empty grants should still include user:<sub>."""
        if not HAS_JOSE:
            pytest.skip("python-jose not installed")

        token = create_token_with_grants(
            sub="user-abc",
            grants=[],  # Empty grants list
        )

        response = client.get(
            "/grants",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        # Should have implied user grant even with empty grants list
        assert "user:user-abc" in data["grants"]


class TestGrantHierarchyExpansion:
    """Tests for grant hierarchy expansion.

    Hierarchy rules:
    - org grant -> includes all project/team/client under that org
    - project grant -> includes all teams under that project
    - team grant -> that team only
    - user grant -> personal memories only
    """

    def test_org_grant_expands_to_include_projects(self):
        """org: grant should expand to include projects under it."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={"user:user-123", "org:cloudfactory"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="cloudfactory",
            claims=claims,
        )

        # org:cloudfactory should allow access to projects under it
        allowed = principal.get_allowed_access_entities()

        # The principal should be able to access org-level memories
        assert "org:cloudfactory" in allowed

        # For hierarchical matching, we need a method that checks if
        # an access_entity is allowed given the grants
        assert principal.can_access("org:cloudfactory") is True
        assert principal.can_access("project:cloudfactory/acme/billing") is True
        assert principal.can_access("team:cloudfactory/acme/billing/backend") is True
        # But NOT a different org
        assert principal.can_access("org:other-org") is False
        assert principal.can_access("project:other-org/something") is False

    def test_project_grant_expands_to_include_teams(self):
        """project: grant should expand to include teams under it."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={"user:user-123", "project:cloudfactory/acme/billing"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="cloudfactory",
            claims=claims,
        )

        # Can access the project itself
        assert principal.can_access("project:cloudfactory/acme/billing") is True
        # Can access teams under the project
        assert principal.can_access("team:cloudfactory/acme/billing/backend") is True
        assert principal.can_access("team:cloudfactory/acme/billing/frontend") is True
        # Cannot access parent org
        assert principal.can_access("org:cloudfactory") is False
        # Cannot access sibling project
        assert principal.can_access("project:cloudfactory/acme/other") is False

    def test_team_grant_does_not_expand(self):
        """team: grant should NOT expand to other levels."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={"user:user-123", "team:cloudfactory/acme/billing/backend"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="cloudfactory",
            claims=claims,
        )

        # Can access the team
        assert principal.can_access("team:cloudfactory/acme/billing/backend") is True
        # Cannot access parent project
        assert principal.can_access("project:cloudfactory/acme/billing") is False
        # Cannot access sibling team
        assert principal.can_access("team:cloudfactory/acme/billing/frontend") is False
        # Cannot access org
        assert principal.can_access("org:cloudfactory") is False

    def test_user_grant_only_matches_self(self):
        """user: grant should only match that user."""
        claims = TokenClaims(
            sub="user-123",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={"user:user-123"},
        )
        principal = Principal(
            user_id="user-123",
            org_id="cloudfactory",
            claims=claims,
        )

        # Can access own user scope
        assert principal.can_access("user:user-123") is True
        # Cannot access other user
        assert principal.can_access("user:other-user") is False
        # Cannot access any team/project/org
        assert principal.can_access("team:cloudfactory/billing") is False


class TestResolveAccessEntities:
    """Tests for resolve_access_entities helper function."""

    def test_resolve_includes_user_grant(self):
        """resolve_access_entities should always include user:<sub>."""
        from app.security.access import resolve_access_entities

        claims = TokenClaims(
            sub="grischa",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={"team:cloudfactory/billing"},  # No explicit user grant
        )
        principal = Principal(
            user_id="grischa",
            org_id="cloudfactory",
            claims=claims,
        )

        resolved = resolve_access_entities(principal)
        assert "user:grischa" in resolved

    def test_resolve_includes_all_explicit_grants(self):
        """resolve_access_entities should include all explicit grants."""
        from app.security.access import resolve_access_entities

        claims = TokenClaims(
            sub="grischa",
            iss=TEST_ISSUER,
            aud=TEST_AUDIENCE,
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="cloudfactory",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
                "project:cloudfactory/acme/billing",
                "org:cloudfactory",
            },
        )
        principal = Principal(
            user_id="grischa",
            org_id="cloudfactory",
            claims=claims,
        )

        resolved = resolve_access_entities(principal)
        assert "user:grischa" in resolved
        assert "team:cloudfactory/acme/billing/backend" in resolved
        assert "project:cloudfactory/acme/billing" in resolved
        assert "org:cloudfactory" in resolved
