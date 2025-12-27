"""
Tests for security types: Principal, TokenClaims, Scope, and error types.

TDD: These tests define the expected behavior of the core security types.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.security.types import (
    AuthenticationError,
    AuthorizationError,
    Principal,
    Scope,
    TokenClaims,
)


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_default_message(self):
        """Default message should be 'Authentication required'."""
        err = AuthenticationError()
        assert err.message == "Authentication required"
        assert err.code == "UNAUTHENTICATED"
        assert str(err) == "Authentication required"

    def test_custom_message(self):
        """Should support custom error messages."""
        err = AuthenticationError(message="Invalid token", code="INVALID_TOKEN")
        assert err.message == "Invalid token"
        assert err.code == "INVALID_TOKEN"

    def test_is_exception(self):
        """Should be a proper Exception subclass."""
        err = AuthenticationError()
        assert isinstance(err, Exception)
        with pytest.raises(AuthenticationError):
            raise err


class TestAuthorizationError:
    """Tests for AuthorizationError exception."""

    def test_default_message(self):
        """Default message should be 'Insufficient permissions'."""
        err = AuthorizationError()
        assert err.message == "Insufficient permissions"
        assert err.code == "FORBIDDEN"
        assert str(err) == "Insufficient permissions"

    def test_custom_message(self):
        """Should support custom error messages."""
        err = AuthorizationError(message="Access denied", code="ACCESS_DENIED")
        assert err.message == "Access denied"
        assert err.code == "ACCESS_DENIED"

    def test_is_exception(self):
        """Should be a proper Exception subclass."""
        err = AuthorizationError()
        assert isinstance(err, Exception)
        with pytest.raises(AuthorizationError):
            raise err


class TestScope:
    """Tests for Scope enum."""

    def test_memory_scopes_exist(self):
        """Memory-related scopes should be defined."""
        assert Scope.MEMORIES_READ.value == "memories:read"
        assert Scope.MEMORIES_WRITE.value == "memories:write"
        assert Scope.MEMORIES_DELETE.value == "memories:delete"

    def test_app_scopes_exist(self):
        """App-related scopes should be defined."""
        assert Scope.APPS_READ.value == "apps:read"
        assert Scope.APPS_WRITE.value == "apps:write"
        assert Scope.APPS_DELETE.value == "apps:delete"

    def test_graph_scopes_exist(self):
        """Graph-related scopes should be defined."""
        assert Scope.GRAPH_READ.value == "graph:read"
        assert Scope.GRAPH_WRITE.value == "graph:write"

    def test_admin_scopes_exist(self):
        """Admin scopes should be defined."""
        assert Scope.ADMIN_READ.value == "admin:read"
        assert Scope.ADMIN_WRITE.value == "admin:write"

    def test_scope_is_string_enum(self):
        """Scope should be a string enum for easy serialization."""
        assert isinstance(Scope.MEMORIES_READ, str)
        assert Scope.MEMORIES_READ == "memories:read"


class TestTokenClaims:
    """Tests for TokenClaims dataclass."""

    @pytest.fixture
    def valid_claims(self):
        """Create valid token claims for testing."""
        now = datetime.now(timezone.utc)
        return TokenClaims(
            sub="user-123",
            iss="https://auth.example.com",
            aud="https://api.example.com",
            exp=now + timedelta(hours=1),
            iat=now,
            jti="unique-token-id",
            org_id="org-456",
            scopes={"memories:read", "memories:write", "apps:read"},
            email="user@example.com",
            name="Test User",
        )

    def test_required_fields(self, valid_claims):
        """All required JWT fields should be accessible."""
        assert valid_claims.sub == "user-123"
        assert valid_claims.iss == "https://auth.example.com"
        assert valid_claims.aud == "https://api.example.com"
        assert valid_claims.jti == "unique-token-id"
        assert valid_claims.org_id == "org-456"

    def test_optional_fields(self, valid_claims):
        """Optional fields should be accessible."""
        assert valid_claims.email == "user@example.com"
        assert valid_claims.name == "Test User"

    def test_optional_fields_default_to_none(self):
        """Optional fields should default to None."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub="user-123",
            iss="https://auth.example.com",
            aud="https://api.example.com",
            exp=now + timedelta(hours=1),
            iat=now,
            jti="token-id",
            org_id="org-456",
        )
        assert claims.email is None
        assert claims.name is None

    def test_scopes_default_to_empty_set(self):
        """Scopes should default to empty set."""
        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub="user-123",
            iss="https://auth.example.com",
            aud="https://api.example.com",
            exp=now + timedelta(hours=1),
            iat=now,
            jti="token-id",
            org_id="org-456",
        )
        assert claims.scopes == set()

    def test_has_scope_with_enum(self, valid_claims):
        """has_scope should work with Scope enum."""
        assert valid_claims.has_scope(Scope.MEMORIES_READ) is True
        assert valid_claims.has_scope(Scope.ADMIN_WRITE) is False

    def test_has_scope_with_string(self, valid_claims):
        """has_scope should work with string scope values."""
        assert valid_claims.has_scope("memories:read") is True
        assert valid_claims.has_scope("admin:write") is False

    def test_has_any_scope(self, valid_claims):
        """has_any_scope should return True if any scope matches."""
        assert valid_claims.has_any_scope({Scope.MEMORIES_READ, Scope.ADMIN_WRITE}) is True
        assert valid_claims.has_any_scope({Scope.ADMIN_READ, Scope.ADMIN_WRITE}) is False

    def test_has_all_scopes(self, valid_claims):
        """has_all_scopes should return True only if all scopes match."""
        assert valid_claims.has_all_scopes({Scope.MEMORIES_READ, Scope.APPS_READ}) is True
        assert valid_claims.has_all_scopes({Scope.MEMORIES_READ, Scope.ADMIN_WRITE}) is False


class TestPrincipal:
    """Tests for Principal dataclass."""

    @pytest.fixture
    def valid_claims(self):
        """Create valid token claims for testing."""
        now = datetime.now(timezone.utc)
        return TokenClaims(
            sub="user-123",
            iss="https://auth.example.com",
            aud="https://api.example.com",
            exp=now + timedelta(hours=1),
            iat=now,
            jti="unique-token-id",
            org_id="org-456",
            scopes={"memories:read", "memories:write"},
        )

    @pytest.fixture
    def principal(self, valid_claims):
        """Create a principal for testing."""
        return Principal(
            user_id="user-123",
            org_id="org-456",
            claims=valid_claims,
        )

    def test_core_identity_fields(self, principal):
        """Principal should have core identity fields."""
        assert principal.user_id == "user-123"
        assert principal.org_id == "org-456"

    def test_claims_attached(self, principal, valid_claims):
        """Principal should have token claims attached."""
        assert principal.claims == valid_claims

    def test_optional_db_user_id(self, principal):
        """db_user_id should be optional and default to None."""
        assert principal.db_user_id is None

    def test_db_user_id_can_be_set(self, valid_claims):
        """db_user_id can be set when known."""
        db_id = uuid4()
        principal = Principal(
            user_id="user-123",
            org_id="org-456",
            claims=valid_claims,
            db_user_id=db_id,
        )
        assert principal.db_user_id == db_id

    def test_dpop_thumbprint_optional(self, principal):
        """dpop_thumbprint should be optional."""
        assert principal.dpop_thumbprint is None

    def test_dpop_thumbprint_can_be_set(self, valid_claims):
        """dpop_thumbprint can be set for DPoP-bound tokens."""
        principal = Principal(
            user_id="user-123",
            org_id="org-456",
            claims=valid_claims,
            dpop_thumbprint="abc123thumbprint",
        )
        assert principal.dpop_thumbprint == "abc123thumbprint"

    def test_has_scope_delegates_to_claims(self, principal):
        """has_scope should delegate to claims."""
        assert principal.has_scope(Scope.MEMORIES_READ) is True
        assert principal.has_scope(Scope.ADMIN_WRITE) is False

    def test_has_any_scope_delegates_to_claims(self, principal):
        """has_any_scope should delegate to claims."""
        assert principal.has_any_scope({Scope.MEMORIES_READ, Scope.ADMIN_WRITE}) is True

    def test_has_all_scopes_delegates_to_claims(self, principal):
        """has_all_scopes should delegate to claims."""
        assert principal.has_all_scopes({Scope.MEMORIES_READ, Scope.MEMORIES_WRITE}) is True

    def test_require_scope_passes_when_granted(self, principal):
        """require_scope should not raise when scope is granted."""
        principal.require_scope(Scope.MEMORIES_READ)  # Should not raise

    def test_require_scope_raises_when_missing(self, principal):
        """require_scope should raise AuthorizationError when scope is missing."""
        with pytest.raises(AuthorizationError) as exc_info:
            principal.require_scope(Scope.ADMIN_WRITE)
        assert "admin:write" in exc_info.value.message
        assert exc_info.value.code == "INSUFFICIENT_SCOPE"

    def test_require_all_scopes_passes_when_all_granted(self, principal):
        """require_all_scopes should not raise when all scopes are granted."""
        principal.require_all_scopes({Scope.MEMORIES_READ, Scope.MEMORIES_WRITE})

    def test_require_all_scopes_raises_on_first_missing(self, principal):
        """require_all_scopes should raise on the first missing scope."""
        with pytest.raises(AuthorizationError):
            principal.require_all_scopes({Scope.MEMORIES_READ, Scope.ADMIN_WRITE})
