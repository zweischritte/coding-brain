"""
Tests for new OAuth 2.0 scopes added in Phase 5.

TDD: These tests define the expected scopes for feedback, experiments, and search.
"""

import pytest
from app.security.types import Scope


class TestFeedbackScopes:
    """Tests for feedback-related scopes."""

    def test_feedback_read_scope_exists(self):
        """FEEDBACK_READ scope should exist in the Scope enum."""
        assert hasattr(Scope, "FEEDBACK_READ")
        assert Scope.FEEDBACK_READ.value == "feedback:read"

    def test_feedback_write_scope_exists(self):
        """FEEDBACK_WRITE scope should exist in the Scope enum."""
        assert hasattr(Scope, "FEEDBACK_WRITE")
        assert Scope.FEEDBACK_WRITE.value == "feedback:write"


class TestExperimentsScopes:
    """Tests for experiments-related scopes."""

    def test_experiments_read_scope_exists(self):
        """EXPERIMENTS_READ scope should exist in the Scope enum."""
        assert hasattr(Scope, "EXPERIMENTS_READ")
        assert Scope.EXPERIMENTS_READ.value == "experiments:read"

    def test_experiments_write_scope_exists(self):
        """EXPERIMENTS_WRITE scope should exist in the Scope enum."""
        assert hasattr(Scope, "EXPERIMENTS_WRITE")
        assert Scope.EXPERIMENTS_WRITE.value == "experiments:write"


class TestSearchScopes:
    """Tests for search-related scopes."""

    def test_search_read_scope_exists(self):
        """SEARCH_READ scope should exist in the Scope enum."""
        assert hasattr(Scope, "SEARCH_READ")
        assert Scope.SEARCH_READ.value == "search:read"


class TestScopeValueFormat:
    """Tests for scope value formatting consistency."""

    def test_all_scopes_follow_resource_action_pattern(self):
        """All scopes should follow the 'resource:action' pattern."""
        for scope in Scope:
            assert ":" in scope.value, f"{scope.name} doesn't contain ':'"
            parts = scope.value.split(":")
            assert len(parts) == 2, f"{scope.name} should have exactly one ':'"
            resource, action = parts
            assert resource, f"{scope.name} has empty resource"
            assert action, f"{scope.name} has empty action"

    def test_new_scopes_count(self):
        """After adding new scopes, we should have 20 total (15 existing + 5 new)."""
        assert len(Scope) == 20


class TestScopeEnumUsage:
    """Tests for scope enum usage in principal."""

    def test_principal_can_check_feedback_read_scope(self):
        """Principal.has_scope should work with FEEDBACK_READ."""
        from app.security.types import Principal, TokenClaims
        from datetime import datetime, timezone

        claims = TokenClaims(
            sub="test-user",
            iss="https://test.com",
            aud="https://api.test.com",
            exp=datetime.now(timezone.utc),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="test-org",
            scopes={"feedback:read"},
        )

        principal = Principal(
            user_id="test-user",
            org_id="test-org",
            claims=claims,
        )

        assert principal.has_scope(Scope.FEEDBACK_READ)
        assert not principal.has_scope(Scope.FEEDBACK_WRITE)

    def test_principal_can_check_experiments_scope(self):
        """Principal.has_scope should work with EXPERIMENTS_* scopes."""
        from app.security.types import Principal, TokenClaims
        from datetime import datetime, timezone

        claims = TokenClaims(
            sub="test-user",
            iss="https://test.com",
            aud="https://api.test.com",
            exp=datetime.now(timezone.utc),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="test-org",
            scopes={"experiments:read", "experiments:write"},
        )

        principal = Principal(
            user_id="test-user",
            org_id="test-org",
            claims=claims,
        )

        assert principal.has_scope(Scope.EXPERIMENTS_READ)
        assert principal.has_scope(Scope.EXPERIMENTS_WRITE)

    def test_principal_can_check_search_read_scope(self):
        """Principal.has_scope should work with SEARCH_READ."""
        from app.security.types import Principal, TokenClaims
        from datetime import datetime, timezone

        claims = TokenClaims(
            sub="test-user",
            iss="https://test.com",
            aud="https://api.test.com",
            exp=datetime.now(timezone.utc),
            iat=datetime.now(timezone.utc),
            jti="test-jti",
            org_id="test-org",
            scopes={"search:read"},
        )

        principal = Principal(
            user_id="test-user",
            org_id="test-org",
            claims=claims,
        )

        assert principal.has_scope(Scope.SEARCH_READ)
