"""
Tests for PRD-13: Scope und Access Entity Consolidation.

TDD: These tests define the expected behavior for derive_scope() function
and optional scope handling. Tests should fail until implementation is complete.

Summary:
- access_entity is the Single Source of Truth for Access Control
- scope is derived from access_entity prefix (unless explicitly provided)
- session and enterprise are special cases that allow explicit scope with different prefix
- scope remains persisted for filters, ranking, and graph indexing
"""

import pytest

from app.utils.structured_memory import (
    StructuredMemoryError,
    build_structured_memory,
    normalize_metadata_for_create,
    SCOPE_TO_ACCESS_ENTITY_PREFIX,
)


# =============================================================================
# TESTS FOR derive_scope() FUNCTION
# =============================================================================


class TestDeriveScopeFunction:
    """Tests for the new derive_scope() utility function."""

    def test_derive_scope_from_user_prefix(self):
        """access_entity='user:X' should derive scope='user'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="user:grischa", explicit_scope=None)
        assert result == "user"

    def test_derive_scope_from_team_prefix(self):
        """access_entity='team:X' should derive scope='team'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="team:cloudfactory/billing/backend", explicit_scope=None)
        assert result == "team"

    def test_derive_scope_from_project_prefix(self):
        """access_entity='project:X' should derive scope='project'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="project:cloudfactory/coding-brain", explicit_scope=None)
        assert result == "project"

    def test_derive_scope_from_org_prefix(self):
        """access_entity='org:X' should derive scope='org'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="org:cloudfactory", explicit_scope=None)
        assert result == "org"

    def test_derive_scope_defaults_to_user_when_no_access_entity(self):
        """No access_entity should default to scope='user'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity=None, explicit_scope=None)
        assert result == "user"

    def test_derive_scope_defaults_to_user_for_invalid_format(self):
        """Invalid access_entity format (no colon) should default to scope='user'."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="invalid-no-colon", explicit_scope=None)
        assert result == "user"

    def test_derive_scope_raises_for_unknown_prefix(self):
        """Unknown prefix should raise ValueError."""
        from app.utils.structured_memory import derive_scope

        with pytest.raises(ValueError) as exc_info:
            derive_scope(access_entity="client:something", explicit_scope=None)
        assert "unknown" in str(exc_info.value).lower()

    def test_derive_scope_respects_explicit_session(self):
        """Explicit scope='session' should be preserved (special case)."""
        from app.utils.structured_memory import derive_scope

        # session scope uses user: prefix but keeps session classification
        result = derive_scope(access_entity="user:grischa", explicit_scope="session")
        assert result == "session"

    def test_derive_scope_respects_explicit_enterprise(self):
        """Explicit scope='enterprise' should be preserved (special case)."""
        from app.utils.structured_memory import derive_scope

        # enterprise scope uses org: prefix but keeps enterprise classification
        result = derive_scope(access_entity="org:cloudfactory", explicit_scope="enterprise")
        assert result == "enterprise"

    def test_derive_scope_explicit_scope_overrides_derivation(self):
        """When explicit scope is provided (not session/enterprise), it should override derivation."""
        from app.utils.structured_memory import derive_scope

        # If user explicitly says scope=user with user:X, keep user
        result = derive_scope(access_entity="user:grischa", explicit_scope="user")
        assert result == "user"

    def test_derive_scope_explicit_scope_validates_consistency(self):
        """Explicit scope that mismatches access_entity should be validated."""
        from app.utils.structured_memory import derive_scope

        # This test ensures explicit scope is returned even if it might
        # later fail consistency validation. derive_scope just derives,
        # validation happens elsewhere.
        result = derive_scope(access_entity="team:cloudfactory/backend", explicit_scope="team")
        assert result == "team"


# =============================================================================
# TESTS FOR OPTIONAL SCOPE IN BUILD_STRUCTURED_MEMORY
# =============================================================================


class TestOptionalScopeInBuildStructuredMemory:
    """Tests for making scope optional in build_structured_memory."""

    def test_build_without_scope_derives_from_user_access_entity(self):
        """When scope is omitted, derive from access_entity='user:X'."""
        text, metadata = build_structured_memory(
            text="My preference",
            category="convention",
            scope=None,  # Explicitly None
            access_entity="user:grischa",
        )
        assert metadata["scope"] == "user"
        assert metadata["access_entity"] == "user:grischa"

    def test_build_without_scope_derives_from_team_access_entity(self):
        """When scope is omitted, derive from access_entity='team:X'."""
        text, metadata = build_structured_memory(
            text="Team standard",
            category="convention",
            scope=None,
            access_entity="team:cloudfactory/billing/backend",
        )
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/billing/backend"

    def test_build_without_scope_derives_from_project_access_entity(self):
        """When scope is omitted, derive from access_entity='project:X'."""
        text, metadata = build_structured_memory(
            text="API uses JWT",
            category="architecture",
            scope=None,
            access_entity="project:cloudfactory/coding-brain",
        )
        assert metadata["scope"] == "project"
        assert metadata["access_entity"] == "project:cloudfactory/coding-brain"

    def test_build_without_scope_derives_from_org_access_entity(self):
        """When scope is omitted, derive from access_entity='org:X'."""
        text, metadata = build_structured_memory(
            text="Org policy",
            category="decision",
            scope=None,
            access_entity="org:cloudfactory",
        )
        assert metadata["scope"] == "org"
        assert metadata["access_entity"] == "org:cloudfactory"

    def test_build_without_scope_and_no_access_entity_defaults_to_user(self):
        """When both scope and access_entity are omitted, default to scope='user'."""
        text, metadata = build_structured_memory(
            text="My personal note",
            category="convention",
            scope=None,
            access_entity=None,
        )
        assert metadata["scope"] == "user"
        # access_entity may be absent (personal default)

    def test_build_with_explicit_scope_and_access_entity_validates(self):
        """When both scope and access_entity are provided, validate consistency."""
        # Matching scope and prefix - should work
        text, metadata = build_structured_memory(
            text="Team convention",
            category="convention",
            scope="team",
            access_entity="team:cloudfactory/backend",
        )
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/backend"

    def test_build_with_explicit_session_scope_and_user_access_entity(self):
        """scope='session' with access_entity='user:X' is valid (special case)."""
        text, metadata = build_structured_memory(
            text="Session note",
            category="workflow",
            scope="session",
            access_entity="user:grischa",
        )
        assert metadata["scope"] == "session"
        assert metadata["access_entity"] == "user:grischa"

    def test_build_with_explicit_enterprise_scope_and_org_access_entity(self):
        """scope='enterprise' with access_entity='org:X' is valid (special case)."""
        text, metadata = build_structured_memory(
            text="Enterprise standard",
            category="security",
            scope="enterprise",
            access_entity="org:cloudfactory",
        )
        assert metadata["scope"] == "enterprise"
        assert metadata["access_entity"] == "org:cloudfactory"

    def test_build_with_mismatched_scope_and_access_entity_raises(self):
        """Mismatched scope and access_entity prefix should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Invalid combo",
                category="convention",
                scope="team",
                access_entity="user:grischa",  # Mismatch!
            )
        assert "mismatch" in str(exc_info.value).lower()


# =============================================================================
# TESTS FOR OPTIONAL SCOPE IN NORMALIZE_METADATA_FOR_CREATE
# =============================================================================


class TestOptionalScopeInNormalizeMetadata:
    """Tests for making scope optional in normalize_metadata_for_create."""

    def test_normalize_without_scope_derives_from_access_entity(self):
        """When scope is omitted in metadata, derive from access_entity."""
        metadata = normalize_metadata_for_create({
            "category": "convention",
            # No scope provided
            "access_entity": "team:cloudfactory/billing/backend",
        })
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/billing/backend"

    def test_normalize_without_scope_and_no_access_entity_defaults_user(self):
        """When both scope and access_entity are omitted, default scope='user'."""
        metadata = normalize_metadata_for_create({
            "category": "convention",
            # No scope, no access_entity
        })
        assert metadata["scope"] == "user"

    def test_normalize_with_scope_and_access_entity_validates(self):
        """When both are provided, validate consistency."""
        metadata = normalize_metadata_for_create({
            "category": "convention",
            "scope": "project",
            "access_entity": "project:cloudfactory/coding-brain",
        })
        assert metadata["scope"] == "project"
        assert metadata["access_entity"] == "project:cloudfactory/coding-brain"

    def test_normalize_with_mismatched_scope_raises(self):
        """Mismatched scope and access_entity should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            normalize_metadata_for_create({
                "category": "convention",
                "scope": "org",
                "access_entity": "project:cloudfactory/coding-brain",  # Mismatch!
            })
        assert "mismatch" in str(exc_info.value).lower()

    def test_normalize_with_session_scope_and_user_access_entity(self):
        """scope='session' with access_entity='user:X' is valid."""
        metadata = normalize_metadata_for_create({
            "category": "workflow",
            "scope": "session",
            "access_entity": "user:grischa",
        })
        assert metadata["scope"] == "session"
        assert metadata["access_entity"] == "user:grischa"

    def test_normalize_with_enterprise_scope_and_org_access_entity(self):
        """scope='enterprise' with access_entity='org:X' is valid."""
        metadata = normalize_metadata_for_create({
            "category": "security",
            "scope": "enterprise",
            "access_entity": "org:cloudfactory",
        })
        assert metadata["scope"] == "enterprise"
        assert metadata["access_entity"] == "org:cloudfactory"


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing behavior."""

    def test_existing_build_with_explicit_scope_still_works(self):
        """Existing code that provides scope should continue to work."""
        text, metadata = build_structured_memory(
            text="Team convention",
            category="convention",
            scope="team",
            access_entity="team:cloudfactory/backend",
        )
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/backend"

    def test_existing_build_with_user_scope_no_access_entity_still_works(self):
        """Existing code with scope='user' and no access_entity should work."""
        text, metadata = build_structured_memory(
            text="My preference",
            category="convention",
            scope="user",
            # No access_entity - was allowed before
        )
        assert metadata["scope"] == "user"

    def test_existing_normalize_with_explicit_scope_still_works(self):
        """Existing metadata normalization with scope should work."""
        metadata = normalize_metadata_for_create({
            "category": "decision",
            "scope": "org",
            "access_entity": "org:cloudfactory",
        })
        assert metadata["scope"] == "org"
        assert metadata["access_entity"] == "org:cloudfactory"

    def test_existing_shared_scope_still_requires_access_entity_when_explicit(self):
        """
        When scope is explicitly provided as a shared scope,
        access_entity should still be required.
        """
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Team standard",
                category="convention",
                scope="team",  # Explicit shared scope
                # No access_entity
            )
        assert "access_entity" in str(exc_info.value).lower()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_access_entity_string_treated_as_none(self):
        """Empty string access_entity should be treated like None."""
        text, metadata = build_structured_memory(
            text="Personal note",
            category="convention",
            scope=None,
            access_entity="",  # Empty string
        )
        # Should default to user scope
        assert metadata["scope"] == "user"

    def test_whitespace_only_access_entity_treated_as_none(self):
        """Whitespace-only access_entity should be treated like None."""
        text, metadata = build_structured_memory(
            text="Personal note",
            category="convention",
            scope=None,
            access_entity="   ",  # Whitespace only
        )
        assert metadata["scope"] == "user"

    def test_scope_is_persisted_in_metadata(self):
        """Derived scope should be persisted in metadata (not just access_entity)."""
        text, metadata = build_structured_memory(
            text="Team note",
            category="convention",
            scope=None,
            access_entity="team:cloudfactory/backend",
        )
        # scope MUST be in metadata for filters, ranking, graph
        assert "scope" in metadata
        assert metadata["scope"] == "team"

    def test_case_insensitive_access_entity_prefix(self):
        """Access entity prefix should be case-insensitive for derivation."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="TEAM:cloudfactory/backend", explicit_scope=None)
        assert result == "team"

    def test_access_entity_with_colons_in_value(self):
        """Access entity with colons in the value part should work."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="project:org/path:with:colons", explicit_scope=None)
        assert result == "project"

    def test_derive_scope_with_empty_explicit_scope(self):
        """Empty string explicit scope should be treated as None."""
        from app.utils.structured_memory import derive_scope

        result = derive_scope(access_entity="team:cloudfactory/backend", explicit_scope="")
        assert result == "team"
