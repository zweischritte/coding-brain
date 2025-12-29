"""
Tests for access_entity validation in structured memory.

TDD: These tests define the expected behavior for access_entity validation.
Tests should fail until implementation is complete.

Rules:
- scope=user: access_entity is optional (defaults to user:<sub>)
- scope!=user (team, project, org, enterprise): access_entity is REQUIRED
- access_entity format must match: user:<name>, team:<path>, project:<path>, org:<name>
"""

import pytest

from app.utils.structured_memory import (
    StructuredMemoryError,
    build_structured_memory,
    normalize_metadata_for_create,
    validate_access_entity,
    validate_update_fields,
)


class TestAccessEntityValidation:
    """Tests for access_entity field validation."""

    def test_validate_access_entity_user_format(self):
        """access_entity with user: prefix should be valid."""
        result = validate_access_entity("user:grischa")
        assert result == "user:grischa"

    def test_validate_access_entity_team_format(self):
        """access_entity with team: prefix should be valid."""
        result = validate_access_entity("team:cloudfactory/acme/billing/backend")
        assert result == "team:cloudfactory/acme/billing/backend"

    def test_validate_access_entity_project_format(self):
        """access_entity with project: prefix should be valid."""
        result = validate_access_entity("project:cloudfactory/acme/billing-api")
        assert result == "project:cloudfactory/acme/billing-api"

    def test_validate_access_entity_org_format(self):
        """access_entity with org: prefix should be valid."""
        result = validate_access_entity("org:cloudfactory")
        assert result == "org:cloudfactory"

    def test_validate_access_entity_client_prefix_rejected(self):
        """access_entity with client: prefix should be rejected."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("client:cloudfactory/acme")

    def test_validate_access_entity_service_prefix_rejected(self):
        """access_entity with service: prefix should be rejected."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("service:cloudfactory/auth-gateway")

    def test_validate_access_entity_invalid_prefix(self):
        """access_entity with invalid prefix should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_access_entity("invalid:something")
        assert "access_entity" in str(exc_info.value).lower()
        assert "prefix" in str(exc_info.value).lower() or "format" in str(exc_info.value).lower()

    def test_validate_access_entity_empty_value(self):
        """access_entity with empty value after prefix should raise error."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("user:")

    def test_validate_access_entity_no_prefix(self):
        """access_entity without prefix should raise error."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("just-a-string")

    def test_validate_access_entity_empty_string(self):
        """Empty access_entity should raise error."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("")

    def test_validate_access_entity_whitespace_only(self):
        """Whitespace-only access_entity should raise error."""
        with pytest.raises(StructuredMemoryError):
            validate_access_entity("   ")


class TestAccessEntityRequiredForSharedScopes:
    """Tests for access_entity requirement based on scope."""

    def test_scope_user_does_not_require_access_entity(self):
        """scope=user should NOT require access_entity."""
        # Should not raise - access_entity optional for personal scope
        text, metadata = build_structured_memory(
            text="My personal preference",
            category="convention",
            scope="user",
            # No access_entity provided
        )
        assert metadata["scope"] == "user"
        # access_entity may be absent or defaulted

    def test_scope_team_requires_access_entity(self):
        """scope=team MUST require access_entity."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Team convention",
                category="convention",
                scope="team",
                # No access_entity - should fail
            )
        assert "access_entity" in str(exc_info.value).lower()

    def test_scope_project_requires_access_entity(self):
        """scope=project MUST require access_entity."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Project architecture",
                category="architecture",
                scope="project",
                # No access_entity - should fail
            )
        assert "access_entity" in str(exc_info.value).lower()

    def test_scope_org_requires_access_entity(self):
        """scope=org MUST require access_entity."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Org-wide policy",
                category="decision",
                scope="org",
                # No access_entity - should fail
            )
        assert "access_entity" in str(exc_info.value).lower()

    def test_scope_enterprise_requires_access_entity(self):
        """scope=enterprise MUST require access_entity."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Enterprise standard",
                category="security",
                scope="enterprise",
                # No access_entity - should fail
            )
        assert "access_entity" in str(exc_info.value).lower()

    def test_scope_session_does_not_require_access_entity(self):
        """scope=session should NOT require access_entity (like user)."""
        # Session scope is personal/ephemeral - no access_entity required
        text, metadata = build_structured_memory(
            text="Session note",
            category="workflow",
            scope="session",
            # No access_entity provided
        )
        assert metadata["scope"] == "session"


class TestAccessEntityWithBuildStructuredMemory:
    """Tests for access_entity in build_structured_memory."""

    def test_build_with_valid_access_entity_team(self):
        """build_structured_memory should accept valid team access_entity."""
        text, metadata = build_structured_memory(
            text="Team standard",
            category="convention",
            scope="team",
            access_entity="team:cloudfactory/acme/billing/backend",
        )
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/acme/billing/backend"

    def test_build_with_valid_access_entity_project(self):
        """build_structured_memory should accept valid project access_entity."""
        text, metadata = build_structured_memory(
            text="API uses JWT",
            category="architecture",
            scope="project",
            access_entity="project:cloudfactory/acme/billing-api",
        )
        assert metadata["scope"] == "project"
        assert metadata["access_entity"] == "project:cloudfactory/acme/billing-api"

    def test_build_with_valid_access_entity_org(self):
        """build_structured_memory should accept valid org access_entity."""
        text, metadata = build_structured_memory(
            text="Use AWS",
            category="decision",
            scope="org",
            access_entity="org:cloudfactory",
        )
        assert metadata["scope"] == "org"
        assert metadata["access_entity"] == "org:cloudfactory"

    def test_build_with_valid_access_entity_user(self):
        """build_structured_memory should accept user access_entity for personal scope."""
        text, metadata = build_structured_memory(
            text="I prefer vim",
            category="convention",
            scope="user",
            access_entity="user:grischa",
        )
        assert metadata["scope"] == "user"
        assert metadata["access_entity"] == "user:grischa"

    def test_build_preserves_entity_separately_from_access_entity(self):
        """entity (semantic) should remain separate from access_entity."""
        text, metadata = build_structured_memory(
            text="Platform team decision",
            category="decision",
            scope="team",
            entity="platform-team",  # Semantic entity
            access_entity="team:cloudfactory/platform/core",  # Access control
        )
        assert metadata["entity"] == "platform-team"
        assert metadata["access_entity"] == "team:cloudfactory/platform/core"


class TestAccessEntityInNormalizeMetadata:
    """Tests for access_entity in normalize_metadata_for_create."""

    def test_normalize_requires_access_entity_for_team_scope(self):
        """normalize_metadata_for_create should require access_entity for team scope."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            normalize_metadata_for_create({
                "category": "convention",
                "scope": "team",
                # Missing access_entity
            })
        assert "access_entity" in str(exc_info.value).lower()

    def test_normalize_accepts_access_entity_for_team_scope(self):
        """normalize_metadata_for_create should accept valid access_entity."""
        metadata = normalize_metadata_for_create({
            "category": "convention",
            "scope": "team",
            "access_entity": "team:cloudfactory/billing/backend",
        })
        assert metadata["scope"] == "team"
        assert metadata["access_entity"] == "team:cloudfactory/billing/backend"

    def test_normalize_does_not_require_access_entity_for_user_scope(self):
        """normalize_metadata_for_create should NOT require access_entity for user scope."""
        metadata = normalize_metadata_for_create({
            "category": "convention",
            "scope": "user",
            # No access_entity - should be OK
        })
        assert metadata["scope"] == "user"

    def test_normalize_validates_access_entity_format(self):
        """normalize_metadata_for_create should validate access_entity format."""
        with pytest.raises(StructuredMemoryError):
            normalize_metadata_for_create({
                "category": "convention",
                "scope": "team",
                "access_entity": "invalid-format",  # Missing prefix
            })


class TestAccessEntityInUpdateFields:
    """Tests for access_entity in validate_update_fields."""

    def test_update_fields_validates_access_entity(self):
        """validate_update_fields should validate access_entity if provided."""
        validated = validate_update_fields(
            access_entity="team:cloudfactory/billing/backend",
        )
        assert validated["access_entity"] == "team:cloudfactory/billing/backend"

    def test_update_fields_rejects_invalid_access_entity(self):
        """validate_update_fields should reject invalid access_entity."""
        with pytest.raises(StructuredMemoryError):
            validate_update_fields(
                access_entity="invalid-no-prefix",
            )

    def test_update_scope_to_team_without_access_entity_info(self):
        """
        Changing scope to team in update should work if access_entity exists.

        Note: The actual enforcement of "team scope requires access_entity"
        happens at the API level, not in validate_update_fields which only
        validates individual field values.
        """
        # This should succeed - we're just validating the scope value
        validated = validate_update_fields(scope="team")
        assert validated["scope"] == "team"


class TestAccessEntityScopeMismatchValidation:
    """Tests for scope/access_entity prefix consistency."""

    def test_team_scope_with_non_team_access_entity_raises(self):
        """scope=team with user: access_entity should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Team standard",
                category="convention",
                scope="team",
                access_entity="user:grischa",  # Mismatch!
            )
        assert "mismatch" in str(exc_info.value).lower() or "scope" in str(exc_info.value).lower()

    def test_org_scope_with_team_access_entity_raises(self):
        """scope=org with team: access_entity should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Org policy",
                category="decision",
                scope="org",
                access_entity="team:cloudfactory/billing",  # Mismatch!
            )
        assert "mismatch" in str(exc_info.value).lower() or "scope" in str(exc_info.value).lower()

    def test_user_scope_with_team_access_entity_raises(self):
        """scope=user with team: access_entity should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Personal note",
                category="convention",
                scope="user",
                access_entity="team:cloudfactory/billing",  # Mismatch!
            )
        assert "mismatch" in str(exc_info.value).lower() or "scope" in str(exc_info.value).lower()

    def test_project_scope_with_org_access_entity_raises(self):
        """scope=project with org: access_entity should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Project architecture",
                category="architecture",
                scope="project",
                access_entity="org:cloudfactory",  # Mismatch!
            )
        assert "mismatch" in str(exc_info.value).lower() or "scope" in str(exc_info.value).lower()
