"""
Tests for access_entity filtering in REST and MCP endpoints.

TDD: These tests define the expected behavior for multi-user memory routing
with access_entity-based access control.

Phase 3 Tests:
- Access filtering in REST endpoints (list/get/filter/related)
- MCP add/update/delete enforcement
- Group-editable write policy

Key rules:
- scope=user -> only owner can access
- scope!=user -> access_entity must match principal's grants
- Write access (group-editable): any member of access_entity can update/delete
"""

import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from sqlalchemy.orm import Session

from app.models import Memory, MemoryState, User, App
from app.security.types import TokenClaims, Principal


# Test UUIDs
USER_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
APP_A_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def make_claims(
    sub: str,
    org_id: str = "test-org",
    grants: set = None,
    scopes: set = None,
) -> TokenClaims:
    """Helper to create TokenClaims for testing."""
    return TokenClaims(
        sub=sub,
        iss="https://auth.test.example.com",
        aud="https://api.test.example.com",
        exp=datetime.now(timezone.utc),
        iat=datetime.now(timezone.utc),
        jti=f"jti-{uuid.uuid4()}",
        org_id=org_id,
        scopes=scopes or {"memories:read", "memories:write", "memories:delete"},
        grants=grants or {f"user:{sub}"},
    )


def make_principal(
    user_id: str,
    org_id: str = "test-org",
    grants: set = None,
    scopes: set = None,
) -> Principal:
    """Helper to create Principal for testing."""
    claims = make_claims(user_id, org_id, grants, scopes)
    return Principal(
        user_id=user_id,
        org_id=org_id,
        claims=claims,
    )


class TestAccessEntityReadFiltering:
    """Tests for reading memories with access_entity filtering."""

    def test_user_can_read_own_personal_memory(self):
        """User should be able to read their own personal (scope=user) memory."""
        principal = make_principal("grischa")

        # Memory with scope=user and access_entity=user:grischa
        memory_access_entity = "user:grischa"

        assert principal.can_access(memory_access_entity) is True

    def test_user_cannot_read_other_user_personal_memory(self):
        """User should NOT be able to read another user's personal memory."""
        principal = make_principal("grischa")

        # Memory owned by different user
        memory_access_entity = "user:other-user"

        assert principal.can_access(memory_access_entity) is False

    def test_user_can_read_team_memory_with_team_grant(self):
        """User with team grant should be able to read team memories."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(memory_access_entity) is True

    def test_user_cannot_read_team_memory_without_grant(self):
        """User without team grant should NOT be able to read team memories."""
        principal = make_principal(
            "grischa",
            grants={"user:grischa"},  # No team grant
        )

        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(memory_access_entity) is False

    def test_user_can_read_project_memory_with_project_grant(self):
        """User with project grant should be able to read project memories."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "project:cloudfactory/acme/billing-api",
            }
        )

        memory_access_entity = "project:cloudfactory/acme/billing-api"

        assert principal.can_access(memory_access_entity) is True

    def test_org_grant_allows_reading_project_memories(self):
        """User with org grant should be able to read project memories under that org."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )

        # Project under cloudfactory org
        memory_access_entity = "project:cloudfactory/acme/billing-api"

        assert principal.can_access(memory_access_entity) is True

    def test_org_grant_allows_reading_team_memories(self):
        """User with org grant should be able to read team memories under that org."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )

        # Team under cloudfactory org
        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(memory_access_entity) is True

    def test_project_grant_allows_reading_team_memories_under_project(self):
        """User with project grant should be able to read team memories under that project."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "project:cloudfactory/acme/billing",
            }
        )

        # Team under the project
        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(memory_access_entity) is True

    def test_team_grant_does_not_allow_reading_other_team(self):
        """User with team grant should NOT be able to read other teams' memories."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # Different team
        memory_access_entity = "team:cloudfactory/acme/billing/frontend"

        assert principal.can_access(memory_access_entity) is False

    def test_team_grant_does_not_allow_reading_parent_project(self):
        """User with team grant should NOT be able to read parent project memories."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # Parent project
        memory_access_entity = "project:cloudfactory/acme/billing"

        assert principal.can_access(memory_access_entity) is False

    def test_org_grant_does_not_allow_reading_other_org(self):
        """User with org grant should NOT be able to read other orgs' memories."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )

        # Different org
        memory_access_entity = "org:other-company"

        assert principal.can_access(memory_access_entity) is False


class TestAccessEntityWriteFiltering:
    """Tests for write (update/delete) access with group-editable policy."""

    def test_owner_can_update_personal_memory(self):
        """Owner should be able to update their own personal memory."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal("grischa")
        memory_access_entity = "user:grischa"

        assert can_write_to_access_entity(principal, memory_access_entity) is True

    def test_other_user_cannot_update_personal_memory(self):
        """Other user should NOT be able to update another's personal memory."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal("grischa")
        memory_access_entity = "user:other-user"

        assert can_write_to_access_entity(principal, memory_access_entity) is False

    def test_team_member_can_update_team_memory(self):
        """Team member should be able to update team memory (group-editable)."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )
        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert can_write_to_access_entity(principal, memory_access_entity) is True

    def test_non_team_member_cannot_update_team_memory(self):
        """Non-team member should NOT be able to update team memory."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={"user:grischa"},  # No team grant
        )
        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        assert can_write_to_access_entity(principal, memory_access_entity) is False

    def test_org_member_can_update_project_memory(self):
        """Org member should be able to update project memory (hierarchy expansion)."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )
        memory_access_entity = "project:cloudfactory/acme/billing-api"

        assert can_write_to_access_entity(principal, memory_access_entity) is True

    def test_team_member_can_delete_team_memory(self):
        """Team member should be able to delete team memory (group-editable)."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )
        memory_access_entity = "team:cloudfactory/acme/billing/backend"

        # Delete uses same access check as update
        assert can_write_to_access_entity(principal, memory_access_entity) is True


class TestAccessEntityCreateValidation:
    """Tests for creating memories with access_entity validation."""

    def test_user_can_create_memory_with_own_access_entity(self):
        """User should be able to create memory with their own access_entity."""
        principal = make_principal("grischa")
        access_entity = "user:grischa"

        # User always has grant for their own user:<sub>
        assert principal.can_access(access_entity) is True

    def test_user_cannot_create_memory_for_other_user(self):
        """User should NOT be able to create memory with another user's access_entity."""
        principal = make_principal("grischa")
        access_entity = "user:other-user"

        assert principal.can_access(access_entity) is False

    def test_team_member_can_create_team_memory(self):
        """Team member should be able to create team memory."""
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )
        access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(access_entity) is True

    def test_non_team_member_cannot_create_team_memory(self):
        """Non-team member should NOT be able to create team memory."""
        principal = make_principal(
            "grischa",
            grants={"user:grischa"},  # No team grant
        )
        access_entity = "team:cloudfactory/acme/billing/backend"

        assert principal.can_access(access_entity) is False


class TestResolveAccessEntities:
    """Tests for resolve_access_entities helper."""

    def test_resolve_always_includes_user_grant(self):
        """resolve_access_entities should always include user:<sub>."""
        from app.security.access import resolve_access_entities

        principal = make_principal(
            "grischa",
            grants={"team:cloudfactory/billing"},  # No explicit user grant
        )

        resolved = resolve_access_entities(principal)

        assert "user:grischa" in resolved

    def test_resolve_includes_explicit_grants(self):
        """resolve_access_entities should include all explicit grants."""
        from app.security.access import resolve_access_entities

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/billing/backend",
                "project:cloudfactory/billing",
            }
        )

        resolved = resolve_access_entities(principal)

        assert "user:grischa" in resolved
        assert "team:cloudfactory/billing/backend" in resolved
        assert "project:cloudfactory/billing" in resolved


class TestBuildAccessFilterQuery:
    """Tests for building database queries with access_entity filtering.

    These tests define how memory queries should filter by access_entity
    based on the principal's grants.
    """

    def test_filter_includes_user_personal_memories(self):
        """Query filter should include user's personal memories (scope=user)."""
        # This test defines the expected SQL/query behavior
        # Implementation will need to build a filter like:
        # WHERE (metadata_->>'scope' = 'user' AND metadata_->>'access_entity' = 'user:grischa')
        #    OR (metadata_->>'access_entity' IN (grants))
        pass  # Placeholder - actual implementation will test query building

    def test_filter_includes_shared_memories_with_matching_grant(self):
        """Query filter should include shared memories where access_entity matches grants."""
        # This test defines the expected behavior
        # A memory with access_entity='team:cloudfactory/billing/backend'
        # should be included if principal has that grant
        pass  # Placeholder

    def test_filter_expands_org_grant_to_match_projects_and_teams(self):
        """Query filter should expand org grant to match child projects/teams."""
        # This is more complex - may need LIKE 'cloudfactory/%' pattern
        # or multiple OR conditions
        pass  # Placeholder


class TestMCPAddMemoriesEnforcement:
    """Tests for MCP add_memories access_entity enforcement."""

    @pytest.mark.asyncio
    async def test_add_memories_with_valid_access_entity_succeeds(self):
        """add_memories should succeed when user has grant for access_entity."""
        # This will test the actual MCP endpoint
        pass  # Placeholder - needs MCP integration

    @pytest.mark.asyncio
    async def test_add_memories_without_grant_for_access_entity_fails(self):
        """add_memories should fail when user lacks grant for access_entity."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_add_memories_validates_scope_access_entity_consistency(self):
        """add_memories should validate scope/access_entity consistency."""
        # scope=team with access_entity=user:* should fail
        pass  # Placeholder


class TestMCPUpdateMemoryEnforcement:
    """Tests for MCP update_memory access_entity enforcement."""

    @pytest.mark.asyncio
    async def test_update_memory_by_team_member_succeeds(self):
        """update_memory should succeed for team member (group-editable)."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_update_memory_by_non_member_fails(self):
        """update_memory should fail for non-member."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_update_memory_cannot_change_access_entity_without_new_grant(self):
        """update_memory should fail if changing access_entity to one without grant."""
        pass  # Placeholder


class TestMCPDeleteMemoriesEnforcement:
    """Tests for MCP delete_memories access_entity enforcement."""

    @pytest.mark.asyncio
    async def test_delete_memory_by_team_member_succeeds(self):
        """delete_memories should succeed for team member (group-editable)."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_delete_memory_by_non_member_fails(self):
        """delete_memories should fail for non-member."""
        pass  # Placeholder


class TestMCPSearchMemoryFiltering:
    """Tests for MCP search_memory access_entity filtering."""

    @pytest.mark.asyncio
    async def test_search_returns_only_accessible_memories(self):
        """search_memory should only return memories user has grant for."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_search_excludes_memories_without_grant(self):
        """search_memory should exclude memories user lacks grant for."""
        pass  # Placeholder


class TestListMemoriesFiltering:
    """Tests for list_memories endpoint access_entity filtering."""

    @pytest.mark.asyncio
    async def test_list_returns_personal_and_shared_memories(self):
        """list_memories should return both personal and shared memories."""
        pass  # Placeholder

    @pytest.mark.asyncio
    async def test_list_excludes_memories_without_grant(self):
        """list_memories should exclude memories user lacks grant for."""
        pass  # Placeholder
