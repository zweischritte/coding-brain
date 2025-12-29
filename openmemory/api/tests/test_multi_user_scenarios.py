"""
End-to-end multi-user scenario tests for access_entity-based memory routing.

These tests simulate realistic team collaboration workflows and verify that:
1. Team members can share memories appropriately
2. Organization hierarchy access works correctly
3. Personal vs shared memory isolation is enforced
4. Access entity changes are properly validated
5. Legacy memories are handled correctly

Key patterns tested:
- Principal.can_access() for hierarchical access checking
- can_write_to_access_entity() for group-editable policy
- filter_memories_by_access() for memory list filtering
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import pytest

from app.security.types import TokenClaims, Principal
from app.security.access import (
    can_write_to_access_entity,
    filter_memories_by_access,
    resolve_access_entities,
    get_default_access_entity,
    check_create_access,
)


# -----------------------------------------------------------------------------
# Test Helpers (copied from test_access_entity_filtering.py)
# -----------------------------------------------------------------------------

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


class MockMemory:
    """Mock memory object for testing filter_memories_by_access."""

    def __init__(
        self,
        access_entity: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        memory_id: str = None,
    ):
        self.id = memory_id or str(uuid.uuid4())
        self.metadata_ = {"access_entity": access_entity} if access_entity else {}
        # Create a mock user object for legacy memory checks
        self.user = type("MockUser", (), {"user_id": owner_user_id})() if owner_user_id else None


class MockMemoryWithContent(MockMemory):
    """Mock memory with content for more detailed testing."""

    def __init__(
        self,
        access_entity: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        content: str = "test content",
        memory_id: str = None,
    ):
        super().__init__(access_entity, owner_user_id, memory_id)
        self.content = content
        self.metadata_["content"] = content


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------

class TestTeamCollaborationScenario:
    """Test team members sharing memories."""

    def test_team_memory_visible_to_all_members(self):
        """Memory created with team access_entity visible to all team members."""
        # Setup: Three users in different teams
        # User A - backend team member
        user_a = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # User B - also backend team member
        user_b = make_principal(
            "bob",
            grants={
                "user:bob",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # User C - frontend team member (different team)
        user_c = make_principal(
            "charlie",
            grants={
                "user:charlie",
                "team:cloudfactory/acme/billing/frontend",
            }
        )

        # Memory created by User A with team access
        team_memory_access_entity = "team:cloudfactory/acme/billing/backend"

        # User A (creator) can access the memory
        assert user_a.can_access(team_memory_access_entity) is True

        # User B (team member) can also access the memory
        assert user_b.can_access(team_memory_access_entity) is True

        # User C (different team) cannot access the memory
        assert user_c.can_access(team_memory_access_entity) is False

    def test_team_member_can_update_shared_memory(self):
        """Any team member can update team memory (group-editable policy)."""
        # User A creates team memory
        user_a = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # User B is also a team member
        user_b = make_principal(
            "bob",
            grants={
                "user:bob",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        team_memory_access_entity = "team:cloudfactory/acme/billing/backend"

        # User A can update (as creator)
        assert can_write_to_access_entity(user_a, team_memory_access_entity) is True

        # User B can also update (as team member - group-editable)
        assert can_write_to_access_entity(user_b, team_memory_access_entity) is True

    def test_team_member_can_delete_shared_memory(self):
        """Any team member can delete team memory (group-editable policy)."""
        user_a = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        user_b = make_principal(
            "bob",
            grants={
                "user:bob",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        team_memory_access_entity = "team:cloudfactory/acme/billing/backend"

        # Both team members can delete team memories
        assert can_write_to_access_entity(user_a, team_memory_access_entity) is True
        assert can_write_to_access_entity(user_b, team_memory_access_entity) is True

    def test_non_member_cannot_update_team_memory(self):
        """Non-team member cannot update team memory."""
        outsider = make_principal(
            "outsider",
            grants={"user:outsider"},  # Only user grant, no team grant
        )

        team_memory_access_entity = "team:cloudfactory/acme/billing/backend"

        # Outsider cannot update the team memory
        assert can_write_to_access_entity(outsider, team_memory_access_entity) is False

    def test_multiple_teams_isolation(self):
        """Members of different teams cannot access each other's team memories."""
        # Backend team member
        backend_dev = make_principal(
            "backend_dev",
            grants={
                "user:backend_dev",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # Frontend team member
        frontend_dev = make_principal(
            "frontend_dev",
            grants={
                "user:frontend_dev",
                "team:cloudfactory/acme/billing/frontend",
            }
        )

        backend_memory = "team:cloudfactory/acme/billing/backend"
        frontend_memory = "team:cloudfactory/acme/billing/frontend"

        # Backend dev can access backend, not frontend
        assert backend_dev.can_access(backend_memory) is True
        assert backend_dev.can_access(frontend_memory) is False

        # Frontend dev can access frontend, not backend
        assert frontend_dev.can_access(frontend_memory) is True
        assert frontend_dev.can_access(backend_memory) is False

    def test_team_member_sees_team_memories_in_list(self):
        """filter_memories_by_access returns team memories for team members."""
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        memories = [
            MockMemory("user:alice"),  # Own personal memory
            MockMemory("team:cloudfactory/acme/billing/backend"),  # Team memory
            MockMemory("team:cloudfactory/acme/billing/frontend"),  # Other team
            MockMemory("user:bob"),  # Another user's personal memory
        ]

        filtered = filter_memories_by_access(user, memories)

        assert len(filtered) == 2
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        assert "user:alice" in access_entities
        assert "team:cloudfactory/acme/billing/backend" in access_entities
        assert "team:cloudfactory/acme/billing/frontend" not in access_entities
        assert "user:bob" not in access_entities


class TestOrganizationHierarchyScenario:
    """Test org-level access expansion."""

    def test_org_admin_sees_all_org_memories(self):
        """User with org grant sees project/team memories under that org."""
        org_admin = make_principal(
            "admin",
            grants={
                "user:admin",
                "org:cloudfactory",
            }
        )

        # Create memories at different hierarchy levels
        memories = [
            MockMemory("org:cloudfactory"),
            MockMemory("project:cloudfactory/acme"),
            MockMemory("project:cloudfactory/acme/billing"),
            MockMemory("team:cloudfactory/acme/billing/backend"),
            MockMemory("team:cloudfactory/acme/billing/frontend"),
            MockMemory("org:other-company"),  # Different org
            MockMemory("project:other-company/project"),  # Different org
        ]

        filtered = filter_memories_by_access(org_admin, memories)

        # Should see all cloudfactory memories (5), not other-company (2)
        assert len(filtered) == 5
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        assert "org:cloudfactory" in access_entities
        assert "project:cloudfactory/acme" in access_entities
        assert "project:cloudfactory/acme/billing" in access_entities
        assert "team:cloudfactory/acme/billing/backend" in access_entities
        assert "team:cloudfactory/acme/billing/frontend" in access_entities
        assert "org:other-company" not in access_entities
        assert "project:other-company/project" not in access_entities

    def test_project_member_sees_project_and_team_memories(self):
        """Project grant includes team memories under project."""
        project_member = make_principal(
            "developer",
            grants={
                "user:developer",
                "project:cloudfactory/acme/billing",
            }
        )

        memories = [
            MockMemory("project:cloudfactory/acme/billing"),  # Project level
            MockMemory("team:cloudfactory/acme/billing/backend"),  # Team under project
            MockMemory("team:cloudfactory/acme/billing/frontend"),  # Team under project
            MockMemory("project:cloudfactory/acme/payments"),  # Different project
            MockMemory("org:cloudfactory"),  # Parent org (no access)
        ]

        filtered = filter_memories_by_access(project_member, memories)

        # Should see project + teams under project (3), not other project or parent org
        assert len(filtered) == 3
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        assert "project:cloudfactory/acme/billing" in access_entities
        assert "team:cloudfactory/acme/billing/backend" in access_entities
        assert "team:cloudfactory/acme/billing/frontend" in access_entities
        assert "project:cloudfactory/acme/payments" not in access_entities
        assert "org:cloudfactory" not in access_entities

    def test_org_admin_can_update_any_project_memory(self):
        """Org admin can update any project/team memory under the org."""
        org_admin = make_principal(
            "admin",
            grants={
                "user:admin",
                "org:cloudfactory",
            }
        )

        # Org admin can write to any hierarchy level under org
        assert can_write_to_access_entity(org_admin, "org:cloudfactory") is True
        assert can_write_to_access_entity(org_admin, "project:cloudfactory/acme") is True
        assert can_write_to_access_entity(org_admin, "team:cloudfactory/acme/billing/backend") is True

        # But not to other orgs
        assert can_write_to_access_entity(org_admin, "org:other-company") is False
        assert can_write_to_access_entity(org_admin, "project:other-company/project") is False

    def test_team_member_cannot_access_parent_org(self):
        """Team grant does not give access to parent org or project."""
        team_member = make_principal(
            "developer",
            grants={
                "user:developer",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        # Can access own team
        assert team_member.can_access("team:cloudfactory/acme/billing/backend") is True

        # Cannot access parent project or org
        assert team_member.can_access("project:cloudfactory/acme/billing") is False
        assert team_member.can_access("org:cloudfactory") is False

        # Cannot access sibling team
        assert team_member.can_access("team:cloudfactory/acme/billing/frontend") is False

    def test_multiple_grant_levels_combined(self):
        """User with grants at multiple levels sees all appropriate memories."""
        # User with both org and team grants
        multi_grant_user = make_principal(
            "superuser",
            grants={
                "user:superuser",
                "org:cloudfactory",
                "team:other-company/project/team",  # Also in another org's team
            }
        )

        memories = [
            MockMemory("org:cloudfactory"),
            MockMemory("project:cloudfactory/acme"),
            MockMemory("team:cloudfactory/acme/backend"),
            MockMemory("team:other-company/project/team"),  # From other org
            MockMemory("org:other-company"),  # No org access to other-company
        ]

        filtered = filter_memories_by_access(multi_grant_user, memories)

        assert len(filtered) == 4
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        # Has org:cloudfactory so sees all cloudfactory memories
        assert "org:cloudfactory" in access_entities
        assert "project:cloudfactory/acme" in access_entities
        assert "team:cloudfactory/acme/backend" in access_entities
        # Has team grant in other-company
        assert "team:other-company/project/team" in access_entities
        # Does not have org:other-company
        assert "org:other-company" not in access_entities


class TestPersonalVsSharedScenario:
    """Test personal vs shared memory isolation."""

    def test_personal_memories_isolated(self):
        """Personal memories only visible to owner."""
        user_a = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",  # Has team grant
            }
        )

        user_b = make_principal(
            "bob",
            grants={
                "user:bob",
                "team:cloudfactory/backend",  # Same team as alice
            }
        )

        # Alice's personal memory
        alice_personal = "user:alice"

        # Alice can access her own personal memory
        assert user_a.can_access(alice_personal) is True

        # Bob cannot access Alice's personal memory even though they share a team
        assert user_b.can_access(alice_personal) is False

    def test_user_can_share_memory_with_team(self):
        """User can create memory shared with team if they have team grant."""
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
            }
        )

        # User can create personal memory
        assert check_create_access(user, "user:alice") is True

        # User can also create team memory (because they have team grant)
        assert check_create_access(user, "team:cloudfactory/backend") is True

        # User cannot create memory for team they don't belong to
        assert check_create_access(user, "team:cloudfactory/frontend") is False

    def test_user_without_team_grant_can_only_create_personal(self):
        """User without team grant can only create personal memories."""
        user = make_principal(
            "alice",
            grants={"user:alice"},  # Only user grant
        )

        # Can create personal memory
        assert check_create_access(user, "user:alice") is True

        # Cannot create team memory
        assert check_create_access(user, "team:cloudfactory/backend") is False

        # Cannot create org memory
        assert check_create_access(user, "org:cloudfactory") is False

    def test_get_default_access_entity_returns_user_entity(self):
        """get_default_access_entity returns user:<user_id>."""
        user = make_principal("alice")

        default = get_default_access_entity(user)

        assert default == "user:alice"

    def test_resolve_access_entities_includes_all_grants(self):
        """resolve_access_entities includes user grant plus all explicit grants."""
        user = make_principal(
            "alice",
            grants={
                "team:cloudfactory/backend",
                "project:cloudfactory/acme",
            }
        )

        resolved = resolve_access_entities(user)

        # Should include implied user grant + explicit grants
        assert "user:alice" in resolved
        assert "team:cloudfactory/backend" in resolved
        assert "project:cloudfactory/acme" in resolved

    def test_mixed_personal_and_shared_memory_filtering(self):
        """filter_memories_by_access correctly handles mix of personal and shared."""
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
            }
        )

        memories = [
            MockMemoryWithContent("user:alice", content="My personal notes"),
            MockMemoryWithContent("team:cloudfactory/backend", content="Team docs"),
            MockMemoryWithContent("user:bob", content="Bob's notes"),  # Not visible
            MockMemoryWithContent("team:cloudfactory/frontend", content="Frontend docs"),  # Not visible
        ]

        filtered = filter_memories_by_access(user, memories)

        assert len(filtered) == 2
        contents = [m.content for m in filtered]
        assert "My personal notes" in contents
        assert "Team docs" in contents
        assert "Bob's notes" not in contents
        assert "Frontend docs" not in contents


class TestAccessEntityChangeScenario:
    """Test changing access_entity on memories."""

    def test_user_can_widen_access_with_grant(self):
        """User with both grants can widen access from user to team."""
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
            }
        )

        # Original memory access_entity
        original_access_entity = "user:alice"
        # New (widened) access_entity
        new_access_entity = "team:cloudfactory/backend"

        # User must have write access to original
        assert can_write_to_access_entity(user, original_access_entity) is True

        # User must also have access to new target
        assert user.can_access(new_access_entity) is True

        # Both conditions met -> change allowed (in real impl, endpoint checks both)

    def test_user_cannot_widen_access_without_target_grant(self):
        """User cannot widen access without target grant."""
        user = make_principal(
            "alice",
            grants={"user:alice"},  # Only has user grant
        )

        original_access_entity = "user:alice"
        target_access_entity = "team:cloudfactory/frontend"  # No grant for this team

        # Can write to original (own memory)
        assert can_write_to_access_entity(user, original_access_entity) is True

        # Cannot access target (no team grant)
        assert user.can_access(target_access_entity) is False

        # Change should be denied (endpoint would check both)

    def test_user_cannot_narrow_access_without_original_grant(self):
        """User cannot change access_entity if they don't have write access to original."""
        # User with only personal grant
        user = make_principal(
            "alice",
            grants={"user:alice"},
        )

        # Try to change team memory (which they can't access)
        team_access_entity = "team:cloudfactory/backend"

        # Cannot write to team memory (no team grant)
        assert can_write_to_access_entity(user, team_access_entity) is False

    def test_org_admin_can_change_any_access_entity_within_org(self):
        """Org admin can change access_entity between any levels within org."""
        org_admin = make_principal(
            "admin",
            grants={
                "user:admin",
                "org:cloudfactory",
            }
        )

        # Can change from project to team
        project_entity = "project:cloudfactory/acme"
        team_entity = "team:cloudfactory/acme/backend"

        assert can_write_to_access_entity(org_admin, project_entity) is True
        assert org_admin.can_access(team_entity) is True

        # Can change from team to org
        assert can_write_to_access_entity(org_admin, team_entity) is True
        assert org_admin.can_access("org:cloudfactory") is True

    def test_project_member_cannot_promote_to_org(self):
        """Project member cannot change access_entity to org level."""
        project_member = make_principal(
            "developer",
            grants={
                "user:developer",
                "project:cloudfactory/acme/billing",
            }
        )

        project_entity = "project:cloudfactory/acme/billing"
        org_entity = "org:cloudfactory"

        # Can write to project memory
        assert can_write_to_access_entity(project_member, project_entity) is True

        # Cannot access org level
        assert project_member.can_access(org_entity) is False

        # Change from project to org should be denied

    def test_change_to_different_team_requires_both_grants(self):
        """Changing access_entity to different team requires both team grants."""
        # User in multiple teams
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
                "team:cloudfactory/devops",
            }
        )

        backend_entity = "team:cloudfactory/backend"
        devops_entity = "team:cloudfactory/devops"
        frontend_entity = "team:cloudfactory/frontend"

        # Can change from backend to devops (has both grants)
        assert can_write_to_access_entity(user, backend_entity) is True
        assert user.can_access(devops_entity) is True

        # Cannot change to frontend (no grant)
        assert user.can_access(frontend_entity) is False


class TestLegacyMemoryMigrationScenario:
    """Test handling of legacy memories without access_entity."""

    def test_legacy_memory_accessible_to_owner_only(self):
        """Memory without access_entity only visible to owner."""
        owner = make_principal("alice")
        other_user = make_principal("bob")

        # Legacy memory without access_entity (metadata_ is empty dict)
        legacy_memory = MockMemory(access_entity=None, owner_user_id="alice")

        memories = [legacy_memory]

        # Owner can see their legacy memory
        owner_filtered = filter_memories_by_access(owner, memories)
        assert len(owner_filtered) == 1

        # Other user cannot see it
        other_filtered = filter_memories_by_access(other_user, memories)
        assert len(other_filtered) == 0

    def test_legacy_memory_without_owner_is_inaccessible(self):
        """Legacy memory without owner info is not accessible to anyone."""
        user = make_principal("alice")

        # Legacy memory with no access_entity and no owner
        orphan_memory = MockMemory(access_entity=None, owner_user_id=None)

        memories = [orphan_memory]

        # No one can access orphan memory
        filtered = filter_memories_by_access(user, memories)
        assert len(filtered) == 0

    def test_mixed_legacy_and_new_memories(self):
        """Filtering handles mix of legacy and new memories correctly."""
        user = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
            }
        )

        memories = [
            # New memories with access_entity
            MockMemory("user:alice"),
            MockMemory("team:cloudfactory/backend"),
            MockMemory("user:bob"),
            # Legacy memory owned by user
            MockMemory(access_entity=None, owner_user_id="alice"),
            # Legacy memory owned by someone else
            MockMemory(access_entity=None, owner_user_id="bob"),
        ]

        filtered = filter_memories_by_access(user, memories)

        # Should get: user:alice, team:cloudfactory/backend, legacy owned by alice
        assert len(filtered) == 3

    def test_team_member_with_no_access_to_legacy_team_memories(self):
        """Team members cannot access legacy memories even if they share a team."""
        alice = make_principal(
            "alice",
            grants={
                "user:alice",
                "team:cloudfactory/backend",
            }
        )

        bob = make_principal(
            "bob",
            grants={
                "user:bob",
                "team:cloudfactory/backend",  # Same team as alice
            }
        )

        # Legacy memory owned by alice (no access_entity)
        legacy_memory = MockMemory(access_entity=None, owner_user_id="alice")

        # Alice can see her own legacy memory
        alice_filtered = filter_memories_by_access(alice, [legacy_memory])
        assert len(alice_filtered) == 1

        # Bob cannot see alice's legacy memory (even though they share a team)
        bob_filtered = filter_memories_by_access(bob, [legacy_memory])
        assert len(bob_filtered) == 0


class TestEdgeCasesScenario:
    """Test edge cases and boundary conditions."""

    def test_empty_grants_only_has_user_access(self):
        """User with empty grants set still has their own user access."""
        # This tests the implicit user grant
        user = make_principal(
            "alice",
            grants=set(),  # Empty grants
        )

        # Should still have access to own user scope
        assert user.can_access("user:alice") is True

        # But not to anything else
        assert user.can_access("team:cloudfactory/backend") is False

    def test_invalid_access_entity_format_denied(self):
        """Invalid access_entity format is denied."""
        user = make_principal("alice")

        # Invalid formats
        assert user.can_access("") is False
        assert user.can_access("no-colon") is False
        assert user.can_access(None) is False  # type: ignore

    def test_case_sensitivity_in_access_entities(self):
        """Access entities are case-sensitive."""
        user = make_principal(
            "Alice",  # Capital A
            grants={"user:Alice", "team:CloudFactory/Backend"},
        )

        # Exact match works
        assert user.can_access("user:Alice") is True
        assert user.can_access("team:CloudFactory/Backend") is True

        # Different case does not match
        assert user.can_access("user:alice") is False
        assert user.can_access("team:cloudfactory/backend") is False

    def test_filter_with_empty_memory_list(self):
        """filter_memories_by_access handles empty list."""
        user = make_principal("alice")

        filtered = filter_memories_by_access(user, [])

        assert filtered == []

    def test_filter_with_custom_accessor_function(self):
        """filter_memories_by_access works with custom accessor function."""
        user = make_principal(
            "alice",
            grants={"user:alice"},
        )

        # Custom objects with different structure
        class CustomMemory:
            def __init__(self, ae):
                self.data = {"ae": ae}

        memories = [
            CustomMemory("user:alice"),
            CustomMemory("user:bob"),
        ]

        # Custom accessor function
        def get_ae(m):
            return m.data.get("ae")

        filtered = filter_memories_by_access(user, memories, get_access_entity=get_ae)

        assert len(filtered) == 1
        assert filtered[0].data["ae"] == "user:alice"

    def test_deeply_nested_hierarchy_access(self):
        """Test access with deeply nested paths."""
        org_admin = make_principal(
            "admin",
            grants={
                "user:admin",
                "org:cloudfactory",
            }
        )

        # Deeply nested team path
        deep_team = "team:cloudfactory/acme/billing/api/v2/backend"

        # Org admin can access deeply nested teams
        assert org_admin.can_access(deep_team) is True

    def test_multiple_orgs_isolation(self):
        """User with grants in multiple orgs sees memories from both."""
        multi_org_user = make_principal(
            "consultant",
            grants={
                "user:consultant",
                "org:company_a",
                "org:company_b",
            }
        )

        memories = [
            MockMemory("org:company_a"),
            MockMemory("project:company_a/project1"),
            MockMemory("org:company_b"),
            MockMemory("project:company_b/project1"),
            MockMemory("org:company_c"),  # No access to this org
        ]

        filtered = filter_memories_by_access(multi_org_user, memories)

        assert len(filtered) == 4
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        assert "org:company_c" not in access_entities

    def test_service_account_with_service_grant(self):
        """Service account with service: prefix grant."""
        service = make_principal(
            "indexer-service",
            grants={
                "service:indexer-service",
                "org:cloudfactory",
            }
        )

        # Service can access its own scope
        assert service.can_access("service:indexer-service") is True

        # Service can also access org resources if it has org grant
        assert service.can_access("project:cloudfactory/acme") is True

    def test_client_access_entity_under_org(self):
        """Client access_entity follows org hierarchy."""
        org_admin = make_principal(
            "admin",
            grants={
                "user:admin",
                "org:cloudfactory",
            }
        )

        # Client under org should be accessible
        client_entity = "client:cloudfactory/customer_abc"

        assert org_admin.can_access(client_entity) is True

        # Client under different org should not be accessible
        other_client = "client:other-org/customer_xyz"

        assert org_admin.can_access(other_client) is False


class TestConcurrentAccessScenario:
    """Test scenarios with concurrent access patterns."""

    def test_multiple_users_same_team_all_see_same_memories(self):
        """All team members see the same set of team memories."""
        team_grant = "team:cloudfactory/backend"

        # Create multiple team members
        users = [
            make_principal(f"user_{i}", grants={f"user:user_{i}", team_grant})
            for i in range(5)
        ]

        # Team memories
        team_memories = [
            MockMemory(team_grant),
            MockMemory(team_grant),
            MockMemory(team_grant),
        ]

        # Each user should see all team memories
        for user in users:
            filtered = filter_memories_by_access(user, team_memories)
            assert len(filtered) == 3, f"User {user.user_id} should see 3 memories"

    def test_user_sees_personal_plus_team_memories(self):
        """User sees their personal memories plus shared team memories."""
        users = [
            make_principal(
                f"user_{i}",
                grants={f"user:user_{i}", "team:cloudfactory/backend"}
            )
            for i in range(3)
        ]

        # Mix of personal and team memories
        all_memories = [
            # Personal memories for each user
            MockMemory("user:user_0"),
            MockMemory("user:user_1"),
            MockMemory("user:user_2"),
            # Shared team memories
            MockMemory("team:cloudfactory/backend"),
            MockMemory("team:cloudfactory/backend"),
        ]

        # User 0 should see: 1 personal + 2 team = 3 memories
        user_0_filtered = filter_memories_by_access(users[0], all_memories)
        assert len(user_0_filtered) == 3

        # Verify correct memories
        access_entities = [m.metadata_["access_entity"] for m in user_0_filtered]
        assert access_entities.count("user:user_0") == 1
        assert access_entities.count("team:cloudfactory/backend") == 2
        assert "user:user_1" not in access_entities
        assert "user:user_2" not in access_entities
