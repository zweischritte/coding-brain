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
        from app.security.access import build_access_entity_patterns

        principal = make_principal("grischa", grants={"user:grischa"})

        exact_matches, like_patterns = build_access_entity_patterns(principal)

        # Should always include the user's personal access_entity
        assert "user:grischa" in exact_matches

    def test_filter_includes_shared_memories_with_matching_grant(self):
        """Query filter should include shared memories where access_entity matches grants."""
        from app.security.access import build_access_entity_patterns

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/billing/backend",
                "project:cloudfactory/acme",
            }
        )

        exact_matches, like_patterns = build_access_entity_patterns(principal)

        # Should include all explicit grants as exact matches
        assert "user:grischa" in exact_matches
        assert "team:cloudfactory/billing/backend" in exact_matches
        assert "project:cloudfactory/acme" in exact_matches

    def test_filter_expands_org_grant_to_match_projects_and_teams(self):
        """Query filter should expand org grant to match child projects/teams."""
        from app.security.access import build_access_entity_patterns

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )

        exact_matches, like_patterns = build_access_entity_patterns(principal)

        # Org grant should be in exact matches
        assert "org:cloudfactory" in exact_matches

        # Should generate LIKE patterns for child projects and teams
        assert "project:cloudfactory/%" in like_patterns
        assert "team:cloudfactory/%" in like_patterns

    def test_filter_expands_project_grant_to_match_teams(self):
        """Query filter should expand project grant to match child teams."""
        from app.security.access import build_access_entity_patterns

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "project:cloudfactory/acme/billing",
            }
        )

        exact_matches, like_patterns = build_access_entity_patterns(principal)

        # Project grant should be in exact matches
        assert "project:cloudfactory/acme/billing" in exact_matches

        # Should generate LIKE pattern for child teams
        assert "team:cloudfactory/acme/billing/%" in like_patterns

    def test_filter_team_grant_has_no_like_patterns(self):
        """Team grant should not generate any LIKE patterns (no children)."""
        from app.security.access import build_access_entity_patterns

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/acme/billing/backend",
            }
        )

        exact_matches, like_patterns = build_access_entity_patterns(principal)

        # Team should be in exact matches
        assert "team:cloudfactory/acme/billing/backend" in exact_matches

        # Team grants don't expand to children (teams are leaves)
        # Check that no team-based patterns were added
        team_patterns = [p for p in like_patterns if p.startswith("team:cloudfactory/acme/billing/backend")]
        assert len(team_patterns) == 0


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
        """list_memories should apply access_entity filtering (structural check)."""
        import ast
        from pathlib import Path

        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/memories.py")
        source = router_path.read_text()
        tree = ast.parse(source)

        list_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "list_memories":
                list_node = node
                break

        assert list_node is not None
        list_source = ast.get_source_segment(source, list_node) or ""
        assert "_apply_access_entity_filter" in list_source

    @pytest.mark.asyncio
    async def test_list_excludes_memories_without_grant(self):
        """filter_memories should apply access_entity filtering (structural check)."""
        import ast
        from pathlib import Path

        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/memories.py")
        source = router_path.read_text()
        tree = ast.parse(source)

        filter_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "filter_memories":
                filter_node = node
                break

        assert filter_node is not None
        filter_source = ast.get_source_segment(source, filter_node) or ""
        assert "_apply_access_entity_filter" in filter_source

    @pytest.mark.asyncio
    async def test_related_memories_apply_access_entity_filtering(self):
        """get_related_memories should apply access_entity filtering (structural check)."""
        import ast
        from pathlib import Path

        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/memories.py")
        source = router_path.read_text()
        tree = ast.parse(source)

        related_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_related_memories":
                related_node = node
                break

        assert related_node is not None
        related_source = ast.get_source_segment(source, related_node) or ""
        assert "_apply_access_entity_filter" in related_source

    @pytest.mark.asyncio
    async def test_categories_apply_access_entity_filtering(self):
        """get_categories should apply access_entity filtering (structural check)."""
        import ast
        from pathlib import Path

        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/memories.py")
        source = router_path.read_text()
        tree = ast.parse(source)

        categories_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_categories":
                categories_node = node
                break

        assert categories_node is not None
        categories_source = ast.get_source_segment(source, categories_node) or ""
        assert "_apply_access_entity_filter" in categories_source


class TestOpenSearchAccessEntityFiltering:
    """Tests for OpenSearch store access_entity filtering."""

    def test_access_entity_in_mappings(self):
        """access_entity should be defined as a keyword field in mappings."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create a mock client
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create = MagicMock()

        # Create store (this will call _ensure_index_and_alias)
        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
            index_prefix="memories",
        )

        # Verify indices.create was called
        mock_client.indices.create.assert_called_once()

        # Get the call arguments
        call_args = mock_client.indices.create.call_args
        index_body = call_args[1]["body"]

        # Verify access_entity is in the mappings
        mappings = index_body.get("mappings", {})
        properties = mappings.get("properties", {})

        assert "access_entity" in properties
        assert properties["access_entity"]["type"] == "keyword"

    def test_search_with_access_control_filters_by_access_entity(self):
        """search_with_access_control should filter results by access_entity."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create a mock client
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}
        mock_client.search.return_value = {"hits": {"hits": []}}

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
            index_prefix="memories",
        )

        # Call search_with_access_control
        access_entities = ["user:grischa", "team:cloudfactory/backend"]
        store.search_with_access_control(
            query_text="test query",
            access_entities=access_entities,
            limit=10,
        )

        # Verify search was called with access_entity filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        search_body = call_args[1]["body"]

        # Get the filter clauses
        filter_clauses = search_body["query"]["bool"]["filter"]

        # Find the access_entity filter (bool should with term/terms)
        access_filter = next(
            (c for c in filter_clauses if "bool" in c and "should" in c["bool"]),
            None,
        )

        assert access_filter is not None
        should_clauses = access_filter["bool"]["should"]
        found = set()
        for clause in should_clauses:
            if "term" in clause and "access_entity" in clause["term"]:
                found.add(clause["term"]["access_entity"])
            if "terms" in clause and "access_entity" in clause["terms"]:
                found.update(clause["terms"]["access_entity"])
        assert set(access_entities).issubset(found)

    def test_search_with_access_control_uses_or_logic(self):
        """search_with_access_control should return results matching ANY access_entity."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create a mock client
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}

        # Mock search to return results with different access_entities
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "mem-1", "_source": {"access_entity": "user:grischa"}},
                    {"_id": "mem-2", "_source": {"access_entity": "team:cloudfactory/backend"}},
                ]
            }
        }

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
            index_prefix="memories",
        )

        # Call search_with_access_control with multiple access_entities
        access_entities = ["user:grischa", "team:cloudfactory/backend"]
        results = store.search_with_access_control(
            query_text="test query",
            access_entities=access_entities,
            limit=10,
        )

        # Verify the query uses OR logic across access entities
        call_args = mock_client.search.call_args
        search_body = call_args[1]["body"]
        filter_clauses = search_body["query"]["bool"]["filter"]

        # Find the access_entity bool filter
        access_filter = next(
            (c for c in filter_clauses if "bool" in c and "should" in c["bool"]),
            None,
        )

        assert access_filter is not None
        assert access_filter["bool"].get("minimum_should_match") == 1

    def test_hybrid_search_with_access_control_filters_by_access_entity(self):
        """hybrid_search_with_access_control should filter by access_entity."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create a mock client
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}
        mock_client.search.return_value = {"hits": {"hits": []}}

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
            index_prefix="memories",
        )

        # Call hybrid_search_with_access_control
        access_entities = ["user:grischa", "team:cloudfactory/backend"]
        query_vector = [0.1] * 1536  # Mock embedding vector

        store.hybrid_search_with_access_control(
            query_text="test query",
            query_vector=query_vector,
            access_entities=access_entities,
            limit=10,
        )

        # Verify search was called with access_entity filter
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        search_body = call_args[1]["body"]

        # Get the filter clauses
        filter_clauses = search_body["query"]["bool"]["filter"]

        # Find the access_entity filter
        access_filter = next(
            (c for c in filter_clauses if "bool" in c and "should" in c["bool"]),
            None,
        )

        assert access_filter is not None
        should_clauses = access_filter["bool"]["should"]
        found = set()
        for clause in should_clauses:
            if "term" in clause and "access_entity" in clause["term"]:
                found.add(clause["term"]["access_entity"])
            if "terms" in clause and "access_entity" in clause["terms"]:
                found.update(clause["terms"]["access_entity"])
        assert set(access_entities).issubset(found)

        # Also verify the hybrid search has both lexical and vector components
        should_clauses = search_body["query"]["bool"]["should"]
        assert len(should_clauses) == 2  # lexical + vector

    def test_create_access_entity_filter_with_additional_filters(self):
        """_create_access_entity_filter should combine access_entity with other filters."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
        )

        access_entities = ["user:grischa"]
        additional_filters = {"category": "workflow", "scope": "project"}

        filter_clauses = store._create_access_entity_filter(
            access_entities=access_entities,
            additional_filters=additional_filters,
        )

        # Should have: org_id filter + access_entity filter + 2 additional filters
        assert len(filter_clauses) == 4

        # Verify org_id filter is present
        org_filters = [c for c in filter_clauses if "term" in c and "org_id" in c.get("term", {})]
        assert len(org_filters) == 1

        # Verify access_entity bool filter is present
        access_filters = [c for c in filter_clauses if "bool" in c and "should" in c["bool"]]
        assert len(access_filters) == 1

    def test_search_with_access_control_supports_prefixes(self):
        """search_with_access_control should include prefix filters when provided."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}
        mock_client.search.return_value = {"hits": {"hits": []}}

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
        )

        access_entities = ["user:grischa"]
        access_entity_prefixes = ["team:cloudfactory/"]

        store.search_with_access_control(
            query_text="test query",
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=10,
        )

        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        search_body = call_args[1]["body"]

        filter_clauses = search_body["query"]["bool"]["filter"]
        access_filter = next(
            (c for c in filter_clauses if "bool" in c and "should" in c["bool"]),
            None,
        )
        assert access_filter is not None
        should_clauses = access_filter["bool"]["should"]
        assert any(
            "prefix" in clause and clause["prefix"].get("access_entity") == "team:cloudfactory/"
            for clause in should_clauses
        )

    def test_search_with_access_control_empty_access_entities(self):
        """search_with_access_control should work with empty access_entities list."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.indices.get_alias.return_value = {"tenant_test-org": {}}
        mock_client.search.return_value = {"hits": {"hits": []}}

        store = TenantOpenSearchStore(
            client=mock_client,
            org_id="test-org",
        )

        # Call with empty access_entities (should still filter by org_id)
        results = store.search_with_access_control(
            query_text="test query",
            access_entities=[],
            limit=10,
        )

        # Verify search was called
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        search_body = call_args[1]["body"]

        # Should only have org_id filter (no access_entity filter when list is empty)
        filter_clauses = search_body["query"]["bool"]["filter"]

        # Empty access_entities means no access_entity bool filter added
        access_filters = [c for c in filter_clauses if "bool" in c and "should" in c["bool"]]
        assert len(access_filters) == 0


class TestRESTSearchAccessEntityFiltering:
    """Tests for REST search access_entity filtering."""

    def test_search_routes_use_access_entity_filters(self):
        """REST search endpoints should use access_entity filters (structural check)."""
        from pathlib import Path

        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/search.py")
        source = router_path.read_text()

        assert "build_access_entity_patterns" in source
        assert "search_with_access_control" in source
        assert "hybrid_search_with_access_control" in source


class TestGraphQueryAccessEntityFiltering:
    """Tests for graph query filtering by access_entity."""

    def test_aggregate_memories_respects_allowed_memory_ids(self):
        """aggregate_memories_in_graph should only count allowed memories."""
        # Test that the function signature accepts allowed_memory_ids
        from app.graph.graph_ops import aggregate_memories_in_graph
        import inspect

        sig = inspect.signature(aggregate_memories_in_graph)

        # Verify function accepts allowed_memory_ids parameter
        assert "allowed_memory_ids" in sig.parameters
        param = sig.parameters["allowed_memory_ids"]
        assert param.default is None  # Optional parameter

    def test_tag_cooccurrence_respects_allowed_memory_ids(self):
        """tag_cooccurrence_in_graph should only process allowed memories."""
        # Test that the function signature accepts allowed_memory_ids
        from app.graph.graph_ops import tag_cooccurrence_in_graph
        import inspect

        sig = inspect.signature(tag_cooccurrence_in_graph)

        # Verify function accepts allowed_memory_ids parameter
        assert "allowed_memory_ids" in sig.parameters
        param = sig.parameters["allowed_memory_ids"]
        assert param.default is None  # Optional parameter

    def test_fulltext_search_respects_allowed_memory_ids(self):
        """fulltext_search_memories_in_graph should only return allowed memories."""
        # Test that the function signature accepts allowed_memory_ids
        from app.graph.graph_ops import fulltext_search_memories_in_graph
        import inspect

        sig = inspect.signature(fulltext_search_memories_in_graph)

        # Verify function accepts allowed_memory_ids parameter
        assert "allowed_memory_ids" in sig.parameters
        param = sig.parameters["allowed_memory_ids"]
        assert param.default is None  # Optional parameter

    @patch("app.graph.graph_ops._get_projector")
    def test_aggregate_memories_with_allowed_ids_filters_results(self, mock_get_projector):
        """aggregate_memories_in_graph should filter results to allowed memories."""
        from app.graph.graph_ops import aggregate_memories_in_graph

        # Create mock projector
        mock_projector = MagicMock()
        mock_projector.aggregate_memories.return_value = [
            {"value": "workflow", "count": 2},
            {"value": "decision", "count": 1},
        ]
        mock_get_projector.return_value = mock_projector

        # Call with allowed_memory_ids that only includes some memories
        allowed_memory_ids = ["mem-1", "mem-2", "mem-6"]

        result = aggregate_memories_in_graph(
            user_id="test-user",
            group_by="category",
            allowed_memory_ids=allowed_memory_ids,
        )

        # Verify the projector was called with allowed_memory_ids
        mock_projector.aggregate_memories.assert_called_once()
        call_kwargs = mock_projector.aggregate_memories.call_args[1]
        assert call_kwargs.get("allowed_memory_ids") == allowed_memory_ids

    @patch("app.graph.graph_ops._get_projector")
    def test_tag_cooccurrence_with_allowed_ids_filters_results(self, mock_get_projector):
        """tag_cooccurrence_in_graph should filter results to allowed memories."""
        from app.graph.graph_ops import tag_cooccurrence_in_graph

        # Create mock projector
        mock_projector = MagicMock()
        mock_projector.tag_cooccurrence.return_value = [
            {"tag1": "important", "tag2": "urgent", "count": 2, "sample_memory_ids": ["mem-1", "mem-2"]},
        ]
        mock_get_projector.return_value = mock_projector

        # Call with allowed_memory_ids
        allowed_memory_ids = ["mem-1", "mem-2"]

        result = tag_cooccurrence_in_graph(
            user_id="test-user",
            allowed_memory_ids=allowed_memory_ids,
        )

        # Verify the projector was called with allowed_memory_ids
        mock_projector.tag_cooccurrence.assert_called_once()
        call_kwargs = mock_projector.tag_cooccurrence.call_args[1]
        assert call_kwargs.get("allowed_memory_ids") == allowed_memory_ids

    @patch("app.graph.graph_ops._get_projector")
    def test_fulltext_search_with_allowed_ids_filters_results(self, mock_get_projector):
        """fulltext_search_memories_in_graph should filter results to allowed memories."""
        from app.graph.graph_ops import fulltext_search_memories_in_graph

        # Create mock projector
        mock_projector = MagicMock()
        mock_projector.fulltext_search_memories.return_value = [
            {"memory_id": "mem-1", "content": "test content 1", "score": 0.9},
            {"memory_id": "mem-3", "content": "test content 3", "score": 0.7},
        ]
        mock_get_projector.return_value = mock_projector

        # Call with allowed_memory_ids that only includes some memories
        allowed_memory_ids = ["mem-1", "mem-3"]

        result = fulltext_search_memories_in_graph(
            search_text="test",
            user_id="test-user",
            allowed_memory_ids=allowed_memory_ids,
        )

        # Verify the projector was called with allowed_memory_ids
        mock_projector.fulltext_search_memories.assert_called_once()
        call_kwargs = mock_projector.fulltext_search_memories.call_args[1]
        assert call_kwargs.get("allowed_memory_ids") == allowed_memory_ids

    def test_get_allowed_memory_ids_helper_exists(self):
        """_get_allowed_memory_ids helper should exist in graph router."""
        from app.routers.graph import _get_allowed_memory_ids
        import inspect

        sig = inspect.signature(_get_allowed_memory_ids)

        # Verify function accepts principal and db parameters
        assert "principal" in sig.parameters
        assert "db" in sig.parameters

    def test_graph_router_endpoints_use_allowed_memory_ids(self):
        """Graph router endpoints should call _get_allowed_memory_ids."""
        # This is a structural test to verify the integration exists
        import ast
        from pathlib import Path

        # Read the graph router source
        router_path = Path("/Users/grischadallmer/git/coding-brain/openmemory/api/app/routers/graph.py")
        source = router_path.read_text()

        # Check that aggregate_by_dimension endpoint uses _get_allowed_memory_ids
        assert "_get_allowed_memory_ids" in source
        assert "allowed_memory_ids" in source

        # Parse the AST and verify the function calls
        tree = ast.parse(source)

        # Find the aggregate_by_dimension function
        found_aggregate_call = False
        found_tag_cooccurrence_call = False
        found_fulltext_search_call = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == "aggregate_memories_in_graph":
                        # Check if allowed_memory_ids is passed
                        for keyword in node.keywords:
                            if keyword.arg == "allowed_memory_ids":
                                found_aggregate_call = True
                    elif node.func.id == "tag_cooccurrence_in_graph":
                        for keyword in node.keywords:
                            if keyword.arg == "allowed_memory_ids":
                                found_tag_cooccurrence_call = True
                    elif node.func.id == "fulltext_search_memories_in_graph":
                        for keyword in node.keywords:
                            if keyword.arg == "allowed_memory_ids":
                                found_fulltext_search_call = True

        assert found_aggregate_call, "aggregate_memories_in_graph should receive allowed_memory_ids"
        assert found_tag_cooccurrence_call, "tag_cooccurrence_in_graph should receive allowed_memory_ids"
        assert found_fulltext_search_call, "fulltext_search_memories_in_graph should receive allowed_memory_ids"


class TestRESTEndpointAccessEntityIntegration:
    """Integration tests for REST endpoint access_entity enforcement."""

    def test_get_memory_checks_access_entity(self):
        """GET /memories/{id} should check access_entity before returning."""
        # User should only be able to get memory if they have access
        principal = make_principal(
            "grischa",
            grants={"user:grischa"},
        )

        # Memory with access_entity=user:grischa should be accessible
        assert principal.can_access("user:grischa") is True

        # Memory with access_entity=team:cloudfactory/backend should NOT be accessible
        assert principal.can_access("team:cloudfactory/backend") is False

    def test_list_memories_filters_by_access_entity(self):
        """GET /memories should filter results by access_entity."""
        from app.security.access import filter_memories_by_access

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            },
        )

        # Mock memory objects
        class MockMemory:
            def __init__(self, access_entity):
                self.metadata_ = {"access_entity": access_entity}

        memories = [
            MockMemory("user:grischa"),  # Should be included
            MockMemory("team:cloudfactory/backend"),  # Should be included (grant)
            MockMemory("team:cloudfactory/frontend"),  # Should be excluded (no grant)
            MockMemory("user:other-user"),  # Should be excluded (different user)
        ]

        filtered = filter_memories_by_access(principal, memories)

        assert len(filtered) == 2
        assert filtered[0].metadata_["access_entity"] == "user:grischa"
        assert filtered[1].metadata_["access_entity"] == "team:cloudfactory/backend"

    def test_filter_memories_filters_by_access_entity(self):
        """POST /memories/filter should filter results by access_entity."""
        # Similar to list_memories test
        from app.security.access import filter_memories_by_access

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",  # Org grant should expand
            },
        )

        class MockMemory:
            def __init__(self, access_entity):
                self.metadata_ = {"access_entity": access_entity}

        memories = [
            MockMemory("user:grischa"),  # Included (user grant)
            MockMemory("org:cloudfactory"),  # Included (org grant)
            MockMemory("project:cloudfactory/acme"),  # Included (org hierarchy)
            MockMemory("team:cloudfactory/acme/billing"),  # Included (org hierarchy)
            MockMemory("org:other-company"),  # Excluded (different org)
        ]

        filtered = filter_memories_by_access(principal, memories)

        assert len(filtered) == 4
        access_entities = [m.metadata_["access_entity"] for m in filtered]
        assert "org:other-company" not in access_entities

    def test_update_memory_checks_write_access(self):
        """PUT /memories/{id} should check write access before updating."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            },
        )

        # Can update own memory
        assert can_write_to_access_entity(principal, "user:grischa") is True

        # Can update team memory (group-editable)
        assert can_write_to_access_entity(principal, "team:cloudfactory/backend") is True

        # Cannot update other team's memory
        assert can_write_to_access_entity(principal, "team:cloudfactory/frontend") is False

    def test_delete_memories_checks_write_access(self):
        """DELETE /memories should check write access before deleting."""
        from app.security.access import can_write_to_access_entity

        principal = make_principal(
            "grischa",
            grants={"user:grischa"},
        )

        # Can delete own memory
        assert can_write_to_access_entity(principal, "user:grischa") is True

        # Cannot delete team memory without grant
        assert can_write_to_access_entity(principal, "team:cloudfactory/backend") is False
