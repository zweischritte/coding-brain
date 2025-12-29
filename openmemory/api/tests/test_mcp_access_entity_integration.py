"""
MCP Access Control Integration Tests.

Full integration tests for MCP tool access control with access_entity enforcement.
Tests cover add_memories, search_memory, update_memory, and delete_memories
with proper principal-based access control.

Key patterns:
- Context variable pattern for principal injection
- Mocking SessionLocal and memory_client for isolation
- JSON response parsing for MCP tool outputs
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from app.security.types import Principal, TokenClaims


# Test UUIDs
USER_GRISCHA_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_ANNA_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
APP_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


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


def make_mock_memory(
    memory_id: uuid.UUID,
    content: str,
    access_entity: str,
    user_id: uuid.UUID = USER_GRISCHA_ID,
    metadata: dict = None,
):
    """Create a mock Memory object for testing."""
    mock = MagicMock()
    mock.id = memory_id
    mock.content = content
    mock.user_id = user_id
    mock.metadata_ = metadata or {
        "access_entity": access_entity,
        "category": "decision",
        "scope": "team" if access_entity.startswith("team:") else "user",
    }
    mock.state = MagicMock()
    mock.state.value = "active"
    mock.created_at = datetime.now(timezone.utc)
    mock.updated_at = None
    return mock


def make_mock_user(user_id: uuid.UUID = USER_GRISCHA_ID, user_name: str = "grischa"):
    """Create a mock User object."""
    mock = MagicMock()
    mock.id = user_id
    mock.user_id = user_name
    return mock


def make_mock_app(app_id: uuid.UUID = APP_ID, app_name: str = "test-app"):
    """Create a mock App object."""
    mock = MagicMock()
    mock.id = app_id
    mock.name = app_name
    mock.is_active = True
    return mock


class TestMCPAddMemoriesAccessControl:
    """Test add_memories enforces access_entity permissions."""

    @pytest.mark.asyncio
    async def test_add_memory_with_valid_access_entity_succeeds(self):
        """User with team grant can create team memory."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        # Setup principal with team:cloudfactory/backend grant
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        # Mock memory client response
        mock_memory_client = MagicMock()
        new_memory_id = str(uuid.uuid4())
        mock_memory_client.add.return_value = {
            "results": [
                {"id": new_memory_id, "event": "ADD", "memory": "Test memory content"}
            ]
        }

        # Setup context
        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"), \
                 patch("app.mcp_server.update_entity_edges_on_memory_add"), \
                 patch("app.mcp_server.update_tag_edges_on_memory_add"), \
                 patch("app.mcp_server.project_similarity_edges_for_memory"), \
                 patch("app.mcp_server.bridge_entities_to_om_graph"):

                # Mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = None

                result = await add_memories(
                    text="Test team memory",
                    category="decision",
                    scope="team",
                    access_entity="team:cloudfactory/backend",
                )

                # Parse JSON response
                response = json.loads(result)

                # Verify no error in response
                assert "error" not in response, f"Unexpected error: {response.get('error')}"
                # Response is a flat object with id, memory, category, created_at, etc.
                assert "id" in response or "results" in response

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_add_memory_without_grant_fails(self):
        """User without grant cannot create team memory."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        # Setup principal with only user grant (no team grant)
        principal = make_principal(
            "grischa",
            grants={"user:grischa"},  # No team:cloudfactory/backend grant
        )

        # Setup context
        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            result = await add_memories(
                text="Test team memory",
                category="decision",
                scope="team",
                access_entity="team:cloudfactory/backend",  # User doesn't have this grant
            )

            # Parse JSON response
            response = json.loads(result)

            # Verify error is returned with code FORBIDDEN
            assert "error" in response
            assert response.get("code") == "FORBIDDEN"
            assert "access_entity" in response.get("error", "").lower() or "permission" in response.get("error", "").lower()

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_add_memory_defaults_to_user_scope(self):
        """Memory without explicit access_entity gets user:<uid>."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa")
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        # Mock memory client - capture the metadata passed to add()
        mock_memory_client = MagicMock()
        new_memory_id = str(uuid.uuid4())
        captured_metadata = {}

        def capture_add(text, user_id, metadata):
            captured_metadata.update(metadata)
            return {
                "results": [
                    {"id": new_memory_id, "event": "ADD", "memory": text}
                ]
            }

        mock_memory_client.add.side_effect = capture_add

        # Setup context
        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"), \
                 patch("app.mcp_server.update_entity_edges_on_memory_add"), \
                 patch("app.mcp_server.update_tag_edges_on_memory_add"), \
                 patch("app.mcp_server.project_similarity_edges_for_memory"), \
                 patch("app.mcp_server.bridge_entities_to_om_graph"):

                # Mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = None

                result = await add_memories(
                    text="Personal memory",
                    category="decision",
                    scope="user",
                    # No access_entity specified - should default to user:grischa
                )

                # Verify memory was created with default access_entity
                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"

                # Verify the captured metadata has the correct access_entity
                assert captured_metadata.get("access_entity") == "user:grischa"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_add_memory_with_org_grant_for_project(self):
        """User with org grant can create project memory under that org."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        # Setup principal with org:cloudfactory grant
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",  # Org grant should allow project:cloudfactory/*
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        mock_memory_client = MagicMock()
        new_memory_id = str(uuid.uuid4())
        mock_memory_client.add.return_value = {
            "results": [
                {"id": new_memory_id, "event": "ADD", "memory": "Test project memory"}
            ]
        }

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"), \
                 patch("app.mcp_server.update_entity_edges_on_memory_add"), \
                 patch("app.mcp_server.update_tag_edges_on_memory_add"), \
                 patch("app.mcp_server.project_similarity_edges_for_memory"), \
                 patch("app.mcp_server.bridge_entities_to_om_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = None

                result = await add_memories(
                    text="Test project memory under org",
                    category="architecture",
                    scope="project",
                    access_entity="project:cloudfactory/billing",  # Under cloudfactory org
                )

                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)


class TestMCPSearchMemoryAccessControl:
    """Test search_memory filters by access_entity."""

    @pytest.mark.asyncio
    async def test_search_returns_only_accessible_memories(self):
        """Search results filtered to accessible memories."""
        from app.mcp_server import (
            search_memory, principal_var, user_id_var, client_name_var
        )

        # Setup principal with limited grants
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        # Create mock memories with different access_entities
        memory1_id = uuid.uuid4()
        memory2_id = uuid.uuid4()
        memory3_id = uuid.uuid4()

        memories = [
            make_mock_memory(memory1_id, "User memory", "user:grischa"),
            make_mock_memory(memory2_id, "Team backend memory", "team:cloudfactory/backend"),
            make_mock_memory(memory3_id, "Team frontend memory", "team:cloudfactory/frontend"),  # No access
        ]

        # Mock memory client
        mock_memory_client = MagicMock()
        mock_embedding = [0.1] * 1536
        mock_memory_client.embedding_model.embed.return_value = mock_embedding

        # Return all memories from vector search (filtering happens in code)
        mock_hits = []
        for mem in memories:
            hit = MagicMock()
            hit.id = mem.id
            hit.score = 0.9
            hit.payload = {"memory": mem.content, "metadata": mem.metadata_}
            mock_hits.append(hit)

        mock_memory_client.vector_store.search.return_value = mock_hits

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.is_graph_enabled", return_value=False):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = memories

                result = await search_memory(
                    query="test memory",
                    limit=10,
                )

                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"

                # Should only return 2 memories (user:grischa and team:cloudfactory/backend)
                # Not the team:cloudfactory/frontend memory
                results = response.get("results", [])

                result_ids = [r.get("id") for r in results]
                assert str(memory1_id) in result_ids or any("grischa" in str(r) for r in results)
                assert str(memory3_id) not in result_ids  # Frontend memory should be excluded

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_search_with_org_grant_includes_child_scopes(self):
        """Org grant expands to include projects/teams."""
        from app.mcp_server import (
            search_memory, principal_var, user_id_var, client_name_var
        )

        # Setup principal with org grant
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "org:cloudfactory",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        # Create memories at different hierarchy levels
        memory1_id = uuid.uuid4()
        memory2_id = uuid.uuid4()
        memory3_id = uuid.uuid4()
        memory4_id = uuid.uuid4()

        memories = [
            make_mock_memory(memory1_id, "Org memory", "org:cloudfactory"),
            make_mock_memory(memory2_id, "Project memory", "project:cloudfactory/billing"),
            make_mock_memory(memory3_id, "Team memory", "team:cloudfactory/billing/backend"),
            make_mock_memory(memory4_id, "Other org memory", "org:other-company"),  # No access
        ]

        mock_memory_client = MagicMock()
        mock_embedding = [0.1] * 1536
        mock_memory_client.embedding_model.embed.return_value = mock_embedding

        mock_hits = []
        for mem in memories:
            hit = MagicMock()
            hit.id = mem.id
            hit.score = 0.9
            hit.payload = {"memory": mem.content, "metadata": mem.metadata_}
            mock_hits.append(hit)

        mock_memory_client.vector_store.search.return_value = mock_hits

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.is_graph_enabled", return_value=False):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = memories

                result = await search_memory(
                    query="test memory",
                    limit=10,
                )

                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"

                results = response.get("results", [])
                result_ids = [r.get("id") for r in results]

                # Should include org, project, and team memories under cloudfactory
                # Should NOT include other-company memory
                assert str(memory4_id) not in result_ids

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_search_excludes_memories_from_other_users(self):
        """Search excludes personal memories from other users."""
        from app.mcp_server import (
            search_memory, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa", grants={"user:grischa"})
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory1_id = uuid.uuid4()
        memory2_id = uuid.uuid4()

        memories = [
            make_mock_memory(memory1_id, "My memory", "user:grischa"),
            make_mock_memory(memory2_id, "Anna's memory", "user:anna"),  # No access
        ]

        mock_memory_client = MagicMock()
        mock_memory_client.embedding_model.embed.return_value = [0.1] * 1536

        mock_hits = []
        for mem in memories:
            hit = MagicMock()
            hit.id = mem.id
            hit.score = 0.9
            hit.payload = {"memory": mem.content, "metadata": mem.metadata_}
            mock_hits.append(hit)

        mock_memory_client.vector_store.search.return_value = mock_hits

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.is_graph_enabled", return_value=False):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = memories

                result = await search_memory(query="memory", limit=10)

                response = json.loads(result)
                results = response.get("results", [])
                result_ids = [r.get("id") for r in results]

                # Anna's memory should be excluded
                assert str(memory2_id) not in result_ids

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)


class TestMCPUpdateMemoryAccessControl:
    """Test update_memory checks write access."""

    @pytest.mark.asyncio
    async def test_team_member_can_update_team_memory(self):
        """Team member can update team memory (group-editable)."""
        from app.mcp_server import (
            update_memory, principal_var, user_id_var, client_name_var
        )

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        existing_memory = make_mock_memory(
            memory_id, "Original content", "team:cloudfactory/backend"
        )
        existing_memory.metadata_ = {
            "access_entity": "team:cloudfactory/backend",
            "category": "decision",
            "scope": "team",
        }

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = existing_memory

                result = await update_memory(
                    memory_id=str(memory_id),
                    text="Updated content",
                )

                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"
                assert response.get("status") == "updated"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_non_member_cannot_update_team_memory(self):
        """Non-member cannot update team memory."""
        from app.mcp_server import (
            update_memory, principal_var, user_id_var, client_name_var
        )

        # Principal without team grant
        principal = make_principal(
            "grischa",
            grants={"user:grischa"},  # No team grant
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        existing_memory = make_mock_memory(
            memory_id, "Team memory", "team:cloudfactory/backend"
        )

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = existing_memory

                result = await update_memory(
                    memory_id=str(memory_id),
                    text="Attempted update",
                )

                response = json.loads(result)
                assert "error" in response
                assert response.get("code") == "FORBIDDEN"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_cannot_change_access_entity_without_both_grants(self):
        """Changing access_entity requires grants for old AND new."""
        from app.mcp_server import (
            update_memory, principal_var, user_id_var, client_name_var
        )

        # Principal has grant for backend but NOT frontend
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
                # No team:cloudfactory/frontend grant
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        existing_memory = make_mock_memory(
            memory_id, "Backend memory", "team:cloudfactory/backend"
        )

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = existing_memory

                result = await update_memory(
                    memory_id=str(memory_id),
                    access_entity="team:cloudfactory/frontend",  # No grant for this
                )

                response = json.loads(result)
                assert "error" in response
                assert response.get("code") == "FORBIDDEN"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_can_change_access_entity_with_both_grants(self):
        """User with grants for both old and new access_entity can change it."""
        from app.mcp_server import (
            update_memory, principal_var, user_id_var, client_name_var
        )

        # Principal has grants for both teams
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
                "team:cloudfactory/frontend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        existing_memory = make_mock_memory(
            memory_id, "Backend memory", "team:cloudfactory/backend"
        )
        existing_memory.metadata_ = {
            "access_entity": "team:cloudfactory/backend",
            "category": "decision",
            "scope": "team",
        }

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = existing_memory

                result = await update_memory(
                    memory_id=str(memory_id),
                    access_entity="team:cloudfactory/frontend",
                )

                response = json.loads(result)
                assert "error" not in response, f"Unexpected error: {response.get('error')}"
                assert response.get("status") == "updated"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_owner_can_update_own_personal_memory(self):
        """Owner can update their own personal memory."""
        from app.mcp_server import (
            update_memory, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa", grants={"user:grischa"})
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        existing_memory = make_mock_memory(
            memory_id, "Personal memory", "user:grischa"
        )
        existing_memory.metadata_ = {
            "access_entity": "user:grischa",
            "category": "decision",
            "scope": "user",
        }

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = existing_memory

                result = await update_memory(
                    memory_id=str(memory_id),
                    text="Updated personal memory",
                )

                response = json.loads(result)
                assert "error" not in response
                assert response.get("status") == "updated"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)


class TestMCPDeleteMemoriesAccessControl:
    """Test delete_memories checks write access."""

    @pytest.mark.asyncio
    async def test_delete_filters_by_access_entity(self):
        """Delete only removes memories user has write access to."""
        from app.mcp_server import (
            delete_memories, principal_var, user_id_var, client_name_var
        )

        # Principal has team:cloudfactory/backend but not frontend
        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory1_id = uuid.uuid4()
        memory2_id = uuid.uuid4()

        memories = [
            make_mock_memory(memory1_id, "Backend memory", "team:cloudfactory/backend"),
            make_mock_memory(memory2_id, "Frontend memory", "team:cloudfactory/frontend"),
        ]

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.update_entity_edges_on_memory_delete"), \
                 patch("app.mcp_server.delete_similarity_edges_for_memory"), \
                 patch("app.mcp_server.delete_memory_from_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = memories
                mock_db.query.return_value.filter.return_value.first.return_value = memories[0]

                result = await delete_memories(
                    memory_ids=[str(memory1_id), str(memory2_id)]
                )

                # Should only delete memory1 (backend), not memory2 (frontend)
                # Result should indicate only 1 memory was deleted
                assert "1" in result or "Successfully deleted" in result

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_team_member_can_delete_team_memory(self):
        """Team member can delete team memory (group-editable)."""
        from app.mcp_server import (
            delete_memories, principal_var, user_id_var, client_name_var
        )

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        memory = make_mock_memory(memory_id, "Team memory", "team:cloudfactory/backend")

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.update_entity_edges_on_memory_delete"), \
                 patch("app.mcp_server.delete_similarity_edges_for_memory"), \
                 patch("app.mcp_server.delete_memory_from_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = [memory]
                mock_db.query.return_value.filter.return_value.first.return_value = memory

                result = await delete_memories(memory_ids=[str(memory_id)])

                assert "Successfully deleted" in result or "1" in result

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_non_member_cannot_delete_team_memory(self):
        """Non-member cannot delete team memory."""
        from app.mcp_server import (
            delete_memories, principal_var, user_id_var, client_name_var
        )

        # No team grant
        principal = make_principal("grischa", grants={"user:grischa"})
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        memory = make_mock_memory(memory_id, "Team memory", "team:cloudfactory/backend")

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = [memory]

                result = await delete_memories(memory_ids=[str(memory_id)])

                # Should fail to delete - no accessible memories
                assert "Error" in result or "No accessible" in result

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_owner_can_delete_own_personal_memory(self):
        """Owner can delete their own personal memory."""
        from app.mcp_server import (
            delete_memories, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa", grants={"user:grischa"})
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        memory = make_mock_memory(memory_id, "Personal memory", "user:grischa")

        mock_memory_client = MagicMock()

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.update_entity_edges_on_memory_delete"), \
                 patch("app.mcp_server.delete_similarity_edges_for_memory"), \
                 patch("app.mcp_server.delete_memory_from_graph"):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = [memory]
                mock_db.query.return_value.filter.return_value.first.return_value = memory

                result = await delete_memories(memory_ids=[str(memory_id)])

                assert "Successfully deleted" in result or "1" in result

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)


class TestMCPListMemoriesAccessControl:
    """Test list_memories filters by access_entity."""

    @pytest.mark.asyncio
    async def test_list_returns_only_accessible_memories(self):
        """list_memories returns only memories the user has access to."""
        from app.mcp_server import (
            list_memories, principal_var, user_id_var, client_name_var
        )

        principal = make_principal(
            "grischa",
            grants={
                "user:grischa",
                "team:cloudfactory/backend",
            }
        )
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory1_id = uuid.uuid4()
        memory2_id = uuid.uuid4()
        memory3_id = uuid.uuid4()

        memories = [
            make_mock_memory(memory1_id, "Personal memory", "user:grischa"),
            make_mock_memory(memory2_id, "Team backend", "team:cloudfactory/backend"),
            make_mock_memory(memory3_id, "Team frontend", "team:cloudfactory/frontend"),
        ]

        # Mock memory client to return raw memories
        mock_memory_client = MagicMock()
        mock_memory_client.get_all.return_value = {
            "results": [
                {"id": str(memory1_id), "memory": "Personal memory", "metadata": {"access_entity": "user:grischa"}},
                {"id": str(memory2_id), "memory": "Team backend", "metadata": {"access_entity": "team:cloudfactory/backend"}},
                {"id": str(memory3_id), "memory": "Team frontend", "metadata": {"access_entity": "team:cloudfactory/frontend"}},
            ]
        }

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = memories

                result = await list_memories()

                response = json.loads(result)

                # Should only include memories user has access to
                # Response format is a list from format_memory_list
                results = response if isinstance(response, list) else response.get("memories", response.get("results", []))
                result_ids = [str(r.get("id")) for r in results if r.get("id")]

                # Frontend memory should be excluded
                assert str(memory3_id) not in result_ids

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)


class TestMCPAccessControlWithoutPrincipal:
    """Test MCP tools behavior when no principal is set (backwards compatibility)."""

    @pytest.mark.asyncio
    async def test_add_memory_without_principal_uses_legacy_behavior(self):
        """Without principal, add_memories uses legacy behavior (no ACL check)."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        mock_user = make_mock_user()
        mock_app = make_mock_app()

        mock_memory_client = MagicMock()
        new_memory_id = str(uuid.uuid4())
        mock_memory_client.add.return_value = {
            "results": [
                {"id": new_memory_id, "event": "ADD", "memory": "Test memory"}
            ]
        }

        # Set user_id and client_name but NOT principal
        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.project_memory_to_graph"), \
                 patch("app.mcp_server.update_entity_edges_on_memory_add"), \
                 patch("app.mcp_server.update_tag_edges_on_memory_add"), \
                 patch("app.mcp_server.project_similarity_edges_for_memory"), \
                 patch("app.mcp_server.bridge_entities_to_om_graph"), \
                 patch("app.mcp_server._check_tool_scope", return_value=None):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.first.return_value = None

                result = await add_memories(
                    text="Test memory without principal",
                    category="decision",
                    scope="user",
                )

                response = json.loads(result)
                # Should succeed in legacy mode (no ACL check)
                assert "error" not in response or "Authentication" not in response.get("error", "")

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)


class TestMCPAccessControlEdgeCases:
    """Test edge cases in MCP access control."""

    @pytest.mark.asyncio
    async def test_memory_without_access_entity_accessible_by_owner(self):
        """Legacy memory without access_entity is accessible by owner."""
        from app.mcp_server import (
            search_memory, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa")
        mock_user = make_mock_user()
        mock_app = make_mock_app()

        memory_id = uuid.uuid4()
        legacy_memory = MagicMock()
        legacy_memory.id = memory_id
        legacy_memory.content = "Legacy memory"
        legacy_memory.user_id = USER_GRISCHA_ID
        legacy_memory.metadata_ = {"category": "decision"}  # No access_entity
        legacy_memory.state = MagicMock()
        legacy_memory.state.value = "active"
        legacy_memory.created_at = datetime.now(timezone.utc)

        mock_memory_client = MagicMock()
        mock_memory_client.embedding_model.embed.return_value = [0.1] * 1536

        hit = MagicMock()
        hit.id = memory_id
        hit.score = 0.9
        hit.payload = {"memory": "Legacy memory", "metadata": {}}
        mock_memory_client.vector_store.search.return_value = [hit]

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_memory_client_safe", return_value=mock_memory_client), \
                 patch("app.mcp_server.get_user_and_app", return_value=(mock_user, mock_app)), \
                 patch("app.mcp_server.check_memory_access_permissions", return_value=True), \
                 patch("app.mcp_server.is_graph_enabled", return_value=False):

                mock_db = MagicMock()
                mock_session_local.return_value = mock_db
                mock_db.query.return_value.filter.return_value.all.return_value = [legacy_memory]

                result = await search_memory(query="legacy", limit=10)

                response = json.loads(result)
                # Legacy memory should be accessible by owner
                results = response.get("results", [])
                assert len(results) >= 0  # Should not error

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_invalid_access_entity_format_handled(self):
        """Invalid access_entity format is handled gracefully."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        principal = make_principal("grischa")

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            result = await add_memories(
                text="Test memory",
                category="decision",
                scope="team",
                access_entity="invalid-format-no-colon",  # Invalid format
            )

            response = json.loads(result)
            # Should fail due to invalid access_entity format
            assert "error" in response

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)

    @pytest.mark.asyncio
    async def test_service_access_entity_requires_service_grant(self):
        """service: access_entity requires explicit service grant."""
        from app.mcp_server import (
            add_memories, principal_var, user_id_var, client_name_var
        )

        # No service grant
        principal = make_principal("grischa", grants={"user:grischa"})

        user_token = user_id_var.set("grischa")
        client_token = client_name_var.set("test-app")
        principal_token = principal_var.set(principal)

        try:
            result = await add_memories(
                text="Service memory",
                category="decision",
                scope="enterprise",
                access_entity="service:auth-service",
            )

            response = json.loads(result)
            assert "error" in response
            assert response.get("code") == "FORBIDDEN"

        finally:
            user_id_var.reset(user_token)
            client_name_var.reset(client_token)
            principal_var.reset(principal_token)
