"""
Tests for get_memory MCP Tool.

TDD: These tests define the expected behavior for the get_memory MCP tool.
The tool should retrieve a single memory by UUID, returning all fields
including access_entity.

Scope requirements (from PRD):
- get_memory: memories:read

Test scenarios:
1. test_get_memory_tool_registered - Tool should be registered with MCP
2. test_get_memory_requires_memories_read_scope - Returns error if scope missing
3. test_get_memory_returns_all_fields - Returns complete memory with all fields
4. test_get_memory_not_found - Returns proper error for non-existent memory
5. test_get_memory_invalid_uuid - Returns error for invalid UUID format
6. test_get_memory_access_denied - Returns NOT_FOUND for access-denied (security)
7. test_get_memory_legacy_owner_only - Legacy memories only accessible by owner
8. test_get_memory_team_access - Team members can access team memories
9. test_get_memory_response_format - Response has lean JSON format
"""

import json
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_principal_no_scopes():
    """Create a mock principal with no scopes."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=False)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {"user:test-user"}
    return principal


@pytest.fixture
def mock_principal_memories_read():
    """Create a mock principal with memories:read scope."""
    principal = MagicMock()
    principal.has_scope = MagicMock(
        side_effect=lambda s: s == "memories:read"
    )
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {"user:test-user"}
    return principal


@pytest.fixture
def mock_principal_all_scopes():
    """Create a mock principal with all required scopes."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=True)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {"user:test-user"}
    return principal


@pytest.fixture
def mock_principal_team_grant():
    """Create a mock principal with team grant."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=True)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {
        "user:test-user",
        "team:test-org/backend",
    }
    return principal


@pytest.fixture
def mock_principal_other_user():
    """Create a mock principal for a different user."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=True)
    principal.user_id = "other-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {"user:other-user"}
    return principal


@pytest.fixture
def sample_memory_id():
    """Sample memory UUID for testing."""
    return "ba93af28-784d-4262-b8f9-adb08c45acab"


@pytest.fixture
def sample_memory_data():
    """Sample memory data as returned from database."""
    return {
        "id": "ba93af28-784d-4262-b8f9-adb08c45acab",
        "content": "Always run pytest before merge",
        "metadata_": {
            "category": "workflow",
            "scope": "project",
            "artifact_type": "repo",
            "artifact_ref": "coding-brain",
            "entity": "Backend Team",
            "source": "user",
            "tags": {"decision": True, "priority": "high"},
            "evidence": ["ADR-014", "PR-456"],
            "access_entity": "project:default_org/coding-brain",
        },
        "created_at": datetime(2026, 1, 3, 10, 22, 3, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 3, 11, 30, 0, tzinfo=timezone.utc),
        "state": "active",
    }


@pytest.fixture
def sample_legacy_memory_data():
    """Sample legacy memory without access_entity."""
    return {
        "id": "ca93af28-784d-4262-b8f9-adb08c45acac",
        "content": "Personal note for owner only",
        "metadata_": {
            "category": "workflow",
            "scope": "user",
        },
        "created_at": datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        "updated_at": None,
        "state": "active",
    }


@pytest.fixture
def sample_team_memory_data():
    """Sample team memory with team access_entity."""
    return {
        "id": "da93af28-784d-4262-b8f9-adb08c45acad",
        "content": "Team coding convention",
        "metadata_": {
            "category": "convention",
            "scope": "team",
            "entity": "Backend Team",
            "access_entity": "team:test-org/backend",
        },
        "created_at": datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        "updated_at": None,
        "state": "active",
    }


# =============================================================================
# Registration Tests
# =============================================================================


class TestGetMemoryMCPRegistration:
    """Test that get_memory MCP tool is properly registered."""

    def _get_registered_tool_names(self):
        """Get list of registered tool names from MCP server."""
        from app.mcp_server import mcp
        # Access internal tool manager registry (sync access)
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            return list(mcp._tool_manager._tools.keys())
        return []

    def test_get_memory_tool_registered(self):
        """get_memory tool should be registered."""
        try:
            tool_names = self._get_registered_tool_names()
            assert "get_memory" in tool_names, (
                f"get_memory not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MCP server tools not available: {e}")


# =============================================================================
# Scope Check Tests
# =============================================================================


class TestGetMemoryScopeChecks:
    """Test that get_memory MCP tool enforces scope requirements."""

    @pytest.mark.asyncio
    async def test_get_memory_requires_memories_read(
        self, mock_principal_no_scopes, sample_memory_id
    ):
        """get_memory should require memories:read scope."""
        try:
            from app.mcp_server import get_memory, principal_var

            token = principal_var.set(mock_principal_no_scopes)
            try:
                result = await get_memory(memory_id=sample_memory_id)
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
                assert result_data.get("code") == "INSUFFICIENT_SCOPE" or \
                       "scope" in result_data.get("error", "").lower()
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_allows_memories_read_scope(
        self, mock_principal_memories_read, sample_memory_id
    ):
        """get_memory should work with memories:read scope."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory not found (to trigger expected error path)
                mock_db.query.return_value.filter.return_value.first.return_value = None

                token = principal_var.set(mock_principal_memories_read)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        error_msg = result_data.get("error", "").lower()
                        assert "scope" not in error_msg
                        assert "memories:read" not in error_msg
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestGetMemoryInputValidation:
    """Test input validation for get_memory MCP tool."""

    @pytest.mark.asyncio
    async def test_get_memory_invalid_uuid_format(
        self, mock_principal_all_scopes
    ):
        """get_memory should return error for invalid UUID format."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            token = principal_var.set(mock_principal_all_scopes)
            user_token = user_id_var.set("test-user")
            client_token = client_name_var.set("test-client")
            try:
                result = await get_memory(memory_id="not-a-valid-uuid")
                result_data = json.loads(result)

                # Should return error for invalid UUID
                assert "error" in result_data
                assert (
                    result_data.get("code") == "INVALID_INPUT" or
                    "invalid" in result_data.get("error", "").lower() or
                    "uuid" in result_data.get("error", "").lower()
                )
            finally:
                principal_var.reset(token)
                user_id_var.reset(user_token)
                client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_requires_user_id(
        self, mock_principal_all_scopes, sample_memory_id
    ):
        """get_memory should require user_id context variable."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            token = principal_var.set(mock_principal_all_scopes)
            # Don't set user_id_var
            client_token = client_name_var.set("test-client")
            try:
                result = await get_memory(memory_id=sample_memory_id)
                result_data = json.loads(result)

                # Should return error for missing user_id
                assert "error" in result_data
                assert "user_id" in result_data.get("error", "").lower()
            finally:
                principal_var.reset(token)
                client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Not Found Tests
# =============================================================================


class TestGetMemoryNotFound:
    """Test get_memory behavior for non-existent memories."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_not_found(
        self, mock_principal_all_scopes, sample_memory_id
    ):
        """get_memory should return NOT_FOUND for non-existent memory."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory not found
                mock_db.query.return_value.filter.return_value.first.return_value = None

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Should return NOT_FOUND error
                    assert "error" in result_data
                    assert (
                        result_data.get("code") == "NOT_FOUND" or
                        "not found" in result_data.get("error", "").lower()
                    )
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Access Control Tests
# =============================================================================


class TestGetMemoryAccessControl:
    """Test get_memory access control behavior."""

    @pytest.mark.asyncio
    async def test_get_memory_access_denied_returns_not_found(
        self, mock_principal_other_user, sample_memory_id, sample_memory_data
    ):
        """get_memory should return NOT_FOUND for access-denied (security)."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory exists but access denied
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_memory_id)
                mock_memory.content = sample_memory_data["content"]
                mock_memory.metadata_ = sample_memory_data["metadata_"]
                mock_memory.created_at = sample_memory_data["created_at"]
                mock_memory.updated_at = sample_memory_data["updated_at"]
                mock_memory.user_id = uuid.uuid4()  # Different user
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                # Access denied
                mock_can_read.return_value = False

                token = principal_var.set(mock_principal_other_user)
                user_token = user_id_var.set("other-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Should return NOT_FOUND (not FORBIDDEN) for security
                    assert "error" in result_data
                    assert (
                        result_data.get("code") == "NOT_FOUND" or
                        "not found" in result_data.get("error", "").lower()
                    )
                    # Should NOT reveal that memory exists
                    assert result_data.get("code") != "FORBIDDEN"
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_legacy_owner_only(
        self, mock_principal_other_user, sample_legacy_memory_data
    ):
        """Legacy memories without access_entity should only be accessible by owner."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()  # other-user's DB id
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Legacy memory exists, owned by different user
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_legacy_memory_data["id"])
                mock_memory.content = sample_legacy_memory_data["content"]
                mock_memory.metadata_ = sample_legacy_memory_data["metadata_"]  # No access_entity
                mock_memory.created_at = sample_legacy_memory_data["created_at"]
                mock_memory.updated_at = sample_legacy_memory_data["updated_at"]
                mock_memory.user_id = uuid.uuid4()  # Different user owns it
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                token = principal_var.set(mock_principal_other_user)
                user_token = user_id_var.set("other-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_legacy_memory_data["id"])
                    result_data = json.loads(result)

                    # Should return NOT_FOUND (legacy = owner only)
                    assert "error" in result_data
                    assert (
                        result_data.get("code") == "NOT_FOUND" or
                        "not found" in result_data.get("error", "").lower()
                    )
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_team_access_allowed(
        self, mock_principal_team_grant, sample_team_memory_data
    ):
        """Team members should be able to access team memories."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_app.is_active = True
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Team memory exists
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_team_memory_data["id"])
                mock_memory.content = sample_team_memory_data["content"]
                mock_memory.metadata_ = sample_team_memory_data["metadata_"]
                mock_memory.created_at = sample_team_memory_data["created_at"]
                mock_memory.updated_at = sample_team_memory_data["updated_at"]
                mock_memory.user_id = uuid.uuid4()  # Different user created it
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                # Access allowed via team grant
                mock_can_read.return_value = True

                token = principal_var.set(mock_principal_team_grant)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_team_memory_data["id"])
                    result_data = json.loads(result)

                    # Should return the memory (not an error)
                    assert "error" not in result_data
                    assert result_data.get("id") == sample_team_memory_data["id"]
                    assert result_data.get("memory") == sample_team_memory_data["content"]
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Response Format Tests
# =============================================================================


class TestGetMemoryResponseFormat:
    """Test get_memory returns properly formatted JSON responses."""

    @pytest.mark.asyncio
    async def test_get_memory_returns_json_string(
        self, mock_principal_all_scopes, sample_memory_id, sample_memory_data
    ):
        """get_memory should return a JSON string."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory exists and accessible
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_memory_id)
                mock_memory.content = sample_memory_data["content"]
                mock_memory.metadata_ = sample_memory_data["metadata_"]
                mock_memory.created_at = sample_memory_data["created_at"]
                mock_memory.updated_at = sample_memory_data["updated_at"]
                mock_memory.user_id = mock_user.id
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                mock_can_read.return_value = True

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)

                    # Result should be a string
                    assert isinstance(result, str)

                    # Result should be valid JSON
                    parsed = json.loads(result)
                    assert isinstance(parsed, dict)
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_returns_all_fields(
        self, mock_principal_all_scopes, sample_memory_id, sample_memory_data
    ):
        """get_memory should return all memory fields including access_entity."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory exists and accessible
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_memory_id)
                mock_memory.content = sample_memory_data["content"]
                mock_memory.metadata_ = sample_memory_data["metadata_"]
                mock_memory.created_at = sample_memory_data["created_at"]
                mock_memory.updated_at = sample_memory_data["updated_at"]
                mock_memory.user_id = mock_user.id
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                mock_can_read.return_value = True

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Required fields
                    assert "id" in result_data
                    assert result_data["id"] == sample_memory_id
                    assert "memory" in result_data
                    assert result_data["memory"] == sample_memory_data["content"]

                    # Structured metadata fields
                    assert "category" in result_data
                    assert result_data["category"] == "workflow"
                    assert "scope" in result_data
                    assert result_data["scope"] == "project"

                    # Optional fields (when present)
                    assert "artifact_type" in result_data
                    assert result_data["artifact_type"] == "repo"
                    assert "artifact_ref" in result_data
                    assert result_data["artifact_ref"] == "coding-brain"
                    assert "entity" in result_data
                    assert result_data["entity"] == "Backend Team"
                    assert "source" in result_data
                    assert result_data["source"] == "user"

                    # Tags and evidence
                    assert "tags" in result_data
                    assert result_data["tags"]["decision"] is True
                    assert "evidence" in result_data
                    assert "ADR-014" in result_data["evidence"]

                    # CRITICAL: access_entity must be included
                    assert "access_entity" in result_data
                    assert result_data["access_entity"] == "project:default_org/coding-brain"

                    # Timestamps
                    assert "created_at" in result_data
                    assert "updated_at" in result_data
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_timestamps_berlin_timezone(
        self, mock_principal_all_scopes, sample_memory_id, sample_memory_data
    ):
        """get_memory timestamps should be in Berlin timezone."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory exists and accessible
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_memory_id)
                mock_memory.content = sample_memory_data["content"]
                mock_memory.metadata_ = sample_memory_data["metadata_"]
                mock_memory.created_at = sample_memory_data["created_at"]
                mock_memory.updated_at = sample_memory_data["updated_at"]
                mock_memory.user_id = mock_user.id
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                mock_can_read.return_value = True

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Timestamps should contain Berlin timezone offset (+01:00 or +02:00)
                    created_at = result_data.get("created_at", "")
                    assert "+01:00" in created_at or "+02:00" in created_at, (
                        f"created_at should be in Berlin timezone: {created_at}"
                    )

                    if result_data.get("updated_at"):
                        updated_at = result_data["updated_at"]
                        assert "+01:00" in updated_at or "+02:00" in updated_at, (
                            f"updated_at should be in Berlin timezone: {updated_at}"
                        )
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_no_score_field(
        self, mock_principal_all_scopes, sample_memory_id, sample_memory_data
    ):
        """get_memory should NOT include score field (unlike search_memory)."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
                 patch("app.security.access.can_read_access_entity") as mock_can_read:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory exists and accessible
                mock_memory = MagicMock()
                mock_memory.id = uuid.UUID(sample_memory_id)
                mock_memory.content = sample_memory_data["content"]
                mock_memory.metadata_ = sample_memory_data["metadata_"]
                mock_memory.created_at = sample_memory_data["created_at"]
                mock_memory.updated_at = sample_memory_data["updated_at"]
                mock_memory.user_id = mock_user.id
                mock_db.query.return_value.filter.return_value.first.return_value = mock_memory

                mock_can_read.return_value = True

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id=sample_memory_id)
                    result_data = json.loads(result)

                    # Should NOT have score field (that's for search results)
                    assert "score" not in result_data
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Error Response Tests
# =============================================================================


class TestGetMemoryErrorResponses:
    """Test get_memory error response format."""

    @pytest.mark.asyncio
    async def test_get_memory_error_has_code(
        self, mock_principal_all_scopes
    ):
        """get_memory errors should include error code."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var

            with patch("app.mcp_server.SessionLocal") as mock_session_local, \
                 patch("app.mcp_server.get_user_and_app") as mock_get_user_app:

                # Set up mock database session
                mock_db = MagicMock()
                mock_session_local.return_value = mock_db

                # Mock user and app
                mock_user = MagicMock()
                mock_user.id = uuid.uuid4()
                mock_app = MagicMock()
                mock_app.id = uuid.uuid4()
                mock_get_user_app.return_value = (mock_user, mock_app)

                # Memory not found
                mock_db.query.return_value.filter.return_value.first.return_value = None

                token = principal_var.set(mock_principal_all_scopes)
                user_token = user_id_var.set("test-user")
                client_token = client_name_var.set("test-client")
                try:
                    result = await get_memory(memory_id="ba93af28-784d-4262-b8f9-adb08c45acab")
                    result_data = json.loads(result)

                    # Error response should have error and code
                    assert "error" in result_data
                    assert "code" in result_data
                    assert result_data["code"] == "NOT_FOUND"
                finally:
                    principal_var.reset(token)
                    user_id_var.reset(user_token)
                    client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_get_memory_missing_auth_error(self):
        """get_memory should return MISSING_AUTH when not authenticated."""
        try:
            from app.mcp_server import get_memory, principal_var, user_id_var, client_name_var, HAS_SECURITY

            if not HAS_SECURITY:
                pytest.skip("Security not enabled")

            # Don't set principal (no auth)
            user_token = user_id_var.set("test-user")
            client_token = client_name_var.set("test-client")
            try:
                result = await get_memory(memory_id="ba93af28-784d-4262-b8f9-adb08c45acab")
                result_data = json.loads(result)

                # Should return MISSING_AUTH error
                assert "error" in result_data
                assert result_data.get("code") == "MISSING_AUTH"
            finally:
                user_id_var.reset(user_token)
                client_name_var.reset(client_token)

        except ImportError:
            pytest.skip("MCP tools not available")
