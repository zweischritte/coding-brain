"""
Tests for add_memories MCP Tool with code_refs parameter.

TDD: These tests define the expected behavior for the code_refs parameter
in the add_memories MCP tool, implementing Phase 1 of Code-Linked Memories.

The code_refs parameter allows linking memories to specific code locations:
- file_path: path to source file
- line_start/line_end: line range
- symbol_id: SCIP-compatible symbol identifier
- git_commit, code_hash: versioning info

Storage: Phase 1 uses tags-based storage (no schema migration).

Test scenarios:
1. test_add_memories_with_code_refs - Basic code_refs parameter works
2. test_add_memories_code_refs_serialized_to_tags - code_refs stored in tags
3. test_add_memories_code_refs_invalid_format - Invalid format rejected
4. test_add_memories_code_refs_multiple - Multiple code_refs supported
5. test_add_memories_code_refs_preserves_existing_tags - Merges with tags
6. test_add_memories_code_refs_minimal - Minimal code_ref (file_path only)
"""

import json
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from app.utils.code_reference import (
    serialize_code_refs_to_tags,
    deserialize_code_refs_from_tags,
    CodeReference,
    LineRange,
)


# =============================================================================
# Test Fixtures
# =============================================================================


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
def sample_code_refs():
    """Sample code references for testing."""
    return [
        {
            "file_path": "/apps/merlin/src/storage/storage.service.ts",
            "line_start": 42,
            "line_end": 87,
            "symbol_id": "StorageService#createFileUploads",
            "git_commit": "abc123def",
            "code_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
    ]


@pytest.fixture
def sample_multiple_code_refs():
    """Multiple code references for testing."""
    return [
        {
            "file_path": "/apps/merlin/src/storage.ts",
            "line_start": 42,
            "line_end": 87,
            "symbol_id": "StorageService#createFileUploads",
        },
        {
            "file_path": "/apps/merlin/src/storage.ts",
            "line_start": 120,
            "line_end": 145,
            "symbol_id": "StorageService#moveFilesToPermanentStorage",
        },
    ]


# =============================================================================
# Registration Tests
# =============================================================================


class TestAddMemoriesCodeRefsRegistration:
    """Test that add_memories accepts code_refs parameter."""

    def _get_tool_function(self, tool_name):
        """Get the actual tool function from MCP server."""
        from app.mcp_server import mcp
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            tool = mcp._tool_manager._tools.get(tool_name)
            return tool
        return None

    def test_add_memories_accepts_code_refs_parameter(self):
        """add_memories tool should accept code_refs parameter."""
        try:
            tool = self._get_tool_function("add_memories")
            if tool is None:
                pytest.skip("add_memories tool not found")

            # Check if the function accepts code_refs parameter
            import inspect
            sig = inspect.signature(tool.fn)
            params = list(sig.parameters.keys())
            assert "code_refs" in params, (
                f"code_refs parameter not found. Available params: {params}"
            )
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MCP server tools not available: {e}")


# =============================================================================
# Code Reference Validation Tests
# =============================================================================


class TestAddMemoriesCodeRefsValidation:
    """Test code_refs parameter validation."""

    @pytest.mark.asyncio
    async def test_add_memories_code_refs_invalid_format_rejected(
        self, mock_principal_all_scopes
    ):
        """Invalid code_refs format should be rejected."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        # Set context
        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        # Test with invalid code_refs (no file_path or symbol_id)
        result_json = await add_memories(
            text="Test memory",
            category="architecture",
            scope="user",
            code_refs=[{"invalid_field": "value"}],
        )
        result = json.loads(result_json)

        assert "error" in result
        assert result.get("code") == "INVALID_CODE_REFS"

    @pytest.mark.asyncio
    async def test_add_memories_code_refs_invalid_type_rejected(
        self, mock_principal_all_scopes
    ):
        """Non-list code_refs should be rejected."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        result_json = await add_memories(
            text="Test memory",
            category="architecture",
            scope="user",
            code_refs="not a list",
        )
        result = json.loads(result_json)

        assert "error" in result
        assert result.get("code") == "INVALID_CODE_REFS"


# =============================================================================
# Code Reference Serialization Tests
# =============================================================================


class TestAddMemoriesCodeRefsSerialization:
    """Test that code_refs are properly serialized to tags."""

    @pytest.mark.asyncio
    async def test_add_memories_code_refs_serialized_to_tags(
        self, mock_principal_all_scopes, sample_code_refs
    ):
        """code_refs should be serialized into tags for storage."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        # Mock memory_client.add to capture the metadata
        captured_metadata = {}

        def capture_add(text, user_id=None, metadata=None, infer=False):
            captured_metadata.update(metadata or {})
            return {
                "results": [{
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            # Setup mocks
            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=capture_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            # Call add_memories with code_refs
            result_json = await add_memories(
                text="createFileUploads uses S3 MultipartUpload",
                category="architecture",
                scope="user",
                entity="StorageService",
                code_refs=sample_code_refs,
            )

            # Verify code_refs are in tags
            tags = captured_metadata.get("tags", {})
            assert tags.get("code_ref_count") == 1
            assert tags.get("code_ref_0_path") == sample_code_refs[0]["file_path"]
            assert tags.get("code_ref_0_lines") == "42-87"
            assert tags.get("code_ref_0_symbol") == sample_code_refs[0]["symbol_id"]
            assert tags.get("code_ref_0_hash") == sample_code_refs[0]["code_hash"]
            assert tags.get("code_ref_0_commit") == sample_code_refs[0]["git_commit"]

    @pytest.mark.asyncio
    async def test_add_memories_multiple_code_refs(
        self, mock_principal_all_scopes, sample_multiple_code_refs
    ):
        """Multiple code_refs should be serialized correctly."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        captured_metadata = {}

        def capture_add(text, user_id=None, metadata=None, infer=False):
            captured_metadata.update(metadata or {})
            return {
                "results": [{
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=capture_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            await add_memories(
                text="Storage service file operations",
                category="architecture",
                scope="user",
                code_refs=sample_multiple_code_refs,
            )

            tags = captured_metadata.get("tags", {})
            assert tags.get("code_ref_count") == 2
            assert tags.get("code_ref_0_path") == sample_multiple_code_refs[0]["file_path"]
            assert tags.get("code_ref_1_path") == sample_multiple_code_refs[1]["file_path"]

    @pytest.mark.asyncio
    async def test_add_memories_code_refs_preserves_existing_tags(
        self, mock_principal_all_scopes, sample_code_refs
    ):
        """code_refs should be merged with existing tags, not replace them."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        captured_metadata = {}

        def capture_add(text, user_id=None, metadata=None, infer=False):
            captured_metadata.update(metadata or {})
            return {
                "results": [{
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=capture_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            # Call with both tags and code_refs
            await add_memories(
                text="Storage service with tags",
                category="architecture",
                scope="user",
                tags={"decision": True, "priority": "high"},
                code_refs=sample_code_refs,
            )

            tags = captured_metadata.get("tags", {})
            # Original tags preserved
            assert tags.get("decision") is True
            assert tags.get("priority") == "high"
            # Code refs also present
            assert tags.get("code_ref_count") == 1
            assert tags.get("code_ref_0_path") == sample_code_refs[0]["file_path"]


# =============================================================================
# Minimal Code Reference Tests
# =============================================================================


class TestAddMemoriesCodeRefsMinimal:
    """Test minimal code_ref support (file_path only)."""

    @pytest.mark.asyncio
    async def test_add_memories_minimal_code_ref(
        self, mock_principal_all_scopes
    ):
        """Minimal code_ref with only file_path should work."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        captured_metadata = {}

        def capture_add(text, user_id=None, metadata=None, infer=False):
            captured_metadata.update(metadata or {})
            return {
                "results": [{
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=capture_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            # Minimal code_ref - only file_path
            await add_memories(
                text="Some code reference",
                category="architecture",
                scope="user",
                code_refs=[{"file_path": "/src/app.ts"}],
            )

            tags = captured_metadata.get("tags", {})
            assert tags.get("code_ref_count") == 1
            assert tags.get("code_ref_0_path") == "/src/app.ts"
            # No lines should be present
            assert "code_ref_0_lines" not in tags

    @pytest.mark.asyncio
    async def test_add_memories_symbol_only_code_ref(
        self, mock_principal_all_scopes
    ):
        """code_ref with only symbol_id should work."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        captured_metadata = {}

        def capture_add(text, user_id=None, metadata=None, infer=False):
            captured_metadata.update(metadata or {})
            return {
                "results": [{
                    "id": str(uuid.uuid4()),
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=capture_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            await add_memories(
                text="Symbol reference",
                category="architecture",
                scope="user",
                code_refs=[{"symbol_id": "scip-typescript npm app App#render()."}],
            )

            tags = captured_metadata.get("tags", {})
            assert tags.get("code_ref_count") == 1
            assert tags.get("code_ref_0_symbol") == "scip-typescript npm app App#render()."
            assert "code_ref_0_path" not in tags


# =============================================================================
# Response Format Tests
# =============================================================================


class TestAddMemoriesCodeRefsResponse:
    """Test that response includes code_refs properly."""

    @pytest.mark.asyncio
    async def test_add_memories_response_format(
        self, mock_principal_all_scopes, sample_code_refs
    ):
        """Response should acknowledge code_refs were stored."""
        from app.mcp_server import (
            add_memories,
            principal_var,
            user_id_var,
            client_name_var,
        )

        principal_var.set(mock_principal_all_scopes)
        user_id_var.set("test-user")
        client_name_var.set("test-client")

        memory_id = str(uuid.uuid4())

        def mock_add(text, user_id=None, metadata=None, infer=False):
            return {
                "results": [{
                    "id": memory_id,
                    "memory": text,
                    "event": "ADD",
                }]
            }

        with patch("app.mcp_server.get_memory_client_safe") as mock_get_client, \
             patch("app.mcp_server.SessionLocal") as mock_session, \
             patch("app.mcp_server.get_user_and_app") as mock_get_user_app, \
             patch("app.mcp_server.project_memory_to_graph"), \
             patch("app.mcp_server.update_entity_edges_on_memory_add"), \
             patch("app.mcp_server.update_tag_edges_on_memory_add"), \
             patch("app.mcp_server.project_similarity_edges_for_memory"):

            mock_memory_client = MagicMock()
            mock_memory_client.add = MagicMock(side_effect=mock_add)
            mock_get_client.return_value = mock_memory_client

            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value = mock_db

            mock_user = MagicMock()
            mock_user.id = uuid.uuid4()
            mock_app = MagicMock()
            mock_app.id = uuid.uuid4()
            mock_app.is_active = True
            mock_get_user_app.return_value = (mock_user, mock_app)

            result_json = await add_memories(
                text="Storage service",
                category="architecture",
                scope="user",
                code_refs=sample_code_refs,
            )

            result = json.loads(result_json)
            # Should not have error
            assert "error" not in result
            # Should have the memory id
            assert result.get("id") == memory_id
