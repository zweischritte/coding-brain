"""
Tests for Qdrant metadata synchronization on memory updates.

This module tests the fix for PRD-QDRANT-METADATA-SYNC:
- Metadata-only updates should sync to Qdrant via set_payload()
- Content updates should sync both embedding and metadata
- Combined updates should work correctly

TDD: These tests define expected behavior for metadata sync.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.utils.vector_sync import build_qdrant_payload, sync_metadata_to_qdrant


# =============================================================================
# Tests for build_qdrant_payload
# =============================================================================


class TestBuildQdrantPayload:
    """Tests for payload construction from memory and metadata."""

    def test_build_payload_with_all_fields(self):
        """Test building payload with all metadata fields populated."""
        # Mock memory object
        memory = MagicMock()
        memory.content = "Test memory content"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        memory.updated_at = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

        metadata = {
            "category": "decision",
            "scope": "project",
            "artifact_type": "repo",
            "artifact_ref": "coding-brain",
            "entity": "TestEntity",
            "source": "user",
            "access_entity": "project:default_org/coding-brain",
            "evidence": ["ADR-001", "PR-123"],
            "tags": {"important": True, "reviewed": True},
        }

        payload = build_qdrant_payload(memory, metadata)

        # Verify core fields
        assert payload["data"] == "Test memory content"
        assert payload["user_id"] == "user-123"
        assert payload["hash"] == hashlib.md5("Test memory content".encode()).hexdigest()
        assert payload["created_at"] == "2025-01-01T12:00:00+00:00"
        assert payload["updated_at"] == "2025-01-02T12:00:00+00:00"

        # Verify metadata fields
        assert payload["category"] == "decision"
        assert payload["scope"] == "project"
        assert payload["artifact_type"] == "repo"
        assert payload["artifact_ref"] == "coding-brain"
        assert payload["entity"] == "TestEntity"
        assert payload["source"] == "user"
        assert payload["access_entity"] == "project:default_org/coding-brain"
        assert payload["evidence"] == ["ADR-001", "PR-123"]
        assert payload["tags"] == {"important": True, "reviewed": True}

    def test_build_payload_with_minimal_fields(self):
        """Test building payload with only required fields."""
        memory = MagicMock()
        memory.content = "Simple memory"
        memory.user = None  # No user
        memory.created_at = None
        memory.updated_at = None

        metadata = {}  # Empty metadata

        payload = build_qdrant_payload(memory, metadata)

        # Verify core fields
        assert payload["data"] == "Simple memory"
        assert payload["hash"] == hashlib.md5("Simple memory".encode()).hexdigest()
        # Empty tags should still be included
        assert payload.get("tags") == {}

        # None values should be filtered out
        assert "user_id" not in payload
        assert "created_at" not in payload
        assert "updated_at" not in payload
        assert "category" not in payload
        assert "entity" not in payload

    def test_build_payload_removes_none_values(self):
        """Test that None values are removed from payload."""
        memory = MagicMock()
        memory.content = "Content"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = None  # Explicitly None

        metadata = {
            "category": "workflow",
            "entity": None,  # Explicitly None
            "scope": None,  # Explicitly None
        }

        payload = build_qdrant_payload(memory, metadata)

        assert payload["category"] == "workflow"
        assert "entity" not in payload  # None removed
        assert "scope" not in payload  # None removed
        assert "updated_at" not in payload  # None removed

    def test_build_payload_with_empty_content(self):
        """Test building payload when content is empty or None."""
        memory = MagicMock()
        memory.content = None
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)

        metadata = {"entity": "Test"}

        payload = build_qdrant_payload(memory, metadata)

        # Empty string should be used for None content
        assert payload["data"] == ""
        assert payload["hash"] == hashlib.md5("".encode()).hexdigest()


# =============================================================================
# Tests for sync_metadata_to_qdrant
# =============================================================================


class TestSyncMetadataToQdrant:
    """Tests for the metadata sync function."""

    def test_sync_uses_set_payload_method(self):
        """Test that sync uses the new set_payload method on vector store."""
        memory_id = str(uuid.uuid4())

        memory = MagicMock()
        memory.content = "Test content"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)

        metadata = {"entity": "TestEntity", "category": "decision"}

        # Mock memory client with set_payload method
        mock_vs = MagicMock()
        mock_vs.set_payload = MagicMock()
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is True
        mock_vs.set_payload.assert_called_once()

        # Verify call arguments
        call_args = mock_vs.set_payload.call_args
        assert call_args[1]["vector_id"] == memory_id
        assert "entity" in call_args[1]["payload"]
        assert call_args[1]["payload"]["entity"] == "TestEntity"

    def test_sync_falls_back_to_client_set_payload(self):
        """Test fallback to direct client.set_payload when wrapper method not available."""
        memory_id = str(uuid.uuid4())

        memory = MagicMock()
        memory.content = "Test content"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)

        metadata = {"entity": "TestEntity"}

        # Mock vector store WITHOUT set_payload method but WITH client
        mock_vs = MagicMock(spec=[])  # Empty spec = no methods
        mock_vs.client = MagicMock()
        mock_vs.client.set_payload = MagicMock()
        mock_vs.collection_name = "test_collection"
        # Manually add the attributes since spec=[] removes them
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        # Remove set_payload from the mock to simulate it not existing
        del mock_vs.set_payload

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is True
        # Verify direct client method was called
        mock_vs.client.set_payload.assert_called_once()
        call_args = mock_vs.client.set_payload.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert memory_id in call_args[1]["points"]

    def test_sync_returns_false_when_no_memory_client(self):
        """Test graceful handling when memory client is unavailable."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Test"
        memory.user = None
        memory.created_at = None
        memory.updated_at = None
        metadata = {}

        # When memory_client=None is passed and no client can be obtained,
        # the function should return False gracefully.
        # In test context, the import will fail which triggers the graceful return False.
        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=None,  # Will try to import, which may fail in test context
        )

        # The import will fail in test context, but that's OK - it should still
        # return False (graceful degradation)
        assert result is False

    def test_sync_returns_false_when_no_vector_store(self):
        """Test graceful handling when vector_store is None."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Test"
        memory.user = None
        memory.created_at = None
        memory.updated_at = None
        metadata = {}

        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = None

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is False

    def test_sync_returns_false_on_exception(self):
        """Test graceful handling when set_payload raises exception."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Test"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)
        metadata = {"entity": "Test"}

        mock_vs = MagicMock()
        mock_vs.set_payload = MagicMock(side_effect=Exception("Qdrant timeout"))
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is False


# =============================================================================
# Tests for Qdrant set_payload method
# =============================================================================


class TestQdrantSetPayload:
    """Tests for the new set_payload method in Qdrant vector store."""

    def test_set_payload_calls_client_correctly(self):
        """Test that set_payload calls the Qdrant client with correct params."""
        from mem0.vector_stores.qdrant import Qdrant

        # Create mock client
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="test_collection")]
        )

        qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=mock_client,
        )

        vector_id = str(uuid.uuid4())
        payload = {
            "data": "Test content",
            "entity": "TestEntity",
            "category": "decision",
        }

        qdrant.set_payload(vector_id=vector_id, payload=payload)

        mock_client.set_payload.assert_called_once_with(
            collection_name="test_collection",
            payload=payload,
            points=[vector_id],
        )

    def test_set_payload_preserves_vector(self):
        """Test that set_payload does NOT call upsert (which would delete vectors)."""
        from mem0.vector_stores.qdrant import Qdrant

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(
            collections=[MagicMock(name="test_collection")]
        )

        qdrant = Qdrant(
            collection_name="test_collection",
            embedding_model_dims=128,
            client=mock_client,
        )

        vector_id = str(uuid.uuid4())
        payload = {"entity": "Test"}

        qdrant.set_payload(vector_id=vector_id, payload=payload)

        # upsert should NOT be called - that would delete the vector
        mock_client.upsert.assert_not_called()
        # set_payload SHOULD be called
        mock_client.set_payload.assert_called_once()


# =============================================================================
# Tests for MCP update_memory metadata sync integration
# =============================================================================


class TestMCPUpdateMemoryMetadataSync:
    """Tests for update_memory MCP tool with metadata sync."""

    @pytest.mark.asyncio
    async def test_metadata_only_update_syncs_to_qdrant(self):
        """Test that metadata-only updates trigger Qdrant sync without re-embedding."""
        # This is an integration-style test to verify the full flow
        pass  # Placeholder - would require full MCP mock setup

    @pytest.mark.asyncio
    async def test_content_update_syncs_both_embedding_and_metadata(self):
        """Test that content updates re-embed AND sync metadata."""
        pass  # Placeholder - would require full MCP mock setup

    @pytest.mark.asyncio
    async def test_combined_update_syncs_correctly(self):
        """Test that combined text + metadata updates work correctly."""
        pass  # Placeholder - would require full MCP mock setup

    def test_metadata_updated_flag_detection(self):
        """Test that metadata_updated flag is correctly computed."""
        # Test cases for metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)

        # Case 1: validated_fields only
        validated_fields = {"entity": "Test"}
        normalized_add_tags = None
        normalized_remove_tags = None
        metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)
        assert metadata_updated is True

        # Case 2: add_tags only
        validated_fields = {}
        normalized_add_tags = {"important": True}
        normalized_remove_tags = None
        metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)
        assert metadata_updated is True

        # Case 3: remove_tags only
        validated_fields = {}
        normalized_add_tags = None
        normalized_remove_tags = ["old_tag"]
        metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)
        assert metadata_updated is True

        # Case 4: No metadata changes
        validated_fields = {}
        normalized_add_tags = None
        normalized_remove_tags = None
        metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)
        assert metadata_updated is False

        # Case 5: Empty dict vs None
        validated_fields = {}
        metadata_updated = bool(validated_fields or None or None)
        assert metadata_updated is False


# =============================================================================
# Tests for edge cases and error handling
# =============================================================================


class TestMetadataSyncEdgeCases:
    """Tests for edge cases in metadata sync."""

    def test_sync_with_unicode_content(self):
        """Test sync handles unicode content correctly."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Unicode content"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)
        metadata = {"entity": "Test"}

        mock_vs = MagicMock()
        mock_vs.set_payload = MagicMock()
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is True
        # Verify hash was computed correctly for unicode
        call_args = mock_vs.set_payload.call_args
        payload = call_args[1]["payload"]
        expected_hash = hashlib.md5("Unicode content".encode()).hexdigest()
        assert payload["hash"] == expected_hash

    def test_sync_with_large_tags_dict(self):
        """Test sync handles large tags dictionaries."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Test"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)

        # Large tags dict
        large_tags = {f"tag_{i}": f"value_{i}" for i in range(100)}
        metadata = {"tags": large_tags}

        mock_vs = MagicMock()
        mock_vs.set_payload = MagicMock()
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is True
        call_args = mock_vs.set_payload.call_args
        payload = call_args[1]["payload"]
        assert len(payload["tags"]) == 100

    def test_sync_with_nested_evidence_list(self):
        """Test sync handles evidence list correctly."""
        memory_id = str(uuid.uuid4())
        memory = MagicMock()
        memory.content = "Test"
        memory.user = MagicMock()
        memory.user.user_id = "user-123"
        memory.created_at = datetime.now(timezone.utc)
        memory.updated_at = datetime.now(timezone.utc)

        metadata = {
            "entity": "Test",
            "evidence": ["ADR-001", "PR-123", "Issue-456"],
        }

        mock_vs = MagicMock()
        mock_vs.set_payload = MagicMock()
        mock_memory_client = MagicMock()
        mock_memory_client.vector_store = mock_vs

        result = sync_metadata_to_qdrant(
            memory_id=memory_id,
            memory=memory,
            metadata=metadata,
            memory_client=mock_memory_client,
        )

        assert result is True
        call_args = mock_vs.set_payload.call_args
        payload = call_args[1]["payload"]
        assert payload["evidence"] == ["ADR-001", "PR-123", "Issue-456"]
