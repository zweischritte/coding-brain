"""
Unit tests for the Neo4j metadata projector.

Tests cover:
- MemoryMetadata normalization from various input formats
- CypherBuilder query generation
- MetadataProjector upsert/delete operations (mocked session)
- Edge cases: evidence as string/list, missing keys

These tests do not require a live Neo4j instance.

Run with: pytest openmemory/api/tests/test_neo4j_metadata_projector.py -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.metadata_projector import (
    MemoryMetadata,
    CypherBuilder,
    MetadataProjector,
    ProjectorConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_memory_data():
    """Sample memory data from OpenMemory export."""
    return {
        "id": "abc-123-def-456",
        "user_id": "9142c989-ce04-416d-b00f-dd4a05d9a1f5",
        "content": "Kritik triggert Schutzreaktion",
        "metadata": {
            "category": "decision",
            "scope": "project",
            "artifact_type": "service",
            "artifact_ref": "api/gateway",
            "entity": "BMG",
            "tags": {"trigger": True, "intensity": 7},
            "evidence": ["evidence-1", "evidence-2"],
            "source": "user",
            "source_app": "openmemory",
            "mcp_client": "claude",
        },
        "state": "active",
        "created_at": "2025-12-04T11:14:00Z",
        "updated_at": "2025-12-05T09:30:00Z",
    }


@pytest.fixture
def mock_session():
    """Create a mock Neo4j session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Create a mock session factory that returns the mock session."""
    def factory():
        return mock_session
    return factory


# =============================================================================
# MEMORYMETADATA TESTS
# =============================================================================

class TestMemoryMetadata:
    """Tests for MemoryMetadata normalization."""

    def test_from_dict_full_metadata(self, sample_memory_data):
        """Parse memory with all metadata fields."""
        metadata = MemoryMetadata.from_dict(
            sample_memory_data,
            memory_id="abc-123",
            user_id="grischadallmer"
        )

        assert metadata.id == "abc-123"
        assert metadata.user_id == "grischadallmer"
        assert metadata.content == "Kritik triggert Schutzreaktion"
        assert metadata.category == "decision"
        assert metadata.scope == "project"
        assert metadata.artifact_type == "service"
        assert metadata.artifact_ref == "api/gateway"
        assert metadata.entity == "BMG"
        assert metadata.evidence == ["evidence-1", "evidence-2"]
        assert metadata.tags == {"trigger": True, "intensity": 7}
        assert metadata.source_app == "openmemory"
        assert metadata.mcp_client == "claude"

    def test_from_dict_ev_as_string(self):
        """Handle evidence as single string."""
        data = {
            "metadata": {
                "category": "decision",
                "scope": "project",
                "evidence": "single-evidence",
            }
        }
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.evidence == ["single-evidence"]

    def test_from_dict_ev_as_list(self):
        """Handle evidence as list of strings."""
        data = {
            "metadata": {
                "category": "decision",
                "scope": "project",
                "evidence": ["ev1", "ev2", "ev3"],
            }
        }
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.evidence == ["ev1", "ev2", "ev3"]

    def test_from_dict_ev_none(self):
        """Handle missing evidence."""
        data = {"metadata": {"category": "decision", "scope": "project"}}
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.evidence == []

    def test_from_dict_tags_as_list(self):
        """Handle tags as list (legacy format)."""
        data = {"metadata": {"tags": ["important", "review"]}}
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.tags == {"important": True, "review": True}

    def test_from_dict_tags_as_dict(self):
        """Handle tags as dict (current format)."""
        data = {"metadata": {"tags": {"ai_obs": True, "conf": 0.8}}}
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.tags == {"ai_obs": True, "conf": 0.8}

    def test_from_dict_src_vs_source(self):
        """Handle both 'src' and 'source' keys."""
        data1 = {"metadata": {"src": "user"}}
        metadata1 = MemoryMetadata.from_dict(data1, "id-1", "user-1")
        assert metadata1.source == "user"

        data2 = {"metadata": {"source": "inference"}}
        metadata2 = MemoryMetadata.from_dict(data2, "id-1", "user-1")
        assert metadata2.source == "inference"

    def test_from_dict_minimal(self):
        """Handle minimal metadata (just required fields)."""
        data = {"metadata": {}}
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")

        assert metadata.id == "id-1"
        assert metadata.user_id == "user-1"
        assert metadata.category is None
        assert metadata.scope is None
        assert metadata.artifact_type is None
        assert metadata.artifact_ref is None
        assert metadata.entity is None
        assert metadata.tags == {}
        assert metadata.evidence == []

    def test_from_dict_content_in_data_key(self):
        """Handle content stored in 'data' key (Qdrant payload format)."""
        data = {"metadata": {"data": "Memory content here"}}
        metadata = MemoryMetadata.from_dict(data, "id-1", "user-1")
        assert metadata.content == "Memory content here"


# =============================================================================
# CYPHERBUILDER TESTS
# =============================================================================

class TestCypherBuilder:
    """Tests for CypherBuilder query generation."""

    def test_constraint_queries_count(self):
        """Verify correct number of constraint queries."""
        queries = CypherBuilder.constraint_queries()
        assert len(queries) >= 9  # At least 9 constraints + 1 index

    def test_constraint_queries_format(self):
        """Verify constraint queries are properly formatted."""
        queries = CypherBuilder.constraint_queries()
        for query in queries:
            assert "IF NOT EXISTS" in query
            assert query.startswith("CREATE ")

    def test_upsert_memory_query_contains_merge(self):
        """Verify upsert query uses MERGE for idempotency."""
        query, params = CypherBuilder.upsert_memory_query()
        assert "MERGE (m:OM_Memory" in query
        assert "$id" in query
        assert "$userId" in query

    def test_entity_relation_query(self):
        """Verify entity relation query structure."""
        query = CypherBuilder.entity_relation_query()
        assert "OM_Entity" in query
        assert "OM_ABOUT" in query
        assert "MERGE" in query

    def test_category_relation_query(self):
        """Verify category relation query structure."""
        query = CypherBuilder.category_relation_query()
        assert "OM_Category" in query
        assert "OM_IN_CATEGORY" in query

    def test_tag_relation_query_stores_value(self):
        """Verify tag relation stores value on relationship."""
        query = CypherBuilder.tag_relation_query()
        assert "OM_Tag" in query
        assert "OM_TAGGED" in query
        assert "r.tagValue" in query

    def test_delete_memory_query_uses_detach(self):
        """Verify delete uses DETACH DELETE for clean removal."""
        query = CypherBuilder.delete_memory_query()
        assert "DETACH DELETE" in query

    def test_get_memory_relations_query(self):
        """Verify relations query returns all necessary fields."""
        query = CypherBuilder.get_memory_relations_query()
        assert "memoryId" in query
        assert "relationType" in query
        assert "targetLabel" in query
        assert "targetValue" in query


# =============================================================================
# METADATAPROJECTOR TESTS (MOCKED SESSION)
# =============================================================================

class TestMetadataProjector:
    """Tests for MetadataProjector with mocked Neo4j session."""

    def test_ensure_constraints_calls_session(self, mock_session_factory, mock_session):
        """Verify ensure_constraints runs all constraint queries."""
        projector = MetadataProjector(mock_session_factory)
        result = projector.ensure_constraints()

        assert result is True
        # Should have run at least 10 queries (constraints + index)
        assert mock_session.run.call_count >= 10

    def test_ensure_constraints_caches_result(self, mock_session_factory, mock_session):
        """Verify constraints are only created once."""
        projector = MetadataProjector(mock_session_factory)

        projector.ensure_constraints()
        first_call_count = mock_session.run.call_count

        projector.ensure_constraints()  # Second call
        assert mock_session.run.call_count == first_call_count  # No new calls

    def test_upsert_memory_minimal(self, mock_session_factory, mock_session):
        """Test upserting memory with minimal metadata."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            content="Test content",
        )

        result = projector.upsert_memory(metadata)

        assert result is True
        # Should call clear relations + upsert memory
        assert mock_session.run.call_count >= 2

    def test_upsert_memory_with_entity(self, mock_session_factory, mock_session):
        """Test upserting memory with entity relation."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            entity="BMG",
        )

        projector.upsert_memory(metadata)

        # Find the entity relation call
        calls = mock_session.run.call_args_list
        entity_call = None
        for c in calls:
            if c[0] and "OM_Entity" in c[0][0]:
                entity_call = c
                break

        assert entity_call is not None
        assert entity_call[0][1]["entityName"] == "BMG"

    def test_upsert_memory_with_category(self, mock_session_factory, mock_session):
        """Test upserting memory with category relation."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            category="architecture",
        )

        projector.upsert_memory(metadata)

        calls = mock_session.run.call_args_list
        category_call = None
        for c in calls:
            if c[0] and "OM_Category" in c[0][0]:
                category_call = c
                break

        assert category_call is not None
        assert category_call[0][1]["categoryName"] == "architecture"

    def test_upsert_memory_with_tags(self, mock_session_factory, mock_session):
        """Test upserting memory with tag relations."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            tags={"trigger": True, "intensity": 7},
        )

        projector.upsert_memory(metadata)

        # Find tag relation calls
        calls = mock_session.run.call_args_list
        tag_calls = [c for c in calls if c[0] and "OM_Tag" in c[0][0]]

        assert len(tag_calls) == 2  # Two tags

    def test_upsert_memory_with_evidence_list(self, mock_session_factory, mock_session):
        """Test upserting memory with multiple evidence items."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            evidence=["ev-1", "ev-2", "ev-3"],
        )

        projector.upsert_memory(metadata)

        calls = mock_session.run.call_args_list
        evidence_calls = [c for c in calls if c[0] and "OM_Evidence" in c[0][0]]

        assert len(evidence_calls) == 3

    def test_upsert_memory_clears_relations_first(self, mock_session_factory, mock_session):
        """Verify upsert clears existing relations before creating new ones."""
        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(id="test-id", user_id="user-1")

        projector.upsert_memory(metadata)

        # First call should be clearing relations
        first_call = mock_session.run.call_args_list[0]
        assert "DELETE r" in first_call[0][0]

    def test_delete_memory(self, mock_session_factory, mock_session):
        """Test deleting a memory."""
        projector = MetadataProjector(mock_session_factory)

        result = projector.delete_memory("memory-to-delete")

        assert result is True
        call = mock_session.run.call_args
        assert "DETACH DELETE" in call[0][0]
        assert call[0][1]["memoryId"] == "memory-to-delete"

    def test_delete_all_user_memories(self, mock_session_factory, mock_session):
        """Test deleting all memories for a user."""
        projector = MetadataProjector(mock_session_factory)

        result = projector.delete_all_user_memories("grischadallmer")

        assert result is True
        call = mock_session.run.call_args
        assert call[0][1]["userId"] == "grischadallmer"

    def test_get_relations_for_memories(self, mock_session_factory, mock_session):
        """Test querying relations for memory IDs."""
        # Mock the result
        mock_records = [
            MagicMock(
                __getitem__=lambda self, k: {
                    "memoryId": "mem-1",
                    "relationType": "OM_IN_CATEGORY",
                    "targetLabel": "OM_Category",
                    "targetValue": "architecture",
                    "relationValue": None,
                }[k]
            ),
            MagicMock(
                __getitem__=lambda self, k: {
                    "memoryId": "mem-1",
                    "relationType": "OM_ABOUT",
                    "targetLabel": "OM_Entity",
                    "targetValue": "BMG",
                    "relationValue": None,
                }[k]
            ),
        ]
        mock_session.run.return_value = mock_records

        projector = MetadataProjector(mock_session_factory)
        relations = projector.get_relations_for_memories(["mem-1"])

        assert "mem-1" in relations
        assert len(relations["mem-1"]) == 2

    def test_get_relations_empty_ids(self, mock_session_factory, mock_session):
        """Test querying with empty memory IDs returns empty dict."""
        projector = MetadataProjector(mock_session_factory)
        relations = projector.get_relations_for_memories([])

        assert relations == {}
        mock_session.run.assert_not_called()

    def test_serialize_tag_value_dict(self, mock_session_factory):
        """Test serializing dict tag value to JSON."""
        projector = MetadataProjector(mock_session_factory)
        result = projector._serialize_tag_value({"nested": "value"})
        assert result == '{"nested": "value"}'

    def test_serialize_tag_value_list(self, mock_session_factory):
        """Test serializing list tag value to JSON."""
        projector = MetadataProjector(mock_session_factory)
        result = projector._serialize_tag_value(["a", "b"])
        assert result == '["a", "b"]'

    def test_serialize_tag_value_primitive(self, mock_session_factory):
        """Test serializing primitive tag values."""
        projector = MetadataProjector(mock_session_factory)
        assert projector._serialize_tag_value(True) == "True"
        assert projector._serialize_tag_value(42) == "42"
        assert projector._serialize_tag_value("text") == "text"

    def test_upsert_truncates_long_content(self, mock_session_factory, mock_session):
        """Test that very long content is truncated."""
        projector = MetadataProjector(mock_session_factory)
        projector.config.max_text_length = 100

        long_content = "x" * 1000
        metadata = MemoryMetadata(
            id="test-id",
            user_id="user-1",
            content=long_content,
        )

        projector.upsert_memory(metadata)

        # Find the upsert call
        for call in mock_session.run.call_args_list:
            if call[0] and "SET m.content" in call[0][0]:
                content_param = call[0][1].get("content")
                if content_param:
                    assert len(content_param) <= 100
                    break

    def test_error_handling_upsert(self, mock_session_factory, mock_session):
        """Test graceful error handling on upsert failure."""
        mock_session.run.side_effect = Exception("Neo4j connection failed")

        projector = MetadataProjector(mock_session_factory)
        metadata = MemoryMetadata(id="test-id", user_id="user-1")

        result = projector.upsert_memory(metadata)

        assert result is False

    def test_error_handling_delete(self, mock_session_factory, mock_session):
        """Test graceful error handling on delete failure."""
        mock_session.run.side_effect = Exception("Neo4j connection failed")

        projector = MetadataProjector(mock_session_factory)
        result = projector.delete_memory("test-id")

        assert result is False


# =============================================================================
# INTEGRATION: FULL MEMORY PROJECTION
# =============================================================================

class TestFullMemoryProjection:
    """Integration tests simulating full memory projection scenarios."""

    def test_project_memory_from_export(self, mock_session_factory, mock_session, sample_memory_data):
        """Test projecting a memory from export format."""
        projector = MetadataProjector(mock_session_factory)

        # Create metadata from export data
        metadata = MemoryMetadata.from_dict(
            sample_memory_data,
            memory_id=sample_memory_data["id"],
            user_id="grischadallmer"  # Use string user_id, not UUID
        )

        result = projector.upsert_memory(metadata)

        assert result is True

        # Verify all expected relations were created
        calls = mock_session.run.call_args_list
        relation_types = []
        for call in calls:
            if call[0]:
                query = call[0][0]
                if "OM_Entity" in query:
                    relation_types.append("entity")
                elif "OM_Category" in query:
                    relation_types.append("category")
                elif "OM_Scope" in query:
                    relation_types.append("scope")
                elif "OM_ArtifactType" in query:
                    relation_types.append("artifact_type")
                elif "OM_ArtifactRef" in query:
                    relation_types.append("artifact_ref")
                elif "OM_Tag" in query:
                    relation_types.append("tag")
                elif "OM_Evidence" in query:
                    relation_types.append("evidence")
                elif "OM_App" in query:
                    relation_types.append("app")

        assert "entity" in relation_types
        assert "category" in relation_types
        assert "scope" in relation_types
        assert "artifact_type" in relation_types
        assert "artifact_ref" in relation_types
        # Tags should appear twice (trigger, intensity)
        assert relation_types.count("tag") == 2
        # Evidence should appear twice
        assert relation_types.count("evidence") == 2
        # App relations (source_app and mcp_client)
        assert relation_types.count("app") == 2


class TestMetadataProjectorQueryMethods:
    """Tests for projector query helper methods (mocked)."""

    def test_get_memory_node(self, mock_session_factory, mock_session):
        """Get a single memory node by id + user_id."""
        mock_session.run.return_value = [
            {
                "id": "mem-1",
                "userId": "user-1",
                "content": "Test content",
                "createdAt": "2025-12-04T11:14:00Z",
                "updatedAt": "2025-12-05T09:30:00Z",
                "state": "active",
                "category": "architecture",
                "scope": "project",
                "artifactType": "service",
                "artifactRef": "api/gateway",
                "entity": "BMG",
                "source": "user",
            }
        ]

        projector = MetadataProjector(mock_session_factory)
        node = projector.get_memory_node(
            memory_id="mem-1",
            user_id="user-1",
            allowed_memory_ids=["mem-1"],
        )

        assert node is not None
        assert node["id"] == "mem-1"
        assert node["category"] == "architecture"

        args, kwargs = mock_session.run.call_args
        assert "MATCH (m:OM_Memory" in args[0]
        params = args[1]
        assert params["memoryId"] == "mem-1"
        assert params["userId"] == "user-1"

    def test_find_related_memories(self, mock_session_factory, mock_session):
        """Find related memories returns normalized shared_relations."""
        mock_session.run.return_value = [
            {
                "memoryId": "mem-2",
                "content": "Related memory",
                "createdAt": "2025-12-05T09:30:00Z",
                "updatedAt": None,
                "state": "active",
                "category": "architecture",
                "scope": "project",
                "artifactType": "service",
                "artifactRef": "api/gateway",
                "entity": "BMG",
                "source": "user",
                "sharedRelations": [
                    {
                        "type": "OM_TAGGED",
                        "targetLabel": "OM_Tag",
                        "targetValue": "neo4j",
                        "seedValue": "True",
                        "otherValue": "True",
                    }
                ],
                "sharedCount": 1,
            }
        ]

        projector = MetadataProjector(mock_session_factory)
        related = projector.find_related_memories(
            memory_id="mem-1",
            user_id="user-1",
            allowed_memory_ids=["mem-1", "mem-2"],
            rel_types=["OM_TAGGED"],
            limit=10,
        )

        assert len(related) == 1
        assert related[0]["id"] == "mem-2"
        assert related[0]["sharedCount"] == 1
        assert related[0]["sharedRelations"][0]["type"] == "OM_TAGGED"
        assert related[0]["sharedRelations"][0]["targetValue"] == "neo4j"

    def test_aggregate_memories(self, mock_session_factory, mock_session):
        """Aggregate returns buckets."""
        mock_session.run.return_value = [
            {"key": "architecture", "count": 7},
            {"key": "decision", "count": 3},
        ]

        projector = MetadataProjector(mock_session_factory)
        buckets = projector.aggregate_memories(
            user_id="user-1",
            group_by="category",
            allowed_memory_ids=["mem-1", "mem-2"],
            limit=10,
        )

        assert buckets[0]["key"] == "architecture"
        assert buckets[0]["count"] == 7

    def test_aggregate_invalid_group_by(self, mock_session_factory):
        """Invalid group_by raises ValueError."""
        projector = MetadataProjector(mock_session_factory)
        with pytest.raises(ValueError):
            projector.aggregate_memories(user_id="user-1", group_by="not-a-dimension")

    def test_tag_cooccurrence(self, mock_session_factory, mock_session):
        """Tag co-occurrence returns pairs with example memory ids."""
        mock_session.run.return_value = [
            {
                "tag1": "trigger",
                "tag2": "important",
                "count": 4,
                "memoryIds": ["mem-1", "mem-2", "mem-3", "mem-4"],
            }
        ]

        projector = MetadataProjector(mock_session_factory)
        pairs = projector.tag_cooccurrence(
            user_id="user-1",
            allowed_memory_ids=["mem-1", "mem-2", "mem-3", "mem-4"],
            limit=10,
            min_count=2,
            sample_size=3,
        )

        assert len(pairs) == 1
        assert pairs[0]["tag1"] == "trigger"
        assert pairs[0]["tag2"] == "important"
        assert pairs[0]["count"] == 4
        assert pairs[0]["exampleMemoryIds"] == ["mem-1", "mem-2", "mem-3"]

    def test_path_between_entities(self, mock_session_factory, mock_session):
        """Path query returns nodes + relationships when present."""
        mock_session.run.return_value = [
            {
                "nodes": [
                    {"label": "OM_Entity", "value": "A", "memory_id": None, "content": None, "category": None, "scope": None},
                    {"label": "OM_Memory", "value": "mem-1", "memory_id": "mem-1", "content": "x", "category": "architecture", "scope": "project"},
                    {"label": "OM_Entity", "value": "B", "memory_id": None, "content": None, "category": None, "scope": None},
                ],
                "relationships": [
                    {"type": "OM_ABOUT", "value": None},
                    {"type": "OM_ABOUT", "value": None},
                ],
            }
        ]

        projector = MetadataProjector(mock_session_factory)
        path = projector.path_between_entities(
            user_id="user-1",
            entity_a="A",
            entity_b="B",
            allowed_memory_ids=["mem-1"],
            max_hops=6,
        )

        assert path is not None
        assert path["entity_a"] == "A"
        assert path["entity_b"] == "B"
        assert len(path["nodes"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
