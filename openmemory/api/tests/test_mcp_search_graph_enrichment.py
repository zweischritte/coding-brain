"""
Unit tests for MCP search graph enrichment.

Tests cover:
- Graph operations helper functions
- Search response enrichment with relations and meta_relations
- Graceful degradation when Neo4j is unavailable

These tests do not require a live Neo4j instance.

Run with: pytest openmemory/api/tests/test_mcp_search_graph_enrichment.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_search_results():
    """Sample search results from MCP search."""
    return [
        {
            "id": "mem-1",
            "memory": "Kritik triggert Schutzreaktion",
            "scores": {"semantic": 0.9, "boost": 0.3, "final": 1.17},
            "metadata": {
                "vault": "FRACTURE_LOG",
                "layer": "emotional",
                "entity": "BMG",
            },
            "created_at": "2025-12-04T11:14:00Z",
        },
        {
            "id": "mem-2",
            "memory": "Pattern recognition active",
            "scores": {"semantic": 0.85, "boost": 0.2, "final": 1.02},
            "metadata": {
                "vault": "FINGERPRINT",
                "layer": "meta",
            },
            "created_at": "2025-12-05T09:30:00Z",
        },
    ]


@pytest.fixture
def sample_meta_relations():
    """Sample meta_relations from projector."""
    return {
        "mem-1": [
            {"type": "OM_IN_VAULT", "target_label": "OM_Vault", "target_value": "FRACTURE_LOG"},
            {"type": "OM_IN_LAYER", "target_label": "OM_Layer", "target_value": "emotional"},
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "BMG"},
        ],
        "mem-2": [
            {"type": "OM_IN_VAULT", "target_label": "OM_Vault", "target_value": "FINGERPRINT"},
            {"type": "OM_IN_LAYER", "target_label": "OM_Layer", "target_value": "meta"},
        ],
    }


@pytest.fixture
def sample_graph_relations():
    """Sample relations from Mem0 Graph Memory."""
    return [
        {
            "source": "Grischa",
            "relation": "WORKS_AT",
            "target": "BMG",
        },
        {
            "source": "BMG",
            "relation": "TRIGGERS",
            "target": "Schutzreaktion",
        },
    ]


# =============================================================================
# GRAPH_OPS TESTS
# =============================================================================

class TestGraphOps:
    """Tests for graph_ops helper functions."""

    def test_project_memory_returns_true_when_no_neo4j(self):
        """project_memory_to_graph returns True when Neo4j not configured."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import project_memory_to_graph, reset_graph_ops
            reset_graph_ops()

            result = project_memory_to_graph(
                memory_id="test-id",
                user_id="user-1",
                content="Test content",
                metadata={"vault": "SOV"},
            )

            assert result is True

    def test_delete_memory_returns_true_when_no_neo4j(self):
        """delete_memory_from_graph returns True when Neo4j not configured."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import delete_memory_from_graph, reset_graph_ops
            reset_graph_ops()

            result = delete_memory_from_graph("test-id")

            assert result is True

    def test_get_meta_relations_returns_empty_when_no_neo4j(self):
        """get_meta_relations_for_memories returns empty dict when Neo4j not configured."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import get_meta_relations_for_memories, reset_graph_ops
            reset_graph_ops()

            result = get_meta_relations_for_memories(["mem-1", "mem-2"])

            assert result == {}

    def test_is_graph_enabled_false_when_no_projector(self):
        """is_graph_enabled returns False when projector not available."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import is_graph_enabled, reset_graph_ops
            reset_graph_ops()

            assert is_graph_enabled() is False

    def test_is_graph_enabled_true_when_projector_available(self):
        """is_graph_enabled returns True when projector is available."""
        mock_projector = MagicMock()
        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import is_graph_enabled, reset_graph_ops
            reset_graph_ops()

            assert is_graph_enabled() is True

    def test_project_memory_calls_projector(self, sample_meta_relations):
        """project_memory_to_graph calls projector.upsert_memory."""
        mock_projector = MagicMock()
        mock_projector.upsert_memory.return_value = True

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import project_memory_to_graph, reset_graph_ops
            reset_graph_ops()

            result = project_memory_to_graph(
                memory_id="test-id",
                user_id="user-1",
                content="Test content",
                metadata={"vault": "SOV", "layer": "identity"},
            )

            assert result is True
            mock_projector.upsert_memory.assert_called_once()

    def test_get_meta_relations_calls_projector(self, sample_meta_relations):
        """get_meta_relations_for_memories calls projector.get_relations_for_memories."""
        mock_projector = MagicMock()
        mock_projector.get_relations_for_memories.return_value = sample_meta_relations

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import get_meta_relations_for_memories, reset_graph_ops
            reset_graph_ops()

            result = get_meta_relations_for_memories(["mem-1", "mem-2"])

            assert result == sample_meta_relations
            mock_projector.get_relations_for_memories.assert_called_once_with(["mem-1", "mem-2"])

    def test_project_memory_handles_projector_error(self):
        """project_memory_to_graph returns False on projector error."""
        mock_projector = MagicMock()
        mock_projector.upsert_memory.side_effect = Exception("Neo4j error")

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import project_memory_to_graph, reset_graph_ops
            reset_graph_ops()

            result = project_memory_to_graph(
                memory_id="test-id",
                user_id="user-1",
                content="Test content",
                metadata={},
            )

            assert result is False


class TestGraphOpsGraphQueries:
    """Tests for new graph query helper functions."""

    def test_graph_queries_return_empty_when_no_neo4j(self):
        """All query helpers should degrade gracefully when projector is missing."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import (
                reset_graph_ops,
                get_memory_node_from_graph,
                find_related_memories_in_graph,
                aggregate_memories_in_graph,
                tag_cooccurrence_in_graph,
                path_between_entities_in_graph,
                get_memory_subgraph_from_graph,
            )

            reset_graph_ops()

            assert get_memory_node_from_graph("mem-1", "user-1") is None
            assert find_related_memories_in_graph("mem-1", "user-1") == []
            assert aggregate_memories_in_graph("user-1", "vault") == []
            assert tag_cooccurrence_in_graph("user-1") == []
            assert path_between_entities_in_graph("user-1", "A", "B") is None

            sg = get_memory_subgraph_from_graph("mem-1", "user-1")
            assert sg["seed_memory_id"] == "mem-1"
            assert sg["nodes"] == []

    def test_find_related_memories_maps_via(self):
        """via=tag,entity maps to OM_TAGGED + OM_ABOUT."""
        mock_projector = MagicMock()
        mock_projector.find_related_memories.return_value = [{"id": "mem-2"}]

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import reset_graph_ops, find_related_memories_in_graph

            reset_graph_ops()
            result = find_related_memories_in_graph(
                memory_id="mem-1",
                user_id="user-1",
                allowed_memory_ids=["mem-1", "mem-2"],
                via="tag,entity",
                limit=5,
            )

            assert result == [{"id": "mem-2"}]
            mock_projector.find_related_memories.assert_called_once()
            _, kwargs = mock_projector.find_related_memories.call_args
            assert kwargs["rel_types"] == ["OM_TAGGED", "OM_ABOUT"]

    def test_find_related_memories_invalid_via_raises(self):
        """Unknown via token should raise ValueError."""
        mock_projector = MagicMock()
        mock_projector.find_related_memories.return_value = []

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import reset_graph_ops, find_related_memories_in_graph

            reset_graph_ops()
            with pytest.raises(ValueError):
                find_related_memories_in_graph(
                    memory_id="mem-1",
                    user_id="user-1",
                    via="not-a-dimension",
                )

    def test_get_memory_subgraph_builds_nodes_and_edges(self):
        """Subgraph builder returns JSON-friendly nodes+edges."""
        mock_projector = MagicMock()
        mock_projector.get_memory_node.return_value = {
            "id": "mem-1",
            "content": "Seed content",
            "vault": "SOVEREIGNTY_CORE",
            "layer": "identity",
            "vector": None,
            "circuit": None,
            "created_at": "2025-12-04T11:14:00Z",
            "updated_at": None,
        }
        mock_projector.get_relations_for_memories.return_value = {
            "mem-1": [
                {"type": "OM_IN_VAULT", "target_label": "OM_Vault", "target_value": "SOVEREIGNTY_CORE"},
                {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "neo4j", "value": "True"},
            ]
        }
        mock_projector.find_related_memories.return_value = [
            {
                "id": "mem-2",
                "content": "Related content",
                "vault": "SOVEREIGNTY_CORE",
                "layer": "identity",
                "vector": None,
                "circuit": None,
                "created_at": "2025-12-05T09:30:00Z",
                "updated_at": None,
                "shared_count": 1,
                "shared_relations": [
                    {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "neo4j", "other_value": "True"},
                ],
            }
        ]

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import reset_graph_ops, get_memory_subgraph_from_graph

            reset_graph_ops()
            sg = get_memory_subgraph_from_graph(
                memory_id="mem-1",
                user_id="user-1",
                allowed_memory_ids=["mem-1", "mem-2"],
                depth=2,
                via="tag",
                related_limit=10,
            )

            assert sg is not None
            assert sg["seed_memory_id"] == "mem-1"
            assert len(sg["nodes"]) >= 3  # seed, related, tag/vault
            assert any(n["label"] == "OM_Tag" and n["value"] == "neo4j" for n in sg["nodes"])
            assert any(e["type"] == "OM_TAGGED" for e in sg["edges"])


# =============================================================================
# SEARCH ENRICHMENT TESTS
# =============================================================================

class TestSearchEnrichment:
    """Tests for search response enrichment logic."""

    def test_enrichment_adds_meta_relations(self, sample_search_results, sample_meta_relations):
        """Verify meta_relations are added to search response."""
        mock_projector = MagicMock()
        mock_projector.get_relations_for_memories.return_value = sample_meta_relations

        with patch('app.graph.graph_ops._get_projector', return_value=mock_projector):
            from app.graph.graph_ops import get_meta_relations_for_memories, is_graph_enabled, reset_graph_ops
            reset_graph_ops()

            # Simulate what search_memory does
            memory_ids = [str(r.get("id")) for r in sample_search_results if r.get("id")]

            if memory_ids and is_graph_enabled():
                meta_relations = get_meta_relations_for_memories(memory_ids)

            assert "mem-1" in meta_relations
            assert "mem-2" in meta_relations
            assert len(meta_relations["mem-1"]) == 3  # vault, layer, entity

    def test_enrichment_preserves_existing_response(self, sample_search_results):
        """Verify enrichment doesn't modify existing response fields."""
        response = {
            "results": sample_search_results,
            "count": 2,
        }

        # Add enrichment
        response["meta_relations"] = {"mem-1": []}

        # Original fields should be unchanged
        assert "results" in response
        assert response["count"] == 2
        assert len(response["results"]) == 2

    def test_enrichment_empty_when_no_results(self):
        """Verify no enrichment happens for empty results."""
        with patch('app.graph.graph_ops._get_projector', return_value=None):
            from app.graph.graph_ops import get_meta_relations_for_memories, reset_graph_ops
            reset_graph_ops()

            result = get_meta_relations_for_memories([])

            assert result == {}


# =============================================================================
# MEM0 GRAPH MEMORY TESTS
# =============================================================================

class TestMem0GraphMemory:
    """Tests for Mem0 Graph Memory integration."""

    def test_is_mem0_graph_enabled_when_no_client(self):
        """is_mem0_graph_enabled returns False when no memory client."""
        with patch('app.utils.memory.get_memory_client', return_value=None):
            from app.graph.graph_ops import is_mem0_graph_enabled

            assert is_mem0_graph_enabled() is False

    def test_is_mem0_graph_enabled_when_no_graph_attr(self):
        """is_mem0_graph_enabled returns False when client has no graph."""
        mock_client = MagicMock(spec=[])  # No 'graph' attribute

        with patch('app.utils.memory.get_memory_client', return_value=mock_client):
            from app.graph.graph_ops import is_mem0_graph_enabled

            assert is_mem0_graph_enabled() is False

    def test_is_mem0_graph_enabled_when_graph_is_none(self):
        """is_mem0_graph_enabled returns False when graph is None."""
        mock_client = MagicMock()
        mock_client.graph = None

        with patch('app.utils.memory.get_memory_client', return_value=mock_client):
            from app.graph.graph_ops import is_mem0_graph_enabled

            assert is_mem0_graph_enabled() is False

    def test_is_mem0_graph_enabled_when_graph_configured(self):
        """is_mem0_graph_enabled returns True when graph is configured."""
        mock_client = MagicMock()
        mock_client.graph = MagicMock()  # Non-None graph

        with patch('app.utils.memory.get_memory_client', return_value=mock_client):
            from app.graph.graph_ops import is_mem0_graph_enabled

            assert is_mem0_graph_enabled() is True

    def test_get_graph_relations_returns_empty_when_no_graph(self):
        """get_graph_relations returns empty list when graph not enabled."""
        mock_client = MagicMock()
        mock_client.graph = None

        with patch('app.utils.memory.get_memory_client', return_value=mock_client):
            from app.graph.graph_ops import get_graph_relations

            result = get_graph_relations(query="test", user_id="user-1")

            assert result == []

    def test_get_graph_relations_calls_graph_search(self, sample_graph_relations):
        """get_graph_relations calls memory_client.graph.search."""
        mock_graph = MagicMock()
        mock_graph.search.return_value = {"relations": sample_graph_relations}

        mock_client = MagicMock()
        mock_client.graph = mock_graph

        with patch('app.utils.memory.get_memory_client', return_value=mock_client):
            from app.graph.graph_ops import get_graph_relations

            result = get_graph_relations(query="kritik", user_id="user-1", limit=10)

            assert result == sample_graph_relations
            mock_graph.search.assert_called_once_with(
                query="kritik",
                filters={"user_id": "user-1"},
                limit=10,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
