"""
Unit tests for Graph Boost functionality in reranking module.

Tests cover:
- Graph boost calculation with various signals
- Integration with existing boost calculation
- Edge cases (unavailable graph context, missing data)
"""

import pytest
from math import log1p
from app.utils.reranking import (
    compute_graph_boost,
    compute_boost,
    BoostConfig,
    SearchContext,
    DEFAULT_BOOST_CONFIG,
)
from app.graph.graph_cache import GraphContext


class TestComputeGraphBoost:
    """Test graph boost calculation."""

    def test_unavailable_graph_context(self):
        """Test that unavailable graph context returns zero boost."""
        context = GraphContext(available=False)

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata={},
            graph_context=context,
        )

        assert boost == 0.0
        assert breakdown == {}

    def test_entity_centrality_boost(self):
        """Test boost from entity PageRank."""
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0.5,
                    "similarityClusterSize": 0,
                    "maxEntityDegree": 0,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=1,
            max_degree=1,
        )

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata={},
            graph_context=context,
        )

        # Expected: 0.5 / 1.0 * 0.25 = 0.125
        assert "entity_centrality" in breakdown
        assert breakdown["entity_centrality"] == 0.125
        assert abs(boost - 0.125) < 0.0001

    def test_similarity_cluster_boost(self):
        """Test boost from similarity cluster size."""
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0,
                    "similarityClusterSize": 10,
                    "maxEntityDegree": 0,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=100,
            max_degree=1,
        )

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata={},
            graph_context=context,
        )

        # Log scale: log1p(10) / log1p(100) * 0.20
        expected = (log1p(10) / log1p(100)) * 0.20
        assert "similarity_cluster" in breakdown
        assert abs(breakdown["similarity_cluster"] - expected) < 0.0001

    def test_entity_density_boost(self):
        """Test boost from entity co-mention degree."""
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0,
                    "similarityClusterSize": 0,
                    "maxEntityDegree": 20,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=1,
            max_degree=50,
        )

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata={},
            graph_context=context,
        )

        # Log scale: log1p(20) / log1p(50) * 0.15
        expected = (log1p(20) / log1p(50)) * 0.15
        assert "entity_density" in breakdown
        assert abs(breakdown["entity_density"] - expected) < 0.0001

    def test_tag_pmi_boost(self):
        """Test boost from tag PMI co-occurrence."""
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0,
                    "similarityClusterSize": 0,
                    "maxEntityDegree": 0,
                }
            },
            tag_pmi_cache={
                ("trigger", "important"): 0.5,
                ("trigger", "emotional"): 0.3,
            },
            max_pagerank=1.0,
            max_cluster_size=1,
            max_degree=1,
        )

        metadata = {"tags": {"trigger": True, "other": True}}

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata=metadata,
            graph_context=context,
        )

        # Should find PMI for trigger with other tags
        # pmi_sum = 0.5 + 0.3 = 0.8
        # normalized = min(1.0, 0.8 / 3.0) ≈ 0.267
        # boost = 0.267 * 0.10 ≈ 0.0267
        assert "tag_pmi_relevance" in breakdown
        assert breakdown["tag_pmi_relevance"] > 0

    def test_combined_graph_boost(self):
        """Test combined boost from all graph signals."""
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0.8,
                    "similarityClusterSize": 15,
                    "maxEntityDegree": 30,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=50,
            max_degree=100,
        )

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata={},
            graph_context=context,
        )

        # Should have all three components
        assert "entity_centrality" in breakdown
        assert "similarity_cluster" in breakdown
        assert "entity_density" in breakdown

        # Total should be sum of components
        expected_total = (
            breakdown["entity_centrality"]
            + breakdown["similarity_cluster"]
            + breakdown["entity_density"]
        )
        assert abs(boost - expected_total) < 0.0001

    def test_graph_boost_capping(self):
        """Test that graph boost is capped at max_graph_boost."""
        # Create context with very high values
        context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 1.0,
                    "similarityClusterSize": 100,
                    "maxEntityDegree": 100,
                }
            },
            tag_pmi_cache={
                ("a", "b"): 1.0,
                ("a", "c"): 1.0,
                ("a", "d"): 1.0,
            },
            max_pagerank=1.0,
            max_cluster_size=100,
            max_degree=100,
        )

        metadata = {"tags": {"a": True}}

        boost, breakdown = compute_graph_boost(
            memory_id="test-id",
            metadata=metadata,
            graph_context=context,
        )

        # Should be capped at max_graph_boost (0.50)
        assert boost <= DEFAULT_BOOST_CONFIG.max_graph_boost
        if boost < sum(v for v in breakdown.values() if isinstance(v, float)):
            assert breakdown.get("capped") == True

    def test_missing_memory_in_cache(self):
        """Test handling of memory not in cache."""
        context = GraphContext(
            available=True,
            memory_cache={},  # Empty cache
            max_pagerank=1.0,
            max_cluster_size=1,
            max_degree=1,
        )

        boost, breakdown = compute_graph_boost(
            memory_id="not-in-cache",
            metadata={},
            graph_context=context,
        )

        # Should return zero boost gracefully
        assert boost == 0.0
        assert breakdown == {}


class TestComputeBoostWithGraph:
    """Test compute_boost integration with graph context."""

    def test_boost_without_graph_context(self):
        """Test that boost works without graph context (backward compatible)."""
        context = SearchContext(category="architecture")
        metadata = {"category": "architecture"}

        boost, breakdown = compute_boost(
            metadata=metadata,
            stored_tags={},
            context=context,
            graph_context=None,
            memory_id=None,
        )

        # Should still work with metadata boost
        assert boost > 0
        assert "metadata" in breakdown
        assert "graph" not in breakdown

    def test_boost_with_graph_context(self):
        """Test that graph context adds to boost."""
        context = SearchContext(category="architecture")
        metadata = {"category": "architecture"}

        graph_context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 0.5,
                    "similarityClusterSize": 5,
                    "maxEntityDegree": 10,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=10,
            max_degree=20,
        )

        boost, breakdown = compute_boost(
            metadata=metadata,
            stored_tags={},
            context=context,
            graph_context=graph_context,
            memory_id="test-id",
        )

        # Should have both metadata and graph boosts
        assert "metadata" in breakdown
        assert "graph" in breakdown
        assert boost > breakdown["metadata"].get("category", 0)  # Graph added to it

    def test_total_boost_capping_includes_graph(self):
        """Test that total boost cap includes graph boost."""
        context = SearchContext(
            category="workflow",
            scope="project",
            entity="test",
            tags=["important", "urgent", "critical"],
            recency_weight=0.7,
        )
        metadata = {
            "category": "workflow",
            "scope": "project",
            "entity": "test",
        }
        stored_tags = {"important": True, "urgent": True, "critical": True}

        graph_context = GraphContext(
            available=True,
            memory_cache={
                "test-id": {
                    "maxEntityPageRank": 1.0,
                    "similarityClusterSize": 100,
                    "maxEntityDegree": 100,
                }
            },
            max_pagerank=1.0,
            max_cluster_size=100,
            max_degree=100,
        )

        boost, breakdown = compute_boost(
            metadata=metadata,
            stored_tags=stored_tags,
            context=context,
            created_at_str="2025-12-20T00:00:00Z",
            graph_context=graph_context,
            memory_id="test-id",
        )

        # Should be capped at max_total_boost (1.5)
        assert boost <= DEFAULT_BOOST_CONFIG.max_total_boost


class TestBoostConfigWithGraphWeights:
    """Test that BoostConfig includes graph weights."""

    def test_default_graph_weights(self):
        """Test default graph weight values."""
        config = BoostConfig()

        assert config.entity_centrality == 0.25
        assert config.similarity_cluster == 0.20
        assert config.entity_density == 0.15
        assert config.tag_pmi_relevance == 0.10
        assert config.max_graph_boost == 0.50

    def test_custom_graph_weights(self):
        """Test custom graph weight values."""
        config = BoostConfig(
            entity_centrality=0.3,
            similarity_cluster=0.25,
            entity_density=0.2,
            tag_pmi_relevance=0.15,
            max_graph_boost=0.6,
        )

        assert config.entity_centrality == 0.3
        assert config.similarity_cluster == 0.25
        assert config.entity_density == 0.2
        assert config.tag_pmi_relevance == 0.15
        assert config.max_graph_boost == 0.6
