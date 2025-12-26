"""Tests for tri-hybrid retrieval (lexical + semantic + graph).

TDD tests covering:
- Tri-hybrid query construction
- Graph context fetching from Neo4j
- Score normalization across retrieval types
- Result fusion (RRF and weighted)
- Fallback when graph unavailable
- Sub-100ms p95 latency target
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test: TriHybridConfig
# =============================================================================


class TestTriHybridConfig:
    """Tests for tri-hybrid configuration."""

    def test_default_weights(self):
        """Default weights should match v9 plan: vector 0.40, lexical 0.35, graph 0.25."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig()
        assert config.vector_weight == 0.40
        assert config.lexical_weight == 0.35
        assert config.graph_weight == 0.25

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig()
        assert config.total_weight == pytest.approx(1.0)

    def test_normalize_weights(self):
        """Weights should normalize to 1.0."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig(
            vector_weight=0.8,
            lexical_weight=0.7,
            graph_weight=0.5,
        )
        config.normalize()
        assert config.total_weight == pytest.approx(1.0)
        # Proportions maintained
        assert config.vector_weight == pytest.approx(0.4)
        assert config.lexical_weight == pytest.approx(0.35)
        assert config.graph_weight == pytest.approx(0.25)

    def test_custom_weights(self):
        """Custom weights should be accepted."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig(
            vector_weight=0.5,
            lexical_weight=0.3,
            graph_weight=0.2,
        )
        assert config.vector_weight == 0.5
        assert config.lexical_weight == 0.3
        assert config.graph_weight == 0.2

    def test_rrf_config_defaults(self):
        """RRF config should have sensible defaults."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig()
        assert config.rrf_rank_constant == 60
        assert config.rrf_window_size == 100

    def test_graph_depth_default(self):
        """Graph traversal depth should default to 2."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig()
        assert config.graph_depth == 2

    def test_graph_edge_types_default(self):
        """Default graph edge types for code retrieval."""
        from openmemory.api.retrieval.trihybrid import TriHybridConfig

        config = TriHybridConfig()
        assert "CALLS" in config.graph_edge_types
        assert "IMPORTS" in config.graph_edge_types
        assert "CONTAINS" in config.graph_edge_types


# =============================================================================
# Test: TriHybridQuery
# =============================================================================


class TestTriHybridQuery:
    """Tests for tri-hybrid query construction."""

    def test_create_query_with_text_and_embedding(self):
        """Query should accept both text and embedding."""
        from openmemory.api.retrieval.trihybrid import TriHybridQuery

        query = TriHybridQuery(
            query_text="def hello_world",
            embedding=[0.1] * 768,
        )
        assert query.query_text == "def hello_world"
        assert len(query.embedding) == 768

    def test_query_requires_text_or_embedding(self):
        """Query must have at least text or embedding."""
        from openmemory.api.retrieval.trihybrid import TriHybridQuery

        with pytest.raises(ValueError, match="text or embedding"):
            TriHybridQuery(query_text="", embedding=[])

    def test_query_with_seed_symbols(self):
        """Query can specify seed symbols for graph expansion."""
        from openmemory.api.retrieval.trihybrid import TriHybridQuery

        query = TriHybridQuery(
            query_text="database connection",
            embedding=[0.1] * 768,
            seed_symbols=["scip:python:pkg:module/class#method"],
        )
        assert len(query.seed_symbols) == 1

    def test_query_with_filters(self):
        """Query can include filters."""
        from openmemory.api.retrieval.trihybrid import TriHybridQuery

        query = TriHybridQuery(
            query_text="async handler",
            embedding=[0.1] * 768,
            filters={"language": "python", "file_path": "/src/*"},
        )
        assert query.filters["language"] == "python"

    def test_query_size_and_offset(self):
        """Query should support pagination."""
        from openmemory.api.retrieval.trihybrid import TriHybridQuery

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
            size=20,
            offset=10,
        )
        assert query.size == 20
        assert query.offset == 10


# =============================================================================
# Test: GraphContextFetcher
# =============================================================================


class TestGraphContextFetcher:
    """Tests for fetching graph context from Neo4j."""

    def test_fetch_neighbors_for_symbol(self):
        """Should fetch neighboring symbols for a given symbol ID."""
        from openmemory.api.retrieval.trihybrid import (
            GraphContextFetcher,
            GraphContext,
        )

        # Mock Neo4j driver
        mock_driver = MagicMock()
        mock_driver.get_outgoing_edges.return_value = [
            MagicMock(
                target_id="scip:python:pkg:module/OtherClass#method",
                edge_type=MagicMock(value="CALLS"),
            ),
        ]
        mock_driver.get_node.return_value = MagicMock(
            id="scip:python:pkg:module/OtherClass#method",
            properties={"name": "method", "kind": "method"},
        )

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["scip:python:pkg:module/MyClass#method"],
            depth=1,
        )

        assert isinstance(context, GraphContext)
        assert len(context.neighbor_ids) >= 1

    def test_fetch_with_edge_type_filter(self):
        """Should filter by edge types."""
        from openmemory.api.retrieval.trihybrid import GraphContextFetcher

        mock_driver = MagicMock()
        mock_driver.get_outgoing_edges.return_value = [
            MagicMock(
                target_id="sym1",
                edge_type=MagicMock(value="CALLS"),
            ),
            MagicMock(
                target_id="sym2",
                edge_type=MagicMock(value="READS"),
            ),
        ]
        mock_driver.get_node.return_value = MagicMock(
            id="sym1",
            properties={},
        )

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["root"],
            depth=1,
            edge_types=["CALLS"],
        )

        # Only CALLS edges should be included
        assert "sym1" in context.neighbor_ids
        assert "sym2" not in context.neighbor_ids

    def test_fetch_multi_hop(self):
        """Should support multi-hop traversal."""
        from openmemory.api.retrieval.trihybrid import GraphContextFetcher

        mock_driver = MagicMock()

        # First hop returns sym1
        # Second hop from sym1 returns sym2
        call_count = [0]

        def mock_get_edges(node_id):
            call_count[0] += 1
            if node_id == "root":
                return [
                    MagicMock(target_id="sym1", edge_type=MagicMock(value="CALLS")),
                ]
            elif node_id == "sym1":
                return [
                    MagicMock(target_id="sym2", edge_type=MagicMock(value="CALLS")),
                ]
            return []

        mock_driver.get_outgoing_edges.side_effect = mock_get_edges
        mock_driver.get_node.return_value = MagicMock(id="", properties={})

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["root"],
            depth=2,
        )

        # Should include both sym1 and sym2
        assert "sym1" in context.neighbor_ids
        assert "sym2" in context.neighbor_ids

    def test_cycle_detection(self):
        """Should handle cycles in the graph."""
        from openmemory.api.retrieval.trihybrid import GraphContextFetcher

        mock_driver = MagicMock()

        # Create a cycle: root -> sym1 -> root
        def mock_get_edges(node_id):
            if node_id == "root":
                return [
                    MagicMock(target_id="sym1", edge_type=MagicMock(value="CALLS")),
                ]
            elif node_id == "sym1":
                return [
                    MagicMock(target_id="root", edge_type=MagicMock(value="CALLS")),
                ]
            return []

        mock_driver.get_outgoing_edges.side_effect = mock_get_edges
        mock_driver.get_node.return_value = MagicMock(id="", properties={})

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["root"],
            depth=10,  # Large depth to test cycle handling
        )

        # Should not hang, should visit each node once
        assert "sym1" in context.neighbor_ids

    def test_empty_graph(self):
        """Should handle empty graph gracefully."""
        from openmemory.api.retrieval.trihybrid import GraphContextFetcher, GraphContext

        mock_driver = MagicMock()
        mock_driver.get_outgoing_edges.return_value = []

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["nonexistent"],
            depth=1,
        )

        assert isinstance(context, GraphContext)
        assert len(context.neighbor_ids) == 0

    def test_max_neighbors_limit(self):
        """Should respect max neighbors limit."""
        from openmemory.api.retrieval.trihybrid import GraphContextFetcher

        mock_driver = MagicMock()
        # Return 100 neighbors
        mock_driver.get_outgoing_edges.return_value = [
            MagicMock(target_id=f"sym{i}", edge_type=MagicMock(value="CALLS"))
            for i in range(100)
        ]
        mock_driver.get_node.return_value = MagicMock(id="", properties={})

        fetcher = GraphContextFetcher(mock_driver)
        context = fetcher.fetch_context(
            symbol_ids=["root"],
            depth=1,
            max_neighbors=10,
        )

        assert len(context.neighbor_ids) <= 10


# =============================================================================
# Test: ScoreNormalizer
# =============================================================================


class TestScoreNormalizer:
    """Tests for score normalization across retrieval types."""

    def test_min_max_normalization(self):
        """Should normalize scores to 0-1 range using min-max."""
        from openmemory.api.retrieval.trihybrid import ScoreNormalizer

        normalizer = ScoreNormalizer(method="min_max")
        scores = [0.5, 1.0, 2.0, 5.0]
        normalized = normalizer.normalize(scores)

        assert min(normalized) == pytest.approx(0.0)
        assert max(normalized) == pytest.approx(1.0)

    def test_z_score_normalization(self):
        """Should normalize scores using z-score."""
        from openmemory.api.retrieval.trihybrid import ScoreNormalizer

        normalizer = ScoreNormalizer(method="z_score")
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = normalizer.normalize(scores)

        # Z-scores should have mean ~0 and std ~1
        import statistics

        assert statistics.mean(normalized) == pytest.approx(0.0, abs=0.01)
        assert statistics.stdev(normalized) == pytest.approx(1.0, abs=0.01)

    def test_single_score(self):
        """Should handle single score."""
        from openmemory.api.retrieval.trihybrid import ScoreNormalizer

        normalizer = ScoreNormalizer(method="min_max")
        scores = [5.0]
        normalized = normalizer.normalize(scores)

        assert normalized[0] == 1.0  # Single score normalizes to 1.0

    def test_empty_scores(self):
        """Should handle empty score list."""
        from openmemory.api.retrieval.trihybrid import ScoreNormalizer

        normalizer = ScoreNormalizer(method="min_max")
        scores = []
        normalized = normalizer.normalize(scores)

        assert normalized == []

    def test_identical_scores(self):
        """Should handle identical scores."""
        from openmemory.api.retrieval.trihybrid import ScoreNormalizer

        normalizer = ScoreNormalizer(method="min_max")
        scores = [5.0, 5.0, 5.0]
        normalized = normalizer.normalize(scores)

        # All identical scores normalize to 1.0 (max value)
        assert all(s == 1.0 for s in normalized)


# =============================================================================
# Test: ResultFusion
# =============================================================================


class TestResultFusion:
    """Tests for result fusion strategies."""

    def test_rrf_fusion_basic(self):
        """RRF fusion should combine results correctly."""
        from openmemory.api.retrieval.trihybrid import (
            ResultFusion,
            FusionMethod,
            RankedResult,
        )

        fusion = ResultFusion(method=FusionMethod.RRF, rrf_k=60)

        lexical_results = [
            RankedResult(id="doc1", score=0.9, rank=1),
            RankedResult(id="doc2", score=0.8, rank=2),
            RankedResult(id="doc3", score=0.7, rank=3),
        ]
        vector_results = [
            RankedResult(id="doc2", score=0.95, rank=1),
            RankedResult(id="doc1", score=0.85, rank=2),
            RankedResult(id="doc4", score=0.75, rank=3),
        ]
        graph_results = [
            RankedResult(id="doc3", score=0.8, rank=1),
            RankedResult(id="doc1", score=0.7, rank=2),
        ]

        fused = fusion.fuse(
            lexical=lexical_results,
            vector=vector_results,
            graph=graph_results,
        )

        # doc1 appears in all three, should be ranked high
        doc1_rank = next(i for i, r in enumerate(fused) if r.id == "doc1")
        assert doc1_rank < 2  # Should be in top 2

    def test_rrf_formula(self):
        """RRF formula should be 1/(k+rank)."""
        from openmemory.api.retrieval.trihybrid import (
            ResultFusion,
            FusionMethod,
            RankedResult,
        )

        fusion = ResultFusion(method=FusionMethod.RRF, rrf_k=60)

        # Single list, single result at rank 1
        # RRF score = 1/(60+1) = 1/61
        results = [RankedResult(id="doc1", score=1.0, rank=1)]
        fused = fusion.fuse(lexical=results, vector=[], graph=[])

        expected_rrf = 1.0 / (60 + 1)
        assert fused[0].score == pytest.approx(expected_rrf)

    def test_weighted_fusion(self):
        """Weighted fusion should apply weights correctly."""
        from openmemory.api.retrieval.trihybrid import (
            ResultFusion,
            FusionMethod,
            RankedResult,
        )

        fusion = ResultFusion(
            method=FusionMethod.WEIGHTED,
            weights={"lexical": 0.35, "vector": 0.40, "graph": 0.25},
        )

        lexical_results = [RankedResult(id="doc1", score=1.0, rank=1)]
        vector_results = [RankedResult(id="doc1", score=0.5, rank=1)]
        graph_results = [RankedResult(id="doc1", score=0.8, rank=1)]

        fused = fusion.fuse(
            lexical=lexical_results,
            vector=vector_results,
            graph=graph_results,
        )

        expected_score = 1.0 * 0.35 + 0.5 * 0.40 + 0.8 * 0.25
        assert fused[0].score == pytest.approx(expected_score)

    def test_fusion_with_missing_results(self):
        """Fusion should handle results that appear in only some lists."""
        from openmemory.api.retrieval.trihybrid import (
            ResultFusion,
            FusionMethod,
            RankedResult,
        )

        fusion = ResultFusion(method=FusionMethod.RRF, rrf_k=60)

        lexical_results = [RankedResult(id="doc1", score=0.9, rank=1)]
        vector_results = [RankedResult(id="doc2", score=0.95, rank=1)]
        graph_results = []

        fused = fusion.fuse(
            lexical=lexical_results,
            vector=vector_results,
            graph=graph_results,
        )

        # Both docs should appear in results
        doc_ids = {r.id for r in fused}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_fusion_result_limit(self):
        """Fusion should respect result limit."""
        from openmemory.api.retrieval.trihybrid import (
            ResultFusion,
            FusionMethod,
            RankedResult,
        )

        fusion = ResultFusion(method=FusionMethod.RRF, rrf_k=60)

        lexical_results = [
            RankedResult(id=f"doc{i}", score=0.9 - i * 0.1, rank=i + 1)
            for i in range(20)
        ]

        fused = fusion.fuse(
            lexical=lexical_results,
            vector=[],
            graph=[],
            limit=5,
        )

        assert len(fused) == 5


# =============================================================================
# Test: TriHybridRetriever
# =============================================================================


class TestTriHybridRetriever:
    """Tests for the main tri-hybrid retriever."""

    def test_retrieve_combines_all_sources(self):
        """Retriever should combine lexical, vector, and graph results."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
            TriHybridResult,
        )

        # Mock OpenSearch client
        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[
                MagicMock(id="doc1", score=0.9, source={"content": "test"}),
            ],
            total=1,
            took_ms=5,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[
                MagicMock(id="doc2", score=0.85, source={"content": "test2"}),
            ],
            total=1,
            took_ms=5,
        )

        # Mock graph driver
        mock_graph = MagicMock()
        mock_graph.get_outgoing_edges.return_value = []

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=mock_graph,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test query",
            embedding=[0.1] * 768,
        )

        result = retriever.retrieve(query, index_name="code_embeddings")

        assert isinstance(result, TriHybridResult)
        assert len(result.hits) >= 1
        mock_opensearch.lexical_search.assert_called_once()
        mock_opensearch.vector_search.assert_called_once()

    def test_retrieve_with_seed_symbols(self):
        """Retriever should use seed symbols for graph expansion."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[], total=0, took_ms=1
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[], total=0, took_ms=1
        )

        mock_graph = MagicMock()
        mock_graph.get_outgoing_edges.return_value = [
            MagicMock(
                target_id="neighbor_sym",
                edge_type=MagicMock(value="CALLS"),
            ),
        ]
        mock_graph.get_node.return_value = MagicMock(
            id="neighbor_sym",
            properties={"name": "func"},
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=mock_graph,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
            seed_symbols=["scip:python:pkg:module/MyClass#method"],
        )

        retriever.retrieve(query, index_name="test")

        # Should have called graph with seed symbols
        mock_graph.get_outgoing_edges.assert_called()

    def test_fallback_when_graph_unavailable(self):
        """Should gracefully degrade when graph is unavailable."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={})],
            total=1,
            took_ms=5,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[MagicMock(id="doc2", score=0.85, source={})],
            total=1,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,  # No graph
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        result = retriever.retrieve(query, index_name="test")

        # Should still work with lexical + vector only
        assert result.hits is not None
        assert result.graph_available is False

    def test_fallback_on_graph_error(self):
        """Should continue if graph query fails."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={})],
            total=1,
            took_ms=5,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[], total=0, took_ms=5
        )

        mock_graph = MagicMock()
        mock_graph.get_outgoing_edges.side_effect = Exception("Graph connection failed")

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=mock_graph,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
            seed_symbols=["scip:python:pkg:module/MyClass#method"],  # Trigger graph fetch
        )

        # Should not raise, should gracefully degrade
        result = retriever.retrieve(query, index_name="test")
        assert result.hits is not None
        assert result.graph_error is not None

    def test_result_includes_timing(self):
        """Result should include timing breakdown."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[], total=0, took_ms=10
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[], total=0, took_ms=15
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        result = retriever.retrieve(query, index_name="test")

        assert result.timing is not None
        assert result.timing.lexical_ms >= 0
        assert result.timing.vector_ms >= 0
        assert result.timing.total_ms >= 0

    def test_result_includes_source_breakdown(self):
        """Result should show which sources contributed to each hit."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={"content": "a"})],
            total=1,
            took_ms=5,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.85, source={"content": "a"})],
            total=1,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        result = retriever.retrieve(query, index_name="test")

        # doc1 appears in both lexical and vector
        hit = next(h for h in result.hits if h.id == "doc1")
        assert hit.sources is not None
        assert "lexical" in hit.sources
        assert "vector" in hit.sources


# =============================================================================
# Test: Performance
# =============================================================================


class TestTriHybridPerformance:
    """Tests for performance requirements."""

    def test_latency_target(self):
        """Retrieval should complete within latency target."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        # Fast mocks
        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[
                MagicMock(id=f"doc{i}", score=0.9, source={})
                for i in range(10)
            ],
            total=10,
            took_ms=5,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[
                MagicMock(id=f"vec{i}", score=0.85, source={})
                for i in range(10)
            ],
            total=10,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        start = time.perf_counter()
        result = retriever.retrieve(query, index_name="test")
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be well under 100ms with mocked backends
        assert elapsed_ms < 100
        # Also check reported timing
        assert result.timing.total_ms < 100

    def test_parallel_retrieval(self):
        """Lexical and vector searches should run in parallel."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridConfig,
            TriHybridQuery,
        )

        # Track call times
        call_times = []

        def track_lexical(*args, **kwargs):
            call_times.append(("lexical", time.perf_counter()))
            time.sleep(0.01)  # Simulate 10ms latency
            return MagicMock(hits=[], total=0, took_ms=10)

        def track_vector(*args, **kwargs):
            call_times.append(("vector", time.perf_counter()))
            time.sleep(0.01)  # Simulate 10ms latency
            return MagicMock(hits=[], total=0, took_ms=10)

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.side_effect = track_lexical
        mock_opensearch.vector_search.side_effect = track_vector

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
            config=TriHybridConfig(),
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        start = time.perf_counter()
        retriever.retrieve(query, index_name="test")
        elapsed = time.perf_counter() - start

        # If parallel, total time should be ~10ms not ~20ms
        # Allow some overhead but should be less than sequential
        assert elapsed < 0.025  # Less than 25ms (sequential would be 20ms+)


# =============================================================================
# Test: Integration with OpenSearch module
# =============================================================================


class TestOpenSearchIntegration:
    """Tests for integration with existing OpenSearch module."""

    def test_uses_existing_search_types(self):
        """Should use existing LexicalSearchQuery and VectorSearchQuery."""
        from openmemory.api.retrieval.trihybrid import TriHybridRetriever, TriHybridQuery
        from openmemory.api.retrieval import LexicalSearchQuery, VectorSearchQuery

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[], total=0, took_ms=1
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[], total=0, took_ms=1
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        retriever.retrieve(query, index_name="test")

        # Check that the calls were made with correct query types
        lexical_call = mock_opensearch.lexical_search.call_args
        vector_call = mock_opensearch.vector_search.call_args

        # Verify arguments passed
        assert lexical_call is not None
        assert vector_call is not None


# =============================================================================
# Test: TriHybridResult
# =============================================================================


class TestTriHybridResult:
    """Tests for tri-hybrid result structure."""

    def test_result_structure(self):
        """Result should have expected structure."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        result = TriHybridResult(
            hits=[
                TriHybridHit(
                    id="doc1",
                    score=0.95,
                    source={"content": "test"},
                    sources={"lexical": 0.9, "vector": 0.8},
                ),
            ],
            total=1,
            timing=TriHybridTiming(
                lexical_ms=10,
                vector_ms=15,
                graph_ms=5,
                fusion_ms=2,
                total_ms=32,
            ),
            graph_available=True,
        )

        assert len(result.hits) == 1
        assert result.hits[0].id == "doc1"
        assert result.timing.total_ms == 32

    def test_hit_includes_graph_context(self):
        """Hits can include graph context."""
        from openmemory.api.retrieval.trihybrid import TriHybridHit

        hit = TriHybridHit(
            id="doc1",
            score=0.9,
            source={},
            sources={"lexical": 0.9},
            graph_context={
                "neighbors": ["sym1", "sym2"],
                "edges": [
                    {"type": "CALLS", "target": "sym1"},
                ],
            },
        )

        assert hit.graph_context is not None
        assert len(hit.graph_context["neighbors"]) == 2


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_text(self):
        """Should handle query with only embedding."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={})],
            total=1,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="",  # Empty text
            embedding=[0.1] * 768,
        )

        result = retriever.retrieve(query, index_name="test")

        # Should skip lexical search, use only vector
        mock_opensearch.lexical_search.assert_not_called()
        mock_opensearch.vector_search.assert_called_once()

    def test_empty_embedding(self):
        """Should handle query with only text."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={})],
            total=1,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="test query",
            embedding=[],  # Empty embedding
        )

        result = retriever.retrieve(query, index_name="test")

        # Should skip vector search, use only lexical
        mock_opensearch.lexical_search.assert_called_once()
        mock_opensearch.vector_search.assert_not_called()

    def test_opensearch_error_handling(self):
        """Should handle OpenSearch errors gracefully."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridQuery,
            RetrievalError,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.side_effect = Exception("Connection failed")
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[MagicMock(id="doc1", score=0.9, source={})],
            total=1,
            took_ms=5,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        # Should still work with partial results
        result = retriever.retrieve(query, index_name="test")
        assert result.hits is not None
        assert result.lexical_error is not None

    def test_all_backends_fail(self):
        """Should raise error if all backends fail."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridQuery,
            RetrievalError,
        )

        mock_opensearch = MagicMock()
        mock_opensearch.lexical_search.side_effect = Exception("Failed")
        mock_opensearch.vector_search.side_effect = Exception("Failed")

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
        )

        with pytest.raises(RetrievalError):
            retriever.retrieve(query, index_name="test")

    def test_very_large_result_set(self):
        """Should handle large result sets efficiently."""
        from openmemory.api.retrieval.trihybrid import (
            TriHybridRetriever,
            TriHybridQuery,
        )

        mock_opensearch = MagicMock()
        # Return 1000 results
        mock_opensearch.lexical_search.return_value = MagicMock(
            hits=[
                MagicMock(id=f"lex{i}", score=0.9 - i * 0.0001, source={})
                for i in range(1000)
            ],
            total=1000,
            took_ms=50,
        )
        mock_opensearch.vector_search.return_value = MagicMock(
            hits=[
                MagicMock(id=f"vec{i}", score=0.85 - i * 0.0001, source={})
                for i in range(1000)
            ],
            total=1000,
            took_ms=50,
        )

        retriever = TriHybridRetriever(
            opensearch_client=mock_opensearch,
            graph_driver=None,
        )

        query = TriHybridQuery(
            query_text="test",
            embedding=[0.1] * 768,
            size=10,  # Only want 10 results
        )

        result = retriever.retrieve(query, index_name="test")

        # Should return limited results
        assert len(result.hits) == 10


# =============================================================================
# Test: Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_trihybrid_retriever(self):
        """Factory should create retriever with defaults."""
        from openmemory.api.retrieval.trihybrid import create_trihybrid_retriever

        mock_opensearch = MagicMock()

        retriever = create_trihybrid_retriever(
            opensearch_client=mock_opensearch,
        )

        assert retriever is not None
        assert retriever.config.vector_weight == 0.40

    def test_create_with_custom_config(self):
        """Factory should accept custom config."""
        from openmemory.api.retrieval.trihybrid import (
            create_trihybrid_retriever,
            TriHybridConfig,
        )

        mock_opensearch = MagicMock()

        retriever = create_trihybrid_retriever(
            opensearch_client=mock_opensearch,
            config=TriHybridConfig(vector_weight=0.6, lexical_weight=0.3, graph_weight=0.1),
        )

        assert retriever.config.vector_weight == 0.6

    def test_create_with_graph_driver(self):
        """Factory should accept graph driver."""
        from openmemory.api.retrieval.trihybrid import create_trihybrid_retriever

        mock_opensearch = MagicMock()
        mock_graph = MagicMock()

        retriever = create_trihybrid_retriever(
            opensearch_client=mock_opensearch,
            graph_driver=mock_graph,
        )

        assert retriever.graph_driver is mock_graph
