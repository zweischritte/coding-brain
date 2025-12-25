"""
Unit tests for RRF (Reciprocal Rank Fusion) module.

Tests cover:
- Basic RRF score calculation
- Weighted fusion with different alpha values
- Handling of single-source results (missing rank penalty)
- Edge cases (empty inputs, disabled mode)
"""

import pytest
from app.utils.rrf_fusion import (
    RRFFusion,
    RRFConfig,
    RetrievalResult,
    FusedResult,
    get_rrf_config,
)


class TestRRFBasics:
    """Test basic RRF functionality."""

    def test_rrf_formula_calculation(self):
        """Test that RRF formula is correctly applied."""
        config = RRFConfig(k=60, alpha=0.5)  # Equal weight
        fusion = RRFFusion(config)

        # Memory in both sources at rank 1
        vector = [RetrievalResult("a", 1, 0.9, "vector")]
        graph = [RetrievalResult("a", 1, 0.8, "graph")]

        result = fusion.fuse(vector, graph)

        assert len(result) == 1
        assert result[0].memory_id == "a"
        assert result[0].in_both == True

        # Expected: 0.5/(60+1) + 0.5/(60+1) = 0.5/61 + 0.5/61 ≈ 0.01639
        expected_score = 0.5 / 61 + 0.5 / 61
        assert abs(result[0].rrf_score - expected_score) < 0.0001

    def test_rrf_ranking_order(self):
        """Test that results are sorted by RRF score descending."""
        fusion = RRFFusion(RRFConfig(k=60, alpha=0.6))

        vector = [
            RetrievalResult("a", 1, 0.9, "vector"),
            RetrievalResult("b", 2, 0.8, "vector"),
            RetrievalResult("c", 3, 0.7, "vector"),
        ]
        graph = [
            RetrievalResult("b", 1, 0.85, "graph"),  # b is top in graph
            RetrievalResult("a", 2, 0.8, "graph"),
        ]

        result = fusion.fuse(vector, graph)

        # First should be either a or b (both in both sources)
        assert result[0].in_both == True
        assert result[1].in_both == True
        # Scores should be descending
        for i in range(len(result) - 1):
            assert result[i].rrf_score >= result[i + 1].rrf_score

    def test_rrf_memory_in_both_sources(self):
        """Test that memories in both sources are correctly identified."""
        fusion = RRFFusion(RRFConfig())

        vector = [RetrievalResult("a", 1, 0.9, "vector")]
        graph = [
            RetrievalResult("a", 1, 0.8, "graph"),  # Same memory
            RetrievalResult("b", 2, 0.7, "graph"),  # Graph only
        ]

        result = fusion.fuse(vector, graph)

        a_result = next(r for r in result if r.memory_id == "a")
        b_result = next(r for r in result if r.memory_id == "b")

        assert a_result.in_both == True
        assert a_result.vector_rank == 1
        assert a_result.graph_rank == 1

        assert b_result.in_both == False
        assert b_result.vector_rank is None
        assert b_result.graph_rank == 2


class TestRRFMissingRankPenalty:
    """Test handling of single-source results."""

    def test_vector_only_result_penalized(self):
        """Test that vector-only results get graph rank penalty."""
        config = RRFConfig(k=60, alpha=0.6, missing_rank_penalty=100)
        fusion = RRFFusion(config)

        vector = [RetrievalResult("a", 1, 0.9, "vector")]
        graph = []  # No graph results

        result = fusion.fuse(vector, graph)

        assert len(result) == 1
        assert result[0].vector_rank == 1
        assert result[0].graph_rank is None  # Not in graph

        # Score should be: 0.6/(60+1) + 0.4/(60+100)
        expected = 0.6 / 61 + 0.4 / 160
        assert abs(result[0].rrf_score - expected) < 0.0001

    def test_graph_only_result_penalized(self):
        """Test that graph-only results get vector rank penalty."""
        config = RRFConfig(k=60, alpha=0.6, missing_rank_penalty=100)
        fusion = RRFFusion(config)

        vector = []
        graph = [RetrievalResult("b", 1, 0.8, "graph")]

        result = fusion.fuse(vector, graph)

        assert len(result) == 1
        assert result[0].vector_rank is None
        assert result[0].graph_rank == 1

        # Score should be: 0.6/(60+100) + 0.4/(60+1)
        expected = 0.6 / 160 + 0.4 / 61
        assert abs(result[0].rrf_score - expected) < 0.0001

    def test_in_both_beats_single_source(self):
        """Test that results in both sources rank higher."""
        config = RRFConfig(k=60, alpha=0.5, missing_rank_penalty=100)
        fusion = RRFFusion(config)

        vector = [
            RetrievalResult("a", 1, 0.9, "vector"),  # Vector only
            RetrievalResult("b", 5, 0.7, "vector"),  # In both
        ]
        graph = [
            RetrievalResult("b", 1, 0.8, "graph"),  # In both
            RetrievalResult("c", 2, 0.75, "graph"),  # Graph only
        ]

        result = fusion.fuse(vector, graph)

        # b should be first (in both sources)
        assert result[0].memory_id == "b"
        assert result[0].in_both == True


class TestRRFAlphaWeight:
    """Test different alpha (vector/graph weight) values."""

    def test_alpha_1_is_vector_only(self):
        """Test that alpha=1.0 ignores graph results."""
        config = RRFConfig(k=60, alpha=1.0, missing_rank_penalty=100)
        fusion = RRFFusion(config)

        vector = [RetrievalResult("a", 1, 0.9, "vector")]
        graph = [RetrievalResult("b", 1, 0.95, "graph")]

        result = fusion.fuse(vector, graph)

        # With alpha=1.0, vector rank 1 should beat graph rank 1
        # a: 1.0/(60+1) + 0/(60+100) = ~0.0164
        # b: 1.0/(60+100) + 0/(60+1) = ~0.00625
        assert result[0].memory_id == "a"

    def test_alpha_0_is_graph_only(self):
        """Test that alpha=0.0 ignores vector results."""
        config = RRFConfig(k=60, alpha=0.0, missing_rank_penalty=100)
        fusion = RRFFusion(config)

        vector = [RetrievalResult("a", 1, 0.9, "vector")]
        graph = [RetrievalResult("b", 1, 0.8, "graph")]

        result = fusion.fuse(vector, graph)

        # With alpha=0.0, graph rank 1 should beat vector rank 1
        assert result[0].memory_id == "b"

    def test_alpha_affects_ranking(self):
        """Test that different alpha values change ranking."""
        vector = [
            RetrievalResult("a", 1, 0.9, "vector"),
            RetrievalResult("b", 10, 0.5, "vector"),
        ]
        graph = [
            RetrievalResult("b", 1, 0.8, "graph"),
            RetrievalResult("a", 10, 0.5, "graph"),
        ]

        # With vector preference (alpha=0.8)
        fusion_vector = RRFFusion(RRFConfig(alpha=0.8))
        result_vector = fusion_vector.fuse(vector, graph)

        # With graph preference (alpha=0.2)
        fusion_graph = RRFFusion(RRFConfig(alpha=0.2))
        result_graph = fusion_graph.fuse(vector, graph)

        # a should rank higher with vector preference
        # b should rank higher with graph preference
        assert result_vector[0].memory_id == "a"
        assert result_graph[0].memory_id == "b"


class TestRRFEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_inputs(self):
        """Test handling of empty result lists."""
        fusion = RRFFusion(RRFConfig())

        result = fusion.fuse([], [])
        assert result == []

    def test_disabled_mode(self):
        """Test that disabled RRF returns vector results as-is."""
        config = RRFConfig(enabled=False)
        fusion = RRFFusion(config)

        vector = [
            RetrievalResult("a", 1, 0.9, "vector"),
            RetrievalResult("b", 2, 0.8, "vector"),
        ]
        graph = [RetrievalResult("c", 1, 0.95, "graph")]

        result = fusion.fuse(vector, graph)

        # Should only contain vector results
        assert len(result) == 2
        assert all(r.vector_rank is not None for r in result)
        assert all(r.graph_rank is None for r in result)

    def test_fuse_with_stats(self):
        """Test that fuse_with_stats returns correct statistics."""
        fusion = RRFFusion(RRFConfig())

        vector = [
            RetrievalResult("a", 1, 0.9, "vector"),
            RetrievalResult("b", 2, 0.8, "vector"),
        ]
        graph = [
            RetrievalResult("b", 1, 0.85, "graph"),
            RetrievalResult("c", 2, 0.7, "graph"),
        ]

        result = fusion.fuse_with_stats(vector, graph)

        assert "results" in result
        assert "stats" in result

        stats = result["stats"]
        assert stats["vector_candidates"] == 2
        assert stats["graph_candidates"] == 2
        assert stats["fused_total"] == 3  # a, b, c
        assert stats["in_both_sources"] == 1  # only b
        assert stats["vector_only"] == 1  # a
        assert stats["graph_only"] == 1  # c


class TestRRFConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RRFConfig()

        assert config.k == 60
        assert config.alpha == 0.6
        assert config.missing_rank_penalty == 100
        assert config.enabled == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RRFConfig(k=30, alpha=0.8, missing_rank_penalty=50, enabled=False)

        assert config.k == 30
        assert config.alpha == 0.8
        assert config.missing_rank_penalty == 50
        assert config.enabled == False
