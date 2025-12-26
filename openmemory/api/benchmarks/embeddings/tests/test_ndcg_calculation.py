"""
Unit tests for Normalized Discounted Cumulative Gain (NDCG) calculation.

NDCG measures ranking quality with graded relevance scores, accounting
for the position of relevant documents (higher positions are more valuable).

Tests written FIRST following TDD approach.
"""

import pytest
from typing import List, Dict

# Import will fail until implementation is written (TDD red phase)
from benchmarks.embeddings.metrics.ndcg import calculate_ndcg, NDCGResult


class TestNDCGBasicCalculation:
    """Test basic NDCG formula: NDCG = DCG / IDCG"""

    def test_ndcg_perfect_ranking_returns_1(self):
        """Perfect ranking should return NDCG = 1.0."""
        # Docs ranked by decreasing relevance: 2, 1, 0, 0, 0
        ranked_results = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevance_scores = [{"doc1": 2, "doc2": 1, "doc3": 0, "doc4": 0, "doc5": 0}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        assert abs(result.score - 1.0) < 0.001

    def test_ndcg_reversed_ranking_less_than_1(self):
        """Worst possible ranking should have NDCG < 1.0."""
        # Docs ranked in reverse relevance order
        ranked_results = [["doc3", "doc2", "doc1"]]  # worst to best
        relevance_scores = [{"doc1": 2, "doc2": 1, "doc3": 0}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        assert result.score < 1.0

    def test_ndcg_all_irrelevant_returns_0(self):
        """When all docs are irrelevant (grade 0), NDCG = 0."""
        ranked_results = [["doc1", "doc2", "doc3"]]
        relevance_scores = [{"doc1": 0, "doc2": 0, "doc3": 0}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        assert result.score == 0.0


class TestNDCGGradedRelevance:
    """Test NDCG with multi-level graded relevance."""

    def test_ndcg_binary_relevance(self):
        """NDCG works with binary relevance (0 or 1)."""
        ranked_results = [["doc1", "doc2", "doc3"]]
        relevance_scores = [{"doc1": 1, "doc2": 0, "doc3": 1}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        # Relevant at rank 1 and 3, irrelevant at rank 2
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # NDCG = 1.5 / 1.631 ≈ 0.92
        assert 0.9 < result.score < 1.0

    def test_ndcg_three_level_relevance(self):
        """NDCG with 0/1/2 graded relevance."""
        ranked_results = [["doc1", "doc2", "doc3", "doc4"]]
        relevance_scores = [{"doc1": 2, "doc2": 1, "doc3": 0, "doc4": 0}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        # Perfect ranking, so NDCG = 1.0
        assert abs(result.score - 1.0) < 0.001

    def test_ndcg_higher_grade_more_valuable(self):
        """Higher relevance grades should contribute more to score."""
        # Same docs, different order
        ranked_results_good = [["high", "low"]]
        ranked_results_bad = [["low", "high"]]
        relevance = [{"high": 2, "low": 1}]

        result_good = calculate_ndcg(ranked_results_good, relevance)
        result_bad = calculate_ndcg(ranked_results_bad, relevance)

        assert result_good.score > result_bad.score


class TestNDCGWithKCutoff:
    """Test NDCG@k - only consider top k results."""

    def test_ndcg_at_k_ignores_beyond_k(self):
        """NDCG@k should only consider top k positions."""
        ranked_results = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevance_scores = [{
            "doc1": 0, "doc2": 0, "doc3": 0, "doc4": 2, "doc5": 2
        }]

        # With k=3, the relevant docs at 4,5 are ignored
        result = calculate_ndcg(ranked_results, relevance_scores, k=3)

        assert result.score == 0.0

    def test_ndcg_at_k_counts_within_k(self):
        """NDCG@k should count relevant docs within position k."""
        ranked_results = [["doc1", "doc2", "doc3", "doc4", "doc5"]]
        relevance_scores = [{"doc1": 2, "doc2": 1, "doc3": 0, "doc4": 0, "doc5": 0}]

        result_k3 = calculate_ndcg(ranked_results, relevance_scores, k=3)
        result_k5 = calculate_ndcg(ranked_results, relevance_scores, k=5)

        # Both should be 1.0 since the first 3 are perfectly ranked
        assert abs(result_k3.score - 1.0) < 0.001
        assert abs(result_k5.score - 1.0) < 0.001

    def test_ndcg_at_k_default_is_10(self):
        """Default k should be 10."""
        ranked_results = [["d" + str(i) for i in range(15)]]
        relevance_scores = [{f"d{i}": (1 if i < 10 else 0) for i in range(15)}]

        result = calculate_ndcg(ranked_results, relevance_scores)  # default k=10

        # Should only consider first 10
        assert result.score > 0


class TestNDCGMultipleQueries:
    """Test NDCG across multiple queries."""

    def test_ndcg_averages_across_queries(self):
        """NDCG should average across all queries."""
        ranked_results = [
            ["a", "b", "c"],  # Query 1
            ["c", "b", "a"],  # Query 2
        ]
        relevance_scores = [
            {"a": 2, "b": 1, "c": 0},  # Query 1: perfect
            {"a": 2, "b": 1, "c": 0},  # Query 2: reversed
        ]

        result = calculate_ndcg(ranked_results, relevance_scores)

        # Average of 1.0 and <1.0
        assert 0.5 < result.score < 1.0
        assert result.num_queries == 2

    def test_ndcg_handles_varying_query_lengths(self):
        """NDCG handles queries with different result lengths."""
        ranked_results = [
            ["a", "b", "c", "d", "e"],  # 5 results
            ["x", "y"],                  # 2 results
        ]
        relevance_scores = [
            {"a": 1, "b": 0, "c": 0, "d": 0, "e": 0},
            {"x": 1, "y": 0},
        ]

        result = calculate_ndcg(ranked_results, relevance_scores)

        # Both queries have perfect ranking (relevant at position 1)
        assert abs(result.score - 1.0) < 0.001


class TestNDCGEdgeCases:
    """Test edge cases and error handling."""

    def test_ndcg_empty_results(self):
        """Empty result set should return 0."""
        ranked_results: List[List[str]] = [[]]
        relevance_scores: List[Dict[str, int]] = [{}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        assert result.score == 0.0

    def test_ndcg_no_queries(self):
        """No queries should return 0."""
        result = calculate_ndcg([], [])

        assert result.score == 0.0
        assert result.num_queries == 0

    def test_ndcg_missing_relevance_treated_as_0(self):
        """Docs not in relevance dict treated as relevance 0."""
        ranked_results = [["doc1", "doc2", "unknown"]]
        relevance_scores = [{"doc1": 2, "doc2": 1}]  # unknown not specified

        result = calculate_ndcg(ranked_results, relevance_scores)

        # Should not crash, unknown treated as 0
        assert result.score > 0

    def test_ndcg_mismatched_lengths_raises(self):
        """Mismatched query and relevance lists should raise."""
        ranked_results = [["a", "b"]]
        relevance_scores = [{"a": 1}, {"b": 1}]  # 2 dicts for 1 query

        with pytest.raises(ValueError):
            calculate_ndcg(ranked_results, relevance_scores)


class TestNDCGDiscountFunction:
    """Test the logarithmic discount function."""

    def test_ndcg_position_discount(self):
        """Later positions should be discounted more heavily."""
        # Same relevance, different positions
        ranked_at_1 = [["relevant", "other"]]
        ranked_at_2 = [["other", "relevant"]]
        relevance = [{"relevant": 1, "other": 0}]

        dcg_1 = calculate_ndcg(ranked_at_1, relevance)
        dcg_2 = calculate_ndcg(ranked_at_2, relevance)

        # Position 1 should contribute more
        assert dcg_1.score > dcg_2.score


class TestNDCGResultDataclass:
    """Test NDCGResult dataclass properties."""

    def test_ndcg_result_has_score(self):
        """NDCGResult should have score attribute."""
        result = calculate_ndcg([["a"]], [{"a": 1}])
        assert hasattr(result, "score")
        assert isinstance(result.score, float)

    def test_ndcg_result_has_num_queries(self):
        """NDCGResult should have num_queries attribute."""
        result = calculate_ndcg([["a"], ["b"]], [{"a": 1}, {"b": 1}])
        assert hasattr(result, "num_queries")
        assert result.num_queries == 2

    def test_ndcg_score_bounded_0_to_1(self):
        """NDCG score should always be between 0 and 1."""
        ranked_results = [["a", "b", "c"]]
        relevance_scores = [{"a": 0, "b": 2, "c": 1}]

        result = calculate_ndcg(ranked_results, relevance_scores)

        assert 0.0 <= result.score <= 1.0
