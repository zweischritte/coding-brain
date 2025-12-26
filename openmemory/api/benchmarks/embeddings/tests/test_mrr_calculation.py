"""
Unit tests for Mean Reciprocal Rank (MRR) calculation.

MRR is a standard IR metric that measures the average of reciprocal ranks
of the first relevant document for each query.

Tests written FIRST following TDD approach.
"""

import pytest
from typing import List, Set

# Import will fail until implementation is written (TDD red phase)
from benchmarks.embeddings.metrics.mrr import calculate_mrr, MRRResult


class TestMRRBasicCalculation:
    """Test basic MRR formula: MRR = (1/N) * sum(1/rank_i)"""

    def test_mrr_perfect_ranking_returns_1(self):
        """When relevant doc is at rank 1 for all queries, MRR = 1.0."""
        ranked_results = [
            ["relevant_doc", "other1", "other2"],
            ["relevant_doc", "other1", "other2"],
        ]
        relevant_docs = [{"relevant_doc"}, {"relevant_doc"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 1.0
        assert result.num_queries == 2
        assert result.queries_with_relevant == 2

    def test_mrr_relevant_at_rank_2_returns_0_5(self):
        """When relevant doc is at rank 2, reciprocal rank = 0.5."""
        ranked_results = [["other", "relevant_doc", "other2"]]
        relevant_docs = [{"relevant_doc"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 0.5

    def test_mrr_relevant_at_rank_3_returns_0_33(self):
        """When relevant doc is at rank 3, reciprocal rank = 1/3."""
        ranked_results = [["a", "b", "relevant_doc"]]
        relevant_docs = [{"relevant_doc"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        assert abs(result.score - 1/3) < 0.001

    def test_mrr_no_relevant_docs_returns_0(self):
        """When no relevant docs are found, contribution = 0."""
        ranked_results = [["other1", "other2", "other3"]]
        relevant_docs = [{"relevant_doc"}]  # Not in results

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 0.0
        assert result.queries_with_relevant == 0


class TestMRRMultipleQueries:
    """Test MRR across multiple queries."""

    def test_mrr_averages_across_queries(self):
        """MRR should average reciprocal ranks across all queries."""
        ranked_results = [
            ["relevant", "b", "c"],      # rank 1 -> 1/1 = 1.0
            ["a", "relevant", "c"],      # rank 2 -> 1/2 = 0.5
            ["a", "b", "relevant"],      # rank 3 -> 1/3 = 0.333
        ]
        relevant_docs = [{"relevant"}, {"relevant"}, {"relevant"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        expected = (1.0 + 0.5 + 1/3) / 3
        assert abs(result.score - expected) < 0.001
        assert result.num_queries == 3

    def test_mrr_mixed_found_and_not_found(self):
        """MRR handles mix of found and not found relevant docs."""
        ranked_results = [
            ["relevant", "b", "c"],  # rank 1 -> 1.0
            ["a", "b", "c"],         # not found -> 0
        ]
        relevant_docs = [{"relevant"}, {"relevant"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        expected = (1.0 + 0.0) / 2
        assert result.score == expected
        assert result.queries_with_relevant == 1


class TestMRRWithKCutoff:
    """Test MRR@k - only consider top k results."""

    def test_mrr_at_k_ignores_beyond_k(self):
        """MRR@k should not count relevant docs beyond position k."""
        ranked_results = [["a", "b", "c", "d", "relevant"]]  # rank 5
        relevant_docs = [{"relevant"}]

        # With k=3, relevant at rank 5 is ignored
        result = calculate_mrr(ranked_results, relevant_docs, k=3)

        assert result.score == 0.0

    def test_mrr_at_k_counts_within_k(self):
        """MRR@k should count relevant docs within position k."""
        ranked_results = [["a", "relevant", "c", "d", "e"]]  # rank 2
        relevant_docs = [{"relevant"}]

        result = calculate_mrr(ranked_results, relevant_docs, k=3)

        assert result.score == 0.5  # 1/2

    def test_mrr_at_k_default_is_10(self):
        """Default k should be 10."""
        ranked_results = [
            ["a"] * 9 + ["relevant"] + ["b"] * 5  # rank 10
        ]
        relevant_docs = [{"relevant"}]

        result = calculate_mrr(ranked_results, relevant_docs)  # default k=10

        assert result.score == 0.1  # 1/10


class TestMRRMultipleRelevant:
    """Test MRR when multiple docs are relevant per query."""

    def test_mrr_uses_first_relevant_doc(self):
        """MRR uses rank of first relevant doc encountered."""
        ranked_results = [["a", "relevant1", "relevant2", "d"]]
        relevant_docs = [{"relevant1", "relevant2"}]  # Both are relevant

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 0.5  # First relevant at rank 2


class TestMRREdgeCases:
    """Test edge cases and error handling."""

    def test_mrr_empty_results(self):
        """Empty result set should not crash, return 0."""
        ranked_results: List[List[str]] = [[]]
        relevant_docs: List[Set[str]] = [{"doc1"}]

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 0.0
        assert result.num_queries == 1

    def test_mrr_empty_relevant_set(self):
        """Empty relevant set should return 0."""
        ranked_results = [["doc1", "doc2"]]
        relevant_docs: List[Set[str]] = [set()]

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.score == 0.0

    def test_mrr_no_queries(self):
        """No queries should return 0."""
        result = calculate_mrr([], [])

        assert result.score == 0.0
        assert result.num_queries == 0

    def test_mrr_mismatched_lengths_raises(self):
        """Mismatched query and relevant doc lists should raise."""
        ranked_results = [["a", "b"]]
        relevant_docs = [{"a"}, {"b"}]  # 2 sets for 1 query

        with pytest.raises(ValueError):
            calculate_mrr(ranked_results, relevant_docs)


class TestMRRResultDataclass:
    """Test MRRResult dataclass properties."""

    def test_mrr_result_has_score(self):
        """MRRResult should have score attribute."""
        result = calculate_mrr([["a"]], [{"a"}])
        assert hasattr(result, "score")
        assert isinstance(result.score, float)

    def test_mrr_result_has_num_queries(self):
        """MRRResult should have num_queries attribute."""
        result = calculate_mrr([["a"], ["b"]], [{"a"}, {"b"}])
        assert hasattr(result, "num_queries")
        assert result.num_queries == 2

    def test_mrr_result_has_queries_with_relevant(self):
        """MRRResult should track how many queries had relevant docs."""
        ranked_results = [["a"], ["b"]]
        relevant_docs = [{"a"}, {"x"}]  # Only first query finds relevant

        result = calculate_mrr(ranked_results, relevant_docs)

        assert result.queries_with_relevant == 1
