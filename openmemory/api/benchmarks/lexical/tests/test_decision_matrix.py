"""
Unit tests for lexical backend decision matrix.

The decision matrix evaluates Tantivy vs OpenSearch using weighted criteria:
- Latency (40%)
- Ops complexity (20%)
- Scalability (20%)
- Feature support (20%)

Tests written FIRST following TDD approach.
"""

import pytest
from typing import Dict

# Import will fail until implementation is written (TDD red phase)
from benchmarks.lexical.decision_matrix.criteria import (
    Criterion,
    CriterionName,
    CRITERIA,
)
from benchmarks.lexical.decision_matrix.evaluator import (
    BackendScore,
    DecisionMatrixEvaluator,
)


class TestCriteriaDefinition:
    """Test criteria are defined correctly per implementation plan."""

    def test_criteria_has_four_items(self):
        """Should have exactly 4 criteria."""
        assert len(CRITERIA) == 4

    def test_latency_criterion_exists(self):
        """Latency criterion should be defined."""
        latency = next(c for c in CRITERIA if c.name == CriterionName.LATENCY)
        assert latency is not None

    def test_ops_complexity_criterion_exists(self):
        """Ops complexity criterion should be defined."""
        ops = next(c for c in CRITERIA if c.name == CriterionName.OPS_COMPLEXITY)
        assert ops is not None

    def test_scalability_criterion_exists(self):
        """Scalability criterion should be defined."""
        scale = next(c for c in CRITERIA if c.name == CriterionName.SCALABILITY)
        assert scale is not None

    def test_feature_support_criterion_exists(self):
        """Feature support criterion should be defined."""
        features = next(c for c in CRITERIA if c.name == CriterionName.FEATURE_SUPPORT)
        assert features is not None


class TestCriteriaWeights:
    """Test criteria weights match implementation plan v7."""

    def test_weights_sum_to_100_percent(self):
        """All criterion weights must sum to 1.0 (100%)."""
        total = sum(c.weight for c in CRITERIA)
        assert abs(total - 1.0) < 0.001

    def test_latency_weight_is_40_percent(self):
        """Latency criterion has 40% weight."""
        latency = next(c for c in CRITERIA if c.name == CriterionName.LATENCY)
        assert latency.weight == 0.40

    def test_ops_complexity_weight_is_20_percent(self):
        """Ops complexity has 20% weight."""
        ops = next(c for c in CRITERIA if c.name == CriterionName.OPS_COMPLEXITY)
        assert ops.weight == 0.20

    def test_scalability_weight_is_20_percent(self):
        """Scalability has 20% weight."""
        scale = next(c for c in CRITERIA if c.name == CriterionName.SCALABILITY)
        assert scale.weight == 0.20

    def test_feature_support_weight_is_20_percent(self):
        """Feature support has 20% weight."""
        features = next(c for c in CRITERIA if c.name == CriterionName.FEATURE_SUPPORT)
        assert features.weight == 0.20


class TestCriterionDataclass:
    """Test Criterion dataclass properties."""

    def test_criterion_has_name(self):
        """Criterion should have name attribute."""
        c = Criterion(CriterionName.LATENCY, 0.4, "Test description")
        assert c.name == CriterionName.LATENCY

    def test_criterion_has_weight(self):
        """Criterion should have weight attribute."""
        c = Criterion(CriterionName.LATENCY, 0.4, "Test")
        assert c.weight == 0.4

    def test_criterion_has_description(self):
        """Criterion should have description attribute."""
        c = Criterion(CriterionName.LATENCY, 0.4, "Query latency P95")
        assert c.description == "Query latency P95"


class TestBackendScoreDataclass:
    """Test BackendScore dataclass properties."""

    def test_backend_score_has_name(self):
        """BackendScore should have backend_name."""
        score = BackendScore(
            backend_name="tantivy",
            criterion_scores={CriterionName.LATENCY: 0.9},
            weighted_total=0.36
        )
        assert score.backend_name == "tantivy"

    def test_backend_score_has_criterion_scores(self):
        """BackendScore should have criterion_scores dict."""
        scores = {CriterionName.LATENCY: 0.9, CriterionName.SCALABILITY: 0.7}
        score = BackendScore("test", scores, 0.5)
        assert score.criterion_scores == scores

    def test_backend_score_has_weighted_total(self):
        """BackendScore should have weighted_total."""
        score = BackendScore("test", {}, 0.85)
        assert score.weighted_total == 0.85


class TestDecisionMatrixEvaluator:
    """Test DecisionMatrixEvaluator functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with default criteria."""
        return DecisionMatrixEvaluator()

    def test_evaluator_uses_default_criteria(self, evaluator):
        """Evaluator should use CRITERIA by default."""
        assert evaluator.criteria == CRITERIA

    def test_evaluator_accepts_custom_criteria(self):
        """Evaluator can use custom criteria."""
        custom = [
            Criterion(CriterionName.LATENCY, 0.5, "Custom"),
            Criterion(CriterionName.SCALABILITY, 0.5, "Custom"),
        ]
        evaluator = DecisionMatrixEvaluator(criteria=custom)
        assert evaluator.criteria == custom


class TestEvaluatorScoreCalculation:
    """Test score calculation logic."""

    @pytest.fixture
    def evaluator(self):
        return DecisionMatrixEvaluator()

    def test_evaluate_calculates_weighted_total(self, evaluator):
        """Evaluate should calculate correct weighted total."""
        raw_scores = {
            CriterionName.LATENCY: 1.0,        # 1.0 * 0.4 = 0.4
            CriterionName.OPS_COMPLEXITY: 1.0,  # 1.0 * 0.2 = 0.2
            CriterionName.SCALABILITY: 1.0,     # 1.0 * 0.2 = 0.2
            CriterionName.FEATURE_SUPPORT: 1.0, # 1.0 * 0.2 = 0.2
        }

        result = evaluator.evaluate("perfect_backend", raw_scores)

        assert abs(result.weighted_total - 1.0) < 0.001

    def test_evaluate_with_varying_scores(self, evaluator):
        """Evaluate with different scores per criterion."""
        raw_scores = {
            CriterionName.LATENCY: 0.9,        # 0.9 * 0.4 = 0.36
            CriterionName.OPS_COMPLEXITY: 0.5,  # 0.5 * 0.2 = 0.10
            CriterionName.SCALABILITY: 0.8,     # 0.8 * 0.2 = 0.16
            CriterionName.FEATURE_SUPPORT: 0.7, # 0.7 * 0.2 = 0.14
        }
        # Total = 0.36 + 0.10 + 0.16 + 0.14 = 0.76

        result = evaluator.evaluate("mixed_backend", raw_scores)

        assert abs(result.weighted_total - 0.76) < 0.001

    def test_evaluate_preserves_raw_scores(self, evaluator):
        """Evaluate should preserve raw scores in result."""
        raw_scores = {
            CriterionName.LATENCY: 0.8,
            CriterionName.OPS_COMPLEXITY: 0.6,
            CriterionName.SCALABILITY: 0.9,
            CriterionName.FEATURE_SUPPORT: 0.7,
        }

        result = evaluator.evaluate("test", raw_scores)

        assert result.criterion_scores == raw_scores

    def test_evaluate_zero_scores(self, evaluator):
        """All zero scores should give zero total."""
        raw_scores = {
            CriterionName.LATENCY: 0.0,
            CriterionName.OPS_COMPLEXITY: 0.0,
            CriterionName.SCALABILITY: 0.0,
            CriterionName.FEATURE_SUPPORT: 0.0,
        }

        result = evaluator.evaluate("zero_backend", raw_scores)

        assert result.weighted_total == 0.0


class TestEvaluatorComparison:
    """Test backend comparison functionality."""

    @pytest.fixture
    def evaluator(self):
        return DecisionMatrixEvaluator()

    def test_compare_returns_winner_name(self, evaluator):
        """Compare should return name of winning backend."""
        tantivy = BackendScore("tantivy", {}, 0.85)
        opensearch = BackendScore("opensearch", {}, 0.75)

        winner = evaluator.compare([tantivy, opensearch])

        assert winner == "tantivy"

    def test_compare_higher_score_wins(self, evaluator):
        """Higher weighted_total should win."""
        backend_a = BackendScore("a", {}, 0.5)
        backend_b = BackendScore("b", {}, 0.9)

        winner = evaluator.compare([backend_a, backend_b])

        assert winner == "b"

    def test_compare_handles_tie(self, evaluator):
        """Tie should return first backend (or handle deterministically)."""
        backend_a = BackendScore("a", {}, 0.8)
        backend_b = BackendScore("b", {}, 0.8)

        winner = evaluator.compare([backend_a, backend_b])

        # Ties go to first in list
        assert winner == "a"

    def test_compare_single_backend(self, evaluator):
        """Single backend should be the winner."""
        only = BackendScore("only_one", {}, 0.5)

        winner = evaluator.compare([only])

        assert winner == "only_one"

    def test_compare_empty_list_raises(self, evaluator):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            evaluator.compare([])


class TestRealWorldScenarios:
    """Test realistic Tantivy vs OpenSearch scenarios."""

    @pytest.fixture
    def evaluator(self):
        return DecisionMatrixEvaluator()

    def test_tantivy_scenario(self, evaluator):
        """Tantivy: great latency and ops, moderate scale, good features."""
        tantivy_scores = {
            CriterionName.LATENCY: 0.95,        # Very fast
            CriterionName.OPS_COMPLEXITY: 0.90,  # Simple embedded library
            CriterionName.SCALABILITY: 0.60,     # Single-node limit
            CriterionName.FEATURE_SUPPORT: 0.80, # Good BM25, basic features
        }

        result = evaluator.evaluate("tantivy", tantivy_scores)

        # Should be competitive for small-medium deployments
        assert result.weighted_total > 0.7

    def test_opensearch_scenario(self, evaluator):
        """OpenSearch: good latency, complex ops, great scale, rich features."""
        opensearch_scores = {
            CriterionName.LATENCY: 0.75,         # Good but network overhead
            CriterionName.OPS_COMPLEXITY: 0.50,  # Cluster management needed
            CriterionName.SCALABILITY: 0.95,     # Excellent horizontal scale
            CriterionName.FEATURE_SUPPORT: 0.90, # Rich feature set
        }

        result = evaluator.evaluate("opensearch", opensearch_scores)

        # Should be competitive for large deployments
        assert result.weighted_total > 0.7

    def test_comparison_for_small_deployment(self, evaluator):
        """For small deployment (<1M docs), Tantivy should win."""
        # Tantivy excels at latency (40% weight) and ops simplicity
        tantivy = evaluator.evaluate("tantivy", {
            CriterionName.LATENCY: 0.95,
            CriterionName.OPS_COMPLEXITY: 0.90,
            CriterionName.SCALABILITY: 0.60,
            CriterionName.FEATURE_SUPPORT: 0.80,
        })

        opensearch = evaluator.evaluate("opensearch", {
            CriterionName.LATENCY: 0.75,
            CriterionName.OPS_COMPLEXITY: 0.50,
            CriterionName.SCALABILITY: 0.95,
            CriterionName.FEATURE_SUPPORT: 0.90,
        })

        winner = evaluator.compare([tantivy, opensearch])

        # Tantivy wins for small deployments
        assert winner == "tantivy"


class TestScoreNormalization:
    """Test that scores are properly bounded."""

    @pytest.fixture
    def evaluator(self):
        return DecisionMatrixEvaluator()

    def test_scores_must_be_0_to_1(self, evaluator):
        """Raw scores outside 0-1 range should raise or be clamped."""
        invalid_scores = {
            CriterionName.LATENCY: 1.5,  # Invalid
            CriterionName.OPS_COMPLEXITY: 0.5,
            CriterionName.SCALABILITY: 0.5,
            CriterionName.FEATURE_SUPPORT: 0.5,
        }

        with pytest.raises(ValueError):
            evaluator.evaluate("invalid", invalid_scores)

    def test_negative_scores_invalid(self, evaluator):
        """Negative scores should raise ValueError."""
        invalid_scores = {
            CriterionName.LATENCY: -0.5,  # Invalid
            CriterionName.OPS_COMPLEXITY: 0.5,
            CriterionName.SCALABILITY: 0.5,
            CriterionName.FEATURE_SUPPORT: 0.5,
        }

        with pytest.raises(ValueError):
            evaluator.evaluate("invalid", invalid_scores)

    def test_weighted_total_bounded_0_to_1(self, evaluator):
        """Weighted total should always be between 0 and 1."""
        scores = {
            CriterionName.LATENCY: 0.5,
            CriterionName.OPS_COMPLEXITY: 0.5,
            CriterionName.SCALABILITY: 0.5,
            CriterionName.FEATURE_SUPPORT: 0.5,
        }

        result = evaluator.evaluate("test", scores)

        assert 0.0 <= result.weighted_total <= 1.0
