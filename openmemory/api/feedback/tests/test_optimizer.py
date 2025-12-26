"""Tests for RRF weight optimizer.

TDD tests for the nightly weight optimizer following v9 plan:
- Learns from feedback data
- Proposes new weights based on acceptance rates
- Validates weight changes against thresholds
- Supports gradual rollout
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmemory.api.feedback.events import FeedbackEvent, FeedbackOutcome
from openmemory.api.feedback.optimizer import (
    OptimizationConfig,
    OptimizationResult,
    RRFWeights,
    RRFWeightOptimizer,
    WeightProposal,
)
from openmemory.api.feedback.store import InMemoryFeedbackStore


class TestRRFWeights:
    """Tests for RRFWeights dataclass."""

    def test_create_weights(self):
        """Can create RRF weights."""
        weights = RRFWeights(vector=0.40, lexical=0.35, graph=0.25)
        assert weights.vector == 0.40
        assert weights.lexical == 0.35
        assert weights.graph == 0.25

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        weights = RRFWeights(vector=0.40, lexical=0.35, graph=0.25)
        assert abs(weights.total - 1.0) < 0.01

    def test_weights_validation_rejects_invalid_sum(self):
        """Weights that don't sum to 1.0 raise error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            RRFWeights(vector=0.50, lexical=0.50, graph=0.50)

    def test_weights_validation_rejects_negative(self):
        """Negative weights raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            RRFWeights(vector=-0.10, lexical=0.60, graph=0.50)

    def test_weights_to_dict(self):
        """Weights can be serialized."""
        weights = RRFWeights(vector=0.40, lexical=0.35, graph=0.25)
        d = weights.to_dict()
        assert d["vector"] == 0.40
        assert d["lexical"] == 0.35
        assert d["graph"] == 0.25

    def test_weights_from_dict(self):
        """Weights can be deserialized."""
        data = {"vector": 0.50, "lexical": 0.30, "graph": 0.20}
        weights = RRFWeights.from_dict(data)
        assert weights.vector == 0.50
        assert weights.lexical == 0.30
        assert weights.graph == 0.20

    def test_default_weights(self):
        """Default weights match v9 plan."""
        weights = RRFWeights.default()
        assert weights.vector == 0.40
        assert weights.lexical == 0.35
        assert weights.graph == 0.25


class TestWeightProposal:
    """Tests for WeightProposal dataclass."""

    def test_create_proposal(self):
        """Can create a weight proposal."""
        current = RRFWeights(vector=0.40, lexical=0.35, graph=0.25)
        proposed = RRFWeights(vector=0.45, lexical=0.35, graph=0.20)
        proposal = WeightProposal(
            current_weights=current,
            proposed_weights=proposed,
            confidence=0.85,
            sample_size=1000,
        )
        assert proposal.current_weights == current
        assert proposal.proposed_weights == proposed
        assert proposal.confidence == 0.85

    def test_proposal_with_expected_improvement(self):
        """Proposal includes expected improvement."""
        proposal = WeightProposal(
            current_weights=RRFWeights.default(),
            proposed_weights=RRFWeights(vector=0.45, lexical=0.35, graph=0.20),
            confidence=0.85,
            sample_size=1000,
            expected_improvement=0.05,  # 5% improvement
        )
        assert proposal.expected_improvement == 0.05

    def test_proposal_with_reason(self):
        """Proposal can include reasoning."""
        proposal = WeightProposal(
            current_weights=RRFWeights.default(),
            proposed_weights=RRFWeights(vector=0.45, lexical=0.35, graph=0.20),
            confidence=0.85,
            sample_size=1000,
            reason="Vector search showed higher acceptance rate",
        )
        assert "Vector" in proposal.reason


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = OptimizationConfig()
        assert config.min_sample_size == 100
        assert config.min_confidence == 0.75
        assert config.max_weight_change == 0.10

    def test_custom_config(self):
        """Can customize config."""
        config = OptimizationConfig(
            min_sample_size=500,
            min_confidence=0.90,
            max_weight_change=0.05,
        )
        assert config.min_sample_size == 500
        assert config.min_confidence == 0.90


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_create_result_with_proposal(self):
        """Result with a proposal."""
        proposal = WeightProposal(
            current_weights=RRFWeights.default(),
            proposed_weights=RRFWeights(vector=0.45, lexical=0.35, graph=0.20),
            confidence=0.85,
            sample_size=1000,
        )
        result = OptimizationResult(
            success=True,
            proposal=proposal,
            message="Optimization complete",
        )
        assert result.success is True
        assert result.proposal is not None

    def test_create_result_no_change(self):
        """Result when no change is recommended."""
        result = OptimizationResult(
            success=True,
            proposal=None,
            message="Current weights are optimal",
        )
        assert result.success is True
        assert result.proposal is None

    def test_create_result_insufficient_data(self):
        """Result when insufficient data."""
        result = OptimizationResult(
            success=False,
            proposal=None,
            message="Insufficient data for optimization",
            error_code="INSUFFICIENT_DATA",
        )
        assert result.success is False
        assert result.error_code == "INSUFFICIENT_DATA"


class TestRRFWeightOptimizer:
    """Tests for RRFWeightOptimizer class."""

    @pytest.fixture
    def store(self) -> InMemoryFeedbackStore:
        """Create a feedback store with sample data."""
        store = InMemoryFeedbackStore()
        return store

    @pytest.fixture
    def optimizer(self, store: InMemoryFeedbackStore) -> RRFWeightOptimizer:
        """Create an optimizer."""
        return RRFWeightOptimizer(store=store)

    def test_optimize_with_no_data(self, optimizer: RRFWeightOptimizer):
        """Returns no proposal when no feedback data."""
        result = optimizer.optimize("org789")
        assert result.success is False
        assert "insufficient" in result.message.lower()

    def test_optimize_with_insufficient_data(
        self, optimizer: RRFWeightOptimizer, store: InMemoryFeedbackStore
    ):
        """Returns no proposal when insufficient data."""
        # Add only a few events (less than min_sample_size)
        for i in range(10):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights={"vector": 0.40, "lexical": 0.35, "graph": 0.25},
                )
            )

        result = optimizer.optimize("org789")
        assert result.success is False
        assert "insufficient" in result.message.lower()

    def test_optimize_returns_proposal(
        self, optimizer: RRFWeightOptimizer, store: InMemoryFeedbackStore
    ):
        """Returns proposal when sufficient data with clear winner."""
        # Add events where vector-heavy weights performed better
        weights_high_vector = {"vector": 0.50, "lexical": 0.30, "graph": 0.20}
        weights_baseline = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}

        # High vector weights: 90% acceptance (200 samples for high confidence)
        for i in range(180):
            store.append(
                FeedbackEvent(
                    query_id=f"high_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights=weights_high_vector,
                )
            )
        for i in range(20):
            store.append(
                FeedbackEvent(
                    query_id=f"high_reject_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.REJECTED,
                    rrf_weights=weights_high_vector,
                )
            )

        # Baseline weights: 60% acceptance
        for i in range(60):
            store.append(
                FeedbackEvent(
                    query_id=f"base_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights=weights_baseline,
                )
            )
        for i in range(40):
            store.append(
                FeedbackEvent(
                    query_id=f"base_reject_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.REJECTED,
                    rrf_weights=weights_baseline,
                )
            )

        result = optimizer.optimize("org789")
        assert result.success is True
        assert result.proposal is not None
        # Should propose moving toward the better-performing weights
        assert result.proposal.proposed_weights.vector > 0.40

    def test_optimize_respects_max_weight_change(
        self, store: InMemoryFeedbackStore
    ):
        """Proposal respects max weight change limit."""
        config = OptimizationConfig(max_weight_change=0.05)
        optimizer = RRFWeightOptimizer(store=store, config=config)

        # Add events favoring very different weights
        for i in range(100):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights={"vector": 0.70, "lexical": 0.20, "graph": 0.10},
                )
            )

        # Add some baseline events
        for i in range(50):
            store.append(
                FeedbackEvent(
                    query_id=f"base_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.REJECTED,
                    rrf_weights={"vector": 0.40, "lexical": 0.35, "graph": 0.25},
                )
            )

        result = optimizer.optimize("org789", current_weights=RRFWeights.default())

        if result.proposal:
            # Change should be at most 0.05 per weight
            assert abs(result.proposal.proposed_weights.vector - 0.40) <= 0.051
            assert abs(result.proposal.proposed_weights.lexical - 0.35) <= 0.051
            assert abs(result.proposal.proposed_weights.graph - 0.25) <= 0.051

    def test_optimize_no_change_when_optimal(
        self, optimizer: RRFWeightOptimizer, store: InMemoryFeedbackStore
    ):
        """Returns no proposal when current weights are optimal."""
        weights = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}

        # All events with same weights, good acceptance
        for i in range(100):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights=weights,
                )
            )

        result = optimizer.optimize("org789")

        # Either no proposal or proposal is same as current
        if result.proposal:
            assert abs(result.proposal.proposed_weights.vector - 0.40) < 0.05
            assert abs(result.proposal.proposed_weights.lexical - 0.35) < 0.05
            assert abs(result.proposal.proposed_weights.graph - 0.25) < 0.05


class TestRRFWeightOptimizerAnalytics:
    """Tests for optimizer analytics methods."""

    @pytest.fixture
    def store(self) -> InMemoryFeedbackStore:
        return InMemoryFeedbackStore()

    @pytest.fixture
    def optimizer(self, store: InMemoryFeedbackStore) -> RRFWeightOptimizer:
        return RRFWeightOptimizer(store=store)

    def test_get_acceptance_by_weights(
        self, optimizer: RRFWeightOptimizer, store: InMemoryFeedbackStore
    ):
        """Can calculate acceptance rate for weight configurations."""
        weights = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}

        # 70% acceptance rate
        for i in range(70):
            store.append(
                FeedbackEvent(
                    query_id=f"accept_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    rrf_weights=weights,
                )
            )
        for i in range(30):
            store.append(
                FeedbackEvent(
                    query_id=f"reject_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.REJECTED,
                    rrf_weights=weights,
                )
            )

        stats = optimizer.get_weight_statistics("org789")
        assert len(stats) >= 1

        # Find our weights in stats
        found = False
        for stat in stats:
            if (
                abs(stat["weights"]["vector"] - 0.40) < 0.01
                and abs(stat["weights"]["lexical"] - 0.35) < 0.01
            ):
                assert stat["acceptance_rate"] == pytest.approx(0.70, rel=0.01)
                assert stat["sample_size"] == 100
                found = True
                break
        assert found, "Expected weight configuration not found in stats"

    def test_get_weight_trends(
        self, optimizer: RRFWeightOptimizer, store: InMemoryFeedbackStore
    ):
        """Can track weight performance trends over time."""
        weights = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}
        now = datetime.now(timezone.utc)

        # Add events over time
        for day in range(7):
            timestamp = now - timedelta(days=6 - day)
            for i in range(10):
                store.append(
                    FeedbackEvent(
                        query_id=f"q_day{day}_{i}",
                        user_id="u456",
                        org_id="org789",
                        tool_name="search_code_hybrid",
                        outcome=FeedbackOutcome.ACCEPTED,
                        rrf_weights=weights,
                        timestamp=timestamp,
                    )
                )

        trends = optimizer.get_weight_trends(
            org_id="org789",
            since=now - timedelta(days=7),
        )
        assert len(trends) >= 1
