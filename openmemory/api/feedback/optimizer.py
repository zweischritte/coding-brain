"""RRF weight optimizer for retrieval tuning.

Implements FR-002: Feedback Integration - nightly RRF weight optimizer.
Learns from feedback data to propose improved weight configurations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from openmemory.api.feedback.events import FeedbackEvent, FeedbackOutcome
from openmemory.api.feedback.store import FeedbackStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RRFWeights:
    """RRF weight configuration.

    Per v9 plan default weights:
    - vector: 0.40
    - lexical: 0.35
    - graph: 0.25
    """

    vector: float
    lexical: float
    graph: float

    def __post_init__(self):
        """Validate weights."""
        for name, value in [
            ("vector", self.vector),
            ("lexical", self.lexical),
            ("graph", self.graph),
        ]:
            if value < 0:
                raise ValueError(f"Weight {name} must be non-negative, got {value}")

        total = self.total
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return self.vector + self.lexical + self.graph

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary."""
        return {
            "vector": self.vector,
            "lexical": self.lexical,
            "graph": self.graph,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> RRFWeights:
        """Deserialize from dictionary."""
        return cls(
            vector=data["vector"],
            lexical=data["lexical"],
            graph=data["graph"],
        )

    @classmethod
    def default(cls) -> RRFWeights:
        """Get default weights per v9 plan."""
        return cls(vector=0.40, lexical=0.35, graph=0.25)


@dataclass
class WeightProposal:
    """A proposed weight change.

    Attributes:
        current_weights: Current weight configuration
        proposed_weights: Proposed new weights
        confidence: Confidence in the proposal (0-1)
        sample_size: Number of samples used
        expected_improvement: Expected improvement in acceptance rate
        reason: Explanation for the proposal
    """

    current_weights: RRFWeights
    proposed_weights: RRFWeights
    confidence: float
    sample_size: int
    expected_improvement: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for the weight optimizer.

    Attributes:
        min_sample_size: Minimum samples before optimization
        min_confidence: Minimum confidence to propose changes
        max_weight_change: Maximum change per weight per iteration
        lookback_days: Days of data to consider
    """

    min_sample_size: int = 100
    min_confidence: float = 0.75
    max_weight_change: float = 0.10
    lookback_days: int = 30


@dataclass
class OptimizationResult:
    """Result of an optimization run.

    Attributes:
        success: Whether optimization completed successfully
        proposal: Weight change proposal (if any)
        message: Human-readable status
        error_code: Error code if failed
    """

    success: bool
    proposal: Optional[WeightProposal]
    message: str
    error_code: Optional[str] = None


class RRFWeightOptimizer:
    """Optimizer for RRF retrieval weights.

    Analyzes feedback data to learn which weight configurations
    produce better user outcomes, and proposes gradual improvements.
    """

    def __init__(
        self,
        store: FeedbackStore,
        config: Optional[OptimizationConfig] = None,
    ):
        """Initialize optimizer.

        Args:
            store: Feedback store to read from
            config: Optional configuration
        """
        self._store = store
        self._config = config or OptimizationConfig()

    def optimize(
        self,
        org_id: str,
        current_weights: Optional[RRFWeights] = None,
        since: Optional[datetime] = None,
    ) -> OptimizationResult:
        """Run optimization to propose new weights.

        Args:
            org_id: Organization to optimize for
            current_weights: Current weight configuration
            since: Start of data window (default: lookback_days ago)

        Returns:
            OptimizationResult with proposal if improvement found
        """
        if current_weights is None:
            current_weights = RRFWeights.default()

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(
                days=self._config.lookback_days
            )

        # Get events with weight data
        events = self._store.query_for_optimization(org_id, since=since)

        if len(events) < self._config.min_sample_size:
            return OptimizationResult(
                success=False,
                proposal=None,
                message=f"Insufficient data: {len(events)} events (need {self._config.min_sample_size})",
                error_code="INSUFFICIENT_DATA",
            )

        # Group events by weight configuration
        weight_groups = self._group_by_weights(events)

        if len(weight_groups) < 1:
            return OptimizationResult(
                success=False,
                proposal=None,
                message="No weight configurations found in feedback data",
                error_code="NO_WEIGHT_DATA",
            )

        # Calculate acceptance rate for each configuration
        config_stats = []
        for weight_key, (weights_dict, group_events) in weight_groups.items():
            acceptance_rate = self._calculate_acceptance_rate(group_events)
            config_stats.append(
                {
                    "weights": weights_dict,
                    "acceptance_rate": acceptance_rate,
                    "sample_size": len(group_events),
                }
            )

        # Sort by acceptance rate (best first)
        config_stats.sort(key=lambda x: x["acceptance_rate"], reverse=True)

        # Find best performing weights
        best_config = config_stats[0]

        # Check if we have enough confidence
        best_weights = best_config["weights"]
        current_key = self._weight_to_key(current_weights.to_dict())

        # Find current weights stats
        current_stats = None
        for stat in config_stats:
            if self._weight_to_key(stat["weights"]) == current_key:
                current_stats = stat
                break

        # If current weights not in data or best is same as current
        if current_stats is None:
            # Use best weights as target
            target_weights = best_weights
            improvement = best_config["acceptance_rate"] - 0.5  # Assume 50% baseline
        elif self._weight_to_key(best_weights) == current_key:
            # Current weights are already optimal
            return OptimizationResult(
                success=True,
                proposal=None,
                message="Current weights are optimal (no improvement found)",
            )
        else:
            target_weights = best_weights
            improvement = (
                best_config["acceptance_rate"] - current_stats["acceptance_rate"]
            )

        # Calculate confidence based on sample size
        confidence = min(
            1.0, best_config["sample_size"] / (self._config.min_sample_size * 2)
        )

        if confidence < self._config.min_confidence:
            return OptimizationResult(
                success=True,
                proposal=None,
                message=f"Confidence too low ({confidence:.2f} < {self._config.min_confidence})",
            )

        # Create proposed weights with max change limit
        proposed = self._clamp_weights(
            current_weights,
            RRFWeights.from_dict(target_weights),
            self._config.max_weight_change,
        )

        proposal = WeightProposal(
            current_weights=current_weights,
            proposed_weights=proposed,
            confidence=confidence,
            sample_size=sum(s["sample_size"] for s in config_stats),
            expected_improvement=improvement,
            reason=f"Weight config with {best_config['acceptance_rate']:.1%} acceptance rate outperformed current by {improvement:.1%}",
        )

        return OptimizationResult(
            success=True,
            proposal=proposal,
            message="Optimization complete - weight change proposed",
        )

    def get_weight_statistics(self, org_id: str) -> list[dict[str, Any]]:
        """Get statistics for each weight configuration.

        Args:
            org_id: Organization to analyze

        Returns:
            List of stats dicts with weights, acceptance_rate, sample_size
        """
        events = self._store.query_for_optimization(org_id)
        weight_groups = self._group_by_weights(events)

        stats = []
        for weight_key, (weights_dict, group_events) in weight_groups.items():
            acceptance_rate = self._calculate_acceptance_rate(group_events)
            stats.append(
                {
                    "weights": weights_dict,
                    "acceptance_rate": acceptance_rate,
                    "sample_size": len(group_events),
                }
            )

        return stats

    def get_weight_trends(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        bucket_days: int = 1,
    ) -> list[dict[str, Any]]:
        """Get weight performance trends over time.

        Args:
            org_id: Organization to analyze
            since: Start of analysis window
            bucket_days: Days per bucket

        Returns:
            List of trend data points
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(
                days=self._config.lookback_days
            )

        events = self._store.query_for_optimization(org_id, since=since)

        # Group by time buckets
        buckets: dict[str, list[FeedbackEvent]] = defaultdict(list)
        for event in events:
            bucket_key = event.timestamp.strftime("%Y-%m-%d")
            buckets[bucket_key].append(event)

        trends = []
        for date_key, bucket_events in sorted(buckets.items()):
            weight_groups = self._group_by_weights(bucket_events)
            for weight_key, (weights_dict, group_events) in weight_groups.items():
                trends.append(
                    {
                        "date": date_key,
                        "weights": weights_dict,
                        "acceptance_rate": self._calculate_acceptance_rate(group_events),
                        "sample_size": len(group_events),
                    }
                )

        return trends

    def _group_by_weights(
        self, events: list[FeedbackEvent]
    ) -> dict[str, tuple[dict, list[FeedbackEvent]]]:
        """Group events by their weight configuration.

        Returns:
            Dict mapping weight key to (weights_dict, events_list) tuples
        """
        groups: dict[str, list[FeedbackEvent]] = defaultdict(list)
        weight_map: dict[str, dict] = {}

        for event in events:
            if event.rrf_weights:
                key = self._weight_to_key(event.rrf_weights)
                groups[key].append(event)
                weight_map[key] = event.rrf_weights

        # Return tuples of (weights, events)
        return {k: (weight_map[k], v) for k, v in groups.items()}

    def _weight_to_key(self, weights: dict[str, float]) -> str:
        """Convert weights dict to a hashable key."""
        return f"v{weights.get('vector', 0):.2f}_l{weights.get('lexical', 0):.2f}_g{weights.get('graph', 0):.2f}"

    def _calculate_acceptance_rate(self, events: list[FeedbackEvent]) -> float:
        """Calculate acceptance rate for a group of events."""
        if not events:
            return 0.0

        # Count accepted outcomes
        accepted = sum(
            1 for e in events if e.outcome == FeedbackOutcome.ACCEPTED
        )
        # Exclude ignored from denominator
        total = sum(
            1 for e in events if e.outcome != FeedbackOutcome.IGNORED
        )

        return accepted / total if total > 0 else 0.0

    def _clamp_weights(
        self,
        current: RRFWeights,
        target: RRFWeights,
        max_change: float,
    ) -> RRFWeights:
        """Clamp weight changes to max_change limit.

        Ensures proposed changes don't exceed max_change per weight.
        Also ensures weights still sum to 1.0.
        """
        # Calculate clamped values
        vector = self._clamp_value(current.vector, target.vector, max_change)
        lexical = self._clamp_value(current.lexical, target.lexical, max_change)
        graph = self._clamp_value(current.graph, target.graph, max_change)

        # Normalize to sum to 1.0
        total = vector + lexical + graph
        if total > 0:
            vector /= total
            lexical /= total
            graph /= total

        return RRFWeights(vector=vector, lexical=lexical, graph=graph)

    def _clamp_value(
        self,
        current: float,
        target: float,
        max_change: float,
    ) -> float:
        """Clamp a single value change."""
        diff = target - current
        if abs(diff) > max_change:
            diff = max_change if diff > 0 else -max_change
        return max(0.0, current + diff)
