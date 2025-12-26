"""
Decision matrix evaluator for lexical backend selection.

Evaluates backends against weighted criteria and determines winner.
"""

from dataclasses import dataclass
from typing import Dict, List

from .criteria import Criterion, CriterionName, CRITERIA


@dataclass
class BackendScore:
    """
    Scores for a single backend.

    Attributes:
        backend_name: Name of the backend (e.g., "tantivy", "opensearch").
        criterion_scores: Raw scores (0-1) per criterion.
        weighted_total: Final weighted score (0-1).
    """
    backend_name: str
    criterion_scores: Dict[CriterionName, float]
    weighted_total: float


class DecisionMatrixEvaluator:
    """
    Evaluate backends against decision matrix criteria.

    Usage:
        evaluator = DecisionMatrixEvaluator()

        tantivy = evaluator.evaluate("tantivy", {
            CriterionName.LATENCY: 0.95,
            CriterionName.OPS_COMPLEXITY: 0.90,
            CriterionName.SCALABILITY: 0.60,
            CriterionName.FEATURE_SUPPORT: 0.80,
        })

        opensearch = evaluator.evaluate("opensearch", {...})

        winner = evaluator.compare([tantivy, opensearch])
    """

    def __init__(self, criteria: List[Criterion] = None):
        """
        Initialize evaluator.

        Args:
            criteria: List of criteria with weights. Defaults to CRITERIA from plan v7.
        """
        self.criteria = criteria if criteria is not None else CRITERIA

    def evaluate(
        self,
        backend_name: str,
        raw_scores: Dict[CriterionName, float]
    ) -> BackendScore:
        """
        Calculate weighted score for a backend.

        Args:
            backend_name: Name of backend being evaluated.
            raw_scores: Normalized scores (0-1) per criterion.

        Returns:
            BackendScore with weighted total.

        Raises:
            ValueError: If any score is outside 0-1 range.
        """
        # Validate scores are in valid range
        for criterion_name, score in raw_scores.items():
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"Score for {criterion_name.value} must be between 0 and 1, "
                    f"got {score}"
                )

        # Calculate weighted total
        weighted_total = 0.0
        for criterion in self.criteria:
            score = raw_scores.get(criterion.name, 0.0)
            weighted_total += score * criterion.weight

        return BackendScore(
            backend_name=backend_name,
            criterion_scores=raw_scores,
            weighted_total=weighted_total
        )

    def compare(self, scores: List[BackendScore]) -> str:
        """
        Compare backends and return winner.

        Args:
            scores: List of BackendScore objects to compare.

        Returns:
            Name of winning backend (highest weighted_total).

        Raises:
            ValueError: If scores list is empty.
        """
        if not scores:
            raise ValueError("Cannot compare empty list of scores")

        # Find backend with highest weighted_total
        winner = max(scores, key=lambda s: s.weighted_total)
        return winner.backend_name
