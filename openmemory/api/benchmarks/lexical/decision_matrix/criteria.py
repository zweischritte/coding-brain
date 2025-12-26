"""
Decision matrix criteria for lexical backend selection.

Criteria and weights from implementation plan v7:
- Latency (40%): Query latency P95
- Ops complexity (20%): Operational complexity
- Scalability (20%): Scalability to 1M+ docs
- Feature support (20%): BM25, RRF, hybrid support
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class CriterionName(Enum):
    """Names of evaluation criteria."""
    LATENCY = "latency"
    OPS_COMPLEXITY = "ops_complexity"
    SCALABILITY = "scalability"
    FEATURE_SUPPORT = "feature_support"


@dataclass
class Criterion:
    """
    A decision matrix criterion.

    Attributes:
        name: Criterion identifier.
        weight: Weight for scoring (0.0 to 1.0). All weights must sum to 1.0.
        description: Human-readable description.
    """
    name: CriterionName
    weight: float
    description: str


# Default criteria as defined in implementation plan v7
CRITERIA: List[Criterion] = [
    Criterion(
        name=CriterionName.LATENCY,
        weight=0.40,
        description="Query latency P95"
    ),
    Criterion(
        name=CriterionName.OPS_COMPLEXITY,
        weight=0.20,
        description="Operational complexity"
    ),
    Criterion(
        name=CriterionName.SCALABILITY,
        weight=0.20,
        description="Scalability to 1M+ docs"
    ),
    Criterion(
        name=CriterionName.FEATURE_SUPPORT,
        weight=0.20,
        description="BM25, RRF, hybrid support"
    ),
]
