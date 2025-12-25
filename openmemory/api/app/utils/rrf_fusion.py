"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Combines multiple retrieval sources using rank-based fusion.
No score normalization needed - RRF is inherently robust to
different score distributions.

Algorithm:
    RRF_score = α/(k + vector_rank) + (1-α)/(k + graph_rank)

Where:
    - k = 60 (smoothing constant from RRF research)
    - α = 0.6 (vector preference, configurable)
    - Missing rank = 100 (penalty for single-source results)

Reference:
    Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods" (2009)

Usage:
    from app.utils.rrf_fusion import RRFFusion, RRFConfig, RetrievalResult

    fusion = RRFFusion(RRFConfig(alpha=0.6))

    vector_results = [RetrievalResult("a", 1, 0.9, "vector")]
    graph_results = [RetrievalResult("b", 1, 0.8, "graph")]

    fused = fusion.fuse(vector_results, graph_results)
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RetrievalSource(Enum):
    """Source of a retrieval result."""
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass
class RetrievalResult:
    """
    Result from a single retrieval source.

    Attributes:
        memory_id: UUID of the memory
        rank: Position in source ranking (1-indexed)
        score: Original source score (not used in RRF, kept for debugging)
        source: Which retrieval system produced this result
        payload: Original result payload (metadata, content, etc.)
    """
    memory_id: str
    rank: int  # 1-indexed
    score: float  # Original source score
    source: str  # 'vector' or 'graph'
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RRFConfig:
    """
    Configuration for RRF fusion.

    Attributes:
        k: Smoothing constant (60 is standard from research)
        alpha: Vector weight (0.0 to 1.0)
            - 0.5 = equal weight
            - 0.6 = slight vector preference (default)
            - 0.7 = strong vector preference
            - 0.4 = slight graph preference (for GRAPH_PRIMARY route)
        missing_rank_penalty: Rank assigned when result is in only one source
        enabled: Master switch for RRF (when False, use vector-only)
    """
    k: int = 60                      # Smoothing constant
    alpha: float = 0.6               # Vector weight (0.0-1.0)
    missing_rank_penalty: int = 100  # Rank for missing source
    enabled: bool = True             # Master switch


@dataclass
class FusedResult:
    """
    Result after RRF fusion with full provenance.

    Attributes:
        memory_id: UUID of the memory
        rrf_score: Computed RRF score for ranking
        vector_rank: Original vector search rank (None if not in vector results)
        graph_rank: Original graph traversal rank (None if not in graph results)
        in_both: Whether this result appeared in both sources
        original_score: Score from the primary source
        payload: Merged payload from sources
        source_scores: Original scores from each source
    """
    memory_id: str
    rrf_score: float
    vector_rank: Optional[int]
    graph_rank: Optional[int]
    in_both: bool
    original_score: float
    payload: Dict[str, Any]
    source_scores: Dict[str, float] = field(default_factory=dict)


class RRFFusion:
    """
    Fuses results from vector and graph sources using RRF.

    RRF is ideal for hybrid retrieval because:
    1. No score normalization needed between heterogeneous sources
    2. Robust to different score distributions
    3. Rank-based = interpretable and debuggable
    4. Penalizes single-source results naturally
    """

    def __init__(self, config: Optional[RRFConfig] = None):
        self.config = config or RRFConfig()

    def fuse(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> List[FusedResult]:
        """
        Fuse vector and graph results using weighted RRF.

        Results appearing in both sources get higher RRF scores.
        Results in only one source are penalized with missing_rank_penalty.

        Args:
            vector_results: Results from vector search (ranked)
            graph_results: Results from graph traversal (ranked)

        Returns:
            List of FusedResult sorted by RRF score descending
        """
        if not self.config.enabled:
            # RRF disabled, return vector results as FusedResult
            return [
                FusedResult(
                    memory_id=r.memory_id,
                    rrf_score=r.score,
                    vector_rank=r.rank,
                    graph_rank=None,
                    in_both=False,
                    original_score=r.score,
                    payload=r.payload,
                    source_scores={"vector": r.score},
                )
                for r in vector_results
            ]

        vector_map = {r.memory_id: r for r in vector_results}
        graph_map = {r.memory_id: r for r in graph_results}
        all_ids = set(vector_map.keys()) | set(graph_map.keys())

        k = self.config.k
        alpha = self.config.alpha
        penalty = self.config.missing_rank_penalty

        fused: List[FusedResult] = []
        for memory_id in all_ids:
            v = vector_map.get(memory_id)
            g = graph_map.get(memory_id)

            v_rank = v.rank if v else penalty
            g_rank = g.rank if g else penalty

            # Weighted RRF formula
            rrf_score = alpha / (k + v_rank) + (1 - alpha) / (k + g_rank)

            # Use vector result as base (preferred), fallback to graph
            base = v or g

            source_scores = {}
            if v:
                source_scores["vector"] = v.score
            if g:
                source_scores["graph"] = g.score

            fused.append(FusedResult(
                memory_id=memory_id,
                rrf_score=round(rrf_score, 6),
                vector_rank=v.rank if v else None,
                graph_rank=g.rank if g else None,
                in_both=v is not None and g is not None,
                original_score=base.score,
                payload=base.payload,
                source_scores=source_scores,
            ))

        # Sort by RRF score descending
        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused

    def fuse_with_stats(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
    ) -> Dict[str, Any]:
        """
        Fuse results and return with statistics.

        Useful for debugging and verbose mode output.

        Returns:
            Dict with:
            - results: List of FusedResult
            - stats: Fusion statistics
        """
        fused = self.fuse(vector_results, graph_results)

        in_both_count = sum(1 for r in fused if r.in_both)
        vector_only = sum(1 for r in fused if r.vector_rank and not r.graph_rank)
        graph_only = sum(1 for r in fused if r.graph_rank and not r.vector_rank)

        return {
            "results": fused,
            "stats": {
                "vector_candidates": len(vector_results),
                "graph_candidates": len(graph_results),
                "fused_total": len(fused),
                "in_both_sources": in_both_count,
                "vector_only": vector_only,
                "graph_only": graph_only,
                "config": {
                    "k": self.config.k,
                    "alpha": self.config.alpha,
                    "missing_rank_penalty": self.config.missing_rank_penalty,
                },
            },
        }


def get_rrf_config() -> RRFConfig:
    """
    Load RRF configuration from environment variables.

    Environment Variables:
        OM_RRF_ENABLED: "true" or "false" (default: true)
        OM_RRF_K: Smoothing constant (default: 60)
        OM_RRF_ALPHA: Vector weight 0.0-1.0 (default: 0.6)
        OM_RRF_MISSING_RANK_PENALTY: Penalty for single-source (default: 100)

    Returns:
        RRFConfig with values from environment
    """
    return RRFConfig(
        k=int(os.getenv("OM_RRF_K", "60")),
        alpha=float(os.getenv("OM_RRF_ALPHA", "0.6")),
        missing_rank_penalty=int(os.getenv("OM_RRF_MISSING_RANK_PENALTY", "100")),
        enabled=os.getenv("OM_RRF_ENABLED", "true").lower() == "true",
    )


def merge_payloads(
    vector_payload: Dict[str, Any],
    graph_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge payloads from vector and graph sources.

    Vector payload takes precedence for shared keys.
    Graph-specific keys (seedConnections, avgSimilarity) are preserved.

    Args:
        vector_payload: Payload from vector search
        graph_payload: Payload from graph traversal

    Returns:
        Merged payload dict
    """
    merged = {**graph_payload, **vector_payload}

    # Preserve graph-specific metadata
    graph_keys = ["seedConnections", "avgSimilarity", "maxSimilarity"]
    for key in graph_keys:
        if key in graph_payload:
            merged[f"graph_{key}"] = graph_payload[key]

    return merged
