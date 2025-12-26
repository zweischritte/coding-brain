"""
Mean Reciprocal Rank (MRR) calculation.

MRR is a standard IR metric that measures the average of reciprocal ranks
of the first relevant document for each query.

Formula: MRR = (1/|Q|) * sum(1/rank_i) for i in Q

Where:
- Q is the set of queries
- rank_i is the position of the first relevant document for query i
"""

from dataclasses import dataclass
from typing import List, Set


@dataclass
class MRRResult:
    """Result of MRR calculation."""
    score: float
    num_queries: int
    queries_with_relevant: int


def calculate_mrr(
    ranked_results: List[List[str]],
    relevant_docs: List[Set[str]],
    k: int = 10
) -> MRRResult:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        ranked_results: List of ranked document IDs per query.
                       Each inner list is ordered by rank (position 0 = rank 1).
        relevant_docs: Set of relevant document IDs per query.
        k: Cutoff for evaluation (MRR@k). Only considers top k results.
           Default is 10.

    Returns:
        MRRResult with:
        - score: The MRR score (0.0 to 1.0)
        - num_queries: Total number of queries evaluated
        - queries_with_relevant: Number of queries where a relevant doc was found

    Raises:
        ValueError: If ranked_results and relevant_docs have different lengths.
    """
    if len(ranked_results) != len(relevant_docs):
        raise ValueError(
            f"Mismatched lengths: {len(ranked_results)} queries but "
            f"{len(relevant_docs)} relevance sets"
        )

    if not ranked_results:
        return MRRResult(score=0.0, num_queries=0, queries_with_relevant=0)

    reciprocal_ranks = []
    queries_with_relevant = 0

    for results, relevant in zip(ranked_results, relevant_docs):
        # Find the rank of the first relevant document
        rr = 0.0
        for rank, doc_id in enumerate(results[:k], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                queries_with_relevant += 1
                break
        reciprocal_ranks.append(rr)

    # Calculate mean
    mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return MRRResult(
        score=mrr_score,
        num_queries=len(ranked_results),
        queries_with_relevant=queries_with_relevant
    )
