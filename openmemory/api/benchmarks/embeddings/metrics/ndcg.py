"""
Normalized Discounted Cumulative Gain (NDCG) calculation.

NDCG is a standard IR metric that measures ranking quality with graded
relevance scores, accounting for the position of relevant documents.

Formula: NDCG@k = DCG@k / IDCG@k

Where:
- DCG@k = sum(rel_i / log2(i + 1)) for i in 1..k
- IDCG@k = DCG of the ideal ranking (sorted by relevance)
"""

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NDCGResult:
    """Result of NDCG calculation."""
    score: float
    num_queries: int


def _dcg(relevances: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain.

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff position.

    Returns:
        DCG score.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        # Using log base 2: discount = log2(i + 1)
        dcg += rel / math.log2(i + 1)
    return dcg


def calculate_ndcg(
    ranked_results: List[List[str]],
    relevance_scores: List[Dict[str, int]],
    k: int = 10
) -> NDCGResult:
    """
    Calculate Normalized Discounted Cumulative Gain.

    Args:
        ranked_results: List of ranked document IDs per query.
                       Each inner list is ordered by rank (position 0 = rank 1).
        relevance_scores: Dict mapping doc_id to relevance grade per query.
                         Documents not in dict are treated as grade 0.
        k: Cutoff for evaluation (NDCG@k). Only considers top k results.
           Default is 10.

    Returns:
        NDCGResult with:
        - score: The average NDCG score across queries (0.0 to 1.0)
        - num_queries: Total number of queries evaluated

    Raises:
        ValueError: If ranked_results and relevance_scores have different lengths.
    """
    if len(ranked_results) != len(relevance_scores):
        raise ValueError(
            f"Mismatched lengths: {len(ranked_results)} queries but "
            f"{len(relevance_scores)} relevance dicts"
        )

    if not ranked_results:
        return NDCGResult(score=0.0, num_queries=0)

    ndcg_scores = []

    for results, rel_dict in zip(ranked_results, relevance_scores):
        # Get relevance scores for the ranked results
        relevances = [rel_dict.get(doc_id, 0) for doc_id in results[:k]]

        # Calculate DCG for the actual ranking
        dcg = _dcg(relevances, k)

        # Calculate IDCG (ideal DCG) - sorted by descending relevance
        all_relevances = list(rel_dict.values())
        ideal_relevances = sorted(all_relevances, reverse=True)[:k]
        idcg = _dcg(ideal_relevances, k)

        # Calculate NDCG
        if idcg == 0:
            # No relevant documents, NDCG is 0
            ndcg = 0.0
        else:
            ndcg = dcg / idcg

        ndcg_scores.append(ndcg)

    # Calculate mean NDCG across all queries
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return NDCGResult(
        score=avg_ndcg,
        num_queries=len(ranked_results)
    )
