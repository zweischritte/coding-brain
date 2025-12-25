"""
Embedding-based similarity module for entity normalization.

Uses semantic embeddings to find similar entity names that may
not be caught by string-based methods.

This is an optional feature that requires an embedding model
and can be expensive with many entities.

Note: This module is designed to be used only for entities
that weren't matched by other methods (string similarity,
prefix matching, domain normalization).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMatch:
    """An embedding-based similarity match between two entities."""
    entity_a: str
    entity_b: str
    cosine_similarity: float


async def find_embedding_similar_entities(
    entities: List[str],
    embedding_model: Optional[Any] = None,
    threshold: float = 0.90,
    batch_size: int = 100,
) -> List[EmbeddingMatch]:
    """
    Find semantically similar entities using embeddings.

    WARNING: This can be expensive with many entities!
    Use only as a last resort for unmatched entities.

    Args:
        entities: List of entity names to compare
        embedding_model: Embedding client (must have embed_batch method)
        threshold: Minimum cosine similarity (0.90 = 90%)
        batch_size: Batch size for API calls

    Returns:
        List of EmbeddingMatch objects

    Cost estimation (OpenAI text-embedding-3-small):
        - 100 entities: ~1 API call, ~$0.01
        - 1000 entities: ~10 API calls, ~$0.10
        - 10000 entities: ~100 API calls, ~$1.00
    """
    if embedding_model is None:
        logger.warning("Embedding model not provided, skipping embedding similarity")
        return []

    if len(entities) < 2:
        return []

    try:
        # Import numpy for vector operations
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("numpy/sklearn not available for embedding similarity")
        return []

    matches = []

    try:
        # Generate embeddings in batches
        all_embeddings = []

        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]

            # Call embedding model (expects embed_batch method)
            if hasattr(embedding_model, 'embed_batch'):
                batch_embeddings = await embedding_model.embed_batch(batch)
            elif hasattr(embedding_model, 'embed'):
                # Fallback for single-item embed method
                batch_embeddings = []
                for text in batch:
                    emb = await embedding_model.embed(text)
                    batch_embeddings.append(emb)
            else:
                logger.error("Embedding model has no embed_batch or embed method")
                return []

            all_embeddings.extend(batch_embeddings)

        # Convert to numpy array
        embeddings = np.array(all_embeddings)

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Find pairs above threshold
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if sim_matrix[i, j] >= threshold:
                    matches.append(EmbeddingMatch(
                        entity_a=entities[i],
                        entity_b=entities[j],
                        cosine_similarity=float(sim_matrix[i, j])
                    ))

        logger.info(f"Embedding similarity found {len(matches)} matches from {len(entities)} entities")

    except Exception as e:
        logger.exception(f"Error in embedding similarity: {e}")

    return matches


def estimate_embedding_cost(
    entity_count: int,
    price_per_1k_tokens: float = 0.00002,  # text-embedding-3-small
    avg_tokens_per_entity: int = 3,
) -> dict:
    """
    Estimate the cost of running embedding similarity.

    Args:
        entity_count: Number of entities to embed
        price_per_1k_tokens: Price per 1000 tokens (default: OpenAI small)
        avg_tokens_per_entity: Average tokens per entity name

    Returns:
        Dict with cost estimates
    """
    total_tokens = entity_count * avg_tokens_per_entity
    estimated_cost = (total_tokens / 1000) * price_per_1k_tokens

    return {
        "entity_count": entity_count,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
        "api_calls": (entity_count + 99) // 100,  # Batch of 100
    }
