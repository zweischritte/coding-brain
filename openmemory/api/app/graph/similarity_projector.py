"""
Similarity Projector for OpenMemory.

Creates and maintains OM_SIMILAR edges between semantically similar memories
using Qdrant embeddings. This enables O(1) graph traversal for finding
related memories without computing embeddings at query time.

Configuration via environment variables:
- OM_SIMILARITY_K: Number of nearest neighbors per memory (default: 20)
- OM_SIMILARITY_THRESHOLD: Minimum cosine similarity score (default: 0.6)
- OM_SIMILARITY_MAX_EDGES: Maximum edges per memory (default: 30)
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SimilarityConfig:
    """Configuration for semantic similarity edge projection."""

    # Number of nearest neighbors to find per memory
    k_neighbors: int = 20

    # Minimum cosine similarity score to create an edge (0.0-1.0)
    min_similarity_threshold: float = 0.6

    # Maximum edges per memory (to limit graph density)
    max_edges_per_memory: int = 30

    # Whether to create bidirectional edges (A→B and B→A)
    # Bidirectional ensures traversal works from either direction
    bidirectional: bool = True

    @classmethod
    def from_env(cls) -> "SimilarityConfig":
        """Load configuration from environment variables."""
        return cls(
            k_neighbors=int(os.environ.get("OM_SIMILARITY_K", 20)),
            min_similarity_threshold=float(os.environ.get("OM_SIMILARITY_THRESHOLD", 0.6)),
            max_edges_per_memory=int(os.environ.get("OM_SIMILARITY_MAX_EDGES", 30)),
            bidirectional=os.environ.get("OM_SIMILARITY_BIDIRECTIONAL", "true").lower() in ("true", "1", "yes"),
        )


class SimilarityCypherBuilder:
    """Builds Cypher queries for similarity edge projection."""

    @staticmethod
    def upsert_similarity_edges_query() -> str:
        """
        Upsert multiple similarity edges in a batch.

        Expects $edges parameter as list of {source_id, target_id, score, rank}.
        """
        return """
        UNWIND $edges AS edge
        MATCH (source:OM_Memory {id: edge.source_id})
        MATCH (target:OM_Memory {id: edge.target_id})
        WHERE coalesce(source.accessEntity, $legacyAccessEntity) = $accessEntity
          AND coalesce(target.accessEntity, $legacyAccessEntity) = $accessEntity
        MERGE (source)-[r:OM_SIMILAR {accessEntity: $accessEntity}]->(target)
        ON CREATE SET r.createdAt = datetime()
        SET r.score = edge.score,
            r.rank = edge.rank,
            r.updatedAt = datetime(),
            r.userId = coalesce(r.userId, $userId)
        RETURN count(r) AS created
        """

    @staticmethod
    def upsert_bidirectional_similarity_edges_query() -> str:
        """
        Upsert bidirectional similarity edges in a batch.

        Creates edges in both directions (A→B and B→A) for complete traversability.
        Expects $edges parameter as list of {source_id, target_id, score, rank}.
        """
        return """
        UNWIND $edges AS edge
        MATCH (source:OM_Memory {id: edge.source_id})
        MATCH (target:OM_Memory {id: edge.target_id})
        WHERE coalesce(source.accessEntity, $legacyAccessEntity) = $accessEntity
          AND coalesce(target.accessEntity, $legacyAccessEntity) = $accessEntity

        // Forward edge (source → target)
        MERGE (source)-[r1:OM_SIMILAR {accessEntity: $accessEntity}]->(target)
        ON CREATE SET r1.createdAt = datetime()
        SET r1.score = edge.score,
            r1.rank = edge.rank,
            r1.updatedAt = datetime(),
            r1.userId = coalesce(r1.userId, $userId)

        // Reverse edge (target → source) with same score
        WITH source, target, edge
        MERGE (target)-[r2:OM_SIMILAR {accessEntity: $accessEntity}]->(source)
        ON CREATE SET r2.createdAt = datetime()
        SET r2.score = edge.score,
            r2.reverseRank = edge.rank,
            r2.updatedAt = datetime(),
            r2.userId = coalesce(r2.userId, $userId)

        RETURN count(*) AS created
        """

    @staticmethod
    def delete_similarity_edges_for_memory_query() -> str:
        """Delete all outgoing similarity edges for a memory."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        WITH m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        MATCH (m)-[r:OM_SIMILAR]->()
        WHERE (r.accessEntity IS NOT NULL AND r.accessEntity = accessEntity)
           OR (r.accessEntity IS NULL AND r.userId = $userId)
        DELETE r
        RETURN count(r) AS deleted
        """

    @staticmethod
    def delete_bidirectional_similarity_edges_query() -> str:
        """Delete all similarity edges (both directions) for a memory."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        WITH m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        MATCH (m)-[r:OM_SIMILAR]-()
        WHERE (r.accessEntity IS NOT NULL AND r.accessEntity = accessEntity)
           OR (r.accessEntity IS NULL AND r.userId = $userId)
        DELETE r
        RETURN count(r) AS deleted
        """

    @staticmethod
    def get_similar_memories_query() -> str:
        """Get pre-computed similar memories via OM_SIMILAR edges."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})-[r:OM_SIMILAR]->(similar:OM_Memory)
        WITH m, r, similar, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        WHERE r.score >= $minScore
          AND ($allowedMemoryIds IS NULL OR similar.id IN $allowedMemoryIds)
          AND (
            (similar.accessEntity IS NOT NULL AND similar.accessEntity = accessEntity)
            OR (similar.accessEntity IS NULL AND similar.userId = $userId)
          )
          AND (
            (r.accessEntity IS NOT NULL AND r.accessEntity = accessEntity)
            OR (r.accessEntity IS NULL AND r.userId = $userId)
          )
        RETURN similar.id AS id,
               similar.content AS content,
               similar.category AS category,
               similar.scope AS scope,
               similar.artifactType AS artifactType,
               similar.artifactRef AS artifactRef,
               similar.source AS source,
               similar.createdAt AS createdAt,
               r.score AS similarityScore,
               r.rank AS rank
        ORDER BY r.score DESC
        LIMIT $limit
        """

    @staticmethod
    def count_similarity_edges_query() -> str:
        """Count total similarity edges for a user."""
        return """
        MATCH ()-[r:OM_SIMILAR]->()
        WHERE (r.accessEntity IS NOT NULL AND r.accessEntity = $accessEntity)
           OR (r.accessEntity IS NULL AND r.userId = $userId)
        RETURN count(r) AS edgeCount
        """


class SimilarityProjector:
    """
    Projects semantic similarity edges between OM_Memory nodes.

    Uses Qdrant embeddings to find K nearest neighbors for each memory
    and creates OM_SIMILAR edges in Neo4j for fast graph traversal.
    """

    def __init__(
        self,
        session_factory,
        memory_client,
        config: SimilarityConfig = None,
    ):
        """
        Initialize the similarity projector.

        Args:
            session_factory: Callable that returns a Neo4j session context manager
            memory_client: Mem0 Memory client with vector_store access
            config: Optional similarity configuration
        """
        self.session_factory = session_factory
        self.memory_client = memory_client
        self.config = config or SimilarityConfig.from_env()

    def find_similar_in_qdrant(
        self,
        memory_id: str,
        user_id: str,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Find K nearest neighbors for a memory using Qdrant.

        Args:
            memory_id: UUID of the seed memory
            user_id: String user ID for filtering

        Returns:
            Tuple of (similar list, access_entity) where similar list contains
            {id, score, rank} dicts.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            vector_store = self.memory_client.vector_store

            # Get the memory's embedding from Qdrant
            points = vector_store.client.retrieve(
                collection_name=vector_store.collection_name,
                ids=[memory_id],
                with_vectors=True,
                with_payload=True,
            )

            if not points or not points[0].vector:
                logger.debug(f"No vector found for memory {memory_id}")
                return [], f"user:{user_id}"

            embedding = points[0].vector
            payload = points[0].payload or {}
            access_entity = payload.get("access_entity") or f"user:{user_id}"
            org_id = payload.get("org_id")
            if not org_id and hasattr(vector_store, "org_id"):
                org_id = getattr(vector_store, "org_id")

            # Query for K+1 neighbors (will include self)
            must_conditions = [FieldCondition(key="access_entity", match=MatchValue(value=access_entity))]
            if org_id:
                must_conditions.append(FieldCondition(key="org_id", match=MatchValue(value=org_id)))

            hits = vector_store.client.query_points(
                collection_name=vector_store.collection_name,
                query=embedding,
                query_filter=Filter(
                    must=must_conditions
                ),
                limit=self.config.k_neighbors + 1,
                with_payload=False,
            )

            # Filter out self and apply threshold
            results = []
            rank = 0
            for hit in hits.points:
                hit_id = str(hit.id)
                if hit_id == memory_id:
                    continue
                if hit.score >= self.config.min_similarity_threshold:
                    results.append({
                        "id": hit_id,
                        "score": hit.score,
                        "rank": rank,
                    })
                    rank += 1
                if len(results) >= self.config.max_edges_per_memory:
                    break

            return results, access_entity

        except Exception as e:
            logger.warning(f"Error finding similar memories in Qdrant for {memory_id}: {e}")
            return [], f"user:{user_id}"

    def project_similarity_edges(
        self,
        memory_id: str,
        user_id: str,
    ) -> int:
        """
        Project similarity edges for a single memory.

        Finds K nearest neighbors in Qdrant and creates OM_SIMILAR edges in Neo4j.
        If bidirectional mode is enabled, creates edges in both directions.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID

        Returns:
            Number of edges created
        """
        similar, access_entity = self.find_similar_in_qdrant(memory_id, user_id)

        if not similar:
            return 0

        edges = [
            {
                "source_id": memory_id,
                "target_id": s["id"],
                "score": s["score"],
                "rank": s["rank"],
            }
            for s in similar
        ]

        try:
            with self.session_factory() as session:
                # Delete existing edges (for clean update)
                if self.config.bidirectional:
                    session.run(
                        SimilarityCypherBuilder.delete_bidirectional_similarity_edges_query(),
                        {
                            "memoryId": memory_id,
                            "userId": user_id,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )
                else:
                    session.run(
                        SimilarityCypherBuilder.delete_similarity_edges_for_memory_query(),
                        {
                            "memoryId": memory_id,
                            "userId": user_id,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )

                # Create new edges (bidirectional or unidirectional)
                if self.config.bidirectional:
                    result = session.run(
                        SimilarityCypherBuilder.upsert_bidirectional_similarity_edges_query(),
                        {
                            "edges": edges,
                            "userId": user_id,
                            "accessEntity": access_entity,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )
                else:
                    result = session.run(
                        SimilarityCypherBuilder.upsert_similarity_edges_query(),
                        {
                            "edges": edges,
                            "userId": user_id,
                            "accessEntity": access_entity,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )

                record = result.single()
                created = record["created"] if record else 0

            logger.debug(f"Created {created} similarity edges for memory {memory_id}")
            return created

        except Exception as e:
            logger.warning(f"Error projecting similarity edges for {memory_id}: {e}")
            return 0

    def delete_similarity_edges(self, memory_id: str, user_id: str) -> bool:
        """
        Delete all similarity edges for a memory.

        Called when a memory is deleted. Deletes bidirectional edges if configured.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID

        Returns:
            True if successful
        """
        try:
            with self.session_factory() as session:
                if self.config.bidirectional:
                    session.run(
                        SimilarityCypherBuilder.delete_bidirectional_similarity_edges_query(),
                        {
                            "memoryId": memory_id,
                            "userId": user_id,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )
                else:
                    session.run(
                        SimilarityCypherBuilder.delete_similarity_edges_for_memory_query(),
                        {
                            "memoryId": memory_id,
                            "userId": user_id,
                            "legacyAccessEntity": f"user:{user_id}",
                        }
                    )
            return True
        except Exception as e:
            logger.warning(f"Error deleting similarity edges for {memory_id}: {e}")
            return False

    def get_similar_memories(
        self,
        memory_id: str,
        user_id: str,
        allowed_memory_ids: Optional[List[str]] = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get pre-computed similar memories from Neo4j.

        Args:
            memory_id: UUID of the seed memory
            user_id: String user ID
            allowed_memory_ids: Optional allowlist for ACL
            min_score: Minimum similarity score
            limit: Maximum memories to return

        Returns:
            List of similar memory dicts with similarityScore and rank
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    SimilarityCypherBuilder.get_similar_memories_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "minScore": float(min_score or 0.0),
                        "limit": max(1, min(int(limit or 10), 100)),
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )

                memories = []
                for record in result:
                    memories.append({
                        "id": record["id"],
                        "content": record["content"],
                        "category": record["category"],
                        "scope": record["scope"],
                        "artifactType": record["artifactType"],
                        "artifactRef": record["artifactRef"],
                        "source": record["source"],
                        "createdAt": record["createdAt"],
                        "similarityScore": record["similarityScore"],
                        "rank": record["rank"],
                    })
                return memories

        except Exception as e:
            logger.warning(f"Error getting similar memories for {memory_id}: {e}")
            return []

    def count_similarity_edges(
        self,
        user_id: str,
        access_entity: Optional[str] = None,
    ) -> int:
        """Count total similarity edges for an access scope."""
        try:
            with self.session_factory() as session:
                result = session.run(
                    SimilarityCypherBuilder.count_similarity_edges_query(),
                    {
                        "userId": user_id,
                        "accessEntity": access_entity or f"user:{user_id}",
                    }
                )
                for record in result:
                    return record["edgeCount"]
                return 0
        except Exception as e:
            logger.warning(f"Error counting similarity edges for {user_id}: {e}")
            return 0


# =============================================================================
# Module-level singleton management
# =============================================================================

_similarity_projector = None
_similarity_projector_initialized = False


def get_similarity_projector() -> Optional[SimilarityProjector]:
    """
    Get the singleton SimilarityProjector instance.

    Returns:
        SimilarityProjector if Neo4j and Qdrant are configured, None otherwise
    """
    global _similarity_projector, _similarity_projector_initialized

    if _similarity_projector_initialized:
        return _similarity_projector

    _similarity_projector_initialized = True

    try:
        from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured
        from app.utils.memory import get_memory_client

        if not is_neo4j_configured():
            logger.debug("Neo4j not configured, similarity projector disabled")
            return None

        memory_client = get_memory_client()
        if memory_client is None:
            logger.debug("Memory client not available, similarity projector disabled")
            return None

        if not hasattr(memory_client, 'vector_store') or memory_client.vector_store is None:
            logger.debug("Vector store not available, similarity projector disabled")
            return None

        config = SimilarityConfig.from_env()
        _similarity_projector = SimilarityProjector(
            session_factory=get_neo4j_session,
            memory_client=memory_client,
            config=config,
        )

        logger.info(
            f"Similarity projector initialized: k={config.k_neighbors}, "
            f"threshold={config.min_similarity_threshold}, "
            f"max_edges={config.max_edges_per_memory}"
        )

    except Exception as e:
        logger.warning(f"Failed to initialize similarity projector: {e}")
        _similarity_projector = None

    return _similarity_projector


def reset_similarity_projector():
    """Reset the similarity projector instance (for testing or config changes)."""
    global _similarity_projector, _similarity_projector_initialized
    _similarity_projector = None
    _similarity_projector_initialized = False
