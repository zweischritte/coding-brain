"""
Business Concept Vector Store for OpenMemory.

Provides vector embedding storage and similarity search for business concepts
using a separate Qdrant collection. This enables:

1. Semantic similarity search across concepts
2. Deduplication via embedding similarity
3. Enhanced contradiction detection through semantic clustering
4. Concept clustering and relationship discovery

The concepts are stored in a separate collection from memories to maintain
clean separation between the structured memory system and Business Concepts.

Collection: business_concepts

Based on implementation plan:
- /docs/implementation/business-concept-implementation-plan.md
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConceptEmbeddingConfig:
    """Configuration for concept vector store."""

    # Qdrant connection
    qdrant_host: str = "mem0_store"
    qdrant_port: int = 6333
    collection_name: str = "business_concepts"

    # Embedding model
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    # Search defaults
    default_top_k: int = 10
    similarity_threshold: float = 0.75  # For deduplication

    @classmethod
    def from_env(cls) -> "ConceptEmbeddingConfig":
        """Load configuration from environment variables."""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", os.getenv("BUSINESS_CONCEPTS_QDRANT_HOST", "mem0_store")),
            qdrant_port=int(os.getenv("QDRANT_PORT", os.getenv("BUSINESS_CONCEPTS_QDRANT_PORT", "6333"))),
            collection_name=os.getenv("BUSINESS_CONCEPTS_COLLECTION", "business_concepts"),
            embedding_provider=os.getenv("BUSINESS_CONCEPTS_EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv("BUSINESS_CONCEPTS_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dims=int(os.getenv("BUSINESS_CONCEPTS_EMBEDDING_DIMS", "1536")),
            default_top_k=int(os.getenv("BUSINESS_CONCEPTS_DEFAULT_TOP_K", "10")),
            similarity_threshold=float(os.getenv("BUSINESS_CONCEPTS_SIMILARITY_THRESHOLD", "0.75")),
        )


def is_concept_embeddings_enabled() -> bool:
    """Check if concept embeddings feature is enabled."""
    from app.config import BusinessConceptsConfig

    # Must have base concepts enabled
    if not BusinessConceptsConfig.is_enabled():
        return False

    # Check specific embedding flag (default: enabled when concepts are enabled)
    return os.getenv("BUSINESS_CONCEPTS_EMBEDDING_ENABLED", "true").lower() == "true"


# =============================================================================
# Embedding Generation
# =============================================================================

class ConceptEmbedder:
    """
    Generates embeddings for business concepts.

    Uses OpenAI text-embedding-3-small by default for consistency
    with the memory system.
    """

    def __init__(self, config: Optional[ConceptEmbeddingConfig] = None):
        """
        Initialize the embedder.

        Args:
            config: Optional configuration (uses env vars if not provided)
        """
        self.config = config or ConceptEmbeddingConfig.from_env()
        self._client = None

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                from app.config import BusinessConceptsConfig

                api_key = BusinessConceptsConfig.get_openai_api_key()
                if not api_key:
                    raise ValueError("OpenAI API key not configured")

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required for concept embeddings")

        return self._client

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        client = self._get_openai_client()

        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=text.strip(),
        )

        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        client = self._get_openai_client()

        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=valid_texts,
        )

        # Return embeddings in same order
        return [item.embedding for item in response.data]

    def embed_concept(self, concept: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a business concept.

        Creates a rich text representation combining:
        - Concept name
        - Type and source type
        - Summary (if available)

        Args:
            concept: Dict with name, type, summary, etc.

        Returns:
            Embedding vector
        """
        # Build rich text representation
        parts = []

        name = concept.get("name", "")
        if name:
            parts.append(name)

        concept_type = concept.get("type", "")
        if concept_type:
            parts.append(f"Type: {concept_type}")

        source_type = concept.get("source_type") or concept.get("sourceType", "")
        if source_type:
            parts.append(f"Source: {source_type}")

        summary = concept.get("summary", "")
        if summary:
            parts.append(summary)

        text = " | ".join(parts) if parts else name

        return self.embed_text(text)


# =============================================================================
# Vector Store
# =============================================================================

class ConceptVectorStore:
    """
    Qdrant vector store for business concepts.

    Provides:
    - Storage of concept embeddings
    - Similarity search
    - Deduplication via embedding similarity
    - Batch operations
    """

    def __init__(self, config: Optional[ConceptEmbeddingConfig] = None):
        """
        Initialize the vector store.

        Args:
            config: Optional configuration (uses env vars if not provided)
        """
        self.config = config or ConceptEmbeddingConfig.from_env()
        self._client = None
        self._collection_initialized = False

    def _get_qdrant_client(self):
        """Lazy-load Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                )
            except ImportError:
                raise ImportError("qdrant-client package required for concept vector store")

        return self._client

    def ensure_collection(self) -> bool:
        """
        Ensure the concepts collection exists.

        Returns:
            True if collection exists or was created
        """
        if self._collection_initialized:
            return True

        try:
            from qdrant_client.models import Distance, VectorParams

            client = self._get_qdrant_client()

            # Check if collection exists
            collections = client.get_collections().collections
            exists = any(c.name == self.config.collection_name for c in collections)

            if not exists:
                # Create collection
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dims,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.config.collection_name}")

            self._collection_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            return False

    def upsert_concept(
        self,
        concept_id: str,
        user_id: str,
        name: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        access_entity: Optional[str] = None,
    ) -> bool:
        """
        Store or update a concept embedding.

        Args:
            concept_id: Unique concept ID (from Neo4j)
            user_id: User ID for legacy fallback
            name: Concept name
            embedding: Embedding vector
            metadata: Optional additional metadata
            access_entity: Access entity scope (defaults to user)

        Returns:
            True if successful
        """
        if not self.ensure_collection():
            return False

        try:
            from qdrant_client.models import PointStruct

            client = self._get_qdrant_client()

            # Build payload
            access_entity = access_entity or f"user:{user_id}"
            payload = {
                "concept_id": concept_id,
                "user_id": user_id,
                "access_entity": access_entity,
                "name": name,
                **(metadata or {}),
            }

            # Use concept_id hash as point ID (Qdrant needs int or uuid)
            import hashlib
            point_id = int(hashlib.md5(concept_id.encode()).hexdigest()[:16], 16)

            client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.debug(f"Upserted concept embedding: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert concept embedding: {e}")
            return False

    def search_similar(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        exclude_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar concepts by embedding.

        Args:
            query_embedding: Query vector
            user_id: User ID for legacy fallback
            top_k: Number of results (default from config)
            min_score: Minimum similarity score (0-1)
            exclude_ids: Concept IDs to exclude
            access_entities: Explicit access_entity matches
            access_entity_prefixes: Access_entity prefixes (without % wildcards)

        Returns:
            List of dicts with concept_id, name, score, and metadata
        """
        if not self.ensure_collection():
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

            client = self._get_qdrant_client()
            top_k = top_k or self.config.default_top_k

            access_entities = access_entities or [f"user:{user_id}"]
            access_entity_prefixes = access_entity_prefixes or []
            include_prefixes = bool(access_entity_prefixes)

            # Build filter for exact access_entity matches when possible.
            filters = []
            if not include_prefixes:
                filters.append(
                    FieldCondition(
                        key="access_entity",
                        match=MatchAny(any=access_entities),
                    )
                )
            else:
                # Prefix matches require post-filtering; avoid pre-filtering to keep scoped results.
                filters = []

            # Search with filter using query_points (qdrant-client >= 1.7)
            fetch_limit = max(top_k or self.config.default_top_k, 1)
            fetch_limit = min(200, fetch_limit * (5 if include_prefixes else 2))
            results = client.query_points(
                collection_name=self.config.collection_name,
                query=query_embedding,
                query_filter=Filter(must=filters) if filters else None,
                limit=fetch_limit + len(exclude_ids or []),  # Request more to account for exclusions
                with_payload=True,
            )

            # Process results - query_points returns QueryResponse with .points
            concepts = []
            for hit in results.points:
                concept_id = hit.payload.get("concept_id")
                access_entity = hit.payload.get("access_entity")
                legacy_user_id = hit.payload.get("user_id")

                # Enforce access_entity visibility (including legacy fallback)
                if access_entity:
                    if access_entity not in access_entities and not any(
                        access_entity.startswith(prefix) for prefix in access_entity_prefixes
                    ):
                        continue
                else:
                    if legacy_user_id != user_id:
                        continue

                # Skip excluded IDs
                if exclude_ids and concept_id in exclude_ids:
                    continue

                # Skip below threshold
                if min_score and hit.score < min_score:
                    continue

                concepts.append({
                    "concept_id": concept_id,
                    "name": hit.payload.get("name"),
                    "access_entity": access_entity,
                    "score": hit.score,
                    **{
                        k: v
                        for k, v in hit.payload.items()
                        if k not in ("concept_id", "name", "user_id", "access_entity")
                    },
                })

                if len(concepts) >= (top_k or self.config.default_top_k):
                    break

            return concepts

        except Exception as e:
            logger.error(f"Failed to search similar concepts: {e}")
            return []

    def find_duplicates(
        self,
        concept_id: str,
        embedding: List[float],
        user_id: str,
        threshold: Optional[float] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate concepts by similarity.

        Args:
            concept_id: ID of the concept to check
            embedding: Embedding of the concept
            user_id: User ID for legacy fallback
            threshold: Similarity threshold (default from config)
            access_entities: Explicit access_entity matches
            access_entity_prefixes: Access_entity prefixes (without % wildcards)

        Returns:
            List of potential duplicates with scores
        """
        threshold = threshold or self.config.similarity_threshold

        return self.search_similar(
            query_embedding=embedding,
            user_id=user_id,
            top_k=5,
            min_score=threshold,
            exclude_ids=[concept_id],
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept embedding.

        Args:
            concept_id: Concept ID to delete

        Returns:
            True if successful
        """
        if not self.ensure_collection():
            return False

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            client = self._get_qdrant_client()

            # Delete by filter on concept_id
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="concept_id",
                            match=MatchValue(value=concept_id),
                        )
                    ]
                ),
            )

            logger.debug(f"Deleted concept embedding: {concept_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete concept embedding: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the concepts collection.

        Returns:
            Dict with collection info
        """
        if not self.ensure_collection():
            return {"error": "Collection not available"}

        try:
            client = self._get_qdrant_client()

            info = client.get_collection(self.config.collection_name)

            return {
                "collection_name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}


# =============================================================================
# High-Level Operations
# =============================================================================

class ConceptSimilarityService:
    """
    High-level service for concept similarity operations.

    Combines embedding generation and vector storage for:
    - Semantic search across concepts
    - Deduplication
    - Relationship discovery
    """

    def __init__(self, config: Optional[ConceptEmbeddingConfig] = None):
        """
        Initialize the service.

        Args:
            config: Optional configuration
        """
        self.config = config or ConceptEmbeddingConfig.from_env()
        self.embedder = ConceptEmbedder(self.config)
        self.store = ConceptVectorStore(self.config)

    def embed_and_store(
        self,
        concept_id: str,
        user_id: str,
        name: str,
        concept_type: Optional[str] = None,
        summary: Optional[str] = None,
        category: Optional[str] = None,
        source_type: Optional[str] = None,
        access_entity: Optional[str] = None,
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        Generate embedding for a concept and store it.

        Also checks for potential duplicates.

        Args:
            concept_id: Unique concept ID
            user_id: User ID for legacy fallback
            name: Concept name
            concept_type: Optional type
            summary: Optional summary
            category: Optional category
            source_type: Optional source type
            access_entity: Access entity scope (defaults to user)

        Returns:
            Tuple of (success, potential_duplicates)
        """
        try:
            # Generate embedding
            concept_data = {
                "name": name,
                "type": concept_type,
                "summary": summary,
                "sourceType": source_type,
            }

            embedding = self.embedder.embed_concept(concept_data)

            # Check for duplicates first
            duplicates = self.store.find_duplicates(
                concept_id=concept_id,
                embedding=embedding,
                user_id=user_id,
                access_entities=[access_entity] if access_entity else None,
            )

            # Store the embedding
            metadata = {
                "type": concept_type,
                "category": category,
                "source_type": source_type,
            }

            success = self.store.upsert_concept(
                concept_id=concept_id,
                user_id=user_id,
                name=name,
                embedding=embedding,
                metadata=metadata,
                access_entity=access_entity,
            )

            return success, duplicates if duplicates else None

        except Exception as e:
            logger.error(f"Failed to embed and store concept: {e}")
            return False, None

    def search_concepts(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        min_score: float = 0.5,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for concepts by text query.

        Args:
            query: Search query text
            user_id: User ID for legacy fallback
            top_k: Number of results
            min_score: Minimum similarity score
            access_entities: Explicit access_entity matches
            access_entity_prefixes: Access_entity prefixes (without % wildcards)

        Returns:
            List of matching concepts with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)

            # Search vector store
            return self.store.search_similar(
                query_embedding=query_embedding,
                user_id=user_id,
                top_k=top_k,
                min_score=min_score,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

        except Exception as e:
            logger.error(f"Failed to search concepts: {e}")
            return []

    def find_similar_concepts(
        self,
        concept_name: str,
        user_id: str,
        top_k: int = 5,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find concepts similar to a given concept by name.

        Args:
            concept_name: Name of the concept to find similar to
            user_id: User ID for legacy fallback
            top_k: Number of results
            access_entities: Explicit access_entity matches
            access_entity_prefixes: Access_entity prefixes (without % wildcards)

        Returns:
            List of similar concepts with scores
        """
        try:
            # Generate embedding from name
            embedding = self.embedder.embed_text(concept_name)

            # Search for similar (excluding exact matches by name)
            results = self.store.search_similar(
                query_embedding=embedding,
                user_id=user_id,
                top_k=top_k + 1,  # +1 to account for self-match
                min_score=0.5,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

            # Filter out the exact match
            return [r for r in results if r.get("name") != concept_name][:top_k]

        except Exception as e:
            logger.error(f"Failed to find similar concepts: {e}")
            return []

    def delete_concept(self, concept_id: str) -> bool:
        """
        Delete a concept's embedding.

        Args:
            concept_id: Concept ID to delete

        Returns:
            True if successful
        """
        return self.store.delete_concept(concept_id)


# =============================================================================
# Factory Functions
# =============================================================================

_similarity_service: Optional[ConceptSimilarityService] = None


def get_concept_similarity_service() -> Optional[ConceptSimilarityService]:
    """
    Get the singleton ConceptSimilarityService instance.

    Returns:
        ConceptSimilarityService if enabled, None otherwise
    """
    global _similarity_service

    if not is_concept_embeddings_enabled():
        return None

    if _similarity_service is None:
        try:
            _similarity_service = ConceptSimilarityService()
            logger.info("Initialized ConceptSimilarityService")
        except Exception as e:
            logger.warning(f"Failed to initialize ConceptSimilarityService: {e}")
            return None

    return _similarity_service


def reset_similarity_service():
    """Reset the similarity service (for testing or config changes)."""
    global _similarity_service
    _similarity_service = None
