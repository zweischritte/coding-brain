"""
Tenant-scoped Qdrant embedding store.

This module implements a Qdrant-backed vector store with org_id-based
tenant isolation. All operations automatically filter by org_id to
ensure data isolation between organizations.

Key features:
- Per-model collection naming (embeddings_{model_name})
- Automatic org_id injection into payloads
- Payload index creation for efficient org_id filtering
- Tenant-safe search, get, list, delete operations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


logger = logging.getLogger(__name__)


class QdrantClientProtocol(Protocol):
    """Protocol for Qdrant client operations."""

    def get_collections(self) -> Any: ...
    def create_collection(self, collection_name: str, vectors_config: Any) -> bool: ...
    def delete_collection(self, collection_name: str) -> bool: ...
    def upsert(self, collection_name: str, points: List[PointStruct]) -> bool: ...
    def retrieve(self, collection_name: str, ids: List[str], with_payload: bool = True) -> List[Any]: ...
    def delete(self, collection_name: str, points_selector: PointIdsList) -> bool: ...
    def query_points(self, collection_name: str, query: List[float], query_filter: Optional[Filter] = None, limit: int = 10) -> Any: ...
    def scroll(self, collection_name: str, scroll_filter: Optional[Filter] = None, limit: int = 100, with_payload: bool = True, with_vectors: bool = False) -> tuple: ...
    def create_payload_index(self, collection_name: str, field_name: str, field_schema: str) -> bool: ...


@dataclass
class TenantQdrantStoreConfig:
    """Configuration for TenantQdrantStore."""

    embedding_dim: int = 1536
    distance_metric: Distance = Distance.COSINE
    on_disk: bool = False
    indexed_fields: tuple = ("org_id", "user_id", "vault", "layer")


class TenantQdrantStore:
    """
    Tenant-scoped Qdrant embedding store.

    Provides vector storage with automatic org_id-based isolation.
    All points are stored with org_id in the payload, and all queries
    automatically filter by org_id.

    Collection naming: embeddings_{sanitized_model_name}

    Example:
        store = TenantQdrantStore(
            client=qdrant_client,
            org_id="11111111-1111-1111-1111-111111111111",
            model_name="text-embedding-3-small",
        )
        store.upsert("point-1", [0.1] * 1536, {"content": "Hello"})
        results = store.search([0.1] * 1536, limit=10)
    """

    def __init__(
        self,
        client: QdrantClientProtocol,
        org_id: str,
        model_name: str,
        embedding_dim: int = 1536,
        config: Optional[TenantQdrantStoreConfig] = None,
    ):
        """
        Initialize the tenant-scoped Qdrant store.

        Args:
            client: Qdrant client instance
            org_id: The organization ID for tenant isolation
            model_name: Name of the embedding model (used in collection name)
            embedding_dim: Dimension of the embedding vectors
            config: Optional configuration
        """
        self.client = client
        self.org_id = org_id
        self.model_name = model_name
        self.config = config or TenantQdrantStoreConfig(embedding_dim=embedding_dim)

        # Sanitize model name for collection naming
        self.collection_name = self._sanitize_collection_name(model_name)

        # Initialize collection
        self._ensure_collection()

    def _sanitize_collection_name(self, model_name: str) -> str:
        """
        Sanitize model name for use as a Qdrant collection name.

        Qdrant collection names must:
        - Not contain '/'
        - Be alphanumeric with hyphens/underscores

        Args:
            model_name: The raw model name

        Returns:
            Sanitized collection name
        """
        # Replace slashes and other invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        return f"embeddings_{sanitized}"

    def _ensure_collection(self) -> None:
        """Ensure the collection exists, creating it if necessary."""
        try:
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dim,
                        distance=self.config.distance_metric,
                        on_disk=self.config.on_disk,
                    ),
                )
                logger.info(f"Created collection {self.collection_name}")

            # Create payload indexes for efficient filtering
            self._create_payload_indexes()

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for commonly filtered fields."""
        for field in self.config.indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
                logger.debug(f"Created index for {field} in {self.collection_name}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index for {field} might already exist: {e}")

    def _create_org_filter(self, additional_filters: Optional[Dict[str, Any]] = None) -> Filter:
        """
        Create a filter that includes org_id constraint.

        Args:
            additional_filters: Optional additional filter conditions

        Returns:
            Filter with org_id and any additional conditions
        """
        conditions = [
            FieldCondition(key="org_id", match=MatchValue(value=self.org_id))
        ]

        if additional_filters:
            for key, value in additional_filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions)

    def upsert(
        self,
        point_id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upsert a point with automatic org_id injection.

        Args:
            point_id: Unique identifier for the point
            vector: The embedding vector
            payload: Optional payload data

        Returns:
            True if successful
        """
        # Inject org_id into payload
        full_payload = payload.copy() if payload else {}
        full_payload["org_id"] = self.org_id

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=full_payload,
        )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            logger.debug(f"Upserted point {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert point {point_id}: {e}")
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        Search for similar vectors, filtered by org_id.

        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            filters: Optional additional filters

        Returns:
            List of matching points with scores
        """
        query_filter = self._create_org_filter(filters)

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
            return results.points
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get(self, point_id: str) -> Optional[Any]:
        """
        Retrieve a point by ID, verifying org_id ownership.

        Args:
            point_id: The point ID to retrieve

        Returns:
            The point if found and owned, None otherwise
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if not results:
                return None

            point = results[0]

            # Verify org_id ownership
            if point.payload.get("org_id") != self.org_id:
                logger.warning(
                    f"Tenant isolation violation: {self.org_id} tried to access "
                    f"point owned by {point.payload.get('org_id')}"
                )
                return None

            return point
        except Exception as e:
            logger.error(f"Failed to get point {point_id}: {e}")
            return None

    def delete(self, point_id: str) -> bool:
        """
        Delete a point, verifying org_id ownership first.

        Args:
            point_id: The point ID to delete

        Returns:
            True if deleted, False if not found or not owned
        """
        # First verify ownership
        point = self.get(point_id)
        if point is None:
            return False

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            logger.debug(f"Deleted point {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete point {point_id}: {e}")
            return False

    def list(
        self,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """
        List points for the current org.

        Args:
            limit: Maximum number of results
            filters: Optional additional filters

        Returns:
            List of points owned by the current org
        """
        query_filter = self._create_org_filter(filters)

        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return results
        except Exception as e:
            logger.error(f"List failed: {e}")
            return []

    def health_check(self) -> bool:
        """
        Check if the Qdrant connection is healthy.

        Returns:
            True if connected, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


def get_tenant_qdrant_store(
    org_id: str,
    model_name: str = "text-embedding-3-small",
    embedding_dim: int = 1536,
) -> Optional[TenantQdrantStore]:
    """
    Factory function to create a TenantQdrantStore.

    Attempts to connect to Qdrant using environment configuration.
    Returns None if Qdrant is not available.

    Args:
        org_id: The organization ID for tenant isolation
        model_name: Name of the embedding model
        embedding_dim: Dimension of embeddings

    Returns:
        TenantQdrantStore instance or None if unavailable
    """
    import os

    try:
        from qdrant_client import QdrantClient

        host = os.getenv("QDRANT_HOST", "qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        client = QdrantClient(host=host, port=port)
        client.get_collections()  # Test connection

        return TenantQdrantStore(
            client=client,
            org_id=org_id,
            model_name=model_name,
            embedding_dim=embedding_dim,
        )
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant: {e}")
        return None
