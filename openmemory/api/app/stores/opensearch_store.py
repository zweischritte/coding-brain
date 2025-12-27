"""
Tenant-scoped OpenSearch store with alias-based routing.

This module implements an OpenSearch-backed document store with org_id-based
tenant isolation using index aliases. The alias strategy allows:
- Small tenants to share a common index via their own alias
- Large tenants to get dedicated indices for performance isolation
- All operations routed through tenant-specific aliases

Key features:
- Tenant alias creation (tenant_{org_id})
- Automatic org_id injection into documents
- Hybrid search (lexical + vector) with tenant filtering
- Configurable shared vs dedicated index strategy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class OpenSearchClientProtocol(Protocol):
    """Protocol for OpenSearch client operations."""

    @property
    def indices(self) -> Any: ...

    def index(self, index: str, body: Dict, id: Optional[str] = None, refresh: Optional[str] = None) -> Dict: ...
    def get(self, index: str, id: str) -> Dict: ...
    def delete(self, index: str, id: str, refresh: Optional[str] = None) -> Dict: ...
    def search(self, index: Optional[str] = None, body: Optional[Dict] = None) -> Dict: ...
    def count(self, index: Optional[str] = None, body: Optional[Dict] = None) -> Dict: ...
    def ping(self) -> bool: ...


@dataclass
class TenantOpenSearchConfig:
    """Configuration for TenantOpenSearchStore."""

    # Index settings
    number_of_shards: int = 1
    number_of_replicas: int = 0

    # Vector settings for kNN
    embedding_dim: int = 1536
    similarity: str = "cosinesimil"

    # Hybrid search weights
    lexical_weight: float = 0.35
    vector_weight: float = 0.40
    graph_weight: float = 0.25


class TenantOpenSearchStore:
    """
    Tenant-scoped OpenSearch document store with alias-based routing.

    Provides document storage with automatic org_id-based isolation.
    Uses index aliases to route tenant operations:
    - Shared index: memories_shared (for small tenants)
    - Dedicated index: memories_{org_id} (for large tenants)

    All tenants get their own alias: tenant_{org_id}

    Example:
        store = TenantOpenSearchStore(
            client=opensearch_client,
            org_id="11111111-1111-1111-1111-111111111111",
            index_prefix="memories",
        )
        store.index("doc-1", {"content": "Hello world"})
        results = store.search("hello")
    """

    def __init__(
        self,
        client: OpenSearchClientProtocol,
        org_id: str,
        index_prefix: str = "memories",
        use_dedicated_index: bool = False,
        config: Optional[TenantOpenSearchConfig] = None,
    ):
        """
        Initialize the tenant-scoped OpenSearch store.

        Args:
            client: OpenSearch client instance
            org_id: The organization ID for tenant isolation
            index_prefix: Prefix for index names
            use_dedicated_index: If True, creates a dedicated index for this tenant
            config: Optional configuration
        """
        self.client = client
        self.org_id = org_id
        self.index_prefix = index_prefix
        self.use_dedicated_index = use_dedicated_index
        self.config = config or TenantOpenSearchConfig()

        # Determine index and alias names
        self.alias_name = f"tenant_{org_id}"
        if use_dedicated_index:
            self.index_name = f"{index_prefix}_{org_id}"
        else:
            self.index_name = f"{index_prefix}_shared"

        # Initialize index and alias
        self._ensure_index_and_alias()

    def _ensure_index_and_alias(self) -> None:
        """Ensure the index exists and alias is configured."""
        try:
            # Check if index exists
            if not self.client.indices.exists(self.index_name):
                # Create index with settings
                body = {
                    "settings": {
                        "number_of_shards": self.config.number_of_shards,
                        "number_of_replicas": self.config.number_of_replicas,
                        "index.knn": True,
                    },
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "org_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "vault": {"type": "keyword"},
                            "layer": {"type": "keyword"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": self.config.embedding_dim,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": self.config.similarity,
                                    "engine": "nmslib",
                                    "parameters": {
                                        "ef_construction": 512,
                                        "m": 16,
                                    },
                                },
                            },
                        }
                    },
                    "aliases": {
                        self.alias_name: {}
                    }
                }
                self.client.indices.create(self.index_name, body=body)
                logger.info(f"Created index {self.index_name} with alias {self.alias_name}")
            else:
                # Index exists, ensure alias exists
                self._ensure_alias()

        except Exception as e:
            logger.error(f"Failed to ensure index and alias: {e}")
            raise

    def _ensure_alias(self) -> None:
        """Ensure the tenant alias exists and points to the correct index."""
        try:
            # Check if alias already exists
            existing_aliases = self.client.indices.get_alias(name=self.alias_name)
            if self.alias_name not in str(existing_aliases):
                # Create alias
                self.client.indices.put_alias(self.index_name, self.alias_name)
                logger.debug(f"Created alias {self.alias_name} -> {self.index_name}")
        except Exception as e:
            # Alias might not exist, try to create it
            try:
                self.client.indices.put_alias(self.index_name, self.alias_name)
                logger.debug(f"Created alias {self.alias_name} -> {self.index_name}")
            except Exception as inner_e:
                logger.debug(f"Alias might already exist: {inner_e}")

    def _create_org_filter(self) -> Dict[str, Any]:
        """Create a filter clause for org_id."""
        return {"term": {"org_id": self.org_id}}

    def index(
        self,
        doc_id: str,
        document: Dict[str, Any],
        refresh: bool = False,
    ) -> bool:
        """
        Index a document with automatic org_id injection.

        Args:
            doc_id: Unique document ID
            document: The document to index
            refresh: Whether to refresh the index immediately

        Returns:
            True if successful
        """
        # Inject org_id
        full_doc = document.copy()
        full_doc["org_id"] = self.org_id

        try:
            self.client.index(
                index=self.alias_name,
                body=full_doc,
                id=doc_id,
                refresh="wait_for" if refresh else None,
            )
            logger.debug(f"Indexed document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            return False

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID, verifying org_id ownership.

        Args:
            doc_id: The document ID

        Returns:
            The document if found and owned, None otherwise
        """
        try:
            result = self.client.get(index=self.alias_name, id=doc_id)

            if not result.get("found", True):
                return None

            # Verify ownership
            source = result.get("_source", {})
            if source.get("org_id") != self.org_id:
                logger.warning(
                    f"Tenant isolation violation: {self.org_id} tried to access "
                    f"document owned by {source.get('org_id')}"
                )
                return None

            return result
        except Exception as e:
            logger.debug(f"Document {doc_id} not found: {e}")
            return None

    def delete(self, doc_id: str, refresh: bool = False) -> bool:
        """
        Delete a document, verifying org_id ownership first.

        Args:
            doc_id: The document ID
            refresh: Whether to refresh immediately

        Returns:
            True if deleted, False if not found or not owned
        """
        # First verify ownership
        doc = self.get(doc_id)
        if doc is None:
            return False

        try:
            self.client.delete(
                index=self.alias_name,
                id=doc_id,
                refresh="wait_for" if refresh else None,
            )
            logger.debug(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def search(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents, filtered by org_id.

        Args:
            query_text: The search query
            limit: Maximum number of results
            filters: Optional additional filters

        Returns:
            List of matching documents
        """
        # Build query with org_id filter
        filter_clauses = [self._create_org_filter()]
        if filters:
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})

        body = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": query_text}}
                    ] if query_text else [{"match_all": {}}],
                    "filter": filter_clauses,
                }
            }
        }

        try:
            result = self.client.search(index=self.alias_name, body=body)
            return result.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining lexical and vector search.

        Args:
            query_text: The text query
            query_vector: The embedding vector
            limit: Maximum number of results
            filters: Optional additional filters

        Returns:
            List of matching documents with combined scores
        """
        # Build filter clauses
        filter_clauses = [self._create_org_filter()]
        if filters:
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})

        # Hybrid query combining lexical and kNN
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        # Lexical component
                        {
                            "match": {
                                "content": {
                                    "query": query_text,
                                    "boost": self.config.lexical_weight,
                                }
                            }
                        },
                        # Vector component (script_score for kNN)
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "knn_score",
                                    "lang": "knn",
                                    "params": {
                                        "field": "embedding",
                                        "query_value": query_vector,
                                        "space_type": self.config.similarity,
                                    }
                                }
                            }
                        }
                    ],
                    "filter": filter_clauses,
                    "minimum_should_match": 1,
                }
            }
        }

        try:
            result = self.client.search(index=self.alias_name, body=body)
            return result.get("hits", {}).get("hits", [])
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents for the current org.

        Args:
            filters: Optional additional filters

        Returns:
            Document count
        """
        filter_clauses = [self._create_org_filter()]
        if filters:
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})

        body = {
            "query": {
                "bool": {
                    "filter": filter_clauses
                }
            }
        }

        try:
            result = self.client.count(index=self.alias_name, body=body)
            return result.get("count", 0)
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0

    def health_check(self) -> bool:
        """
        Check if the OpenSearch connection is healthy.

        Returns:
            True if connected, False otherwise
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False


def get_tenant_opensearch_store(
    org_id: str,
    index_prefix: str = "memories",
    use_dedicated_index: bool = False,
) -> Optional[TenantOpenSearchStore]:
    """
    Factory function to create a TenantOpenSearchStore.

    Attempts to connect to OpenSearch using environment configuration.
    Returns None if OpenSearch is not available.

    Args:
        org_id: The organization ID for tenant isolation
        index_prefix: Prefix for index names
        use_dedicated_index: Whether to use dedicated index

    Returns:
        TenantOpenSearchStore instance or None if unavailable
    """
    import os

    try:
        from opensearchpy import OpenSearch

        hosts = os.getenv("OPENSEARCH_HOSTS", "localhost:9200").split(",")
        username = os.getenv("OPENSEARCH_USERNAME")
        password = os.getenv("OPENSEARCH_PASSWORD")

        auth = (username, password) if username and password else None

        client = OpenSearch(
            hosts=[{"host": h.split(":")[0], "port": int(h.split(":")[1])} for h in hosts],
            http_auth=auth,
            use_ssl=os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
            verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true",
            timeout=30,
        )

        if not client.ping():
            logger.warning("OpenSearch ping failed")
            return None

        return TenantOpenSearchStore(
            client=client,
            org_id=org_id,
            index_prefix=index_prefix,
            use_dedicated_index=use_dedicated_index,
        )
    except Exception as e:
        logger.warning(f"Failed to connect to OpenSearch: {e}")
        return None
