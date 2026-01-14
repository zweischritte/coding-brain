"""OpenSearch client and retrieval operations.

This module provides:
- OpenSearch client wrapper with connection pooling
- Index management (create, update, delete)
- Document indexing with embeddings
- Lexical search (BM25)
- Vector search (kNN)
- Hybrid search (BM25 + vector with RRF)
- Search result ranking and scoring
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable, TYPE_CHECKING
from enum import Enum
from contextlib import contextmanager

import opensearchpy
from opensearchpy.exceptions import (
    ConnectionError as OSConnectionError,
    RequestError,
    NotFoundError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OpenSearchError(Exception):
    """Base exception for OpenSearch operations."""

    pass


class ConnectionError(OpenSearchError):
    """Exception for connection failures."""

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        retries: Optional[int] = None,
    ):
        super().__init__(message)
        self.host = host
        self.retries = retries


class IndexError(OpenSearchError):
    """Exception for index operations."""

    def __init__(
        self,
        message: str,
        index_name: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.index_name = index_name
        self.operation = operation


class DocumentError(OpenSearchError):
    """Exception for document operations."""

    def __init__(
        self,
        message: str,
        doc_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.doc_id = doc_id
        self.operation = operation


class SearchError(OpenSearchError):
    """Exception for search operations."""

    def __init__(
        self,
        message: str,
        query_type: Optional[str] = None,
        took_ms: Optional[int] = None,
    ):
        super().__init__(message)
        self.query_type = query_type
        self.took_ms = took_ms


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OpenSearchConfig:
    """Configuration for OpenSearch client."""

    hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    use_ssl: bool = False
    verify_certs: bool = True
    username: Optional[str] = None
    password: Optional[str] = None
    pool_maxsize: int = 10
    pool_connections: int = 10
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.hosts:
            raise ValueError("hosts cannot be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.pool_maxsize <= 0:
            raise ValueError("pool_maxsize must be positive")
        if self.pool_connections <= 0:
            raise ValueError("pool_connections must be positive")

    @classmethod
    def from_env(cls) -> "OpenSearchConfig":
        """Load configuration from environment variables."""
        hosts_str = os.environ.get("OPENSEARCH_HOSTS", "localhost:9200")
        hosts = [h.strip() for h in hosts_str.split(",")]

        use_ssl = os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true"
        verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"
        username = os.environ.get("OPENSEARCH_USERNAME")
        password = os.environ.get("OPENSEARCH_PASSWORD")

        pool_maxsize = int(os.environ.get("OPENSEARCH_POOL_MAXSIZE", "10"))
        pool_connections = int(os.environ.get("OPENSEARCH_POOL_CONNECTIONS", "10"))
        timeout = int(os.environ.get("OPENSEARCH_TIMEOUT", "30"))
        max_retries = int(os.environ.get("OPENSEARCH_MAX_RETRIES", "3"))

        return cls(
            hosts=hosts,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            username=username,
            password=password,
            pool_maxsize=pool_maxsize,
            pool_connections=pool_connections,
            timeout=timeout,
            max_retries=max_retries,
        )


@dataclass
class IndexConfig:
    """Configuration for an OpenSearch index."""

    name: str
    mappings: Dict[str, Any]
    settings: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    @classmethod
    def for_code(
        cls,
        name: str,
        embedding_dim: int = 768,
        languages: Optional[List[str]] = None,
    ) -> "IndexConfig":
        """Create an index configuration for code."""
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16,
                        },
                    },
                },
                "file_path": {"type": "keyword"},
                "language": {"type": "keyword"},
                "symbol_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"},
                    },
                },
                "symbol_type": {"type": "keyword"},
                "line_start": {"type": "integer"},
                "line_end": {"type": "integer"},
                "repo_id": {"type": "keyword"},
                "chunk_hash": {"type": "keyword"},
                "last_modified": {"type": "date"},
                "is_generated": {"type": "boolean"},
                "generated_reason": {"type": "keyword"},
                "source_tier": {"type": "keyword"},
                "confidence": {"type": "keyword"},
            }
        }

        settings = {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }

        return cls(name=name, mappings=mappings, settings=settings)

    @classmethod
    def for_memory(
        cls,
        name: str,
        embedding_dim: int = 1536,
    ) -> "IndexConfig":
        """Create an index configuration for memory."""
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                "scope": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "team_id": {"type": "keyword"},
                "project_id": {"type": "keyword"},
                "session_id": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        }

        settings = {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }

        return cls(name=name, mappings=mappings, settings=settings)


# =============================================================================
# Document Operations
# =============================================================================


@dataclass
class Document:
    """A document to be indexed in OpenSearch."""

    id: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_opensearch_body(self) -> Dict[str, Any]:
        """Convert to OpenSearch document body."""
        body = {
            "content": self.content,
        }
        if self.embedding is not None:
            body["embedding"] = self.embedding
        body.update(self.metadata)
        return body

    @classmethod
    def from_opensearch_hit(cls, hit: Dict[str, Any]) -> "Document":
        """Create a Document from an OpenSearch hit."""
        source = hit.get("_source", {})
        content = source.pop("content", "")
        embedding = source.pop("embedding", [])

        return cls(
            id=hit["_id"],
            content=content,
            embedding=embedding,
            metadata=source,
        )


@dataclass
class BulkResult:
    """Result of a bulk indexing operation."""

    total: int
    succeeded: int
    failed: int
    errors: List[str] = field(default_factory=list)
    took_ms: int = 0


# =============================================================================
# Search Queries
# =============================================================================


@dataclass
class FieldWeight:
    """Field weight for boosting in search."""

    field: str
    weight: float = 1.0

    def to_boost_string(self) -> str:
        """Convert to OpenSearch boost string format."""
        return f"{self.field}^{self.weight}"


@dataclass
class LexicalSearchParams:
    """Parameters for lexical search."""

    analyzer: Optional[str] = None
    minimum_should_match: Optional[str] = None
    fuzziness: Optional[str] = None
    field_weights: List[FieldWeight] = field(default_factory=list)

    def get_boosted_fields(self) -> List[str]:
        """Get list of fields with boost values."""
        return [fw.to_boost_string() for fw in self.field_weights]


@dataclass
class LexicalSearchQuery:
    """Query for lexical (BM25) search."""

    query_text: str
    fields: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    size: int = 10
    offset: int = 0
    highlight: bool = False

    def to_opensearch_body(
        self,
        params: Optional[LexicalSearchParams] = None,
    ) -> Dict[str, Any]:
        """Convert to OpenSearch query body."""
        # Build multi_match query
        fields = self.fields
        if params and params.field_weights:
            fields = params.get_boosted_fields()

        multi_match = {
            "query": self.query_text,
            "fields": fields,
            "type": "best_fields",
        }

        if params:
            if params.minimum_should_match:
                multi_match["minimum_should_match"] = params.minimum_should_match
            if params.fuzziness:
                multi_match["fuzziness"] = params.fuzziness
            if params.analyzer:
                multi_match["analyzer"] = params.analyzer

        # Build bool query with filters
        bool_query: Dict[str, Any] = {
            "must": [{"multi_match": multi_match}],
        }

        if self.filters:
            filter_clauses = []
            for field_name, value in self.filters.items():
                filter_clauses.append({"term": {field_name: value}})
            bool_query["filter"] = filter_clauses

        body: Dict[str, Any] = {
            "query": {"bool": bool_query},
            "size": self.size,
            "from": self.offset,
        }

        if self.highlight:
            body["highlight"] = {
                "fields": {f: {} for f in self.fields},
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            }

        return body


@dataclass
class VectorSearchParams:
    """Parameters for vector search."""

    ef_search: Optional[int] = None
    space_type: Optional[str] = None


@dataclass
class VectorSearchQuery:
    """Query for vector (kNN) search."""

    embedding: List[float]
    k: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    min_score: Optional[float] = None

    def to_opensearch_body(
        self,
        field: str = "embedding",
        params: Optional[VectorSearchParams] = None,
    ) -> Dict[str, Any]:
        """Convert to OpenSearch kNN query body.

        Uses the OpenSearch k-NN plugin's script_score approach for vector similarity.
        For knn_vector fields, we use knn_score which is optimized for the k-NN plugin.
        """
        # Use script_score with knn_score for k-NN plugin vector fields
        # knn_score works with knn_vector field types
        script_query: Dict[str, Any] = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": field,
                        "query_value": self.embedding,
                        "space_type": "cosinesimil",
                    },
                },
            }
        }

        # Apply filters if present
        if self.filters:
            filter_clauses = []
            for field_name, value in self.filters.items():
                filter_clauses.append({"term": {field_name: value}})
            script_query["script_score"]["query"] = {
                "bool": {
                    "must": [{"match_all": {}}],
                    "filter": filter_clauses,
                }
            }

        body: Dict[str, Any] = {
            "query": script_query,
            "size": self.k,
        }

        if self.min_score is not None:
            body["min_score"] = self.min_score

        return body


@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion."""

    rank_constant: int = 60
    window_size: int = 100

    def calculate_score(self, rank: int) -> float:
        """Calculate RRF score for a given rank."""
        return 1.0 / (self.rank_constant + rank)


@dataclass
class HybridSearchParams:
    """Parameters for hybrid search."""

    lexical_weight: float = 0.35
    vector_weight: float = 0.40
    graph_weight: float = 0.25

    @property
    def total_weight(self) -> float:
        """Calculate total weight sum."""
        return self.lexical_weight + self.vector_weight + self.graph_weight

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.total_weight
        if total > 0:
            self.lexical_weight /= total
            self.vector_weight /= total
            self.graph_weight /= total


@dataclass
class HybridSearchQuery:
    """Query for hybrid (lexical + vector) search."""

    query_text: str
    embedding: List[float]
    lexical_fields: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    rrf_config: Optional[RRFConfig] = None
    size: int = 10
    offset: int = 0

    def to_opensearch_body(
        self,
        params: Optional[HybridSearchParams] = None,
        embedding_field: str = "embedding",
    ) -> Dict[str, Any]:
        """Convert to OpenSearch hybrid query body.

        Uses a bool query combining lexical and vector scores without
        requiring the neural search plugin.
        """
        if params is None:
            params = HybridSearchParams()

        # Lexical part with boosting
        lexical_query = {
            "multi_match": {
                "query": self.query_text,
                "fields": self.lexical_fields,
                "type": "best_fields",
                "boost": params.lexical_weight,
            }
        }

        # Build filter clauses
        filter_clauses = []
        if self.filters:
            for field_name, value in self.filters.items():
                filter_clauses.append({"term": {field_name: value}})

        # Build base query for script_score
        if filter_clauses:
            base_query = {
                "bool": {
                    "must": [lexical_query],
                    "filter": filter_clauses,
                }
            }
        else:
            base_query = lexical_query

        # Combine lexical and vector using script_score
        # The lexical score comes from the base query, vector similarity is added via knn_score
        body: Dict[str, Any] = {
            "query": {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        "params": {
                            "field": embedding_field,
                            "query_value": self.embedding,
                            "space_type": "cosinesimil",
                        },
                    },
                }
            },
            "size": self.size,
            "from": self.offset,
        }

        return body


# =============================================================================
# Search Results
# =============================================================================


@dataclass
class SearchHit:
    """A single search hit."""

    id: str
    score: float
    source: Dict[str, Any]
    highlights: Optional[Dict[str, List[str]]] = None

    @classmethod
    def from_opensearch(cls, hit: Dict[str, Any]) -> "SearchHit":
        """Create a SearchHit from an OpenSearch hit."""
        return cls(
            id=hit["_id"],
            score=hit.get("_score", 0.0),
            source=hit.get("_source", {}),
            highlights=hit.get("highlight"),
        )


@dataclass
class SearchResponse:
    """Response from a search operation."""

    total: int
    hits: List[SearchHit]
    took_ms: int
    aggregations: Optional[Dict[str, Any]] = None

    @property
    def is_empty(self) -> bool:
        """Check if response has no hits."""
        return len(self.hits) == 0


@dataclass
class SearchResult:
    """Higher-level search result wrapper."""

    query: str
    response: SearchResponse
    search_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Scoring Functions
# =============================================================================


@dataclass
class ScoringFunction:
    """A scoring function for result boosting."""

    type: str  # field_value_factor, decay, script_score
    field: Optional[str] = None
    factor: Optional[float] = None
    modifier: Optional[str] = None
    decay_function: Optional[str] = None  # exp, linear, gauss
    origin: Optional[str] = None
    scale: Optional[str] = None
    decay: Optional[float] = None

    def to_opensearch_body(self) -> Dict[str, Any]:
        """Convert to OpenSearch function score body."""
        if self.type == "field_value_factor":
            body: Dict[str, Any] = {
                "field_value_factor": {
                    "field": self.field,
                }
            }
            if self.factor:
                body["field_value_factor"]["factor"] = self.factor
            if self.modifier:
                body["field_value_factor"]["modifier"] = self.modifier
            return body
        elif self.type == "decay":
            return {
                self.decay_function: {
                    self.field: {
                        "origin": self.origin,
                        "scale": self.scale,
                        "decay": self.decay,
                    }
                }
            }
        return {}


# =============================================================================
# Index Information
# =============================================================================


@dataclass
class IndexInfo:
    """Information about an index."""

    name: str
    mappings: Dict[str, Any]
    settings: Dict[str, Any]
    aliases: List[str]


@dataclass
class IndexStats:
    """Statistics for an index."""

    doc_count: int
    deleted_docs: int
    size_bytes: int


# =============================================================================
# Index Manager
# =============================================================================


class IndexManager:
    """Manager for index operations."""

    def __init__(self, client: "OpenSearchClient"):
        """Initialize the index manager."""
        self.client = client

    def create_index(
        self,
        config: IndexConfig,
        ignore_existing: bool = False,
    ) -> bool:
        """Create an index with the given configuration."""
        try:
            body: Dict[str, Any] = {
                "mappings": config.mappings,
                "settings": config.settings,
            }

            if config.aliases:
                body["aliases"] = {alias: {} for alias in config.aliases}

            self.client._client.indices.create(
                index=config.name,
                body=body,
            )
            return True

        except Exception as e:
            # Check if it's a resource_already_exists_exception
            error_str = str(e)
            if "resource_already_exists" in error_str.lower():
                if ignore_existing:
                    return False
            raise IndexError(
                f"Failed to create index: {e}",
                index_name=config.name,
                operation="create",
            )

    def delete_index(
        self,
        index_name: str,
        ignore_not_found: bool = False,
    ) -> bool:
        """Delete an index."""
        try:
            self.client._client.indices.delete(index=index_name)
            return True
        except Exception as e:
            if "not_found" in str(e).lower() or "index_not_found" in str(e).lower():
                if ignore_not_found:
                    return False
            raise IndexError(
                f"Failed to delete index: {e}",
                index_name=index_name,
                operation="delete",
            )

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        return self.client._client.indices.exists(index=index_name)

    def get_index_info(self, index_name: str) -> IndexInfo:
        """Get information about an index."""
        response = self.client._client.indices.get(index=index_name)
        index_data = response[index_name]

        aliases = list(index_data.get("aliases", {}).keys())

        return IndexInfo(
            name=index_name,
            mappings=index_data.get("mappings", {}),
            settings=index_data.get("settings", {}),
            aliases=aliases,
        )

    def get_index_stats(self, index_name: str) -> IndexStats:
        """Get statistics for an index."""
        response = self.client._client.indices.stats(index=index_name)
        primaries = response["_all"]["primaries"]

        return IndexStats(
            doc_count=primaries["docs"]["count"],
            deleted_docs=primaries["docs"]["deleted"],
            size_bytes=primaries["store"]["size_in_bytes"],
        )

    def update_mapping(
        self,
        index_name: str,
        mapping: Dict[str, Any],
    ) -> bool:
        """Update the mapping for an index."""
        self.client._client.indices.put_mapping(
            index=index_name,
            body=mapping,
        )
        return True

    def update_settings(
        self,
        index_name: str,
        settings: Dict[str, Any],
    ) -> bool:
        """Update the settings for an index."""
        self.client._client.indices.put_settings(
            index=index_name,
            body=settings,
        )
        return True

    def refresh_index(self, index_name: str) -> None:
        """Refresh an index to make recent changes searchable."""
        self.client._client.indices.refresh(index=index_name)

    def list_indices(self, pattern: str = "*") -> List[str]:
        """List indices matching a pattern."""
        response = self.client._client.indices.get(index=pattern)
        return list(response.keys())


# =============================================================================
# OpenSearch Client
# =============================================================================


class OpenSearchClient:
    """OpenSearch client wrapper with connection pooling and retry logic."""

    def __init__(self, config: OpenSearchConfig):
        """Initialize the client with configuration."""
        self.config = config
        self._client: Any = None
        self._async_client: Any = None

    def connect(self) -> None:
        """Connect to OpenSearch."""
        retries = 0

        while retries < self.config.max_retries:
            try:
                kwargs: Dict[str, Any] = {
                    "hosts": self.config.hosts,
                    "use_ssl": self.config.use_ssl,
                    "verify_certs": self.config.verify_certs,
                    "pool_maxsize": self.config.pool_maxsize,
                    "pool_connections": self.config.pool_connections,
                    "timeout": self.config.timeout,
                    "retry_on_timeout": self.config.retry_on_timeout,
                }

                if self.config.username and self.config.password:
                    kwargs["http_auth"] = (
                        self.config.username,
                        self.config.password,
                    )

                self._client = opensearchpy.OpenSearch(**kwargs)
                return

            except OSConnectionError as e:
                retries += 1
                if retries >= self.config.max_retries:
                    raise ConnectionError(
                        f"Failed to connect after {retries} retries: {e}",
                        host=self.config.hosts[0] if self.config.hosts else None,
                        retries=retries,
                    )
                time.sleep(0.5 * retries)  # Exponential backoff

    async def connect_async(self) -> None:
        """Connect to OpenSearch asynchronously."""
        kwargs: Dict[str, Any] = {
            "hosts": self.config.hosts,
            "use_ssl": self.config.use_ssl,
            "verify_certs": self.config.verify_certs,
            "timeout": self.config.timeout,
        }

        if self.config.username and self.config.password:
            kwargs["http_auth"] = (self.config.username, self.config.password)

        self._async_client = opensearchpy.AsyncOpenSearch(**kwargs)

    def health(self) -> Dict[str, Any]:
        """Check cluster health."""
        return self._client.cluster.health()

    def close(self) -> None:
        """Close the client connection."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OpenSearchClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # -------------------------------------------------------------------------
    # Document Operations
    # -------------------------------------------------------------------------

    def index_document(
        self,
        index_name: str,
        document: Document,
    ) -> bool:
        """Index a single document."""
        self._client.index(
            index=index_name,
            id=document.id,
            body=document.to_opensearch_body(),
        )
        return True

    def bulk_index(
        self,
        index_name: str,
        documents: List[Document],
        batch_size: int = 100,
    ) -> BulkResult:
        """Bulk index documents."""
        total = len(documents)
        succeeded = 0
        failed = 0
        errors: List[str] = []
        total_time = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            actions = []

            for doc in batch:
                actions.append({"index": {"_index": index_name, "_id": doc.id}})
                actions.append(doc.to_opensearch_body())

            response = self._client.bulk(body=actions)
            total_time += response.get("took", 0)

            for item in response.get("items", []):
                action_result = item.get("index", {})
                if "error" in action_result:
                    failed += 1
                    doc_id = action_result.get("_id", "unknown")
                    error_reason = action_result["error"].get("reason", "unknown")
                    errors.append(f"{doc_id}: {error_reason}")
                else:
                    succeeded += 1

        return BulkResult(
            total=total,
            succeeded=succeeded,
            failed=failed,
            errors=errors,
            took_ms=total_time,
        )

    def get_document(
        self,
        index_name: str,
        doc_id: str,
    ) -> Optional[Document]:
        """Get a document by ID."""
        try:
            response = self._client.get(index=index_name, id=doc_id)
            if response.get("found"):
                return Document.from_opensearch_hit(response)
            return None
        except Exception as e:
            if "not_found" in str(e).lower():
                return None
            raise DocumentError(
                f"Failed to get document: {e}",
                doc_id=doc_id,
                operation="get",
            )

    def delete_document(
        self,
        index_name: str,
        doc_id: str,
    ) -> bool:
        """Delete a document by ID."""
        self._client.delete(index=index_name, id=doc_id)
        return True

    def delete_by_query(
        self,
        index_name: str,
        query: Dict[str, Any],
    ) -> int:
        """Delete documents matching a query."""
        response = self._client.delete_by_query(
            index=index_name,
            body={"query": query},
        )
        return response.get("deleted", 0)

    def update_document(
        self,
        index_name: str,
        doc_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a document partially."""
        self._client.update(
            index=index_name,
            id=doc_id,
            body={"doc": updates},
        )
        return True

    def count(
        self,
        index_name: str,
        query: Dict[str, Any],
    ) -> int:
        """Count documents matching a query."""
        response = self._client.count(
            index=index_name,
            body={"query": query},
        )
        return response.get("count", 0)

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def lexical_search(
        self,
        index_name: str,
        query: LexicalSearchQuery,
        params: Optional[LexicalSearchParams] = None,
    ) -> SearchResponse:
        """Perform lexical (BM25) search."""
        body = query.to_opensearch_body(params)

        response = self._client.search(
            index=index_name,
            body=body,
        )

        hits = [
            SearchHit.from_opensearch(hit)
            for hit in response["hits"]["hits"]
        ]

        return SearchResponse(
            total=response["hits"]["total"]["value"],
            hits=hits,
            took_ms=response["took"],
        )

    def vector_search(
        self,
        index_name: str,
        query: VectorSearchQuery,
        embedding_field: str = "embedding",
        params: Optional[VectorSearchParams] = None,
    ) -> SearchResponse:
        """Perform vector (kNN) search."""
        body = query.to_opensearch_body(field=embedding_field, params=params)

        response = self._client.search(
            index=index_name,
            body=body,
        )

        hits = [
            SearchHit.from_opensearch(hit)
            for hit in response["hits"]["hits"]
        ]

        return SearchResponse(
            total=response["hits"]["total"]["value"],
            hits=hits,
            took_ms=response["took"],
        )

    def hybrid_search(
        self,
        index_name: str,
        query: HybridSearchQuery,
        params: Optional[HybridSearchParams] = None,
        embedding_field: str = "embedding",
        fallback: bool = False,
    ) -> SearchResponse:
        """Perform hybrid (lexical + vector) search."""
        body = query.to_opensearch_body(params=params, embedding_field=embedding_field)

        try:
            response = self._client.search(
                index=index_name,
                body=body,
            )
        except Exception as e:
            if fallback:
                # Fall back to lexical search
                logger.warning(f"Hybrid search failed, falling back to lexical: {e}")
                lexical_query = LexicalSearchQuery(
                    query_text=query.query_text,
                    fields=query.lexical_fields,
                    filters=query.filters,
                    size=query.size,
                    offset=query.offset,
                )
                return self.lexical_search(index_name, lexical_query)
            raise SearchError(
                f"Hybrid search failed: {e}",
                query_type="hybrid",
            )

        hits = [
            SearchHit.from_opensearch(hit)
            for hit in response["hits"]["hits"]
        ]

        return SearchResponse(
            total=response["hits"]["total"]["value"],
            hits=hits,
            took_ms=response["took"],
        )


# =============================================================================
# Factory
# =============================================================================


def create_opensearch_client(
    config: Optional[OpenSearchConfig] = None,
    from_env: bool = False,
) -> OpenSearchClient:
    """Create an OpenSearch client."""
    if from_env:
        config = OpenSearchConfig.from_env()
    elif config is None:
        config = OpenSearchConfig()

    return OpenSearchClient(config)
