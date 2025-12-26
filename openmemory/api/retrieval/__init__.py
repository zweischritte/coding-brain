"""Retrieval module for OpenSearch-based hybrid search.

This module provides:
- OpenSearch client wrapper with connection pooling
- Index management (create, update, delete)
- Document indexing with embeddings
- Lexical search (BM25)
- Vector search (kNN)
- Hybrid search (BM25 + vector with RRF)
- Tri-hybrid search (lexical + vector + graph with RRF)
- Search result ranking and scoring
"""

from openmemory.api.retrieval.opensearch import (
    # Configuration
    OpenSearchConfig,
    IndexConfig,
    # Client
    OpenSearchClient,
    # Index management
    IndexManager,
    IndexInfo,
    IndexStats,
    # Document operations
    Document,
    BulkResult,
    # Search operations
    SearchResult,
    SearchHit,
    SearchResponse,
    # Lexical search
    LexicalSearchQuery,
    LexicalSearchParams,
    # Vector search
    VectorSearchQuery,
    VectorSearchParams,
    # Hybrid search
    HybridSearchQuery,
    HybridSearchParams,
    RRFConfig,
    # Ranking
    ScoringFunction,
    FieldWeight,
    # Exceptions
    OpenSearchError,
    ConnectionError,
    IndexError,
    DocumentError,
    SearchError,
    # Factory
    create_opensearch_client,
)

from openmemory.api.retrieval.trihybrid import (
    # Configuration
    TriHybridConfig,
    # Query
    TriHybridQuery,
    # Graph context
    GraphContext,
    GraphContextFetcher,
    # Score normalization
    ScoreNormalizer,
    # Result fusion
    FusionMethod,
    RankedResult,
    ResultFusion,
    # Result types
    TriHybridTiming,
    TriHybridHit,
    TriHybridResult,
    # Main retriever
    TriHybridRetriever,
    # Exceptions
    RetrievalError,
    # Factory
    create_trihybrid_retriever,
)

from openmemory.api.retrieval.reranker import (
    # Configuration
    RerankerConfig,
    # Result types
    RerankedResult,
    RerankedTriHybridTiming,
    RerankedTriHybridResult,
    # Adapters
    RerankerAdapter,
    CrossEncoderReranker,
    CohereReranker,
    NoOpReranker,
    # Integration
    rerank_trihybrid_results,
    # Exceptions
    RerankerError,
    RerankerTimeoutError,
    # Factory
    create_reranker,
)

from openmemory.api.retrieval.prefetch_cache import (
    # Configuration
    CacheConfig,
    # Core types
    CacheEntry,
    CacheKey,
    CacheMetrics,
    CacheStats,
    # Cache implementations
    PrefetchCache,
    LRUPrefetchCache,
    # Speculative patterns
    SpeculativePattern,
    PatternType,
    SpeculativeQueryGenerator,
    # Cached retriever
    CachedTriHybridRetriever,
    # Exceptions
    CacheError,
    CacheKeyError,
    CacheMissError,
    # Factory
    create_prefetch_cache,
    create_cached_retriever,
)

from openmemory.api.retrieval.embedding_pipeline import (
    # Configuration
    EmbeddingConfig,
    EmbeddingModel,
    # Content hashing
    ContentHash,
    # Result types
    EmbeddingResult,
    EmbeddingBatch,
    # Storage
    EmbeddingStore,
    InMemoryEmbeddingStore,
    ContentAddressedEmbeddingStore,
    # Providers
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    # Pipeline
    EmbeddingPipeline,
    # Shadow pipeline
    ShadowPipeline,
    ShadowResult,
    ShadowComparison,
    # Metrics
    EmbeddingMetrics,
    SimilarityMetric,
    # Exceptions
    EmbeddingError,
    ProviderError,
    StorageError,
    # Factory
    create_embedding_pipeline,
    create_shadow_pipeline,
)

from openmemory.api.retrieval.graph_scaling import (
    # Configuration
    GraphScalingConfig,
    PartitionConfig,
    ReplicaConfig,
    MaterializedViewConfig,
    # Partitioning
    PartitionStrategy,
    HashPartitioner,
    RangePartitioner,
    PartitionInfo,
    PartitionRouter,
    # Replicas
    ReplicaManager,
    ReplicaNode,
    ReplicaStatus,
    ReadPreference,
    # Materialized views
    MaterializedView,
    ViewDefinition,
    ViewRefreshPolicy,
    MaterializedViewManager,
    # Connection pooling
    GraphConnectionPool,
    PooledConnection,
    PoolStats,
    # Query routing
    QueryRouter,
    RoutingDecision,
    # Health
    NodeHealth,
    HealthMonitor,
    # Manager
    GraphScalingManager,
    PartitionedGraph,
    # Exceptions
    GraphScalingError,
    PartitionError,
    ReplicaError,
    ViewError,
    # Factory
    create_graph_scaling_manager,
    create_partitioned_graph,
)

__all__ = [
    # Configuration
    "OpenSearchConfig",
    "IndexConfig",
    # Client
    "OpenSearchClient",
    # Index management
    "IndexManager",
    "IndexInfo",
    "IndexStats",
    # Document operations
    "Document",
    "BulkResult",
    # Search operations
    "SearchResult",
    "SearchHit",
    "SearchResponse",
    # Lexical search
    "LexicalSearchQuery",
    "LexicalSearchParams",
    # Vector search
    "VectorSearchQuery",
    "VectorSearchParams",
    # Hybrid search
    "HybridSearchQuery",
    "HybridSearchParams",
    "RRFConfig",
    # Ranking
    "ScoringFunction",
    "FieldWeight",
    # Exceptions
    "OpenSearchError",
    "ConnectionError",
    "IndexError",
    "DocumentError",
    "SearchError",
    # Factory
    "create_opensearch_client",
    # Tri-hybrid configuration
    "TriHybridConfig",
    # Tri-hybrid query
    "TriHybridQuery",
    # Graph context
    "GraphContext",
    "GraphContextFetcher",
    # Score normalization
    "ScoreNormalizer",
    # Result fusion
    "FusionMethod",
    "RankedResult",
    "ResultFusion",
    # Tri-hybrid result types
    "TriHybridTiming",
    "TriHybridHit",
    "TriHybridResult",
    # Tri-hybrid retriever
    "TriHybridRetriever",
    # Tri-hybrid exceptions
    "RetrievalError",
    # Tri-hybrid factory
    "create_trihybrid_retriever",
    # Reranker configuration
    "RerankerConfig",
    # Reranker result types
    "RerankedResult",
    "RerankedTriHybridTiming",
    "RerankedTriHybridResult",
    # Reranker adapters
    "RerankerAdapter",
    "CrossEncoderReranker",
    "CohereReranker",
    "NoOpReranker",
    # Reranker integration
    "rerank_trihybrid_results",
    # Reranker exceptions
    "RerankerError",
    "RerankerTimeoutError",
    # Reranker factory
    "create_reranker",
    # Prefetch cache configuration
    "CacheConfig",
    # Prefetch cache core types
    "CacheEntry",
    "CacheKey",
    "CacheMetrics",
    "CacheStats",
    # Prefetch cache implementations
    "PrefetchCache",
    "LRUPrefetchCache",
    # Speculative patterns
    "SpeculativePattern",
    "PatternType",
    "SpeculativeQueryGenerator",
    # Cached retriever
    "CachedTriHybridRetriever",
    # Prefetch cache exceptions
    "CacheError",
    "CacheKeyError",
    "CacheMissError",
    # Prefetch cache factory
    "create_prefetch_cache",
    "create_cached_retriever",
    # Embedding pipeline configuration
    "EmbeddingConfig",
    "EmbeddingModel",
    # Content hashing
    "ContentHash",
    # Embedding result types
    "EmbeddingResult",
    "EmbeddingBatch",
    # Embedding storage
    "EmbeddingStore",
    "InMemoryEmbeddingStore",
    "ContentAddressedEmbeddingStore",
    # Embedding providers
    "EmbeddingProvider",
    "OllamaProvider",
    "OpenAIProvider",
    # Embedding pipeline
    "EmbeddingPipeline",
    # Shadow pipeline
    "ShadowPipeline",
    "ShadowResult",
    "ShadowComparison",
    # Embedding metrics
    "EmbeddingMetrics",
    "SimilarityMetric",
    # Embedding exceptions
    "EmbeddingError",
    "ProviderError",
    "StorageError",
    # Embedding factory
    "create_embedding_pipeline",
    "create_shadow_pipeline",
    # Graph scaling configuration
    "GraphScalingConfig",
    "PartitionConfig",
    "ReplicaConfig",
    "MaterializedViewConfig",
    # Partitioning
    "PartitionStrategy",
    "HashPartitioner",
    "RangePartitioner",
    "PartitionInfo",
    "PartitionRouter",
    # Replicas
    "ReplicaManager",
    "ReplicaNode",
    "ReplicaStatus",
    "ReadPreference",
    # Materialized views
    "MaterializedView",
    "ViewDefinition",
    "ViewRefreshPolicy",
    "MaterializedViewManager",
    # Connection pooling
    "GraphConnectionPool",
    "PooledConnection",
    "PoolStats",
    # Query routing
    "QueryRouter",
    "RoutingDecision",
    # Health
    "NodeHealth",
    "HealthMonitor",
    # Graph scaling manager
    "GraphScalingManager",
    "PartitionedGraph",
    # Graph scaling exceptions
    "GraphScalingError",
    "PartitionError",
    "ReplicaError",
    "ViewError",
    # Graph scaling factory
    "create_graph_scaling_manager",
    "create_partitioned_graph",
]
