"""Code indexing module for AST parsing and symbol extraction.

This module provides:
- AST parsing with Tree-sitter (Python, TypeScript, Java, Go)
- Incremental indexing with Merkle tree
- SCIP symbol ID format extraction
- Bootstrap state and priority queue for tiered indexing
- Cross-language API boundary detection (REST, fetch, axios)
- Code embeddings pipeline
"""

from openmemory.api.indexing.ast_parser import (
    # Core types
    Language,
    SymbolType,
    Symbol,
    ParseResult,
    ParseError,
    ParseStatistics,
    # Parser interface
    LanguagePlugin,
    ASTParser,
    ASTParserConfig,
    # Concrete plugins
    PythonPlugin,
    TypeScriptPlugin,
    JavaPlugin,
    GoPlugin,
    # Factory
    create_parser,
    # Exceptions
    UnsupportedLanguageError,
    ParseFailedError,
)

from openmemory.api.indexing.merkle_tree import (
    # Core types
    HashValue,
    FileNode,
    DirectoryNode,
    MerkleTree,
    ChangeType,
    Change,
    ChangeSet,
    # State management
    TreeState,
    StateStore,
    MemoryStateStore,
    FileStateStore,
    # Incremental indexer
    IncrementalIndexer,
    IndexerConfig,
    IndexTransaction,
    TransactionStatus,
    ScanResult,
    # Exceptions
    TransactionFailedError,
    StateCorruptedError,
    # Factory
    create_indexer,
)

from openmemory.api.indexing.scip_symbols import (
    # Core types
    SCIPScheme,
    SCIPDescriptor,
    SCIPSymbolID,
    # Builder
    SymbolIDBuilder,
    # Package resolution
    PackageResolver,
    PythonPackageResolver,
    TypeScriptPackageResolver,
    JavaPackageResolver,
    # Extractor
    SCIPSymbolExtractor,
    # Factory
    create_extractor,
    # Exceptions
    InvalidSymbolError,
)

from openmemory.api.indexing.graph_projection import (
    # Core types
    CodeNode,
    CodeNodeType,
    CodeEdge,
    CodeEdgeType,
    # Configuration
    GraphProjectionConfig,
    # Builders
    FileNodeBuilder,
    SymbolNodeBuilder,
    PackageNodeBuilder,
    SchemaFieldNodeBuilder,
    FieldPathNodeBuilder,
    OpenAPIDefNodeBuilder,
    EdgeBuilder,
    # Batch operations
    BatchOperation,
    BatchOperationType,
    BatchResult,
    # Transaction
    ProjectionTransaction,
    TransactionState,
    # Driver abstraction
    Neo4jDriver,
    MemoryGraphStore,
    # Service
    GraphProjection,
    # Exceptions
    GraphProjectionError,
    ConstraintViolationError,
    TransactionError,
    # Factory
    create_graph_projection,
)

from openmemory.api.indexing.code_graph_driver import (
    CodeGraphDriver,
    create_code_graph_driver,
)

from openmemory.api.indexing.code_indexer import (
    CodeIndexingService,
    CodeIndexSummary,
    FileIndexStats,
)

from openmemory.api.indexing.bootstrap import (
    # Core types
    FilePriority,
    PriorityTier,
    IndexingPhase,
    BootstrapProgress,
    # Priority Queue
    PriorityQueue,
    FilePriorityScorer,
    # Bootstrap State
    BootstrapState,
    BootstrapStateStore,
    MemoryBootstrapStateStore,
    FileBootstrapStateStore,
    # Bootstrap Status API
    BootstrapStatus,
    BootstrapStatusResponse,
    FileProgress,
    # Bootstrap Manager
    BootstrapConfig,
    BootstrapManager,
    # Callback types
    ProgressCallback,
    FileCompleteCallback,
    PhaseChangeCallback,
    # Result types
    ScanResult as BootstrapScanResult,
    IndexBatchResult,
    # Exceptions
    BootstrapError,
    StateNotFoundError,
    # Factory
    create_bootstrap_manager,
)

from openmemory.api.indexing.api_boundaries import (
    # Core types
    HTTPMethod,
    APIEndpoint,
    APIClient,
    APIBoundaryLink,
    APIBoundaryAnalysisResult,
    # Edge and node types
    APIBoundaryEdgeType,
    APIBoundaryNodeType,
    APIBoundaryNode,
    APIBoundaryEdge,
    # Builders
    APIEndpointNodeBuilder,
    APIClientNodeBuilder,
    APIBoundaryEdgeBuilder,
    # Detectors
    RestEndpointDetector,
    APIClientDetector,
    # Path matching
    PathMatcher,
    # Linker
    APIBoundaryLinker,
    # Projection
    APIBoundaryProjection,
    # Analyzer
    APIBoundaryAnalyzer,
    # Factory
    create_api_boundary_analyzer,
    create_rest_endpoint_detector,
    create_api_client_detector,
)

__all__ = [
    # AST Parser
    "Language",
    "SymbolType",
    "Symbol",
    "ParseResult",
    "ParseError",
    "ParseStatistics",
    "LanguagePlugin",
    "ASTParser",
    "ASTParserConfig",
    "PythonPlugin",
    "TypeScriptPlugin",
    "JavaPlugin",
    "create_parser",
    "UnsupportedLanguageError",
    "ParseFailedError",
    # Merkle Tree
    "HashValue",
    "FileNode",
    "DirectoryNode",
    "MerkleTree",
    "ChangeType",
    "Change",
    "ChangeSet",
    "TreeState",
    "StateStore",
    "MemoryStateStore",
    "FileStateStore",
    "IncrementalIndexer",
    "IndexerConfig",
    "IndexTransaction",
    "TransactionStatus",
    "ScanResult",
    "TransactionFailedError",
    "StateCorruptedError",
    "create_indexer",
    # SCIP Symbols
    "SCIPScheme",
    "SCIPDescriptor",
    "SCIPSymbolID",
    "SymbolIDBuilder",
    "PackageResolver",
    "PythonPackageResolver",
    "TypeScriptPackageResolver",
    "JavaPackageResolver",
    "SCIPSymbolExtractor",
    "create_extractor",
    "InvalidSymbolError",
    # Graph Projection
    "CodeNode",
    "CodeNodeType",
    "CodeEdge",
    "CodeEdgeType",
    "GraphProjectionConfig",
    "FileNodeBuilder",
    "SymbolNodeBuilder",
    "PackageNodeBuilder",
    "SchemaFieldNodeBuilder",
    "OpenAPIDefNodeBuilder",
    "EdgeBuilder",
    "BatchOperation",
    "BatchOperationType",
    "BatchResult",
    "ProjectionTransaction",
    "TransactionState",
    "Neo4jDriver",
    "MemoryGraphStore",
    "GraphProjection",
    "GraphProjectionError",
    "ConstraintViolationError",
    "TransactionError",
    "create_graph_projection",
    # Neo4j-backed code graph driver
    "CodeGraphDriver",
    "create_code_graph_driver",
    # Code indexing service
    "CodeIndexingService",
    "CodeIndexSummary",
    "FileIndexStats",
    # Bootstrap
    "FilePriority",
    "PriorityTier",
    "IndexingPhase",
    "BootstrapProgress",
    "PriorityQueue",
    "FilePriorityScorer",
    "BootstrapState",
    "BootstrapStateStore",
    "MemoryBootstrapStateStore",
    "FileBootstrapStateStore",
    "BootstrapStatus",
    "BootstrapStatusResponse",
    "FileProgress",
    "BootstrapConfig",
    "BootstrapManager",
    "ProgressCallback",
    "FileCompleteCallback",
    "PhaseChangeCallback",
    "BootstrapScanResult",
    "IndexBatchResult",
    "BootstrapError",
    "StateNotFoundError",
    "create_bootstrap_manager",
    # API Boundaries
    "HTTPMethod",
    "APIEndpoint",
    "APIClient",
    "APIBoundaryLink",
    "APIBoundaryAnalysisResult",
    "APIBoundaryEdgeType",
    "APIBoundaryNodeType",
    "APIBoundaryNode",
    "APIBoundaryEdge",
    "APIEndpointNodeBuilder",
    "APIClientNodeBuilder",
    "APIBoundaryEdgeBuilder",
    "RestEndpointDetector",
    "APIClientDetector",
    "PathMatcher",
    "APIBoundaryLinker",
    "APIBoundaryProjection",
    "APIBoundaryAnalyzer",
    "create_api_boundary_analyzer",
    "create_rest_endpoint_detector",
    "create_api_client_detector",
]
