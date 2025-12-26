"""Phase 8: Cross-Repository Intelligence.

This module implements:
- Repository registry and discovery
- Cross-repository symbol resolution
- Dependency graph across repositories
- Cross-repository impact analysis
- Unified search across repositories

Per implementation plan v9 section 6.6 (Multi-Repository Graph):
- New nodes: CODE_Repository, CODE_APISpec
- New edges: CODE_DEPENDS_ON, CODE_PUBLISHES_API
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # Registry
    "Repository",
    "RepositoryMetadata",
    "RepositoryStatus",
    "RepositoryConfig",
    "RepositoryRegistry",
    "InMemoryRepositoryRegistry",
    "RepositoryRegistryError",
    "RepositoryNotFoundError",
    "RepositoryAlreadyExistsError",
    # Symbol resolution
    "CrossRepoSymbol",
    "SymbolMapping",
    "SymbolType",
    "SymbolResolutionConfig",
    "SymbolResolutionResult",
    "SymbolStore",
    "InMemorySymbolStore",
    "CrossRepoSymbolResolver",
    "SymbolResolutionError",
    "SymbolNotFoundError",
    "AmbiguousSymbolError",
    # Dependency graph
    "RepoDependency",
    "DependencyType",
    "DependencyEdge",
    "DependencyGraphConfig",
    "DependencyGraphStore",
    "InMemoryDependencyGraphStore",
    "RepoDependencyGraph",
    "DependencyGraphError",
    "CyclicDependencyError",
    # Impact analysis
    "CrossRepoImpactConfig",
    "CrossRepoImpactInput",
    "CrossRepoImpactOutput",
    "AffectedRepository",
    "BreakingChange",
    "BreakingChangeType",
    "ChangeSeverity",
    "ChangeType",
    "SymbolChange",
    "CrossRepoImpactAnalyzer",
    "CrossRepoImpactError",
    # Unified search
    "UnifiedSearchConfig",
    "UnifiedSearchInput",
    "UnifiedSearchResult",
    "SearchFilter",
    "SearchResultItem",
    "SearchResultType",
    "SearchRanking",
    "RepoSearchResult",
    "UnifiedSearcher",
    "UnifiedSearchError",
]


def __getattr__(name: str):
    """Lazy import module components."""
    # Registry components
    if name in (
        "Repository",
        "RepositoryMetadata",
        "RepositoryStatus",
        "RepositoryConfig",
        "RepositoryRegistry",
        "InMemoryRepositoryRegistry",
        "RepositoryRegistryError",
        "RepositoryNotFoundError",
        "RepositoryAlreadyExistsError",
    ):
        from .registry import (
            InMemoryRepositoryRegistry,
            Repository,
            RepositoryAlreadyExistsError,
            RepositoryConfig,
            RepositoryMetadata,
            RepositoryNotFoundError,
            RepositoryRegistry,
            RepositoryRegistryError,
            RepositoryStatus,
        )

        return locals()[name]

    # Symbol resolution components
    if name in (
        "CrossRepoSymbol",
        "SymbolMapping",
        "SymbolType",
        "SymbolResolutionConfig",
        "SymbolResolutionResult",
        "SymbolStore",
        "InMemorySymbolStore",
        "CrossRepoSymbolResolver",
        "SymbolResolutionError",
        "SymbolNotFoundError",
        "AmbiguousSymbolError",
    ):
        from .symbol_resolution import (
            AmbiguousSymbolError,
            CrossRepoSymbol,
            CrossRepoSymbolResolver,
            InMemorySymbolStore,
            SymbolMapping,
            SymbolNotFoundError,
            SymbolResolutionConfig,
            SymbolResolutionError,
            SymbolResolutionResult,
            SymbolStore,
            SymbolType,
        )

        return locals()[name]

    # Dependency graph components
    if name in (
        "RepoDependency",
        "DependencyType",
        "DependencyEdge",
        "DependencyGraphConfig",
        "DependencyGraphStore",
        "InMemoryDependencyGraphStore",
        "RepoDependencyGraph",
        "DependencyGraphError",
        "CyclicDependencyError",
    ):
        from .dependency_graph import (
            CyclicDependencyError,
            DependencyEdge,
            DependencyGraphConfig,
            DependencyGraphError,
            DependencyGraphStore,
            DependencyType,
            InMemoryDependencyGraphStore,
            RepoDependency,
            RepoDependencyGraph,
        )

        return locals()[name]

    # Impact analysis components
    if name in (
        "CrossRepoImpactConfig",
        "CrossRepoImpactInput",
        "CrossRepoImpactOutput",
        "AffectedRepository",
        "BreakingChange",
        "BreakingChangeType",
        "ChangeSeverity",
        "ChangeType",
        "SymbolChange",
        "CrossRepoImpactAnalyzer",
        "CrossRepoImpactError",
    ):
        from .impact_analysis import (
            AffectedRepository,
            BreakingChange,
            BreakingChangeType,
            ChangeSeverity,
            ChangeType,
            CrossRepoImpactAnalyzer,
            CrossRepoImpactConfig,
            CrossRepoImpactError,
            CrossRepoImpactInput,
            CrossRepoImpactOutput,
            SymbolChange,
        )

        return locals()[name]

    # Unified search components
    if name in (
        "UnifiedSearchConfig",
        "UnifiedSearchInput",
        "UnifiedSearchResult",
        "SearchFilter",
        "SearchResultItem",
        "SearchResultType",
        "SearchRanking",
        "RepoSearchResult",
        "UnifiedSearcher",
        "UnifiedSearchError",
    ):
        from .unified_search import (
            RepoSearchResult,
            SearchFilter,
            SearchRanking,
            SearchResultItem,
            SearchResultType,
            UnifiedSearchConfig,
            UnifiedSearchError,
            UnifiedSearcher,
            UnifiedSearchInput,
            UnifiedSearchResult,
        )

        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
