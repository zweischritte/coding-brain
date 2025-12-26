"""Unified search across repositories.

This module provides unified search capabilities that allow searching
for code, symbols, and content across multiple repositories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .registry import Repository, RepositoryRegistry
from .symbol_resolution import CrossRepoSymbol, SymbolStore, SymbolType


# =============================================================================
# Exceptions
# =============================================================================


class UnifiedSearchError(Exception):
    """Base exception for unified search errors."""

    pass


# =============================================================================
# Enums
# =============================================================================


class SearchResultType(str, Enum):
    """Type of search result."""

    SYMBOL = "symbol"
    FILE = "file"
    CODE = "code"
    MEMORY = "memory"
    ADR = "adr"


class SearchRanking(str, Enum):
    """Ranking strategy for search results."""

    RELEVANCE = "relevance"
    RECENCY = "recency"
    POPULARITY = "popularity"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SearchFilter:
    """Filter criteria for search."""

    repo_ids: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    symbol_types: list[SymbolType] = field(default_factory=list)
    result_types: list[SearchResultType] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    min_score: float = 0.0


@dataclass
class SearchResultItem:
    """A single search result item."""

    result_type: SearchResultType
    id: str
    repo_id: str
    name: str
    score: float
    file_path: str = ""
    line_number: int = 0
    snippet: str = ""
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_type": self.result_type.value,
            "id": self.id,
            "repo_id": self.repo_id,
            "name": self.name,
            "score": self.score,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class RepoSearchResult:
    """Search results for a single repository."""

    repo_id: str
    items: list[SearchResultItem]
    total_matches: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def max_score(self) -> float:
        """Get the maximum score in this repository's results."""
        if not self.items:
            return 0.0
        return max(item.score for item in self.items)


@dataclass
class UnifiedSearchConfig:
    """Configuration for unified search."""

    max_results: int = 100
    max_results_per_repo: int = 20
    ranking: SearchRanking = SearchRanking.RELEVANCE
    include_snippets: bool = True
    snippet_context_lines: int = 2


@dataclass
class UnifiedSearchInput:
    """Input for unified search."""

    query: str
    filter: SearchFilter = field(default_factory=SearchFilter)
    exact_match: bool = False
    case_sensitive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSearchResult:
    """Result of unified search."""

    query: str
    items: list[SearchResultItem]
    repo_results: list[RepoSearchResult]
    total_matches: int
    search_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        """Check if there are any results."""
        return len(self.items) > 0

    @property
    def top_score(self) -> float:
        """Get the highest score across all results."""
        if not self.items:
            return 0.0
        return max(item.score for item in self.items)


# =============================================================================
# Unified Searcher
# =============================================================================


class UnifiedSearcher:
    """Searcher for unified cross-repository search."""

    def __init__(
        self,
        registry: RepositoryRegistry,
        symbol_store: SymbolStore,
        config: Optional[UnifiedSearchConfig] = None,
    ):
        self._registry = registry
        self._symbol_store = symbol_store
        self._config = config or UnifiedSearchConfig()

    @property
    def config(self) -> UnifiedSearchConfig:
        """Get searcher configuration."""
        return self._config

    def search(self, input_data: UnifiedSearchInput) -> UnifiedSearchResult:
        """Perform unified search across repositories."""
        import time

        start_time = time.time()

        # Get queryable repositories
        repos = self._get_target_repos(input_data.filter)

        # Search each repository
        all_items: list[SearchResultItem] = []
        repo_results: list[RepoSearchResult] = []

        for repo in repos:
            repo_items = self._search_repo(
                repo_id=repo.repo_id,
                query=input_data.query,
                filter=input_data.filter,
                exact_match=input_data.exact_match,
                case_sensitive=input_data.case_sensitive,
            )

            if repo_items:
                all_items.extend(repo_items)
                repo_results.append(
                    RepoSearchResult(
                        repo_id=repo.repo_id,
                        items=repo_items,
                        total_matches=len(repo_items),
                    )
                )

        # Sort by score (descending)
        all_items.sort(key=lambda x: x.score, reverse=True)

        # Apply result limit
        all_items = all_items[: self._config.max_results]

        search_time_ms = (time.time() - start_time) * 1000

        return UnifiedSearchResult(
            query=input_data.query,
            items=all_items,
            repo_results=repo_results,
            total_matches=len(all_items),
            search_time_ms=search_time_ms,
        )

    def get_queryable_repos(self) -> list[Repository]:
        """Get all queryable repositories."""
        return self._registry.list_queryable()

    def _get_target_repos(self, filter: SearchFilter) -> list[Repository]:
        """Get target repositories based on filter."""
        all_repos = self._registry.list_queryable()

        # Filter by repo_ids if specified
        if filter.repo_ids:
            all_repos = [r for r in all_repos if r.repo_id in filter.repo_ids]

        # Filter by language if specified
        if filter.languages:
            all_repos = [
                r
                for r in all_repos
                if any(lang in r.metadata.languages for lang in filter.languages)
            ]

        return all_repos

    def _search_repo(
        self,
        repo_id: str,
        query: str,
        filter: SearchFilter,
        exact_match: bool,
        case_sensitive: bool,
    ) -> list[SearchResultItem]:
        """Search within a single repository."""
        items: list[SearchResultItem] = []

        # Get symbols from this repo
        symbols = self._symbol_store.search_by_repo(repo_id)

        # Filter by symbol type if specified
        if filter.symbol_types:
            symbols = [s for s in symbols if s.symbol_type in filter.symbol_types]

        # Match against query
        for symbol in symbols:
            score = self._calculate_match_score(
                symbol=symbol,
                query=query,
                exact_match=exact_match,
                case_sensitive=case_sensitive,
            )

            if score >= filter.min_score and score > 0:
                items.append(
                    SearchResultItem(
                        result_type=SearchResultType.SYMBOL,
                        id=symbol.symbol_id,
                        repo_id=symbol.repo_id,
                        name=symbol.name,
                        score=score,
                        file_path=symbol.file_path,
                        line_number=symbol.line_number,
                        language=self._detect_language(symbol.file_path),
                    )
                )

        # Sort by score and limit per-repo
        items.sort(key=lambda x: x.score, reverse=True)
        items = items[: self._config.max_results_per_repo]

        return items

    def _calculate_match_score(
        self,
        symbol: CrossRepoSymbol,
        query: str,
        exact_match: bool,
        case_sensitive: bool,
    ) -> float:
        """Calculate match score for a symbol against a query."""
        if not query:
            # Empty query matches everything with base score
            return 0.5

        name = symbol.name
        if not case_sensitive:
            name = name.lower()
            query = query.lower()

        # Exact match
        if exact_match:
            if name == query:
                return 1.0
            return 0.0

        # Partial match scoring
        if query == name:
            return 1.0
        elif name.startswith(query):
            return 0.9
        elif query in name:
            return 0.7
        elif self._fuzzy_match(query, name):
            return 0.5

        return 0.0

    def _fuzzy_match(self, query: str, name: str) -> bool:
        """Simple fuzzy matching."""
        # Check if all characters in query appear in order in name
        query_idx = 0
        for char in name:
            if query_idx < len(query) and char == query[query_idx]:
                query_idx += 1
        return query_idx == len(query)

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file path."""
        if not file_path:
            return ""

        extension_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
        }

        for ext, lang in extension_map.items():
            if file_path.endswith(ext):
                return lang

        return ""
