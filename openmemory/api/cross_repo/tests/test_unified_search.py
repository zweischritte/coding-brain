"""Tests for unified search across repositories.

This module tests unified search capabilities that allow searching
for code, symbols, and content across multiple repositories.
"""

from __future__ import annotations

from typing import Any

import pytest

from openmemory.api.cross_repo.registry import (
    InMemoryRepositoryRegistry,
    RepositoryMetadata,
    RepositoryStatus,
)
from openmemory.api.cross_repo.symbol_resolution import (
    CrossRepoSymbol,
    InMemorySymbolStore,
    SymbolType,
)
from openmemory.api.cross_repo.unified_search import (
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


# =============================================================================
# SearchResultType Tests
# =============================================================================


class TestSearchResultType:
    """Tests for SearchResultType enum."""

    def test_result_types(self):
        """Test all result types exist."""
        assert SearchResultType.SYMBOL.value == "symbol"
        assert SearchResultType.FILE.value == "file"
        assert SearchResultType.CODE.value == "code"
        assert SearchResultType.MEMORY.value == "memory"
        assert SearchResultType.ADR.value == "adr"


# =============================================================================
# SearchRanking Tests
# =============================================================================


class TestSearchRanking:
    """Tests for SearchRanking enum."""

    def test_ranking_types(self):
        """Test all ranking types exist."""
        assert SearchRanking.RELEVANCE.value == "relevance"
        assert SearchRanking.RECENCY.value == "recency"
        assert SearchRanking.POPULARITY.value == "popularity"


# =============================================================================
# SearchFilter Tests
# =============================================================================


class TestSearchFilter:
    """Tests for SearchFilter dataclass."""

    def test_create_filter(self):
        """Test creating a search filter."""
        filter = SearchFilter(
            repo_ids=["org/repo1", "org/repo2"],
            languages=["python", "typescript"],
            symbol_types=[SymbolType.FUNCTION, SymbolType.CLASS],
        )
        assert len(filter.repo_ids) == 2
        assert len(filter.languages) == 2
        assert len(filter.symbol_types) == 2

    def test_filter_defaults(self):
        """Test default values for filter."""
        filter = SearchFilter()
        assert filter.repo_ids == []
        assert filter.languages == []
        assert filter.symbol_types == []
        assert filter.result_types == []
        assert filter.file_patterns == []
        assert filter.exclude_patterns == []
        assert filter.min_score == 0.0

    def test_filter_with_file_patterns(self):
        """Test filter with file patterns."""
        filter = SearchFilter(
            file_patterns=["*.py", "*.ts"],
            exclude_patterns=["*_test.py", "test_*.py"],
        )
        assert len(filter.file_patterns) == 2
        assert len(filter.exclude_patterns) == 2


# =============================================================================
# SearchResultItem Tests
# =============================================================================


class TestSearchResultItem:
    """Tests for SearchResultItem dataclass."""

    def test_create_result_item(self):
        """Test creating a search result item."""
        item = SearchResultItem(
            result_type=SearchResultType.SYMBOL,
            id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            score=0.95,
        )
        assert item.result_type == SearchResultType.SYMBOL
        assert item.id == "scip-python org/repo module/func."
        assert item.score == 0.95

    def test_result_item_defaults(self):
        """Test default values for result item."""
        item = SearchResultItem(
            result_type=SearchResultType.FILE,
            id="/path/to/file.py",
            repo_id="org/repo",
            name="file.py",
            score=0.8,
        )
        assert item.file_path == ""
        assert item.line_number == 0
        assert item.snippet == ""
        assert item.language == ""
        assert item.metadata == {}

    def test_result_item_with_snippet(self):
        """Test result item with code snippet."""
        item = SearchResultItem(
            result_type=SearchResultType.CODE,
            id="code-match-1",
            repo_id="org/repo",
            name="process_data",
            score=0.9,
            file_path="src/utils.py",
            line_number=42,
            snippet="def process_data(input: str) -> list:\n    ...",
            language="python",
        )
        assert "def process_data" in item.snippet
        assert item.line_number == 42
        assert item.language == "python"

    def test_result_item_to_dict(self):
        """Test converting result item to dictionary."""
        item = SearchResultItem(
            result_type=SearchResultType.SYMBOL,
            id="sym1",
            repo_id="org/repo",
            name="MyClass",
            score=0.85,
        )
        data = item.to_dict()
        assert data["result_type"] == "symbol"
        assert data["id"] == "sym1"
        assert data["score"] == 0.85


# =============================================================================
# RepoSearchResult Tests
# =============================================================================


class TestRepoSearchResult:
    """Tests for RepoSearchResult dataclass."""

    def test_create_repo_result(self):
        """Test creating a repository search result."""
        items = [
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym1",
                repo_id="org/repo",
                name="func1",
                score=0.9,
            ),
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym2",
                repo_id="org/repo",
                name="func2",
                score=0.8,
            ),
        ]
        result = RepoSearchResult(
            repo_id="org/repo",
            items=items,
            total_matches=2,
        )
        assert result.repo_id == "org/repo"
        assert len(result.items) == 2
        assert result.total_matches == 2

    def test_repo_result_max_score(self):
        """Test max_score property."""
        items = [
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym1",
                repo_id="org/repo",
                name="func1",
                score=0.7,
            ),
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym2",
                repo_id="org/repo",
                name="func2",
                score=0.9,
            ),
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym3",
                repo_id="org/repo",
                name="func3",
                score=0.8,
            ),
        ]
        result = RepoSearchResult(
            repo_id="org/repo",
            items=items,
            total_matches=3,
        )
        assert result.max_score == 0.9


# =============================================================================
# UnifiedSearchConfig Tests
# =============================================================================


class TestUnifiedSearchConfig:
    """Tests for UnifiedSearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedSearchConfig()
        assert config.max_results == 100
        assert config.max_results_per_repo == 20
        assert config.ranking == SearchRanking.RELEVANCE
        assert config.include_snippets is True
        assert config.snippet_context_lines == 2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedSearchConfig(
            max_results=50,
            max_results_per_repo=10,
            ranking=SearchRanking.RECENCY,
            include_snippets=False,
            snippet_context_lines=5,
        )
        assert config.max_results == 50
        assert config.max_results_per_repo == 10
        assert config.ranking == SearchRanking.RECENCY
        assert config.include_snippets is False
        assert config.snippet_context_lines == 5


# =============================================================================
# UnifiedSearchInput Tests
# =============================================================================


class TestUnifiedSearchInput:
    """Tests for UnifiedSearchInput dataclass."""

    def test_create_input(self):
        """Test creating search input."""
        input_data = UnifiedSearchInput(
            query="process data function",
        )
        assert input_data.query == "process data function"

    def test_input_with_filter(self):
        """Test input with filter."""
        filter = SearchFilter(
            repo_ids=["org/repo1"],
            languages=["python"],
        )
        input_data = UnifiedSearchInput(
            query="process",
            filter=filter,
        )
        assert len(input_data.filter.repo_ids) == 1
        assert input_data.filter.languages == ["python"]

    def test_input_with_options(self):
        """Test input with search options."""
        input_data = UnifiedSearchInput(
            query="class User",
            exact_match=True,
            case_sensitive=True,
        )
        assert input_data.exact_match is True
        assert input_data.case_sensitive is True


# =============================================================================
# UnifiedSearchResult Tests
# =============================================================================


class TestUnifiedSearchResult:
    """Tests for UnifiedSearchResult dataclass."""

    def test_create_result(self):
        """Test creating a search result."""
        result = UnifiedSearchResult(
            query="test",
            items=[],
            repo_results=[],
            total_matches=0,
        )
        assert result.query == "test"
        assert len(result.items) == 0
        assert result.total_matches == 0

    def test_result_with_items(self):
        """Test result with items."""
        items = [
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym1",
                repo_id="org/repo1",
                name="func1",
                score=0.9,
            ),
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym2",
                repo_id="org/repo2",
                name="func2",
                score=0.8,
            ),
        ]
        result = UnifiedSearchResult(
            query="func",
            items=items,
            repo_results=[],
            total_matches=2,
        )
        assert len(result.items) == 2
        assert result.total_matches == 2

    def test_result_has_results(self):
        """Test has_results property."""
        empty_result = UnifiedSearchResult(
            query="nothing",
            items=[],
            repo_results=[],
            total_matches=0,
        )
        assert empty_result.has_results is False

        with_results = UnifiedSearchResult(
            query="something",
            items=[
                SearchResultItem(
                    result_type=SearchResultType.SYMBOL,
                    id="sym1",
                    repo_id="repo",
                    name="func",
                    score=0.5,
                )
            ],
            repo_results=[],
            total_matches=1,
        )
        assert with_results.has_results is True

    def test_result_top_score(self):
        """Test top_score property."""
        items = [
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym1",
                repo_id="repo",
                name="func1",
                score=0.7,
            ),
            SearchResultItem(
                result_type=SearchResultType.SYMBOL,
                id="sym2",
                repo_id="repo",
                name="func2",
                score=0.95,
            ),
        ]
        result = UnifiedSearchResult(
            query="func",
            items=items,
            repo_results=[],
            total_matches=2,
        )
        assert result.top_score == 0.95


# =============================================================================
# UnifiedSearcher Tests
# =============================================================================


class TestUnifiedSearcher:
    """Tests for UnifiedSearcher."""

    @pytest.fixture
    def registry(self) -> InMemoryRepositoryRegistry:
        """Create a registry with test repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/backend",
            name="backend",
            metadata=RepositoryMetadata(
                languages=["python"],
                tags=["api", "backend"],
            ),
        )
        registry.update_status("org/backend", RepositoryStatus.ACTIVE)

        registry.register(
            repo_id="org/frontend",
            name="frontend",
            metadata=RepositoryMetadata(
                languages=["typescript"],
                tags=["web", "frontend"],
            ),
        )
        registry.update_status("org/frontend", RepositoryStatus.ACTIVE)

        registry.register(
            repo_id="org/shared-lib",
            name="shared-lib",
            metadata=RepositoryMetadata(
                languages=["python", "typescript"],
                tags=["shared", "library"],
            ),
        )
        registry.update_status("org/shared-lib", RepositoryStatus.ACTIVE)

        return registry

    @pytest.fixture
    def symbol_store(self) -> InMemorySymbolStore:
        """Create a symbol store with test symbols."""
        store = InMemorySymbolStore()

        # Backend symbols
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/backend handlers/process_request.",
                repo_id="org/backend",
                name="process_request",
                symbol_type=SymbolType.FUNCTION,
                file_path="handlers/api.py",
                line_number=10,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/backend models/UserModel#",
                repo_id="org/backend",
                name="UserModel",
                symbol_type=SymbolType.CLASS,
                file_path="models/user.py",
                line_number=5,
            )
        )

        # Frontend symbols
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-typescript org/frontend components/UserCard#",
                repo_id="org/frontend",
                name="UserCard",
                symbol_type=SymbolType.CLASS,
                file_path="components/UserCard.tsx",
                line_number=15,
            )
        )

        # Shared lib symbols
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/shared-lib utils/validate.",
                repo_id="org/shared-lib",
                name="validate",
                symbol_type=SymbolType.FUNCTION,
                file_path="utils/validation.py",
                line_number=1,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/shared-lib utils/process_data.",
                repo_id="org/shared-lib",
                name="process_data",
                symbol_type=SymbolType.FUNCTION,
                file_path="utils/processing.py",
                line_number=20,
            )
        )

        return store

    @pytest.fixture
    def searcher(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
    ) -> UnifiedSearcher:
        """Create a unified searcher."""
        return UnifiedSearcher(
            registry=registry,
            symbol_store=symbol_store,
        )

    def test_create_searcher(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
    ):
        """Test creating a searcher."""
        searcher = UnifiedSearcher(
            registry=registry,
            symbol_store=symbol_store,
        )
        assert searcher is not None

    def test_create_searcher_with_config(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
    ):
        """Test creating a searcher with custom config."""
        config = UnifiedSearchConfig(max_results=50)
        searcher = UnifiedSearcher(
            registry=registry,
            symbol_store=symbol_store,
            config=config,
        )
        assert searcher.config.max_results == 50

    def test_search_by_name(self, searcher: UnifiedSearcher):
        """Test searching symbols by name."""
        input_data = UnifiedSearchInput(query="User")

        result = searcher.search(input_data)

        assert result.has_results is True
        # Should find UserModel and UserCard
        names = [item.name for item in result.items]
        assert "UserModel" in names or "UserCard" in names

    def test_search_with_repo_filter(self, searcher: UnifiedSearcher):
        """Test searching with repository filter."""
        input_data = UnifiedSearchInput(
            query="User",
            filter=SearchFilter(repo_ids=["org/backend"]),
        )

        result = searcher.search(input_data)

        # Should only find results from backend
        for item in result.items:
            assert item.repo_id == "org/backend"

    def test_search_with_language_filter(self, searcher: UnifiedSearcher):
        """Test searching with language filter."""
        input_data = UnifiedSearchInput(
            query="process",
            filter=SearchFilter(languages=["python"]),
        )

        result = searcher.search(input_data)

        # Should only find Python results
        assert result.has_results is True
        for item in result.items:
            assert item.repo_id in ["org/backend", "org/shared-lib"]

    def test_search_with_symbol_type_filter(self, searcher: UnifiedSearcher):
        """Test searching with symbol type filter."""
        input_data = UnifiedSearchInput(
            query="",  # Empty query to match all
            filter=SearchFilter(symbol_types=[SymbolType.CLASS]),
        )

        result = searcher.search(input_data)

        # Should only find classes
        for item in result.items:
            assert item.result_type == SearchResultType.SYMBOL

    def test_search_no_results(self, searcher: UnifiedSearcher):
        """Test searching with no matching results."""
        input_data = UnifiedSearchInput(query="nonexistent_symbol_xyz")

        result = searcher.search(input_data)

        assert result.has_results is False
        assert result.total_matches == 0

    def test_search_limit_results(self, searcher: UnifiedSearcher):
        """Test limiting search results."""
        searcher._config.max_results = 2

        input_data = UnifiedSearchInput(query="")  # Match all

        result = searcher.search(input_data)

        assert len(result.items) <= 2

    def test_search_grouped_by_repo(self, searcher: UnifiedSearcher):
        """Test search results grouped by repository."""
        input_data = UnifiedSearchInput(query="")  # Match all

        result = searcher.search(input_data)

        # Check repo_results
        assert len(result.repo_results) > 0
        for repo_result in result.repo_results:
            assert repo_result.repo_id in ["org/backend", "org/frontend", "org/shared-lib"]
            assert len(repo_result.items) > 0

    def test_search_exact_match(self, searcher: UnifiedSearcher):
        """Test exact match search."""
        input_data = UnifiedSearchInput(
            query="process_request",
            exact_match=True,
        )

        result = searcher.search(input_data)

        # Should find exact match
        if result.has_results:
            assert any(item.name == "process_request" for item in result.items)

    def test_search_case_sensitive(self, searcher: UnifiedSearcher):
        """Test case-sensitive search."""
        # Case-sensitive search for "usermodel" should not find "UserModel"
        input_data = UnifiedSearchInput(
            query="usermodel",
            case_sensitive=True,
        )

        result = searcher.search(input_data)

        # Should not find UserModel due to case mismatch
        assert not any(item.name == "UserModel" for item in result.items)

    def test_search_across_all_repos(self, searcher: UnifiedSearcher):
        """Test searching across all repositories."""
        input_data = UnifiedSearchInput(query="")  # Match all

        result = searcher.search(input_data)

        # Should have results from multiple repos
        repo_ids = set(item.repo_id for item in result.items)
        assert len(repo_ids) > 1

    def test_search_returns_scores(self, searcher: UnifiedSearcher):
        """Test that search returns relevance scores."""
        input_data = UnifiedSearchInput(query="process")

        result = searcher.search(input_data)

        if result.has_results:
            for item in result.items:
                assert 0.0 <= item.score <= 1.0

    def test_search_ordered_by_score(self, searcher: UnifiedSearcher):
        """Test that results are ordered by score."""
        input_data = UnifiedSearchInput(query="process")

        result = searcher.search(input_data)

        if len(result.items) > 1:
            scores = [item.score for item in result.items]
            assert scores == sorted(scores, reverse=True)

    def test_get_queryable_repos(self, searcher: UnifiedSearcher):
        """Test getting list of queryable repositories."""
        repos = searcher.get_queryable_repos()

        assert len(repos) == 3
        repo_ids = [r.repo_id for r in repos]
        assert "org/backend" in repo_ids
        assert "org/frontend" in repo_ids
        assert "org/shared-lib" in repo_ids


# =============================================================================
# Exception Tests
# =============================================================================


class TestUnifiedSearchExceptions:
    """Tests for unified search exceptions."""

    def test_unified_search_error(self):
        """Test base unified search error."""
        error = UnifiedSearchError("Test error")
        assert str(error) == "Test error"
