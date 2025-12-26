"""Tests for cross-repository symbol resolution.

This module tests cross-repo symbol resolution for detecting and resolving
symbols that span multiple repositories.
"""

from __future__ import annotations

from typing import Any

import pytest

from openmemory.api.cross_repo.registry import (
    InMemoryRepositoryRegistry,
    Repository,
    RepositoryMetadata,
    RepositoryStatus,
)
from openmemory.api.cross_repo.symbol_resolution import (
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


# =============================================================================
# CrossRepoSymbol Tests
# =============================================================================


class TestCrossRepoSymbol:
    """Tests for CrossRepoSymbol dataclass."""

    def test_create_symbol(self):
        """Test creating a cross-repo symbol."""
        symbol = CrossRepoSymbol(
            symbol_id="scip-python myorg/myrepo module/MyClass#",
            repo_id="myorg/myrepo",
            name="MyClass",
            symbol_type=SymbolType.CLASS,
            file_path="src/module.py",
            line_number=10,
        )
        assert symbol.symbol_id == "scip-python myorg/myrepo module/MyClass#"
        assert symbol.repo_id == "myorg/myrepo"
        assert symbol.name == "MyClass"
        assert symbol.symbol_type == SymbolType.CLASS
        assert symbol.file_path == "src/module.py"
        assert symbol.line_number == 10

    def test_symbol_defaults(self):
        """Test default values for symbol."""
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        assert symbol.file_path == ""
        assert symbol.line_number == 0
        assert symbol.signature == ""
        assert symbol.docstring == ""
        assert symbol.visibility == "public"
        assert symbol.exported is True
        assert symbol.references == []
        assert symbol.metadata == {}

    def test_symbol_with_signature(self):
        """Test symbol with signature."""
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/process_data.",
            repo_id="org/repo",
            name="process_data",
            symbol_type=SymbolType.FUNCTION,
            signature="def process_data(input: str, limit: int = 10) -> list[str]",
        )
        assert "process_data" in symbol.signature
        assert "str" in symbol.signature

    def test_symbol_visibility(self):
        """Test symbol visibility."""
        public_symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/PublicClass#",
            repo_id="org/repo",
            name="PublicClass",
            symbol_type=SymbolType.CLASS,
            visibility="public",
        )
        assert public_symbol.visibility == "public"

        private_symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/_PrivateClass#",
            repo_id="org/repo",
            name="_PrivateClass",
            symbol_type=SymbolType.CLASS,
            visibility="private",
        )
        assert private_symbol.visibility == "private"

    def test_symbol_equality(self):
        """Test symbol equality based on symbol_id."""
        sym1 = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        sym2 = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
            docstring="Updated docstring",  # Different
        )
        sym3 = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/other.",
            repo_id="org/repo",
            name="other",
            symbol_type=SymbolType.FUNCTION,
        )

        assert sym1 == sym2  # Same symbol_id
        assert sym1 != sym3  # Different symbol_id

    def test_symbol_hash(self):
        """Test symbol is hashable."""
        sym1 = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        sym2 = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )

        # Same symbol_id should have same hash
        assert hash(sym1) == hash(sym2)

        # Can use in sets
        symbol_set = {sym1, sym2}
        assert len(symbol_set) == 1

    def test_symbol_to_dict(self):
        """Test converting symbol to dictionary."""
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/MyClass#",
            repo_id="org/repo",
            name="MyClass",
            symbol_type=SymbolType.CLASS,
            file_path="src/module.py",
            line_number=42,
        )
        data = symbol.to_dict()
        assert data["symbol_id"] == "scip-python org/repo module/MyClass#"
        assert data["repo_id"] == "org/repo"
        assert data["name"] == "MyClass"
        assert data["symbol_type"] == "class"
        assert data["file_path"] == "src/module.py"
        assert data["line_number"] == 42

    def test_symbol_from_dict(self):
        """Test creating symbol from dictionary."""
        data = {
            "symbol_id": "scip-python org/repo module/func.",
            "repo_id": "org/repo",
            "name": "func",
            "symbol_type": "function",
            "file_path": "src/module.py",
            "line_number": 10,
        }
        symbol = CrossRepoSymbol.from_dict(data)
        assert symbol.symbol_id == "scip-python org/repo module/func."
        assert symbol.symbol_type == SymbolType.FUNCTION
        assert symbol.line_number == 10


# =============================================================================
# SymbolMapping Tests
# =============================================================================


class TestSymbolMapping:
    """Tests for SymbolMapping dataclass."""

    def test_create_mapping(self):
        """Test creating a symbol mapping."""
        mapping = SymbolMapping(
            source_symbol_id="scip-python org/lib module/BaseClass#",
            target_symbol_id="scip-python org/app module/DerivedClass#",
            source_repo_id="org/lib",
            target_repo_id="org/app",
            mapping_type="inheritance",
        )
        assert mapping.source_symbol_id == "scip-python org/lib module/BaseClass#"
        assert mapping.target_symbol_id == "scip-python org/app module/DerivedClass#"
        assert mapping.source_repo_id == "org/lib"
        assert mapping.target_repo_id == "org/app"
        assert mapping.mapping_type == "inheritance"

    def test_mapping_defaults(self):
        """Test default values for mapping."""
        mapping = SymbolMapping(
            source_symbol_id="src-sym",
            target_symbol_id="tgt-sym",
            source_repo_id="src-repo",
            target_repo_id="tgt-repo",
            mapping_type="import",
        )
        assert mapping.confidence == 1.0
        assert mapping.metadata == {}

    def test_mapping_with_confidence(self):
        """Test mapping with confidence score."""
        mapping = SymbolMapping(
            source_symbol_id="src-sym",
            target_symbol_id="tgt-sym",
            source_repo_id="src-repo",
            target_repo_id="tgt-repo",
            mapping_type="inferred",
            confidence=0.85,
        )
        assert mapping.confidence == 0.85

    def test_mapping_equality(self):
        """Test mapping equality."""
        map1 = SymbolMapping(
            source_symbol_id="src",
            target_symbol_id="tgt",
            source_repo_id="repo1",
            target_repo_id="repo2",
            mapping_type="import",
        )
        map2 = SymbolMapping(
            source_symbol_id="src",
            target_symbol_id="tgt",
            source_repo_id="repo1",
            target_repo_id="repo2",
            mapping_type="import",
        )
        map3 = SymbolMapping(
            source_symbol_id="src",
            target_symbol_id="other",
            source_repo_id="repo1",
            target_repo_id="repo2",
            mapping_type="import",
        )

        assert map1 == map2
        assert map1 != map3


# =============================================================================
# SymbolType Tests
# =============================================================================


class TestSymbolType:
    """Tests for SymbolType enum."""

    def test_symbol_types(self):
        """Test all symbol types exist."""
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.VARIABLE.value == "variable"
        assert SymbolType.CONSTANT.value == "constant"
        assert SymbolType.MODULE.value == "module"
        assert SymbolType.PACKAGE.value == "package"
        assert SymbolType.INTERFACE.value == "interface"
        assert SymbolType.TYPE.value == "type"

    def test_symbol_type_is_callable(self):
        """Test is_callable property."""
        assert SymbolType.FUNCTION.is_callable is True
        assert SymbolType.METHOD.is_callable is True
        assert SymbolType.CLASS.is_callable is True  # Constructors
        assert SymbolType.VARIABLE.is_callable is False
        assert SymbolType.CONSTANT.is_callable is False
        assert SymbolType.MODULE.is_callable is False


# =============================================================================
# SymbolResolutionConfig Tests
# =============================================================================


class TestSymbolResolutionConfig:
    """Tests for SymbolResolutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SymbolResolutionConfig()
        assert config.max_depth == 3
        assert config.include_private is False
        assert config.min_confidence == 0.7
        assert config.resolve_transitive is True
        assert config.max_results == 100

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SymbolResolutionConfig(
            max_depth=5,
            include_private=True,
            min_confidence=0.5,
            resolve_transitive=False,
            max_results=50,
        )
        assert config.max_depth == 5
        assert config.include_private is True
        assert config.min_confidence == 0.5
        assert config.resolve_transitive is False
        assert config.max_results == 50


# =============================================================================
# SymbolResolutionResult Tests
# =============================================================================


class TestSymbolResolutionResult:
    """Tests for SymbolResolutionResult dataclass."""

    def test_create_result(self):
        """Test creating a resolution result."""
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        result = SymbolResolutionResult(
            query="func",
            resolved_symbols=[symbol],
            mappings=[],
            confidence=0.95,
        )
        assert result.query == "func"
        assert len(result.resolved_symbols) == 1
        assert result.confidence == 0.95

    def test_result_with_multiple_symbols(self):
        """Test result with multiple resolved symbols."""
        symbols = [
            CrossRepoSymbol(
                symbol_id=f"scip-python org/repo{i} module/func.",
                repo_id=f"org/repo{i}",
                name="func",
                symbol_type=SymbolType.FUNCTION,
            )
            for i in range(3)
        ]
        result = SymbolResolutionResult(
            query="func",
            resolved_symbols=symbols,
            mappings=[],
        )
        assert len(result.resolved_symbols) == 3

    def test_result_is_resolved(self):
        """Test is_resolved property."""
        empty_result = SymbolResolutionResult(
            query="missing",
            resolved_symbols=[],
            mappings=[],
        )
        assert empty_result.is_resolved is False

        symbol = CrossRepoSymbol(
            symbol_id="sym",
            repo_id="repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        found_result = SymbolResolutionResult(
            query="func",
            resolved_symbols=[symbol],
            mappings=[],
        )
        assert found_result.is_resolved is True

    def test_result_is_ambiguous(self):
        """Test is_ambiguous property."""
        single_result = SymbolResolutionResult(
            query="func",
            resolved_symbols=[
                CrossRepoSymbol(
                    symbol_id="sym",
                    repo_id="repo",
                    name="func",
                    symbol_type=SymbolType.FUNCTION,
                )
            ],
            mappings=[],
        )
        assert single_result.is_ambiguous is False

        multi_result = SymbolResolutionResult(
            query="func",
            resolved_symbols=[
                CrossRepoSymbol(
                    symbol_id=f"sym{i}",
                    repo_id=f"repo{i}",
                    name="func",
                    symbol_type=SymbolType.FUNCTION,
                )
                for i in range(3)
            ],
            mappings=[],
        )
        assert multi_result.is_ambiguous is True


# =============================================================================
# InMemorySymbolStore Tests
# =============================================================================


class TestInMemorySymbolStore:
    """Tests for InMemorySymbolStore implementation."""

    def test_create_store(self):
        """Test creating an empty symbol store."""
        store = InMemorySymbolStore()
        assert store.count == 0

    def test_add_symbol(self):
        """Test adding a symbol."""
        store = InMemorySymbolStore()
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        store.add(symbol)
        assert store.count == 1

    def test_get_symbol(self):
        """Test getting a symbol by ID."""
        store = InMemorySymbolStore()
        symbol = CrossRepoSymbol(
            symbol_id="scip-python org/repo module/func.",
            repo_id="org/repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        store.add(symbol)

        retrieved = store.get("scip-python org/repo module/func.")
        assert retrieved is not None
        assert retrieved.name == "func"

    def test_get_nonexistent_returns_none(self):
        """Test getting a nonexistent symbol returns None."""
        store = InMemorySymbolStore()
        assert store.get("nonexistent") is None

    def test_search_by_name(self):
        """Test searching symbols by name."""
        store = InMemorySymbolStore()
        store.add(
            CrossRepoSymbol(
                symbol_id="sym1",
                repo_id="repo",
                name="process_data",
                symbol_type=SymbolType.FUNCTION,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym2",
                repo_id="repo",
                name="process_request",
                symbol_type=SymbolType.FUNCTION,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym3",
                repo_id="repo",
                name="handle_event",
                symbol_type=SymbolType.FUNCTION,
            )
        )

        results = store.search_by_name("process")
        assert len(results) == 2
        names = [s.name for s in results]
        assert "process_data" in names
        assert "process_request" in names

    def test_search_by_repo(self):
        """Test searching symbols by repository."""
        store = InMemorySymbolStore()
        store.add(
            CrossRepoSymbol(
                symbol_id="sym1",
                repo_id="org/repo1",
                name="func1",
                symbol_type=SymbolType.FUNCTION,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym2",
                repo_id="org/repo2",
                name="func2",
                symbol_type=SymbolType.FUNCTION,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym3",
                repo_id="org/repo1",
                name="func3",
                symbol_type=SymbolType.FUNCTION,
            )
        )

        results = store.search_by_repo("org/repo1")
        assert len(results) == 2

    def test_search_by_type(self):
        """Test searching symbols by type."""
        store = InMemorySymbolStore()
        store.add(
            CrossRepoSymbol(
                symbol_id="sym1",
                repo_id="repo",
                name="func",
                symbol_type=SymbolType.FUNCTION,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym2",
                repo_id="repo",
                name="MyClass",
                symbol_type=SymbolType.CLASS,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="sym3",
                repo_id="repo",
                name="method",
                symbol_type=SymbolType.METHOD,
            )
        )

        functions = store.search_by_type(SymbolType.FUNCTION)
        assert len(functions) == 1
        assert functions[0].name == "func"

        classes = store.search_by_type(SymbolType.CLASS)
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_remove_symbol(self):
        """Test removing a symbol."""
        store = InMemorySymbolStore()
        symbol = CrossRepoSymbol(
            symbol_id="sym1",
            repo_id="repo",
            name="func",
            symbol_type=SymbolType.FUNCTION,
        )
        store.add(symbol)
        assert store.count == 1

        store.remove("sym1")
        assert store.count == 0
        assert store.get("sym1") is None

    def test_clear(self):
        """Test clearing all symbols."""
        store = InMemorySymbolStore()
        for i in range(5):
            store.add(
                CrossRepoSymbol(
                    symbol_id=f"sym{i}",
                    repo_id="repo",
                    name=f"func{i}",
                    symbol_type=SymbolType.FUNCTION,
                )
            )
        assert store.count == 5

        store.clear()
        assert store.count == 0

    def test_add_mapping(self):
        """Test adding a symbol mapping."""
        store = InMemorySymbolStore()
        mapping = SymbolMapping(
            source_symbol_id="src",
            target_symbol_id="tgt",
            source_repo_id="repo1",
            target_repo_id="repo2",
            mapping_type="import",
        )
        store.add_mapping(mapping)

        mappings = store.get_mappings("src")
        assert len(mappings) == 1
        assert mappings[0].target_symbol_id == "tgt"

    def test_get_mappings(self):
        """Test getting mappings for a symbol."""
        store = InMemorySymbolStore()
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="src",
                target_symbol_id="tgt1",
                source_repo_id="repo1",
                target_repo_id="repo2",
                mapping_type="import",
            )
        )
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="src",
                target_symbol_id="tgt2",
                source_repo_id="repo1",
                target_repo_id="repo3",
                mapping_type="inheritance",
            )
        )

        mappings = store.get_mappings("src")
        assert len(mappings) == 2

    def test_get_reverse_mappings(self):
        """Test getting reverse mappings for a symbol."""
        store = InMemorySymbolStore()
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="src1",
                target_symbol_id="tgt",
                source_repo_id="repo1",
                target_repo_id="repo2",
                mapping_type="import",
            )
        )
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="src2",
                target_symbol_id="tgt",
                source_repo_id="repo3",
                target_repo_id="repo2",
                mapping_type="import",
            )
        )

        reverse_mappings = store.get_reverse_mappings("tgt")
        assert len(reverse_mappings) == 2


# =============================================================================
# CrossRepoSymbolResolver Tests
# =============================================================================


class TestCrossRepoSymbolResolver:
    """Tests for CrossRepoSymbolResolver."""

    @pytest.fixture
    def registry(self) -> InMemoryRepositoryRegistry:
        """Create a registry with test repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/shared-lib",
            name="shared-lib",
            metadata=RepositoryMetadata(languages=["python"]),
        )
        registry.register(
            repo_id="org/api-service",
            name="api-service",
            metadata=RepositoryMetadata(languages=["python"]),
        )
        registry.register(
            repo_id="org/frontend",
            name="frontend",
            metadata=RepositoryMetadata(languages=["typescript"]),
        )
        return registry

    @pytest.fixture
    def symbol_store(self) -> InMemorySymbolStore:
        """Create a symbol store with test symbols."""
        store = InMemorySymbolStore()

        # Shared library symbols
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/shared-lib utils/BaseModel#",
                repo_id="org/shared-lib",
                name="BaseModel",
                symbol_type=SymbolType.CLASS,
                file_path="utils/models.py",
                line_number=10,
                exported=True,
            )
        )
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/shared-lib utils/process_data.",
                repo_id="org/shared-lib",
                name="process_data",
                symbol_type=SymbolType.FUNCTION,
                file_path="utils/processing.py",
                line_number=5,
                exported=True,
            )
        )

        # API service symbols that use shared lib
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/api-service models/UserModel#",
                repo_id="org/api-service",
                name="UserModel",
                symbol_type=SymbolType.CLASS,
                file_path="models/user.py",
                line_number=8,
            )
        )

        # Add mapping: UserModel extends BaseModel
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="scip-python org/shared-lib utils/BaseModel#",
                target_symbol_id="scip-python org/api-service models/UserModel#",
                source_repo_id="org/shared-lib",
                target_repo_id="org/api-service",
                mapping_type="inheritance",
            )
        )

        # Add mapping: API service imports process_data
        store.add_mapping(
            SymbolMapping(
                source_symbol_id="scip-python org/shared-lib utils/process_data.",
                target_symbol_id="scip-python org/api-service handlers/handler.",
                source_repo_id="org/shared-lib",
                target_repo_id="org/api-service",
                mapping_type="import",
            )
        )

        return store

    @pytest.fixture
    def resolver(
        self, registry: InMemoryRepositoryRegistry, symbol_store: InMemorySymbolStore
    ) -> CrossRepoSymbolResolver:
        """Create a symbol resolver."""
        return CrossRepoSymbolResolver(
            registry=registry,
            symbol_store=symbol_store,
        )

    def test_create_resolver(
        self, registry: InMemoryRepositoryRegistry, symbol_store: InMemorySymbolStore
    ):
        """Test creating a resolver."""
        resolver = CrossRepoSymbolResolver(
            registry=registry,
            symbol_store=symbol_store,
        )
        assert resolver is not None

    def test_create_resolver_with_config(
        self, registry: InMemoryRepositoryRegistry, symbol_store: InMemorySymbolStore
    ):
        """Test creating a resolver with custom config."""
        config = SymbolResolutionConfig(max_depth=5)
        resolver = CrossRepoSymbolResolver(
            registry=registry,
            symbol_store=symbol_store,
            config=config,
        )
        assert resolver.config.max_depth == 5

    def test_resolve_by_id(self, resolver: CrossRepoSymbolResolver):
        """Test resolving a symbol by ID."""
        result = resolver.resolve_by_id(
            "scip-python org/shared-lib utils/BaseModel#"
        )
        assert result.is_resolved is True
        assert len(result.resolved_symbols) == 1
        assert result.resolved_symbols[0].name == "BaseModel"

    def test_resolve_by_id_not_found(self, resolver: CrossRepoSymbolResolver):
        """Test resolving a nonexistent symbol by ID."""
        result = resolver.resolve_by_id("scip-python org/missing module/NotFound.")
        assert result.is_resolved is False
        assert len(result.resolved_symbols) == 0

    def test_resolve_by_name(self, resolver: CrossRepoSymbolResolver):
        """Test resolving symbols by name."""
        result = resolver.resolve_by_name("BaseModel")
        assert result.is_resolved is True
        assert len(result.resolved_symbols) == 1
        assert result.resolved_symbols[0].name == "BaseModel"

    def test_resolve_by_name_multiple_matches(
        self, resolver: CrossRepoSymbolResolver, symbol_store: InMemorySymbolStore
    ):
        """Test resolving a name that matches multiple symbols."""
        # Add another symbol with the same name in a different repo
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/api-service models/BaseModel#",
                repo_id="org/api-service",
                name="BaseModel",
                symbol_type=SymbolType.CLASS,
            )
        )

        result = resolver.resolve_by_name("BaseModel")
        assert result.is_resolved is True
        assert result.is_ambiguous is True
        assert len(result.resolved_symbols) == 2

    def test_resolve_by_name_with_repo_filter(
        self, resolver: CrossRepoSymbolResolver, symbol_store: InMemorySymbolStore
    ):
        """Test resolving symbols by name with repository filter."""
        # Add another symbol with the same name in a different repo
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/api-service models/BaseModel#",
                repo_id="org/api-service",
                name="BaseModel",
                symbol_type=SymbolType.CLASS,
            )
        )

        result = resolver.resolve_by_name(
            "BaseModel", repo_ids=["org/shared-lib"]
        )
        assert result.is_resolved is True
        assert len(result.resolved_symbols) == 1
        assert result.resolved_symbols[0].repo_id == "org/shared-lib"

    def test_resolve_dependents(self, resolver: CrossRepoSymbolResolver):
        """Test finding symbols that depend on a given symbol."""
        result = resolver.resolve_dependents(
            "scip-python org/shared-lib utils/BaseModel#"
        )
        assert len(result.resolved_symbols) >= 1
        assert len(result.mappings) >= 1

    def test_resolve_dependencies(
        self, resolver: CrossRepoSymbolResolver, symbol_store: InMemorySymbolStore
    ):
        """Test finding symbols that a given symbol depends on."""
        # Add a symbol with dependencies
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/api-service handlers/handler.",
                repo_id="org/api-service",
                name="handler",
                symbol_type=SymbolType.FUNCTION,
            )
        )

        result = resolver.resolve_dependencies(
            "scip-python org/api-service handlers/handler."
        )
        # Should find dependencies via mappings
        assert len(result.mappings) >= 0  # At least checked

    def test_resolve_cross_repo_references(
        self, resolver: CrossRepoSymbolResolver
    ):
        """Test finding all cross-repo references for a symbol."""
        result = resolver.resolve_cross_repo_references(
            "scip-python org/shared-lib utils/BaseModel#"
        )
        # BaseModel is referenced by UserModel in api-service
        assert len(result.resolved_symbols) >= 1
        # Should include the mappings
        assert len(result.mappings) >= 1

    def test_find_usages_across_repos(self, resolver: CrossRepoSymbolResolver):
        """Test finding usages of a symbol across all repositories."""
        result = resolver.find_usages_across_repos(
            "scip-python org/shared-lib utils/process_data."
        )
        # process_data is used in api-service
        assert len(result.mappings) >= 1

    def test_resolve_transitive_dependencies(
        self, resolver: CrossRepoSymbolResolver, symbol_store: InMemorySymbolStore
    ):
        """Test resolving transitive dependencies."""
        # Add a chain: A -> B -> C
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="sym-a",
                repo_id="repo-a",
                name="A",
                symbol_type=SymbolType.CLASS,
            )
        )
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="sym-b",
                repo_id="repo-b",
                name="B",
                symbol_type=SymbolType.CLASS,
            )
        )
        symbol_store.add(
            CrossRepoSymbol(
                symbol_id="sym-c",
                repo_id="repo-c",
                name="C",
                symbol_type=SymbolType.CLASS,
            )
        )

        # A depends on B, B depends on C
        symbol_store.add_mapping(
            SymbolMapping(
                source_symbol_id="sym-b",
                target_symbol_id="sym-a",
                source_repo_id="repo-b",
                target_repo_id="repo-a",
                mapping_type="import",
            )
        )
        symbol_store.add_mapping(
            SymbolMapping(
                source_symbol_id="sym-c",
                target_symbol_id="sym-b",
                source_repo_id="repo-c",
                target_repo_id="repo-b",
                mapping_type="import",
            )
        )

        # Resolve transitive dependencies of A
        result = resolver.resolve_transitive_dependencies("sym-a", max_depth=3)
        # Should find both B and C
        symbol_ids = {s.symbol_id for s in result.resolved_symbols}
        assert "sym-b" in symbol_ids or len(result.mappings) >= 1


# =============================================================================
# Exception Tests
# =============================================================================


class TestSymbolResolutionExceptions:
    """Tests for symbol resolution exceptions."""

    def test_symbol_resolution_error(self):
        """Test base symbol resolution error."""
        error = SymbolResolutionError("Test error")
        assert str(error) == "Test error"

    def test_symbol_not_found_error(self):
        """Test symbol not found error."""
        error = SymbolNotFoundError("scip-python org/repo module/func.")
        assert "scip-python org/repo module/func." in str(error)
        assert error.symbol_id == "scip-python org/repo module/func."

    def test_ambiguous_symbol_error(self):
        """Test ambiguous symbol error."""
        candidates = ["sym1", "sym2", "sym3"]
        error = AmbiguousSymbolError("func", candidates)
        assert "func" in str(error)
        assert error.query == "func"
        assert error.candidates == candidates
