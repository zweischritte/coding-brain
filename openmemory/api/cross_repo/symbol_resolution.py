"""Cross-repository symbol resolution.

This module provides symbol resolution across multiple repositories for detecting
and resolving symbols that span repository boundaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .registry import RepositoryRegistry


# =============================================================================
# Exceptions
# =============================================================================


class SymbolResolutionError(Exception):
    """Base exception for symbol resolution errors."""

    pass


class SymbolNotFoundError(SymbolResolutionError):
    """Raised when a symbol is not found."""

    def __init__(self, symbol_id: str):
        self.symbol_id = symbol_id
        super().__init__(f"Symbol not found: {symbol_id}")


class AmbiguousSymbolError(SymbolResolutionError):
    """Raised when a symbol query matches multiple symbols."""

    def __init__(self, query: str, candidates: list[str]):
        self.query = query
        self.candidates = candidates
        super().__init__(
            f"Ambiguous symbol query '{query}': {len(candidates)} candidates found"
        )


# =============================================================================
# Enums
# =============================================================================


class SymbolType(str, Enum):
    """Type of symbol."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    PACKAGE = "package"
    INTERFACE = "interface"
    TYPE = "type"

    @property
    def is_callable(self) -> bool:
        """Check if the symbol type is callable."""
        return self in (SymbolType.FUNCTION, SymbolType.METHOD, SymbolType.CLASS)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CrossRepoSymbol:
    """A symbol that may exist across repositories."""

    symbol_id: str
    repo_id: str
    name: str
    symbol_type: SymbolType
    file_path: str = ""
    line_number: int = 0
    signature: str = ""
    docstring: str = ""
    visibility: str = "public"
    exported: bool = True
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrossRepoSymbol):
            return False
        return self.symbol_id == other.symbol_id

    def __hash__(self) -> int:
        return hash(self.symbol_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert symbol to dictionary."""
        return {
            "symbol_id": self.symbol_id,
            "repo_id": self.repo_id,
            "name": self.name,
            "symbol_type": self.symbol_type.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "signature": self.signature,
            "docstring": self.docstring,
            "visibility": self.visibility,
            "exported": self.exported,
            "references": self.references,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossRepoSymbol":
        """Create symbol from dictionary."""
        return cls(
            symbol_id=data["symbol_id"],
            repo_id=data["repo_id"],
            name=data["name"],
            symbol_type=SymbolType(data["symbol_type"]),
            file_path=data.get("file_path", ""),
            line_number=data.get("line_number", 0),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            visibility=data.get("visibility", "public"),
            exported=data.get("exported", True),
            references=data.get("references", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SymbolMapping:
    """A mapping between symbols across repositories."""

    source_symbol_id: str
    target_symbol_id: str
    source_repo_id: str
    target_repo_id: str
    mapping_type: str  # "import", "inheritance", "implementation", "call", "inferred"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymbolMapping):
            return False
        return (
            self.source_symbol_id == other.source_symbol_id
            and self.target_symbol_id == other.target_symbol_id
            and self.mapping_type == other.mapping_type
        )

    def __hash__(self) -> int:
        return hash((self.source_symbol_id, self.target_symbol_id, self.mapping_type))


@dataclass
class SymbolResolutionConfig:
    """Configuration for symbol resolution."""

    max_depth: int = 3
    include_private: bool = False
    min_confidence: float = 0.7
    resolve_transitive: bool = True
    max_results: int = 100


@dataclass
class SymbolResolutionResult:
    """Result of a symbol resolution operation."""

    query: str
    resolved_symbols: list[CrossRepoSymbol]
    mappings: list[SymbolMapping]
    confidence: float = 1.0
    search_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Check if the resolution found any symbols."""
        return len(self.resolved_symbols) > 0

    @property
    def is_ambiguous(self) -> bool:
        """Check if the resolution is ambiguous (multiple results)."""
        return len(self.resolved_symbols) > 1


# =============================================================================
# Symbol Store Interface
# =============================================================================


class SymbolStore(ABC):
    """Abstract interface for symbol storage operations."""

    @abstractmethod
    def add(self, symbol: CrossRepoSymbol) -> None:
        """Add a symbol to the store."""
        pass

    @abstractmethod
    def get(self, symbol_id: str) -> Optional[CrossRepoSymbol]:
        """Get a symbol by ID."""
        pass

    @abstractmethod
    def remove(self, symbol_id: str) -> None:
        """Remove a symbol from the store."""
        pass

    @abstractmethod
    def search_by_name(self, name: str) -> list[CrossRepoSymbol]:
        """Search symbols by name (partial match)."""
        pass

    @abstractmethod
    def search_by_repo(self, repo_id: str) -> list[CrossRepoSymbol]:
        """Search symbols by repository."""
        pass

    @abstractmethod
    def search_by_type(self, symbol_type: SymbolType) -> list[CrossRepoSymbol]:
        """Search symbols by type."""
        pass

    @abstractmethod
    def add_mapping(self, mapping: SymbolMapping) -> None:
        """Add a symbol mapping."""
        pass

    @abstractmethod
    def get_mappings(self, source_symbol_id: str) -> list[SymbolMapping]:
        """Get mappings from a source symbol."""
        pass

    @abstractmethod
    def get_reverse_mappings(self, target_symbol_id: str) -> list[SymbolMapping]:
        """Get mappings to a target symbol."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all symbols and mappings."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get total number of symbols."""
        pass


# =============================================================================
# In-Memory Symbol Store
# =============================================================================


class InMemorySymbolStore(SymbolStore):
    """In-memory symbol store for testing and development."""

    def __init__(self):
        self._symbols: dict[str, CrossRepoSymbol] = {}
        self._mappings: list[SymbolMapping] = []

    @property
    def count(self) -> int:
        """Get total number of symbols."""
        return len(self._symbols)

    def add(self, symbol: CrossRepoSymbol) -> None:
        """Add a symbol to the store."""
        self._symbols[symbol.symbol_id] = symbol

    def get(self, symbol_id: str) -> Optional[CrossRepoSymbol]:
        """Get a symbol by ID."""
        return self._symbols.get(symbol_id)

    def remove(self, symbol_id: str) -> None:
        """Remove a symbol from the store."""
        if symbol_id in self._symbols:
            del self._symbols[symbol_id]

    def search_by_name(self, name: str) -> list[CrossRepoSymbol]:
        """Search symbols by name (partial match)."""
        name_lower = name.lower()
        return [s for s in self._symbols.values() if name_lower in s.name.lower()]

    def search_by_repo(self, repo_id: str) -> list[CrossRepoSymbol]:
        """Search symbols by repository."""
        return [s for s in self._symbols.values() if s.repo_id == repo_id]

    def search_by_type(self, symbol_type: SymbolType) -> list[CrossRepoSymbol]:
        """Search symbols by type."""
        return [s for s in self._symbols.values() if s.symbol_type == symbol_type]

    def add_mapping(self, mapping: SymbolMapping) -> None:
        """Add a symbol mapping."""
        self._mappings.append(mapping)

    def get_mappings(self, source_symbol_id: str) -> list[SymbolMapping]:
        """Get mappings from a source symbol."""
        return [m for m in self._mappings if m.source_symbol_id == source_symbol_id]

    def get_reverse_mappings(self, target_symbol_id: str) -> list[SymbolMapping]:
        """Get mappings to a target symbol."""
        return [m for m in self._mappings if m.target_symbol_id == target_symbol_id]

    def clear(self) -> None:
        """Clear all symbols and mappings."""
        self._symbols.clear()
        self._mappings.clear()


# =============================================================================
# Cross-Repository Symbol Resolver
# =============================================================================


class CrossRepoSymbolResolver:
    """Resolver for cross-repository symbol resolution."""

    def __init__(
        self,
        registry: RepositoryRegistry,
        symbol_store: SymbolStore,
        config: Optional[SymbolResolutionConfig] = None,
    ):
        self._registry = registry
        self._symbol_store = symbol_store
        self._config = config or SymbolResolutionConfig()

    @property
    def config(self) -> SymbolResolutionConfig:
        """Get resolver configuration."""
        return self._config

    def resolve_by_id(self, symbol_id: str) -> SymbolResolutionResult:
        """Resolve a symbol by its ID."""
        symbol = self._symbol_store.get(symbol_id)
        if symbol:
            return SymbolResolutionResult(
                query=symbol_id,
                resolved_symbols=[symbol],
                mappings=self._symbol_store.get_mappings(symbol_id),
                confidence=1.0,
            )
        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=[],
            mappings=[],
            confidence=0.0,
        )

    def resolve_by_name(
        self,
        name: str,
        repo_ids: Optional[list[str]] = None,
        symbol_type: Optional[SymbolType] = None,
    ) -> SymbolResolutionResult:
        """Resolve symbols by name."""
        # Search by name
        symbols = self._symbol_store.search_by_name(name)

        # Filter exact matches first
        exact_matches = [s for s in symbols if s.name == name]
        if exact_matches:
            symbols = exact_matches

        # Filter by repository if specified
        if repo_ids:
            symbols = [s for s in symbols if s.repo_id in repo_ids]

        # Filter by type if specified
        if symbol_type:
            symbols = [s for s in symbols if s.symbol_type == symbol_type]

        # Filter private symbols if not included
        if not self._config.include_private:
            symbols = [s for s in symbols if s.visibility != "private"]

        # Limit results
        symbols = symbols[: self._config.max_results]

        # Collect all mappings
        all_mappings: list[SymbolMapping] = []
        for sym in symbols:
            all_mappings.extend(self._symbol_store.get_mappings(sym.symbol_id))

        return SymbolResolutionResult(
            query=name,
            resolved_symbols=symbols,
            mappings=all_mappings,
            confidence=1.0 if len(symbols) == 1 else (0.8 if symbols else 0.0),
        )

    def resolve_dependents(self, symbol_id: str) -> SymbolResolutionResult:
        """Find symbols that depend on the given symbol.

        For a mapping like: source=BaseModel -> target=UserModel (inheritance)
        This means UserModel depends on BaseModel.
        So to find dependents of BaseModel, we look for mappings where
        BaseModel is the source.
        """
        # Get forward mappings (things that depend on this symbol)
        mappings = self._symbol_store.get_mappings(symbol_id)

        # Resolve the dependent symbols
        dependents: list[CrossRepoSymbol] = []
        for mapping in mappings:
            symbol = self._symbol_store.get(mapping.target_symbol_id)
            if symbol:
                dependents.append(symbol)

        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=dependents,
            mappings=mappings,
            confidence=1.0 if mappings else 0.0,
        )

    def resolve_dependencies(self, symbol_id: str) -> SymbolResolutionResult:
        """Find symbols that the given symbol depends on."""
        # Get forward mappings where this symbol is the target
        all_mappings = self._symbol_store.get_reverse_mappings(symbol_id)

        # Also check if there are mappings where this symbol is source
        forward_mappings = self._symbol_store.get_mappings(symbol_id)

        # Combine mappings
        combined_mappings = list(set(all_mappings + forward_mappings))

        # Resolve the dependency symbols
        dependencies: list[CrossRepoSymbol] = []
        for mapping in forward_mappings:
            # This symbol imports/uses the source
            symbol = self._symbol_store.get(mapping.source_symbol_id)
            if symbol:
                dependencies.append(symbol)

        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=dependencies,
            mappings=combined_mappings,
            confidence=1.0 if forward_mappings else 0.0,
        )

    def resolve_cross_repo_references(
        self, symbol_id: str
    ) -> SymbolResolutionResult:
        """Find all cross-repo references for a symbol."""
        symbol = self._symbol_store.get(symbol_id)
        if not symbol:
            return SymbolResolutionResult(
                query=symbol_id,
                resolved_symbols=[],
                mappings=[],
                confidence=0.0,
            )

        # Get all mappings (forward and reverse)
        forward_mappings = self._symbol_store.get_mappings(symbol_id)
        reverse_mappings = self._symbol_store.get_reverse_mappings(symbol_id)
        all_mappings = list(set(forward_mappings + reverse_mappings))

        # Get cross-repo references only
        cross_repo_mappings = [
            m for m in all_mappings if m.source_repo_id != m.target_repo_id
        ]

        # Resolve referenced symbols
        referenced_symbols: list[CrossRepoSymbol] = []
        for mapping in cross_repo_mappings:
            # Get the symbol from the other repo
            other_id = (
                mapping.target_symbol_id
                if mapping.source_symbol_id == symbol_id
                else mapping.source_symbol_id
            )
            other_symbol = self._symbol_store.get(other_id)
            if other_symbol and other_symbol.repo_id != symbol.repo_id:
                referenced_symbols.append(other_symbol)

        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=referenced_symbols,
            mappings=cross_repo_mappings,
            confidence=1.0 if cross_repo_mappings else 0.0,
        )

    def find_usages_across_repos(self, symbol_id: str) -> SymbolResolutionResult:
        """Find usages of a symbol across all repositories."""
        symbol = self._symbol_store.get(symbol_id)
        if not symbol:
            return SymbolResolutionResult(
                query=symbol_id,
                resolved_symbols=[],
                mappings=[],
                confidence=0.0,
            )

        # Get mappings where this symbol is the source (it's being used)
        mappings = self._symbol_store.get_mappings(symbol_id)

        # Filter to cross-repo usages
        cross_repo_usages = [
            m for m in mappings if m.target_repo_id != symbol.repo_id
        ]

        # Resolve the using symbols
        using_symbols: list[CrossRepoSymbol] = []
        for mapping in cross_repo_usages:
            using_symbol = self._symbol_store.get(mapping.target_symbol_id)
            if using_symbol:
                using_symbols.append(using_symbol)

        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=using_symbols,
            mappings=cross_repo_usages,
            confidence=1.0 if cross_repo_usages else 0.0,
        )

    def resolve_transitive_dependencies(
        self,
        symbol_id: str,
        max_depth: Optional[int] = None,
    ) -> SymbolResolutionResult:
        """Resolve transitive dependencies of a symbol."""
        if max_depth is None:
            max_depth = self._config.max_depth

        visited: set[str] = set()
        all_symbols: list[CrossRepoSymbol] = []
        all_mappings: list[SymbolMapping] = []

        def traverse(current_id: str, depth: int) -> None:
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get mappings where this symbol depends on something
            # (reverse mappings - where this symbol is the target)
            reverse_mappings = self._symbol_store.get_reverse_mappings(current_id)

            for mapping in reverse_mappings:
                if mapping not in all_mappings:
                    all_mappings.append(mapping)

                source_symbol = self._symbol_store.get(mapping.source_symbol_id)
                if source_symbol and source_symbol not in all_symbols:
                    all_symbols.append(source_symbol)
                    traverse(mapping.source_symbol_id, depth + 1)

        traverse(symbol_id, 0)

        return SymbolResolutionResult(
            query=symbol_id,
            resolved_symbols=all_symbols,
            mappings=all_mappings,
            search_depth=max_depth,
            confidence=1.0 if all_symbols else 0.0,
        )
