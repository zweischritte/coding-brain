"""Cross-repository dependency graph.

This module provides dependency graph construction and traversal for detecting
and analyzing dependencies across repositories.

Per implementation plan v9 section 6.6 (Multi-Repository Graph):
- New edges: CODE_DEPENDS_ON, CODE_PUBLISHES_API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .registry import RepositoryRegistry


# =============================================================================
# Exceptions
# =============================================================================


class DependencyGraphError(Exception):
    """Base exception for dependency graph errors."""

    pass


class CyclicDependencyError(DependencyGraphError):
    """Raised when a cyclic dependency is detected."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Cyclic dependency detected: {cycle_str}")


# =============================================================================
# Enums
# =============================================================================


class DependencyType(str, Enum):
    """Type of dependency between repositories."""

    RUNTIME = "runtime"  # Required at runtime
    DEV = "dev"  # Development dependency
    OPTIONAL = "optional"  # Optional dependency
    PEER = "peer"  # Peer dependency
    BUILD = "build"  # Build-time dependency
    API = "api"  # API dependency (HTTP, gRPC, etc.)
    INTERNAL = "internal"  # Internal/monorepo dependency

    @property
    def is_required(self) -> bool:
        """Check if this dependency type is required."""
        return self in (DependencyType.RUNTIME, DependencyType.BUILD)

    @property
    def is_production(self) -> bool:
        """Check if this dependency is for production."""
        return self in (DependencyType.RUNTIME, DependencyType.API, DependencyType.INTERNAL)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RepoDependency:
    """A dependency between repositories."""

    source_repo_id: str
    target_repo_id: str
    dependency_type: DependencyType
    version_constraint: str = ""
    declared_at: Optional[datetime] = None
    symbols: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RepoDependency):
            return False
        return (
            self.source_repo_id == other.source_repo_id
            and self.target_repo_id == other.target_repo_id
            and self.dependency_type == other.dependency_type
        )

    def __hash__(self) -> int:
        return hash((self.source_repo_id, self.target_repo_id, self.dependency_type))

    def to_dict(self) -> dict[str, Any]:
        """Convert dependency to dictionary."""
        return {
            "source_repo_id": self.source_repo_id,
            "target_repo_id": self.target_repo_id,
            "dependency_type": self.dependency_type.value,
            "version_constraint": self.version_constraint,
            "declared_at": self.declared_at.isoformat() if self.declared_at else None,
            "symbols": self.symbols,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoDependency":
        """Create dependency from dictionary."""
        return cls(
            source_repo_id=data["source_repo_id"],
            target_repo_id=data["target_repo_id"],
            dependency_type=DependencyType(data["dependency_type"]),
            version_constraint=data.get("version_constraint", ""),
            declared_at=datetime.fromisoformat(data["declared_at"])
            if data.get("declared_at")
            else None,
            symbols=data.get("symbols", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""

    source_repo_id: str
    target_repo_id: str
    dependency_type: DependencyType
    weight: float = 1.0
    symbol_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyGraphConfig:
    """Configuration for dependency graph operations."""

    max_depth: int = 10
    include_dev_dependencies: bool = False
    include_optional_dependencies: bool = False
    detect_cycles: bool = True


# =============================================================================
# Dependency Graph Store Interface
# =============================================================================


class DependencyGraphStore(ABC):
    """Abstract interface for dependency graph storage."""

    @abstractmethod
    def add_dependency(self, dependency: RepoDependency) -> None:
        """Add or update a dependency."""
        pass

    @abstractmethod
    def remove_dependency(self, source_repo_id: str, target_repo_id: str) -> None:
        """Remove a dependency."""
        pass

    @abstractmethod
    def get_dependencies(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get dependencies of a repository."""
        pass

    @abstractmethod
    def get_dependents(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get repositories that depend on the given repository."""
        pass

    @abstractmethod
    def has_dependency(self, source_repo_id: str, target_repo_id: str) -> bool:
        """Check if a dependency exists."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all dependencies."""
        pass

    @property
    @abstractmethod
    def dependency_count(self) -> int:
        """Get total number of dependencies."""
        pass

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Get total number of edges."""
        pass


# =============================================================================
# In-Memory Dependency Graph Store
# =============================================================================


class InMemoryDependencyGraphStore(DependencyGraphStore):
    """In-memory dependency graph store for testing and development."""

    def __init__(self):
        # Map: (source, target) -> RepoDependency
        self._dependencies: dict[tuple[str, str], RepoDependency] = {}

    @property
    def dependency_count(self) -> int:
        """Get total number of dependencies."""
        return len(self._dependencies)

    @property
    def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._dependencies)

    def add_dependency(self, dependency: RepoDependency) -> None:
        """Add or update a dependency."""
        key = (dependency.source_repo_id, dependency.target_repo_id)
        self._dependencies[key] = dependency

    def remove_dependency(self, source_repo_id: str, target_repo_id: str) -> None:
        """Remove a dependency."""
        key = (source_repo_id, target_repo_id)
        if key in self._dependencies:
            del self._dependencies[key]

    def get_dependencies(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get dependencies of a repository."""
        deps = [d for d in self._dependencies.values() if d.source_repo_id == repo_id]
        if dep_type:
            deps = [d for d in deps if d.dependency_type == dep_type]
        return deps

    def get_dependents(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get repositories that depend on the given repository."""
        deps = [d for d in self._dependencies.values() if d.target_repo_id == repo_id]
        if dep_type:
            deps = [d for d in deps if d.dependency_type == dep_type]
        return deps

    def has_dependency(self, source_repo_id: str, target_repo_id: str) -> bool:
        """Check if a dependency exists."""
        return (source_repo_id, target_repo_id) in self._dependencies

    def clear(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()


# =============================================================================
# Repository Dependency Graph
# =============================================================================


class RepoDependencyGraph:
    """Graph for managing repository dependencies."""

    def __init__(
        self,
        registry: RepositoryRegistry,
        store: DependencyGraphStore,
        config: Optional[DependencyGraphConfig] = None,
    ):
        self._registry = registry
        self._store = store
        self._config = config or DependencyGraphConfig()

    @property
    def config(self) -> DependencyGraphConfig:
        """Get graph configuration."""
        return self._config

    def add_dependency(
        self,
        source_repo_id: str,
        target_repo_id: str,
        dependency_type: DependencyType,
        version_constraint: str = "",
        symbols: Optional[list[str]] = None,
    ) -> RepoDependency:
        """Add a dependency between repositories."""
        dep = RepoDependency(
            source_repo_id=source_repo_id,
            target_repo_id=target_repo_id,
            dependency_type=dependency_type,
            version_constraint=version_constraint,
            symbols=symbols or [],
            declared_at=datetime.now(timezone.utc),
        )
        self._store.add_dependency(dep)
        return dep

    def remove_dependency(self, source_repo_id: str, target_repo_id: str) -> None:
        """Remove a dependency."""
        self._store.remove_dependency(source_repo_id, target_repo_id)

    def get_direct_dependencies(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get direct dependencies of a repository."""
        return self._store.get_dependencies(repo_id, dep_type)

    def get_direct_dependents(
        self,
        repo_id: str,
        dep_type: Optional[DependencyType] = None,
    ) -> list[RepoDependency]:
        """Get direct dependents of a repository."""
        return self._store.get_dependents(repo_id, dep_type)

    def get_all_dependencies(
        self,
        repo_id: str,
        max_depth: Optional[int] = None,
    ) -> list[RepoDependency]:
        """Get all transitive dependencies of a repository.

        Args:
            repo_id: Repository to get dependencies for
            max_depth: Maximum depth to traverse. 1 = direct deps only,
                       2 = deps + their deps, etc.
        """
        if max_depth is None:
            max_depth = self._config.max_depth

        visited: set[str] = set()
        all_deps: list[RepoDependency] = []

        def traverse(current_id: str, depth: int) -> None:
            if depth >= max_depth or current_id in visited:
                return

            visited.add(current_id)
            deps = self._store.get_dependencies(current_id)

            for dep in deps:
                if self._should_include_dependency(dep):
                    if dep not in all_deps:
                        all_deps.append(dep)
                    traverse(dep.target_repo_id, depth + 1)

        traverse(repo_id, 0)
        return all_deps

    def get_all_dependents(
        self,
        repo_id: str,
        max_depth: Optional[int] = None,
    ) -> list[RepoDependency]:
        """Get all transitive dependents of a repository.

        Args:
            repo_id: Repository to get dependents for
            max_depth: Maximum depth to traverse. 1 = direct dependents only,
                       2 = dependents + their dependents, etc.
        """
        if max_depth is None:
            max_depth = self._config.max_depth

        visited: set[str] = set()
        all_deps: list[RepoDependency] = []

        def traverse(current_id: str, depth: int) -> None:
            if depth >= max_depth or current_id in visited:
                return

            visited.add(current_id)
            deps = self._store.get_dependents(current_id)

            for dep in deps:
                if self._should_include_dependency(dep):
                    if dep not in all_deps:
                        all_deps.append(dep)
                    traverse(dep.source_repo_id, depth + 1)

        traverse(repo_id, 0)
        return all_deps

    def get_dependency_path(
        self,
        source_repo_id: str,
        target_repo_id: str,
    ) -> list[RepoDependency]:
        """Find the shortest dependency path between two repositories."""
        if source_repo_id == target_repo_id:
            return []

        # BFS to find shortest path
        visited: set[str] = set()
        queue: deque[tuple[str, list[RepoDependency]]] = deque()
        queue.append((source_repo_id, []))
        visited.add(source_repo_id)

        while queue:
            current_id, path = queue.popleft()
            deps = self._store.get_dependencies(current_id)

            for dep in deps:
                if dep.target_repo_id == target_repo_id:
                    return path + [dep]

                if dep.target_repo_id not in visited:
                    visited.add(dep.target_repo_id)
                    queue.append((dep.target_repo_id, path + [dep]))

        return []

    def has_path(self, source_repo_id: str, target_repo_id: str) -> bool:
        """Check if a path exists between two repositories."""
        return len(self.get_dependency_path(source_repo_id, target_repo_id)) > 0

    def detect_cycles(self) -> list[list[str]]:
        """Detect all cycles in the dependency graph."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()

        # Get all repositories from dependencies
        all_repos: set[str] = set()
        for dep in self._get_all_dependencies():
            all_repos.add(dep.source_repo_id)
            all_repos.add(dep.target_repo_id)

        def dfs(repo_id: str, path: list[str]) -> None:
            visited.add(repo_id)
            rec_stack.add(repo_id)
            path.append(repo_id)

            deps = self._store.get_dependencies(repo_id)
            for dep in deps:
                if dep.target_repo_id not in visited:
                    dfs(dep.target_repo_id, path.copy())
                elif dep.target_repo_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dep.target_repo_id)
                    cycle = path[cycle_start:] + [dep.target_repo_id]
                    cycles.append(cycle)

            rec_stack.remove(repo_id)

        for repo_id in all_repos:
            if repo_id not in visited:
                dfs(repo_id, [])

        return cycles

    def get_dependency_tree(
        self,
        repo_id: str,
        max_depth: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get dependency tree as a nested dictionary."""
        if max_depth is None:
            max_depth = self._config.max_depth

        visited: set[str] = set()

        def build_tree(current_id: str, depth: int) -> dict[str, Any]:
            if depth >= max_depth or current_id in visited:
                return {
                    "repo_id": current_id,
                    "dependencies": [],
                }

            visited.add(current_id)
            deps = self._store.get_dependencies(current_id)

            children = []
            for dep in deps:
                if self._should_include_dependency(dep):
                    child = build_tree(dep.target_repo_id, depth + 1)
                    children.append(child)

            visited.discard(current_id)  # Allow same repo in different branches
            return {
                "repo_id": current_id,
                "dependencies": children,
            }

        return build_tree(repo_id, 0)

    def topological_sort(self) -> list[str]:
        """Topologically sort repositories based on dependencies.

        Returns repositories in order where dependencies come before dependents.
        Raises CyclicDependencyError if a cycle is detected.
        """
        # Check for cycles first
        cycles = self.detect_cycles()
        if cycles:
            raise CyclicDependencyError(cycles[0])

        # Kahn's algorithm
        # Get all repositories
        all_repos: set[str] = set()
        in_degree: dict[str, int] = {}

        for dep in self._get_all_dependencies():
            all_repos.add(dep.source_repo_id)
            all_repos.add(dep.target_repo_id)

        # Initialize in-degrees
        for repo in all_repos:
            in_degree[repo] = 0

        # Calculate in-degrees (how many things depend on this repo)
        for dep in self._get_all_dependencies():
            in_degree[dep.source_repo_id] += 1  # source depends on target

        # Start with repos that nothing depends on (lowest dependencies)
        queue = deque([repo for repo in all_repos if in_degree[repo] == 0])
        result: list[str] = []

        while queue:
            repo = queue.popleft()
            result.append(repo)

            # Reduce in-degree for all repos that depend on this one
            deps = self._store.get_dependents(repo)
            for dep in deps:
                in_degree[dep.source_repo_id] -= 1
                if in_degree[dep.source_repo_id] == 0:
                    queue.append(dep.source_repo_id)

        return result

    def get_affected_repos(
        self,
        repo_id: str,
        max_depth: Optional[int] = None,
    ) -> list[str]:
        """Get all repositories affected by a change to the given repository."""
        deps = self.get_all_dependents(repo_id, max_depth)
        return list(set(d.source_repo_id for d in deps))

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the dependency graph."""
        all_deps = self._get_all_dependencies()

        by_type: dict[str, int] = {}
        for dep in all_deps:
            type_key = dep.dependency_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        all_repos: set[str] = set()
        for dep in all_deps:
            all_repos.add(dep.source_repo_id)
            all_repos.add(dep.target_repo_id)

        return {
            "total_dependencies": len(all_deps),
            "total_repos": len(all_repos),
            "by_type": by_type,
        }

    def _should_include_dependency(self, dep: RepoDependency) -> bool:
        """Check if a dependency should be included based on config."""
        if dep.dependency_type == DependencyType.DEV:
            return self._config.include_dev_dependencies
        if dep.dependency_type == DependencyType.OPTIONAL:
            return self._config.include_optional_dependencies
        return True

    def _get_all_dependencies(self) -> list[RepoDependency]:
        """Get all dependencies in the graph."""
        # This is a workaround since we don't have direct access
        all_deps: list[RepoDependency] = []
        seen_repos: set[str] = set()

        # Get all repos from registry
        for repo in self._registry.list_all():
            seen_repos.add(repo.repo_id)

        # Get dependencies from all known repos
        for repo_id in seen_repos:
            deps = self._store.get_dependencies(repo_id)
            for dep in deps:
                if dep not in all_deps:
                    all_deps.append(dep)

        return all_deps
