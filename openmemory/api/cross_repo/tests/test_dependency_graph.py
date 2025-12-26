"""Tests for cross-repository dependency graph.

This module tests dependency graph construction and traversal
for detecting dependencies across repositories.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from openmemory.api.cross_repo.registry import (
    InMemoryRepositoryRegistry,
    RepositoryMetadata,
    RepositoryStatus,
)
from openmemory.api.cross_repo.dependency_graph import (
    CyclicDependencyError,
    DependencyEdge,
    DependencyGraphConfig,
    DependencyGraphError,
    DependencyType,
    InMemoryDependencyGraphStore,
    RepoDependency,
    RepoDependencyGraph,
)


# =============================================================================
# DependencyType Tests
# =============================================================================


class TestDependencyType:
    """Tests for DependencyType enum."""

    def test_dependency_types(self):
        """Test all dependency types exist."""
        assert DependencyType.RUNTIME.value == "runtime"
        assert DependencyType.DEV.value == "dev"
        assert DependencyType.OPTIONAL.value == "optional"
        assert DependencyType.PEER.value == "peer"
        assert DependencyType.BUILD.value == "build"
        assert DependencyType.API.value == "api"
        assert DependencyType.INTERNAL.value == "internal"

    def test_is_required(self):
        """Test is_required property."""
        assert DependencyType.RUNTIME.is_required is True
        assert DependencyType.BUILD.is_required is True
        assert DependencyType.DEV.is_required is False
        assert DependencyType.OPTIONAL.is_required is False
        assert DependencyType.PEER.is_required is False

    def test_is_production(self):
        """Test is_production property."""
        assert DependencyType.RUNTIME.is_production is True
        assert DependencyType.API.is_production is True
        assert DependencyType.DEV.is_production is False
        assert DependencyType.BUILD.is_production is False


# =============================================================================
# RepoDependency Tests
# =============================================================================


class TestRepoDependency:
    """Tests for RepoDependency dataclass."""

    def test_create_dependency(self):
        """Test creating a repository dependency."""
        dep = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/shared-lib",
            dependency_type=DependencyType.RUNTIME,
            version_constraint=">=1.0.0,<2.0.0",
        )
        assert dep.source_repo_id == "org/app"
        assert dep.target_repo_id == "org/shared-lib"
        assert dep.dependency_type == DependencyType.RUNTIME
        assert dep.version_constraint == ">=1.0.0,<2.0.0"

    def test_dependency_defaults(self):
        """Test default values for dependency."""
        dep = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )
        assert dep.version_constraint == ""
        assert dep.declared_at is None
        assert dep.symbols == []
        assert dep.metadata == {}

    def test_dependency_with_symbols(self):
        """Test dependency with linked symbols."""
        dep = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
            symbols=["scip-python org/lib utils/func.", "scip-python org/lib models/Model#"],
        )
        assert len(dep.symbols) == 2
        assert "scip-python org/lib utils/func." in dep.symbols

    def test_dependency_equality(self):
        """Test dependency equality."""
        dep1 = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )
        dep2 = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )
        dep3 = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/c",
            dependency_type=DependencyType.RUNTIME,
        )

        assert dep1 == dep2
        assert dep1 != dep3

    def test_dependency_hash(self):
        """Test dependency is hashable."""
        dep1 = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )
        dep2 = RepoDependency(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )

        assert hash(dep1) == hash(dep2)

        dep_set = {dep1, dep2}
        assert len(dep_set) == 1

    def test_dependency_to_dict(self):
        """Test converting dependency to dictionary."""
        dep = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.API,
            version_constraint="^1.0.0",
        )
        data = dep.to_dict()
        assert data["source_repo_id"] == "org/app"
        assert data["target_repo_id"] == "org/lib"
        assert data["dependency_type"] == "api"
        assert data["version_constraint"] == "^1.0.0"

    def test_dependency_from_dict(self):
        """Test creating dependency from dictionary."""
        data = {
            "source_repo_id": "org/app",
            "target_repo_id": "org/lib",
            "dependency_type": "runtime",
            "version_constraint": ">=1.0.0",
        }
        dep = RepoDependency.from_dict(data)
        assert dep.source_repo_id == "org/app"
        assert dep.dependency_type == DependencyType.RUNTIME


# =============================================================================
# DependencyEdge Tests
# =============================================================================


class TestDependencyEdge:
    """Tests for DependencyEdge dataclass."""

    def test_create_edge(self):
        """Test creating a dependency edge."""
        edge = DependencyEdge(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
            weight=1.0,
        )
        assert edge.source_repo_id == "org/app"
        assert edge.target_repo_id == "org/lib"
        assert edge.weight == 1.0

    def test_edge_defaults(self):
        """Test default values for edge."""
        edge = DependencyEdge(
            source_repo_id="org/a",
            target_repo_id="org/b",
            dependency_type=DependencyType.RUNTIME,
        )
        assert edge.weight == 1.0
        assert edge.symbol_count == 0
        assert edge.metadata == {}

    def test_edge_with_symbol_count(self):
        """Test edge with symbol count."""
        edge = DependencyEdge(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
            symbol_count=42,
        )
        assert edge.symbol_count == 42


# =============================================================================
# DependencyGraphConfig Tests
# =============================================================================


class TestDependencyGraphConfig:
    """Tests for DependencyGraphConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DependencyGraphConfig()
        assert config.max_depth == 10
        assert config.include_dev_dependencies is False
        assert config.include_optional_dependencies is False
        assert config.detect_cycles is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DependencyGraphConfig(
            max_depth=5,
            include_dev_dependencies=True,
            include_optional_dependencies=True,
            detect_cycles=False,
        )
        assert config.max_depth == 5
        assert config.include_dev_dependencies is True
        assert config.include_optional_dependencies is True
        assert config.detect_cycles is False


# =============================================================================
# InMemoryDependencyGraphStore Tests
# =============================================================================


class TestInMemoryDependencyGraphStore:
    """Tests for InMemoryDependencyGraphStore implementation."""

    def test_create_store(self):
        """Test creating an empty dependency store."""
        store = InMemoryDependencyGraphStore()
        assert store.dependency_count == 0
        assert store.edge_count == 0

    def test_add_dependency(self):
        """Test adding a dependency."""
        store = InMemoryDependencyGraphStore()
        dep = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
        )
        store.add_dependency(dep)
        assert store.dependency_count == 1

    def test_add_duplicate_dependency(self):
        """Test adding a duplicate dependency updates it."""
        store = InMemoryDependencyGraphStore()
        dep1 = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
            version_constraint=">=1.0.0",
        )
        store.add_dependency(dep1)

        dep2 = RepoDependency(
            source_repo_id="org/app",
            target_repo_id="org/lib",
            dependency_type=DependencyType.RUNTIME,
            version_constraint=">=2.0.0",  # Updated version
        )
        store.add_dependency(dep2)

        # Should still be 1 dependency, but updated
        assert store.dependency_count == 1
        deps = store.get_dependencies("org/app")
        assert deps[0].version_constraint == ">=2.0.0"

    def test_get_dependencies(self):
        """Test getting dependencies of a repository."""
        store = InMemoryDependencyGraphStore()
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/lib1",
                dependency_type=DependencyType.RUNTIME,
            )
        )
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/lib2",
                dependency_type=DependencyType.RUNTIME,
            )
        )
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/other",
                target_repo_id="org/lib1",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        deps = store.get_dependencies("org/app")
        assert len(deps) == 2
        target_repos = [d.target_repo_id for d in deps]
        assert "org/lib1" in target_repos
        assert "org/lib2" in target_repos

    def test_get_dependents(self):
        """Test getting dependents of a repository (reverse dependencies)."""
        store = InMemoryDependencyGraphStore()
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app1",
                target_repo_id="org/lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app2",
                target_repo_id="org/lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        dependents = store.get_dependents("org/lib")
        assert len(dependents) == 2
        source_repos = [d.source_repo_id for d in dependents]
        assert "org/app1" in source_repos
        assert "org/app2" in source_repos

    def test_get_dependencies_by_type(self):
        """Test filtering dependencies by type."""
        store = InMemoryDependencyGraphStore()
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/test-lib",
                dependency_type=DependencyType.DEV,
            )
        )

        runtime_deps = store.get_dependencies("org/app", dep_type=DependencyType.RUNTIME)
        assert len(runtime_deps) == 1
        assert runtime_deps[0].target_repo_id == "org/lib"

        dev_deps = store.get_dependencies("org/app", dep_type=DependencyType.DEV)
        assert len(dev_deps) == 1
        assert dev_deps[0].target_repo_id == "org/test-lib"

    def test_remove_dependency(self):
        """Test removing a dependency."""
        store = InMemoryDependencyGraphStore()
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )
        assert store.dependency_count == 1

        store.remove_dependency("org/app", "org/lib")
        assert store.dependency_count == 0

    def test_has_dependency(self):
        """Test checking if dependency exists."""
        store = InMemoryDependencyGraphStore()
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/app",
                target_repo_id="org/lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        assert store.has_dependency("org/app", "org/lib") is True
        assert store.has_dependency("org/app", "org/other") is False

    def test_clear(self):
        """Test clearing all dependencies."""
        store = InMemoryDependencyGraphStore()
        for i in range(5):
            store.add_dependency(
                RepoDependency(
                    source_repo_id=f"org/app{i}",
                    target_repo_id="org/lib",
                    dependency_type=DependencyType.RUNTIME,
                )
            )
        assert store.dependency_count == 5

        store.clear()
        assert store.dependency_count == 0


# =============================================================================
# RepoDependencyGraph Tests
# =============================================================================


class TestRepoDependencyGraph:
    """Tests for RepoDependencyGraph."""

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
        registry.register(
            repo_id="org/core-utils",
            name="core-utils",
            metadata=RepositoryMetadata(languages=["python"]),
        )
        return registry

    @pytest.fixture
    def graph_store(self) -> InMemoryDependencyGraphStore:
        """Create a dependency graph store with test data."""
        store = InMemoryDependencyGraphStore()

        # api-service depends on shared-lib
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/api-service",
                target_repo_id="org/shared-lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        # shared-lib depends on core-utils
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/shared-lib",
                target_repo_id="org/core-utils",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        # frontend depends on api-service (via API)
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/frontend",
                target_repo_id="org/api-service",
                dependency_type=DependencyType.API,
            )
        )

        return store

    @pytest.fixture
    def graph(
        self,
        registry: InMemoryRepositoryRegistry,
        graph_store: InMemoryDependencyGraphStore,
    ) -> RepoDependencyGraph:
        """Create a dependency graph."""
        return RepoDependencyGraph(
            registry=registry,
            store=graph_store,
        )

    def test_create_graph(
        self,
        registry: InMemoryRepositoryRegistry,
        graph_store: InMemoryDependencyGraphStore,
    ):
        """Test creating a dependency graph."""
        graph = RepoDependencyGraph(
            registry=registry,
            store=graph_store,
        )
        assert graph is not None

    def test_create_graph_with_config(
        self,
        registry: InMemoryRepositoryRegistry,
        graph_store: InMemoryDependencyGraphStore,
    ):
        """Test creating a graph with custom config."""
        config = DependencyGraphConfig(max_depth=5)
        graph = RepoDependencyGraph(
            registry=registry,
            store=graph_store,
            config=config,
        )
        assert graph.config.max_depth == 5

    def test_get_direct_dependencies(self, graph: RepoDependencyGraph):
        """Test getting direct dependencies."""
        deps = graph.get_direct_dependencies("org/api-service")
        assert len(deps) == 1
        assert deps[0].target_repo_id == "org/shared-lib"

    def test_get_direct_dependents(self, graph: RepoDependencyGraph):
        """Test getting direct dependents."""
        deps = graph.get_direct_dependents("org/shared-lib")
        assert len(deps) == 1
        assert deps[0].source_repo_id == "org/api-service"

    def test_get_all_dependencies(self, graph: RepoDependencyGraph):
        """Test getting all transitive dependencies."""
        deps = graph.get_all_dependencies("org/api-service")

        # api-service -> shared-lib -> core-utils
        assert len(deps) == 2
        target_repos = {d.target_repo_id for d in deps}
        assert "org/shared-lib" in target_repos
        assert "org/core-utils" in target_repos

    def test_get_all_dependencies_with_depth_limit(self, graph: RepoDependencyGraph):
        """Test getting dependencies with depth limit."""
        deps = graph.get_all_dependencies("org/api-service", max_depth=1)

        # Only direct dependency
        assert len(deps) == 1
        assert deps[0].target_repo_id == "org/shared-lib"

    def test_get_all_dependents(self, graph: RepoDependencyGraph):
        """Test getting all transitive dependents (reverse)."""
        deps = graph.get_all_dependents("org/core-utils")

        # core-utils <- shared-lib <- api-service
        # core-utils <- shared-lib (and api-service through frontend)
        assert len(deps) >= 2  # At least shared-lib and api-service

    def test_get_dependency_path(self, graph: RepoDependencyGraph):
        """Test finding dependency path between repositories."""
        path = graph.get_dependency_path("org/api-service", "org/core-utils")

        # api-service -> shared-lib -> core-utils
        assert len(path) == 2
        assert path[0].source_repo_id == "org/api-service"
        assert path[0].target_repo_id == "org/shared-lib"
        assert path[1].source_repo_id == "org/shared-lib"
        assert path[1].target_repo_id == "org/core-utils"

    def test_get_dependency_path_no_path(self, graph: RepoDependencyGraph):
        """Test finding path when none exists."""
        path = graph.get_dependency_path("org/core-utils", "org/frontend")
        assert path == []

    def test_has_path(self, graph: RepoDependencyGraph):
        """Test checking if path exists."""
        assert graph.has_path("org/api-service", "org/core-utils") is True
        assert graph.has_path("org/frontend", "org/core-utils") is True  # Through api-service
        assert graph.has_path("org/core-utils", "org/frontend") is False

    def test_detect_cycles_no_cycle(self, graph: RepoDependencyGraph):
        """Test cycle detection with no cycles."""
        cycles = graph.detect_cycles()
        assert len(cycles) == 0

    def test_detect_cycles_with_cycle(
        self,
        registry: InMemoryRepositoryRegistry,
        graph_store: InMemoryDependencyGraphStore,
    ):
        """Test cycle detection with a cycle."""
        # Add a cycle: core-utils -> api-service
        graph_store.add_dependency(
            RepoDependency(
                source_repo_id="org/core-utils",
                target_repo_id="org/api-service",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        graph = RepoDependencyGraph(
            registry=registry,
            store=graph_store,
        )

        cycles = graph.detect_cycles()
        assert len(cycles) >= 1

    def test_get_dependency_tree(self, graph: RepoDependencyGraph):
        """Test getting dependency tree."""
        tree = graph.get_dependency_tree("org/api-service")

        assert tree["repo_id"] == "org/api-service"
        assert len(tree["dependencies"]) == 1
        assert tree["dependencies"][0]["repo_id"] == "org/shared-lib"
        assert len(tree["dependencies"][0]["dependencies"]) == 1
        assert tree["dependencies"][0]["dependencies"][0]["repo_id"] == "org/core-utils"

    def test_get_dependency_tree_with_depth(self, graph: RepoDependencyGraph):
        """Test getting dependency tree with depth limit."""
        tree = graph.get_dependency_tree("org/api-service", max_depth=1)

        assert tree["repo_id"] == "org/api-service"
        assert len(tree["dependencies"]) == 1
        # Should not have nested dependencies due to depth limit
        assert tree["dependencies"][0]["dependencies"] == []

    def test_topological_sort(self, graph: RepoDependencyGraph):
        """Test topological sorting of repositories."""
        sorted_repos = graph.topological_sort()

        # core-utils should come before shared-lib
        # shared-lib should come before api-service
        # api-service should come before frontend
        core_idx = sorted_repos.index("org/core-utils")
        shared_idx = sorted_repos.index("org/shared-lib")
        api_idx = sorted_repos.index("org/api-service")
        frontend_idx = sorted_repos.index("org/frontend")

        assert core_idx < shared_idx
        assert shared_idx < api_idx
        assert api_idx < frontend_idx

    def test_topological_sort_with_cycle_raises(
        self,
        registry: InMemoryRepositoryRegistry,
        graph_store: InMemoryDependencyGraphStore,
    ):
        """Test topological sort raises on cycle."""
        # Add a cycle
        graph_store.add_dependency(
            RepoDependency(
                source_repo_id="org/core-utils",
                target_repo_id="org/api-service",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        graph = RepoDependencyGraph(
            registry=registry,
            store=graph_store,
        )

        with pytest.raises(CyclicDependencyError):
            graph.topological_sort()

    def test_get_affected_repos(self, graph: RepoDependencyGraph):
        """Test getting repositories affected by a change."""
        affected = graph.get_affected_repos("org/core-utils")

        # Changes to core-utils affect shared-lib and api-service
        assert "org/shared-lib" in affected
        assert "org/api-service" in affected
        # frontend depends on api-service via API
        assert "org/frontend" in affected

    def test_get_dependency_stats(self, graph: RepoDependencyGraph):
        """Test getting dependency statistics."""
        stats = graph.get_stats()

        assert stats["total_dependencies"] == 3
        assert stats["total_repos"] == 4
        assert "by_type" in stats
        assert "runtime" in stats["by_type"]

    def test_add_dependency(self, graph: RepoDependencyGraph):
        """Test adding a dependency through the graph."""
        graph.add_dependency(
            source_repo_id="org/api-service",
            target_repo_id="org/frontend",
            dependency_type=DependencyType.DEV,
        )

        deps = graph.get_direct_dependencies("org/api-service")
        target_repos = [d.target_repo_id for d in deps]
        assert "org/frontend" in target_repos

    def test_remove_dependency(self, graph: RepoDependencyGraph):
        """Test removing a dependency."""
        # First verify dependency exists
        assert graph.has_path("org/api-service", "org/shared-lib")

        graph.remove_dependency("org/api-service", "org/shared-lib")

        # Now it should not exist
        deps = graph.get_direct_dependencies("org/api-service")
        target_repos = [d.target_repo_id for d in deps]
        assert "org/shared-lib" not in target_repos


# =============================================================================
# Exception Tests
# =============================================================================


class TestDependencyGraphExceptions:
    """Tests for dependency graph exceptions."""

    def test_dependency_graph_error(self):
        """Test base dependency graph error."""
        error = DependencyGraphError("Test error")
        assert str(error) == "Test error"

    def test_cyclic_dependency_error(self):
        """Test cyclic dependency error."""
        cycle = ["org/a", "org/b", "org/c", "org/a"]
        error = CyclicDependencyError(cycle)
        assert "cyclic" in str(error).lower()
        assert error.cycle == cycle
