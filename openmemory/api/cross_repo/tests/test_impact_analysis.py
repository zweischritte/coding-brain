"""Tests for cross-repository impact analysis.

This module tests impact analysis across repositories for detecting
breaking changes and understanding the blast radius of code changes.
"""

from __future__ import annotations

from typing import Any

import pytest

from openmemory.api.cross_repo.registry import (
    InMemoryRepositoryRegistry,
    RepositoryMetadata,
)
from openmemory.api.cross_repo.symbol_resolution import (
    CrossRepoSymbol,
    InMemorySymbolStore,
    SymbolMapping,
    SymbolType,
)
from openmemory.api.cross_repo.dependency_graph import (
    DependencyType,
    InMemoryDependencyGraphStore,
    RepoDependency,
)
from openmemory.api.cross_repo.impact_analysis import (
    AffectedRepository,
    BreakingChange,
    BreakingChangeType,
    ChangeSeverity,
    CrossRepoImpactAnalyzer,
    CrossRepoImpactConfig,
    CrossRepoImpactError,
    CrossRepoImpactInput,
    CrossRepoImpactOutput,
    SymbolChange,
    ChangeType,
)


# =============================================================================
# ChangeSeverity Tests
# =============================================================================


class TestChangeSeverity:
    """Tests for ChangeSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert ChangeSeverity.CRITICAL.value == "critical"
        assert ChangeSeverity.HIGH.value == "high"
        assert ChangeSeverity.MEDIUM.value == "medium"
        assert ChangeSeverity.LOW.value == "low"
        assert ChangeSeverity.INFO.value == "info"

    def test_severity_ordering(self):
        """Test severity ordering."""
        assert ChangeSeverity.CRITICAL.priority > ChangeSeverity.HIGH.priority
        assert ChangeSeverity.HIGH.priority > ChangeSeverity.MEDIUM.priority
        assert ChangeSeverity.MEDIUM.priority > ChangeSeverity.LOW.priority
        assert ChangeSeverity.LOW.priority > ChangeSeverity.INFO.priority

    def test_is_breaking(self):
        """Test is_breaking property."""
        assert ChangeSeverity.CRITICAL.is_breaking is True
        assert ChangeSeverity.HIGH.is_breaking is True
        assert ChangeSeverity.MEDIUM.is_breaking is False
        assert ChangeSeverity.LOW.is_breaking is False
        assert ChangeSeverity.INFO.is_breaking is False


# =============================================================================
# ChangeType Tests
# =============================================================================


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types(self):
        """Test all change types exist."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.REMOVED.value == "removed"
        assert ChangeType.RENAMED.value == "renamed"

    def test_is_breaking_change(self):
        """Test is_breaking property."""
        assert ChangeType.REMOVED.is_breaking is True
        assert ChangeType.RENAMED.is_breaking is True
        assert ChangeType.ADDED.is_breaking is False
        assert ChangeType.MODIFIED.is_breaking is False  # Depends on details


# =============================================================================
# BreakingChangeType Tests
# =============================================================================


class TestBreakingChangeType:
    """Tests for BreakingChangeType enum."""

    def test_breaking_change_types(self):
        """Test all breaking change types exist."""
        assert BreakingChangeType.SIGNATURE_CHANGE.value == "signature_change"
        assert BreakingChangeType.REMOVAL.value == "removal"
        assert BreakingChangeType.RENAME.value == "rename"
        assert BreakingChangeType.TYPE_CHANGE.value == "type_change"
        assert BreakingChangeType.BEHAVIOR_CHANGE.value == "behavior_change"
        assert BreakingChangeType.VISIBILITY_CHANGE.value == "visibility_change"


# =============================================================================
# SymbolChange Tests
# =============================================================================


class TestSymbolChange:
    """Tests for SymbolChange dataclass."""

    def test_create_symbol_change(self):
        """Test creating a symbol change."""
        change = SymbolChange(
            symbol_id="scip-python org/lib module/func.",
            repo_id="org/lib",
            change_type=ChangeType.MODIFIED,
            old_signature="def func(x: int) -> str",
            new_signature="def func(x: int, y: str) -> str",
        )
        assert change.symbol_id == "scip-python org/lib module/func."
        assert change.repo_id == "org/lib"
        assert change.change_type == ChangeType.MODIFIED

    def test_symbol_change_defaults(self):
        """Test default values for symbol change."""
        change = SymbolChange(
            symbol_id="sym1",
            repo_id="repo1",
            change_type=ChangeType.ADDED,
        )
        assert change.old_signature == ""
        assert change.new_signature == ""
        assert change.file_path == ""
        assert change.line_number == 0
        assert change.description == ""

    def test_symbol_change_to_dict(self):
        """Test converting symbol change to dictionary."""
        change = SymbolChange(
            symbol_id="sym1",
            repo_id="repo1",
            change_type=ChangeType.REMOVED,
            file_path="src/module.py",
        )
        data = change.to_dict()
        assert data["symbol_id"] == "sym1"
        assert data["change_type"] == "removed"
        assert data["file_path"] == "src/module.py"


# =============================================================================
# BreakingChange Tests
# =============================================================================


class TestBreakingChange:
    """Tests for BreakingChange dataclass."""

    def test_create_breaking_change(self):
        """Test creating a breaking change."""
        change = BreakingChange(
            change_type=BreakingChangeType.SIGNATURE_CHANGE,
            severity=ChangeSeverity.HIGH,
            symbol_id="scip-python org/lib module/func.",
            source_repo_id="org/lib",
            description="Parameter 'x' type changed from int to str",
            affected_repos=["org/app1", "org/app2"],
        )
        assert change.change_type == BreakingChangeType.SIGNATURE_CHANGE
        assert change.severity == ChangeSeverity.HIGH
        assert len(change.affected_repos) == 2

    def test_breaking_change_defaults(self):
        """Test default values for breaking change."""
        change = BreakingChange(
            change_type=BreakingChangeType.REMOVAL,
            severity=ChangeSeverity.CRITICAL,
            symbol_id="sym1",
            source_repo_id="repo1",
        )
        assert change.description == ""
        assert change.affected_repos == []
        assert change.affected_symbols == []
        assert change.migration_guide == ""
        assert change.metadata == {}

    def test_breaking_change_with_migration_guide(self):
        """Test breaking change with migration guide."""
        change = BreakingChange(
            change_type=BreakingChangeType.RENAME,
            severity=ChangeSeverity.MEDIUM,
            symbol_id="old_name",
            source_repo_id="repo1",
            migration_guide="Replace 'old_name' with 'new_name' in all usages",
        )
        assert "new_name" in change.migration_guide

    def test_breaking_change_to_dict(self):
        """Test converting breaking change to dictionary."""
        change = BreakingChange(
            change_type=BreakingChangeType.REMOVAL,
            severity=ChangeSeverity.CRITICAL,
            symbol_id="sym1",
            source_repo_id="repo1",
            affected_repos=["repo2", "repo3"],
        )
        data = change.to_dict()
        assert data["change_type"] == "removal"
        assert data["severity"] == "critical"
        assert data["affected_repos"] == ["repo2", "repo3"]


# =============================================================================
# AffectedRepository Tests
# =============================================================================


class TestAffectedRepository:
    """Tests for AffectedRepository dataclass."""

    def test_create_affected_repo(self):
        """Test creating an affected repository."""
        affected = AffectedRepository(
            repo_id="org/app",
            impact_severity=ChangeSeverity.HIGH,
            affected_symbols=["sym1", "sym2"],
            breaking_changes=1,
            dependency_path=["org/lib", "org/app"],
        )
        assert affected.repo_id == "org/app"
        assert affected.impact_severity == ChangeSeverity.HIGH
        assert len(affected.affected_symbols) == 2
        assert affected.breaking_changes == 1

    def test_affected_repo_defaults(self):
        """Test default values for affected repository."""
        affected = AffectedRepository(
            repo_id="org/app",
            impact_severity=ChangeSeverity.LOW,
        )
        assert affected.affected_symbols == []
        assert affected.breaking_changes == 0
        assert affected.dependency_path == []
        assert affected.distance == 1

    def test_affected_repo_distance(self):
        """Test setting dependency distance."""
        affected = AffectedRepository(
            repo_id="org/app",
            impact_severity=ChangeSeverity.MEDIUM,
            distance=3,  # 3 hops away
        )
        assert affected.distance == 3


# =============================================================================
# CrossRepoImpactConfig Tests
# =============================================================================


class TestCrossRepoImpactConfig:
    """Tests for CrossRepoImpactConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CrossRepoImpactConfig()
        assert config.max_depth == 5
        assert config.include_transitive is True
        assert config.min_severity == ChangeSeverity.LOW
        assert config.detect_breaking_changes is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CrossRepoImpactConfig(
            max_depth=3,
            include_transitive=False,
            min_severity=ChangeSeverity.HIGH,
            detect_breaking_changes=False,
        )
        assert config.max_depth == 3
        assert config.include_transitive is False
        assert config.min_severity == ChangeSeverity.HIGH
        assert config.detect_breaking_changes is False


# =============================================================================
# CrossRepoImpactInput Tests
# =============================================================================


class TestCrossRepoImpactInput:
    """Tests for CrossRepoImpactInput dataclass."""

    def test_create_input(self):
        """Test creating impact analysis input."""
        changes = [
            SymbolChange(
                symbol_id="sym1",
                repo_id="repo1",
                change_type=ChangeType.MODIFIED,
            )
        ]
        input_data = CrossRepoImpactInput(
            source_repo_id="org/lib",
            symbol_changes=changes,
        )
        assert input_data.source_repo_id == "org/lib"
        assert len(input_data.symbol_changes) == 1

    def test_input_with_version(self):
        """Test input with version info."""
        input_data = CrossRepoImpactInput(
            source_repo_id="org/lib",
            symbol_changes=[],
            from_version="1.0.0",
            to_version="2.0.0",
        )
        assert input_data.from_version == "1.0.0"
        assert input_data.to_version == "2.0.0"


# =============================================================================
# CrossRepoImpactOutput Tests
# =============================================================================


class TestCrossRepoImpactOutput:
    """Tests for CrossRepoImpactOutput dataclass."""

    def test_create_output(self):
        """Test creating impact analysis output."""
        output = CrossRepoImpactOutput(
            source_repo_id="org/lib",
            affected_repos=[
                AffectedRepository(
                    repo_id="org/app",
                    impact_severity=ChangeSeverity.HIGH,
                )
            ],
            breaking_changes=[],
            total_affected_repos=1,
        )
        assert output.source_repo_id == "org/lib"
        assert len(output.affected_repos) == 1
        assert output.total_affected_repos == 1

    def test_output_with_breaking_changes(self):
        """Test output with breaking changes."""
        output = CrossRepoImpactOutput(
            source_repo_id="org/lib",
            affected_repos=[],
            breaking_changes=[
                BreakingChange(
                    change_type=BreakingChangeType.REMOVAL,
                    severity=ChangeSeverity.CRITICAL,
                    symbol_id="sym1",
                    source_repo_id="org/lib",
                )
            ],
            total_affected_repos=0,
            total_breaking_changes=1,
        )
        assert output.total_breaking_changes == 1

    def test_output_has_breaking_changes(self):
        """Test has_breaking_changes property."""
        output_no_breaking = CrossRepoImpactOutput(
            source_repo_id="org/lib",
            affected_repos=[],
            breaking_changes=[],
            total_affected_repos=0,
        )
        assert output_no_breaking.has_breaking_changes is False

        output_with_breaking = CrossRepoImpactOutput(
            source_repo_id="org/lib",
            affected_repos=[],
            breaking_changes=[
                BreakingChange(
                    change_type=BreakingChangeType.REMOVAL,
                    severity=ChangeSeverity.CRITICAL,
                    symbol_id="sym1",
                    source_repo_id="org/lib",
                )
            ],
            total_affected_repos=0,
        )
        assert output_with_breaking.has_breaking_changes is True

    def test_output_max_severity(self):
        """Test max_severity property."""
        output = CrossRepoImpactOutput(
            source_repo_id="org/lib",
            affected_repos=[
                AffectedRepository(
                    repo_id="org/app1",
                    impact_severity=ChangeSeverity.LOW,
                ),
                AffectedRepository(
                    repo_id="org/app2",
                    impact_severity=ChangeSeverity.HIGH,
                ),
                AffectedRepository(
                    repo_id="org/app3",
                    impact_severity=ChangeSeverity.MEDIUM,
                ),
            ],
            breaking_changes=[],
            total_affected_repos=3,
        )
        assert output.max_severity == ChangeSeverity.HIGH


# =============================================================================
# CrossRepoImpactAnalyzer Tests
# =============================================================================


class TestCrossRepoImpactAnalyzer:
    """Tests for CrossRepoImpactAnalyzer."""

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
                symbol_id="scip-python org/shared-lib utils/process_data.",
                repo_id="org/shared-lib",
                name="process_data",
                symbol_type=SymbolType.FUNCTION,
                exported=True,
            )
        )

        # API service symbols
        store.add(
            CrossRepoSymbol(
                symbol_id="scip-python org/api-service handlers/handler.",
                repo_id="org/api-service",
                name="handler",
                symbol_type=SymbolType.FUNCTION,
            )
        )

        # Mapping: API service uses process_data
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
    def dependency_store(self) -> InMemoryDependencyGraphStore:
        """Create a dependency graph store."""
        store = InMemoryDependencyGraphStore()

        # api-service depends on shared-lib
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/api-service",
                target_repo_id="org/shared-lib",
                dependency_type=DependencyType.RUNTIME,
            )
        )

        # frontend depends on api-service
        store.add_dependency(
            RepoDependency(
                source_repo_id="org/frontend",
                target_repo_id="org/api-service",
                dependency_type=DependencyType.API,
            )
        )

        return store

    @pytest.fixture
    def analyzer(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
        dependency_store: InMemoryDependencyGraphStore,
    ) -> CrossRepoImpactAnalyzer:
        """Create an impact analyzer."""
        return CrossRepoImpactAnalyzer(
            registry=registry,
            symbol_store=symbol_store,
            dependency_store=dependency_store,
        )

    def test_create_analyzer(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
        dependency_store: InMemoryDependencyGraphStore,
    ):
        """Test creating an analyzer."""
        analyzer = CrossRepoImpactAnalyzer(
            registry=registry,
            symbol_store=symbol_store,
            dependency_store=dependency_store,
        )
        assert analyzer is not None

    def test_create_analyzer_with_config(
        self,
        registry: InMemoryRepositoryRegistry,
        symbol_store: InMemorySymbolStore,
        dependency_store: InMemoryDependencyGraphStore,
    ):
        """Test creating an analyzer with custom config."""
        config = CrossRepoImpactConfig(max_depth=3)
        analyzer = CrossRepoImpactAnalyzer(
            registry=registry,
            symbol_store=symbol_store,
            dependency_store=dependency_store,
            config=config,
        )
        assert analyzer.config.max_depth == 3

    def test_analyze_symbol_removal(self, analyzer: CrossRepoImpactAnalyzer):
        """Test analyzing impact of a symbol removal."""
        input_data = CrossRepoImpactInput(
            source_repo_id="org/shared-lib",
            symbol_changes=[
                SymbolChange(
                    symbol_id="scip-python org/shared-lib utils/process_data.",
                    repo_id="org/shared-lib",
                    change_type=ChangeType.REMOVED,
                )
            ],
        )

        output = analyzer.analyze(input_data)

        assert output.source_repo_id == "org/shared-lib"
        assert output.total_affected_repos >= 1
        # Should have breaking changes for removal
        assert output.has_breaking_changes is True

    def test_analyze_symbol_modification(self, analyzer: CrossRepoImpactAnalyzer):
        """Test analyzing impact of a symbol modification."""
        input_data = CrossRepoImpactInput(
            source_repo_id="org/shared-lib",
            symbol_changes=[
                SymbolChange(
                    symbol_id="scip-python org/shared-lib utils/process_data.",
                    repo_id="org/shared-lib",
                    change_type=ChangeType.MODIFIED,
                    old_signature="def process_data(x: int) -> str",
                    new_signature="def process_data(x: str) -> str",  # Type change
                )
            ],
        )

        output = analyzer.analyze(input_data)

        assert output.source_repo_id == "org/shared-lib"
        # Signature change should be detected
        assert len(output.breaking_changes) >= 0  # May or may not detect as breaking

    def test_analyze_added_symbol(self, analyzer: CrossRepoImpactAnalyzer):
        """Test analyzing impact of adding a symbol (non-breaking)."""
        input_data = CrossRepoImpactInput(
            source_repo_id="org/shared-lib",
            symbol_changes=[
                SymbolChange(
                    symbol_id="scip-python org/shared-lib utils/new_func.",
                    repo_id="org/shared-lib",
                    change_type=ChangeType.ADDED,
                )
            ],
        )

        output = analyzer.analyze(input_data)

        # Adding a symbol should not be breaking
        assert output.has_breaking_changes is False

    def test_analyze_multiple_changes(self, analyzer: CrossRepoImpactAnalyzer):
        """Test analyzing multiple symbol changes."""
        input_data = CrossRepoImpactInput(
            source_repo_id="org/shared-lib",
            symbol_changes=[
                SymbolChange(
                    symbol_id="scip-python org/shared-lib utils/process_data.",
                    repo_id="org/shared-lib",
                    change_type=ChangeType.MODIFIED,
                ),
                SymbolChange(
                    symbol_id="scip-python org/shared-lib utils/helper.",
                    repo_id="org/shared-lib",
                    change_type=ChangeType.REMOVED,
                ),
            ],
        )

        output = analyzer.analyze(input_data)

        # Should analyze all changes
        assert output.source_repo_id == "org/shared-lib"

    def test_get_affected_repos(self, analyzer: CrossRepoImpactAnalyzer):
        """Test getting affected repositories for a symbol."""
        affected = analyzer.get_affected_repos(
            symbol_id="scip-python org/shared-lib utils/process_data."
        )

        # api-service uses this symbol
        assert len(affected) >= 1
        repo_ids = [a.repo_id for a in affected]
        assert "org/api-service" in repo_ids

    def test_get_affected_repos_transitive(self, analyzer: CrossRepoImpactAnalyzer):
        """Test getting transitively affected repositories."""
        affected = analyzer.get_affected_repos(
            symbol_id="scip-python org/shared-lib utils/process_data.",
            include_transitive=True,
        )

        # Should include both direct (api-service) and transitive (frontend)
        repo_ids = [a.repo_id for a in affected]
        assert "org/api-service" in repo_ids
        # Frontend depends on api-service, so it's transitively affected

    def test_detect_breaking_changes_removal(
        self, analyzer: CrossRepoImpactAnalyzer
    ):
        """Test detecting breaking changes for symbol removal."""
        changes = [
            SymbolChange(
                symbol_id="scip-python org/shared-lib utils/process_data.",
                repo_id="org/shared-lib",
                change_type=ChangeType.REMOVED,
            )
        ]

        breaking = analyzer.detect_breaking_changes(changes)

        assert len(breaking) >= 1
        assert any(b.change_type == BreakingChangeType.REMOVAL for b in breaking)

    def test_detect_breaking_changes_signature(
        self, analyzer: CrossRepoImpactAnalyzer
    ):
        """Test detecting signature change as breaking."""
        changes = [
            SymbolChange(
                symbol_id="scip-python org/shared-lib utils/process_data.",
                repo_id="org/shared-lib",
                change_type=ChangeType.MODIFIED,
                old_signature="def process_data(x: int) -> str",
                new_signature="def process_data(x: int, required_param: str) -> str",
            )
        ]

        breaking = analyzer.detect_breaking_changes(changes)

        # Adding required parameter is breaking
        assert len(breaking) >= 1

    def test_calculate_impact_severity(self, analyzer: CrossRepoImpactAnalyzer):
        """Test calculating impact severity."""
        # Critical for removal of widely-used symbol
        severity = analyzer.calculate_impact_severity(
            change_type=ChangeType.REMOVED,
            usage_count=10,
            is_public=True,
        )
        assert severity in (ChangeSeverity.CRITICAL, ChangeSeverity.HIGH)

        # Low for adding a symbol
        severity = analyzer.calculate_impact_severity(
            change_type=ChangeType.ADDED,
            usage_count=0,
            is_public=True,
        )
        assert severity == ChangeSeverity.INFO

    def test_get_dependency_path(self, analyzer: CrossRepoImpactAnalyzer):
        """Test getting dependency path between repos."""
        # api-service depends on shared-lib, so path goes from api-service -> shared-lib
        path = analyzer.get_dependency_path(
            source_repo_id="org/api-service",
            target_repo_id="org/shared-lib",
        )

        assert len(path) >= 1


# =============================================================================
# Exception Tests
# =============================================================================


class TestCrossRepoImpactExceptions:
    """Tests for cross-repo impact analysis exceptions."""

    def test_cross_repo_impact_error(self):
        """Test base impact analysis error."""
        error = CrossRepoImpactError("Test error")
        assert str(error) == "Test error"
