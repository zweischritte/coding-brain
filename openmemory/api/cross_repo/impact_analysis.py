"""Cross-repository impact analysis.

This module provides impact analysis across repositories for detecting
breaking changes and understanding the blast radius of code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .registry import RepositoryRegistry
from .symbol_resolution import SymbolStore
from .dependency_graph import DependencyGraphStore, RepoDependencyGraph


# =============================================================================
# Exceptions
# =============================================================================


class CrossRepoImpactError(Exception):
    """Base exception for cross-repo impact analysis errors."""

    pass


# =============================================================================
# Enums
# =============================================================================


class ChangeSeverity(str, Enum):
    """Severity level of a change."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def priority(self) -> int:
        """Get priority value for ordering (higher = more severe)."""
        priority_map = {
            ChangeSeverity.CRITICAL: 5,
            ChangeSeverity.HIGH: 4,
            ChangeSeverity.MEDIUM: 3,
            ChangeSeverity.LOW: 2,
            ChangeSeverity.INFO: 1,
        }
        return priority_map[self]

    @property
    def is_breaking(self) -> bool:
        """Check if this severity indicates a breaking change."""
        return self in (ChangeSeverity.CRITICAL, ChangeSeverity.HIGH)


class ChangeType(str, Enum):
    """Type of change to a symbol."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"

    @property
    def is_breaking(self) -> bool:
        """Check if this change type is potentially breaking."""
        return self in (ChangeType.REMOVED, ChangeType.RENAMED)


class BreakingChangeType(str, Enum):
    """Type of breaking change."""

    SIGNATURE_CHANGE = "signature_change"
    REMOVAL = "removal"
    RENAME = "rename"
    TYPE_CHANGE = "type_change"
    BEHAVIOR_CHANGE = "behavior_change"
    VISIBILITY_CHANGE = "visibility_change"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SymbolChange:
    """A change to a symbol."""

    symbol_id: str
    repo_id: str
    change_type: ChangeType
    old_signature: str = ""
    new_signature: str = ""
    file_path: str = ""
    line_number: int = 0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol_id": self.symbol_id,
            "repo_id": self.repo_id,
            "change_type": self.change_type.value,
            "old_signature": self.old_signature,
            "new_signature": self.new_signature,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class BreakingChange:
    """A detected breaking change."""

    change_type: BreakingChangeType
    severity: ChangeSeverity
    symbol_id: str
    source_repo_id: str
    description: str = ""
    affected_repos: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)
    migration_guide: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type.value,
            "severity": self.severity.value,
            "symbol_id": self.symbol_id,
            "source_repo_id": self.source_repo_id,
            "description": self.description,
            "affected_repos": self.affected_repos,
            "affected_symbols": self.affected_symbols,
            "migration_guide": self.migration_guide,
            "metadata": self.metadata,
        }


@dataclass
class AffectedRepository:
    """A repository affected by changes."""

    repo_id: str
    impact_severity: ChangeSeverity
    affected_symbols: list[str] = field(default_factory=list)
    breaking_changes: int = 0
    dependency_path: list[str] = field(default_factory=list)
    distance: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossRepoImpactConfig:
    """Configuration for cross-repo impact analysis."""

    max_depth: int = 5
    include_transitive: bool = True
    min_severity: ChangeSeverity = ChangeSeverity.LOW
    detect_breaking_changes: bool = True


@dataclass
class CrossRepoImpactInput:
    """Input for impact analysis."""

    source_repo_id: str
    symbol_changes: list[SymbolChange]
    from_version: str = ""
    to_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossRepoImpactOutput:
    """Output of impact analysis."""

    source_repo_id: str
    affected_repos: list[AffectedRepository]
    breaking_changes: list[BreakingChange]
    total_affected_repos: int
    total_breaking_changes: int = 0
    analysis_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_breaking_changes(self) -> bool:
        """Check if there are any breaking changes."""
        return len(self.breaking_changes) > 0

    @property
    def max_severity(self) -> ChangeSeverity:
        """Get the maximum severity across all affected repos."""
        if not self.affected_repos:
            return ChangeSeverity.INFO

        return max(
            self.affected_repos,
            key=lambda r: r.impact_severity.priority,
        ).impact_severity


# =============================================================================
# Cross-Repository Impact Analyzer
# =============================================================================


class CrossRepoImpactAnalyzer:
    """Analyzer for cross-repository impact analysis."""

    def __init__(
        self,
        registry: RepositoryRegistry,
        symbol_store: SymbolStore,
        dependency_store: DependencyGraphStore,
        config: Optional[CrossRepoImpactConfig] = None,
    ):
        self._registry = registry
        self._symbol_store = symbol_store
        self._dependency_store = dependency_store
        self._config = config or CrossRepoImpactConfig()

        # Create dependency graph wrapper
        self._dep_graph = RepoDependencyGraph(
            registry=registry,
            store=dependency_store,
        )

    @property
    def config(self) -> CrossRepoImpactConfig:
        """Get analyzer configuration."""
        return self._config

    def analyze(self, input_data: CrossRepoImpactInput) -> CrossRepoImpactOutput:
        """Analyze the impact of changes across repositories."""
        all_affected: list[AffectedRepository] = []
        all_breaking: list[BreakingChange] = []

        for change in input_data.symbol_changes:
            # Get affected repos for this symbol
            affected = self.get_affected_repos(
                symbol_id=change.symbol_id,
                include_transitive=self._config.include_transitive,
            )
            all_affected.extend(affected)

            # Detect breaking changes if enabled
            if self._config.detect_breaking_changes:
                breaking = self.detect_breaking_changes([change])
                all_breaking.extend(breaking)

        # Deduplicate affected repos
        unique_affected: dict[str, AffectedRepository] = {}
        for affected in all_affected:
            if affected.repo_id not in unique_affected:
                unique_affected[affected.repo_id] = affected
            else:
                # Keep the one with higher severity
                existing = unique_affected[affected.repo_id]
                if affected.impact_severity.priority > existing.impact_severity.priority:
                    unique_affected[affected.repo_id] = affected

        return CrossRepoImpactOutput(
            source_repo_id=input_data.source_repo_id,
            affected_repos=list(unique_affected.values()),
            breaking_changes=all_breaking,
            total_affected_repos=len(unique_affected),
            total_breaking_changes=len(all_breaking),
            analysis_depth=self._config.max_depth,
        )

    def get_affected_repos(
        self,
        symbol_id: str,
        include_transitive: bool = True,
    ) -> list[AffectedRepository]:
        """Get repositories affected by a change to the given symbol."""
        affected: list[AffectedRepository] = []

        # Get symbol info
        symbol = self._symbol_store.get(symbol_id)
        if not symbol:
            return affected

        # Get direct usages via symbol mappings
        mappings = self._symbol_store.get_mappings(symbol_id)
        direct_repos: set[str] = set()

        for mapping in mappings:
            if mapping.target_repo_id != symbol.repo_id:
                direct_repos.add(mapping.target_repo_id)

        # Also check via dependency graph
        dependents = self._dep_graph.get_direct_dependents(symbol.repo_id)
        for dep in dependents:
            direct_repos.add(dep.source_repo_id)

        # Add direct affected repos
        for repo_id in direct_repos:
            affected.append(
                AffectedRepository(
                    repo_id=repo_id,
                    impact_severity=ChangeSeverity.HIGH,
                    affected_symbols=[symbol_id],
                    distance=1,
                )
            )

        # Get transitive dependencies if requested
        if include_transitive:
            all_dependents = self._dep_graph.get_all_dependents(
                symbol.repo_id,
                max_depth=self._config.max_depth,
            )

            for dep in all_dependents:
                if dep.source_repo_id not in direct_repos:
                    # Calculate distance
                    path = self.get_dependency_path(
                        symbol.repo_id, dep.source_repo_id
                    )
                    distance = len(path) + 1

                    affected.append(
                        AffectedRepository(
                            repo_id=dep.source_repo_id,
                            impact_severity=ChangeSeverity.MEDIUM,
                            affected_symbols=[],  # Transitively affected
                            distance=distance,
                            dependency_path=[d.source_repo_id for d in path],
                        )
                    )

        return affected

    def detect_breaking_changes(
        self, changes: list[SymbolChange]
    ) -> list[BreakingChange]:
        """Detect breaking changes from a list of symbol changes."""
        breaking: list[BreakingChange] = []

        for change in changes:
            # Removal is always breaking
            if change.change_type == ChangeType.REMOVED:
                # Get affected repos
                affected_repos = [
                    a.repo_id for a in self.get_affected_repos(change.symbol_id)
                ]

                breaking.append(
                    BreakingChange(
                        change_type=BreakingChangeType.REMOVAL,
                        severity=ChangeSeverity.CRITICAL,
                        symbol_id=change.symbol_id,
                        source_repo_id=change.repo_id,
                        description=f"Symbol '{change.symbol_id}' was removed",
                        affected_repos=affected_repos,
                    )
                )

            # Rename is breaking
            elif change.change_type == ChangeType.RENAMED:
                affected_repos = [
                    a.repo_id for a in self.get_affected_repos(change.symbol_id)
                ]

                breaking.append(
                    BreakingChange(
                        change_type=BreakingChangeType.RENAME,
                        severity=ChangeSeverity.HIGH,
                        symbol_id=change.symbol_id,
                        source_repo_id=change.repo_id,
                        description=f"Symbol '{change.symbol_id}' was renamed",
                        affected_repos=affected_repos,
                    )
                )

            # Modification might be breaking (signature change)
            elif change.change_type == ChangeType.MODIFIED:
                if self._is_signature_breaking(
                    change.old_signature, change.new_signature
                ):
                    affected_repos = [
                        a.repo_id for a in self.get_affected_repos(change.symbol_id)
                    ]

                    breaking.append(
                        BreakingChange(
                            change_type=BreakingChangeType.SIGNATURE_CHANGE,
                            severity=ChangeSeverity.HIGH,
                            symbol_id=change.symbol_id,
                            source_repo_id=change.repo_id,
                            description="Function signature changed in a breaking way",
                            affected_repos=affected_repos,
                        )
                    )

        return breaking

    def calculate_impact_severity(
        self,
        change_type: ChangeType,
        usage_count: int,
        is_public: bool,
    ) -> ChangeSeverity:
        """Calculate the severity of an impact."""
        # Adding is always info
        if change_type == ChangeType.ADDED:
            return ChangeSeverity.INFO

        # Private symbols have lower impact
        if not is_public:
            if change_type == ChangeType.REMOVED:
                return ChangeSeverity.LOW
            return ChangeSeverity.INFO

        # Removal is critical if widely used
        if change_type == ChangeType.REMOVED:
            if usage_count >= 5:
                return ChangeSeverity.CRITICAL
            elif usage_count >= 2:
                return ChangeSeverity.HIGH
            else:
                return ChangeSeverity.MEDIUM

        # Rename depends on usage
        if change_type == ChangeType.RENAMED:
            if usage_count >= 5:
                return ChangeSeverity.HIGH
            else:
                return ChangeSeverity.MEDIUM

        # Modification depends on usage
        if usage_count >= 5:
            return ChangeSeverity.MEDIUM
        else:
            return ChangeSeverity.LOW

    def get_dependency_path(
        self,
        source_repo_id: str,
        target_repo_id: str,
    ) -> list:
        """Get the dependency path between two repositories."""
        return self._dep_graph.get_dependency_path(source_repo_id, target_repo_id)

    def _is_signature_breaking(
        self,
        old_signature: str,
        new_signature: str,
    ) -> bool:
        """Check if a signature change is breaking.

        This is a simplified check. A more complete implementation would
        parse the signatures and compare:
        - Required parameters added
        - Parameter types changed
        - Return type changed
        - Required parameters removed (might affect callers with positional args)
        """
        if not old_signature or not new_signature:
            return False

        # Simple heuristics:
        # 1. If the new signature is shorter (parameters removed), might be breaking
        # 2. If parameters are added without defaults, it's breaking

        # Extract parameter counts (very simplified)
        old_params = old_signature.count(",") + (
            1 if "(" in old_signature and ")" in old_signature else 0
        )
        new_params = new_signature.count(",") + (
            1 if "(" in new_signature and ")" in new_signature else 0
        )

        # Adding required parameters (without default) is breaking
        # This is a simplified check - in reality we'd parse the signature
        if new_params > old_params:
            # Check if new params have defaults
            if "=" not in new_signature.split(",")[-1]:
                return True

        return False
