"""Impact Analysis Tool.

This module provides the impact_analysis MCP tool:
- ImpactAnalysisConfig: Configuration for the tool
- ImpactInput: Input parameters
- ImpactOutput: Result structure with affected files
- ImpactAnalysisTool: Main tool entry point

Integration points:
- openmemory.api.indexing.graph_projection: CODE_* graph queries
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from openmemory.api.indexing.graph_projection import CodeEdgeType

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ImpactAnalysisError(Exception):
    """Base exception for impact analysis tool errors."""

    pass


class InvalidInputError(ImpactAnalysisError):
    """Raised when input is invalid."""

    pass


# =============================================================================
# Configuration
# =============================================================================


# Confidence level thresholds
CONFIDENCE_THRESHOLDS = {
    "definite": 0.9,
    "probable": 0.5,
    "possible": 0.0,
}


@dataclass
class ImpactAnalysisConfig:
    """Configuration for impact_analysis tool.

    Args:
        max_depth: Maximum traversal depth (1-10)
        confidence_threshold: Minimum confidence level ("definite", "probable", "possible")
        include_cross_language: Include cross-language dependencies
        max_affected_files: Maximum affected files to return
    """

    max_depth: int = 3
    confidence_threshold: str = "probable"  # "definite", "probable", "possible"
    include_cross_language: bool = False
    max_affected_files: int = 100


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class ImpactInput:
    """Input parameters for impact_analysis.

    At least one of changed_files or symbol_id must be provided.

    Args:
        repo_id: Repository ID (required)
        changed_files: List of changed file paths
        symbol_id: Symbol ID for single symbol analysis
        include_cross_language: Override config for cross-language
        max_depth: Override config for max depth
        confidence_threshold: Override config for confidence
    """

    repo_id: str = ""
    changed_files: list[str] = field(default_factory=list)
    symbol_id: Optional[str] = None
    include_cross_language: Optional[bool] = None
    max_depth: Optional[int] = None
    confidence_threshold: Optional[str] = None


# =============================================================================
# Result Types (MCP Schema Compliant)
# =============================================================================


@dataclass
class AffectedFile:
    """An affected file from impact analysis.

    Maps to affected_files items in MCP schema.
    """

    file_path: str
    reason: str
    confidence: float


@dataclass
class ResponseMeta:
    """Response metadata.

    Maps to ResponseMeta in MCP schema.
    """

    request_id: str
    degraded_mode: bool = False
    missing_sources: list[str] = field(default_factory=list)


@dataclass
class ImpactOutput:
    """Result from impact_analysis.

    Maps to ImpactOutput in MCP schema.
    """

    affected_files: list[AffectedFile]
    meta: ResponseMeta


# =============================================================================
# Main Tool
# =============================================================================


class ImpactAnalysisTool:
    """MCP tool for analyzing impact of changes.

    Traverses the code graph to find files affected by changes
    to a symbol or set of files.
    """

    def __init__(
        self,
        graph_driver: Any,
        config: Optional[ImpactAnalysisConfig] = None,
    ):
        """Initialize impact_analysis tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.config = config or ImpactAnalysisConfig()

    def analyze(
        self,
        input_data: ImpactInput,
        config: Optional[ImpactAnalysisConfig] = None,
    ) -> ImpactOutput:
        """Analyze impact of changes.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            ImpactOutput with affected files

        Raises:
            InvalidInputError: If input is invalid
            ImpactAnalysisError: If analysis fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Determine effective settings
        max_depth = input_data.max_depth or cfg.max_depth
        confidence_threshold = input_data.confidence_threshold or cfg.confidence_threshold
        min_confidence = CONFIDENCE_THRESHOLDS.get(confidence_threshold, 0.5)

        # Collect affected files
        affected_map: dict[str, AffectedFile] = {}

        try:
            if input_data.symbol_id:
                # Analyze single symbol impact
                self._analyze_symbol_impact(
                    symbol_id=input_data.symbol_id,
                    depth=max_depth,
                    affected_map=affected_map,
                    max_files=cfg.max_affected_files,
                )
            elif input_data.changed_files:
                # Analyze file changes
                self._analyze_file_changes(
                    file_paths=input_data.changed_files,
                    depth=max_depth,
                    affected_map=affected_map,
                    max_files=cfg.max_affected_files,
                )
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            raise ImpactAnalysisError(f"Analysis failed: {e}") from e

        # Filter by confidence threshold
        affected_files = [
            af for af in affected_map.values()
            if af.confidence >= min_confidence
        ]

        # Sort by confidence descending
        affected_files.sort(key=lambda x: x.confidence, reverse=True)

        # Limit results
        affected_files = affected_files[: cfg.max_affected_files]

        return ImpactOutput(
            affected_files=affected_files,
            meta=ResponseMeta(request_id=request_id),
        )

    def _validate_input(self, input_data: ImpactInput) -> None:
        """Validate input parameters."""
        if not input_data.repo_id or not input_data.repo_id.strip():
            raise InvalidInputError("repo_id is required")

        if not input_data.changed_files and not input_data.symbol_id:
            raise InvalidInputError(
                "Either changed_files or symbol_id is required"
            )

    def _analyze_symbol_impact(
        self,
        symbol_id: str,
        depth: int,
        affected_map: dict[str, AffectedFile],
        max_files: int,
    ) -> None:
        """Analyze impact of changes to a symbol.

        Traverses callers to find affected files.
        """
        # Verify symbol exists
        node = self.graph_driver.get_node(symbol_id)
        if node is None:
            raise ImpactAnalysisError(f"Symbol not found: {symbol_id}")

        # The file containing the symbol is directly affected
        file_path = node.properties.get("file_path")
        if file_path:
            affected_map[file_path] = AffectedFile(
                file_path=file_path,
                reason="Contains changed symbol",
                confidence=1.0,
            )

        # Find callers (things that depend on this symbol)
        visited: set[str] = set()
        self._traverse_callers(
            symbol_id=symbol_id,
            depth=depth,
            current_depth=1,
            affected_map=affected_map,
            visited=visited,
            max_files=max_files,
        )

    def _traverse_callers(
        self,
        symbol_id: str,
        depth: int,
        current_depth: int,
        affected_map: dict[str, AffectedFile],
        visited: set[str],
        max_files: int,
    ) -> None:
        """Recursively traverse callers to find affected files."""
        if current_depth > depth or symbol_id in visited:
            return

        if len(affected_map) >= max_files:
            return

        visited.add(symbol_id)

        # Get incoming CALLS edges
        if not hasattr(self.graph_driver, "get_incoming_edges"):
            return

        incoming_edges = self.graph_driver.get_incoming_edges(symbol_id)

        for edge in incoming_edges:
            edge_type_value = (
                edge.edge_type.value
                if hasattr(edge.edge_type, "value")
                else str(edge.edge_type)
            )

            if edge_type_value != "CALLS":
                continue

            caller_id = edge.source_id

            if caller_id in visited:
                continue

            # Get caller node
            caller_node = self.graph_driver.get_node(caller_id)
            if caller_node:
                file_path = caller_node.properties.get("file_path")
                if file_path and file_path not in affected_map:
                    # Calculate confidence based on depth
                    confidence = self._calculate_confidence(current_depth)

                    affected_map[file_path] = AffectedFile(
                        file_path=file_path,
                        reason=f"Calls changed symbol (depth {current_depth})",
                        confidence=confidence,
                    )

            # Recurse
            if len(affected_map) < max_files:
                self._traverse_callers(
                    symbol_id=caller_id,
                    depth=depth,
                    current_depth=current_depth + 1,
                    affected_map=affected_map,
                    visited=visited,
                    max_files=max_files,
                )

    def _analyze_file_changes(
        self,
        file_paths: list[str],
        depth: int,
        affected_map: dict[str, AffectedFile],
        max_files: int,
    ) -> None:
        """Analyze impact of file changes.

        For each file, find symbols and their callers.
        """
        for file_path in file_paths:
            # The changed file itself is affected
            if file_path not in affected_map:
                affected_map[file_path] = AffectedFile(
                    file_path=file_path,
                    reason="Directly changed",
                    confidence=1.0,
                )

            # Find symbols in this file
            file_node = self.graph_driver.get_node(file_path)
            if not file_node and hasattr(self.graph_driver, "find_file_id"):
                fallback_id = self.graph_driver.find_file_id(
                    file_path,
                    repo_id=getattr(input_data, "repo_id", None),
                )
                if fallback_id:
                    file_node = self.graph_driver.get_node(fallback_id)

            if not file_node:
                continue

            # Get CONTAINS edges to find symbols
            try:
                outgoing_edges = self.graph_driver.get_outgoing_edges(file_path)
                for edge in outgoing_edges:
                    edge_type_value = (
                        edge.edge_type.value
                        if hasattr(edge.edge_type, "value")
                        else str(edge.edge_type)
                    )

                    if edge_type_value == "CONTAINS":
                        symbol_id = edge.target_id
                        # Find callers of this symbol
                        visited: set[str] = set()
                        self._traverse_callers(
                            symbol_id=symbol_id,
                            depth=depth,
                            current_depth=1,
                            affected_map=affected_map,
                            visited=visited,
                            max_files=max_files,
                        )
            except Exception as e:
                logger.warning(f"Failed to analyze file {file_path}: {e}")

    def _calculate_confidence(self, depth: int) -> float:
        """Calculate confidence based on call depth.

        Direct callers (depth 1) have higher confidence.
        Transitive callers have lower confidence.
        """
        if depth == 1:
            return 0.95  # Direct caller - very likely affected
        elif depth == 2:
            return 0.8  # One level removed
        elif depth == 3:
            return 0.6  # Two levels removed
        else:
            return max(0.3, 1.0 - (depth * 0.15))  # Decreasing confidence


# =============================================================================
# Factory Function
# =============================================================================


def create_impact_analysis_tool(
    graph_driver: Any,
    config: Optional[ImpactAnalysisConfig] = None,
) -> ImpactAnalysisTool:
    """Create an impact_analysis tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        config: Optional configuration

    Returns:
        Configured ImpactAnalysisTool
    """
    return ImpactAnalysisTool(
        graph_driver=graph_driver,
        config=config,
    )
