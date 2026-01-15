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

try:
    from indexing.graph_projection import CodeEdgeType, CodeNodeType
except ImportError:
    from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType

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

PATH_CONFIDENCE_SCORES = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
}
PATH_MATCH_LIMIT = 50
PARENT_HINT_BOOST = 0.05


@dataclass
class ImpactAnalysisConfig:
    """Configuration for impact_analysis tool.

    Args:
        max_depth: Maximum traversal depth (1-10)
        confidence_threshold: Minimum confidence level ("definite", "probable", "possible")
        include_cross_language: Include cross-language dependencies
        max_affected_files: Maximum affected files to return
        include_inferred_edges: Include edges inferred heuristically
        include_field_edges: Include field/property edges (READS/WRITES/HAS_FIELD)
        include_schema_edges: Include schema exposure edges (SCHEMA_EXPOSES, SCHEMA_ALIASES)
        include_path_edges: Include string path literal references (CODE_FIELD_PATH)
        include_path_edges: Include string path literal references (CODE_FIELD_PATH)
    """

    max_depth: int = 10
    confidence_threshold: str = "probable"  # "definite", "probable", "possible"
    include_cross_language: bool = False
    max_affected_files: int = 100
    include_inferred_edges: bool = True
    include_field_edges: bool = True
    include_schema_edges: bool = True
    include_path_edges: bool = True


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
        symbol_name: Symbol name for lookup when symbol_id is unknown
        parent_name: Parent type name for disambiguation (fields/methods)
        symbol_kind: Symbol kind filter (field|method|class|function|interface|enum|type_alias|property)
        file_path: File path to scope symbol lookup
        include_cross_language: Override config for cross-language
        max_depth: Override config for max depth
        confidence_threshold: Override config for confidence
        include_inferred_edges: Override config for inferred edge usage
        include_field_edges: Include field/property edges (READS/WRITES/HAS_FIELD)
        include_schema_edges: Include schema exposure edges (SCHEMA_EXPOSES, SCHEMA_ALIASES)
    """

    repo_id: str = ""
    changed_files: list[str] = field(default_factory=list)
    symbol_id: Optional[str] = None
    symbol_name: Optional[str] = None
    parent_name: Optional[str] = None
    symbol_kind: Optional[str] = None
    file_path: Optional[str] = None
    include_cross_language: Optional[bool] = None
    max_depth: Optional[int] = None
    confidence_threshold: Optional[str] = None
    include_inferred_edges: Optional[bool] = None
    include_field_edges: Optional[bool] = None
    include_schema_edges: Optional[bool] = None
    include_path_edges: Optional[bool] = None


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
    warnings: list[str] = field(default_factory=list)


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
        include_inferred_edges = (
            input_data.include_inferred_edges
            if input_data.include_inferred_edges is not None
            else cfg.include_inferred_edges
        )
        include_field_edges = (
            input_data.include_field_edges
            if input_data.include_field_edges is not None
            else cfg.include_field_edges
        )
        include_schema_edges = (
            input_data.include_schema_edges
            if input_data.include_schema_edges is not None
            else cfg.include_schema_edges
        )
        include_path_edges = (
            input_data.include_path_edges
            if input_data.include_path_edges is not None
            else cfg.include_path_edges
        )

        # Collect affected files
        affected_map: dict[str, AffectedFile] = {}
        traversal_state = {"used_inferred_edges": False}

        try:
            resolved_symbol_id = self._resolve_symbol_id(input_data)
            if resolved_symbol_id:
                # Analyze single symbol impact
                self._analyze_symbol_impact(
                    symbol_id=resolved_symbol_id,
                    repo_id=input_data.repo_id,
                    depth=max_depth,
                    affected_map=affected_map,
                    max_files=cfg.max_affected_files,
                    include_inferred_edges=include_inferred_edges,
                    include_field_edges=include_field_edges,
                    include_schema_edges=include_schema_edges,
                    include_path_edges=include_path_edges,
                    traversal_state=traversal_state,
                )
            elif input_data.changed_files:
                # Analyze file changes
                self._analyze_file_changes(
                    file_paths=input_data.changed_files,
                    repo_id=input_data.repo_id,
                    depth=max_depth,
                    affected_map=affected_map,
                    max_files=cfg.max_affected_files,
                    include_inferred_edges=include_inferred_edges,
                    include_field_edges=include_field_edges,
                    include_schema_edges=include_schema_edges,
                    include_path_edges=include_path_edges,
                    traversal_state=traversal_state,
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

        meta = ResponseMeta(request_id=request_id)
        if traversal_state["used_inferred_edges"]:
            meta.degraded_mode = True
            meta.missing_sources.append("inferred_call_edges")
        required_files: list[str] = []
        seen_files: set[str] = set()
        for af in affected_files:
            if not af.reason:
                continue
            if (
                "Writes changed field" not in af.reason
                and "Schema alias match" not in af.reason
            ):
                continue
            if not af.file_path or af.file_path in seen_files:
                continue
            seen_files.add(af.file_path)
            required_files.append(af.file_path)

        if required_files:
            max_listed = 20
            listed = required_files[:max_listed]
            remaining = len(required_files) - len(listed)
            suffix = f"\n(+{remaining} more)" if remaining > 0 else ""
            required_block = "\n".join(f"- {path}" for path in listed)
            meta.warnings.append(
                "STOP: Do not finalize the answer until you have read all required files.\n"
                "Required files:\n"
                f"{required_block}{suffix}\n"
                "If any required file is not read, explicitly mark it UNVERIFIED in the answer."
            )
            meta.warnings.append(
                "Only implement code changes if you were asked to explicitly."
            )

        return ImpactOutput(affected_files=affected_files, meta=meta)

    def _validate_input(self, input_data: ImpactInput) -> None:
        """Validate input parameters."""
        if not input_data.repo_id or not input_data.repo_id.strip():
            raise InvalidInputError("repo_id is required")

        if (
            not input_data.changed_files
            and not input_data.symbol_id
            and not input_data.symbol_name
        ):
            raise InvalidInputError(
                "Either changed_files, symbol_id, or symbol_name is required"
            )

    def _resolve_symbol_id(self, input_data: ImpactInput) -> Optional[str]:
        if input_data.symbol_id:
            return input_data.symbol_id
        if not input_data.symbol_name:
            return None

        symbol_id = None
        if input_data.file_path and hasattr(self.graph_driver, "get_outgoing_edges"):
            file_node = self.graph_driver.get_node(input_data.file_path)
            if not file_node and hasattr(self.graph_driver, "find_file_id"):
                fallback_id = self.graph_driver.find_file_id(
                    input_data.file_path,
                    repo_id=input_data.repo_id,
                )
                if fallback_id:
                    file_node = self.graph_driver.get_node(fallback_id)

            if file_node:
                file_id = getattr(file_node, "id", None) or input_data.file_path
                outgoing_edges = self.graph_driver.get_outgoing_edges(file_id)
                symbol_id = self._find_symbol_in_edges(
                    outgoing_edges,
                    input_data.symbol_name,
                    input_data.parent_name,
                    input_data.symbol_kind,
                )

        if not symbol_id and hasattr(self.graph_driver, "find_symbol_id_by_name"):
            symbol_id = self.graph_driver.find_symbol_id_by_name(
                input_data.symbol_name,
                repo_id=input_data.repo_id,
                parent_name=input_data.parent_name,
                kind=input_data.symbol_kind,
                file_path=input_data.file_path,
            )

        if not symbol_id and hasattr(self.graph_driver, "find_symbol_id"):
            symbol_id = self.graph_driver.find_symbol_id(
                input_data.symbol_name,
                repo_id=input_data.repo_id,
            )

        if not symbol_id:
            raise ImpactAnalysisError(
                f"Symbol not found: {input_data.symbol_name}"
            )
        return symbol_id

    def _find_symbol_in_edges(
        self,
        edges: list[Any],
        symbol_name: str,
        parent_name: Optional[str],
        symbol_kind: Optional[str],
    ) -> Optional[str]:
        for edge in edges:
            edge_type_value = (
                edge.edge_type.value
                if hasattr(edge.edge_type, "value")
                else str(edge.edge_type)
            )
            if edge_type_value != "CONTAINS":
                continue
            symbol_node = self.graph_driver.get_node(edge.target_id)
            if not symbol_node:
                continue
            props = symbol_node.properties
            if props.get("name") != symbol_name:
                continue
            if parent_name and props.get("parent_name") != parent_name:
                continue
            if symbol_kind and props.get("kind") != symbol_kind:
                continue
            return symbol_node.id
        return None

    def _analyze_symbol_impact(
        self,
        symbol_id: str,
        repo_id: Optional[str],
        depth: int,
        affected_map: dict[str, AffectedFile],
        max_files: int,
        include_inferred_edges: bool,
        include_field_edges: bool,
        include_schema_edges: bool,
        include_path_edges: bool,
        traversal_state: dict[str, bool],
    ) -> None:
        """Analyze impact of changes to a symbol.

        Traverses callers to find affected files.
        """
        # Verify symbol exists
        node = self.graph_driver.get_node(symbol_id)
        if node is None:
            raise ImpactAnalysisError(f"Symbol not found: {symbol_id}")

        # The file containing the symbol is directly affected
        file_path = node.properties.get("file_path") or node.properties.get("path")
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
            include_inferred_edges=include_inferred_edges,
            include_field_edges=include_field_edges,
            include_schema_edges=include_schema_edges,
            include_path_edges=include_path_edges,
            traversal_state=traversal_state,
        )

        if include_path_edges:
            self._add_path_literal_impacts(
                symbol_node=node,
                repo_id=repo_id,
                affected_map=affected_map,
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
        include_inferred_edges: bool,
        include_field_edges: bool,
        include_schema_edges: bool,
        include_path_edges: bool,
        traversal_state: dict[str, bool],
    ) -> None:
        """Recursively traverse callers to find affected files."""
        if current_depth > depth or symbol_id in visited:
            return

        if len(affected_map) >= max_files:
            return

        visited.add(symbol_id)

        # Get incoming dependency edges (CALLS/READS/WRITES/HAS_FIELD)
        if not hasattr(self.graph_driver, "get_incoming_edges"):
            return

        incoming_edges = self.graph_driver.get_incoming_edges(symbol_id)

        for edge in incoming_edges:
            edge_type_value = (
                edge.edge_type.value
                if hasattr(edge.edge_type, "value")
                else str(edge.edge_type)
            )

            is_call = edge_type_value == "CALLS"
            is_field_edge = edge_type_value in ("READS", "WRITES", "HAS_FIELD")
            is_schema_edge = edge_type_value in ("SCHEMA_EXPOSES", "SCHEMA_ALIASES")
            is_path_edge = edge_type_value == "PATH_READS"

            if is_call:
                if not include_inferred_edges and edge.properties and edge.properties.get("inferred"):
                    continue
            elif not (
                (include_field_edges and is_field_edge)
                or (include_schema_edges and is_schema_edge)
                or (include_path_edges and is_path_edge)
            ):
                continue

            caller_id = edge.source_id

            if caller_id in visited:
                continue

            # Get caller node
            caller_node = self.graph_driver.get_node(caller_id)
            if caller_node:
                if edge.properties and edge.properties.get("inferred"):
                    traversal_state["used_inferred_edges"] = True

                file_path = caller_node.properties.get("file_path") or caller_node.properties.get("path")
                if file_path and file_path not in affected_map:
                    # Calculate confidence based on depth
                    confidence = self._calculate_confidence(current_depth)

                    if is_call:
                        reason = f"Calls changed symbol (depth {current_depth})"
                    elif edge_type_value == "READS":
                        reason = f"Reads changed field (depth {current_depth})"
                    elif edge_type_value == "WRITES":
                        reason = f"Writes changed field (depth {current_depth})"
                    elif edge_type_value == "SCHEMA_EXPOSES":
                        reason = f"Exposes schema field (depth {current_depth})"
                    elif edge_type_value == "SCHEMA_ALIASES":
                        reason = f"Schema alias match (depth {current_depth})"
                    elif edge_type_value == "PATH_READS":
                        reason = "String path reference (heuristic)"
                    else:
                        reason = f"Contains changed field (depth {current_depth})"

                    affected_map[file_path] = AffectedFile(
                        file_path=file_path,
                        reason=reason,
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
                    include_inferred_edges=include_inferred_edges,
                    include_field_edges=include_field_edges,
                    include_schema_edges=include_schema_edges,
                    include_path_edges=include_path_edges,
                    traversal_state=traversal_state,
                )

        if include_schema_edges and hasattr(self.graph_driver, "get_outgoing_edges"):
            outgoing_edges = self.graph_driver.get_outgoing_edges(symbol_id)
            for edge in outgoing_edges:
                edge_type_value = (
                    edge.edge_type.value
                    if hasattr(edge.edge_type, "value")
                    else str(edge.edge_type)
                )
                if edge_type_value not in ("SCHEMA_EXPOSES", "SCHEMA_ALIASES"):
                    continue
                schema_id = edge.target_id
                if schema_id in visited:
                    continue
                schema_node = self.graph_driver.get_node(schema_id)
                if not schema_node:
                    continue
                schema_path = schema_node.properties.get("file_path")
                if schema_path and schema_path not in affected_map:
                    reason = (
                        f"Exposed via schema field (depth {current_depth})"
                        if edge_type_value == "SCHEMA_EXPOSES"
                        else f"Schema alias match (depth {current_depth})"
                    )
                    affected_map[schema_path] = AffectedFile(
                        file_path=schema_path,
                        reason=reason,
                        confidence=self._calculate_confidence(current_depth),
                    )
                if len(affected_map) < max_files:
                        self._traverse_callers(
                            symbol_id=schema_id,
                            depth=depth,
                            current_depth=current_depth + 1,
                            affected_map=affected_map,
                            visited=visited,
                            max_files=max_files,
                            include_inferred_edges=include_inferred_edges,
                            include_field_edges=include_field_edges,
                            include_schema_edges=include_schema_edges,
                            include_path_edges=include_path_edges,
                            traversal_state=traversal_state,
                        )

    def _add_path_literal_impacts(
        self,
        symbol_node: Any,
        repo_id: Optional[str],
        affected_map: dict[str, AffectedFile],
        max_files: int,
    ) -> None:
        if not hasattr(self.graph_driver, "query_nodes_by_type"):
            return

        props = getattr(symbol_node, "properties", {}) or {}
        if props.get("kind") not in ("field", "property"):
            return

        field_name = props.get("name")
        if not field_name:
            return

        parent_name = props.get("parent_name")
        try:
            path_nodes = self.graph_driver.query_nodes_by_type(CodeNodeType.FIELD_PATH)
        except Exception:
            return

        matches: list[tuple[bool, float, Any]] = []
        for node in path_nodes:
            node_props = getattr(node, "properties", {}) or {}
            if repo_id and node_props.get("repo_id") and node_props.get("repo_id") != repo_id:
                continue
            if node_props.get("leaf") != field_name:
                continue
            has_parent_hint = self._path_has_parent_hint(node_props, parent_name)
            confidence = self._path_confidence(node_props.get("confidence"), has_parent_hint)
            matches.append((has_parent_hint, confidence, node))

        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)

        for _, confidence, node in matches[:PATH_MATCH_LIMIT]:
            if len(affected_map) >= max_files:
                break
            node_props = getattr(node, "properties", {}) or {}
            file_path = node_props.get("file_path")
            if not file_path or file_path in affected_map:
                continue
            affected_map[file_path] = AffectedFile(
                file_path=file_path,
                reason="String path reference (heuristic)",
                confidence=confidence,
            )

    def _path_confidence(self, confidence: Optional[str], has_parent_hint: bool) -> float:
        base = PATH_CONFIDENCE_SCORES.get(str(confidence or "").lower(), 0.5)
        if has_parent_hint:
            return min(0.95, base + PARENT_HINT_BOOST)
        return base

    def _path_has_parent_hint(self, props: dict[str, Any], parent_name: Optional[str]) -> bool:
        if not parent_name:
            return False
        parent = parent_name.lower()
        plural = parent if parent.endswith("s") else f"{parent}s"
        segments = props.get("segments") or []
        for segment in segments:
            segment_lower = str(segment).lower()
            if segment_lower in (parent, plural):
                return True
        normalized = str(props.get("normalized_path") or "").lower()
        if parent in normalized or plural in normalized:
            return True
        return False

    def _analyze_file_changes(
        self,
        file_paths: list[str],
        repo_id: Optional[str],
        depth: int,
        affected_map: dict[str, AffectedFile],
        max_files: int,
        include_inferred_edges: bool,
        include_field_edges: bool,
        include_schema_edges: bool,
        include_path_edges: bool,
        traversal_state: dict[str, bool],
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
                    repo_id=repo_id,
                )
                if fallback_id:
                    file_node = self.graph_driver.get_node(fallback_id)

            if not file_node:
                continue

            file_id = getattr(file_node, "id", None) or file_path

            # Get CONTAINS edges to find symbols
            try:
                outgoing_edges = self.graph_driver.get_outgoing_edges(file_id)
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
                            include_inferred_edges=include_inferred_edges,
                            include_field_edges=include_field_edges,
                            include_schema_edges=include_schema_edges,
                            include_path_edges=include_path_edges,
                            traversal_state=traversal_state,
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
