"""Call Graph Tools (find_callers, find_callees).

This module provides MCP tools for traversing the code call graph:
- FindCallersTool: Find functions that call a symbol
- FindCalleesTool: Find functions called by a symbol

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


class CallGraphError(Exception):
    """Base exception for call graph tool errors."""

    pass


class SymbolNotFoundError(CallGraphError):
    """Raised when a symbol cannot be found in the graph."""

    pass


class InvalidInputError(CallGraphError):
    """Raised when input is invalid."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CallGraphConfig:
    """Configuration for call graph tools.

    Args:
        depth: Maximum traversal depth (1-5)
        max_nodes: Maximum nodes to return
        include_properties: Include node properties in output
    """

    depth: int = 1
    max_nodes: int = 100
    include_properties: bool = True


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class CallGraphInput:
    """Input parameters for call graph tools.

    Args:
        symbol_id: SCIP symbol ID
        symbol_name: Alternative: symbol name for lookup
        repo_id: Repository ID (required)
        depth: Traversal depth override
    """

    repo_id: str = ""
    symbol_id: Optional[str] = None
    symbol_name: Optional[str] = None
    depth: Optional[int] = None


# =============================================================================
# Result Types (MCP Schema Compliant)
# =============================================================================


@dataclass
class GraphNode:
    """A node in the graph output.

    Maps to GraphNode in MCP schema.
    """

    id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the graph output.

    Maps to GraphEdge in MCP schema.
    """

    from_id: str
    to_id: str
    type: str
    confidence: str = "definite"  # "definite", "probable", "possible"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseMeta:
    """Response metadata.

    Maps to ResponseMeta in MCP schema.
    """

    request_id: str
    degraded_mode: bool = False
    missing_sources: list[str] = field(default_factory=list)
    next_cursor: Optional[str] = None


@dataclass
class GraphOutput:
    """Result from call graph tools.

    Maps to GraphOutput in MCP schema.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    meta: ResponseMeta
    next_cursor: Optional[str] = None


# =============================================================================
# Find Callers Tool
# =============================================================================


class FindCallersTool:
    """MCP tool for finding functions that call a symbol.

    Traverses incoming CALLS edges in the code graph.
    """

    def __init__(
        self,
        graph_driver: Any,
        config: Optional[CallGraphConfig] = None,
    ):
        """Initialize find_callers tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.config = config or CallGraphConfig()

    def find(
        self,
        input_data: CallGraphInput,
        config: Optional[CallGraphConfig] = None,
    ) -> GraphOutput:
        """Find functions that call the given symbol.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            GraphOutput with caller nodes and edges

        Raises:
            InvalidInputError: If input is invalid
            SymbolNotFoundError: If symbol not found
            CallGraphError: If graph query fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Get target symbol ID
        symbol_id = self._resolve_symbol_id(input_data)

        # Verify symbol exists
        target_node = self.graph_driver.get_node(symbol_id)
        if target_node is None:
            raise SymbolNotFoundError(f"Symbol not found: {symbol_id}")

        # Traverse callers
        depth = input_data.depth or cfg.depth
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        visited: set[str] = set()

        try:
            self._traverse_callers(
                symbol_id=symbol_id,
                depth=depth,
                nodes=nodes,
                edges=edges,
                visited=visited,
                max_nodes=cfg.max_nodes,
                include_properties=cfg.include_properties,
            )
        except Exception as e:
            logger.error(f"Failed to traverse callers: {e}")
            raise CallGraphError(f"Graph query failed: {e}") from e

        # Add target node
        if cfg.include_properties:
            target_props = dict(target_node.properties)
        else:
            target_props = {"name": target_node.properties.get("name", "")}

        nodes.insert(
            0,
            GraphNode(
                id=symbol_id,
                type="CODE_SYMBOL",
                properties=target_props,
            ),
        )

        return GraphOutput(
            nodes=nodes,
            edges=edges,
            meta=ResponseMeta(request_id=request_id),
        )

    def _validate_input(self, input_data: CallGraphInput) -> None:
        """Validate input parameters."""
        if not input_data.repo_id or not input_data.repo_id.strip():
            raise InvalidInputError("repo_id is required")

        if not input_data.symbol_id and not input_data.symbol_name:
            raise InvalidInputError("Either symbol_id or symbol_name is required")

        if input_data.symbol_id is not None and not input_data.symbol_id.strip():
            raise InvalidInputError("symbol_id cannot be empty")

    def _resolve_symbol_id(self, input_data: CallGraphInput) -> str:
        """Resolve symbol_id from input."""
        if input_data.symbol_id:
            return input_data.symbol_id

        # TODO: Implement symbol name lookup
        raise InvalidInputError("symbol_name lookup not yet implemented")

    def _traverse_callers(
        self,
        symbol_id: str,
        depth: int,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        visited: set[str],
        max_nodes: int,
        include_properties: bool,
    ) -> None:
        """Recursively traverse caller graph."""
        if depth <= 0 or symbol_id in visited:
            return

        if len(nodes) >= max_nodes:
            return

        visited.add(symbol_id)

        # Get incoming CALLS edges
        if not hasattr(self.graph_driver, "get_incoming_edges"):
            return

        incoming_edges = self.graph_driver.get_incoming_edges(symbol_id)

        for edge in incoming_edges:
            # Check edge type
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

            # Add edge
            edge_props = dict(edge.properties) if edge.properties else {}
            edges.append(
                GraphEdge(
                    from_id=caller_id,
                    to_id=symbol_id,
                    type="CALLS",
                    properties=edge_props,
                )
            )

            # Get caller node
            caller_node = self.graph_driver.get_node(caller_id)
            if caller_node:
                if include_properties:
                    props = dict(caller_node.properties)
                else:
                    props = {"name": caller_node.properties.get("name", "")}

                nodes.append(
                    GraphNode(
                        id=caller_id,
                        type="CODE_SYMBOL",
                        properties=props,
                    )
                )

            # Recurse for transitive callers
            if depth > 1 and len(nodes) < max_nodes:
                self._traverse_callers(
                    symbol_id=caller_id,
                    depth=depth - 1,
                    nodes=nodes,
                    edges=edges,
                    visited=visited,
                    max_nodes=max_nodes,
                    include_properties=include_properties,
                )


# =============================================================================
# Find Callees Tool
# =============================================================================


class FindCalleesTool:
    """MCP tool for finding functions called by a symbol.

    Traverses outgoing CALLS edges in the code graph.
    """

    def __init__(
        self,
        graph_driver: Any,
        config: Optional[CallGraphConfig] = None,
    ):
        """Initialize find_callees tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.config = config or CallGraphConfig()

    def find(
        self,
        input_data: CallGraphInput,
        config: Optional[CallGraphConfig] = None,
    ) -> GraphOutput:
        """Find functions called by the given symbol.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            GraphOutput with callee nodes and edges

        Raises:
            InvalidInputError: If input is invalid
            SymbolNotFoundError: If symbol not found
            CallGraphError: If graph query fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Get source symbol ID
        symbol_id = self._resolve_symbol_id(input_data)

        # Verify symbol exists
        source_node = self.graph_driver.get_node(symbol_id)
        if source_node is None:
            raise SymbolNotFoundError(f"Symbol not found: {symbol_id}")

        # Traverse callees
        depth = input_data.depth or cfg.depth
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        visited: set[str] = set()

        try:
            self._traverse_callees(
                symbol_id=symbol_id,
                depth=depth,
                nodes=nodes,
                edges=edges,
                visited=visited,
                max_nodes=cfg.max_nodes,
                include_properties=cfg.include_properties,
            )
        except Exception as e:
            logger.error(f"Failed to traverse callees: {e}")
            raise CallGraphError(f"Graph query failed: {e}") from e

        # Add source node
        if cfg.include_properties:
            source_props = dict(source_node.properties)
        else:
            source_props = {"name": source_node.properties.get("name", "")}

        nodes.insert(
            0,
            GraphNode(
                id=symbol_id,
                type="CODE_SYMBOL",
                properties=source_props,
            ),
        )

        return GraphOutput(
            nodes=nodes,
            edges=edges,
            meta=ResponseMeta(request_id=request_id),
        )

    def _validate_input(self, input_data: CallGraphInput) -> None:
        """Validate input parameters."""
        if not input_data.repo_id or not input_data.repo_id.strip():
            raise InvalidInputError("repo_id is required")

        if not input_data.symbol_id and not input_data.symbol_name:
            raise InvalidInputError("Either symbol_id or symbol_name is required")

        if input_data.symbol_id is not None and not input_data.symbol_id.strip():
            raise InvalidInputError("symbol_id cannot be empty")

    def _resolve_symbol_id(self, input_data: CallGraphInput) -> str:
        """Resolve symbol_id from input."""
        if input_data.symbol_id:
            return input_data.symbol_id

        # TODO: Implement symbol name lookup
        raise InvalidInputError("symbol_name lookup not yet implemented")

    def _traverse_callees(
        self,
        symbol_id: str,
        depth: int,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        visited: set[str],
        max_nodes: int,
        include_properties: bool,
    ) -> None:
        """Recursively traverse callee graph."""
        if depth <= 0 or symbol_id in visited:
            return

        if len(nodes) >= max_nodes:
            return

        visited.add(symbol_id)

        # Get outgoing CALLS edges
        outgoing_edges = self.graph_driver.get_outgoing_edges(symbol_id)

        for edge in outgoing_edges:
            # Check edge type
            edge_type_value = (
                edge.edge_type.value
                if hasattr(edge.edge_type, "value")
                else str(edge.edge_type)
            )

            if edge_type_value != "CALLS":
                continue

            callee_id = edge.target_id

            if callee_id in visited:
                continue

            # Add edge
            edge_props = dict(edge.properties) if edge.properties else {}
            edges.append(
                GraphEdge(
                    from_id=symbol_id,
                    to_id=callee_id,
                    type="CALLS",
                    properties=edge_props,
                )
            )

            # Get callee node
            callee_node = self.graph_driver.get_node(callee_id)
            if callee_node:
                if include_properties:
                    props = dict(callee_node.properties)
                else:
                    props = {"name": callee_node.properties.get("name", "")}

                nodes.append(
                    GraphNode(
                        id=callee_id,
                        type="CODE_SYMBOL",
                        properties=props,
                    )
                )

            # Recurse for transitive callees
            if depth > 1 and len(nodes) < max_nodes:
                self._traverse_callees(
                    symbol_id=callee_id,
                    depth=depth - 1,
                    nodes=nodes,
                    edges=edges,
                    visited=visited,
                    max_nodes=max_nodes,
                    include_properties=include_properties,
                )


# =============================================================================
# Factory Functions
# =============================================================================


def create_find_callers_tool(
    graph_driver: Any,
    config: Optional[CallGraphConfig] = None,
) -> FindCallersTool:
    """Create a find_callers tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        config: Optional configuration

    Returns:
        Configured FindCallersTool
    """
    return FindCallersTool(
        graph_driver=graph_driver,
        config=config,
    )


def create_find_callees_tool(
    graph_driver: Any,
    config: Optional[CallGraphConfig] = None,
) -> FindCalleesTool:
    """Create a find_callees tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        config: Optional configuration

    Returns:
        Configured FindCalleesTool
    """
    return FindCalleesTool(
        graph_driver=graph_driver,
        config=config,
    )
