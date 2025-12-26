"""Graph export functionality.

This module provides graph export with:
- Configurable traversal depth
- Node and edge filtering
- Multiple output formats (JSON, DOT, Mermaid)
- Subgraph export from root nodes
- Call graph and containment hierarchy export
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
    Neo4jDriver,
)
from openmemory.api.visualization.config import (
    ExportConfig,
    ExportFormat,
    FilterConfig,
)
from openmemory.api.visualization.formatters import (
    DOTFormatter,
    Formatter,
    JSONFormatter,
    MermaidFormatter,
)
from openmemory.api.visualization.pagination import (
    Cursor,
    CursorEncoder,
    PageInfo,
    PaginatedResult,
    PaginationError,
)


# =============================================================================
# Exceptions
# =============================================================================


class GraphExportError(Exception):
    """Base exception for graph export errors."""

    pass


class InvalidFormatError(GraphExportError):
    """Raised when an invalid export format is specified."""

    pass


class TraversalError(GraphExportError):
    """Raised when graph traversal fails."""

    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExportResult:
    """Result of a graph export operation.

    Attributes:
        content: Formatted graph content.
        format: Export format used.
        node_count: Number of nodes exported.
        edge_count: Number of edges exported.
        metadata: Optional additional metadata.
        duration_ms: Export duration in milliseconds.
    """

    content: str
    format: ExportFormat
    node_count: int
    edge_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


# =============================================================================
# Graph Exporter
# =============================================================================


class GraphExporter:
    """Export code graphs to various formats.

    Supports:
    - Full graph export
    - Subgraph export from root nodes
    - Depth-limited traversal
    - Node and edge filtering
    - Multiple output formats
    """

    def __init__(
        self,
        driver: Neo4jDriver,
        config: Optional[ExportConfig] = None,
    ):
        """Initialize graph exporter.

        Args:
            driver: Neo4j driver for graph access.
            config: Optional export configuration.
        """
        self.driver = driver
        self.config = config or ExportConfig()

    def export(self, format_str: Optional[str] = None) -> ExportResult:
        """Export the full graph.

        Args:
            format_str: Optional format override (json, dot, mermaid).

        Returns:
            ExportResult with formatted content.

        Raises:
            InvalidFormatError: If format is not recognized.
        """
        start_time = time.time()

        # Determine format
        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Collect all nodes
        nodes: list[CodeNode] = []
        for node_type in CodeNodeType:
            type_nodes = self.driver.query_nodes_by_type(node_type)
            nodes.extend(type_nodes)

        # Apply node filters
        filtered_nodes = self._filter_nodes(nodes)

        # Collect all edges for filtered nodes
        node_ids = {n.id for n in filtered_nodes}
        edges = self._collect_edges_for_nodes(filtered_nodes)

        # Apply edge filters
        filtered_edges = self._filter_edges(edges, node_ids)

        # Apply property filters
        filtered_nodes = self._filter_node_properties(filtered_nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(filtered_nodes, filtered_edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(filtered_nodes),
            edge_count=len(filtered_edges),
            duration_ms=duration_ms,
        )

    def export_from_node(
        self,
        root_id: str,
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export subgraph starting from a root node.

        Args:
            root_id: ID of the root node.
            format_str: Optional format override.

        Returns:
            ExportResult with formatted content.

        Raises:
            TraversalError: If root node not found.
        """
        return self.export_from_nodes([root_id], format_str)

    def export_from_nodes(
        self,
        root_ids: list[str],
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export subgraph starting from multiple root nodes.

        Args:
            root_ids: IDs of root nodes.
            format_str: Optional format override.

        Returns:
            ExportResult with formatted content.

        Raises:
            TraversalError: If any root node not found.
        """
        start_time = time.time()

        # Determine format
        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Verify root nodes exist
        for root_id in root_ids:
            root_node = self.driver.get_node(root_id)
            if root_node is None:
                raise TraversalError(f"Root node not found: {root_id}")

        # BFS traversal from roots
        nodes, edges = self._traverse_from_roots(root_ids)

        # Apply node filters
        filtered_nodes = self._filter_nodes(nodes)

        # Apply edge filters
        node_ids = {n.id for n in filtered_nodes}
        filtered_edges = self._filter_edges(edges, node_ids)

        # Apply property filters
        filtered_nodes = self._filter_node_properties(filtered_nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(filtered_nodes, filtered_edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(filtered_nodes),
            edge_count=len(filtered_edges),
            duration_ms=duration_ms,
        )

    def export_call_graph(self, format_str: Optional[str] = None) -> ExportResult:
        """Export only call relationships.

        Args:
            format_str: Optional format override.

        Returns:
            ExportResult with call graph.
        """
        # Override filter to only include CALLS edges
        saved_filter = self.config.filter
        self.config.filter = FilterConfig(
            node_types=saved_filter.node_types,
            edge_types={CodeEdgeType.CALLS},
            max_depth=saved_filter.max_depth,
            include_properties=saved_filter.include_properties,
            exclude_properties=saved_filter.exclude_properties,
        )

        try:
            return self.export(format_str)
        finally:
            self.config.filter = saved_filter

    def export_callers(
        self,
        symbol_id: str,
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export callers of a symbol.

        Args:
            symbol_id: ID of the symbol.
            format_str: Optional format override.

        Returns:
            ExportResult with callers subgraph.

        Raises:
            TraversalError: If symbol not found.
        """
        start_time = time.time()

        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Verify symbol exists
        target_node = self.driver.get_node(symbol_id)
        if target_node is None:
            raise TraversalError(f"Symbol not found: {symbol_id}")

        # Find callers (reverse traversal)
        nodes = [target_node]
        edges: list[CodeEdge] = []

        # Check all nodes for outgoing CALLS edges to target
        for node_type in CodeNodeType:
            for node in self.driver.query_nodes_by_type(node_type):
                outgoing = self.driver.get_outgoing_edges(node.id)
                for edge in outgoing:
                    if (
                        edge.edge_type == CodeEdgeType.CALLS
                        and edge.target_id == symbol_id
                    ):
                        if node not in nodes:
                            nodes.append(node)
                        edges.append(edge)

        # Apply property filters
        nodes = self._filter_node_properties(nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(nodes, edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(nodes),
            edge_count=len(edges),
            duration_ms=duration_ms,
        )

    def export_callees(
        self,
        symbol_id: str,
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export callees of a symbol.

        Args:
            symbol_id: ID of the symbol.
            format_str: Optional format override.

        Returns:
            ExportResult with callees subgraph.

        Raises:
            TraversalError: If symbol not found.
        """
        start_time = time.time()

        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Verify symbol exists
        source_node = self.driver.get_node(symbol_id)
        if source_node is None:
            raise TraversalError(f"Symbol not found: {symbol_id}")

        # Find callees
        nodes = [source_node]
        edges: list[CodeEdge] = []

        outgoing = self.driver.get_outgoing_edges(symbol_id)
        for edge in outgoing:
            if edge.edge_type == CodeEdgeType.CALLS:
                edges.append(edge)
                callee = self.driver.get_node(edge.target_id)
                if callee and callee not in nodes:
                    nodes.append(callee)

        # Apply property filters
        nodes = self._filter_node_properties(nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(nodes, edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(nodes),
            edge_count=len(edges),
            duration_ms=duration_ms,
        )

    def export_file_contents(
        self,
        file_path: str,
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export contents of a file.

        Args:
            file_path: Path of the file.
            format_str: Optional format override.

        Returns:
            ExportResult with file contents subgraph.

        Raises:
            TraversalError: If file not found.
        """
        start_time = time.time()

        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Verify file exists
        file_node = self.driver.get_node(file_path)
        if file_node is None:
            raise TraversalError(f"File not found: {file_path}")

        # Find contained symbols
        nodes = [file_node]
        edges: list[CodeEdge] = []

        outgoing = self.driver.get_outgoing_edges(file_path)
        for edge in outgoing:
            if edge.edge_type == CodeEdgeType.CONTAINS:
                edges.append(edge)
                contained = self.driver.get_node(edge.target_id)
                if contained:
                    nodes.append(contained)
                    # Also get nested containment (e.g., methods in classes)
                    nested_edges = self.driver.get_outgoing_edges(contained.id)
                    for nested_edge in nested_edges:
                        if nested_edge.edge_type == CodeEdgeType.CONTAINS:
                            edges.append(nested_edge)
                            nested_node = self.driver.get_node(nested_edge.target_id)
                            if nested_node and nested_node not in nodes:
                                nodes.append(nested_node)

        # Apply property filters
        nodes = self._filter_node_properties(nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(nodes, edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(nodes),
            edge_count=len(edges),
            duration_ms=duration_ms,
        )

    def export_package_contents(
        self,
        package_name: str,
        format_str: Optional[str] = None,
    ) -> ExportResult:
        """Export contents of a package.

        Args:
            package_name: Name of the package.
            format_str: Optional format override.

        Returns:
            ExportResult with package contents subgraph.

        Raises:
            TraversalError: If package not found.
        """
        start_time = time.time()

        if format_str:
            export_format = self._parse_format(format_str)
        else:
            export_format = self.config.format

        # Verify package exists
        package_node = self.driver.get_node(package_name)
        if package_node is None:
            raise TraversalError(f"Package not found: {package_name}")

        # Find defined symbols
        nodes = [package_node]
        edges: list[CodeEdge] = []

        outgoing = self.driver.get_outgoing_edges(package_name)
        for edge in outgoing:
            if edge.edge_type == CodeEdgeType.DEFINES:
                edges.append(edge)
                defined = self.driver.get_node(edge.target_id)
                if defined:
                    nodes.append(defined)

        # Apply property filters
        nodes = self._filter_node_properties(nodes)

        # Format output
        formatter = self._get_formatter(export_format)
        content = formatter.format_graph(nodes, edges)

        duration_ms = (time.time() - start_time) * 1000

        return ExportResult(
            content=content,
            format=export_format,
            node_count=len(nodes),
            edge_count=len(edges),
            duration_ms=duration_ms,
        )

    def export_paginated(
        self,
        after: Optional[str] = None,
        first: Optional[int] = None,
    ) -> PaginatedResult:
        """Export graph with cursor-based pagination.

        Args:
            after: Cursor from previous page's end_cursor.
            first: Number of items to return (default from config).

        Returns:
            PaginatedResult with nodes, edges, and page info.

        Raises:
            PaginationError: If cursor is invalid.
        """
        # Get page size
        page_size = self.config.pagination.get_effective_page_size(first)

        # Collect all nodes
        all_nodes: list[CodeNode] = []
        for node_type in CodeNodeType:
            type_nodes = self.driver.query_nodes_by_type(node_type)
            all_nodes.extend(type_nodes)

        # Apply node filters
        filtered_nodes = self._filter_nodes(all_nodes)

        # Sort for stable pagination
        sorted_nodes = sorted(filtered_nodes, key=lambda n: n.id)

        total_count = len(sorted_nodes)

        # Decode cursor to get start position
        encoder = CursorEncoder()
        start_position = 0

        if after:
            try:
                cursor = encoder.decode(after)
                start_position = cursor.position + 1
            except Exception as e:
                raise PaginationError(f"Invalid cursor: {e}") from e

        # Get page slice
        end_position = min(start_position + page_size, total_count)
        page_nodes = sorted_nodes[start_position:end_position]

        # Apply property filters
        page_nodes = self._filter_node_properties(page_nodes)

        # Collect edges for page nodes
        page_node_ids = {n.id for n in page_nodes}
        all_edges = self._collect_edges_for_nodes(page_nodes)
        page_edges = self._filter_edges(all_edges, page_node_ids)

        # Build cursors
        start_cursor = None
        end_cursor = None

        if page_nodes:
            start_cursor = encoder.encode(Cursor(
                position=start_position,
                sort_key=page_nodes[0].id,
            ))
            end_cursor = encoder.encode(Cursor(
                position=end_position - 1,
                sort_key=page_nodes[-1].id,
            ))

        # Build page info
        page_info = PageInfo(
            has_next_page=end_position < total_count,
            has_previous_page=start_position > 0,
            start_cursor=start_cursor,
            end_cursor=end_cursor,
            total_count=total_count,
        )

        # Convert to dicts
        node_dicts = [
            {
                "id": n.id,
                "type": n.node_type.value,
                "properties": n.properties,
            }
            for n in page_nodes
        ]
        edge_dicts = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "properties": e.properties,
            }
            for e in page_edges
        ]

        return PaginatedResult(
            nodes=node_dicts,
            edges=edge_dicts,
            page_info=page_info,
        )

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _parse_format(self, format_str: str) -> ExportFormat:
        """Parse format string to ExportFormat enum."""
        format_str = format_str.lower().strip()
        try:
            return ExportFormat(format_str)
        except ValueError:
            valid_formats = ", ".join(f.value for f in ExportFormat)
            raise InvalidFormatError(
                f"Invalid format '{format_str}'. Valid formats: {valid_formats}"
            )

    def _get_formatter(self, format: ExportFormat) -> Formatter:
        """Get formatter for export format."""
        style = self.config.style
        if format == ExportFormat.JSON:
            indent = 2 if self.config.pretty_print else None
            return JSONFormatter(
                indent=indent,
                style_config=style,
                include_style=self.config.include_metadata,
            )
        elif format == ExportFormat.DOT:
            return DOTFormatter(
                graph_name=self.config.graph_name,
                style_config=style,
            )
        elif format == ExportFormat.MERMAID:
            return MermaidFormatter(
                style_config=style,
            )
        else:
            raise InvalidFormatError(f"Unsupported format: {format}")

    def _traverse_from_roots(
        self,
        root_ids: list[str],
    ) -> tuple[list[CodeNode], list[CodeEdge]]:
        """BFS traversal from root nodes.

        Args:
            root_ids: IDs of root nodes.

        Returns:
            Tuple of (nodes, edges) found during traversal.
        """
        max_depth = self.config.filter.max_depth
        visited: set[str] = set()
        nodes: list[CodeNode] = []
        edges: list[CodeEdge] = []

        # Queue entries: (node_id, depth)
        queue: deque[tuple[str, int]] = deque()
        for root_id in root_ids:
            queue.append((root_id, 0))
            visited.add(root_id)

        while queue:
            node_id, depth = queue.popleft()
            node = self.driver.get_node(node_id)
            if node:
                nodes.append(node)

            # Stop at max depth
            if depth >= max_depth:
                continue

            # Get outgoing edges
            if node:
                outgoing = self.driver.get_outgoing_edges(node_id)
                for edge in outgoing:
                    edges.append(edge)
                    target_id = edge.target_id
                    if target_id not in visited:
                        visited.add(target_id)
                        queue.append((target_id, depth + 1))

        return nodes, edges

    def _filter_nodes(self, nodes: list[CodeNode]) -> list[CodeNode]:
        """Filter nodes based on configuration."""
        filter_config = self.config.filter
        return [
            node
            for node in nodes
            if filter_config.matches_node(node.node_type, node.properties)
        ]

    def _filter_edges(
        self,
        edges: list[CodeEdge],
        valid_node_ids: set[str],
    ) -> list[CodeEdge]:
        """Filter edges based on configuration and valid nodes."""
        filter_config = self.config.filter
        result = []
        for edge in edges:
            # Check edge type filter
            if not filter_config.matches_edge(edge.edge_type):
                continue
            # Check both endpoints are in valid nodes
            if edge.source_id in valid_node_ids and edge.target_id in valid_node_ids:
                result.append(edge)
        return result

    def _filter_node_properties(self, nodes: list[CodeNode]) -> list[CodeNode]:
        """Apply property filters to nodes."""
        filter_config = self.config.filter
        result = []
        for node in nodes:
            filtered_props = filter_config.filter_properties(node.properties)
            # Create new node with filtered properties
            filtered_node = CodeNode(
                node_type=node.node_type,
                id=node.id,
                properties=filtered_props,
            )
            result.append(filtered_node)
        return result

    def _collect_edges_for_nodes(self, nodes: list[CodeNode]) -> list[CodeEdge]:
        """Collect all edges for a set of nodes."""
        edges: list[CodeEdge] = []
        for node in nodes:
            outgoing = self.driver.get_outgoing_edges(node.id)
            edges.extend(outgoing)
        return edges
