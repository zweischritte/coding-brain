"""Configuration for graph visualization and export.

This module provides configuration dataclasses for:
- Export formats (JSON, DOT, Mermaid)
- Filtering options (node types, edge types, depth)
- Pagination settings
- Visual styling (colors, shapes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    DOT = "dot"
    MERMAID = "mermaid"


@dataclass
class StyleConfig:
    """Visual styling configuration.

    Attributes:
        file_color: Color for file nodes (hex).
        symbol_color: Color for symbol nodes (hex).
        package_color: Color for package nodes (hex).
        edge_colors: Optional mapping of edge types to colors.
        file_shape: Shape for file nodes (DOT format).
        symbol_shape: Shape for symbol nodes (DOT format).
        package_shape: Shape for package nodes (DOT format).
        font_name: Font family for labels.
        font_size: Font size for labels.
    """

    file_color: str = "#E8F4FD"
    symbol_color: str = "#E8F5E9"
    package_color: str = "#FFF3E0"
    edge_colors: dict[CodeEdgeType, str] = field(default_factory=lambda: {
        CodeEdgeType.CONTAINS: "#90CAF9",
        CodeEdgeType.DEFINES: "#A5D6A7",
        CodeEdgeType.IMPORTS: "#FFCC80",
        CodeEdgeType.CALLS: "#EF9A9A",
        CodeEdgeType.READS: "#CE93D8",
        CodeEdgeType.WRITES: "#F48FB1",
        CodeEdgeType.DATA_FLOWS_TO: "#80DEEA",
    })
    file_shape: str = "folder"
    symbol_shape: str = "box"
    package_shape: str = "box3d"
    font_name: str = "Arial"
    font_size: int = 12


@dataclass
class FilterConfig:
    """Filtering configuration for graph traversal.

    Attributes:
        node_types: Include only these node types (None = all).
        edge_types: Include only these edge types (None = all).
        max_depth: Maximum traversal depth from root nodes.
        include_properties: Properties to include (None = all).
        exclude_properties: Properties to exclude.
        file_patterns: Glob patterns for files to include.
        exclude_file_patterns: Glob patterns for files to exclude.
        languages: Include only these languages.
        symbol_kinds: Include only these symbol kinds (function, class, etc.).
    """

    node_types: Optional[set[CodeNodeType]] = None
    edge_types: Optional[set[CodeEdgeType]] = None
    max_depth: int = 10
    include_properties: Optional[set[str]] = None
    exclude_properties: set[str] = field(default_factory=set)
    file_patterns: Optional[list[str]] = None
    exclude_file_patterns: list[str] = field(default_factory=list)
    languages: Optional[set[str]] = None
    symbol_kinds: Optional[set[str]] = None

    def matches_node(self, node_type: CodeNodeType, properties: dict) -> bool:
        """Check if a node matches this filter.

        Args:
            node_type: The node type to check.
            properties: The node properties.

        Returns:
            True if the node matches all filter criteria.
        """
        # Check node type
        if self.node_types is not None and node_type not in self.node_types:
            return False

        # Check language
        if self.languages is not None:
            lang = properties.get("language")
            if lang and lang not in self.languages:
                return False

        # Check symbol kind
        if self.symbol_kinds is not None:
            kind = properties.get("kind")
            if kind and kind not in self.symbol_kinds:
                return False

        # Check file patterns (for file nodes)
        if node_type == CodeNodeType.FILE and self.file_patterns is not None:
            path = properties.get("path", "")
            if not self._matches_any_pattern(path, self.file_patterns):
                return False

        # Check exclude patterns
        if node_type == CodeNodeType.FILE and self.exclude_file_patterns:
            path = properties.get("path", "")
            if self._matches_any_pattern(path, self.exclude_file_patterns):
                return False

        return True

    def matches_edge(self, edge_type: CodeEdgeType) -> bool:
        """Check if an edge matches this filter.

        Args:
            edge_type: The edge type to check.

        Returns:
            True if the edge type is allowed.
        """
        if self.edge_types is None:
            return True
        return edge_type in self.edge_types

    def filter_properties(self, properties: dict) -> dict:
        """Filter properties based on include/exclude lists.

        Args:
            properties: The properties to filter.

        Returns:
            Filtered properties dict.
        """
        if self.include_properties is not None:
            result = {
                k: v for k, v in properties.items()
                if k in self.include_properties
            }
        else:
            result = dict(properties)

        # Remove excluded properties
        for key in self.exclude_properties:
            result.pop(key, None)

        return result

    def _matches_any_pattern(self, path: str, patterns: list[str]) -> bool:
        """Check if path matches any glob pattern."""
        import fnmatch
        return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


@dataclass
class PaginationConfig:
    """Pagination configuration.

    Attributes:
        page_size: Number of items per page (default 50, max 100).
        max_page_size: Maximum allowed page size.
        default_page_size: Default page size if not specified.
    """

    page_size: int = 50
    max_page_size: int = 100
    default_page_size: int = 50

    def __post_init__(self):
        """Validate configuration."""
        if self.page_size > self.max_page_size:
            self.page_size = self.max_page_size
        if self.page_size < 1:
            self.page_size = 1

    def get_effective_page_size(self, requested: Optional[int] = None) -> int:
        """Get effective page size considering limits.

        Args:
            requested: Requested page size (None = use default).

        Returns:
            Effective page size within limits.
        """
        if requested is None:
            return self.default_page_size
        return min(max(1, requested), self.max_page_size)


@dataclass
class ExportConfig:
    """Main export configuration.

    Attributes:
        format: Output format (JSON, DOT, Mermaid).
        filter: Filtering configuration.
        pagination: Pagination configuration.
        style: Visual styling configuration.
        include_metadata: Include export metadata in output.
        pretty_print: Pretty-print output (where applicable).
        graph_name: Name for the graph (used in DOT/Mermaid).
    """

    format: ExportFormat = ExportFormat.JSON
    filter: FilterConfig = field(default_factory=FilterConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    include_metadata: bool = True
    pretty_print: bool = True
    graph_name: str = "CodeGraph"
