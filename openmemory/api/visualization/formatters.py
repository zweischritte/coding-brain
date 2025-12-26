"""Graph export formatters.

This module provides formatters for exporting code graphs:
- JSONFormatter: Export to JSON format
- DOTFormatter: Export to DOT (Graphviz) format
- MermaidFormatter: Export to Mermaid diagram format

Each formatter implements the Formatter abstract interface.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
)
from openmemory.api.visualization.config import ExportFormat, StyleConfig


class Formatter(ABC):
    """Abstract base class for graph formatters."""

    @abstractmethod
    def format_nodes(self, nodes: list[CodeNode]) -> str:
        """Format a list of nodes.

        Args:
            nodes: List of nodes to format.

        Returns:
            Formatted string representation.
        """
        pass

    @abstractmethod
    def format_edges(self, edges: list[CodeEdge]) -> str:
        """Format a list of edges.

        Args:
            edges: List of edges to format.

        Returns:
            Formatted string representation.
        """
        pass

    @abstractmethod
    def format_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> str:
        """Format a complete graph with nodes and edges.

        Args:
            nodes: List of nodes.
            edges: List of edges.

        Returns:
            Formatted string representation.
        """
        pass

    @abstractmethod
    def get_format(self) -> ExportFormat:
        """Get the export format type.

        Returns:
            The ExportFormat enum value.
        """
        pass


class JSONFormatter(Formatter):
    """Format graphs as JSON.

    Supports:
    - Flat or hierarchical output
    - Configurable indentation
    - Optional style information
    - Metadata inclusion
    """

    def __init__(
        self,
        indent: Optional[int] = 2,
        style_config: Optional[StyleConfig] = None,
        include_style: bool = False,
    ):
        """Initialize JSON formatter.

        Args:
            indent: JSON indentation level (None for compact).
            style_config: Optional styling configuration.
            include_style: Whether to include style info in output.
        """
        self.indent = indent
        self.style_config = style_config or StyleConfig()
        self.include_style = include_style

    def format_nodes(self, nodes: list[CodeNode]) -> str:
        """Format nodes as JSON array."""
        result = []
        for node in nodes:
            node_dict = self._node_to_dict(node)
            result.append(node_dict)
        return json.dumps(result, indent=self.indent, default=str)

    def format_edges(self, edges: list[CodeEdge]) -> str:
        """Format edges as JSON array."""
        result = []
        for edge in edges:
            edge_dict = self._edge_to_dict(edge)
            result.append(edge_dict)
        return json.dumps(result, indent=self.indent, default=str)

    def format_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> str:
        """Format complete graph as JSON object."""
        graph = {
            "nodes": [self._node_to_dict(n) for n in nodes],
            "edges": [self._edge_to_dict(e) for e in edges],
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "format": "json",
                "exported_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        return json.dumps(graph, indent=self.indent, default=str)

    def _node_to_dict(self, node: CodeNode) -> dict[str, Any]:
        """Convert CodeNode to dictionary."""
        result: dict[str, Any] = {
            "id": node.id,
            "type": node.node_type.value,
            "properties": {
                k: v for k, v in node.properties.items()
                if v is not None
            },
        }

        if self.include_style:
            result["style"] = self._get_node_style(node.node_type)

        return result

    def _edge_to_dict(self, edge: CodeEdge) -> dict[str, Any]:
        """Convert CodeEdge to dictionary."""
        return {
            "source": edge.source_id,
            "target": edge.target_id,
            "type": edge.edge_type.value,
            "properties": edge.properties,
        }

    def _get_node_style(self, node_type: CodeNodeType) -> dict[str, str]:
        """Get style info for node type."""
        if node_type == CodeNodeType.FILE:
            return {"color": self.style_config.file_color}
        elif node_type == CodeNodeType.SYMBOL:
            return {"color": self.style_config.symbol_color}
        elif node_type == CodeNodeType.PACKAGE:
            return {"color": self.style_config.package_color}
        return {}

    def get_format(self) -> ExportFormat:
        """Get format type."""
        return ExportFormat.JSON


class DOTFormatter(Formatter):
    """Format graphs as DOT (Graphviz) format.

    Supports:
    - Node shapes by type
    - Edge labels
    - Custom colors and styling
    - Graph direction (TB, LR, etc.)
    """

    def __init__(
        self,
        graph_name: str = "CodeGraph",
        rankdir: str = "TB",
        style_config: Optional[StyleConfig] = None,
        show_edge_labels: bool = False,
    ):
        """Initialize DOT formatter.

        Args:
            graph_name: Name of the graph.
            rankdir: Graph direction (TB, LR, BT, RL).
            style_config: Optional styling configuration.
            show_edge_labels: Whether to show edge type labels.
        """
        self.graph_name = graph_name
        self.rankdir = rankdir
        self.style_config = style_config or StyleConfig()
        self.show_edge_labels = show_edge_labels
        self._node_id_map: dict[str, str] = {}
        self._node_counter = 0

    def format_nodes(self, nodes: list[CodeNode]) -> str:
        """Format nodes as DOT node definitions."""
        lines = []
        for node in nodes:
            line = self._format_node(node)
            lines.append(line)
        return "\n".join(lines)

    def format_edges(self, edges: list[CodeEdge]) -> str:
        """Format edges as DOT edge definitions."""
        lines = []
        for edge in edges:
            line = self._format_edge(edge)
            lines.append(line)
        return "\n".join(lines)

    def format_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> str:
        """Format complete graph as DOT."""
        lines = [
            f"digraph {self._escape_id(self.graph_name)} {{",
            f"  rankdir={self.rankdir};",
            f'  node [fontname="{self.style_config.font_name}", fontsize={self.style_config.font_size}];',
            "",
            "  // Nodes",
        ]

        # Add nodes
        for node in nodes:
            lines.append("  " + self._format_node(node))

        lines.append("")
        lines.append("  // Edges")

        # Add edges
        for edge in edges:
            lines.append("  " + self._format_edge(edge))

        lines.append("}")
        return "\n".join(lines)

    def _format_node(self, node: CodeNode) -> str:
        """Format a single node."""
        node_id = self._get_safe_id(node.id)
        label = self._get_node_label(node)
        attrs = self._get_node_attrs(node)

        attr_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in attrs.items())
        return f'{node_id} [label="{label}", {attr_str}];'

    def _format_edge(self, edge: CodeEdge) -> str:
        """Format a single edge."""
        source_id = self._get_safe_id(edge.source_id)
        target_id = self._get_safe_id(edge.target_id)

        attrs: list[str] = []

        if self.show_edge_labels:
            attrs.append(f'label="{edge.edge_type.value}"')

        # Get edge color
        color = self.style_config.edge_colors.get(edge.edge_type, "#333333")
        attrs.append(f'color="{color}"')

        if attrs:
            return f"{source_id} -> {target_id} [{', '.join(attrs)}];"
        return f"{source_id} -> {target_id};"

    def _get_safe_id(self, original_id: str) -> str:
        """Get a safe DOT node ID."""
        if original_id not in self._node_id_map:
            safe_id = f"n_{self._node_counter}"
            self._node_id_map[original_id] = safe_id
            self._node_counter += 1
        return self._node_id_map[original_id]

    def _get_node_label(self, node: CodeNode) -> str:
        """Get display label for node."""
        name = node.properties.get("name", node.id)
        # Truncate long labels
        if len(name) > 50:
            name = name[:47] + "..."
        return self._escape_label(name)

    def _get_node_attrs(self, node: CodeNode) -> dict[str, Any]:
        """Get DOT attributes for node."""
        attrs: dict[str, Any] = {}

        if node.node_type == CodeNodeType.FILE:
            attrs["shape"] = self.style_config.file_shape
            attrs["fillcolor"] = self.style_config.file_color
            attrs["style"] = "filled"
        elif node.node_type == CodeNodeType.SYMBOL:
            kind = node.properties.get("kind", "symbol")
            if kind == "class":
                attrs["shape"] = "box"
                attrs["style"] = "filled,rounded"
            elif kind in ("function", "method"):
                attrs["shape"] = "ellipse"
                attrs["style"] = "filled"
            else:
                attrs["shape"] = self.style_config.symbol_shape
                attrs["style"] = "filled"
            attrs["fillcolor"] = self.style_config.symbol_color
        elif node.node_type == CodeNodeType.PACKAGE:
            attrs["shape"] = self.style_config.package_shape
            attrs["fillcolor"] = self.style_config.package_color
            attrs["style"] = "filled"

        return attrs

    def _escape_id(self, s: str) -> str:
        """Escape a string for use as DOT ID."""
        # Replace non-alphanumeric with underscore
        return re.sub(r"[^a-zA-Z0-9_]", "_", s)

    def _escape_label(self, s: str) -> str:
        """Escape a string for use in DOT label."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def get_format(self) -> ExportFormat:
        """Get format type."""
        return ExportFormat.DOT


class MermaidFormatter(Formatter):
    """Format graphs as Mermaid diagram syntax.

    Supports:
    - Node shapes by type
    - Edge labels
    - Subgraph grouping
    - Flow direction (TB, LR, etc.)
    """

    def __init__(
        self,
        direction: str = "TB",
        style_config: Optional[StyleConfig] = None,
        show_edge_labels: bool = False,
        use_subgraphs: bool = False,
    ):
        """Initialize Mermaid formatter.

        Args:
            direction: Diagram direction (TB, LR, BT, RL).
            style_config: Optional styling configuration.
            show_edge_labels: Whether to show edge type labels.
            use_subgraphs: Whether to group nodes in subgraphs.
        """
        self.direction = direction
        self.style_config = style_config or StyleConfig()
        self.show_edge_labels = show_edge_labels
        self.use_subgraphs = use_subgraphs
        self._node_id_map: dict[str, str] = {}
        self._node_counter = 0

    def format_nodes(self, nodes: list[CodeNode]) -> str:
        """Format nodes as Mermaid node definitions."""
        lines = []
        for node in nodes:
            line = self._format_node(node)
            lines.append(line)
        return "\n".join(lines)

    def format_edges(self, edges: list[CodeEdge]) -> str:
        """Format edges as Mermaid edge definitions."""
        lines = []
        for edge in edges:
            line = self._format_edge(edge)
            lines.append(line)
        return "\n".join(lines)

    def format_graph(self, nodes: list[CodeNode], edges: list[CodeEdge]) -> str:
        """Format complete graph as Mermaid."""
        lines = [f"flowchart {self.direction}"]

        # Add style definitions
        lines.append(f"  classDef fileNode fill:{self.style_config.file_color}")
        lines.append(f"  classDef symbolNode fill:{self.style_config.symbol_color}")
        lines.append(f"  classDef packageNode fill:{self.style_config.package_color}")

        # Add nodes
        if self.use_subgraphs:
            lines.extend(self._format_with_subgraphs(nodes))
        else:
            for node in nodes:
                lines.append("  " + self._format_node(node))

        # Add edges
        for edge in edges:
            lines.append("  " + self._format_edge(edge))

        return "\n".join(lines)

    def _format_node(self, node: CodeNode) -> str:
        """Format a single node."""
        node_id = self._get_safe_id(node.id)
        label = self._get_node_label(node)
        shape_start, shape_end = self._get_shape_brackets(node)
        class_name = self._get_node_class(node)

        return f"{node_id}{shape_start}{label}{shape_end}:::{class_name}"

    def _format_edge(self, edge: CodeEdge) -> str:
        """Format a single edge."""
        source_id = self._get_safe_id(edge.source_id)
        target_id = self._get_safe_id(edge.target_id)

        if self.show_edge_labels:
            label = edge.edge_type.value
            return f"{source_id} -->|{label}| {target_id}"
        return f"{source_id} --> {target_id}"

    def _format_with_subgraphs(self, nodes: list[CodeNode]) -> list[str]:
        """Group nodes into subgraphs by file."""
        lines: list[str] = []
        file_nodes: dict[str, list[CodeNode]] = {}
        other_nodes: list[CodeNode] = []

        for node in nodes:
            if node.node_type == CodeNodeType.FILE:
                other_nodes.append(node)
            else:
                file_path = node.properties.get("file_path")
                if file_path:
                    if file_path not in file_nodes:
                        file_nodes[file_path] = []
                    file_nodes[file_path].append(node)
                else:
                    other_nodes.append(node)

        # Add file nodes
        for node in other_nodes:
            lines.append("  " + self._format_node(node))

        # Add subgraphs for files with symbols
        for file_path, symbols in file_nodes.items():
            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
            subgraph_id = self._escape_label(file_name)
            lines.append(f"  subgraph {subgraph_id}")
            for node in symbols:
                lines.append("    " + self._format_node(node))
            lines.append("  end")

        return lines

    def _get_safe_id(self, original_id: str) -> str:
        """Get a safe Mermaid node ID."""
        if original_id not in self._node_id_map:
            safe_id = f"n{self._node_counter}"
            self._node_id_map[original_id] = safe_id
            self._node_counter += 1
        return self._node_id_map[original_id]

    def _get_node_label(self, node: CodeNode) -> str:
        """Get display label for node."""
        name = node.properties.get("name", node.id)
        # Truncate long labels
        if len(name) > 40:
            name = name[:37] + "..."
        return self._escape_label(name)

    def _get_shape_brackets(self, node: CodeNode) -> tuple[str, str]:
        """Get Mermaid shape brackets for node type."""
        if node.node_type == CodeNodeType.FILE:
            return "[(", ")]"  # Cylindrical for files
        elif node.node_type == CodeNodeType.SYMBOL:
            kind = node.properties.get("kind", "symbol")
            if kind == "class":
                return "[", "]"  # Rectangle for classes
            elif kind in ("function", "method"):
                return "(", ")"  # Rounded for functions
            return "[", "]"
        elif node.node_type == CodeNodeType.PACKAGE:
            return "[[", "]]"  # Subroutine shape for packages
        return "[", "]"

    def _get_node_class(self, node: CodeNode) -> str:
        """Get CSS class for node type."""
        if node.node_type == CodeNodeType.FILE:
            return "fileNode"
        elif node.node_type == CodeNodeType.SYMBOL:
            return "symbolNode"
        elif node.node_type == CodeNodeType.PACKAGE:
            return "packageNode"
        return "symbolNode"

    def _escape_label(self, s: str) -> str:
        """Escape a string for use in Mermaid label."""
        # Remove or replace characters that break Mermaid
        s = re.sub(r'["\[\]{}()<>|]', "", s)
        s = s.replace("\\", "")
        return s

    def get_format(self) -> ExportFormat:
        """Get format type."""
        return ExportFormat.MERMAID
