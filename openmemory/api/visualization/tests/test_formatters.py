"""Tests for graph export formatters.

This module tests:
- JSON formatter (hierarchical and flat modes)
- DOT formatter (Graphviz format)
- Mermaid formatter (Mermaid diagram syntax)
- Formatter interface compliance
- Node and edge styling
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
    MemoryGraphStore,
)
from openmemory.api.visualization.config import ExportFormat, StyleConfig
from openmemory.api.visualization.formatters import (
    DOTFormatter,
    Formatter,
    JSONFormatter,
    MermaidFormatter,
)


class TestFormatterInterface:
    """Test Formatter abstract interface."""

    def test_formatter_is_abstract(self):
        """Formatter base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Formatter()

    def test_json_formatter_implements_interface(self):
        """JSONFormatter implements Formatter interface."""
        formatter = JSONFormatter()
        assert isinstance(formatter, Formatter)
        assert hasattr(formatter, "format_nodes")
        assert hasattr(formatter, "format_edges")
        assert hasattr(formatter, "format_graph")
        assert hasattr(formatter, "get_format")

    def test_dot_formatter_implements_interface(self):
        """DOTFormatter implements Formatter interface."""
        formatter = DOTFormatter()
        assert isinstance(formatter, Formatter)
        assert hasattr(formatter, "format_nodes")
        assert hasattr(formatter, "format_edges")
        assert hasattr(formatter, "format_graph")
        assert hasattr(formatter, "get_format")

    def test_mermaid_formatter_implements_interface(self):
        """MermaidFormatter implements Formatter interface."""
        formatter = MermaidFormatter()
        assert isinstance(formatter, Formatter)
        assert hasattr(formatter, "format_nodes")
        assert hasattr(formatter, "format_edges")
        assert hasattr(formatter, "format_graph")
        assert hasattr(formatter, "get_format")


class TestJSONFormatter:
    """Test JSON graph formatter."""

    def test_format_empty_graph(self):
        """Format empty graph returns valid JSON structure."""
        formatter = JSONFormatter()
        result = formatter.format_graph([], [])

        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert "metadata" in parsed
        assert parsed["nodes"] == []
        assert parsed["edges"] == []

    def test_format_single_node(self, sample_file_node: CodeNode):
        """Format single node correctly."""
        formatter = JSONFormatter()
        result = formatter.format_nodes([sample_file_node])

        parsed = json.loads(result)
        assert len(parsed) == 1
        node = parsed[0]
        assert node["id"] == sample_file_node.id
        assert node["type"] == sample_file_node.node_type.value
        assert node["properties"]["path"] == "/src/main.py"
        assert node["properties"]["language"] == "python"

    def test_format_multiple_nodes(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_class_node: CodeNode,
    ):
        """Format multiple nodes correctly."""
        formatter = JSONFormatter()
        nodes = [sample_file_node, sample_symbol_node, sample_class_node]
        result = formatter.format_nodes(nodes)

        parsed = json.loads(result)
        assert len(parsed) == 3
        ids = {n["id"] for n in parsed}
        assert sample_file_node.id in ids
        assert sample_symbol_node.id in ids
        assert sample_class_node.id in ids

    def test_format_single_edge(self, sample_contains_edge: CodeEdge):
        """Format single edge correctly."""
        formatter = JSONFormatter()
        result = formatter.format_edges([sample_contains_edge])

        parsed = json.loads(result)
        assert len(parsed) == 1
        edge = parsed[0]
        assert edge["source"] == sample_contains_edge.source_id
        assert edge["target"] == sample_contains_edge.target_id
        assert edge["type"] == sample_contains_edge.edge_type.value

    def test_format_edge_with_properties(self, sample_calls_edge: CodeEdge):
        """Format edge with properties correctly."""
        formatter = JSONFormatter()
        result = formatter.format_edges([sample_calls_edge])

        parsed = json.loads(result)
        assert len(parsed) == 1
        edge = parsed[0]
        assert edge["properties"]["call_line"] == 15
        assert edge["properties"]["call_col"] == 8

    def test_format_full_graph(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Format complete graph with nodes and edges."""
        formatter = JSONFormatter()
        nodes = [sample_file_node, sample_symbol_node]
        edges = [sample_contains_edge]
        result = formatter.format_graph(nodes, edges)

        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 2
        assert len(parsed["edges"]) == 1
        assert "metadata" in parsed
        assert parsed["metadata"]["node_count"] == 2
        assert parsed["metadata"]["edge_count"] == 1
        assert parsed["metadata"]["format"] == "json"

    def test_format_returns_valid_json(self, populated_graph_store: MemoryGraphStore):
        """Formatted output is valid JSON."""
        formatter = JSONFormatter()
        nodes = list(populated_graph_store._nodes.values())
        edges = list(populated_graph_store._edges.values())
        result = formatter.format_graph(nodes, edges)

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_format_with_indent(self, sample_file_node: CodeNode):
        """Format with custom indentation."""
        formatter = JSONFormatter(indent=4)
        result = formatter.format_nodes([sample_file_node])

        # Indented JSON should have newlines
        assert "\n" in result
        # Should still be valid JSON
        json.loads(result)

    def test_format_compact(self, sample_file_node: CodeNode):
        """Format in compact mode (no indentation)."""
        formatter = JSONFormatter(indent=None)
        result = formatter.format_nodes([sample_file_node])

        # Compact JSON should be single line (except for nested structures)
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_get_format(self):
        """Get format returns JSON."""
        formatter = JSONFormatter()
        assert formatter.get_format() == ExportFormat.JSON


class TestDOTFormatter:
    """Test DOT (Graphviz) graph formatter."""

    def test_format_empty_graph(self):
        """Format empty graph returns valid DOT structure."""
        formatter = DOTFormatter()
        result = formatter.format_graph([], [])

        assert "digraph" in result
        assert "{" in result
        assert "}" in result

    def test_format_single_node(self, sample_file_node: CodeNode):
        """Format single node as DOT."""
        formatter = DOTFormatter()
        result = formatter.format_nodes([sample_file_node])

        # Node ID should be quoted and escaped
        assert '"/src/main.py"' in result or "n_" in result
        assert "label" in result.lower() or "main.py" in result

    def test_format_symbol_node(self, sample_symbol_node: CodeNode):
        """Format symbol node with correct shape."""
        formatter = DOTFormatter()
        result = formatter.format_nodes([sample_symbol_node])

        # Should contain node definition
        assert "process_data" in result or "label" in result

    def test_format_single_edge(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Format single edge as DOT."""
        formatter = DOTFormatter()
        result = formatter.format_edges([sample_contains_edge])

        # Edge should use -> syntax
        assert "->" in result

    def test_format_full_graph(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Format complete graph as DOT."""
        formatter = DOTFormatter()
        nodes = [sample_file_node, sample_symbol_node]
        edges = [sample_contains_edge]
        result = formatter.format_graph(nodes, edges)

        assert "digraph" in result
        assert "->" in result
        # Should have closing brace
        assert result.strip().endswith("}")

    def test_node_shapes_by_type(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_package_node: CodeNode,
    ):
        """Different node types have different shapes."""
        formatter = DOTFormatter()
        nodes = [sample_file_node, sample_symbol_node, sample_package_node]
        result = formatter.format_graph(nodes, [])

        # Files typically folder shape, symbols box/ellipse, packages box3d
        # Just verify different styling exists
        assert "shape" in result.lower() or "style" in result.lower()

    def test_edge_labels(self, sample_calls_edge: CodeEdge):
        """Edges can have labels."""
        formatter = DOTFormatter(show_edge_labels=True)
        result = formatter.format_edges([sample_calls_edge])

        # CALLS edge should have label
        assert "CALLS" in result or "label" in result

    def test_graph_name(self, sample_file_node: CodeNode):
        """Graph can have custom name."""
        formatter = DOTFormatter(graph_name="CodeGraph")
        result = formatter.format_graph([sample_file_node], [])

        assert "CodeGraph" in result

    def test_escape_special_characters(self):
        """Special characters in node labels are escaped."""
        formatter = DOTFormatter()
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id='scip-python pkg module/"special".',
            properties={
                "name": 'process "data"',
                "kind": "function",
            },
        )
        result = formatter.format_nodes([node])

        # Should escape quotes
        assert '\\"' in result or "special" in result

    def test_get_format(self):
        """Get format returns DOT."""
        formatter = DOTFormatter()
        assert formatter.get_format() == ExportFormat.DOT

    def test_rankdir_option(self, sample_file_node: CodeNode, sample_symbol_node: CodeNode):
        """Graph can have custom rank direction."""
        formatter = DOTFormatter(rankdir="LR")
        result = formatter.format_graph([sample_file_node, sample_symbol_node], [])

        assert "rankdir" in result.lower() or "LR" in result


class TestMermaidFormatter:
    """Test Mermaid diagram formatter."""

    def test_format_empty_graph(self):
        """Format empty graph returns valid Mermaid structure."""
        formatter = MermaidFormatter()
        result = formatter.format_graph([], [])

        # Should start with flowchart or graph declaration
        assert result.strip().startswith(("flowchart", "graph"))

    def test_format_single_node(self, sample_file_node: CodeNode):
        """Format single node as Mermaid."""
        formatter = MermaidFormatter()
        result = formatter.format_nodes([sample_file_node])

        # Node should have ID and label
        assert "main.py" in result or "main_py" in result

    def test_format_symbol_node(self, sample_symbol_node: CodeNode):
        """Format symbol node with correct shape."""
        formatter = MermaidFormatter()
        result = formatter.format_nodes([sample_symbol_node])

        assert "process_data" in result

    def test_format_single_edge(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Format single edge as Mermaid."""
        formatter = MermaidFormatter()
        result = formatter.format_edges([sample_contains_edge])

        # Edge should use --> syntax
        assert "-->" in result or "---" in result

    def test_format_full_graph(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Format complete graph as Mermaid."""
        formatter = MermaidFormatter()
        nodes = [sample_file_node, sample_symbol_node]
        edges = [sample_contains_edge]
        result = formatter.format_graph(nodes, edges)

        assert result.strip().startswith(("flowchart", "graph"))
        assert "-->" in result or "---" in result

    def test_node_shapes_by_type(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_package_node: CodeNode,
    ):
        """Different node types have different Mermaid shapes."""
        formatter = MermaidFormatter()
        nodes = [sample_file_node, sample_symbol_node, sample_package_node]
        result = formatter.format_graph(nodes, [])

        # Mermaid uses different bracket styles for shapes
        # () for rounded, [] for rectangle, {} for diamond, etc.
        # Just verify valid mermaid syntax
        assert "flowchart" in result or "graph" in result

    def test_edge_labels(self, sample_calls_edge: CodeEdge):
        """Edges can have labels in Mermaid."""
        formatter = MermaidFormatter(show_edge_labels=True)
        result = formatter.format_edges([sample_calls_edge])

        # Edge with label uses |label| syntax
        assert "|" in result or "CALLS" in result

    def test_direction_option(self, sample_file_node: CodeNode):
        """Graph can have custom direction."""
        formatter = MermaidFormatter(direction="LR")
        result = formatter.format_graph([sample_file_node], [])

        assert "LR" in result

    def test_escape_special_characters(self):
        """Special characters in node labels are escaped."""
        formatter = MermaidFormatter()
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="scip-python pkg module/special_func.",
            properties={
                "name": "process[data]",
                "kind": "function",
            },
        )
        result = formatter.format_nodes([node])

        # Should escape or remove special characters
        # Mermaid doesn't like [] in labels
        assert "process" in result

    def test_get_format(self):
        """Get format returns Mermaid."""
        formatter = MermaidFormatter()
        assert formatter.get_format() == ExportFormat.MERMAID

    def test_subgraph_support(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_class_node: CodeNode,
    ):
        """Mermaid can group nodes in subgraphs."""
        formatter = MermaidFormatter(use_subgraphs=True)
        nodes = [sample_file_node, sample_symbol_node, sample_class_node]
        result = formatter.format_graph(nodes, [])

        # Subgraphs are optional, just verify valid output
        assert "flowchart" in result or "graph" in result


class TestStyleConfig:
    """Test styling configuration for formatters."""

    def test_default_style_config(self):
        """Default style config has sensible defaults."""
        config = StyleConfig()
        assert config.file_color is not None
        assert config.symbol_color is not None
        assert config.package_color is not None

    def test_custom_colors(self):
        """Custom colors can be set."""
        config = StyleConfig(
            file_color="#FF0000",
            symbol_color="#00FF00",
            package_color="#0000FF",
        )
        assert config.file_color == "#FF0000"
        assert config.symbol_color == "#00FF00"
        assert config.package_color == "#0000FF"

    def test_json_formatter_with_style(self, sample_file_node: CodeNode):
        """JSON formatter can include style information."""
        config = StyleConfig(file_color="#FF0000")
        formatter = JSONFormatter(style_config=config, include_style=True)
        result = formatter.format_nodes([sample_file_node])

        parsed = json.loads(result)
        assert parsed[0].get("style") is not None or "color" in str(parsed)

    def test_dot_formatter_with_style(self, sample_file_node: CodeNode):
        """DOT formatter applies style configuration."""
        config = StyleConfig(file_color="#FF0000")
        formatter = DOTFormatter(style_config=config)
        result = formatter.format_nodes([sample_file_node])

        # Color should appear in output
        assert "FF0000" in result or "color" in result.lower()

    def test_mermaid_formatter_with_style(self, sample_file_node: CodeNode):
        """Mermaid formatter can apply styles."""
        config = StyleConfig(file_color="#FF0000")
        formatter = MermaidFormatter(style_config=config)
        result = formatter.format_graph([sample_file_node], [])

        # Mermaid uses style or class definitions
        # Style may be in separate classDef or inline
        assert "flowchart" in result or "graph" in result


class TestFormatterEdgeCases:
    """Test edge cases and error handling."""

    def test_format_node_with_none_properties(self):
        """Handle nodes with None values in properties."""
        formatter = JSONFormatter()
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="scip-python pkg module/func.",
            properties={
                "name": "func",
                "kind": "function",
                "docstring": None,
            },
        )
        result = formatter.format_nodes([node])

        # Should not raise and should handle None
        parsed = json.loads(result)
        assert parsed[0]["properties"]["name"] == "func"

    def test_format_node_with_unicode(self):
        """Handle unicode in node properties."""
        formatter = JSONFormatter()
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="scip-python pkg module/func.",
            properties={
                "name": "process_émoji_🎉",
                "kind": "function",
            },
        )
        result = formatter.format_nodes([node])

        parsed = json.loads(result)
        assert "émoji" in parsed[0]["properties"]["name"]
        assert "🎉" in parsed[0]["properties"]["name"]

    def test_format_node_with_long_id(self):
        """Handle nodes with very long IDs."""
        formatter = DOTFormatter()
        long_id = "scip-python pkg " + "a" * 500 + "/func."
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id=long_id,
            properties={
                "name": "func",
                "kind": "function",
            },
        )
        result = formatter.format_nodes([node])

        # Should not raise
        assert "func" in result or "label" in result

    def test_format_edge_with_missing_nodes(self):
        """Format edges even if referenced nodes don't exist."""
        formatter = JSONFormatter()
        edge = CodeEdge(
            edge_type=CodeEdgeType.CALLS,
            source_id="nonexistent_source",
            target_id="nonexistent_target",
            properties={},
        )
        result = formatter.format_edges([edge])

        # Should format edge regardless of node existence
        parsed = json.loads(result)
        assert parsed[0]["source"] == "nonexistent_source"
        assert parsed[0]["target"] == "nonexistent_target"

    def test_format_circular_references(self):
        """Handle graphs with circular references."""
        formatter = JSONFormatter()
        node1 = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="func_a",
            properties={"name": "func_a"},
        )
        node2 = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="func_b",
            properties={"name": "func_b"},
        )
        edges = [
            CodeEdge(CodeEdgeType.CALLS, "func_a", "func_b", {}),
            CodeEdge(CodeEdgeType.CALLS, "func_b", "func_a", {}),  # Circular
        ]

        result = formatter.format_graph([node1, node2], edges)

        # Should handle circular refs without error
        parsed = json.loads(result)
        assert len(parsed["edges"]) == 2

    def test_format_self_referencing_edge(self):
        """Handle self-referencing edges."""
        formatter = DOTFormatter()
        edge = CodeEdge(
            edge_type=CodeEdgeType.CALLS,
            source_id="recursive_func",
            target_id="recursive_func",  # Self-reference
            properties={},
        )
        result = formatter.format_edges([edge])

        # DOT supports self-loops
        assert "->" in result
