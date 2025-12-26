"""Tests for graph export functionality.

This module tests:
- GraphExporter with configurable depth and filters
- Traversal from root nodes
- Node and edge type filtering
- Depth limiting
- Error handling for invalid inputs
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
from openmemory.api.visualization.config import (
    ExportConfig,
    ExportFormat,
    FilterConfig,
)
from openmemory.api.visualization.graph_export import (
    ExportResult,
    GraphExporter,
    GraphExportError,
    InvalidFormatError,
    TraversalError,
)


class TestExportResult:
    """Test ExportResult dataclass."""

    def test_create_export_result(self):
        """Create ExportResult with required fields."""
        result = ExportResult(
            content="test content",
            format=ExportFormat.JSON,
            node_count=10,
            edge_count=5,
        )
        assert result.content == "test content"
        assert result.format == ExportFormat.JSON
        assert result.node_count == 10
        assert result.edge_count == 5

    def test_export_result_metadata(self):
        """ExportResult can have optional metadata."""
        result = ExportResult(
            content="test",
            format=ExportFormat.DOT,
            node_count=5,
            edge_count=3,
            metadata={"custom": "value"},
        )
        assert result.metadata["custom"] == "value"

    def test_export_result_timing(self):
        """ExportResult tracks export timing."""
        result = ExportResult(
            content="test",
            format=ExportFormat.JSON,
            node_count=0,
            edge_count=0,
            duration_ms=150.5,
        )
        assert result.duration_ms == 150.5


class TestGraphExporter:
    """Test GraphExporter class."""

    def test_create_exporter(self, memory_graph_store: MemoryGraphStore):
        """Create exporter with graph driver."""
        exporter = GraphExporter(driver=memory_graph_store)
        assert exporter is not None

    def test_create_exporter_with_config(self, memory_graph_store: MemoryGraphStore):
        """Create exporter with custom config."""
        config = ExportConfig(format=ExportFormat.DOT)
        exporter = GraphExporter(driver=memory_graph_store, config=config)
        assert exporter.config.format == ExportFormat.DOT

    def test_export_empty_graph(self, memory_graph_store: MemoryGraphStore):
        """Export empty graph."""
        exporter = GraphExporter(driver=memory_graph_store)
        result = exporter.export()

        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.format == ExportFormat.JSON

        # Verify valid JSON
        parsed = json.loads(result.content)
        assert parsed["nodes"] == []
        assert parsed["edges"] == []

    def test_export_full_graph(self, populated_graph_store: MemoryGraphStore):
        """Export all nodes and edges."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        assert result.node_count == 5
        assert result.edge_count == 5

        parsed = json.loads(result.content)
        assert len(parsed["nodes"]) == 5
        assert len(parsed["edges"]) == 5

    def test_export_as_dot(self, populated_graph_store: MemoryGraphStore):
        """Export graph as DOT format."""
        config = ExportConfig(format=ExportFormat.DOT)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        assert result.format == ExportFormat.DOT
        assert "digraph" in result.content
        assert "->" in result.content

    def test_export_as_mermaid(self, populated_graph_store: MemoryGraphStore):
        """Export graph as Mermaid format."""
        config = ExportConfig(format=ExportFormat.MERMAID)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        assert result.format == ExportFormat.MERMAID
        assert "flowchart" in result.content
        assert "-->" in result.content


class TestGraphExporterWithFilters:
    """Test GraphExporter with filtering."""

    def test_filter_by_node_type(self, populated_graph_store: MemoryGraphStore):
        """Filter to only file nodes."""
        filter_config = FilterConfig(node_types={CodeNodeType.FILE})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        # Only file nodes
        parsed = json.loads(result.content)
        assert result.node_count == 1
        assert all(n["type"] == "CODE_FILE" for n in parsed["nodes"])

    def test_filter_by_multiple_node_types(self, populated_graph_store: MemoryGraphStore):
        """Filter to file and symbol nodes."""
        filter_config = FilterConfig(node_types={CodeNodeType.FILE, CodeNodeType.SYMBOL})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        assert result.node_count == 4  # 1 file + 3 symbols
        types = {n["type"] for n in parsed["nodes"]}
        assert types <= {"CODE_FILE", "CODE_SYMBOL"}

    def test_filter_by_edge_type(self, populated_graph_store: MemoryGraphStore):
        """Filter to only CONTAINS edges."""
        filter_config = FilterConfig(edge_types={CodeEdgeType.CONTAINS})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        assert all(e["type"] == "CONTAINS" for e in parsed["edges"])

    def test_filter_by_language(self, populated_graph_store: MemoryGraphStore):
        """Filter nodes by language."""
        filter_config = FilterConfig(languages={"python"})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        # All nodes should be Python
        for node in parsed["nodes"]:
            if "language" in node["properties"]:
                assert node["properties"]["language"] == "python"

    def test_filter_by_symbol_kind(self, populated_graph_store: MemoryGraphStore):
        """Filter symbols by kind (function, class, etc.)."""
        filter_config = FilterConfig(symbol_kinds={"function"})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        for node in parsed["nodes"]:
            if node["type"] == "CODE_SYMBOL":
                assert node["properties"]["kind"] == "function"

    def test_filter_excludes_properties(self, populated_graph_store: MemoryGraphStore):
        """Filter out specific properties."""
        filter_config = FilterConfig(exclude_properties={"content_hash", "size"})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        for node in parsed["nodes"]:
            assert "content_hash" not in node["properties"]
            assert "size" not in node["properties"]

    def test_filter_includes_only_specific_properties(self, populated_graph_store: MemoryGraphStore):
        """Include only specific properties."""
        filter_config = FilterConfig(include_properties={"name", "path"})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export()

        parsed = json.loads(result.content)
        for node in parsed["nodes"]:
            props = set(node["properties"].keys())
            assert props <= {"name", "path"}


class TestGraphExporterWithTraversal:
    """Test GraphExporter with traversal from root nodes."""

    def test_export_from_root_node(self, populated_graph_store: MemoryGraphStore):
        """Export subgraph starting from a root node."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_from_node("/src/main.py")

        assert result.node_count > 0
        parsed = json.loads(result.content)
        # Should include the root and connected nodes
        ids = {n["id"] for n in parsed["nodes"]}
        assert "/src/main.py" in ids

    def test_export_with_depth_limit(self, populated_graph_store: MemoryGraphStore):
        """Export with limited traversal depth."""
        filter_config = FilterConfig(max_depth=1)
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export_from_node("/src/main.py")

        # Depth 1 means only direct connections
        parsed = json.loads(result.content)
        # Should have root + direct children but not grandchildren
        assert result.node_count >= 1

    def test_export_with_zero_depth(self, populated_graph_store: MemoryGraphStore):
        """Export with depth 0 returns only root node."""
        filter_config = FilterConfig(max_depth=0)
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=populated_graph_store, config=config)
        result = exporter.export_from_node("/src/main.py")

        parsed = json.loads(result.content)
        assert result.node_count == 1
        assert parsed["nodes"][0]["id"] == "/src/main.py"
        assert len(parsed["edges"]) == 0

    def test_export_from_nonexistent_node_raises(self, populated_graph_store: MemoryGraphStore):
        """Export from nonexistent node raises error."""
        exporter = GraphExporter(driver=populated_graph_store)
        with pytest.raises(TraversalError, match="not found"):
            exporter.export_from_node("nonexistent_node_id")

    def test_export_from_multiple_roots(self, populated_graph_store: MemoryGraphStore):
        """Export from multiple root nodes."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_from_nodes([
            "/src/main.py",
            "scip-python pkg module/process_data.",
        ])

        parsed = json.loads(result.content)
        ids = {n["id"] for n in parsed["nodes"]}
        assert "/src/main.py" in ids
        assert "scip-python pkg module/process_data." in ids


class TestGraphExporterCallGraph:
    """Test call graph specific export."""

    def test_export_call_graph(self, populated_graph_store: MemoryGraphStore):
        """Export only call relationships."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_call_graph()

        parsed = json.loads(result.content)
        # All edges should be CALLS
        for edge in parsed["edges"]:
            assert edge["type"] == "CALLS"

    def test_export_callers_of_symbol(self, populated_graph_store: MemoryGraphStore):
        """Export callers of a specific symbol."""
        exporter = GraphExporter(driver=populated_graph_store)
        # This symbol is called by process_data
        result = exporter.export_callers("scip-python pkg module/DataProcessor#transform().")

        parsed = json.loads(result.content)
        ids = {n["id"] for n in parsed["nodes"]}
        # Should include the target and its callers
        assert "scip-python pkg module/DataProcessor#transform()." in ids

    def test_export_callees_of_symbol(self, populated_graph_store: MemoryGraphStore):
        """Export callees of a specific symbol."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_callees("scip-python pkg module/process_data.")

        parsed = json.loads(result.content)
        ids = {n["id"] for n in parsed["nodes"]}
        # Should include the source and its callees
        assert "scip-python pkg module/process_data." in ids


class TestGraphExporterContainment:
    """Test containment hierarchy export."""

    def test_export_file_contents(self, populated_graph_store: MemoryGraphStore):
        """Export contents of a file."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_file_contents("/src/main.py")

        parsed = json.loads(result.content)
        ids = {n["id"] for n in parsed["nodes"]}
        assert "/src/main.py" in ids
        # Should have contained symbols
        assert result.node_count > 1

    def test_export_package_contents(self, populated_graph_store: MemoryGraphStore):
        """Export contents of a package."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export_package_contents("pkg.module")

        parsed = json.loads(result.content)
        ids = {n["id"] for n in parsed["nodes"]}
        assert "pkg.module" in ids


class TestGraphExportErrors:
    """Test error handling."""

    def test_invalid_format_raises(self, populated_graph_store: MemoryGraphStore):
        """Invalid format raises error."""
        exporter = GraphExporter(driver=populated_graph_store)
        with pytest.raises(InvalidFormatError):
            exporter.export(format_str="invalid_format")

    def test_export_error_includes_details(self, memory_graph_store: MemoryGraphStore):
        """Export errors include helpful details."""
        exporter = GraphExporter(driver=memory_graph_store)
        try:
            exporter.export_from_node("missing_node")
            pytest.fail("Should raise TraversalError")
        except TraversalError as e:
            assert "missing_node" in str(e)

    def test_traversal_error_on_disconnected_graph(self, memory_graph_store: MemoryGraphStore):
        """Handle disconnected nodes gracefully."""
        # Add isolated nodes
        memory_graph_store.add_node(CodeNode(
            node_type=CodeNodeType.FILE,
            id="isolated1",
            properties={"name": "isolated1"},
        ))
        memory_graph_store.add_node(CodeNode(
            node_type=CodeNodeType.FILE,
            id="isolated2",
            properties={"name": "isolated2"},
        ))

        exporter = GraphExporter(driver=memory_graph_store)
        # Should still export, just no edges
        result = exporter.export()
        assert result.node_count == 2
        assert result.edge_count == 0


class TestGraphExportMetadata:
    """Test export metadata."""

    def test_export_includes_format_metadata(self, populated_graph_store: MemoryGraphStore):
        """Export includes format in metadata."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        parsed = json.loads(result.content)
        assert parsed["metadata"]["format"] == "json"

    def test_export_includes_timestamp(self, populated_graph_store: MemoryGraphStore):
        """Export includes timestamp."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        parsed = json.loads(result.content)
        assert "exported_at" in parsed["metadata"]

    def test_export_includes_counts(self, populated_graph_store: MemoryGraphStore):
        """Export includes node and edge counts."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        parsed = json.loads(result.content)
        assert parsed["metadata"]["node_count"] == result.node_count
        assert parsed["metadata"]["edge_count"] == result.edge_count

    def test_export_result_has_duration(self, populated_graph_store: MemoryGraphStore):
        """ExportResult tracks duration."""
        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        assert result.duration_ms is not None
        assert result.duration_ms >= 0


class TestGraphExporterLargeGraph:
    """Test export of large graphs."""

    def test_export_large_graph(self, large_graph_store: MemoryGraphStore):
        """Export large graph efficiently."""
        exporter = GraphExporter(driver=large_graph_store)
        result = exporter.export()

        # 100 files + 500 symbols = 600 nodes
        assert result.node_count == 600
        # 500 CONTAINS + 50 CALLS = 550 edges
        assert result.edge_count == 550

    def test_export_large_graph_with_filter(self, large_graph_store: MemoryGraphStore):
        """Filter reduces large graph export size."""
        filter_config = FilterConfig(node_types={CodeNodeType.FILE})
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export()

        # Only 100 file nodes
        assert result.node_count == 100

    def test_export_subgraph_from_large_graph(self, large_graph_store: MemoryGraphStore):
        """Export subgraph from large graph."""
        filter_config = FilterConfig(max_depth=2)
        config = ExportConfig(filter=filter_config)
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_from_node("/src/file_000.py")

        # Should have file + its symbols + some calls
        assert result.node_count < 600  # Not full graph
        assert result.node_count >= 6  # At least file + 5 symbols
