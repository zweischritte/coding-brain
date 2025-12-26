"""Tests for hierarchical code graph JSON schema.

This module tests:
- HierarchicalNode structure
- HierarchicalGraph with parent-child relationships
- Schema validation
- Schema compliance for exports
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
from openmemory.api.visualization.schema import (
    HierarchicalGraph,
    HierarchicalNode,
    SchemaValidationError,
    SchemaValidator,
)


class TestHierarchicalNode:
    """Test HierarchicalNode dataclass."""

    def test_create_minimal_node(self):
        """Create node with minimal required fields."""
        node = HierarchicalNode(
            id="test-id",
            type="CODE_FILE",
            name="test.py",
        )
        assert node.id == "test-id"
        assert node.type == "CODE_FILE"
        assert node.name == "test.py"
        assert node.children == []
        assert node.properties == {}

    def test_create_node_with_children(self):
        """Create node with children."""
        child = HierarchicalNode(
            id="child-id",
            type="CODE_SYMBOL",
            name="func",
        )
        parent = HierarchicalNode(
            id="parent-id",
            type="CODE_FILE",
            name="test.py",
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0].id == "child-id"

    def test_create_node_with_properties(self):
        """Create node with custom properties."""
        node = HierarchicalNode(
            id="test-id",
            type="CODE_SYMBOL",
            name="MyClass",
            properties={
                "kind": "class",
                "line_start": 10,
                "line_end": 50,
                "language": "python",
            },
        )
        assert node.properties["kind"] == "class"
        assert node.properties["line_start"] == 10

    def test_node_with_parent_reference(self):
        """Node can track parent ID."""
        node = HierarchicalNode(
            id="test-id",
            type="CODE_SYMBOL",
            name="func",
            parent_id="parent-file",
        )
        assert node.parent_id == "parent-file"

    def test_node_to_dict(self):
        """Convert node to dictionary."""
        child = HierarchicalNode(
            id="child-id",
            type="CODE_SYMBOL",
            name="func",
        )
        node = HierarchicalNode(
            id="parent-id",
            type="CODE_FILE",
            name="test.py",
            children=[child],
            properties={"language": "python"},
        )
        result = node.to_dict()

        assert result["id"] == "parent-id"
        assert result["type"] == "CODE_FILE"
        assert result["name"] == "test.py"
        assert result["properties"]["language"] == "python"
        assert len(result["children"]) == 1
        assert result["children"][0]["id"] == "child-id"

    def test_node_to_dict_excludes_empty_optional(self):
        """to_dict excludes empty optional fields when configured."""
        node = HierarchicalNode(
            id="test-id",
            type="CODE_FILE",
            name="test.py",
        )
        result = node.to_dict(exclude_empty=True)

        # Empty children and properties should be excluded
        assert "children" not in result or result["children"] == []
        assert "properties" not in result or result["properties"] == {}


class TestHierarchicalGraph:
    """Test HierarchicalGraph structure."""

    def test_create_empty_graph(self):
        """Create empty hierarchical graph."""
        graph = HierarchicalGraph(
            version="1.0",
            roots=[],
        )
        assert graph.version == "1.0"
        assert graph.roots == []

    def test_create_graph_with_roots(self):
        """Create graph with root nodes."""
        file_node = HierarchicalNode(
            id="file-1",
            type="CODE_FILE",
            name="main.py",
        )
        graph = HierarchicalGraph(
            version="1.0",
            roots=[file_node],
        )
        assert len(graph.roots) == 1
        assert graph.roots[0].id == "file-1"

    def test_create_graph_with_metadata(self):
        """Create graph with metadata."""
        graph = HierarchicalGraph(
            version="1.0",
            roots=[],
            metadata={
                "exported_at": "2024-01-01T00:00:00Z",
                "node_count": 100,
                "edge_count": 50,
            },
        )
        assert graph.metadata["node_count"] == 100

    def test_graph_to_dict(self):
        """Convert graph to dictionary."""
        node = HierarchicalNode(
            id="file-1",
            type="CODE_FILE",
            name="main.py",
        )
        graph = HierarchicalGraph(
            version="1.0",
            roots=[node],
            metadata={"count": 1},
        )
        result = graph.to_dict()

        assert result["version"] == "1.0"
        assert len(result["roots"]) == 1
        assert result["metadata"]["count"] == 1

    def test_graph_to_json(self):
        """Convert graph to JSON string."""
        node = HierarchicalNode(
            id="file-1",
            type="CODE_FILE",
            name="main.py",
        )
        graph = HierarchicalGraph(
            version="1.0",
            roots=[node],
        )
        json_str = graph.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["version"] == "1.0"
        assert parsed["roots"][0]["id"] == "file-1"

    def test_graph_to_json_pretty(self):
        """Convert graph to pretty-printed JSON."""
        node = HierarchicalNode(
            id="file-1",
            type="CODE_FILE",
            name="main.py",
        )
        graph = HierarchicalGraph(
            version="1.0",
            roots=[node],
        )
        json_str = graph.to_json(indent=2)

        # Should have newlines for pretty printing
        assert "\n" in json_str


class TestHierarchicalGraphFromCodeNodes:
    """Test building hierarchical graph from CodeNodes."""

    def test_build_from_single_file(self, sample_file_node: CodeNode):
        """Build hierarchical graph from single file node."""
        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_file_node],
            edges=[],
        )
        assert len(graph.roots) == 1
        assert graph.roots[0].id == sample_file_node.id
        assert graph.roots[0].type == "CODE_FILE"

    def test_build_from_file_with_symbols(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Build hierarchical graph with file containing symbols."""
        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_file_node, sample_symbol_node],
            edges=[sample_contains_edge],
        )

        # File should be root
        assert len(graph.roots) == 1
        assert graph.roots[0].id == sample_file_node.id

        # Symbol should be child of file
        assert len(graph.roots[0].children) == 1
        assert graph.roots[0].children[0].id == sample_symbol_node.id

    def test_build_nested_hierarchy(
        self,
        sample_file_node: CodeNode,
        sample_class_node: CodeNode,
        sample_method_node: CodeNode,
    ):
        """Build nested hierarchy: file > class > method."""
        file_contains_class = CodeEdge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=sample_file_node.id,
            target_id=sample_class_node.id,
            properties={},
        )
        class_contains_method = CodeEdge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=sample_class_node.id,
            target_id=sample_method_node.id,
            properties={},
        )

        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_file_node, sample_class_node, sample_method_node],
            edges=[file_contains_class, class_contains_method],
        )

        # File is root
        assert len(graph.roots) == 1
        file_hier = graph.roots[0]
        assert file_hier.id == sample_file_node.id

        # Class is child of file
        assert len(file_hier.children) == 1
        class_hier = file_hier.children[0]
        assert class_hier.id == sample_class_node.id

        # Method is child of class
        assert len(class_hier.children) == 1
        method_hier = class_hier.children[0]
        assert method_hier.id == sample_method_node.id

    def test_build_with_multiple_roots(
        self,
        sample_file_node: CodeNode,
        sample_package_node: CodeNode,
    ):
        """Build graph with multiple root nodes."""
        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_file_node, sample_package_node],
            edges=[],  # No containment edges
        )

        # Both should be roots
        assert len(graph.roots) == 2
        ids = {r.id for r in graph.roots}
        assert sample_file_node.id in ids
        assert sample_package_node.id in ids

    def test_build_includes_non_containment_edges(
        self,
        sample_symbol_node: CodeNode,
        sample_method_node: CodeNode,
        sample_calls_edge: CodeEdge,
    ):
        """Non-containment edges are preserved in properties."""
        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_symbol_node, sample_method_node],
            edges=[sample_calls_edge],
        )

        # Both nodes are roots (no CONTAINS edge)
        assert len(graph.roots) == 2

        # CALLS edge should be in graph edges
        assert sample_calls_edge.source_id in graph.edge_list[0]


class TestSchemaValidator:
    """Test schema validation."""

    def test_create_validator(self):
        """Create schema validator."""
        validator = SchemaValidator()
        assert validator is not None

    def test_validate_minimal_node(self):
        """Validate minimal valid node."""
        validator = SchemaValidator()
        node_dict = {
            "id": "test-id",
            "type": "CODE_FILE",
            "name": "test.py",
        }
        assert validator.validate_node(node_dict)

    def test_validate_node_missing_id(self):
        """Node without id is invalid."""
        validator = SchemaValidator()
        node_dict = {
            "type": "CODE_FILE",
            "name": "test.py",
        }
        with pytest.raises(SchemaValidationError, match="id"):
            validator.validate_node(node_dict)

    def test_validate_node_missing_type(self):
        """Node without type is invalid."""
        validator = SchemaValidator()
        node_dict = {
            "id": "test-id",
            "name": "test.py",
        }
        with pytest.raises(SchemaValidationError, match="type"):
            validator.validate_node(node_dict)

    def test_validate_node_missing_name(self):
        """Node without name is invalid."""
        validator = SchemaValidator()
        node_dict = {
            "id": "test-id",
            "type": "CODE_FILE",
        }
        with pytest.raises(SchemaValidationError, match="name"):
            validator.validate_node(node_dict)

    def test_validate_node_invalid_type(self):
        """Node with invalid type is rejected."""
        validator = SchemaValidator()
        node_dict = {
            "id": "test-id",
            "type": "INVALID_TYPE",
            "name": "test.py",
        }
        with pytest.raises(SchemaValidationError, match="type"):
            validator.validate_node(node_dict)

    def test_validate_node_with_children(self):
        """Validate node with valid children."""
        validator = SchemaValidator()
        node_dict = {
            "id": "parent-id",
            "type": "CODE_FILE",
            "name": "test.py",
            "children": [
                {
                    "id": "child-id",
                    "type": "CODE_SYMBOL",
                    "name": "func",
                }
            ],
        }
        assert validator.validate_node(node_dict)

    def test_validate_node_with_invalid_child(self):
        """Node with invalid child is rejected."""
        validator = SchemaValidator()
        node_dict = {
            "id": "parent-id",
            "type": "CODE_FILE",
            "name": "test.py",
            "children": [
                {
                    "id": "child-id",
                    # Missing type and name
                }
            ],
        }
        with pytest.raises(SchemaValidationError):
            validator.validate_node(node_dict)

    def test_validate_graph(self):
        """Validate complete graph structure."""
        validator = SchemaValidator()
        graph_dict = {
            "version": "1.0",
            "roots": [
                {
                    "id": "file-1",
                    "type": "CODE_FILE",
                    "name": "main.py",
                }
            ],
            "metadata": {},
        }
        assert validator.validate_graph(graph_dict)

    def test_validate_graph_missing_version(self):
        """Graph without version is invalid."""
        validator = SchemaValidator()
        graph_dict = {
            "roots": [],
            "metadata": {},
        }
        with pytest.raises(SchemaValidationError, match="version"):
            validator.validate_graph(graph_dict)

    def test_validate_graph_missing_roots(self):
        """Graph without roots is invalid."""
        validator = SchemaValidator()
        graph_dict = {
            "version": "1.0",
            "metadata": {},
        }
        with pytest.raises(SchemaValidationError, match="roots"):
            validator.validate_graph(graph_dict)

    def test_validate_graph_with_invalid_root(self):
        """Graph with invalid root node is rejected."""
        validator = SchemaValidator()
        graph_dict = {
            "version": "1.0",
            "roots": [
                {"id": "invalid"}  # Missing type and name
            ],
        }
        with pytest.raises(SchemaValidationError):
            validator.validate_graph(graph_dict)


class TestSchemaValidatorTypes:
    """Test type validation."""

    def test_valid_node_types(self):
        """All valid node types are accepted."""
        validator = SchemaValidator()
        valid_types = ["CODE_FILE", "CODE_SYMBOL", "CODE_PACKAGE"]

        for node_type in valid_types:
            node_dict = {
                "id": "test-id",
                "type": node_type,
                "name": "test",
            }
            assert validator.validate_node(node_dict)

    def test_valid_property_types(self):
        """Properties with valid types are accepted."""
        validator = SchemaValidator()
        node_dict = {
            "id": "test-id",
            "type": "CODE_SYMBOL",
            "name": "test",
            "properties": {
                "string_prop": "value",
                "int_prop": 42,
                "bool_prop": True,
                "float_prop": 3.14,
                "list_prop": [1, 2, 3],
                "none_prop": None,
            },
        }
        assert validator.validate_node(node_dict)


class TestSchemaCompliance:
    """Test schema compliance for exported graphs."""

    def test_exported_graph_passes_validation(
        self,
        populated_graph_store: MemoryGraphStore,
    ):
        """Exported graph passes schema validation."""
        from openmemory.api.visualization.graph_export import GraphExporter

        exporter = GraphExporter(driver=populated_graph_store)
        result = exporter.export()

        # Parse and validate
        parsed = json.loads(result.content)
        # JSON format uses flat structure, so we convert
        graph = HierarchicalGraph.from_flat_export(parsed)

        validator = SchemaValidator()
        graph_dict = graph.to_dict()
        assert validator.validate_graph(graph_dict)

    def test_hierarchical_export_passes_validation(
        self,
        sample_file_node: CodeNode,
        sample_symbol_node: CodeNode,
        sample_contains_edge: CodeEdge,
    ):
        """Hierarchical export passes schema validation."""
        graph = HierarchicalGraph.from_code_nodes(
            nodes=[sample_file_node, sample_symbol_node],
            edges=[sample_contains_edge],
        )

        validator = SchemaValidator()
        graph_dict = graph.to_dict()
        assert validator.validate_graph(graph_dict)
