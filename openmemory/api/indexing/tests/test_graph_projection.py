"""Tests for CODE_* graph projection in Neo4j.

This module tests:
- Node creation (CODE_FILE, CODE_SYMBOL, CODE_PACKAGE)
- Edge creation (CONTAINS, DEFINES, IMPORTS, CALLS, READS, WRITES, DATA_FLOWS_TO)
- Incremental updates (add/modify/delete)
- Transaction support (atomic operations)
- Neo4j constraint creation
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType
from openmemory.api.indexing.scip_symbols import (
    SCIPScheme,
    SCIPSymbolID,
    SCIPDescriptor,
)
from openmemory.api.indexing.graph_projection import (
    # Core types
    CodeNode,
    CodeNodeType,
    CodeEdge,
    CodeEdgeType,
    # Projection service
    GraphProjection,
    GraphProjectionConfig,
    # Builders
    FileNodeBuilder,
    SymbolNodeBuilder,
    PackageNodeBuilder,
    SchemaFieldNodeBuilder,
    FieldPathNodeBuilder,
    OpenAPIDefNodeBuilder,
    EdgeBuilder,
    # Batch operations
    BatchOperation,
    BatchOperationType,
    BatchResult,
    # Transaction
    ProjectionTransaction,
    TransactionState,
    # Driver abstraction
    Neo4jDriver,
    MemoryGraphStore,
    # Exceptions
    GraphProjectionError,
    ConstraintViolationError,
    TransactionError,
    # Factory
    create_graph_projection,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_store() -> MemoryGraphStore:
    """Create in-memory graph store for testing."""
    return MemoryGraphStore()


@pytest.fixture
def projection(memory_store: MemoryGraphStore) -> GraphProjection:
    """Create graph projection with memory store."""
    return GraphProjection(driver=memory_store)


@pytest.fixture
def sample_python_file() -> Path:
    """Sample Python file path."""
    return Path("/project/src/utils/helpers.py")


@pytest.fixture
def sample_symbol() -> Symbol:
    """Sample parsed symbol."""
    return Symbol(
        name="process_data",
        symbol_type=SymbolType.FUNCTION,
        line_start=10,
        line_end=25,
        language=Language.PYTHON,
        signature="def process_data(input: str) -> dict:",
        docstring="Process input data and return results.",
    )


@pytest.fixture
def sample_scip_id() -> SCIPSymbolID:
    """Sample SCIP symbol ID."""
    return SCIPSymbolID(
        scheme=SCIPScheme.SCIP_PYTHON,
        package="project.src.utils.helpers",
        descriptors=[
            SCIPDescriptor.namespace("helpers"),
            SCIPDescriptor.term("process_data"),
        ],
    )


# =============================================================================
# Test CodeNode Types
# =============================================================================


class TestCodeNodeType:
    """Tests for CodeNodeType enum."""

    def test_file_type(self):
        """Test CODE_FILE node type."""
        assert CodeNodeType.FILE.value == "CODE_FILE"
        assert CodeNodeType.FILE.label == "CODE_FILE"

    def test_schema_field_type(self):
        """Test CODE_SCHEMA_FIELD node type."""
        assert CodeNodeType.SCHEMA_FIELD.value == "CODE_SCHEMA_FIELD"
        assert CodeNodeType.SCHEMA_FIELD.label == "CODE_SCHEMA_FIELD"

    def test_field_path_type(self):
        """Test CODE_FIELD_PATH node type."""
        assert CodeNodeType.FIELD_PATH.value == "CODE_FIELD_PATH"
        assert CodeNodeType.FIELD_PATH.label == "CODE_FIELD_PATH"

    def test_openapi_def_type(self):
        """Test CODE_OPENAPI_DEF node type."""
        assert CodeNodeType.OPENAPI_DEF.value == "CODE_OPENAPI_DEF"
        assert CodeNodeType.OPENAPI_DEF.label == "CODE_OPENAPI_DEF"

    def test_symbol_type(self):
        """Test CODE_SYMBOL node type."""
        assert CodeNodeType.SYMBOL.value == "CODE_SYMBOL"
        assert CodeNodeType.SYMBOL.label == "CODE_SYMBOL"

    def test_package_type(self):
        """Test CODE_PACKAGE node type."""
        assert CodeNodeType.PACKAGE.value == "CODE_PACKAGE"
        assert CodeNodeType.PACKAGE.label == "CODE_PACKAGE"


class TestCodeNode:
    """Tests for CodeNode dataclass."""

    def test_create_file_node(self, sample_python_file: Path):
        """Test creating a file node."""
        node = CodeNode(
            node_type=CodeNodeType.FILE,
            id=str(sample_python_file),
            properties={
                "path": str(sample_python_file),
                "name": sample_python_file.name,
                "language": "python",
            },
        )

        assert node.node_type == CodeNodeType.FILE
        assert node.id == str(sample_python_file)
        assert node.properties["language"] == "python"

    def test_create_symbol_node(self, sample_scip_id: SCIPSymbolID):
        """Test creating a symbol node."""
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id=str(sample_scip_id),
            properties={
                "scip_id": str(sample_scip_id),
                "name": "process_data",
                "kind": "function",
                "signature": "def process_data(input: str) -> dict:",
                "docstring": "Process input data.",
                "line_start": 10,
                "line_end": 25,
            },
        )

        assert node.node_type == CodeNodeType.SYMBOL
        assert node.properties["kind"] == "function"
        assert node.properties["line_start"] == 10

    def test_create_package_node(self):
        """Test creating a package node."""
        node = CodeNode(
            node_type=CodeNodeType.PACKAGE,
            id="project.src.utils",
            properties={
                "name": "utils",
                "full_name": "project.src.utils",
                "language": "python",
            },
        )

        assert node.node_type == CodeNodeType.PACKAGE
        assert node.properties["full_name"] == "project.src.utils"

    def test_node_equality(self):
        """Test node equality based on id."""
        node1 = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )
        node2 = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )
        node3 = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="different_id",
            properties={"name": "test"},
        )

        assert node1 == node2
        assert node1 != node3


# =============================================================================
# Test CodeEdge Types
# =============================================================================


class TestCodeEdgeType:
    """Tests for CodeEdgeType enum."""

    def test_contains_edge(self):
        """Test CONTAINS edge type."""
        assert CodeEdgeType.CONTAINS.value == "CONTAINS"

    def test_defines_edge(self):
        """Test DEFINES edge type."""
        assert CodeEdgeType.DEFINES.value == "DEFINES"

    def test_imports_edge(self):
        """Test IMPORTS edge type."""
        assert CodeEdgeType.IMPORTS.value == "IMPORTS"

    def test_calls_edge(self):
        """Test CALLS edge type."""
        assert CodeEdgeType.CALLS.value == "CALLS"

    def test_reads_edge(self):
        """Test READS edge type (data-flow)."""
        assert CodeEdgeType.READS.value == "READS"

    def test_writes_edge(self):
        """Test WRITES edge type (data-flow)."""
        assert CodeEdgeType.WRITES.value == "WRITES"

    def test_data_flows_to_edge(self):
        """Test DATA_FLOWS_TO edge type."""
        assert CodeEdgeType.DATA_FLOWS_TO.value == "DATA_FLOWS_TO"

    def test_has_field_edge(self):
        """Test HAS_FIELD edge type."""
        assert CodeEdgeType.HAS_FIELD.value == "HAS_FIELD"

    def test_schema_exposes_edge(self):
        """Test SCHEMA_EXPOSES edge type."""
        assert CodeEdgeType.SCHEMA_EXPOSES.value == "SCHEMA_EXPOSES"

    def test_schema_aliases_edge(self):
        """Test SCHEMA_ALIASES edge type."""
        assert CodeEdgeType.SCHEMA_ALIASES.value == "SCHEMA_ALIASES"

    def test_path_reads_edge(self):
        """Test PATH_READS edge type."""
        assert CodeEdgeType.PATH_READS.value == "PATH_READS"


class TestCodeEdge:
    """Tests for CodeEdge dataclass."""

    def test_create_contains_edge(self):
        """Test creating CONTAINS edge (file -> symbol)."""
        edge = CodeEdge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id="/project/src/utils.py",
            target_id="scip-python project.src.utils utils/process_data.",
            properties={},
        )

        assert edge.edge_type == CodeEdgeType.CONTAINS
        assert edge.source_id == "/project/src/utils.py"

    def test_create_calls_edge_with_properties(self):
        """Test creating CALLS edge with properties."""
        edge = CodeEdge(
            edge_type=CodeEdgeType.CALLS,
            source_id="caller_scip_id",
            target_id="callee_scip_id",
            properties={
                "call_line": 15,
                "call_col": 4,
            },
        )

        assert edge.properties["call_line"] == 15

    def test_create_data_flow_edge(self):
        """Test creating data flow edge."""
        edge = CodeEdge(
            edge_type=CodeEdgeType.DATA_FLOWS_TO,
            source_id="source_var_id",
            target_id="sink_var_id",
            properties={
                "flow_type": "assignment",
            },
        )

        assert edge.edge_type == CodeEdgeType.DATA_FLOWS_TO


# =============================================================================
# Test Node Builders
# =============================================================================


class TestFileNodeBuilder:
    """Tests for FileNodeBuilder."""

    def test_build_file_node(self, sample_python_file: Path):
        """Test building a file node."""
        builder = FileNodeBuilder()
        node = (
            builder.path(sample_python_file)
            .language(Language.PYTHON)
            .size(1024)
            .content_hash("abc123")
            .build()
        )

        assert node.node_type == CodeNodeType.FILE
        assert node.id == str(sample_python_file)
        assert node.properties["path"] == str(sample_python_file)
        assert node.properties["language"] == "python"
        assert node.properties["size"] == 1024

    def test_file_node_requires_path(self):
        """Test that path is required."""
        builder = FileNodeBuilder()
        with pytest.raises(ValueError, match="path is required"):
            builder.build()


class TestSymbolNodeBuilder:
    """Tests for SymbolNodeBuilder."""

    def test_build_symbol_node(
        self, sample_symbol: Symbol, sample_scip_id: SCIPSymbolID
    ):
        """Test building a symbol node from parsed symbol."""
        builder = SymbolNodeBuilder()
        node = (
            builder.from_symbol(sample_symbol)
            .scip_id(sample_scip_id)
            .file_path(Path("/project/src/utils.py"))
            .build()
        )

        assert node.node_type == CodeNodeType.SYMBOL
        assert node.id == str(sample_scip_id)
        assert node.properties["name"] == "process_data"
        assert node.properties["kind"] == "function"
        assert node.properties["line_start"] == 10
        assert node.properties["line_end"] == 25
        assert node.properties["signature"] == "def process_data(input: str) -> dict:"
        assert node.properties["docstring"] == "Process input data and return results."

    def test_symbol_node_requires_scip_id(self, sample_symbol: Symbol):
        """Test that scip_id is required."""
        builder = SymbolNodeBuilder().from_symbol(sample_symbol)
        with pytest.raises(ValueError, match="scip_id is required"):
            builder.build()


class TestPackageNodeBuilder:
    """Tests for PackageNodeBuilder."""

    def test_build_package_node(self):
        """Test building a package node."""
        builder = PackageNodeBuilder()
        node = (
            builder.name("utils")
            .full_name("project.src.utils")
            .language(Language.PYTHON)
            .build()
        )

        assert node.node_type == CodeNodeType.PACKAGE
        assert node.id == "project.src.utils"
        assert node.properties["name"] == "utils"

    def test_package_node_requires_full_name(self):
        """Test that full_name is required."""
        builder = PackageNodeBuilder().name("utils")
        with pytest.raises(ValueError, match="full_name is required"):
            builder.build()


class TestSchemaFieldNodeBuilder:
    """Tests for SchemaFieldNodeBuilder."""

    def test_build_schema_field_node(self):
        """Test building a schema field node."""
        builder = SchemaFieldNodeBuilder()
        node = (
            builder.schema_id("schema::graphql:/path/to/file.ts:User:name")
            .name("name")
            .schema_type("graphql")
            .schema_name("User")
            .nullable(True)
            .field_type("string")
            .file_path(Path("/path/to/file.ts"))
            .line_start(10)
            .line_end(10)
            .build()
        )

        assert node.node_type == CodeNodeType.SCHEMA_FIELD
        assert node.id == "schema::graphql:/path/to/file.ts:User:name"
        assert node.properties["name"] == "name"
        assert node.properties["schema_type"] == "graphql"
        assert node.properties["schema_name"] == "User"
        assert node.properties["nullable"] is True
        assert node.properties["field_type"] == "string"
        assert node.properties["file_path"] == "/path/to/file.ts"

    def test_schema_field_requires_id(self):
        """Test that schema_id is required."""
        builder = SchemaFieldNodeBuilder().name("name").schema_type("graphql")
        with pytest.raises(ValueError, match="schema_id is required"):
            builder.build()


class TestFieldPathNodeBuilder:
    """Tests for FieldPathNodeBuilder."""

    def test_build_field_path_node(self):
        """Test building a field path node."""
        builder = FieldPathNodeBuilder()
        node = (
            builder.path_id("path::/path/to/file.ts:10:20")
            .path("user.address.city")
            .normalized_path("user.address.city")
            .segments(["user", "address", "city"])
            .leaf("city")
            .confidence("high")
            .file_path(Path("/path/to/file.ts"))
            .line_start(10)
            .line_end(10)
            .build()
        )

        assert node.node_type == CodeNodeType.FIELD_PATH
        assert node.id == "path::/path/to/file.ts:10:20"
        assert node.properties["path"] == "user.address.city"
        assert node.properties["normalized_path"] == "user.address.city"
        assert node.properties["leaf"] == "city"
        assert node.properties["confidence"] == "high"

    def test_field_path_requires_id(self):
        """Test that path_id is required."""
        builder = FieldPathNodeBuilder().path("user.address")
        with pytest.raises(ValueError, match="path_id is required"):
            builder.build()


class TestOpenAPIDefNodeBuilder:
    """Tests for OpenAPIDefNodeBuilder."""

    def test_build_openapi_def_node(self):
        """Test building an OpenAPI definition node."""
        builder = OpenAPIDefNodeBuilder()
        node = (
            builder.openapi_id("openapi::/path/to/openapi.json:User")
            .name("User")
            .title("Spec")
            .file_path(Path("/path/to/openapi.json"))
            .build()
        )

        assert node.node_type == CodeNodeType.OPENAPI_DEF
        assert node.id == "openapi::/path/to/openapi.json:User"
        assert node.properties["name"] == "User"
        assert node.properties["title"] == "Spec"
        assert node.properties["file_path"] == "/path/to/openapi.json"

    def test_openapi_def_requires_id(self):
        """Test that openapi_id is required."""
        builder = OpenAPIDefNodeBuilder().name("User")
        with pytest.raises(ValueError, match="openapi_id is required"):
            builder.build()


# =============================================================================
# Test Edge Builder
# =============================================================================


class TestEdgeBuilder:
    """Tests for EdgeBuilder."""

    def test_build_contains_edge(self):
        """Test building CONTAINS edge."""
        edge = (
            EdgeBuilder()
            .contains()
            .from_node("file_id")
            .to_node("symbol_id")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.CONTAINS
        assert edge.source_id == "file_id"
        assert edge.target_id == "symbol_id"

    def test_build_defines_edge(self):
        """Test building DEFINES edge."""
        edge = (
            EdgeBuilder()
            .defines()
            .from_node("package_id")
            .to_node("symbol_id")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.DEFINES

    def test_build_calls_edge_with_location(self):
        """Test building CALLS edge with call location."""
        edge = (
            EdgeBuilder()
            .calls()
            .from_node("caller_id")
            .to_node("callee_id")
            .at_line(42)
            .at_column(8)
            .build()
        )

        assert edge.edge_type == CodeEdgeType.CALLS
        assert edge.properties["call_line"] == 42
        assert edge.properties["call_col"] == 8

    def test_build_imports_edge(self):
        """Test building IMPORTS edge."""
        edge = (
            EdgeBuilder()
            .imports()
            .from_node("importing_file")
            .to_node("imported_module")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.IMPORTS

    def test_build_data_flow_edge(self):
        """Test building data flow edges."""
        edge = (
            EdgeBuilder()
            .data_flows_to()
            .from_node("source")
            .to_node("sink")
            .with_property("flow_type", "return_value")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.DATA_FLOWS_TO
        assert edge.properties["flow_type"] == "return_value"

    def test_build_has_field_edge(self):
        """Test building HAS_FIELD edge."""
        edge = (
            EdgeBuilder()
            .has_field()
            .from_node("type_id")
            .to_node("field_id")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.HAS_FIELD

    def test_build_schema_exposes_edge(self):
        """Test building SCHEMA_EXPOSES edge."""
        edge = (
            EdgeBuilder()
            .schema_exposes()
            .from_node("field_id")
            .to_node("schema_field_id")
            .build()
        )

        assert edge.edge_type == CodeEdgeType.SCHEMA_EXPOSES

    def test_edge_requires_type(self):
        """Test that edge type is required."""
        builder = EdgeBuilder().from_node("a").to_node("b")
        with pytest.raises(ValueError, match="edge_type is required"):
            builder.build()

    def test_edge_requires_source(self):
        """Test that source is required."""
        builder = EdgeBuilder().calls().to_node("b")
        with pytest.raises(ValueError, match="source_id is required"):
            builder.build()


# =============================================================================
# Test Memory Graph Store (Mock Neo4j Driver)
# =============================================================================


class TestMemoryGraphStore:
    """Tests for in-memory graph store."""

    def test_add_node(self, memory_store: MemoryGraphStore):
        """Test adding a node."""
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )

        memory_store.add_node(node)

        assert memory_store.get_node("test_id") == node
        assert memory_store.node_count == 1

    def test_add_duplicate_node_raises(self, memory_store: MemoryGraphStore):
        """Test adding duplicate node raises error."""
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )

        memory_store.add_node(node)

        with pytest.raises(ConstraintViolationError, match="already exists"):
            memory_store.add_node(node)

    def test_update_node(self, memory_store: MemoryGraphStore):
        """Test updating a node."""
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )
        memory_store.add_node(node)

        updated = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "updated"},
        )
        memory_store.update_node(updated)

        result = memory_store.get_node("test_id")
        assert result.properties["name"] == "updated"

    def test_delete_node(self, memory_store: MemoryGraphStore):
        """Test deleting a node."""
        node = CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id="test_id",
            properties={"name": "test"},
        )
        memory_store.add_node(node)

        memory_store.delete_node("test_id")

        assert memory_store.get_node("test_id") is None
        assert memory_store.node_count == 0

    def test_delete_node_removes_edges(self, memory_store: MemoryGraphStore):
        """Test that deleting node removes related edges."""
        node1 = CodeNode(CodeNodeType.FILE, "file_id", {})
        node2 = CodeNode(CodeNodeType.SYMBOL, "symbol_id", {})
        edge = CodeEdge(CodeEdgeType.CONTAINS, "file_id", "symbol_id", {})

        memory_store.add_node(node1)
        memory_store.add_node(node2)
        memory_store.add_edge(edge)

        memory_store.delete_node("file_id")

        assert memory_store.edge_count == 0

    def test_add_edge(self, memory_store: MemoryGraphStore):
        """Test adding an edge."""
        node1 = CodeNode(CodeNodeType.FILE, "file_id", {})
        node2 = CodeNode(CodeNodeType.SYMBOL, "symbol_id", {})
        edge = CodeEdge(CodeEdgeType.CONTAINS, "file_id", "symbol_id", {})

        memory_store.add_node(node1)
        memory_store.add_node(node2)
        memory_store.add_edge(edge)

        assert memory_store.edge_count == 1
        assert memory_store.has_edge("file_id", "symbol_id", CodeEdgeType.CONTAINS)

    def test_get_outgoing_edges(self, memory_store: MemoryGraphStore):
        """Test getting outgoing edges from a node."""
        node1 = CodeNode(CodeNodeType.FILE, "file_id", {})
        node2 = CodeNode(CodeNodeType.SYMBOL, "symbol1_id", {})
        node3 = CodeNode(CodeNodeType.SYMBOL, "symbol2_id", {})

        memory_store.add_node(node1)
        memory_store.add_node(node2)
        memory_store.add_node(node3)
        memory_store.add_edge(CodeEdge(CodeEdgeType.CONTAINS, "file_id", "symbol1_id", {}))
        memory_store.add_edge(CodeEdge(CodeEdgeType.CONTAINS, "file_id", "symbol2_id", {}))

        edges = memory_store.get_outgoing_edges("file_id")

        assert len(edges) == 2

    def test_query_nodes_by_type(self, memory_store: MemoryGraphStore):
        """Test querying nodes by type."""
        memory_store.add_node(CodeNode(CodeNodeType.FILE, "file1", {}))
        memory_store.add_node(CodeNode(CodeNodeType.FILE, "file2", {}))
        memory_store.add_node(CodeNode(CodeNodeType.SYMBOL, "symbol1", {}))

        files = memory_store.query_nodes_by_type(CodeNodeType.FILE)

        assert len(files) == 2


# =============================================================================
# Test Graph Projection Service
# =============================================================================


class TestGraphProjection:
    """Tests for GraphProjection service."""

    def test_create_file_node(self, projection: GraphProjection, sample_python_file: Path):
        """Test creating a file node."""
        node = projection.create_file_node(
            path=sample_python_file,
            language=Language.PYTHON,
            size=1024,
            content_hash="abc123",
        )

        assert node.node_type == CodeNodeType.FILE
        assert projection.driver.node_count == 1

    def test_create_symbol_node(
        self,
        projection: GraphProjection,
        sample_symbol: Symbol,
        sample_scip_id: SCIPSymbolID,
        sample_python_file: Path,
    ):
        """Test creating a symbol node."""
        node = projection.create_symbol_node(
            symbol=sample_symbol,
            scip_id=sample_scip_id,
            file_path=sample_python_file,
        )

        assert node.node_type == CodeNodeType.SYMBOL
        assert node.properties["scip_id"] == str(sample_scip_id)

    def test_create_package_node(self, projection: GraphProjection):
        """Test creating a package node."""
        node = projection.create_package_node(
            name="utils",
            full_name="project.src.utils",
            language=Language.PYTHON,
        )

        assert node.node_type == CodeNodeType.PACKAGE

    def test_create_contains_edge(
        self,
        projection: GraphProjection,
        sample_python_file: Path,
        sample_symbol: Symbol,
        sample_scip_id: SCIPSymbolID,
    ):
        """Test creating CONTAINS edge between file and symbol."""
        file_node = projection.create_file_node(
            path=sample_python_file,
            language=Language.PYTHON,
        )
        symbol_node = projection.create_symbol_node(
            symbol=sample_symbol,
            scip_id=sample_scip_id,
            file_path=sample_python_file,
        )

        edge = projection.create_edge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=file_node.id,
            target_id=symbol_node.id,
        )

        assert edge.edge_type == CodeEdgeType.CONTAINS
        assert projection.driver.edge_count == 1

    def test_create_imports_edge(self, projection: GraphProjection):
        """Test creating IMPORTS edge."""
        file1 = projection.create_file_node(Path("/a.py"), Language.PYTHON)
        file2 = projection.create_file_node(Path("/b.py"), Language.PYTHON)

        edge = projection.create_edge(
            edge_type=CodeEdgeType.IMPORTS,
            source_id=file1.id,
            target_id=file2.id,
        )

        assert edge.edge_type == CodeEdgeType.IMPORTS

    def test_create_calls_edge(
        self,
        projection: GraphProjection,
        sample_symbol: Symbol,
    ):
        """Test creating CALLS edge between symbols."""
        caller_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON,
            "pkg",
            [SCIPDescriptor.term("caller")],
        )
        callee_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON,
            "pkg",
            [SCIPDescriptor.term("callee")],
        )

        caller_symbol = Symbol(
            name="caller",
            symbol_type=SymbolType.FUNCTION,
            line_start=1,
            line_end=5,
            language=Language.PYTHON,
        )
        callee_symbol = Symbol(
            name="callee",
            symbol_type=SymbolType.FUNCTION,
            line_start=10,
            line_end=15,
            language=Language.PYTHON,
        )

        caller_node = projection.create_symbol_node(
            caller_symbol, caller_id, Path("/test.py")
        )
        callee_node = projection.create_symbol_node(
            callee_symbol, callee_id, Path("/test.py")
        )

        edge = projection.create_edge(
            edge_type=CodeEdgeType.CALLS,
            source_id=caller_node.id,
            target_id=callee_node.id,
            properties={"call_line": 3},
        )

        assert edge.edge_type == CodeEdgeType.CALLS
        assert edge.properties["call_line"] == 3

    def test_update_symbol_node(
        self,
        projection: GraphProjection,
        sample_symbol: Symbol,
        sample_scip_id: SCIPSymbolID,
        sample_python_file: Path,
    ):
        """Test updating an existing symbol node."""
        # Create initial node
        node = projection.create_symbol_node(
            symbol=sample_symbol,
            scip_id=sample_scip_id,
            file_path=sample_python_file,
        )

        # Update with new properties
        updated_symbol = Symbol(
            name="process_data",
            symbol_type=SymbolType.FUNCTION,
            line_start=10,
            line_end=30,  # Changed
            language=Language.PYTHON,
            signature="def process_data(input: str) -> dict:",
            docstring="Updated docstring.",  # Changed
        )

        updated = projection.update_symbol_node(
            scip_id=sample_scip_id,
            symbol=updated_symbol,
        )

        assert updated.properties["line_end"] == 30
        assert updated.properties["docstring"] == "Updated docstring."

    def test_delete_symbol_node(
        self,
        projection: GraphProjection,
        sample_symbol: Symbol,
        sample_scip_id: SCIPSymbolID,
        sample_python_file: Path,
    ):
        """Test deleting a symbol node."""
        projection.create_symbol_node(
            symbol=sample_symbol,
            scip_id=sample_scip_id,
            file_path=sample_python_file,
        )

        projection.delete_node(str(sample_scip_id))

        assert projection.driver.node_count == 0

    def test_delete_file_cascades_to_symbols(
        self,
        projection: GraphProjection,
        sample_python_file: Path,
        sample_symbol: Symbol,
        sample_scip_id: SCIPSymbolID,
    ):
        """Test deleting file removes contained symbols."""
        file_node = projection.create_file_node(
            path=sample_python_file,
            language=Language.PYTHON,
        )
        symbol_node = projection.create_symbol_node(
            symbol=sample_symbol,
            scip_id=sample_scip_id,
            file_path=sample_python_file,
        )
        projection.create_edge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=file_node.id,
            target_id=symbol_node.id,
        )

        # Delete file with cascade
        projection.delete_file_with_contents(sample_python_file)

        assert projection.driver.node_count == 0
        assert projection.driver.edge_count == 0


# =============================================================================
# Test Batch Operations
# =============================================================================


class TestBatchOperations:
    """Tests for batch operations."""

    def test_batch_add_nodes(self, projection: GraphProjection):
        """Test batch adding multiple nodes."""
        nodes = [
            CodeNode(CodeNodeType.FILE, f"/file{i}.py", {"name": f"file{i}"})
            for i in range(10)
        ]

        batch = BatchOperation(
            operation_type=BatchOperationType.ADD_NODES,
            items=nodes,
        )

        result = projection.execute_batch(batch)

        assert result.success
        assert result.processed == 10
        assert projection.driver.node_count == 10

    def test_batch_add_edges(self, projection: GraphProjection):
        """Test batch adding multiple edges."""
        # First add nodes
        for i in range(5):
            projection.create_file_node(Path(f"/file{i}.py"), Language.PYTHON)

        edges = [
            CodeEdge(
                CodeEdgeType.IMPORTS,
                f"/file{i}.py",
                f"/file{(i+1)%5}.py",
                {},
            )
            for i in range(5)
        ]

        batch = BatchOperation(
            operation_type=BatchOperationType.ADD_EDGES,
            items=edges,
        )

        result = projection.execute_batch(batch)

        assert result.success
        assert result.processed == 5

    def test_batch_delete_nodes(self, projection: GraphProjection):
        """Test batch deleting nodes."""
        # Create nodes first
        for i in range(5):
            projection.create_file_node(Path(f"/file{i}.py"), Language.PYTHON)

        node_ids = [f"/file{i}.py" for i in range(3)]

        batch = BatchOperation(
            operation_type=BatchOperationType.DELETE_NODES,
            items=node_ids,
        )

        result = projection.execute_batch(batch)

        assert result.success
        assert result.processed == 3
        assert projection.driver.node_count == 2

    def test_batch_with_failures_reports_errors(self, projection: GraphProjection):
        """Test batch operation reports partial failures."""
        # Add one node
        projection.create_file_node(Path("/existing.py"), Language.PYTHON)

        # Try to add duplicate
        nodes = [
            CodeNode(CodeNodeType.FILE, "/existing.py", {}),
            CodeNode(CodeNodeType.FILE, "/new.py", {}),
        ]

        batch = BatchOperation(
            operation_type=BatchOperationType.ADD_NODES,
            items=nodes,
        )

        result = projection.execute_batch(batch, continue_on_error=True)

        assert result.processed == 1
        assert result.failed == 1
        assert len(result.errors) == 1


# =============================================================================
# Test Transactions
# =============================================================================


class TestProjectionTransaction:
    """Tests for projection transactions."""

    def test_transaction_commit(self, projection: GraphProjection):
        """Test successful transaction commit."""
        with projection.transaction() as tx:
            tx.add_node(CodeNode(CodeNodeType.FILE, "/test.py", {}))
            tx.add_node(CodeNode(CodeNodeType.SYMBOL, "sym1", {}))
            tx.add_edge(CodeEdge(CodeEdgeType.CONTAINS, "/test.py", "sym1", {}))

        assert projection.driver.node_count == 2
        assert projection.driver.edge_count == 1

    def test_transaction_rollback_on_error(self, projection: GraphProjection):
        """Test transaction rollback on error."""
        try:
            with projection.transaction() as tx:
                tx.add_node(CodeNode(CodeNodeType.FILE, "/test.py", {}))
                raise ValueError("Simulated error")
        except ValueError:
            pass

        assert projection.driver.node_count == 0

    def test_transaction_explicit_rollback(self, projection: GraphProjection):
        """Test explicit transaction rollback."""
        tx = projection.begin_transaction()
        tx.add_node(CodeNode(CodeNodeType.FILE, "/test.py", {}))
        tx.rollback()

        assert projection.driver.node_count == 0
        assert tx.state == TransactionState.ROLLED_BACK

    def test_nested_transactions_not_allowed(self, projection: GraphProjection):
        """Test that nested transactions are not allowed."""
        with projection.transaction() as tx1:
            with pytest.raises(TransactionError, match="already in transaction"):
                with projection.transaction() as tx2:
                    pass


# =============================================================================
# Test Constraint Creation
# =============================================================================


class TestConstraints:
    """Tests for Neo4j constraint creation."""

    def test_create_scip_id_unique_constraint(self, memory_store: MemoryGraphStore):
        """Test creating unique constraint on scip_id."""
        memory_store.create_constraint(
            name="code_symbol_scip_id_unique",
            node_type=CodeNodeType.SYMBOL,
            property_name="scip_id",
            constraint_type="UNIQUE",
        )

        assert memory_store.has_constraint("code_symbol_scip_id_unique")

    def test_constraint_prevents_duplicates(self, memory_store: MemoryGraphStore):
        """Test that constraint prevents duplicate scip_ids."""
        memory_store.create_constraint(
            name="code_symbol_scip_id_unique",
            node_type=CodeNodeType.SYMBOL,
            property_name="scip_id",
            constraint_type="UNIQUE",
        )

        node1 = CodeNode(
            CodeNodeType.SYMBOL,
            "id1",
            {"scip_id": "same_scip_id", "name": "sym1"},
        )
        node2 = CodeNode(
            CodeNodeType.SYMBOL,
            "id2",
            {"scip_id": "same_scip_id", "name": "sym2"},
        )

        memory_store.add_node(node1)

        with pytest.raises(ConstraintViolationError):
            memory_store.add_node(node2)


# =============================================================================
# Test Incremental Updates
# =============================================================================


class TestIncrementalUpdates:
    """Tests for incremental graph updates."""

    def test_apply_added_change(self, projection: GraphProjection):
        """Test applying an ADDED change."""
        from openmemory.api.indexing.merkle_tree import Change, ChangeType

        change = Change(
            change_type=ChangeType.ADDED,
            path=Path("src/new_file.py"),
            new_hash=None,
        )

        # Simulate what the indexer would do
        projection.create_file_node(
            path=change.path,
            language=Language.PYTHON,
        )

        assert projection.driver.node_count == 1

    def test_apply_modified_change(
        self,
        projection: GraphProjection,
        sample_python_file: Path,
    ):
        """Test applying a MODIFIED change updates node."""
        # Create initial file
        projection.create_file_node(
            path=sample_python_file,
            language=Language.PYTHON,
            content_hash="hash1",
        )

        # Modify (simulating update)
        projection.update_file_node(
            path=sample_python_file,
            content_hash="hash2",
        )

        node = projection.driver.get_node(str(sample_python_file))
        assert node.properties["content_hash"] == "hash2"

    def test_apply_deleted_change(
        self,
        projection: GraphProjection,
        sample_python_file: Path,
    ):
        """Test applying a DELETED change removes node."""
        projection.create_file_node(
            path=sample_python_file,
            language=Language.PYTHON,
        )

        projection.delete_file_with_contents(sample_python_file)

        assert projection.driver.node_count == 0


# =============================================================================
# Test Cross-File References
# =============================================================================


class TestCrossFileReferences:
    """Tests for cross-file symbol references."""

    def test_create_cross_file_call(self, projection: GraphProjection):
        """Test creating a call edge between symbols in different files."""
        # File A with function foo
        file_a = projection.create_file_node(Path("/a.py"), Language.PYTHON)
        foo_id = SCIPSymbolID(SCIPScheme.SCIP_PYTHON, "a", [SCIPDescriptor.term("foo")])
        foo = projection.create_symbol_node(
            Symbol("foo", SymbolType.FUNCTION, 1, 5, Language.PYTHON),
            foo_id,
            Path("/a.py"),
        )

        # File B with function bar that calls foo
        file_b = projection.create_file_node(Path("/b.py"), Language.PYTHON)
        bar_id = SCIPSymbolID(SCIPScheme.SCIP_PYTHON, "b", [SCIPDescriptor.term("bar")])
        bar = projection.create_symbol_node(
            Symbol("bar", SymbolType.FUNCTION, 1, 5, Language.PYTHON),
            bar_id,
            Path("/b.py"),
        )

        # bar calls foo
        edge = projection.create_edge(
            edge_type=CodeEdgeType.CALLS,
            source_id=bar.id,
            target_id=foo.id,
            properties={"call_line": 3},
        )

        assert projection.driver.has_edge(bar.id, foo.id, CodeEdgeType.CALLS)


# =============================================================================
# Test Factory Function
# =============================================================================


class TestFactory:
    """Tests for factory function."""

    def test_create_with_memory_store(self):
        """Test creating projection with memory store."""
        projection = create_graph_projection(driver_type="memory")

        assert isinstance(projection.driver, MemoryGraphStore)

    def test_create_with_config(self):
        """Test creating projection with config."""
        config = GraphProjectionConfig(
            create_constraints=True,
            batch_size=500,
        )

        projection = create_graph_projection(
            driver_type="memory",
            config=config,
        )

        assert projection.config.batch_size == 500


# =============================================================================
# Test Data Flow Edges
# =============================================================================


class TestDataFlowEdges:
    """Tests for data flow edge types."""

    def test_reads_edge(self, projection: GraphProjection):
        """Test creating READS edge (function reads variable)."""
        var_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("config")]
        )
        func_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("process")]
        )

        var_node = projection.create_symbol_node(
            Symbol("config", SymbolType.VARIABLE, 1, 1, Language.PYTHON),
            var_id,
            Path("/test.py"),
        )
        func_node = projection.create_symbol_node(
            Symbol("process", SymbolType.FUNCTION, 5, 10, Language.PYTHON),
            func_id,
            Path("/test.py"),
        )

        edge = projection.create_edge(
            CodeEdgeType.READS,
            func_node.id,
            var_node.id,
        )

        assert edge.edge_type == CodeEdgeType.READS

    def test_writes_edge(self, projection: GraphProjection):
        """Test creating WRITES edge (function writes variable)."""
        var_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("result")]
        )
        func_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("compute")]
        )

        var_node = projection.create_symbol_node(
            Symbol("result", SymbolType.VARIABLE, 1, 1, Language.PYTHON),
            var_id,
            Path("/test.py"),
        )
        func_node = projection.create_symbol_node(
            Symbol("compute", SymbolType.FUNCTION, 5, 10, Language.PYTHON),
            func_id,
            Path("/test.py"),
        )

        edge = projection.create_edge(
            CodeEdgeType.WRITES,
            func_node.id,
            var_node.id,
        )

        assert edge.edge_type == CodeEdgeType.WRITES

    def test_data_flows_to_edge(self, projection: GraphProjection):
        """Test creating DATA_FLOWS_TO edge between variables."""
        source_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("input")]
        )
        sink_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON, "pkg", [SCIPDescriptor.term("output")]
        )

        source = projection.create_symbol_node(
            Symbol("input", SymbolType.VARIABLE, 1, 1, Language.PYTHON),
            source_id,
            Path("/test.py"),
        )
        sink = projection.create_symbol_node(
            Symbol("output", SymbolType.VARIABLE, 5, 5, Language.PYTHON),
            sink_id,
            Path("/test.py"),
        )

        edge = projection.create_edge(
            CodeEdgeType.DATA_FLOWS_TO,
            source.id,
            sink.id,
            properties={"flow_type": "assignment"},
        )

        assert edge.edge_type == CodeEdgeType.DATA_FLOWS_TO
        assert edge.properties["flow_type"] == "assignment"
