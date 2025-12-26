"""Pytest fixtures for visualization tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmemory.api.indexing.ast_parser import Language
from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
    MemoryGraphStore,
)


@pytest.fixture
def memory_graph_store() -> MemoryGraphStore:
    """Create an in-memory graph store for testing."""
    return MemoryGraphStore()


@pytest.fixture
def sample_file_node() -> CodeNode:
    """Create a sample file node."""
    return CodeNode(
        node_type=CodeNodeType.FILE,
        id="/src/main.py",
        properties={
            "path": "/src/main.py",
            "name": "main.py",
            "language": "python",
            "size": 1024,
            "content_hash": "abc123",
        },
    )


@pytest.fixture
def sample_symbol_node() -> CodeNode:
    """Create a sample symbol node."""
    return CodeNode(
        node_type=CodeNodeType.SYMBOL,
        id="scip-python pkg module/process_data.",
        properties={
            "scip_id": "scip-python pkg module/process_data.",
            "name": "process_data",
            "kind": "function",
            "line_start": 10,
            "line_end": 25,
            "language": "python",
            "signature": "def process_data(data: list) -> dict",
            "file_path": "/src/main.py",
        },
    )


@pytest.fixture
def sample_class_node() -> CodeNode:
    """Create a sample class node."""
    return CodeNode(
        node_type=CodeNodeType.SYMBOL,
        id="scip-python pkg module/DataProcessor#",
        properties={
            "scip_id": "scip-python pkg module/DataProcessor#",
            "name": "DataProcessor",
            "kind": "class",
            "line_start": 30,
            "line_end": 80,
            "language": "python",
            "file_path": "/src/main.py",
        },
    )


@pytest.fixture
def sample_method_node() -> CodeNode:
    """Create a sample method node."""
    return CodeNode(
        node_type=CodeNodeType.SYMBOL,
        id="scip-python pkg module/DataProcessor#transform().",
        properties={
            "scip_id": "scip-python pkg module/DataProcessor#transform().",
            "name": "transform",
            "kind": "method",
            "line_start": 45,
            "line_end": 55,
            "language": "python",
            "signature": "def transform(self, data: dict) -> dict",
            "parent_name": "DataProcessor",
            "file_path": "/src/main.py",
        },
    )


@pytest.fixture
def sample_package_node() -> CodeNode:
    """Create a sample package node."""
    return CodeNode(
        node_type=CodeNodeType.PACKAGE,
        id="pkg.module",
        properties={
            "full_name": "pkg.module",
            "name": "module",
            "language": "python",
        },
    )


@pytest.fixture
def sample_contains_edge(sample_file_node: CodeNode, sample_symbol_node: CodeNode) -> CodeEdge:
    """Create a sample CONTAINS edge."""
    return CodeEdge(
        edge_type=CodeEdgeType.CONTAINS,
        source_id=sample_file_node.id,
        target_id=sample_symbol_node.id,
        properties={},
    )


@pytest.fixture
def sample_calls_edge(sample_symbol_node: CodeNode, sample_method_node: CodeNode) -> CodeEdge:
    """Create a sample CALLS edge."""
    return CodeEdge(
        edge_type=CodeEdgeType.CALLS,
        source_id=sample_symbol_node.id,
        target_id=sample_method_node.id,
        properties={
            "call_line": 15,
            "call_col": 8,
        },
    )


@pytest.fixture
def populated_graph_store(
    memory_graph_store: MemoryGraphStore,
    sample_file_node: CodeNode,
    sample_symbol_node: CodeNode,
    sample_class_node: CodeNode,
    sample_method_node: CodeNode,
    sample_package_node: CodeNode,
    sample_contains_edge: CodeEdge,
    sample_calls_edge: CodeEdge,
) -> MemoryGraphStore:
    """Create a populated graph store with sample nodes and edges."""
    # Add nodes
    memory_graph_store.add_node(sample_file_node)
    memory_graph_store.add_node(sample_symbol_node)
    memory_graph_store.add_node(sample_class_node)
    memory_graph_store.add_node(sample_method_node)
    memory_graph_store.add_node(sample_package_node)

    # Add edges
    memory_graph_store.add_edge(sample_contains_edge)
    memory_graph_store.add_edge(sample_calls_edge)

    # Add more edges for a richer graph
    memory_graph_store.add_edge(
        CodeEdge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=sample_file_node.id,
            target_id=sample_class_node.id,
            properties={},
        )
    )
    memory_graph_store.add_edge(
        CodeEdge(
            edge_type=CodeEdgeType.CONTAINS,
            source_id=sample_class_node.id,
            target_id=sample_method_node.id,
            properties={},
        )
    )
    memory_graph_store.add_edge(
        CodeEdge(
            edge_type=CodeEdgeType.DEFINES,
            source_id=sample_package_node.id,
            target_id=sample_class_node.id,
            properties={},
        )
    )

    return memory_graph_store


@pytest.fixture
def large_graph_store() -> MemoryGraphStore:
    """Create a large graph store for pagination testing."""
    store = MemoryGraphStore()

    # Create 100 file nodes
    for i in range(100):
        store.add_node(
            CodeNode(
                node_type=CodeNodeType.FILE,
                id=f"/src/file_{i:03d}.py",
                properties={
                    "path": f"/src/file_{i:03d}.py",
                    "name": f"file_{i:03d}.py",
                    "language": "python",
                    "size": 1000 + i,
                },
            )
        )

    # Create 500 symbol nodes (5 per file)
    for i in range(100):
        for j in range(5):
            symbol_id = f"scip-python pkg file_{i:03d}/func_{j}."
            store.add_node(
                CodeNode(
                    node_type=CodeNodeType.SYMBOL,
                    id=symbol_id,
                    properties={
                        "scip_id": symbol_id,
                        "name": f"func_{j}",
                        "kind": "function",
                        "line_start": j * 20 + 1,
                        "line_end": j * 20 + 15,
                        "language": "python",
                        "file_path": f"/src/file_{i:03d}.py",
                    },
                )
            )
            # Add CONTAINS edge
            store.add_edge(
                CodeEdge(
                    edge_type=CodeEdgeType.CONTAINS,
                    source_id=f"/src/file_{i:03d}.py",
                    target_id=symbol_id,
                    properties={},
                )
            )

    # Add some CALLS edges
    for i in range(50):
        source_file = i
        target_file = (i + 1) % 100
        store.add_edge(
            CodeEdge(
                edge_type=CodeEdgeType.CALLS,
                source_id=f"scip-python pkg file_{source_file:03d}/func_0.",
                target_id=f"scip-python pkg file_{target_file:03d}/func_0.",
                properties={"call_line": 5},
            )
        )

    return store
