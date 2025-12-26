"""Tests for call graph tools (find_callers, find_callees).

This module tests the find_callers and find_callees MCP tools with TDD approach:
- CallGraphConfig: Configuration and defaults
- CallGraphInput: Input validation
- GraphOutput: Result structure with nodes and edges
- FindCallersTool: Find functions that call a symbol
- FindCalleesTool: Find functions called by a symbol
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from openmemory.api.indexing.graph_projection import CodeEdgeType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()

    # Setup default node data
    symbol_node = MagicMock()
    symbol_node.properties = {
        "scip_id": "scip-python myapp module/MyClass#my_method.",
        "name": "my_method",
        "kind": "method",
        "signature": "def my_method(self, arg: str) -> int:",
        "file_path": "/path/to/module.py",
        "line_start": 10,
        "line_end": 20,
        "language": "python",
    }

    driver.get_node.return_value = symbol_node
    driver.get_outgoing_edges.return_value = []
    driver.get_incoming_edges.return_value = []

    return driver


@pytest.fixture
def sample_scip_id() -> str:
    """Return a sample SCIP symbol ID."""
    return "scip-python myapp module/MyClass#my_method."


@pytest.fixture
def mock_graph_driver_with_callers(mock_graph_driver, sample_scip_id):
    """Create a mock graph driver with callers."""
    # Setup caller edges
    caller_edge = MagicMock()
    caller_edge.source_id = "scip-python myapp module/caller_func."
    caller_edge.target_id = sample_scip_id
    caller_edge.edge_type = CodeEdgeType.CALLS
    caller_edge.properties = {"call_line": 25, "call_col": 8}

    mock_graph_driver.get_incoming_edges.return_value = [caller_edge]

    # Setup caller node
    caller_node = MagicMock()
    caller_node.properties = {
        "scip_id": "scip-python myapp module/caller_func.",
        "name": "caller_func",
        "kind": "function",
        "file_path": "/path/to/caller.py",
        "line_start": 20,
        "line_end": 30,
    }

    def get_node_side_effect(node_id):
        if node_id == sample_scip_id:
            return mock_graph_driver.get_node.return_value
        elif node_id == "scip-python myapp module/caller_func.":
            return caller_node
        return None

    mock_graph_driver.get_node.side_effect = get_node_side_effect

    return mock_graph_driver


@pytest.fixture
def mock_graph_driver_with_callees(mock_graph_driver, sample_scip_id):
    """Create a mock graph driver with callees."""
    # Setup callee edges
    callee_edge = MagicMock()
    callee_edge.source_id = sample_scip_id
    callee_edge.target_id = "scip-python myapp module/helper_func."
    callee_edge.edge_type = CodeEdgeType.CALLS
    callee_edge.properties = {"call_line": 15, "call_col": 4}

    mock_graph_driver.get_outgoing_edges.return_value = [callee_edge]

    # Setup callee node
    callee_node = MagicMock()
    callee_node.properties = {
        "scip_id": "scip-python myapp module/helper_func.",
        "name": "helper_func",
        "kind": "function",
        "file_path": "/path/to/helpers.py",
        "line_start": 5,
        "line_end": 15,
    }

    def get_node_side_effect(node_id):
        if node_id == sample_scip_id:
            return mock_graph_driver.get_node.return_value
        elif node_id == "scip-python myapp module/helper_func.":
            return callee_node
        return None

    mock_graph_driver.get_node.side_effect = get_node_side_effect

    return mock_graph_driver


# =============================================================================
# CallGraphConfig Tests
# =============================================================================


class TestCallGraphConfig:
    """Tests for CallGraphConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.call_graph import CallGraphConfig

        config = CallGraphConfig()

        assert config.depth == 1
        assert config.max_nodes == 100
        assert config.include_properties is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.call_graph import CallGraphConfig

        config = CallGraphConfig(
            depth=3,
            max_nodes=50,
            include_properties=False,
        )

        assert config.depth == 3
        assert config.max_nodes == 50
        assert config.include_properties is False


# =============================================================================
# CallGraphInput Tests
# =============================================================================


class TestCallGraphInput:
    """Tests for CallGraphInput dataclass."""

    def test_symbol_id_input(self):
        """Test input with symbol_id."""
        from openmemory.api.tools.call_graph import CallGraphInput

        input_data = CallGraphInput(
            symbol_id="scip-python myapp module/func.",
            repo_id="myrepo",
        )

        assert input_data.symbol_id == "scip-python myapp module/func."
        assert input_data.repo_id == "myrepo"

    def test_symbol_name_input(self):
        """Test input with symbol_name."""
        from openmemory.api.tools.call_graph import CallGraphInput

        input_data = CallGraphInput(
            symbol_name="my_function",
            repo_id="myrepo",
        )

        assert input_data.symbol_name == "my_function"

    def test_depth_validation(self):
        """Test depth parameter."""
        from openmemory.api.tools.call_graph import CallGraphInput

        input_data = CallGraphInput(
            symbol_id="scip-python myapp module/func.",
            repo_id="myrepo",
            depth=3,
        )

        assert input_data.depth == 3


# =============================================================================
# GraphOutput Tests
# =============================================================================


class TestGraphOutput:
    """Tests for GraphOutput dataclass."""

    def test_graph_output_structure(self):
        """Test GraphOutput has all required fields."""
        from openmemory.api.tools.call_graph import (
            GraphOutput,
            GraphNode,
            GraphEdge,
            ResponseMeta,
        )

        node = GraphNode(
            id="scip-python myapp module/func.",
            type="CODE_SYMBOL",
            properties={"name": "func"},
        )

        edge = GraphEdge(
            from_id="scip-python myapp module/caller.",
            to_id="scip-python myapp module/func.",
            type="CALLS",
        )

        meta = ResponseMeta(request_id="req-123")

        output = GraphOutput(
            nodes=[node],
            edges=[edge],
            meta=meta,
        )

        assert len(output.nodes) == 1
        assert len(output.edges) == 1
        assert output.meta.request_id == "req-123"


# =============================================================================
# FindCallersTool Tests
# =============================================================================


class TestFindCallersTool:
    """Tests for FindCallersTool."""

    def test_find_callers_basic(self, mock_graph_driver_with_callers, sample_scip_id):
        """Test basic find callers functionality."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver_with_callers)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert result is not None
        assert len(result.nodes) >= 1
        assert result.meta is not None

    def test_find_callers_with_depth(self, mock_graph_driver_with_callers, sample_scip_id):
        """Test find callers with specified depth."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
            CallGraphConfig,
        )

        config = CallGraphConfig(depth=2)
        tool = FindCallersTool(
            graph_driver=mock_graph_driver_with_callers,
            config=config,
        )

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo", depth=2)
        )

        assert result is not None

    def test_find_callers_no_callers(self, mock_graph_driver, sample_scip_id):
        """Test find callers when symbol has no callers."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert result is not None
        # Should return empty graph but not error
        assert len(result.edges) == 0

    def test_find_callers_returns_edges(self, mock_graph_driver_with_callers, sample_scip_id):
        """Test find callers returns proper edges."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver_with_callers)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert len(result.edges) >= 1
        edge = result.edges[0]
        assert edge.type == "CALLS"
        assert edge.to_id == sample_scip_id

    def test_find_callers_symbol_not_found(self, mock_graph_driver):
        """Test find callers when symbol doesn't exist."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
            SymbolNotFoundError,
        )

        mock_graph_driver.get_node.return_value = None

        tool = FindCallersTool(graph_driver=mock_graph_driver)

        with pytest.raises(SymbolNotFoundError):
            tool.find(
                CallGraphInput(
                    symbol_id="scip-python myapp module/nonexistent.",
                    repo_id="myrepo",
                )
            )


# =============================================================================
# FindCalleesTool Tests
# =============================================================================


class TestFindCalleesTool:
    """Tests for FindCalleesTool."""

    def test_find_callees_basic(self, mock_graph_driver_with_callees, sample_scip_id):
        """Test basic find callees functionality."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
        )

        tool = FindCalleesTool(graph_driver=mock_graph_driver_with_callees)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert result is not None
        assert len(result.nodes) >= 1
        assert result.meta is not None

    def test_find_callees_with_depth(self, mock_graph_driver_with_callees, sample_scip_id):
        """Test find callees with specified depth."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
            CallGraphConfig,
        )

        config = CallGraphConfig(depth=3)
        tool = FindCalleesTool(
            graph_driver=mock_graph_driver_with_callees,
            config=config,
        )

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo", depth=3)
        )

        assert result is not None

    def test_find_callees_no_callees(self, mock_graph_driver, sample_scip_id):
        """Test find callees when symbol has no callees."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
        )

        tool = FindCalleesTool(graph_driver=mock_graph_driver)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert result is not None
        # Should return empty graph but not error
        assert len(result.edges) == 0

    def test_find_callees_returns_edges(self, mock_graph_driver_with_callees, sample_scip_id):
        """Test find callees returns proper edges."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
        )

        tool = FindCalleesTool(graph_driver=mock_graph_driver_with_callees)

        result = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )

        assert len(result.edges) >= 1
        edge = result.edges[0]
        assert edge.type == "CALLS"
        assert edge.from_id == sample_scip_id

    def test_find_callees_symbol_not_found(self, mock_graph_driver):
        """Test find callees when symbol doesn't exist."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
            SymbolNotFoundError,
        )

        mock_graph_driver.get_node.return_value = None

        tool = FindCalleesTool(graph_driver=mock_graph_driver)

        with pytest.raises(SymbolNotFoundError):
            tool.find(
                CallGraphInput(
                    symbol_id="scip-python myapp module/nonexistent.",
                    repo_id="myrepo",
                )
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_invalid_symbol_id_error(self, mock_graph_driver):
        """Test InvalidSymbolIDError for empty IDs."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
            InvalidInputError,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.find(CallGraphInput(symbol_id="", repo_id="myrepo"))

    def test_missing_repo_id_error(self, mock_graph_driver, sample_scip_id):
        """Test error when repo_id is missing."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
            InvalidInputError,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.find(CallGraphInput(symbol_id=sample_scip_id, repo_id=""))

    def test_graph_error_handled(self, mock_graph_driver, sample_scip_id):
        """Test graph errors are handled gracefully."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
            CallGraphError,
        )

        mock_graph_driver.get_incoming_edges.side_effect = Exception("Graph unavailable")

        tool = FindCallersTool(graph_driver=mock_graph_driver)

        with pytest.raises(CallGraphError) as exc_info:
            tool.find(
                CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
            )

        assert "Graph unavailable" in str(exc_info.value)


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_find_callers_latency(self, mock_graph_driver_with_callers, sample_scip_id):
        """Test find callers completes quickly with mocks."""
        from openmemory.api.tools.call_graph import (
            FindCallersTool,
            CallGraphInput,
        )

        tool = FindCallersTool(graph_driver=mock_graph_driver_with_callers)

        start = time.perf_counter()
        _ = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 100

    def test_find_callees_latency(self, mock_graph_driver_with_callees, sample_scip_id):
        """Test find callees completes quickly with mocks."""
        from openmemory.api.tools.call_graph import (
            FindCalleesTool,
            CallGraphInput,
        )

        tool = FindCalleesTool(graph_driver=mock_graph_driver_with_callees)

        start = time.perf_counter()
        _ = tool.find(
            CallGraphInput(symbol_id=sample_scip_id, repo_id="myrepo")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_find_callers_tool(self, mock_graph_driver):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.call_graph import create_find_callers_tool

        tool = create_find_callers_tool(graph_driver=mock_graph_driver)

        assert tool is not None
        assert tool.config.depth == 1  # Default

    def test_create_find_callees_tool(self, mock_graph_driver):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.call_graph import create_find_callees_tool

        tool = create_find_callees_tool(graph_driver=mock_graph_driver)

        assert tool is not None
        assert tool.config.depth == 1  # Default

    def test_create_with_custom_config(self, mock_graph_driver):
        """Test factory function with custom config."""
        from openmemory.api.tools.call_graph import (
            create_find_callers_tool,
            CallGraphConfig,
        )

        config = CallGraphConfig(depth=5, max_nodes=200)

        tool = create_find_callers_tool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        assert tool.config.depth == 5
        assert tool.config.max_nodes == 200
