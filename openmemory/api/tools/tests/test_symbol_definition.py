"""Tests for get_symbol_definition tool.

This module tests the get_symbol_definition MCP tool with TDD approach:
- SymbolDefinitionConfig: Configuration and defaults
- SymbolLookupInput: Input validation
- SymbolDefinitionOutput: Result structure
- GetSymbolDefinitionTool: Main tool entry point
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


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
        "docstring": "This is the docstring.",
    }

    driver.get_node.return_value = symbol_node

    return driver


@pytest.fixture
def mock_parser():
    """Create a mock AST parser."""
    parser = MagicMock()

    # Setup mock symbol
    mock_symbol = MagicMock()
    mock_symbol.name = "my_method"
    mock_symbol.docstring = "This is the docstring."
    mock_symbol.signature = "def my_method(self, arg: str) -> int:"
    mock_symbol.line_start = 10
    mock_symbol.line_end = 20

    mock_result = MagicMock()
    mock_result.symbols = [mock_symbol]
    mock_result.success = True

    parser.parse_file.return_value = mock_result

    return parser


@pytest.fixture
def sample_scip_id() -> str:
    """Return a sample SCIP symbol ID."""
    return "scip-python myapp module/MyClass#my_method."


# =============================================================================
# SymbolDefinitionConfig Tests
# =============================================================================


class TestSymbolDefinitionConfig:
    """Tests for SymbolDefinitionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.symbol_definition import SymbolDefinitionConfig

        config = SymbolDefinitionConfig()

        assert config.include_snippet is True
        assert config.snippet_context_lines == 5
        assert config.include_docstring is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.symbol_definition import SymbolDefinitionConfig

        config = SymbolDefinitionConfig(
            include_snippet=False,
            snippet_context_lines=10,
            include_docstring=False,
        )

        assert config.include_snippet is False
        assert config.snippet_context_lines == 10
        assert config.include_docstring is False


# =============================================================================
# SymbolLookupInput Tests
# =============================================================================


class TestSymbolLookupInput:
    """Tests for SymbolLookupInput dataclass."""

    def test_symbol_id_input(self):
        """Test input with symbol_id."""
        from openmemory.api.tools.symbol_definition import SymbolLookupInput

        input_data = SymbolLookupInput(
            symbol_id="scip-python myapp module/func.",
        )

        assert input_data.symbol_id == "scip-python myapp module/func."

    def test_symbol_name_input(self):
        """Test input with symbol_name."""
        from openmemory.api.tools.symbol_definition import SymbolLookupInput

        input_data = SymbolLookupInput(
            symbol_name="my_function",
            file_path="/path/to/file.py",
        )

        assert input_data.symbol_name == "my_function"
        assert input_data.file_path == "/path/to/file.py"

    def test_repo_id_optional(self):
        """Test repo_id is optional."""
        from openmemory.api.tools.symbol_definition import SymbolLookupInput

        input_data = SymbolLookupInput(
            symbol_id="scip-python myapp module/func.",
            repo_id="myrepo",
        )

        assert input_data.repo_id == "myrepo"


# =============================================================================
# SymbolDefinitionOutput Tests
# =============================================================================


class TestSymbolDefinitionOutput:
    """Tests for SymbolDefinitionOutput dataclass."""

    def test_output_structure(self):
        """Test output has all required fields."""
        from openmemory.api.tools.symbol_definition import (
            SymbolDefinitionOutput,
            CodeSymbol,
            ResponseMeta,
        )

        symbol = CodeSymbol(
            symbol_id="scip-python myapp module/func.",
            symbol_name="func",
            symbol_type="function",
        )

        meta = ResponseMeta(request_id="req-123")

        output = SymbolDefinitionOutput(
            symbol=symbol,
            meta=meta,
        )

        assert output.symbol.symbol_name == "func"
        assert output.meta.request_id == "req-123"

    def test_output_with_snippet(self):
        """Test output with code snippet."""
        from openmemory.api.tools.symbol_definition import (
            SymbolDefinitionOutput,
            CodeSymbol,
            ResponseMeta,
        )

        symbol = CodeSymbol(
            symbol_id="scip-python myapp module/func.",
            symbol_name="func",
            symbol_type="function",
            signature="def func(x: int) -> int:",
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
        )

        output = SymbolDefinitionOutput(
            symbol=symbol,
            snippet="def func(x: int) -> int:\n    return x * 2",
            meta=ResponseMeta(request_id="req-456"),
        )

        assert output.snippet is not None
        assert "def func" in output.snippet


# =============================================================================
# GetSymbolDefinitionTool Tests
# =============================================================================


class TestGetSymbolDefinitionTool:
    """Tests for GetSymbolDefinitionTool."""

    def test_get_definition_by_symbol_id(self, mock_graph_driver, sample_scip_id):
        """Test getting definition by symbol ID."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        result = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert result is not None
        assert result.symbol.symbol_name == "my_method"
        assert result.symbol.symbol_type == "method"
        mock_graph_driver.get_node.assert_called_once_with(sample_scip_id)

    def test_get_definition_returns_location(self, mock_graph_driver, sample_scip_id):
        """Test definition includes file location."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        result = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert result.symbol.file_path == "/path/to/module.py"
        assert result.symbol.line_start == 10
        assert result.symbol.line_end == 20

    def test_get_definition_returns_signature(self, mock_graph_driver, sample_scip_id):
        """Test definition includes signature."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        result = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert result.symbol.signature is not None
        assert "my_method" in result.symbol.signature

    def test_get_definition_symbol_not_found(self, mock_graph_driver):
        """Test error when symbol doesn't exist."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
            SymbolNotFoundError,
        )

        mock_graph_driver.get_node.return_value = None

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        with pytest.raises(SymbolNotFoundError):
            tool.get_definition(
                SymbolLookupInput(symbol_id="scip-python myapp module/nonexistent.")
            )

    def test_get_definition_includes_language(self, mock_graph_driver, sample_scip_id):
        """Test definition includes language."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        result = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert result.symbol.language == "python"

    def test_get_definition_with_snippet_disabled(
        self, mock_graph_driver, sample_scip_id
    ):
        """Test definition without snippet."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
            SymbolDefinitionConfig,
        )

        config = SymbolDefinitionConfig(include_snippet=False)
        tool = GetSymbolDefinitionTool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        result = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert result.snippet is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_empty_symbol_id_error(self, mock_graph_driver):
        """Test error when symbol_id is empty."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
            InvalidInputError,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.get_definition(SymbolLookupInput(symbol_id=""))

    def test_no_identifier_error(self, mock_graph_driver):
        """Test error when neither symbol_id nor symbol_name provided."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
            InvalidInputError,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.get_definition(SymbolLookupInput())

    def test_graph_error_handled(self, mock_graph_driver, sample_scip_id):
        """Test graph errors are handled gracefully."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
            SymbolDefinitionError,
        )

        mock_graph_driver.get_node.side_effect = Exception("Graph unavailable")

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        with pytest.raises(SymbolDefinitionError) as exc_info:
            tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))

        assert "Graph unavailable" in str(exc_info.value)


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_get_definition_latency(self, mock_graph_driver, sample_scip_id):
        """Test get definition completes quickly with mocks."""
        from openmemory.api.tools.symbol_definition import (
            GetSymbolDefinitionTool,
            SymbolLookupInput,
        )

        tool = GetSymbolDefinitionTool(graph_driver=mock_graph_driver)

        start = time.perf_counter()
        _ = tool.get_definition(SymbolLookupInput(symbol_id=sample_scip_id))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 50


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_get_symbol_definition_tool(self, mock_graph_driver):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.symbol_definition import (
            create_get_symbol_definition_tool,
        )

        tool = create_get_symbol_definition_tool(graph_driver=mock_graph_driver)

        assert tool is not None
        assert tool.config.include_snippet is True  # Default

    def test_create_with_custom_config(self, mock_graph_driver):
        """Test factory function with custom config."""
        from openmemory.api.tools.symbol_definition import (
            create_get_symbol_definition_tool,
            SymbolDefinitionConfig,
        )

        config = SymbolDefinitionConfig(snippet_context_lines=10)

        tool = create_get_symbol_definition_tool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        assert tool.config.snippet_context_lines == 10
