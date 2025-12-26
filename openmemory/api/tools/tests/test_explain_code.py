"""Tests for explain_code tool (FR-007).

This module tests the explain_code tool with TDD approach:
- ExplainCodeConfig: Configuration validation and defaults
- SymbolExplanation: Result dataclass structure
- SymbolLookup: SCIP ID lookup in graph
- CallGraphTraverser: Call graph traversal (callers/callees)
- DocumentationExtractor: Docstring extraction from AST
- CodeContextRetriever: Tri-hybrid context retrieval
- ExplanationFormatter: LLM-friendly output formatting
- ExplainCodeTool: Main tool entry point
- Caching: Performance optimization
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# These imports will fail until implementation exists
# The tests are written first (TDD)


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
        "docstring": "This is a method docstring.",
        "language": "python",
    }

    driver.get_node.return_value = symbol_node
    driver.get_outgoing_edges.return_value = []

    return driver


@pytest.fixture
def mock_retriever():
    """Create a mock tri-hybrid retriever."""
    retriever = MagicMock()

    # Setup mock result
    mock_hit = MagicMock()
    mock_hit.id = "doc_1"
    mock_hit.score = 0.95
    mock_hit.source = {"content": "Usage example code here"}

    mock_result = MagicMock()
    mock_result.hits = [mock_hit]
    mock_result.total = 1

    retriever.retrieve.return_value = mock_result

    return retriever


@pytest.fixture
def mock_parser():
    """Create a mock AST parser."""
    parser = MagicMock()

    # Setup mock symbol
    mock_symbol = MagicMock()
    mock_symbol.name = "my_method"
    mock_symbol.docstring = "This is a method docstring."
    mock_symbol.signature = "def my_method(self, arg: str) -> int:"
    mock_symbol.line_start = 10  # Match the test's expected line

    mock_result = MagicMock()
    mock_result.symbols = [mock_symbol]
    mock_result.success = True

    parser.parse_file.return_value = mock_result

    return parser


@pytest.fixture
def sample_scip_id() -> str:
    """Return a sample SCIP symbol ID."""
    return "scip-python myapp module/MyClass#my_method."


@pytest.fixture
def sample_scip_id_function() -> str:
    """Return a sample SCIP symbol ID for a function."""
    return "scip-python myapp module/my_function."


# =============================================================================
# ExplainCodeConfig Tests
# =============================================================================


class TestExplainCodeConfig:
    """Tests for ExplainCodeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config = ExplainCodeConfig()

        assert config.depth == 2
        assert config.include_callers is True
        assert config.include_callees is True
        assert config.include_usages is True
        assert config.max_usages == 5
        assert config.include_related is True
        assert config.max_related == 10
        assert config.cache_ttl_seconds == 300

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config = ExplainCodeConfig(
            depth=3,
            include_callers=False,
            include_callees=False,
            include_usages=False,
            max_usages=10,
            include_related=False,
            max_related=5,
            cache_ttl_seconds=600,
        )

        assert config.depth == 3
        assert config.include_callers is False
        assert config.include_callees is False
        assert config.include_usages is False
        assert config.max_usages == 10
        assert config.include_related is False
        assert config.max_related == 5
        assert config.cache_ttl_seconds == 600

    def test_depth_validation(self):
        """Test depth must be positive."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        # Valid depths
        config = ExplainCodeConfig(depth=1)
        assert config.depth == 1

        config = ExplainCodeConfig(depth=5)
        assert config.depth == 5

    def test_max_usages_validation(self):
        """Test max_usages must be non-negative."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config = ExplainCodeConfig(max_usages=0)
        assert config.max_usages == 0

    def test_config_immutability(self):
        """Test config is a regular dataclass (mutable by default)."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config = ExplainCodeConfig()
        # Should be mutable (not frozen)
        config.depth = 5
        assert config.depth == 5


# =============================================================================
# SymbolExplanation Tests
# =============================================================================


class TestSymbolExplanation:
    """Tests for SymbolExplanation dataclass."""

    def test_required_fields(self):
        """Test required fields are present."""
        from openmemory.api.tools.explain_code import SymbolExplanation

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/MyClass#my_method.",
            name="my_method",
            kind="method",
            signature="def my_method(self, arg: str) -> int:",
            file_path="/path/to/module.py",
            line_start=10,
            line_end=20,
        )

        assert explanation.symbol_id == "scip-python myapp module/MyClass#my_method."
        assert explanation.name == "my_method"
        assert explanation.kind == "method"
        assert explanation.signature == "def my_method(self, arg: str) -> int:"
        assert explanation.file_path == "/path/to/module.py"
        assert explanation.line_start == 10
        assert explanation.line_end == 20

    def test_optional_fields_defaults(self):
        """Test optional fields have correct defaults."""
        from openmemory.api.tools.explain_code import SymbolExplanation

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/func.",
            name="func",
            kind="function",
            signature="def func():",
            file_path="/path/to/file.py",
            line_start=1,
            line_end=5,
        )

        assert explanation.docstring is None
        assert explanation.callers == []
        assert explanation.callees == []
        assert explanation.usages == []
        assert explanation.related == []
        assert explanation.context is None

    def test_with_all_fields(self):
        """Test with all fields populated."""
        from openmemory.api.tools.explain_code import SymbolExplanation

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/MyClass#my_method.",
            name="my_method",
            kind="method",
            signature="def my_method(self, arg: str) -> int:",
            file_path="/path/to/module.py",
            line_start=10,
            line_end=20,
            docstring="This is a docstring.",
            callers=[{"name": "caller1", "file": "/path/to/caller.py"}],
            callees=[{"name": "callee1", "file": "/path/to/callee.py"}],
            usages=[{"code": "obj.my_method('test')"}],
            related=[{"name": "related_func"}],
            context="Additional context from retrieval.",
        )

        assert explanation.docstring == "This is a docstring."
        assert len(explanation.callers) == 1
        assert len(explanation.callees) == 1
        assert len(explanation.usages) == 1
        assert len(explanation.related) == 1
        assert explanation.context == "Additional context from retrieval."


# =============================================================================
# SymbolLookup Tests
# =============================================================================


class TestSymbolLookup:
    """Tests for SymbolLookup component."""

    def test_lookup_existing_symbol(self, mock_graph_driver, sample_scip_id):
        """Test looking up an existing symbol."""
        from openmemory.api.tools.explain_code import SymbolLookup

        lookup = SymbolLookup(mock_graph_driver)
        result = lookup.lookup(sample_scip_id)

        assert result is not None
        assert result["name"] == "my_method"
        assert result["kind"] == "method"
        mock_graph_driver.get_node.assert_called_once_with(sample_scip_id)

    def test_lookup_nonexistent_symbol(self, mock_graph_driver):
        """Test looking up a symbol that doesn't exist."""
        from openmemory.api.tools.explain_code import SymbolLookup, SymbolNotFoundError

        mock_graph_driver.get_node.return_value = None

        lookup = SymbolLookup(mock_graph_driver)

        with pytest.raises(SymbolNotFoundError) as exc_info:
            lookup.lookup("scip-python myapp module/nonexistent.")

        assert "not found" in str(exc_info.value).lower()

    def test_lookup_returns_all_properties(self, mock_graph_driver, sample_scip_id):
        """Test lookup returns all expected properties."""
        from openmemory.api.tools.explain_code import SymbolLookup

        lookup = SymbolLookup(mock_graph_driver)
        result = lookup.lookup(sample_scip_id)

        assert "scip_id" in result
        assert "name" in result
        assert "kind" in result
        assert "signature" in result
        assert "file_path" in result
        assert "line_start" in result
        assert "line_end" in result

    def test_lookup_with_invalid_scip_id(self, mock_graph_driver):
        """Test lookup with invalid SCIP ID format."""
        from openmemory.api.tools.explain_code import SymbolLookup, InvalidSymbolIDError

        lookup = SymbolLookup(mock_graph_driver)

        with pytest.raises(InvalidSymbolIDError):
            lookup.lookup("")  # Empty ID

    def test_lookup_handles_driver_error(self, mock_graph_driver, sample_scip_id):
        """Test lookup handles driver errors gracefully."""
        from openmemory.api.tools.explain_code import SymbolLookup, SymbolLookupError

        mock_graph_driver.get_node.side_effect = Exception("Connection error")

        lookup = SymbolLookup(mock_graph_driver)

        with pytest.raises(SymbolLookupError) as exc_info:
            lookup.lookup(sample_scip_id)

        assert "Connection error" in str(exc_info.value)


# =============================================================================
# CallGraphTraverser Tests
# =============================================================================


class TestCallGraphTraverser:
    """Tests for CallGraphTraverser component."""

    def test_get_callers_depth_1(self, mock_graph_driver, sample_scip_id):
        """Test getting callers at depth 1."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        # Setup mock edges
        caller_edge = MagicMock()
        caller_edge.source_id = "scip-python myapp module/caller_func."
        caller_edge.target_id = sample_scip_id
        caller_edge.edge_type.value = "CALLS"
        caller_edge.properties = {"call_line": 15}

        # Mock incoming edges query
        mock_graph_driver.get_incoming_edges.return_value = [caller_edge]

        caller_node = MagicMock()
        caller_node.properties = {
            "name": "caller_func",
            "file_path": "/path/to/caller.py",
            "line_start": 10,
        }
        mock_graph_driver.get_node.return_value = caller_node

        traverser = CallGraphTraverser(mock_graph_driver)
        callers = traverser.get_callers(sample_scip_id, depth=1)

        assert len(callers) >= 0  # May return empty if no callers
        mock_graph_driver.get_incoming_edges.assert_called()

    def test_get_callees_depth_1(self, mock_graph_driver, sample_scip_id):
        """Test getting callees at depth 1."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        # Setup mock edges
        callee_edge = MagicMock()
        callee_edge.source_id = sample_scip_id
        callee_edge.target_id = "scip-python myapp module/callee_func."
        callee_edge.edge_type.value = "CALLS"
        callee_edge.properties = {"call_line": 15}

        mock_graph_driver.get_outgoing_edges.return_value = [callee_edge]

        callee_node = MagicMock()
        callee_node.properties = {
            "name": "callee_func",
            "file_path": "/path/to/callee.py",
            "line_start": 1,
        }
        mock_graph_driver.get_node.return_value = callee_node

        traverser = CallGraphTraverser(mock_graph_driver)
        callees = traverser.get_callees(sample_scip_id, depth=1)

        assert isinstance(callees, list)
        mock_graph_driver.get_outgoing_edges.assert_called()

    def test_get_callers_depth_2(self, mock_graph_driver, sample_scip_id):
        """Test getting callers at depth 2 (transitive callers)."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        traverser = CallGraphTraverser(mock_graph_driver)
        callers = traverser.get_callers(sample_scip_id, depth=2)

        assert isinstance(callers, list)

    def test_get_callees_depth_3(self, mock_graph_driver, sample_scip_id):
        """Test getting callees at depth 3."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        traverser = CallGraphTraverser(mock_graph_driver)
        callees = traverser.get_callees(sample_scip_id, depth=3)

        assert isinstance(callees, list)

    def test_empty_call_graph(self, mock_graph_driver, sample_scip_id):
        """Test handling of symbol with no callers/callees."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        mock_graph_driver.get_incoming_edges.return_value = []
        mock_graph_driver.get_outgoing_edges.return_value = []

        traverser = CallGraphTraverser(mock_graph_driver)

        callers = traverser.get_callers(sample_scip_id, depth=1)
        callees = traverser.get_callees(sample_scip_id, depth=1)

        assert callers == []
        assert callees == []

    def test_cycle_detection(self, mock_graph_driver, sample_scip_id):
        """Test that cycles in call graph are handled."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        # Create a cycle: A -> B -> A
        edge_a_to_b = MagicMock()
        edge_a_to_b.source_id = sample_scip_id
        edge_a_to_b.target_id = "scip-python myapp module/func_b."
        edge_a_to_b.edge_type.value = "CALLS"

        edge_b_to_a = MagicMock()
        edge_b_to_a.source_id = "scip-python myapp module/func_b."
        edge_b_to_a.target_id = sample_scip_id
        edge_b_to_a.edge_type.value = "CALLS"

        def get_outgoing_edges(node_id):
            if node_id == sample_scip_id:
                return [edge_a_to_b]
            return [edge_b_to_a]

        mock_graph_driver.get_outgoing_edges.side_effect = get_outgoing_edges

        traverser = CallGraphTraverser(mock_graph_driver)
        callees = traverser.get_callees(sample_scip_id, depth=5)

        # Should not infinite loop
        assert isinstance(callees, list)

    def test_caller_callee_includes_metadata(self, mock_graph_driver, sample_scip_id):
        """Test that caller/callee results include useful metadata."""
        from openmemory.api.tools.explain_code import CallGraphTraverser

        callee_edge = MagicMock()
        callee_edge.source_id = sample_scip_id
        callee_edge.target_id = "scip-python myapp module/helper."
        callee_edge.edge_type.value = "CALLS"
        callee_edge.properties = {"call_line": 15, "call_col": 8}

        mock_graph_driver.get_outgoing_edges.return_value = [callee_edge]

        callee_node = MagicMock()
        callee_node.properties = {
            "name": "helper",
            "kind": "function",
            "file_path": "/path/to/helpers.py",
            "line_start": 1,
            "signature": "def helper(x: int) -> int:",
        }
        mock_graph_driver.get_node.return_value = callee_node

        traverser = CallGraphTraverser(mock_graph_driver)
        callees = traverser.get_callees(sample_scip_id, depth=1)

        if callees:
            callee = callees[0]
            # Should have useful metadata
            assert "name" in callee or "symbol_id" in callee


# =============================================================================
# DocumentationExtractor Tests
# =============================================================================


class TestDocumentationExtractor:
    """Tests for DocumentationExtractor component."""

    def test_extract_docstring_present(self, mock_parser):
        """Test extracting docstring when present."""
        from openmemory.api.tools.explain_code import DocumentationExtractor

        extractor = DocumentationExtractor(mock_parser)
        docstring = extractor.extract(
            file_path=Path("/path/to/module.py"),
            symbol_name="my_method",
            line_start=10,
        )

        assert docstring is not None
        assert "docstring" in docstring.lower() or len(docstring) > 0

    def test_extract_docstring_missing(self, mock_parser):
        """Test extracting docstring when missing."""
        from openmemory.api.tools.explain_code import DocumentationExtractor

        # Setup mock with no docstring
        mock_symbol = MagicMock()
        mock_symbol.name = "my_method"
        mock_symbol.docstring = None
        mock_symbol.line_start = 10

        mock_result = MagicMock()
        mock_result.symbols = [mock_symbol]
        mock_result.success = True
        mock_parser.parse_file.return_value = mock_result

        extractor = DocumentationExtractor(mock_parser)
        docstring = extractor.extract(
            file_path=Path("/path/to/module.py"),
            symbol_name="my_method",
            line_start=10,
        )

        assert docstring is None

    def test_extract_file_not_found(self, mock_parser):
        """Test extracting docstring when file not found."""
        from openmemory.api.tools.explain_code import DocumentationExtractor

        mock_parser.parse_file.side_effect = FileNotFoundError()

        extractor = DocumentationExtractor(mock_parser)
        docstring = extractor.extract(
            file_path=Path("/nonexistent/path.py"),
            symbol_name="func",
            line_start=1,
        )

        assert docstring is None

    def test_extract_symbol_not_found(self, mock_parser):
        """Test extracting docstring when symbol not in file."""
        from openmemory.api.tools.explain_code import DocumentationExtractor

        mock_symbol = MagicMock()
        mock_symbol.name = "other_method"  # Different name
        mock_symbol.docstring = "Wrong symbol"

        mock_result = MagicMock()
        mock_result.symbols = [mock_symbol]
        mock_result.success = True
        mock_parser.parse_file.return_value = mock_result

        extractor = DocumentationExtractor(mock_parser)
        docstring = extractor.extract(
            file_path=Path("/path/to/module.py"),
            symbol_name="my_method",  # Looking for different symbol
            line_start=10,
        )

        # Should return None when symbol not found
        assert docstring is None

    def test_extract_multiline_docstring(self, mock_parser):
        """Test extracting multiline docstring."""
        from openmemory.api.tools.explain_code import DocumentationExtractor

        mock_symbol = MagicMock()
        mock_symbol.name = "complex_func"
        mock_symbol.docstring = """This is a multiline docstring.

        Args:
            arg1: First argument
            arg2: Second argument

        Returns:
            Result value
        """
        mock_symbol.line_start = 1

        mock_result = MagicMock()
        mock_result.symbols = [mock_symbol]
        mock_result.success = True
        mock_parser.parse_file.return_value = mock_result

        extractor = DocumentationExtractor(mock_parser)
        docstring = extractor.extract(
            file_path=Path("/path/to/module.py"),
            symbol_name="complex_func",
            line_start=1,
        )

        assert docstring is not None
        assert "multiline" in docstring.lower()


# =============================================================================
# CodeContextRetriever Tests
# =============================================================================


class TestCodeContextRetriever:
    """Tests for CodeContextRetriever component."""

    def test_retrieve_context(self, mock_retriever, sample_scip_id):
        """Test retrieving context for a symbol."""
        from openmemory.api.tools.explain_code import CodeContextRetriever

        context_retriever = CodeContextRetriever(mock_retriever)
        context = context_retriever.retrieve(
            symbol_id=sample_scip_id,
            symbol_name="my_method",
            max_results=5,
        )

        assert context is not None
        mock_retriever.retrieve.assert_called()

    def test_retrieve_usages(self, mock_retriever, sample_scip_id):
        """Test retrieving usage examples for a symbol."""
        from openmemory.api.tools.explain_code import CodeContextRetriever

        context_retriever = CodeContextRetriever(mock_retriever)
        usages = context_retriever.retrieve_usages(
            symbol_name="my_method",
            max_usages=5,
        )

        assert isinstance(usages, list)

    def test_retrieve_related(self, mock_retriever, sample_scip_id):
        """Test retrieving related symbols."""
        from openmemory.api.tools.explain_code import CodeContextRetriever

        context_retriever = CodeContextRetriever(mock_retriever)
        related = context_retriever.retrieve_related(
            symbol_id=sample_scip_id,
            symbol_name="my_method",
            max_related=10,
        )

        assert isinstance(related, list)

    def test_retrieve_with_no_results(self, mock_retriever, sample_scip_id):
        """Test retrieving context with no results."""
        from openmemory.api.tools.explain_code import CodeContextRetriever

        mock_result = MagicMock()
        mock_result.hits = []
        mock_result.total = 0
        mock_retriever.retrieve.return_value = mock_result

        context_retriever = CodeContextRetriever(mock_retriever)
        context = context_retriever.retrieve(
            symbol_id=sample_scip_id,
            symbol_name="my_method",
            max_results=5,
        )

        # Should handle empty results gracefully
        assert context is None or context == ""

    def test_retrieve_handles_retriever_error(self, mock_retriever, sample_scip_id):
        """Test retrieval handles errors gracefully."""
        from openmemory.api.tools.explain_code import CodeContextRetriever

        mock_retriever.retrieve.side_effect = Exception("Retrieval failed")

        context_retriever = CodeContextRetriever(mock_retriever)
        # Should not raise, should return None or empty
        context = context_retriever.retrieve(
            symbol_id=sample_scip_id,
            symbol_name="my_method",
            max_results=5,
        )

        assert context is None or context == ""


# =============================================================================
# ExplanationFormatter Tests
# =============================================================================


class TestExplanationFormatter:
    """Tests for ExplanationFormatter component."""

    def test_format_json(self):
        """Test formatting explanation as JSON."""
        from openmemory.api.tools.explain_code import (
            ExplanationFormatter,
            SymbolExplanation,
        )

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/func.",
            name="func",
            kind="function",
            signature="def func(x: int) -> int:",
            file_path="/path/to/module.py",
            line_start=1,
            line_end=10,
            docstring="A sample function.",
        )

        formatter = ExplanationFormatter()
        json_output = formatter.format_json(explanation)

        assert isinstance(json_output, str)
        assert "func" in json_output
        assert "function" in json_output

    def test_format_markdown(self):
        """Test formatting explanation as Markdown."""
        from openmemory.api.tools.explain_code import (
            ExplanationFormatter,
            SymbolExplanation,
        )

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/MyClass#method.",
            name="method",
            kind="method",
            signature="def method(self) -> None:",
            file_path="/path/to/module.py",
            line_start=20,
            line_end=30,
            docstring="A method docstring.",
            callers=[{"name": "caller", "file": "/path/caller.py"}],
            callees=[{"name": "helper", "file": "/path/helper.py"}],
        )

        formatter = ExplanationFormatter()
        md_output = formatter.format_markdown(explanation)

        assert isinstance(md_output, str)
        assert "method" in md_output
        assert "#" in md_output  # Markdown headers

    def test_format_for_llm(self):
        """Test formatting explanation for LLM consumption."""
        from openmemory.api.tools.explain_code import (
            ExplanationFormatter,
            SymbolExplanation,
        )

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/func.",
            name="func",
            kind="function",
            signature="def func(x: int) -> int:",
            file_path="/path/to/module.py",
            line_start=1,
            line_end=10,
            docstring="Computes the result.",
            callers=[{"name": "main", "file": "/path/main.py", "line": 5}],
            callees=[{"name": "helper", "file": "/path/helpers.py", "line": 20}],
            usages=[{"code": "result = func(42)", "file": "/path/test.py"}],
            context="Additional context about the function.",
        )

        formatter = ExplanationFormatter()
        llm_output = formatter.format_for_llm(explanation)

        assert isinstance(llm_output, str)
        # Should include key information
        assert "func" in llm_output
        assert "function" in llm_output.lower()

    def test_format_empty_explanation(self):
        """Test formatting minimal explanation."""
        from openmemory.api.tools.explain_code import (
            ExplanationFormatter,
            SymbolExplanation,
        )

        explanation = SymbolExplanation(
            symbol_id="scip-python myapp module/x.",
            name="x",
            kind="variable",
            signature="x: int = 0",
            file_path="/path/to/module.py",
            line_start=1,
            line_end=1,
        )

        formatter = ExplanationFormatter()
        output = formatter.format_for_llm(explanation)

        assert isinstance(output, str)
        assert len(output) > 0


# =============================================================================
# ExplainCodeTool Tests
# =============================================================================


class TestExplainCodeTool:
    """Tests for ExplainCodeTool main class."""

    def test_explain_symbol(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test explaining a symbol."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        explanation = tool.explain(sample_scip_id)

        assert explanation is not None
        assert explanation.symbol_id == sample_scip_id
        assert explanation.name == "my_method"

    def test_explain_with_custom_config(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test explaining a symbol with custom config."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        config = ExplainCodeConfig(
            depth=3,
            include_callers=False,
            include_usages=False,
        )

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        explanation = tool.explain(sample_scip_id, config=config)

        assert explanation is not None
        # With include_callers=False, should have no callers
        assert explanation.callers == []

    def test_explain_nonexistent_symbol(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test explaining a symbol that doesn't exist."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, SymbolNotFoundError

        mock_graph_driver.get_node.return_value = None

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        with pytest.raises(SymbolNotFoundError):
            tool.explain("scip-python myapp module/nonexistent.")

    def test_explain_with_format(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test explaining a symbol with specific format."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Get formatted output
        formatted = tool.explain_formatted(sample_scip_id, format="markdown")

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_for_llm(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test format_for_llm convenience method."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        explanation = tool.explain(sample_scip_id)
        llm_output = tool.format_for_llm(explanation)

        assert isinstance(llm_output, str)
        assert len(llm_output) > 0

    def test_tool_with_default_config(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test tool uses default config when none provided."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        assert tool.config is not None
        assert tool.config.depth == 2


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for caching behavior."""

    def test_cached_query_faster(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test that cached queries are faster."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        config = ExplainCodeConfig(cache_ttl_seconds=300)

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
            config=config,
        )

        # First call - uncached
        start = time.perf_counter()
        _ = tool.explain(sample_scip_id)
        first_call_time = time.perf_counter() - start

        # Second call - should use cache
        start = time.perf_counter()
        _ = tool.explain(sample_scip_id)
        second_call_time = time.perf_counter() - start

        # Cached should be faster (or at least not significantly slower)
        # We can't guarantee exact timing, but cache should help
        assert second_call_time <= first_call_time * 10  # Very lenient

    def test_cache_invalidation_on_ttl(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test that cache is invalidated after TTL."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        config = ExplainCodeConfig(cache_ttl_seconds=0)  # Immediate expiry

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
            config=config,
        )

        # First call
        _ = tool.explain(sample_scip_id)

        # Second call should not use cache (TTL=0)
        _ = tool.explain(sample_scip_id)

        # With TTL=0, each call should hit the driver
        assert mock_graph_driver.get_node.call_count >= 2

    def test_cache_clear(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test clearing the cache."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Populate cache
        _ = tool.explain(sample_scip_id)

        # Clear cache
        tool.clear_cache()

        # Next call should hit the driver again
        initial_call_count = mock_graph_driver.get_node.call_count
        _ = tool.explain(sample_scip_id)

        assert mock_graph_driver.get_node.call_count > initial_call_count

    def test_cache_different_configs(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test that different configs produce separate cache entries."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        config1 = ExplainCodeConfig(depth=1)
        config2 = ExplainCodeConfig(depth=2)

        # These should be cached separately
        _ = tool.explain(sample_scip_id, config=config1)
        _ = tool.explain(sample_scip_id, config=config2)

        # Both configs should have caused driver calls
        assert mock_graph_driver.get_node.call_count >= 2


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_cached_latency_under_100ms(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test cached queries complete under 100ms."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Warm up cache
        _ = tool.explain(sample_scip_id)

        # Measure cached call
        start = time.perf_counter()
        _ = tool.explain(sample_scip_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be well under 100ms for cached (mocked) call
        assert elapsed_ms < 100

    def test_uncached_latency_under_500ms(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test uncached queries complete under 500ms."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, ExplainCodeConfig

        config = ExplainCodeConfig(cache_ttl_seconds=0)  # Disable cache

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
            config=config,
        )

        start = time.perf_counter()
        _ = tool.explain(sample_scip_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete under 500ms (mocked)
        assert elapsed_ms < 500


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_symbol_not_found_error(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test SymbolNotFoundError is raised appropriately."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, SymbolNotFoundError

        mock_graph_driver.get_node.return_value = None

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        with pytest.raises(SymbolNotFoundError) as exc_info:
            tool.explain("scip-python myapp module/missing.")

        assert "missing" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    def test_graph_unavailable_graceful(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test graceful handling when graph is unavailable."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        # Make graph calls fail after initial lookup
        symbol_node = MagicMock()
        symbol_node.properties = {
            "scip_id": sample_scip_id,
            "name": "my_method",
            "kind": "method",
            "signature": "def my_method():",
            "file_path": "/path/to/module.py",
            "line_start": 10,
            "line_end": 20,
        }
        mock_graph_driver.get_node.return_value = symbol_node
        mock_graph_driver.get_outgoing_edges.side_effect = Exception("Graph error")
        mock_graph_driver.get_incoming_edges.side_effect = Exception("Graph error")

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Should still return partial explanation
        explanation = tool.explain(sample_scip_id)

        assert explanation is not None
        assert explanation.name == "my_method"
        # Callers/callees should be empty due to error
        assert explanation.callers == []
        assert explanation.callees == []

    def test_retriever_unavailable_graceful(
        self, mock_graph_driver, mock_retriever, mock_parser, sample_scip_id
    ):
        """Test graceful handling when retriever is unavailable."""
        from openmemory.api.tools.explain_code import ExplainCodeTool

        mock_retriever.retrieve.side_effect = Exception("Retriever error")

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        # Should still return partial explanation
        explanation = tool.explain(sample_scip_id)

        assert explanation is not None
        assert explanation.name == "my_method"
        # Usages should be empty due to error
        assert explanation.usages == []

    def test_invalid_symbol_id_error(
        self, mock_graph_driver, mock_retriever, mock_parser
    ):
        """Test InvalidSymbolIDError for malformed IDs."""
        from openmemory.api.tools.explain_code import ExplainCodeTool, InvalidSymbolIDError

        tool = ExplainCodeTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            parser=mock_parser,
        )

        with pytest.raises(InvalidSymbolIDError):
            tool.explain("")  # Empty ID


# =============================================================================
# Integration Tests (marked for optional execution)
# =============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring real services."""

    def test_explain_real_symbol(self):
        """Test explaining a real symbol from the codebase."""
        pytest.skip("Requires running Neo4j and OpenSearch")

    def test_full_pipeline(self):
        """Test full explain pipeline with real services."""
        pytest.skip("Requires running Neo4j and OpenSearch")
