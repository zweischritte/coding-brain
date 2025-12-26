"""Tests for impact_analysis tool.

This module tests the impact_analysis MCP tool with TDD approach:
- ImpactAnalysisConfig: Configuration and defaults
- ImpactInput: Input validation
- ImpactOutput: Result structure
- ImpactAnalysisTool: Main tool entry point
- Confidence levels and affected files analysis
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
def mock_graph_driver_with_dependencies(mock_graph_driver, sample_scip_id):
    """Create a mock graph driver with dependencies for impact analysis."""
    # Setup caller edges (things that depend on this symbol)
    caller_edge = MagicMock()
    caller_edge.source_id = "scip-python myapp caller/caller_func."
    caller_edge.target_id = sample_scip_id
    caller_edge.edge_type = CodeEdgeType.CALLS
    caller_edge.properties = {}

    mock_graph_driver.get_incoming_edges.return_value = [caller_edge]

    # Setup nodes
    caller_node = MagicMock()
    caller_node.properties = {
        "scip_id": "scip-python myapp caller/caller_func.",
        "name": "caller_func",
        "kind": "function",
        "file_path": "/path/to/caller.py",
        "line_start": 5,
        "line_end": 15,
    }

    file_node = MagicMock()
    file_node.properties = {
        "path": "/path/to/module.py",
        "language": "python",
    }

    def get_node_side_effect(node_id):
        if node_id == sample_scip_id:
            return mock_graph_driver.get_node.return_value
        elif node_id == "scip-python myapp caller/caller_func.":
            return caller_node
        elif node_id == "/path/to/module.py":
            return file_node
        return None

    mock_graph_driver.get_node.side_effect = get_node_side_effect

    return mock_graph_driver


# =============================================================================
# ImpactAnalysisConfig Tests
# =============================================================================


class TestImpactAnalysisConfig:
    """Tests for ImpactAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.impact_analysis import ImpactAnalysisConfig

        config = ImpactAnalysisConfig()

        assert config.max_depth == 3
        assert config.confidence_threshold == "probable"
        assert config.include_cross_language is False
        assert config.max_affected_files == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.impact_analysis import ImpactAnalysisConfig

        config = ImpactAnalysisConfig(
            max_depth=5,
            confidence_threshold="definite",
            include_cross_language=True,
            max_affected_files=50,
        )

        assert config.max_depth == 5
        assert config.confidence_threshold == "definite"
        assert config.include_cross_language is True
        assert config.max_affected_files == 50


# =============================================================================
# ImpactInput Tests
# =============================================================================


class TestImpactInput:
    """Tests for ImpactInput dataclass."""

    def test_changed_files_input(self):
        """Test input with changed_files."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            changed_files=["/path/to/file.py", "/path/to/other.py"],
        )

        assert input_data.repo_id == "myrepo"
        assert len(input_data.changed_files) == 2

    def test_symbol_id_input(self):
        """Test input with symbol_id."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            symbol_id="scip-python myapp module/func.",
        )

        assert input_data.symbol_id == "scip-python myapp module/func."

    def test_depth_override(self):
        """Test depth override in input."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            symbol_id="scip-python myapp module/func.",
            max_depth=5,
        )

        assert input_data.max_depth == 5


# =============================================================================
# ImpactOutput Tests
# =============================================================================


class TestImpactOutput:
    """Tests for ImpactOutput dataclass."""

    def test_output_structure(self):
        """Test output has all required fields."""
        from openmemory.api.tools.impact_analysis import (
            ImpactOutput,
            AffectedFile,
            ResponseMeta,
        )

        affected = AffectedFile(
            file_path="/path/to/file.py",
            reason="Called by changed symbol",
            confidence=0.9,
        )

        meta = ResponseMeta(request_id="req-123")

        output = ImpactOutput(
            affected_files=[affected],
            meta=meta,
        )

        assert len(output.affected_files) == 1
        assert output.affected_files[0].file_path == "/path/to/file.py"
        assert output.meta.request_id == "req-123"

    def test_output_with_multiple_files(self):
        """Test output with multiple affected files."""
        from openmemory.api.tools.impact_analysis import (
            ImpactOutput,
            AffectedFile,
            ResponseMeta,
        )

        affected1 = AffectedFile(
            file_path="/path/to/file1.py",
            reason="Direct caller",
            confidence=0.95,
        )
        affected2 = AffectedFile(
            file_path="/path/to/file2.py",
            reason="Transitive dependency",
            confidence=0.7,
        )

        output = ImpactOutput(
            affected_files=[affected1, affected2],
            meta=ResponseMeta(request_id="req-456"),
        )

        assert len(output.affected_files) == 2


# =============================================================================
# ImpactAnalysisTool Tests
# =============================================================================


class TestImpactAnalysisTool:
    """Tests for ImpactAnalysisTool."""

    def test_analyze_symbol_impact(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test analyzing impact of a symbol change."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver_with_dependencies)

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id)
        )

        assert result is not None
        assert len(result.affected_files) >= 1
        assert result.meta is not None

    def test_analyze_file_changes(self, mock_graph_driver):
        """Test analyzing impact of file changes."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver)

        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                changed_files=["/path/to/module.py"],
            )
        )

        assert result is not None

    def test_analyze_with_depth_limit(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test impact analysis with depth limit."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            ImpactAnalysisConfig,
        )

        config = ImpactAnalysisConfig(max_depth=1)
        tool = ImpactAnalysisTool(
            graph_driver=mock_graph_driver_with_dependencies,
            config=config,
        )

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id, max_depth=1)
        )

        assert result is not None

    def test_analyze_returns_confidence(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test that analysis returns confidence levels."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver_with_dependencies)

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id)
        )

        if result.affected_files:
            assert result.affected_files[0].confidence is not None
            assert 0.0 <= result.affected_files[0].confidence <= 1.0

    def test_analyze_returns_reasons(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test that analysis returns reasons for impact."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver_with_dependencies)

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id)
        )

        if result.affected_files:
            assert result.affected_files[0].reason is not None
            assert len(result.affected_files[0].reason) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_missing_input_error(self, mock_graph_driver):
        """Test error when neither changed_files nor symbol_id provided."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            InvalidInputError,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.analyze(ImpactInput(repo_id="myrepo"))

    def test_missing_repo_id_error(self, mock_graph_driver, sample_scip_id):
        """Test error when repo_id is missing."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            InvalidInputError,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver)

        with pytest.raises(InvalidInputError):
            tool.analyze(ImpactInput(repo_id="", symbol_id=sample_scip_id))

    def test_graph_error_handled(self, mock_graph_driver, sample_scip_id):
        """Test graph errors are handled gracefully."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            ImpactAnalysisError,
        )

        mock_graph_driver.get_node.side_effect = Exception("Graph unavailable")

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver)

        with pytest.raises(ImpactAnalysisError) as exc_info:
            tool.analyze(ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id))

        assert "Graph unavailable" in str(exc_info.value)


# =============================================================================
# Confidence Threshold Tests
# =============================================================================


class TestConfidenceThreshold:
    """Tests for confidence threshold filtering."""

    def test_filter_by_definite_confidence(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test filtering to only definite confidence."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            ImpactAnalysisConfig,
        )

        config = ImpactAnalysisConfig(confidence_threshold="definite")
        tool = ImpactAnalysisTool(
            graph_driver=mock_graph_driver_with_dependencies,
            config=config,
        )

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id)
        )

        # All results should have high confidence
        for affected in result.affected_files:
            assert affected.confidence >= 0.9

    def test_filter_by_probable_confidence(
        self, mock_graph_driver_with_dependencies, sample_scip_id
    ):
        """Test filtering to probable confidence."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
            ImpactAnalysisConfig,
        )

        config = ImpactAnalysisConfig(confidence_threshold="probable")
        tool = ImpactAnalysisTool(
            graph_driver=mock_graph_driver_with_dependencies,
            config=config,
        )

        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id)
        )

        for affected in result.affected_files:
            assert affected.confidence >= 0.5


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_analyze_latency(self, mock_graph_driver_with_dependencies, sample_scip_id):
        """Test analyze completes quickly with mocks."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        tool = ImpactAnalysisTool(graph_driver=mock_graph_driver_with_dependencies)

        start = time.perf_counter()
        _ = tool.analyze(ImpactInput(repo_id="myrepo", symbol_id=sample_scip_id))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 100


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_impact_analysis_tool(self, mock_graph_driver):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.impact_analysis import create_impact_analysis_tool

        tool = create_impact_analysis_tool(graph_driver=mock_graph_driver)

        assert tool is not None
        assert tool.config.max_depth == 3  # Default

    def test_create_with_custom_config(self, mock_graph_driver):
        """Test factory function with custom config."""
        from openmemory.api.tools.impact_analysis import (
            create_impact_analysis_tool,
            ImpactAnalysisConfig,
        )

        config = ImpactAnalysisConfig(max_depth=10, max_affected_files=200)

        tool = create_impact_analysis_tool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        assert tool.config.max_depth == 10
        assert tool.config.max_affected_files == 200
