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


@pytest.fixture
def mock_graph_driver_with_field_and_schema_edges():
    """Create a mock graph driver with field and schema edges."""
    driver = MagicMock()

    field_id = "scip-typescript mypkg module/User#field:name."
    reader_id = "scip-typescript mypkg module/User#getName."
    writer_id = "scip-typescript mypkg module/User#setName."
    schema_id = "schema::graphql:/path/to/schema.ts:User:name"

    field_node = MagicMock()
    field_node.properties = {
        "scip_id": field_id,
        "name": "name",
        "kind": "field",
        "file_path": "/path/to/model.ts",
        "line_start": 3,
        "line_end": 3,
    }

    reader_node = MagicMock()
    reader_node.properties = {
        "scip_id": reader_id,
        "name": "getName",
        "kind": "method",
        "file_path": "/path/to/reader.ts",
    }

    writer_node = MagicMock()
    writer_node.properties = {
        "scip_id": writer_id,
        "name": "setName",
        "kind": "method",
        "file_path": "/path/to/writer.ts",
    }

    schema_node = MagicMock()
    schema_node.properties = {
        "name": "name",
        "schema_type": "graphql",
        "file_path": "/path/to/schema.ts",
    }

    reads_edge = MagicMock()
    reads_edge.source_id = reader_id
    reads_edge.target_id = field_id
    reads_edge.edge_type = CodeEdgeType.READS
    reads_edge.properties = {}

    writes_edge = MagicMock()
    writes_edge.source_id = writer_id
    writes_edge.target_id = field_id
    writes_edge.edge_type = CodeEdgeType.WRITES
    writes_edge.properties = {}

    schema_edge = MagicMock()
    schema_edge.source_id = field_id
    schema_edge.target_id = schema_id
    schema_edge.edge_type = CodeEdgeType.SCHEMA_EXPOSES
    schema_edge.properties = {}

    def get_node_side_effect(node_id):
        if node_id == field_id:
            return field_node
        if node_id == reader_id:
            return reader_node
        if node_id == writer_id:
            return writer_node
        if node_id == schema_id:
            return schema_node
        return None

    driver.get_node.side_effect = get_node_side_effect
    driver.get_incoming_edges.return_value = [reads_edge, writes_edge]
    driver.get_outgoing_edges.return_value = [schema_edge]

    return driver, field_id


# =============================================================================
# ImpactAnalysisConfig Tests
# =============================================================================


class TestImpactAnalysisConfig:
    """Tests for ImpactAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.impact_analysis import ImpactAnalysisConfig

        config = ImpactAnalysisConfig()

        assert config.max_depth == 10
        assert config.confidence_threshold == "probable"
        assert config.include_cross_language is False
        assert config.max_affected_files == 100
        assert config.include_inferred_edges is True
        assert config.include_field_edges is True
        assert config.include_schema_edges is True
        assert config.include_path_edges is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.impact_analysis import ImpactAnalysisConfig

        config = ImpactAnalysisConfig(
            max_depth=5,
            confidence_threshold="definite",
            include_cross_language=True,
            max_affected_files=50,
            include_inferred_edges=False,
            include_field_edges=True,
            include_schema_edges=True,
            include_path_edges=True,
        )

        assert config.max_depth == 5
        assert config.confidence_threshold == "definite"
        assert config.include_cross_language is True
        assert config.max_affected_files == 50
        assert config.include_inferred_edges is False
        assert config.include_field_edges is True
        assert config.include_schema_edges is True
        assert config.include_path_edges is True


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

    def test_symbol_name_input(self):
        """Test input with symbol_name."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            symbol_name="my_field",
            parent_name="MyType",
            symbol_kind="field",
            file_path="/path/to/module.py",
        )

        assert input_data.symbol_name == "my_field"
        assert input_data.parent_name == "MyType"
        assert input_data.symbol_kind == "field"
        assert input_data.file_path == "/path/to/module.py"

    def test_depth_override(self):
        """Test depth override in input."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            symbol_id="scip-python myapp module/func.",
            max_depth=5,
        )

        assert input_data.max_depth == 5

    def test_field_edge_flag(self):
        """Test field/schema edge flags in input."""
        from openmemory.api.tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id="myrepo",
            symbol_id="scip-python myapp module/func.",
            include_field_edges=True,
            include_schema_edges=True,
            include_path_edges=True,
        )

        assert input_data.include_field_edges is True
        assert input_data.include_schema_edges is True
        assert input_data.include_path_edges is True


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
        assert output.required_files == []
        assert output.coverage_summary.reads == 0
        assert output.coverage_summary.writes == 0
        assert output.coverage_summary.schema == 0
        assert output.coverage_summary.path_matches == 0
        assert output.coverage_summary.calls == 0
        assert output.coverage_summary.contains == 0
        assert output.coverage_low is False
        assert output.action_required is None
        assert output.action_message is None
        assert output.status == "ok"
        assert output.do_not_finalize is False
        assert output.required_action is None
        assert output.symbol_candidates == []
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
        assert output.required_files == []
        assert output.coverage_low is False
        assert output.action_required is None
        assert output.action_message is None
        assert output.status == "ok"
        assert output.do_not_finalize is False
        assert output.required_action is None
        assert output.symbol_candidates == []


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

    def test_resolution_mismatch_requires_rerun(self):
        """Return action_required when resolved symbol does not match file_path."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        graph_driver = MagicMock()
        symbol_id = "scip-typescript myapp module/User#field:channelId."
        graph_driver.find_symbol_id_by_name.return_value = symbol_id
        graph_driver.get_outgoing_edges.return_value = []
        graph_driver.get_incoming_edges.return_value = []

        symbol_node = MagicMock()
        symbol_node.properties = {
            "name": "channelId",
            "kind": "field",
            "file_path": "/path/to/other.ts",
        }

        def get_node_side_effect(node_id):
            if node_id == symbol_id:
                return symbol_node
            return None

        graph_driver.get_node.side_effect = get_node_side_effect

        tool = ImpactAnalysisTool(graph_driver=graph_driver)
        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_name="channelId",
                file_path="/path/to/expected.ts",
            )
        )

        assert result.action_required == "RESOLUTION_MISMATCH"
        assert result.coverage_low is True
        assert result.affected_files == []
        assert result.status == "blocked"
        assert result.do_not_finalize is True
        assert result.required_action is not None
        assert result.required_action.kind == "RESOLUTION_MISMATCH"

        assert result is not None

    def test_required_files_block_finalize(self):
        """Block finalize when required_files are present."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        graph_driver = MagicMock()
        symbol_id = "scip-typescript myapp module/User#field:channelId."
        graph_driver.find_symbol_id_by_name.return_value = symbol_id

        symbol_node = MagicMock()
        symbol_node.properties = {
            "name": "channelId",
            "kind": "field",
            "file_path": "/path/to/source.ts",
        }

        caller_node = MagicMock()
        caller_node.properties = {"file_path": "/path/to/consumer.ts"}

        write_edge = MagicMock()
        write_edge.edge_type = CodeEdgeType.WRITES
        write_edge.source_id = "caller"
        write_edge.target_id = symbol_id
        write_edge.properties = {}

        def get_node_side_effect(node_id):
            if node_id == symbol_id:
                return symbol_node
            if node_id == "caller":
                return caller_node
            return None

        def get_incoming_edges_side_effect(node_id):
            if node_id == symbol_id:
                return [write_edge]
            return []

        graph_driver.get_node.side_effect = get_node_side_effect
        graph_driver.get_incoming_edges.side_effect = get_incoming_edges_side_effect
        graph_driver.get_outgoing_edges.return_value = []

        tool = ImpactAnalysisTool(graph_driver=graph_driver)
        result = tool.analyze(
            ImpactInput(repo_id="myrepo", symbol_name="channelId")
        )

        assert "/path/to/consumer.ts" in result.required_files
        assert result.action_required == "READ_REQUIRED_FILES"
        assert result.status == "blocked"
        assert result.do_not_finalize is True
        assert result.required_action is not None
        assert result.required_action.kind == "READ_REQUIRED_FILES"

    def test_analyze_file_changes_uses_resolved_file_id(self):
        """Use resolved file id for CONTAINS lookup when file path is relative."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver = MagicMock()
        file_path = "relative/path/module.py"
        file_id = "/abs/path/module.py"

        file_node = MagicMock()
        file_node.id = file_id
        file_node.properties = {"path": file_id}

        contains_edge = MagicMock()
        contains_edge.edge_type = CodeEdgeType.CONTAINS
        contains_edge.target_id = "scip-python myapp module/my_func."

        def get_node_side_effect(node_id):
            if node_id == file_id:
                return file_node
            return None

        def get_outgoing_edges_side_effect(node_id):
            if node_id == file_id:
                return [contains_edge]
            return []

        driver.get_node.side_effect = get_node_side_effect
        driver.find_file_id.return_value = file_id
        driver.get_outgoing_edges.side_effect = get_outgoing_edges_side_effect
        driver.get_incoming_edges.return_value = []

        tool = ImpactAnalysisTool(graph_driver=driver)
        result = tool.analyze(
            ImpactInput(repo_id="myrepo", changed_files=[file_path])
        )

        assert result is not None
        assert any(
            call.args and call.args[0] == file_id
            for call in driver.get_outgoing_edges.call_args_list
        )

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

    def test_analyze_resolves_symbol_name_in_file(self):
        """Test resolving symbol_id from symbol_name and file_path."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver = MagicMock()
        file_id = "/path/to/module.py"
        symbol_id = "scip-python myapp module/MyClass#field:value."

        file_node = MagicMock()
        file_node.id = file_id
        file_node.properties = {"path": file_id}

        symbol_node = MagicMock()
        symbol_node.id = symbol_id
        symbol_node.properties = {
            "name": "value",
            "kind": "field",
            "parent_name": "MyClass",
            "file_path": file_id,
        }

        contains_edge = MagicMock()
        contains_edge.edge_type = CodeEdgeType.CONTAINS
        contains_edge.target_id = symbol_id

        def get_node_side_effect(node_id):
            if node_id == file_id:
                return file_node
            if node_id == symbol_id:
                return symbol_node
            return None

        driver.get_node.side_effect = get_node_side_effect
        driver.get_outgoing_edges.return_value = [contains_edge]
        driver.get_incoming_edges.return_value = []

        tool = ImpactAnalysisTool(graph_driver=driver)
        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_name="value",
                parent_name="MyClass",
                symbol_kind="field",
                file_path=file_id,
            )
        )

        file_paths = {af.file_path for af in result.affected_files}
        assert file_id in file_paths

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

    def test_analyze_includes_field_edges(
        self, mock_graph_driver_with_field_and_schema_edges
    ):
        """Test field READS/WRITES edges are traversed when enabled."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver, field_id = mock_graph_driver_with_field_and_schema_edges
        tool = ImpactAnalysisTool(graph_driver=driver)

        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_id=field_id,
                include_field_edges=True,
            )
        )

        file_paths = {af.file_path for af in result.affected_files}
        assert "/path/to/reader.ts" in file_paths
        assert "/path/to/writer.ts" in file_paths

    def test_analyze_includes_schema_edges(
        self, mock_graph_driver_with_field_and_schema_edges
    ):
        """Test schema edges are traversed when enabled."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver, field_id = mock_graph_driver_with_field_and_schema_edges
        tool = ImpactAnalysisTool(graph_driver=driver)

        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_id=field_id,
                include_schema_edges=True,
            )
        )

        file_paths = {af.file_path for af in result.affected_files}
        assert "/path/to/schema.ts" in file_paths

    def test_analyze_schema_node_resolves_to_code(self):
        """Test schema node analysis follows SCHEMA_EXPOSES back to code."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver = MagicMock()

        field_id = "scip-typescript mypkg module/Producer#field:firstname."
        schema_id = "schema::graphql:/path/to/schema.ts:Producer:firstname"

        schema_node = MagicMock()
        schema_node.properties = {
            "name": "firstname",
            "schema_type": "graphql",
            "file_path": "/path/to/schema.ts",
        }

        field_node = MagicMock()
        field_node.properties = {
            "scip_id": field_id,
            "name": "firstname",
            "kind": "field",
            "file_path": "/path/to/producer.ts",
            "parent_name": "Producer",
        }

        schema_edge = MagicMock()
        schema_edge.source_id = field_id
        schema_edge.target_id = schema_id
        schema_edge.edge_type = CodeEdgeType.SCHEMA_EXPOSES
        schema_edge.properties = {}

        def get_node_side_effect(node_id):
            if node_id == schema_id:
                return schema_node
            if node_id == field_id:
                return field_node
            return None

        driver.get_node.side_effect = get_node_side_effect
        driver.get_incoming_edges.return_value = [schema_edge]
        driver.get_outgoing_edges.return_value = []

        tool = ImpactAnalysisTool(graph_driver=driver)
        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_id=schema_id,
                include_schema_edges=True,
            )
        )

        file_paths = {af.file_path for af in result.affected_files}
        assert "/path/to/producer.ts" in file_paths

    def test_analyze_includes_path_edges(self):
        """Test path literal matches are included when enabled."""
        from openmemory.api.tools.impact_analysis import (
            ImpactAnalysisTool,
            ImpactInput,
        )

        driver = MagicMock()
        field_id = "scip-typescript mypkg module/Producer#field:firstname."

        field_node = MagicMock()
        field_node.properties = {
            "scip_id": field_id,
            "name": "firstname",
            "kind": "field",
            "parent_name": "Producer",
            "file_path": "/path/to/producer.ts",
        }

        path_node = MagicMock()
        path_node.properties = {
            "path": "movie.producers.firstname",
            "normalized_path": "movie.producers.firstname",
            "segments": ["movie", "producers", "firstname"],
            "leaf": "firstname",
            "file_path": "/path/to/form.ts",
            "confidence": "medium",
            "repo_id": "myrepo",
        }

        def get_node_side_effect(node_id):
            if node_id == field_id:
                return field_node
            return None

        def query_nodes_by_type(node_type):
            node_value = node_type.value if hasattr(node_type, "value") else str(node_type)
            if node_value == "CODE_FIELD_PATH":
                return [path_node]
            return []

        driver.get_node.side_effect = get_node_side_effect
        driver.get_incoming_edges.return_value = []
        driver.get_outgoing_edges.return_value = []
        driver.query_nodes_by_type.side_effect = query_nodes_by_type

        tool = ImpactAnalysisTool(graph_driver=driver)
        result = tool.analyze(
            ImpactInput(
                repo_id="myrepo",
                symbol_id=field_id,
                include_path_edges=True,
            )
        )

        file_paths = {af.file_path for af in result.affected_files}
        assert "/path/to/form.ts" in file_paths


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
        assert tool.config.max_depth == 10  # Default

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
