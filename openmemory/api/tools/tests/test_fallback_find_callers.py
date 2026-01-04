"""Tests for FallbackFindCallersTool.

This module tests the fallback cascade for find_callers:
- Stage 1: Graph-based search
- Stage 2: Grep fallback
- Stage 3: Semantic search fallback
- Stage 4: Structured error response
"""

import pytest
from unittest.mock import MagicMock, patch
import time

# Use relative imports that work in both contexts
try:
    from openmemory.api.tools.call_graph import (
        CallGraphConfig,
        CallGraphInput,
        GraphOutput,
        GraphNode,
        GraphEdge,
        ResponseMeta,
        SymbolNotFoundError,
        CallGraphError,
    )
    from openmemory.api.tools.fallback_find_callers import (
        FallbackFindCallersTool,
        FallbackConfig,
        GrepTool,
        GrepMatch,
        create_fallback_find_callers_tool,
    )
except ImportError:
    from tools.call_graph import (
        CallGraphConfig,
        CallGraphInput,
        GraphOutput,
        GraphNode,
        GraphEdge,
        ResponseMeta,
        SymbolNotFoundError,
        CallGraphError,
    )
    from tools.fallback_find_callers import (
        FallbackFindCallersTool,
        FallbackConfig,
        GrepTool,
        GrepMatch,
        create_fallback_find_callers_tool,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()
    return driver


@pytest.fixture
def mock_grep_tool():
    """Create a mock grep tool."""
    tool = MagicMock(spec=GrepTool)
    return tool


@pytest.fixture
def mock_search_tool():
    """Create a mock search code hybrid tool."""
    tool = MagicMock()
    return tool


@pytest.fixture
def sample_input():
    """Create sample input data."""
    return CallGraphInput(
        repo_id="test-repo",
        symbol_name="testFunction",
        depth=2,
    )


@pytest.fixture
def sample_graph_output():
    """Create sample graph output with nodes."""
    return GraphOutput(
        nodes=[
            GraphNode(
                id="caller-1",
                type="CODE_SYMBOL",
                properties={"name": "callerFunction"},
            ),
        ],
        edges=[
            GraphEdge(
                from_id="caller-1",
                to_id="target",
                type="CALLS",
            ),
        ],
        meta=ResponseMeta(request_id="test-123"),
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestFallbackConfig:
    """Tests for FallbackConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FallbackConfig()
        assert config.stage_timeout_ms == 150
        assert config.total_timeout_ms == 500
        assert config.grep_max_results == 50
        assert config.semantic_min_score == 0.5
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_reset_s == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FallbackConfig(
            stage_timeout_ms=200,
            total_timeout_ms=1000,
            grep_max_results=100,
        )
        assert config.stage_timeout_ms == 200
        assert config.total_timeout_ms == 1000
        assert config.grep_max_results == 100


# =============================================================================
# GrepTool Tests
# =============================================================================


class TestGrepTool:
    """Tests for GrepTool class."""

    def test_search_with_custom_func(self):
        """Test grep search with custom function."""
        mock_results = [
            {"file": "test.py", "line": 10, "context": "def test()"},
            {"file": "test.py", "line": 20, "context": "call test()"},
        ]
        search_func = MagicMock(return_value=mock_results)
        tool = GrepTool(search_func=search_func)

        results = tool.search(
            pattern="test",
            repo_id="repo-1",
            include_patterns=["*.py"],
        )

        assert len(results) == 2
        assert results[0].file == "test.py"
        assert results[0].line == 10
        search_func.assert_called_once()

    def test_search_without_func_returns_empty(self):
        """Test grep search without custom function returns empty list."""
        tool = GrepTool()
        results = tool.search(pattern="test", repo_id="repo-1")
        assert results == []

    def test_search_handles_exception(self):
        """Test grep search handles exceptions gracefully."""
        search_func = MagicMock(side_effect=Exception("Search failed"))
        tool = GrepTool(search_func=search_func)

        results = tool.search(pattern="test", repo_id="repo-1")
        assert results == []


# =============================================================================
# FallbackFindCallersTool Initialization Tests
# =============================================================================


class TestFallbackFindCallersToolInit:
    """Tests for FallbackFindCallersTool initialization."""

    def test_default_initialization(self, mock_graph_driver):
        """Test default initialization."""
        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        assert tool.graph_driver == mock_graph_driver
        assert tool.grep_tool is not None
        assert tool.search_tool is None
        assert tool.fallback_used is False
        assert tool.fallback_stage is None

    def test_full_initialization(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool
    ):
        """Test initialization with all dependencies."""
        config = FallbackConfig(stage_timeout_ms=100)

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
            config=config,
        )

        assert tool.grep_tool == mock_grep_tool
        assert tool.search_tool == mock_search_tool
        assert tool.config.stage_timeout_ms == 100


# =============================================================================
# Stage 1: Graph Search Tests
# =============================================================================


class TestStage1GraphSearch:
    """Tests for Stage 1: Graph-based search."""

    def test_graph_search_success(
        self, mock_graph_driver, sample_input, sample_graph_output
    ):
        """Test successful graph search returns immediately."""
        # Setup mock to return valid result
        mock_graph_driver.get_node.return_value = MagicMock(
            properties={"name": "testFunction"}
        )
        mock_graph_driver.get_incoming_edges.return_value = [
            MagicMock(
                source_id="caller-1",
                target_id="target",
                edge_type=MagicMock(value="CALLS"),
                properties={},
            )
        ]

        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        with patch.object(tool, "_graph_search", return_value=sample_graph_output):
            result = tool.find(sample_input)

        assert tool.fallback_used is False
        assert tool.fallback_stage is None
        assert len(result.nodes) > 0

    def test_graph_search_symbol_not_found_triggers_fallback(
        self, mock_graph_driver, mock_grep_tool, sample_input
    ):
        """Test SymbolNotFoundError triggers fallback cascade."""
        # Setup grep to return results
        mock_grep_tool.search.return_value = [
            GrepMatch(file="test.py", line=10, context="testFunction()"),
        ]

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
        )

        # Mock graph search to raise SymbolNotFoundError
        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Symbol not found: testFunction"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_used is True
        assert tool.fallback_stage == 2  # Grep fallback
        assert result.meta.degraded_mode is True
        assert result.meta.fallback_strategy == "grep"


# =============================================================================
# Stage 2: Grep Fallback Tests
# =============================================================================


class TestStage2GrepFallback:
    """Tests for Stage 2: Grep fallback."""

    def test_grep_fallback_success(
        self, mock_graph_driver, mock_grep_tool, sample_input
    ):
        """Test grep fallback returns grep-based results."""
        mock_grep_tool.search.return_value = [
            GrepMatch(file="main.py", line=42, context="testFunction(args)"),
            GrepMatch(file="util.py", line=100, context="result = testFunction()"),
        ]

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 2
        assert len(result.nodes) == 2
        assert result.nodes[0].type == "grep_match"
        assert "main.py" in result.nodes[0].id
        assert result.meta.fallback_strategy == "grep"
        assert "false positives" in result.meta.warning

    def test_grep_fallback_too_many_results_triggers_semantic(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool, sample_input
    ):
        """Test too many grep results triggers semantic search."""
        # Return more than grep_max_results
        mock_grep_tool.search.return_value = [
            GrepMatch(file=f"file{i}.py", line=i, context="match")
            for i in range(60)
        ]

        # Setup semantic search to return results
        mock_search_result = MagicMock()
        mock_search_result.results = [
            MagicMock(
                symbol=MagicMock(
                    symbol_id="sym-1",
                    symbol_name="testFunction",
                    file_path="src/test.py",
                ),
                score=0.8,
            )
        ]
        mock_search_tool.search.return_value = mock_search_result

        config = FallbackConfig(grep_max_results=50)
        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
            config=config,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 3  # Semantic search
        assert result.meta.fallback_strategy == "semantic_search"

    def test_grep_fallback_no_results_triggers_semantic(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool, sample_input
    ):
        """Test empty grep results trigger semantic search."""
        mock_grep_tool.search.return_value = []

        mock_search_result = MagicMock()
        mock_search_result.results = [
            MagicMock(
                symbol=MagicMock(
                    symbol_id="sym-1",
                    symbol_name="testFunction",
                    file_path="src/test.py",
                ),
                score=0.7,
            )
        ]
        mock_search_tool.search.return_value = mock_search_result

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 3


# =============================================================================
# Stage 3: Semantic Search Fallback Tests
# =============================================================================


class TestStage3SemanticFallback:
    """Tests for Stage 3: Semantic search fallback."""

    def test_semantic_fallback_success(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool, sample_input
    ):
        """Test semantic search fallback returns semantic-based results."""
        mock_grep_tool.search.return_value = []

        mock_search_result = MagicMock()
        mock_search_result.results = [
            MagicMock(
                symbol=MagicMock(
                    symbol_id="sym-1",
                    symbol_name="testFunction",
                    file_path="src/handlers.py",
                ),
                score=0.85,
            ),
            MagicMock(
                symbol=MagicMock(
                    symbol_id="sym-2",
                    symbol_name="testHelper",
                    file_path="src/utils.py",
                ),
                score=0.6,
            ),
        ]
        mock_search_tool.search.return_value = mock_search_result

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 3
        assert len(result.nodes) == 2
        assert result.nodes[0].type == "semantic_match"
        assert result.meta.fallback_strategy == "semantic_search"

    def test_semantic_fallback_filters_low_score(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool, sample_input
    ):
        """Test semantic search filters results below threshold."""
        mock_grep_tool.search.return_value = []

        mock_search_result = MagicMock()
        mock_search_result.results = [
            MagicMock(
                symbol=MagicMock(symbol_id="sym-1", symbol_name="test", file_path="a.py"),
                score=0.8,  # Above threshold
            ),
            MagicMock(
                symbol=MagicMock(symbol_id="sym-2", symbol_name="other", file_path="b.py"),
                score=0.3,  # Below threshold
            ),
        ]
        mock_search_tool.search.return_value = mock_search_result

        config = FallbackConfig(semantic_min_score=0.5)
        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
            config=config,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 3
        assert len(result.nodes) == 1  # Only high-score result


# =============================================================================
# Stage 4: Structured Error Tests
# =============================================================================


class TestStage4StructuredError:
    """Tests for Stage 4: Structured error response."""

    def test_structured_error_when_all_fallbacks_fail(
        self, mock_graph_driver, mock_grep_tool, sample_input
    ):
        """Test structured error response when all fallbacks fail."""
        mock_grep_tool.search.return_value = []

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=None,  # No semantic search
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert tool.fallback_stage == 4
        assert result.nodes == []
        assert result.edges == []
        assert result.meta.degraded_mode is True
        assert result.meta.fallback_strategy == "structured_error"
        assert len(result.suggestions) > 0
        assert any("grep" in s for s in result.suggestions)
        assert any("search_code_hybrid" in s for s in result.suggestions)
        assert any("decorator" in s.lower() for s in result.suggestions)

    def test_structured_error_includes_missing_sources(
        self, mock_graph_driver, sample_input
    ):
        """Test structured error includes all missing sources."""
        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=GrepTool(),
            search_tool=None,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            result = tool.find(sample_input)

        assert "graph_index" in result.meta.missing_sources
        assert "grep" in result.meta.missing_sources
        assert "semantic_search" in result.meta.missing_sources


# =============================================================================
# Keyword Extraction Tests
# =============================================================================


class TestKeywordExtraction:
    """Tests for keyword extraction from symbol names."""

    def test_extract_camelcase_keywords(self, mock_graph_driver):
        """Test extraction of keywords from camelCase."""
        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        keywords = tool._extract_keywords("moveFilesToPermanentStorage")
        assert "move" in keywords
        assert "files" in keywords
        assert "to" in keywords
        assert "permanent" in keywords
        assert "storage" in keywords

    def test_extract_snake_case_keywords(self, mock_graph_driver):
        """Test extraction of keywords from snake_case."""
        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        keywords = tool._extract_keywords("move_files_to_storage")
        assert "move" in keywords
        assert "files" in keywords
        assert "to" in keywords
        assert "storage" in keywords

    def test_extract_mixed_case_keywords(self, mock_graph_driver):
        """Test extraction of keywords from mixed case."""
        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        keywords = tool._extract_keywords("get_UserProfile")
        assert "get" in keywords
        assert "user" in keywords
        assert "profile" in keywords


# =============================================================================
# Timing and Performance Tests
# =============================================================================


class TestTimingAndPerformance:
    """Tests for timing and performance tracking."""

    def test_stage_timings_recorded(
        self, mock_graph_driver, mock_grep_tool, sample_input
    ):
        """Test that stage timings are recorded."""
        mock_grep_tool.search.return_value = [
            GrepMatch(file="test.py", line=1, context="match")
        ]

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
        )

        with patch.object(
            tool,
            "_graph_search",
            side_effect=SymbolNotFoundError("Not found"),
        ):
            tool.find(sample_input)

        timings = tool.get_stage_timings()
        assert 1 in timings
        assert 2 in timings
        assert timings[1] >= 0
        assert timings[2] >= 0

    def test_timeout_response(self, mock_graph_driver, sample_input):
        """Test timeout response when cascade takes too long."""
        config = FallbackConfig(total_timeout_ms=1)  # Very short timeout

        tool = FallbackFindCallersTool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        # Mock graph search to take time
        def slow_search(*args, **kwargs):
            time.sleep(0.1)  # 100ms > 1ms timeout
            raise SymbolNotFoundError("Not found")

        with patch.object(tool, "_graph_search", side_effect=slow_search):
            result = tool.find(sample_input)

        assert result.meta.fallback_strategy == "timeout"
        assert "timed out" in result.meta.warning.lower()


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_created_per_repo(self, mock_graph_driver):
        """Test circuit breaker is created per repository."""
        tool = FallbackFindCallersTool(graph_driver=mock_graph_driver)

        breaker1 = tool._get_circuit_breaker("repo-1")
        breaker2 = tool._get_circuit_breaker("repo-2")
        breaker1_again = tool._get_circuit_breaker("repo-1")

        assert breaker1 is breaker1_again
        assert breaker1 is not breaker2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for create_fallback_find_callers_tool factory."""

    def test_factory_creates_tool(self, mock_graph_driver):
        """Test factory creates configured tool."""
        config = FallbackConfig(grep_max_results=25)

        tool = create_fallback_find_callers_tool(
            graph_driver=mock_graph_driver,
            config=config,
        )

        assert isinstance(tool, FallbackFindCallersTool)
        assert tool.config.grep_max_results == 25

    def test_factory_with_all_options(
        self, mock_graph_driver, mock_grep_tool, mock_search_tool
    ):
        """Test factory with all options."""
        tool = create_fallback_find_callers_tool(
            graph_driver=mock_graph_driver,
            grep_tool=mock_grep_tool,
            search_tool=mock_search_tool,
        )

        assert tool.grep_tool == mock_grep_tool
        assert tool.search_tool == mock_search_tool


# =============================================================================
# SymbolNotFoundError Enhancement Tests
# =============================================================================


class TestSymbolNotFoundErrorEnhancement:
    """Tests for enhanced SymbolNotFoundError."""

    def test_error_includes_suggestions(self):
        """Test error includes fallback suggestions."""
        error = SymbolNotFoundError(
            "Symbol not found: testFunction",
            symbol_name="testFunction",
            repo_id="test-repo",
        )

        assert error.symbol_name == "testFunction"
        assert error.repo_id == "test-repo"
        assert len(error.suggestions) > 0
        assert any("search_code_hybrid" in s for s in error.suggestions)

    def test_error_to_dict(self):
        """Test error converts to dict for MCP response."""
        error = SymbolNotFoundError(
            "Symbol not found: myFunc",
            symbol_name="myFunc",
        )

        result = error.to_dict()

        assert result["error"] == "SYMBOL_NOT_FOUND"
        assert result["symbol"] == "myFunc"
        assert "suggestions" in result
        assert "next_actions" in result
        assert len(result["next_actions"]) > 0

    def test_error_extracts_symbol_from_message(self):
        """Test error extracts symbol name from message if not provided."""
        error = SymbolNotFoundError("Symbol not found: extractedName")

        assert error.symbol_name == "extractedName"
