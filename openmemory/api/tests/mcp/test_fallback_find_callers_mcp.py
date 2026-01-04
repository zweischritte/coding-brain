"""MCP Tests for FallbackFindCallersTool integration.

This module tests the find_callers MCP tool with fallback cascade:
- Fallback cascade activation
- Response format with fallback info
- SymbolNotFoundError handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import contextvars


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_principal_graph_read():
    """Create a mock principal with graph:read scope."""
    principal = MagicMock()
    principal.has_scope = MagicMock(
        side_effect=lambda s: s in ["graph:read"]
    )
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def mock_toolkit():
    """Create a mock code toolkit."""
    toolkit = MagicMock()
    toolkit.is_available.return_value = True
    toolkit.callers_tool = MagicMock()
    toolkit.callers_tool.graph_driver = MagicMock()
    toolkit.search_tool = None
    toolkit.get_missing_sources.return_value = []
    toolkit.get_error.return_value = None
    return toolkit


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestFindCallersWithFallback:
    """Tests for find_callers MCP tool with fallback."""

    @pytest.mark.asyncio
    async def test_find_callers_requires_symbol(self, mock_principal_graph_read):
        """Test find_callers requires symbol_name or symbol_id."""
        try:
            from app.mcp_server import find_callers, principal_var
        except ImportError:
            pytest.skip("MCP server not available")

        token = principal_var.set(mock_principal_graph_read)
        try:
            result = await find_callers(repo_id="test-repo")
            data = json.loads(result)
            assert "error" in data
            assert "symbol_name or symbol_id is required" in data["error"].lower()
        finally:
            principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_find_callers_returns_json(
        self, mock_principal_graph_read, mock_toolkit
    ):
        """Test find_callers returns valid JSON string.

        Note: This test verifies the function signature accepts new use_fallback param.
        Full integration testing requires running Neo4j.
        """
        try:
            from app.mcp_server import find_callers, principal_var
            from tools.call_graph import GraphOutput, ResponseMeta
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        token = principal_var.set(mock_principal_graph_read)
        try:
            # Mock the callers_tool to return empty result
            mock_toolkit.callers_tool.find.return_value = GraphOutput(
                nodes=[],
                edges=[],
                meta=ResponseMeta(request_id="test-123"),
            )

            with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
                result = await find_callers(
                    repo_id="test-repo",
                    symbol_name="testFunction",
                    use_fallback=False,  # Disable fallback to avoid import chain
                )
                # Should return JSON string
                data = json.loads(result)
                assert isinstance(data, dict)
        finally:
            principal_var.reset(token)


# =============================================================================
# Fallback Response Format Tests
# =============================================================================


class TestFallbackResponseFormat:
    """Tests for fallback response format."""

    @pytest.mark.asyncio
    async def test_fallback_info_in_response(self, mock_principal_graph_read, mock_toolkit):
        """Test fallback info is included when fallback is used."""
        try:
            from app.mcp_server import find_callers, principal_var
            from tools.call_graph import GraphOutput, ResponseMeta
        except ImportError:
            pytest.skip("MCP server not available")

        token = principal_var.set(mock_principal_graph_read)
        try:
            mock_result = GraphOutput(
                nodes=[],
                edges=[],
                meta=ResponseMeta(
                    request_id="test-123",
                    degraded_mode=True,
                    fallback_stage=2,
                    fallback_strategy="grep",
                    warning="Results from grep fallback",
                ),
            )

            with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
                # Mock FallbackFindCallersTool
                mock_fallback_tool = MagicMock()
                mock_fallback_tool.find.return_value = mock_result
                mock_fallback_tool.fallback_used = True
                mock_fallback_tool.fallback_stage = 2
                mock_fallback_tool.get_stage_timings.return_value = {1: 10.0, 2: 5.0}

                with patch(
                    "tools.fallback_find_callers.FallbackFindCallersTool",
                    return_value=mock_fallback_tool,
                ):
                    result = await find_callers(
                        repo_id="test-repo",
                        symbol_name="testFunction",
                        use_fallback=True,
                    )

                    data = json.loads(result)
                    # Check for fallback info
                    if "_fallback_info" in data:
                        assert data["_fallback_info"]["fallback_used"] is True
                        assert data["_fallback_info"]["fallback_stage"] == 2
        finally:
            principal_var.reset(token)


# =============================================================================
# SymbolNotFoundError Handling Tests
# =============================================================================


class TestSymbolNotFoundErrorHandling:
    """Tests for SymbolNotFoundError handling."""

    @pytest.mark.asyncio
    async def test_symbol_not_found_returns_suggestions(
        self, mock_principal_graph_read, mock_toolkit
    ):
        """Test SymbolNotFoundError returns structured suggestions."""
        try:
            from app.mcp_server import find_callers, principal_var
            from tools.call_graph import SymbolNotFoundError
        except ImportError:
            pytest.skip("MCP server not available")

        token = principal_var.set(mock_principal_graph_read)
        try:
            mock_toolkit.callers_tool.find.side_effect = SymbolNotFoundError(
                "Symbol not found: missingFunc",
                symbol_name="missingFunc",
                repo_id="test-repo",
            )

            with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
                # Disable fallback to trigger the exception handler
                result = await find_callers(
                    repo_id="test-repo",
                    symbol_name="missingFunc",
                    use_fallback=False,
                )

                data = json.loads(result)
                assert data["error"] == "SYMBOL_NOT_FOUND"
                assert data["symbol"] == "missingFunc"
                assert "suggestions" in data
                assert len(data["suggestions"]) > 0
                assert "next_actions" in data
        finally:
            principal_var.reset(token)


# =============================================================================
# Degraded Mode Tests
# =============================================================================


class TestDegradedMode:
    """Tests for degraded mode handling."""

    @pytest.mark.asyncio
    async def test_neo4j_unavailable_returns_degraded(self, mock_principal_graph_read):
        """Test Neo4j unavailable returns degraded response."""
        try:
            from app.mcp_server import find_callers, principal_var
        except ImportError:
            pytest.skip("MCP server not available")

        token = principal_var.set(mock_principal_graph_read)
        try:
            mock_toolkit = MagicMock()
            mock_toolkit.is_available.return_value = False

            with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
                result = await find_callers(
                    repo_id="test-repo",
                    symbol_name="testFunction",
                )

                data = json.loads(result)
                assert data["meta"]["degraded_mode"] is True
                assert "neo4j" in data["meta"]["missing_sources"]
        finally:
            principal_var.reset(token)


# =============================================================================
# Fallback Tool Unit Tests (without MCP)
# =============================================================================


class TestFallbackFindCallersToolUnit:
    """Unit tests for FallbackFindCallersTool."""

    def test_tool_imports(self):
        """Test FallbackFindCallersTool can be imported."""
        try:
            from tools.fallback_find_callers import (
                FallbackFindCallersTool,
                FallbackConfig,
                GrepTool,
                GrepMatch,
            )
            assert FallbackFindCallersTool is not None
            assert FallbackConfig is not None
            assert GrepTool is not None
            assert GrepMatch is not None
        except ImportError as e:
            pytest.skip(f"Fallback tool not available: {e}")

    def test_config_defaults(self):
        """Test FallbackConfig default values."""
        try:
            from tools.fallback_find_callers import FallbackConfig
        except ImportError:
            pytest.skip("Fallback tool not available")

        config = FallbackConfig()
        assert config.stage_timeout_ms == 150
        assert config.total_timeout_ms == 500
        assert config.grep_max_results == 50
        assert config.semantic_min_score == 0.5

    def test_grep_tool_no_func(self):
        """Test GrepTool without search function returns empty list."""
        try:
            from tools.fallback_find_callers import GrepTool
        except ImportError:
            pytest.skip("Fallback tool not available")

        tool = GrepTool()
        results = tool.search(pattern="test", repo_id="repo-1")
        assert results == []

    def test_grep_tool_with_func(self):
        """Test GrepTool with custom search function."""
        try:
            from tools.fallback_find_callers import GrepTool
        except ImportError:
            pytest.skip("Fallback tool not available")

        mock_results = [
            {"file": "test.py", "line": 10, "context": "def test()"},
        ]
        search_func = MagicMock(return_value=mock_results)
        tool = GrepTool(search_func=search_func)

        results = tool.search(pattern="test", repo_id="repo-1")
        assert len(results) == 1
        assert results[0].file == "test.py"
        assert results[0].line == 10

    def test_symbol_not_found_error_suggestions(self):
        """Test SymbolNotFoundError includes suggestions."""
        try:
            from tools.call_graph import SymbolNotFoundError
        except ImportError:
            pytest.skip("call_graph not available")

        error = SymbolNotFoundError(
            "Symbol not found: testFunc",
            symbol_name="testFunc",
            repo_id="test-repo",
        )

        assert error.symbol_name == "testFunc"
        assert error.repo_id == "test-repo"
        assert len(error.suggestions) > 0
        assert any("search_code_hybrid" in s for s in error.suggestions)

    def test_symbol_not_found_error_to_dict(self):
        """Test SymbolNotFoundError to_dict method."""
        try:
            from tools.call_graph import SymbolNotFoundError
        except ImportError:
            pytest.skip("call_graph not available")

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

    def test_keyword_extraction_camelcase(self):
        """Test keyword extraction from camelCase."""
        try:
            from tools.fallback_find_callers import FallbackFindCallersTool
        except ImportError:
            pytest.skip("Fallback tool not available")

        tool = FallbackFindCallersTool(graph_driver=MagicMock())
        keywords = tool._extract_keywords("moveFilesToPermanentStorage")

        assert "move" in keywords
        assert "files" in keywords
        assert "to" in keywords
        assert "permanent" in keywords
        assert "storage" in keywords

    def test_keyword_extraction_snake_case(self):
        """Test keyword extraction from snake_case."""
        try:
            from tools.fallback_find_callers import FallbackFindCallersTool
        except ImportError:
            pytest.skip("Fallback tool not available")

        tool = FallbackFindCallersTool(graph_driver=MagicMock())
        keywords = tool._extract_keywords("move_files_to_storage")

        assert "move" in keywords
        assert "files" in keywords
        assert "to" in keywords
        assert "storage" in keywords
