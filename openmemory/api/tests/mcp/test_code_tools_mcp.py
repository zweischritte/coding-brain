"""
Tests for Code Intelligence MCP Tools.

TDD: These tests define the expected behavior for code-intel MCP tools.
All tools should check scopes and return JSON strings.

Scope requirements (from PRD):
- search_code_hybrid: search:read
- find_callers/callees: graph:read
- impact_analysis: graph:read
- explain_code: search:read + graph:read
- adr_automation: search:read + graph:read
- test_generation: search:read + graph:read
- pr_analysis: search:read + graph:read
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import contextvars


class TestMCPCodeToolsRegistration:
    """Test that code-intel MCP tools are properly registered."""

    def test_mcp_server_imports(self):
        """MCP server should import without errors."""
        try:
            from app.mcp_server import mcp
            assert mcp is not None
        except ImportError as e:
            pytest.skip(f"MCP server not available: {e}")

    def test_search_code_hybrid_tool_registered(self):
        """search_code_hybrid tool should be registered."""
        try:
            from app.mcp_server import mcp
            tools = mcp.list_tools()
            tool_names = [t.name for t in tools] if hasattr(tools, '__iter__') else []
            assert "search_code_hybrid" in tool_names, (
                f"search_code_hybrid not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError):
            pytest.skip("MCP server tools not available")

    def test_find_callers_tool_registered(self):
        """find_callers tool should be registered."""
        try:
            from app.mcp_server import mcp
            tools = mcp.list_tools()
            tool_names = [t.name for t in tools] if hasattr(tools, '__iter__') else []
            assert "find_callers" in tool_names, (
                f"find_callers not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError):
            pytest.skip("MCP server tools not available")

    def test_find_callees_tool_registered(self):
        """find_callees tool should be registered."""
        try:
            from app.mcp_server import mcp
            tools = mcp.list_tools()
            tool_names = [t.name for t in tools] if hasattr(tools, '__iter__') else []
            assert "find_callees" in tool_names, (
                f"find_callees not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError):
            pytest.skip("MCP server tools not available")

    def test_impact_analysis_tool_registered(self):
        """impact_analysis tool should be registered."""
        try:
            from app.mcp_server import mcp
            tools = mcp.list_tools()
            tool_names = [t.name for t in tools] if hasattr(tools, '__iter__') else []
            assert "impact_analysis" in tool_names, (
                f"impact_analysis not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError):
            pytest.skip("MCP server tools not available")

    def test_explain_code_tool_registered(self):
        """explain_code tool should be registered."""
        try:
            from app.mcp_server import mcp
            tools = mcp.list_tools()
            tool_names = [t.name for t in tools] if hasattr(tools, '__iter__') else []
            assert "explain_code" in tool_names, (
                f"explain_code not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError):
            pytest.skip("MCP server tools not available")


class TestMCPCodeToolsScopeChecks:
    """Test that MCP code tools enforce scope requirements."""

    @pytest.fixture
    def mock_principal_no_scopes(self):
        """Create a mock principal with no scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=False)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def mock_principal_search_read(self):
        """Create a mock principal with search:read scope."""
        principal = MagicMock()
        principal.has_scope = MagicMock(
            side_effect=lambda s: s in ["search:read"]
        )
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def mock_principal_graph_read(self):
        """Create a mock principal with graph:read scope."""
        principal = MagicMock()
        principal.has_scope = MagicMock(
            side_effect=lambda s: s in ["graph:read"]
        )
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def mock_principal_both_scopes(self):
        """Create a mock principal with both search:read and graph:read scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(
            side_effect=lambda s: s in ["search:read", "graph:read"]
        )
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_search_code_requires_search_read(self, mock_principal_no_scopes):
        """search_code_hybrid should require search:read scope."""
        try:
            from app.mcp_server import search_code_hybrid, principal_var

            # Set principal in context
            token = principal_var.set(mock_principal_no_scopes)
            try:
                result = await search_code_hybrid(query="test")
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
                assert (
                    "scope" in result_data.get("error", "").lower() or
                    "code" in result_data
                )
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_search_code_allows_search_read(self, mock_principal_search_read):
        """search_code_hybrid should work with search:read scope."""
        try:
            from app.mcp_server import search_code_hybrid, principal_var

            # Mock the toolkit to avoid actual search
            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_search_read)
                try:
                    result = await search_code_hybrid(query="test")
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        assert "scope" not in result_data.get("error", "").lower()
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_find_callers_requires_graph_read(self, mock_principal_search_read):
        """find_callers should require graph:read scope (not just search:read)."""
        try:
            from app.mcp_server import find_callers, principal_var

            token = principal_var.set(mock_principal_search_read)
            try:
                result = await find_callers(
                    repo_id="test-repo",
                    symbol_name="main"
                )
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_find_callers_allows_graph_read(self, mock_principal_graph_read):
        """find_callers should work with graph:read scope."""
        try:
            from app.mcp_server import find_callers, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_graph_read)
                try:
                    result = await find_callers(
                        repo_id="test-repo",
                        symbol_name="main"
                    )
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        assert "scope" not in result_data.get("error", "").lower()
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_explain_code_requires_both_scopes(self, mock_principal_search_read):
        """explain_code should require both search:read and graph:read scopes."""
        try:
            from app.mcp_server import explain_code, principal_var

            token = principal_var.set(mock_principal_search_read)
            try:
                result = await explain_code(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should return error for missing graph:read scope
                assert "error" in result_data
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_explain_code_allows_both_scopes(self, mock_principal_both_scopes):
        """explain_code should work with both scopes."""
        try:
            from app.mcp_server import explain_code, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_both_scopes)
                try:
                    result = await explain_code(symbol_id="test-symbol")
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        assert "scope" not in result_data.get("error", "").lower()
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


class TestMCPCodeToolsReturnFormat:
    """Test that MCP code tools return properly formatted JSON strings."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_search_code_returns_json_string(self, mock_principal_all_scopes):
        """search_code_hybrid should return a JSON string."""
        try:
            from app.mcp_server import search_code_hybrid, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["opensearch"]
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await search_code_hybrid(query="test")

                    # Result should be a string
                    assert isinstance(result, str)

                    # Result should be valid JSON
                    parsed = json.loads(result)
                    assert isinstance(parsed, dict)
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_find_callers_returns_json_string(self, mock_principal_all_scopes):
        """find_callers should return a JSON string."""
        try:
            from app.mcp_server import find_callers, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await find_callers(
                        repo_id="test-repo",
                        symbol_name="main"
                    )

                    # Result should be a string
                    assert isinstance(result, str)

                    # Result should be valid JSON
                    parsed = json.loads(result)
                    assert isinstance(parsed, dict)
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


class TestMCPCodeToolsGracefulDegradation:
    """Test that MCP code tools handle unavailable dependencies gracefully."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_search_code_degraded_when_opensearch_down(
        self, mock_principal_all_scopes
    ):
        """search_code_hybrid should return degraded response when OpenSearch unavailable."""
        try:
            from app.mcp_server import search_code_hybrid, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["opensearch"]
                mock_instance.search_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await search_code_hybrid(query="test")
                    result_data = json.loads(result)

                    # Should indicate degraded mode
                    assert (
                        result_data.get("meta", {}).get("degraded_mode", False) or
                        "missing" in str(result_data).lower() or
                        "error" in result_data
                    )
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_find_callers_degraded_when_neo4j_down(
        self, mock_principal_all_scopes
    ):
        """find_callers should return degraded response when Neo4j unavailable."""
        try:
            from app.mcp_server import find_callers, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_instance.callers_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await find_callers(
                        repo_id="test-repo",
                        symbol_name="main"
                    )
                    result_data = json.loads(result)

                    # Should indicate degraded mode
                    assert (
                        result_data.get("meta", {}).get("degraded_mode", False) or
                        "missing" in str(result_data).lower() or
                        "error" in result_data
                    )
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")
