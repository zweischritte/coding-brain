"""
Tests for graph_query MCP tool.
"""

import json
from unittest.mock import MagicMock

import pytest


class TestGraphQueryToolRegistration:
    """Ensure graph_query is registered in MCP server."""

    def test_graph_query_tool_registered(self):
        try:
            from app.mcp_server import mcp
            if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
                tool_names = list(mcp._tool_manager._tools.keys())
                assert "graph_query" in tool_names
        except ImportError as exc:
            pytest.skip(f"MCP server not available: {exc}")


class TestGraphQueryScopeChecks:
    """Validate scope and input enforcement for graph_query."""

    @pytest.fixture
    def principal_no_scopes(self):
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=False)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        principal.can_access = MagicMock(return_value=False)
        return principal

    @pytest.fixture
    def principal_graph_read(self):
        principal = MagicMock()
        principal.has_scope = MagicMock(side_effect=lambda s: s in ["graph:read"])
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        principal.can_access = MagicMock(return_value=True)
        return principal

    @pytest.mark.asyncio
    async def test_graph_query_requires_graph_read(self, principal_no_scopes):
        try:
            from app.mcp_server import graph_query, principal_var

            token = principal_var.set(principal_no_scopes)
            try:
                result = await graph_query(
                    scope="code",
                    query="MATCH (n:CODE_SYMBOL {repo_id: $repo_id}) RETURN n LIMIT 1",
                    repo_id="test-repo",
                )
                result_data = json.loads(result)
                assert result_data.get("code") == "INSUFFICIENT_SCOPE"
            finally:
                principal_var.reset(token)
        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_graph_query_requires_repo_id(self, principal_graph_read):
        try:
            from app.mcp_server import graph_query, principal_var

            token = principal_var.set(principal_graph_read)
            try:
                result = await graph_query(
                    scope="code",
                    query="MATCH (n:CODE_SYMBOL {repo_id: $repo_id}) RETURN n LIMIT 1",
                )
                result_data = json.loads(result)
                assert result_data.get("code") == "MISSING_REPO_ID"
            finally:
                principal_var.reset(token)
        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_graph_query_requires_access_entity(self, principal_graph_read):
        try:
            from app.mcp_server import graph_query, principal_var

            token = principal_var.set(principal_graph_read)
            try:
                result = await graph_query(
                    scope="memory",
                    query="MATCH (m:OM_Memory {accessEntity: $access_entity}) RETURN m LIMIT 1",
                )
                result_data = json.loads(result)
                assert result_data.get("code") == "MISSING_ACCESS_ENTITY"
            finally:
                principal_var.reset(token)
        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_graph_query_rejects_write(self, principal_graph_read):
        try:
            from app.mcp_server import graph_query, principal_var

            token = principal_var.set(principal_graph_read)
            try:
                result = await graph_query(
                    scope="code",
                    query="CREATE (n:CODE_SYMBOL {repo_id: $repo_id}) RETURN n",
                    repo_id="test-repo",
                )
                result_data = json.loads(result)
                assert result_data.get("code") == "READ_ONLY_REQUIRED"
            finally:
                principal_var.reset(token)
        except ImportError:
            pytest.skip("MCP tools not available")
