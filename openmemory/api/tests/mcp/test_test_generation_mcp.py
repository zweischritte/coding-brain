"""
Tests for Test Generation MCP Tool.

TDD: These tests define the expected behavior for the test_generation MCP tool.
The tool should:
1. Check scope (search:read + graph:read required)
2. Use CodeToolkit's test_gen_tool.execute() method
3. Return JSON string with test_cases and meta

Scope requirements (from PRD):
- test_generation: search:read + graph:read
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_principal_no_scopes():
    """Create a mock principal with no scopes."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=False)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def mock_principal_search_read_only():
    """Create a mock principal with only search:read scope."""
    principal = MagicMock()
    principal.has_scope = MagicMock(
        side_effect=lambda s: s in ["search:read"]
    )
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def mock_principal_graph_read_only():
    """Create a mock principal with only graph:read scope."""
    principal = MagicMock()
    principal.has_scope = MagicMock(
        side_effect=lambda s: s in ["graph:read"]
    )
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def mock_principal_both_scopes():
    """Create a mock principal with both search:read and graph:read scopes."""
    principal = MagicMock()
    principal.has_scope = MagicMock(
        side_effect=lambda s: s in ["search:read", "graph:read"]
    )
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def mock_test_suite_result():
    """Create a mock test suite result from execute()."""
    return {
        "symbol_id": "scip-python myapp module/my_function.",
        "symbol_name": "my_function",
        "test_cases": [
            {
                "name": "test_my_function_happy_path",
                "description": "Test my_function with valid inputs",
                "category": "happy_path",
                "code": "def test_my_function_happy_path():\n    result = my_function(1, 2)\n    assert result == 3",
            },
            {
                "name": "test_my_function_edge_case_empty",
                "description": "Test my_function with empty input",
                "category": "edge_case",
                "code": "def test_my_function_edge_case_empty():\n    with pytest.raises(ValueError):\n        my_function(None, None)",
            },
        ],
        "file_content": '"""Tests for my_function."""\n\nimport pytest\n\nclass TestMyFunction:\n    def test_my_function_happy_path(self):\n        pass\n',
    }


@pytest.fixture
def mock_toolkit_with_test_gen(mock_test_suite_result):
    """Create a mock CodeToolkit with test_gen_tool available."""
    toolkit = MagicMock()
    toolkit.is_available = MagicMock(return_value=True)
    toolkit.get_missing_sources = MagicMock(return_value=[])
    toolkit.get_error = MagicMock(return_value=None)

    # Mock test_gen_tool with execute method
    toolkit.test_gen_tool = MagicMock()
    toolkit.test_gen_tool.execute = MagicMock(return_value=mock_test_suite_result)

    return toolkit


@pytest.fixture
def mock_toolkit_without_test_gen():
    """Create a mock CodeToolkit without test_gen_tool."""
    toolkit = MagicMock()
    toolkit.is_available = MagicMock(return_value=True)
    toolkit.get_missing_sources = MagicMock(return_value=["neo4j"])
    toolkit.get_error = MagicMock(return_value="Test generation tool not available")
    toolkit.test_gen_tool = None
    return toolkit


@pytest.fixture
def mock_toolkit_unavailable_backend():
    """Create a mock CodeToolkit with unavailable backend."""
    toolkit = MagicMock()
    toolkit.is_available = MagicMock(return_value=False)
    toolkit.get_missing_sources = MagicMock(return_value=["neo4j", "opensearch"])
    toolkit.get_error = MagicMock(return_value="Backend unavailable")
    toolkit.test_gen_tool = None
    return toolkit


# =============================================================================
# Scope Check Tests
# =============================================================================


class TestTestGenerationScopeCheck:
    """Test that test_generation MCP tool enforces scope requirements."""

    @pytest.mark.asyncio
    async def test_test_generation_requires_search_read(self, mock_principal_graph_read_only):
        """test_generation should require search:read scope."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        token = principal_var.set(mock_principal_graph_read_only)
        try:
            result = await test_generation(symbol_id="test-symbol")
            result_data = json.loads(result)

            # Should return error for missing search:read scope
            assert "error" in result_data
            assert (
                "scope" in result_data.get("error", "").lower() or
                result_data.get("code") == "INSUFFICIENT_SCOPE"
            )
        finally:
            principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_requires_graph_read(self, mock_principal_search_read_only):
        """test_generation should require graph:read scope."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        token = principal_var.set(mock_principal_search_read_only)
        try:
            result = await test_generation(symbol_id="test-symbol")
            result_data = json.loads(result)

            # Should return error for missing graph:read scope
            assert "error" in result_data
            assert (
                "scope" in result_data.get("error", "").lower() or
                result_data.get("code") == "INSUFFICIENT_SCOPE"
            )
        finally:
            principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_scope_check_returns_error_when_no_principal(self):
        """test_generation should return error when no principal is set."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        # Don't set any principal - use default (None)
        token = principal_var.set(None)
        try:
            result = await test_generation(symbol_id="test-symbol")
            result_data = json.loads(result)

            # Should return authentication error
            assert "error" in result_data
            assert (
                "auth" in result_data.get("error", "").lower() or
                result_data.get("code") == "MISSING_AUTH"
            )
        finally:
            principal_var.reset(token)


# =============================================================================
# Successful Execution Tests
# =============================================================================


class TestTestGenerationWithSymbolId:
    """Test test_generation with symbol_id parameter."""

    @pytest.mark.asyncio
    async def test_test_generation_with_symbol_id_returns_test_suite(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should return test suite when given symbol_id."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="scip-python myapp module/my_function.")
                result_data = json.loads(result)

                # Should have test_cases
                assert "test_cases" in result_data or "error" not in result_data

                # Should have meta
                assert "meta" in result_data

                # Meta should have expected fields
                meta = result_data["meta"]
                assert "request_id" in meta
                assert "degraded_mode" in meta
                assert meta["degraded_mode"] is False
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_calls_execute_with_correct_input(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should call test_gen_tool.execute() with correct input."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                await test_generation(
                    symbol_id="scip-python myapp module/my_function.",
                    framework="pytest",
                    include_edge_cases=True,
                    include_error_cases=False,
                )

                # Verify execute was called with expected input
                mock_toolkit_with_test_gen.test_gen_tool.execute.assert_called_once()
                call_args = mock_toolkit_with_test_gen.test_gen_tool.execute.call_args[0][0]

                assert call_args["symbol_id"] == "scip-python myapp module/my_function."
                assert call_args.get("framework") == "pytest"
                assert call_args.get("include_edge_cases") is True
                assert call_args.get("include_error_cases") is False
            finally:
                principal_var.reset(token)


class TestTestGenerationWithFilePath:
    """Test test_generation with file_path parameter."""

    @pytest.mark.asyncio
    async def test_test_generation_with_file_path_returns_test_suite(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should return test suite when given file_path."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(file_path="/path/to/module.py")
                result_data = json.loads(result)

                # Should have test_cases
                assert "test_cases" in result_data or "error" not in result_data

                # Should have meta
                assert "meta" in result_data
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_calls_execute_with_file_path(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should call test_gen_tool.execute() with file_path."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                await test_generation(file_path="/path/to/module.py")

                # Verify execute was called with file_path
                mock_toolkit_with_test_gen.test_gen_tool.execute.assert_called_once()
                call_args = mock_toolkit_with_test_gen.test_gen_tool.execute.call_args[0][0]

                assert call_args.get("file_path") == "/path/to/module.py"
            finally:
                principal_var.reset(token)


# =============================================================================
# Degraded Mode Tests
# =============================================================================


class TestTestGenerationDegradedMode:
    """Test test_generation graceful degradation."""

    @pytest.mark.asyncio
    async def test_test_generation_with_missing_backend(
        self, mock_principal_both_scopes, mock_toolkit_unavailable_backend
    ):
        """test_generation should return degraded response when backend unavailable."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_unavailable_backend):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should indicate degraded mode
                assert (
                    result_data.get("meta", {}).get("degraded_mode", False) or
                    "missing" in str(result_data).lower() or
                    "error" in result_data
                )

                # Should list missing sources
                if "meta" in result_data:
                    meta = result_data["meta"]
                    assert "missing_sources" in meta
                    assert len(meta["missing_sources"]) > 0
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_with_toolkit_unavailable(
        self, mock_principal_both_scopes, mock_toolkit_without_test_gen
    ):
        """test_generation should return error when test_gen_tool is None."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_without_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should indicate degraded mode or have error
                meta = result_data.get("meta", {})
                assert (
                    meta.get("degraded_mode", False) or
                    meta.get("error") is not None or
                    "error" in result_data
                )
            finally:
                principal_var.reset(token)


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestTestGenerationInputValidation:
    """Test test_generation input validation."""

    @pytest.mark.asyncio
    async def test_test_generation_requires_symbol_id_or_file_path(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should return error if neither symbol_id nor file_path provided."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation()
                result_data = json.loads(result)

                # Should return error for missing required input
                assert "error" in result_data
                error_msg = result_data["error"].lower()
                assert "symbol_id" in error_msg or "file_path" in error_msg or "required" in error_msg
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_accepts_both_symbol_id_and_file_path(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should accept both symbol_id and file_path (symbol_id takes precedence)."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(
                    symbol_id="scip-python myapp module/func.",
                    file_path="/path/to/module.py"
                )
                result_data = json.loads(result)

                # Should succeed (not return an error about conflicting params)
                # When both are provided, symbol_id should take precedence
                assert "error" not in result_data or "symbol_id" not in result_data.get("error", "").lower()
            finally:
                principal_var.reset(token)


# =============================================================================
# Response Structure Tests
# =============================================================================


class TestTestGenerationResponseStructure:
    """Test test_generation response structure."""

    @pytest.mark.asyncio
    async def test_test_generation_returns_json_string(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation should return a valid JSON string."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")

                # Result should be a string
                assert isinstance(result, str)

                # Result should be valid JSON
                parsed = json.loads(result)
                assert isinstance(parsed, dict)
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_response_has_meta(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation response should have meta field."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should have meta field
                assert "meta" in result_data

                meta = result_data["meta"]

                # Meta should have required fields
                assert "request_id" in meta
                assert "degraded_mode" in meta
                assert "missing_sources" in meta

                # request_id should be a valid UUID string
                import uuid
                try:
                    uuid.UUID(meta["request_id"])
                except ValueError:
                    pytest.fail(f"request_id is not a valid UUID: {meta['request_id']}")
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_response_has_test_cases(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation response should have test_cases field."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should have test_cases field
                assert "test_cases" in result_data

                test_cases = result_data["test_cases"]
                assert isinstance(test_cases, list)

                # Each test case should have expected structure
                if len(test_cases) > 0:
                    test_case = test_cases[0]
                    assert "name" in test_case
                    assert "description" in test_case
                    assert "category" in test_case
                    assert "code" in test_case
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_response_has_file_content(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation response should have file_content field with rendered test file."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should have file_content field
                assert "file_content" in result_data

                file_content = result_data["file_content"]
                assert isinstance(file_content, str)

                # file_content should look like a test file
                assert "test" in file_content.lower() or "def " in file_content
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_response_has_symbol_info(
        self, mock_principal_both_scopes, mock_toolkit_with_test_gen
    ):
        """test_generation response should have symbol_id and symbol_name."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit_with_test_gen):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should have symbol info
                assert "symbol_id" in result_data
                assert "symbol_name" in result_data
            finally:
                principal_var.reset(token)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTestGenerationErrorHandling:
    """Test test_generation error handling."""

    @pytest.mark.asyncio
    async def test_test_generation_handles_execute_exception(
        self, mock_principal_both_scopes
    ):
        """test_generation should handle exceptions from test_gen_tool.execute()."""
        try:
            from app.mcp_server import test_generation, principal_var
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        # Create toolkit that raises exception
        mock_toolkit = MagicMock()
        mock_toolkit.is_available = MagicMock(return_value=True)
        mock_toolkit.get_missing_sources = MagicMock(return_value=[])
        mock_toolkit.get_error = MagicMock(return_value=None)
        mock_toolkit.test_gen_tool = MagicMock()
        mock_toolkit.test_gen_tool.execute = MagicMock(
            side_effect=Exception("Unexpected error during test generation")
        )

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="test-symbol")
                result_data = json.loads(result)

                # Should return degraded response with error
                assert (
                    result_data.get("meta", {}).get("degraded_mode", False) or
                    result_data.get("meta", {}).get("error") is not None or
                    "error" in result_data
                )

                # Error message should be present somewhere
                error_str = str(result_data)
                assert "error" in error_str.lower() or "unexpected" in error_str.lower()
            finally:
                principal_var.reset(token)

    @pytest.mark.asyncio
    async def test_test_generation_handles_symbol_not_found(
        self, mock_principal_both_scopes
    ):
        """test_generation should handle SymbolNotFoundError gracefully."""
        try:
            from app.mcp_server import test_generation, principal_var
            from tools.test_generation import SymbolNotFoundError
        except ImportError:
            pytest.skip("test_generation MCP tool not implemented yet")

        # Create toolkit that raises SymbolNotFoundError
        mock_toolkit = MagicMock()
        mock_toolkit.is_available = MagicMock(return_value=True)
        mock_toolkit.get_missing_sources = MagicMock(return_value=[])
        mock_toolkit.get_error = MagicMock(return_value=None)
        mock_toolkit.test_gen_tool = MagicMock()
        mock_toolkit.test_gen_tool.execute = MagicMock(
            side_effect=SymbolNotFoundError("Symbol not found: unknown-symbol")
        )

        with patch("app.mcp_server.get_code_toolkit", return_value=mock_toolkit):
            token = principal_var.set(mock_principal_both_scopes)
            try:
                result = await test_generation(symbol_id="unknown-symbol")
                result_data = json.loads(result)

                # Should return degraded response or error
                assert (
                    result_data.get("meta", {}).get("degraded_mode", False) or
                    result_data.get("meta", {}).get("error") is not None or
                    "error" in result_data
                )
            finally:
                principal_var.reset(token)


# =============================================================================
# MCP Tool Registration Tests
# =============================================================================


class TestTestGenerationMCPRegistration:
    """Test that test_generation MCP tool is properly registered."""

    def test_test_generation_tool_registered(self):
        """test_generation tool should be registered in MCP server."""
        try:
            from app.mcp_server import mcp
        except ImportError:
            pytest.skip("MCP server not available")

        # Access internal tool manager registry
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            tool_names = list(mcp._tool_manager._tools.keys())
            assert "test_generation" in tool_names, (
                f"test_generation not found in tools: {tool_names}"
            )
        else:
            pytest.skip("Cannot access MCP tool registry")

    def test_test_generation_tool_has_description(self):
        """test_generation tool should have a description."""
        try:
            from app.mcp_server import mcp
        except ImportError:
            pytest.skip("MCP server not available")

        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            tools = mcp._tool_manager._tools
            if "test_generation" in tools:
                tool = tools["test_generation"]
                # Tool should have description attribute
                assert hasattr(tool, 'description') or hasattr(tool, '__doc__')
            else:
                pytest.skip("test_generation tool not registered")
        else:
            pytest.skip("Cannot access MCP tool registry")
