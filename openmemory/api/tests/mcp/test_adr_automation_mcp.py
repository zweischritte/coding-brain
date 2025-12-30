"""
Tests for ADR Automation MCP Tool.

TDD: These tests define the expected behavior for the adr_automation MCP tool.
The tool should check scopes and return JSON strings with proper response structure.

Scope requirements (from PRD):
- adr_automation: search:read + graph:read

Test scenarios:
1. test_adr_automation_scope_check - Returns error if scope missing
2. test_adr_automation_with_valid_input - Returns ADR analysis with meta
3. test_adr_automation_with_missing_backend - Returns degraded_mode response
4. test_adr_automation_with_toolkit_unavailable - Returns error when toolkit has no adr_tool
5. test_adr_automation_input_validation - Validates required inputs (changes list)
6. test_adr_automation_response_structure - Response has correct JSON structure with meta
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Any, Optional


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
def mock_principal_all_scopes():
    """Create a mock principal with all required scopes."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=True)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    return principal


@pytest.fixture
def sample_changes():
    """Sample changes list for testing ADR detection."""
    return [
        {
            "file_path": "requirements.txt",
            "change_type": "modified",
            "diff": "+redis>=4.0.0",
            "added_lines": ["redis>=4.0.0"],
            "removed_lines": [],
        }
    ]


@pytest.fixture
def sample_changes_api():
    """Sample API changes for testing ADR detection."""
    return [
        {
            "file_path": "app/routers/users.py",
            "change_type": "modified",
            "diff": "+@router.post('/api/v2/users')",
            "added_lines": ["@router.post('/api/v2/users')"],
            "removed_lines": ["@router.post('/api/v1/users')"],
        }
    ]


@pytest.fixture
def sample_changes_no_adr():
    """Sample changes that should not trigger ADR."""
    return [
        {
            "file_path": "README.md",
            "change_type": "modified",
            "diff": "+Updated documentation",
            "added_lines": ["Updated documentation"],
            "removed_lines": [],
        }
    ]


@pytest.fixture
def mock_adr_tool_result():
    """Mock ADR tool result for successful analysis."""
    return {
        "should_create_adr": True,
        "confidence": 0.85,
        "triggered_heuristics": ["dependency"],
        "reasons": ["Significant new dependencies added: redis"],
        "generated_adr": {
            "title": "Add New Dependency: redis",
            "status": "Proposed",
            "context": "This decision addresses changes to the following files...",
            "decision": "We will add a new dependency to the project...",
            "consequences": [
                "New dependency adds functionality",
                "Additional maintenance burden for dependency updates",
            ],
            "markdown": "# ADR: Add New Dependency...",
        },
        "code_links": [
            {
                "file_path": "requirements.txt",
                "change_type": "modified",
                "line_start": 0,
                "line_end": 0,
            }
        ],
        "related_adrs": [],
        "impact_analysis": {
            "affected_files": ["requirements.txt"],
            "file_count": 1,
        },
    }


@pytest.fixture
def mock_adr_tool_no_adr_result():
    """Mock ADR tool result when no ADR is needed."""
    return {
        "should_create_adr": False,
        "confidence": 0.2,
        "triggered_heuristics": [],
        "reasons": [],
    }


# =============================================================================
# Registration Tests
# =============================================================================


class TestADRAutomationMCPRegistration:
    """Test that adr_automation MCP tool is properly registered."""

    def _get_registered_tool_names(self):
        """Get list of registered tool names from MCP server."""
        from app.mcp_server import mcp
        # Access internal tool manager registry (sync access)
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            return list(mcp._tool_manager._tools.keys())
        return []

    def test_adr_automation_tool_registered(self):
        """adr_automation tool should be registered."""
        try:
            tool_names = self._get_registered_tool_names()
            assert "adr_automation" in tool_names, (
                f"adr_automation not found in tools: {tool_names}"
            )
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MCP server tools not available: {e}")


# =============================================================================
# Scope Check Tests
# =============================================================================


class TestADRAutomationScopeChecks:
    """Test that adr_automation MCP tool enforces scope requirements."""

    @pytest.mark.asyncio
    async def test_adr_automation_requires_search_read(
        self, mock_principal_graph_read_only, sample_changes
    ):
        """adr_automation should require search:read scope."""
        try:
            from app.mcp_server import adr_automation, principal_var

            token = principal_var.set(mock_principal_graph_read_only)
            try:
                result = await adr_automation(changes=sample_changes)
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
                assert (
                    "scope" in result_data.get("error", "").lower() or
                    "search:read" in result_data.get("error", "").lower()
                )
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_requires_graph_read(
        self, mock_principal_search_read_only, sample_changes
    ):
        """adr_automation should require graph:read scope."""
        try:
            from app.mcp_server import adr_automation, principal_var

            token = principal_var.set(mock_principal_search_read_only)
            try:
                result = await adr_automation(changes=sample_changes)
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
                assert (
                    "scope" in result_data.get("error", "").lower() or
                    "graph:read" in result_data.get("error", "").lower()
                )
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_no_scopes(
        self, mock_principal_no_scopes, sample_changes
    ):
        """adr_automation should fail with no scopes."""
        try:
            from app.mcp_server import adr_automation, principal_var

            token = principal_var.set(mock_principal_no_scopes)
            try:
                result = await adr_automation(changes=sample_changes)
                result_data = json.loads(result)

                # Should return error for missing scope
                assert "error" in result_data
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_allows_both_scopes(
        self, mock_principal_both_scopes, sample_changes
    ):
        """adr_automation should work with both scopes."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_instance.adr_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_both_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        error_msg = result_data.get("error", "").lower()
                        assert "scope" not in error_msg
                        assert "search:read" not in error_msg
                        assert "graph:read" not in error_msg
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestADRAutomationInputValidation:
    """Test input validation for adr_automation MCP tool."""

    @pytest.mark.asyncio
    async def test_adr_automation_empty_changes(self, mock_principal_all_scopes):
        """adr_automation should handle empty changes list."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = {
                    "should_create_adr": False,
                    "confidence": 0.0,
                    "triggered_heuristics": [],
                    "reasons": [],
                }
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=[])
                    result_data = json.loads(result)

                    # Should return valid response with no ADR needed
                    assert result_data.get("should_create_adr") is False
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_with_min_confidence(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation should accept min_confidence parameter."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = {
                    "should_create_adr": True,
                    "confidence": 0.85,
                    "triggered_heuristics": ["dependency"],
                    "reasons": ["Significant dependency added"],
                }
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(
                        changes=sample_changes,
                        min_confidence=0.8
                    )
                    result_data = json.loads(result)

                    # Should return valid response
                    assert isinstance(result_data, dict)
                    # min_confidence should have been passed to tool
                    mock_instance.adr_tool.execute.assert_called_once()
                    call_args = mock_instance.adr_tool.execute.call_args[0][0]
                    assert call_args.get("min_confidence") == 0.8
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Backend Availability Tests
# =============================================================================


class TestADRAutomationBackendAvailability:
    """Test adr_automation handles backend availability gracefully."""

    @pytest.mark.asyncio
    async def test_adr_automation_neo4j_unavailable(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation should return degraded response when Neo4j unavailable."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_instance.adr_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Should indicate degraded mode
                    assert (
                        result_data.get("meta", {}).get("degraded_mode", False) or
                        "missing" in str(result_data).lower() or
                        "neo4j" in str(result_data.get("meta", {})).lower()
                    )
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_toolkit_unavailable(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation should return error when toolkit has no adr_tool."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.adr_tool = None  # Tool not available
                mock_instance.get_error.return_value = "ADR tool not initialized"
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Should indicate tool unavailable
                    meta = result_data.get("meta", {})
                    assert (
                        meta.get("degraded_mode", False) or
                        meta.get("error") is not None or
                        "unavailable" in str(result_data).lower() or
                        "not available" in str(result_data).lower()
                    )
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Response Structure Tests
# =============================================================================


class TestADRAutomationResponseStructure:
    """Test adr_automation returns properly structured JSON responses."""

    @pytest.mark.asyncio
    async def test_adr_automation_returns_json_string(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation should return a JSON string."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = {
                    "should_create_adr": False,
                    "confidence": 0.0,
                    "triggered_heuristics": [],
                    "reasons": [],
                }
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)

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
    async def test_adr_automation_response_has_meta(
        self, mock_principal_all_scopes, sample_changes, mock_adr_tool_result
    ):
        """adr_automation response should include meta field."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Should have meta field
                    assert "meta" in result_data

                    meta = result_data["meta"]
                    # Meta should have expected fields
                    assert "request_id" in meta
                    assert "degraded_mode" in meta
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_response_structure_on_detection(
        self, mock_principal_all_scopes, sample_changes, mock_adr_tool_result
    ):
        """adr_automation response should have correct structure when ADR detected."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Core fields
                    assert "should_create_adr" in result_data
                    assert result_data["should_create_adr"] is True
                    assert "confidence" in result_data
                    assert isinstance(result_data["confidence"], (int, float))
                    assert 0.0 <= result_data["confidence"] <= 1.0

                    # Heuristic results
                    assert "triggered_heuristics" in result_data
                    assert isinstance(result_data["triggered_heuristics"], list)
                    assert "reasons" in result_data
                    assert isinstance(result_data["reasons"], list)

                    # Generated ADR (when should_create_adr is True)
                    assert "generated_adr" in result_data
                    adr = result_data["generated_adr"]
                    assert "title" in adr
                    assert "status" in adr
                    assert "context" in adr
                    assert "decision" in adr
                    assert "consequences" in adr

                    # Meta field
                    assert "meta" in result_data
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_response_structure_no_detection(
        self, mock_principal_all_scopes, sample_changes_no_adr, mock_adr_tool_no_adr_result
    ):
        """adr_automation response should have correct structure when no ADR detected."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_no_adr_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes_no_adr)
                    result_data = json.loads(result)

                    # Core fields
                    assert "should_create_adr" in result_data
                    assert result_data["should_create_adr"] is False
                    assert "confidence" in result_data

                    # Empty or minimal heuristic results
                    assert "triggered_heuristics" in result_data
                    assert len(result_data["triggered_heuristics"]) == 0

                    # No generated_adr when should_create_adr is False
                    assert (
                        "generated_adr" not in result_data or
                        result_data.get("generated_adr") is None
                    )

                    # Meta field should still be present
                    assert "meta" in result_data
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Valid Input Tests
# =============================================================================


class TestADRAutomationWithValidInput:
    """Test adr_automation with valid input returns expected results."""

    @pytest.mark.asyncio
    async def test_adr_automation_dependency_detection(
        self, mock_principal_all_scopes, sample_changes, mock_adr_tool_result
    ):
        """adr_automation should detect dependency changes."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    assert result_data["should_create_adr"] is True
                    assert "dependency" in result_data["triggered_heuristics"]
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_api_change_detection(
        self, mock_principal_all_scopes, sample_changes_api
    ):
        """adr_automation should detect API changes."""
        try:
            from app.mcp_server import adr_automation, principal_var

            api_result = {
                "should_create_adr": True,
                "confidence": 0.9,
                "triggered_heuristics": ["api_change"],
                "reasons": ["Breaking API change detected"],
                "generated_adr": {
                    "title": "API Change: v2 endpoints",
                    "status": "Proposed",
                    "context": "API endpoints modified...",
                    "decision": "We will introduce new API endpoints...",
                    "consequences": ["API documentation needs updating"],
                },
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = api_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes_api)
                    result_data = json.loads(result)

                    assert result_data["should_create_adr"] is True
                    assert "api_change" in result_data["triggered_heuristics"]
                    assert result_data["confidence"] >= 0.8
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_code_links_included(
        self, mock_principal_all_scopes, sample_changes, mock_adr_tool_result
    ):
        """adr_automation should include code_links when ADR detected."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    assert result_data["should_create_adr"] is True
                    assert "code_links" in result_data
                    assert isinstance(result_data["code_links"], list)
                    if result_data["code_links"]:
                        link = result_data["code_links"][0]
                        assert "file_path" in link
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_impact_analysis_included(
        self, mock_principal_all_scopes, sample_changes, mock_adr_tool_result
    ):
        """adr_automation should include impact_analysis when ADR detected."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.return_value = mock_adr_tool_result
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    if result_data["should_create_adr"]:
                        assert "impact_analysis" in result_data
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestADRAutomationErrorHandling:
    """Test adr_automation handles errors gracefully."""

    @pytest.mark.asyncio
    async def test_adr_automation_tool_exception(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation should handle tool exceptions gracefully."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.adr_tool = MagicMock()
                mock_instance.adr_tool.execute.side_effect = Exception("Tool error")
                mock_instance.get_missing_sources.return_value = []
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Should return error or degraded response
                    meta = result_data.get("meta", {})
                    assert (
                        meta.get("error") is not None or
                        meta.get("degraded_mode", False) or
                        "error" in result_data
                    )
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")

    @pytest.mark.asyncio
    async def test_adr_automation_degraded_response_structure(
        self, mock_principal_all_scopes, sample_changes
    ):
        """adr_automation degraded response should have proper structure."""
        try:
            from app.mcp_server import adr_automation, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j", "opensearch"]
                mock_instance.adr_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await adr_automation(changes=sample_changes)
                    result_data = json.loads(result)

                    # Even in degraded mode, should have expected structure
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    assert "degraded_mode" in meta
                    assert meta["degraded_mode"] is True
                    assert "missing_sources" in meta
                    assert isinstance(meta["missing_sources"], list)
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("MCP tools not available")
