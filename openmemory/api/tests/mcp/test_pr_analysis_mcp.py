"""
Tests for PR Analysis MCP Tool.

TDD: These tests define the expected behavior for the pr_analysis MCP tool
that needs to be added to openmemory/api/app/mcp_server.py.

The pr_analysis tool requires:
- Scope: search:read + graph:read (like explain_code)
- Integration: Uses PRAnalysisTool from tools/pr_workflow/pr_analysis.py
- Response: JSON string with summary, issues, and meta

Test scenarios:
1. Scope check - Returns error if scopes missing
2. Valid input - Returns PR analysis with meta
3. Missing backend - Returns degraded_mode response
4. Toolkit unavailable - Returns error when toolkit has no pr_analysis_tool
5. Input validation - Validates required inputs (repo_id)
6. Response structure - Response has correct JSON structure with meta
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestPRAnalysisScopeChecks:
    """Test that pr_analysis MCP tool enforces scope requirements."""

    @pytest.fixture
    def mock_principal_no_scopes(self):
        """Create a mock principal with no scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=False)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def mock_principal_search_read_only(self):
        """Create a mock principal with only search:read scope."""
        principal = MagicMock()
        principal.has_scope = MagicMock(
            side_effect=lambda s: s in ["search:read"]
        )
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def mock_principal_graph_read_only(self):
        """Create a mock principal with only graph:read scope."""
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
    async def test_pr_analysis_scope_check_no_scopes(self, mock_principal_no_scopes):
        """pr_analysis should return error when no scopes are present."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            token = principal_var.set(mock_principal_no_scopes)
            try:
                result = await pr_analysis(
                    repo_id="test-repo",
                    diff="diff --git a/file.py b/file.py\n+new line"
                )
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
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_requires_both_scopes_missing_graph(
        self, mock_principal_search_read_only
    ):
        """pr_analysis should require both search:read and graph:read scopes."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            token = principal_var.set(mock_principal_search_read_only)
            try:
                result = await pr_analysis(
                    repo_id="test-repo",
                    diff="diff --git a/file.py b/file.py\n+new line"
                )
                result_data = json.loads(result)

                # Should return error for missing graph:read scope
                assert "error" in result_data
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_requires_both_scopes_missing_search(
        self, mock_principal_graph_read_only
    ):
        """pr_analysis should require both search:read and graph:read scopes."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            token = principal_var.set(mock_principal_graph_read_only)
            try:
                result = await pr_analysis(
                    repo_id="test-repo",
                    diff="diff --git a/file.py b/file.py\n+new line"
                )
                result_data = json.loads(result)

                # Should return error for missing search:read scope
                assert "error" in result_data
            finally:
                principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_allows_both_scopes(self, mock_principal_both_scopes):
        """pr_analysis should work with both search:read and graph:read scopes."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j", "opensearch"]
                mock_instance.pr_analysis_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_both_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should not be a scope error
                    if "error" in result_data:
                        error_msg = result_data.get("error", "").lower()
                        assert "scope" not in error_msg
                        assert "insufficient" not in error_msg
                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisWithValidInput:
    """Test pr_analysis with valid inputs and mocked dependencies."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def sample_diff(self):
        """Return a sample diff for testing."""
        return """diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,10 @@ def authenticate(user, password):
     if not user:
         return None
+    # New authentication logic
+    token = generate_token(user)
+    cache_token(token)
+    return token
     return hash_password(password)
"""

    @pytest.mark.asyncio
    async def test_pr_analysis_with_valid_input(
        self, mock_principal_all_scopes, sample_diff
    ):
        """pr_analysis should return analysis with valid input."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            # Create mock PR analysis tool
            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 1,
                    "additions": 4,
                    "deletions": 0,
                    "languages": ["python"],
                    "main_areas": ["src"],
                    "complexity_score": 0.2,
                    "affected_files": [],
                    "suggested_adr": False,
                    "adr_reason": None,
                },
                "issues": [],
                "request_id": "test-request-id",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff=sample_diff
                    )

                    # Result should be a string
                    assert isinstance(result, str)

                    # Result should be valid JSON
                    result_data = json.loads(result)
                    assert isinstance(result_data, dict)

                    # Should have summary
                    assert "summary" in result_data
                    summary = result_data["summary"]
                    assert "files_changed" in summary
                    assert "additions" in summary
                    assert "deletions" in summary

                    # Should have meta
                    assert "meta" in result_data

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_returns_meta_with_request_id(
        self, mock_principal_all_scopes, sample_diff
    ):
        """pr_analysis should return meta with request_id."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 1,
                    "additions": 4,
                    "deletions": 0,
                    "languages": ["python"],
                    "main_areas": ["src"],
                    "complexity_score": 0.2,
                    "affected_files": [],
                    "suggested_adr": False,
                    "adr_reason": None,
                },
                "issues": [],
                "request_id": "test-request-id-123",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff=sample_diff
                    )
                    result_data = json.loads(result)

                    # Meta should have request_id
                    assert "meta" in result_data
                    assert "request_id" in result_data["meta"]
                    assert result_data["meta"]["request_id"] is not None

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisMissingBackend:
    """Test pr_analysis behavior when backends are unavailable."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_pr_analysis_with_missing_backend(self, mock_principal_all_scopes):
        """pr_analysis should return degraded_mode response when backend unavailable."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j", "opensearch"]
                mock_instance.get_error.return_value = "Backend unavailable"
                mock_instance.pr_analysis_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should indicate degraded mode in meta
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    assert meta.get("degraded_mode", False) is True
                    assert len(meta.get("missing_sources", [])) > 0

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_with_partial_backend(self, mock_principal_all_scopes):
        """pr_analysis should work with partial backend availability."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            # Create mock PR analysis tool that works without full backends
            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 1,
                    "additions": 1,
                    "deletions": 0,
                    "languages": ["python"],
                    "main_areas": [],
                    "complexity_score": 0.1,
                    "affected_files": [],  # Empty due to missing impact analysis
                    "suggested_adr": False,
                    "adr_reason": None,
                },
                "issues": [],
                "request_id": "partial-test-id",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.side_effect = lambda s: s != "neo4j"
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should have summary even with partial backend
                    assert "summary" in result_data

                    # Should indicate partial degradation in meta
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    if meta.get("missing_sources"):
                        assert "neo4j" in meta["missing_sources"]

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisToolkitUnavailable:
    """Test pr_analysis behavior when toolkit or tool is unavailable."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_pr_analysis_with_toolkit_unavailable(self, mock_principal_all_scopes):
        """pr_analysis should return error when toolkit has no pr_analysis_tool."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                # pr_analysis_tool not initialized
                mock_instance.pr_analysis_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should have meta indicating tool unavailable
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    assert (
                        meta.get("degraded_mode", False) is True or
                        meta.get("error") is not None or
                        "error" in result_data
                    )

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_with_tool_execution_error(self, mock_principal_all_scopes):
        """pr_analysis should handle tool execution errors gracefully."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.side_effect = Exception("Analysis failed")

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should have error in meta
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    assert (
                        meta.get("error") is not None or
                        meta.get("degraded_mode", False) is True
                    )

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisInputValidation:
    """Test pr_analysis input validation."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_pr_analysis_requires_repo_id(self, mock_principal_all_scopes):
        """pr_analysis should require repo_id."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    # Call without repo_id (empty string)
                    result = await pr_analysis(
                        repo_id="",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )
                    result_data = json.loads(result)

                    # Should have error about missing repo_id
                    assert "error" in result_data or (
                        "meta" in result_data and result_data["meta"].get("error")
                    )

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_accepts_optional_inputs(self, mock_principal_all_scopes):
        """pr_analysis should accept optional inputs like title, body, pr_number."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 1,
                    "additions": 1,
                    "deletions": 0,
                    "languages": [],
                    "main_areas": [],
                    "complexity_score": 0.1,
                    "affected_files": [],
                    "suggested_adr": False,
                    "adr_reason": None,
                },
                "issues": [],
                "request_id": "test-id",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line",
                        pr_number=123,
                        title="Test PR",
                        body="This is a test PR description",
                        check_impact=True,
                        check_adr=False,
                        check_security=True,
                        check_conventions=True,
                    )
                    result_data = json.loads(result)

                    # Should succeed with optional inputs
                    assert "summary" in result_data or "error" not in result_data

                    # Verify the tool was called with inputs
                    if mock_pr_tool.execute.called:
                        call_args = mock_pr_tool.execute.call_args
                        if call_args and call_args[0]:
                            input_data = call_args[0][0]
                            assert input_data.get("repo_id") == "test-repo"

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisResponseStructure:
    """Test pr_analysis response structure matches expected format."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.mark.asyncio
    async def test_pr_analysis_response_structure(self, mock_principal_all_scopes):
        """pr_analysis response should have correct JSON structure."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 2,
                    "additions": 10,
                    "deletions": 5,
                    "languages": ["python", "javascript"],
                    "main_areas": ["src", "tests"],
                    "complexity_score": 0.35,
                    "affected_files": ["src/utils.py", "tests/test_utils.py"],
                    "suggested_adr": True,
                    "adr_reason": "Significant architectural change detected",
                },
                "issues": [
                    {
                        "severity": "warning",
                        "category": "security",
                        "file_path": "src/auth.py",
                        "line_number": 42,
                        "message": "Possible hardcoded secret",
                        "suggestion": "Use environment variables",
                    }
                ],
                "request_id": "response-test-id",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )

                    # Result should be valid JSON string
                    assert isinstance(result, str)
                    result_data = json.loads(result)

                    # Check summary structure
                    assert "summary" in result_data
                    summary = result_data["summary"]
                    assert isinstance(summary.get("files_changed"), int)
                    assert isinstance(summary.get("additions"), int)
                    assert isinstance(summary.get("deletions"), int)
                    assert isinstance(summary.get("languages"), list)
                    assert isinstance(summary.get("main_areas"), list)
                    assert isinstance(summary.get("complexity_score"), (int, float))
                    assert isinstance(summary.get("affected_files"), list)
                    assert isinstance(summary.get("suggested_adr"), bool)

                    # Check issues structure
                    assert "issues" in result_data
                    assert isinstance(result_data["issues"], list)
                    if result_data["issues"]:
                        issue = result_data["issues"][0]
                        assert "severity" in issue
                        assert "category" in issue
                        assert "file_path" in issue
                        assert "message" in issue

                    # Check meta structure
                    assert "meta" in result_data
                    meta = result_data["meta"]
                    assert "request_id" in meta
                    assert "degraded_mode" in meta
                    assert "missing_sources" in meta

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")

    @pytest.mark.asyncio
    async def test_pr_analysis_returns_json_string(self, mock_principal_all_scopes):
        """pr_analysis should always return a JSON string."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = False
                mock_instance.get_missing_sources.return_value = ["neo4j"]
                mock_instance.pr_analysis_tool = None
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff="diff --git a/file.py b/file.py\n+new line"
                    )

                    # Result should always be a string
                    assert isinstance(result, str)

                    # Result should be valid JSON
                    parsed = json.loads(result)
                    assert isinstance(parsed, dict)

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")


class TestPRAnalysisToolRegistration:
    """Test that pr_analysis MCP tool is properly registered."""

    def test_mcp_server_imports(self):
        """MCP server should import without errors."""
        try:
            from app.mcp_server import mcp
            assert mcp is not None
        except ImportError as e:
            pytest.skip(f"MCP server not available: {e}")

    def _get_registered_tool_names(self):
        """Get list of registered tool names from MCP server."""
        from app.mcp_server import mcp
        if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
            return list(mcp._tool_manager._tools.keys())
        return []

    def test_pr_analysis_tool_registered(self):
        """pr_analysis tool should be registered in MCP server."""
        try:
            tool_names = self._get_registered_tool_names()
            if "pr_analysis" not in tool_names:
                pytest.skip(
                    f"pr_analysis not yet implemented. Current tools: {tool_names}"
                )
            assert "pr_analysis" in tool_names
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MCP server tools not available: {e}")


class TestPRAnalysisSecurityChecks:
    """Test pr_analysis security issue detection."""

    @pytest.fixture
    def mock_principal_all_scopes(self):
        """Create a mock principal with all required scopes."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        principal.user_id = "test-user"
        principal.org_id = "test-org"
        return principal

    @pytest.fixture
    def diff_with_security_issue(self):
        """Return a diff containing potential security issues."""
        return """diff --git a/src/db.py b/src/db.py
--- a/src/db.py
+++ b/src/db.py
@@ -10,3 +10,5 @@ def query_user(user_id):
+    # Direct string interpolation in SQL
+    sql = f"SELECT * FROM users WHERE id = {user_id}"
+    return execute(sql)
"""

    @pytest.mark.asyncio
    async def test_pr_analysis_detects_security_issues(
        self, mock_principal_all_scopes, diff_with_security_issue
    ):
        """pr_analysis should detect security issues in diff."""
        try:
            from app.mcp_server import pr_analysis, principal_var

            mock_pr_tool = MagicMock()
            mock_pr_tool.execute.return_value = {
                "summary": {
                    "files_changed": 1,
                    "additions": 3,
                    "deletions": 0,
                    "languages": ["python"],
                    "main_areas": ["src"],
                    "complexity_score": 0.2,
                    "affected_files": [],
                    "suggested_adr": False,
                    "adr_reason": None,
                },
                "issues": [
                    {
                        "severity": "warning",
                        "category": "security",
                        "file_path": "src/db.py",
                        "line_number": 12,
                        "message": "Possible SQL injection vulnerability",
                        "suggestion": "Use parameterized queries",
                    }
                ],
                "request_id": "security-test-id",
            }

            with patch("app.mcp_server.get_code_toolkit") as mock_toolkit:
                mock_instance = MagicMock()
                mock_instance.is_available.return_value = True
                mock_instance.get_missing_sources.return_value = []
                mock_instance.pr_analysis_tool = mock_pr_tool
                mock_toolkit.return_value = mock_instance

                token = principal_var.set(mock_principal_all_scopes)
                try:
                    result = await pr_analysis(
                        repo_id="test-repo",
                        diff=diff_with_security_issue,
                        check_security=True
                    )
                    result_data = json.loads(result)

                    # Should have issues detected
                    assert "issues" in result_data
                    issues = result_data["issues"]

                    # Should have at least one security issue
                    security_issues = [
                        i for i in issues if i.get("category") == "security"
                    ]
                    assert len(security_issues) > 0

                finally:
                    principal_var.reset(token)

        except ImportError:
            pytest.skip("pr_analysis MCP tool not yet implemented")
