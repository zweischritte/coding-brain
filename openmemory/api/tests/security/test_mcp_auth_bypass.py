"""
Tests for MCP auth bypass regression.

Ensures _check_tool_scope fails closed when principal is not set.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# We'll test the function directly by importing it
# Need to mock HAS_SECURITY to True for these tests


class TestCheckToolScopeAuthBypass:
    """Tests for _check_tool_scope authentication bypass fix."""

    @pytest.fixture
    def mock_principal(self):
        """Create a mock principal with scope checking."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=True)
        return principal

    @pytest.fixture
    def mock_principal_no_scope(self):
        """Create a mock principal without the required scope."""
        principal = MagicMock()
        principal.has_scope = MagicMock(return_value=False)
        return principal

    def test_check_tool_scope_returns_error_when_principal_missing(self):
        """_check_tool_scope should return error JSON when principal_var is not set."""
        # Import here to allow patching
        from app.mcp_server import _check_tool_scope, principal_var

        # Ensure principal_var is not set (or set to None)
        token = principal_var.set(None)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                result = _check_tool_scope("memories:write")

                # Should return error JSON, not None
                assert result is not None
                error = json.loads(result)
                assert error["code"] == "MISSING_AUTH"
                assert "Authentication required" in error["error"]
        finally:
            principal_var.reset(token)

    def test_check_tool_scope_returns_none_when_principal_has_scope(self, mock_principal):
        """_check_tool_scope should return None when principal has required scope."""
        from app.mcp_server import _check_tool_scope, principal_var

        token = principal_var.set(mock_principal)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                result = _check_tool_scope("memories:write")

                assert result is None
                mock_principal.has_scope.assert_called_once_with("memories:write")
        finally:
            principal_var.reset(token)

    def test_check_tool_scope_returns_error_when_scope_missing(self, mock_principal_no_scope):
        """_check_tool_scope should return error when principal lacks scope."""
        from app.mcp_server import _check_tool_scope, principal_var

        token = principal_var.set(mock_principal_no_scope)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                result = _check_tool_scope("memories:delete")

                assert result is not None
                error = json.loads(result)
                assert error["code"] == "INSUFFICIENT_SCOPE"
                assert error["required_scope"] == "memories:delete"
        finally:
            principal_var.reset(token)

    def test_check_tool_scope_skips_check_when_security_disabled(self):
        """_check_tool_scope should return None when HAS_SECURITY is False."""
        from app.mcp_server import _check_tool_scope, principal_var

        # Don't set principal
        token = principal_var.set(None)
        try:
            with patch("app.mcp_server.HAS_SECURITY", False):
                result = _check_tool_scope("memories:write")

                # Should skip check entirely when security disabled
                assert result is None
        finally:
            principal_var.reset(token)

    def test_check_tool_scope_checks_exact_scope_string(self, mock_principal):
        """_check_tool_scope should pass the exact scope string to principal."""
        from app.mcp_server import _check_tool_scope, principal_var

        token = principal_var.set(mock_principal)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                _check_tool_scope("admin:write")

                mock_principal.has_scope.assert_called_once_with("admin:write")
        finally:
            principal_var.reset(token)


class TestCheckToolScopeRegressionPrevention:
    """Tests to prevent regression of the auth bypass fix."""

    def test_no_backwards_compat_bypass(self):
        """Verify backwards compatibility code that bypasses auth is removed."""
        from app.mcp_server import _check_tool_scope, principal_var

        token = principal_var.set(None)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                result = _check_tool_scope("memories:write")

                # The old code returned None here (bypass)
                # New code should return an error
                assert result is not None, (
                    "SECURITY REGRESSION: _check_tool_scope returned None when "
                    "principal_var is not set. This is an auth bypass vulnerability!"
                )
        finally:
            principal_var.reset(token)

    def test_missing_principal_error_is_json_parseable(self):
        """Verify the error response is valid JSON for MCP tools."""
        from app.mcp_server import _check_tool_scope, principal_var

        token = principal_var.set(None)
        try:
            with patch("app.mcp_server.HAS_SECURITY", True):
                result = _check_tool_scope("memories:write")

                # Should be valid JSON
                assert result is not None
                error = json.loads(result)  # Should not raise

                # Should have expected structure
                assert "error" in error
                assert "code" in error
        finally:
            principal_var.reset(token)

    def test_fail_closed_principle(self):
        """_check_tool_scope should fail closed (deny by default)."""
        from app.mcp_server import _check_tool_scope, principal_var

        # Test various "unset" states
        test_cases = [
            (None, "principal_var set to None"),
            # Could add more edge cases if the implementation changes
        ]

        for value, description in test_cases:
            token = principal_var.set(value)
            try:
                with patch("app.mcp_server.HAS_SECURITY", True):
                    result = _check_tool_scope("memories:write")
                    assert result is not None, f"Auth bypass when {description}"
            finally:
                principal_var.reset(token)
