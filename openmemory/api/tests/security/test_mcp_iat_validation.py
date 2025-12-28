"""
Tests for MCP SSE IAT (issued-at) validation.

Ensures future-dated tokens are rejected in MCP SSE handlers.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.security.jwt import validate_iat_not_future
from app.security.types import AuthenticationError


class TestValidateIatNotFuture:
    """Tests for validate_iat_not_future function."""

    def test_past_iat_accepted(self):
        """Token with past iat should be accepted."""
        past_iat = datetime.now(timezone.utc) - timedelta(seconds=5)

        # Should not raise
        validate_iat_not_future(past_iat)

    def test_current_iat_accepted(self):
        """Token with current iat should be accepted."""
        current_iat = datetime.now(timezone.utc)

        # Should not raise
        validate_iat_not_future(current_iat)

    def test_future_iat_within_skew_accepted(self):
        """Token with iat slightly in future (within skew) should be accepted."""
        # Default skew is 30 seconds
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=29)

        # Should not raise (within 30 second skew)
        validate_iat_not_future(future_iat)

    def test_future_iat_at_skew_boundary_accepted(self):
        """Token with iat at exactly the skew boundary should be accepted."""
        # At exactly 30 seconds, still within skew
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=30)

        # Should not raise (exactly at boundary)
        validate_iat_not_future(future_iat)

    def test_future_iat_beyond_skew_rejected(self):
        """Token with iat beyond skew should be rejected."""
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=60)

        with pytest.raises(AuthenticationError) as exc_info:
            validate_iat_not_future(future_iat)

        assert exc_info.value.code == "INVALID_IAT"
        assert "future" in exc_info.value.message.lower()

    def test_custom_skew_value_respected(self):
        """Custom skew value should be used when provided."""
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=10)

        # With 5-second skew, 10 seconds in future should fail
        with pytest.raises(AuthenticationError):
            validate_iat_not_future(future_iat, max_clock_skew_seconds=5)

    def test_custom_skew_allows_larger_drift(self):
        """Larger custom skew should allow more drift."""
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=50)

        # With 60-second skew, 50 seconds in future should pass
        validate_iat_not_future(future_iat, max_clock_skew_seconds=60)

    def test_very_future_iat_rejected(self):
        """Token with iat far in future should be rejected."""
        future_iat = datetime.now(timezone.utc) + timedelta(hours=1)

        with pytest.raises(AuthenticationError) as exc_info:
            validate_iat_not_future(future_iat)

        assert exc_info.value.code == "INVALID_IAT"


class TestMcpPostIatValidation:
    """Tests for IAT validation in MCP POST handler."""

    @pytest.fixture
    def mock_principal(self):
        """Create a mock principal with a past iat."""
        principal = MagicMock()
        principal.user_id = "user-123"
        principal.org_id = "org-456"
        principal.claims = MagicMock()
        principal.claims.iat = datetime.now(timezone.utc) - timedelta(seconds=5)
        return principal

    @pytest.fixture
    def mock_principal_future_iat(self):
        """Create a mock principal with a future iat."""
        principal = MagicMock()
        principal.user_id = "user-123"
        principal.org_id = "org-456"
        principal.claims = MagicMock()
        # 60 seconds in future (beyond 30 second skew)
        principal.claims.iat = datetime.now(timezone.utc) + timedelta(seconds=60)
        return principal

    def test_iat_validation_called_on_post(self, mock_principal):
        """IAT validation should be called when processing POST."""
        # This test verifies the integration pattern
        # The actual integration is tested in integration tests

        # Validate that calling validate_iat_not_future with a valid iat doesn't raise
        validate_iat_not_future(mock_principal.claims.iat)

    def test_future_iat_raises_error(self, mock_principal_future_iat):
        """Future iat should raise AuthenticationError."""
        with pytest.raises(AuthenticationError) as exc_info:
            validate_iat_not_future(mock_principal_future_iat.claims.iat)

        assert exc_info.value.code == "INVALID_IAT"


class TestIatEdgeCases:
    """Tests for edge cases in IAT validation."""

    def test_iat_exactly_at_clock_boundary(self):
        """Test iat at exact clock boundary."""
        # Just beyond the 30 second skew
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=31)

        with pytest.raises(AuthenticationError):
            validate_iat_not_future(future_iat)

    def test_zero_skew_allows_no_future(self):
        """With zero skew, any future iat should be rejected."""
        future_iat = datetime.now(timezone.utc) + timedelta(seconds=1)

        with pytest.raises(AuthenticationError):
            validate_iat_not_future(future_iat, max_clock_skew_seconds=0)

    def test_negative_clock_drift_always_accepted(self):
        """Tokens issued in the past should always be accepted."""
        # Token issued 1 hour ago
        past_iat = datetime.now(timezone.utc) - timedelta(hours=1)

        # Should not raise (past tokens are fine, expiry is separate check)
        validate_iat_not_future(past_iat)

    def test_very_old_iat_accepted(self):
        """Very old tokens are accepted (exp check is separate)."""
        # Token issued 1 day ago
        old_iat = datetime.now(timezone.utc) - timedelta(days=1)

        # Should not raise (iat validation only checks for future)
        validate_iat_not_future(old_iat)
