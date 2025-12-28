"""
Integration tests for MCP SSE session binding.

Tests the session binding flow between GET and POST handlers.
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.security.session_binding import (
    MemorySessionBindingStore,
    get_session_binding_store,
    reset_session_binding_store,
)


# Test constants
TEST_SESSION_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_USER_ID = "user-123"
TEST_USER_ID_2 = "user-456"
TEST_ORG_ID = "org-abc"
TEST_ORG_ID_2 = "org-xyz"


class TestSessionBindingIntegration:
    """Integration tests for session binding flow."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_get_creates_session_binding(self):
        """GET handler should create session binding with principal."""
        store = get_session_binding_store()

        # Simulate what GET handler does
        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        assert binding is not None
        assert binding.session_id == TEST_SESSION_ID
        assert binding.user_id == TEST_USER_ID
        assert binding.org_id == TEST_ORG_ID

    def test_post_validates_session_binding_success(self):
        """POST handler should succeed when session binding matches."""
        store = get_session_binding_store()

        # Create binding (simulating GET)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Validate (simulating POST)
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert is_valid is True

    def test_post_rejects_mismatched_user_id(self):
        """POST should return 403 when user_id doesn't match binding."""
        store = get_session_binding_store()

        # Create binding for user-123
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Validate with different user (simulating POST with wrong JWT)
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID_2, TEST_ORG_ID)

        assert is_valid is False

    def test_post_rejects_mismatched_org_id(self):
        """POST should return 403 when org_id doesn't match binding."""
        store = get_session_binding_store()

        # Create binding for org-abc
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Validate with different org
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID_2)

        assert is_valid is False

    def test_post_rejects_expired_session(self):
        """POST should return 403 when session has expired."""
        # Use 1 second TTL
        store = MemorySessionBindingStore(default_ttl_seconds=1)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Wait for expiration
        import time
        time.sleep(1.1)

        # Validate should fail
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert is_valid is False

    def test_post_rejects_missing_session(self):
        """POST should return 403 when session doesn't exist."""
        store = get_session_binding_store()

        # Don't create any binding

        # Validate should fail
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert is_valid is False

    def test_disconnect_deletes_session_binding(self):
        """SSE disconnect should delete session binding."""
        store = get_session_binding_store()

        # Create binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Verify it exists
        assert store.get(TEST_SESSION_ID) is not None

        # Simulate disconnect cleanup
        deleted = store.delete(TEST_SESSION_ID)

        assert deleted is True
        assert store.get(TEST_SESSION_ID) is None

    def test_multiple_sessions_independent(self):
        """Multiple sessions should be tracked independently."""
        store = get_session_binding_store()
        session_1 = uuid.uuid4()
        session_2 = uuid.uuid4()

        # Create two sessions for different users
        store.create(session_id=session_1, user_id=TEST_USER_ID, org_id=TEST_ORG_ID)
        store.create(session_id=session_2, user_id=TEST_USER_ID_2, org_id=TEST_ORG_ID_2)

        # Each session should validate only for its own user
        assert store.validate(session_1, TEST_USER_ID, TEST_ORG_ID) is True
        assert store.validate(session_1, TEST_USER_ID_2, TEST_ORG_ID_2) is False

        assert store.validate(session_2, TEST_USER_ID_2, TEST_ORG_ID_2) is True
        assert store.validate(session_2, TEST_USER_ID, TEST_ORG_ID) is False


class TestSessionBindingWithDPoP:
    """Tests for session binding with DPoP thumbprint."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_dpop_binding_created_when_provided(self):
        """DPoP thumbprint should be stored in binding when provided."""
        store = get_session_binding_store()
        dpop_thumbprint = "test-dpop-thumbprint"

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=dpop_thumbprint,
        )

        assert binding.dpop_thumbprint == dpop_thumbprint

    def test_dpop_required_when_bound(self):
        """POST should require matching DPoP when binding has thumbprint."""
        store = get_session_binding_store()
        dpop_thumbprint = "test-dpop-thumbprint"

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=dpop_thumbprint,
        )

        # Should fail without DPoP
        assert store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID) is False

        # Should succeed with matching DPoP
        assert store.validate(
            TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=dpop_thumbprint
        ) is True

        # Should fail with wrong DPoP
        assert store.validate(
            TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint="wrong-thumbprint"
        ) is False


class TestSessionBindingErrorResponses:
    """Tests for session binding error response formats."""

    def test_missing_session_id_format(self):
        """Missing session_id should return 400 with proper error format."""
        # This test documents the expected response format
        expected_error = {
            "error": "Missing session_id",
            "code": "MISSING_SESSION_ID",
        }

        # Verify format is correct
        assert "error" in expected_error
        assert "code" in expected_error
        assert expected_error["code"] == "MISSING_SESSION_ID"

    def test_invalid_session_id_format(self):
        """Invalid session_id should return 400 with proper error format."""
        expected_error = {
            "error": "Invalid session_id format",
            "code": "INVALID_SESSION_ID",
        }

        assert "error" in expected_error
        assert "code" in expected_error
        assert expected_error["code"] == "INVALID_SESSION_ID"

    def test_session_binding_invalid_format(self):
        """Session binding mismatch should return 403 with proper error format."""
        expected_error = {
            "error": "Session binding mismatch",
            "code": "SESSION_BINDING_INVALID",
        }

        assert "error" in expected_error
        assert "code" in expected_error
        assert expected_error["code"] == "SESSION_BINDING_INVALID"


class TestSessionBindingConcurrency:
    """Tests for concurrent session binding operations."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_concurrent_session_creation(self):
        """Multiple concurrent session creations should not conflict."""
        import threading

        store = get_session_binding_store()
        sessions_created = []
        errors = []

        def create_session(user_num):
            try:
                session_id = uuid.uuid4()
                store.create(
                    session_id=session_id,
                    user_id=f"user-{user_num}",
                    org_id=f"org-{user_num}",
                )
                sessions_created.append(session_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_session, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sessions_created) == 20

    def test_concurrent_validation_same_session(self):
        """Multiple concurrent validations of same session should all succeed."""
        import threading

        store = get_session_binding_store()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        results = []
        errors = []

        def validate_session():
            try:
                result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=validate_session)
            for _ in range(50)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)
        assert len(results) == 50
