"""
Unit tests for the session binding store.

Tests the MemorySessionBindingStore for MCP SSE session binding.
"""
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.security.session_binding import (
    MemorySessionBindingStore,
    SessionBinding,
    get_session_binding_store,
    reset_session_binding_store,
)


# Test constants
TEST_SESSION_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_SESSION_ID_2 = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
TEST_USER_ID = "user-123"
TEST_USER_ID_2 = "user-456"
TEST_ORG_ID = "org-abc"
TEST_ORG_ID_2 = "org-xyz"
TEST_DPOP_THUMBPRINT = "test-dpop-thumbprint-abc123"


class TestSessionBindingCreate:
    """Tests for session binding creation."""

    def test_create_binding_stores_all_fields(self):
        """Create binding should store session_id, user_id, org_id, and timestamps."""
        store = MemorySessionBindingStore(default_ttl_seconds=1800)

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        assert binding.session_id == TEST_SESSION_ID
        assert binding.user_id == TEST_USER_ID
        assert binding.org_id == TEST_ORG_ID
        assert binding.issued_at is not None
        assert binding.expires_at is not None
        assert binding.dpop_thumbprint is None

    def test_create_binding_with_dpop_thumbprint(self):
        """Create binding should store DPoP thumbprint when provided."""
        store = MemorySessionBindingStore()

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        assert binding.dpop_thumbprint == TEST_DPOP_THUMBPRINT

    def test_create_binding_with_custom_ttl(self):
        """Create binding should use custom TTL when provided."""
        store = MemorySessionBindingStore(default_ttl_seconds=1800)
        custom_ttl = 300  # 5 minutes

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            ttl_seconds=custom_ttl,
        )

        expected_expiry = binding.issued_at + timedelta(seconds=custom_ttl)
        # Allow 1 second tolerance for test execution time
        assert abs((binding.expires_at - expected_expiry).total_seconds()) < 1

    def test_create_binding_uses_default_ttl(self):
        """Create binding should use default TTL when none provided."""
        default_ttl = 1800
        store = MemorySessionBindingStore(default_ttl_seconds=default_ttl)

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        expected_expiry = binding.issued_at + timedelta(seconds=default_ttl)
        assert abs((binding.expires_at - expected_expiry).total_seconds()) < 1

    def test_create_binding_overwrites_existing(self):
        """Create binding should overwrite existing binding for same session_id."""
        store = MemorySessionBindingStore()

        # Create initial binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Overwrite with new user
        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID_2,
            org_id=TEST_ORG_ID_2,
        )

        # Get should return the new binding
        retrieved = store.get(TEST_SESSION_ID)
        assert retrieved is not None
        assert retrieved.user_id == TEST_USER_ID_2
        assert retrieved.org_id == TEST_ORG_ID_2


class TestSessionBindingGet:
    """Tests for session binding retrieval."""

    def test_get_existing_binding_returns_binding(self):
        """Get should return binding when it exists and is not expired."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        binding = store.get(TEST_SESSION_ID)

        assert binding is not None
        assert binding.session_id == TEST_SESSION_ID
        assert binding.user_id == TEST_USER_ID
        assert binding.org_id == TEST_ORG_ID

    def test_get_nonexistent_binding_returns_none(self):
        """Get should return None when binding does not exist."""
        store = MemorySessionBindingStore()

        binding = store.get(TEST_SESSION_ID)

        assert binding is None

    def test_get_expired_binding_returns_none(self):
        """Get should return None when binding has expired."""
        store = MemorySessionBindingStore(default_ttl_seconds=1)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Wait for expiration
        time.sleep(1.1)

        binding = store.get(TEST_SESSION_ID)

        assert binding is None

    def test_get_expired_binding_cleans_up_entry(self):
        """Get on expired binding should remove it from store."""
        store = MemorySessionBindingStore(default_ttl_seconds=1)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Wait for expiration
        time.sleep(1.1)

        # First get returns None and cleans up
        assert store.get(TEST_SESSION_ID) is None

        # Verify it's actually removed (internal check)
        assert TEST_SESSION_ID not in store._bindings


class TestSessionBindingValidate:
    """Tests for session binding validation."""

    def test_validate_matching_principal_returns_true(self):
        """Validate should return True when user_id and org_id match."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is True

    def test_validate_mismatched_user_id_returns_false(self):
        """Validate should return False when user_id doesn't match."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID_2, TEST_ORG_ID)

        assert result is False

    def test_validate_mismatched_org_id_returns_false(self):
        """Validate should return False when org_id doesn't match."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID_2)

        assert result is False

    def test_validate_nonexistent_session_returns_false(self):
        """Validate should return False when session doesn't exist."""
        store = MemorySessionBindingStore()

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is False

    def test_validate_expired_session_returns_false(self):
        """Validate should return False when session has expired."""
        store = MemorySessionBindingStore(default_ttl_seconds=1)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        time.sleep(1.1)

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is False

    def test_validate_with_matching_dpop_thumbprint_returns_true(self):
        """Validate should return True when DPoP thumbprint matches."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        result = store.validate(
            TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT
        )

        assert result is True

    def test_validate_with_mismatched_dpop_thumbprint_returns_false(self):
        """Validate should return False when DPoP thumbprint doesn't match."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        result = store.validate(
            TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint="wrong-thumbprint"
        )

        assert result is False

    def test_validate_without_dpop_when_binding_has_dpop_returns_false(self):
        """Validate should return False when binding requires DPoP but none provided."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is False

    def test_validate_with_dpop_when_binding_has_none_returns_true(self):
        """Validate should return True when binding doesn't require DPoP."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Providing DPoP when binding doesn't require it should still pass
        result = store.validate(
            TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT
        )

        assert result is True


class TestSessionBindingDelete:
    """Tests for session binding deletion."""

    def test_delete_existing_binding_returns_true(self):
        """Delete should return True when binding exists."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.delete(TEST_SESSION_ID)

        assert result is True

    def test_delete_removes_binding(self):
        """Delete should remove binding from store."""
        store = MemorySessionBindingStore()
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        store.delete(TEST_SESSION_ID)

        assert store.get(TEST_SESSION_ID) is None

    def test_delete_nonexistent_binding_returns_false(self):
        """Delete should return False when binding doesn't exist."""
        store = MemorySessionBindingStore()

        result = store.delete(TEST_SESSION_ID)

        assert result is False


class TestSessionBindingCleanup:
    """Tests for session binding cleanup."""

    def test_cleanup_expired_removes_expired_bindings(self):
        """Cleanup should remove all expired bindings."""
        store = MemorySessionBindingStore(default_ttl_seconds=1)

        # Create two bindings
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
        store.create(TEST_SESSION_ID_2, TEST_USER_ID_2, TEST_ORG_ID_2)

        # Wait for expiration
        time.sleep(1.1)

        removed = store.cleanup_expired()

        assert removed == 2
        assert store.get(TEST_SESSION_ID) is None
        assert store.get(TEST_SESSION_ID_2) is None

    def test_cleanup_expired_preserves_valid_bindings(self):
        """Cleanup should not remove non-expired bindings."""
        store = MemorySessionBindingStore(default_ttl_seconds=3600)

        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        removed = store.cleanup_expired()

        assert removed == 0
        assert store.get(TEST_SESSION_ID) is not None

    def test_cleanup_expired_returns_count(self):
        """Cleanup should return count of removed bindings."""
        store = MemorySessionBindingStore(default_ttl_seconds=1)

        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        time.sleep(1.1)

        removed = store.cleanup_expired()

        assert removed == 1


class TestSessionBindingThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_create_does_not_raise(self):
        """Concurrent creates should not raise exceptions."""
        store = MemorySessionBindingStore()
        errors = []

        def create_binding(session_id):
            try:
                for _ in range(100):
                    store.create(session_id, TEST_USER_ID, TEST_ORG_ID)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_binding, args=(uuid.uuid4(),))
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_get_and_validate_does_not_raise(self):
        """Concurrent get and validate operations should not raise."""
        store = MemorySessionBindingStore()
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
        errors = []

        def read_operations():
            try:
                for _ in range(100):
                    store.get(TEST_SESSION_ID)
                    store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=read_operations)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_create_and_delete_does_not_raise(self):
        """Concurrent create and delete operations should not raise."""
        store = MemorySessionBindingStore()
        errors = []

        def mixed_operations():
            try:
                session_id = uuid.uuid4()
                for _ in range(50):
                    store.create(session_id, TEST_USER_ID, TEST_ORG_ID)
                    store.delete(session_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mixed_operations)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSessionBindingSingleton:
    """Tests for the singleton store accessor."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_session_binding_store()

    def test_get_session_binding_store_returns_singleton(self):
        """get_session_binding_store should return the same instance."""
        store1 = get_session_binding_store()
        store2 = get_session_binding_store()

        assert store1 is store2

    def test_get_session_binding_store_uses_env_ttl(self):
        """get_session_binding_store should use MCP_SESSION_TTL_SECONDS env var."""
        with patch.dict("os.environ", {"MCP_SESSION_TTL_SECONDS": "600"}):
            reset_session_binding_store()
            store = get_session_binding_store()

            assert store._default_ttl == 600

    def test_get_session_binding_store_defaults_to_1800(self):
        """get_session_binding_store should default to 1800 seconds TTL."""
        with patch.dict("os.environ", {}, clear=True):
            reset_session_binding_store()
            store = get_session_binding_store()

            assert store._default_ttl == 1800

    def test_reset_session_binding_store_clears_singleton(self):
        """reset_session_binding_store should clear the singleton."""
        store1 = get_session_binding_store()
        reset_session_binding_store()
        store2 = get_session_binding_store()

        assert store1 is not store2

    def test_get_session_binding_store_defaults_to_memory(self):
        """get_session_binding_store should default to memory store when env not set."""
        with patch.dict("os.environ", {}, clear=True):
            reset_session_binding_store()
            store = get_session_binding_store()

            assert isinstance(store, MemorySessionBindingStore)

    def test_get_session_binding_store_returns_memory_when_configured(self):
        """get_session_binding_store should return memory store when MCP_SESSION_STORE=memory."""
        with patch.dict("os.environ", {"MCP_SESSION_STORE": "memory"}):
            reset_session_binding_store()
            store = get_session_binding_store()

            assert isinstance(store, MemorySessionBindingStore)

    def test_get_session_binding_store_returns_valkey_when_configured(self):
        """get_session_binding_store should return valkey store when MCP_SESSION_STORE=valkey."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        # Mock the valkey connection to succeed
        with patch.dict("os.environ", {"MCP_SESSION_STORE": "valkey"}):
            with patch("app.security.session_binding.get_valkey_session_binding_store") as mock_valkey:
                mock_store = ValkeySessionBindingStore(
                    client=type("MockClient", (), {"ping": lambda s: True, "get": lambda s, k: None, "setex": lambda s, k, t, v: True, "delete": lambda s, *k: 0})(),
                    default_ttl_seconds=1800,
                )
                mock_valkey.return_value = mock_store

                reset_session_binding_store()
                store = get_session_binding_store()

                assert isinstance(store, ValkeySessionBindingStore)

    def test_get_session_binding_store_falls_back_to_memory_on_valkey_failure(self):
        """get_session_binding_store should fall back to memory if valkey unavailable."""
        with patch.dict("os.environ", {"MCP_SESSION_STORE": "valkey"}):
            with patch("app.security.session_binding.get_valkey_session_binding_store") as mock_valkey:
                mock_valkey.return_value = None  # Valkey unavailable

                reset_session_binding_store()
                store = get_session_binding_store()

                # Should fall back to memory store
                assert isinstance(store, MemorySessionBindingStore)
