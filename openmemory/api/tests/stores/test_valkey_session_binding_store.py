"""
Unit tests for the Valkey-backed session binding store.

Tests the ValkeySessionBindingStore for MCP SSE session binding in multi-worker deployments.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# Test constants
TEST_SESSION_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_SESSION_ID_2 = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
TEST_USER_ID = "user-123"
TEST_USER_ID_2 = "user-456"
TEST_ORG_ID = "org-abc"
TEST_ORG_ID_2 = "org-xyz"
TEST_DPOP_THUMBPRINT = "test-dpop-thumbprint-abc123"


class MockValkeyClient:
    """Mock Valkey client for testing without real Redis connection."""

    def __init__(self):
        self._data: dict[str, tuple[str, Optional[float]]] = {}  # key -> (value, expire_at)
        self._raise_on_next = False

    def get(self, key: str) -> Optional[bytes]:
        """Get a value from the mock store."""
        if self._raise_on_next:
            self._raise_on_next = False
            raise ConnectionError("Mock connection error")
        item = self._data.get(key)
        if item is None:
            return None
        value, expire_at = item
        if expire_at and datetime.now(timezone.utc).timestamp() > expire_at:
            del self._data[key]
            return None
        return value.encode("utf-8")

    def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set a value with expiration."""
        if self._raise_on_next:
            self._raise_on_next = False
            raise ConnectionError("Mock connection error")
        expire_at = datetime.now(timezone.utc).timestamp() + seconds
        self._data[key] = (value, expire_at)
        return True

    def delete(self, *keys: str) -> int:
        """Delete keys from the store."""
        if self._raise_on_next:
            self._raise_on_next = False
            raise ConnectionError("Mock connection error")
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
        return deleted

    def keys(self, pattern: str) -> list[bytes]:
        """Get keys matching pattern (simple prefix match for testing)."""
        if self._raise_on_next:
            self._raise_on_next = False
            raise ConnectionError("Mock connection error")
        prefix = pattern.replace("*", "")
        return [k.encode("utf-8") for k in self._data.keys() if k.startswith(prefix)]

    def ping(self) -> bool:
        """Health check."""
        if self._raise_on_next:
            self._raise_on_next = False
            raise ConnectionError("Mock connection error")
        return True

    def simulate_error(self):
        """Simulate a connection error on next operation."""
        self._raise_on_next = True


class TestValkeySessionBindingCreate:
    """Tests for Valkey session binding creation."""

    def test_create_binding_stores_all_fields(self):
        """Create binding should store session_id, user_id, org_id, and timestamps."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client, default_ttl_seconds=1800)

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
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        binding = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        assert binding.dpop_thumbprint == TEST_DPOP_THUMBPRINT

    def test_create_binding_with_custom_ttl(self):
        """Create binding should use custom TTL when provided."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client, default_ttl_seconds=1800)
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

    def test_create_binding_uses_correct_key_format(self):
        """Create binding should use key format: mcp:session:{session_id}."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        expected_key = f"mcp:session:{TEST_SESSION_ID}"
        assert expected_key in client._data

    def test_create_binding_serializes_to_json(self):
        """Create binding should store JSON-serialized data."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        key = f"mcp:session:{TEST_SESSION_ID}"
        stored_value, _ = client._data[key]
        data = json.loads(stored_value)

        assert data["user_id"] == TEST_USER_ID
        assert data["org_id"] == TEST_ORG_ID
        assert "issued_at" in data
        assert "expires_at" in data

    def test_create_binding_handles_connection_error(self):
        """Create binding should fail closed on connection error."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        client.simulate_error()

        # Should raise or return None on error (fail closed)
        with pytest.raises(Exception):
            store.create(
                session_id=TEST_SESSION_ID,
                user_id=TEST_USER_ID,
                org_id=TEST_ORG_ID,
            )


class TestValkeySessionBindingGet:
    """Tests for Valkey session binding retrieval."""

    def test_get_existing_binding_returns_binding(self):
        """Get should return binding when it exists and is not expired."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
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
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        binding = store.get(TEST_SESSION_ID)

        assert binding is None

    def test_get_deserializes_json_correctly(self):
        """Get should correctly deserialize JSON data including datetime fields."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        created = store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        retrieved = store.get(TEST_SESSION_ID)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.user_id == created.user_id
        assert retrieved.org_id == created.org_id
        assert retrieved.dpop_thumbprint == created.dpop_thumbprint
        # Datetime comparison with tolerance
        assert abs((retrieved.issued_at - created.issued_at).total_seconds()) < 1
        assert abs((retrieved.expires_at - created.expires_at).total_seconds()) < 1

    def test_get_handles_connection_error_returns_none(self):
        """Get should return None on connection error (fail closed)."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        client.simulate_error()

        binding = store.get(TEST_SESSION_ID)

        # Fail closed: return None on error
        assert binding is None


class TestValkeySessionBindingValidate:
    """Tests for Valkey session binding validation."""

    def test_validate_matching_principal_returns_true(self):
        """Validate should return True when user_id and org_id match."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is True

    def test_validate_mismatched_user_id_returns_false(self):
        """Validate should return False when user_id doesn't match."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID_2, TEST_ORG_ID)

        assert result is False

    def test_validate_mismatched_org_id_returns_false(self):
        """Validate should return False when org_id doesn't match."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID_2)

        assert result is False

    def test_validate_nonexistent_session_returns_false(self):
        """Validate should return False when session doesn't exist."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is False

    def test_validate_with_matching_dpop_thumbprint_returns_true(self):
        """Validate should return True when DPoP thumbprint matches."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
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
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
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
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=TEST_DPOP_THUMBPRINT,
        )

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        assert result is False

    def test_validate_handles_connection_error_returns_false(self):
        """Validate should return False on connection error (fail closed)."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        client.simulate_error()

        result = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        # Fail closed: return False on error
        assert result is False


class TestValkeySessionBindingDelete:
    """Tests for Valkey session binding deletion."""

    def test_delete_existing_binding_returns_true(self):
        """Delete should return True when binding exists."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        result = store.delete(TEST_SESSION_ID)

        assert result is True

    def test_delete_removes_binding(self):
        """Delete should remove binding from store."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        store.delete(TEST_SESSION_ID)

        assert store.get(TEST_SESSION_ID) is None

    def test_delete_nonexistent_binding_returns_false(self):
        """Delete should return False when binding doesn't exist."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        result = store.delete(TEST_SESSION_ID)

        assert result is False

    def test_delete_handles_connection_error_returns_false(self):
        """Delete should return False on connection error."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        client.simulate_error()

        result = store.delete(TEST_SESSION_ID)

        assert result is False


class TestValkeySessionBindingCleanup:
    """Tests for Valkey session binding cleanup."""

    def test_cleanup_expired_returns_zero_for_valkey(self):
        """Cleanup should return 0 for Valkey (TTL handles expiration automatically)."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        removed = store.cleanup_expired()

        # Valkey handles TTL automatically, no manual cleanup needed
        assert removed == 0


class TestValkeySessionBindingHealthCheck:
    """Tests for Valkey connection health check."""

    def test_health_check_returns_true_when_connected(self):
        """Health check should return True when Valkey is connected."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)

        assert store.health_check() is True

    def test_health_check_returns_false_on_connection_error(self):
        """Health check should return False when Valkey is disconnected."""
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        client = MockValkeyClient()
        store = ValkeySessionBindingStore(client)
        client.simulate_error()

        assert store.health_check() is False
