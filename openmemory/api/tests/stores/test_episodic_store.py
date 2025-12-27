"""
TDD tests for ValkeyEpisodicStore.

Tests are written FIRST per strict TDD methodology.
These tests verify the Valkey-backed episodic memory store with:
- Session-scoped storage with TTL
- Tenant isolation via user_id
- Recency decay support
- Reference resolution
"""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from memory.episodic import (
    EpisodicMemory,
    SessionContext,
    EpisodicMemoryConfig,
)


# Test fixtures
TEST_USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TEST_ORG_A_ID = "11111111-1111-1111-1111-111111111111"
TEST_SESSION_A = "session-a-12345"
TEST_SESSION_B = "session-b-67890"


@pytest.fixture
def mock_valkey_client():
    """Create a mock Valkey client for unit tests."""
    client = MagicMock()

    # Store data in a dict to simulate Valkey behavior
    storage = {}
    expiry_times = {}

    def mock_get(key):
        if key in expiry_times:
            if datetime.now(timezone.utc) > expiry_times[key]:
                del storage[key]
                del expiry_times[key]
                return None
        return storage.get(key)

    def mock_set(key, value, ex=None):
        storage[key] = value
        if ex:
            expiry_times[key] = datetime.now(timezone.utc) + timedelta(seconds=ex)
        return True

    def mock_setex(key, seconds, value):
        storage[key] = value
        expiry_times[key] = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return True

    def mock_delete(*keys):
        deleted = 0
        for key in keys:
            if key in storage:
                del storage[key]
                deleted += 1
            if key in expiry_times:
                del expiry_times[key]
        return deleted

    def mock_keys(pattern):
        import fnmatch
        return [k for k in storage.keys() if fnmatch.fnmatch(k, pattern.replace("*", "*"))]

    def mock_mget(keys):
        return [storage.get(k) for k in keys]

    def mock_ttl(key):
        if key not in storage:
            return -2  # Key doesn't exist
        if key not in expiry_times:
            return -1  # Key exists but has no TTL
        remaining = (expiry_times[key] - datetime.now(timezone.utc)).total_seconds()
        return int(remaining) if remaining > 0 else -2

    def mock_expire(key, seconds):
        if key not in storage:
            return False
        expiry_times[key] = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return True

    def mock_zadd(key, mapping, nx=False, xx=False):
        if key not in storage:
            storage[key] = {}
        for member, score in mapping.items():
            if nx and member in storage[key]:
                continue
            if xx and member not in storage[key]:
                continue
            storage[key][member] = score
        return len(mapping)

    def mock_zrange(key, start, end, withscores=False, desc=False):
        if key not in storage or not isinstance(storage[key], dict):
            return []
        items = sorted(storage[key].items(), key=lambda x: x[1], reverse=desc)
        if end == -1:
            items = items[start:]
        else:
            items = items[start:end+1]
        if withscores:
            return [(k, v) for k, v in items]
        return [k for k, v in items]

    def mock_zincrby(key, amount, member):
        if key not in storage:
            storage[key] = {}
        storage[key][member] = storage[key].get(member, 0.0) + amount
        return storage[key][member]

    def mock_zrem(key, *members):
        if key not in storage or not isinstance(storage[key], dict):
            return 0
        removed = 0
        for member in members:
            if member in storage[key]:
                del storage[key][member]
                removed += 1
        return removed

    client.get = MagicMock(side_effect=mock_get)
    client.set = MagicMock(side_effect=mock_set)
    client.setex = MagicMock(side_effect=mock_setex)
    client.delete = MagicMock(side_effect=mock_delete)
    client.keys = MagicMock(side_effect=mock_keys)
    client.mget = MagicMock(side_effect=mock_mget)
    client.ttl = MagicMock(side_effect=mock_ttl)
    client.expire = MagicMock(side_effect=mock_expire)
    client.zadd = MagicMock(side_effect=mock_zadd)
    client.zrange = MagicMock(side_effect=mock_zrange)
    client.zincrby = MagicMock(side_effect=mock_zincrby)
    client.zrem = MagicMock(side_effect=mock_zrem)
    client.ping = MagicMock(return_value=True)

    # Expose storage for assertions
    client._storage = storage
    client._expiry_times = expiry_times

    return client


@pytest.fixture
def sample_memory_a():
    """Create a sample episodic memory for user A."""
    return EpisodicMemory(
        memory_id="mem-a-001",
        session_id=TEST_SESSION_A,
        user_id=TEST_USER_A_ID,
        org_id=TEST_ORG_A_ID,
        content="User A's first memory",
        recency_score=1.0,
        tool_name="chat",
    )


@pytest.fixture
def sample_memory_b():
    """Create a sample episodic memory for user B."""
    return EpisodicMemory(
        memory_id="mem-b-001",
        session_id=TEST_SESSION_B,
        user_id=TEST_USER_B_ID,
        org_id=TEST_ORG_A_ID,  # Same org, different user
        content="User B's first memory",
        recency_score=1.0,
        tool_name="search",
    )


@pytest.fixture
def sample_session_context():
    """Create a sample session context."""
    return SessionContext(
        session_id=TEST_SESSION_A,
        user_id=TEST_USER_A_ID,
        org_id=TEST_ORG_A_ID,
    )


class TestValkeyEpisodicStoreAdd:
    """Tests for ValkeyEpisodicStore.add() method."""

    def test_add_memory_stores_with_ttl(self, mock_valkey_client, sample_memory_a):
        """Adding a memory should store it with the configured TTL."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        config = EpisodicMemoryConfig(session_ttl=timedelta(hours=24))
        store = ValkeyEpisodicStore(
            client=mock_valkey_client,
            user_id=TEST_USER_A_ID,
            config=config,
        )

        store.add(sample_memory_a)

        # Verify setex was called with correct TTL
        expected_key = f"episodic:{TEST_USER_A_ID}:{TEST_SESSION_A}:mem-a-001"
        assert expected_key in mock_valkey_client._storage

    def test_add_memory_uses_correct_key_format(self, mock_valkey_client, sample_memory_a):
        """Memory key should follow format: episodic:{user_id}:{session_id}:{memory_id}."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store.add(sample_memory_a)

        expected_key = f"episodic:{TEST_USER_A_ID}:{TEST_SESSION_A}:mem-a-001"
        assert expected_key in mock_valkey_client._storage

    def test_add_memory_includes_recency_in_sorted_set(self, mock_valkey_client, sample_memory_a):
        """Adding a memory should add it to a session sorted set for recency queries."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store.add(sample_memory_a)

        # Check sorted set for session
        session_key = f"episodic:session:{TEST_USER_A_ID}:{TEST_SESSION_A}"
        assert session_key in mock_valkey_client._storage
        assert isinstance(mock_valkey_client._storage[session_key], dict)

    def test_add_memory_with_references(self, mock_valkey_client):
        """Memory with references should store reference IDs."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        memory = EpisodicMemory(
            memory_id="mem-with-refs",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="References other memories",
            references=["ref-1", "ref-2"],
        )

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store.add(memory)

        # Memory should be stored
        expected_key = f"episodic:{TEST_USER_A_ID}:{TEST_SESSION_A}:mem-with-refs"
        assert expected_key in mock_valkey_client._storage


class TestValkeyEpisodicStoreGet:
    """Tests for ValkeyEpisodicStore.get() method."""

    def test_get_returns_memory(self, mock_valkey_client, sample_memory_a):
        """get() should return the memory if it exists."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store.add(sample_memory_a)

        result = store.get("mem-a-001", session_id=TEST_SESSION_A)

        assert result is not None
        assert result.memory_id == "mem-a-001"
        assert result.content == "User A's first memory"

    def test_get_returns_none_for_nonexistent(self, mock_valkey_client):
        """get() should return None for non-existent memory."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        result = store.get("nonexistent", session_id=TEST_SESSION_A)

        assert result is None

    def test_get_returns_none_for_other_users_memory(
        self, mock_valkey_client, sample_memory_a, sample_memory_b
    ):
        """get() should return None when trying to access another user's memory."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        # Store memory as user A
        store_a = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store_a.add(sample_memory_a)

        # Try to get it as user B
        store_b = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_B_ID)
        result = store_b.get("mem-a-001", session_id=TEST_SESSION_A)

        assert result is None

    def test_get_returns_none_for_expired_memory(self, mock_valkey_client, sample_memory_a):
        """get() should return None for expired memories."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        # Use very short TTL
        config = EpisodicMemoryConfig(session_ttl=timedelta(seconds=0))
        store = ValkeyEpisodicStore(
            client=mock_valkey_client,
            user_id=TEST_USER_A_ID,
            config=config,
        )

        store.add(sample_memory_a)

        # Manually expire the key
        key = f"episodic:{TEST_USER_A_ID}:{TEST_SESSION_A}:mem-a-001"
        mock_valkey_client._expiry_times[key] = datetime.now(timezone.utc) - timedelta(seconds=1)

        result = store.get("mem-a-001", session_id=TEST_SESSION_A)

        assert result is None


class TestValkeyEpisodicStoreGetSessionMemories:
    """Tests for ValkeyEpisodicStore.get_session_memories() method."""

    def test_get_session_memories_returns_all(self, mock_valkey_client):
        """get_session_memories() should return all memories for a session."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add multiple memories
        for i in range(3):
            memory = EpisodicMemory(
                memory_id=f"mem-{i}",
                session_id=TEST_SESSION_A,
                user_id=TEST_USER_A_ID,
                org_id=TEST_ORG_A_ID,
                content=f"Memory {i}",
            )
            store.add(memory)

        result = store.get_session_memories(TEST_SESSION_A)

        assert len(result) == 3

    def test_get_session_memories_excludes_summarized(self, mock_valkey_client):
        """get_session_memories() should exclude summarized memories by default."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add one normal and one summarized memory
        memory1 = EpisodicMemory(
            memory_id="mem-1",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Normal memory",
            summarized=False,
        )
        memory2 = EpisodicMemory(
            memory_id="mem-2",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Summarized memory",
            summarized=True,
        )
        store.add(memory1)
        store.add(memory2)

        result = store.get_session_memories(TEST_SESSION_A, include_summarized=False)

        assert len(result) == 1
        assert result[0].memory_id == "mem-1"

    def test_get_session_memories_includes_summarized_when_requested(self, mock_valkey_client):
        """get_session_memories() should include summarized when requested."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add one summarized memory
        memory = EpisodicMemory(
            memory_id="mem-summarized",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Summarized memory",
            summarized=True,
        )
        store.add(memory)

        result = store.get_session_memories(TEST_SESSION_A, include_summarized=True)

        assert len(result) == 1

    def test_get_session_memories_sorted_by_recency(self, mock_valkey_client):
        """get_session_memories() should return memories sorted by recency score."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add memories with different recency scores
        for i, score in enumerate([0.5, 1.0, 0.3]):
            memory = EpisodicMemory(
                memory_id=f"mem-{i}",
                session_id=TEST_SESSION_A,
                user_id=TEST_USER_A_ID,
                org_id=TEST_ORG_A_ID,
                content=f"Memory {i}",
                recency_score=score,
            )
            store.add(memory)

        result = store.get_session_memories(TEST_SESSION_A)

        # Should be sorted by recency (highest first)
        assert result[0].recency_score >= result[1].recency_score
        assert result[1].recency_score >= result[2].recency_score

    def test_get_session_memories_empty_for_other_user(self, mock_valkey_client, sample_memory_a):
        """get_session_memories() should return empty for another user's session."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        # Add memory as user A
        store_a = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store_a.add(sample_memory_a)

        # Try to get as user B
        store_b = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_B_ID)
        result = store_b.get_session_memories(TEST_SESSION_A)

        assert result == []


class TestValkeyEpisodicStoreUpdateRecency:
    """Tests for ValkeyEpisodicStore.update_recency() method."""

    def test_update_recency_applies_decay(self, mock_valkey_client):
        """update_recency() should apply decay factor to all session memories."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        memory = EpisodicMemory(
            memory_id="mem-1",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Memory",
            recency_score=1.0,
        )
        store.add(memory)

        store.update_recency(TEST_SESSION_A, decay_factor=0.9)

        result = store.get("mem-1", session_id=TEST_SESSION_A)
        assert result.recency_score == pytest.approx(0.9, rel=0.01)

    def test_update_recency_cumulative(self, mock_valkey_client):
        """Multiple update_recency() calls should compound the decay."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        memory = EpisodicMemory(
            memory_id="mem-1",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Memory",
            recency_score=1.0,
        )
        store.add(memory)

        # Apply decay twice
        store.update_recency(TEST_SESSION_A, decay_factor=0.9)
        store.update_recency(TEST_SESSION_A, decay_factor=0.9)

        result = store.get("mem-1", session_id=TEST_SESSION_A)
        assert result.recency_score == pytest.approx(0.81, rel=0.01)


class TestValkeyEpisodicStoreMarkSummarized:
    """Tests for ValkeyEpisodicStore.mark_summarized() method."""

    def test_mark_summarized_updates_memories(self, mock_valkey_client):
        """mark_summarized() should set summarized=True on specified memories."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        memory = EpisodicMemory(
            memory_id="mem-1",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Memory",
            summarized=False,
        )
        store.add(memory)

        store.mark_summarized(["mem-1"], session_id=TEST_SESSION_A)

        result = store.get("mem-1", session_id=TEST_SESSION_A)
        assert result.summarized is True

    def test_mark_summarized_ignores_nonexistent(self, mock_valkey_client):
        """mark_summarized() should ignore non-existent memory IDs."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Should not raise
        store.mark_summarized(["nonexistent"], session_id=TEST_SESSION_A)


class TestValkeyEpisodicStoreDelete:
    """Tests for ValkeyEpisodicStore.delete() method."""

    def test_delete_removes_memory(self, mock_valkey_client, sample_memory_a):
        """delete() should remove the memory from storage."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store.add(sample_memory_a)

        result = store.delete("mem-a-001", session_id=TEST_SESSION_A)

        assert result is True
        assert store.get("mem-a-001", session_id=TEST_SESSION_A) is None

    def test_delete_returns_false_for_nonexistent(self, mock_valkey_client):
        """delete() should return False for non-existent memory."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        result = store.delete("nonexistent", session_id=TEST_SESSION_A)

        assert result is False

    def test_delete_cannot_delete_other_users_memory(
        self, mock_valkey_client, sample_memory_a
    ):
        """delete() should not delete another user's memory."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        # Add as user A
        store_a = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)
        store_a.add(sample_memory_a)

        # Try to delete as user B
        store_b = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_B_ID)
        result = store_b.delete("mem-a-001", session_id=TEST_SESSION_A)

        assert result is False

        # Memory should still exist for user A
        assert store_a.get("mem-a-001", session_id=TEST_SESSION_A) is not None


class TestValkeyEpisodicStoreExpireSession:
    """Tests for ValkeyEpisodicStore.expire_session() method."""

    def test_expire_session_removes_all_session_memories(self, mock_valkey_client):
        """expire_session() should remove all memories for a session."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add multiple memories
        for i in range(3):
            memory = EpisodicMemory(
                memory_id=f"mem-{i}",
                session_id=TEST_SESSION_A,
                user_id=TEST_USER_A_ID,
                org_id=TEST_ORG_A_ID,
                content=f"Memory {i}",
            )
            store.add(memory)

        store.expire_session(TEST_SESSION_A)

        result = store.get_session_memories(TEST_SESSION_A)
        assert result == []

    def test_expire_session_does_not_affect_other_sessions(self, mock_valkey_client):
        """expire_session() should not affect other sessions."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Add memories to different sessions
        memory_a = EpisodicMemory(
            memory_id="mem-a",
            session_id=TEST_SESSION_A,
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Session A memory",
        )
        memory_other = EpisodicMemory(
            memory_id="mem-other",
            session_id="other-session",
            user_id=TEST_USER_A_ID,
            org_id=TEST_ORG_A_ID,
            content="Other session memory",
        )
        store.add(memory_a)
        store.add(memory_other)

        store.expire_session(TEST_SESSION_A)

        # Other session should still have its memory
        result = store.get_session_memories("other-session")
        assert len(result) == 1


class TestValkeyEpisodicStoreHealth:
    """Tests for ValkeyEpisodicStore health check."""

    def test_health_check_returns_true_when_connected(self, mock_valkey_client):
        """Health check should return True when Valkey is connected."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        assert store.health_check() is True

    def test_health_check_returns_false_on_connection_error(self, mock_valkey_client):
        """Health check should return False when ping fails."""
        from app.stores.episodic_store import ValkeyEpisodicStore

        mock_valkey_client.ping.side_effect = Exception("Connection refused")
        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        assert store.health_check() is False


class TestValkeyEpisodicStoreContractCompliance:
    """Tests verifying ValkeyEpisodicStore implements the EpisodicMemoryStore ABC."""

    def test_implements_episodic_store_interface(self, mock_valkey_client):
        """ValkeyEpisodicStore should implement all EpisodicMemoryStore methods."""
        from app.stores.episodic_store import ValkeyEpisodicStore
        from memory.episodic import EpisodicMemoryStore

        store = ValkeyEpisodicStore(client=mock_valkey_client, user_id=TEST_USER_A_ID)

        # Verify it's an instance of the ABC
        assert isinstance(store, EpisodicMemoryStore)

        # Verify all required methods exist
        assert hasattr(store, 'add')
        assert hasattr(store, 'get')
        assert hasattr(store, 'get_session_memories')
        assert hasattr(store, 'update_recency')
        assert hasattr(store, 'mark_summarized')
