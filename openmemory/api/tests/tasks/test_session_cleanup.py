"""
Tests for session cleanup background task.

The cleanup task periodically removes expired session bindings for memory store.
For Valkey store, TTL handles expiration automatically.
"""
import asyncio
import time
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.security.session_binding import (
    MemorySessionBindingStore,
    reset_session_binding_store,
)


# Test constants
TEST_SESSION_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_USER_ID = "user-123"
TEST_ORG_ID = "org-abc"


class TestSessionCleanupTask:
    """Tests for the session cleanup background task."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_cleanup_removes_expired_sessions(self):
        """Cleanup should remove expired session bindings."""
        from app.tasks.session_cleanup import cleanup_expired_sessions

        # Create store with 1 second TTL
        store = MemorySessionBindingStore(default_ttl_seconds=1)
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        # Wait for expiration
        time.sleep(1.1)

        # Run cleanup
        removed = cleanup_expired_sessions(store)

        assert removed == 1
        assert store.get(TEST_SESSION_ID) is None

    def test_cleanup_preserves_valid_sessions(self):
        """Cleanup should not remove non-expired sessions."""
        from app.tasks.session_cleanup import cleanup_expired_sessions

        store = MemorySessionBindingStore(default_ttl_seconds=3600)
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        removed = cleanup_expired_sessions(store)

        assert removed == 0
        assert store.get(TEST_SESSION_ID) is not None

    def test_cleanup_returns_count_of_removed(self):
        """Cleanup should return the count of removed bindings."""
        from app.tasks.session_cleanup import cleanup_expired_sessions

        store = MemorySessionBindingStore(default_ttl_seconds=1)
        session_ids = [uuid.uuid4() for _ in range(5)]

        for sid in session_ids:
            store.create(sid, TEST_USER_ID, TEST_ORG_ID)

        time.sleep(1.1)

        removed = cleanup_expired_sessions(store)

        assert removed == 5

    def test_cleanup_with_empty_store(self):
        """Cleanup should handle empty store gracefully."""
        from app.tasks.session_cleanup import cleanup_expired_sessions

        store = MemorySessionBindingStore()
        removed = cleanup_expired_sessions(store)

        assert removed == 0


class TestSessionCleanupWithValkeyStore:
    """Tests for cleanup with Valkey store (TTL handles expiration)."""

    def test_cleanup_returns_zero_for_valkey(self):
        """Cleanup should return 0 for Valkey store (TTL handles it)."""
        from app.tasks.session_cleanup import cleanup_expired_sessions
        from app.security.valkey_session_binding import ValkeySessionBindingStore

        # Create mock Valkey client
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        store = ValkeySessionBindingStore(mock_client, default_ttl_seconds=1800)
        removed = cleanup_expired_sessions(store)

        # Valkey TTL handles expiration, so cleanup returns 0
        assert removed == 0


class TestCleanupScheduler:
    """Tests for the cleanup scheduler functionality."""

    @pytest.mark.asyncio
    async def test_scheduler_runs_at_interval(self):
        """Scheduler should run cleanup at configured interval."""
        from app.tasks.session_cleanup import SessionCleanupScheduler

        store = MemorySessionBindingStore(default_ttl_seconds=1)
        cleanup_calls = []

        def track_cleanup(s):
            cleanup_calls.append(1)
            return 0

        scheduler = SessionCleanupScheduler(
            store=store,
            interval_seconds=0.1,  # 100ms for testing
            cleanup_fn=track_cleanup,
        )

        # Start scheduler
        await scheduler.start()

        # Wait for a few cleanup cycles
        await asyncio.sleep(0.35)

        # Stop scheduler
        await scheduler.stop()

        # Should have run at least 2 times (immediate + interval)
        assert len(cleanup_calls) >= 2

    @pytest.mark.asyncio
    async def test_scheduler_can_be_stopped(self):
        """Scheduler should stop cleanly when requested."""
        from app.tasks.session_cleanup import SessionCleanupScheduler

        store = MemorySessionBindingStore()
        scheduler = SessionCleanupScheduler(
            store=store,
            interval_seconds=0.1,
        )

        await scheduler.start()
        await scheduler.stop()

        assert scheduler._task is None or scheduler._task.done()

    @pytest.mark.asyncio
    async def test_scheduler_logs_cleanup_stats(self):
        """Scheduler should log cleanup statistics."""
        from app.tasks.session_cleanup import SessionCleanupScheduler

        store = MemorySessionBindingStore(default_ttl_seconds=1)
        store.create(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)

        time.sleep(1.1)

        with patch("app.tasks.session_cleanup.logger") as mock_logger:
            scheduler = SessionCleanupScheduler(
                store=store,
                interval_seconds=0.1,
            )

            await scheduler.start()
            await asyncio.sleep(0.15)  # Wait for first cleanup
            await scheduler.stop()

            # Should have logged cleanup info
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_scheduler_handles_cleanup_errors_gracefully(self):
        """Scheduler should continue running even if cleanup raises."""
        from app.tasks.session_cleanup import SessionCleanupScheduler

        store = MemorySessionBindingStore()
        error_count = [0]
        success_count = [0]

        def failing_cleanup(s):
            if error_count[0] < 2:
                error_count[0] += 1
                raise RuntimeError("Test error")
            success_count[0] += 1
            return 0

        scheduler = SessionCleanupScheduler(
            store=store,
            interval_seconds=0.05,
            cleanup_fn=failing_cleanup,
        )

        await scheduler.start()
        await asyncio.sleep(0.2)
        await scheduler.stop()

        # Should have recovered and continued after errors
        assert error_count[0] >= 2
        assert success_count[0] >= 1


class TestGetDefaultCleanupInterval:
    """Tests for default cleanup interval configuration."""

    def test_default_interval_is_5_minutes(self):
        """Default cleanup interval should be 5 minutes (300 seconds)."""
        from app.tasks.session_cleanup import get_cleanup_interval

        with patch.dict("os.environ", {}, clear=True):
            interval = get_cleanup_interval()
            assert interval == 300

    def test_interval_from_env_var(self):
        """Cleanup interval should be configurable via env var."""
        from app.tasks.session_cleanup import get_cleanup_interval

        with patch.dict("os.environ", {"MCP_SESSION_CLEANUP_INTERVAL": "60"}):
            interval = get_cleanup_interval()
            assert interval == 60

    def test_invalid_env_var_uses_default(self):
        """Invalid env var should fall back to default."""
        from app.tasks.session_cleanup import get_cleanup_interval

        with patch.dict("os.environ", {"MCP_SESSION_CLEANUP_INTERVAL": "invalid"}):
            interval = get_cleanup_interval()
            assert interval == 300
