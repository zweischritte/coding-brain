"""
Session cleanup background task.

Periodically removes expired session bindings for memory store.
For Valkey store, TTL handles expiration automatically (cleanup is a no-op).

Phase 2: Includes metrics for cleanup cycle monitoring.
"""
import asyncio
import logging
import os
from typing import Callable, Optional, Union

from ..observability.session_metrics import record_cleanup_run
from ..security.session_binding import MemorySessionBindingStore

logger = logging.getLogger(__name__)


# Default cleanup interval (5 minutes)
DEFAULT_CLEANUP_INTERVAL = 300


def get_cleanup_interval() -> int:
    """Get the cleanup interval from environment or use default.

    Returns:
        Cleanup interval in seconds (default 300 = 5 minutes)
    """
    try:
        return int(os.environ.get("MCP_SESSION_CLEANUP_INTERVAL", str(DEFAULT_CLEANUP_INTERVAL)))
    except ValueError:
        logger.warning(
            f"Invalid MCP_SESSION_CLEANUP_INTERVAL, using default {DEFAULT_CLEANUP_INTERVAL}"
        )
        return DEFAULT_CLEANUP_INTERVAL


def cleanup_expired_sessions(store) -> int:
    """Remove expired session bindings from the store.

    For memory store, this removes expired entries.
    For Valkey store, this is a no-op since TTL handles expiration.

    Args:
        store: The session binding store (Memory or Valkey)

    Returns:
        Number of expired sessions removed
    """
    return store.cleanup_expired()


class SessionCleanupScheduler:
    """Background scheduler for periodic session cleanup.

    Runs cleanup at configurable intervals to remove expired sessions.
    For memory store, this is essential for memory management.
    For Valkey store, this is mostly for monitoring/logging purposes.
    """

    def __init__(
        self,
        store,
        interval_seconds: Optional[int] = None,
        cleanup_fn: Optional[Callable] = None,
    ):
        """Initialize the cleanup scheduler.

        Args:
            store: The session binding store to clean
            interval_seconds: Cleanup interval (default from env or 300s)
            cleanup_fn: Custom cleanup function (default: cleanup_expired_sessions)
        """
        self.store = store
        self.interval = interval_seconds or get_cleanup_interval()
        self.cleanup_fn = cleanup_fn or cleanup_expired_sessions
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background cleanup scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Session cleanup scheduler started (interval: {self.interval}s)"
        )

    async def stop(self) -> None:
        """Stop the background cleanup scheduler."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("Session cleanup scheduler stopped")

    async def _run_loop(self) -> None:
        """Main cleanup loop with metrics."""
        while self._running:
            try:
                removed = self.cleanup_fn(self.store)
                if removed > 0:
                    logger.info(f"Session cleanup: removed {removed} expired sessions")
                else:
                    logger.debug("Session cleanup: no expired sessions")
                # Record successful cleanup run
                record_cleanup_run(success=True)
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                # Record failed cleanup run
                record_cleanup_run(success=False)

            # Wait for next interval
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break


# Global scheduler instance
_cleanup_scheduler: Optional[SessionCleanupScheduler] = None


async def start_cleanup_scheduler(store) -> SessionCleanupScheduler:
    """Start the global cleanup scheduler.

    Args:
        store: The session binding store to clean

    Returns:
        The started scheduler instance
    """
    global _cleanup_scheduler
    if _cleanup_scheduler is None:
        _cleanup_scheduler = SessionCleanupScheduler(store)
        await _cleanup_scheduler.start()
    return _cleanup_scheduler


async def stop_cleanup_scheduler() -> None:
    """Stop the global cleanup scheduler."""
    global _cleanup_scheduler
    if _cleanup_scheduler is not None:
        await _cleanup_scheduler.stop()
        _cleanup_scheduler = None
