"""
SSE transport wrapper for session ID capture.

The MCP library's SseServerTransport generates session_id internally but doesn't
expose it. This wrapper captures it for session binding.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional, Tuple
from uuid import UUID

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.sse import SseServerTransport


class SessionAwareSseTransport(SseServerTransport):
    """SSE transport wrapper that exposes session_id for binding.

    The session_id is generated inside connect_sse() and stored in the
    _read_stream_writers dict. This wrapper captures it for use by the
    session binding system.
    """

    def __init__(self, endpoint: str):
        super().__init__(endpoint)
        self._session_ids_by_scope: Dict[int, UUID] = {}

    @asynccontextmanager
    async def connect_sse(
        self, scope, receive, send
    ) -> AsyncGenerator[Tuple[MemoryObjectReceiveStream, MemoryObjectSendStream], None]:
        """Wrap connect_sse to capture session_id before yielding."""
        scope_id = id(scope)

        async with super().connect_sse(scope, receive, send) as streams:
            # After connect, find the newly added session_id
            # The library stores it in _read_stream_writers dict
            if hasattr(self, "_read_stream_writers") and self._read_stream_writers:
                # Find session_id not already tracked
                for sid in self._read_stream_writers.keys():
                    if sid not in self._session_ids_by_scope.values():
                        self._session_ids_by_scope[scope_id] = sid
                        break

            yield streams

            # Cleanup after connection ends
            self._session_ids_by_scope.pop(scope_id, None)

    def get_session_id(self, scope) -> Optional[UUID]:
        """Get the session_id associated with a request scope."""
        return self._session_ids_by_scope.get(id(scope))

    @property
    def current_session_id(self) -> Optional[UUID]:
        """Get the most recently captured session_id.

        DEPRECATED: Use get_session_id(scope) instead for multi-connection safety.
        This property is kept for backwards compatibility but may return
        incorrect results under concurrent connections.
        """
        if self._session_ids_by_scope:
            return list(self._session_ids_by_scope.values())[-1]
        return None
