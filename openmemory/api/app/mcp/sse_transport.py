"""
SSE transport wrapper for session ID capture.

The MCP library's SseServerTransport generates session_id internally but doesn't
expose it. This wrapper captures it for session binding.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from urllib.parse import quote
from uuid import UUID, uuid4

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.sse import SseServerTransport
from sse_starlette import EventSourceResponse
from starlette.requests import Request

logger = logging.getLogger(__name__)


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

        if scope["type"] != "http":
            logger.error("connect_sse received non-HTTP request")
            raise ValueError("connect_sse can only handle HTTP requests")

        request = Request(scope, receive)
        error_response = await self._security.validate_request(request, is_post=False)
        if error_response:
            await error_response(scope, receive, send)
            raise ValueError("Request validation failed")

        logger.debug("Setting up SSE connection")
        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        session_id = uuid4()
        self._read_stream_writers[session_id] = read_stream_writer
        self._session_ids_by_scope[scope_id] = session_id
        logger.debug("Created new session with ID: %s", session_id)

        root_path = scope.get("root_path", "")
        full_message_path_for_client = root_path.rstrip("/") + self._endpoint
        client_post_uri_data = f"{quote(full_message_path_for_client)}?session_id={session_id.hex}"

        sse_stream_writer, sse_stream_reader = anyio.create_memory_object_stream[dict[str, Any]](0)

        async def sse_writer():
            logger.debug("Starting SSE writer")
            async with sse_stream_writer, write_stream_reader:
                await sse_stream_writer.send({"event": "endpoint", "data": client_post_uri_data})
                logger.debug("Sent endpoint event: %s", client_post_uri_data)

                async for session_message in write_stream_reader:
                    logger.debug("Sending message via SSE: %s", session_message)
                    await sse_stream_writer.send(
                        {
                            "event": "message",
                            "data": session_message.message.model_dump_json(by_alias=True, exclude_none=True),
                        }
                    )

        async with anyio.create_task_group() as tg:

            async def response_wrapper(scope, receive, send):
                await EventSourceResponse(content=sse_stream_reader, data_sender_callable=sse_writer)(
                    scope, receive, send
                )
                await read_stream_writer.aclose()
                await write_stream_reader.aclose()
                logger.debug("Client session disconnected %s", session_id)

            logger.debug("Starting SSE response task")
            tg.start_soon(response_wrapper, scope, receive, send)

            logger.debug("Yielding read and write streams")
            try:
                yield (read_stream, write_stream)
            finally:
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
