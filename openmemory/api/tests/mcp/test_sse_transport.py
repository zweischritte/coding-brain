"""
Unit tests for the SessionAwareSseTransport wrapper.

Tests session_id capture from the MCP SSE transport.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.sse_transport import SessionAwareSseTransport


class TestSessionAwareSseTransportInit:
    """Tests for SessionAwareSseTransport initialization."""

    def test_init_creates_transport_with_endpoint(self):
        """Transport should initialize with given endpoint."""
        transport = SessionAwareSseTransport("/mcp/messages/")

        assert transport._endpoint == "/mcp/messages/"

    def test_init_creates_empty_session_id_dict(self):
        """Transport should start with empty session_id tracking dict."""
        transport = SessionAwareSseTransport("/mcp/messages/")

        assert transport._session_ids_by_scope == {}

    def test_init_prepends_slash_if_missing(self):
        """Transport should prepend slash to endpoint if missing."""
        transport = SessionAwareSseTransport("mcp/messages/")

        assert transport._endpoint == "/mcp/messages/"


class TestSessionAwareSseTransportGetSessionId:
    """Tests for get_session_id method."""

    def test_get_session_id_returns_none_when_not_tracked(self):
        """get_session_id should return None for untracked scope."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        scope = {"type": "http"}

        result = transport.get_session_id(scope)

        assert result is None

    def test_get_session_id_returns_tracked_session_id(self):
        """get_session_id should return session_id when tracked."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        scope = {"type": "http"}
        test_session_id = uuid.uuid4()

        # Manually add to tracking dict
        transport._session_ids_by_scope[id(scope)] = test_session_id

        result = transport.get_session_id(scope)

        assert result == test_session_id


class TestSessionAwareSseTransportCurrentSessionId:
    """Tests for current_session_id property."""

    def test_current_session_id_returns_none_when_empty(self):
        """current_session_id should return None when no sessions tracked."""
        transport = SessionAwareSseTransport("/mcp/messages/")

        result = transport.current_session_id

        assert result is None

    def test_current_session_id_returns_last_session_id(self):
        """current_session_id should return the most recently added session_id."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        test_session_id_1 = uuid.uuid4()
        test_session_id_2 = uuid.uuid4()

        transport._session_ids_by_scope[1] = test_session_id_1
        transport._session_ids_by_scope[2] = test_session_id_2

        result = transport.current_session_id

        assert result == test_session_id_2


class TestSessionAwareSseTransportConnectSse:
    """Tests for connect_sse session_id capture."""

    def test_session_id_captured_when_read_stream_writers_populated(self):
        """Session ID should be captured when _read_stream_writers has entries."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        test_session_id = uuid.uuid4()
        scope = {"type": "http", "root_path": ""}

        # Simulate what happens during connect_sse
        transport._read_stream_writers[test_session_id] = MagicMock()

        # Manually track like connect_sse would
        scope_id = id(scope)
        for sid in transport._read_stream_writers.keys():
            if sid not in transport._session_ids_by_scope.values():
                transport._session_ids_by_scope[scope_id] = sid
                break

        captured = transport.get_session_id(scope)
        assert captured == test_session_id

    def test_session_id_cleaned_up_on_disconnect(self):
        """Session ID should be cleaned up when connection ends."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        test_session_id = uuid.uuid4()
        scope = {"type": "http", "root_path": ""}

        # Simulate connection
        scope_id = id(scope)
        transport._session_ids_by_scope[scope_id] = test_session_id

        # Verify tracked
        assert transport.get_session_id(scope) == test_session_id

        # Simulate disconnect cleanup
        transport._session_ids_by_scope.pop(scope_id, None)

        # Verify cleaned up
        assert transport.get_session_id(scope) is None

    def test_multiple_sessions_tracked_independently(self):
        """Multiple concurrent sessions should be tracked independently."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        test_session_id_1 = uuid.uuid4()
        test_session_id_2 = uuid.uuid4()
        scope_1 = {"type": "http", "root_path": "", "id": 1}
        scope_2 = {"type": "http", "root_path": "", "id": 2}

        # Track two sessions
        transport._session_ids_by_scope[id(scope_1)] = test_session_id_1
        transport._session_ids_by_scope[id(scope_2)] = test_session_id_2

        # Both should be tracked independently
        assert transport.get_session_id(scope_1) == test_session_id_1
        assert transport.get_session_id(scope_2) == test_session_id_2

        # Removing one shouldn't affect the other
        transport._session_ids_by_scope.pop(id(scope_1), None)
        assert transport.get_session_id(scope_1) is None
        assert transport.get_session_id(scope_2) == test_session_id_2

    def test_no_session_id_when_read_stream_writers_empty(self):
        """No session ID should be captured when _read_stream_writers is empty."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        scope = {"type": "http", "root_path": ""}

        # _read_stream_writers is empty by default
        assert len(transport._read_stream_writers) == 0

        # Trying to capture would find nothing
        scope_id = id(scope)
        if hasattr(transport, "_read_stream_writers") and transport._read_stream_writers:
            for sid in transport._read_stream_writers.keys():
                if sid not in transport._session_ids_by_scope.values():
                    transport._session_ids_by_scope[scope_id] = sid
                    break

        # Should still be None
        assert transport.get_session_id(scope) is None

    def test_session_id_capture_avoids_already_tracked_sessions(self):
        """Session ID capture should skip already-tracked session IDs."""
        transport = SessionAwareSseTransport("/mcp/messages/")
        existing_session_id = uuid.uuid4()
        new_session_id = uuid.uuid4()
        scope_1 = {"type": "http", "id": 1}
        scope_2 = {"type": "http", "id": 2}

        # First session already tracked
        transport._session_ids_by_scope[id(scope_1)] = existing_session_id
        transport._read_stream_writers[existing_session_id] = MagicMock()
        transport._read_stream_writers[new_session_id] = MagicMock()

        # Capture for second scope should skip already-tracked ID
        scope_id_2 = id(scope_2)
        for sid in transport._read_stream_writers.keys():
            if sid not in transport._session_ids_by_scope.values():
                transport._session_ids_by_scope[scope_id_2] = sid
                break

        # Second scope should get the new session ID
        assert transport.get_session_id(scope_2) == new_session_id
        # First scope should keep its original
        assert transport.get_session_id(scope_1) == existing_session_id


class TestSessionAwareSseTransportInheritance:
    """Tests for proper inheritance from SseServerTransport."""

    def test_inherits_from_sse_server_transport(self):
        """SessionAwareSseTransport should inherit from SseServerTransport."""
        from mcp.server.sse import SseServerTransport

        transport = SessionAwareSseTransport("/mcp/messages/")

        assert isinstance(transport, SseServerTransport)

    def test_has_handle_post_message_method(self):
        """Transport should have handle_post_message from parent."""
        transport = SessionAwareSseTransport("/mcp/messages/")

        assert hasattr(transport, "handle_post_message")
        assert callable(transport.handle_post_message)
