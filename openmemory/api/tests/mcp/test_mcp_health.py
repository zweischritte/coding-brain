"""
Tests for MCP Session Health Endpoint (Phase 2).

Verifies the /mcp/health endpoint for session binding store status.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestMCPHealthEndpoint:
    """Test the /mcp/health endpoint."""

    @pytest.fixture
    def mock_memory_store(self):
        """Create a mock memory store."""
        store = Mock()
        store.STORE_TYPE = "memory"
        # Memory store doesn't have health_check method
        if hasattr(store, 'health_check'):
            delattr(store, 'health_check')
        return store

    @pytest.fixture
    def mock_valkey_store(self):
        """Create a mock Valkey store."""
        store = Mock()
        store.STORE_TYPE = "valkey"
        store.health_check = Mock(return_value=True)
        return store

    @pytest.fixture
    def mock_unhealthy_valkey_store(self):
        """Create a mock unhealthy Valkey store."""
        store = Mock()
        store.STORE_TYPE = "valkey"
        store.health_check = Mock(return_value=False)
        return store

    def test_health_endpoint_memory_store_healthy(self, mock_memory_store):
        """Test health endpoint returns healthy for memory store."""
        with patch("app.mcp_server.get_session_binding_store", return_value=mock_memory_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", None):
                # Import after patching
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["store_type"] == "memory"
                assert "timestamp" in data
                assert "latency_ms" in data

    def test_health_endpoint_valkey_store_healthy(self, mock_valkey_store):
        """Test health endpoint returns healthy for Valkey store."""
        with patch("app.mcp_server.get_session_binding_store", return_value=mock_valkey_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", None):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["store_type"] == "valkey"
                mock_valkey_store.health_check.assert_called_once()

    def test_health_endpoint_valkey_store_unhealthy(self, mock_unhealthy_valkey_store):
        """Test health endpoint returns unhealthy when Valkey is down."""
        with patch("app.mcp_server.get_session_binding_store", return_value=mock_unhealthy_valkey_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", None):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 503
                data = response.json()
                assert data["status"] == "unhealthy"
                assert data["store_type"] == "valkey"
                assert "reason" in data

    def test_health_endpoint_with_running_scheduler(self, mock_memory_store):
        """Test health endpoint shows scheduler status when running."""
        mock_scheduler = Mock()
        mock_scheduler._running = True

        with patch("app.mcp_server.get_session_binding_store", return_value=mock_memory_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", mock_scheduler):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                assert data["scheduler_running"] is True

    def test_health_endpoint_with_stopped_scheduler(self, mock_memory_store):
        """Test health endpoint shows scheduler status when stopped."""
        mock_scheduler = Mock()
        mock_scheduler._running = False

        with patch("app.mcp_server.get_session_binding_store", return_value=mock_memory_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", mock_scheduler):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                assert data["scheduler_running"] is False

    def test_health_endpoint_store_exception(self):
        """Test health endpoint handles store initialization errors."""
        with patch("app.mcp_server.get_session_binding_store", side_effect=Exception("Store error")):
            from app.mcp_server import mcp_router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(mcp_router)
            client = TestClient(app)

            response = client.get("/mcp/health")

            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "Store error" in data["reason"]

    def test_health_endpoint_latency_measured(self, mock_memory_store):
        """Test that latency is measured and returned."""
        with patch("app.mcp_server.get_session_binding_store", return_value=mock_memory_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", None):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                assert "latency_ms" in data
                assert isinstance(data["latency_ms"], (int, float))
                assert data["latency_ms"] >= 0

    def test_health_endpoint_timestamp_format(self, mock_memory_store):
        """Test that timestamp is in ISO format."""
        with patch("app.mcp_server.get_session_binding_store", return_value=mock_memory_store):
            with patch("app.tasks.session_cleanup._cleanup_scheduler", None):
                from app.mcp_server import mcp_router
                from fastapi import FastAPI
                from datetime import datetime

                app = FastAPI()
                app.include_router(mcp_router)
                client = TestClient(app)

                response = client.get("/mcp/health")

                assert response.status_code == 200
                data = response.json()
                # Should be parseable as ISO format
                timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                assert timestamp is not None
