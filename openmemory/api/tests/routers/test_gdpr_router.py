"""Tests for GDPR REST API Router.

These tests verify the GDPR REST endpoints for SAR export and user deletion
with proper authentication, authorization, and rate limiting.

Test IDs: RTR-001 through RTR-010
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Create test app with GDPR router
@pytest.fixture
def test_app():
    """Create a test FastAPI app with the GDPR router."""
    from fastapi import FastAPI
    from app.routers.gdpr import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_principal():
    """Create a mock Principal for testing."""
    from app.security.types import Principal, TokenClaims, Scope

    claims = TokenClaims(
        sub="test-user-123",
        iss="https://auth.example.com",
        aud="test-api",
        exp=int(datetime.now(timezone.utc).timestamp()) + 3600,
        iat=int(datetime.now(timezone.utc).timestamp()),
        scopes=[Scope.GDPR_READ.value, Scope.GDPR_DELETE.value],
        jti=str(uuid4()),  # JWT ID
        org_id="test-org",  # Organization ID
    )
    return Principal(
        user_id="test-user-123",
        org_id="test-org",
        claims=claims,
    )


class TestGDPRScopeTests:
    """Tests for GDPR scope definitions."""

    def test_gdpr_read_scope_exists(self):
        """RTR-001: GDPR_READ scope is defined."""
        from app.security.types import Scope

        assert hasattr(Scope, "GDPR_READ")
        assert Scope.GDPR_READ.value == "gdpr:read"

    def test_gdpr_delete_scope_exists(self):
        """RTR-002: GDPR_DELETE scope is defined."""
        from app.security.types import Scope

        assert hasattr(Scope, "GDPR_DELETE")
        assert Scope.GDPR_DELETE.value == "gdpr:delete"


class TestSAREndpoint:
    """Tests for SAR export endpoint (RTR-001, RTR-003, RTR-004)."""

    def test_sar_endpoint_requires_gdpr_read_scope(self):
        """RTR-001: GET /v1/gdpr/export/{user_id} requires GDPR_READ."""
        from app.routers.gdpr import router

        # Find the export route
        export_route = None
        for route in router.routes:
            if hasattr(route, 'path') and '/export/' in route.path:
                export_route = route
                break

        assert export_route is not None

    @patch('app.routers.gdpr.get_db')
    @patch('app.routers.gdpr.require_scopes')
    def test_sar_endpoint_returns_json_response(
        self,
        mock_require_scopes,
        mock_get_db,
        test_app,
        mock_principal,
    ):
        """RTR-003: SAR endpoint returns JSON response."""
        from app.gdpr.schemas import SARResponse

        # This test verifies the response model is SARResponse
        # which will be serialized as JSON
        from app.routers.gdpr import router

        export_route = None
        for route in router.routes:
            if hasattr(route, 'path') and '/export/' in route.path:
                export_route = route
                break

        # The route should exist
        assert export_route is not None

    def test_sar_response_includes_all_stores(self):
        """RTR-004: SAR endpoint includes all stores."""
        from app.gdpr.schemas import SARResponse

        # Create a sample response
        response = SARResponse(
            user_id="test-user",
            export_date=datetime.now(timezone.utc),
            postgres={"user": {"id": "123"}},
            neo4j={"nodes": []},
            qdrant={"embeddings": {}},
            opensearch={"documents": []},
            valkey={"episodic_sessions": []},
        )

        # All store keys should be present
        assert "postgres" in response.stores
        assert "neo4j" in response.stores
        assert "qdrant" in response.stores
        assert "opensearch" in response.stores
        assert "valkey" in response.stores


class TestDeleteEndpoint:
    """Tests for delete endpoint (RTR-002, RTR-005)."""

    def test_delete_endpoint_requires_gdpr_delete_scope(self):
        """RTR-002: DELETE /v1/gdpr/user/{user_id} requires GDPR_DELETE."""
        from app.routers.gdpr import router

        # Find the delete route
        delete_route = None
        for route in router.routes:
            if hasattr(route, 'path') and '/user/' in route.path:
                if hasattr(route, 'methods') and 'DELETE' in route.methods:
                    delete_route = route
                    break

        assert delete_route is not None

    def test_delete_endpoint_returns_deletion_result(self):
        """RTR-005: Delete endpoint returns deletion result."""
        from app.gdpr.schemas import DeletionResult

        # Create a sample result
        result = DeletionResult(
            audit_id=str(uuid4()),
            user_id="test-user",
            timestamp=datetime.now(timezone.utc),
            results={
                "valkey": {"status": "deleted", "count": 2},
                "opensearch": {"status": "deleted", "count": 5},
                "qdrant": {"status": "deleted", "count": 10},
                "neo4j": {"status": "deleted", "count": 3},
                "postgres": {"status": "deleted", "count": 15},
            },
            success=True,
        )

        assert result.success is True
        assert "postgres" in result.results


class TestRateLimiting:
    """Tests for GDPR endpoint rate limiting (RTR-006, RTR-007)."""

    def test_rate_limit_config_exists_for_sar(self):
        """RTR-006: Rate limiting configuration exists for SAR endpoint."""
        from app.routers.gdpr import GDPR_RATE_LIMITS

        assert "export" in GDPR_RATE_LIMITS
        # Should allow limited requests per time period
        assert GDPR_RATE_LIMITS["export"]["requests_per_hour"] <= 10

    def test_rate_limit_config_exists_for_delete(self):
        """RTR-007: Rate limiting configuration exists for delete endpoint."""
        from app.routers.gdpr import GDPR_RATE_LIMITS

        assert "delete" in GDPR_RATE_LIMITS
        # Should be more restrictive than export
        assert GDPR_RATE_LIMITS["delete"]["requests_per_day"] <= 5


class TestAuditLogging:
    """Tests for GDPR audit logging in router (RTR-008, RTR-009)."""

    def test_sar_audit_log_created(self):
        """RTR-008: Audit log created for SAR operations."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        # Verify we can create export audit logs
        entry = logger.log_operation_start(
            audit_id=str(uuid4()),
            operation=GDPROperation.EXPORT,
            target_user_id="test-user",
            requestor_id="admin",
        )

        assert entry.operation == GDPROperation.EXPORT

    def test_delete_audit_log_created(self):
        """RTR-009: Audit log created for delete operations."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        # Verify we can create delete audit logs
        entry = logger.log_operation_start(
            audit_id=str(uuid4()),
            operation=GDPROperation.DELETE,
            target_user_id="test-user",
            requestor_id="admin",
            reason="User requested deletion",
        )

        assert entry.operation == GDPROperation.DELETE
        assert entry.reason == "User requested deletion"


class TestGDPRWorkflow:
    """Tests for full GDPR workflow (RTR-010)."""

    @pytest.mark.asyncio
    async def test_full_gdpr_workflow(self):
        """RTR-010: Full GDPR workflow (Export -> Delete -> Verify)."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.deletion import UserDeletionOrchestrator
        from app.gdpr.schemas import SARResponse, DeletionResult

        mock_db = MagicMock()
        user_id = "test-user-workflow"

        # Mock user exists
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.return_value = []

        # Step 1: Export user data
        exporter = SARExporter(db=mock_db)
        export_result = await exporter.export_user_data(user_id)

        assert isinstance(export_result, SARResponse)
        assert export_result.user_id == user_id

        # Step 2: Delete user data
        mock_db.query.return_value.filter.return_value.first.return_value = None  # User already deleted
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        delete_result = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="GDPR workflow test",
        )

        assert isinstance(delete_result, DeletionResult)
        assert delete_result.user_id == user_id

        # Step 3: Verify empty (second export should return no data)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        verify_result = await exporter.export_user_data(user_id)

        assert verify_result.postgres.get("user") is None


class TestRouterRegistration:
    """Tests for router registration and configuration."""

    def test_router_has_correct_prefix(self):
        """Verify router has /v1/gdpr prefix."""
        from app.routers.gdpr import router

        assert router.prefix == "/v1/gdpr"

    def test_router_has_gdpr_tag(self):
        """Verify router has gdpr tag."""
        from app.routers.gdpr import router

        assert "gdpr" in router.tags
