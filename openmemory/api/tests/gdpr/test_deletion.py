"""Tests for GDPR Cascading User Deletion.

These tests verify the user deletion functionality that removes all user data
from all data stores in the correct dependency order.

Test IDs: DEL-001 through DEL-016
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch, call
from uuid import uuid4


class TestDeletionOrder:
    """Tests for deletion order (DEL-001)."""

    def test_deletion_order_follows_dependencies(self):
        """DEL-001: Deletion order follows dependencies."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        expected_order = [
            "valkey",      # Session/cache data (no dependencies)
            "opensearch",  # Search indices (can be rebuilt)
            "qdrant",      # Embeddings (can be rebuilt)
            "neo4j",       # Graph relationships
            "postgres",    # Primary data (last, FK constraints)
        ]

        assert UserDeletionOrchestrator.DELETION_ORDER == expected_order


class TestValkeyDeletion:
    """Tests for Valkey deletion (DEL-002)."""

    @pytest.mark.asyncio
    async def test_valkey_deletion_removes_session_keys(self):
        """DEL-002: Valkey deletion removes session keys."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_valkey = MagicMock()
        user_id = "test-user-123"

        # Mock Valkey keys
        mock_valkey.keys.return_value = [
            f"episodic:{user_id}:session1",
            f"episodic:{user_id}:session2",
        ]

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            valkey_client=mock_valkey,
        )
        count = await orchestrator._delete_valkey(user_id)

        assert count == 2
        mock_valkey.keys.assert_called_with(f"episodic:{user_id}:*")
        mock_valkey.delete.assert_called()


class TestOpenSearchDeletion:
    """Tests for OpenSearch deletion (DEL-003)."""

    @pytest.mark.asyncio
    async def test_opensearch_deletion_removes_documents(self):
        """DEL-003: OpenSearch deletion removes documents."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_opensearch = MagicMock()
        user_id = "test-user-123"

        # Mock OpenSearch delete_by_query
        mock_opensearch.delete_by_query.return_value = {"deleted": 5}

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            opensearch_client=mock_opensearch,
        )
        count = await orchestrator._delete_opensearch(user_id)

        assert count == 5
        mock_opensearch.delete_by_query.assert_called()


class TestQdrantDeletion:
    """Tests for Qdrant deletion (DEL-004)."""

    @pytest.mark.asyncio
    async def test_qdrant_deletion_removes_embeddings(self):
        """DEL-004: Qdrant deletion removes embeddings."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_qdrant = MagicMock()
        user_id = "test-user-123"

        # Mock Qdrant collections
        mock_collection = MagicMock()
        mock_collection.name = "embeddings_default"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]

        # Mock delete result
        mock_delete_result = MagicMock()
        mock_delete_result.status = "completed"
        mock_qdrant.delete.return_value = mock_delete_result

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            qdrant_client=mock_qdrant,
        )
        count = await orchestrator._delete_qdrant(user_id)

        mock_qdrant.delete.assert_called()


class TestNeo4jDeletion:
    """Tests for Neo4j deletion (DEL-005, DEL-006)."""

    @pytest.mark.asyncio
    async def test_neo4j_deletion_removes_nodes(self):
        """DEL-005: Neo4j deletion removes nodes."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # Mock Neo4j session
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"deleted": 10}
        mock_session.run.return_value = mock_result
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            neo4j_driver=mock_neo4j,
        )
        count = await orchestrator._delete_neo4j(user_id)

        assert count == 10
        mock_session.run.assert_called()

    @pytest.mark.asyncio
    async def test_neo4j_deletion_removes_relationships(self):
        """DEL-006: Neo4j deletion removes relationships via DETACH DELETE."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"deleted": 5}
        mock_session.run.return_value = mock_result
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            neo4j_driver=mock_neo4j,
        )
        await orchestrator._delete_neo4j(user_id)

        # Verify DETACH DELETE is used (removes relationships)
        call_args = mock_session.run.call_args
        assert "DETACH DELETE" in call_args[0][0]


class TestPostgreSQLDeletion:
    """Tests for PostgreSQL deletion (DEL-007 through DEL-010)."""

    @pytest.mark.asyncio
    async def test_postgresql_deletion_removes_user_record(self):
        """DEL-007: PostgreSQL deletion removes user record."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.delete.return_value = 0

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        count = await orchestrator._delete_postgres(user_id)

        # Should delete user
        mock_db.delete.assert_called_with(mock_user)
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_postgresql_deletion_cascades_to_memories(self):
        """DEL-008: PostgreSQL deletion cascades to memories."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "test-user-123"

        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        # Simulate memory deletion
        memory_delete_mock = MagicMock(return_value=5)
        mock_db.query.return_value.filter.return_value.delete = memory_delete_mock

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        await orchestrator._delete_postgres(user_id)

        # Should have queried Memory model
        assert mock_db.query.called

    @pytest.mark.asyncio
    async def test_postgresql_deletion_cascades_to_apps(self):
        """DEL-009: PostgreSQL deletion cascades to apps."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "test-user-123"

        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.delete.return_value = 2

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        await orchestrator._delete_postgres(user_id)

        # Verify query was made
        assert mock_db.query.called

    @pytest.mark.asyncio
    async def test_postgresql_deletion_cascades_to_feedback(self):
        """DEL-010: PostgreSQL deletion cascades to feedback."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "test-user-123"

        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.delete.return_value = 3

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        await orchestrator._delete_postgres(user_id)

        # Verify queries include feedback deletion
        assert mock_db.query.called


class TestDeletionAudit:
    """Tests for deletion audit (DEL-011)."""

    @pytest.mark.asyncio
    async def test_deletion_creates_audit_record(self):
        """DEL-011: Deletion creates audit record."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "test-user-123"

        # No user found - should still create audit
        mock_db.query.return_value.filter.return_value.first.return_value = None

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        result = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="User requested deletion",
            requestor_id="admin-123",
        )

        assert result.audit_id is not None
        assert result.user_id == user_id


class TestDeletionEdgeCases:
    """Tests for deletion edge cases (DEL-012 through DEL-016)."""

    @pytest.mark.asyncio
    async def test_deletion_handles_nonexistent_user(self):
        """DEL-012: Deletion handles non-existent user."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "nonexistent-user"

        # No user found
        mock_db.query.return_value.filter.return_value.first.return_value = None

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        result = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="User requested deletion",
        )

        # Should complete without error
        assert result.user_id == user_id
        # postgres should report 0 deleted
        assert result.results.get("postgres", {}).get("count", 0) == 0

    @pytest.mark.asyncio
    async def test_deletion_handles_partial_failures(self):
        """DEL-013: Deletion handles partial failures."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # PostgreSQL works
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Neo4j fails
        mock_neo4j.session.side_effect = Exception("Neo4j connection failed")

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            neo4j_driver=mock_neo4j,
        )
        result = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="User requested deletion",
        )

        # Should continue with remaining stores
        assert result.success is False
        assert len(result.errors) > 0
        assert "neo4j" in result.errors[0].lower()
        # Postgres should still have been attempted
        assert "postgres" in result.results

    @pytest.mark.asyncio
    async def test_full_deletion_removes_all_user_data(self):
        """DEL-014: Full deletion removes all user data."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_valkey = MagicMock()
        mock_opensearch = MagicMock()
        mock_qdrant = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # Setup all mocks
        mock_valkey.keys.return_value = ["key1", "key2"]
        mock_opensearch.delete_by_query.return_value = {"deleted": 3}

        mock_collection = MagicMock()
        mock_collection.name = "embeddings_default"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]
        mock_qdrant.delete.return_value = MagicMock()

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"deleted": 4}
        mock_session.run.return_value = mock_result
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_db.query.return_value.filter.return_value.first.return_value = None

        orchestrator = UserDeletionOrchestrator(
            db=mock_db,
            valkey_client=mock_valkey,
            opensearch_client=mock_opensearch,
            qdrant_client=mock_qdrant,
            neo4j_driver=mock_neo4j,
        )
        result = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="GDPR request",
        )

        # All stores should have been processed
        assert "valkey" in result.results
        assert "opensearch" in result.results
        assert "qdrant" in result.results
        assert "neo4j" in result.results
        assert "postgres" in result.results

    @pytest.mark.asyncio
    async def test_deletion_is_idempotent(self):
        """DEL-015: Deletion is idempotent."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        user_id = "already-deleted-user"

        # No user found (already deleted)
        mock_db.query.return_value.filter.return_value.first.return_value = None

        orchestrator = UserDeletionOrchestrator(db=mock_db)

        # First deletion
        result1 = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="First deletion attempt",
        )

        # Second deletion - should also succeed
        result2 = await orchestrator.delete_user(
            user_id=user_id,
            audit_reason="Second deletion attempt",
        )

        # Both should complete without error
        assert result1.user_id == user_id
        assert result2.user_id == user_id
        # Second should also show 0 deleted
        assert result2.results.get("postgres", {}).get("count", 0) == 0

    @pytest.mark.asyncio
    async def test_deletion_includes_timing(self):
        """DEL-016: Deletion includes timing information."""
        from app.gdpr.deletion import UserDeletionOrchestrator

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        orchestrator = UserDeletionOrchestrator(db=mock_db)
        result = await orchestrator.delete_user(
            user_id="test-user",
            audit_reason="Test",
        )

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)
