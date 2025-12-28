"""Tests for GDPR Subject Access Request (SAR) Export.

These tests verify the SAR export functionality that retrieves all user data
from all data stores in the system.

Test IDs: SAR-001 through SAR-016
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4


class TestSARExporterInitialization:
    """Tests for SARExporter initialization (SAR-001)."""

    def test_sar_exporter_initializes_with_db_only(self):
        """SAR-001: SARExporter initializes with only database client."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        exporter = SARExporter(db=mock_db)

        assert exporter._db is mock_db
        assert exporter._neo4j is None
        assert exporter._qdrant is None
        assert exporter._opensearch is None
        assert exporter._valkey is None

    def test_sar_exporter_initializes_with_all_store_clients(self):
        """SAR-001: SARExporter initializes with all store clients."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        mock_qdrant = MagicMock()
        mock_opensearch = MagicMock()
        mock_valkey = MagicMock()

        exporter = SARExporter(
            db=mock_db,
            neo4j_driver=mock_neo4j,
            qdrant_client=mock_qdrant,
            opensearch_client=mock_opensearch,
            valkey_client=mock_valkey,
        )

        assert exporter._db is mock_db
        assert exporter._neo4j is mock_neo4j
        assert exporter._qdrant is mock_qdrant
        assert exporter._opensearch is mock_opensearch
        assert exporter._valkey is mock_valkey


class TestPostgreSQLExport:
    """Tests for PostgreSQL data export (SAR-002 through SAR-006)."""

    def test_postgresql_export_returns_user_record(self):
        """SAR-002: PostgreSQL export returns user record."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user record
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_user.name = "Test User"
        mock_user.email = "test@example.com"
        mock_user.created_at = datetime.now(timezone.utc)
        mock_user.updated_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.return_value = []

        exporter = SARExporter(db=mock_db)
        result = exporter._export_postgres(user_id)

        assert result["user"] is not None
        assert result["user"]["user_id"] == user_id

    def test_postgresql_export_returns_all_memories(self):
        """SAR-003: PostgreSQL export returns all memories."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id

        # Mock memories
        mock_memory = MagicMock()
        mock_memory.id = uuid4()
        mock_memory.content = "Test memory content"
        mock_memory.state = "active"
        mock_memory.created_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.side_effect = [
            [mock_memory],  # memories
            [],  # apps
            [],  # feedback
            [],  # experiments
        ]

        exporter = SARExporter(db=mock_db)
        result = exporter._export_postgres(user_id)

        assert "memories" in result
        assert len(result["memories"]) == 1

    def test_postgresql_export_returns_all_apps(self):
        """SAR-004: PostgreSQL export returns all apps."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id

        # Mock app
        mock_app = MagicMock()
        mock_app.id = uuid4()
        mock_app.name = "Test App"
        mock_app.created_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.side_effect = [
            [],  # memories
            [mock_app],  # apps
            [],  # feedback
            [],  # experiments
        ]

        exporter = SARExporter(db=mock_db)
        result = exporter._export_postgres(user_id)

        assert "apps" in result
        assert len(result["apps"]) == 1

    def test_postgresql_export_returns_feedback_events(self):
        """SAR-005: PostgreSQL export returns feedback events."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id

        # Mock feedback event
        mock_feedback = MagicMock()
        mock_feedback.event_id = str(uuid4())
        mock_feedback.created_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.side_effect = [
            [],  # memories
            [],  # apps
            [mock_feedback],  # feedback
            [],  # experiments
        ]

        exporter = SARExporter(db=mock_db)
        result = exporter._export_postgres(user_id)

        assert "feedback" in result
        assert len(result["feedback"]) == 1

    def test_postgresql_export_returns_experiment_assignments(self):
        """SAR-006: PostgreSQL export returns experiment assignments."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id

        # Mock assignment
        mock_assignment = MagicMock()
        mock_assignment.id = uuid4()
        mock_assignment.variant_config = {"variant": "A"}
        mock_assignment.created_at = datetime.now(timezone.utc)

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.side_effect = [
            [],  # memories
            [],  # apps
            [],  # feedback
            [mock_assignment],  # experiments
        ]

        exporter = SARExporter(db=mock_db)
        result = exporter._export_postgres(user_id)

        assert "experiments" in result
        assert len(result["experiments"]) == 1


class TestNeo4jExport:
    """Tests for Neo4j data export (SAR-007, SAR-008)."""

    def test_neo4j_export_returns_user_graph_nodes(self):
        """SAR-007: Neo4j export returns user graph nodes."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # Mock Neo4j session and result
        mock_session = MagicMock()

        # First query returns nodes
        mock_node_result = MagicMock()
        mock_node_record = MagicMock()
        mock_node_record.__getitem__ = lambda self, key: {"user_id": user_id} if key == "n" else ["User"]
        mock_node_result.__iter__ = lambda self: iter([mock_node_record])

        # Second query returns relationships
        mock_rel_result = MagicMock()
        mock_rel_result.__iter__ = lambda self: iter([])

        mock_session.run.side_effect = [mock_node_result, mock_rel_result]
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        exporter = SARExporter(db=mock_db, neo4j_driver=mock_neo4j)
        result = exporter._export_neo4j(user_id)

        assert "nodes" in result
        mock_session.run.assert_called()

    def test_neo4j_export_returns_relationships(self):
        """SAR-008: Neo4j export returns relationships."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # Mock Neo4j session
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run.return_value = mock_result
        mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

        exporter = SARExporter(db=mock_db, neo4j_driver=mock_neo4j)
        result = exporter._export_neo4j(user_id)

        assert "relationships" in result


class TestQdrantExport:
    """Tests for Qdrant data export (SAR-009)."""

    def test_qdrant_export_returns_embedding_metadata(self):
        """SAR-009: Qdrant export returns embedding metadata."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_qdrant = MagicMock()
        user_id = "test-user-123"

        # Mock Qdrant collections and scroll
        mock_collection = MagicMock()
        mock_collection.name = "embeddings_default"
        mock_qdrant.get_collections.return_value.collections = [mock_collection]
        mock_qdrant.scroll.return_value = ([], None)

        exporter = SARExporter(db=mock_db, qdrant_client=mock_qdrant)
        result = exporter._export_qdrant(user_id)

        assert "embeddings" in result
        mock_qdrant.scroll.assert_called()


class TestOpenSearchExport:
    """Tests for OpenSearch data export (SAR-010)."""

    def test_opensearch_export_returns_indexed_documents(self):
        """SAR-010: OpenSearch export returns indexed documents."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_opensearch = MagicMock()
        user_id = "test-user-123"

        # Mock OpenSearch search
        mock_opensearch.search.return_value = {
            "hits": {
                "hits": [
                    {"_id": "doc-1", "_source": {"user_id": user_id, "content": "test"}}
                ]
            }
        }

        exporter = SARExporter(db=mock_db, opensearch_client=mock_opensearch)
        result = exporter._export_opensearch(user_id)

        assert "documents" in result
        mock_opensearch.search.assert_called()


class TestValkeyExport:
    """Tests for Valkey data export (SAR-011)."""

    def test_valkey_export_returns_session_data(self):
        """SAR-011: Valkey export returns session data."""
        from app.gdpr.sar_export import SARExporter

        mock_db = MagicMock()
        mock_valkey = MagicMock()
        user_id = "test-user-123"

        # Mock Valkey keys and get
        mock_valkey.keys.return_value = [f"episodic:{user_id}:session1"]
        mock_valkey.get.return_value = b'{"memory_id": "mem-1"}'

        exporter = SARExporter(db=mock_db, valkey_client=mock_valkey)
        result = exporter._export_valkey(user_id)

        assert "episodic_sessions" in result
        mock_valkey.keys.assert_called()


class TestFullSARExport:
    """Tests for full SAR export integration (SAR-012 through SAR-016)."""

    @pytest.mark.asyncio
    async def test_full_sar_export_returns_all_stores(self):
        """SAR-012: Full SAR export returns all stores."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.schemas import SARResponse

        mock_db = MagicMock()
        user_id = "test-user-123"

        # Setup minimal mocks for PostgreSQL only
        mock_db.query.return_value.filter.return_value.first.return_value = None

        exporter = SARExporter(db=mock_db)
        result = await exporter.export_user_data(user_id)

        assert isinstance(result, SARResponse)
        assert result.user_id == user_id
        assert "postgres" in result.stores or hasattr(result, 'postgres')

    def test_sar_response_follows_json_schema(self):
        """SAR-013: SAR response follows defined JSON schema."""
        from app.gdpr.schemas import SARResponse

        response = SARResponse(
            user_id="test-user-123",
            export_date=datetime.now(timezone.utc),
            format_version="1.0",
        )

        # Required fields
        assert response.user_id is not None
        assert response.export_date is not None
        assert response.format_version is not None

        # Default values
        assert response.partial is False
        assert response.errors == []

    @pytest.mark.asyncio
    async def test_sar_handles_nonexistent_user(self):
        """SAR-014: SAR handles non-existent user."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.schemas import SARResponse

        mock_db = MagicMock()
        user_id = "nonexistent-user"

        # No user found
        mock_db.query.return_value.filter.return_value.first.return_value = None

        exporter = SARExporter(db=mock_db)
        result = await exporter.export_user_data(user_id)

        # Should return empty data, not error
        assert isinstance(result, SARResponse)
        assert result.user_id == user_id
        # postgres data should be empty or have None user
        assert result.postgres.get("user") is None or result.postgres == {}

    @pytest.mark.asyncio
    async def test_sar_handles_partial_data(self):
        """SAR-015: SAR handles partial data (missing stores)."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.schemas import SARResponse

        mock_db = MagicMock()
        user_id = "test-user-123"

        # User exists in PostgreSQL
        mock_user = MagicMock()
        mock_user.id = uuid4()
        mock_user.user_id = user_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.query.return_value.filter.return_value.all.return_value = []

        # No external stores connected
        exporter = SARExporter(db=mock_db)
        result = await exporter.export_user_data(user_id)

        # Should return data from available stores only
        assert isinstance(result, SARResponse)
        assert result.postgres.get("user") is not None
        assert result.neo4j == {}
        assert result.qdrant == {}
        assert result.opensearch == {}
        assert result.valkey == {}

    @pytest.mark.asyncio
    async def test_sar_export_includes_duration(self):
        """SAR-016: SAR export includes duration measurement."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.schemas import SARResponse

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        exporter = SARExporter(db=mock_db)
        result = await exporter.export_user_data("test-user")

        assert result.export_duration_ms is not None
        assert result.export_duration_ms >= 0


class TestSARExportErrorHandling:
    """Tests for SAR export error handling."""

    @pytest.mark.asyncio
    async def test_sar_continues_on_store_failure(self):
        """SAR export continues even if one store fails."""
        from app.gdpr.sar_export import SARExporter
        from app.gdpr.schemas import SARResponse

        mock_db = MagicMock()
        mock_neo4j = MagicMock()
        user_id = "test-user-123"

        # PostgreSQL works
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Neo4j fails
        mock_neo4j.session.side_effect = Exception("Neo4j connection failed")

        exporter = SARExporter(db=mock_db, neo4j_driver=mock_neo4j)
        result = await exporter.export_user_data(user_id)

        # Should still return result with error logged
        assert isinstance(result, SARResponse)
        assert result.partial is True
        assert len(result.errors) > 0
        assert "neo4j" in result.errors[0].lower()
