"""Tests for Query Playground router.

This module tests the query playground per section 16 (FR-013):
- Parameter tuning interface
- Side-by-side retrieval comparison
- Query execution with configurable parameters
- Result comparison and analysis
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openmemory.api.app.routers.playground import (
    PlaygroundConfig,
    QueryParams,
    RetrievalResult,
    ComparisonResult,
    QueryExecution,
    PlaygroundSession,
    PlaygroundService,
    create_playground_service,
)


# ============================================================================
# PlaygroundConfig Tests
# ============================================================================


class TestPlaygroundConfig:
    """Tests for PlaygroundConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PlaygroundConfig()
        assert config.max_results == 20
        assert config.enable_caching is True
        assert config.default_search_mode == "hybrid"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlaygroundConfig(
            max_results=50,
            enable_caching=False,
            default_search_mode="semantic",
        )
        assert config.max_results == 50
        assert config.enable_caching is False
        assert config.default_search_mode == "semantic"


# ============================================================================
# QueryParams Tests
# ============================================================================


class TestQueryParams:
    """Tests for QueryParams."""

    def test_default_params(self):
        """Test default query parameters."""
        params = QueryParams(query="test query")
        assert params.query == "test query"
        assert params.limit == 10
        assert params.mode == "hybrid"
        assert params.threshold == 0.0

    def test_semantic_params(self):
        """Test semantic search parameters."""
        params = QueryParams(
            query="find authentication",
            mode="semantic",
            limit=5,
            threshold=0.7,
        )
        assert params.mode == "semantic"
        assert params.limit == 5
        assert params.threshold == 0.7

    def test_lexical_params(self):
        """Test lexical search parameters."""
        params = QueryParams(
            query="class UserAuth",
            mode="lexical",
            limit=20,
        )
        assert params.mode == "lexical"
        assert params.limit == 20

    def test_hybrid_params_with_weights(self):
        """Test hybrid search with custom weights."""
        params = QueryParams(
            query="authentication flow",
            mode="hybrid",
            semantic_weight=0.7,
            lexical_weight=0.3,
        )
        assert params.semantic_weight == 0.7
        assert params.lexical_weight == 0.3

    def test_params_to_dict(self):
        """Test parameter serialization."""
        params = QueryParams(query="test", limit=5, mode="semantic")
        d = params.to_dict()
        assert d["query"] == "test"
        assert d["limit"] == 5
        assert d["mode"] == "semantic"


# ============================================================================
# RetrievalResult Tests
# ============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult."""

    def test_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            id="result-1",
            content="Test content",
            score=0.95,
            source="code",
        )
        assert result.id == "result-1"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.source == "code"

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = RetrievalResult(
            id="result-2",
            content="Function definition",
            score=0.88,
            source="code",
            metadata={"file": "/src/auth.py", "line": 42},
        )
        assert result.metadata["file"] == "/src/auth.py"
        assert result.metadata["line"] == 42

    def test_result_to_dict(self):
        """Test result serialization."""
        result = RetrievalResult(
            id="result-3",
            content="Test",
            score=0.75,
            source="memory",
        )
        d = result.to_dict()
        assert d["id"] == "result-3"
        assert d["score"] == 0.75
        assert d["source"] == "memory"


# ============================================================================
# ComparisonResult Tests
# ============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_comparison_creation(self):
        """Test creating a comparison result."""
        left_results = [
            RetrievalResult(id="l1", content="A", score=0.9, source="code"),
            RetrievalResult(id="l2", content="B", score=0.8, source="code"),
        ]
        right_results = [
            RetrievalResult(id="r1", content="A", score=0.85, source="code"),
            RetrievalResult(id="r2", content="C", score=0.75, source="code"),
        ]

        comparison = ComparisonResult(
            left_params=QueryParams(query="test", mode="semantic"),
            right_params=QueryParams(query="test", mode="lexical"),
            left_results=left_results,
            right_results=right_results,
        )

        assert len(comparison.left_results) == 2
        assert len(comparison.right_results) == 2
        assert comparison.left_params.mode == "semantic"
        assert comparison.right_params.mode == "lexical"

    def test_comparison_overlap(self):
        """Test calculating result overlap."""
        left = [
            RetrievalResult(id="1", content="A", score=0.9, source="code"),
            RetrievalResult(id="2", content="B", score=0.8, source="code"),
            RetrievalResult(id="3", content="C", score=0.7, source="code"),
        ]
        right = [
            RetrievalResult(id="1", content="A", score=0.85, source="code"),
            RetrievalResult(id="4", content="D", score=0.75, source="code"),
        ]

        comparison = ComparisonResult(
            left_params=QueryParams(query="test", mode="semantic"),
            right_params=QueryParams(query="test", mode="lexical"),
            left_results=left,
            right_results=right,
        )

        overlap = comparison.calculate_overlap()
        assert overlap["common_ids"] == {"1"}
        assert overlap["left_only_ids"] == {"2", "3"}
        assert overlap["right_only_ids"] == {"4"}

    def test_comparison_statistics(self):
        """Test comparison statistics."""
        left = [
            RetrievalResult(id="1", content="A", score=0.9, source="code"),
            RetrievalResult(id="2", content="B", score=0.8, source="code"),
        ]
        right = [
            RetrievalResult(id="3", content="C", score=0.7, source="code"),
        ]

        comparison = ComparisonResult(
            left_params=QueryParams(query="test", mode="semantic"),
            right_params=QueryParams(query="test", mode="lexical"),
            left_results=left,
            right_results=right,
        )

        stats = comparison.get_statistics()
        assert stats["left_count"] == 2
        assert stats["right_count"] == 1
        assert abs(stats["left_avg_score"] - 0.85) < 0.001
        assert abs(stats["right_avg_score"] - 0.7) < 0.001

    def test_comparison_to_dict(self):
        """Test comparison serialization."""
        comparison = ComparisonResult(
            left_params=QueryParams(query="test", mode="semantic"),
            right_params=QueryParams(query="test", mode="hybrid"),
            left_results=[],
            right_results=[],
        )

        d = comparison.to_dict()
        assert "left_params" in d
        assert "right_params" in d
        assert "left_results" in d
        assert "right_results" in d
        assert "statistics" in d


# ============================================================================
# QueryExecution Tests
# ============================================================================


class TestQueryExecution:
    """Tests for QueryExecution."""

    def test_execution_creation(self):
        """Test creating a query execution."""
        execution = QueryExecution(
            execution_id="exec-1",
            params=QueryParams(query="test", mode="semantic"),
            results=[
                RetrievalResult(id="r1", content="A", score=0.9, source="code"),
            ],
            execution_time_ms=45.5,
        )

        assert execution.execution_id == "exec-1"
        assert execution.params.mode == "semantic"
        assert len(execution.results) == 1
        assert execution.execution_time_ms == 45.5

    def test_execution_to_dict(self):
        """Test execution serialization."""
        execution = QueryExecution(
            execution_id="exec-2",
            params=QueryParams(query="test"),
            results=[],
            execution_time_ms=10.0,
        )

        d = execution.to_dict()
        assert d["execution_id"] == "exec-2"
        assert d["execution_time_ms"] == 10.0
        assert "params" in d
        assert "results" in d


# ============================================================================
# PlaygroundSession Tests
# ============================================================================


class TestPlaygroundSession:
    """Tests for PlaygroundSession."""

    def test_session_creation(self):
        """Test creating a session."""
        session = PlaygroundSession(
            session_id="session-1",
            user_id="user-123",
        )

        assert session.session_id == "session-1"
        assert session.user_id == "user-123"
        assert len(session.executions) == 0
        assert len(session.comparisons) == 0

    def test_add_execution(self):
        """Test adding an execution to session."""
        session = PlaygroundSession(
            session_id="session-2",
            user_id="user-123",
        )

        execution = QueryExecution(
            execution_id="exec-1",
            params=QueryParams(query="test"),
            results=[],
            execution_time_ms=20.0,
        )

        session.add_execution(execution)
        assert len(session.executions) == 1
        assert session.executions[0].execution_id == "exec-1"

    def test_add_comparison(self):
        """Test adding a comparison to session."""
        session = PlaygroundSession(
            session_id="session-3",
            user_id="user-123",
        )

        comparison = ComparisonResult(
            left_params=QueryParams(query="test", mode="semantic"),
            right_params=QueryParams(query="test", mode="lexical"),
            left_results=[],
            right_results=[],
        )

        session.add_comparison(comparison)
        assert len(session.comparisons) == 1

    def test_session_history(self):
        """Test session history tracking."""
        session = PlaygroundSession(
            session_id="session-4",
            user_id="user-123",
        )

        # Add multiple executions
        for i in range(3):
            execution = QueryExecution(
                execution_id=f"exec-{i}",
                params=QueryParams(query=f"query {i}"),
                results=[],
                execution_time_ms=10.0 + i,
            )
            session.add_execution(execution)

        history = session.get_history()
        assert len(history) == 3
        assert history[0]["execution_id"] == "exec-0"

    def test_session_to_dict(self):
        """Test session serialization."""
        session = PlaygroundSession(
            session_id="session-5",
            user_id="user-456",
        )

        d = session.to_dict()
        assert d["session_id"] == "session-5"
        assert d["user_id"] == "user-456"
        assert "executions" in d
        assert "comparisons" in d


# ============================================================================
# PlaygroundService Tests
# ============================================================================


class TestPlaygroundService:
    """Tests for PlaygroundService."""

    def test_service_creation(self):
        """Test creating the service."""
        service = PlaygroundService()
        assert service is not None

    def test_service_with_config(self):
        """Test service with custom config."""
        config = PlaygroundConfig(max_results=50)
        service = PlaygroundService(config=config)
        assert service._config.max_results == 50

    def test_create_session(self):
        """Test creating a new session."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")

        assert session.user_id == "user-123"
        assert session.session_id is not None

    def test_get_session(self):
        """Test retrieving a session."""
        service = PlaygroundService()
        created = service.create_session(user_id="user-123")
        retrieved = service.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Test retrieving non-existent session."""
        service = PlaygroundService()
        session = service.get_session("nonexistent")
        assert session is None

    def test_execute_query(self):
        """Test executing a query."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")

        params = QueryParams(query="authentication", mode="semantic", limit=5)

        # Mock the search function
        with patch.object(
            service,
            "_do_search",
            return_value=[
                {"id": "1", "content": "Auth code", "score": 0.9},
            ],
        ):
            execution = service.execute_query(session.session_id, params)

        assert execution is not None
        assert execution.params.query == "authentication"
        assert len(session.executions) == 1

    def test_compare_queries(self):
        """Test comparing two queries."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")

        left_params = QueryParams(query="auth", mode="semantic")
        right_params = QueryParams(query="auth", mode="lexical")

        # Mock search
        with patch.object(
            service,
            "_do_search",
            side_effect=[
                [{"id": "1", "content": "A", "score": 0.9}],
                [{"id": "2", "content": "B", "score": 0.8}],
            ],
        ):
            comparison = service.compare_queries(
                session.session_id,
                left_params,
                right_params,
            )

        assert comparison is not None
        assert len(session.comparisons) == 1
        assert comparison.left_params.mode == "semantic"
        assert comparison.right_params.mode == "lexical"

    def test_parameter_presets(self):
        """Test getting parameter presets."""
        service = PlaygroundService()
        presets = service.get_presets()

        assert "semantic" in presets
        assert "lexical" in presets
        assert "hybrid" in presets
        assert presets["semantic"]["mode"] == "semantic"
        assert presets["lexical"]["mode"] == "lexical"
        assert presets["hybrid"]["mode"] == "hybrid"

    def test_apply_preset(self):
        """Test applying a preset."""
        service = PlaygroundService()
        params = service.apply_preset("semantic", query="test query")

        assert params.mode == "semantic"
        assert params.query == "test query"

    def test_list_sessions(self):
        """Test listing user sessions."""
        service = PlaygroundService()
        service.create_session(user_id="user-1")
        service.create_session(user_id="user-1")
        service.create_session(user_id="user-2")

        user1_sessions = service.list_sessions(user_id="user-1")
        user2_sessions = service.list_sessions(user_id="user-2")

        assert len(user1_sessions) == 2
        assert len(user2_sessions) == 1

    def test_delete_session(self):
        """Test deleting a session."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")
        session_id = session.session_id

        success = service.delete_session(session_id)
        assert success is True

        retrieved = service.get_session(session_id)
        assert retrieved is None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreatePlaygroundService:
    """Tests for create_playground_service factory."""

    def test_create_default_service(self):
        """Test creating default service."""
        service = create_playground_service()
        assert service is not None
        assert isinstance(service, PlaygroundService)

    def test_create_with_config(self):
        """Test creating service with config."""
        config = PlaygroundConfig(max_results=100)
        service = create_playground_service(config=config)
        assert service._config.max_results == 100


# ============================================================================
# Integration Tests
# ============================================================================


class TestPlaygroundIntegration:
    """Integration tests for playground functionality."""

    def test_full_workflow(self):
        """Test complete query playground workflow."""
        service = PlaygroundService()

        # Create session
        session = service.create_session(user_id="user-123")

        # Execute queries with different parameters
        with patch.object(
            service,
            "_do_search",
            return_value=[
                {"id": "1", "content": "Result A", "score": 0.95},
                {"id": "2", "content": "Result B", "score": 0.85},
            ],
        ):
            # First query
            exec1 = service.execute_query(
                session.session_id,
                QueryParams(query="authentication", mode="semantic"),
            )
            assert exec1 is not None

            # Second query with different params
            exec2 = service.execute_query(
                session.session_id,
                QueryParams(query="authentication", mode="hybrid", limit=5),
            )
            assert exec2 is not None

        # Check session history
        history = session.get_history()
        assert len(history) == 2

    def test_comparison_workflow(self):
        """Test query comparison workflow."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")

        semantic_results = [
            {"id": "1", "content": "A", "score": 0.95},
            {"id": "2", "content": "B", "score": 0.90},
        ]
        lexical_results = [
            {"id": "1", "content": "A", "score": 0.80},
            {"id": "3", "content": "C", "score": 0.75},
        ]

        with patch.object(
            service,
            "_do_search",
            side_effect=[semantic_results, lexical_results],
        ):
            comparison = service.compare_queries(
                session.session_id,
                QueryParams(query="auth", mode="semantic"),
                QueryParams(query="auth", mode="lexical"),
            )

        # Check overlap analysis
        overlap = comparison.calculate_overlap()
        assert "1" in overlap["common_ids"]

        # Check statistics
        stats = comparison.get_statistics()
        assert stats["left_count"] == 2
        assert stats["right_count"] == 2

    def test_preset_application_workflow(self):
        """Test applying presets in queries."""
        service = PlaygroundService()
        session = service.create_session(user_id="user-123")

        # Get presets
        presets = service.get_presets()
        assert "semantic" in presets

        # Apply preset
        params = service.apply_preset("semantic", query="find user login")
        assert params.mode == "semantic"
        assert params.query == "find user login"

        # Execute with preset params
        with patch.object(
            service,
            "_do_search",
            return_value=[{"id": "1", "content": "Login", "score": 0.9}],
        ):
            execution = service.execute_query(session.session_id, params)
            assert len(execution.results) == 1
