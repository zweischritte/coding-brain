"""
Tests for PostgreSQL-backed FeedbackStore.

The PostgresFeedbackStore provides persistent storage for feedback events
with tenant isolation via user_id and org_id, and retention query support.

TDD: These tests are written BEFORE the implementation.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from sqlalchemy.orm import Session

from app.stores.feedback_store import (
    FeedbackEvent,
    FeedbackOutcome,
    FeedbackType,
    FeedbackStoreInterface,
    PostgresFeedbackStore,
)


# Test UUIDs
USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
ORG_A_ID = "11111111-1111-1111-1111-111111111111"
ORG_B_ID = "22222222-2222-2222-2222-222222222222"


def make_feedback_event(
    user_id: str = USER_A_ID,
    org_id: str = ORG_A_ID,
    outcome: FeedbackOutcome = FeedbackOutcome.ACCEPTED,
    timestamp: Optional[datetime] = None,
    query_id: Optional[str] = None,
    tool_name: str = "test_tool",
    experiment_id: Optional[str] = None,
    rrf_weights: Optional[dict] = None,
) -> FeedbackEvent:
    """Helper to create feedback events for testing."""
    return FeedbackEvent(
        event_id=str(uuid.uuid4()),
        query_id=query_id or str(uuid.uuid4()),
        user_id=user_id,
        org_id=org_id,
        tool_name=tool_name,
        outcome=outcome,
        feedback_type=FeedbackType.IMPLICIT,
        timestamp=timestamp or datetime.now(timezone.utc),
        experiment_id=experiment_id,
        rrf_weights=rrf_weights,
    )


class TestPostgresFeedbackStoreAppend:
    """Test PostgresFeedbackStore.append() method."""

    def test_append_persists_event(self, sqlite_test_db: Session):
        """
        append() should persist an event and return the event_id.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        event = make_feedback_event()

        result = store.append(event)

        assert result == event.event_id

    def test_append_stores_all_fields(self, sqlite_test_db: Session):
        """
        append() should store all event fields correctly.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        now = datetime.now(timezone.utc)
        event = FeedbackEvent(
            event_id="test-event-id",
            query_id="test-query-id",
            user_id=USER_A_ID,
            org_id=ORG_A_ID,
            tool_name="test_tool",
            outcome=FeedbackOutcome.MODIFIED,
            feedback_type=FeedbackType.EXPLICIT,
            timestamp=now,
            session_id="test-session",
            decision_time_ms=500,
            rrf_weights={"semantic": 0.5, "keyword": 0.5},
            reranker_used=True,
            result_index=2,
            result_id="result-123",
            experiment_id="exp-123",
            variant_id="variant-a",
            metadata={"extra": "data"},
        )

        store.append(event)
        results = store.query_by_user(USER_A_ID, ORG_A_ID)

        assert len(results) == 1
        stored = results[0]
        assert stored.event_id == "test-event-id"
        assert stored.query_id == "test-query-id"
        assert stored.tool_name == "test_tool"
        assert stored.outcome == FeedbackOutcome.MODIFIED
        assert stored.feedback_type == FeedbackType.EXPLICIT
        assert stored.session_id == "test-session"
        assert stored.decision_time_ms == 500
        assert stored.rrf_weights == {"semantic": 0.5, "keyword": 0.5}
        assert stored.reranker_used is True
        assert stored.result_index == 2
        assert stored.result_id == "result-123"
        assert stored.experiment_id == "exp-123"
        assert stored.variant_id == "variant-a"
        assert stored.metadata == {"extra": "data"}


class TestPostgresFeedbackStoreQueryByUser:
    """Test PostgresFeedbackStore.query_by_user() method."""

    def test_query_by_user_filters_by_user_id(self, sqlite_test_db: Session):
        """
        query_by_user() should only return events for the specified user.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        event_a = make_feedback_event(user_id=USER_A_ID, org_id=ORG_A_ID)
        event_b = make_feedback_event(user_id=USER_B_ID, org_id=ORG_A_ID)

        store.append(event_a)
        store.append(event_b)

        results = store.query_by_user(USER_A_ID, ORG_A_ID)

        assert len(results) == 1
        assert results[0].user_id == USER_A_ID

    def test_query_by_user_filters_by_org_id(self, sqlite_test_db: Session):
        """
        query_by_user() should scope by org_id (tenant isolation).
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        event_org_a = make_feedback_event(user_id=USER_A_ID, org_id=ORG_A_ID)
        event_org_b = make_feedback_event(user_id=USER_A_ID, org_id=ORG_B_ID)

        store.append(event_org_a)
        store.append(event_org_b)

        results = store.query_by_user(USER_A_ID, ORG_A_ID)

        assert len(results) == 1
        assert results[0].org_id == ORG_A_ID

    def test_query_by_user_with_time_range(self, sqlite_test_db: Session):
        """
        query_by_user() should filter by time range.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        now = datetime.now(timezone.utc)

        old_event = make_feedback_event(timestamp=now - timedelta(days=10))
        recent_event = make_feedback_event(timestamp=now - timedelta(days=1))

        store.append(old_event)
        store.append(recent_event)

        results = store.query_by_user(
            USER_A_ID,
            ORG_A_ID,
            since=now - timedelta(days=5),
        )

        assert len(results) == 1
        assert results[0].event_id == recent_event.event_id

    def test_query_by_user_with_retention(self, sqlite_test_db: Session):
        """
        query_by_user(apply_retention=True) should exclude old events.
        """
        store = PostgresFeedbackStore(sqlite_test_db, retention_days=30)
        now = datetime.now(timezone.utc)

        old_event = make_feedback_event(timestamp=now - timedelta(days=60))
        recent_event = make_feedback_event(timestamp=now - timedelta(days=5))

        store.append(old_event)
        store.append(recent_event)

        results = store.query_by_user(
            USER_A_ID,
            ORG_A_ID,
            apply_retention=True,
        )

        assert len(results) == 1
        assert results[0].event_id == recent_event.event_id

    def test_query_by_user_pagination(self, sqlite_test_db: Session):
        """
        query_by_user() should support limit and offset.
        """
        store = PostgresFeedbackStore(sqlite_test_db)

        # Create 5 events
        for i in range(5):
            event = make_feedback_event(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=5 - i)
            )
            store.append(event)

        # Get page 1 (first 2)
        page1 = store.query_by_user(USER_A_ID, ORG_A_ID, limit=2, offset=0)
        # Get page 2 (next 2)
        page2 = store.query_by_user(USER_A_ID, ORG_A_ID, limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2


class TestPostgresFeedbackStoreQueryByOrg:
    """Test PostgresFeedbackStore.query_by_org() method."""

    def test_query_by_org_returns_all_org_events(self, sqlite_test_db: Session):
        """
        query_by_org() should return all events for an org.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        event_a = make_feedback_event(user_id=USER_A_ID, org_id=ORG_A_ID)
        event_b = make_feedback_event(user_id=USER_B_ID, org_id=ORG_A_ID)
        event_other_org = make_feedback_event(user_id=USER_A_ID, org_id=ORG_B_ID)

        store.append(event_a)
        store.append(event_b)
        store.append(event_other_org)

        results = store.query_by_org(ORG_A_ID)

        assert len(results) == 2
        assert all(e.org_id == ORG_A_ID for e in results)


class TestPostgresFeedbackStoreAggregateMetrics:
    """Test PostgresFeedbackStore.get_aggregate_metrics() method."""

    def test_get_aggregate_metrics_calculates_acceptance_rate(
        self, sqlite_test_db: Session
    ):
        """
        get_aggregate_metrics() should calculate correct acceptance rate.
        """
        store = PostgresFeedbackStore(sqlite_test_db)

        # Add 3 accepted, 1 rejected, 1 ignored
        for _ in range(3):
            store.append(make_feedback_event(outcome=FeedbackOutcome.ACCEPTED))
        store.append(make_feedback_event(outcome=FeedbackOutcome.REJECTED))
        store.append(make_feedback_event(outcome=FeedbackOutcome.IGNORED))

        metrics = store.get_aggregate_metrics(ORG_A_ID)

        # acceptance_rate = accepted / (total - ignored) = 3 / 4 = 0.75
        assert metrics["acceptance_rate"] == pytest.approx(0.75, rel=0.01)
        assert metrics["total_events"] == 5

    def test_get_aggregate_metrics_outcome_distribution(self, sqlite_test_db: Session):
        """
        get_aggregate_metrics() should return outcome distribution.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        store.append(make_feedback_event(outcome=FeedbackOutcome.ACCEPTED))
        store.append(make_feedback_event(outcome=FeedbackOutcome.ACCEPTED))
        store.append(make_feedback_event(outcome=FeedbackOutcome.MODIFIED))
        store.append(make_feedback_event(outcome=FeedbackOutcome.REJECTED))

        metrics = store.get_aggregate_metrics(ORG_A_ID)

        assert metrics["outcome_distribution"]["accepted"] == 2
        assert metrics["outcome_distribution"]["modified"] == 1
        assert metrics["outcome_distribution"]["rejected"] == 1

    def test_get_aggregate_metrics_by_tool(self, sqlite_test_db: Session):
        """
        get_aggregate_metrics() should group by tool name.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        store.append(make_feedback_event(tool_name="search"))
        store.append(make_feedback_event(tool_name="search"))
        store.append(make_feedback_event(tool_name="recall"))

        metrics = store.get_aggregate_metrics(ORG_A_ID)

        assert metrics["by_tool"]["search"] == 2
        assert metrics["by_tool"]["recall"] == 1


class TestPostgresFeedbackStoreQueryByQueryId:
    """Test PostgresFeedbackStore.query_by_query_id() method."""

    def test_query_by_query_id_returns_events_for_query(self, sqlite_test_db: Session):
        """
        query_by_query_id() should return all events for a specific query.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        query_id = str(uuid.uuid4())

        event1 = make_feedback_event(query_id=query_id)
        event2 = make_feedback_event(query_id=query_id)
        event_other = make_feedback_event()  # Different query

        store.append(event1)
        store.append(event2)
        store.append(event_other)

        results = store.query_by_query_id(query_id, ORG_A_ID)

        assert len(results) == 2
        assert all(e.query_id == query_id for e in results)


class TestPostgresFeedbackStoreQueryForOptimization:
    """Test PostgresFeedbackStore.query_for_optimization() method."""

    def test_query_for_optimization_returns_events_with_rrf_weights(
        self, sqlite_test_db: Session
    ):
        """
        query_for_optimization() should only return events with rrf_weights.
        """
        store = PostgresFeedbackStore(sqlite_test_db)

        event_with_weights = make_feedback_event(
            rrf_weights={"semantic": 0.5, "keyword": 0.5}
        )
        event_without_weights = make_feedback_event(rrf_weights=None)

        store.append(event_with_weights)
        store.append(event_without_weights)

        results = store.query_for_optimization(ORG_A_ID)

        assert len(results) == 1
        assert results[0].rrf_weights is not None


class TestPostgresFeedbackStoreQueryByExperiment:
    """Test PostgresFeedbackStore.query_by_experiment() method."""

    def test_query_by_experiment_filters_correctly(self, sqlite_test_db: Session):
        """
        query_by_experiment() should return events for a specific experiment.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        exp_id = "exp-test-123"

        event_in_exp = make_feedback_event(experiment_id=exp_id)
        event_other_exp = make_feedback_event(experiment_id="other-exp")
        event_no_exp = make_feedback_event(experiment_id=None)

        store.append(event_in_exp)
        store.append(event_other_exp)
        store.append(event_no_exp)

        results = store.query_by_experiment(exp_id, ORG_A_ID)

        assert len(results) == 1
        assert results[0].experiment_id == exp_id


class TestPostgresFeedbackStoreTenantIsolation:
    """Test tenant isolation in PostgresFeedbackStore."""

    def test_cross_org_query_returns_empty(self, sqlite_test_db: Session):
        """
        Querying with wrong org_id should return empty results.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        event = make_feedback_event(org_id=ORG_A_ID)
        store.append(event)

        results = store.query_by_user(USER_A_ID, ORG_B_ID)

        assert len(results) == 0

    def test_metrics_only_includes_org_events(self, sqlite_test_db: Session):
        """
        Aggregate metrics should only include events from the specified org.
        """
        store = PostgresFeedbackStore(sqlite_test_db)
        store.append(make_feedback_event(org_id=ORG_A_ID))
        store.append(make_feedback_event(org_id=ORG_A_ID))
        store.append(make_feedback_event(org_id=ORG_B_ID))

        metrics = store.get_aggregate_metrics(ORG_A_ID)

        assert metrics["total_events"] == 2


class TestPostgresFeedbackStoreInterface:
    """Test that PostgresFeedbackStore implements the FeedbackStoreInterface."""

    def test_implements_feedback_store_interface(self, sqlite_test_db: Session):
        """
        PostgresFeedbackStore should implement all FeedbackStoreInterface methods.
        """
        assert issubclass(PostgresFeedbackStore, FeedbackStoreInterface)

        store = PostgresFeedbackStore(sqlite_test_db)
        required_methods = [
            "append",
            "query_by_user",
            "query_by_org",
            "get_aggregate_metrics",
            "query_by_query_id",
            "query_for_optimization",
            "query_by_experiment",
        ]
        for method in required_methods:
            assert hasattr(store, method)
            assert callable(getattr(store, method))
