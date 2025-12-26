"""Tests for FeedbackStore.

TDD tests for feedback storage following the v9 plan spec:
- Append-only log storage
- Scoped to org and user
- 30-day query content retention cap
- Anonymized aggregate metrics
"""

from datetime import datetime, timedelta, timezone
from typing import Generator

import pytest

from openmemory.api.feedback.events import (
    FeedbackEvent,
    FeedbackOutcome,
    FeedbackType,
)
from openmemory.api.feedback.store import (
    FeedbackStore,
    InMemoryFeedbackStore,
)


@pytest.fixture
def store() -> InMemoryFeedbackStore:
    """Create a fresh in-memory store for tests."""
    return InMemoryFeedbackStore()


@pytest.fixture
def sample_event() -> FeedbackEvent:
    """Create a sample feedback event."""
    return FeedbackEvent(
        query_id="q123",
        user_id="u456",
        org_id="org789",
        tool_name="search_code_hybrid",
        outcome=FeedbackOutcome.ACCEPTED,
    )


class TestFeedbackStoreInterface:
    """Tests for FeedbackStore abstract interface."""

    def test_store_is_abstract(self):
        """FeedbackStore is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeedbackStore()

    def test_store_has_append_method(self):
        """Store interface defines append method."""
        assert hasattr(FeedbackStore, "append")

    def test_store_has_query_by_user_method(self):
        """Store interface defines query_by_user method."""
        assert hasattr(FeedbackStore, "query_by_user")

    def test_store_has_query_by_org_method(self):
        """Store interface defines query_by_org method."""
        assert hasattr(FeedbackStore, "query_by_org")

    def test_store_has_get_aggregate_metrics_method(self):
        """Store interface defines get_aggregate_metrics method."""
        assert hasattr(FeedbackStore, "get_aggregate_metrics")


class TestInMemoryFeedbackStoreAppend:
    """Tests for appending events to the store."""

    def test_append_single_event(self, store: InMemoryFeedbackStore, sample_event: FeedbackEvent):
        """Can append a single event."""
        store.append(sample_event)
        events = store.query_by_user(sample_event.user_id, sample_event.org_id)
        assert len(events) == 1
        assert events[0].event_id == sample_event.event_id

    def test_append_multiple_events(self, store: InMemoryFeedbackStore):
        """Can append multiple events."""
        events = [
            FeedbackEvent(
                query_id=f"q{i}",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
            for i in range(5)
        ]
        for event in events:
            store.append(event)

        result = store.query_by_user("u456", "org789")
        assert len(result) == 5

    def test_append_preserves_order(self, store: InMemoryFeedbackStore):
        """Events are returned in append order (chronological)."""
        events = [
            FeedbackEvent(
                query_id=f"q{i}",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
            for i in range(3)
        ]
        for event in events:
            store.append(event)

        result = store.query_by_user("u456", "org789")
        for i, event in enumerate(result):
            assert event.query_id == f"q{i}"

    def test_append_returns_event_id(self, store: InMemoryFeedbackStore, sample_event: FeedbackEvent):
        """Append returns the event ID."""
        event_id = store.append(sample_event)
        assert event_id == sample_event.event_id


class TestInMemoryFeedbackStoreQueryByUser:
    """Tests for querying events by user."""

    def test_query_by_user_filters_by_user_id(self, store: InMemoryFeedbackStore):
        """Query filters by user ID."""
        event1 = FeedbackEvent(
            query_id="q1",
            user_id="user_a",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        event2 = FeedbackEvent(
            query_id="q2",
            user_id="user_b",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        store.append(event1)
        store.append(event2)

        result = store.query_by_user("user_a", "org789")
        assert len(result) == 1
        assert result[0].user_id == "user_a"

    def test_query_by_user_filters_by_org_id(self, store: InMemoryFeedbackStore):
        """Query requires matching org ID."""
        event1 = FeedbackEvent(
            query_id="q1",
            user_id="user_a",
            org_id="org_1",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        event2 = FeedbackEvent(
            query_id="q2",
            user_id="user_a",
            org_id="org_2",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        store.append(event1)
        store.append(event2)

        result = store.query_by_user("user_a", "org_1")
        assert len(result) == 1
        assert result[0].org_id == "org_1"

    def test_query_by_user_with_limit(self, store: InMemoryFeedbackStore):
        """Query respects limit parameter."""
        for i in range(10):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )

        result = store.query_by_user("u456", "org789", limit=5)
        assert len(result) == 5

    def test_query_by_user_with_offset(self, store: InMemoryFeedbackStore):
        """Query respects offset parameter."""
        for i in range(10):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )

        result = store.query_by_user("u456", "org789", offset=5)
        assert len(result) == 5
        assert result[0].query_id == "q5"

    def test_query_by_user_with_time_range(self, store: InMemoryFeedbackStore):
        """Query can filter by time range."""
        now = datetime.now(timezone.utc)
        old_event = FeedbackEvent(
            query_id="old",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            timestamp=now - timedelta(days=40),
        )
        recent_event = FeedbackEvent(
            query_id="recent",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            timestamp=now - timedelta(days=5),
        )
        store.append(old_event)
        store.append(recent_event)

        # Query last 30 days only
        result = store.query_by_user(
            "u456",
            "org789",
            since=now - timedelta(days=30),
        )
        assert len(result) == 1
        assert result[0].query_id == "recent"


class TestInMemoryFeedbackStoreQueryByOrg:
    """Tests for querying events by org."""

    def test_query_by_org_returns_all_users(self, store: InMemoryFeedbackStore):
        """Query by org returns events from all users in the org."""
        for user_id in ["user_a", "user_b", "user_c"]:
            store.append(
                FeedbackEvent(
                    query_id=f"q_{user_id}",
                    user_id=user_id,
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )

        result = store.query_by_org("org789")
        assert len(result) == 3

    def test_query_by_org_filters_other_orgs(self, store: InMemoryFeedbackStore):
        """Query by org excludes events from other orgs."""
        store.append(
            FeedbackEvent(
                query_id="q1",
                user_id="user_a",
                org_id="org_1",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
        )
        store.append(
            FeedbackEvent(
                query_id="q2",
                user_id="user_b",
                org_id="org_2",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
        )

        result = store.query_by_org("org_1")
        assert len(result) == 1
        assert result[0].org_id == "org_1"


class TestInMemoryFeedbackStoreAggregateMetrics:
    """Tests for aggregate metrics calculation."""

    def test_acceptance_rate_calculation(self, store: InMemoryFeedbackStore):
        """Calculates acceptance rate correctly."""
        # 7 accepted, 3 rejected = 70% acceptance
        for i in range(7):
            store.append(
                FeedbackEvent(
                    query_id=f"q_accept_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )
        for i in range(3):
            store.append(
                FeedbackEvent(
                    query_id=f"q_reject_{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.REJECTED,
                )
            )

        metrics = store.get_aggregate_metrics("org789")
        assert metrics["acceptance_rate"] == pytest.approx(0.7, rel=0.01)

    def test_outcome_distribution(self, store: InMemoryFeedbackStore):
        """Calculates outcome distribution."""
        outcomes = [
            FeedbackOutcome.ACCEPTED,
            FeedbackOutcome.ACCEPTED,
            FeedbackOutcome.MODIFIED,
            FeedbackOutcome.REJECTED,
            FeedbackOutcome.IGNORED,
        ]
        for i, outcome in enumerate(outcomes):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=outcome,
                )
            )

        metrics = store.get_aggregate_metrics("org789")
        assert metrics["outcome_distribution"]["accepted"] == 2
        assert metrics["outcome_distribution"]["modified"] == 1
        assert metrics["outcome_distribution"]["rejected"] == 1
        assert metrics["outcome_distribution"]["ignored"] == 1

    def test_total_events_count(self, store: InMemoryFeedbackStore):
        """Counts total events."""
        for i in range(15):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )

        metrics = store.get_aggregate_metrics("org789")
        assert metrics["total_events"] == 15

    def test_metrics_by_tool(self, store: InMemoryFeedbackStore):
        """Calculates metrics grouped by tool name."""
        tools = ["search_code_hybrid", "search_code_hybrid", "find_callers", "impact_analysis"]
        for i, tool in enumerate(tools):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name=tool,
                    outcome=FeedbackOutcome.ACCEPTED,
                )
            )

        metrics = store.get_aggregate_metrics("org789")
        assert metrics["by_tool"]["search_code_hybrid"] == 2
        assert metrics["by_tool"]["find_callers"] == 1
        assert metrics["by_tool"]["impact_analysis"] == 1

    def test_metrics_anonymized_no_user_ids(self, store: InMemoryFeedbackStore):
        """Aggregate metrics don't include user IDs (anonymized)."""
        store.append(
            FeedbackEvent(
                query_id="q1",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
        )

        metrics = store.get_aggregate_metrics("org789")
        # Should not contain any user_id fields
        assert "user_id" not in metrics
        assert "user_ids" not in metrics

    def test_average_decision_time(self, store: InMemoryFeedbackStore):
        """Calculates average decision time."""
        times = [100, 200, 300]
        for i, time_ms in enumerate(times):
            store.append(
                FeedbackEvent(
                    query_id=f"q{i}",
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    decision_time_ms=time_ms,
                )
            )

        metrics = store.get_aggregate_metrics("org789")
        assert metrics["average_decision_time_ms"] == pytest.approx(200.0, rel=0.01)


class TestInMemoryFeedbackStoreRetention:
    """Tests for 30-day retention policy."""

    def test_query_content_respects_retention(self, store: InMemoryFeedbackStore):
        """Query content older than 30 days is not returned by default."""
        now = datetime.now(timezone.utc)

        old_event = FeedbackEvent(
            query_id="old",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            timestamp=now - timedelta(days=35),
        )
        new_event = FeedbackEvent(
            query_id="new",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            timestamp=now - timedelta(days=5),
        )
        store.append(old_event)
        store.append(new_event)

        # Default query should only return recent events
        result = store.query_by_user("u456", "org789", apply_retention=True)
        assert len(result) == 1
        assert result[0].query_id == "new"

    def test_retention_can_be_overridden(self, store: InMemoryFeedbackStore):
        """Retention can be disabled for permitted queries."""
        now = datetime.now(timezone.utc)

        old_event = FeedbackEvent(
            query_id="old",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            timestamp=now - timedelta(days=35),
        )
        store.append(old_event)

        # Override retention for admin/audit purposes
        result = store.query_by_user("u456", "org789", apply_retention=False)
        assert len(result) == 1


class TestInMemoryFeedbackStoreQueryByQuery:
    """Tests for querying events by query_id."""

    def test_query_by_query_id(self, store: InMemoryFeedbackStore):
        """Can retrieve all events for a specific query."""
        query_id = "q123"
        for i in range(3):
            store.append(
                FeedbackEvent(
                    query_id=query_id,
                    user_id="u456",
                    org_id="org789",
                    tool_name="search_code_hybrid",
                    outcome=FeedbackOutcome.ACCEPTED,
                    result_index=i,
                )
            )
        store.append(
            FeedbackEvent(
                query_id="other",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
            )
        )

        result = store.query_by_query_id(query_id, "org789")
        assert len(result) == 3
        for event in result:
            assert event.query_id == query_id


class TestInMemoryFeedbackStoreQueryForOptimizer:
    """Tests for queries used by the RRF weight optimizer."""

    def test_get_events_with_weights(self, store: InMemoryFeedbackStore):
        """Can retrieve events that have RRF weights recorded."""
        weights = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}
        store.append(
            FeedbackEvent(
                query_id="q1",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
                rrf_weights=weights,
            )
        )
        store.append(
            FeedbackEvent(
                query_id="q2",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.REJECTED,
                # No weights recorded
            )
        )

        result = store.query_for_optimization("org789")
        assert len(result) == 1
        assert result[0].rrf_weights == weights

    def test_get_events_with_experiment_data(self, store: InMemoryFeedbackStore):
        """Can retrieve events that have experiment data."""
        store.append(
            FeedbackEvent(
                query_id="q1",
                user_id="u456",
                org_id="org789",
                tool_name="search_code_hybrid",
                outcome=FeedbackOutcome.ACCEPTED,
                experiment_id="exp_001",
                variant_id="control",
            )
        )

        result = store.query_by_experiment("exp_001", "org789")
        assert len(result) == 1
        assert result[0].experiment_id == "exp_001"
        assert result[0].variant_id == "control"
