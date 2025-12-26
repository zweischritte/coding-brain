"""Feedback storage for retrieval quality tracking.

Implements append-only storage for feedback events following v9 plan:
- Scoped to org and user
- 30-day query content retention cap (default)
- Anonymized aggregate metrics for cross-org analysis
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from openmemory.api.feedback.events import FeedbackEvent, FeedbackOutcome

logger = logging.getLogger(__name__)

# Default retention period (30 days per v9 plan)
DEFAULT_RETENTION_DAYS = 30


class FeedbackStore(ABC):
    """Abstract base class for feedback storage.

    All implementations must provide:
    - append: Add a new event (append-only)
    - query_by_user: Get events for a specific user
    - query_by_org: Get events for an entire org
    - get_aggregate_metrics: Get anonymized aggregate metrics
    """

    @abstractmethod
    def append(self, event: FeedbackEvent) -> str:
        """Append a feedback event to the store.

        Args:
            event: The feedback event to store

        Returns:
            The event_id of the stored event
        """
        pass

    @abstractmethod
    def query_by_user(
        self,
        user_id: str,
        org_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        apply_retention: bool = False,
    ) -> list[FeedbackEvent]:
        """Query events by user ID within an org.

        Args:
            user_id: The user ID to query
            org_id: The org ID (required for scoping)
            limit: Maximum number of events to return
            offset: Number of events to skip
            since: Only return events after this time
            until: Only return events before this time
            apply_retention: Apply 30-day retention limit

        Returns:
            List of matching events in chronological order
        """
        pass

    @abstractmethod
    def query_by_org(
        self,
        org_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query all events for an org.

        Args:
            org_id: The org ID to query
            limit: Maximum number of events to return
            offset: Number of events to skip
            since: Only return events after this time
            until: Only return events before this time

        Returns:
            List of matching events in chronological order
        """
        pass

    @abstractmethod
    def get_aggregate_metrics(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get anonymized aggregate metrics for an org.

        Per v9 plan: Aggregate metrics must be anonymized for cross-org analysis.
        No user IDs or personally identifiable information should be included.

        Args:
            org_id: The org ID to get metrics for
            since: Only include events after this time
            until: Only include events before this time

        Returns:
            Dictionary with aggregate metrics:
            - acceptance_rate: Ratio of accepted outcomes
            - outcome_distribution: Count of each outcome type
            - total_events: Total number of events
            - by_tool: Events grouped by tool name
            - average_decision_time_ms: Average decision time
        """
        pass

    @abstractmethod
    def query_by_query_id(
        self,
        query_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query all events for a specific query.

        Args:
            query_id: The query ID to look up
            org_id: The org ID (required for scoping)

        Returns:
            List of events for that query
        """
        pass

    @abstractmethod
    def query_for_optimization(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query events that have RRF weights for optimization.

        Used by the RRF weight optimizer to gather training data.

        Args:
            org_id: The org ID to query
            since: Only return events after this time
            until: Only return events before this time

        Returns:
            List of events with rrf_weights recorded
        """
        pass

    @abstractmethod
    def query_by_experiment(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query events for a specific A/B test experiment.

        Args:
            experiment_id: The experiment ID to query
            org_id: The org ID (required for scoping)

        Returns:
            List of events for that experiment
        """
        pass


class InMemoryFeedbackStore(FeedbackStore):
    """In-memory implementation of FeedbackStore for testing and development.

    Note: This is not suitable for production use. Use a persistent
    implementation (e.g., PostgreSQL, ClickHouse) for production.
    """

    def __init__(self, retention_days: int = DEFAULT_RETENTION_DAYS):
        """Initialize the store.

        Args:
            retention_days: Number of days to retain events (default 30)
        """
        self._events: list[FeedbackEvent] = []
        self._retention_days = retention_days

    def append(self, event: FeedbackEvent) -> str:
        """Append a feedback event to the store."""
        self._events.append(event)
        logger.debug(f"Appended feedback event: {event.event_id}")
        return event.event_id

    def query_by_user(
        self,
        user_id: str,
        org_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        apply_retention: bool = False,
    ) -> list[FeedbackEvent]:
        """Query events by user ID within an org."""
        # Apply retention cutoff if requested
        if apply_retention:
            retention_cutoff = datetime.now(timezone.utc) - timedelta(
                days=self._retention_days
            )
            if since is None or since < retention_cutoff:
                since = retention_cutoff

        results = [
            e
            for e in self._events
            if e.user_id == user_id
            and e.org_id == org_id
            and (since is None or e.timestamp >= since)
            and (until is None or e.timestamp <= until)
        ]

        # Apply pagination
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    def query_by_org(
        self,
        org_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query all events for an org."""
        results = [
            e
            for e in self._events
            if e.org_id == org_id
            and (since is None or e.timestamp >= since)
            and (until is None or e.timestamp <= until)
        ]

        # Apply pagination
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    def get_aggregate_metrics(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get anonymized aggregate metrics for an org."""
        events = self.query_by_org(org_id, since=since, until=until)

        if not events:
            return {
                "acceptance_rate": 0.0,
                "outcome_distribution": {},
                "total_events": 0,
                "by_tool": {},
                "average_decision_time_ms": 0.0,
            }

        # Outcome distribution
        outcome_counts: dict[str, int] = defaultdict(int)
        for e in events:
            outcome_counts[e.outcome.value] += 1

        # Tool distribution
        tool_counts: dict[str, int] = defaultdict(int)
        for e in events:
            tool_counts[e.tool_name] += 1

        # Acceptance rate (accepted / total, excluding ignored)
        total_actionable = sum(
            1 for e in events if e.outcome != FeedbackOutcome.IGNORED
        )
        accepted_count = outcome_counts.get("accepted", 0)
        acceptance_rate = (
            accepted_count / total_actionable if total_actionable > 0 else 0.0
        )

        # Average decision time (only for events with decision_time_ms)
        decision_times = [e.decision_time_ms for e in events if e.decision_time_ms]
        average_decision_time = (
            sum(decision_times) / len(decision_times) if decision_times else 0.0
        )

        return {
            "acceptance_rate": acceptance_rate,
            "outcome_distribution": dict(outcome_counts),
            "total_events": len(events),
            "by_tool": dict(tool_counts),
            "average_decision_time_ms": average_decision_time,
        }

    def query_by_query_id(
        self,
        query_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query all events for a specific query."""
        return [
            e for e in self._events if e.query_id == query_id and e.org_id == org_id
        ]

    def query_for_optimization(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query events that have RRF weights for optimization."""
        return [
            e
            for e in self._events
            if e.org_id == org_id
            and e.rrf_weights is not None
            and (since is None or e.timestamp >= since)
            and (until is None or e.timestamp <= until)
        ]

    def query_by_experiment(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query events for a specific A/B test experiment."""
        return [
            e
            for e in self._events
            if e.experiment_id == experiment_id and e.org_id == org_id
        ]

    def clear(self) -> None:
        """Clear all events from the store (testing only)."""
        self._events.clear()

    def count(self) -> int:
        """Get total event count (testing only)."""
        return len(self._events)
