"""
PostgreSQL-backed FeedbackStore with tenant isolation.

This module provides persistent storage for feedback events with:
- Tenant isolation via user_id and org_id
- Retention query support (30-day default)
- Aggregation methods for metrics

Note: This implementation defines its own types that are compatible with
the existing openmemory.api.feedback module, but doesn't depend on it
to avoid circular import issues.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from app.database import Base


# Default retention period (30 days per v9 plan)
DEFAULT_RETENTION_DAYS = 30


# ============================================================================
# Domain Types (compatible with openmemory.api.feedback.events)
# ============================================================================


class FeedbackOutcome(Enum):
    """Outcome of a retrieval result."""

    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    IGNORED = "ignored"


class FeedbackType(Enum):
    """Type of feedback event."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class FeedbackEvent:
    """A feedback event for retrieval quality tracking."""

    # Required fields
    query_id: str
    user_id: str
    org_id: str
    tool_name: str
    outcome: FeedbackOutcome

    # Optional fields with defaults
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.IMPLICIT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Timing and context
    decision_time_ms: Optional[int] = None
    rrf_weights: Optional[dict[str, float]] = None
    reranker_used: Optional[bool] = None

    # Result tracking
    result_index: Optional[int] = None
    result_id: Optional[str] = None

    # A/B testing
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None

    # Arbitrary metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Abstract Store Interface
# ============================================================================


class FeedbackStoreInterface(ABC):
    """Abstract interface for feedback storage."""

    @abstractmethod
    def append(self, event: FeedbackEvent) -> str:
        """Append a feedback event to the store."""
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
        """Query events by user ID within an org."""
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
        """Query all events for an org."""
        pass

    @abstractmethod
    def get_aggregate_metrics(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get anonymized aggregate metrics for an org."""
        pass

    @abstractmethod
    def query_by_query_id(
        self,
        query_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query all events for a specific query."""
        pass

    @abstractmethod
    def query_for_optimization(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query events that have RRF weights for optimization."""
        pass

    @abstractmethod
    def query_by_experiment(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[FeedbackEvent]:
        """Query events for a specific A/B test experiment."""
        pass


# ============================================================================
# SQLAlchemy Model
# ============================================================================


class FeedbackEventModel(Base):
    """SQLAlchemy model for feedback events."""

    __tablename__ = "feedback_events"

    # Primary key
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String, nullable=False, unique=True, index=True)

    # Required fields for tenant isolation
    user_id = Column(String, nullable=False, index=True)
    org_id = Column(String, nullable=False, index=True)

    # Core event fields
    query_id = Column(String, nullable=False, index=True)
    tool_name = Column(String, nullable=False, index=True)
    outcome = Column(SQLEnum(FeedbackOutcome), nullable=False, index=True)
    feedback_type = Column(
        SQLEnum(FeedbackType), nullable=False, default=FeedbackType.IMPLICIT
    )
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Optional context fields
    session_id = Column(String, nullable=True, index=True)
    decision_time_ms = Column(Integer, nullable=True)

    # RRF weights for optimization queries
    rrf_weights = Column(JSON, nullable=True)
    reranker_used = Column(Boolean, nullable=True)

    # Result tracking
    result_index = Column(Integer, nullable=True)
    result_id = Column(String, nullable=True)

    # A/B testing
    experiment_id = Column(String, nullable=True, index=True)
    variant_id = Column(String, nullable=True)

    # Arbitrary metadata
    metadata_ = Column("metadata", JSON, nullable=True, default=dict)

    __table_args__ = (
        # Composite indexes for common query patterns
        Index("idx_feedback_user_org", "user_id", "org_id"),
        Index("idx_feedback_org_timestamp", "org_id", "timestamp"),
        Index("idx_feedback_org_experiment", "org_id", "experiment_id"),
        Index("idx_feedback_query", "query_id", "org_id"),
    )

    def to_domain(self) -> FeedbackEvent:
        """Convert SQLAlchemy model to domain object."""
        return FeedbackEvent(
            event_id=self.event_id,
            query_id=self.query_id,
            user_id=self.user_id,
            org_id=self.org_id,
            tool_name=self.tool_name,
            outcome=self.outcome,
            feedback_type=self.feedback_type,
            timestamp=self.timestamp,
            session_id=self.session_id,
            decision_time_ms=self.decision_time_ms,
            rrf_weights=self.rrf_weights,
            reranker_used=self.reranker_used,
            result_index=self.result_index,
            result_id=self.result_id,
            experiment_id=self.experiment_id,
            variant_id=self.variant_id,
            metadata=self.metadata_ or {},
        )

    @classmethod
    def from_domain(cls, event: FeedbackEvent) -> "FeedbackEventModel":
        """Create SQLAlchemy model from domain object."""
        return cls(
            event_id=event.event_id,
            query_id=event.query_id,
            user_id=event.user_id,
            org_id=event.org_id,
            tool_name=event.tool_name,
            outcome=event.outcome,
            feedback_type=event.feedback_type,
            timestamp=event.timestamp,
            session_id=event.session_id,
            decision_time_ms=event.decision_time_ms,
            rrf_weights=event.rrf_weights,
            reranker_used=event.reranker_used,
            result_index=event.result_index,
            result_id=event.result_id,
            experiment_id=event.experiment_id,
            variant_id=event.variant_id,
            metadata_=event.metadata,
        )


# ============================================================================
# PostgreSQL Store Implementation
# ============================================================================


class PostgresFeedbackStore(FeedbackStoreInterface):
    """PostgreSQL-backed implementation of FeedbackStore.

    Provides persistent storage for feedback events with tenant isolation
    via user_id and org_id filtering.

    Args:
        db: SQLAlchemy session
        retention_days: Number of days for retention queries (default 30)

    Example:
        store = PostgresFeedbackStore(db)
        store.append(feedback_event)
        events = store.query_by_user(user_id, org_id)
    """

    def __init__(self, db: Session, retention_days: int = DEFAULT_RETENTION_DAYS):
        """Initialize the store.

        Args:
            db: SQLAlchemy session for database operations
            retention_days: Number of days to retain events for retention queries
        """
        self._db = db
        self._retention_days = retention_days

    def append(self, event: FeedbackEvent) -> str:
        """Append a feedback event to the store.

        Args:
            event: The feedback event to store

        Returns:
            The event_id of the stored event
        """
        model = FeedbackEventModel.from_domain(event)
        self._db.add(model)
        self._db.commit()
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
        """Query events by user ID within an org.

        Args:
            user_id: The user ID to query
            org_id: The org ID (required for scoping)
            limit: Maximum number of events to return
            offset: Number of events to skip
            since: Only return events after this time
            until: Only return events before this time
            apply_retention: Apply retention limit (default 30 days)

        Returns:
            List of matching events in chronological order
        """
        # Apply retention cutoff if requested
        if apply_retention:
            retention_cutoff = datetime.now(timezone.utc) - timedelta(
                days=self._retention_days
            )
            if since is None or since < retention_cutoff:
                since = retention_cutoff

        query = (
            self._db.query(FeedbackEventModel)
            .filter(FeedbackEventModel.user_id == user_id)
            .filter(FeedbackEventModel.org_id == org_id)
        )

        if since is not None:
            query = query.filter(FeedbackEventModel.timestamp >= since)
        if until is not None:
            query = query.filter(FeedbackEventModel.timestamp <= until)

        query = query.order_by(FeedbackEventModel.timestamp)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return [model.to_domain() for model in query.all()]

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
        query = self._db.query(FeedbackEventModel).filter(
            FeedbackEventModel.org_id == org_id
        )

        if since is not None:
            query = query.filter(FeedbackEventModel.timestamp >= since)
        if until is not None:
            query = query.filter(FeedbackEventModel.timestamp <= until)

        query = query.order_by(FeedbackEventModel.timestamp)

        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)

        return [model.to_domain() for model in query.all()]

    def get_aggregate_metrics(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get anonymized aggregate metrics for an org.

        Args:
            org_id: The org ID to get metrics for
            since: Only include events after this time
            until: Only include events before this time

        Returns:
            Dictionary with aggregate metrics
        """
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
        """Query all events for a specific query.

        Args:
            query_id: The query ID to look up
            org_id: The org ID (required for scoping)

        Returns:
            List of events for that query
        """
        query = (
            self._db.query(FeedbackEventModel)
            .filter(FeedbackEventModel.query_id == query_id)
            .filter(FeedbackEventModel.org_id == org_id)
            .order_by(FeedbackEventModel.timestamp)
        )
        return [model.to_domain() for model in query.all()]

    def query_for_optimization(
        self,
        org_id: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[FeedbackEvent]:
        """Query events that have RRF weights for optimization.

        Args:
            org_id: The org ID to query
            since: Only return events after this time
            until: Only return events before this time

        Returns:
            List of events with rrf_weights recorded
        """
        query = self._db.query(FeedbackEventModel).filter(
            FeedbackEventModel.org_id == org_id
        )

        if since is not None:
            query = query.filter(FeedbackEventModel.timestamp >= since)
        if until is not None:
            query = query.filter(FeedbackEventModel.timestamp <= until)

        query = query.order_by(FeedbackEventModel.timestamp)

        # Filter for events with rrf_weights (SQLite-compatible approach)
        # Check in Python since SQLite JSON NULL handling is inconsistent
        return [
            model.to_domain()
            for model in query.all()
            if model.rrf_weights is not None
        ]

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
        query = (
            self._db.query(FeedbackEventModel)
            .filter(FeedbackEventModel.experiment_id == experiment_id)
            .filter(FeedbackEventModel.org_id == org_id)
            .order_by(FeedbackEventModel.timestamp)
        )
        return [model.to_domain() for model in query.all()]
