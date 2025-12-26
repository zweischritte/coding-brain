"""Feedback event types and dataclasses.

Defines the core types for feedback events:
- FeedbackOutcome: accepted, modified, rejected, ignored
- FeedbackType: implicit (click, copy, use) or explicit (user feedback)
- FeedbackEvent: Main dataclass for storing feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class FeedbackOutcome(Enum):
    """Outcome of a retrieval result.

    Per v9 plan spec:
    - accepted: User used the result as-is
    - modified: User used but changed the result
    - rejected: User explicitly rejected the result
    - ignored: User skipped/ignored the result
    """

    ACCEPTED = "accepted"
    MODIFIED = "modified"
    REJECTED = "rejected"
    IGNORED = "ignored"


class FeedbackType(Enum):
    """Type of feedback event.

    - implicit: Automatic feedback from user actions (click, copy, use)
    - explicit: Direct feedback via MCP provide_feedback tool
    """

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class FeedbackEvent:
    """A feedback event for retrieval quality tracking.

    Per v9 plan spec (Section 6.3):
    - query_id: Unique identifier for the query
    - user_id: User who submitted the query
    - org_id: Organization the user belongs to
    - session_id: Optional session identifier
    - tool_name: MCP tool that was used
    - outcome: Result of user interaction
    - decision_time_ms: Time taken to decide
    - rrf_weights: RRF weights used for retrieval
    - reranker_used: Whether reranker was applied
    - timestamp: When the event occurred
    """

    # Required fields
    query_id: str
    user_id: str
    org_id: str
    tool_name: str
    outcome: FeedbackOutcome

    # Optional fields with defaults
    event_id: str = field(default_factory=lambda: str(uuid4()))
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "query_id": self.query_id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "outcome": self.outcome.value,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "decision_time_ms": self.decision_time_ms,
            "rrf_weights": self.rrf_weights,
            "reranker_used": self.reranker_used,
            "result_index": self.result_index,
            "result_id": self.result_id,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackEvent:
        """Deserialize event from dictionary."""
        # Parse enums
        outcome = FeedbackOutcome(data["outcome"])
        feedback_type = FeedbackType(data.get("feedback_type", "implicit"))

        # Parse timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now(timezone.utc)

        return cls(
            event_id=data.get("event_id", str(uuid4())),
            query_id=data["query_id"],
            user_id=data["user_id"],
            org_id=data["org_id"],
            session_id=data.get("session_id"),
            tool_name=data["tool_name"],
            outcome=outcome,
            feedback_type=feedback_type,
            timestamp=timestamp,
            decision_time_ms=data.get("decision_time_ms"),
            rrf_weights=data.get("rrf_weights"),
            reranker_used=data.get("reranker_used"),
            result_index=data.get("result_index"),
            result_id=data.get("result_id"),
            experiment_id=data.get("experiment_id"),
            variant_id=data.get("variant_id"),
            metadata=data.get("metadata", {}),
        )

    # ==========================================================================
    # Factory methods for implicit feedback events
    # ==========================================================================

    @classmethod
    def create_click(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        result_index: int,
        result_id: str,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create a click event (result was clicked)."""
        metadata = kwargs.pop("metadata", {})
        metadata["action"] = "click"
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=FeedbackOutcome.ACCEPTED,
            feedback_type=FeedbackType.IMPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    def create_copy(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        result_index: int,
        result_id: str,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create a copy event (result was copied)."""
        metadata = kwargs.pop("metadata", {})
        metadata["action"] = "copy"
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=FeedbackOutcome.ACCEPTED,
            feedback_type=FeedbackType.IMPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    def create_use(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        result_index: int,
        result_id: str,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create a use event (result was used in code)."""
        metadata = kwargs.pop("metadata", {})
        metadata["action"] = "use"
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=FeedbackOutcome.ACCEPTED,
            feedback_type=FeedbackType.IMPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    def create_skip(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        result_index: int,
        result_id: str,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create a skip event (result was skipped/scrolled past)."""
        metadata = kwargs.pop("metadata", {})
        metadata["action"] = "skip"
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=FeedbackOutcome.IGNORED,
            feedback_type=FeedbackType.IMPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )

    @classmethod
    def create_dwell(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        result_index: int,
        result_id: str,
        dwell_time_ms: int,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create a dwell event (user spent time viewing result)."""
        metadata = kwargs.pop("metadata", {})
        metadata["action"] = "dwell"
        metadata["dwell_time_ms"] = dwell_time_ms
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=FeedbackOutcome.ACCEPTED,
            feedback_type=FeedbackType.IMPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )

    # ==========================================================================
    # Factory method for explicit feedback events
    # ==========================================================================

    @classmethod
    def create_explicit(
        cls,
        query_id: str,
        user_id: str,
        org_id: str,
        tool_name: str,
        outcome: FeedbackOutcome,
        result_index: Optional[int] = None,
        result_id: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> FeedbackEvent:
        """Create an explicit feedback event from user via MCP tool."""
        metadata = kwargs.pop("metadata", {})
        if comment:
            metadata["comment"] = comment
        return cls(
            query_id=query_id,
            user_id=user_id,
            org_id=org_id,
            tool_name=tool_name,
            outcome=outcome,
            feedback_type=FeedbackType.EXPLICIT,
            result_index=result_index,
            result_id=result_id,
            metadata=metadata,
            **kwargs,
        )
