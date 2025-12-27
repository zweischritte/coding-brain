"""
Feedback Router - REST API for feedback events.

Provides endpoints for creating and querying feedback events,
as well as aggregate metrics for retrieval quality tracking.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.stores.feedback_store import (
    FeedbackEvent,
    FeedbackOutcome,
    FeedbackType,
    PostgresFeedbackStore,
)


router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


# ============================================================================
# Pydantic Schemas
# ============================================================================


class OutcomeEnum(str, Enum):
    """Valid feedback outcome values."""
    accepted = "accepted"
    rejected = "rejected"
    modified = "modified"
    ignored = "ignored"


class FeedbackCreate(BaseModel):
    """Request body for creating a feedback event."""
    query_id: str = Field(..., min_length=1, description="The query ID this feedback is for")
    outcome: OutcomeEnum = Field(..., description="The outcome of the retrieval result")
    tool_name: str = Field(..., min_length=1, description="The tool that produced the result")

    # Optional fields
    session_id: Optional[str] = Field(None, description="Session ID for grouping")
    decision_time_ms: Optional[int] = Field(None, ge=0, description="Time to decision in ms")
    experiment_id: Optional[str] = Field(None, description="A/B experiment ID if applicable")
    variant_id: Optional[str] = Field(None, description="A/B experiment variant ID")
    result_id: Optional[str] = Field(None, description="ID of the specific result")
    result_index: Optional[int] = Field(None, ge=0, description="Position of result in list")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Response body for a feedback event."""
    event_id: str
    query_id: str
    user_id: str
    org_id: str
    tool_name: str
    outcome: str
    feedback_type: str
    timestamp: datetime
    session_id: Optional[str] = None
    decision_time_ms: Optional[int] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None
    result_id: Optional[str] = None
    result_index: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_domain(cls, event: FeedbackEvent) -> "FeedbackResponse":
        """Create response from domain event."""
        return cls(
            event_id=event.event_id,
            query_id=event.query_id,
            user_id=event.user_id,
            org_id=event.org_id,
            tool_name=event.tool_name,
            outcome=event.outcome.value,
            feedback_type=event.feedback_type.value,
            timestamp=event.timestamp,
            session_id=event.session_id,
            decision_time_ms=event.decision_time_ms,
            experiment_id=event.experiment_id,
            variant_id=event.variant_id,
            result_id=event.result_id,
            result_index=event.result_index,
            metadata=event.metadata,
        )


class FeedbackListResponse(BaseModel):
    """Response body for listing feedback events."""
    items: list[FeedbackResponse]
    total: int
    limit: int
    offset: int


class FeedbackMetrics(BaseModel):
    """Aggregate metrics for feedback events."""
    total_events: int
    acceptance_rate: float
    outcome_distribution: dict[str, int]
    average_decision_time_ms: float
    by_tool: dict[str, int]


class ToolMetrics(BaseModel):
    """Metrics for a specific tool."""
    tool_name: str
    total_count: int
    accepted_count: int
    rejected_count: int
    modified_count: int
    ignored_count: int
    acceptance_rate: float


class ToolMetricsResponse(BaseModel):
    """Response body for tool-grouped metrics."""
    tools: list[ToolMetrics]


# ============================================================================
# Dependencies
# ============================================================================


def get_feedback_store(db: Session = Depends(get_db)) -> PostgresFeedbackStore:
    """Dependency for feedback store."""
    return PostgresFeedbackStore(db)


# ============================================================================
# Endpoints
# ============================================================================


@router.post("", status_code=201, response_model=FeedbackResponse)
async def create_feedback(
    feedback: FeedbackCreate,
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_WRITE)),
    store: PostgresFeedbackStore = Depends(get_feedback_store),
):
    """
    Create a new feedback event.

    Records feedback for a retrieval result. User and org are taken from
    the authenticated principal, not the request body.
    """
    # Map outcome string to enum
    outcome_map = {
        "accepted": FeedbackOutcome.ACCEPTED,
        "rejected": FeedbackOutcome.REJECTED,
        "modified": FeedbackOutcome.MODIFIED,
        "ignored": FeedbackOutcome.IGNORED,
    }

    # Create domain event with user/org from principal
    event = FeedbackEvent(
        query_id=feedback.query_id,
        user_id=principal.user_id,
        org_id=principal.org_id,
        tool_name=feedback.tool_name,
        outcome=outcome_map[feedback.outcome.value],
        session_id=feedback.session_id,
        decision_time_ms=feedback.decision_time_ms,
        experiment_id=feedback.experiment_id,
        variant_id=feedback.variant_id,
        result_id=feedback.result_id,
        result_index=feedback.result_index,
        metadata=feedback.metadata,
    )

    # Store the event
    event_id = store.append(event)

    return FeedbackResponse.from_domain(event)


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_READ)),
    store: PostgresFeedbackStore = Depends(get_feedback_store),
    limit: int = 100,
    offset: int = 0,
    since: Optional[str] = None,
    until: Optional[str] = None,
    query_id: Optional[str] = None,
):
    """
    List feedback events for the authenticated org.

    Results are scoped to the principal's org_id and sorted by timestamp.
    """
    # Parse datetime strings if provided
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' datetime format")

    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'until' datetime format")

    # Query by query_id if specified, otherwise query all org events
    if query_id:
        events = store.query_by_query_id(query_id, principal.org_id)
    else:
        events = store.query_by_org(
            org_id=principal.org_id,
            limit=limit,
            offset=offset,
            since=since_dt,
            until=until_dt,
        )

    return FeedbackListResponse(
        items=[FeedbackResponse.from_domain(e) for e in events],
        total=len(events),
        limit=limit,
        offset=offset,
    )


@router.get("/metrics", response_model=FeedbackMetrics)
async def get_metrics(
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_READ)),
    store: PostgresFeedbackStore = Depends(get_feedback_store),
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    """
    Get aggregate metrics for feedback events.

    Returns acceptance rate, outcome distribution, and per-tool metrics
    for the authenticated org.
    """
    # Parse datetime strings if provided
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' datetime format")

    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'until' datetime format")

    metrics = store.get_aggregate_metrics(
        org_id=principal.org_id,
        since=since_dt,
        until=until_dt,
    )

    return FeedbackMetrics(
        total_events=metrics["total_events"],
        acceptance_rate=metrics["acceptance_rate"],
        outcome_distribution=metrics["outcome_distribution"],
        average_decision_time_ms=metrics["average_decision_time_ms"],
        by_tool=metrics["by_tool"],
    )


@router.get("/by-tool", response_model=ToolMetricsResponse)
async def get_metrics_by_tool(
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_READ)),
    store: PostgresFeedbackStore = Depends(get_feedback_store),
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    """
    Get feedback metrics grouped by tool.

    Returns per-tool acceptance rates and outcome counts for the
    authenticated org.
    """
    # Parse datetime strings if provided
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'since' datetime format")

    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'until' datetime format")

    # Query all events and group by tool
    events = store.query_by_org(
        org_id=principal.org_id,
        since=since_dt,
        until=until_dt,
    )

    # Group by tool
    tool_data: dict[str, dict[str, int]] = {}
    for event in events:
        if event.tool_name not in tool_data:
            tool_data[event.tool_name] = {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "modified": 0,
                "ignored": 0,
            }
        tool_data[event.tool_name]["total"] += 1
        tool_data[event.tool_name][event.outcome.value] += 1

    # Build response
    tools = []
    for tool_name, counts in tool_data.items():
        total_actionable = counts["total"] - counts["ignored"]
        acceptance_rate = (
            counts["accepted"] / total_actionable
            if total_actionable > 0
            else 0.0
        )
        tools.append(ToolMetrics(
            tool_name=tool_name,
            total_count=counts["total"],
            accepted_count=counts["accepted"],
            rejected_count=counts["rejected"],
            modified_count=counts["modified"],
            ignored_count=counts["ignored"],
            acceptance_rate=acceptance_rate,
        ))

    return ToolMetricsResponse(tools=tools)
