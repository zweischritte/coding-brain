"""
Experiments Router - REST API for A/B test experiments.

Provides endpoints for creating, managing, and tracking A/B test experiments
including variant assignment and status history.
"""

import random
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.stores.experiment_store import (
    Experiment,
    ExperimentStatus,
    ExperimentVariant,
    PostgresExperimentStore,
    VariantAssignment,
)


router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


# ============================================================================
# Pydantic Schemas
# ============================================================================


class StatusEnum(str, Enum):
    """Valid experiment status values."""
    draft = "draft"
    running = "running"
    paused = "paused"
    completed = "completed"
    rolled_back = "rolled_back"


class VariantCreate(BaseModel):
    """Request body for creating a variant."""
    name: str = Field(..., min_length=1, description="Variant name")
    weight: float = Field(..., ge=0.0, le=1.0, description="Traffic weight (0-1)")
    description: str = Field("", description="Variant description")
    config: dict[str, Any] = Field(default_factory=dict, description="Variant config")


class ExperimentCreate(BaseModel):
    """Request body for creating an experiment."""
    name: str = Field(..., min_length=1, description="Experiment name")
    description: str = Field("", description="Experiment description")
    variants: list[VariantCreate] = Field(..., min_length=2, description="At least 2 variants")
    traffic_percentage: float = Field(1.0, ge=0.0, le=1.0, description="Percentage of traffic to include")

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 variants are required")
        return v


class VariantResponse(BaseModel):
    """Response for a variant."""
    variant_id: str
    name: str
    weight: float
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentResponse(BaseModel):
    """Response body for an experiment."""
    experiment_id: str
    name: str
    description: str
    status: str
    variants: list[VariantResponse]
    traffic_percentage: float
    created_at: datetime
    updated_at: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    org_id: str

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_domain(cls, exp: Experiment) -> "ExperimentResponse":
        """Create response from domain experiment."""
        return cls(
            experiment_id=exp.experiment_id,
            name=exp.name,
            description=exp.description,
            status=exp.status.value,
            variants=[
                VariantResponse(
                    variant_id=v.variant_id,
                    name=v.name,
                    weight=v.weight,
                    description=v.description,
                    config=v.config,
                )
                for v in exp.variants
            ],
            traffic_percentage=exp.traffic_percentage,
            created_at=exp.created_at,
            updated_at=exp.updated_at,
            start_time=exp.start_time,
            end_time=exp.end_time,
            org_id=exp.org_id,
        )


class ExperimentListResponse(BaseModel):
    """Response body for listing experiments."""
    items: list[ExperimentResponse]
    total: int


class StatusUpdate(BaseModel):
    """Request body for updating experiment status."""
    status: StatusEnum = Field(..., description="New status")
    reason: Optional[str] = Field(None, description="Reason for status change")


class AssignmentResponse(BaseModel):
    """Response body for a variant assignment."""
    experiment_id: str
    variant_id: str
    variant_name: str
    assigned_at: datetime
    variant_config: dict[str, Any] = Field(default_factory=dict)


class StatusHistoryEntry(BaseModel):
    """A status history entry."""
    from_status: Optional[str]
    to_status: str
    reason: Optional[str]
    changed_at: str


class StatusHistoryResponse(BaseModel):
    """Response body for status history."""
    history: list[StatusHistoryEntry]


# ============================================================================
# Dependencies
# ============================================================================


def get_experiment_store(db: Session = Depends(get_db)) -> PostgresExperimentStore:
    """Dependency for experiment store."""
    return PostgresExperimentStore(db)


# ============================================================================
# Endpoints
# ============================================================================


@router.post("", status_code=201, response_model=ExperimentResponse)
async def create_experiment(
    data: ExperimentCreate,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_WRITE)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Create a new A/B test experiment.

    The experiment is created in DRAFT status. Use the status update endpoint
    to start the experiment.
    """
    # Generate IDs for variants
    variants = [
        ExperimentVariant(
            variant_id=str(uuid.uuid4()),
            name=v.name,
            weight=v.weight,
            description=v.description,
            config=v.config,
        )
        for v in data.variants
    ]

    # Create experiment with org from principal
    experiment = Experiment(
        experiment_id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        org_id=principal.org_id,
        variants=variants,
        traffic_percentage=data.traffic_percentage,
        status=ExperimentStatus.DRAFT,
    )

    store.create(experiment)
    return ExperimentResponse.from_domain(experiment)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_READ)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
    status: Optional[str] = None,
):
    """
    List experiments for the authenticated org.

    Optionally filter by status.
    """
    # Convert status string to enum if provided
    status_enum = None
    if status:
        try:
            status_enum = ExperimentStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: {[s.value for s in ExperimentStatus]}"
            )

    experiments = store.list(
        org_id=principal.org_id,
        status=status_enum,
    )

    return ExperimentListResponse(
        items=[ExperimentResponse.from_domain(e) for e in experiments],
        total=len(experiments),
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_READ)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Get experiment details by ID.

    Returns 404 if the experiment doesn't exist or belongs to a different org.
    """
    experiment = store.get(experiment_id, principal.org_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return ExperimentResponse.from_domain(experiment)


@router.put("/{experiment_id}/status", response_model=ExperimentResponse)
async def update_experiment_status(
    experiment_id: str,
    data: StatusUpdate,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_WRITE)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Update experiment status.

    Valid transitions:
    - draft -> running, completed
    - running -> paused, completed, rolled_back
    - paused -> running, completed, rolled_back
    """
    # Map enum to domain status
    status_map = {
        StatusEnum.draft: ExperimentStatus.DRAFT,
        StatusEnum.running: ExperimentStatus.RUNNING,
        StatusEnum.paused: ExperimentStatus.PAUSED,
        StatusEnum.completed: ExperimentStatus.COMPLETED,
        StatusEnum.rolled_back: ExperimentStatus.ROLLED_BACK,
    }

    new_status = status_map[data.status]

    # Update status
    success = store.update_status(
        experiment_id=experiment_id,
        org_id=principal.org_id,
        status=new_status,
        reason=data.reason,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Return updated experiment
    experiment = store.get(experiment_id, principal.org_id)
    return ExperimentResponse.from_domain(experiment)


@router.post("/{experiment_id}/assign", response_model=AssignmentResponse)
async def assign_variant(
    experiment_id: str,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_WRITE)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Assign the authenticated user to a variant.

    If the user is already assigned, returns the existing assignment (sticky).
    Assignment only works for RUNNING experiments.
    """
    # Get experiment
    experiment = store.get(experiment_id, principal.org_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Check if experiment is running
    if experiment.status != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot assign to experiment with status '{experiment.status.value}'. Must be 'running'."
        )

    # Check for existing assignment (sticky)
    existing = store.get_user_assignment(
        experiment_id=experiment_id,
        user_id=principal.user_id,
        org_id=principal.org_id,
    )

    if existing:
        # Get variant name from experiment
        variant_name = next(
            (v.name for v in experiment.variants if v.variant_id == existing.variant_id),
            "unknown"
        )
        return AssignmentResponse(
            experiment_id=existing.experiment_id,
            variant_id=existing.variant_id,
            variant_name=variant_name,
            assigned_at=existing.assigned_at,
            variant_config=existing.variant_config,
        )

    # Check if user is in traffic percentage
    if random.random() > experiment.traffic_percentage:
        raise HTTPException(
            status_code=400,
            detail="User not included in experiment traffic percentage"
        )

    # Select variant based on weights
    weights = [v.weight for v in experiment.variants]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    rand = random.random()
    cumulative = 0.0
    selected_variant = experiment.variants[0]

    for variant, weight in zip(experiment.variants, normalized_weights):
        cumulative += weight
        if rand < cumulative:
            selected_variant = variant
            break

    # Record assignment
    assignment = VariantAssignment(
        experiment_id=experiment_id,
        variant_id=selected_variant.variant_id,
        user_id=principal.user_id,
        variant_config=selected_variant.config,
    )
    store.record_assignment(assignment, principal.org_id)

    return AssignmentResponse(
        experiment_id=assignment.experiment_id,
        variant_id=assignment.variant_id,
        variant_name=selected_variant.name,
        assigned_at=assignment.assigned_at,
        variant_config=assignment.variant_config,
    )


@router.get("/{experiment_id}/assignment", response_model=AssignmentResponse)
async def get_assignment(
    experiment_id: str,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_READ)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Get the authenticated user's assignment for an experiment.

    Returns 404 if the user is not assigned or experiment doesn't exist.
    """
    # Get experiment first to verify it exists
    experiment = store.get(experiment_id, principal.org_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Get assignment
    assignment = store.get_user_assignment(
        experiment_id=experiment_id,
        user_id=principal.user_id,
        org_id=principal.org_id,
    )

    if not assignment:
        raise HTTPException(status_code=404, detail="No assignment found for user")

    # Get variant name from experiment
    variant_name = next(
        (v.name for v in experiment.variants if v.variant_id == assignment.variant_id),
        "unknown"
    )

    return AssignmentResponse(
        experiment_id=assignment.experiment_id,
        variant_id=assignment.variant_id,
        variant_name=variant_name,
        assigned_at=assignment.assigned_at,
        variant_config=assignment.variant_config,
    )


@router.get("/{experiment_id}/history", response_model=StatusHistoryResponse)
async def get_status_history(
    experiment_id: str,
    principal: Principal = Depends(require_scopes(Scope.EXPERIMENTS_READ)),
    store: PostgresExperimentStore = Depends(get_experiment_store),
):
    """
    Get status change history for an experiment.

    Returns chronologically ordered list of status changes.
    """
    # Verify experiment exists
    experiment = store.get(experiment_id, principal.org_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    history = store.get_status_history(experiment_id, principal.org_id)

    return StatusHistoryResponse(
        history=[
            StatusHistoryEntry(
                from_status=entry["from_status"],
                to_status=entry["to_status"],
                reason=entry["reason"],
                changed_at=entry["changed_at"],
            )
            for entry in history
        ]
    )
