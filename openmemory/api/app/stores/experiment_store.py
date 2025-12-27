"""
PostgreSQL-backed ExperimentStore with tenant isolation.

This module provides persistent storage for A/B test experiments with:
- Tenant isolation via org_id
- Status history tracking for audit purposes
- Variant assignment persistence

Note: This implementation defines its own types that are compatible with
the existing openmemory.api.feedback.ab_testing module, but doesn't depend
on it to avoid circular import issues.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session, relationship

from app.database import Base


# ============================================================================
# Domain Types (compatible with openmemory.api.feedback.ab_testing)
# ============================================================================


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExperimentVariant:
    """A variant within an experiment."""

    variant_id: str
    name: str
    weight: float
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """An A/B test experiment."""

    experiment_id: str
    name: str
    org_id: str
    variants: list[ExperimentVariant]
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    traffic_percentage: float = 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VariantAssignment:
    """Assignment of a user to a variant."""

    experiment_id: str
    variant_id: str
    user_id: str
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    variant_config: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Abstract Store Interface
# ============================================================================


class ExperimentStoreInterface(ABC):
    """Abstract interface for experiment storage."""

    @abstractmethod
    def create(self, experiment: Experiment) -> str:
        """Create a new experiment."""
        pass

    @abstractmethod
    def get(self, experiment_id: str, org_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        pass

    @abstractmethod
    def list(
        self,
        org_id: str,
        status: Optional[ExperimentStatus] = None,
    ) -> list[Experiment]:
        """List experiments for an org."""
        pass

    @abstractmethod
    def update(self, experiment: Experiment) -> Optional[Experiment]:
        """Update an existing experiment."""
        pass

    @abstractmethod
    def update_status(
        self,
        experiment_id: str,
        org_id: str,
        status: ExperimentStatus,
        reason: Optional[str] = None,
    ) -> bool:
        """Update experiment status and record history."""
        pass

    @abstractmethod
    def get_status_history(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[dict[str, Any]]:
        """Get status change history for an experiment."""
        pass

    @abstractmethod
    def record_assignment(
        self,
        assignment: VariantAssignment,
        org_id: str,
    ) -> None:
        """Record a variant assignment."""
        pass

    @abstractmethod
    def get_assignments(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[VariantAssignment]:
        """Get all assignments for an experiment."""
        pass

    @abstractmethod
    def get_user_assignment(
        self,
        experiment_id: str,
        user_id: str,
        org_id: str,
    ) -> Optional[VariantAssignment]:
        """Get a user's assignment for an experiment."""
        pass


# ============================================================================
# SQLAlchemy Models
# ============================================================================


class ExperimentModel(Base):
    """SQLAlchemy model for experiments."""

    __tablename__ = "experiments"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(String, nullable=False, unique=True, index=True)
    org_id = Column(String, nullable=False, index=True)

    name = Column(String, nullable=False)
    description = Column(String, nullable=True, default="")
    status = Column(
        SQLEnum(ExperimentStatus), nullable=False, default=ExperimentStatus.DRAFT
    )
    traffic_percentage = Column(Float, nullable=False, default=1.0)

    # Variants stored as JSON array
    variants = Column(JSON, nullable=False, default=list)

    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)
    )

    # Relationships
    status_history = relationship(
        "ExperimentStatusHistoryModel",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )
    assignments = relationship(
        "VariantAssignmentModel",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_experiments_org", "org_id"),
        Index("idx_experiments_org_status", "org_id", "status"),
    )

    def to_domain(self) -> Experiment:
        """Convert SQLAlchemy model to domain object."""
        variants = [
            ExperimentVariant(
                variant_id=v["variant_id"],
                name=v["name"],
                weight=v["weight"],
                description=v.get("description", ""),
                config=v.get("config", {}),
            )
            for v in (self.variants or [])
        ]
        return Experiment(
            experiment_id=self.experiment_id,
            name=self.name,
            org_id=self.org_id,
            variants=variants,
            description=self.description or "",
            status=self.status,
            traffic_percentage=self.traffic_percentage,
            start_time=self.start_time,
            end_time=self.end_time,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, experiment: Experiment) -> "ExperimentModel":
        """Create SQLAlchemy model from domain object."""
        variants = [
            {
                "variant_id": v.variant_id,
                "name": v.name,
                "weight": v.weight,
                "description": v.description,
                "config": v.config,
            }
            for v in experiment.variants
        ]
        return cls(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            org_id=experiment.org_id,
            variants=variants,
            description=experiment.description,
            status=experiment.status,
            traffic_percentage=experiment.traffic_percentage,
            start_time=experiment.start_time,
            end_time=experiment.end_time,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at,
        )


class ExperimentStatusHistoryModel(Base):
    """SQLAlchemy model for experiment status history."""

    __tablename__ = "experiment_status_history"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        String, ForeignKey("experiments.experiment_id"), nullable=False, index=True
    )
    org_id = Column(String, nullable=False, index=True)

    from_status = Column(SQLEnum(ExperimentStatus), nullable=True)
    to_status = Column(SQLEnum(ExperimentStatus), nullable=False)
    reason = Column(String, nullable=True)
    changed_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)
    )

    experiment = relationship("ExperimentModel", back_populates="status_history")

    __table_args__ = (Index("idx_status_history_exp_org", "experiment_id", "org_id"),)


class VariantAssignmentModel(Base):
    """SQLAlchemy model for variant assignments."""

    __tablename__ = "variant_assignments"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(
        String, ForeignKey("experiments.experiment_id"), nullable=False, index=True
    )
    org_id = Column(String, nullable=False, index=True)

    variant_id = Column(String, nullable=False)
    user_id = Column(String, nullable=False, index=True)
    variant_config = Column(JSON, nullable=True, default=dict)
    assigned_at = Column(
        DateTime(timezone=True), nullable=False, default=datetime.now(timezone.utc)
    )

    experiment = relationship("ExperimentModel", back_populates="assignments")

    __table_args__ = (
        Index("idx_assignments_exp_user", "experiment_id", "user_id"),
        Index("idx_assignments_exp_org", "experiment_id", "org_id"),
    )

    def to_domain(self) -> VariantAssignment:
        """Convert SQLAlchemy model to domain object."""
        return VariantAssignment(
            experiment_id=self.experiment_id,
            variant_id=self.variant_id,
            user_id=self.user_id,
            assigned_at=self.assigned_at,
            variant_config=self.variant_config or {},
        )


# ============================================================================
# PostgreSQL Store Implementation
# ============================================================================


class PostgresExperimentStore(ExperimentStoreInterface):
    """PostgreSQL-backed implementation of ExperimentStore.

    Provides persistent storage for A/B test experiments with tenant
    isolation via org_id filtering and status history tracking.

    Args:
        db: SQLAlchemy session

    Example:
        store = PostgresExperimentStore(db)
        store.create(experiment)
        exp = store.get(experiment_id, org_id)
    """

    def __init__(self, db: Session):
        """Initialize the store.

        Args:
            db: SQLAlchemy session for database operations
        """
        self._db = db

    def create(self, experiment: Experiment) -> str:
        """Create a new experiment.

        Args:
            experiment: The experiment to create

        Returns:
            The experiment_id of the created experiment
        """
        model = ExperimentModel.from_domain(experiment)
        self._db.add(model)

        # Record initial status in history
        history = ExperimentStatusHistoryModel(
            experiment_id=experiment.experiment_id,
            org_id=experiment.org_id,
            from_status=None,
            to_status=experiment.status,
            reason="Experiment created",
        )
        self._db.add(history)

        self._db.commit()
        return experiment.experiment_id

    def get(self, experiment_id: str, org_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID
            org_id: The org ID (required for scoping)

        Returns:
            The experiment or None if not found
        """
        model = (
            self._db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment_id)
            .filter(ExperimentModel.org_id == org_id)
            .first()
        )
        return model.to_domain() if model else None

    def list(
        self,
        org_id: str,
        status: Optional[ExperimentStatus] = None,
    ) -> list[Experiment]:
        """List experiments for an org.

        Args:
            org_id: The org ID to query
            status: Optional status filter

        Returns:
            List of experiments
        """
        query = self._db.query(ExperimentModel).filter(
            ExperimentModel.org_id == org_id
        )

        if status is not None:
            query = query.filter(ExperimentModel.status == status)

        query = query.order_by(ExperimentModel.created_at.desc())
        return [model.to_domain() for model in query.all()]

    def update(self, experiment: Experiment) -> Optional[Experiment]:
        """Update an existing experiment.

        Args:
            experiment: The experiment with updates

        Returns:
            The updated experiment or None if not found/unauthorized
        """
        model = (
            self._db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment.experiment_id)
            .filter(ExperimentModel.org_id == experiment.org_id)
            .first()
        )

        if not model:
            return None

        # Update fields
        model.name = experiment.name
        model.description = experiment.description
        model.status = experiment.status
        model.traffic_percentage = experiment.traffic_percentage
        model.variants = [
            {
                "variant_id": v.variant_id,
                "name": v.name,
                "weight": v.weight,
                "description": v.description,
                "config": v.config,
            }
            for v in experiment.variants
        ]
        model.start_time = experiment.start_time
        model.end_time = experiment.end_time
        model.updated_at = datetime.now(timezone.utc)

        self._db.commit()
        self._db.refresh(model)
        return model.to_domain()

    def update_status(
        self,
        experiment_id: str,
        org_id: str,
        status: ExperimentStatus,
        reason: Optional[str] = None,
    ) -> bool:
        """Update experiment status and record history.

        Args:
            experiment_id: The experiment ID
            org_id: The org ID (required for scoping)
            status: The new status
            reason: Optional reason for the change

        Returns:
            True if updated, False if not found/unauthorized
        """
        model = (
            self._db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment_id)
            .filter(ExperimentModel.org_id == org_id)
            .first()
        )

        if not model:
            return False

        old_status = model.status

        # Update experiment status
        model.status = status
        model.updated_at = datetime.now(timezone.utc)

        # Handle timestamps
        if status == ExperimentStatus.RUNNING and model.start_time is None:
            model.start_time = datetime.now(timezone.utc)
        if status in (
            ExperimentStatus.COMPLETED,
            ExperimentStatus.ROLLED_BACK,
        ):
            model.end_time = datetime.now(timezone.utc)

        # Record in history
        history = ExperimentStatusHistoryModel(
            experiment_id=experiment_id,
            org_id=org_id,
            from_status=old_status,
            to_status=status,
            reason=reason,
        )
        self._db.add(history)

        self._db.commit()
        return True

    def get_status_history(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[dict[str, Any]]:
        """Get status change history for an experiment.

        Args:
            experiment_id: The experiment ID
            org_id: The org ID (required for scoping)

        Returns:
            List of status change records
        """
        entries = (
            self._db.query(ExperimentStatusHistoryModel)
            .filter(ExperimentStatusHistoryModel.experiment_id == experiment_id)
            .filter(ExperimentStatusHistoryModel.org_id == org_id)
            .order_by(ExperimentStatusHistoryModel.changed_at)
            .all()
        )

        return [
            {
                "from_status": (
                    entry.from_status.value if entry.from_status else None
                ),
                "to_status": entry.to_status.value,
                "reason": entry.reason,
                "changed_at": entry.changed_at.isoformat(),
            }
            for entry in entries
        ]

    def record_assignment(
        self,
        assignment: VariantAssignment,
        org_id: str,
    ) -> None:
        """Record a variant assignment.

        Args:
            assignment: The assignment to record
            org_id: The org ID (required for scoping)
        """
        model = VariantAssignmentModel(
            experiment_id=assignment.experiment_id,
            org_id=org_id,
            variant_id=assignment.variant_id,
            user_id=assignment.user_id,
            variant_config=assignment.variant_config,
            assigned_at=assignment.assigned_at,
        )
        self._db.add(model)
        self._db.commit()

    def get_assignments(
        self,
        experiment_id: str,
        org_id: str,
    ) -> list[VariantAssignment]:
        """Get all assignments for an experiment.

        Args:
            experiment_id: The experiment ID
            org_id: The org ID (required for scoping)

        Returns:
            List of variant assignments
        """
        models = (
            self._db.query(VariantAssignmentModel)
            .filter(VariantAssignmentModel.experiment_id == experiment_id)
            .filter(VariantAssignmentModel.org_id == org_id)
            .order_by(VariantAssignmentModel.assigned_at)
            .all()
        )
        return [model.to_domain() for model in models]

    def get_user_assignment(
        self,
        experiment_id: str,
        user_id: str,
        org_id: str,
    ) -> Optional[VariantAssignment]:
        """Get a user's assignment for an experiment.

        Args:
            experiment_id: The experiment ID
            user_id: The user ID
            org_id: The org ID (required for scoping)

        Returns:
            The assignment or None if not found
        """
        model = (
            self._db.query(VariantAssignmentModel)
            .filter(VariantAssignmentModel.experiment_id == experiment_id)
            .filter(VariantAssignmentModel.user_id == user_id)
            .filter(VariantAssignmentModel.org_id == org_id)
            .order_by(VariantAssignmentModel.assigned_at.desc())
            .first()
        )
        return model.to_domain() if model else None
