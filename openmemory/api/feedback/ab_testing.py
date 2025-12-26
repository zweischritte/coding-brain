"""A/B testing framework for retrieval experiments.

Implements FR-002: Feedback Integration - A/B testing framework.
Supports experiment management, deterministic variant assignment,
traffic allocation, and guardrails with auto-rollback.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment."""

    DRAFT = "draft"  # Not yet started
    RUNNING = "running"  # Actively assigning users
    PAUSED = "paused"  # Temporarily stopped
    COMPLETED = "completed"  # Finished successfully
    ROLLED_BACK = "rolled_back"  # Stopped due to guardrail trigger


@dataclass
class ExperimentVariant:
    """A variant within an experiment.

    Attributes:
        variant_id: Unique identifier for this variant
        name: Human-readable name
        description: Optional description
        weight: Traffic weight (0.0 to 1.0)
        config: Optional configuration dict for this variant
    """

    variant_id: str
    name: str
    weight: float
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate variant."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Variant weight must be between 0 and 1, got {self.weight}")


@dataclass
class GuardrailConfig:
    """Configuration for experiment guardrails.

    Guardrails define thresholds that trigger auto-rollback
    when violated.

    Attributes:
        metric_name: Name of the metric to monitor
        min_threshold: Minimum acceptable value (if any)
        max_threshold: Maximum acceptable value (if any)
        sample_size: Minimum samples before evaluation
    """

    metric_name: str
    sample_size: int
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    def __post_init__(self):
        """Validate guardrail."""
        if self.min_threshold is None and self.max_threshold is None:
            raise ValueError("Guardrail must have at least min_threshold or max_threshold")


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the guardrail check passed
        metric_name: Name of the metric checked
        metric_value: Actual value of the metric
        threshold: Threshold that was checked
        reason: Explanation if failed
    """

    passed: bool
    metric_name: str
    metric_value: float
    threshold: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class Experiment:
    """An A/B test experiment.

    Attributes:
        experiment_id: Unique identifier
        name: Human-readable name
        org_id: Organization this experiment belongs to
        variants: List of variants
        description: Optional description
        status: Current status
        traffic_percentage: What % of users to include (0.0 to 1.0)
        start_time: When experiment started/will start
        end_time: When experiment ended/will end
        guardrails: Optional guardrail configurations
        created_at: When experiment was created
        updated_at: Last update time
    """

    experiment_id: str
    name: str
    org_id: str
    variants: list[ExperimentVariant]
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    traffic_percentage: float = 1.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    guardrails: list[GuardrailConfig] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate experiment."""
        if not self.variants:
            raise ValueError("Experiment must have at least one variant")

        # Validate weights sum to 1.0
        total_weight = sum(v.weight for v in self.variants)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Variant weights must sum to 1.0, got {total_weight:.2f}"
            )


@dataclass
class VariantAssignment:
    """Assignment of a user to a variant.

    Attributes:
        experiment_id: The experiment ID
        variant_id: The assigned variant ID
        user_id: The user who was assigned
        assigned_at: When assignment was made
        variant_config: Configuration from the variant
    """

    experiment_id: str
    variant_id: str
    user_id: str
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    variant_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for the A/B testing framework."""

    pass  # Placeholder for future config options


class ABTestingFramework:
    """A/B testing framework for retrieval experiments.

    Provides deterministic variant assignment, traffic allocation,
    and guardrail monitoring.
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize framework.

        Args:
            config: Optional configuration
        """
        self._config = config or ExperimentConfig()
        self._experiments: dict[str, Experiment] = {}

    def register_experiment(self, experiment: Experiment) -> None:
        """Register an experiment.

        Args:
            experiment: The experiment to register
        """
        self._experiments[experiment.experiment_id] = experiment
        logger.info(f"Registered experiment: {experiment.experiment_id}")

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID

        Returns:
            The experiment or None if not found
        """
        return self._experiments.get(experiment_id)

    def list_experiments(self, org_id: str) -> list[Experiment]:
        """List all experiments for an org.

        Args:
            org_id: The organization ID

        Returns:
            List of experiments for this org
        """
        return [e for e in self._experiments.values() if e.org_id == org_id]

    def get_assignment(
        self,
        experiment_id: str,
        user_id: str,
        org_id: str,
    ) -> Optional[VariantAssignment]:
        """Get variant assignment for a user.

        Assignment is deterministic - same user always gets same variant.
        Uses consistent hashing based on user_id + experiment_id.

        Args:
            experiment_id: The experiment ID
            user_id: The user ID
            org_id: The organization ID

        Returns:
            VariantAssignment or None if not eligible
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        # Check experiment is running
        if experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check org matches
        if experiment.org_id != org_id:
            return None

        # Check traffic percentage
        if not self._is_in_traffic(user_id, experiment_id, experiment.traffic_percentage):
            return None

        # Determine variant using consistent hashing
        variant = self._select_variant(user_id, experiment_id, experiment.variants)

        return VariantAssignment(
            experiment_id=experiment_id,
            variant_id=variant.variant_id,
            user_id=user_id,
            variant_config=variant.config,
        )

    def get_all_assignments(
        self,
        user_id: str,
        org_id: str,
    ) -> list[VariantAssignment]:
        """Get all active experiment assignments for a user.

        Args:
            user_id: The user ID
            org_id: The organization ID

        Returns:
            List of all active assignments
        """
        assignments = []
        for exp in self.list_experiments(org_id):
            assignment = self.get_assignment(exp.experiment_id, user_id, org_id)
            if assignment:
                assignments.append(assignment)
        return assignments

    def _is_in_traffic(
        self,
        user_id: str,
        experiment_id: str,
        traffic_percentage: float,
    ) -> bool:
        """Check if user is in the traffic sample.

        Uses consistent hashing so same user is always in/out.

        Args:
            user_id: The user ID
            experiment_id: The experiment ID
            traffic_percentage: Percentage of traffic (0.0 to 1.0)

        Returns:
            True if user is in the traffic sample
        """
        if traffic_percentage >= 1.0:
            return True
        if traffic_percentage <= 0.0:
            return False

        # Use consistent hash to determine inclusion
        hash_input = f"traffic:{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0

        return bucket < traffic_percentage

    def _select_variant(
        self,
        user_id: str,
        experiment_id: str,
        variants: list[ExperimentVariant],
    ) -> ExperimentVariant:
        """Select a variant for a user using consistent hashing.

        Args:
            user_id: The user ID
            experiment_id: The experiment ID
            variants: Available variants

        Returns:
            The selected variant
        """
        # Use consistent hash to select variant
        hash_input = f"variant:{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0

        # Walk through variants by weight
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant

        # Fallback to last variant (shouldn't happen with valid weights)
        return variants[-1]

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment (move from draft to running).

        Args:
            experiment_id: The experiment ID
        """
        experiment = self._experiments.get(experiment_id)
        if experiment:
            self._experiments[experiment_id] = Experiment(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                org_id=experiment.org_id,
                variants=experiment.variants,
                description=experiment.description,
                status=ExperimentStatus.RUNNING,
                traffic_percentage=experiment.traffic_percentage,
                start_time=datetime.now(timezone.utc),
                end_time=experiment.end_time,
                guardrails=experiment.guardrails,
                created_at=experiment.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            logger.info(f"Started experiment: {experiment_id}")

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause a running experiment.

        Args:
            experiment_id: The experiment ID
        """
        experiment = self._experiments.get(experiment_id)
        if experiment:
            self._experiments[experiment_id] = Experiment(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                org_id=experiment.org_id,
                variants=experiment.variants,
                description=experiment.description,
                status=ExperimentStatus.PAUSED,
                traffic_percentage=experiment.traffic_percentage,
                start_time=experiment.start_time,
                end_time=experiment.end_time,
                guardrails=experiment.guardrails,
                created_at=experiment.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            logger.info(f"Paused experiment: {experiment_id}")

    def complete_experiment(self, experiment_id: str) -> None:
        """Complete an experiment.

        Args:
            experiment_id: The experiment ID
        """
        experiment = self._experiments.get(experiment_id)
        if experiment:
            self._experiments[experiment_id] = Experiment(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                org_id=experiment.org_id,
                variants=experiment.variants,
                description=experiment.description,
                status=ExperimentStatus.COMPLETED,
                traffic_percentage=experiment.traffic_percentage,
                start_time=experiment.start_time,
                end_time=datetime.now(timezone.utc),
                guardrails=experiment.guardrails,
                created_at=experiment.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            logger.info(f"Completed experiment: {experiment_id}")

    def rollback_experiment(self, experiment_id: str, reason: str) -> None:
        """Rollback an experiment (guardrail triggered).

        Args:
            experiment_id: The experiment ID
            reason: Why the experiment was rolled back
        """
        experiment = self._experiments.get(experiment_id)
        if experiment:
            self._experiments[experiment_id] = Experiment(
                experiment_id=experiment.experiment_id,
                name=experiment.name,
                org_id=experiment.org_id,
                variants=experiment.variants,
                description=experiment.description,
                status=ExperimentStatus.ROLLED_BACK,
                traffic_percentage=experiment.traffic_percentage,
                start_time=experiment.start_time,
                end_time=datetime.now(timezone.utc),
                guardrails=experiment.guardrails,
                created_at=experiment.created_at,
                updated_at=datetime.now(timezone.utc),
            )
            logger.warning(f"Rolled back experiment {experiment_id}: {reason}")

    def check_guardrail(
        self,
        guardrail: GuardrailConfig,
        metric_value: float,
        sample_size: int,
    ) -> GuardrailResult:
        """Check if a guardrail is violated.

        Args:
            guardrail: The guardrail configuration
            metric_value: Current value of the metric
            sample_size: Number of samples collected

        Returns:
            GuardrailResult indicating pass/fail
        """
        # Check if we have enough samples
        if sample_size < guardrail.sample_size:
            return GuardrailResult(
                passed=True,  # Don't fail with insufficient data
                metric_name=guardrail.metric_name,
                metric_value=metric_value,
                reason=f"Insufficient samples ({sample_size}/{guardrail.sample_size})",
            )

        # Check min threshold
        if guardrail.min_threshold is not None:
            if metric_value < guardrail.min_threshold:
                return GuardrailResult(
                    passed=False,
                    metric_name=guardrail.metric_name,
                    metric_value=metric_value,
                    threshold=guardrail.min_threshold,
                    reason=f"{guardrail.metric_name} ({metric_value:.3f}) is below threshold ({guardrail.min_threshold:.3f})",
                )

        # Check max threshold
        if guardrail.max_threshold is not None:
            if metric_value > guardrail.max_threshold:
                return GuardrailResult(
                    passed=False,
                    metric_name=guardrail.metric_name,
                    metric_value=metric_value,
                    threshold=guardrail.max_threshold,
                    reason=f"{guardrail.metric_name} ({metric_value:.3f}) is above threshold ({guardrail.max_threshold:.3f})",
                )

        return GuardrailResult(
            passed=True,
            metric_name=guardrail.metric_name,
            metric_value=metric_value,
        )
