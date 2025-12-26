"""Tests for A/B testing framework.

TDD tests for experiment management and variant assignment following v9 plan:
- Experiment creation and management
- Variant assignment (deterministic by user)
- Traffic allocation
- Experiment status (running, paused, completed)
- Guardrails and auto-rollback
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmemory.api.feedback.ab_testing import (
    ABTestingFramework,
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentVariant,
    GuardrailConfig,
    VariantAssignment,
)


class TestExperimentVariant:
    """Tests for ExperimentVariant dataclass."""

    def test_create_variant(self):
        """Can create a variant."""
        variant = ExperimentVariant(
            variant_id="control",
            name="Control",
            description="Default RRF weights",
            weight=0.5,
        )
        assert variant.variant_id == "control"
        assert variant.name == "Control"
        assert variant.weight == 0.5

    def test_variant_with_config(self):
        """Variant can include configuration."""
        config = {"rrf_weights": {"vector": 0.40, "lexical": 0.35, "graph": 0.25}}
        variant = ExperimentVariant(
            variant_id="treatment",
            name="New Weights",
            weight=0.5,
            config=config,
        )
        assert variant.config == config

    def test_variant_weight_validation(self):
        """Variant weight must be between 0 and 1."""
        with pytest.raises(ValueError, match="weight"):
            ExperimentVariant(variant_id="bad", name="Bad", weight=1.5)

        with pytest.raises(ValueError, match="weight"):
            ExperimentVariant(variant_id="bad", name="Bad", weight=-0.1)


class TestExperimentStatus:
    """Tests for ExperimentStatus enum."""

    def test_has_draft_status(self):
        """Has draft status."""
        assert ExperimentStatus.DRAFT.value == "draft"

    def test_has_running_status(self):
        """Has running status."""
        assert ExperimentStatus.RUNNING.value == "running"

    def test_has_paused_status(self):
        """Has paused status."""
        assert ExperimentStatus.PAUSED.value == "paused"

    def test_has_completed_status(self):
        """Has completed status."""
        assert ExperimentStatus.COMPLETED.value == "completed"

    def test_has_rolled_back_status(self):
        """Has rolled_back status for guardrail triggers."""
        assert ExperimentStatus.ROLLED_BACK.value == "rolled_back"


class TestExperiment:
    """Tests for Experiment dataclass."""

    def test_create_experiment(self):
        """Can create an experiment."""
        exp = Experiment(
            experiment_id="exp_001",
            name="RRF Weight Test",
            description="Test new RRF weights",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=0.5),
                ExperimentVariant(variant_id="treatment", name="Treatment", weight=0.5),
            ],
        )
        assert exp.experiment_id == "exp_001"
        assert exp.name == "RRF Weight Test"
        assert len(exp.variants) == 2

    def test_experiment_default_status_is_draft(self):
        """New experiments default to draft status."""
        exp = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
        )
        assert exp.status == ExperimentStatus.DRAFT

    def test_experiment_validates_variant_weights_sum(self):
        """Variant weights should sum to 1.0."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            Experiment(
                experiment_id="exp_001",
                name="Test",
                org_id="org789",
                variants=[
                    ExperimentVariant(variant_id="a", name="A", weight=0.3),
                    ExperimentVariant(variant_id="b", name="B", weight=0.3),
                ],  # Sum is 0.6, not 1.0
            )

    def test_experiment_requires_at_least_one_variant(self):
        """Experiment must have at least one variant."""
        with pytest.raises(ValueError, match="at least one variant"):
            Experiment(
                experiment_id="exp_001",
                name="Test",
                org_id="org789",
                variants=[],
            )

    def test_experiment_with_traffic_percentage(self):
        """Experiment can specify traffic percentage."""
        exp = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            traffic_percentage=0.1,  # 10% of users
        )
        assert exp.traffic_percentage == 0.1

    def test_experiment_with_start_and_end_dates(self):
        """Experiment can have scheduled start/end."""
        now = datetime.now(timezone.utc)
        exp = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            start_time=now,
            end_time=now + timedelta(days=7),
        )
        assert exp.start_time == now
        assert exp.end_time == now + timedelta(days=7)


class TestGuardrailConfig:
    """Tests for experiment guardrails."""

    def test_create_guardrail(self):
        """Can create a guardrail config."""
        guardrail = GuardrailConfig(
            metric_name="acceptance_rate",
            min_threshold=0.5,
            max_threshold=None,
            sample_size=100,
        )
        assert guardrail.metric_name == "acceptance_rate"
        assert guardrail.min_threshold == 0.5

    def test_guardrail_with_max_threshold(self):
        """Guardrail can have max threshold."""
        guardrail = GuardrailConfig(
            metric_name="latency_p95",
            min_threshold=None,
            max_threshold=500.0,  # 500ms max
            sample_size=100,
        )
        assert guardrail.max_threshold == 500.0

    def test_guardrail_requires_at_least_one_threshold(self):
        """Guardrail must have at least min or max threshold."""
        with pytest.raises(ValueError, match="threshold"):
            GuardrailConfig(
                metric_name="acceptance_rate",
                min_threshold=None,
                max_threshold=None,
                sample_size=100,
            )


class TestVariantAssignment:
    """Tests for VariantAssignment result."""

    def test_create_assignment(self):
        """Can create an assignment."""
        assignment = VariantAssignment(
            experiment_id="exp_001",
            variant_id="treatment",
            user_id="u456",
            assigned_at=datetime.now(timezone.utc),
        )
        assert assignment.experiment_id == "exp_001"
        assert assignment.variant_id == "treatment"

    def test_assignment_includes_config(self):
        """Assignment includes variant config."""
        config = {"rrf_weights": {"vector": 0.50}}
        assignment = VariantAssignment(
            experiment_id="exp_001",
            variant_id="treatment",
            user_id="u456",
            variant_config=config,
        )
        assert assignment.variant_config == config


class TestABTestingFramework:
    """Tests for ABTestingFramework class."""

    @pytest.fixture
    def framework(self) -> ABTestingFramework:
        """Create an A/B testing framework."""
        return ABTestingFramework()

    @pytest.fixture
    def sample_experiment(self) -> Experiment:
        """Create a sample experiment."""
        return Experiment(
            experiment_id="exp_001",
            name="RRF Weight Test",
            org_id="org789",
            variants=[
                ExperimentVariant(
                    variant_id="control",
                    name="Control",
                    weight=0.5,
                    config={"rrf_weights": {"vector": 0.40, "lexical": 0.35, "graph": 0.25}},
                ),
                ExperimentVariant(
                    variant_id="treatment",
                    name="Treatment",
                    weight=0.5,
                    config={"rrf_weights": {"vector": 0.50, "lexical": 0.30, "graph": 0.20}},
                ),
            ],
            status=ExperimentStatus.RUNNING,
        )

    def test_register_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can register an experiment."""
        framework.register_experiment(sample_experiment)
        exp = framework.get_experiment(sample_experiment.experiment_id)
        assert exp is not None
        assert exp.name == "RRF Weight Test"

    def test_list_experiments(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can list all experiments for an org."""
        framework.register_experiment(sample_experiment)
        experiments = framework.list_experiments("org789")
        assert len(experiments) == 1
        assert experiments[0].experiment_id == "exp_001"

    def test_get_assignment_for_user(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can get variant assignment for a user."""
        framework.register_experiment(sample_experiment)
        assignment = framework.get_assignment(
            experiment_id="exp_001",
            user_id="u456",
            org_id="org789",
        )
        assert assignment is not None
        assert assignment.experiment_id == "exp_001"
        assert assignment.variant_id in ["control", "treatment"]

    def test_assignment_is_deterministic(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Same user always gets same variant (deterministic)."""
        framework.register_experiment(sample_experiment)

        assignments = [
            framework.get_assignment("exp_001", "u456", "org789")
            for _ in range(10)
        ]

        # All assignments should be the same variant
        first_variant = assignments[0].variant_id
        for assignment in assignments:
            assert assignment.variant_id == first_variant

    def test_different_users_get_distributed_variants(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Different users should be distributed across variants."""
        framework.register_experiment(sample_experiment)

        variant_counts = {"control": 0, "treatment": 0}
        for i in range(100):
            assignment = framework.get_assignment("exp_001", f"user_{i}", "org789")
            variant_counts[assignment.variant_id] += 1

        # With 50/50 split, both should have reasonable counts
        assert variant_counts["control"] > 20
        assert variant_counts["treatment"] > 20

    def test_no_assignment_for_paused_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Paused experiments don't assign variants."""
        sample_experiment = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.PAUSED,
        )
        framework.register_experiment(sample_experiment)

        assignment = framework.get_assignment("exp_001", "u456", "org789")
        assert assignment is None

    def test_no_assignment_for_completed_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Completed experiments don't assign variants."""
        sample_experiment = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.COMPLETED,
        )
        framework.register_experiment(sample_experiment)

        assignment = framework.get_assignment("exp_001", "u456", "org789")
        assert assignment is None

    def test_traffic_percentage_limits_assignment(
        self, framework: ABTestingFramework
    ):
        """Only specified percentage of traffic gets assigned."""
        exp = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.RUNNING,
            traffic_percentage=0.1,  # 10%
        )
        framework.register_experiment(exp)

        assigned_count = 0
        for i in range(100):
            assignment = framework.get_assignment("exp_001", f"user_{i}", "org789")
            if assignment is not None:
                assigned_count += 1

        # Roughly 10% should be assigned (allowing for variance)
        assert 5 <= assigned_count <= 20

    def test_start_experiment(self, framework: ABTestingFramework):
        """Can start a draft experiment."""
        exp = Experiment(
            experiment_id="exp_001",
            name="Test",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.DRAFT,
        )
        framework.register_experiment(exp)
        framework.start_experiment("exp_001")

        exp = framework.get_experiment("exp_001")
        assert exp.status == ExperimentStatus.RUNNING

    def test_pause_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can pause a running experiment."""
        framework.register_experiment(sample_experiment)
        framework.pause_experiment("exp_001")

        exp = framework.get_experiment("exp_001")
        assert exp.status == ExperimentStatus.PAUSED

    def test_complete_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can complete an experiment."""
        framework.register_experiment(sample_experiment)
        framework.complete_experiment("exp_001")

        exp = framework.get_experiment("exp_001")
        assert exp.status == ExperimentStatus.COMPLETED

    def test_rollback_experiment(
        self, framework: ABTestingFramework, sample_experiment: Experiment
    ):
        """Can rollback an experiment (guardrail triggered)."""
        framework.register_experiment(sample_experiment)
        framework.rollback_experiment("exp_001", reason="Acceptance rate below threshold")

        exp = framework.get_experiment("exp_001")
        assert exp.status == ExperimentStatus.ROLLED_BACK


class TestABTestingGuardrails:
    """Tests for experiment guardrails and auto-rollback."""

    @pytest.fixture
    def framework(self) -> ABTestingFramework:
        return ABTestingFramework()

    def test_check_guardrail_passes(self, framework: ABTestingFramework):
        """Guardrail check passes when metric is within bounds."""
        guardrail = GuardrailConfig(
            metric_name="acceptance_rate",
            min_threshold=0.5,
            sample_size=100,
        )

        result = framework.check_guardrail(
            guardrail=guardrail,
            metric_value=0.7,  # Above 0.5 threshold
            sample_size=150,
        )
        assert result.passed is True

    def test_check_guardrail_fails_below_min(self, framework: ABTestingFramework):
        """Guardrail check fails when metric below min threshold."""
        guardrail = GuardrailConfig(
            metric_name="acceptance_rate",
            min_threshold=0.5,
            sample_size=100,
        )

        result = framework.check_guardrail(
            guardrail=guardrail,
            metric_value=0.3,  # Below 0.5 threshold
            sample_size=150,
        )
        assert result.passed is False
        assert "below" in result.reason.lower()

    def test_check_guardrail_fails_above_max(self, framework: ABTestingFramework):
        """Guardrail check fails when metric above max threshold."""
        guardrail = GuardrailConfig(
            metric_name="latency_p95",
            max_threshold=500.0,
            sample_size=100,
        )

        result = framework.check_guardrail(
            guardrail=guardrail,
            metric_value=750.0,  # Above 500ms threshold
            sample_size=150,
        )
        assert result.passed is False
        assert "above" in result.reason.lower()

    def test_check_guardrail_insufficient_sample(self, framework: ABTestingFramework):
        """Guardrail check inconclusive with insufficient sample."""
        guardrail = GuardrailConfig(
            metric_name="acceptance_rate",
            min_threshold=0.5,
            sample_size=100,
        )

        result = framework.check_guardrail(
            guardrail=guardrail,
            metric_value=0.3,
            sample_size=50,  # Below required 100
        )
        assert result.passed is True  # Don't fail with insufficient data
        assert result.reason is not None


class TestABTestingGetAllActiveAssignments:
    """Tests for getting all active experiment assignments for a user."""

    @pytest.fixture
    def framework(self) -> ABTestingFramework:
        return ABTestingFramework()

    def test_get_all_assignments_for_user(self, framework: ABTestingFramework):
        """Can get all active assignments for a user."""
        # Register two experiments
        exp1 = Experiment(
            experiment_id="exp_001",
            name="Test 1",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.RUNNING,
        )
        exp2 = Experiment(
            experiment_id="exp_002",
            name="Test 2",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.RUNNING,
        )
        framework.register_experiment(exp1)
        framework.register_experiment(exp2)

        assignments = framework.get_all_assignments("u456", "org789")
        assert len(assignments) == 2

    def test_get_all_assignments_excludes_inactive(self, framework: ABTestingFramework):
        """Inactive experiments are excluded from assignments."""
        exp1 = Experiment(
            experiment_id="exp_001",
            name="Test 1",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.RUNNING,
        )
        exp2 = Experiment(
            experiment_id="exp_002",
            name="Test 2",
            org_id="org789",
            variants=[
                ExperimentVariant(variant_id="control", name="Control", weight=1.0),
            ],
            status=ExperimentStatus.PAUSED,
        )
        framework.register_experiment(exp1)
        framework.register_experiment(exp2)

        assignments = framework.get_all_assignments("u456", "org789")
        assert len(assignments) == 1
        assert assignments[0].experiment_id == "exp_001"
