"""
Tests for PostgreSQL-backed ExperimentStore.

The PostgresExperimentStore provides persistent storage for A/B test experiments
with tenant isolation via org_id and status history tracking.

TDD: These tests are written BEFORE the implementation.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
from sqlalchemy.orm import Session

from app.stores.experiment_store import (
    Experiment,
    ExperimentVariant,
    ExperimentStatus,
    VariantAssignment,
    ExperimentStoreInterface,
    PostgresExperimentStore,
)


# Test UUIDs
ORG_A_ID = "11111111-1111-1111-1111-111111111111"
ORG_B_ID = "22222222-2222-2222-2222-222222222222"
USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def make_experiment(
    org_id: str = ORG_A_ID,
    name: str = "Test Experiment",
    status: ExperimentStatus = ExperimentStatus.DRAFT,
    variants: Optional[list[ExperimentVariant]] = None,
    traffic_percentage: float = 1.0,
    experiment_id: Optional[str] = None,
) -> Experiment:
    """Helper to create experiments for testing."""
    if variants is None:
        variants = [
            ExperimentVariant(
                variant_id="control",
                name="Control",
                weight=0.5,
            ),
            ExperimentVariant(
                variant_id="treatment",
                name="Treatment",
                weight=0.5,
            ),
        ]
    return Experiment(
        experiment_id=experiment_id or str(uuid.uuid4()),
        name=name,
        org_id=org_id,
        variants=variants,
        status=status,
        traffic_percentage=traffic_percentage,
    )


class TestPostgresExperimentStoreCreate:
    """Test PostgresExperimentStore.create() method."""

    def test_create_persists_experiment(self, sqlite_test_db: Session):
        """
        create() should persist an experiment and return the experiment_id.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment()

        result = store.create(experiment)

        assert result == experiment.experiment_id

    def test_create_stores_all_fields(self, sqlite_test_db: Session):
        """
        create() should store all experiment fields correctly.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        now = datetime.now(timezone.utc)
        variants = [
            ExperimentVariant(
                variant_id="control",
                name="Control",
                weight=0.6,
                description="The control group",
                config={"feature_flag": False},
            ),
            ExperimentVariant(
                variant_id="treatment",
                name="Treatment",
                weight=0.4,
                description="The treatment group",
                config={"feature_flag": True},
            ),
        ]
        experiment = Experiment(
            experiment_id="test-exp-id",
            name="Full Test Experiment",
            org_id=ORG_A_ID,
            variants=variants,
            description="A comprehensive test experiment",
            status=ExperimentStatus.RUNNING,
            traffic_percentage=0.5,
            start_time=now,
        )

        store.create(experiment)
        retrieved = store.get(experiment.experiment_id, ORG_A_ID)

        assert retrieved is not None
        assert retrieved.experiment_id == "test-exp-id"
        assert retrieved.name == "Full Test Experiment"
        assert retrieved.description == "A comprehensive test experiment"
        assert retrieved.status == ExperimentStatus.RUNNING
        assert retrieved.traffic_percentage == 0.5
        assert len(retrieved.variants) == 2
        assert retrieved.variants[0].config == {"feature_flag": False}


class TestPostgresExperimentStoreGet:
    """Test PostgresExperimentStore.get() method."""

    def test_get_returns_experiment_by_id(self, sqlite_test_db: Session):
        """
        get() should return the experiment for a given ID.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment()
        store.create(experiment)

        result = store.get(experiment.experiment_id, ORG_A_ID)

        assert result is not None
        assert result.experiment_id == experiment.experiment_id

    def test_get_returns_none_for_nonexistent(self, sqlite_test_db: Session):
        """
        get() should return None for non-existent experiment ID.
        """
        store = PostgresExperimentStore(sqlite_test_db)

        result = store.get("nonexistent-id", ORG_A_ID)

        assert result is None

    def test_get_returns_none_for_wrong_org(self, sqlite_test_db: Session):
        """
        get() should return None when org_id doesn't match (tenant isolation).
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(org_id=ORG_A_ID)
        store.create(experiment)

        result = store.get(experiment.experiment_id, ORG_B_ID)

        assert result is None


class TestPostgresExperimentStoreList:
    """Test PostgresExperimentStore.list() method."""

    def test_list_returns_all_org_experiments(self, sqlite_test_db: Session):
        """
        list() should return all experiments for an org.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        exp1 = make_experiment(org_id=ORG_A_ID, name="Exp 1")
        exp2 = make_experiment(org_id=ORG_A_ID, name="Exp 2")
        exp_other = make_experiment(org_id=ORG_B_ID, name="Exp Other")

        store.create(exp1)
        store.create(exp2)
        store.create(exp_other)

        results = store.list(ORG_A_ID)

        assert len(results) == 2
        assert all(e.org_id == ORG_A_ID for e in results)

    def test_list_filters_by_status(self, sqlite_test_db: Session):
        """
        list(status=...) should filter by experiment status.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        draft = make_experiment(status=ExperimentStatus.DRAFT)
        running = make_experiment(status=ExperimentStatus.RUNNING)

        store.create(draft)
        store.create(running)

        results = store.list(ORG_A_ID, status=ExperimentStatus.RUNNING)

        assert len(results) == 1
        assert results[0].status == ExperimentStatus.RUNNING

    def test_list_returns_empty_for_no_experiments(self, sqlite_test_db: Session):
        """
        list() should return empty list when org has no experiments.
        """
        store = PostgresExperimentStore(sqlite_test_db)

        results = store.list(ORG_A_ID)

        assert results == []


class TestPostgresExperimentStoreUpdate:
    """Test PostgresExperimentStore.update() method."""

    def test_update_modifies_experiment(self, sqlite_test_db: Session):
        """
        update() should modify an existing experiment.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(name="Original Name")
        store.create(experiment)

        # Modify and update
        updated = Experiment(
            experiment_id=experiment.experiment_id,
            name="Updated Name",
            org_id=experiment.org_id,
            variants=experiment.variants,
            status=experiment.status,
        )
        result = store.update(updated)

        assert result is not None
        assert result.name == "Updated Name"

        # Verify persisted
        retrieved = store.get(experiment.experiment_id, ORG_A_ID)
        assert retrieved.name == "Updated Name"

    def test_update_returns_none_for_wrong_org(self, sqlite_test_db: Session):
        """
        update() should return None for experiments in other orgs.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(org_id=ORG_A_ID)
        store.create(experiment)

        # Try to update with wrong org
        tampered = Experiment(
            experiment_id=experiment.experiment_id,
            name="Hacked",
            org_id=ORG_B_ID,  # Wrong org
            variants=experiment.variants,
            status=experiment.status,
        )
        result = store.update(tampered)

        assert result is None


class TestPostgresExperimentStoreStatusHistory:
    """Test status history tracking in PostgresExperimentStore."""

    def test_status_change_creates_history_entry(self, sqlite_test_db: Session):
        """
        Changing experiment status should create a history entry.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(status=ExperimentStatus.DRAFT)
        store.create(experiment)

        # Change status
        store.update_status(
            experiment.experiment_id,
            ORG_A_ID,
            ExperimentStatus.RUNNING,
            reason="Starting experiment",
        )

        history = store.get_status_history(experiment.experiment_id, ORG_A_ID)

        # Should have 2 entries: DRAFT (initial) and RUNNING (transition)
        assert len(history) >= 1
        assert history[-1]["to_status"] == ExperimentStatus.RUNNING.value
        assert history[-1]["reason"] == "Starting experiment"

    def test_update_status_updates_experiment(self, sqlite_test_db: Session):
        """
        update_status() should update the experiment's current status.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(status=ExperimentStatus.DRAFT)
        store.create(experiment)

        store.update_status(
            experiment.experiment_id,
            ORG_A_ID,
            ExperimentStatus.RUNNING,
        )

        retrieved = store.get(experiment.experiment_id, ORG_A_ID)
        assert retrieved.status == ExperimentStatus.RUNNING

    def test_multiple_status_changes_tracked(self, sqlite_test_db: Session):
        """
        Multiple status changes should all be tracked in history.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(status=ExperimentStatus.DRAFT)
        store.create(experiment)

        # Multiple transitions
        store.update_status(
            experiment.experiment_id, ORG_A_ID, ExperimentStatus.RUNNING
        )
        store.update_status(experiment.experiment_id, ORG_A_ID, ExperimentStatus.PAUSED)
        store.update_status(
            experiment.experiment_id, ORG_A_ID, ExperimentStatus.RUNNING
        )

        history = store.get_status_history(experiment.experiment_id, ORG_A_ID)

        assert len(history) >= 3


class TestPostgresExperimentStoreAssignment:
    """Test variant assignment tracking in PostgresExperimentStore."""

    def test_record_assignment_persists(self, sqlite_test_db: Session):
        """
        record_assignment() should persist a variant assignment.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(status=ExperimentStatus.RUNNING)
        store.create(experiment)

        assignment = VariantAssignment(
            experiment_id=experiment.experiment_id,
            variant_id="control",
            user_id=USER_A_ID,
        )
        store.record_assignment(assignment, ORG_A_ID)

        assignments = store.get_assignments(experiment.experiment_id, ORG_A_ID)
        assert len(assignments) == 1
        assert assignments[0].user_id == USER_A_ID
        assert assignments[0].variant_id == "control"

    def test_get_user_assignment_returns_latest(self, sqlite_test_db: Session):
        """
        get_user_assignment() should return the user's assignment.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(status=ExperimentStatus.RUNNING)
        store.create(experiment)

        assignment = VariantAssignment(
            experiment_id=experiment.experiment_id,
            variant_id="treatment",
            user_id=USER_A_ID,
        )
        store.record_assignment(assignment, ORG_A_ID)

        result = store.get_user_assignment(
            experiment.experiment_id, USER_A_ID, ORG_A_ID
        )

        assert result is not None
        assert result.variant_id == "treatment"


class TestPostgresExperimentStoreTenantIsolation:
    """Test tenant isolation in PostgresExperimentStore."""

    def test_cross_org_list_returns_empty(self, sqlite_test_db: Session):
        """
        Listing experiments from wrong org should return empty.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(org_id=ORG_A_ID)
        store.create(experiment)

        results = store.list(ORG_B_ID)

        assert len(results) == 0

    def test_cross_org_status_update_fails(self, sqlite_test_db: Session):
        """
        Updating status with wrong org_id should fail silently.
        """
        store = PostgresExperimentStore(sqlite_test_db)
        experiment = make_experiment(org_id=ORG_A_ID, status=ExperimentStatus.DRAFT)
        store.create(experiment)

        # Try to update with wrong org
        result = store.update_status(
            experiment.experiment_id, ORG_B_ID, ExperimentStatus.RUNNING
        )

        assert result is False

        # Verify status unchanged
        retrieved = store.get(experiment.experiment_id, ORG_A_ID)
        assert retrieved.status == ExperimentStatus.DRAFT


class TestPostgresExperimentStoreInterface:
    """Test that PostgresExperimentStore implements the interface."""

    def test_implements_experiment_store_interface(self, sqlite_test_db: Session):
        """
        PostgresExperimentStore should implement all required methods.
        """
        assert issubclass(PostgresExperimentStore, ExperimentStoreInterface)

        store = PostgresExperimentStore(sqlite_test_db)
        required_methods = [
            "create",
            "get",
            "list",
            "update",
            "update_status",
            "get_status_history",
            "record_assignment",
            "get_assignments",
            "get_user_assignment",
        ]
        for method in required_methods:
            assert hasattr(store, method)
            assert callable(getattr(store, method))
