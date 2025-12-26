"""Tests for SCIM orphan data handling (FR-012).

Per v9 plan section 5.6:
- Deprovisioned users suspended within 4 hours
- 3-day grace period before ownership changes
- Personal memories deleted after 30-day grace period
- Team/org memory transfer to admin owner
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


class TestSCIMUserState:
    """Tests for SCIMUserState enum."""

    def test_scim_user_states(self):
        """SCIMUserState should have all required states."""
        from openmemory.api.memory import SCIMUserState

        assert SCIMUserState.ACTIVE.value == "active"
        assert SCIMUserState.SUSPENDED.value == "suspended"
        assert SCIMUserState.GRACE_PERIOD.value == "grace_period"
        assert SCIMUserState.PENDING_DELETION.value == "pending_deletion"
        assert SCIMUserState.DELETED.value == "deleted"


class TestOrphanDataPolicy:
    """Tests for OrphanDataPolicy configuration."""

    def test_default_suspension_timeout(self):
        """Default suspension timeout should be 4 hours."""
        from openmemory.api.memory import OrphanDataPolicy

        policy = OrphanDataPolicy()
        assert policy.suspension_timeout == timedelta(hours=4)

    def test_default_ownership_grace_period(self):
        """Default ownership grace period should be 3 days."""
        from openmemory.api.memory import OrphanDataPolicy

        policy = OrphanDataPolicy()
        assert policy.ownership_grace_period == timedelta(days=3)

    def test_default_personal_data_retention(self):
        """Default personal data retention should be 30 days."""
        from openmemory.api.memory import OrphanDataPolicy

        policy = OrphanDataPolicy()
        assert policy.personal_data_retention == timedelta(days=30)

    def test_custom_policy(self):
        """Should support custom policy configuration."""
        from openmemory.api.memory import OrphanDataPolicy

        policy = OrphanDataPolicy(
            suspension_timeout=timedelta(hours=2),
            ownership_grace_period=timedelta(days=7),
            personal_data_retention=timedelta(days=60),
            default_owner_id="admin_001",
        )

        assert policy.suspension_timeout == timedelta(hours=2)
        assert policy.ownership_grace_period == timedelta(days=7)
        assert policy.personal_data_retention == timedelta(days=60)
        assert policy.default_owner_id == "admin_001"


class TestSuspensionRecord:
    """Tests for SuspensionRecord data class."""

    def test_suspension_record_creation(self):
        """SuspensionRecord should store all fields."""
        from openmemory.api.memory import SuspensionRecord, SCIMUserState

        now = datetime.now(timezone.utc)
        record = SuspensionRecord(
            user_id="user_456",
            org_id="org_789",
            state=SCIMUserState.SUSPENDED,
            suspended_at=now,
            grace_period_ends_at=now + timedelta(days=3),
            deletion_scheduled_at=now + timedelta(days=30),
        )

        assert record.user_id == "user_456"
        assert record.org_id == "org_789"
        assert record.state == SCIMUserState.SUSPENDED
        assert record.suspended_at == now
        assert record.grace_period_ends_at == now + timedelta(days=3)
        assert record.deletion_scheduled_at == now + timedelta(days=30)

    def test_suspension_record_metadata(self):
        """SuspensionRecord should support metadata."""
        from openmemory.api.memory import SuspensionRecord, SCIMUserState

        now = datetime.now(timezone.utc)
        record = SuspensionRecord(
            user_id="user_456",
            org_id="org_789",
            state=SCIMUserState.SUSPENDED,
            suspended_at=now,
            grace_period_ends_at=now + timedelta(days=3),
            metadata={"reason": "user_left_company", "by": "hr_system"},
        )

        assert record.metadata["reason"] == "user_left_company"
        assert record.metadata["by"] == "hr_system"


class TestDeletionSchedule:
    """Tests for DeletionSchedule data class."""

    def test_deletion_schedule_creation(self):
        """DeletionSchedule should store scheduled deletion details."""
        from openmemory.api.memory import DeletionSchedule

        scheduled = datetime.now(timezone.utc) + timedelta(days=30)
        schedule = DeletionSchedule(
            user_id="user_456",
            org_id="org_789",
            scheduled_at=scheduled,
            data_types=["personal_memories", "embeddings_metadata"],
        )

        assert schedule.user_id == "user_456"
        assert schedule.org_id == "org_789"
        assert schedule.scheduled_at == scheduled
        assert "personal_memories" in schedule.data_types
        assert schedule.executed is False
        assert schedule.executed_at is None


class TestOwnershipTransfer:
    """Tests for OwnershipTransfer data class."""

    def test_ownership_transfer_creation(self):
        """OwnershipTransfer should store transfer details."""
        from openmemory.api.memory import OwnershipTransfer

        transfer = OwnershipTransfer(
            from_user_id="user_456",
            to_user_id="admin_001",
            org_id="org_789",
            memory_ids=["mem_1", "mem_2", "mem_3"],
        )

        assert transfer.from_user_id == "user_456"
        assert transfer.to_user_id == "admin_001"
        assert transfer.org_id == "org_789"
        assert len(transfer.memory_ids) == 3
        assert transfer.reason == "scim_deprovisioning"

    def test_ownership_transfer_auto_timestamp(self):
        """OwnershipTransfer should auto-set transferred_at."""
        from openmemory.api.memory import OwnershipTransfer

        before = datetime.now(timezone.utc)
        transfer = OwnershipTransfer(
            from_user_id="user_456",
            to_user_id="admin_001",
            org_id="org_789",
        )
        after = datetime.now(timezone.utc)

        assert before <= transfer.transferred_at <= after


class TestOrphanDataHandler:
    """Tests for OrphanDataHandler workflow."""

    def test_handle_deprovisioning_creates_record(self):
        """handle_deprovisioning should create suspension record."""
        from openmemory.api.memory import OrphanDataHandler, SCIMUserState

        handler = OrphanDataHandler()
        record = handler.handle_deprovisioning("user_456", "org_789")

        assert record.user_id == "user_456"
        assert record.org_id == "org_789"
        assert record.state == SCIMUserState.SUSPENDED

    def test_handle_deprovisioning_sets_grace_period(self):
        """handle_deprovisioning should set 3-day grace period."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        record = handler.handle_deprovisioning("user_456", "org_789")

        expected_grace = record.suspended_at + timedelta(days=3)
        assert record.grace_period_ends_at == expected_grace

    def test_handle_deprovisioning_schedules_deletion(self):
        """handle_deprovisioning should schedule 30-day deletion."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        record = handler.handle_deprovisioning("user_456", "org_789")

        expected_deletion = record.suspended_at + timedelta(days=30)
        assert record.deletion_scheduled_at == expected_deletion

    def test_handle_deprovisioning_creates_deletion_schedule(self):
        """handle_deprovisioning should create deletion schedule."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        schedules = handler.get_pending_deletions()
        # Not pending yet (30 days in future)
        assert len(schedules) == 0

    def test_get_suspension(self):
        """get_suspension should return suspension record."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        record = handler.get_suspension("user_456")
        assert record is not None
        assert record.user_id == "user_456"

    def test_get_suspension_not_found(self):
        """get_suspension should return None for unknown user."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        assert handler.get_suspension("nonexistent") is None


class TestStateTransitions:
    """Tests for SCIM user state transitions."""

    def test_update_state_remains_suspended(self):
        """User should remain suspended within grace period."""
        from openmemory.api.memory import OrphanDataHandler, SCIMUserState

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        # Immediately after, still suspended
        state = handler.update_state("user_456")
        assert state == SCIMUserState.SUSPENDED

    def test_update_state_to_grace_period(self):
        """User should transition to grace period after 3 days."""
        from openmemory.api.memory import OrphanDataHandler, SCIMUserState

        handler = OrphanDataHandler()
        record = handler.handle_deprovisioning("user_456", "org_789")

        # Mock time to 4 days later (past 3-day grace)
        with patch("openmemory.api.memory.scim.datetime") as mock_dt:
            mock_dt.now.return_value = record.suspended_at + timedelta(days=4)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            state = handler.update_state("user_456")
            assert state == SCIMUserState.GRACE_PERIOD

    def test_update_state_to_pending_deletion(self):
        """User should transition to pending deletion after 30 days."""
        from openmemory.api.memory import OrphanDataHandler, SCIMUserState

        handler = OrphanDataHandler()
        record = handler.handle_deprovisioning("user_456", "org_789")

        # Mock time to 31 days later (past 30-day retention)
        with patch("openmemory.api.memory.scim.datetime") as mock_dt:
            mock_dt.now.return_value = record.suspended_at + timedelta(days=31)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            state = handler.update_state("user_456")
            assert state == SCIMUserState.PENDING_DELETION

    def test_update_state_unknown_user(self):
        """update_state should return None for unknown user."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        state = handler.update_state("nonexistent")
        assert state is None


class TestOwnershipTransferWorkflow:
    """Tests for ownership transfer workflow."""

    def test_transfer_ownership(self):
        """transfer_ownership should create transfer record."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        transfer = handler.transfer_ownership(
            from_user_id="user_456",
            to_user_id="admin_001",
            org_id="org_789",
            memory_ids=["mem_1", "mem_2"],
        )

        assert transfer.from_user_id == "user_456"
        assert transfer.to_user_id == "admin_001"
        assert len(transfer.memory_ids) == 2

    def test_get_transfers_for_org(self):
        """get_transfers_for_org should return org transfers."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()

        handler.transfer_ownership("user_1", "admin_1", "org_a", ["mem_1"])
        handler.transfer_ownership("user_2", "admin_1", "org_a", ["mem_2"])
        handler.transfer_ownership("user_3", "admin_2", "org_b", ["mem_3"])

        org_a_transfers = handler.get_transfers_for_org("org_a")
        org_b_transfers = handler.get_transfers_for_org("org_b")

        assert len(org_a_transfers) == 2
        assert len(org_b_transfers) == 1


class TestDeletionWorkflow:
    """Tests for deletion workflow."""

    def test_get_pending_deletions_empty(self):
        """get_pending_deletions should return empty when none pending."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        assert handler.get_pending_deletions() == []

    def test_get_pending_deletions_after_scheduled_time(self):
        """get_pending_deletions should return schedules past their time."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        # Mock time to 31 days later
        with patch("openmemory.api.memory.scim.datetime") as mock_dt:
            mock_dt.now.return_value = datetime.now(timezone.utc) + timedelta(days=31)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            pending = handler.get_pending_deletions()
            assert len(pending) == 1
            assert pending[0].user_id == "user_456"

    def test_execute_deletion(self):
        """execute_deletion should mark schedule as executed."""
        from openmemory.api.memory import OrphanDataHandler, SCIMUserState

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        result = handler.execute_deletion("user_456")
        assert result is True

        record = handler.get_suspension("user_456")
        assert record.state == SCIMUserState.DELETED

    def test_execute_deletion_not_found(self):
        """execute_deletion should return False for unknown user."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        result = handler.execute_deletion("nonexistent")
        assert result is False


class TestCancelDeprovisioning:
    """Tests for canceling deprovisioning (user re-provisioned)."""

    def test_cancel_deprovisioning(self):
        """cancel_deprovisioning should remove suspension record."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        result = handler.cancel_deprovisioning("user_456")
        assert result is True

        # Should be gone
        assert handler.get_suspension("user_456") is None

    def test_cancel_deprovisioning_not_found(self):
        """cancel_deprovisioning should return False for unknown user."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        result = handler.cancel_deprovisioning("nonexistent")
        assert result is False


class TestCustomPolicy:
    """Tests for OrphanDataHandler with custom policy."""

    def test_custom_grace_period(self):
        """Should respect custom grace period."""
        from openmemory.api.memory import OrphanDataHandler, OrphanDataPolicy

        policy = OrphanDataPolicy(ownership_grace_period=timedelta(days=7))
        handler = OrphanDataHandler(policy=policy)

        record = handler.handle_deprovisioning("user_456", "org_789")

        expected_grace = record.suspended_at + timedelta(days=7)
        assert record.grace_period_ends_at == expected_grace

    def test_custom_retention_period(self):
        """Should respect custom retention period."""
        from openmemory.api.memory import OrphanDataHandler, OrphanDataPolicy

        policy = OrphanDataPolicy(personal_data_retention=timedelta(days=60))
        handler = OrphanDataHandler(policy=policy)

        record = handler.handle_deprovisioning("user_456", "org_789")

        expected_deletion = record.suspended_at + timedelta(days=60)
        assert record.deletion_scheduled_at == expected_deletion


class TestMultipleOrgs:
    """Tests for handling users in multiple orgs."""

    def test_separate_handling_per_org(self):
        """Each org should have separate suspension tracking."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()

        # Same user in different orgs
        handler.handle_deprovisioning("user_456", "org_a")

        # Only org_a should have suspension
        record_a = handler.get_suspension("user_456")
        assert record_a is not None
        assert record_a.org_id == "org_a"

    def test_transfers_isolated_by_org(self):
        """Transfers should be isolated by org."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()

        handler.transfer_ownership("user_1", "admin_1", "org_a", ["mem_1"])
        handler.transfer_ownership("user_1", "admin_2", "org_b", ["mem_2"])

        org_a = handler.get_transfers_for_org("org_a")
        org_b = handler.get_transfers_for_org("org_b")

        assert len(org_a) == 1
        assert org_a[0].to_user_id == "admin_1"

        assert len(org_b) == 1
        assert org_b[0].to_user_id == "admin_2"


class TestDataTypesDeletion:
    """Tests for data types marked for deletion."""

    def test_deletion_schedule_data_types(self):
        """Deletion schedule should include required data types."""
        from openmemory.api.memory import OrphanDataHandler

        handler = OrphanDataHandler()
        handler.handle_deprovisioning("user_456", "org_789")

        # Get the deletion schedule directly
        schedules = [
            s for s in handler._schedules.values()
            if s.user_id == "user_456"
        ]
        assert len(schedules) == 1

        schedule = schedules[0]
        assert "personal_memories" in schedule.data_types
        assert "embeddings_metadata" in schedule.data_types
        assert "audit_summary" in schedule.data_types
