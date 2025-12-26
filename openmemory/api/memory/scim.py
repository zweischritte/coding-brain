"""SCIM orphan data handling for Phase 3 (FR-012).

Per v9 plan section 5.6:
- Deprovisioned users suspended within 4 hours
- 3-day grace period before ownership changes
- Personal memories deleted after 30-day grace period
- Team/org memory transfer to admin owner
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class SCIMUserState(str, Enum):
    """SCIM user provisioning states."""

    ACTIVE = "active"
    SUSPENDED = "suspended"  # Within 4h of deprovisioning
    GRACE_PERIOD = "grace_period"  # 3-day grace before ownership change
    PENDING_DELETION = "pending_deletion"  # 30-day grace for personal data
    DELETED = "deleted"


@dataclass
class OrphanDataPolicy:
    """Policy for handling orphaned data."""

    suspension_timeout: timedelta = field(
        default_factory=lambda: timedelta(hours=4)
    )
    ownership_grace_period: timedelta = field(
        default_factory=lambda: timedelta(days=3)
    )
    personal_data_retention: timedelta = field(
        default_factory=lambda: timedelta(days=30)
    )
    default_owner_id: str | None = None  # Admin to transfer ownership to


@dataclass
class SuspensionRecord:
    """Record of a user suspension event."""

    user_id: str
    org_id: str
    state: SCIMUserState
    suspended_at: datetime
    grace_period_ends_at: datetime
    deletion_scheduled_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeletionSchedule:
    """Schedule for data deletion."""

    user_id: str
    org_id: str
    scheduled_at: datetime
    data_types: list[str] = field(default_factory=list)  # e.g., ["memories", "embeddings"]
    executed: bool = False
    executed_at: datetime | None = None


@dataclass
class OwnershipTransfer:
    """Record of ownership transfer."""

    from_user_id: str
    to_user_id: str
    org_id: str
    memory_ids: list[str] = field(default_factory=list)
    transferred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = "scim_deprovisioning"


class OrphanDataHandler:
    """Handles orphaned data from SCIM deprovisioning.

    Workflow:
    1. User deprovisioned via SCIM
    2. Suspend within 4 hours
    3. 3-day grace period for ownership transfer
    4. Transfer team/org memories to admin
    5. 30-day grace for personal memory deletion
    """

    def __init__(self, policy: OrphanDataPolicy | None = None):
        self.policy = policy or OrphanDataPolicy()
        self._suspensions: dict[str, SuspensionRecord] = {}
        self._schedules: dict[str, DeletionSchedule] = {}
        self._transfers: list[OwnershipTransfer] = []

    def handle_deprovisioning(
        self,
        user_id: str,
        org_id: str,
    ) -> SuspensionRecord:
        """Handle a user deprovisioning event.

        Args:
            user_id: The deprovisioned user
            org_id: The user's organization

        Returns:
            SuspensionRecord with calculated deadlines
        """
        now = datetime.now(timezone.utc)

        record = SuspensionRecord(
            user_id=user_id,
            org_id=org_id,
            state=SCIMUserState.SUSPENDED,
            suspended_at=now,
            grace_period_ends_at=now + self.policy.ownership_grace_period,
            deletion_scheduled_at=now + self.policy.personal_data_retention,
        )

        self._suspensions[user_id] = record

        # Schedule deletion
        self._schedules[user_id] = DeletionSchedule(
            user_id=user_id,
            org_id=org_id,
            scheduled_at=record.deletion_scheduled_at,
            data_types=["personal_memories", "embeddings_metadata", "audit_summary"],
        )

        return record

    def get_suspension(self, user_id: str) -> SuspensionRecord | None:
        """Get suspension record for a user."""
        return self._suspensions.get(user_id)

    def update_state(self, user_id: str) -> SCIMUserState | None:
        """Update user state based on current time.

        Returns the new state, or None if user not found.
        """
        record = self._suspensions.get(user_id)
        if record is None:
            return None

        now = datetime.now(timezone.utc)

        if record.state == SCIMUserState.DELETED:
            return SCIMUserState.DELETED

        if record.deletion_scheduled_at and now >= record.deletion_scheduled_at:
            record.state = SCIMUserState.PENDING_DELETION
        elif now >= record.grace_period_ends_at:
            record.state = SCIMUserState.GRACE_PERIOD
        else:
            record.state = SCIMUserState.SUSPENDED

        return record.state

    def transfer_ownership(
        self,
        from_user_id: str,
        to_user_id: str,
        org_id: str,
        memory_ids: list[str],
    ) -> OwnershipTransfer:
        """Transfer ownership of team/org memories.

        Args:
            from_user_id: Deprovisioned user
            to_user_id: New owner (admin)
            org_id: Organization
            memory_ids: Memory IDs to transfer

        Returns:
            OwnershipTransfer record
        """
        transfer = OwnershipTransfer(
            from_user_id=from_user_id,
            to_user_id=to_user_id,
            org_id=org_id,
            memory_ids=memory_ids,
        )
        self._transfers.append(transfer)
        return transfer

    def get_pending_deletions(self) -> list[DeletionSchedule]:
        """Get all pending deletion schedules."""
        now = datetime.now(timezone.utc)
        return [
            s for s in self._schedules.values()
            if not s.executed and s.scheduled_at <= now
        ]

    def execute_deletion(self, user_id: str) -> bool:
        """Mark a deletion as executed.

        Returns True if schedule existed and was updated.
        """
        schedule = self._schedules.get(user_id)
        if schedule is None:
            return False

        schedule.executed = True
        schedule.executed_at = datetime.now(timezone.utc)

        if user_id in self._suspensions:
            self._suspensions[user_id].state = SCIMUserState.DELETED

        return True

    def cancel_deprovisioning(self, user_id: str) -> bool:
        """Cancel deprovisioning (e.g., user re-provisioned).

        Returns True if user was in suspended state.
        """
        if user_id not in self._suspensions:
            return False

        del self._suspensions[user_id]
        self._schedules.pop(user_id, None)
        return True

    def get_transfers_for_org(self, org_id: str) -> list[OwnershipTransfer]:
        """Get all ownership transfers for an org."""
        return [t for t in self._transfers if t.org_id == org_id]
