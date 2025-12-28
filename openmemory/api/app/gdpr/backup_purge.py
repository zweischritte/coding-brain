"""Backup Purge Strategy for GDPR Compliance.

This module implements the backup purge tracking strategy for GDPR compliance.
When a user is deleted, we record the deletion timestamp so that backup
rotation procedures can ensure old backups containing the user's data
are properly aged out.

Strategy: Retention Tracking
----------------------------
For MVP, we use retention tracking instead of crypto-shredding:
1. When a user is deleted, we record the deletion timestamp
2. Backups older than the retention period are considered "purged"
3. Documentation guides operators on backup rotation

Future Enhancement: Crypto-Shredding
------------------------------------
For true cryptographic erasure, implement per-user encryption keys:
1. Each user's data is encrypted with a unique key before backup
2. On deletion, the key is destroyed
3. Backups become unreadable without the key

See: docs/RUNBOOK-BACKUP-RESTORE.md for operational procedures
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from app.database import Base

# Default retention: backups older than 30 days are rotated out
DEFAULT_BACKUP_RETENTION_DAYS = 30


@dataclass
class BackupPurgeRecord:
    """Record of a user deletion for backup purge tracking.

    Attributes:
        id: Unique identifier for this record
        user_id: The external user ID that was deleted
        deleted_at: When the user was deleted
        retention_days: How long until backups should be considered purged
        purge_after: Date after which backups should not contain this user's data
    """
    id: str
    user_id: str
    deleted_at: datetime
    retention_days: int
    purge_after: datetime


class BackupPurgeTrackingModel(Base):
    """SQLAlchemy model for backup purge tracking.

    This table records all user deletions so that backup rotation
    procedures can verify compliance with the retention policy.
    """

    __tablename__ = "gdpr_backup_purge_tracking"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True, unique=True)
    deleted_at = Column(DateTime(timezone=True), nullable=False)
    retention_days = Column(Integer, nullable=False, default=DEFAULT_BACKUP_RETENTION_DAYS)
    purge_after = Column(DateTime(timezone=True), nullable=False)


class BackupPurgeTracker:
    """Track user deletions for backup purge compliance.

    This class manages the recording of user deletions and provides
    queries to identify which deletions are pending backup rotation
    and which have been fully purged.

    Attributes:
        DEFAULT_RETENTION_DAYS: Default backup retention period
    """

    def __init__(
        self,
        db: Session,
        retention_days: int = DEFAULT_BACKUP_RETENTION_DAYS,
    ):
        """Initialize the backup purge tracker.

        Args:
            db: SQLAlchemy database session
            retention_days: How long to wait before considering backups purged
        """
        self._db = db
        self._retention_days = retention_days

    def record_deletion(self, user_id: str) -> BackupPurgeRecord:
        """Record a user deletion for backup purge tracking.

        Args:
            user_id: The external user ID that was deleted

        Returns:
            BackupPurgeRecord with deletion details
        """
        deleted_at = datetime.now(timezone.utc)
        purge_after = deleted_at + timedelta(days=self._retention_days)

        record = BackupPurgeTrackingModel(
            user_id=user_id,
            deleted_at=deleted_at,
            retention_days=self._retention_days,
            purge_after=purge_after,
        )
        self._db.add(record)
        self._db.commit()

        return BackupPurgeRecord(
            id=str(record.id),
            user_id=user_id,
            deleted_at=deleted_at,
            retention_days=self._retention_days,
            purge_after=purge_after,
        )

    def get_pending_purges(self) -> List[BackupPurgeRecord]:
        """Get deletions still within retention period.

        These are user deletions where backups may still contain
        the user's data and should not be restored without filtering.

        Returns:
            List of BackupPurgeRecord for users within retention period
        """
        now = datetime.now(timezone.utc)
        records = self._db.query(BackupPurgeTrackingModel).filter(
            BackupPurgeTrackingModel.purge_after > now
        ).all()

        return [
            BackupPurgeRecord(
                id=str(r.id),
                user_id=r.user_id,
                deleted_at=r.deleted_at,
                retention_days=r.retention_days,
                purge_after=r.purge_after,
            )
            for r in records
        ]

    def get_completed_purges(self) -> List[BackupPurgeRecord]:
        """Get deletions past retention period.

        These are user deletions where backups should have been
        rotated out and no longer contain the user's data.

        Returns:
            List of BackupPurgeRecord for users past retention period
        """
        now = datetime.now(timezone.utc)
        records = self._db.query(BackupPurgeTrackingModel).filter(
            BackupPurgeTrackingModel.purge_after <= now
        ).all()

        return [
            BackupPurgeRecord(
                id=str(r.id),
                user_id=r.user_id,
                deleted_at=r.deleted_at,
                retention_days=r.retention_days,
                purge_after=r.purge_after,
            )
            for r in records
        ]

    def is_user_pending_purge(self, user_id: str) -> bool:
        """Check if a user's data may still be in backups.

        Args:
            user_id: The external user ID to check

        Returns:
            True if the user was deleted but may still be in backups
        """
        now = datetime.now(timezone.utc)
        record = self._db.query(BackupPurgeTrackingModel).filter(
            BackupPurgeTrackingModel.user_id == user_id,
            BackupPurgeTrackingModel.purge_after > now,
        ).first()
        return record is not None

    def cleanup_old_records(self, keep_days: int = 365) -> int:
        """Remove old purge tracking records.

        After backups have been fully rotated, we can clean up
        the tracking records to save space.

        Args:
            keep_days: How long to keep records after purge_after

        Returns:
            Number of records deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        deleted = self._db.query(BackupPurgeTrackingModel).filter(
            BackupPurgeTrackingModel.purge_after < cutoff
        ).delete()
        self._db.commit()
        return deleted
