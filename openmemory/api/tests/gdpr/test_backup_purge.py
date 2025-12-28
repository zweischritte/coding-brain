"""Tests for GDPR Backup Purge Strategy.

These tests verify the backup purge tracking functionality that records
user deletions for backup rotation compliance.

Test IDs: BKP-001 through BKP-006
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4


class TestBackupPurgeTracking:
    """Tests for backup purge tracking (BKP-001 through BKP-006)."""

    def test_deletion_creates_backup_purge_record(self):
        """BKP-001: Deletion creates backup purge record."""
        from app.gdpr.backup_purge import BackupPurgeTracker

        mock_db = MagicMock()
        tracker = BackupPurgeTracker(db=mock_db)

        user_id = "test-user-123"
        record = tracker.record_deletion(user_id)

        assert record.user_id == user_id
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_backup_purge_record_contains_timestamp(self):
        """BKP-002: Backup purge record contains timestamp."""
        from app.gdpr.backup_purge import BackupPurgeTracker

        mock_db = MagicMock()
        tracker = BackupPurgeTracker(db=mock_db)

        user_id = "test-user-123"
        record = tracker.record_deletion(user_id)

        assert record.deleted_at is not None
        assert isinstance(record.deleted_at, datetime)
        # Should be recent (within last minute)
        assert (datetime.now(timezone.utc) - record.deleted_at).total_seconds() < 60

    def test_backup_purge_record_contains_user_id(self):
        """BKP-003: Backup purge record contains user_id."""
        from app.gdpr.backup_purge import BackupPurgeTracker

        mock_db = MagicMock()
        tracker = BackupPurgeTracker(db=mock_db)

        user_id = "specific-user-456"
        record = tracker.record_deletion(user_id)

        assert record.user_id == user_id

    def test_retention_policy_configuration(self):
        """BKP-004: Retention policy configuration."""
        from app.gdpr.backup_purge import BackupPurgeTracker, DEFAULT_BACKUP_RETENTION_DAYS

        mock_db = MagicMock()

        # Default retention
        tracker_default = BackupPurgeTracker(db=mock_db)
        record_default = tracker_default.record_deletion("user-1")
        assert record_default.retention_days == DEFAULT_BACKUP_RETENTION_DAYS

        # Custom retention
        custom_days = 60
        tracker_custom = BackupPurgeTracker(db=mock_db, retention_days=custom_days)
        record_custom = tracker_custom.record_deletion("user-2")
        assert record_custom.retention_days == custom_days

    def test_backup_purge_record_has_purge_after_date(self):
        """BKP-004: Backup purge record has purge_after date."""
        from app.gdpr.backup_purge import BackupPurgeTracker

        mock_db = MagicMock()
        retention_days = 30
        tracker = BackupPurgeTracker(db=mock_db, retention_days=retention_days)

        record = tracker.record_deletion("test-user")

        assert record.purge_after is not None
        # purge_after should be deleted_at + retention_days
        expected_purge = record.deleted_at + timedelta(days=retention_days)
        # Allow 1 second tolerance
        assert abs((record.purge_after - expected_purge).total_seconds()) < 1

    def test_backup_purge_list_query_pending(self):
        """BKP-005: Backup purge list query for pending purges."""
        from app.gdpr.backup_purge import BackupPurgeTracker, BackupPurgeTrackingModel

        mock_db = MagicMock()

        # Mock pending records (purge_after in the future)
        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_record.user_id = "user-123"
        mock_record.deleted_at = datetime.now(timezone.utc)
        mock_record.retention_days = 30
        mock_record.purge_after = datetime.now(timezone.utc) + timedelta(days=15)

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_record]

        tracker = BackupPurgeTracker(db=mock_db)
        pending = tracker.get_pending_purges()

        assert len(pending) == 1
        assert pending[0].user_id == "user-123"
        mock_db.query.assert_called_with(BackupPurgeTrackingModel)

    def test_backup_purge_list_query_completed(self):
        """BKP-005: Backup purge list query for completed purges."""
        from app.gdpr.backup_purge import BackupPurgeTracker, BackupPurgeTrackingModel

        mock_db = MagicMock()

        # Mock completed records (purge_after in the past)
        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_record.user_id = "old-user"
        mock_record.deleted_at = datetime.now(timezone.utc) - timedelta(days=60)
        mock_record.retention_days = 30
        mock_record.purge_after = datetime.now(timezone.utc) - timedelta(days=30)

        mock_db.query.return_value.filter.return_value.all.return_value = [mock_record]

        tracker = BackupPurgeTracker(db=mock_db)
        completed = tracker.get_completed_purges()

        assert len(completed) == 1
        assert completed[0].user_id == "old-user"

    def test_retention_policy_default_value(self):
        """BKP-006: Default retention policy is 30 days."""
        from app.gdpr.backup_purge import DEFAULT_BACKUP_RETENTION_DAYS

        assert DEFAULT_BACKUP_RETENTION_DAYS == 30


class TestBackupPurgeRecord:
    """Tests for BackupPurgeRecord dataclass."""

    def test_backup_purge_record_fields(self):
        """Verify BackupPurgeRecord has all required fields."""
        from app.gdpr.backup_purge import BackupPurgeRecord

        now = datetime.now(timezone.utc)
        record = BackupPurgeRecord(
            id="test-id",
            user_id="user-123",
            deleted_at=now,
            retention_days=30,
            purge_after=now + timedelta(days=30),
        )

        assert record.id == "test-id"
        assert record.user_id == "user-123"
        assert record.deleted_at == now
        assert record.retention_days == 30
        assert record.purge_after == now + timedelta(days=30)


class TestBackupPurgeModel:
    """Tests for BackupPurgeTrackingModel SQLAlchemy model."""

    def test_model_table_name(self):
        """Verify model has correct table name."""
        from app.gdpr.backup_purge import BackupPurgeTrackingModel

        assert BackupPurgeTrackingModel.__tablename__ == "gdpr_backup_purge_tracking"

    def test_model_has_required_columns(self):
        """Verify model has all required columns."""
        from app.gdpr.backup_purge import BackupPurgeTrackingModel

        columns = [c.name for c in BackupPurgeTrackingModel.__table__.columns]

        assert "id" in columns
        assert "user_id" in columns
        assert "deleted_at" in columns
        assert "retention_days" in columns
        assert "purge_after" in columns
