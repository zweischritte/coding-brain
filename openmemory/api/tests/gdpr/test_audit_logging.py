"""Tests for GDPR Audit Logging.

These tests verify the GDPR audit logging functionality that tracks
all GDPR operations (SAR export, deletion) with full audit trail.

Test IDs: AUD-001 through AUD-006
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4


class TestGDPRAuditLogModel:
    """Tests for GDPRAuditLogModel (AUD-001)."""

    def test_gdpr_audit_log_model_has_required_fields(self):
        """AUD-001: GDPRAuditLog model has required fields."""
        from app.gdpr.audit import GDPRAuditLogModel

        columns = [c.name for c in GDPRAuditLogModel.__table__.columns]

        # Required fields
        assert "id" in columns
        assert "audit_id" in columns
        assert "operation" in columns
        assert "target_user_id" in columns
        assert "started_at" in columns

        # Optional but important fields
        assert "requestor_id" in columns
        assert "reason" in columns
        assert "status" in columns
        assert "completed_at" in columns
        assert "details" in columns


class TestAuditLogging:
    """Tests for audit logging operations (AUD-002 through AUD-006)."""

    def test_sar_creates_audit_log_before_operation(self):
        """AUD-002: SAR creates audit log before operation."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        audit_id = str(uuid4())
        target_user_id = "user-123"
        requestor_id = "admin-456"

        entry = logger.log_operation_start(
            audit_id=audit_id,
            operation=GDPROperation.EXPORT,
            target_user_id=target_user_id,
            requestor_id=requestor_id,
            reason="User requested data export",
        )

        # Should create log entry
        assert entry.audit_id == audit_id
        assert entry.operation == GDPROperation.EXPORT
        assert entry.target_user_id == target_user_id
        assert entry.status == "started"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_delete_creates_audit_log_before_operation(self):
        """AUD-003: Delete creates audit log before operation."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        audit_id = str(uuid4())
        target_user_id = "user-123"

        entry = logger.log_operation_start(
            audit_id=audit_id,
            operation=GDPROperation.DELETE,
            target_user_id=target_user_id,
            reason="Right to erasure request",
        )

        assert entry.audit_id == audit_id
        assert entry.operation == GDPROperation.DELETE
        assert entry.status == "started"

    def test_audit_logs_survive_user_deletion(self):
        """AUD-004: Audit logs survive user deletion.

        Audit logs are stored in a separate table (gdpr_audit_logs)
        that does not have FK constraints to the users table.
        """
        from app.gdpr.audit import GDPRAuditLogModel

        # The model should NOT have a foreign key to users table
        # It stores user_id as a string, not as a FK
        columns = {c.name: c for c in GDPRAuditLogModel.__table__.columns}

        target_user_id_col = columns["target_user_id"]
        # Should be a string column, not a FK
        assert str(target_user_id_col.type) in ("VARCHAR", "String", "String()")

        # Should not have foreign keys
        assert len(target_user_id_col.foreign_keys) == 0

    def test_audit_logs_include_requestor_identity(self):
        """AUD-005: Audit logs include requestor identity."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        requestor_id = "admin-789"
        entry = logger.log_operation_start(
            audit_id=str(uuid4()),
            operation=GDPROperation.EXPORT,
            target_user_id="user-123",
            requestor_id=requestor_id,
        )

        assert entry.requestor_id == requestor_id

    def test_audit_logs_include_operation_result(self):
        """AUD-006: Audit logs include operation result."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation, GDPRAuditLogModel

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        audit_id = str(uuid4())

        # Mock the existing record for update
        mock_record = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_record

        # Complete the operation
        details = {
            "stores_processed": ["postgres", "neo4j", "qdrant"],
            "records_deleted": 42,
        }
        logger.log_operation_complete(
            audit_id=audit_id,
            status="completed",
            details=details,
        )

        # Should update the record
        assert mock_record.status == "completed"
        assert mock_record.details == details
        mock_db.commit.assert_called()


class TestGDPRAuditLogger:
    """Additional tests for GDPRAuditLogger."""

    def test_get_audit_log_by_id(self):
        """Test retrieving an audit log entry by ID."""
        from app.gdpr.audit import GDPRAuditLogger, GDPROperation, GDPRAuditLogModel

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        audit_id = str(uuid4())

        # Mock the record
        mock_record = MagicMock()
        mock_record.audit_id = audit_id
        mock_record.operation = "export"
        mock_record.target_user_id = "user-123"
        mock_record.requestor_id = "admin-456"
        mock_record.reason = "Test reason"
        mock_record.status = "completed"
        mock_record.started_at = datetime.now(timezone.utc)
        mock_record.completed_at = datetime.now(timezone.utc)
        mock_record.details = {"count": 10}

        mock_db.query.return_value.filter.return_value.first.return_value = mock_record

        entry = logger.get_audit_log(audit_id)

        assert entry is not None
        assert entry.audit_id == audit_id
        assert entry.status == "completed"

    def test_get_audit_log_returns_none_for_missing(self):
        """Test that get_audit_log returns None for missing entries."""
        from app.gdpr.audit import GDPRAuditLogger

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        logger = GDPRAuditLogger(db=mock_db)
        entry = logger.get_audit_log("nonexistent-id")

        assert entry is None

    def test_list_audit_logs_for_user(self):
        """Test listing all audit logs for a target user."""
        from app.gdpr.audit import GDPRAuditLogger, GDPRAuditLogModel

        mock_db = MagicMock()
        logger = GDPRAuditLogger(db=mock_db)

        target_user_id = "user-123"

        # Mock multiple records
        mock_records = [
            MagicMock(
                audit_id=str(uuid4()),
                operation="export",
                target_user_id=target_user_id,
                requestor_id="admin-1",
                reason="First export",
                status="completed",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                details=None,
            ),
            MagicMock(
                audit_id=str(uuid4()),
                operation="delete",
                target_user_id=target_user_id,
                requestor_id="admin-2",
                reason="Deletion",
                status="completed",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                details={"stores": 5},
            ),
        ]

        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_records

        entries = logger.list_audit_logs_for_user(target_user_id)

        assert len(entries) == 2
        mock_db.query.assert_called_with(GDPRAuditLogModel)


class TestGDPROperationEnum:
    """Tests for GDPROperation enum."""

    def test_operation_types_defined(self):
        """Verify all operation types are defined."""
        from app.gdpr.audit import GDPROperation

        assert hasattr(GDPROperation, "EXPORT")
        assert hasattr(GDPROperation, "DELETE")
        assert hasattr(GDPROperation, "ACCESS")

    def test_operation_values(self):
        """Verify operation enum values."""
        from app.gdpr.audit import GDPROperation

        assert GDPROperation.EXPORT.value == "export"
        assert GDPROperation.DELETE.value == "delete"
        assert GDPROperation.ACCESS.value == "access"
