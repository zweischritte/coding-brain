"""GDPR Audit Logging for Compliance.

This module implements comprehensive audit logging for all GDPR operations.
Audit logs are stored in a separate table that survives user deletion,
ensuring we maintain a complete compliance trail.

Audit logs capture:
- Who requested the operation (requestor_id)
- What operation was performed (export, delete, access)
- When the operation started and completed
- The target user's ID
- The result and any details

These logs are essential for:
- Demonstrating GDPR compliance to regulators
- Internal auditing and security reviews
- Debugging and troubleshooting
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, String, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from app.database import Base


class GDPROperation(Enum):
    """Types of GDPR operations that are logged."""
    EXPORT = "export"   # Subject Access Request (SAR)
    DELETE = "delete"   # Right to Erasure
    ACCESS = "access"   # Data access audit


class GDPRAuditLogModel(Base):
    """SQLAlchemy model for GDPR audit logs.

    This table stores an immutable audit trail of all GDPR operations.
    It intentionally does NOT have foreign keys to the users table
    so that audit logs survive user deletion.
    """

    __tablename__ = "gdpr_audit_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id = Column(String, nullable=False, unique=True, index=True)
    operation = Column(String, nullable=False, index=True)
    target_user_id = Column(String, nullable=False, index=True)
    requestor_id = Column(String, nullable=True, index=True)
    reason = Column(String, nullable=True)
    status = Column(String, nullable=False, default="started")
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    details = Column(JSON, nullable=True)


@dataclass
class GDPRAuditEntry:
    """Audit log entry for GDPR operations.

    Attributes:
        audit_id: Unique identifier for this audit entry
        operation: Type of GDPR operation
        target_user_id: The user whose data was accessed/modified
        requestor_id: Who requested the operation (may be None for automated)
        reason: Reason for the operation (optional)
        status: Current status (started, completed, failed)
        started_at: When the operation began
        completed_at: When the operation finished (None if still running)
        details: Additional details about the operation result
    """
    audit_id: str
    operation: GDPROperation
    target_user_id: str
    requestor_id: Optional[str]
    reason: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    details: Optional[Dict[str, Any]]


class GDPRAuditLogger:
    """Audit logger for GDPR operations.

    This class provides methods to log the start and completion of
    GDPR operations, and to query the audit log.
    """

    def __init__(self, db: Session):
        """Initialize the audit logger.

        Args:
            db: SQLAlchemy database session
        """
        self._db = db

    def log_operation_start(
        self,
        audit_id: str,
        operation: GDPROperation,
        target_user_id: str,
        requestor_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> GDPRAuditEntry:
        """Log the start of a GDPR operation.

        This should be called BEFORE the operation begins to ensure
        we have a record even if the operation fails.

        Args:
            audit_id: Unique identifier for this operation
            operation: Type of GDPR operation
            target_user_id: The user whose data will be accessed/modified
            requestor_id: Who is requesting the operation
            reason: Reason for the operation

        Returns:
            GDPRAuditEntry with the logged information
        """
        started_at = datetime.now(timezone.utc)

        record = GDPRAuditLogModel(
            audit_id=audit_id,
            operation=operation.value,
            target_user_id=target_user_id,
            requestor_id=requestor_id,
            reason=reason,
            status="started",
            started_at=started_at,
        )
        self._db.add(record)
        self._db.commit()

        return GDPRAuditEntry(
            audit_id=audit_id,
            operation=operation,
            target_user_id=target_user_id,
            requestor_id=requestor_id,
            reason=reason,
            status="started",
            started_at=started_at,
            completed_at=None,
            details=None,
        )

    def log_operation_complete(
        self,
        audit_id: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the completion of a GDPR operation.

        This should be called AFTER the operation completes (success or failure).

        Args:
            audit_id: The audit ID from log_operation_start
            status: Final status (completed, failed, partial)
            details: Additional details about the result
        """
        record = self._db.query(GDPRAuditLogModel).filter(
            GDPRAuditLogModel.audit_id == audit_id
        ).first()

        if record:
            record.status = status
            record.completed_at = datetime.now(timezone.utc)
            record.details = details
            self._db.commit()

    def get_audit_log(self, audit_id: str) -> Optional[GDPRAuditEntry]:
        """Get an audit log entry by ID.

        Args:
            audit_id: The unique audit ID

        Returns:
            GDPRAuditEntry if found, None otherwise
        """
        record = self._db.query(GDPRAuditLogModel).filter(
            GDPRAuditLogModel.audit_id == audit_id
        ).first()

        if not record:
            return None

        return GDPRAuditEntry(
            audit_id=record.audit_id,
            operation=GDPROperation(record.operation),
            target_user_id=record.target_user_id,
            requestor_id=record.requestor_id,
            reason=record.reason,
            status=record.status,
            started_at=record.started_at,
            completed_at=record.completed_at,
            details=record.details,
        )

    def list_audit_logs_for_user(self, target_user_id: str) -> List[GDPRAuditEntry]:
        """List all audit logs for a target user.

        Args:
            target_user_id: The user whose audit logs to retrieve

        Returns:
            List of GDPRAuditEntry ordered by started_at descending
        """
        records = self._db.query(GDPRAuditLogModel).filter(
            GDPRAuditLogModel.target_user_id == target_user_id
        ).order_by(GDPRAuditLogModel.started_at.desc()).all()

        return [
            GDPRAuditEntry(
                audit_id=r.audit_id,
                operation=GDPROperation(r.operation),
                target_user_id=r.target_user_id,
                requestor_id=r.requestor_id,
                reason=r.reason,
                status=r.status,
                started_at=r.started_at,
                completed_at=r.completed_at,
                details=r.details,
            )
            for r in records
        ]

    def list_recent_operations(
        self,
        limit: int = 100,
        operation: Optional[GDPROperation] = None,
    ) -> List[GDPRAuditEntry]:
        """List recent GDPR operations.

        Args:
            limit: Maximum number of entries to return
            operation: Filter by operation type (optional)

        Returns:
            List of GDPRAuditEntry ordered by started_at descending
        """
        query = self._db.query(GDPRAuditLogModel)

        if operation:
            query = query.filter(GDPRAuditLogModel.operation == operation.value)

        records = query.order_by(
            GDPRAuditLogModel.started_at.desc()
        ).limit(limit).all()

        return [
            GDPRAuditEntry(
                audit_id=r.audit_id,
                operation=GDPROperation(r.operation),
                target_user_id=r.target_user_id,
                requestor_id=r.requestor_id,
                reason=r.reason,
                status=r.status,
                started_at=r.started_at,
                completed_at=r.completed_at,
                details=r.details,
            )
            for r in records
        ]
