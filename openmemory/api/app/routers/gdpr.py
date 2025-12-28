"""GDPR REST API Router.

This router provides REST endpoints for GDPR operations:
- Subject Access Request (SAR) export: GET /v1/gdpr/export/{user_id}
- Right to Erasure (deletion): DELETE /v1/gdpr/user/{user_id}
- Audit log access: GET /v1/gdpr/audit/{user_id}

All endpoints require appropriate GDPR scopes and create audit logs.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.gdpr.audit import GDPRAuditLogger, GDPROperation
from app.gdpr.backup_purge import BackupPurgeTracker
from app.gdpr.deletion import UserDeletionOrchestrator
from app.gdpr.sar_export import SARExporter
from app.gdpr.schemas import DeletionResult, SARResponse
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/gdpr", tags=["gdpr"])

# Rate limiting configuration
# These limits are intentionally restrictive for GDPR operations
GDPR_RATE_LIMITS = {
    "export": {
        "requests_per_hour": 5,  # SAR exports are expensive
        "description": "Subject Access Request export",
    },
    "delete": {
        "requests_per_day": 1,  # Deletion is irreversible
        "description": "Right to erasure",
    },
}


@router.get(
    "/export/{user_id}",
    response_model=Dict[str, Any],
    summary="Export user data (Subject Access Request)",
    description="""
    Export all user data across all stores (Subject Access Request).

    GDPR Article 15: Right of access by the data subject.

    Returns all PII for the specified user from:
    - PostgreSQL (users, memories, apps, feedback, experiments)
    - Neo4j (graph nodes and relationships)
    - Qdrant (embedding metadata)
    - OpenSearch (indexed documents)
    - Valkey (session data)

    Requires the `gdpr:read` scope.
    """,
    responses={
        200: {"description": "User data export"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "User not found"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def export_user_data(
    user_id: str,
    principal: Principal = Depends(require_scopes(Scope.GDPR_READ)),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Export all user data (Subject Access Request).

    Args:
        user_id: The user ID to export data for
        principal: Authenticated principal with GDPR_READ scope
        db: Database session

    Returns:
        SARResponse containing all user data from all stores
    """
    audit_id = str(uuid4())
    audit_logger = GDPRAuditLogger(db=db)

    # Create audit log BEFORE operation
    audit_logger.log_operation_start(
        audit_id=audit_id,
        operation=GDPROperation.EXPORT,
        target_user_id=user_id,
        requestor_id=principal.user_id,
        reason="Subject Access Request via API",
    )

    try:
        # Perform export
        exporter = SARExporter(db=db)
        result = await exporter.export_user_data(user_id)

        # Update audit log with success
        audit_logger.log_operation_complete(
            audit_id=audit_id,
            status="completed" if not result.partial else "partial",
            details={
                "export_duration_ms": result.export_duration_ms,
                "partial": result.partial,
                "errors": result.errors,
            },
        )

        logger.info(
            f"SAR export completed: user_id={user_id}, "
            f"requestor={principal.user_id}, audit_id={audit_id}"
        )

        return result.to_dict()

    except Exception as e:
        # Update audit log with failure
        audit_logger.log_operation_complete(
            audit_id=audit_id,
            status="failed",
            details={"error": str(e)},
        )
        logger.error(f"SAR export failed: user_id={user_id}, error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {e}",
        )


@router.delete(
    "/user/{user_id}",
    response_model=Dict[str, Any],
    summary="Delete user data (Right to Erasure)",
    description="""
    Delete all user data across all stores (Right to Erasure).

    GDPR Article 17: Right to erasure ('right to be forgotten').

    Deletes all PII for the specified user from all stores in the
    correct dependency order:
    1. Valkey (session/cache data)
    2. OpenSearch (search indices)
    3. Qdrant (embeddings)
    4. Neo4j (graph relationships)
    5. PostgreSQL (primary data)

    This operation is IRREVERSIBLE. An audit trail is maintained.

    Requires the `gdpr:delete` scope.
    """,
    responses={
        200: {"description": "Deletion result"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "User not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Partial deletion failure"},
    },
)
async def delete_user_data(
    user_id: str,
    reason: str = Query(
        ...,
        description="Reason for deletion (required for audit)",
        min_length=10,
        max_length=500,
    ),
    principal: Principal = Depends(require_scopes(Scope.GDPR_DELETE)),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete all user data (Right to Erasure).

    Args:
        user_id: The user ID to delete
        reason: Reason for deletion (for audit trail)
        principal: Authenticated principal with GDPR_DELETE scope
        db: Database session

    Returns:
        DeletionResult containing status of each store's deletion
    """
    # Perform deletion with audit trail
    orchestrator = UserDeletionOrchestrator(db=db)
    result = await orchestrator.delete_user(
        user_id=user_id,
        audit_reason=reason,
        requestor_id=principal.user_id,
    )

    # Record for backup purge tracking
    try:
        purge_tracker = BackupPurgeTracker(db=db)
        purge_tracker.record_deletion(user_id)
    except Exception as e:
        logger.warning(f"Failed to record backup purge: {e}")

    logger.info(
        f"User deletion completed: user_id={user_id}, "
        f"requestor={principal.user_id}, success={result.success}"
    )

    if not result.success:
        # Return 500 with partial results
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Partial deletion - some stores failed",
                "result": result.to_dict(),
            },
        )

    return result.to_dict()


@router.get(
    "/audit/{user_id}",
    response_model=Dict[str, Any],
    summary="Get GDPR audit logs for a user",
    description="""
    Retrieve all GDPR audit logs for a specific user.

    Returns a list of all SAR exports and deletions performed
    for the specified user, including timestamps and requestor IDs.

    Requires the `gdpr:read` scope.
    """,
    responses={
        200: {"description": "Audit log entries"},
        403: {"description": "Insufficient permissions"},
    },
)
async def get_user_audit_logs(
    user_id: str,
    principal: Principal = Depends(require_scopes(Scope.GDPR_READ)),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get GDPR audit logs for a user.

    Args:
        user_id: The user ID to get audit logs for
        principal: Authenticated principal with GDPR_READ scope
        db: Database session

    Returns:
        Dictionary containing list of audit entries
    """
    audit_logger = GDPRAuditLogger(db=db)
    entries = audit_logger.list_audit_logs_for_user(user_id)

    return {
        "user_id": user_id,
        "entries": [
            {
                "audit_id": e.audit_id,
                "operation": e.operation.value,
                "requestor_id": e.requestor_id,
                "reason": e.reason,
                "status": e.status,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "details": e.details,
            }
            for e in entries
        ],
        "count": len(entries),
    }


@router.get(
    "/pending-purges",
    response_model=Dict[str, Any],
    summary="List pending backup purges",
    description="""
    List user deletions that are still within the backup retention period.

    These are users whose data may still exist in backups and should not
    be restored without proper filtering.

    Requires the `gdpr:read` scope.
    """,
    responses={
        200: {"description": "List of pending purges"},
        403: {"description": "Insufficient permissions"},
    },
)
async def list_pending_purges(
    principal: Principal = Depends(require_scopes(Scope.GDPR_READ)),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """List pending backup purges.

    Args:
        principal: Authenticated principal with GDPR_READ scope
        db: Database session

    Returns:
        Dictionary containing list of pending purges
    """
    tracker = BackupPurgeTracker(db=db)
    pending = tracker.get_pending_purges()

    return {
        "pending_purges": [
            {
                "id": p.id,
                "user_id": p.user_id,
                "deleted_at": p.deleted_at.isoformat(),
                "purge_after": p.purge_after.isoformat(),
                "retention_days": p.retention_days,
            }
            for p in pending
        ],
        "count": len(pending),
    }
