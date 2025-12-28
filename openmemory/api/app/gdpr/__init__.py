"""GDPR Compliance module for data protection operations.

This module provides:
- PII Inventory: Documentation of all PII fields across all data stores
- SAR Export: Subject Access Request export functionality
- Cascading Delete: Right to erasure implementation
- Audit Logging: GDPR operation audit trail
- Backup Purge: Strategy for backup data purging
"""

from app.gdpr.pii_inventory import (
    PIIField,
    PIIType,
    EncryptionLevel,
    Store,
    PII_INVENTORY,
    get_pii_fields_by_store,
    get_deletable_fields,
)
from app.gdpr.schemas import SARResponse, DeletionResult
from app.gdpr.sar_export import SARExporter
from app.gdpr.deletion import UserDeletionOrchestrator
from app.gdpr.audit import GDPRAuditLogger, GDPROperation, GDPRAuditEntry
from app.gdpr.backup_purge import BackupPurgeTracker, BackupPurgeRecord

__all__ = [
    # PII Inventory
    "PIIField",
    "PIIType",
    "EncryptionLevel",
    "Store",
    "PII_INVENTORY",
    "get_pii_fields_by_store",
    "get_deletable_fields",
    # Schemas
    "SARResponse",
    "DeletionResult",
    # SAR Export
    "SARExporter",
    # Deletion
    "UserDeletionOrchestrator",
    # Audit
    "GDPRAuditLogger",
    "GDPROperation",
    "GDPRAuditEntry",
    # Backup Purge
    "BackupPurgeTracker",
    "BackupPurgeRecord",
]
