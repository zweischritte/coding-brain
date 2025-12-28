"""PII Inventory for GDPR Compliance.

This module documents all Personally Identifiable Information (PII) fields
across all data stores in the system. This inventory is used for:
- Subject Access Request (SAR) exports
- Right to Erasure (cascading deletes)
- Audit and compliance reporting

Each PIIField documents:
- Which store contains the data (PostgreSQL, Neo4j, Qdrant, OpenSearch, Valkey)
- The table/collection/key pattern where data is stored
- The field name containing PII
- The type of PII (email, name, user_id, content, etc.)
- Retention policy (if any)
- Encryption level applied to the field
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    NAME = "name"
    USER_ID = "user_id"
    CONTENT = "content"  # Memory content may contain PII
    IP_ADDRESS = "ip_address"
    METADATA = "metadata"  # Arbitrary JSON that may contain PII


class EncryptionLevel(Enum):
    """Encryption levels applied to PII fields."""
    NONE = "none"  # No field-level encryption (may still have disk encryption)
    AT_REST = "at_rest"  # Encrypted at rest by the database
    FIELD_LEVEL = "field_level"  # Application-level field encryption


class Store(Enum):
    """Data stores in the system."""
    POSTGRES = "postgres"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    OPENSEARCH = "opensearch"
    VALKEY = "valkey"


@dataclass(frozen=True)
class PIIField:
    """Represents a PII field in a data store.

    Attributes:
        store: The data store containing this field
        table_or_collection: Table name (PostgreSQL), node type (Neo4j),
                            collection (Qdrant), index (OpenSearch), or
                            key pattern (Valkey)
        field_name: Name of the field containing PII
        pii_type: Type of PII contained in the field
        retention_days: Retention period in days (None means indefinite)
        encryption: Encryption level applied to this field
        description: Human-readable description of the field
    """
    store: Store
    table_or_collection: str
    field_name: str
    pii_type: PIIType
    retention_days: Optional[int] = None
    encryption: EncryptionLevel = EncryptionLevel.NONE
    description: str = ""


# =============================================================================
# COMPLETE PII INVENTORY
# =============================================================================
# This inventory documents ALL fields containing PII across ALL stores.
# It is used for SAR exports, cascading deletes, and compliance auditing.
# =============================================================================

PII_INVENTORY: List[PIIField] = [
    # =========================================================================
    # PostgreSQL - Primary Data Store
    # =========================================================================

    # --- users table ---
    PIIField(
        Store.POSTGRES, "users", "user_id", PIIType.USER_ID,
        description="External user identifier from identity provider"
    ),
    PIIField(
        Store.POSTGRES, "users", "email", PIIType.EMAIL,
        encryption=EncryptionLevel.AT_REST,
        description="User email address"
    ),
    PIIField(
        Store.POSTGRES, "users", "name", PIIType.NAME,
        description="User display name"
    ),
    PIIField(
        Store.POSTGRES, "users", "metadata", PIIType.METADATA,
        description="Arbitrary user metadata (JSON)"
    ),

    # --- memories table ---
    PIIField(
        Store.POSTGRES, "memories", "user_id", PIIType.USER_ID,
        description="FK to users table - memory owner"
    ),
    PIIField(
        Store.POSTGRES, "memories", "content", PIIType.CONTENT,
        description="Memory content - may contain user-generated PII"
    ),
    PIIField(
        Store.POSTGRES, "memories", "metadata", PIIType.METADATA,
        description="Memory metadata (JSON) - may contain PII"
    ),

    # --- apps table ---
    PIIField(
        Store.POSTGRES, "apps", "owner_id", PIIType.USER_ID,
        description="FK to users table - app owner"
    ),

    # --- feedback_events table ---
    PIIField(
        Store.POSTGRES, "feedback_events", "user_id", PIIType.USER_ID,
        retention_days=30,
        description="User who generated feedback event"
    ),
    PIIField(
        Store.POSTGRES, "feedback_events", "org_id", PIIType.USER_ID,
        retention_days=30,
        description="Organization ID for tenant isolation"
    ),

    # --- variant_assignments table ---
    PIIField(
        Store.POSTGRES, "variant_assignments", "user_id", PIIType.USER_ID,
        description="User assigned to experiment variant"
    ),

    # --- memory_status_history table ---
    PIIField(
        Store.POSTGRES, "memory_status_history", "changed_by", PIIType.USER_ID,
        description="User who changed memory status"
    ),

    # --- memory_access_logs table ---
    PIIField(
        Store.POSTGRES, "memory_access_logs", "metadata", PIIType.METADATA,
        description="Access log metadata - may contain PII"
    ),

    # =========================================================================
    # Neo4j - Graph Store
    # =========================================================================

    # --- User nodes ---
    PIIField(
        Store.NEO4J, "User", "user_id", PIIType.USER_ID,
        description="User node identifier property"
    ),

    # --- Memory nodes ---
    PIIField(
        Store.NEO4J, "Memory", "user_id", PIIType.USER_ID,
        description="Memory node owner property"
    ),

    # --- Entity nodes ---
    PIIField(
        Store.NEO4J, "Entity", "user_id", PIIType.USER_ID,
        description="Entity node owner property (extracted entities from memories)"
    ),

    # =========================================================================
    # Qdrant - Vector Store
    # =========================================================================

    # --- Embedding collections (per-model naming: embeddings_{model_name}) ---
    PIIField(
        Store.QDRANT, "embeddings_*", "user_id", PIIType.USER_ID,
        description="User ID in embedding payload for tenant isolation"
    ),
    PIIField(
        Store.QDRANT, "embeddings_*", "org_id", PIIType.USER_ID,
        description="Organization ID in embedding payload for tenant isolation"
    ),
    PIIField(
        Store.QDRANT, "embeddings_*", "content", PIIType.CONTENT,
        description="Original content stored in payload (may contain PII)"
    ),

    # =========================================================================
    # OpenSearch - Search Index
    # =========================================================================

    # --- Memories index ---
    PIIField(
        Store.OPENSEARCH, "memories", "user_id", PIIType.USER_ID,
        description="User ID field for tenant-scoped search"
    ),
    PIIField(
        Store.OPENSEARCH, "memories", "content", PIIType.CONTENT,
        description="Indexed memory content (may contain PII)"
    ),

    # =========================================================================
    # Valkey - Cache/Session Store
    # =========================================================================

    # --- Episodic memory keys ---
    PIIField(
        Store.VALKEY, "episodic:{user_id}:*", "value", PIIType.CONTENT,
        retention_days=1,  # 24-hour TTL
        description="Episodic memory session data (ephemeral)"
    ),

    # --- DPoP replay cache (security) ---
    PIIField(
        Store.VALKEY, "dpop:{thumbprint}:*", "user_id", PIIType.USER_ID,
        retention_days=1,
        description="DPoP proof cache - may reference user"
    ),
]


def get_pii_fields_by_store(store: Store) -> List[PIIField]:
    """Get all PII fields for a specific store.

    Args:
        store: The store to filter by

    Returns:
        List of PIIField instances for the specified store
    """
    return [f for f in PII_INVENTORY if f.store == store]


def get_deletable_fields() -> List[PIIField]:
    """Get all PII fields that should be deleted on user deletion.

    This returns fields that contain data attributable to a specific user
    and should be deleted as part of a Right to Erasure request.

    Returns:
        List of PIIField instances that should be deleted
    """
    return [
        f for f in PII_INVENTORY
        if f.pii_type != PIIType.METADATA or f.retention_days is None
    ]


def get_pii_fields_with_retention() -> List[PIIField]:
    """Get PII fields that have retention policies.

    Returns:
        List of PIIField instances with retention_days set
    """
    return [f for f in PII_INVENTORY if f.retention_days is not None]


def get_encrypted_pii_fields() -> List[PIIField]:
    """Get PII fields that have encryption applied.

    Returns:
        List of PIIField instances with encryption other than NONE
    """
    return [f for f in PII_INVENTORY if f.encryption != EncryptionLevel.NONE]
