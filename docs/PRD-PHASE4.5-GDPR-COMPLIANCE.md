# PRD: Phase 4.5 GDPR Compliance

## Overview

**Phase**: 4.5 - GDPR Compliance
**Priority**: High (Legal/Compliance Requirement)
**Plan Reference**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress Tracking**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Continuation Prompt**: `docs/CONTINUATION-PROMPT-PHASE4.5-GDPR.md`

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Success Criteria

All criteria must be met for Phase 4.5 to be considered complete:

1. [ ] **PII Inventory Complete**: All PII fields documented across all 5 data stores (PostgreSQL, Neo4j, Qdrant, OpenSearch, Valkey) with validation that inventory matches actual schemas
2. [ ] **SAR Export Functional**: Subject Access Request export retrieves all user data from all stores and returns it in a defined JSON format within 30 seconds
3. [ ] **Cascading Delete Works**: User deletion removes all data from all 5 stores in the correct dependency order with audit trail
4. [ ] **Audit Trail Complete**: All GDPR operations (export, delete) are logged with timestamps, user IDs, and operation details
5. [ ] **Backup Purge Strategy Documented**: Either crypto-shredding or retention tracking is implemented and documented
6. [ ] **REST Endpoints Secured**: GDPR endpoints protected with GDPR_READ and GDPR_DELETE scopes
7. [ ] **Rate Limiting Applied**: GDPR endpoints have appropriate rate limits (e.g., 1 SAR/hour, 1 delete/day per user)
8. [ ] **Response Times Met**: SAR export completes within 30 seconds; deletion completes within 60 seconds
9. [ ] **All Tests Pass**: 40-60 new tests covering all GDPR functionality
10. [ ] **No Regressions**: All existing 3,374 tests continue to pass

### 1.2 Edge Cases

Document edge cases that must be handled:

1. **Non-existent User**: SAR/delete for user_id that doesn't exist should return 404, not error
2. **Partial Data**: User may have data in some stores but not others (e.g., no graph nodes)
3. **Concurrent Operations**: Two SAR requests for same user; SAR during deletion
4. **Large Data Sets**: User with thousands of memories should still export within timeout
5. **External Store Failures**: Qdrant/OpenSearch/Neo4j down during SAR or delete
6. **Orphaned Data**: Data in external stores without corresponding PostgreSQL record
7. **Active Sessions**: User with active Valkey sessions during deletion
8. **Audit Log Retention**: GDPR audit logs should be retained even after user deletion
9. **Encryption Key Management**: If using crypto-shredding, key loss scenario
10. **Replay Prevention**: Prevent re-deletion of already deleted user

### 1.3 Test Suite Structure

```
openmemory/api/tests/
├── gdpr/
│   ├── __init__.py
│   ├── test_pii_inventory.py          # PII field documentation tests
│   ├── test_sar_export.py             # Subject Access Request export tests
│   ├── test_deletion.py               # Cascading delete tests
│   ├── test_backup_purge.py           # Backup purge strategy tests
│   └── test_audit_logging.py          # GDPR audit trail tests
├── routers/
│   └── test_gdpr_router.py            # GDPR REST endpoint tests
└── security/
    └── test_gdpr_scopes.py            # GDPR scope enforcement tests
```

### 1.4 Test Specifications

#### Feature 1: PII Inventory

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| PII-001 | Unit | PIIField dataclass validates store names | ValueError for invalid store |
| PII-002 | Unit | PII inventory contains all PostgreSQL tables | users, memories, apps, feedback_events, experiments, variant_assignments |
| PII-003 | Unit | PII inventory contains all Neo4j node types | User, Memory, Entity nodes with user_id properties |
| PII-004 | Unit | PII inventory contains all Qdrant payload fields | user_id, org_id in embedding payloads |
| PII-005 | Unit | PII inventory contains all OpenSearch document fields | user_id, content in indexed documents |
| PII-006 | Unit | PII inventory contains all Valkey key patterns | episodic:{user_id}:*, session:{user_id}:* |
| PII-007 | Integration | Inventory matches actual PostgreSQL schema | All tables validated against information_schema |
| PII-008 | Integration | Inventory matches actual Neo4j schema | Node properties validated via db.schema.nodeTypeProperties() |
| PII-009 | Unit | PII types are correctly categorized | email, name, user_id, content, ip_address |
| PII-010 | Unit | Encryption levels documented | none, at_rest, field_level |

#### Feature 2: SAR Export

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| SAR-001 | Unit | SARExporter initializes with all store clients | No exceptions |
| SAR-002 | Unit | PostgreSQL export returns user record | User with email, name, metadata |
| SAR-003 | Unit | PostgreSQL export returns all memories | All memories with content, metadata |
| SAR-004 | Unit | PostgreSQL export returns all apps | All apps owned by user |
| SAR-005 | Unit | PostgreSQL export returns feedback events | Events filtered by user_id |
| SAR-006 | Unit | PostgreSQL export returns experiment assignments | Variant assignments for user |
| SAR-007 | Unit | Neo4j export returns user graph nodes | Nodes with user_id property |
| SAR-008 | Unit | Neo4j export returns relationships | Edges involving user's nodes |
| SAR-009 | Unit | Qdrant export returns embedding metadata | Vectors with user_id in payload |
| SAR-010 | Unit | OpenSearch export returns indexed documents | Documents with user_id filter |
| SAR-011 | Unit | Valkey export returns session data | All keys matching episodic:{user_id}:* |
| SAR-012 | Integration | Full SAR export returns all stores | Complete SARResponse with all data |
| SAR-013 | Unit | SAR response follows defined JSON schema | Validates against SARResponse model |
| SAR-014 | Unit | SAR handles non-existent user | Returns empty data, not error |
| SAR-015 | Unit | SAR handles partial data | Missing stores return empty objects |
| SAR-016 | Performance | SAR completes within 30 seconds | Timeout at 30s with partial results |

#### Feature 3: Cascading Delete

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| DEL-001 | Unit | Deletion order follows dependencies | Valkey -> OpenSearch -> Qdrant -> Neo4j -> PostgreSQL |
| DEL-002 | Unit | Valkey deletion removes session keys | All episodic:{user_id}:* keys deleted |
| DEL-003 | Unit | OpenSearch deletion removes documents | All documents with user_id removed |
| DEL-004 | Unit | Qdrant deletion removes embeddings | All points with user_id in payload removed |
| DEL-005 | Unit | Neo4j deletion removes nodes | All nodes with user_id property removed |
| DEL-006 | Unit | Neo4j deletion removes relationships | All edges involving deleted nodes removed |
| DEL-007 | Unit | PostgreSQL deletion removes user record | User row deleted |
| DEL-008 | Unit | PostgreSQL deletion cascades to memories | All Memory rows deleted |
| DEL-009 | Unit | PostgreSQL deletion cascades to apps | All App rows deleted |
| DEL-010 | Unit | PostgreSQL deletion cascades to feedback | All FeedbackEvent rows deleted |
| DEL-011 | Unit | Deletion creates audit record | DeletionAudit row with counts per store |
| DEL-012 | Unit | Deletion handles non-existent user | Returns 404, not error |
| DEL-013 | Unit | Deletion handles partial failures | Continues with remaining stores, logs errors |
| DEL-014 | Integration | Full deletion removes all user data | No data remains in any store |
| DEL-015 | Unit | Deletion is idempotent | Second call returns 404 |
| DEL-016 | Performance | Deletion completes within 60 seconds | Timeout with partial deletion status |

#### Feature 4: Backup Purge Strategy

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| BKP-001 | Unit | Crypto-shred key generation | Per-user key created on first backup |
| BKP-002 | Unit | Crypto-shred key retrieval | Key retrieved by user_id |
| BKP-003 | Unit | Crypto-shred key deletion | Key deleted, renders backups unreadable |
| BKP-004 | Unit | Backup metadata tracking | Deletion timestamp recorded |
| BKP-005 | Unit | Backup purge scheduling | Deletions queued for backup purge |
| BKP-006 | Unit | Retention policy enforcement | Old keys auto-deleted after retention period |

#### Feature 5: GDPR Router

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| RTR-001 | Unit | GET /v1/gdpr/export/{user_id} requires GDPR_READ | 403 without scope |
| RTR-002 | Unit | DELETE /v1/gdpr/user/{user_id} requires GDPR_DELETE | 403 without scope |
| RTR-003 | Unit | SAR endpoint returns JSON response | Content-Type: application/json |
| RTR-004 | Unit | SAR endpoint includes all stores | Response has postgres, neo4j, qdrant, opensearch, valkey keys |
| RTR-005 | Unit | Delete endpoint returns deletion result | DeletionResult with counts |
| RTR-006 | Unit | Rate limiting on SAR endpoint | 429 after 1 request/hour |
| RTR-007 | Unit | Rate limiting on delete endpoint | 429 after 1 request/day |
| RTR-008 | Unit | Audit log created for SAR | GDPRAuditLog with operation=export |
| RTR-009 | Unit | Audit log created for delete | GDPRAuditLog with operation=delete |
| RTR-010 | Integration | Full GDPR workflow | Export -> Delete -> Verify empty |

#### Feature 6: Audit Logging

| Test ID | Test Type | Description | Expected Outcome |
|---------|-----------|-------------|------------------|
| AUD-001 | Unit | GDPRAuditLog model has required fields | operation, user_id, timestamp, details |
| AUD-002 | Unit | SAR creates audit log before operation | Log exists even if SAR fails |
| AUD-003 | Unit | Delete creates audit log before operation | Log exists even if delete fails |
| AUD-004 | Unit | Audit logs survive user deletion | Logs retained after user deleted |
| AUD-005 | Unit | Audit logs include requestor identity | Principal user_id in log |
| AUD-006 | Unit | Audit logs include operation result | success/failure status, counts |

---

## PHASE 2: Feature Specifications

### Feature 1: PII Inventory

**Description**: Document all PII fields across all 5 data stores with validation that the inventory matches actual schemas.

**Dependencies**:
- All store implementations (Phase 4)
- Database models (`app/models.py`)

**Files to Create**:
- `openmemory/api/app/gdpr/__init__.py`
- `openmemory/api/app/gdpr/pii_inventory.py`
- `openmemory/api/tests/gdpr/__init__.py`
- `openmemory/api/tests/gdpr/test_pii_inventory.py`

**Test Cases**:
- [x] Unit: PIIField dataclass validates store names
- [x] Unit: PII inventory contains all PostgreSQL tables
- [x] Unit: PII inventory contains all Neo4j node types
- [x] Unit: PII inventory contains all Qdrant payload fields
- [x] Unit: PII inventory contains all OpenSearch document fields
- [x] Unit: PII inventory contains all Valkey key patterns
- [x] Integration: Inventory matches actual PostgreSQL schema
- [x] Integration: Inventory matches actual Neo4j schema

**Implementation Approach**:

```python
# app/gdpr/pii_inventory.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PIIType(Enum):
    EMAIL = "email"
    NAME = "name"
    USER_ID = "user_id"
    CONTENT = "content"  # Memory content may contain PII
    IP_ADDRESS = "ip_address"
    METADATA = "metadata"  # Arbitrary JSON that may contain PII

class EncryptionLevel(Enum):
    NONE = "none"
    AT_REST = "at_rest"
    FIELD_LEVEL = "field_level"

class Store(Enum):
    POSTGRES = "postgres"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    OPENSEARCH = "opensearch"
    VALKEY = "valkey"

@dataclass(frozen=True)
class PIIField:
    """Represents a PII field in a data store."""
    store: Store
    table_or_collection: str
    field_name: str
    pii_type: PIIType
    retention_days: Optional[int] = None  # None means indefinite
    encryption: EncryptionLevel = EncryptionLevel.NONE
    description: str = ""

# Complete PII inventory
PII_INVENTORY: list[PIIField] = [
    # PostgreSQL - users table
    PIIField(Store.POSTGRES, "users", "user_id", PIIType.USER_ID,
             description="External user identifier"),
    PIIField(Store.POSTGRES, "users", "email", PIIType.EMAIL,
             encryption=EncryptionLevel.AT_REST,
             description="User email address"),
    PIIField(Store.POSTGRES, "users", "name", PIIType.NAME,
             description="User display name"),
    PIIField(Store.POSTGRES, "users", "metadata", PIIType.METADATA,
             description="Arbitrary user metadata"),

    # PostgreSQL - memories table
    PIIField(Store.POSTGRES, "memories", "user_id", PIIType.USER_ID,
             description="FK to users table"),
    PIIField(Store.POSTGRES, "memories", "content", PIIType.CONTENT,
             description="Memory content may contain PII"),
    PIIField(Store.POSTGRES, "memories", "metadata", PIIType.METADATA,
             description="Memory metadata may contain PII"),

    # PostgreSQL - apps table
    PIIField(Store.POSTGRES, "apps", "owner_id", PIIType.USER_ID,
             description="FK to users table"),

    # PostgreSQL - feedback_events table
    PIIField(Store.POSTGRES, "feedback_events", "user_id", PIIType.USER_ID,
             retention_days=30,
             description="User who generated feedback"),
    PIIField(Store.POSTGRES, "feedback_events", "org_id", PIIType.USER_ID,
             retention_days=30,
             description="Organization ID"),

    # PostgreSQL - variant_assignments table
    PIIField(Store.POSTGRES, "variant_assignments", "user_id", PIIType.USER_ID,
             description="User assigned to experiment variant"),

    # PostgreSQL - memory_status_history table
    PIIField(Store.POSTGRES, "memory_status_history", "changed_by", PIIType.USER_ID,
             description="User who changed memory status"),

    # PostgreSQL - memory_access_logs table
    PIIField(Store.POSTGRES, "memory_access_logs", "metadata", PIIType.METADATA,
             description="Access log metadata may contain PII"),

    # Neo4j - nodes with user_id property
    PIIField(Store.NEO4J, "User", "user_id", PIIType.USER_ID,
             description="User node identifier"),
    PIIField(Store.NEO4J, "Memory", "user_id", PIIType.USER_ID,
             description="Memory node owner"),
    PIIField(Store.NEO4J, "Entity", "user_id", PIIType.USER_ID,
             description="Entity node owner"),

    # Qdrant - embedding payloads
    PIIField(Store.QDRANT, "embeddings_*", "user_id", PIIType.USER_ID,
             description="Vector payload user identifier"),
    PIIField(Store.QDRANT, "embeddings_*", "org_id", PIIType.USER_ID,
             description="Vector payload org identifier"),
    PIIField(Store.QDRANT, "embeddings_*", "content", PIIType.CONTENT,
             description="Original content in payload"),

    # OpenSearch - indexed documents
    PIIField(Store.OPENSEARCH, "memories", "user_id", PIIType.USER_ID,
             description="Document user identifier"),
    PIIField(Store.OPENSEARCH, "memories", "content", PIIType.CONTENT,
             description="Indexed memory content"),

    # Valkey - key patterns
    PIIField(Store.VALKEY, "episodic:{user_id}:*", "value", PIIType.CONTENT,
             description="Episodic memory session data"),
    PIIField(Store.VALKEY, "dpop:{thumbprint}:*", "user_id", PIIType.USER_ID,
             description="DPoP proof cache may reference user"),
]

def get_pii_fields_by_store(store: Store) -> list[PIIField]:
    """Get all PII fields for a specific store."""
    return [f for f in PII_INVENTORY if f.store == store]

def get_deletable_fields() -> list[PIIField]:
    """Get all PII fields that should be deleted on user deletion."""
    return [f for f in PII_INVENTORY if f.pii_type != PIIType.METADATA or f.retention_days is None]
```

**Git Commit Message**: `feat(gdpr): add PII inventory for all stores`

---

### Feature 2: SAR Export Orchestrator

**Description**: Implement Subject Access Request (SAR) export that retrieves all user data from all 5 stores and returns it in a defined JSON format.

**Dependencies**:
- PII Inventory (Feature 1)
- All store clients (Phase 4)
- Database session (`app/database.py`)

**Files to Create**:
- `openmemory/api/app/gdpr/sar_export.py`
- `openmemory/api/app/gdpr/schemas.py`
- `openmemory/api/tests/gdpr/test_sar_export.py`

**Test Cases**:
- [x] Unit: SARExporter initializes with all store clients
- [x] Unit: PostgreSQL export returns user record
- [x] Unit: PostgreSQL export returns all memories
- [x] Unit: PostgreSQL export returns all apps
- [x] Unit: PostgreSQL export returns feedback events
- [x] Unit: PostgreSQL export returns experiment assignments
- [x] Unit: Neo4j export returns user graph nodes
- [x] Unit: Qdrant export returns embedding metadata
- [x] Unit: OpenSearch export returns indexed documents
- [x] Unit: Valkey export returns session data
- [x] Integration: Full SAR export returns all stores
- [x] Unit: SAR response follows defined JSON schema
- [x] Unit: SAR handles non-existent user
- [x] Unit: SAR handles partial data

**Implementation Approach**:

```python
# app/gdpr/schemas.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

@dataclass
class SARResponse:
    """Subject Access Request response containing all user data."""
    user_id: str
    export_date: datetime
    format_version: str = "1.0"

    # Data from each store
    postgres: dict[str, Any] = field(default_factory=dict)
    neo4j: dict[str, Any] = field(default_factory=dict)
    qdrant: dict[str, Any] = field(default_factory=dict)
    opensearch: dict[str, Any] = field(default_factory=dict)
    valkey: dict[str, Any] = field(default_factory=dict)

    # Metadata
    export_duration_ms: Optional[int] = None
    partial: bool = False  # True if any store failed
    errors: list[str] = field(default_factory=list)

# app/gdpr/sar_export.py
class SARExporter:
    """Subject Access Request exporter for all stores."""

    EXPORT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        db: Session,
        neo4j_driver: Optional[neo4j.Driver] = None,
        qdrant_client: Optional[QdrantClient] = None,
        opensearch_client: Optional[OpenSearch] = None,
        valkey_client: Optional[Redis] = None,
    ):
        self._db = db
        self._neo4j = neo4j_driver
        self._qdrant = qdrant_client
        self._opensearch = opensearch_client
        self._valkey = valkey_client

    async def export_user_data(self, user_id: str) -> SARResponse:
        """Export all PII for a user across all stores."""
        start = datetime.now()
        errors = []

        # Export from each store with error handling
        postgres_data = await self._export_postgres(user_id)

        neo4j_data = {}
        if self._neo4j:
            try:
                neo4j_data = await self._export_neo4j(user_id)
            except Exception as e:
                errors.append(f"Neo4j export failed: {e}")

        qdrant_data = {}
        if self._qdrant:
            try:
                qdrant_data = await self._export_qdrant(user_id)
            except Exception as e:
                errors.append(f"Qdrant export failed: {e}")

        opensearch_data = {}
        if self._opensearch:
            try:
                opensearch_data = await self._export_opensearch(user_id)
            except Exception as e:
                errors.append(f"OpenSearch export failed: {e}")

        valkey_data = {}
        if self._valkey:
            try:
                valkey_data = await self._export_valkey(user_id)
            except Exception as e:
                errors.append(f"Valkey export failed: {e}")

        duration = (datetime.now() - start).total_seconds() * 1000

        return SARResponse(
            user_id=user_id,
            export_date=datetime.utcnow(),
            postgres=postgres_data,
            neo4j=neo4j_data,
            qdrant=qdrant_data,
            opensearch=opensearch_data,
            valkey=valkey_data,
            export_duration_ms=int(duration),
            partial=len(errors) > 0,
            errors=errors,
        )

    async def _export_postgres(self, user_id: str) -> dict[str, Any]:
        """Export user data from PostgreSQL tables."""
        from app.models import User, Memory, App
        from app.stores.feedback_store import FeedbackEventModel
        from app.stores.experiment_store import VariantAssignmentModel

        # Get user record
        user = self._db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return {"user": None, "memories": [], "apps": [], "feedback": [], "experiments": []}

        # Get all memories
        memories = self._db.query(Memory).filter(Memory.user_id == user.id).all()

        # Get all apps
        apps = self._db.query(App).filter(App.owner_id == user.id).all()

        # Get feedback events
        feedback = self._db.query(FeedbackEventModel).filter(
            FeedbackEventModel.user_id == user_id
        ).all()

        # Get experiment assignments
        assignments = self._db.query(VariantAssignmentModel).filter(
            VariantAssignmentModel.user_id == user_id
        ).all()

        return {
            "user": self._serialize_user(user),
            "memories": [self._serialize_memory(m) for m in memories],
            "apps": [self._serialize_app(a) for a in apps],
            "feedback": [self._serialize_feedback(f) for f in feedback],
            "experiments": [self._serialize_assignment(a) for a in assignments],
        }
```

**Git Commit Message**: `feat(gdpr): add SAR export orchestrator`

---

### Feature 3: Cascading Delete

**Description**: Implement user deletion that removes all data from all 5 stores in the correct dependency order with full audit trail.

**Dependencies**:
- PII Inventory (Feature 1)
- All store clients (Phase 4)
- Database session (`app/database.py`)

**Files to Create**:
- `openmemory/api/app/gdpr/deletion.py`
- `openmemory/api/tests/gdpr/test_deletion.py`

**Test Cases**:
- [x] Unit: Deletion order follows dependencies
- [x] Unit: Valkey deletion removes session keys
- [x] Unit: OpenSearch deletion removes documents
- [x] Unit: Qdrant deletion removes embeddings
- [x] Unit: Neo4j deletion removes nodes and relationships
- [x] Unit: PostgreSQL deletion cascades properly
- [x] Unit: Deletion creates audit record
- [x] Unit: Deletion handles non-existent user
- [x] Unit: Deletion handles partial failures
- [x] Integration: Full deletion removes all user data
- [x] Unit: Deletion is idempotent

**Implementation Approach**:

```python
# app/gdpr/deletion.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

@dataclass
class DeletionResult:
    """Result of a user deletion operation."""
    audit_id: str
    user_id: str
    timestamp: datetime
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    success: bool = True
    errors: list[str] = field(default_factory=list)

class UserDeletionOrchestrator:
    """Orchestrate user deletion across all stores with audit trail."""

    # Deletion order: dependencies first, primary data last
    DELETION_ORDER = [
        "valkey",      # Session/cache data (no dependencies)
        "opensearch",  # Search indices (can be rebuilt)
        "qdrant",      # Embeddings (can be rebuilt)
        "neo4j",       # Graph relationships
        "postgres",    # Primary data (last, FK constraints)
    ]

    def __init__(
        self,
        db: Session,
        neo4j_driver: Optional[neo4j.Driver] = None,
        qdrant_client: Optional[QdrantClient] = None,
        opensearch_client: Optional[OpenSearch] = None,
        valkey_client: Optional[Redis] = None,
    ):
        self._db = db
        self._neo4j = neo4j_driver
        self._qdrant = qdrant_client
        self._opensearch = opensearch_client
        self._valkey = valkey_client

    async def delete_user(
        self,
        user_id: str,
        audit_reason: str,
        requestor_id: Optional[str] = None,
    ) -> DeletionResult:
        """Delete all user data with audit trail."""
        audit_id = str(uuid4())
        results = {}
        errors = []

        # Create audit record BEFORE deletion
        await self._create_audit_log(
            audit_id=audit_id,
            user_id=user_id,
            operation="delete",
            reason=audit_reason,
            requestor_id=requestor_id,
            status="started",
        )

        for store in self.DELETION_ORDER:
            try:
                count = await self._delete_from_store(store, user_id)
                results[store] = {"status": "deleted", "count": count}
            except Exception as e:
                results[store] = {"status": "failed", "error": str(e)}
                errors.append(f"{store}: {e}")

        # Update audit record with result
        await self._update_audit_log(
            audit_id=audit_id,
            status="completed" if not errors else "partial",
            results=results,
        )

        return DeletionResult(
            audit_id=audit_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            results=results,
            success=len(errors) == 0,
            errors=errors,
        )

    async def _delete_from_store(self, store: str, user_id: str) -> int:
        """Delete user data from a specific store."""
        if store == "valkey":
            return await self._delete_valkey(user_id)
        elif store == "opensearch":
            return await self._delete_opensearch(user_id)
        elif store == "qdrant":
            return await self._delete_qdrant(user_id)
        elif store == "neo4j":
            return await self._delete_neo4j(user_id)
        elif store == "postgres":
            return await self._delete_postgres(user_id)
        else:
            raise ValueError(f"Unknown store: {store}")

    async def _delete_valkey(self, user_id: str) -> int:
        """Delete all Valkey keys for user."""
        if not self._valkey:
            return 0

        # Delete episodic memory keys
        pattern = f"episodic:{user_id}:*"
        keys = self._valkey.keys(pattern)
        if keys:
            self._valkey.delete(*keys)
        return len(keys)

    async def _delete_opensearch(self, user_id: str) -> int:
        """Delete all OpenSearch documents for user."""
        if not self._opensearch:
            return 0

        response = self._opensearch.delete_by_query(
            index="memories",
            body={"query": {"term": {"user_id": user_id}}},
        )
        return response.get("deleted", 0)

    async def _delete_qdrant(self, user_id: str) -> int:
        """Delete all Qdrant points for user."""
        if not self._qdrant:
            return 0

        # Delete from all embedding collections
        collections = self._qdrant.get_collections().collections
        count = 0
        for coll in collections:
            if coll.name.startswith("embeddings_"):
                result = self._qdrant.delete(
                    collection_name=coll.name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="user_id",
                                    match=models.MatchValue(value=user_id),
                                )
                            ]
                        )
                    ),
                )
                count += result.deleted or 0
        return count

    async def _delete_neo4j(self, user_id: str) -> int:
        """Delete all Neo4j nodes for user."""
        if not self._neo4j:
            return 0

        with self._neo4j.session() as session:
            result = session.run(
                """
                MATCH (n {user_id: $user_id})
                DETACH DELETE n
                RETURN count(n) as deleted
                """,
                user_id=user_id,
            )
            return result.single()["deleted"]

    async def _delete_postgres(self, user_id: str) -> int:
        """Delete all PostgreSQL data for user."""
        from app.models import User, Memory, App
        from app.stores.feedback_store import FeedbackEventModel
        from app.stores.experiment_store import VariantAssignmentModel

        # Get user
        user = self._db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return 0

        count = 0

        # Delete feedback events
        count += self._db.query(FeedbackEventModel).filter(
            FeedbackEventModel.user_id == user_id
        ).delete()

        # Delete variant assignments
        count += self._db.query(VariantAssignmentModel).filter(
            VariantAssignmentModel.user_id == user_id
        ).delete()

        # Delete memories (cascade to memory_categories, status_history, access_logs)
        count += self._db.query(Memory).filter(Memory.user_id == user.id).delete()

        # Delete apps
        count += self._db.query(App).filter(App.owner_id == user.id).delete()

        # Delete user
        self._db.delete(user)
        count += 1

        self._db.commit()
        return count
```

**Git Commit Message**: `feat(gdpr): add cascading user deletion with audit`

---

### Feature 4: Backup Purge Strategy

**Description**: Implement backup purge strategy using retention tracking (simpler than crypto-shredding for MVP).

**Dependencies**:
- Deletion orchestrator (Feature 3)
- Audit logging

**Files to Create**:
- `openmemory/api/app/gdpr/backup_purge.py`
- `openmemory/api/tests/gdpr/test_backup_purge.py`

**Test Cases**:
- [x] Unit: Deletion creates backup purge record
- [x] Unit: Backup purge record contains timestamp
- [x] Unit: Backup purge record contains user_id
- [x] Unit: Retention policy configuration
- [x] Unit: Backup purge list query

**Implementation Approach**:

```python
# app/gdpr/backup_purge.py
"""
Backup Purge Strategy: Retention Tracking

For MVP, we use retention tracking instead of crypto-shredding:
1. When a user is deleted, we record the deletion timestamp
2. Backups older than the retention period are considered "purged"
3. Documentation guides operators on backup rotation

Future: Implement crypto-shredding for true cryptographic erasure.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session
import uuid

from app.database import Base

# Default retention: backups older than 30 days are rotated out
DEFAULT_BACKUP_RETENTION_DAYS = 30

@dataclass
class BackupPurgeRecord:
    """Record of a user deletion for backup purge tracking."""
    id: str
    user_id: str
    deleted_at: datetime
    retention_days: int
    purge_after: datetime

class BackupPurgeTrackingModel(Base):
    """SQLAlchemy model for backup purge tracking."""

    __tablename__ = "gdpr_backup_purge_tracking"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True, unique=True)
    deleted_at = Column(DateTime(timezone=True), nullable=False)
    retention_days = Column(Integer, nullable=False, default=DEFAULT_BACKUP_RETENTION_DAYS)
    purge_after = Column(DateTime(timezone=True), nullable=False)

class BackupPurgeTracker:
    """Track user deletions for backup purge compliance."""

    def __init__(self, db: Session, retention_days: int = DEFAULT_BACKUP_RETENTION_DAYS):
        self._db = db
        self._retention_days = retention_days

    def record_deletion(self, user_id: str) -> BackupPurgeRecord:
        """Record a user deletion for backup purge tracking."""
        deleted_at = datetime.utcnow()
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

    def get_pending_purges(self) -> list[BackupPurgeRecord]:
        """Get deletions still within retention period."""
        now = datetime.utcnow()
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

    def get_completed_purges(self) -> list[BackupPurgeRecord]:
        """Get deletions past retention period (backups should be rotated)."""
        now = datetime.utcnow()
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
```

**Git Commit Message**: `feat(gdpr): add backup purge strategy`

---

### Feature 5: GDPR Router and Scopes

**Description**: Create REST API endpoints for GDPR operations with proper auth scopes and rate limiting.

**Dependencies**:
- SAR Exporter (Feature 2)
- Deletion Orchestrator (Feature 3)
- Security module (`app/security/`)

**Files to Create/Modify**:
- `openmemory/api/app/routers/gdpr.py`
- `openmemory/api/app/security/types.py` (add GDPR scopes)
- `openmemory/api/tests/routers/test_gdpr_router.py`
- `openmemory/api/tests/security/test_gdpr_scopes.py`

**Test Cases**:
- [x] Unit: GET /v1/gdpr/export/{user_id} requires GDPR_READ
- [x] Unit: DELETE /v1/gdpr/user/{user_id} requires GDPR_DELETE
- [x] Unit: SAR endpoint returns JSON response
- [x] Unit: Delete endpoint returns deletion result
- [x] Unit: Rate limiting on SAR endpoint
- [x] Unit: Rate limiting on delete endpoint
- [x] Unit: Audit log created for operations
- [x] Integration: Full GDPR workflow

**Implementation Approach**:

```python
# Add to app/security/types.py
class Scope(str, Enum):
    # ... existing scopes ...

    # GDPR operations (Phase 4.5)
    GDPR_READ = "gdpr:read"      # SAR export
    GDPR_DELETE = "gdpr:delete"  # User deletion

# app/routers/gdpr.py
from fastapi import APIRouter, Depends, HTTPException, status
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.gdpr.sar_export import SARExporter
from app.gdpr.deletion import UserDeletionOrchestrator
from app.gdpr.schemas import SARResponse, DeletionResult

router = APIRouter(prefix="/v1/gdpr", tags=["gdpr"])

@router.get("/export/{user_id}", response_model=SARResponse)
async def export_user_data(
    user_id: str,
    principal: Principal = Depends(require_scopes(Scope.GDPR_READ)),
    db: Session = Depends(get_db),
) -> SARResponse:
    """
    Export all user data (Subject Access Request).

    GDPR Article 15: Right of access by the data subject.
    """
    exporter = SARExporter(db, ...)  # Initialize with store clients
    response = await exporter.export_user_data(user_id)

    # Audit log
    await create_gdpr_audit_log(
        operation="export",
        target_user_id=user_id,
        requestor_id=principal.user_id,
        result="success" if not response.partial else "partial",
    )

    return response

@router.delete("/user/{user_id}", response_model=DeletionResult)
async def delete_user_data(
    user_id: str,
    reason: str,
    principal: Principal = Depends(require_scopes(Scope.GDPR_DELETE)),
    db: Session = Depends(get_db),
) -> DeletionResult:
    """
    Delete all user data (Right to Erasure).

    GDPR Article 17: Right to erasure ('right to be forgotten').
    """
    orchestrator = UserDeletionOrchestrator(db, ...)
    result = await orchestrator.delete_user(
        user_id=user_id,
        audit_reason=reason,
        requestor_id=principal.user_id,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Partial deletion: {result.errors}",
        )

    return result
```

**Git Commit Message**: `feat(gdpr): add GDPR REST endpoints`

---

### Feature 6: Audit Logging

**Description**: Implement comprehensive audit logging for all GDPR operations that survives user deletion.

**Dependencies**:
- Database (`app/database.py`)
- SAR Exporter (Feature 2)
- Deletion Orchestrator (Feature 3)

**Files to Create**:
- `openmemory/api/app/gdpr/audit.py`
- `openmemory/api/tests/gdpr/test_audit_logging.py`

**Test Cases**:
- [x] Unit: GDPRAuditLog model has required fields
- [x] Unit: SAR creates audit log before operation
- [x] Unit: Delete creates audit log before operation
- [x] Unit: Audit logs survive user deletion
- [x] Unit: Audit logs include requestor identity
- [x] Unit: Audit logs include operation result

**Implementation Approach**:

```python
# app/gdpr/audit.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from sqlalchemy import Column, DateTime, String, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session
import uuid

from app.database import Base

class GDPROperation(Enum):
    EXPORT = "export"   # SAR
    DELETE = "delete"   # Right to erasure
    ACCESS = "access"   # Data access audit

class GDPRAuditLogModel(Base):
    """SQLAlchemy model for GDPR audit logs."""

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
    """Audit log entry for GDPR operations."""
    audit_id: str
    operation: GDPROperation
    target_user_id: str
    requestor_id: Optional[str]
    reason: Optional[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    details: Optional[dict[str, Any]]

class GDPRAuditLogger:
    """Audit logger for GDPR operations."""

    def __init__(self, db: Session):
        self._db = db

    def log_operation_start(
        self,
        audit_id: str,
        operation: GDPROperation,
        target_user_id: str,
        requestor_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> GDPRAuditEntry:
        """Log the start of a GDPR operation."""
        record = GDPRAuditLogModel(
            audit_id=audit_id,
            operation=operation.value,
            target_user_id=target_user_id,
            requestor_id=requestor_id,
            reason=reason,
            status="started",
            started_at=datetime.utcnow(),
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
            started_at=record.started_at,
            completed_at=None,
            details=None,
        )

    def log_operation_complete(
        self,
        audit_id: str,
        status: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log the completion of a GDPR operation."""
        record = self._db.query(GDPRAuditLogModel).filter(
            GDPRAuditLogModel.audit_id == audit_id
        ).first()

        if record:
            record.status = status
            record.completed_at = datetime.utcnow()
            record.details = details
            self._db.commit()

    def get_audit_log(self, audit_id: str) -> Optional[GDPRAuditEntry]:
        """Get an audit log entry by ID."""
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

    def list_audit_logs_for_user(self, target_user_id: str) -> list[GDPRAuditEntry]:
        """List all audit logs for a target user."""
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
```

**Git Commit Message**: `feat(gdpr): add GDPR audit logging`

---

## PHASE 3: Development Protocol

### Execution Order

Follow this exact order for implementation:

1. **PII Inventory** (Feature 1)
   - Create gdpr module structure
   - Implement PIIField dataclass and inventory
   - Write validation tests

2. **SAR Export** (Feature 2)
   - Implement store-specific export methods
   - Create SARResponse schema
   - Write unit and integration tests

3. **Cascading Delete** (Feature 3)
   - Implement deletion orchestrator
   - Create audit logging
   - Write unit and integration tests

4. **Backup Purge** (Feature 4)
   - Implement retention tracking
   - Document procedures in runbook
   - Write unit tests

5. **GDPR Router** (Feature 5)
   - Add GDPR scopes to security module
   - Create REST endpoints
   - Add rate limiting
   - Write endpoint tests

6. **Audit Logging** (Feature 6)
   - Integrate audit logging with all operations
   - Write audit persistence tests

### Git Checkpoint Protocol

```bash
# After each feature passes tests
git add -A
git commit -m "feat(gdpr): [feature description]"

# After all features complete
git tag -a v0.4.5 -m "Phase 4.5: GDPR Compliance"
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-28
**Current Phase**: Phase 4.5 - GDPR Compliance
**Last Action**: PRD created

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | PII Inventory | [ ] | [ ] | [ ] | |
| 2 | SAR Export | [ ] | [ ] | [ ] | |
| 3 | Cascading Delete | [ ] | [ ] | [ ] | |
| 4 | Backup Purge | [ ] | [ ] | [ ] | |
| 5 | GDPR Router | [ ] | [ ] | [ ] | |
| 6 | Audit Logging | [ ] | [ ] | [ ] | |

### Decisions Made

1. **Decision**: Use retention tracking instead of crypto-shredding for MVP
   - **Rationale**: Simpler implementation, crypto-shredding can be added later
   - **Alternatives Considered**: Per-user encryption keys (more complex, better erasure guarantee)

2. **Decision**: Store audit logs in separate table from user data
   - **Rationale**: GDPR audit logs must survive user deletion
   - **Alternatives Considered**: External audit service (overkill for MVP)

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Codebase structure | Found all PII fields in models.py, store implementations |
| | | Security scopes in types.py, router patterns in routers/ |

### Known Issues & Blockers

- [ ] Issue: MCP SSE auth still pending (Phase 1)
  - Status: Deferred, not blocking GDPR
  - Attempted solutions: N/A for this phase

### Notes for Next Session

> Continue from here in the next session:

- [ ] Start with Feature 1: PII Inventory
- [ ] Create `openmemory/api/app/gdpr/__init__.py`
- [ ] Write tests first in `openmemory/api/tests/gdpr/test_pii_inventory.py`
- [ ] Run baseline test suite before starting

### Test Results Log

```
[Run docker compose exec codingbrain-mcp pytest tests/ -v --tb=short to establish baseline]
```

### Recent Git History

```
703dbca7 docs: add Phase 4.5 GDPR Compliance continuation prompt
b98c962b feat(phase7): complete Deployment, DR, and Hardening
02e60949 docs: add Phase 7 DR and Hardening PRD and continuation prompt
```

---

## Execution Checklist

When executing this PRD, follow this order:

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. **Spawn parallel Explore agents** to understand codebase
- [x] 3. Review/complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [ ] 5. Write feature specifications (Phase 2) - DONE
- [ ] 6. For each feature:
  - [ ] Write tests first
  - [ ] Implement feature
  - [ ] Run tests
  - [ ] Commit on green
  - [ ] Run regression tests
  - [ ] Update scratchpad
- [ ] 7. Tag milestone when complete

---

## Quick Reference Commands

### Test Strategy: Tiered Approach

Use this tiered testing strategy to balance speed with safety:

| When                   | What to Run            | Time     | Command                          |
| ---------------------- | ---------------------- | -------- | -------------------------------- |
| After each code change | GDPR tests only        | ~5s      | `pytest tests/gdpr/ -v`          |
| Before each commit     | GDPR + related modules | ~30s     | See "Pre-commit" below           |
| After feature complete | Full regression        | ~2-3min  | `pytest tests/ -v`               |
| Before PR/merge        | Full + coverage        | ~5min    | `pytest tests/ --cov=app`        |

### Test Commands

```bash
# =============================================================================
# TIER 1: Fast feedback (run after every change) - ~5 seconds
# =============================================================================
docker compose exec codingbrain-mcp pytest tests/gdpr/ -v --tb=short

# =============================================================================
# TIER 2: Pre-commit (run before committing) - ~30 seconds
# Includes GDPR + modules that GDPR depends on or modifies
# =============================================================================
docker compose exec codingbrain-mcp pytest \
  tests/gdpr/ \
  tests/stores/ \
  tests/security/test_gdpr_scopes.py \
  tests/routers/test_gdpr_router.py \
  -v --tb=short

# =============================================================================
# TIER 3: Feature complete (run after finishing a feature) - ~1-2 minutes
# Includes GDPR + all potentially affected areas
# =============================================================================
docker compose exec codingbrain-mcp pytest \
  tests/gdpr/ \
  tests/stores/ \
  tests/security/ \
  tests/routers/ \
  -v --tb=short

# =============================================================================
# TIER 4: Full regression (run before tagging/merging) - ~2-3 minutes
# All tests - required before phase completion
# =============================================================================
docker compose exec codingbrain-mcp pytest tests/ -v --tb=short

# =============================================================================
# TIER 5: Full with coverage (optional, for metrics) - ~5 minutes
# =============================================================================
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing
```

### Feature-Specific Test Commands

```bash
# Feature 1: PII Inventory
docker compose exec codingbrain-mcp pytest tests/gdpr/test_pii_inventory.py -v

# Feature 2: SAR Export
docker compose exec codingbrain-mcp pytest tests/gdpr/test_sar_export.py -v

# Feature 3: Cascading Delete
docker compose exec codingbrain-mcp pytest tests/gdpr/test_deletion.py -v

# Feature 4: Backup Purge
docker compose exec codingbrain-mcp pytest tests/gdpr/test_backup_purge.py -v

# Feature 5: GDPR Router
docker compose exec codingbrain-mcp pytest tests/routers/test_gdpr_router.py tests/security/test_gdpr_scopes.py -v

# Feature 6: Audit Logging
docker compose exec codingbrain-mcp pytest tests/gdpr/test_audit_logging.py -v
```

### Quick Checks

```bash
# Check test count (fast, no execution)
docker compose exec codingbrain-mcp pytest tests/ --collect-only -q | tail -3

# Run failed tests only (after a failure)
docker compose exec codingbrain-mcp pytest tests/ --lf -v

# Run tests matching a keyword
docker compose exec codingbrain-mcp pytest tests/ -k "gdpr or deletion" -v
```

### Git Workflow

```bash
git status
git add -A
git commit -m "feat(gdpr): description"
git log --oneline -5
```

---

**Testing Rules**:

1. **TIER 1** after every code change (fast feedback loop)
2. **TIER 2** before every commit (catch integration issues)
3. **TIER 3** after completing each feature (verify no regressions in related code)
4. **TIER 4** required before tagging phase complete (full confidence)
5. Never skip TIER 4 before merging or marking phase complete
