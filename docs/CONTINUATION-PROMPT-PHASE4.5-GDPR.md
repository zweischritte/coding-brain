# Phase 4.5 Continuation: GDPR Compliance

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**PRD**: `docs/PRD-PHASE4.5-GDPR-COMPLIANCE.md` (to be created)
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
2. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress
3. Read `docs/PRD-PHASE4.5-GDPR-COMPLIANCE.md` for detailed specifications (create if missing)
4. Check Section 4 (Next Tasks) below for what to work on
5. Run the test suite to verify baseline: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update Phase 4.5 task status table
   - Add entry to Daily Log with date, work completed, and notes
2. Update Section 4 (Next Tasks) with remaining work
3. **If Phase 4.5 complete**: Update continuation prompt for next priority (MCP auth or Phase 8)
4. Commit all changes:

```bash
git add docs/CONTINUATION-PROMPT-PHASE4.5-GDPR.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md docs/PRD-PHASE4.5-GDPR-COMPLIANCE.md
git commit -m "docs: update Phase 4.5 GDPR session progress"
```

---

## 1. Current Gaps

**Phase 1 Gap**: MCP SSE authentication not implemented (deferred - requires architecture rework)
**Phase 4.5 Gap**: GDPR compliance not started (SAR export, cascading delete) - THIS PHASE
**Phase 6 Gap**: Alert thresholds + SLOs not implemented (deferred)

---

## 2. Command Reference

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/gdpr/test_sar_export.py -v

# Check health endpoints
curl http://localhost:8865/health/live
curl http://localhost:8865/health/ready
curl http://localhost:8865/health/deps

# Database queries for PII audit
docker compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\d+ memories"
docker compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\d+ users"

# Neo4j PII check
docker compose exec neo4j cypher-shell -u ${NEO4J_USERNAME} -p ${NEO4J_PASSWORD} \
  "CALL db.schema.nodeTypeProperties()"

# Qdrant collection schema
curl http://localhost:6433/collections/embeddings

# OpenSearch mapping
curl http://localhost:9200/memories/_mapping

# Valkey keys pattern
docker compose exec valkey valkey-cli KEYS "*"
```

---

## 3. Architecture Patterns

### PII Inventory Pattern

```python
# app/gdpr/pii_inventory.py
@dataclass
class PIIField:
    store: str           # postgres, neo4j, qdrant, opensearch, valkey
    table_or_collection: str
    field_name: str
    pii_type: str        # email, name, user_id, ip_address, content
    retention_days: int | None
    encryption: str      # none, at_rest, field_level

PII_INVENTORY: list[PIIField] = [
    PIIField("postgres", "users", "email", "email", None, "at_rest"),
    PIIField("postgres", "users", "user_id", "user_id", None, "none"),
    PIIField("postgres", "memories", "memory", "content", None, "none"),
    PIIField("postgres", "memories", "user_id", "user_id", None, "none"),
    # ... etc
]
```

### SAR Export Orchestrator Pattern

```python
# app/gdpr/sar_export.py
class SARExporter:
    """Subject Access Request exporter for all stores."""

    async def export_user_data(self, user_id: str) -> SARResponse:
        """Export all PII for a user across all stores."""
        data = {
            "postgres": await self._export_postgres(user_id),
            "neo4j": await self._export_neo4j(user_id),
            "qdrant": await self._export_qdrant(user_id),
            "opensearch": await self._export_opensearch(user_id),
            "valkey": await self._export_valkey(user_id),
        }
        return SARResponse(
            user_id=user_id,
            export_date=datetime.utcnow(),
            data=data,
            format_version="1.0"
        )

    async def _export_postgres(self, user_id: str) -> dict:
        """Export user data from PostgreSQL tables."""
        # memories, apps, feedback, experiments, users
        pass
```

### Cascading Delete Pattern

```python
# app/gdpr/deletion.py
class UserDeletionOrchestrator:
    """Orchestrate user deletion across all stores with audit trail."""

    # Deletion order matters - dependencies first
    DELETION_ORDER = [
        "valkey",      # Session/cache data (no dependencies)
        "opensearch",  # Search indices (can be rebuilt)
        "qdrant",      # Embeddings (can be rebuilt)
        "neo4j",       # Graph relationships
        "postgres",    # Primary data (last)
    ]

    async def delete_user(
        self,
        user_id: str,
        audit_reason: str
    ) -> DeletionResult:
        """Delete all user data with audit trail."""
        audit_id = uuid4()
        results = {}

        for store in self.DELETION_ORDER:
            try:
                count = await self._delete_from_store(store, user_id)
                results[store] = {"status": "deleted", "count": count}
                await self._log_audit(audit_id, store, user_id, count)
            except Exception as e:
                results[store] = {"status": "failed", "error": str(e)}
                # Decide: continue or rollback?

        return DeletionResult(
            audit_id=audit_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            results=results
        )
```

### Crypto-Shredding Pattern (for backups)

```python
# app/gdpr/crypto_shred.py
class CryptoShredder:
    """Manage per-user encryption keys for backup purging."""

    async def get_user_key(self, user_id: str) -> bytes:
        """Get or create user-specific encryption key."""
        pass

    async def encrypt_for_backup(self, user_id: str, data: bytes) -> bytes:
        """Encrypt data with user's key before backup."""
        pass

    async def shred_user_key(self, user_id: str) -> None:
        """Delete user's key, making all backups unreadable."""
        # This effectively deletes user data from all backups
        pass
```

---

## 4. Next Tasks

Execute in this order:

### Step 1: PII Inventory

- [ ] Create `openmemory/api/app/gdpr/__init__.py`
- [ ] Create `openmemory/api/app/gdpr/pii_inventory.py`
- [ ] Write tests first: `openmemory/api/tests/gdpr/test_pii_inventory.py`
- [ ] Document all PII fields in PostgreSQL (users, memories, apps, feedback, experiments)
- [ ] Document all PII fields in Neo4j (nodes with user_id)
- [ ] Document all PII fields in Qdrant (payloads with user/org data)
- [ ] Document all PII fields in OpenSearch (indexed user content)
- [ ] Document all PII fields in Valkey (session keys, cached user data)
- [ ] Add validation that inventory matches actual schema
- [ ] Commit: `feat(gdpr): add PII inventory for all stores`

### Step 2: SAR Export Orchestrator

- [ ] Write tests first: `openmemory/api/tests/gdpr/test_sar_export.py`
- [ ] Create `openmemory/api/app/gdpr/sar_export.py`
- [ ] Implement PostgreSQL export (users, memories, apps, feedback, experiments)
- [ ] Implement Neo4j export (user's graph nodes and relationships)
- [ ] Implement Qdrant export (user's embeddings metadata)
- [ ] Implement OpenSearch export (user's indexed documents)
- [ ] Implement Valkey export (user's session/cache data)
- [ ] Define SAR response format (JSON schema)
- [ ] Add SAR REST endpoint: `GET /v1/gdpr/export/{user_id}`
- [ ] Commit: `feat(gdpr): add SAR export orchestrator`

### Step 3: Cascading Delete

- [ ] Write tests first: `openmemory/api/tests/gdpr/test_deletion.py`
- [ ] Create `openmemory/api/app/gdpr/deletion.py`
- [ ] Implement Valkey deletion (session/cache cleanup)
- [ ] Implement OpenSearch deletion (document removal)
- [ ] Implement Qdrant deletion (embedding removal)
- [ ] Implement Neo4j deletion (node/relationship removal)
- [ ] Implement PostgreSQL deletion (cascade with proper order)
- [ ] Add audit logging for all deletions
- [ ] Add deletion REST endpoint: `DELETE /v1/gdpr/user/{user_id}`
- [ ] Commit: `feat(gdpr): add cascading user deletion with audit`

### Step 4: Backup Purge Strategy

- [ ] Write tests first: `openmemory/api/tests/gdpr/test_backup_purge.py`
- [ ] Create `openmemory/api/app/gdpr/backup_purge.py`
- [ ] Decide: crypto-shredding vs retention tracking
- [ ] If crypto-shred: implement per-user key management
- [ ] If retention: implement backup metadata tracking
- [ ] Document backup purge procedures in RUNBOOK-BACKUP-RESTORE.md
- [ ] Commit: `feat(gdpr): add backup purge strategy`

### Step 5: GDPR Router and Integration

- [ ] Write tests first: `openmemory/api/tests/routers/test_gdpr_router.py`
- [ ] Create `openmemory/api/app/routers/gdpr.py`
- [ ] Add scopes: GDPR_READ, GDPR_DELETE
- [ ] Register router in main.py
- [ ] Add rate limiting for GDPR endpoints
- [ ] Add audit logging for all GDPR operations
- [ ] Commit: `feat(gdpr): add GDPR REST endpoints`

---

## 5. Known Issues

1. **MCP SSE auth pending**: SSE endpoints need auth but test_mcp_auth.py hangs; deferred
2. **Pydantic V1 deprecation**: `app/schemas.py:54` uses V1 @validator (non-blocking)
3. **Qdrant version mismatch**: Client 1.16.2 vs server 1.12.5 (non-blocking)
4. **Alert thresholds deferred**: SLO definitions postponed from Phase 6

---

## 6. Last Session Summary (2025-12-28)

**Completed**: Phase 7 Deployment, DR, and Hardening

- Added 47 new tests across 2 test files
- Created backup/restore runbook for all 5 data stores
- Implemented BackupVerifier class with 31 tests
- Added CI security scanning workflow (Trivy, pip-audit, bandit, Hadolint)
- Hardened Dockerfile with non-root user, COPY --chown, HEALTHCHECK
- Created deployment playbook with blue-green and canary procedures

**Result**: Phase 7 complete, 3,374 total tests

---

## 7. Reference Files

**Existing store implementations (for reference):**

- `openmemory/api/app/stores/memory_store.py` - ScopedMemoryStore (PostgreSQL)
- `openmemory/api/app/stores/feedback_store.py` - PostgresFeedbackStore
- `openmemory/api/app/stores/experiment_store.py` - PostgresExperimentStore
- `openmemory/api/app/stores/episodic_store.py` - ValkeyEpisodicStore
- `openmemory/api/app/stores/qdrant_store.py` - TenantQdrantStore
- `openmemory/api/app/stores/opensearch_store.py` - TenantOpenSearchStore

**Database models:**

- `openmemory/api/app/models.py` - SQLAlchemy models (Memory, App, User, etc.)
- `openmemory/api/app/database.py` - Database connection and tenant_session

**Security patterns:**

- `openmemory/api/app/security/dependencies.py` - require_scopes pattern
- `openmemory/api/app/security/types.py` - Scope enum (add GDPR_READ, GDPR_DELETE)

---

## 8. Commit Template

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(gdpr): description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## 9. Success Criteria Checklist

Phase 4.5 is complete when ALL are checked:

- [ ] PII inventory documented for all 5 data stores
- [ ] SAR export working for all stores with defined response format
- [ ] Cascading delete working with proper dependency order
- [ ] Audit trail logged for all GDPR operations
- [ ] Backup purge strategy documented and implemented
- [ ] GDPR REST endpoints with proper auth scopes
- [ ] All procedures tested end-to-end

**Target**: ~40-60 new tests for GDPR functionality

---

## 10. Phase Transition

When Phase 4.5 is complete:

1. **Verify completion:**
   - All checkboxes in Section 4 are checked
   - All tests passing: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`
   - SAR export and deletion validated end-to-end

2. **Update Progress file:**
   - Mark Phase 4.5 as ✅ in header
   - Fill in all commit hashes
   - Add final Daily Log entry

3. **Next priority:**
   - Option A: Fix MCP SSE auth (Phase 1 completion)
   - Option B: Phase 8 Scale-Out (if auth is acceptable risk)

4. **Commit together:**

```bash
git add docs/CONTINUATION-PROMPT-PHASE4.5-GDPR.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "$(cat <<'EOF'
docs: complete Phase 4.5 GDPR Compliance

- PII inventory for all 5 data stores
- SAR export orchestrator with JSON response format
- Cascading user deletion with audit trail
- Backup purge strategy documented
- GDPR REST endpoints with auth scopes

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```
