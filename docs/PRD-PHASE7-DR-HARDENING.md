# PRD: Phase 7 - Deployment, DR, and Hardening

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Continuation Prompt**: `docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## Executive Summary

Phase 7 focuses on production hardening: backup/restore procedures for all data stores, automated backup verification, CI vulnerability scanning, container security hardening, and deployment playbooks for blue-green/canary releases.

**Target**: Documentation-heavy phase with ~10-20 new tests for verification scripts.

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Define Success Criteria

All measurable acceptance criteria for Phase 7:

1. [x] Backup/restore runbook covers all 5 data stores (PostgreSQL, Neo4j, Qdrant, OpenSearch, Valkey)
2. [x] Each backup procedure has verification steps documented
3. [x] Nightly backup verification script exists with alerting on failure
4. [x] Script validates backup existence, freshness, and integrity
5. [x] CI vulnerability scanning workflow configured (Trivy for containers)
6. [x] Dependency scanning enabled (pip-audit or safety for Python)
7. [x] SAST scanning enabled (bandit for Python)
8. [x] Dockerfile uses non-root user
9. [x] Container drops unnecessary Linux capabilities
10. [x] Blue-green deployment playbook documents rollback procedures
11. [x] Canary deployment playbook documents health verification between stages
12. [x] All procedures tested in development environment

### 1.2 Define Edge Cases

Edge cases that must be handled:

**Backup/Restore:**
- Backup fails mid-stream (partial backup detection)
- Backup location disk full
- Network timeout during backup to remote storage
- Corrupted backup file detection during restore
- Cross-version restore compatibility (e.g., Neo4j 5.x to 5.y)
- Large dataset backup (> 10GB) timeout handling

**Verification Script:**
- Backup file missing entirely
- Backup file exists but zero bytes
- Backup file older than expected (stale)
- Verification environment fails to start
- Restore succeeds but data integrity check fails
- Multiple backup files (choosing correct one)

**Vulnerability Scanning:**
- Scanner times out on large images
- False positives that shouldn't block CI
- Unfixable vulnerabilities (vendored dependencies)
- Scanner reports different severity levels

**Container Hardening:**
- Application requires root for certain operations
- Read-only filesystem breaking log writes
- Dropped capabilities breaking network operations

**Deployment:**
- Health check passes but application not fully ready
- Database migration conflicts during blue-green switch
- Canary traffic routing not working as expected
- Rollback during active user sessions

### 1.3 Design Test Suite Structure

```
openmemory/api/tests/
├── infrastructure/
│   ├── test_backup_verification.py      # NEW - Backup verification logic tests
│   └── test_backup_health_checks.py     # NEW - Backup freshness/size checks
└── ci/
    └── test_security_scan_config.py     # NEW - Verify CI config correctness (optional)

scripts/
└── verify_backup.py                     # NEW - Main verification script
    └── tests/
        └── test_verify_backup.py        # NEW - Script unit tests
```

### 1.4 Write Test Specifications

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| Backup exists check | Unit | Check if backup file exists at path | Returns True/False |
| Backup freshness check | Unit | Validate backup modified within N hours | Returns True if fresh, False if stale |
| Backup size validation | Unit | Check backup file size > 0 and reasonable | Returns True/validation error |
| Backup integrity check | Unit | Validate backup format (gzip header, JSON structure) | Returns True/error message |
| PostgreSQL restore test | Integration | Restore pg_dump to ephemeral database | Verify row counts match |
| Alert on failure | Unit | Mock alert function called when verification fails | Alert function invoked with error details |
| CI config validation | Unit | YAML structure has required fields | Validates Trivy/bandit/pip-audit steps |
| Container user check | Unit | Dockerfile USER directive is non-root | USER statement exists and isn't root |
| Capabilities check | Unit | Docker compose drops unnecessary caps | securityContext has dropped caps |

---

## PHASE 2: Feature Specifications

### Feature 1: Backup/Restore Runbook

**Description**: Comprehensive documentation for backup and restore procedures across all 5 data stores.

**Dependencies**: None (documentation only)

**File**: `docs/RUNBOOK-BACKUP-RESTORE.md`

**Test Cases**:
- [ ] N/A (documentation feature)

**Implementation Approach**:
1. Document PostgreSQL backup using pg_dump with format options
2. Document Neo4j backup using neo4j-admin dump
3. Document Qdrant backup using snapshot API endpoints
4. Document OpenSearch backup using snapshot/restore API
5. Document Valkey backup using BGSAVE and AOF
6. Add verification steps for each backup type
7. Document restore procedures with validation

**Git Commit Message**: `docs: add backup/restore runbook for all data stores`

---

### Feature 2: Nightly Backup Verification Script

**Description**: Python script that verifies backup existence, freshness, size, and optionally performs a restore test.

**Dependencies**: PostgreSQL, Valkey (for alerting)

**Test Cases**:
- [x] Unit: test_check_backup_exists_returns_true_for_existing_file
- [x] Unit: test_check_backup_exists_returns_false_for_missing_file
- [x] Unit: test_check_backup_freshness_passes_for_recent_backup
- [x] Unit: test_check_backup_freshness_fails_for_stale_backup
- [x] Unit: test_validate_backup_size_passes_for_nonzero
- [x] Unit: test_validate_backup_size_fails_for_empty
- [x] Unit: test_validate_backup_integrity_passes_for_valid_gzip
- [x] Unit: test_validate_backup_integrity_fails_for_corrupted
- [x] Unit: test_postgres_restore_test_runs_to_ephemeral
- [x] Unit: test_alert_called_on_verification_failure
- [x] Integration: test_full_verification_pipeline

**Implementation Approach**:
```python
# scripts/verify_backup.py
class BackupVerifier:
    def check_exists(self, path: str) -> bool
    def check_freshness(self, path: str, max_age_hours: int) -> bool
    def validate_size(self, path: str, min_bytes: int) -> bool
    def validate_integrity(self, path: str, backup_type: str) -> bool
    def restore_test(self, path: str, backup_type: str) -> bool
    def send_alert(self, error: str) -> None
    def verify_all(self, config: dict) -> dict[str, bool]
```

**Git Commit Message**: `feat(backup): add nightly restore verification script`

---

### Feature 3: CI Vulnerability Scanning

**Description**: GitHub Actions workflow for container scanning (Trivy), dependency scanning (pip-audit), and SAST (bandit).

**Dependencies**: GitHub Actions

**File**: `.github/workflows/security-scan.yml`

**Test Cases**:
- [x] N/A (CI configuration - tested by CI itself)

**Implementation Approach**:
1. Add Trivy scanner for container images
2. Configure CRITICAL/HIGH severity to block PRs
3. Add pip-audit for Python dependency vulnerabilities
4. Add bandit for Python static analysis
5. Add SARIF output for GitHub Security tab integration
6. Configure scan caching for faster runs

**Git Commit Message**: `ci: add vulnerability scanning workflow`

---

### Feature 4: Container Hardening

**Description**: Secure the Dockerfile following OWASP container security best practices.

**Dependencies**: Docker

**Files**: `openmemory/api/Dockerfile`

**Test Cases**:
- [x] Unit: test_dockerfile_has_nonroot_user
- [x] Unit: test_dockerfile_has_no_root_cmd
- [x] Unit: test_no_sensitive_files_in_image

**Implementation Approach**:
1. Create non-root user and group (appuser:1000)
2. Set ownership of application files to non-root user
3. Add USER directive before CMD
4. Remove unnecessary packages (curl, wget if not needed)
5. Use COPY --chown instead of RUN chown
6. Add .dockerignore to exclude secrets
7. Add read-only root filesystem guidance in compose
8. Document dropped capabilities in docker-compose.yml

**Git Commit Message**: `security: harden container image`

---

### Feature 5: Blue-Green/Canary Deployment Playbook

**Description**: Comprehensive deployment procedures for zero-downtime releases.

**Dependencies**: None (documentation only)

**File**: `docs/RUNBOOK-DEPLOYMENT.md`

**Test Cases**:
- [x] N/A (documentation feature)

**Implementation Approach**:
1. Document blue-green deployment topology
2. Add health check verification procedures
3. Document traffic switching process
4. Add rollback procedures with timelines
5. Document database migration coordination
6. Add canary deployment percentages (1%, 5%, 25%, 100%)
7. Include feature flag integration notes
8. Add monitoring and alerting checkpoints

**Git Commit Message**: `docs: add deployment playbook`

---

## PHASE 3: Development Protocol

### The Recursive Testing Loop

Execute this loop for EVERY testable feature:

```
1. WRITE TESTS FIRST
   └── Create failing tests that define expected behavior

2. IMPLEMENT FEATURE
   └── Write minimum code to pass tests

3. RUN ALL TESTS
   ├── docker compose exec codingbrain-mcp pytest tests/infrastructure/test_backup_verification.py -v
   └── docker compose exec codingbrain-mcp pytest tests/ -v --tb=short

4. ON PASS:
   ├── git add -A
   ├── git commit -m "feat(scope): description"
   └── Update Agent Scratchpad below

5. ON FAIL:
   ├── Spawn general-purpose agent: "Debug why [test] is failing"
   ├── If complex: Spawn Explore agent to find similar working patterns
   ├── DO NOT proceed until green
   └── Return to step 3

6. REGRESSION VERIFICATION
   ├── docker compose exec codingbrain-mcp pytest tests/ -v --tb=short
   ├── Verify all past features still work
   └── If regression found: fix before continuing

7. REPEAT for next feature
```

### Git Checkpoint Protocol

```bash
# After each passing feature
git add -A
git commit -m "type(scope): description"

# Conventional commit types:
# feat:     New feature
# fix:      Bug fix
# test:     Adding tests
# refactor: Code refactoring
# docs:     Documentation
# ci:       CI configuration
# security: Security improvements

# Tag milestones
git tag -a v0.7.0 -m "Milestone: Phase 7 DR and Hardening complete"
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-28
**Current Phase**: Phase 7 - Deployment, DR, and Hardening
**Last Action**: PRD created

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | Backup/Restore Runbook | N/A | N/A | [ ] | |
| 2 | Nightly Verification Script | [ ] | [ ] | [ ] | |
| 3 | CI Vulnerability Scanning | N/A | N/A | [ ] | |
| 4 | Container Hardening | [ ] | [ ] | [ ] | |
| 5 | Deployment Playbook | N/A | N/A | [ ] | |

### Decisions Made

1. **Decision**: Use Python for verification script instead of bash
   - **Rationale**: Better error handling, testability, and integration with existing codebase
   - **Alternatives Considered**: Bash script (simpler but harder to test)

2. **Decision**: Trivy for container scanning over Grype/Snyk
   - **Rationale**: Open source, comprehensive, good GitHub Actions integration, SARIF support
   - **Alternatives Considered**: Grype (similar), Snyk (requires license)

3. **Decision**: pip-audit over safety for dependency scanning
   - **Rationale**: Google-maintained, direct PyPI advisory database integration
   - **Alternatives Considered**: safety (requires account for full features)

### Attempted Approaches

| Approach | Outcome | Notes |
|----------|---------|-------|
|          |         |       |

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Test structure and patterns | 54 test files, pytest conventions, mock-based testing with fixtures |
| Explore | Docker and CI configuration | 15 Dockerfiles, 2 workflows (ci.yml, cd.yml), no security scanning configured |
| Explore | Backup implementations | backup.py router with export/import, alembic/utils.py with MigrationVerifier, 5 data stores with volume mounts |

### Known Issues & Blockers

- [ ] None identified yet

### Notes for Next Session

- [ ] Start with Feature 1: Create RUNBOOK-BACKUP-RESTORE.md
- [ ] Then Feature 2: Write tests for backup verification
- [ ] Feature 3: Add security-scan.yml workflow
- [ ] Feature 4: Harden Dockerfile with non-root user
- [ ] Feature 5: Create RUNBOOK-DEPLOYMENT.md

### Test Results Log

```
[Pending - will paste test output here during implementation]
```

### Recent Git History

```
c4e1b8b6 docs: complete Phase 6 Operability and Resilience
7cff3fdc feat(security): add token bucket rate limiting for Phase 6
46110af7 feat(resilience): add circuit breakers for external service calls
b4d7c82d feat(observability): add Prometheus metrics endpoint and middleware
16fd32d8 feat(observability): add OpenTelemetry distributed tracing
```

---

## Execution Checklist

When executing this PRD, follow this order:

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. **Spawn parallel Explore agents** to understand codebase (DONE)
- [x] 3. Review/complete success criteria (Phase 1) (DONE)
- [x] 4. Design test suite structure (Phase 1) (DONE)
- [x] 5. Write feature specifications (Phase 2) (DONE)
- [ ] 6. For each feature:
  - [ ] Write tests first (if applicable)
  - [ ] Implement feature
  - [ ] Run unit/integration tests
  - [ ] Commit on green
  - [ ] Run regression tests
  - [ ] Update scratchpad
- [ ] 7. Tag milestone when complete

---

## Quick Reference Commands

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v --tb=short

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_backup_verification.py -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Git workflow
git status
git add -A
git commit -m "feat(scope): description"
git log --oneline -5

# Database backup (PostgreSQL)
docker compose exec postgres pg_dump -U postgres openmemory > backup.sql

# Database restore (PostgreSQL)
docker compose exec -T postgres psql -U postgres openmemory < backup.sql

# Neo4j backup
docker compose exec neo4j neo4j-admin dump --database=neo4j --to=/data/neo4j-backup.dump

# Qdrant snapshot
curl -X POST http://localhost:6433/collections/embeddings/snapshots

# OpenSearch snapshot
curl -X PUT "localhost:9200/_snapshot/backups" -H 'Content-Type: application/json' -d'{"type": "fs", "settings": {"location": "/mnt/backups"}}'

# Valkey backup
docker compose exec valkey valkey-cli BGSAVE
```

---

## Appendix: Data Store Backup Reference

### PostgreSQL (Primary Relational DB)

**Backup Command**:
```bash
docker compose exec postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} -Fc > backup.dump
```

**Restore Command**:
```bash
docker compose exec -T postgres pg_restore -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c < backup.dump
```

**Volume**: `codingbrain_postgres_data:/var/lib/postgresql/data`

---

### Neo4j (Graph Database)

**Backup Command**:
```bash
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/data/
```

**Restore Command**:
```bash
docker compose exec neo4j neo4j-admin database load neo4j --from-path=/data/neo4j.dump --overwrite-destination=true
```

**Volume**: `codingbrain_neo4j_data:/data`

---

### Qdrant (Vector Database)

**Backup via Snapshot API**:
```bash
curl -X POST "http://localhost:6433/collections/{collection_name}/snapshots"
```

**List Snapshots**:
```bash
curl "http://localhost:6433/collections/{collection_name}/snapshots"
```

**Restore from Snapshot**:
```bash
curl -X PUT "http://localhost:6433/collections/{collection_name}/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///qdrant/snapshots/{snapshot_name}"}'
```

**Volume**: `codingbrain_qdrant_storage:/qdrant/storage`

---

### OpenSearch (Full-Text Search)

**Register Repository**:
```bash
curl -X PUT "localhost:9200/_snapshot/backups" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/mnt/backups"}}'
```

**Create Snapshot**:
```bash
curl -X PUT "localhost:9200/_snapshot/backups/snapshot_1?wait_for_completion=true"
```

**Restore Snapshot**:
```bash
curl -X POST "localhost:9200/_snapshot/backups/snapshot_1/_restore"
```

**Volume**: `codingbrain_opensearch_data:/usr/share/opensearch/data`

---

### Valkey (Cache/Session Store)

**Trigger Backup**:
```bash
docker compose exec valkey valkey-cli BGSAVE
```

**Check Backup Status**:
```bash
docker compose exec valkey valkey-cli LASTSAVE
```

**Backup Files**: `/data/dump.rdb` (RDB), `/data/appendonly.aof` (AOF)

**Volume**: `codingbrain_valkey_data:/data`

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
