# Phase 7 Continuation: Deployment, DR, and Hardening

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**PRD**: `docs/PRD-PHASE7-DR-HARDENING.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
2. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress
3. Read `docs/PRD-PHASE7-DR-HARDENING.md` for detailed specifications
4. Check Section 4 (Next Tasks) below for what to work on
5. Run the test suite to verify baseline: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update Phase 7 task status table
   - Add entry to Daily Log with date, work completed, and notes
2. Update Section 4 (Next Tasks) with remaining work
3. **If Phase 7 complete**: Create `docs/CONTINUATION-PROMPT-PHASE8-SCALEOUT.md` for next phase
4. Commit all changes:

```bash
git add docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md docs/PRD-PHASE7-DR-HARDENING.md
git commit -m "docs: update Phase 7 session progress"
```

---

## 1. Current Gaps

**Phase 1 Gap**: MCP SSE authentication not implemented (deferred - requires architecture rework)
**Phase 4.5 Gap**: GDPR compliance not started (SAR export, cascading delete)
**Phase 6 Gap**: Alert thresholds + SLOs not implemented (deferred)

---

## 2. Command Reference

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_backup_verification.py -v

# Check health endpoints
curl http://localhost:8865/health/live
curl http://localhost:8865/health/ready
curl http://localhost:8865/health/deps

# Database backup (PostgreSQL)
docker compose exec postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} -Fc > backup.dump

# Database restore (PostgreSQL)
docker compose exec -T postgres pg_restore -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c < backup.dump

# Neo4j backup
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/data/

# Qdrant snapshot
curl -X POST "http://localhost:6433/collections/{collection}/snapshots"

# OpenSearch snapshot
curl -X PUT "localhost:9200/_snapshot/backups/snapshot_1?wait_for_completion=true"

# Valkey backup
docker compose exec valkey valkey-cli BGSAVE

# Docker vulnerability scanning (local)
docker scan codingbrain/mcp:latest
trivy image codingbrain/mcp:latest
```

---

## 3. Architecture Patterns

### Backup Verification Pattern

```python
# scripts/verify_backup.py
class BackupVerifier:
    def check_exists(self, path: str) -> bool:
        """Return True if backup file exists."""
        return Path(path).exists()

    def check_freshness(self, path: str, max_age_hours: int = 24) -> bool:
        """Return True if backup modified within max_age_hours."""
        mtime = Path(path).stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        return age_hours <= max_age_hours

    def validate_integrity(self, path: str, backup_type: str) -> bool:
        """Validate backup format based on type."""
        # PostgreSQL: check pg_restore --list works
        # Neo4j: check dump file header
        # Qdrant: verify JSON structure
        pass

    def send_alert(self, error: str) -> None:
        """Send alert on verification failure."""
        # Integrate with alerting system (email, Slack, PagerDuty)
        pass
```

### CI Security Scanning Pattern

```yaml
# .github/workflows/security-scan.yml
jobs:
  trivy-scan:
    steps:
      - uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'codingbrain/mcp:${{ github.sha }}'
          format: 'sarif'
          severity: 'CRITICAL,HIGH'

  dependency-scan:
    steps:
      - run: pip install pip-audit
      - run: pip-audit --requirement requirements.txt

  sast-scan:
    steps:
      - run: pip install bandit
      - run: bandit -r app/ -f sarif -o bandit-results.sarif
```

### Container Hardening Pattern

```dockerfile
# Dockerfile with security hardening
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy with ownership
COPY --chown=appuser:appgroup requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
```

---

## 4. Next Tasks

**STATUS: PHASE 7 COMPLETE** - All tasks completed on 2025-12-28

### Step 1: Backup/Restore Runbooks ✅ COMPLETE

- [x] Create `docs/RUNBOOK-BACKUP-RESTORE.md`
- [x] Document PostgreSQL backup procedures (pg_dump, pg_restore)
- [x] Document Neo4j backup procedures (neo4j-admin dump/load)
- [x] Document Qdrant backup procedures (snapshot API)
- [x] Document OpenSearch backup procedures (snapshot repository)
- [x] Document Valkey backup procedures (BGSAVE, AOF)
- [x] Add backup verification steps
- [x] Commit: `docs: add backup/restore runbook`

### Step 2: Nightly Restore Verification Script ✅ COMPLETE

- [x] Create `openmemory/api/app/backup/verifier.py` (placed in app module, not scripts)
- [x] Write tests first: `openmemory/api/tests/infrastructure/test_backup_verification.py` (31 tests)
- [x] Implement backup existence check
- [x] Implement backup size/freshness validation
- [x] Implement integrity validation (gzip, JSON, PGDMP headers)
- [x] Add alerting on verification failure (logging + webhook support)
- [x] Commit: `feat(backup): add nightly restore verification script`

### Step 3: CI Vulnerability Scanning ✅ COMPLETE

- [x] Add Trivy scanner to CI workflow (`.github/workflows/security-scan.yml`)
- [x] Configure severity thresholds (CRITICAL, HIGH blocking)
- [x] Add dependency scanning (pip-audit)
- [x] Add SAST scanning (bandit for Python)
- [x] Add Dockerfile linting (Hadolint)
- [x] Configure SARIF output for GitHub Security tab
- [x] Commit: `ci: add vulnerability scanning workflow`

### Step 4: Container Hardening ✅ COMPLETE

- [x] Audit `openmemory/api/Dockerfile` for security best practices
- [x] Use non-root user in container (appuser:1000)
- [x] Use COPY --chown for proper file ownership
- [x] Add HEALTHCHECK instruction
- [x] Enhanced .dockerignore to exclude secrets, credentials, dev files
- [x] Write tests: `test_container_hardening.py` (16 tests)
- [x] Commit: `security: harden container image`

### Step 5: Blue-Green/Canary Deployment Playbook ✅ COMPLETE

- [x] Create `docs/RUNBOOK-DEPLOYMENT.md`
- [x] Document blue-green deployment procedure
- [x] Document canary deployment procedure
- [x] Add rollback procedures
- [x] Document health check verification between stages
- [x] Add database migration coordination
- [x] Commit: `docs: add deployment playbook`

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
- Implemented BackupVerifier class with 31 tests (existence, freshness, size, integrity, alerting)
- Added CI security scanning workflow (Trivy, pip-audit, bandit, Hadolint)
- Hardened Dockerfile with non-root user, COPY --chown, HEALTHCHECK
- Created deployment playbook with blue-green and canary procedures

**Previous**: Phase 6 Operability and Resilience

- Added 56 new tests across 6 features
- Implemented health endpoint enhancements with latency measurement
- Added structured JSON logging with correlation IDs
- Implemented OpenTelemetry tracing with OTLP exporter support
- Added Prometheus metrics endpoint with request metrics
- Implemented circuit breakers with state machine (closed/open/half_open)
- Added token bucket rate limiting with per-endpoint configuration

**Result**: Phase 6 complete, 3,327 total tests

---

## 7. Reference Files

**Phase 6 implementations (for reference):**

- `openmemory/api/app/observability/logging.py` - Structured logging
- `openmemory/api/app/observability/tracing.py` - OpenTelemetry
- `openmemory/api/app/observability/metrics.py` - Prometheus metrics
- `openmemory/api/app/resilience/circuit_breaker.py` - Circuit breakers
- `openmemory/api/app/security/rate_limit.py` - Rate limiting

**Existing backup patterns:**

- `openmemory/api/app/routers/backup.py` - Backup endpoints (user-level backup)
- `openmemory/api/app/alembic/utils.py` - Migration utilities with backup validation

**Docker configuration:**

- `openmemory/docker-compose.yml` - Service definitions
- `openmemory/api/Dockerfile` - Container build (to be hardened)

---

## 8. Commit Template

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(scope): description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## 9. Success Criteria Checklist

Phase 7 is complete when ALL are checked:

- [ ] Backup/restore runbook with procedures for all data stores
- [ ] Nightly restore verification script with alerting
- [ ] CI vulnerability scanning (Trivy, pip-audit, bandit)
- [ ] Container hardening (non-root, capabilities dropped)
- [ ] Blue-green/canary deployment playbook
- [ ] All procedures tested and documented

**Target**: Documentation-heavy phase with ~10-20 new tests for verification scripts

---

## 10. Phase Transition

When Phase 7 is complete:

1. **Verify completion:**
   - All checkboxes in Section 4 are checked
   - All tests passing: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`
   - Security scan workflow runs successfully

2. **Update Progress file:**
   - Mark Phase 7 as ✅ in header
   - Fill in all commit hashes
   - Add final Daily Log entry

3. **Create next continuation prompt:**
   - Copy template to `docs/CONTINUATION-PROMPT-PHASE8-SCALEOUT.md`
   - Phase 8 covers: Hot/cold graph partitioning, read replicas + routing (currently deferred)

4. **Commit together:**

```bash
git add docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md docs/CONTINUATION-PROMPT-PHASE8-SCALEOUT.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "$(cat <<'EOF'
docs: complete Phase 7 DR and Hardening

- Backup/restore runbooks for all 5 data stores
- Nightly verification script with alerting
- CI vulnerability scanning workflow
- Container hardening with non-root user
- Deployment playbooks for blue-green/canary

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```
