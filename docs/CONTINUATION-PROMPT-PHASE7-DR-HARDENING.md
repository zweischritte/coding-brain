# Phase 7 Continuation: Deployment, DR, and Hardening

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
2. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress
3. Check Section 4 (Next Tasks) below for what to work on
4. Run the test suite to verify baseline: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update Phase 7 task status table
   - Add entry to Daily Log with date, work completed, and notes
2. Update Section 4 (Next Tasks) with remaining work
3. **Create new continuation prompt** for next session (if moving to Phase 8)
4. Commit all changes:

```bash
git add docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
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

# Check health endpoints
curl http://localhost:8765/health/live
curl http://localhost:8765/health/ready
curl http://localhost:8765/health/deps

# Database backup (PostgreSQL)
docker compose exec postgres pg_dump -U postgres openmemory > backup.sql

# Database restore (PostgreSQL)
docker compose exec -T postgres psql -U postgres openmemory < backup.sql

# Docker vulnerability scanning
docker scan codingbrain-mcp:latest

# Trivy scanning
trivy image codingbrain-mcp:latest
```

---

## 3. TDD Execution Pattern

For EACH feature, follow this exact pattern:

```text
1. CREATE TEST FILE
   └── Write failing tests based on feature specifications

2. RUN TESTS (expect failures)
   └── docker compose exec codingbrain-mcp pytest tests/[new_test_file].py -v

3. IMPLEMENT MINIMUM CODE
   └── Only enough to make tests pass

4. RUN TESTS AGAIN
   └── Verify all new tests pass

5. RUN FULL SUITE
   └── docker compose exec codingbrain-mcp pytest tests/ -v
   └── Ensure no regressions

6. COMMIT ON GREEN
   └── git add -A && git commit -m "feat(scope): description"

7. UPDATE PROGRESS
   └── Update progress file
```

---

## 4. Next Tasks

Execute in this order:

### Step 1: Backup/Restore Runbooks

- [ ] Create `docs/RUNBOOK-BACKUP-RESTORE.md`
- [ ] Document PostgreSQL backup procedures (pg_dump, pg_restore)
- [ ] Document Neo4j backup procedures (neo4j-admin dump/load)
- [ ] Document Qdrant backup procedures (snapshot API)
- [ ] Document OpenSearch backup procedures (snapshot repository)
- [ ] Document Valkey backup procedures (BGSAVE, AOF)
- [ ] Add backup verification steps
- [ ] Commit: `docs: add backup/restore runbook`

### Step 2: Nightly Restore Verification Script

- [ ] Create `scripts/verify-backup.sh` or `scripts/verify_backup.py`
- [ ] Implement backup existence check
- [ ] Implement backup size/freshness validation
- [ ] Implement restore test to ephemeral environment
- [ ] Add data integrity verification (row counts, checksums)
- [ ] Add alerting on verification failure
- [ ] Write tests for verification logic
- [ ] Commit: `feat(backup): add nightly restore verification script`

### Step 3: CI Vulnerability Scanning

- [ ] Add Trivy scanner to CI workflow (`.github/workflows/security-scan.yml`)
- [ ] Configure severity thresholds (CRITICAL, HIGH blocking)
- [ ] Add dependency scanning (pip-audit, safety)
- [ ] Add SAST scanning (bandit for Python)
- [ ] Configure scan caching for performance
- [ ] Document scan results interpretation
- [ ] Commit: `ci: add vulnerability scanning workflow`

### Step 4: Container Hardening

- [ ] Audit Dockerfile for security best practices
- [ ] Use non-root user in container
- [ ] Remove unnecessary packages and tools
- [ ] Add read-only root filesystem where possible
- [ ] Drop unnecessary Linux capabilities
- [ ] Add security context constraints
- [ ] Document hardening decisions
- [ ] Commit: `security: harden container image`

### Step 5: Blue-Green/Canary Deployment Playbook

- [ ] Create `docs/RUNBOOK-DEPLOYMENT.md`
- [ ] Document blue-green deployment procedure
- [ ] Document canary deployment procedure
- [ ] Add rollback procedures
- [ ] Document health check verification between stages
- [ ] Add database migration coordination
- [ ] Document feature flag integration (if applicable)
- [ ] Commit: `docs: add deployment playbook`

---

## 5. Known Issues

1. **MCP SSE auth pending**: SSE endpoints need auth but test_mcp_auth.py hangs; deferred
2. **Pydantic V1 deprecation**: `app/schemas.py:54` uses V1 @validator (non-blocking)
3. **Qdrant version mismatch**: Client 1.16.2 vs server 1.12.5 (non-blocking)
4. **Alert thresholds deferred**: SLO definitions postponed from Phase 6

---

## 6. Last Session Summary (2025-12-28)

**Completed**: Phase 6 Operability and Resilience

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
- `openmemory/api/Dockerfile` - Container build

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

1. Update Progress file - mark Phase 7 as complete
2. Create `docs/CONTINUATION-PROMPT-PHASE8-SCALEOUT.md` for next phase (if needed)
3. Commit all documentation together
4. Phase 8 covers: Hot/cold graph partitioning, read replicas + routing (currently deferred)
