# Phase 6 Continuation: Operability and Resilience

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**PRD**: `docs/PRD-PHASE6-OPERABILITY-RESILIENCE.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
2. Read `docs/PRD-PHASE6-OPERABILITY-RESILIENCE.md` for full feature specs and test cases
3. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress
4. Check Section 4 (Next Tasks) below for what to work on
5. Run the test suite to verify baseline: `docker compose exec codingbrain-mcp pytest tests/ -v --tb=short`

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update Phase 6 task status table
   - Add entry to Daily Log with date, work completed, and notes
2. Update Section 4 (Next Tasks) with remaining work
3. **Create new continuation prompt** for next session (copy this file and update)
4. Commit all changes:

```bash
git add docs/CONTINUATION-PROMPT-PHASE6-OPERABILITY.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md docs/PRD-PHASE6-OPERABILITY-RESILIENCE.md
git commit -m "docs: update Phase 6 session progress"
```

---

## 1. Current Gaps

**Phase 1 Gap**: MCP SSE authentication not implemented (deferred - requires architecture rework)
**Phase 4.5 Gap**: GDPR compliance not started (SAR export, cascading delete)

---

## 2. Command Reference

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run Phase 6 specific tests
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_health_endpoints.py -v
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_circuit_breakers.py -v
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_rate_limiting.py -v
docker compose exec codingbrain-mcp pytest tests/observability/ -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Check health endpoints manually
curl http://localhost:8765/health/live
curl http://localhost:8765/health/ready
curl http://localhost:8765/health/deps

# Install new dependencies (after adding to pyproject.toml)
docker compose exec codingbrain-mcp pip install -e ".[test]"
```

---

## 3. TDD Execution Pattern

For EACH feature, follow this exact pattern:

```text
1. CREATE TEST FILE
   └── Write failing tests based on PRD test specifications

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
   └── Update PRD scratchpad and progress file
```

---

## 4. Next Tasks

Execute in this order (dependencies flow downward):

### Step 0: Add Dependencies

- [ ] Add new packages to `openmemory/api/pyproject.toml`:

  ```toml
  circuitbreaker = "^2.0.0"
  opentelemetry-api = "^1.20.0"
  opentelemetry-sdk = "^1.20.0"
  opentelemetry-instrumentation-fastapi = "^0.41b0"
  opentelemetry-exporter-otlp = "^1.20.0"
  prometheus-client = "^0.19.0"
  python-json-logger = "^2.0.7"
  ```

- [ ] Rebuild container: `docker compose build codingbrain-mcp`

### Step 1: Health Endpoints Enhancement (7 tests)

- [ ] Write tests in `tests/infrastructure/test_health_endpoints.py` (enhance existing)
- [ ] Add timeout-based async health checks per dependency
- [ ] Add latency measurement to health responses
- [ ] Commit: `feat(health): add comprehensive dependency health checks`

### Step 2: Structured Logging (8 tests)

- [ ] Create `tests/observability/test_structured_logging.py`
- [ ] Create `app/observability/logging.py` with JSON formatter
- [ ] Add request_id middleware
- [ ] Commit: `feat(observability): add structured JSON logging`

### Step 3: OpenTelemetry Instrumentation (7 tests)

- [ ] Create `tests/observability/test_otel_instrumentation.py`
- [ ] Create `app/observability/tracing.py`
- [ ] Add FastAPIInstrumentor to main.py
- [ ] Commit: `feat(observability): add OpenTelemetry distributed tracing`

### Step 4: Prometheus Metrics (7 tests)

- [ ] Create `tests/observability/test_metrics.py`
- [ ] Create `app/observability/metrics.py`
- [ ] Add /metrics endpoint
- [ ] Commit: `feat(observability): add Prometheus metrics endpoint`

### Step 5: Circuit Breakers (7 tests)

- [ ] Create `tests/infrastructure/test_circuit_breakers.py`
- [ ] Create `app/resilience/circuit_breaker.py`
- [ ] Wrap store methods with circuit breakers
- [ ] Add degraded mode response schema
- [ ] Commit: `feat(resilience): add circuit breakers for external services`

### Step 6: Rate Limiting (9 tests)

- [ ] Create `tests/infrastructure/test_rate_limiting.py`
- [ ] Create `app/security/rate_limit.py`
- [ ] Add rate limit middleware
- [ ] Add X-RateLimit-* headers
- [ ] Commit: `feat(security): add token bucket rate limiting`

---

## 5. Known Issues

1. **MCP SSE auth pending**: SSE endpoints need auth but test_mcp_auth.py hangs; deferred
2. **Pydantic V1 deprecation**: `app/schemas.py:54` uses V1 @validator (non-blocking)
3. **Qdrant version mismatch**: Client 1.16.2 vs server 1.12.5 (non-blocking)

---

## 6. Last Session Summary (2025-12-28)

**Completed**: Phase 6 COMPLETE

All 6 features implemented with 56 new tests:

- Step 0: Added dependencies to requirements.txt, rebuilt container
- Step 1: Health endpoints enhancement (6 tests) - latency_ms, timestamps
- Step 2: Structured JSON logging (9 tests) - CorrelatedJsonFormatter, correlation IDs
- Step 3: OpenTelemetry tracing (9 tests) - TracerProvider, OTLP exporter
- Step 4: Prometheus metrics (9 tests) - counters, histograms, /metrics endpoint
- Step 5: Circuit breakers (11 tests) - state machine, DegradedResponse
- Step 6: Rate limiting (12 tests) - token bucket, X-RateLimit headers

**Result**: Phase 6 complete, total 3,327 tests. Proceed to Phase 7.

---

## 7. Reference Files

**PRD with full specs:**

- `docs/PRD-PHASE6-OPERABILITY-RESILIENCE.md` - Complete feature specs, test cases, schemas

**Existing health implementation:**

- `openmemory/api/app/routers/health.py` - Current health endpoints
- `openmemory/api/tests/infrastructure/test_health_endpoints.py` - Existing tests

**Store clients to wrap:**

- `openmemory/api/app/stores/qdrant_store.py` - TenantQdrantStore with health_check()
- `openmemory/api/app/stores/opensearch_store.py` - TenantOpenSearchStore with health_check()
- `openmemory/api/app/stores/episodic_store.py` - ValkeyEpisodicStore with health_check()
- `openmemory/api/app/graph/neo4j_client.py` - Neo4j with existing circuit breaker pattern

**Test patterns:**

- `openmemory/api/tests/routers/test_feedback_router.py` - Router test pattern with auth
- `openmemory/api/tests/stores/conftest.py` - Mock fixtures

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

Phase 6 is complete when ALL are checked:

- [x] `/health/live` returns 200 with < 10ms latency
- [x] `/health/ready` checks PostgreSQL and Neo4j
- [x] `/health/deps` checks all 5 dependencies with individual status
- [x] Circuit breakers on Neo4j, Qdrant, OpenSearch, reranker
- [x] Degraded-mode response schema implemented
- [x] Rate limiting with per-endpoint and per-org quotas
- [x] X-RateLimit-* headers on rate-limited responses
- [x] OpenTelemetry traces with correlation IDs
- [x] Structured JSON logging with trace context
- [x] Prometheus /metrics endpoint with request metrics

**Result**: 56 new tests added, total 3,327 tests

---

## 10. Phase Transition

Phase 6 is COMPLETE. Next steps:

1. ✅ Progress file updated - Phase 6 marked complete
2. ✅ Created `docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md` for next phase
3. Phase 7 covers: Backup/restore, DR runbooks, vulnerability scanning, container hardening

**To continue**: Read `docs/CONTINUATION-PROMPT-PHASE7-DR-HARDENING.md`
