# PRD: Phase 6 - Operability and Resilience

**Plan Reference**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress Tracker**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Continuation Prompt**: `docs/CONTINUATION-PROMPT-PHASE6-OPERABILITY.md`
**Style**: STRICT TDD - Write failing tests first, then implement

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Success Criteria

Phase 6 is complete when all of the following are met:

1. [x] `/health/live` returns 200 with minimal latency (< 10ms)
2. [ ] `/health/ready` checks PostgreSQL and Neo4j connectivity
3. [ ] `/health/deps` checks all 5 dependencies (PostgreSQL, Neo4j, Qdrant, OpenSearch, Valkey)
4. [ ] Circuit breakers wrap Neo4j, Qdrant, OpenSearch, and reranker calls
5. [ ] Degraded-mode response schema returns partial results when dependencies fail
6. [ ] Rate limiting enforces per-endpoint and per-org quotas
7. [ ] X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers on responses
8. [ ] OpenTelemetry instrumentation propagates trace IDs
9. [ ] Structured JSON logging with correlation IDs
10. [ ] Prometheus metrics endpoint exposes request counts, latencies, and error rates

### 1.2 Edge Cases

**Health Endpoints:**
- Dependency timeout (check should timeout before HTTP timeout)
- Partial dependency failure (some healthy, some not)
- Connection pool exhaustion
- Network partition during health check

**Circuit Breakers:**
- Rapid successive failures should open circuit quickly
- Circuit in half-open state should allow probe requests
- Multiple concurrent failures shouldn't race on state transition
- Recovery after extended outage

**Rate Limiting:**
- Burst traffic exceeding limit
- Multiple clients from same org hitting limits
- Rate limit reset at window boundary
- Clock skew between distributed rate limiters

**OpenTelemetry:**
- Missing trace context in incoming request
- Span propagation across async boundaries
- Trace sampling under high load

### 1.3 Test Suite Structure

```
openmemory/api/tests/
├── infrastructure/
│   ├── test_health_endpoints.py      # EXISTING - enhance
│   ├── test_circuit_breakers.py      # NEW
│   └── test_rate_limiting.py         # NEW
├── observability/
│   ├── test_otel_instrumentation.py  # NEW
│   ├── test_structured_logging.py    # NEW
│   └── test_metrics.py               # NEW
└── routers/
    └── test_health.py                # Router-level tests
```

### 1.4 Test Specifications

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| **Health Endpoints** ||||
| /health/live | Unit | Returns 200 with status ok | `{"status": "ok"}` |
| /health/live | Unit | Response time < 10ms | Latency assertion |
| /health/ready | Unit | Returns 200 when DB connected | Status 200 |
| /health/ready | Unit | Returns 503 when DB unavailable | Status 503, `{"status": "degraded"}` |
| /health/deps | Unit | Reports all 5 dependencies | All services listed |
| /health/deps | Unit | Individual dep failure reported | Failed dep status = "unavailable" |
| /health/deps | Unit | Timeout per dependency check | < 2s per check |
| **Circuit Breakers** ||||
| Neo4j CB | Unit | Opens after 5 failures | Circuit state = open |
| Neo4j CB | Unit | Half-open after 30s | Allows single probe |
| Neo4j CB | Unit | Closes after successful probe | Circuit state = closed |
| Qdrant CB | Unit | Opens after 5 failures | Circuit state = open |
| OpenSearch CB | Unit | Opens after 5 failures | Circuit state = open |
| Degraded Mode | Integration | Returns partial results when Qdrant down | Status 200, degraded flag |
| **Rate Limiting** ||||
| Per-endpoint | Unit | Returns 429 when limit exceeded | Status 429 |
| Per-endpoint | Unit | X-RateLimit headers present | All 3 headers |
| Per-org | Unit | Org quota shared across endpoints | Quota consumed |
| Burst | Unit | Token bucket allows bursts | First N requests pass |
| Reset | Unit | Limit resets at window boundary | Counter resets |
| **OpenTelemetry** ||||
| Trace context | Unit | Propagates traceparent header | Trace ID matches |
| Span creation | Unit | Request creates parent span | Span exported |
| Async spans | Unit | Async calls linked to parent | Parent-child relationship |
| **Logging** ||||
| JSON format | Unit | Log entries are valid JSON | JSON parseable |
| Correlation ID | Unit | Request ID in all log entries | request_id field present |
| Trace context | Unit | Trace ID in log entries | trace_id field present |
| **Metrics** ||||
| /metrics | Unit | Prometheus format response | Content-type text/plain |
| Request count | Unit | Counter increments per request | http_requests_total++ |
| Latency | Unit | Histogram captures latency | http_request_duration_seconds |
| Error rate | Unit | Error counter increments on 5xx | http_errors_total++ |

---

## PHASE 2: Feature Specifications

### Feature 1: Health Endpoints Enhancement

**Description**: Enhance existing `/health/live`, `/health/ready`, `/health/deps` endpoints with proper dependency checks, timeouts, and structured responses.

**Dependencies**:
- Existing `app/routers/health.py`
- Database engine from `app/database.py`
- Neo4j client from `app/graph/neo4j_client.py`
- Store health checks from `app/stores/*.py`

**Test Cases** (write these first):
- [ ] Unit test: `/health/live` returns 200 in < 10ms
- [ ] Unit test: `/health/ready` returns 200 when PostgreSQL connected
- [ ] Unit test: `/health/ready` returns 503 when PostgreSQL unavailable
- [ ] Unit test: `/health/deps` returns status for all 5 dependencies
- [ ] Unit test: Individual dependency timeout doesn't block others
- [ ] Integration test: Health endpoints work with real containers

**Implementation Approach**:
1. Add async health check functions for each dependency with timeout
2. Use `asyncio.wait_for()` with 2-second timeout per check
3. Aggregate results with overall status calculation
4. Return 503 if any critical dependency (PostgreSQL, Neo4j) is down

**Response Schemas**:

```python
class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

class DependencyHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: HealthStatus
    timestamp: datetime
    dependencies: Optional[Dict[str, DependencyHealth]] = None
```

**Git Commit Message**: `feat(health): add comprehensive dependency health checks`

---

### Feature 2: Circuit Breakers

**Description**: Wrap external service calls (Neo4j, Qdrant, OpenSearch, reranker) with circuit breakers to prevent cascading failures.

**Dependencies**:
- `circuitbreaker` package (add to pyproject.toml)
- Store implementations in `app/stores/`
- Graph client in `app/graph/neo4j_client.py`

**Test Cases** (write these first):
- [ ] Unit test: Circuit opens after 5 consecutive failures
- [ ] Unit test: Circuit stays open for recovery_timeout (30s)
- [ ] Unit test: Half-open circuit allows single probe request
- [ ] Unit test: Successful probe closes circuit
- [ ] Unit test: Failed probe re-opens circuit
- [ ] Unit test: Circuit breaker state persists across requests
- [ ] Integration test: Service recovers after circuit closes

**Implementation Approach**:
1. Create `app/resilience/circuit_breaker.py` with configured breakers
2. Define breaker configs per service (threshold, timeout, expected exceptions)
3. Wrap store methods with `@circuit` decorator
4. Add degraded mode response when circuit is open

**Circuit Breaker Configuration**:

```python
from circuitbreaker import circuit

# Per-service configuration
CIRCUIT_CONFIGS = {
    "neo4j": {"failure_threshold": 5, "recovery_timeout": 30, "expected_exception": Neo4jError},
    "qdrant": {"failure_threshold": 5, "recovery_timeout": 30, "expected_exception": QdrantException},
    "opensearch": {"failure_threshold": 5, "recovery_timeout": 30, "expected_exception": OpenSearchException},
    "reranker": {"failure_threshold": 3, "recovery_timeout": 60, "expected_exception": RerankerError},
}

@circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Neo4jError)
async def execute_neo4j_query(query: str, params: dict):
    ...
```

**Degraded Response Schema**:

```python
class DegradedResponse(BaseModel):
    status: Literal["degraded"] = "degraded"
    message: str
    available_sources: List[str]
    unavailable_sources: List[str]
    partial_results: Optional[Any] = None
```

**Git Commit Message**: `feat(resilience): add circuit breakers for external services`

---

### Feature 3: Rate Limiting

**Description**: Implement token bucket rate limiting with per-endpoint and per-org quotas, backed by Valkey for distributed coordination.

**Dependencies**:
- Valkey client for distributed state
- `app/security/` for Principal injection

**Test Cases** (write these first):
- [ ] Unit test: Request within limit returns 200
- [ ] Unit test: Request exceeding limit returns 429
- [ ] Unit test: X-RateLimit-Limit header shows configured limit
- [ ] Unit test: X-RateLimit-Remaining header decrements
- [ ] Unit test: X-RateLimit-Reset header shows window end
- [ ] Unit test: Different endpoints have different limits
- [ ] Unit test: Org quota shared across all endpoints
- [ ] Unit test: Burst tokens allow temporary spike
- [ ] Integration test: Rate limit works across multiple processes

**Implementation Approach**:
1. Create `app/security/rate_limit.py` with `RateLimiter` class
2. Implement token bucket algorithm with Valkey backend
3. Create FastAPI middleware for rate limit enforcement
4. Add response headers via middleware
5. Configure per-endpoint limits in settings

**Rate Limit Configuration**:

```python
@dataclass
class RateLimitConfig:
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20

ENDPOINT_LIMITS = {
    "/v1/memories": RateLimitConfig(requests_per_minute=60),
    "/v1/search": RateLimitConfig(requests_per_minute=30),
    "/v1/graph": RateLimitConfig(requests_per_minute=20),
    "default": RateLimitConfig(requests_per_minute=100),
}

ORG_QUOTAS = {
    "free": RateLimitConfig(requests_per_hour=1000),
    "pro": RateLimitConfig(requests_per_hour=10000),
    "enterprise": RateLimitConfig(requests_per_hour=100000),
}
```

**Response Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1703808000
```

**Git Commit Message**: `feat(security): add token bucket rate limiting with Valkey backend`

---

### Feature 4: OpenTelemetry Instrumentation

**Description**: Add distributed tracing with OpenTelemetry, propagating trace context through all service calls.

**Dependencies**:
- `opentelemetry-api`, `opentelemetry-sdk` packages
- `opentelemetry-instrumentation-fastapi` for auto-instrumentation
- `opentelemetry-exporter-otlp` for trace export

**Test Cases** (write these first):
- [ ] Unit test: Incoming traceparent header is propagated
- [ ] Unit test: Request creates span with correct attributes
- [ ] Unit test: Nested async calls create child spans
- [ ] Unit test: Span status reflects HTTP status code
- [ ] Unit test: Exception sets span status to error
- [ ] Unit test: Trace ID available in request state
- [ ] Integration test: Traces exported to collector

**Implementation Approach**:
1. Create `app/observability/tracing.py` with OTel setup
2. Configure TracerProvider with OTLP exporter
3. Add FastAPIInstrumentor to main.py
4. Create utility to get current trace ID for logging
5. Configure sampling for production (10% of requests)

**Tracing Configuration**:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def setup_tracing(app: FastAPI, service_name: str = "codingbrain-api"):
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )

    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
```

**Git Commit Message**: `feat(observability): add OpenTelemetry distributed tracing`

---

### Feature 5: Structured Logging

**Description**: Configure structured JSON logging with correlation IDs and trace context for all log entries.

**Dependencies**:
- Python `logging` module
- `python-json-logger` for JSON formatting
- OpenTelemetry trace context

**Test Cases** (write these first):
- [ ] Unit test: Log entries are valid JSON
- [ ] Unit test: Log entry includes timestamp in ISO format
- [ ] Unit test: Log entry includes log level
- [ ] Unit test: Log entry includes request_id when in request context
- [ ] Unit test: Log entry includes trace_id when tracing enabled
- [ ] Unit test: Log entry includes span_id when tracing enabled
- [ ] Unit test: Exception includes stack trace in structured field
- [ ] Unit test: Sensitive fields are redacted (password, token)

**Implementation Approach**:
1. Create `app/observability/logging.py` with JSON formatter
2. Add middleware to inject request_id into log context
3. Configure root logger with JSON handler
4. Add utility to get logger with context
5. Configure log levels per module via settings

**Logging Configuration**:

```python
import logging
from pythonjsonlogger import jsonlogger

class CorrelatedJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        # Add trace context if available
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            log_record["trace_id"] = format(ctx.trace_id, "032x")
            log_record["span_id"] = format(ctx.span_id, "016x")

        # Add request context if available
        if hasattr(_request_context, "request_id"):
            log_record["request_id"] = _request_context.request_id
```

**Git Commit Message**: `feat(observability): add structured JSON logging with correlation`

---

### Feature 6: Prometheus Metrics

**Description**: Expose Prometheus metrics endpoint with request counts, latencies, and error rates for SLO monitoring.

**Dependencies**:
- `prometheus-client` package
- FastAPI middleware for metric collection

**Test Cases** (write these first):
- [ ] Unit test: /metrics returns Prometheus format
- [ ] Unit test: http_requests_total counter increments
- [ ] Unit test: http_request_duration_seconds histogram updated
- [ ] Unit test: Metrics include endpoint label
- [ ] Unit test: Metrics include status_code label
- [ ] Unit test: Custom business metrics can be registered
- [ ] Integration test: Metrics scraped by Prometheus

**Implementation Approach**:
1. Create `app/observability/metrics.py` with metric definitions
2. Add middleware to record request metrics
3. Create `/metrics` endpoint with Prometheus format
4. Define SLO-relevant metrics (P95 latency, error rate)
5. Add custom metrics for business events

**Metric Definitions**:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Dependency health metrics
dependency_health = Gauge(
    "dependency_health",
    "Dependency health status (1=healthy, 0=unhealthy)",
    ["dependency"]
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 0.5=half-open)",
    ["service"]
)

# Business metrics
memories_created_total = Counter(
    "memories_created_total",
    "Total memories created",
    ["org_id", "vault"]
)

search_queries_total = Counter(
    "search_queries_total",
    "Total search queries",
    ["org_id", "search_type"]
)
```

**Git Commit Message**: `feat(observability): add Prometheus metrics endpoint`

---

## PHASE 3: Development Protocol

### Execution Order

Execute features in this order to minimize dependencies:

1. **Feature 1: Health Endpoints** (no external deps)
2. **Feature 5: Structured Logging** (foundation for debugging)
3. **Feature 4: OpenTelemetry** (builds on logging)
4. **Feature 6: Prometheus Metrics** (parallel to OTel)
5. **Feature 2: Circuit Breakers** (needs metrics)
6. **Feature 3: Rate Limiting** (needs metrics + Valkey)

### The Recursive Testing Loop

For EVERY feature:

```
1. WRITE TESTS FIRST
   └── Create failing tests in appropriate test file

2. IMPLEMENT FEATURE
   └── Write minimum code to pass tests

3. RUN ALL TESTS
   docker compose exec codingbrain-mcp pytest tests/ -v

4. ON PASS:
   ├── git add -A
   ├── git commit -m "feat(scope): description"
   └── Update IMPLEMENTATION-PROGRESS-PROD-READINESS.md

5. ON FAIL:
   ├── Debug and fix
   └── Return to step 3

6. REGRESSION VERIFICATION
   └── Ensure all previous tests still pass
```

### Dependencies to Add

Add to `openmemory/api/pyproject.toml`:

```toml
[project.dependencies]
circuitbreaker = "^2.0.0"
opentelemetry-api = "^1.20.0"
opentelemetry-sdk = "^1.20.0"
opentelemetry-instrumentation-fastapi = "^0.41b0"
opentelemetry-exporter-otlp = "^1.20.0"
prometheus-client = "^0.19.0"
python-json-logger = "^2.0.7"
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-28
**Current Phase**: Phase 6 - PRD Creation
**Last Action**: Created PRD document

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | Health Endpoints | [ ] | [ ] | [ ] | |
| 2 | Circuit Breakers | [ ] | [ ] | [ ] | |
| 3 | Rate Limiting | [ ] | [ ] | [ ] | |
| 4 | OpenTelemetry | [ ] | [ ] | [ ] | |
| 5 | Structured Logging | [ ] | [ ] | [ ] | |
| 6 | Prometheus Metrics | [ ] | [ ] | [ ] | |

### Decisions Made

1. **Decision**: Use token bucket for rate limiting (not sliding window)
   - **Rationale**: Better burst handling, simpler Valkey implementation
   - **Alternatives Considered**: Sliding window log, fixed window counter

2. **Decision**: OTLP exporter over Jaeger/Zipkin native
   - **Rationale**: OTLP is the standard, supports all backends
   - **Alternatives Considered**: Jaeger exporter, Zipkin exporter

3. **Decision**: python-json-logger over structlog
   - **Rationale**: Simpler, lighter weight, sufficient for needs
   - **Alternatives Considered**: structlog, loguru

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Health router patterns | Existing /health/live, /health/ready, /health/deps with 5 dependency checks |
| Explore | Store client patterns | Factory functions return None on unavailable; health_check() methods exist |
| Explore | Test infrastructure | TDD patterns, mock_jwt_config, TestClient fixtures, tenant isolation tests |

### Known Issues & Blockers

- [ ] Issue: MCP SSE auth still pending (Phase 1 blocker)
  - Status: Deferred
  - Impact: None for Phase 6

### Notes for Next Session

> Continue from here in the next session:

- [ ] Add dependencies to pyproject.toml
- [ ] Write tests for Feature 1 (Health Endpoints)
- [ ] Implement health endpoint enhancements
- [ ] Run full test suite to verify no regressions

### Test Results Log

```
# Run before starting implementation:
docker compose exec codingbrain-mcp pytest tests/ -v --tb=short
```

### Recent Git History

```
# Paste `git log --oneline -5` here before starting
```

---

## Execution Checklist

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. Spawn parallel Explore agents to understand codebase
- [x] 3. Complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [x] 5. Write feature specifications (Phase 2)
- [ ] 6. Add dependencies to pyproject.toml
- [ ] 7. For each feature:
  - [ ] Write tests first
  - [ ] Implement feature
  - [ ] Run tests
  - [ ] Commit on green
  - [ ] Run regression tests
  - [ ] Update scratchpad
- [ ] 8. Tag milestone when complete

---

## Quick Reference Commands

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/infrastructure/test_health_endpoints.py -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Check health endpoints
curl http://localhost:8765/health/live
curl http://localhost:8765/health/ready
curl http://localhost:8765/health/deps

# View circuit breaker state (after implementation)
curl http://localhost:8765/metrics | grep circuit_breaker

# Git workflow
git status
git add -A
git commit -m "feat(scope): description"
git log --oneline -5
```

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
