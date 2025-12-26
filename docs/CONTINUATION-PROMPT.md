# Continuation Prompt for Claude Code

Copy and paste one of the prompts below to continue implementation.

---

## Current State (as of 2025-12-26)

**Phase 0: COMPLETE** ✅

**784 tests passing** across Phase 0a benchmarks (337), Phase 0b security (234), and Phase 0c observability (213).

### Phase 0a Completed (DO NOT REDO):

- ✅ MRR metric (17 tests)
- ✅ NDCG metric (19 tests)
- ✅ Latency tracker (20 tests)
- ✅ Embedder adapter interface (22 tests)
- ✅ Lexical decision matrix (33 tests)
- ✅ Concrete adapters: Qwen3, Nomic, Gemini (23 tests)
- ✅ CodeSearchNet dataset loader (37 tests)
- ✅ Lexical backend interface: Tantivy, OpenSearch (66 tests)
- ✅ Benchmark runner (48 tests)
- ✅ Benchmark reporter (52 tests)
- ✅ Benchmark execution script (`run_benchmarks.py`)
- ✅ Baseline metrics collected (`docs/BENCHMARK-RESULTS.md`)

### Phase 0b Completed (DO NOT REDO):

- ✅ JWT validation with OAuth 2.1/PKCE S256 (49 tests)
- ✅ DPoP token binding per RFC 9449 (34 tests)
- ✅ RBAC permission matrix with 7 roles (64 tests)
- ✅ SCIM 2.0 integration stubs (34 tests)
- ✅ Prompt injection defenses (53 tests)

### Phase 0c Completed (DO NOT REDO):

- ✅ OpenTelemetry tracing with GenAI conventions (70 tests)
- ✅ Structured logging with trace correlation (49 tests)
- ✅ Audit hooks for security events (51 tests)
- ✅ SLO tracking with burn rate alerts (43 tests)

### Decisions Made

1. **Production embedding**: qwen3-embedding:8b (MRR: 0.824, NDCG: 0.848)
2. **Development embedding**: nomic-embed-text (20x faster)
3. **Lexical backend**: OpenSearch (better scalability/features)

### Git Commits

Phase 0a:
- `9df4c1e1` feat(benchmarks): add Phase 0a benchmark framework with TDD
- `cac96f6a` feat(benchmarks): add concrete embedder adapters
- `1c4ddc13` feat(benchmarks): add CodeSearchNet dataset loader
- `fcb569d4` feat(benchmarks): add lexical backend interface
- `21df18eb` feat(benchmarks): add benchmark runner
- `43e36e54` feat(benchmarks): add benchmark reporter

Phase 0b:

- `8e611782` feat(security): add Phase 0b security baseline with TDD

Phase 0c:

- Pending commit for observability baseline

---

## NEXT: Phase 1 - Code Indexing Core

```text
Begin Phase 1 code indexing core per docs/IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md.

## Prerequisites

Phase 0 must be complete. Current test count: 784 tests passing.

### Phase 1 Tasks (per section 6 & 7)

1. AST parsing with Tree-sitter (Python first)
2. Incremental indexer with Merkle tree
3. SCIP symbol ID format
4. Code embeddings pipeline
5. CODE_* graph projection in Neo4j
6. Data-flow edges (READS, WRITES, DATA_FLOWS_TO)

### First Task: AST Parsing (Python)

**Location**: openmemory/api/indexing/

**TDD Steps**:
1. Create directory: openmemory/api/indexing/
2. Write tests in: openmemory/api/indexing/tests/test_ast_parser.py
3. Tests should cover:
   - Parse Python file into AST
   - Extract symbols: functions, classes, methods, imports
   - Extract symbol properties: name, signature, docstring, line numbers
   - Handle malformed/partial files gracefully (skip with logging)
   - Parse error rate tracking
   - Language plugin interface
4. Implement: openmemory/api/indexing/ast_parser.py
5. Write tests for incremental indexing with Merkle tree

### Key Requirements (from v7 plan section 7)

- Tree-sitter for AST parsing with language plugins
- Parser resilience: try/catch, skip malformed files, log errors
- Track parse_error_rate metric
- Partial indexing for recoverable files
- Language plugin interface: parse(file) -> AST, symbols(AST), references(AST)
- Incremental parsing with ts_tree_edit() for structural sharing
- Symbol IDs follow SCIP format: <scheme> <package> <descriptor>+
- Transactions per commit; atomic delete/insert; rollback on failure

### Key Dependencies

Consider adding to pyproject.toml [project.optional-dependencies.indexing]:
- tree-sitter>=0.21.0
- tree-sitter-python>=0.21.0

### Exit Criteria (from v7 plan section 16)

- Parser error rate <= 2% on target repos
- Indexing success >= 99%
- CodeSearchNet MRR >= 0.70 (target 0.75+)
```

---

## Development Practices (ALWAYS FOLLOW)

1. **ALWAYS write tests FIRST** - Never implement code before tests exist
2. **Track progress** in docs/IMPLEMENTATION-PROGRESS.md after each milestone
3. **Use TodoWrite tool** to track all tasks
4. **Commit frequently** with descriptive messages
5. **Use subagents** for exploration and parallel work
6. **Mock external dependencies** in unit tests
7. **Mark integration tests** with @pytest.mark.integration

## Running Tests

```bash
# Run all Phase 0 tests
.venv/bin/python -m pytest openmemory/api/benchmarks/ openmemory/api/security/tests/ openmemory/api/observability/tests/ -v

# Run only observability tests
.venv/bin/python -m pytest openmemory/api/observability/tests/ -v

# Run only security tests
.venv/bin/python -m pytest openmemory/api/security/tests/ -v

# Run only benchmark tests (excluding integration)
.venv/bin/python -m pytest openmemory/api/benchmarks/ -v -m "not integration"

# Quick test count
.venv/bin/python -m pytest openmemory/api/benchmarks/ openmemory/api/security/tests/ openmemory/api/observability/tests/ -v 2>&1 | tail -3
```

## Key Thresholds (from v7 plan)

- MRR >= 0.75 for production readiness ✅ (achieved 0.824)
- NDCG@10 >= 0.80 ✅ (achieved 0.848)
- Parser error rate <= 2%
- Indexing success >= 99%
- SLO: fast burn 2x over 1h, slow burn 4x over 6h
- Error budget freeze at 50% burn

## Directory Structure

```text
openmemory/api/
├── benchmarks/               # ✅ Phase 0a complete (337 tests)
│   ├── embeddings/
│   ├── lexical/
│   ├── runner/
│   ├── reporter/
│   └── run_benchmarks.py
├── security/                 # ✅ Phase 0b complete (234 tests)
│   ├── __init__.py
│   ├── jwt_validator.py      # OAuth 2.1/PKCE S256
│   ├── dpop_validator.py     # RFC 9449 DPoP
│   ├── rbac.py               # Role permission matrix
│   ├── scim.py               # SCIM 2.0 stubs
│   ├── prompt_injection.py   # Pattern library, risk scoring
│   └── tests/
├── observability/            # ✅ Phase 0c complete (213 tests)
│   ├── __init__.py
│   ├── tracing.py            # OpenTelemetry with GenAI conventions
│   ├── logging.py            # Structured logging with trace correlation
│   ├── audit.py              # Audit hooks with hash chaining
│   ├── slo.py                # SLO tracking with burn rate alerts
│   └── tests/
└── indexing/                 # 🎯 Phase 1 (next)
    ├── __init__.py
    ├── ast_parser.py         # Tree-sitter AST parsing
    ├── merkle_tree.py        # Incremental indexing
    ├── symbol_extractor.py   # SCIP symbol ID format
    ├── embeddings.py         # Code embedding pipeline
    ├── graph_projection.py   # CODE_* Neo4j projection
    └── tests/
```

## Observability Module Reference

The Phase 0c observability module provides:

```python
# Tracing
from openmemory.api.observability.tracing import (
    Tracer, TracingConfig, TracingContext,
    SpanKind, SpanStatus, Span, NoOpSpan,
    create_tracer, get_current_span, get_current_trace_id,
    inject_context, extract_context, trace,
)

# Logging
from openmemory.api.observability.logging import (
    StructuredLogger, LogConfig, LogRecord, LogLevel,
    create_logger, get_logger, bind_context, unbind_context,
)

# Audit
from openmemory.api.observability.audit import (
    AuditLogger, AuditEvent, AuditConfig, AuditEventType,
    AuditStore, MemoryAuditStore, AuditChainVerifier,
    create_audit_logger,
)

# SLO
from openmemory.api.observability.slo import (
    SLOTracker, SLODefinition, SLOBudget, SLOConfig,
    BurnRate, BurnRateWindow, SLOAlert, AlertSeverity,
    create_slo_tracker,
)
```

## Security Module Reference

The Phase 0b security module provides:

```python
# JWT validation
from openmemory.api.security.jwt_validator import JWTValidator, PKCEValidator
from openmemory.api.security.jwt_validator import JWTValidationError, JWTExpiredError

# DPoP token binding
from openmemory.api.security.dpop_validator import DPoPValidator, create_dpop_proof
from openmemory.api.security.dpop_validator import DPoPValidationError, DPoPReplayError

# RBAC
from openmemory.api.security.rbac import RBACEnforcer, Principal, Role, Permission, Scope
from openmemory.api.security.rbac import PermissionDeniedError, ScopeMismatchError

# SCIM
from openmemory.api.security.scim import SCIMService, SCIMUser, UserStatus

# Prompt injection
from openmemory.api.security.prompt_injection import (
    PromptInjectionDetector,
    InputSanitizer,
    ContextIsolator,
    RiskLevel,
    InjectionType,
)
```
