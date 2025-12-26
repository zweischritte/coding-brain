# Continuation Prompt for Claude Code

Copy and paste one of the prompts below to continue implementation.

---

## Current State (as of 2025-12-26)

**285 tests passing** across Phase 0a benchmark framework.

### Completed (DO NOT REDO):
- ✅ MRR metric (17 tests)
- ✅ NDCG metric (19 tests)
- ✅ Latency tracker (20 tests)
- ✅ Embedder adapter interface (22 tests)
- ✅ Lexical decision matrix (33 tests)
- ✅ Concrete adapters: Qwen3, Nomic, Gemini (23 tests)
- ✅ CodeSearchNet dataset loader (37 tests)
- ✅ Lexical backend interface: Tantivy, OpenSearch (66 tests)
- ✅ Benchmark runner (48 tests)

### Git Commits:
- `9df4c1e1` feat(benchmarks): add Phase 0a benchmark framework with TDD
- `cac96f6a` feat(benchmarks): add concrete embedder adapters
- `1c4ddc13` feat(benchmarks): add CodeSearchNet dataset loader
- `fcb569d4` feat(benchmarks): add lexical backend interface with Tantivy and OpenSearch adapters
- `21df18eb` feat(benchmarks): add benchmark runner for model and backend comparisons

---

## Next Task: Benchmark Reporter

```
Continue implementing docs/IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md following TDD.

## Current Status
**285 tests passing**. Benchmark runner complete.

### Completed (DO NOT REDO):
- ✅ MRR, NDCG, Latency metrics
- ✅ Embedder adapters (Qwen3, Nomic, Gemini)
- ✅ Lexical decision matrix
- ✅ CodeSearchNet dataset loader
- ✅ Lexical backend interface (Tantivy, OpenSearch)
- ✅ Benchmark runner

### Next Task: Benchmark Reporter
**Location**: openmemory/api/benchmarks/reporter/

Create benchmark reporter for generating comparison reports:

**TDD Steps**:
1. Write tests in: openmemory/api/benchmarks/reporter/tests/test_benchmark_reporter.py
2. Tests should cover:
   - BenchmarkReporter class
   - generate_report(results) method
   - Model comparison table generation
   - Winner selection based on thresholds
   - Threshold validation (MRR >= 0.75, NDCG >= 0.80)
   - JSON output format
   - Markdown output format
   - Console-friendly summary output
3. Implement: openmemory/api/benchmarks/reporter/benchmark_reporter.py

### Targets:
- Clear winner identification
- Threshold validation (MRR >= 0.75, NDCG >= 0.80)
- Structured output for automation
```

---

## After Reporter: Run Actual Benchmarks

```
Run the benchmark framework to collect baseline metrics.

## Prerequisites
Benchmark runner and reporter must be complete.

### Tasks:
1. Create a benchmark script: openmemory/api/benchmarks/run_benchmarks.py
2. Run embedding model comparisons on CodeSearchNet sample
3. Run lexical backend evaluation through decision matrix
4. Generate benchmark report
5. Document results in docs/BENCHMARK-RESULTS.md

### Expected Outputs:
- MRR, NDCG, P95 latency per embedding model
- Decision matrix scores for Tantivy vs OpenSearch
- Winner recommendation with rationale
```

---

## Phase 0b: Security Baseline (After 0a Complete)

```
Begin Phase 0b security baseline per docs/IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md.

## Prerequisites
Phase 0a must be complete with all benchmarks passing.

### Phase 0b Tasks:
1. JWT validation with OAuth 2.1 requirements
2. RBAC permission matrix
3. SCIM 2.0 integration stubs
4. Prompt injection defense patterns

### First Task: JWT Validation
**Location**: openmemory/api/security/

**TDD Steps**:
1. Write tests for JWT validation (iss, aud, exp, iat, nbf, sub claims)
2. Write tests for PKCE S256 validation
3. Implement JWTValidator class using Authlib
4. Write tests for DPoP token binding
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
# Run all benchmark tests (excluding integration)
uv run --no-sync pytest openmemory/api/benchmarks/ -v -m "not integration"

# Run specific test file
uv run --no-sync pytest openmemory/api/benchmarks/path/to/test_file.py -v

# Run with coverage
uv run --no-sync pytest openmemory/api/benchmarks/ -v --cov=openmemory/api/benchmarks
```

## Key Thresholds (from v7 plan)

- MRR >= 0.75 for production readiness
- NDCG@10 >= 0.80
- Reranker uplift >= 10% precision
- Latency P95 tracked per model

## Directory Structure

```
openmemory/api/benchmarks/
├── embeddings/
│   ├── adapters/          # Qwen3, Nomic, Gemini adapters
│   ├── datasets/          # CodeSearchNet loader
│   └── metrics/           # MRR, NDCG, Latency
├── lexical/
│   ├── backends/          # Tantivy, OpenSearch adapters
│   ├── decision_matrix/   # Criteria, Evaluator
│   └── tests/
├── runner/                # ✅ Complete (48 tests)
│   ├── tests/
│   ├── benchmark_runner.py
│   └── results.py
└── reporter/              # [NEXT] Report generation
    ├── tests/
    └── benchmark_reporter.py
```
