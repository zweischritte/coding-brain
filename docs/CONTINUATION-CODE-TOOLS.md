# Continuation Prompt: Code-Intel Tools Exposure (Phase 2)

## Instructions

This is a self-contained continuation prompt for implementing Phase 2 of the PRD. Read this file completely, then proceed with TDD implementation.

**Related Documents:**

- **PRD (Full Requirements):** `docs/PRD-NEXT-STEPS.md` - Contains complete functional requirements, acceptance criteria, and system context
- **Progress Tracker:** `docs/PRD-NEXT-STEPS.md` (Implementation Progress section) - Track completed work and update status

**Before starting:**

1. Read `docs/PRD-NEXT-STEPS.md` for full requirements context
2. Check the Implementation Progress table to understand current state
3. Follow the TDD approach: write tests first, then implement

---

## Project Context

**Repository:** coding-brain (OpenMemory API)
**Working Directory:** `/Users/grischadallmer/git/coding-brain`
**API Location:** `openmemory/api/`

**Key Architecture:**

- FastAPI backend in `openmemory/api/main.py`
- REST routers in `openmemory/api/app/routers/`
- MCP server in `openmemory/api/app/mcp_server.py`
- Code tools (already implemented) in `openmemory/api/tools/`
- Tests in `openmemory/api/tests/`

---

## Phase 1 Completed (Reference)

| Feature | Location | Tests |
|---------|----------|-------|
| AUTO_MIGRATE | `app/database.py:81-164` | `tests/infrastructure/test_auto_migrate.py` |
| /metrics endpoint | `main.py:122-134` | `tests/infrastructure/test_metrics_integration.py` |
| MetricsMiddleware | `main.py:49-50` | (same as above) |
| docker-compose AUTO_MIGRATE | `docker-compose.yml:169-170` | - |

---

## Phase 2: Code-Intel Tools Exposure

### Goal

Expose existing code-intelligence tools (already implemented in `tools/`) via:

1. REST API endpoints at `/api/v1/code/*`
2. MCP tools registered in `mcp_server.py`

### Scope Mapping (from PRD Decisions)

| Tool | REST Scopes | MCP Scope Value |
|------|-------------|-----------------|
| search_code_hybrid | SEARCH_READ | `"search:read"` |
| find_callers | GRAPH_READ | `"graph:read"` |
| find_callees | GRAPH_READ | `"graph:read"` |
| impact_analysis | GRAPH_READ | `"graph:read"` |
| explain_code | SEARCH_READ + GRAPH_READ | both |
| adr_automation | SEARCH_READ + GRAPH_READ | both |
| test_generation | SEARCH_READ + GRAPH_READ | both |
| pr_analysis | SEARCH_READ + GRAPH_READ | both |

---

## Task 1: REST Router (TDD)

### Step 1.1: Write Tests First

Create `openmemory/api/tests/routers/test_code_router.py`:

```python
"""
Tests for Code Intelligence REST Router.

TDD: These tests are written first and should fail until implementation.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestCodeSearchEndpoint:
    """Test POST /api/v1/code/search endpoint."""

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)

    def test_endpoint_exists(self, client):
        """The /api/v1/code/search endpoint must exist."""
        response = client.post("/api/v1/code/search", json={"query": "test"})
        assert response.status_code != 404

    def test_requires_query_parameter(self, client):
        """Search endpoint requires query parameter."""
        response = client.post("/api/v1/code/search", json={})
        assert response.status_code == 422  # Validation error

    def test_returns_json_response(self, client):
        """Search endpoint returns JSON with results and meta."""
        response = client.post("/api/v1/code/search", json={"query": "function"})
        if response.status_code == 200:
            data = response.json()
            assert "results" in data or "hits" in data
            assert "meta" in data


class TestCodeExplainEndpoint:
    """Test POST /api/v1/code/explain endpoint."""
    # ... similar pattern


class TestGracefulDegradation:
    """Test graceful degradation when dependencies unavailable."""

    @pytest.fixture
    def client(self):
        from main import app
        return TestClient(app)

    def test_returns_degraded_response_when_neo4j_unavailable(self, client):
        """Should return 200 with degraded_mode=true when Neo4j down."""
        # Mock Neo4j as unavailable
        with patch("app.routers.code.get_code_toolkit") as mock:
            mock.return_value.neo4j_available = False
            response = client.post("/api/v1/code/callers", json={
                "repo_id": "test",
                "symbol_name": "main"
            })
            # Should not crash - return degraded response
            assert response.status_code in (200, 503)
```

### Step 1.2: Create Pydantic Schemas

Create `openmemory/api/app/schemas/code.py`:

```python
from typing import Optional, List
from pydantic import BaseModel, Field


class CodeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    repo_id: Optional[str] = None
    language: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    seed_symbols: Optional[List[str]] = None


class CallGraphRequest(BaseModel):
    repo_id: str
    symbol_id: Optional[str] = None
    symbol_name: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=5)


class ImpactAnalysisRequest(BaseModel):
    repo_id: str
    changed_files: Optional[List[str]] = None
    symbol_id: Optional[str] = None
    include_cross_language: bool = False
    max_depth: int = Field(default=3, ge=1, le=10)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ... add more schemas per PRD
```

### Step 1.3: Implement Router

Create `openmemory/api/app/routers/code.py`:

```python
"""
Code Intelligence REST API Router.

Exposes code-intel tools via REST endpoints.
All endpoints require appropriate scopes.
"""
import dataclasses
import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from app.schemas.code import (
    CodeSearchRequest, CallGraphRequest, ImpactAnalysisRequest
)
from app.security.dependencies import require_scopes
from app.security.models import Scope

router = APIRouter(prefix="/api/v1/code", tags=["code-intelligence"])


def get_code_toolkit():
    """Lazy-load code toolkit to avoid import-time failures."""
    from app.code_toolkit import get_toolkit
    return get_toolkit()


@router.post("/search")
async def search_code(
    request: CodeSearchRequest,
    _: None = Depends(require_scopes([Scope.SEARCH_READ]))
):
    """
    Tri-hybrid code search.

    Combines lexical, semantic, and graph-based search.
    """
    toolkit = get_code_toolkit()

    if not toolkit.is_available("opensearch"):
        return {
            "results": [],
            "meta": {
                "degraded_mode": True,
                "missing_sources": ["opensearch"]
            }
        }

    try:
        from tools.search_code_hybrid import SearchCodeHybridInput
        input_data = SearchCodeHybridInput(
            query=request.query,
            repo_id=request.repo_id,
            language=request.language,
            limit=request.limit,
            offset=request.offset,
            seed_symbols=request.seed_symbols
        )
        result = toolkit.search_tool.search(input_data)
        return dataclasses.asdict(result)
    except Exception as e:
        return {
            "results": [],
            "meta": {
                "degraded_mode": True,
                "error": str(e)
            }
        }


@router.post("/callers")
async def find_callers(
    request: CallGraphRequest,
    _: None = Depends(require_scopes([Scope.GRAPH_READ]))
):
    """Find functions that call a given symbol."""
    # Similar pattern...


# ... implement remaining endpoints
```

### Step 1.4: Register Router

In `openmemory/api/main.py`, add:

```python
from app.routers import code_router  # Add to imports

# In router section:
app.include_router(code_router)
```

And in `openmemory/api/app/routers/__init__.py`:

```python
from .code import router as code_router
```

---

## Task 2: Code Toolkit Factory

Create `openmemory/api/app/code_toolkit.py`:

```python
"""
Code Intelligence Toolkit Factory.

Provides lazy initialization of code tool dependencies with graceful degradation.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeToolkit:
    """Container for code intelligence dependencies."""
    opensearch_client: Optional[any] = None
    neo4j_driver: Optional[any] = None
    trihybrid_retriever: Optional[any] = None
    embedding_service: Optional[any] = None
    ast_parser: Optional[any] = None

    # Tool instances
    search_tool: Optional[any] = None
    explain_tool: Optional[any] = None
    callers_tool: Optional[any] = None
    callees_tool: Optional[any] = None
    impact_tool: Optional[any] = None

    _available_services: dict = None

    def __post_init__(self):
        self._available_services = {}

    def is_available(self, service: str) -> bool:
        """Check if a service is available."""
        return self._available_services.get(service, False)


@lru_cache(maxsize=1)
def get_toolkit() -> CodeToolkit:
    """
    Lazy-initialize code intelligence dependencies.

    Returns toolkit with available services. Missing services
    are logged but don't prevent initialization.
    """
    toolkit = CodeToolkit()
    toolkit._available_services = {}

    # Try OpenSearch
    try:
        from retrieval.opensearch import create_opensearch_client
        toolkit.opensearch_client = create_opensearch_client(from_env=True)
        toolkit._available_services["opensearch"] = True
        logger.info("OpenSearch client initialized")
    except Exception as e:
        logger.warning(f"OpenSearch unavailable: {e}")
        toolkit._available_services["opensearch"] = False

    # Try Neo4j
    try:
        from app.graph.neo4j_client import get_neo4j_driver
        toolkit.neo4j_driver = get_neo4j_driver()
        toolkit._available_services["neo4j"] = True
        logger.info("Neo4j driver initialized")
    except Exception as e:
        logger.warning(f"Neo4j unavailable: {e}")
        toolkit._available_services["neo4j"] = False

    # Initialize tools if dependencies available
    if toolkit._available_services.get("opensearch"):
        try:
            from tools.search_code_hybrid import create_search_code_hybrid_tool
            toolkit.search_tool = create_search_code_hybrid_tool(
                retriever=toolkit.trihybrid_retriever,
                embedding_service=toolkit.embedding_service
            )
        except Exception as e:
            logger.warning(f"Search tool init failed: {e}")

    # ... initialize other tools

    return toolkit
```

---

## Task 3: MCP Tool Registration

In `openmemory/api/app/mcp_server.py`, add code tools:

```python
# Add after existing tool registrations

@mcp.tool()
async def search_code_hybrid(
    query: str,
    repo_id: str = None,
    language: str = None,
    limit: int = 10
) -> str:
    """
    Search code using tri-hybrid retrieval (lexical + semantic + graph).

    Args:
        query: Search query text
        repo_id: Optional repository filter
        language: Optional language filter
        limit: Maximum results (default 10)

    Returns:
        JSON string with search results and metadata
    """
    _check_tool_scope(Scope.SEARCH_READ.value)

    from app.code_toolkit import get_toolkit
    toolkit = get_toolkit()

    if not toolkit.is_available("opensearch"):
        return json.dumps({
            "results": [],
            "meta": {"degraded_mode": True, "missing_sources": ["opensearch"]}
        })

    try:
        from tools.search_code_hybrid import SearchCodeHybridInput
        input_data = SearchCodeHybridInput(
            query=query,
            repo_id=repo_id,
            language=language,
            limit=limit
        )
        result = toolkit.search_tool.search(input_data)
        return json.dumps(dataclasses.asdict(result))
    except Exception as e:
        return json.dumps({"error": str(e), "meta": {"degraded_mode": True}})


@mcp.tool()
async def find_callers(
    repo_id: str,
    symbol_id: str = None,
    symbol_name: str = None,
    depth: int = 2
) -> str:
    """Find functions that call a given symbol."""
    _check_tool_scope(Scope.GRAPH_READ.value)
    # ... implementation


# ... register remaining tools
```

---

## Execution Checklist

Update `docs/PRD-NEXT-STEPS.md` Implementation Progress as you complete:

- [ ] Read full PRD requirements
- [ ] Write TDD tests: `tests/routers/test_code_router.py`
- [ ] Create schemas: `app/schemas/code.py`
- [ ] Implement router: `app/routers/code.py`
- [ ] Register router in `main.py` and `routers/__init__.py`
- [ ] Write TDD tests: `tests/mcp/test_code_tools_mcp.py`
- [ ] Create toolkit: `app/code_toolkit.py`
- [ ] Register MCP tools in `mcp_server.py`
- [ ] Verify graceful degradation
- [ ] Run full test suite
- [ ] Update PRD-NEXT-STEPS.md progress table
- [ ] Commit with message: `feat(api): expose code-intel tools via REST and MCP (Phase 2)`

---

## Commands Reference

```bash
# Run in openmemory directory
cd openmemory

# Run specific tests
docker compose exec codingbrain-mcp pytest tests/routers/test_code_router.py -v
docker compose exec codingbrain-mcp pytest tests/mcp/test_code_tools_mcp.py -v

# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Check endpoints
curl http://localhost:8865/health/live
curl http://localhost:8865/metrics
curl -X POST http://localhost:8865/api/v1/code/search \
  -H "Content-Type: application/json" \
  -d '{"query": "function"}'
```

---

## Existing Tool Files Reference

| Tool | Implementation | Tests |
|------|---------------|-------|
| search_code_hybrid | `tools/search_code_hybrid.py` | `tools/tests/test_search_code_hybrid.py` |
| explain_code | `tools/explain_code.py` | `tools/tests/test_explain_code.py` |
| find_callers/callees | `tools/call_graph.py` | `tools/tests/test_call_graph_tools.py` |
| impact_analysis | `tools/impact_analysis.py` | `tools/tests/test_impact_analysis.py` |
| adr_automation | `tools/adr_automation.py` | `tools/tests/test_adr_automation.py` |
| test_generation | `tools/test_generation.py` | `tools/tests/test_test_generation.py` |

---

## Error Handling Requirements (from PRD)

- **Neo4j missing**: Return degraded response, do not crash
- **OpenSearch missing**: Return empty results + degraded meta
- **Embedding service missing**: Disable vector query embedding
- **LLM key missing**: Return 503 for ADR/test generation

---

## Session Completion Checklist

**IMPORTANT:** Before ending your session, complete these steps:

### 1. Track Progress

Update `docs/PRD-NEXT-STEPS.md` Implementation Progress table:

```markdown
| Feature | Status | Commit | Notes |
|---------|--------|--------|-------|
| Code tools REST router | COMPLETED | <hash> | `app/routers/code.py` |
| Code tools MCP registration | COMPLETED | <hash> | Added to `mcp_server.py` |
| ... | ... | ... | ... |
```

### 2. Create Next Continuation Prompt

If work remains, create a new continuation prompt file:

- **Filename:** `docs/CONTINUATION-<PHASE-NAME>.md`
- **Format:** Follow this same template structure
- **Content must include:**
  - Instructions section with related document references
  - Project context
  - What was completed (reference commits)
  - What needs to be done (detailed steps)
  - TDD test templates
  - Implementation code templates
  - Execution checklist
  - Commands reference
  - This same "Session Completion Checklist" section

### 3. Commit Changes

```bash
# Stage all changes
git add -A

# Commit with conventional format
git commit -m "$(cat <<'EOF'
feat(api): expose code-intel tools via REST and MCP (Phase 2)

- Add code tools REST router at /api/v1/code/*
- Register code tools as MCP endpoints
- Create CodeToolkit factory for dependency management
- Implement graceful degradation for missing services
- Add TDD tests for router and MCP tools
- Create continuation prompt for Phase 3

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# Verify commit
git log --oneline -1
```

### 4. Summary Output

At end of session, output:

```
## Session Summary

### Completed
- [ item 1 ]
- [ item 2 ]

### Files Modified
- path/to/file1.py
- path/to/file2.py

### Files Created
- path/to/new/file.py

### Commit
<hash> <message>

### Next Steps
See: docs/CONTINUATION-<NEXT-PHASE>.md
```
