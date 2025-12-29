# Continuation Prompt: Multi-User Memory Routing - Final Phase

## Objective

Complete the remaining work for multi-user memory routing:

1. **Integration tests** for MCP access_entity enforcement
2. **End-to-end tests** for multi-user scenarios
3. **Validation** that all components work together
4. **Documentation** updates

## Current State (Complete)

### Core Implementation ✅

- `access_entity` validation in `structured_memory.py`
- JWT grants parsing in `security/jwt.py`, `security/types.py`
- `Principal.can_access()` with hierarchy expansion
- Access control helpers in `security/access.py`
- Qdrant store with `access_entity` filtering
- OpenSearch store with `access_entity` filtering
- MCP tools enforcement (add/update/delete/search/list)
- REST endpoints with access_entity checks
- Graph router with `allowed_memory_ids` filtering

### Files Already Modified

```
openmemory/api/app/
├── security/
│   ├── types.py          # Principal.can_access(), TokenClaims.grants
│   ├── jwt.py            # JWT grants parsing
│   └── access.py         # Helper functions
├── stores/
│   ├── qdrant_store.py   # access_entity filtering
│   └── opensearch_store.py # access_entity filtering
├── routers/
│   ├── memories.py       # All endpoints with access checks
│   └── graph.py          # Graph endpoints with allowed_memory_ids
├── mcp_server.py         # MCP tools with access enforcement
└── utils/
    └── structured_memory.py # access_entity validation
```

## Remaining Work

### 1. MCP Integration Tests

Create tests that verify MCP tools enforce access_entity correctly:

```python
# File: openmemory/api/tests/test_mcp_access_entity_integration.py

"""
Integration tests for MCP access_entity enforcement.

Tests the full flow from MCP tool call through to storage,
verifying access control is enforced at each step.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.security.types import Principal, TokenClaims
from datetime import datetime, timezone
import uuid

# Test scenarios to implement:

class TestMCPAddMemoriesAccessControl:
    """Test add_memories enforces access_entity permissions."""

    @pytest.mark.asyncio
    async def test_add_memory_with_valid_access_entity_succeeds(self):
        """User with team grant can create team memory."""
        # Setup principal with team:cloudfactory/backend grant
        # Call add_memories with access_entity=team:cloudfactory/backend
        # Verify memory is created
        pass

    @pytest.mark.asyncio
    async def test_add_memory_without_grant_fails(self):
        """User without grant cannot create team memory."""
        # Setup principal with only user grant
        # Call add_memories with access_entity=team:cloudfactory/backend
        # Verify error is returned
        pass

    @pytest.mark.asyncio
    async def test_add_memory_defaults_to_user_scope(self):
        """Memory without explicit access_entity gets user:<uid>."""
        # Call add_memories without access_entity
        # Verify memory has access_entity=user:<principal.user_id>
        pass

class TestMCPSearchMemoryAccessControl:
    """Test search_memory filters by access_entity."""

    @pytest.mark.asyncio
    async def test_search_returns_only_accessible_memories(self):
        """Search results filtered to accessible memories."""
        # Create memories with different access_entities
        # Search as user with limited grants
        # Verify only accessible memories returned
        pass

    @pytest.mark.asyncio
    async def test_search_with_org_grant_includes_child_scopes(self):
        """Org grant expands to include projects/teams."""
        # Create project and team memories under org
        # Search as user with org grant
        # Verify child scope memories included
        pass

class TestMCPUpdateMemoryAccessControl:
    """Test update_memory checks write access."""

    @pytest.mark.asyncio
    async def test_team_member_can_update_team_memory(self):
        """Team member can update team memory (group-editable)."""
        pass

    @pytest.mark.asyncio
    async def test_non_member_cannot_update_team_memory(self):
        """Non-member cannot update team memory."""
        pass

    @pytest.mark.asyncio
    async def test_cannot_change_access_entity_without_both_grants(self):
        """Changing access_entity requires grants for old AND new."""
        pass

class TestMCPDeleteMemoriesAccessControl:
    """Test delete_memories checks write access."""

    @pytest.mark.asyncio
    async def test_delete_filters_by_access_entity(self):
        """Delete only removes memories user has write access to."""
        pass
```

### 2. End-to-End Multi-User Tests

Create tests that simulate real multi-user scenarios:

```python
# File: openmemory/api/tests/test_multi_user_scenarios.py

"""
End-to-end tests for multi-user memory routing scenarios.

Simulates realistic team collaboration workflows.
"""

class TestTeamCollaborationScenario:
    """Test team members sharing memories."""

    @pytest.mark.asyncio
    async def test_team_memory_visible_to_all_members(self):
        """Memory created with team access_entity visible to all team members."""
        # User A creates memory with access_entity=team:backend
        # User B (team member) can search and find the memory
        # User C (different team) cannot find the memory
        pass

    @pytest.mark.asyncio
    async def test_team_member_can_update_shared_memory(self):
        """Any team member can update team memory."""
        # User A creates team memory
        # User B updates the memory
        # Verify update succeeds
        pass

class TestOrganizationHierarchyScenario:
    """Test org-level access expansion."""

    @pytest.mark.asyncio
    async def test_org_admin_sees_all_org_memories(self):
        """User with org grant sees project/team memories."""
        # Create memories at org, project, team levels
        # User with org grant can see all
        pass

    @pytest.mark.asyncio
    async def test_project_member_sees_project_and_team_memories(self):
        """Project grant includes team memories under project."""
        pass

class TestPersonalVsSharedScenario:
    """Test personal vs shared memory isolation."""

    @pytest.mark.asyncio
    async def test_personal_memories_isolated(self):
        """Personal memories only visible to owner."""
        # User A creates personal memory (scope=user)
        # User B cannot access it even with team grants
        pass

    @pytest.mark.asyncio
    async def test_user_can_share_memory_with_team(self):
        """User can create memory shared with team."""
        # User creates with access_entity=team:X
        # Team members can access
        pass

class TestLegacyMemoryMigrationScenario:
    """Test handling of legacy memories without access_entity."""

    @pytest.mark.asyncio
    async def test_legacy_memory_accessible_to_owner_only(self):
        """Memory without access_entity only visible to owner."""
        pass
```

### 3. Validation Checklist

Run these validations to ensure everything works:

```bash
# 1. Run all tests
cd openmemory/api
python -m pytest tests/ -v --tb=short

# 2. Verify specific test files
python -m pytest tests/test_access_entity_validation.py -v
python -m pytest tests/test_access_entity_filtering.py -v
python -m pytest tests/security/test_jwt_grants.py -v

# 3. Check for import errors
python -c "from app.routers.memories import router"
python -c "from app.routers.graph import router"
python -c "from app.stores.opensearch_store import TenantOpenSearchStore"
python -c "from app.mcp_server import mcp"

# 4. Verify JWT generation with grants
python scripts/generate_jwt.py --user testuser --grants "team:backend,org:cloudfactory"
```

## Subagent Strategy

Use parallel subagents to complete this work efficiently:

### Phase 1: Create Test Files (Parallel)

```
Spawn 2 Explore agents in parallel:
- Agent 1: "Find existing test patterns in openmemory/api/tests/ for MCP tools and async tests"
- Agent 2: "Find existing test fixtures and conftest.py patterns for database/mock setup"
```

### Phase 2: Implement Tests (Parallel)

```
Spawn 3 general-purpose agents in parallel:
- Agent 1: "Create openmemory/api/tests/test_mcp_access_entity_integration.py with full MCP access control tests"
- Agent 2: "Create openmemory/api/tests/test_multi_user_scenarios.py with end-to-end team collaboration tests"
- Agent 3: "Fill in placeholder tests in test_access_entity_filtering.py (TestOpenSearchAccessEntityFiltering, TestGraphQueryAccessEntityFiltering)"
```

### Phase 3: Validation

```
Spawn 1 general-purpose agent:
- "Run all tests and fix any failures. Ensure all 100+ tests pass."
```

## Key Test Helpers

Use these helpers in tests:

```python
# From test_access_entity_filtering.py
def make_claims(sub, org_id="test-org", grants=None, scopes=None):
    return TokenClaims(
        sub=sub,
        iss="https://auth.test.example.com",
        aud="https://api.test.example.com",
        exp=datetime.now(timezone.utc),
        iat=datetime.now(timezone.utc),
        jti=f"jti-{uuid.uuid4()}",
        org_id=org_id,
        scopes=scopes or {"memories:read", "memories:write", "memories:delete"},
        grants=grants or {f"user:{sub}"},
    )

def make_principal(user_id, org_id="test-org", grants=None, scopes=None):
    claims = make_claims(user_id, org_id, grants, scopes)
    return Principal(user_id=user_id, org_id=org_id, claims=claims)
```

## Success Criteria

- [ ] All existing tests still pass (95+ tests)
- [ ] New MCP integration tests pass (10+ tests)
- [ ] New multi-user scenario tests pass (10+ tests)
- [ ] Placeholder tests filled in and passing
- [ ] No import errors in any module
- [ ] Documentation updated with final status

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `tests/test_mcp_access_entity_integration.py` | Create | MCP tool access control tests |
| `tests/test_multi_user_scenarios.py` | Create | E2E multi-user tests |
| `tests/test_access_entity_filtering.py` | Modify | Fill placeholder tests |
| `docs/plans/multi-user-memory-routing-progress.md` | Update | Mark phase complete |

## References

- Progress: `docs/plans/multi-user-memory-routing-progress.md`
- Plan: `docs/plans/multi-user-memory-routing.md`
- Security helpers: `openmemory/api/app/security/access.py`
- Principal model: `openmemory/api/app/security/types.py`
- Existing tests: `openmemory/api/tests/test_access_entity_*.py`

## Execution Instructions

1. Read this prompt completely
2. Spawn parallel Explore agents to understand test patterns
3. Spawn parallel general-purpose agents to create test files
4. Run tests and fix any failures
5. Update progress documentation
6. Commit changes with message: `test: add integration tests for multi-user memory routing`
