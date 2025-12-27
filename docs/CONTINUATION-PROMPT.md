# Phase 2 Continuation: Configuration and Secrets

**Purpose**: Continue Phase 2 - Configuration and Secrets Management.
**Usage**: Paste this entire prompt to resume implementation exactly where interrupted.
**Development Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## 1. Current State Summary

| Component | Status | Tests |
|-----------|--------|-------|
| Security Core Types | COMPLETE | 32 |
| JWT Validation | COMPLETE | 24 |
| DPoP RFC 9449 | COMPLETE | 16 |
| Security Headers | COMPLETE | 27 |
| Router Auth (all routers) | COMPLETE | 25 |
| MCP Server Auth | COMPLETE | 13+ |

**Phase 1 Complete**: All routers and MCP SSE endpoints require JWT authentication. Tool scope checks implemented for all MCP tools. 99 security tests passing.

---

## 2. Completed Work Registry - DO NOT REDO

### Phase 0b: Security Module
- `openmemory/api/app/security/types.py` - Principal, TokenClaims, Scope, errors
- `openmemory/api/app/security/jwt.py` - JWT validation
- `openmemory/api/app/security/dpop.py` - DPoP RFC 9449 with Valkey replay cache
- `openmemory/api/app/security/dependencies.py` - get_current_principal(), require_scopes()
- `openmemory/api/app/security/middleware.py` - Security headers
- `openmemory/api/app/security/exception_handlers.py` - 401/403 formatting

### Phase 1a: Router Auth - ALL COMPLETE
- `openmemory/api/app/routers/memories.py` - 15+ endpoints, MEMORIES_READ/WRITE/DELETE
- `openmemory/api/app/routers/apps.py` - 5 endpoints, APPS_READ/WRITE
- `openmemory/api/app/routers/graph.py` - 12 endpoints, GRAPH_READ
- `openmemory/api/app/routers/entities.py` - 14 endpoints, ENTITIES_READ/WRITE
- `openmemory/api/app/routers/stats.py` - 1 endpoint, STATS_READ
- `openmemory/api/app/routers/backup.py` - 2 endpoints, BACKUP_READ/WRITE

### Phase 1b: MCP Server Auth - COMPLETE
- `openmemory/api/app/mcp_server.py` - SSE endpoints require JWT
- MCP tools check scopes:
  - add_memories: memories:write
  - search_memory: memories:read
  - list_memories: memories:read
  - delete_memories: memories:delete
  - delete_all_memories: memories:delete
  - update_memory: memories:write

### Infrastructure
- `openmemory/docker-compose.yml` - Project name `codingbrain`, all containers prefixed
- `openmemory/.env` - Complete local dev environment
- `openmemory/api/requirements.txt` - Includes python-jose[cryptography]

---

## 3. Next Task: Configuration and Secrets (Phase 2)

### STEP 1: Analyze Current Configuration

**Use a subagent** to explore current config:

```
Use Task tool with subagent_type=Explore to:
1. Find all config loading patterns in the codebase
2. Identify hardcoded secrets or defaults
3. List environment variables used
4. Check for existing Pydantic settings
```

---

### STEP 2: Write Configuration Tests (TDD Red Phase)

Create `openmemory/api/tests/security/test_config.py`:

```python
"""Tests for configuration and secrets management."""

import pytest
from unittest.mock import patch
import os


class TestSettingsValidation:
    """Test that settings validate required secrets at startup."""

    def test_missing_jwt_secret_fails_fast(self):
        """Settings should raise if JWT_SECRET_KEY is missing."""
        ...

    def test_missing_postgres_password_fails_fast(self):
        """Settings should raise if POSTGRES_PASSWORD is missing."""
        ...

    def test_weak_jwt_secret_rejected(self):
        """JWT secret must be at least 32 characters."""
        ...

    def test_settings_loads_from_env(self):
        """Settings should load all values from environment."""
        ...


class TestSecretRotation:
    """Test secret rotation metadata tracking."""

    def test_jwt_key_rotation_timestamp_tracked(self):
        """JWT key rotation should be tracked with timestamp."""
        ...

    def test_rotation_schedule_validation(self):
        """Validate rotation schedules are enforced."""
        ...
```

Run tests:
```bash
docker compose exec codingbrain-mcp pytest tests/security/test_config.py -v --tb=short
```

**Confirm tests fail.** This is the RED phase.

---

### STEP 3: Implement Pydantic Settings

Create `openmemory/api/app/config/settings.py`:

```python
"""Pydantic settings for all services and secrets."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Required secrets - fail fast if missing
    jwt_secret_key: str
    postgres_password: str
    neo4j_password: str
    openai_api_key: str

    # Database connections
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "codingbrain"
    postgres_user: str = "codingbrain"

    # Neo4j
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_username: str = "neo4j"

    # Valkey/Redis
    valkey_host: str = "valkey"
    valkey_port: int = 6379

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # OpenSearch
    opensearch_hosts: str = "opensearch:9200"

    # JWT settings
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "https://codingbrain.local"
    jwt_audience: str = "https://api.codingbrain.local"
    jwt_expiry_minutes: int = 60

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

---

### STEP 4: Update JWT Module to Use Settings

Modify `openmemory/api/app/security/jwt.py` to use Pydantic settings:

```python
from app.config.settings import get_settings

def validate_jwt(token: str) -> TokenClaims:
    settings = get_settings()
    # Use settings.jwt_secret_key, settings.jwt_algorithm, etc.
    ...
```

---

### STEP 5: Add Startup Validation

Modify `openmemory/api/main.py`:

```python
from app.config.settings import get_settings

@app.on_event("startup")
async def validate_settings():
    """Validate required secrets at startup - fail fast."""
    try:
        settings = get_settings()
        logger.info("Settings validated successfully")
    except Exception as e:
        logger.critical(f"Settings validation failed: {e}")
        raise SystemExit(1)
```

---

### STEP 6: Run Configuration Tests (TDD Green Phase)

```bash
docker compose exec codingbrain-mcp pytest tests/security/test_config.py -v
```

**All tests should pass.**

---

### STEP 7: Run All Security Tests

```bash
docker compose exec codingbrain-mcp pytest tests/security/ -v
```

Ensure no regressions.

---

### STEP 8: Create .env.example Template

Update `openmemory/.env.example`:

```bash
# Coding Brain Configuration
# Copy to .env and fill in required values

# Required Secrets (fail-fast if missing)
JWT_SECRET_KEY=           # Min 32 characters, rotate every 90 days
POSTGRES_PASSWORD=        # Rotate every 180 days
NEO4J_PASSWORD=           # Rotate every 180 days
OPENAI_API_KEY=           # Rotate every 30-90 days

# PostgreSQL
POSTGRES_USER=codingbrain
POSTGRES_DB=codingbrain
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Neo4j
NEO4J_USERNAME=neo4j
NEO4J_URL=bolt://neo4j:7687

# Valkey
VALKEY_HOST=valkey
VALKEY_PORT=6379

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# OpenSearch
OPENSEARCH_HOSTS=opensearch:9200
OPENSEARCH_INITIAL_ADMIN_PASSWORD=

# JWT
JWT_ALGORITHM=HS256
JWT_ISSUER=https://codingbrain.local
JWT_AUDIENCE=https://api.codingbrain.local
JWT_EXPIRY_MINUTES=60

# CORS
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3433

# UI
NEXT_PUBLIC_API_URL=http://localhost:8865
USER=coding-brain-user
```

---

### STEP 9: Document Secret Rotation

Create `docs/SECRET-ROTATION.md`:

```markdown
# Secret Rotation Procedures

## Rotation Schedule

| Secret | Rotation Period | Last Rotated |
|--------|-----------------|--------------|
| JWT_SECRET_KEY | 90 days | YYYY-MM-DD |
| POSTGRES_PASSWORD | 180 days | YYYY-MM-DD |
| NEO4J_PASSWORD | 180 days | YYYY-MM-DD |
| OPENAI_API_KEY | 30-90 days | YYYY-MM-DD |

## Rotation Procedures

### JWT Secret Key
1. Generate new key: `openssl rand -base64 48`
2. Add new key to JWT_SECRET_KEY_NEW in .env
3. Update code to accept both old and new keys
4. Deploy and wait for all old tokens to expire
5. Remove old key from JWT_SECRET_KEY

### Database Passwords
1. Create new password
2. Update in database
3. Update in .env
4. Restart services
```

---

### STEP 10: Commit and Write Next Continuation Prompt

Commit with message:
```
feat(config): add Pydantic settings with secret validation
```

Write next continuation prompt for Phase 3 (PostgreSQL Migration).

---

## 4. TDD Workflow - MANDATORY

1. **RED**: Run tests, confirm they fail
2. **GREEN**: Write minimal code to pass tests
3. **REFACTOR**: Clean up while keeping tests green

**NEVER skip the RED phase.** If tests already pass, verify the feature works.

---

## 5. Subagent Usage - RECOMMENDED

```
# Explore configuration patterns
Use Task tool with subagent_type=Explore to find all config loading in the codebase

# Check for hardcoded secrets
Use Task tool with subagent_type=Explore to find hardcoded passwords or keys
```

---

## 6. Exit Gates for Phase 2

| Metric | Threshold |
|--------|-----------|
| test_config.py | All tests pass |
| Settings validated | At startup |
| No hardcoded secrets | 0 remaining |
| .env.example | Complete template |
| SECRET-ROTATION.md | Documented |

---

## 7. Command Reference

```bash
# Configuration tests
docker compose exec codingbrain-mcp pytest tests/security/test_config.py -v

# All security tests
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Validate settings load
docker compose exec codingbrain-mcp python -c "from app.config.settings import get_settings; print(get_settings())"
```

---

## 8. Phase 1 Completion Summary (2025-12-27)

**What was implemented:**
- JWT authentication on all MCP SSE endpoints (/mcp/ and /concepts/)
- Scope checks on all MCP tools (add_memories, search_memory, list_memories, delete_memories, delete_all_memories, update_memory)
- User ID now comes from JWT token, NOT from URL path parameters
- 99 security tests passing

**Files modified:**
- `openmemory/api/app/mcp_server.py` - Added _extract_principal_from_request() and _check_tool_scope() helpers
- `openmemory/docker-compose.yml` - Added `name: codingbrain` to prevent container namespace conflicts
- `openmemory/api/requirements.txt` - Added python-jose[cryptography]
- `openmemory/.env` - Complete local development environment

**Verification:**
```bash
curl http://localhost:8865/mcp/test-client/sse/attacker-user  # Returns 401
curl http://localhost:8865/concepts/test-client/sse/attacker-user  # Returns 401
```
