"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import asyncio
import contextvars
import datetime
import json
import logging
import uuid
from pathlib import Path

from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
from sqlalchemy import and_, or_
from app.utils.structured_memory import (
    build_structured_memory,
    validate_update_fields,
    apply_metadata_updates,
    validate_text,
    StructuredMemoryError,
    normalize_tags_input,
    normalize_tag_list_input,
    SHARED_SCOPES,
)
from app.utils.code_reference import (
    CodeReference,
    CodeReferenceError,
    validate_code_refs_input,
    serialize_code_refs_to_tags,
    deserialize_code_refs_from_tags,
    format_code_refs_for_response,
    has_code_refs_in_tags,
)
from app.utils.reranking import (
    SearchContext,
    ExclusionFilters,
    compute_boost,
    should_exclude,
    calculate_final_score,
    normalize_tags,
    parse_datetime,
)
from app.utils.response_format import (
    format_search_results,
    format_memory_list,
    format_add_memories_response,
    format_compact_relations,
)
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from app.indexing_jobs import create_index_job, get_index_job, run_index_job
from app.security.access import resolve_access_entity_for_scope
from app.graph.graph_ops import (
    project_memory_to_graph,
    delete_memory_from_graph,
    get_meta_relations_for_memories,
    find_related_memories_in_graph,
    get_memory_subgraph_from_graph,
    aggregate_memories_in_graph,
    tag_cooccurrence_in_graph,
    path_between_entities_in_graph,
    get_graph_relations,
    is_graph_enabled,
    is_mem0_graph_enabled,
    # New edge operations
    update_entity_edges_on_memory_add,
    update_entity_edges_on_memory_delete,
    get_entity_network_from_graph,
    update_tag_edges_on_memory_add,
    get_related_tags_from_graph,
    project_similarity_edges_for_memory,
    delete_similarity_edges_for_memory,
    get_similar_memories_from_graph,
    is_similarity_enabled,
    # Entity normalization operations
    find_duplicate_entities_in_graph,
    normalize_entities_in_graph,
    # Typed relations operations
    get_entity_relations_from_graph,
    # Biographical timeline operations
    get_biography_timeline_from_graph,
)
from app.graph.entity_bridge import bridge_entities_to_om_graph
from app.path_utils import resolve_repo_root
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from app.mcp.sse_transport import SessionAwareSseTransport
from app.security.session_binding import get_session_binding_store

# Load environment variables
load_dotenv()

# Initialize MCP for memory tools
mcp = FastMCP("mem0-mcp-server")

# Initialize separate MCP for Business Concept tools (reduces context window bloat)
concept_mcp = FastMCP("business-concepts-mcp-server")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id, client_name, and org_id
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
org_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("org_id")

# Import security dependencies for JWT validation
try:
    from app.security.jwt import validate_jwt, validate_iat_not_future
    from app.security.types import Principal, AuthenticationError, AuthorizationError, Scope
    HAS_SECURITY = True
except ImportError:
    HAS_SECURITY = False
    logging.warning("Security module not available - MCP endpoints will not require authentication")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Create a separate router for Business Concept endpoints
concept_router = APIRouter(prefix="/concepts")

# Initialize SSE transport for MCP tools (with session ID capture)
sse = SessionAwareSseTransport("/mcp/messages/")


# --- MCP Session Health Endpoint (Phase 2) ---

@mcp_router.get("/health")
async def mcp_session_health():
    """Health check endpoint for MCP session binding store.

    Checks:
    - Session binding store connectivity (memory always ok, Valkey ping)
    - Cleanup scheduler status (if available)

    Returns:
        JSON response with status and details:
        - status: "healthy" or "unhealthy"
        - store_type: "memory" or "valkey"
        - reason: Error description if unhealthy
    """
    from datetime import datetime, timezone
    import time

    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        store = get_session_binding_store()
        store_type = getattr(store, 'STORE_TYPE', 'memory')

        # Check store health
        if hasattr(store, 'health_check'):
            # Valkey store has health_check method
            store_healthy = store.health_check()
        else:
            # Memory store is always healthy if instantiated
            store_healthy = True

        # Check cleanup scheduler status
        try:
            from app.tasks.session_cleanup import _cleanup_scheduler
            scheduler_running = _cleanup_scheduler is not None and _cleanup_scheduler._running
        except Exception:
            scheduler_running = None  # Unknown status

        latency_ms = (time.perf_counter() - start_time) * 1000

        if store_healthy:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "healthy",
                    "store_type": store_type,
                    "scheduler_running": scheduler_running,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": timestamp,
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "store_type": store_type,
                    "reason": "Store health check failed",
                    "scheduler_running": scheduler_running,
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": timestamp,
                }
            )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": str(e),
                "latency_ms": round(latency_ms, 2),
                "timestamp": timestamp,
            }
        )

# Initialize SSE transport for Business Concept tools (with session ID capture)
concept_sse = SessionAwareSseTransport("/concepts/messages/")


def _extract_principal_from_request(request: Request) -> "Principal":
    """
    Extract and validate the authenticated principal from the request.

    Raises:
        AuthenticationError: If authentication fails or is missing
    """
    if not HAS_SECURITY:
        # Security module not available - create a mock principal from path params
        # This is for backwards compatibility during migration
        uid = request.path_params.get("user_id", "anonymous")
        return type('Principal', (), {
            'user_id': uid,
            'org_id': 'default',
            'has_scope': lambda self, scope: True,
            'claims': None,
        })()

    # Extract Authorization header
    auth_header = request.headers.get("authorization", "")
    if not auth_header:
        raise AuthenticationError(
            message="Missing authorization header",
            code="MISSING_AUTH"
        )

    # Parse Bearer token
    if not auth_header.lower().startswith("bearer "):
        raise AuthenticationError(
            message="Invalid authorization header format - expected 'Bearer <token>'",
            code="INVALID_AUTH_FORMAT"
        )

    token = auth_header[7:]  # Remove "Bearer " prefix
    if not token:
        raise AuthenticationError(
            message="Empty authorization token",
            code="EMPTY_TOKEN"
        )

    # Validate the JWT token
    claims = validate_jwt(token)
    # Reject tokens issued in the future (clock skew protection)
    validate_iat_not_future(claims.iat)

    # Build the principal
    principal = Principal(
        user_id=claims.sub,
        org_id=claims.org_id,
        claims=claims,
    )

    return principal


async def _extract_dpop_thumbprint(request: Request, token: str) -> str | None:
    """
    Extract and validate DPoP proof from request, returning thumbprint if valid.

    For POST requests bound to a session with DPoP, this validates:
    - DPoP header is present
    - Proof has correct htm (HTTP method)
    - Proof has correct htu (HTTP URI)
    - Proof signature is valid
    - Returns the JWK thumbprint for session binding validation

    Returns None if no DPoP header is present.

    Raises:
        AuthenticationError: If DPoP header is present but invalid
    """
    dpop_header = request.headers.get("dpop")
    if not dpop_header:
        return None

    try:
        from .security.dpop import DPoPValidator, get_dpop_cache

        cache = await get_dpop_cache()
        if not cache:
            # DPoP cache not configured - skip validation
            return None

        validator = DPoPValidator(cache)

        # Build the full URL for htu validation
        http_uri = str(request.url)
        http_method = request.method

        # Validate the DPoP proof
        await validator.validate(
            dpop_proof=dpop_header,
            http_method=http_method,
            http_uri=http_uri,
            access_token=token,
        )

        # Extract and return the thumbprint
        return await validator.get_thumbprint(dpop_header)
    except Exception as e:
        if isinstance(e, AuthenticationError):
            raise
        raise AuthenticationError(
            message=f"DPoP validation failed: {e}",
            code="INVALID_DPOP"
        )


def _check_scope(principal, scope: "Scope") -> None:
    """
    Check if the principal has the required scope.

    Raises:
        AuthorizationError: If the principal lacks the required scope
    """
    if not HAS_SECURITY:
        return  # Security not available, skip check

    if not principal.has_scope(scope):
        raise AuthorizationError(
            message=f"Required scope '{scope.value}' not granted",
            code="INSUFFICIENT_SCOPE",
        )


# Context variable to store the current principal for tool scope checks
principal_var: contextvars.ContextVar["Principal"] = contextvars.ContextVar("principal")


def _check_tool_scope(required_scope: str) -> str | None:
    """
    Check if the current principal has the required scope for a tool.

    Returns None if scope check passes, or a JSON error string if it fails.
    This is designed for MCP tools which return JSON strings.
    """
    if not HAS_SECURITY:
        return None  # Security not available, skip check

    # Get principal from context var (set by SSE handlers)
    # Fail closed: if no principal, deny access
    principal = principal_var.get(None)
    if not principal:
        return json.dumps({
            "error": "Authentication required",
            "code": "MISSING_AUTH",
        })

    # Check scope
    if not principal.has_scope(required_scope):
        return json.dumps({
            "error": f"Insufficient scope",
            "code": "INSUFFICIENT_SCOPE",
            "required_scope": required_scope,
        })


def _can_access_job(job: dict, admin_scope: str) -> bool:
    """Check whether the current principal can access the job."""
    if not HAS_SECURITY:
        return True
    principal = principal_var.get(None)
    if not principal:
        return False
    if principal.has_scope(admin_scope):
        return True
    requested_by = job.get("requested_by")
    if not requested_by:
        return True
    return requested_by == principal.user_id


def _apply_access_entity_filter(query, principal: "Principal", user):
    from app.security.access import build_access_entity_patterns

    exact_matches, like_patterns = build_access_entity_patterns(principal)
    access_entity_col = Memory.metadata_["access_entity"].as_string()
    legacy_owner_filter = and_(
        or_(Memory.metadata_.is_(None), access_entity_col.is_(None)),
        Memory.user_id == user.id,
    )
    access_clauses = [access_entity_col.in_(exact_matches), legacy_owner_filter]
    access_clauses.extend(access_entity_col.like(pattern) for pattern in like_patterns)
    return query.filter(or_(*access_clauses))


def _get_accessible_memories(db, principal: "Principal | None", user, app, include_archived: bool = True):
    query = db.query(Memory).filter(Memory.state != MemoryState.deleted)
    if not include_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    if principal:
        query = _apply_access_entity_filter(query, principal, user)
    else:
        query = query.filter(Memory.user_id == user.id)

    memories = query.all()

    if principal:
        from app.security.access import can_read_access_entity

        filtered = []
        for memory in memories:
            access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
            if access_entity:
                if can_read_access_entity(principal, access_entity):
                    filtered.append(memory)
            else:
                if memory.user_id == user.id:
                    filtered.append(memory)
        memories = filtered

    return [memory for memory in memories if check_memory_access_permissions(db, memory, app.id)]


def _grant_matches_access_entity(grant: str, access_entity: str) -> bool:
    if not grant or ":" not in grant or not access_entity or ":" not in access_entity:
        return False

    grant_prefix, grant_path = grant.split(":", 1)
    access_prefix, access_path = access_entity.split(":", 1)

    if grant_prefix == "user":
        return access_prefix == "user" and access_path == grant_path

    if grant_prefix == "team":
        return access_prefix == "team" and access_path == grant_path

    if grant_prefix == "project":
        if access_prefix == "project" and access_path == grant_path:
            return True
        return access_prefix == "team" and access_path.startswith(f"{grant_path}/")

    if grant_prefix == "org":
        if access_prefix == "org" and access_path == grant_path:
            return True
        return access_prefix in ("project", "team") and access_path.startswith(f"{grant_path}/")

    return False


def _matching_grants_for_access_entity(principal: "Principal", access_entity: str) -> list[str]:
    grants = set(principal.claims.grants)
    grants.add(f"user:{principal.user_id}")
    return [
        grant for grant in sorted(grants)
        if _grant_matches_access_entity(grant, access_entity)
    ]


def _structured_memory_hint(message: str) -> str | None:
    if "access_entity is required for scope" in message:
        return (
            "Provide access_entity that matches the scope (team:<org>/<team>, "
            "project:<org>/<path>, org:<org>)."
        )
    if "Scope/access_entity mismatch" in message:
        return "Make access_entity prefix match scope (user:, team:, project:, org:)."
    if "Invalid access_entity format" in message:
        return "Use <prefix>:<value> (e.g., user:alice, team:org/team)."
    if "Tags must be a dictionary" in message:
        return "Pass tags as a dict or list of strings (e.g., tags={\"priority\": \"high\"})."
    if "Evidence must be a list" in message:
        return "Pass evidence as a list of strings or a comma-separated string."
    if "remove_tags must be a list" in message:
        return "Pass remove_tags as a list of strings or a comma-separated string."
    return None


def _format_structured_memory_error(error: StructuredMemoryError) -> dict:
    message = str(error)
    payload = {"error": message}
    hint = _structured_memory_hint(message)
    if hint:
        payload["hint"] = hint
    return payload


@mcp.tool(description="""Return the current auth context for this MCP session.

Returns:
- user_id: JWT subject (user id)
- org_id: Organization id from the token
- client_name: MCP client name from the session
- scopes: Granted OAuth scopes
- grants: access_entity grants (includes user:<sub>)
- security_enabled: Whether auth is enforced
""")
async def whoami() -> str:
    principal = principal_var.get(None)
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    org_id = org_id_var.get(None)

    if HAS_SECURITY and not principal:
        return json.dumps({
            "error": "Authentication required",
            "code": "MISSING_AUTH",
        })

    scopes = []
    grants = []
    if principal and principal.claims:
        scopes = sorted(list(principal.claims.scopes))
        grants = sorted(list(principal.claims.grants))
        uid = principal.user_id or uid
        org_id = principal.org_id or org_id

    return json.dumps(
        {
            "user_id": uid,
            "org_id": org_id,
            "client_name": client_name,
            "scopes": scopes,
            "grants": grants,
            "security_enabled": HAS_SECURITY,
        }
    )

@mcp.tool(description="""Return the active embedder configuration and runtime details.

Returns:
- provider: Configured embedder provider (e.g., "gemini")
- model: Embedder model name
- embedding_dims: Configured embedding dimensions
- output_dimensionality: Configured output dims (if set)
- class_name: Runtime embedder class
- vector_store_provider: Vector store provider name
- vector_store_dims: Vector store embedding dimensions (if available)
- memory_client_ready: Whether the memory client is initialized
- error: Error message if the memory client is unavailable
""")
async def get_embedder_info() -> str:
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({
            "memory_client_ready": False,
            "error": "Memory client is not available",
        })

    embedder = getattr(memory_client, "embedding_model", None)
    embedder_config = getattr(embedder, "config", None)

    provider = None
    try:
        provider = getattr(getattr(memory_client.config, "embedder", None), "provider", None)
    except Exception:
        provider = None

    model = getattr(embedder_config, "model", None) if embedder_config else None
    embedding_dims = getattr(embedder_config, "embedding_dims", None) if embedder_config else None
    output_dimensionality = getattr(embedder_config, "output_dimensionality", None) if embedder_config else None

    class_name = None
    if embedder:
        class_name = f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"

    vector_store_provider = None
    vector_store_dims = None
    try:
        vector_store_provider = getattr(getattr(memory_client.config, "vector_store", None), "provider", None)
    except Exception:
        vector_store_provider = None

    if hasattr(getattr(memory_client, "vector_store", None), "embedding_model_dims"):
        vector_store_dims = memory_client.vector_store.embedding_model_dims
    else:
        try:
            vector_store_config = getattr(getattr(memory_client.config, "vector_store", None), "config", None)
            if hasattr(vector_store_config, "embedding_model_dims"):
                vector_store_dims = vector_store_config.embedding_model_dims
            elif isinstance(vector_store_config, dict):
                vector_store_dims = vector_store_config.get("embedding_model_dims")
        except Exception:
            vector_store_dims = None

    return json.dumps({
        "memory_client_ready": True,
        "provider": provider,
        "model": model,
        "embedding_dims": embedding_dims,
        "output_dimensionality": output_dimensionality,
        "class_name": class_name,
        "vector_store_provider": vector_store_provider,
        "vector_store_dims": vector_store_dims,
    })

@mcp.tool(description="""Add a new memory with structured parameters.

Required:
- text: Pure content, no markers
- category: decision | convention | architecture | dependency | workflow | testing | security | performance | runbook | glossary
- scope: session | user | team | project | org | enterprise

Optional metadata:
- artifact_type: repo | service | module | component | api | db | infra | file
- artifact_ref: reference string (repo name, file path, symbol, service)
- entity: team/service/component/person
- source: user (default) or inference
- evidence: List of ADR/PR/issue refs (or comma-separated string)
- tags: Dict with string keys (or list of strings)
- access_entity: Access control entity (required for shared scopes: team, project, org, enterprise)
  Format: <prefix>:<value> where prefix is user, team, project, or org
  Examples: user:grischa, team:cloudfactory/acme/billing/backend, org:cloudfactory
  You may pass access_entity="auto" (or omit it) to default when exactly one matching grant exists.
- code_refs: List of code references linking memory to source code locations
  Each code_ref is a dict with: file_path, line_start, line_end, symbol_id, git_commit, code_hash
  Example: [{"file_path": "/src/auth.ts", "line_start": 42, "line_end": 87, "symbol_id": "AuthService#login"}]
- infer: If true, use LLM extraction to infer memories (default: false)

Examples:
- add_memories(text="Use module boundaries for auth clients", category="architecture", scope="org", artifact_type="module", artifact_ref="auth/clients", entity="Platform", access_entity="org:cloudfactory", tags={"decision": true})
- add_memories(text="Always run python -m pytest before merge", category="workflow", scope="team", entity="Backend", access_entity="team:cloudfactory/backend", evidence=["ADR-014"])
- add_memories(text="createFileUploads uses S3 MultipartUpload", category="architecture", scope="project", entity="StorageService", code_refs=[{"file_path": "/apps/merlin/src/storage.ts", "line_start": 42, "line_end": 87}], access_entity="project:default_org/merlin")
""")
async def add_memories(
    text: str,
    category: str,
    scope: str,
    artifact_type: str = None,
    artifact_ref: str = None,
    entity: str = None,
    source: str = "user",
    evidence: list = None,
    tags: dict = None,
    access_entity: str = None,
    code_refs: list = None,
    infer: bool = False,
) -> str:
    # Check scope - requires memories:write
    scope_error = _check_tool_scope("memories:write")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    principal = principal_var.get(None)
    if access_entity == "auto":
        access_entity = None

    if principal and (access_entity is None):
        resolved_access_entity, options = resolve_access_entity_for_scope(
            principal=principal,
            scope=scope,
            access_entity=access_entity,
        )
        if resolved_access_entity:
            access_entity = resolved_access_entity
        elif scope in SHARED_SCOPES and options:
            return json.dumps({
                "error": f"access_entity is required for scope='{scope}'. Multiple grants available.",
                "code": "ACCESS_ENTITY_AMBIGUOUS",
                "options": options,
                "hint": "Pick one of the available access_entity values.",
            })

    # Default access_entity to user:<uid> for personal scopes
    if access_entity is None and scope in ("user", "session"):
        access_entity = f"user:{uid}"

    # Access control check: verify principal can create memory with this access_entity
    if principal and access_entity:
        from app.security.access import check_create_access
        if not check_create_access(principal, access_entity):
            return json.dumps({
                "error": f"Access denied: you don't have permission to create memories with access_entity='{access_entity}'",
                "code": "FORBIDDEN",
            })

    # Validate and build structured memory
    try:
        clean_text, structured_metadata = build_structured_memory(
            text=text,
            category=category,
            scope=scope,
            artifact_type=artifact_type,
            artifact_ref=artifact_ref,
            entity=entity,
            access_entity=access_entity,
            source=source,
            evidence=evidence,
            tags=tags,
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    # Validate and serialize code_refs into tags (Phase 1: tags-based storage)
    validated_code_refs = []
    if code_refs:
        try:
            validated_code_refs = validate_code_refs_input(code_refs)
            code_ref_tags = serialize_code_refs_to_tags(validated_code_refs)
            # Merge code_ref tags into existing tags
            existing_tags = structured_metadata.get("tags", {})
            if not isinstance(existing_tags, dict):
                existing_tags = {}
            structured_metadata["tags"] = {**existing_tags, **code_ref_tags}
        except CodeReferenceError as e:
            return json.dumps({
                "error": str(e),
                "code": "INVALID_CODE_REFS",
            })

    combined_metadata = {
        "source_app": "openmemory",
        "mcp_client": client_name,
        **structured_metadata,
    }
    if principal:
        combined_metadata["org_id"] = principal.org_id

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({"error": "Memory system is currently unavailable. Please try again later."})

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Check if app is active
            if not app.is_active:
                return json.dumps({"error": f"App {app.name} is currently paused on OpenMemory. Cannot create new memories."})

            response = memory_client.add(
                clean_text,
                user_id=uid,
                metadata=combined_metadata,
                infer=infer,
            )

            # Process the response and update database
            if isinstance(response, dict) and 'results' in response:
                results = response.get('results', [])
                if not results:
                    return json.dumps({
                        "error": "Memory client returned no results",
                        "code": "MEMORY_ADD_EMPTY_RESULT",
                        "hint": "If you expect raw storage, set infer=false",
                    })

                valid_results = [
                    result for result in results
                    if result.get("id") and result.get("memory")
                ]
                invalid_results = [
                    result for result in results
                    if not result.get("id") or not result.get("memory")
                ]
                if not valid_results:
                    return json.dumps({
                        "error": "Memory creation failed",
                        "code": "MEMORY_ADD_FAILED",
                        "details": invalid_results,
                    })

                for result in valid_results:
                    if 'id' not in result:
                        continue

                    memory_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()

                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                metadata_=combined_metadata,
                                state=MemoryState.active,
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']
                            memory.metadata_ = combined_metadata

                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if memory else None,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.UTC)
                            # Create history entry
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted
                            )
                            db.add(history)

                db.commit()

                # Project to Neo4j graph (non-blocking - failures are logged but don't fail the operation)
                logging.info(f"Graph projection: checking {len(results)} results, graph_enabled={is_graph_enabled()}")
                for result in valid_results:
                    logging.info(f"Graph projection: result id={result.get('id')}, event={result.get('event')}")
                    if 'id' in result and result.get('event') == 'ADD':
                        try:
                            logging.info(f"Graph projection: projecting {result['id']}")
                            project_memory_to_graph(
                                memory_id=result['id'],
                                user_id=uid,  # Use string user_id for graph scoping
                                content=result.get('memory', ''),
                                metadata=combined_metadata,
                                state="active",
                            )
                            # Bridge multi-entity extraction from Mem0 to OM graph
                            # This creates OM_ABOUT edges for ALL extracted entities (not just metadata.re)
                            # and typed OM_RELATION edges between related entities
                            if is_mem0_graph_enabled():
                                try:
                                    bridge_result = bridge_entities_to_om_graph(
                                        memory_id=result['id'],
                                        user_id=uid,
                                        content=result.get('memory', ''),
                                        existing_entity=combined_metadata.get('entity'),
                                    )
                                    logging.info(f"Entity bridge: {bridge_result.get('entities_bridged', 0)} entities, "
                                                f"{bridge_result.get('relations_created', 0)} relations")
                                except Exception as bridge_error:
                                    logging.warning(f"Entity bridge failed for {result['id']}: {bridge_error}")
                            # Update entity-to-entity co-mention edges (now works with multi-entity bridging)
                            update_entity_edges_on_memory_add(result['id'], uid)
                            # Update tag-to-tag co-occurrence edges
                            update_tag_edges_on_memory_add(result['id'], uid)
                            # Project similarity edges (K nearest neighbors)
                            project_similarity_edges_for_memory(result['id'], uid)
                        except Exception as graph_error:
                            logging.warning(f"Graph projection failed for {result['id']}: {graph_error}")
                    elif 'id' in result and result.get('event') == 'DELETE':
                        try:
                            # Update entity edges before deleting memory
                            update_entity_edges_on_memory_delete(result['id'], uid)
                            # Delete similarity edges
                            delete_similarity_edges_for_memory(result['id'], uid)
                            # Delete the memory node
                            delete_memory_from_graph(result['id'])
                        except Exception as graph_error:
                            logging.warning(f"Graph deletion failed for {result['id']}: {graph_error}")

                # Format response using the new lean format
                formatted_response = format_add_memories_response(
                    results=valid_results,
                    structured_metadata=structured_metadata,
                )
                if invalid_results:
                    formatted_response["error"] = "Some memories failed to save"
                    formatted_response["failed"] = invalid_results
                return json.dumps(formatted_response)

            # Handle case where response is not in expected format
            return json.dumps({"error": "Unexpected response format from memory client"})
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return json.dumps({"error": f"Error adding to memory: {e}"})


@mcp.tool(description="""Search memories with semantic search and metadata-based re-ranking.

BOOST parameters (category, scope, artifact_type, artifact_ref, entity, tags) influence
ranking but do NOT exclude results. All semantically relevant memories
are returned, with contextually matching ones ranked higher.

RECENCY boost (recency_weight) prioritizes newer memories without excluding
older ones. Use this for "current situation" queries. Set to 0 for
"core pattern" queries where old insights are equally valuable.

DATE FILTER parameters (created_after/before, updated_after/before) are
TRUE FILTERS that exclude results outside the range. Use these for
explicit time windows like "last week" or "Q4 2025".

Parameters:
- query: Search query (required)
- category: Boost memories with this category (decision, convention, etc.)
- scope: Boost memories with this scope (team, project, org, etc.)
- artifact_type: Boost memories for this artifact type (repo, api, file, etc.)
- artifact_ref: Boost memories for this artifact reference
- entity: Boost memories about this entity
- tags: Comma-separated tags to boost (e.g., "trigger,important")
- recency_weight: How much to prioritize recent memories (0.0-1.0)
                  0.0 = off (default), 0.4 = moderate, 0.7 = strong
- recency_halflife_days: Days until recency boost is halved (default: 45)
- created_after: ISO datetime - exclude memories created before this
- created_before: ISO datetime - exclude memories created after this
- updated_after: ISO datetime - exclude memories updated before this
- updated_before: ISO datetime - exclude memories updated after this
- exclude_states: Comma-separated states to exclude (default: "deleted")
- exclude_tags: Comma-separated tags to exclude (e.g., "rejected")
- limit: Max results to return (default: 10, max: 50, for initial searches use 10)
- verbose: If true, return full debug info (scores breakdown, query, filters)
- use_rrf: Enable RRF multi-source fusion (default: true)
- graph_seed_count: Top vector results to use as graph traversal seeds (default: 5)
- auto_route: Enable intelligent query routing based on entity detection (default: true)
- relation_detail: Output verbosity for meta_relations (default: "standard")
  - "none": No meta_relations (minimal tokens)
  - "minimal": Only artifact path + similar memory IDs
  - "standard": + entities + tags + evidence (recommended, ~60% token reduction)
  - "full": Verbose format with all OM_* relations (for debugging)

HYBRID RETRIEVAL: When enabled (use_rrf=true), search combines:
1. Vector similarity (Qdrant embeddings)
2. Graph topology (Neo4j OM_SIMILAR edges)
3. Entity centrality (PageRank boost for important entities)

Query routing automatically selects search strategy:
- VECTOR_ONLY: No entities detected, pure semantic search
- HYBRID: Single entity, balanced vector+graph fusion
- GRAPH_PRIMARY: Multiple entities or relationship keywords, graph-preferred

Returns lean JSON optimized for LLM processing:
- results[].id: UUID for chaining with update_memory/delete_memories
- results[].memory: The actual content
- results[].score: Final relevance score (2 decimals)
- results[].category: decision | convention | architecture | dependency | workflow | testing | security | performance | runbook | glossary
- results[].scope: session | user | team | project | org | enterprise
- results[].artifact_type: repo | service | module | component | api | db | infra | file
- results[].artifact_ref: reference string (if present)
- results[].entity: team/service/component/person (if present)
- results[].access_entity: ACL identifier (e.g., "org:cloudfactory", "team:acme/backend")
- results[].tags: Qualitative info (if present)
- results[].created_at: Berlin time (Europe/Berlin)
- results[].updated_at: Berlin time (if present)
- results[].visibility_reason: (verbose only) Which grants matched for access

Examples:
- Core pattern search (no recency bias):
  search_memory(query="Kritik-Trigger", entity="BMG")

- Current situation (prefer recent):
  search_memory(query="Projekt-Status", recency_weight=0.5)

- Explicit time window:
  search_memory(query="Meeting", created_after="2025-11-28T00:00:00Z")

- Debug mode with full scoring:
  search_memory(query="Pattern", verbose=true)

meta_relations compact format (default):
- meta_relations[memory_id].artifact: File path for code navigation
- meta_relations[memory_id].similar[]: List of {id, score, preview} for related memories
- meta_relations[memory_id].entities[]: Related entity names
- meta_relations[memory_id].tags: Tag key-value dict
- meta_relations[memory_id].evidence[]: ADR/PR/issue references

Note: With relation_detail="full", meta_relations includes all OM_* relations in verbose format.
Use get_memory(memory_id) to fetch full content of related memories from similar[].id.
""")
async def search_memory(
    query: str,
    # BOOST context (soft ranking - does NOT exclude)
    category: str = None,
    scope: str = None,
    artifact_type: str = None,
    artifact_ref: str = None,
    entity: str = None,
    tags: str = None,
    # RECENCY boost (soft ranking - does NOT exclude)
    recency_weight: float = 0.0,
    recency_halflife_days: int = 45,
    # DATE FILTERS (hard exclusion - these DO exclude)
    created_after: str = None,
    created_before: str = None,
    updated_after: str = None,
    updated_before: str = None,
    # EXCLUSIONS (hard exclusion)
    exclude_states: str = "deleted",
    exclude_tags: str = None,
    # CONTROL
    limit: int = 10,
    verbose: bool = False,
    # HYBRID RETRIEVAL (Phase 2: RRF Multi-Source Fusion)
    use_rrf: bool = True,
    graph_seed_count: int = 5,
    # QUERY ROUTING (Phase 3: Intelligent Query Routing)
    auto_route: bool = True,
    # RELATION OUTPUT (controls meta_relations format)
    relation_detail: str = "standard",
) -> str:
    """Search memories with re-ranking. See tool description for full docs."""
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return json.dumps({"error": "Memory system is currently unavailable"})

    # Validate and cap parameters
    limit = min(max(1, limit), 50)
    recency_weight = min(max(0.0, recency_weight), 1.0)

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # ACL check - get accessible memory IDs based on access_entity grants
            principal = principal_var.get(None)
            app_accessible = _get_accessible_memories(db, principal, user, app, include_archived=True)
            accessible_memory_ids = [memory.id for memory in app_accessible]

            allowed = set(str(mid) for mid in accessible_memory_ids)

            # Build search context for boosting
            boost_tags = [t.strip() for t in tags.split(",")] if tags else []

            context = SearchContext(
                category=category,
                scope=scope,
                artifact_type=artifact_type,
                artifact_ref=artifact_ref,
                entity=entity,
                tags=boost_tags,
                recency_weight=recency_weight,
                recency_halflife_days=recency_halflife_days,
            )

            # Build exclusion filters
            filters = ExclusionFilters(
                exclude_states=[s.strip() for s in exclude_states.split(",")] if exclude_states else ["deleted"],
                exclude_tags=[t.strip() for t in exclude_tags.split(",")] if exclude_tags else [],
                boost_tags=boost_tags,
                created_after=parse_datetime(created_after),
                created_before=parse_datetime(created_before),
                updated_after=parse_datetime(updated_after),
                updated_before=parse_datetime(updated_before),
            )

            # =========================================================
            # PHASE 3: Query Routing (Intelligent Route Selection)
            # =========================================================
            route = None
            query_analysis = None
            rrf_alpha = 0.6  # Default RRF alpha

            if auto_route and use_rrf:
                try:
                    from app.utils.query_router import (
                        analyze_query, get_routing_config, RouteType, get_rrf_alpha_for_route
                    )
                    routing_config = get_routing_config()

                    if routing_config.enabled:
                        query_analysis = analyze_query(query, uid, routing_config)
                        route = query_analysis.route
                        rrf_alpha = get_rrf_alpha_for_route(route)

                        if verbose:
                            logging.debug(
                                f"Query routing: route={route.value}, "
                                f"entities={[e[0] for e in query_analysis.detected_entities]}, "
                                f"keywords={query_analysis.relationship_keywords}"
                            )
                except Exception as routing_error:
                    logging.warning(f"Query routing failed, using default: {routing_error}")

            # =========================================================
            # PHASE 1: Vector Search
            # =========================================================
            # No metadata filters - we do re-ranking instead
            embeddings = memory_client.embedding_model.embed(query, "search")
            search_limit = limit * 3  # Pool for reranking

            search_filters = {"user_id": uid} if not principal else None

            hits = memory_client.vector_store.search(
                query=query,
                vectors=embeddings,
                limit=search_limit,
                filters=search_filters,
            )

            # =========================================================
            # PHASE 2: Graph Retrieval & RRF Fusion
            # =========================================================
            graph_results = []
            rrf_stats = None
            rrf_enabled = (
                use_rrf
                and is_graph_enabled()
                and (route is None or route.value != "vector")  # Skip for VECTOR_ONLY
            )

            if rrf_enabled:
                try:
                    from app.graph.graph_ops import (
                        retrieve_via_similarity_graph, is_similarity_enabled
                    )
                    from app.utils.rrf_fusion import (
                        RRFFusion, RRFConfig, RetrievalResult
                    )

                    # Check if similarity edges are available
                    if is_similarity_enabled():
                        # Build seed list from top vector results
                        seed_ids = []
                        for h in hits:
                            if str(h.id) in allowed:
                                seed_ids.append(str(h.id))
                            if len(seed_ids) >= graph_seed_count:
                                break

                        if len(seed_ids) >= 2:  # Need at least 2 seeds
                            # Graph retrieval via OM_SIMILAR edges
                            graph_raw = retrieve_via_similarity_graph(
                                user_id=uid,
                                seed_memory_ids=seed_ids,
                                allowed_memory_ids=allowed,
                                limit=search_limit,
                                min_score=0.5,
                            )

                            # Convert to RetrievalResult format
                            graph_results = [
                                RetrievalResult(
                                    memory_id=r["id"],
                                    rank=rank,
                                    score=r.get("avgSimilarity", 0),
                                    source="graph",
                                    payload=r,
                                )
                                for rank, r in enumerate(graph_raw, 1)
                            ]

                            if verbose and graph_results:
                                logging.debug(
                                    f"Graph retrieval: {len(graph_results)} candidates "
                                    f"from {len(seed_ids)} seeds"
                                )
                except Exception as graph_error:
                    logging.warning(f"Graph retrieval failed, using vector-only: {graph_error}")

            # =========================================================
            # Entity-Aware Query Expansion (for GRAPH_PRIMARY route)
            # =========================================================
            # When 2+ entities detected and route is GRAPH_PRIMARY,
            # find bridge entities and expand retrieval
            bridge_entities = []

            if (
                rrf_enabled
                and query_analysis
                and len(query_analysis.detected_entities) >= 2
                and route is not None
                and route.value == "graph"  # GRAPH_PRIMARY
            ):
                try:
                    from app.graph.graph_ops import (
                        find_bridge_entities, retrieve_via_entity_graph
                    )

                    entity_names = [e[0] for e in query_analysis.detected_entities]

                    # Find bridge entities (3-hop max for transitive paths like A→B→C→D)
                    bridge_entities = find_bridge_entities(
                        user_id=uid,
                        entity_names=entity_names,
                        max_bridges=5,
                        min_count=2,
                        max_hops=3,
                    )

                    # Expand entity list with bridges
                    expanded_names = entity_names.copy()
                    for bridge in bridge_entities:
                        if bridge["name"] not in expanded_names:
                            expanded_names.append(bridge["name"])

                    # Retrieve memories via entity graph
                    entity_raw = retrieve_via_entity_graph(
                        user_id=uid,
                        entity_names=expanded_names,
                        allowed_memory_ids=allowed,
                        limit=search_limit,
                    )

                    # Convert to RetrievalResult for RRF fusion
                    entity_graph_results = []
                    for rank, r in enumerate(entity_raw, 1):
                        # Avoid duplicates already in graph_results
                        if not any(gr.memory_id == r["id"] for gr in graph_results):
                            entity_graph_results.append(
                                RetrievalResult(
                                    memory_id=r["id"],
                                    rank=rank,
                                    score=min(1.0, r.get("matchedEntities", 1) / len(expanded_names)),
                                    source="entity_graph",
                                    payload=r,
                                )
                            )

                    # Merge entity_graph_results into graph_results for RRF
                    graph_results.extend(entity_graph_results)

                    if verbose and (bridge_entities or entity_graph_results):
                        logging.debug(
                            f"Entity expansion: {entity_names} + bridges {[b['name'] for b in bridge_entities]} "
                            f"-> {len(entity_graph_results)} additional memories"
                        )

                except Exception as entity_error:
                    logging.warning(f"Entity graph retrieval failed: {entity_error}")

            # =========================================================
            # PHASE 1 (continued): Graph-Enhanced Reranking
            # =========================================================
            # Fetch graph context for graph boost calculation
            graph_context = None
            if is_graph_enabled():
                try:
                    from app.graph.graph_cache import fetch_graph_context
                    memory_ids_in_pool = [str(h.id) for h in hits if h.id and str(h.id) in allowed]
                    graph_context = fetch_graph_context(
                        memory_ids=memory_ids_in_pool,
                        user_id=uid,
                        context_tags=boost_tags,
                    )
                except Exception as cache_error:
                    logging.warning(f"Graph context fetch failed: {cache_error}")

            # Process results
            results = []
            core_keys = {"data", "hash", "created_at", "updated_at"}

            for h in hits:
                memory_id = h.id
                semantic_score = h.score
                payload = h.payload

                # ACL check
                if memory_id is None or str(memory_id) not in allowed:
                    continue

                # Extract metadata
                metadata = {k: v for k, v in payload.items() if k not in core_keys}
                stored_tags = normalize_tags(metadata.get("tags", {}))

                # Exclusion check (states, tags, date ranges)
                excluded, reason = should_exclude(payload, metadata, stored_tags, filters)
                if excluded:
                    continue

                # Boost calculation (with graph context if available)
                boost, boost_breakdown = compute_boost(
                    metadata=metadata,
                    stored_tags=stored_tags,
                    context=context,
                    created_at_str=payload.get("created_at"),
                    graph_context=graph_context,
                    memory_id=str(memory_id),
                )

                final_score = calculate_final_score(semantic_score, boost)

                access_entity = metadata.get("access_entity")
                result_entry = {
                    "id": memory_id,
                    "memory": payload.get("data"),
                    "access_entity": access_entity,
                    "scores": {
                        "semantic": round(semantic_score, 4),
                        "boost": round(boost, 4),
                        "final": round(final_score, 4),
                    },
                    "metadata": {
                        "category": metadata.get("category"),
                        "scope": metadata.get("scope"),
                        "artifact_type": metadata.get("artifact_type"),
                        "artifact_ref": metadata.get("artifact_ref"),
                        "source": metadata.get("source"),
                        "evidence": metadata.get("evidence"),
                        "tags": stored_tags,
                        "entity": metadata.get("entity"),
                        "access_entity": access_entity,
                    },
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                }

                if verbose and principal and access_entity:
                    result_entry["visibility_reason"] = {
                        "access_entity": access_entity,
                        "matched_grants": _matching_grants_for_access_entity(principal, access_entity),
                    }

                results.append(result_entry)

            # =========================================================
            # PHASE 2 (continued): RRF Fusion
            # =========================================================
            # If we have graph results, fuse them with vector results using RRF
            if graph_results:
                try:
                    from app.utils.rrf_fusion import RRFFusion, RRFConfig, RetrievalResult

                    # Convert vector results to RetrievalResult format
                    vector_results_for_rrf = []
                    results_by_id = {str(r["id"]): r for r in results}
                    for rank, r in enumerate(results, 1):
                        vector_results_for_rrf.append(RetrievalResult(
                            memory_id=str(r["id"]),
                            rank=rank,
                            score=r["scores"]["final"],
                            source="vector",
                            payload=r,
                        ))

                    # Create RRF fusion with configured alpha
                    fusion = RRFFusion(RRFConfig(alpha=rrf_alpha))
                    fused_with_stats = fusion.fuse_with_stats(
                        vector_results_for_rrf, graph_results
                    )
                    fused_results = fused_with_stats["results"]
                    rrf_stats = fused_with_stats["stats"]

                    # Process graph-only results that weren't in vector results
                    graph_only_ids = set()
                    for fr in fused_results:
                        if fr.memory_id not in results_by_id:
                            graph_only_ids.add(fr.memory_id)

                    # Fetch and process graph-only results
                    if graph_only_ids:
                        # Get payloads from graph results
                        graph_payload_map = {r.memory_id: r.payload for r in graph_results}

                        for fr in fused_results:
                            if fr.memory_id in graph_only_ids:
                                gp = graph_payload_map.get(fr.memory_id, {})
                                access_entity = gp.get("access_entity")
                                # Create result entry for graph-only hit
                                graph_entry = {
                                    "id": fr.memory_id,
                                    "memory": gp.get("content"),
                                    "access_entity": access_entity,
                                    "scores": {
                                        "semantic": 0.0,  # No vector score
                                        "boost": 0.0,
                                        "final": fr.rrf_score,
                                        "rrf": fr.rrf_score,
                                        "graph_similarity": fr.original_score,
                                    },
                                    "metadata": {
                                        "category": gp.get("category"),
                                        "scope": gp.get("scope"),
                                        "artifact_type": gp.get("artifact_type"),
                                        "artifact_ref": gp.get("artifact_ref"),
                                        "source": gp.get("source"),
                                        "evidence": gp.get("evidence"),
                                        "tags": {},
                                        "entity": None,
                                        "access_entity": access_entity,
                                    },
                                    "created_at": None,
                                    "updated_at": None,
                                    "source": "graph",
                                    "graph_info": {
                                        "seed_connections": gp.get("seedConnections"),
                                        "avg_similarity": gp.get("avgSimilarity"),
                                    },
                                }
                                if verbose and principal and access_entity:
                                    graph_entry["visibility_reason"] = {
                                        "access_entity": access_entity,
                                        "matched_grants": _matching_grants_for_access_entity(principal, access_entity),
                                    }
                                results_by_id[fr.memory_id] = graph_entry

                    # Re-order results by RRF score
                    reordered = []
                    for fr in fused_results:
                        if fr.memory_id in results_by_id:
                            r = results_by_id[fr.memory_id]
                            # Add RRF score to existing results
                            r["scores"]["rrf"] = fr.rrf_score
                            r["in_both_sources"] = fr.in_both
                            reordered.append(r)

                    results = reordered

                    if verbose:
                        logging.debug(
                            f"RRF fusion: {rrf_stats['fused_total']} total, "
                            f"{rrf_stats['in_both_sources']} in both sources"
                        )

                except Exception as rrf_error:
                    logging.warning(f"RRF fusion failed, using vector-only ranking: {rrf_error}")
                    # Fall back to original vector-only sorting
                    results.sort(key=lambda x: x["scores"]["final"], reverse=True)
            else:
                # No graph results - standard vector-only sorting
                results.sort(key=lambda x: x["scores"]["final"], reverse=True)

            # Apply limit
            results = results[:limit]

            # 5. Access logging
            for r in results:
                if r.get("id"):
                    try:
                        access_log = MemoryAccessLog(
                            memory_id=uuid.UUID(str(r["id"])),
                            app_id=app.id,
                            access_type="search",
                            metadata_={
                                "query": query,
                                "semantic_score": r["scores"]["semantic"],
                                "boost": r["scores"]["boost"],
                                "final_score": r["scores"]["final"],
                                "recency_weight": recency_weight,
                            }
                        )
                        db.add(access_log)
                    except Exception as log_error:
                        logging.warning(f"Failed to log access: {log_error}")

            db.commit()

            # Format response (lean by default, verbose for debugging)
            if verbose:
                # Build verbose response metadata
                context_applied = {}
                if category:
                    context_applied["category"] = category
                if scope:
                    context_applied["scope"] = scope
                if artifact_type:
                    context_applied["artifact_type"] = artifact_type
                if artifact_ref:
                    context_applied["artifact_ref"] = artifact_ref
                if entity:
                    context_applied["entity"] = entity
                if boost_tags:
                    context_applied["tags"] = boost_tags
                if recency_weight > 0:
                    context_applied["recency"] = {
                        "weight": recency_weight,
                        "halflife_days": recency_halflife_days,
                    }

                filters_applied = {}
                if filters.created_after:
                    filters_applied["created_after"] = created_after
                if filters.created_before:
                    filters_applied["created_before"] = created_before
                if filters.updated_after:
                    filters_applied["updated_after"] = updated_after
                if filters.updated_before:
                    filters_applied["updated_before"] = updated_before
                if filters.exclude_tags:
                    filters_applied["exclude_tags"] = filters.exclude_tags

                response = format_search_results(
                    results=results,
                    verbose=True,
                    query=query,
                    context_applied=context_applied if context_applied else None,
                    filters_applied=filters_applied if filters_applied else None,
                    total_candidates=len(hits),
                )

                # Add hybrid retrieval info to verbose response
                if use_rrf or auto_route:
                    hybrid_info = {}
                    if query_analysis:
                        hybrid_info["routing"] = {
                            "route": route.value if route else "hybrid",
                            "detected_entities": [e[0] for e in query_analysis.detected_entities],
                            "relationship_keywords": query_analysis.relationship_keywords,
                            "confidence": query_analysis.confidence,
                            "analysis_time_ms": query_analysis.analysis_time_ms,
                        }
                    if rrf_stats:
                        hybrid_info["rrf"] = {
                            "alpha": rrf_alpha,
                            "vector_candidates": rrf_stats.get("vector_candidates", 0),
                            "graph_candidates": rrf_stats.get("graph_candidates", 0),
                            "in_both_sources": rrf_stats.get("in_both_sources", 0),
                        }
                    if graph_context and graph_context.available:
                        hybrid_info["graph_boost"] = {
                            "max_pagerank": graph_context.max_pagerank,
                            "max_degree": graph_context.max_degree,
                            "max_cluster_size": graph_context.max_cluster_size,
                        }
                    # Add entity expansion info if bridges were found
                    if bridge_entities and query_analysis:
                        hybrid_info["entity_expansion"] = {
                            "detected_entities": [e[0] for e in query_analysis.detected_entities],
                            "bridge_entities": [b["name"] for b in bridge_entities],
                            "expanded_count": len(query_analysis.detected_entities) + len(bridge_entities),
                        }
                    if hybrid_info:
                        response["hybrid_retrieval"] = hybrid_info
            else:
                # Lean format (default) - optimized for LLM processing
                response = format_search_results(results=results, verbose=False)

            # Enrich with graph relations (only if graph is available)
            # This adds two new optional fields without changing existing response shape
            # The relation_detail parameter controls the output format:
            #   - "none": No meta_relations (minimal tokens)
            #   - "minimal": Only artifact + similar IDs
            #   - "standard": artifact + similar + entities + tags + evidence (default, compact)
            #   - "full": Verbose format with all relation types (for debugging)
            try:
                # Skip meta_relations if relation_detail is "none"
                if relation_detail != "none":
                    # Get memory IDs from results for meta_relations lookup
                    memory_ids = [str(r.get("id")) for r in results if r.get("id")]

                    # 1. meta_relations: deterministic metadata relations from Neo4j projection
                    if memory_ids and is_graph_enabled():
                        raw_meta_relations = get_meta_relations_for_memories(memory_ids)
                        if raw_meta_relations:
                            if relation_detail == "full":
                                # Full verbose format (for debugging)
                                response["meta_relations"] = raw_meta_relations
                            else:
                                # Compact format (standard or minimal)
                                compact_relations = {}
                                for memory_id, relations in raw_meta_relations.items():
                                    compact = format_compact_relations(relations)
                                    if compact:
                                        if relation_detail == "minimal":
                                            # Only keep artifact(s) and similar
                                            compact = {
                                                k: v for k, v in compact.items()
                                                if k in ("artifact", "artifacts", "similar")
                                            }
                                        if compact:  # Still has content after filtering
                                            compact_relations[memory_id] = compact
                                if compact_relations:
                                    response["meta_relations"] = compact_relations

                # 2. relations: Mem0 Graph Memory relations (LLM-extracted entities)
                if is_mem0_graph_enabled():
                    graph_relations = get_graph_relations(
                        query=query,
                        user_id=uid,
                        limit=10,
                    )
                    if graph_relations:
                        response["relations"] = graph_relations

            except Exception as graph_error:
                logging.warning(f"Graph enrichment failed: {graph_error}")
                # Don't fail the search, just skip graph enrichment

            return json.dumps(response, default=str)

        finally:
            db.close()

    except Exception as e:
        logging.error(f"Error in search_memory: {e}", exc_info=True)
        return json.dumps({"error": f"Error searching memories: {str(e)}"})


@mcp.tool(description="""Find memories related to a seed memory via the Neo4j metadata subgraph.

This tool is deterministic (no LLM). It uses the OM_* metadata graph that is
projected from your Qdrant/OpenMemory metadata.

Parameters:
- memory_id: UUID of the seed memory (required)
- via: Optional comma-separated list of dimensions to traverse:
       tag, entity, category, scope, artifact_type, artifact_ref, evidence, app
       (you can also pass explicit OM_* relationship types)
- limit: Max related memories to return (default: 10, max: 100)

Returns:
- seed_memory_id
- related[]: each item includes shared_relations + shared_count
""")
async def graph_related_memories(
    memory_id: str,
    via: str = None,
    limit: int = 10,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps(
            {
                "seed_memory_id": memory_id,
                "related": [],
                "count": 0,
                "graph_enabled": False,
            }
        )

    limit = min(max(1, int(limit or 10)), 100)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = set(str(memory.id) for memory in accessible_memories)

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            related = find_related_memories_in_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=list(allowed),
                via=via,
                limit=limit,
            )

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_related",
                        metadata_={"via": via, "limit": limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_related access: {log_error}")

            return json.dumps(
                {
                    "seed_memory_id": memory_id,
                    "related": related,
                    "count": len(related),
                    "graph_enabled": True,
                },
                default=str,
            )
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_related_memories: {e}")
        return json.dumps({"error": f"Error finding related memories: {str(e)}"})


@mcp.tool(description="""Return a small neighborhood subgraph around a memory (Neo4j metadata graph).

depth:
- 1: only seed memory + its dimension nodes
- 2: also include other memories connected through shared dimensions

Parameters:
- memory_id: UUID of the seed memory (required)
- depth: 1 or 2 (default: 2)
- via: Optional comma-separated dimensions to use for the depth=2 expansion
- related_limit: Max number of related memory nodes to include (default: 25)
""")
async def graph_subgraph(
    memory_id: str,
    depth: int = 2,
    via: str = None,
    related_limit: int = 25,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps(
            {
                "seed_memory_id": memory_id,
                "nodes": [],
                "edges": [],
                "related": [],
                "graph_enabled": False,
            }
        )

    depth = max(1, min(int(depth or 2), 2))
    related_limit = max(0, min(int(related_limit or 25), 200))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = set(str(memory.id) for memory in accessible_memories)

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            subgraph = get_memory_subgraph_from_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=list(allowed),
                depth=depth,
                via=via,
                related_limit=related_limit,
            )
            if subgraph is None:
                return json.dumps({"error": f"Memory '{memory_id}' not found in graph"})

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_subgraph",
                        metadata_={"via": via, "depth": depth, "related_limit": related_limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_subgraph access: {log_error}")

            subgraph["graph_enabled"] = True
            return json.dumps(subgraph, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_subgraph: {e}")
        return json.dumps({"error": f"Error building subgraph: {str(e)}"})


@mcp.tool(description="""Aggregate memories by a Neo4j metadata dimension.

Parameters:
- group_by: category | scope | artifact_type | artifact_ref | tag | entity | app | evidence | source | state
- limit: max buckets to return (default: 20)
""")
async def graph_aggregate(group_by: str, limit: int = 20) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"group_by": group_by, "buckets": [], "count": 0, "graph_enabled": False})

    limit = max(1, min(int(limit or 20), 200))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]

            buckets = aggregate_memories_in_graph(
                user_id=uid,
                group_by=group_by,
                allowed_memory_ids=allowed,
                limit=limit,
            )

            return json.dumps(
                {
                    "group_by": group_by,
                    "buckets": buckets,
                    "count": len(buckets),
                    "graph_enabled": True,
                },
                default=str,
            )
        finally:
            db.close()
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logging.exception(f"Error in graph_aggregate: {e}")
        return json.dumps({"error": f"Error aggregating memories: {str(e)}"})


@mcp.tool(description="""Find frequently co-occurring tags across memories (Neo4j metadata graph).

Parameters:
- limit: max tag pairs to return (default: 20)
- min_count: minimum co-occurrence count to include (default: 2)
- sample_size: number of example memory IDs per pair (default: 3)
""")
async def graph_tag_cooccurrence(limit: int = 20, min_count: int = 2, sample_size: int = 3) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"pairs": [], "count": 0, "graph_enabled": False})

    limit = max(1, min(int(limit or 20), 200))
    min_count = max(1, int(min_count or 1))
    sample_size = max(0, min(int(sample_size or 0), 10))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]

            pairs = tag_cooccurrence_in_graph(
                user_id=uid,
                allowed_memory_ids=allowed,
                limit=limit,
                min_count=min_count,
                sample_size=sample_size,
            )

            return json.dumps(
                {"pairs": pairs, "count": len(pairs), "graph_enabled": True},
                default=str,
            )
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_tag_cooccurrence: {e}")
        return json.dumps({"error": f"Error computing tag co-occurrence: {str(e)}"})


@mcp.tool(description="""Find a shortest path between two entities through the Neo4j metadata graph.

Parameters:
- entity_a: Name of entity A (matches metadata.entity projection)
- entity_b: Name of entity B
- max_hops: cap for traversal (default: 6, max: 12)
""")
async def graph_path_between_entities(entity_a: str, entity_b: str, max_hops: int = 6) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"path": None, "graph_enabled": False})

    max_hops = max(2, min(int(max_hops or 6), 12))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]

            path = path_between_entities_in_graph(
                user_id=uid,
                entity_a=entity_a,
                entity_b=entity_b,
                allowed_memory_ids=allowed,
                max_hops=max_hops,
            )

            return json.dumps({"path": path, "graph_enabled": True}, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_path_between_entities: {e}")
        return json.dumps({"error": f"Error finding path: {str(e)}"})


@mcp.tool(description="""List all memories in the user's memory.

Returns lean JSON optimized for LLM processing:
- id: UUID for chaining with update_memory/delete_memories
- memory: The actual content
- category: decision | convention | architecture | dependency | workflow | testing | security | performance | runbook | glossary
- scope: session | user | team | project | org | enterprise
- artifact_type: repo | service | module | component | api | db | infra | file
- artifact_ref: reference string (if present)
- entity: team/service/component/person (if present)
- access_entity: ACL identifier (e.g., "org:cloudfactory", "team:acme/backend")
- tags: Qualitative info (if present)
- created_at: Berlin time (Europe/Berlin)
- updated_at: Berlin time (if present)
""")
async def list_memories() -> str:
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=False)
            raw_memories = []

            for memory in accessible_memories:
                md = memory.metadata_ or {}
                raw_memories.append(
                    {
                        "id": str(memory.id),
                        "memory": memory.content,
                        "category": md.get("category"),
                        "scope": md.get("scope"),
                        "artifact_type": md.get("artifact_type"),
                        "artifact_ref": md.get("artifact_ref"),
                        "entity": md.get("entity"),
                        "access_entity": md.get("access_entity"),
                        "source": md.get("source"),
                        "evidence": md.get("evidence"),
                        "tags": md.get("tags"),
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                    }
                )

                access_log = MemoryAccessLog(
                    memory_id=memory.id,
                    app_id=app.id,
                    access_type="list",
                    metadata_={},
                )
                db.add(access_log)
            db.commit()

            # Apply lean formatting
            formatted_memories = format_memory_list(raw_memories)
            return json.dumps(formatted_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="""Retrieve a single memory by UUID with all fields.

Required:
- memory_id: UUID of the memory to retrieve

Returns all memory fields (consistent with list_memories format):
- id: UUID for chaining with update_memory/delete_memories
- memory: The actual content
- category: decision | convention | architecture | dependency | workflow | testing | security | performance | runbook | glossary
- scope: session | user | team | project | org | enterprise
- artifact_type: repo | service | module | component | api | db | infra | file
- artifact_ref: reference string (if present)
- entity: team/service/component/person (if present)
- source: user | inference (if present)
- evidence: List of ADR/PR/issue refs (if present)
- tags: Qualitative info (if present)
- access_entity: ACL identifier (e.g., "org:cloudfactory", "team:acme/backend")
- created_at: Berlin time (Europe/Berlin)
- updated_at: Berlin time (if present)

Errors:
- {"error": "Memory not found", "code": "NOT_FOUND"} - memory doesn't exist or user lacks access
- {"error": "Invalid memory_id format", "code": "INVALID_INPUT"} - malformed UUID

Use cases:
- Verify that an update_memory call succeeded
- Debug ACL issues (see access_entity field)
- Follow evidence chains between memories
- Inspect memory state after operations

Examples:
- get_memory(memory_id="ba93af28-784d-4262-b8f9-adb08c45acab")
""")
async def get_memory(memory_id: str) -> str:
    """Retrieve a single memory by UUID with all fields."""
    # Check scope - requires memories:read
    scope_error = _check_tool_scope("memories:read")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Validate UUID format
    try:
        memory_uuid = uuid.UUID(memory_id)
    except (ValueError, TypeError):
        return json.dumps({
            "error": "Invalid memory_id format",
            "code": "INVALID_INPUT",
        })

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Query the memory
            memory = db.query(Memory).filter(Memory.id == memory_uuid).first()

            if not memory:
                return json.dumps({
                    "error": "Memory not found",
                    "code": "NOT_FOUND",
                })

            # Access control check
            principal = principal_var.get(None)
            md = memory.metadata_ or {}
            access_entity = md.get("access_entity")

            if access_entity:
                # Multi-user routing: check if principal has access to the access_entity
                if principal:
                    from app.security.access import can_read_access_entity
                    if not can_read_access_entity(principal, access_entity):
                        # Return NOT_FOUND to avoid leaking existence (security best practice)
                        return json.dumps({
                            "error": "Memory not found",
                            "code": "NOT_FOUND",
                        })
                else:
                    # No principal but memory has access_entity - check legacy owner access
                    if memory.user_id != user.id:
                        return json.dumps({
                            "error": "Memory not found",
                            "code": "NOT_FOUND",
                        })
            else:
                # Legacy behavior: only owner can access memories without access_entity
                if memory.user_id != user.id:
                    return json.dumps({
                        "error": "Memory not found",
                        "code": "NOT_FOUND",
                    })

            # Build raw memory object for formatting
            raw_memory = {
                "id": str(memory.id),
                "memory": memory.content,
                "category": md.get("category"),
                "scope": md.get("scope"),
                "artifact_type": md.get("artifact_type"),
                "artifact_ref": md.get("artifact_ref"),
                "entity": md.get("entity"),
                "access_entity": md.get("access_entity"),
                "source": md.get("source"),
                "evidence": md.get("evidence"),
                "tags": md.get("tags"),
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            }

            # Create access log
            access_log = MemoryAccessLog(
                memory_id=memory.id,
                app_id=app.id,
                access_type="get",
                metadata_={},
            )
            db.add(access_log)
            db.commit()

            # Apply lean formatting (without score since this is not a search result)
            from app.utils.response_format import format_memory_result
            formatted_memory = format_memory_result(raw_memory, include_score=False)

            return json.dumps(formatted_memory, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memory: {e}")
        return json.dumps({"error": f"Error getting memory: {e}"})


@mcp.tool(description="Delete specific memories by their IDs")
async def delete_memories(memory_ids: list[str]) -> str:
    # Check scope - requires memories:delete
    scope_error = _check_tool_scope("memories:delete")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Convert string IDs to UUIDs and filter accessible ones
            requested_ids = [uuid.UUID(mid) for mid in memory_ids]
            principal = principal_var.get(None)
            accessible_by_app = _get_accessible_memories(db, principal, user, app, include_archived=True)

            # Filter by write permissions (group-editable policy)
            if principal:
                from app.security.access import can_write_to_access_entity

                accessible_memory_ids = []
                for memory in accessible_by_app:
                    access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
                    if access_entity:
                        if can_write_to_access_entity(principal, access_entity):
                            accessible_memory_ids.append(memory.id)
                    else:
                        # Legacy memory without access_entity - only owner can delete
                        accessible_memory_ids.append(memory.id)
            else:
                accessible_memory_ids = [memory.id for memory in accessible_by_app]

            # Only delete memories that are both requested and accessible
            ids_to_delete = [mid for mid in requested_ids if mid in accessible_memory_ids]

            if not ids_to_delete:
                return "Error: No accessible memories found with provided IDs"

            # Delete from vector store
            for memory_id in ids_to_delete:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    # Update memory state
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now

                    # Create history entry
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.active,
                        new_state=MemoryState.deleted
                    )
                    db.add(history)

                    # Create access log entry
                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="delete",
                        metadata_={"operation": "delete_by_id"}
                    )
                    db.add(access_log)

            db.commit()

            # Delete from Neo4j graph (non-blocking)
            for memory_id in ids_to_delete:
                try:
                    # Update entity edges before deleting memory
                    update_entity_edges_on_memory_delete(str(memory_id), uid)
                    # Delete similarity edges
                    delete_similarity_edges_for_memory(str(memory_id), uid)
                    # Delete the memory node
                    delete_memory_from_graph(str(memory_id))
                except Exception as graph_error:
                    logging.warning(f"Graph deletion failed for {memory_id}: {graph_error}")

            return f"Successfully deleted {len(ids_to_delete)} memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    # Check scope - requires memories:delete
    scope_error = _check_tool_scope("memories:delete")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)

            if principal:
                from app.security.access import can_write_to_access_entity

                accessible_memory_ids = []
                for memory in accessible_memories:
                    access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None
                    if access_entity:
                        if can_write_to_access_entity(principal, access_entity):
                            accessible_memory_ids.append(memory.id)
                    else:
                        if memory.user_id == user.id:
                            accessible_memory_ids.append(memory.id)
            else:
                accessible_memory_ids = [memory.id for memory in accessible_memories]

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()

            # Delete from Neo4j graph (non-blocking)
            for memory_id in accessible_memory_ids:
                try:
                    update_entity_edges_on_memory_delete(str(memory_id), uid)
                    delete_similarity_edges_for_memory(str(memory_id), uid)
                    delete_memory_from_graph(str(memory_id))
                except Exception as graph_error:
                    logging.warning(f"Graph deletion failed for memory {memory_id}: {graph_error}")

            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool(description="""Update a memory's content, structure, or tags.

Required:
- memory_id: UUID of the memory to update

Optional content:
- text: New content text

Optional structure (only provided fields are updated):
- category: decision | convention | architecture | dependency | workflow | testing | security | performance | runbook | glossary
- scope: session | user | team | project | org | enterprise
- artifact_type: repo | service | module | component | api | db | infra | file
- artifact_ref: reference string (path, symbol, service)
- access_entity: Access control entity (format: <prefix>:<value>)
  When changing access_entity, you must have permission for both the old AND new access_entity

Optional metadata:
- entity: Reference entity
- source: user|inference
- evidence: List of evidence items

Tag operations:
- add_tags: Dict of tags to add/update (or list of strings)
- remove_tags: List of tag keys to remove (or comma-separated string)

Maintenance mode:
- preserve_timestamps: If true, don't update updated_at (for migrations/fixes)

Examples:
- update_memory(memory_id="abc-123", text="Kritik triggert jetzt Neugier", add_tags={"evolved": true})
- update_memory(memory_id="abc-123", category="decision", scope="project", preserve_timestamps=true)
- update_memory(memory_id="abc-123", add_tags={"confirmed": true})
- update_memory(memory_id="abc-123", remove_tags=["silent"])
- update_memory(memory_id="abc-123", access_entity="team:cloudfactory/backend")
""")
async def update_memory(
    memory_id: str,
    # Content
    text: str = None,
    # Structure
    category: str = None,
    scope: str = None,
    artifact_type: str = None,
    artifact_ref: str = None,
    access_entity: str = None,
    # Metadata
    entity: str = None,
    source: str = None,
    evidence: list = None,
    # Tag operations
    add_tags: dict = None,
    remove_tags: list = None,
    # Maintenance mode
    preserve_timestamps: bool = False,
) -> str:
    # Check scope - requires memories:write
    scope_error = _check_tool_scope("memories:write")
    if scope_error:
        return scope_error

    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Validate update fields
    try:
        validated_fields = validate_update_fields(
            category=category,
            scope=scope,
            artifact_type=artifact_type,
            artifact_ref=artifact_ref,
            entity=entity,
            access_entity=access_entity,
            source=source,
            evidence=evidence,
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    content_text = None
    if text is not None:
        try:
            content_text = validate_text(text)
        except StructuredMemoryError as e:
            return json.dumps(_format_structured_memory_error(e))

    try:
        normalized_add_tags = (
            normalize_tags_input(add_tags) if add_tags is not None else None
        )
        normalized_remove_tags = (
            normalize_tag_list_input(remove_tags) if remove_tags is not None else None
        )
    except StructuredMemoryError as e:
        return json.dumps(_format_structured_memory_error(e))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            memory = db.query(Memory).filter(
                Memory.id == uuid.UUID(memory_id),
            ).first()

            if not memory:
                return json.dumps({"error": f"Memory '{memory_id}' not found"})

            # Access control check: verify principal can write to memory's access_entity
            principal = principal_var.get(None)
            current_metadata = memory.metadata_ or {}
            current_access_entity = current_metadata.get("access_entity")

            if not current_access_entity and memory.user_id != user.id:
                return json.dumps({"error": f"Memory '{memory_id}' not found"})

            if principal:
                from app.security.access import can_write_to_access_entity

                # Check write access to current access_entity
                if current_access_entity:
                    if not can_write_to_access_entity(principal, current_access_entity):
                        return json.dumps({
                            "error": f"Access denied: you don't have permission to update memories with access_entity='{current_access_entity}'",
                            "code": "FORBIDDEN",
                        })

                # If changing access_entity, also check access to new access_entity
                new_access_entity = validated_fields.get("access_entity")
                if new_access_entity and new_access_entity != current_access_entity:
                    if not can_write_to_access_entity(principal, new_access_entity):
                        return json.dumps({
                            "error": f"Access denied: you don't have permission to set access_entity='{new_access_entity}'",
                            "code": "FORBIDDEN",
                        })

            # Get current metadata

            # Apply metadata updates
            updated_metadata = apply_metadata_updates(
                current_metadata=current_metadata,
                validated_fields=validated_fields,
                add_tags=normalized_add_tags,
                remove_tags=normalized_remove_tags,
            )

            # Update content if provided
            content_updated = False
            if content_text is not None:
                memory.content = content_text
                content_updated = True

            # Update metadata
            memory.metadata_ = updated_metadata

            # Handle timestamps
            if not preserve_timestamps:
                memory.updated_at = datetime.datetime.now(datetime.UTC)

            # Determine if metadata was updated
            metadata_updated = bool(validated_fields or normalized_add_tags or normalized_remove_tags)

            # Update vector store
            memory_client = get_memory_client_safe()
            if memory_client:
                if content_updated:
                    # Content changed -> re-embed and update payload
                    try:
                        memory_client.update(str(memory_id), content_text)
                        # Also sync metadata (mem0's update doesn't include our structured metadata)
                        from app.utils.vector_sync import sync_metadata_to_qdrant
                        sync_metadata_to_qdrant(
                            memory_id=str(memory_id),
                            memory=memory,
                            metadata=updated_metadata,
                            memory_client=memory_client,
                        )
                    except Exception as vs_error:
                        logging.warning(f"Failed to update vector store: {vs_error}")
                elif metadata_updated:
                    # Only metadata changed -> update payload without re-embedding
                    try:
                        from app.utils.vector_sync import sync_metadata_to_qdrant
                        sync_metadata_to_qdrant(
                            memory_id=str(memory_id),
                            memory=memory,
                            metadata=updated_metadata,
                            memory_client=memory_client,
                        )
                    except Exception as vs_error:
                        logging.warning(f"Failed to sync metadata to vector store: {vs_error}")

            # Create access log
            access_log = MemoryAccessLog(
                memory_id=memory.id,
                app_id=app.id,
                access_type="update",
                metadata_={
                    "fields_updated": (
                        list(validated_fields.keys())
                        + (["text"] if content_text is not None else [])
                    ),
                    "add_tags": normalized_add_tags,
                    "remove_tags": normalized_remove_tags,
                    "preserve_timestamps": preserve_timestamps,
                },
            )
            db.add(access_log)

            db.commit()

            # Re-project to Neo4j graph with updated metadata (non-blocking)
            try:
                project_memory_to_graph(
                    memory_id=memory_id,
                    user_id=uid,
                    content=memory.content,
                    metadata=updated_metadata,
                    created_at=memory.created_at.isoformat() if memory.created_at else None,
                    updated_at=memory.updated_at.isoformat() if memory.updated_at else None,
                    state=memory.state.value if memory.state else "active",
                )
            except Exception as graph_error:
                logging.warning(f"Graph projection failed for updated memory {memory_id}: {graph_error}")

            # Build response
            response = {
                "status": "updated",
                "memory_id": memory_id,
            }

            if validated_fields:
                response["fields_updated"] = list(validated_fields.keys())

            if normalized_add_tags or normalized_remove_tags:
                response["current_tags"] = updated_metadata.get("tags", {})

            return json.dumps(response)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return json.dumps({"error": f"Error updating memory: {e}"})


@mcp.tool(description="""Get pre-computed similar memories via OM_SIMILAR edges.

Uses the materialized similarity graph from Qdrant embeddings for O(1) lookup.
No embedding computation at query time - edges are pre-computed.

Parameters:
- memory_id: UUID of the seed memory (required)
- min_score: Minimum similarity score (0.0-1.0, default: 0.0)
- limit: Max results to return (default: 10, max: 50, for initial searches use 10)

Returns:
- seed_memory_id
- similar[]: each item includes content, similarity_score, rank
- count: number of results
- similarity_enabled: whether feature is active
""")
async def graph_similar_memories(
    memory_id: str,
    min_score: float = 0.0,
    limit: int = 10,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_similarity_enabled():
        return json.dumps({
            "seed_memory_id": memory_id,
            "similar": [],
            "count": 0,
            "similarity_enabled": False,
        })

    limit = min(max(1, int(limit or 10)), 50)
    min_score = max(0.0, min(1.0, float(min_score or 0.0)))

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            principal = principal_var.get(None)
            accessible_memories = _get_accessible_memories(db, principal, user, app, include_archived=True)
            allowed = [str(memory.id) for memory in accessible_memories]

            if memory_id not in allowed:
                return json.dumps({"error": f"Memory '{memory_id}' not found or not accessible"})

            similar = get_similar_memories_from_graph(
                memory_id=memory_id,
                user_id=uid,
                allowed_memory_ids=allowed,
                min_score=min_score,
                limit=limit,
            )

            # Access log (best-effort)
            try:
                db.add(
                    MemoryAccessLog(
                        memory_id=uuid.UUID(memory_id),
                        app_id=app.id,
                        access_type="graph_similar",
                        metadata_={"min_score": min_score, "limit": limit},
                    )
                )
                db.commit()
            except Exception as log_error:
                logging.warning(f"Failed to log graph_similar access: {log_error}")

            return json.dumps({
                "seed_memory_id": memory_id,
                "similar": similar,
                "count": len(similar),
                "similarity_enabled": True,
            }, default=str)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in graph_similar_memories: {e}")
        return json.dumps({"error": f"Error finding similar memories: {str(e)}"})


@mcp.tool(description="""Get an entity's co-mention network via OM_CO_MENTIONED edges.

Shows other entities that frequently appear in the same memories as this entity.
Useful for understanding relationship networks (people, places, concepts).

Parameters:
- entity_name: Name of the entity to explore (required)
- min_count: Minimum co-mention count (default: 1)
- limit: Max connections to return (default: 20, max: 100)

Returns:
- entity: the queried entity name
- connections[]: each has {name, count, sample_memory_ids}
- total: number of connections
- graph_enabled: whether feature is active
""")
async def graph_entity_network(
    entity_name: str,
    min_count: int = 1,
    limit: int = 20,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "connections": [],
            "total": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 20)), 100)
    min_count = max(1, int(min_count or 1))

    try:
        result = get_entity_network_from_graph(
            entity_name=entity_name,
            user_id=uid,
            min_count=min_count,
            limit=limit,
        )

        if result is None:
            return json.dumps({
                "entity": entity_name,
                "connections": [],
                "total": 0,
                "graph_enabled": True,
                "message": f"Entity '{entity_name}' not found",
            })

        return json.dumps({
            "entity": entity_name,
            "connections": result.get("connections", []),
            "total": result.get("total", 0),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_entity_network: {e}")
        return json.dumps({"error": f"Error getting entity network: {str(e)}"})


@mcp.tool(description="""Get co-occurring tags via OM_COOCCURS edges with PMI scores.

Shows tags that frequently appear together with this tag across memories.
PMI (Pointwise Mutual Information) indicates how strongly tags are associated.

Parameters:
- tag_key: The tag key to find related tags for (required)
- min_count: Minimum co-occurrence count (default: 1)
- limit: Max tags to return (default: 20, max: 50)

Returns:
- tag: the queried tag key
- related[]: each has {key, count, pmi}
- count: number of related tags
- graph_enabled: whether feature is active
""")
async def graph_related_tags(
    tag_key: str,
    min_count: int = 1,
    limit: int = 20,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "tag": tag_key,
            "related": [],
            "count": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 20)), 50)
    min_count = max(1, int(min_count or 1))

    try:
        related = get_related_tags_from_graph(
            tag_key=tag_key,
            user_id=uid,
            min_count=min_count,
            limit=limit,
        )

        return json.dumps({
            "tag": tag_key,
            "related": related,
            "count": len(related),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_related_tags: {e}")
        return json.dumps({"error": f"Error getting related tags: {str(e)}"})


@mcp.tool(description="""Find and optionally merge duplicate entities.

Identifies entity variants that should be merged (e.g., "Matthias", "matthias", "matthias_coers").

Parameters:
- dry_run: If true (default), only show what would be merged without changes
- auto: If true, automatically merge all detected duplicates
- canonical: For manual merge: the target entity name
- variants: For manual merge: comma-separated list of variant names to merge

Returns:
- duplicates: List of detected duplicate groups (if dry_run or no merge specified)
- merged: Count of merged entities (if merge was performed)

Examples:
- Detect duplicates: graph_normalize_entities(dry_run=true)
- Auto-merge all: graph_normalize_entities(auto=true, dry_run=false)
- Manual merge: graph_normalize_entities(canonical="matthias_coers", variants="Matthias,matthias", dry_run=false)
""")
async def graph_normalize_entities(
    dry_run: bool = True,
    auto: bool = False,
    canonical: str = None,
    variants: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"duplicates": [], "graph_enabled": False})

    try:
        # If just checking duplicates
        if dry_run and not auto and not canonical:
            duplicates = find_duplicate_entities_in_graph(uid)
            return json.dumps({
                "duplicates": duplicates,
                "count": len(duplicates),
                "dry_run": True,
                "graph_enabled": True,
            }, default=str)

        # Perform normalization
        variant_list = [v.strip() for v in variants.split(",")] if variants else None

        result = normalize_entities_in_graph(
            user_id=uid,
            canonical_name=canonical,
            variant_names=variant_list,
            auto=auto,
            dry_run=dry_run,
        )

        result["graph_enabled"] = True
        return json.dumps(result, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_normalize_entities: {e}")
        return json.dumps({"error": f"Error normalizing entities: {str(e)}"})


@mcp.tool(description="""Semantic entity normalization with multi-phase duplicate detection.

Extends basic case normalization with advanced similarity detection:
1. String similarity (Levenshtein/fuzzy matching, e.g., "el_juego" ≈ "eljuego")
2. Prefix/suffix matching (e.g., "marie" → "marie_schubenz")
3. Domain normalization (e.g., "eljuego.community" → "el_juego")

Parameters:
- mode: Detection mode - "detect" (find duplicates), "preview" (dry-run merge), "execute" (perform merge)
- threshold: Minimum confidence for match (0.0-1.0, default: 0.7). Lower = more matches.
- canonical: For manual merge: target entity name
- variants: For manual merge: comma-separated variant names

Returns (mode=detect):
- duplicates[]: Each group has {canonical, variants, confidence, sources}
- count: Number of duplicate groups

Returns (mode=preview/execute):
- merge_groups: Number of groups to merge
- total_variants_merged: Total variant count
- total_edges_migrated: Edges that were/will be migrated
- merges[]: Details per group

Examples:
- Detect: graph_normalize_entities_semantic(mode="detect")
- Lower threshold: graph_normalize_entities_semantic(mode="detect", threshold=0.6)
- Preview all: graph_normalize_entities_semantic(mode="preview")
- Execute all: graph_normalize_entities_semantic(mode="execute")
- Manual merge: graph_normalize_entities_semantic(mode="execute", canonical="el_juego", variants="eljuego,el-juego")
""")
async def graph_normalize_entities_semantic(
    mode: str = "detect",
    threshold: float = 0.7,
    canonical: str = None,
    variants: str = None,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({"duplicates": [], "graph_enabled": False})

    if mode not in ["detect", "preview", "execute"]:
        return json.dumps({"error": f"Invalid mode: {mode}. Use 'detect', 'preview', or 'execute'"})

    try:
        from app.graph.graph_ops import find_semantic_duplicates, normalize_entities_semantic

        if mode == "detect":
            # Only detect duplicates
            duplicates = await find_semantic_duplicates(uid, threshold=threshold)
            return json.dumps({
                "duplicates": duplicates,
                "count": len(duplicates),
                "threshold": threshold,
                "mode": "detect",
                "graph_enabled": True,
            }, default=str)

        # Preview or execute merge
        variant_list = [v.strip() for v in variants.split(",")] if variants else None
        dry_run = mode == "preview"

        if canonical and variant_list:
            # Manual merge
            result = await normalize_entities_semantic(
                user_id=uid,
                canonical=canonical,
                variants=variant_list,
                threshold=threshold,
                dry_run=dry_run,
            )
        else:
            # Auto-detect and merge
            result = await normalize_entities_semantic(
                user_id=uid,
                auto=True,
                threshold=threshold,
                dry_run=dry_run,
            )

        result["mode"] = mode
        result["graph_enabled"] = True
        return json.dumps(result, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_normalize_entities_semantic: {e}")
        return json.dumps({"error": f"Error in semantic normalization: {str(e)}"})


@mcp.tool(description="""Get typed relationships for an entity.

Shows semantic relationships extracted from memories, not just co-mentions.
Examples: "Grischa -[schwester_von]-> Julia", "Grischa -[arbeitet_bei]-> CloudFactory"

Parameters:
- entity_name: Name of the entity (required)
- relation_types: Optional comma-separated relation types to filter
                  (e.g., "schwester_von,bruder_von")
- category: Optional category to filter (family, social, work, location, creative, membership)
- direction: "outgoing", "incoming", or "both" (default: both)
- limit: Max relations to return (default: 50)

Returns:
- entity: the queried entity
- relations[]: each has {target, type, direction, memory_id, count}
- count: number of relations
""")
async def graph_entity_relations(
    entity_name: str,
    relation_types: str = None,
    category: str = None,
    direction: str = "both",
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "relations": [],
            "count": 0,
            "graph_enabled": False,
        })

    limit = min(max(1, int(limit or 50)), 200)

    # Build type filter
    type_filter = []
    if relation_types:
        type_filter = [t.strip() for t in relation_types.split(",")]

    try:
        relations = get_entity_relations_from_graph(
            entity_name=entity_name,
            user_id=uid,
            relation_types=type_filter if type_filter else None,
            category=category,
            direction=direction,
            limit=limit,
        )

        return json.dumps({
            "entity": entity_name,
            "relations": relations,
            "count": len(relations),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_entity_relations: {e}")
        return json.dumps({"error": f"Error getting entity relations: {str(e)}"})


@mcp.tool(description="""Get biographical timeline for a person or user.

Shows chronological events like residences, projects, work history.

Parameters:
- entity_name: Optional person name (if omitted, shows all events for user)
- event_types: Optional comma-separated types (residence, education, work, project, relationship, health, travel, milestone)
- start_year: Optional start year filter
- end_year: Optional end year filter
- limit: Max events (default: 50)

Returns:
- events[]: chronologically sorted events with dates and descriptions
- count: number of events

Example:
- Get Grischa's timeline: graph_biography_timeline(entity_name="grischa")
- Get all projects: graph_biography_timeline(event_types="project")
""")
async def graph_biography_timeline(
    entity_name: str = None,
    event_types: str = None,
    start_year: int = None,
    end_year: int = None,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    if not is_graph_enabled():
        return json.dumps({
            "entity": entity_name,
            "events": [],
            "count": 0,
            "graph_enabled": False,
        })

    try:
        type_list = [t.strip() for t in event_types.split(",")] if event_types else None

        events = get_biography_timeline_from_graph(
            user_id=uid,
            entity_name=entity_name,
            event_types=type_list,
            start_year=start_year,
            end_year=end_year,
            limit=min(max(1, limit or 50), 200),
        )

        return json.dumps({
            "entity": entity_name,
            "events": events,
            "count": len(events),
            "graph_enabled": True,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in graph_biography_timeline: {e}")
        return json.dumps({"error": f"Error getting timeline: {str(e)}"})


# =============================================================================
# CODE INTELLIGENCE TOOLS
# =============================================================================


def get_code_toolkit():
    """Lazy-load code toolkit to avoid import-time failures."""
    from app.code_toolkit import get_code_toolkit as _get_toolkit
    return _get_toolkit()


def _create_code_meta(request_id: str = None, degraded: bool = False, missing: list = None, error: str = None) -> dict:
    """Create metadata dict for code tool responses."""
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "degraded_mode": degraded,
        "missing_sources": missing or [],
        "error": error,
    }


@mcp.tool(description="""Index a local codebase into the CODE_* graph and search index.

Required parameters:
- repo_id: Repository ID (required)
- root_path: Filesystem path to the repository root (required)

Optional parameters:
- index_name: OpenSearch index name (default: "code")
- reset: Clear existing repo data before indexing (default: false)
- max_files: Maximum number of files to index
- include_api_boundaries: Enable API boundary detection (default: true)
- async_mode: Run indexing in the background and return a job_id (default: false)

Returns:
- repo_id: Repository ID
- files_indexed: Files successfully indexed
- files_failed: Files that failed indexing
- symbols_indexed: Symbols added to the graph
- documents_indexed: Documents added to OpenSearch
- call_edges_indexed: CALLS edges inferred
- duration_ms: Total indexing time
- meta: Response metadata
""")
async def index_codebase(
    repo_id: str,
    root_path: str,
    index_name: str = "code",
    reset: bool = False,
    max_files: int = None,
    include_api_boundaries: bool = True,
    async_mode: bool = False,
) -> str:
    """Index a local repository for code intelligence."""
    scope_error = _check_tool_scope("graph:write")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    root = resolve_repo_root(root_path, repo_id=repo_id)
    if not root.exists() or not root.is_dir():
        return json.dumps({
            "error": "root_path not found",
            "hint": "If running in Docker, use the container path (e.g., /usr/src/<repo>) or set OPENMEMORY_REPO_ROOT.",
        })

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "repo_id": repo_id,
            "files_indexed": 0,
            "files_failed": 0,
            "symbols_indexed": 0,
            "documents_indexed": 0,
            "call_edges_indexed": 0,
            "duration_ms": 0.0,
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    missing_sources = []
    if not toolkit.is_available("opensearch"):
        missing_sources.append("opensearch")
    if not toolkit.is_available("embedding"):
        missing_sources.append("embedding")

    try:
        from indexing.code_indexer import CodeIndexingService

        meta = _create_code_meta(
            degraded=bool(missing_sources),
            missing=missing_sources,
            error="Some backends unavailable" if missing_sources else None,
        )

        indexer = CodeIndexingService(
            root_path=root,
            repo_id=repo_id,
            graph_driver=toolkit.neo4j_driver,
            opensearch_client=toolkit.opensearch_client if toolkit.is_available("opensearch") else None,
            embedding_service=toolkit.embedding_service if toolkit.is_available("embedding") else None,
            index_name=index_name or "code",
            include_api_boundaries=include_api_boundaries,
        )

        if async_mode:
            # Use persistent job queue - worker will pick up the job
            from app.database import SessionLocal
            from app.services.job_queue_service import IndexingJobQueueService, QueueFullError

            uid = user_id_var.get("anonymous")
            db = SessionLocal()
            try:
                import os
                max_queued_jobs = int(os.getenv("MAX_QUEUED_JOBS", "100"))
                valkey_client = None
                try:
                    import redis

                    host = os.getenv("VALKEY_HOST", "valkey")
                    port = int(os.getenv("VALKEY_PORT", "6379"))
                    client = redis.Redis(host=host, port=port, socket_timeout=5)
                    client.ping()
                    valkey_client = client
                except Exception:
                    valkey_client = None

                queue_service = IndexingJobQueueService(
                    db=db,
                    valkey_client=valkey_client,
                    max_queued_jobs=max_queued_jobs,
                )
                try:
                    job_uuid = queue_service.create_job(
                        repo_id=repo_id,
                        root_path=str(root),
                        index_name=index_name or "code",
                        requested_by=uid,
                        request={
                            "max_files": max_files,
                            "reset": reset,
                            "include_api_boundaries": include_api_boundaries,
                        },
                        meta=meta,
                    )
                    return json.dumps({
                        "repo_id": repo_id,
                        "job_id": str(job_uuid),
                        "status": "queued",
                        "meta": meta,
                    })
                except QueueFullError as e:
                    return json.dumps({
                        "error": str(e),
                        "meta": _create_code_meta(
                            degraded=True,
                            missing=missing_sources,
                            error=str(e),
                        ),
                    })
            finally:
                db.close()

        summary = indexer.index_repository(
            max_files=max_files,
            reset=reset,
        )

        return json.dumps({
            "repo_id": summary.repo_id,
            "files_indexed": summary.files_indexed,
            "files_failed": summary.files_failed,
            "symbols_indexed": summary.symbols_indexed,
            "documents_indexed": summary.documents_indexed,
            "call_edges_indexed": summary.call_edges_indexed,
            "duration_ms": summary.duration_ms,
            "meta": meta,
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in index_codebase: {e}")
        return json.dumps({
            "repo_id": repo_id,
            "files_indexed": 0,
            "files_failed": 0,
            "symbols_indexed": 0,
            "documents_indexed": 0,
            "call_edges_indexed": 0,
            "duration_ms": 0.0,
            "meta": _create_code_meta(
                degraded=True,
                missing=missing_sources,
                error=str(e),
            ),
        })


@mcp.tool(description="""Get the status of a background code indexing job.

Required parameters:
- job_id: Indexing job ID (required)

Returns:
- job_id
- repo_id
- status: queued | running | succeeded | failed
- timestamps and summary/error when available
""")
async def index_codebase_status(job_id: str) -> str:
    """Fetch status for a background code indexing job."""
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    import uuid as uuid_mod

    # Try persistent store first
    try:
        job_uuid = uuid_mod.UUID(job_id)
        from app.database import SessionLocal
        from app.services.job_queue_service import IndexingJobQueueService

        db = SessionLocal()
        try:
            queue_service = IndexingJobQueueService(db=db, valkey_client=None)
            job = queue_service.get_job(job_uuid)
            if job:
                if not _can_access_job(job, "admin:read"):
                    return json.dumps({"error": "Access denied"})
                return json.dumps(job, default=str)
        finally:
            db.close()
    except ValueError:
        pass  # Invalid UUID, try legacy store
    except Exception:
        pass  # Persistent store error, try legacy store

    # Fallback to legacy in-memory store
    job = get_index_job(job_id)
    if not job:
        return json.dumps({"error": "job_id not found"})

    return json.dumps(job, default=str)


@mcp.tool(description="""Cancel a running or queued indexing job.

Required parameters:
- job_id: Indexing job ID (required)

Returns:
- job_id
- status: "cancel_requested" on success
- error: Error message if job not found

Examples:
- index_codebase_cancel(job_id="<job-id>")
""")
async def index_codebase_cancel(job_id: str) -> str:
    """Request cancellation of an indexing job."""
    scope_error = _check_tool_scope("graph:write")
    if scope_error:
        return scope_error

    import uuid as uuid_mod
    try:
        job_uuid = uuid_mod.UUID(job_id)
    except ValueError:
        return json.dumps({"error": "Invalid job_id format"})

    # Try persistent store
    from app.database import SessionLocal
    from app.services.job_queue_service import IndexingJobQueueService

    try:
        db = SessionLocal()
        try:
            queue_service = IndexingJobQueueService(db=db, valkey_client=None)
            job = queue_service.get_job(job_uuid)
            if not job:
                return json.dumps({"error": "job_id not found"})

            if not _can_access_job(job, "admin:write"):
                return json.dumps({"error": "Access denied"})

            success = queue_service.cancel_job(job_uuid)
            if success:
                return json.dumps({"job_id": job_id, "status": "cancel_requested"})
            else:
                return json.dumps({"error": "Failed to request cancellation"})
        finally:
            db.close()
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(description="""Search code using tri-hybrid retrieval (lexical + semantic + graph).

Combines lexical, semantic, and graph-based search for comprehensive code discovery.

Required parameters:
- query: Search query text (required)

Optional parameters:
- repo_id: Filter by repository ID
- language: Filter by programming language
- limit: Maximum results (default: 10, max: 100)

Returns:
- results[]: Array of code hits with symbol info, scores, and snippets
- meta: Response metadata including degraded_mode indicator

Examples:
- search_code_hybrid(query="database connection pooling")
- search_code_hybrid(query="authentication handler", language="python", limit=20)
""")
async def search_code_hybrid(
    query: str,
    repo_id: str = None,
    language: str = None,
    limit: int = 10,
) -> str:
    """Search code using tri-hybrid retrieval."""
    # Check scope
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    # Check if search backend is available
    if not toolkit.is_available("opensearch") and not toolkit.is_available("trihybrid"):
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["opensearch"],
                error="Search backend unavailable",
            ),
        })

    if not toolkit.search_tool:
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("search_tool"),
            ),
        })

    try:
        from tools.search_code_hybrid import SearchCodeHybridInput
        import dataclasses

        input_data = SearchCodeHybridInput(
            query=query,
            repo_id=repo_id,
            language=language,
            limit=min(max(1, limit or 10), 100),
        )

        result = toolkit.search_tool.search(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in search_code_hybrid: {e}")
        return json.dumps({
            "results": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=str(e),
            ),
        })


@mcp.tool(description="""Explain a code symbol with full context.

Get detailed explanation of a code symbol including callers, callees, and usages.

Required parameters:
- symbol_id: ID of the symbol to explain (required)

Optional parameters:
- depth: Analysis depth (default: 2, max: 5)
- include_callers: Include functions that call this symbol (default: true)
- include_callees: Include functions called by this symbol (default: true)

Returns:
- explanation: Symbol details with call graph context
- meta: Response metadata

Examples:
- explain_code(symbol_id="sym-abc123")
- explain_code(symbol_id="sym-abc123", depth=3)
""")
async def explain_code(
    symbol_id: str,
    depth: int = 2,
    include_callers: bool = True,
    include_callees: bool = True,
) -> str:
    """Explain a code symbol with context."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.explain_tool:
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("explain_tool"),
            ),
        })

    try:
        from tools.explain_code import ExplainCodeConfig
        import dataclasses

        config = ExplainCodeConfig(
            depth=min(max(1, depth or 2), 5),
            include_callers=include_callers,
            include_callees=include_callees,
        )

        result = toolkit.explain_tool.explain(symbol_id, config)
        return json.dumps({
            "explanation": dataclasses.asdict(result) if result else None,
            "meta": _create_code_meta(),
        }, default=str)

    except Exception as e:
        logging.exception(f"Error in explain_code: {e}")
        return json.dumps({
            "explanation": None,
            "meta": _create_code_meta(
                degraded=True,
                error=str(e),
            ),
        })


@mcp.tool(description="""Find functions that call a given symbol with automatic fallback.

Traverse the call graph to find all callers of a symbol. Uses a 4-stage
fallback cascade when the symbol is not found in the graph:

1. Graph-based search (SCIP index) - Primary
2. Grep-based pattern matching - First fallback
3. Semantic search (embeddings) - Second fallback
4. Structured error with suggestions - Final fallback

Required parameters:
- repo_id: Repository ID (required)
- symbol_name OR symbol_id: The symbol to find callers for (one required)

Optional parameters:
- depth: Traversal depth (default: 2, max: 5)
- use_fallback: Enable fallback cascade (default: true)

Returns:
- nodes[]: Graph nodes (functions/methods)
- edges[]: Graph edges (CALLS relationships)
- meta: Response metadata with fallback info
- suggestions[]: Fallback suggestions (when fallback stage 4)

Fallback meta fields:
- degraded_mode: true when using fallback
- fallback_stage: 1-4 indicating which stage succeeded
- fallback_strategy: "grep" | "semantic_search" | "structured_error"
- warning: Describes accuracy caveats

Examples:
- find_callers(repo_id="repo-123", symbol_name="main")
- find_callers(repo_id="repo-123", symbol_id="sym-abc", depth=3)
""")
async def find_callers(
    repo_id: str,
    symbol_name: str = None,
    symbol_id: str = None,
    depth: int = 2,
    use_fallback: bool = True,
) -> str:
    """Find functions that call a given symbol with automatic fallback."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    if not symbol_name and not symbol_id:
        return json.dumps({"error": "Either symbol_name or symbol_id is required"})

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.callers_tool:
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("callers_tool"),
            ),
        })

    try:
        from tools.call_graph import CallGraphInput, SymbolNotFoundError
        import dataclasses

        input_data = CallGraphInput(
            repo_id=repo_id,
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            depth=min(max(1, depth or 2), 5),
        )

        # Use fallback tool if enabled
        if use_fallback:
            try:
                from tools.fallback_find_callers import (
                    FallbackFindCallersTool,
                    GrepTool,
                )

                # Create fallback tool with available dependencies
                fallback_tool = FallbackFindCallersTool(
                    graph_driver=toolkit.callers_tool.graph_driver,
                    grep_tool=GrepTool(),  # Basic grep tool
                    search_tool=toolkit.search_tool if hasattr(toolkit, 'search_tool') else None,
                )

                result = fallback_tool.find(input_data)

                # Add fallback info to response
                response = dataclasses.asdict(result)
                if fallback_tool.fallback_used:
                    response["_fallback_info"] = {
                        "fallback_used": True,
                        "fallback_stage": fallback_tool.fallback_stage,
                        "stage_timings_ms": fallback_tool.get_stage_timings(),
                        "warning": "Results from fallback strategy - verify accuracy",
                    }

                return json.dumps(response, default=str)

            except ImportError:
                logging.warning("FallbackFindCallersTool not available, using basic tool")
                # Fall through to basic tool

        # Basic tool (no fallback)
        result = toolkit.callers_tool.find(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except SymbolNotFoundError as e:
        # Return structured error with suggestions
        logging.info(f"Symbol not found in find_callers: {e}")
        error_response = e.to_dict()
        error_response["nodes"] = []
        error_response["edges"] = []
        error_response["meta"] = _create_code_meta(
            degraded=True,
            error=str(e),
        )
        return json.dumps(error_response, default=str)

    except Exception as e:
        logging.exception(f"Error in find_callers: {e}")
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Find functions called by a given symbol.

Traverse the call graph to find all functions that a symbol calls.

Required parameters:
- repo_id: Repository ID (required)
- symbol_name OR symbol_id: The symbol to find callees for (one required)

Optional parameters:
- depth: Traversal depth (default: 2, max: 5)

Returns:
- nodes[]: Graph nodes (functions/methods)
- edges[]: Graph edges (CALLS relationships)
- meta: Response metadata

Examples:
- find_callees(repo_id="repo-123", symbol_name="main")
- find_callees(repo_id="repo-123", symbol_id="sym-abc", depth=3)
""")
async def find_callees(
    repo_id: str,
    symbol_name: str = None,
    symbol_id: str = None,
    depth: int = 2,
) -> str:
    """Find functions called by a given symbol."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    if not symbol_name and not symbol_id:
        return json.dumps({"error": "Either symbol_name or symbol_id is required"})

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.callees_tool:
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("callees_tool"),
            ),
        })

    try:
        from tools.call_graph import CallGraphInput
        import dataclasses

        input_data = CallGraphInput(
            repo_id=repo_id,
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            depth=min(max(1, depth or 2), 5),
        )

        result = toolkit.callees_tool.find(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in find_callees: {e}")
        return json.dumps({
            "nodes": [],
            "edges": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze the impact of code changes.

Determine which files and symbols are affected by code changes.

Required parameters:
- repo_id: Repository ID (required)

Optional parameters:
- changed_files: List of changed file paths
- symbol_id: Changed symbol ID
- max_depth: Maximum traversal depth (default: 3, max: 10)
- include_cross_language: Include cross-language dependencies (default: false)

Returns:
- affected_files[]: List of affected files with impact scores
- meta: Response metadata

Examples:
- impact_analysis(repo_id="repo-123", changed_files=["src/utils.py"])
- impact_analysis(repo_id="repo-123", symbol_id="sym-abc", max_depth=5)
""")
async def impact_analysis(
    repo_id: str,
    changed_files: list = None,
    symbol_id: str = None,
    max_depth: int = 3,
    include_cross_language: bool = False,
) -> str:
    """Analyze the impact of code changes."""
    # Check scope
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.is_available("neo4j"):
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=["neo4j"],
                error="Graph backend unavailable",
            ),
        })

    if not toolkit.impact_tool:
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("impact_tool"),
            ),
        })

    try:
        from tools.impact_analysis import ImpactInput
        import dataclasses

        input_data = ImpactInput(
            repo_id=repo_id,
            changed_files=changed_files or [],
            symbol_id=symbol_id,
            include_cross_language=include_cross_language,
            max_depth=min(max(1, max_depth or 3), 10),
        )

        result = toolkit.impact_tool.analyze(input_data)
        return json.dumps(dataclasses.asdict(result), default=str)

    except Exception as e:
        logging.exception(f"Error in impact_analysis: {e}")
        return json.dumps({
            "affected_files": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze code changes for ADR (Architecture Decision Record) recommendation.

Detects significant architectural changes that warrant documentation.

Required parameters:
- changes: List of change objects with file_path, change_type, diff, added_lines, removed_lines

Optional parameters:
- min_confidence: Minimum confidence threshold for ADR recommendation (default: 0.7)

Returns:
- should_create_adr: Whether an ADR should be created
- confidence: Confidence score (0.0-1.0)
- triggered_heuristics: List of detected change types (dependency, api_change, etc.)
- reasons: Human-readable reasons for the recommendation
- generated_adr: Generated ADR content (when should_create_adr is True)
- code_links: Links to relevant code sections (optional)
- impact_analysis: Impact analysis results (optional)
- meta: Response metadata

Examples:
- adr_automation(changes=[{"file_path": "requirements.txt", "change_type": "modified", "diff": "+redis>=4.0.0"}])
- adr_automation(changes=[...], min_confidence=0.8)
""")
async def adr_automation(
    changes: list,
    min_confidence: float = 0.7,
) -> str:
    """Analyze code changes for ADR recommendation."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    toolkit = get_code_toolkit()

    if not toolkit.adr_tool:
        return json.dumps({
            "should_create_adr": False,
            "confidence": 0.0,
            "triggered_heuristics": [],
            "reasons": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("adr_tool") or "ADR tool not available",
            ),
        }, default=str)

    try:
        result = toolkit.adr_tool.execute({
            "changes": changes,
            "min_confidence": min_confidence,
        })

        # Build response with meta
        response = {
            "should_create_adr": result.get("should_create_adr", False),
            "confidence": result.get("confidence", 0.0),
            "triggered_heuristics": result.get("triggered_heuristics", []),
            "reasons": result.get("reasons", []),
            "meta": _create_code_meta(),
        }

        # Include optional fields when present
        if result.get("should_create_adr") and result.get("generated_adr"):
            response["generated_adr"] = result["generated_adr"]
        if result.get("code_links"):
            response["code_links"] = result["code_links"]
        if result.get("impact_analysis"):
            response["impact_analysis"] = result["impact_analysis"]

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in adr_automation: {e}")
        return json.dumps({
            "should_create_adr": False,
            "confidence": 0.0,
            "triggered_heuristics": [],
            "reasons": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Generate test cases for a code symbol or file.

Creates comprehensive test cases based on code analysis.

Required parameters (at least one):
- symbol_id: SCIP symbol ID to generate tests for
- file_path: File path to generate tests for

Optional parameters:
- framework: Test framework to use (default: "pytest")
- include_edge_cases: Include edge case tests (default: true)
- include_error_cases: Include error handling tests (default: true)

Returns:
- symbol_id: The symbol that tests were generated for
- symbol_name: Human-readable symbol name
- test_cases: List of generated test cases with name, description, category, code
- file_content: Rendered test file content
- meta: Response metadata

Examples:
- test_generation(symbol_id="scip-python myapp module/my_function.")
- test_generation(file_path="/path/to/module.py", framework="pytest")
- test_generation(symbol_id="...", include_edge_cases=true, include_error_cases=false)
""")
async def test_generation(
    symbol_id: str = None,
    file_path: str = None,
    framework: str = "pytest",
    include_edge_cases: bool = True,
    include_error_cases: bool = True,
) -> str:
    """Generate test cases for a code symbol or file."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    # Validate input - at least one of symbol_id or file_path required
    if not symbol_id and not file_path:
        return json.dumps({
            "error": "Either symbol_id or file_path is required",
        })

    toolkit = get_code_toolkit()

    if not toolkit.test_gen_tool:
        return json.dumps({
            "symbol_id": symbol_id,
            "symbol_name": None,
            "test_cases": [],
            "file_content": "",
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("test_gen_tool") or "Test generation tool not available",
            ),
        })

    try:
        # Build input for the tool (symbol_id takes precedence)
        input_data = {
            "framework": framework,
            "include_edge_cases": include_edge_cases,
            "include_error_cases": include_error_cases,
        }
        if symbol_id:
            input_data["symbol_id"] = symbol_id
        if file_path:
            input_data["file_path"] = file_path

        result = toolkit.test_gen_tool.execute(input_data)

        # Build response with meta
        response = {
            "symbol_id": result.get("symbol_id", symbol_id),
            "symbol_name": result.get("symbol_name"),
            "test_cases": result.get("test_cases", []),
            "file_content": result.get("file_content", ""),
            "meta": _create_code_meta(),
        }

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in test_generation: {e}")
        return json.dumps({
            "symbol_id": symbol_id,
            "symbol_name": None,
            "test_cases": [],
            "file_content": "",
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


@mcp.tool(description="""Analyze a pull request for issues, impact, and recommendations.

Provides comprehensive PR analysis including security checks, convention compliance,
impact analysis, and ADR recommendations.

Required parameters:
- repo_id: Repository ID (required, non-empty)
- diff: The PR diff content (required)

Optional parameters:
- pr_number: Pull request number
- title: PR title
- body: PR description body
- check_impact: Run impact analysis (default: true)
- check_adr: Check for ADR recommendations (default: true)
- check_security: Run security checks (default: true)
- check_conventions: Check coding conventions (default: true)

Returns:
- summary: PR summary with files_changed, additions, deletions, languages, main_areas,
          complexity_score, affected_files, suggested_adr, adr_reason
- issues: List of detected issues with severity, category, file_path, line_number,
          message, suggestion
- meta: Response metadata

Examples:
- pr_analysis(repo_id="my-repo", diff="diff --git a/file.py...")
- pr_analysis(repo_id="my-repo", diff="...", pr_number=123, title="Add feature")
- pr_analysis(repo_id="my-repo", diff="...", check_security=true, check_adr=false)
""")
async def pr_analysis(
    repo_id: str,
    diff: str,
    pr_number: int = None,
    title: str = None,
    body: str = None,
    check_impact: bool = True,
    check_adr: bool = True,
    check_security: bool = True,
    check_conventions: bool = True,
) -> str:
    """Analyze a pull request for issues and recommendations."""
    # Check scopes - needs both search:read and graph:read
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    scope_error = _check_tool_scope("graph:read")
    if scope_error:
        return scope_error

    # Validate required inputs
    if not repo_id:
        return json.dumps({
            "error": "repo_id is required and cannot be empty",
        })

    toolkit = get_code_toolkit()

    if not toolkit.pr_analysis_tool:
        return json.dumps({
            "summary": None,
            "issues": [],
            "meta": _create_code_meta(
                degraded=True,
                missing=toolkit.get_missing_sources(),
                error=toolkit.get_error("pr_analysis_tool") or "PR analysis tool not available",
            ),
        }, default=str)

    try:
        # Build input for the tool
        input_data = {
            "repo_id": repo_id,
            "diff": diff,
            "check_impact": check_impact,
            "check_adr": check_adr,
            "check_security": check_security,
            "check_conventions": check_conventions,
        }
        if pr_number is not None:
            input_data["pr_number"] = pr_number
        if title:
            input_data["title"] = title
        if body:
            input_data["body"] = body

        result = toolkit.pr_analysis_tool.execute(input_data)

        # Build response with meta
        response = {
            "summary": result.get("summary"),
            "issues": result.get("issues", []),
            "meta": _create_code_meta(
                request_id=result.get("request_id"),
            ),
        }

        return json.dumps(response, default=str)

    except Exception as e:
        logging.exception(f"Error in pr_analysis: {e}")
        return json.dumps({
            "summary": None,
            "issues": [],
            "meta": _create_code_meta(degraded=True, error=str(e)),
        })


# =============================================================================
# BUSINESS CONCEPT TOOLS (Separate MCP endpoint - /concepts/claude/sse/{user_id})
# =============================================================================


@concept_mcp.tool(description="""Extract business concepts from a memory or text.

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- memory_id: UUID of the memory to extract from (required if content not provided)
- content: Text content to extract from (required if memory_id not provided)
- category: Optional category for concept scoping
- store: Whether to store extracted concepts in graph (default: true)

Returns:
- entities: Extracted business entities with types and importance
- concepts: Extracted business concepts with types and confidence
- summary: Brief summary of main topics
- language: Detected language (en, de, mixed)
- stored_entities: Count of entities stored (if store=true)
- stored_concepts: Count of concepts stored (if store=true)
""")
async def extract_business_concepts(
    memory_id: str = None,
    content: str = None,
    category: str = None,
    store: bool = True,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    # Check if concepts are enabled
    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled. Set BUSINESS_CONCEPTS_ENABLED=true",
            "enabled": False,
        })

    if not content and not memory_id:
        return json.dumps({"error": "Either memory_id or content is required"})

    try:
        # If memory_id provided but no content, fetch content from DB
        if memory_id and not content:
            db = SessionLocal()
            try:
                memory = db.query(Memory).filter(Memory.id == uuid.UUID(memory_id)).first()
                if not memory:
                    return json.dumps({"error": f"Memory {memory_id} not found"})
                content = memory.content
                if category is None and isinstance(memory.metadata_, dict):
                    category = memory.metadata_.get("category")
            finally:
                db.close()

        # Use provided memory_id or generate a temporary one
        effective_memory_id = memory_id or str(uuid.uuid4())

        from app.utils.concept_extractor import extract_from_memory

        result = extract_from_memory(
            memory_id=effective_memory_id,
            user_id=uid,
            content=content,
            category=category,
            store_in_graph=store,
        )

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error extracting concepts: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""List business concepts with optional filters.

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- category: Filter by category
- concept_type: Filter by type (causal, pattern, comparison, trend, contradiction, hypothesis, fact)
- min_confidence: Minimum confidence threshold (0.0-1.0)
- limit: Maximum results (default: 50)
- offset: Pagination offset

Returns:
- concepts[]: List of concepts with name, type, confidence, category, etc.
- count: Number of concepts returned
""")
async def list_business_concepts(
    category: str = None,
    concept_type: str = None,
    min_confidence: float = None,
    limit: int = 50,
    offset: int = 0,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import list_concepts

        concepts = list_concepts(
            user_id=uid,
            category=category,
            concept_type=concept_type,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )

        return json.dumps({
            "concepts": concepts,
            "count": len(concepts),
            "enabled": True,
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error listing concepts: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Get a specific business concept by name.

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- name: Name of the concept (required)

Returns:
- Concept details including name, type, confidence, evidence, etc.
""")
async def get_business_concept(
    name: str,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import get_concept

        concept = get_concept(user_id=uid, name=name)

        if concept:
            return json.dumps(concept, indent=2, default=str)
        else:
            return json.dumps({"error": f"Concept '{name}' not found"})

    except Exception as e:
        logging.exception(f"Error getting concept: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Search business concepts by text query.

Uses semantic (vector) search when embeddings are enabled, falling back to
full-text search otherwise. Semantic search understands meaning, not just keywords.

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- query: Search query (required)
- limit: Maximum results (default: 20)
- use_semantic: Use semantic search if available (default: true)
- min_score: Minimum similarity score 0-1 for semantic search (default: 0.5)

Returns:
- concepts[]: Matching concepts with search/similarity scores
- count: Number of results
- search_type: "semantic" or "fulltext"
""")
async def search_business_concepts(
    query: str,
    limit: int = 20,
    use_semantic: bool = True,
    min_score: float = 0.5,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import (
            search_concepts,
            semantic_search_concepts,
            is_vector_search_enabled,
        )

        search_type = "fulltext"

        # Use semantic search if requested and available
        if use_semantic and is_vector_search_enabled():
            concepts = semantic_search_concepts(
                user_id=uid,
                query=query,
                top_k=limit,
                min_score=min_score,
            )
            search_type = "semantic"
        else:
            concepts = search_concepts(
                user_id=uid,
                query=query,
                limit=limit,
            )

        return json.dumps({
            "query": query,
            "concepts": concepts,
            "count": len(concepts),
            "search_type": search_type,
            "embedding_enabled": is_vector_search_enabled(),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error searching concepts: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Find concepts semantically similar to a given concept.

Uses vector embeddings to find concepts that are semantically related,
even if they don't share exact keywords. Useful for:
- Discovering related concepts
- Finding potential duplicates
- Building concept clusters

Requires BUSINESS_CONCEPTS_ENABLED=true and BUSINESS_CONCEPTS_EMBEDDING_ENABLED=true.

Parameters:
- concept_name: Name of the seed concept (required)
- top_k: Number of similar concepts to return (default: 5)
- find_duplicates: If true, uses higher threshold to find potential duplicates (default: false)

Returns:
- similar_concepts[]: List of similar concepts with similarity scores
- count: Number of results
- seed_concept: The concept used as seed
""")
async def find_similar_business_concepts(
    concept_name: str,
    top_k: int = 5,
    find_duplicates: bool = False,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import (
            find_similar_concepts,
            find_concept_duplicates,
            is_vector_search_enabled,
        )

        if not is_vector_search_enabled():
            return json.dumps({
                "error": "Vector embeddings not enabled",
                "help": "Set BUSINESS_CONCEPTS_EMBEDDING_ENABLED=true to use this feature",
                "embedding_enabled": False,
            })

        if find_duplicates:
            similar = find_concept_duplicates(
                user_id=uid,
                concept_name=concept_name,
            )
        else:
            similar = find_similar_concepts(
                user_id=uid,
                concept_name=concept_name,
                top_k=top_k,
            )

        return json.dumps({
            "seed_concept": concept_name,
            "similar_concepts": similar,
            "count": len(similar),
            "mode": "duplicates" if find_duplicates else "similar",
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error finding similar concepts: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""List business entities (companies, people, products, etc.)

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- entity_type: Filter by type (company, person, product, market, metric, business_model, technology, strategy)
- min_importance: Minimum importance threshold (0.0-1.0)
- limit: Maximum results (default: 50)

Returns:
- entities[]: List of business entities
- count: Number of entities returned
""")
async def list_business_entities(
    entity_type: str = None,
    min_importance: float = None,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import list_business_entities as list_entities

        entities = list_entities(
            user_id=uid,
            entity_type=entity_type,
            min_importance=min_importance,
            limit=limit,
        )

        return json.dumps({
            "entities": entities,
            "count": len(entities),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error listing entities: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Get the business concept network graph for visualization.

Requires BUSINESS_CONCEPTS_ENABLED=true environment variable.

Parameters:
- concept_name: Optional seed concept (if omitted, returns full network)
- depth: Traversal depth 1-3 (default: 2)
- limit: Maximum nodes (default: 50)

Returns:
- nodes[]: Concept and entity nodes with properties
- edges[]: Relationships between nodes
""")
async def get_concept_network(
    concept_name: str = None,
    depth: int = 2,
    limit: int = 50,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import get_concept_network as get_network

        result = get_network(
            user_id=uid,
            concept_name=concept_name,
            depth=depth,
            limit=limit,
        )

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error getting concept network: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Find contradictions between business concepts.

Requires BUSINESS_CONCEPTS_ENABLED=true and BUSINESS_CONCEPTS_CONTRADICTION_DETECTION=true.

Parameters:
- concept_name: Optional concept to analyze (if omitted, finds all contradictions)
- category: Optional category filter
- min_severity: Minimum severity threshold 0.0-1.0 (default: 0.5)

Returns:
- contradictions[]: List of detected contradictions with severity and evidence
- count: Number of contradictions found
""")
async def find_concept_contradictions(
    concept_name: str = None,
    category: str = None,
    min_severity: float = 0.5,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.convergence_detector import (
            detect_contradictions_for_concept,
            find_all_contradictions,
        )

        if concept_name:
            contradictions = detect_contradictions_for_concept(
                user_id=uid,
                concept_name=concept_name,
                store=True,
            )
        else:
            contradictions = find_all_contradictions(
                user_id=uid,
                category=category,
                min_severity=min_severity,
            )

        return json.dumps({
            "concept": concept_name,
            "contradictions": contradictions,
            "count": len(contradictions),
        }, indent=2, default=str)

    except Exception as e:
        logging.exception(f"Error finding contradictions: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Analyze convergence of evidence for a concept.

Requires BUSINESS_CONCEPTS_ENABLED=true.

Analyzes whether a concept has strong convergent evidence from multiple
independent sources across time and domains.

Parameters:
- concept_name: Name of the concept to analyze (required)
- min_evidence: Minimum supporting memories required (default: 3)

Returns:
- convergence_score: Overall convergence score 0.0-1.0
- is_strong: Whether meets strong convergence threshold
- temporal_spread_days: Days between first and last evidence
- category_diversity: Diversity of evidence sources
- recommended_confidence: Suggested confidence boost
- evidence_count: Number of supporting memories
""")
async def analyze_concept_convergence(
    concept_name: str,
    min_evidence: int = 3,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.convergence_detector import ConvergenceDetector

        detector = ConvergenceDetector(user_id=uid)
        result = detector.analyze_concept_convergence(
            concept_name=concept_name,
            min_evidence=min_evidence,
        )

        if result:
            return json.dumps(result.to_dict(), indent=2, default=str)
        else:
            return json.dumps({
                "concept_name": concept_name,
                "error": f"Insufficient evidence (need at least {min_evidence} supporting memories)",
            })

    except Exception as e:
        logging.exception(f"Error analyzing convergence: {e}")
        return json.dumps({"error": str(e)})


@concept_mcp.tool(description="""Delete a business concept.

Requires BUSINESS_CONCEPTS_ENABLED=true.

Parameters:
- name: Name of the concept to delete (required)

Returns:
- deleted: Whether the concept was deleted
""")
async def delete_business_concept(
    name: str,
) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"})
    if not client_name:
        return json.dumps({"error": "client_name not provided"})

    from app.config import BusinessConceptsConfig
    if not BusinessConceptsConfig.is_enabled():
        return json.dumps({
            "error": "Business concepts system not enabled",
            "enabled": False,
        })

    try:
        from app.graph.concept_ops import delete_concept

        deleted = delete_concept(user_id=uid, name=name)

        return json.dumps({
            "name": name,
            "deleted": deleted,
        })

    except Exception as e:
        logging.exception(f"Error deleting concept: {e}")
        return json.dumps({"error": str(e)})


@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client.

    Authentication is required - the user_id in the path is IGNORED.
    The authenticated user comes from the JWT token in the Authorization header.
    """
    # Authenticate the request - user_id comes from JWT, not path
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
            )
        raise

    # Extract and validate DPoP proof if present (binds session to DPoP key)
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
            if dpop_thumbprint:
                principal.dpop_thumbprint = dpop_thumbprint
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
                )
            raise

    # Set context variables from authenticated principal (NOT from path)
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    session_id = None
    try:
        # Handle SSE connection
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # Capture session_id and create binding
            session_id = sse.get_session_id(request.scope)
            if session_id and HAS_SECURITY:
                store = get_session_binding_store()
                store.create(
                    session_id=session_id,
                    user_id=principal.user_id,
                    org_id=principal.org_id,
                    dpop_thumbprint=dpop_thumbprint,
                )

            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up session binding on disconnect
        if session_id and HAS_SECURITY:
            get_session_binding_store().delete(session_id)

        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_mcp_post_message(request: Request):
    return await handle_post_message(request)


async def handle_post_message(request: Request):
    """Handle POST messages for SSE.

    Authentication is required - context vars must be set from authenticated principal.
    Session binding is validated to prevent session hijacking.
    DPoP validation is performed when session was bound with DPoP.
    """
    # Validate session_id from query params (required for session binding)
    if HAS_SECURITY:
        session_id_str = request.query_params.get("session_id")
        if not session_id_str:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"},
            )

        try:
            from uuid import UUID
            session_id = UUID(session_id_str)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"},
            )

    # Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
            )
        raise

    # Extract and validate DPoP proof if present
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            # Get token for DPoP ath validation
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
                )
            raise

    # Validate session binding (after authentication)
    if HAS_SECURITY:
        store = get_session_binding_store()
        if not store.validate(
            session_id,
            principal.user_id,
            principal.org_id,
            dpop_thumbprint=dpop_thumbprint,
        ):
            return JSONResponse(
                status_code=403,
                content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"},
            )

    # Set context variables from authenticated principal
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name", "")
    client_token = client_name_var.set(client_name)
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


# =============================================================================
# BUSINESS CONCEPTS MCP ENDPOINT - /concepts/claude/sse/{user_id}
# =============================================================================
# This separate endpoint keeps Business Concept tools isolated from the
# main memory tools to reduce context window bloat in daily use.


@concept_router.get("/{client_name}/sse/{user_id}")
async def handle_concept_sse(request: Request):
    """Handle SSE connections for Business Concepts MCP.

    Authentication is required - the user_id in the path is IGNORED.
    The authenticated user comes from the JWT token in the Authorization header.
    """
    # Authenticate the request - user_id comes from JWT, not path
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
            )
        raise

    # Extract and validate DPoP proof if present (binds session to DPoP key)
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
            if dpop_thumbprint:
                principal.dpop_thumbprint = dpop_thumbprint
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
                )
            raise

    # Set context variables from authenticated principal (NOT from path)
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    session_id = None
    try:
        # Handle SSE connection using concept_mcp server
        async with concept_sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            # Capture session_id and create binding
            session_id = concept_sse.get_session_id(request.scope)
            if session_id and HAS_SECURITY:
                store = get_session_binding_store()
                store.create(
                    session_id=session_id,
                    user_id=principal.user_id,
                    org_id=principal.org_id,
                    dpop_thumbprint=dpop_thumbprint,
                )

            await concept_mcp._mcp_server.run(
                read_stream,
                write_stream,
                concept_mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up session binding on disconnect
        if session_id and HAS_SECURITY:
            get_session_binding_store().delete(session_id)

        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


@concept_router.post("/messages/")
async def handle_concept_get_message(request: Request):
    return await handle_concept_post_message(request)


@concept_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_concept_post_message_route(request: Request):
    return await handle_concept_post_message(request)


async def handle_concept_post_message(request: Request):
    """Handle POST messages for Business Concepts SSE.

    Authentication is required - context vars must be set from authenticated principal.
    Session binding is validated to prevent session hijacking.
    DPoP validation is performed when session was bound with DPoP.
    """
    # Validate session_id from query params (required for session binding)
    if HAS_SECURITY:
        session_id_str = request.query_params.get("session_id")
        if not session_id_str:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"},
            )

        try:
            from uuid import UUID
            session_id = UUID(session_id_str)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"},
            )

    # Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
    except Exception as e:
        if HAS_SECURITY and isinstance(e, AuthenticationError):
            return JSONResponse(
                status_code=401,
                content={"error": e.message, "code": e.code},
                headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
            )
        raise

    # Extract and validate DPoP proof if present
    dpop_thumbprint = None
    if HAS_SECURITY:
        try:
            # Get token for DPoP ath validation
            auth_header = request.headers.get("authorization", "")
            token = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
            dpop_thumbprint = await _extract_dpop_thumbprint(request, token)
        except Exception as e:
            if isinstance(e, AuthenticationError):
                return JSONResponse(
                    status_code=401,
                    content={"error": e.message, "code": e.code},
                    headers={"WWW-Authenticate": 'Bearer realm="concepts"'},
                )
            raise

    # Validate session binding (after authentication)
    if HAS_SECURITY:
        store = get_session_binding_store()
        if not store.validate(
            session_id,
            principal.user_id,
            principal.org_id,
            dpop_thumbprint=dpop_thumbprint,
        ):
            return JSONResponse(
                status_code=403,
                content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"},
            )

    # Set context variables from authenticated principal
    user_token = user_id_var.set(principal.user_id)
    client_name = request.path_params.get("client_name", "")
    client_token = client_name_var.set(client_name)
    org_token = org_id_var.set(principal.org_id)
    principal_token = principal_var.set(principal)

    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await concept_sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        org_id_var.reset(org_token)
        principal_var.reset(principal_token)


def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"
    concept_mcp._mcp_server.name = "business-concepts-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)

    # Include Business Concepts router (separate endpoint for concept tools)
    app.include_router(concept_router)
