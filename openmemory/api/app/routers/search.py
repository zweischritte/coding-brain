"""
Search Router - REST API for search operations.

Provides endpoints for hybrid (lexical + vector), lexical-only, and
semantic-only search operations with tenant isolation.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.security.access import build_access_entity_patterns
from app.stores.opensearch_store import TenantOpenSearchStore, get_tenant_opensearch_store
from app.utils.memory import get_memory_client


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ============================================================================
# Enums
# ============================================================================


class SearchMode(str, Enum):
    """Search mode for the hybrid search endpoint."""
    auto = "auto"
    lexical = "lexical"
    semantic = "semantic"


# ============================================================================
# Pydantic Schemas
# ============================================================================


class SearchMeta(BaseModel):
    """Metadata about the search execution."""
    degraded_mode: bool = Field(
        False,
        description="True if search fell back to lexical-only due to embedding failure"
    )
    missing_sources: List[str] = Field(
        default_factory=list,
        description="List of sources that were unavailable (e.g., 'embedding')"
    )


class SearchFilters(BaseModel):
    """Optional filters for search queries."""
    category: Optional[str] = Field(None, description="Filter by memory category")
    scope: Optional[str] = Field(None, description="Filter by memory scope")
    artifact_type: Optional[str] = Field(None, description="Filter by artifact type")
    artifact_ref: Optional[str] = Field(None, description="Filter by artifact ref")
    entity: Optional[str] = Field(None, description="Filter by entity")
    source: Optional[str] = Field(None, description="Filter by source")
    app_id: Optional[str] = Field(None, description="Filter by app ID")


class SearchRequest(BaseModel):
    """Request body for search operations."""
    query: str = Field(..., min_length=1, description="The search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    filters: Optional[SearchFilters] = Field(None, description="Optional filters")
    mode: SearchMode = Field(
        SearchMode.auto,
        description="Search mode: auto (try embedding, fallback to lexical), lexical (text only), semantic (requires embedding)"
    )


class SemanticSearchRequest(BaseModel):
    """Request body for semantic (vector) search."""
    query: str = Field(..., min_length=1, description="The search query")
    query_vector: List[float] = Field(..., min_length=1, description="Query embedding vector")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    filters: Optional[SearchFilters] = Field(None, description="Optional filters")


class SearchResult(BaseModel):
    """A single search result."""
    memory_id: str
    score: float
    content: str
    highlights: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response body for search operations."""
    results: List[SearchResult]
    total_count: int
    took_ms: int
    meta: SearchMeta = Field(
        default_factory=SearchMeta,
        description="Metadata about the search execution"
    )


# ============================================================================
# Dependencies
# ============================================================================


def get_search_store(
    principal: Principal = Depends(require_scopes(Scope.SEARCH_READ)),
) -> TenantOpenSearchStore:
    """
    Dependency for search store with tenant isolation.

    Returns a TenantOpenSearchStore scoped to the principal's org_id.
    Raises 503 if OpenSearch is unavailable.
    """
    store = get_tenant_opensearch_store(
        org_id=principal.org_id,
        index_prefix="memories",
    )

    if store is None:
        logger.warning("OpenSearch unavailable for search")
        raise HTTPException(
            status_code=503,
            detail="Search service temporarily unavailable"
        )

    return store


def _filters_to_dict(filters: Optional[SearchFilters]) -> Optional[Dict[str, Any]]:
    """Convert SearchFilters to dict for store."""
    if filters is None:
        return None

    result = {}
    if filters.category:
        result["category"] = filters.category
    if filters.scope:
        result["scope"] = filters.scope
    if filters.artifact_type:
        result["artifact_type"] = filters.artifact_type
    if filters.artifact_ref:
        result["artifact_ref"] = filters.artifact_ref
    if filters.entity:
        result["entity"] = filters.entity
    if filters.source:
        result["source"] = filters.source
    if filters.app_id:
        result["app_id"] = filters.app_id

    return result if result else None


def _access_entity_filters(principal: Principal) -> tuple[list[str], list[str]]:
    """Build exact and prefix access_entity filters for OpenSearch."""
    exact, like_patterns = build_access_entity_patterns(principal)
    prefixes = []
    for pattern in like_patterns:
        if pattern.endswith("%"):
            prefixes.append(pattern[:-1])
        else:
            prefixes.append(pattern)
    return exact, prefixes


def _format_results(hits: List[Dict[str, Any]]) -> List[SearchResult]:
    """Format OpenSearch hits into SearchResult objects."""
    results = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append(SearchResult(
            memory_id=hit.get("_id", ""),
            score=hit.get("_score", 0.0),
            content=source.get("content", ""),
            highlights=hit.get("highlight", {}).get("content", []),
            metadata={
                k: v for k, v in source.items()
                if k not in ("content", "embedding", "org_id")
            },
        ))
    return results


# ============================================================================
# Endpoints
# ============================================================================


@router.post("", response_model=SearchResponse)
async def hybrid_search(
    request: SearchRequest,
    principal: Principal = Depends(require_scopes(Scope.SEARCH_READ)),
    db: Session = Depends(get_db),
):
    """
    Hybrid search combining lexical and semantic search.

    Supports three modes:
    - auto (default): Try to generate embedding for hybrid search, fall back to lexical if unavailable
    - lexical: Use text matching only, skip embedding
    - semantic: Require embedding, return 503 if unavailable

    Results are automatically scoped to the authenticated org.
    If OpenSearch is unavailable, returns empty results.
    """
    start_time = time.time()

    # Get tenant-scoped store
    store = get_tenant_opensearch_store(
        org_id=principal.org_id,
        index_prefix="memories",
    )

    if store is None:
        # OpenSearch unavailable - return empty results gracefully
        logger.warning(f"OpenSearch unavailable for org {principal.org_id}")
        return SearchResponse(
            results=[],
            total_count=0,
            took_ms=0,
            meta=SearchMeta(degraded_mode=False, missing_sources=[]),
        )

    filters_dict = _filters_to_dict(request.filters)
    exact, prefixes = _access_entity_filters(principal)

    # Track search metadata
    degraded_mode = False
    missing_sources: List[str] = []
    query_vector: Optional[List[float]] = None

    # Determine if we should attempt embedding
    should_embed = request.mode in (SearchMode.auto, SearchMode.semantic)

    if should_embed:
        # Try to get embedding
        embedding_error = None
        try:
            memory_client = get_memory_client()
            if memory_client and memory_client.embedding_model:
                query_vector = memory_client.embedding_model.embed(request.query, "search")
                logger.info(f"Generated embedding for search query (mode={request.mode.value})")
            else:
                embedding_error = "Memory client or embedding model unavailable"
        except Exception as e:
            embedding_error = str(e)
            logger.warning(f"Embedding generation failed: {e}")

        # Handle embedding failure based on mode
        if query_vector is None:
            if request.mode == SearchMode.semantic:
                # Semantic mode requires embedding - fail with 503
                logger.error(f"Semantic search unavailable: {embedding_error}")
                raise HTTPException(
                    status_code=503,
                    detail="Embedding service unavailable - semantic search not possible"
                )
            else:
                # Auto mode - fall back to lexical
                degraded_mode = True
                missing_sources.append("embedding")
                logger.info("Falling back to lexical search (embedding unavailable)")

    # Perform search
    if query_vector is not None:
        # Use hybrid search with embedding
        logger.info(f"Performing hybrid search (mode={request.mode.value})")
        hits = store.hybrid_search_with_access_control(
            query_text=request.query,
            query_vector=query_vector,
            access_entities=exact,
            access_entity_prefixes=prefixes,
            limit=request.limit,
            filters=filters_dict,
        )
    else:
        # Use lexical-only search
        logger.info(f"Performing lexical search (mode={request.mode.value})")
        hits = store.search_with_access_control(
            query_text=request.query,
            access_entities=exact,
            access_entity_prefixes=prefixes,
            limit=request.limit,
            filters=filters_dict,
        )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
        meta=SearchMeta(
            degraded_mode=degraded_mode,
            missing_sources=missing_sources,
        ),
    )


@router.post("/lexical", response_model=SearchResponse)
async def lexical_search(
    request: SearchRequest,
    principal: Principal = Depends(require_scopes(Scope.SEARCH_READ)),
    db: Session = Depends(get_db),
):
    """
    Lexical-only search using text matching.

    Best for keyword-based queries where exact word matches matter.
    Results are automatically scoped to the authenticated org.
    """
    start_time = time.time()

    # Get tenant-scoped store
    store = get_tenant_opensearch_store(
        org_id=principal.org_id,
        index_prefix="memories",
    )

    if store is None:
        # OpenSearch unavailable - return empty results gracefully
        logger.warning(f"OpenSearch unavailable for org {principal.org_id}")
        return SearchResponse(
            results=[],
            total_count=0,
            took_ms=0,
            meta=SearchMeta(degraded_mode=False, missing_sources=[]),
        )

    # Perform lexical search
    filters_dict = _filters_to_dict(request.filters)
    exact, prefixes = _access_entity_filters(principal)
    hits = store.search_with_access_control(
        query_text=request.query,
        access_entities=exact,
        access_entity_prefixes=prefixes,
        limit=request.limit,
        filters=filters_dict,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
        meta=SearchMeta(degraded_mode=False, missing_sources=[]),
    )


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    principal: Principal = Depends(require_scopes(Scope.SEARCH_READ)),
    db: Session = Depends(get_db),
):
    """
    Semantic-only search using vector similarity.

    Requires a pre-computed query embedding vector.
    Best for conceptual similarity searches.
    Results are automatically scoped to the authenticated org.
    """
    start_time = time.time()

    # Get tenant-scoped store
    store = get_tenant_opensearch_store(
        org_id=principal.org_id,
        index_prefix="memories",
    )

    if store is None:
        # OpenSearch unavailable - return empty results gracefully
        logger.warning(f"OpenSearch unavailable for org {principal.org_id}")
        return SearchResponse(
            results=[],
            total_count=0,
            took_ms=0,
            meta=SearchMeta(degraded_mode=False, missing_sources=[]),
        )

    # Perform hybrid search with the provided vector
    filters_dict = _filters_to_dict(request.filters)
    exact, prefixes = _access_entity_filters(principal)
    hits = store.hybrid_search_with_access_control(
        query_text=request.query,
        query_vector=request.query_vector,
        access_entities=exact,
        access_entity_prefixes=prefixes,
        limit=request.limit,
        filters=filters_dict,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
        meta=SearchMeta(degraded_mode=False, missing_sources=[]),
    )
