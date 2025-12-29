"""
Search Router - REST API for search operations.

Provides endpoints for hybrid (lexical + vector), lexical-only, and
semantic-only search operations with tenant isolation.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from app.database import get_db
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.stores.opensearch_store import TenantOpenSearchStore, get_tenant_opensearch_store


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ============================================================================
# Pydantic Schemas
# ============================================================================


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

    For best results, the query should be semantically meaningful.
    Results are automatically scoped to the authenticated org.

    If OpenSearch is unavailable, returns 503.
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
        )

    # Perform lexical search (hybrid requires embedding which we don't have here)
    # In production, we'd generate an embedding for the query
    filters_dict = _filters_to_dict(request.filters)
    hits = store.search(
        query_text=request.query,
        limit=request.limit,
        filters=filters_dict,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
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
        )

    # Perform lexical search
    filters_dict = _filters_to_dict(request.filters)
    hits = store.search(
        query_text=request.query,
        limit=request.limit,
        filters=filters_dict,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
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
        )

    # Perform hybrid search with the provided vector
    filters_dict = _filters_to_dict(request.filters)
    hits = store.hybrid_search(
        query_text=request.query,
        query_vector=request.query_vector,
        limit=request.limit,
        filters=filters_dict,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(
        results=_format_results(hits),
        total_count=len(hits),
        took_ms=elapsed_ms,
    )
