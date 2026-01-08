"""
Graph Router for OpenMemory API.

Provides REST endpoints for graph operations:
- Graph statistics
- Aggregations (category, scope, tag distributions)
- Tag co-occurrence
- Full-text search
- Biography timeline
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, Memory
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.security.access import (
    resolve_access_entities,
    can_read_access_entity,
    build_access_entity_patterns,
)

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


def _get_allowed_memory_ids(principal: Principal, db: Session) -> Optional[List[str]]:
    """
    Get list of memory IDs the principal has access to based on access_entity.

    For multi-user routing, this queries memories where access_entity matches
    the principal's grants. Returns None if no filtering is needed (e.g., legacy
    single-user mode where user owns all their memories).

    Args:
        principal: The authenticated principal
        db: Database session

    Returns:
        List of memory ID strings, or None if no filtering needed
    """
    # Get user's explicit access entities
    access_entities = resolve_access_entities(principal)

    # Query memories that the principal has access to
    # For efficiency, we look for memories with access_entity in the user's grants
    # OR memories without access_entity that belong to the user
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        return []

    allowed_ids = set()

    # Get all memories accessible via access_entity grants
    memories_with_access = db.query(Memory).filter(
        Memory.state != 'deleted'
    ).all()

    for memory in memories_with_access:
        memory_access_entity = memory.metadata_.get("access_entity") if memory.metadata_ else None

        if memory_access_entity:
            # Check if principal can access this access_entity
            if can_read_access_entity(principal, memory_access_entity):
                allowed_ids.add(str(memory.id))
        else:
            # Legacy memory without access_entity - only owner has access
            if memory.user_id == user.id:
                allowed_ids.add(str(memory.id))

    return list(allowed_ids) if allowed_ids else None


def _get_graph_access_filters(principal: Principal) -> tuple[list[str], list[str]]:
    exact, like_patterns = build_access_entity_patterns(principal)
    prefixes = [p[:-1] for p in like_patterns if p.endswith("%")]
    return exact, prefixes


# =============================================================================
# Graph Statistics
# =============================================================================


@router.get("/stats")
async def get_stats(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    db: Session = Depends(get_db),
):
    """Get overall graph statistics."""
    from app.graph.graph_ops import get_graph_statistics, is_graph_enabled

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not is_graph_enabled():
        return {
            "enabled": False,
            "message": "Graph features not available (Neo4j not configured)",
        }

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    stats = get_graph_statistics(
        user_id=principal.user_id,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )
    stats["enabled"] = True

    return stats


@router.get("/health")
async def check_health():
    """Check graph system health."""
    from app.graph.graph_ops import (
        is_graph_enabled,
        is_mem0_graph_enabled,
        is_similarity_enabled,
        is_gds_available,
    )

    return {
        "neo4j_metadata_projection": is_graph_enabled(),
        "neo4j_similarity_edges": is_similarity_enabled(),
        "neo4j_gds": is_gds_available(),
        "mem0_graph_memory": is_mem0_graph_enabled(),
    }


# =============================================================================
# Aggregations
# =============================================================================


@router.get("/aggregate/{dimension}")
async def aggregate_by_dimension(
    dimension: str,
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Aggregate memories by a dimension.

    Dimension can be: category, scope, artifact_type, artifact_ref, tag, entity, app, evidence, source, state
    """
    from app.graph.graph_ops import aggregate_memories_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    valid_dimensions = [
        "category", "scope", "artifact_type", "artifact_ref", "tag",
        "entity", "app", "evidence", "source", "state"
    ]
    if dimension not in valid_dimensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {', '.join(valid_dimensions)}"
        )

    # Get allowed memory IDs based on access_entity grants
    allowed_memory_ids = _get_allowed_memory_ids(principal, db)
    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)

    results = aggregate_memories_in_graph(
        user_id=principal.user_id,
        group_by=dimension,
        allowed_memory_ids=allowed_memory_ids,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"dimension": dimension, "buckets": results}


# =============================================================================
# Tag Co-occurrence
# =============================================================================


@router.get("/tags/cooccurrence")
async def get_tag_cooccurrence(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    min_count: int = Query(default=2, ge=1),
    db: Session = Depends(get_db),
):
    """Get tag co-occurrence pairs."""
    from app.graph.graph_ops import tag_cooccurrence_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get allowed memory IDs based on access_entity grants
    allowed_memory_ids = _get_allowed_memory_ids(principal, db)
    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)

    pairs = tag_cooccurrence_in_graph(
        user_id=principal.user_id,
        allowed_memory_ids=allowed_memory_ids,
        limit=limit,
        min_count=min_count,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"pairs": pairs}


@router.get("/tags/{tag_key}/related")
async def get_related_tags(
    tag_key: str,
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    min_count: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Get tags related to a specific tag."""
    from app.graph.graph_ops import get_related_tags_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    related = get_related_tags_from_graph(
        tag_key=tag_key,
        user_id=principal.user_id,
        min_count=min_count,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"tag": tag_key, "related": related}


# =============================================================================
# Full-Text Search
# =============================================================================


@router.get("/search/memories")
async def fulltext_search_memories(
    query: str,
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Full-text search across memory content.

    Supports Lucene syntax: AND, OR, wildcards (*), fuzzy (~), phrases ("...")
    """
    from app.graph.graph_ops import fulltext_search_memories_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get allowed memory IDs based on access_entity grants
    allowed_memory_ids = _get_allowed_memory_ids(principal, db)
    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)

    results = fulltext_search_memories_in_graph(
        search_text=query,
        user_id=principal.user_id,
        allowed_memory_ids=allowed_memory_ids,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"query": query, "results": results}


@router.get("/search/entities")
async def fulltext_search_entities(
    query: str,
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Full-text search across entity names."""
    from app.graph.graph_ops import fulltext_search_entities_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    results = fulltext_search_entities_in_graph(
        search_text=query,
        user_id=principal.user_id,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"query": query, "results": results}


# =============================================================================
# Biography Timeline
# =============================================================================


@router.get("/timeline")
async def get_timeline(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    entity_name: Optional[str] = None,
    event_types: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get biographical timeline events."""
    from app.graph.graph_ops import get_biography_timeline_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Parse event_types if provided
    types_list = None
    if event_types:
        types_list = [t.strip() for t in event_types.split(",") if t.strip()]

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    events = get_biography_timeline_from_graph(
        user_id=principal.user_id,
        entity_name=entity_name,
        event_types=types_list,
        start_year=start_year,
        end_year=end_year,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"events": events}


# =============================================================================
# Graph Analytics (GDS-based)
# =============================================================================


@router.get("/analytics/communities/entities")
async def get_entity_communities(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    db: Session = Depends(get_db),
):
    """Detect entity communities using Louvain (requires GDS)."""
    from app.graph.graph_ops import detect_entity_communities, is_gds_available

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not is_gds_available():
        raise HTTPException(
            status_code=503,
            detail="Neo4j GDS not available for community detection"
        )

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    result = detect_entity_communities(
        user_id=principal.user_id,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return result


@router.get("/analytics/communities/memories")
async def get_memory_communities(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    db: Session = Depends(get_db),
):
    """Detect memory communities based on similarity (requires GDS)."""
    from app.graph.graph_ops import detect_memory_communities, is_gds_available

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not is_gds_available():
        raise HTTPException(
            status_code=503,
            detail="Neo4j GDS not available for community detection"
        )

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    result = detect_memory_communities(
        user_id=principal.user_id,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return result


@router.get("/analytics/similarity/entities")
async def get_similar_entities(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    entity_name: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Find similar entities based on shared memory connections (requires GDS)."""
    from app.graph.graph_ops import find_similar_entities_gds, is_gds_available

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not is_gds_available():
        raise HTTPException(
            status_code=503,
            detail="Neo4j GDS not available for similarity computation"
        )

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    result = find_similar_entities_gds(
        user_id=principal.user_id,
        entity_name=entity_name,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"pairs": result}


@router.get("/analytics/connectivity")
async def get_memory_connectivity(
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get memory connectivity statistics."""
    from app.graph.graph_ops import get_memory_connectivity_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    result = get_memory_connectivity_from_graph(
        user_id=principal.user_id,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"memories": result}
