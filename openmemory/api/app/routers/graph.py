"""
Graph Router for OpenMemory API.

Provides REST endpoints for graph operations:
- Graph statistics
- Aggregations (vault, layer, tag distributions)
- Tag co-occurrence
- Full-text search
- Biography timeline
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


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

    stats = get_graph_statistics(user_id=principal.user_id)
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

    Dimension can be: vault, layer, tag, entity, app, vector, circuit, origin, evidence, source, state
    """
    from app.graph.graph_ops import aggregate_memories_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    valid_dimensions = [
        "vault", "layer", "tag", "entity", "app",
        "vector", "circuit", "origin", "evidence", "source", "state"
    ]
    if dimension not in valid_dimensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension. Must be one of: {', '.join(valid_dimensions)}"
        )

    results = aggregate_memories_in_graph(
        user_id=principal.user_id,
        group_by=dimension,
        limit=limit,
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

    pairs = tag_cooccurrence_in_graph(
        user_id=principal.user_id,
        limit=limit,
        min_count=min_count,
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

    related = get_related_tags_from_graph(
        tag_key=tag_key,
        user_id=principal.user_id,
        min_count=min_count,
        limit=limit,
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

    results = fulltext_search_memories_in_graph(
        search_text=query,
        user_id=principal.user_id,
        limit=limit,
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

    results = fulltext_search_entities_in_graph(
        search_text=query,
        user_id=principal.user_id,
        limit=limit,
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

    events = get_biography_timeline_from_graph(
        user_id=principal.user_id,
        entity_name=entity_name,
        event_types=types_list,
        start_year=start_year,
        end_year=end_year,
        limit=limit,
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

    result = detect_entity_communities(user_id=principal.user_id)

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

    result = detect_memory_communities(user_id=principal.user_id)

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

    result = find_similar_entities_gds(
        user_id=principal.user_id,
        entity_name=entity_name,
        limit=limit,
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

    result = get_memory_connectivity_from_graph(
        user_id=principal.user_id,
        limit=limit,
    )

    return {"memories": result}
