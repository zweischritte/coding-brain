"""
Entities Router for OpenMemory API.

Provides REST endpoints for entity operations:
- Entity network visualization
- Entity relations (typed relationships)
- Entity centrality metrics
- Entity normalization
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope
from app.security.access import build_access_entity_patterns

router = APIRouter(prefix="/api/v1/entities", tags=["entities"])


def _get_graph_access_filters(principal: Principal) -> tuple[list[str], list[str]]:
    exact, like_patterns = build_access_entity_patterns(principal)
    prefixes = [p[:-1] for p in like_patterns if p.endswith("%")]
    return exact, prefixes


# =============================================================================
# IMPORTANT: Static routes MUST be defined BEFORE dynamic routes!
# FastAPI matches routes in definition order, so /analytics/centrality
# must come before /{entity_name} to avoid "analytics" being parsed as entity_name
# =============================================================================


# =============================================================================
# Entity Analytics Endpoints (STATIC - must be before dynamic routes)
# =============================================================================


@router.get("/analytics/centrality")
async def get_centrality(
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get entity centrality rankings."""
    from app.graph.graph_ops import get_entity_centrality_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    centrality = get_entity_centrality_from_graph(
        user_id=principal.user_id,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"entities": centrality}


@router.get("/analytics/pagerank")
async def get_pagerank(
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get entity PageRank scores (requires GDS)."""
    from app.graph.graph_ops import get_entity_pagerank, is_gds_available

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not is_gds_available():
        raise HTTPException(
            status_code=503,
            detail="Neo4j GDS not available for PageRank"
        )

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    pagerank = get_entity_pagerank(
        user_id=principal.user_id,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {"entities": pagerank}


# =============================================================================
# Entity Normalization Endpoints (STATIC - must be before dynamic routes)
# =============================================================================


class NormalizeRequest(BaseModel):
    canonical: Optional[str] = None
    variants: Optional[List[str]] = None
    auto: bool = False
    dry_run: bool = True
    access_entity: Optional[str] = None


@router.get("/normalization/duplicates")
async def find_duplicates(
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    access_entity: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Find duplicate entity candidates."""
    from app.graph.graph_ops import find_duplicate_entities_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    duplicates = find_duplicate_entities_in_graph(
        user_id=principal.user_id,
        access_entity=access_entity,
    )

    return {"duplicates": duplicates}


@router.post("/normalization/merge")
async def normalize_entities_endpoint(
    request: NormalizeRequest,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_WRITE)),
    db: Session = Depends(get_db),
):
    """Merge duplicate entities."""
    from app.graph.graph_ops import normalize_entities_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = normalize_entities_in_graph(
        user_id=principal.user_id,
        canonical_name=request.canonical,
        variant_names=request.variants,
        auto=request.auto,
        dry_run=request.dry_run,
        access_entity=request.access_entity,
    )

    return result


@router.get("/normalization/semantic-duplicates")
async def find_semantic_duplicates_endpoint(
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    threshold: float = Query(default=0.7, ge=0.0, le=1.0),
    access_entity: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Find semantic entity duplicates using advanced detection."""
    from app.graph.graph_ops import find_semantic_duplicates

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    duplicates = await find_semantic_duplicates(
        user_id=principal.user_id,
        threshold=threshold,
        access_entity=access_entity,
    )

    return {"duplicates": duplicates}


class SemanticNormalizeRequest(BaseModel):
    canonical: Optional[str] = None
    variants: Optional[List[str]] = None
    auto: bool = False
    threshold: float = 0.7
    dry_run: bool = True


@router.post("/normalization/semantic-merge")
async def normalize_semantic(
    request: SemanticNormalizeRequest,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_WRITE)),
    db: Session = Depends(get_db),
):
    """Merge entities using semantic normalization."""
    from app.graph.graph_ops import normalize_entities_semantic

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await normalize_entities_semantic(
        user_id=principal.user_id,
        canonical=request.canonical,
        variants=request.variants,
        auto=request.auto,
        threshold=request.threshold,
        dry_run=request.dry_run,
    )

    return result


# =============================================================================
# Entity Path Finding (STATIC - must be before dynamic routes)
# =============================================================================


@router.get("/path/{entity_a}/{entity_b}")
async def find_path(
    entity_a: str,
    entity_b: str,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    max_hops: int = Query(default=6, ge=2, le=12),
    db: Session = Depends(get_db),
):
    """Find shortest path between two entities."""
    from app.graph.graph_ops import path_between_entities_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    path = path_between_entities_in_graph(
        user_id=principal.user_id,
        entity_a=entity_a,
        entity_b=entity_b,
        max_hops=max_hops,
    )

    if path is None:
        raise HTTPException(status_code=404, detail="No path found between entities")

    return path


# =============================================================================
# Entity List Endpoint (Root - must be before dynamic entity routes)
# =============================================================================


@router.get("")
async def list_entities(
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    limit: int = Query(default=50, ge=1, le=200),
    min_memories: int = Query(default=1, ge=1),
    db: Session = Depends(get_db),
):
    """List all entities for a user with memory counts."""
    from app.graph.graph_ops import aggregate_memories_in_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Aggregate by entity to get list with counts
    entities = aggregate_memories_in_graph(
        user_id=principal.user_id,
        group_by="entity",
        limit=limit,
    )

    # Filter by min_memories
    entities = [e for e in entities if e.get("count", 0) >= min_memories]

    return {"entities": entities, "total": len(entities)}


# =============================================================================
# Dynamic Entity Routes (MUST be LAST - these have path parameters)
# =============================================================================


@router.get("/{entity_name}")
async def get_entity(
    entity_name: str,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    db: Session = Depends(get_db),
):
    """Get detailed info for a single entity."""
    from app.graph.graph_ops import (
        get_entity_network_from_graph,
        get_entity_relations_from_graph,
        aggregate_memories_in_graph,
    )

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)

    # Get network (co-mentions)
    network = get_entity_network_from_graph(
        entity_name=entity_name,
        user_id=principal.user_id,
        min_count=1,
        limit=50,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    # Get typed relations
    relations = get_entity_relations_from_graph(
        entity_name=entity_name,
        user_id=principal.user_id,
        direction="both",
        limit=50,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    # Get memory count for this entity
    entities = aggregate_memories_in_graph(
        user_id=principal.user_id,
        group_by="entity",
        limit=1000,  # High limit to find the entity
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )
    memory_count = 0
    display_name = None
    for e in entities:
        if e.get("key") == entity_name:
            memory_count = e.get("count", 0)
            display_name = e.get("displayName")
            break
    if not display_name and network and network.get("entityDisplayName"):
        display_name = network.get("entityDisplayName")

    return {
        "name": entity_name,
        "displayName": display_name or entity_name,
        "memory_count": memory_count,
        "network": network,
        "relations": relations,
    }


@router.get("/{entity_name}/network")
async def get_entity_network(
    entity_name: str,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    min_count: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get the co-mention network for an entity."""
    from app.graph.graph_ops import get_entity_network_from_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    network = get_entity_network_from_graph(
        entity_name=entity_name,
        user_id=principal.user_id,
        min_count=min_count,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    if network is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    return network


@router.get("/{entity_name}/relations")
async def get_entity_relations(
    entity_name: str,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    relation_types: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    direction: str = Query(default="both", pattern="^(outgoing|incoming|both)$"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get typed relations for an entity."""
    from app.graph.graph_ops import (
        get_entity_relations_from_graph,
        get_entity_display_name_from_graph,
    )

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)

    # Parse relation_types if provided
    types_list = None
    if relation_types:
        types_list = [t.strip() for t in relation_types.split(",") if t.strip()]

    relations = get_entity_relations_from_graph(
        entity_name=entity_name,
        user_id=principal.user_id,
        relation_types=types_list,
        category=category,
        direction=direction,
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    display_name = get_entity_display_name_from_graph(
        entity_name=entity_name,
        user_id=principal.user_id,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    return {
        "entity": entity_name,
        "entityDisplayName": display_name or entity_name,
        "relations": relations,
    }


@router.get("/{entity_name}/memories")
async def get_entity_memories(
    entity_name: str,
    principal: Principal = Depends(require_scopes(Scope.ENTITIES_READ)),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get memories associated with an entity."""
    from app.graph.graph_ops import retrieve_via_entity_graph

    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    access_entities, access_entity_prefixes = _get_graph_access_filters(principal)
    memories = retrieve_via_entity_graph(
        user_id=principal.user_id,
        entity_names=[entity_name],
        limit=limit,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )

    display_name = None
    for memory in memories:
        for detail in memory.get("entityDetails", []) or []:
            if detail.get("name") == entity_name and detail.get("displayName"):
                display_name = detail["displayName"]
                break
        if display_name:
            break
    if not display_name:
        from app.graph.graph_ops import get_entity_display_name_from_graph
        display_name = get_entity_display_name_from_graph(
            entity_name=entity_name,
            user_id=principal.user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    return {
        "entity": entity_name,
        "entityDisplayName": display_name or entity_name,
        "memories": memories,
    }
