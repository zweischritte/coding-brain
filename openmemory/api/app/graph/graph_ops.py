"""
Graph operations helper for MCP server integration.

Provides high-level functions for:
- Projecting memory metadata to Neo4j
- Deleting memory projections
- Querying graph relations for search enrichment
- Querying Mem0 Graph Memory relations

All operations are designed to fail gracefully - if Neo4j is unavailable,
operations log warnings but don't fail the parent operation.
"""

import logging
import itertools
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid import errors when Neo4j is not installed
_projector = None
_projector_initialized = False

_VIA_TO_REL_TYPES: Dict[str, str] = {
    "entity": "OM_ABOUT",
    "category": "OM_IN_CATEGORY",
    "scope": "OM_IN_SCOPE",
    "artifact_type": "OM_HAS_ARTIFACT_TYPE",
    "artifact_ref": "OM_REFERENCES_ARTIFACT",
    "tag": "OM_TAGGED",
    "evidence": "OM_HAS_EVIDENCE",
    "app": "OM_WRITTEN_VIA",
}


def _parse_via_to_rel_types(via: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated list of traversal dimensions into relationship types.

    Accepts either:
    - dimension names (tag, entity, category, scope, artifact_type, artifact_ref, evidence, app)
    - explicit relationship types (OM_TAGGED, OM_ABOUT, ...)

    Returns:
        None if via is empty/None (meaning "use defaults")
        Otherwise a list of relationship type strings.
    """
    if not via:
        return None

    tokens = [t.strip() for t in str(via).split(",") if t.strip()]
    rel_types: List[str] = []
    for token in tokens:
        upper = token.upper()
        lower = token.lower()
        if upper.startswith("OM_"):
            rel_types.append(upper)
            continue
        if lower in _VIA_TO_REL_TYPES:
            rel_types.append(_VIA_TO_REL_TYPES[lower])
            continue
        raise ValueError(
            f"Unsupported via='{token}'. Supported: {', '.join(sorted(_VIA_TO_REL_TYPES.keys()))} "
            "or explicit OM_* relationship types."
        )

    # De-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for rt in rel_types:
        if rt in seen:
            continue
        seen.add(rt)
        deduped.append(rt)
    return deduped


def _get_projector():
    """Get the metadata projector (lazy initialization)."""
    global _projector, _projector_initialized

    if _projector_initialized:
        return _projector

    _projector_initialized = True

    try:
        from app.graph.metadata_projector import get_projector
        _projector = get_projector()
        if _projector:
            logger.info("Metadata projector initialized for graph operations")
    except ImportError as e:
        logger.debug(f"Graph module not available: {e}")
        _projector = None
    except Exception as e:
        logger.warning(f"Failed to initialize metadata projector: {e}")
        _projector = None

    return _projector


def reset_graph_ops():
    """Reset the graph operations state (for testing or config changes)."""
    global _projector, _projector_initialized
    _projector = None
    _projector_initialized = False


def project_memory_to_graph(
    memory_id: str,
    user_id: str,
    content: str,
    metadata: Dict[str, Any],
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    state: str = "active",
) -> bool:
    """
    Project a memory's metadata to Neo4j graph.

    This is called after a memory is successfully added/updated in the vector store.
    If Neo4j is unavailable, this logs a warning but returns True to not block
    the main operation.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID (e.g., "grischadallmer")
        content: Memory content text
        metadata: Memory metadata dict
        created_at: ISO datetime string
        updated_at: ISO datetime string
        state: Memory state (active, deleted, etc.)

    Returns:
        True if projection succeeded or Neo4j is not configured
        False only if Neo4j is configured but projection failed
    """
    projector = _get_projector()

    if projector is None:
        # Neo4j not configured, silently succeed
        return True

    try:
        from app.graph.metadata_projector import MemoryMetadata

        # Build the data dict in the format MemoryMetadata expects
        data = {
            "content": content,
            "metadata": metadata,
            "created_at": created_at,
            "updated_at": updated_at,
            "state": state,
        }

        memory_metadata = MemoryMetadata.from_dict(
            data=data,
            memory_id=memory_id,
            user_id=user_id,
        )

        success = projector.upsert_memory(memory_metadata)

        if not success:
            logger.warning(f"Failed to project memory {memory_id} to graph")

        return success

    except Exception as e:
        logger.warning(f"Error projecting memory {memory_id} to graph: {e}")
        return False


def delete_memory_from_graph(memory_id: str) -> bool:
    """
    Delete a memory's projection from Neo4j graph.

    Args:
        memory_id: UUID of the memory to delete

    Returns:
        True if deletion succeeded or Neo4j is not configured
    """
    projector = _get_projector()

    if projector is None:
        return True

    try:
        return projector.delete_memory(memory_id)
    except Exception as e:
        logger.warning(f"Error deleting memory {memory_id} from graph: {e}")
        return False


def delete_all_user_memories_from_graph(user_id: str) -> bool:
    """
    Delete all memory projections for a user from Neo4j graph.

    Args:
        user_id: String user ID

    Returns:
        True if deletion succeeded or Neo4j is not configured
    """
    projector = _get_projector()

    if projector is None:
        return True

    try:
        return projector.delete_all_user_memories(user_id)
    except Exception as e:
        logger.warning(f"Error deleting all memories for user {user_id} from graph: {e}")
        return False


def get_meta_relations_for_memories(memory_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    Get metadata graph relations for a list of memory IDs.

    Args:
        memory_ids: List of memory UUIDs

    Returns:
        Dict mapping memory_id -> list of relation dicts
        Empty dict if Neo4j is not configured or query fails
    """
    if not memory_ids:
        return {}

    projector = _get_projector()

    if projector is None:
        return {}

    try:
        return projector.get_relations_for_memories(memory_ids)
    except Exception as e:
        logger.warning(f"Error getting meta relations: {e}")
        return {}


def get_memory_node_from_graph(
    memory_id: str,
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a memory node from the Neo4j metadata projection.

    Returns None if Neo4j is not configured or the memory is not found / not allowed.
    """
    projector = _get_projector()
    if projector is None:
        return None
    try:
        return projector.get_memory_node(
            memory_id=memory_id,
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Error getting memory node {memory_id} from graph: {e}")
        return None


def find_related_memories_in_graph(
    memory_id: str,
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    via: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find related memories for a seed memory using the metadata subgraph.

    Args:
        memory_id: Seed memory UUID
        user_id: String user id
        allowed_memory_ids: Optional allowlist of accessible memory IDs (ACL)
        via: Optional CSV of dimensions/relationship types to traverse
        limit: Max related memories to return
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        rel_types = _parse_via_to_rel_types(via)
        return projector.find_related_memories(
            memory_id=memory_id,
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            rel_types=rel_types,
            limit=limit,
        )
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Error finding related memories for {memory_id}: {e}")
        return []


def aggregate_memories_in_graph(
    user_id: str,
    group_by: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Aggregate memories by a dimension using the metadata subgraph.

    Returns empty list if Neo4j is not configured or query fails.
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.aggregate_memories(
            user_id=user_id,
            group_by=group_by,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
        )
    except ValueError as e:
        # Surface input validation errors cleanly
        raise
    except Exception as e:
        logger.warning(f"Error aggregating memories by {group_by}: {e}")
        return []


def tag_cooccurrence_in_graph(
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
    min_count: int = 2,
    sample_size: int = 3,
) -> List[Dict[str, Any]]:
    """
    Compute co-occurring tags across memories using the metadata subgraph.
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.tag_cooccurrence(
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
            min_count=min_count,
            sample_size=sample_size,
        )
    except Exception as e:
        logger.warning(f"Error computing tag cooccurrence: {e}")
        return []


def path_between_entities_in_graph(
    user_id: str,
    entity_a: str,
    entity_b: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    max_hops: int = 6,
) -> Optional[Dict[str, Any]]:
    """
    Find a shortest path between two entities through the metadata subgraph.
    """
    projector = _get_projector()
    if projector is None:
        return None

    try:
        return projector.path_between_entities(
            user_id=user_id,
            entity_a=entity_a,
            entity_b=entity_b,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            max_hops=max_hops,
        )
    except Exception as e:
        logger.warning(f"Error finding entity path: {e}")
        return None


def get_memory_subgraph_from_graph(
    memory_id: str,
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    depth: int = 2,
    via: Optional[str] = None,
    related_limit: int = 25,
) -> Optional[Dict[str, Any]]:
    """
    Build a small, JSON-friendly subgraph around a memory.

    depth:
      - 1: seed memory + its dimension nodes (outgoing relations)
      - 2: additionally include up to `related_limit` other memories connected
           through any shared dimension node (controlled by `via`)
    """
    projector = _get_projector()
    if projector is None:
        return {
            "seed_memory_id": memory_id,
            "nodes": [],
            "edges": [],
            "related": [],
        }

    depth = max(1, min(int(depth or 2), 2))
    related_limit = max(0, min(int(related_limit or 25), 200))

    seed = get_memory_node_from_graph(
        memory_id=memory_id,
        user_id=user_id,
        allowed_memory_ids=allowed_memory_ids,
        access_entities=access_entities,
        access_entity_prefixes=access_entity_prefixes,
    )
    if seed is None:
        return None

    relations = projector.get_relations_for_memories([memory_id]).get(memory_id, [])

    related: List[Dict[str, Any]] = []
    if depth >= 2 and related_limit > 0:
        related = find_related_memories_in_graph(
            memory_id=memory_id,
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            via=via,
            limit=related_limit,
        )

    def node_id_for(label: str, value: Any, access_entity: Optional[str] = None) -> str:
        if label == "OM_Memory":
            return f"OM_Memory:{value}"
        if label == "OM_Entity":
            if access_entity:
                return f"OM_Entity:{access_entity}:{value}"
            return f"OM_Entity:{value}"
        return f"{label}:{value}"

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    seed_node_id = node_id_for("OM_Memory", seed["id"])
    seed_access_entity = seed.get("accessEntity")
    nodes[seed_node_id] = {
        "id": seed_node_id,
        "label": "OM_Memory",
        "memory_id": seed["id"],
        "content": seed.get("content"),
        "category": seed.get("category"),
        "scope": seed.get("scope"),
        "artifact_type": seed.get("artifactType"),
        "artifact_ref": seed.get("artifactRef"),
        "source": seed.get("source"),
        "created_at": seed.get("created_at") or seed.get("createdAt"),
        "updated_at": seed.get("updated_at") or seed.get("updatedAt"),
    }

    # Add dimension nodes + edges from seed
    for rel in relations:
        target_label = rel.get("target_label")
        target_value = rel.get("target_value")
        if not target_label or target_value is None:
            continue
        dim_id = node_id_for(target_label, target_value, seed_access_entity if target_label == "OM_Entity" else None)
        if dim_id not in nodes:
            nodes[dim_id] = {
                "id": dim_id,
                "label": target_label,
                "value": target_value,
            }
        if target_label == "OM_Entity" and rel.get("target_display_name"):
            nodes[dim_id]["displayName"] = rel.get("target_display_name")
        edge = {
            "source": seed_node_id,
            "target": dim_id,
            "type": rel.get("type"),
        }
        if rel.get("value") is not None:
            edge["value"] = rel.get("value")
        edges.append(edge)

    # Add related memory nodes + edges to shared dims
    for rm in related:
        rid = rm.get("id")
        if not rid:
            continue
        rm_access_entity = rm.get("accessEntity")
        rm_node_id = node_id_for("OM_Memory", rid)
        if rm_node_id not in nodes:
            nodes[rm_node_id] = {
                "id": rm_node_id,
                "label": "OM_Memory",
                "memory_id": rid,
                "content": rm.get("content"),
                "category": rm.get("category"),
                "scope": rm.get("scope"),
                "artifact_type": rm.get("artifactType"),
                "artifact_ref": rm.get("artifactRef"),
                "source": rm.get("source"),
                "created_at": rm.get("created_at") or rm.get("createdAt"),
                "updated_at": rm.get("updated_at") or rm.get("updatedAt"),
                "shared_count": rm.get("shared_count"),
            }

        for shared in rm.get("shared_relations", []) or []:
            target_label = shared.get("target_label")
            target_value = shared.get("target_value")
            if not target_label or target_value is None:
                continue
            dim_id = node_id_for(
                target_label,
                target_value,
                rm_access_entity if target_label == "OM_Entity" else None,
            )
            if dim_id not in nodes:
                nodes[dim_id] = {
                    "id": dim_id,
                    "label": target_label,
                    "value": target_value,
                }
            if target_label == "OM_Entity" and shared.get("targetDisplayName"):
                nodes[dim_id]["displayName"] = shared.get("targetDisplayName")
            edge = {
                "source": rm_node_id,
                "target": dim_id,
                "type": shared.get("type"),
            }
            if shared.get("other_value") is not None:
                edge["value"] = shared.get("other_value")
            edges.append(edge)

    # Extract only non-redundant fields from related memories.
    # Content and metadata are already in nodes[]; related[] only needs
    # the ID and relationship info (sharedCount, sharedRelations).
    related_lean = [
        {
            "id": r.get("id"),
            "sharedCount": r.get("sharedCount"),
            "sharedRelations": r.get("sharedRelations"),
        }
        for r in related
    ]

    return {
        "seed": seed,
        "seed_memory_id": memory_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "relations": relations,
        "related": related_lean,
    }


def get_graph_relations(
    query: str,
    user_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get Mem0 Graph Memory relations for a query.

    This queries the LLM-extracted entity graph, not the metadata projection.

    Args:
        query: Search query
        user_id: String user ID
        limit: Maximum number of relations to return

    Returns:
        List of relation dicts from Mem0 Graph Memory
        Empty list if graph memory is not enabled or query fails
    """
    try:
        from app.utils.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client is None:
            return []

        # Check if graph memory is enabled (client has graph attribute)
        if not hasattr(memory_client, 'graph') or memory_client.graph is None:
            return []

        # Query the Mem0 graph
        result = memory_client.graph.search(
            query=query,
            filters={"user_id": user_id},
            limit=limit,
        )

        # Normalize the result format
        if isinstance(result, dict) and "relations" in result:
            return result["relations"]
        elif isinstance(result, list):
            return result
        else:
            return []

    except AttributeError:
        # Graph not enabled on memory client
        return []
    except Exception as e:
        logger.warning(f"Error querying graph relations: {e}")
        return []


def is_graph_enabled() -> bool:
    """
    Check if Neo4j graph projection is enabled.

    Returns:
        True if metadata projector is available
    """
    return _get_projector() is not None


def is_mem0_graph_enabled() -> bool:
    """
    Check if Mem0 Graph Memory is enabled.

    Returns:
        True if Mem0 client has graph memory configured
    """
    try:
        from app.utils.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client is None:
            return False

        return hasattr(memory_client, 'graph') and memory_client.graph is not None

    except Exception:
        return False


# =============================================================================
# Entity Co-Mention Edge Operations (OM_CO_MENTIONED)
# =============================================================================

def update_entity_edges_on_memory_add(memory_id: str, user_id: str) -> bool:
    """
    Update entity-to-entity edges after memory add.

    Creates/increments OM_CO_MENTIONED edges between co-mentioned entities.
    Fails gracefully - returns True if Neo4j not configured.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        True if successful or Neo4j not configured
    """
    projector = _get_projector()
    if projector is None:
        return True  # Not configured, succeed silently

    try:
        return projector.update_entity_edges_on_add(memory_id, user_id)
    except Exception as e:
        logger.warning(f"Error updating entity edges on add for {memory_id}: {e}")
        return False


def update_entity_edges_on_memory_delete(memory_id: str, user_id: str) -> bool:
    """
    Update entity-to-entity edges after memory delete.

    Decrements counts and removes edges with count=0.
    Fails gracefully - returns True if Neo4j not configured.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        True if successful or Neo4j not configured
    """
    projector = _get_projector()
    if projector is None:
        return True

    try:
        return projector.update_entity_edges_on_delete(memory_id, user_id)
    except Exception as e:
        logger.warning(f"Error updating entity edges on delete for {memory_id}: {e}")
        return False


def get_entity_network_from_graph(
    entity_name: str,
    user_id: str,
    min_count: int = 1,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the co-mention network for an entity.

    Returns entities that appear in the same memories as this entity,
    with connection counts and sample memory IDs.

    Args:
        entity_name: Name of the entity
        user_id: String user ID
        min_count: Minimum co-mention count
        limit: Maximum connections to return

    Returns:
        Dict with entity name, connections list, and total count
    """
    projector = _get_projector()
    if projector is None:
        return None

    try:
        return projector.get_entity_connections(
            entity_name=entity_name,
            user_id=user_id,
            min_count=min_count,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Error getting entity network for {entity_name}: {e}")
        return None


def get_entity_display_name_from_graph(
    entity_name: str,
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Resolve displayName for a given entity name.
    """
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured():
        return None
    if not is_neo4j_healthy():
        return None

    query = """
    MATCH (e:OM_Entity {name: $entityName})
    WHERE (
      (e.accessEntity IS NOT NULL AND (
        e.accessEntity IN $accessEntities
        OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
      ))
      OR (e.accessEntity IS NULL AND e.userId = $userId)
    )
    RETURN coalesce(e.displayName, e.name) AS displayName
    LIMIT 1
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                entityName=entity_name,
                accessEntities=access_entities or [f"user:{user_id}"],
                accessEntityPrefixes=access_entity_prefixes or [],
            )
            record = result.single()
            if record and record.get("displayName"):
                return record["displayName"]
    except Exception as e:
        logger.warning(f"Error getting entity displayName for {entity_name}: {e}")
    return None


def backfill_entity_edges_in_graph(
    user_id: str,
    min_count: int = 1,
    access_entity: Optional[str] = None,
) -> int:
    """
    Backfill all OM_CO_MENTIONED edges for a user.

    Args:
        user_id: String user ID
        min_count: Minimum co-occurrence count to create edge

    Returns:
        Number of edges created
    """
    projector = _get_projector()
    if projector is None:
        return 0

    try:
        return projector.backfill_entity_edges(user_id, min_count, access_entity=access_entity)
    except Exception as e:
        logger.warning(f"Error backfilling entity edges for {user_id}: {e}")
        return 0


def get_entities_for_memory_from_graph(
    memory_id: str,
    user_id: str,
    limit: int = 200,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Get all entity names connected to a memory via OM_ABOUT.

    Args:
        memory_id: UUID of the memory (string)
        user_id: String user ID
        limit: Max entities to return (safety cap)

    Returns:
        List of entity names (distinct, sorted)
    """
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured():
        return []
    if not is_neo4j_healthy():
        logger.debug("Neo4j unhealthy, skipping get_entities_for_memory_from_graph")
        return []

    query = """
    MATCH (m:OM_Memory {id: $memoryId})-[:OM_ABOUT]->(e:OM_Entity)
    WHERE (
      (m.accessEntity IS NOT NULL AND (
        m.accessEntity IN $accessEntities
        OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
      ))
      OR (m.accessEntity IS NULL AND m.userId = $userId)
    )
      AND (
        (e.accessEntity IS NOT NULL AND e.accessEntity = m.accessEntity)
        OR (e.accessEntity IS NULL AND e.userId = $userId)
      )
    RETURN DISTINCT e.name AS name
    ORDER BY name
    LIMIT $limit
    """

    try:
        access_entities = access_entities or [f"user:{user_id}"]
        access_entity_prefixes = access_entity_prefixes or []
        with get_neo4j_session() as session:
            result = session.run(
                query,
                memoryId=memory_id,
                userId=user_id,
                limit=limit,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
            )
            names: List[str] = []
            for record in result:
                name = record.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            return names
    except Exception as e:
        logger.warning(f"Error getting entities for memory {memory_id}: {e}")
        return []


def refresh_co_mention_edges_for_entities(
    user_id: str,
    entity_names: List[str],
    access_entity: Optional[str] = None,
    *,
    max_entities: int = 80,
    max_pairs: int = 3000,
) -> bool:
    """
    Recompute OM_CO_MENTIONED edges for a subset of entities.

    This is intended for update flows where a single memory's entity set changed and we want
    counts to reflect the true number of shared memories across the user's graph.

    Args:
        user_id: String user ID
        entity_names: List of entity names to refresh (will be de-duplicated)
        max_entities: Safety cap to avoid O(n^2) blowups
        max_pairs: Safety cap on number of pairs to refresh

    Returns:
        True if successful or Neo4j not configured
    """
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured():
        return True
    if not is_neo4j_healthy():
        logger.debug("Neo4j unhealthy, skipping refresh_co_mention_edges_for_entities")
        return False

    if not entity_names:
        return True

    unique_entities = sorted({e.strip() for e in entity_names if isinstance(e, str) and e.strip()})
    if not unique_entities:
        return True

    if len(unique_entities) > max_entities:
        logger.warning(
            "Co-mention refresh capped: %s entities provided, limiting to %s",
            len(unique_entities),
            max_entities,
        )
        unique_entities = unique_entities[:max_entities]

    pairs: List[List[str]] = []
    for a, b in itertools.combinations(unique_entities, 2):
        pairs.append([a, b])
        if len(pairs) >= max_pairs:
            logger.warning("Co-mention refresh capped: limiting to %s pairs", max_pairs)
            break

    if not pairs:
        return True

    query = """
    UNWIND $pairs AS pair
    MATCH (e1:OM_Entity {accessEntity: $accessEntity, name: pair[0]})
    MATCH (e2:OM_Entity {accessEntity: $accessEntity, name: pair[1]})
    OPTIONAL MATCH (e1)<-[:OM_ABOUT]-(m:OM_Memory)-[:OM_ABOUT]->(e2)
    WHERE coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
    WITH e1, e2, count(DISTINCT m) AS cnt, collect(DISTINCT m.id)[0..5] AS memoryIds
    FOREACH (_ IN CASE WHEN cnt > 0 THEN [1] ELSE [] END |
      MERGE (e1)-[r:OM_CO_MENTIONED {accessEntity: $accessEntity}]->(e2)
      ON CREATE SET r.createdAt = datetime(),
                    r.userId = $userId
      SET r.count = cnt,
          r.memoryIds = memoryIds,
          r.updatedAt = datetime()
    )
    FOREACH (_ IN CASE WHEN cnt = 0 THEN [1] ELSE [] END |
      MATCH (e1)-[r:OM_CO_MENTIONED {accessEntity: $accessEntity}]->(e2)
      DELETE r
    )
    RETURN count(*) AS processed
    """

    try:
        with get_neo4j_session() as session:
            session.run(
                query,
                userId=user_id,
                pairs=pairs,
                accessEntity=access_entity or f"user:{user_id}",
                legacyAccessEntity=f"user:{user_id}",
            ).consume()
        return True
    except Exception as e:
        logger.warning(f"Error refreshing co-mention edges: {e}")
        return False


# =============================================================================
# Tag Co-Occurrence Edge Operations (OM_COOCCURS)
# =============================================================================

def update_tag_edges_on_memory_add(memory_id: str, user_id: str) -> bool:
    """
    Update tag-to-tag edges after memory add.

    Creates/increments OM_COOCCURS edges between co-occurring tags.
    Fails gracefully - returns True if Neo4j not configured.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        True if successful or Neo4j not configured
    """
    projector = _get_projector()
    if projector is None:
        return True

    try:
        return projector.update_tag_edges_on_add(memory_id, user_id)
    except Exception as e:
        logger.warning(f"Error updating tag edges on add for {memory_id}: {e}")
        return False


def get_related_tags_from_graph(
    tag_key: str,
    user_id: str,
    min_count: int = 1,
    limit: int = 20,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get co-occurring tags for a given tag.

    Args:
        tag_key: The tag key to find related tags for
        user_id: String user ID
        min_count: Minimum co-occurrence count
        limit: Maximum tags to return

    Returns:
        List of related tags with count and PMI
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.get_related_tags(
            tag_key=tag_key,
            user_id=user_id,
            min_count=min_count,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Error getting related tags for {tag_key}: {e}")
        return []


def backfill_tag_edges_in_graph(
    user_id: str,
    min_count: int = 2,
    min_pmi: float = 0.0,
    access_entity: Optional[str] = None,
) -> int:
    """
    Backfill all OM_COOCCURS edges for a user with PMI calculation.

    Args:
        user_id: String user ID
        min_count: Minimum co-occurrence count to create edge
        min_pmi: Minimum PMI score to create edge

    Returns:
        Number of edges created
    """
    projector = _get_projector()
    if projector is None:
        return 0

    try:
        return projector.backfill_tag_edges(
            user_id,
            min_count,
            min_pmi,
            access_entity=access_entity,
        )
    except Exception as e:
        logger.warning(f"Error backfilling tag edges for {user_id}: {e}")
        return 0


# =============================================================================
# Memory-to-Memory Similarity Edge Operations (OM_SIMILAR)
# =============================================================================

_similarity_projector = None
_similarity_projector_initialized = False


def _get_similarity_projector():
    """Get the similarity projector (lazy initialization)."""
    global _similarity_projector, _similarity_projector_initialized

    if _similarity_projector_initialized:
        return _similarity_projector

    _similarity_projector_initialized = True

    try:
        from app.graph.similarity_projector import get_similarity_projector
        _similarity_projector = get_similarity_projector()
        if _similarity_projector:
            logger.info("Similarity projector initialized for graph operations")
    except ImportError as e:
        logger.debug(f"Similarity projector module not available: {e}")
        _similarity_projector = None
    except Exception as e:
        logger.warning(f"Failed to initialize similarity projector: {e}")
        _similarity_projector = None

    return _similarity_projector


def project_similarity_edges_for_memory(memory_id: str, user_id: str) -> bool:
    """
    Project similarity edges for a memory (real-time).

    Finds K nearest neighbors in Qdrant and creates OM_SIMILAR edges in Neo4j.
    Fails gracefully - returns True if not configured.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        True if successful or not configured
    """
    projector = _get_similarity_projector()
    if projector is None:
        return True  # Not configured, succeed silently

    try:
        projector.project_similarity_edges(memory_id, user_id)
        return True
    except Exception as e:
        logger.warning(f"Error projecting similarity edges for {memory_id}: {e}")
        return False


def delete_similarity_edges_for_memory(memory_id: str, user_id: str) -> bool:
    """
    Delete similarity edges for a memory.

    Called when a memory is deleted.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        True if successful or not configured
    """
    projector = _get_similarity_projector()
    if projector is None:
        return True

    try:
        return projector.delete_similarity_edges(memory_id, user_id)
    except Exception as e:
        logger.warning(f"Error deleting similarity edges for {memory_id}: {e}")
        return False


def get_similar_memories_from_graph(
    memory_id: str,
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    min_score: float = 0.0,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get pre-computed similar memories from Neo4j.

    Uses OM_SIMILAR edges created from Qdrant embeddings.

    Args:
        memory_id: UUID of the seed memory
        user_id: String user ID
        allowed_memory_ids: Optional allowlist for ACL
        min_score: Minimum similarity score
        limit: Maximum memories to return

    Returns:
        List of similar memory dicts with similarity_score and rank
    """
    projector = _get_similarity_projector()
    if projector is None:
        return []

    try:
        return projector.get_similar_memories(
            memory_id=memory_id,
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            min_score=min_score,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Error getting similar memories for {memory_id}: {e}")
        return []


def is_similarity_enabled() -> bool:
    """Check if similarity projection is enabled."""
    return _get_similarity_projector() is not None


def retrieve_via_similarity_graph(
    user_id: str,
    seed_memory_ids: List[str],
    allowed_memory_ids: Optional[set] = None,
    limit: int = 30,
    min_score: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Retrieve memories via OM_SIMILAR graph traversal from seed memories.

    This is the graph retrieval component of RRF hybrid search.
    It finds memories that are similar to the top vector search results,
    potentially surfacing relevant memories that vector search missed.

    Strategy:
    1. Start from seed memories (top vector results)
    2. Traverse OM_SIMILAR edges to find connected memories
    3. Rank by: number of seed connections + average similarity

    Args:
        user_id: User ID for filtering
        seed_memory_ids: Top-K memories from vector search as seeds
        allowed_memory_ids: ACL filter (optional set)
        limit: Max results to return
        min_score: Minimum similarity score on OM_SIMILAR edges

    Returns:
        List of memory dicts with graph ranking info:
        - id: Memory UUID
        - content: Memory content
        - category, scope, artifact_type, artifact_ref: metadata fields
        - seedConnections: How many seed memories link to this one
        - avgSimilarity: Average similarity score from seeds
        - maxSimilarity: Maximum similarity score from any seed

    Performance:
        - Typical latency: 50-100ms for 5 seeds, 30 result limit
        - Uses single Cypher query with UNWIND for batch traversal
    """
    from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured() or not seed_memory_ids:
        return []

    if not is_neo4j_healthy():
        logger.debug("Neo4j unhealthy, skipping graph retrieval")
        return []

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            # Multi-seed traversal query
            # Finds memories connected to ANY of the seeds via OM_SIMILAR
            # Ranks by number of seed connections (more = more relevant)
            query = """
            UNWIND $seedIds AS seedId
            MATCH (seed:OM_Memory {id: seedId})
            WITH seed, seedId, coalesce(seed.accessEntity, $legacyAccessEntity) AS accessEntity
            MATCH (seed)-[r:OM_SIMILAR]->(candidate:OM_Memory)
            WHERE r.score >= $minScore
              AND candidate.id NOT IN $seedIds
              AND (
                (candidate.accessEntity IS NOT NULL AND candidate.accessEntity = accessEntity)
                OR (candidate.accessEntity IS NULL AND candidate.userId = $userId)
              )
              AND (
                (r.accessEntity IS NOT NULL AND r.accessEntity = accessEntity)
                OR (r.accessEntity IS NULL AND r.userId = $userId)
              )
            WITH candidate,
                 count(DISTINCT seedId) AS seedConnections,
                 max(r.score) AS maxSimilarity,
                 avg(r.score) AS avgSimilarity
            ORDER BY seedConnections DESC, avgSimilarity DESC
            LIMIT $limit
            RETURN candidate.id AS id,
                   candidate.content AS content,
                   candidate.category AS category,
                   candidate.scope AS scope,
                   candidate.artifactType AS artifactType,
                   candidate.artifactRef AS artifactRef,
                   candidate.source AS source,
                   seedConnections,
                   avgSimilarity,
                   maxSimilarity
            """

            result = session.run(
                query,
                seedIds=seed_memory_ids,
                userId=user_id,
                minScore=min_score,
                limit=limit,
                legacyAccessEntity=f"user:{user_id}",
            )

            memories: List[Dict[str, Any]] = []
            for record in result:
                mem = {
                    "id": record["id"],
                    "content": record["content"],
                    "category": record["category"],
                    "scope": record["scope"],
                    "artifactType": record["artifactType"],
                    "artifactRef": record["artifactRef"],
                    "source": record["source"],
                    "seedConnections": record["seedConnections"],
                    "avgSimilarity": record["avgSimilarity"],
                    "maxSimilarity": record["maxSimilarity"],
                }

                # Apply ACL filter if provided
                if allowed_memory_ids is None or mem["id"] in allowed_memory_ids:
                    memories.append(mem)

            logger.debug(
                f"Graph retrieval: {len(seed_memory_ids)} seeds -> "
                f"{len(memories)} candidates (limit={limit})"
            )
            return memories

    except Exception as e:
        logger.warning(f"Graph retrieval failed: {e}")
        # Mark Neo4j as unhealthy to trigger circuit breaker
        try:
            from app.graph.neo4j_client import mark_neo4j_unhealthy
            mark_neo4j_unhealthy()
        except ImportError:
            pass
        return []


def retrieve_via_entity_graph(
    user_id: str,
    entity_names: List[str],
    allowed_memory_ids: Optional[set] = None,
    limit: int = 30,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve memories connected to specific entities.

    Used when query routing detects entity mentions and wants
    entity-focused retrieval over pure vector similarity.

    Args:
        user_id: User ID for filtering
        entity_names: List of entity names to find memories about
        allowed_memory_ids: ACL filter (optional set)
        limit: Max results to return

    Returns:
        List of memory dicts with entity connection info:
        - id: Memory UUID
        - content: Memory content
        - matchedEntities: Count of matched entities
        - entityNames: List of matched entity names
    """
    from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured() or not entity_names:
        return []

    if not is_neo4j_healthy():
        return []

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            # Find memories connected to any of the specified entities
            query = """
            UNWIND $entityNames AS entityName
            MATCH (e:OM_Entity {name: entityName})
            WHERE (
              (e.accessEntity IS NOT NULL AND (
                e.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
              ))
              OR (e.accessEntity IS NULL AND e.userId = $userId)
            )
            MATCH (m:OM_Memory)-[:OM_ABOUT]->(e)
            WHERE (
              (m.accessEntity IS NOT NULL AND m.accessEntity = e.accessEntity)
              OR (m.accessEntity IS NULL AND m.userId = $userId)
            )
            WITH m,
                 count(DISTINCT entityName) AS matchedEntities,
                 collect(DISTINCT entityName) AS entityNames,
                 collect(DISTINCT {
                     name: e.name,
                     displayName: coalesce(e.displayName, e.name)
                 }) AS entityDetails
            ORDER BY matchedEntities DESC
            LIMIT $limit
            RETURN m.id AS id,
                   m.content AS content,
                   m.category AS category,
                   m.scope AS scope,
                   m.artifactType AS artifactType,
                   m.artifactRef AS artifactRef,
                   matchedEntities,
                   entityNames,
                   entityDetails
            """

            result = session.run(
                query,
                entityNames=entity_names,
                userId=user_id,
                limit=limit,
                accessEntities=access_entities or [f"user:{user_id}"],
                accessEntityPrefixes=access_entity_prefixes or [],
            )

            memories: List[Dict[str, Any]] = []
            for record in result:
                mem = {
                    "id": record["id"],
                    "content": record["content"],
                    "category": record["category"],
                    "scope": record["scope"],
                    "artifactType": record["artifactType"],
                    "artifactRef": record["artifactRef"],
                    "matchedEntities": record["matchedEntities"],
                    "entityNames": record["entityNames"],
                    "entityDetails": record.get("entityDetails") or [],
                }

                if allowed_memory_ids is None or mem["id"] in allowed_memory_ids:
                    memories.append(mem)

            return memories

    except Exception as e:
        logger.warning(f"Entity graph retrieval failed: {e}")
        return []


def find_bridge_entities(
    user_id: str,
    entity_names: List[str],
    max_bridges: int = 5,
    min_count: int = 2,
    max_hops: int = 3,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find entities that bridge between query entities via OM_CO_MENTIONED.

    For 2+ entities, finds intermediate entities that connect them
    through the co-mention network (up to max_hops).

    Args:
        user_id: User ID for filtering
        entity_names: List of query entity names (need 2+)
        max_bridges: Maximum bridge entities to return
        min_count: Minimum co-mention count threshold
        max_hops: Maximum path length (default 3 for A→B→C→D)

    Returns:
        List of bridge entity dicts with:
        - name: Bridge entity name
        - connectionStrength: Sum of co-mention counts
        - connects: List of query entities this bridges
    """
    logger.debug(f"find_bridge_entities called with entity_names={entity_names}, user_id={user_id}")

    if len(entity_names) < 2:
        return []

    from app.graph.neo4j_client import is_neo4j_configured

    if not is_neo4j_configured():
        return []

    # Note: We intentionally don't check is_neo4j_healthy() here because:
    # 1. The circuit breaker may have been tripped by unrelated operations
    # 2. This function has its own exception handling
    # 3. We want to give Neo4j a chance to work even if previous ops failed

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            # Find entities connected to 2+ query entities via OM_CO_MENTIONED paths
            # max_hops controls path length (3 = A→B→C→D)
            query = f"""
            UNWIND $entityNames AS name
            MATCH (e:OM_Entity {{name: name}})
            WHERE (
              (e.accessEntity IS NOT NULL AND (
                e.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
              ))
              OR (e.accessEntity IS NULL AND e.userId = $userId)
            )
            MATCH (e)-[r:OM_CO_MENTIONED*1..{max_hops}]-(bridge:OM_Entity)
            WHERE NOT bridge.name IN $entityNames
              AND (
                (bridge.accessEntity IS NOT NULL AND bridge.accessEntity = e.accessEntity)
                OR (bridge.accessEntity IS NULL AND bridge.userId = $userId)
              )
              AND all(rel IN r WHERE (
                (rel.accessEntity IS NOT NULL AND rel.accessEntity = e.accessEntity)
                OR (rel.accessEntity IS NULL AND rel.userId = $userId)
              ))
            WITH bridge,
                 collect(DISTINCT name) AS connectedEntities,
                 sum(reduce(s = 0, rel IN r | s + coalesce(rel.count, 1))) AS totalStrength
            WHERE size(connectedEntities) >= 2
            RETURN bridge.name AS name,
                   coalesce(bridge.displayName, bridge.name) AS displayName,
                   totalStrength AS connectionStrength,
                   connectedEntities AS connects
            ORDER BY size(connectedEntities) DESC, totalStrength DESC
            LIMIT $maxBridges
            """

            result = session.run(
                query,
                entityNames=[e.lower() for e in entity_names],
                userId=user_id,
                maxBridges=max_bridges,
                accessEntities=access_entities or [f"user:{user_id}"],
                accessEntityPrefixes=access_entity_prefixes or [],
            )

            bridges = [dict(record) for record in result]
            if bridges:
                logger.debug(
                    f"Found {len(bridges)} bridge entities for {entity_names}: "
                    f"{[b['name'] for b in bridges]}"
                )
            return bridges

    except Exception as e:
        logger.warning(f"Bridge entity detection failed: {e}")
        return []


# =============================================================================
# Full-Text Search Operations
# =============================================================================

def fulltext_search_memories_in_graph(
    search_text: str,
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Full-text search across memory content using Neo4j full-text index.

    Supports Lucene query syntax:
    - AND/OR: "work AND meeting"
    - Wildcards: "mem*"
    - Fuzzy: "memory~2"
    - Phrase: '"exact phrase"'

    Args:
        search_text: Search query (Lucene syntax supported)
        user_id: String user ID
        allowed_memory_ids: Optional allowlist for ACL
        limit: Maximum results to return

    Returns:
        List of memory dicts with searchScore
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.fulltext_search_memories(
            search_text=search_text,
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Full-text memory search failed: {e}")
        return []


def fulltext_search_entities_in_graph(
    search_text: str,
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Full-text search across entity names.

    Args:
        search_text: Search query
        user_id: String user ID
        limit: Maximum results to return

    Returns:
        List of entity dicts with searchScore
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.fulltext_search_entities(
            search_text=search_text,
            user_id=user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Full-text entity search failed: {e}")
        return []


# =============================================================================
# Graph Analytics Operations
# =============================================================================

def get_entity_centrality_from_graph(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get entity importance based on co-mention network.

    Uses degree centrality and total mention count.

    Args:
        user_id: String user ID
        limit: Maximum entities to return

    Returns:
        List of entities with connections and mentionCount
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.get_entity_centrality(
            user_id=user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Failed to get entity centrality: {e}")
        return []


def get_memory_connectivity_from_graph(
    user_id: str,
    allowed_memory_ids: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get memory connectivity statistics.

    Shows how connected each memory is through entities and similarity.

    Args:
        user_id: String user ID
        allowed_memory_ids: Optional allowlist for ACL
        limit: Maximum memories to return

    Returns:
        List of memories with connectivity scores
    """
    projector = _get_projector()
    if projector is None:
        return []

    try:
        return projector.get_memory_connectivity(
            user_id=user_id,
            allowed_memory_ids=allowed_memory_ids,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Failed to get memory connectivity: {e}")
        return []


def get_graph_statistics(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get statistics about the user's graph.

    Args:
        user_id: String user ID

    Returns:
        Dict with graph statistics (memoryCount, entityCount, edge counts)
    """
    projector = _get_projector()
    if projector is None:
        return {}

    try:
        return projector.get_graph_statistics(
            user_id=user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to get graph statistics: {e}")
        return {}


# =============================================================================
# Neo4j GDS (Graph Data Science) Operations
# =============================================================================

_gds_operations = None
_gds_operations_initialized = False


def _get_gds_operations():
    """Get the GDS operations instance (lazy initialization)."""
    global _gds_operations, _gds_operations_initialized

    if _gds_operations_initialized:
        return _gds_operations

    _gds_operations_initialized = True

    try:
        from app.graph.gds_operations import get_gds_operations
        _gds_operations = get_gds_operations()
        if _gds_operations and _gds_operations.is_gds_available():
            logger.info("GDS operations initialized for graph analytics")
    except ImportError as e:
        logger.debug(f"GDS operations module not available: {e}")
        _gds_operations = None
    except Exception as e:
        logger.warning(f"Failed to initialize GDS operations: {e}")
        _gds_operations = None

    return _gds_operations


def is_gds_available() -> bool:
    """
    Check if Neo4j GDS (Graph Data Science) library is available.

    Returns:
        True if GDS is installed and accessible
    """
    gds = _get_gds_operations()
    return gds is not None and gds.is_gds_available()


def get_entity_pagerank(
    user_id: str,
    limit: int = 50,
    write_to_nodes: bool = False,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run PageRank on entity co-mention network.

    Identifies the most influential entities based on how they're
    connected through shared memories.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        limit: Maximum results to return
        write_to_nodes: Whether to write pageRank property to nodes
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of entities with pageRankScore
    """
    gds = _get_gds_operations()
    if gds is None:
        return []

    try:
        return gds.entity_pagerank(
            user_id=user_id,
            limit=limit,
            write_to_nodes=write_to_nodes,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to run entity PageRank: {e}")
        return []


def detect_entity_communities(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect communities in the entity co-mention network using Louvain.

    Groups entities that frequently appear together in memories.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Dict with communities list and statistics
    """
    gds = _get_gds_operations()
    if gds is None:
        return {"communities": [], "stats": {}}

    try:
        return gds.detect_entity_communities(
            user_id=user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to detect entity communities: {e}")
        return {"communities": [], "stats": {}}


def detect_memory_communities(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect communities in the memory similarity network.

    Groups memories that are semantically similar to each other.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Dict with communities list and statistics
    """
    gds = _get_gds_operations()
    if gds is None:
        return {"communities": [], "stats": {}}

    try:
        return gds.detect_memory_communities(
            user_id=user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to detect memory communities: {e}")
        return {"communities": [], "stats": {}}


def find_similar_entities_gds(
    user_id: str,
    entity_name: Optional[str] = None,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find similar entities based on shared memory connections using GDS.

    Uses Jaccard similarity on shared neighbors.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        entity_name: Optional specific entity to find similar to
        limit: Maximum results to return
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of entity pairs with similarity scores
    """
    gds = _get_gds_operations()
    if gds is None:
        return []

    try:
        return gds.find_similar_entities(
            user_id=user_id,
            entity_name=entity_name,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to find similar entities: {e}")
        return []


def get_entity_betweenness(
    user_id: str,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Calculate betweenness centrality for entities.

    Entities with high betweenness act as bridges between
    different clusters of memories.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        limit: Maximum results to return
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of entities with betweenness scores
    """
    gds = _get_gds_operations()
    if gds is None:
        return []

    try:
        return gds.entity_betweenness(
            user_id=user_id,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to calculate betweenness centrality: {e}")
        return []


def generate_entity_embeddings(
    user_id: str,
    write_to_nodes: bool = False,
    limit: int = 100,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate FastRP graph embeddings for entities.

    These embeddings capture the structural position of entities
    in the co-mention graph and can be used for ML tasks.

    Requires Neo4j GDS plugin.

    Args:
        user_id: String user ID (legacy fallback)
        write_to_nodes: Whether to write embeddings to node properties
        limit: Maximum results to return
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of entities with embeddings
    """
    gds = _get_gds_operations()
    if gds is None:
        return []

    try:
        return gds.generate_entity_embeddings(
            user_id=user_id,
            write_to_nodes=write_to_nodes,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except Exception as e:
        logger.warning(f"Failed to generate entity embeddings: {e}")
        return []


# =============================================================================
# Entity Normalization Operations
# =============================================================================

def find_duplicate_entities_in_graph(
    user_id: str,
    min_variants: int = 2,
    access_entity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find duplicate entity candidates for normalization.

    Args:
        user_id: String user ID
        min_variants: Minimum number of variants to report

    Returns:
        List of duplicate groups with canonical suggestions
    """
    try:
        from app.graph.entity_normalizer import identify_duplicates
        duplicates = identify_duplicates(user_id, min_variants, access_entity=access_entity)
        return [
            {
                "canonical": d.canonical,
                "variants": [{"name": v.name, "memories": v.memory_count} for v in d.variants],
                "total_memories": d.total_memories,
            }
            for d in duplicates
        ]
    except Exception as e:
        logger.warning(f"Error finding duplicate entities: {e}")
        return []


def normalize_entities_in_graph(
    user_id: str,
    canonical_name: Optional[str] = None,
    variant_names: Optional[List[str]] = None,
    auto: bool = False,
    dry_run: bool = True,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize entity duplicates.

    Either specify canonical_name + variant_names for manual merge,
    or set auto=True for automatic normalization of all duplicates.

    Args:
        user_id: String user ID
        canonical_name: Target entity name (for manual merge)
        variant_names: Variants to merge (for manual merge)
        auto: If True, automatically normalize all duplicates
        dry_run: If True, only simulate

    Returns:
        Merge statistics
    """
    try:
        from app.graph.entity_normalizer import (
            merge_entity_variants,
            auto_normalize_entities,
        )

        if auto:
            return auto_normalize_entities(user_id, dry_run=dry_run, access_entity=access_entity)
        elif canonical_name and variant_names:
            return merge_entity_variants(
                user_id=user_id,
                canonical_name=canonical_name,
                variant_names=variant_names,
                dry_run=dry_run,
                access_entity=access_entity,
            )
        else:
            return {"error": "Either set auto=True or provide canonical_name + variant_names"}
    except Exception as e:
        logger.warning(f"Error normalizing entities: {e}")
        return {"error": str(e)}


# =============================================================================
# Semantic Entity Normalization (Multi-Phase Detection)
# =============================================================================

async def find_semantic_duplicates(
    user_id: str,
    threshold: float = 0.7,
    access_entity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Find semantic entity duplicates using multi-phase detection.

    Uses:
    1. String similarity (Levenshtein/fuzzy)
    2. Prefix/suffix matching
    3. Domain normalization (.community, etc.)

    Args:
        user_id: String user ID
        threshold: Minimum confidence for reporting (0.0-1.0)

    Returns:
        List of duplicate groups with confidence scores
    """
    try:
        from app.graph.semantic_entity_normalizer import (
            SemanticEntityNormalizer,
            get_all_user_entities,
        )

        access_entities = [access_entity] if access_entity else None
        entities = await get_all_user_entities(
            user_id,
            access_entities=access_entities,
            access_entity_prefixes=None,
        )
        if len(entities) < 2:
            return []

        normalizer = SemanticEntityNormalizer()
        normalizer.MERGE_CONFIDENCE_THRESHOLD = threshold

        candidates = await normalizer.find_merge_candidates(entities)
        groups = normalizer.cluster_candidates(candidates)

        return [
            {
                "canonical": g.canonical,
                "variants": g.variants,
                "confidence": round(g.confidence, 2),
                "sources": g.merge_sources,
            }
            for g in groups
        ]
    except Exception as e:
        logger.warning(f"Error finding semantic duplicates: {e}")
        return []


async def normalize_entities_semantic(
    user_id: str,
    canonical: Optional[str] = None,
    variants: Optional[List[str]] = None,
    auto: bool = False,
    threshold: float = 0.7,
    dry_run: bool = True,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize entities using semantic detection.

    Args:
        user_id: String user ID
        canonical: Target entity name (for manual merge)
        variants: Variants to merge (for manual merge)
        auto: If True, auto-merge all detected duplicates
        threshold: Minimum confidence for merge
        dry_run: If True, only simulate

    Returns:
        Normalization results with statistics
    """
    try:
        from app.graph.semantic_entity_normalizer import (
            SemanticEntityNormalizer,
            SemanticCanonicalEntity,
            get_all_user_entities,
        )
        from app.graph.entity_edge_migrator import estimate_migration_impact
        from app.graph.gds_signal_refresh import refresh_graph_signals

        results = {
            "dry_run": dry_run,
            "merges": [],
            "total_edges_migrated": 0,
            "total_variants_merged": 0,
        }

        normalizer = SemanticEntityNormalizer()
        normalizer.MERGE_CONFIDENCE_THRESHOLD = threshold

        if canonical and variants:
            # Manual merge
            group = SemanticCanonicalEntity(
                canonical=canonical,
                variants=variants,
                confidence=1.0,  # User-specified
                merge_sources={v: ["manual"] for v in variants}
            )
            groups = [group]
        elif auto:
            # Auto-detect and merge
            access_entities = [access_entity] if access_entity else None
            entities = await get_all_user_entities(
                user_id,
                access_entities=access_entities,
                access_entity_prefixes=None,
            )
            if len(entities) < 2:
                return {"message": "Less than 2 entities, nothing to normalize"}

            candidates = await normalizer.find_merge_candidates(entities)
            groups = normalizer.cluster_candidates(candidates)
        else:
            return {"error": "Either set auto=True or provide canonical + variants"}

        results["merge_groups"] = len(groups)

        for group in groups:
            if dry_run:
                impact = await estimate_migration_impact(
                    user_id=user_id,
                    canonical=group.canonical,
                    variants=group.variants,
                    access_entity=access_entity,
                )
                results["merges"].append({
                    "canonical": group.canonical,
                    "variants": group.variants,
                    "confidence": round(group.confidence, 2),
                    "estimated_impact": impact["estimated_changes"],
                })
                results["total_edges_migrated"] += impact["estimated_changes"]["total_edges"]
            else:
                merge_result = await normalizer.execute_merge(
                    user_id=user_id,
                    group=group,
                    allowed_memory_ids=None,
                    dry_run=False,
                    access_entity=access_entity,
                )
                results["merges"].append(merge_result)
                if merge_result.get("edge_migration"):
                    results["total_edges_migrated"] += merge_result["edge_migration"]["total_migrated"]

            results["total_variants_merged"] += len(group.variants)

        # Refresh signals after merge
        if not dry_run and groups:
            gds_stats = await refresh_graph_signals(
                user_id,
                access_entities=[access_entity] if access_entity else None,
            )
            results["gds_refresh"] = gds_stats

        return results

    except Exception as e:
        logger.exception(f"Error in semantic normalization: {e}")
        return {"error": str(e)}


# =============================================================================
# Typed Entity Relations Operations (OM_RELATION)
# =============================================================================

def get_entity_relations_from_graph(
    entity_name: str,
    user_id: str,
    relation_types: Optional[List[str]] = None,
    category: Optional[str] = None,
    direction: str = "both",
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get typed relationships for an entity.

    Shows semantic relationships extracted from memories, not just co-mentions.

    Args:
        entity_name: Name of the entity
        user_id: String user ID
        relation_types: Optional list of relation types to filter
        category: Optional category to filter (family, social, work, etc.)
        direction: "outgoing", "incoming", or "both"
        limit: Maximum relations to return

    Returns:
        List of relation dicts with target, type, direction, etc.
    """
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

    if not is_neo4j_configured():
        return []

    # Build type filter from category if specified
    type_filter = relation_types or []
    if category and not type_filter:
        try:
            from app.graph.relation_types import get_relations_for_category
            type_filter = get_relations_for_category(category)
        except ImportError:
            pass

    # Build direction-aware query
    if direction == "outgoing":
        match_pattern = "(e)-[r:OM_RELATION]->(other)"
    elif direction == "incoming":
        match_pattern = "(e)<-[r:OM_RELATION]-(other)"
    else:  # both
        match_pattern = "(e)-[r:OM_RELATION]-(other)"

    type_filter_clause = ""
    if type_filter:
        type_filter_clause = "AND r.type IN $typeFilter"

    query = f"""
    MATCH (e:OM_Entity {{name: $entityName}})
    WHERE (
      (e.accessEntity IS NOT NULL AND (
        e.accessEntity IN $accessEntities
        OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
      ))
      OR (e.accessEntity IS NULL AND e.userId = $userId)
    )
    MATCH {match_pattern}
    WHERE e <> other {type_filter_clause}
      AND (
        (r.accessEntity IS NOT NULL AND r.accessEntity = e.accessEntity)
        OR (r.accessEntity IS NULL AND r.userId = $userId)
      )
      AND (
        (other.accessEntity IS NOT NULL AND other.accessEntity = e.accessEntity)
        OR (other.accessEntity IS NULL AND other.userId = $userId)
      )
    RETURN
        other.name AS target,
        coalesce(other.displayName, other.name) AS targetDisplayName,
        r.type AS relationType,
        r.memoryId AS memoryId,
        coalesce(r.count, 1) AS count,
        CASE
            WHEN startNode(r) = e THEN 'outgoing'
            ELSE 'incoming'
        END AS direction
    ORDER BY count DESC
    LIMIT $limit
    """

    relations = []
    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                entityName=entity_name,
                typeFilter=type_filter,
                limit=limit,
                accessEntities=access_entities or [f"user:{user_id}"],
                accessEntityPrefixes=access_entity_prefixes or [],
            )
            for record in result:
                relations.append({
                    "target": record["target"],
                    "targetDisplayName": record.get("targetDisplayName") or record["target"],
                    "type": record["relationType"],
                    "direction": record["direction"],
                    "memory_id": record["memoryId"],
                    "count": record["count"],
                })
    except Exception as e:
        logger.warning(f"Error getting entity relations: {e}")

    return relations


# =============================================================================
# Biographical Timeline Operations (OM_TemporalEvent)
# =============================================================================

def get_biography_timeline_from_graph(
    user_id: str,
    entity_name: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get the biographical timeline for a user or specific entity.

    Args:
        user_id: String user ID
        entity_name: Optional entity name to filter
        event_types: Optional list of event types to filter
        start_year: Optional start year filter
        end_year: Optional end year filter
        limit: Maximum events to return

    Returns:
        Chronologically sorted list of events
    """
    try:
        from app.graph.temporal_events import get_biography_timeline
        return get_biography_timeline(
            user_id=user_id,
            entity_name=entity_name,
            event_types=event_types,
            start_year=start_year,
            end_year=end_year,
            limit=limit,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
    except ImportError:
        logger.debug("Temporal events module not available")
        return []
    except Exception as e:
        logger.warning(f"Error getting biography timeline: {e}")
        return []


def create_temporal_event_in_graph(
    user_id: str,
    name: str,
    event_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    description: Optional[str] = None,
    entity: Optional[str] = None,
    memory_ids: Optional[List[str]] = None,
    access_entity: Optional[str] = None,
) -> bool:
    """
    Create a temporal event in the graph.

    Args:
        user_id: String user ID
        name: Event name
        event_type: Type of event (residence, work, project, etc.)
        start_date: Start date (YYYY, YYYY-MM, or YYYY-MM-DD)
        end_date: End date
        description: Event description
        entity: Related entity name
        memory_ids: Source memory IDs

    Returns:
        True if created successfully
    """
    try:
        from app.graph.temporal_events import TemporalEvent, EventType, create_temporal_event

        # Map string to EventType enum
        try:
            etype = EventType(event_type.lower())
        except ValueError:
            etype = EventType.MILESTONE

        event = TemporalEvent(
            name=name,
            event_type=etype,
            start_date=start_date,
            end_date=end_date,
            description=description,
            entity=entity,
            memory_ids=memory_ids or [],
        )

        return create_temporal_event(user_id, event, access_entity=access_entity)
    except ImportError:
        logger.debug("Temporal events module not available")
        return False
    except Exception as e:
        logger.warning(f"Error creating temporal event: {e}")
        return False
