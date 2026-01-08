"""
Graph cache operations for hybrid retrieval.

Provides batch-fetch operations for graph signals used in reranking.
All functions gracefully return empty results if Neo4j is unavailable.

Key Features:
- Batch queries to avoid per-memory Neo4j calls during reranking
- Pre-computed graph metrics (PageRank, cluster size, entity degree)
- Tag PMI (Pointwise Mutual Information) lookup for co-occurrence relevance
- Refresh operations for maintaining cached properties

Usage:
    from app.graph.graph_cache import fetch_graph_context

    # Before reranking loop
    graph_context = fetch_graph_context(
        memory_ids=["uuid1", "uuid2", ...],
        user_id="user123",
        context_tags=["important", "trigger"],
    )

    # Use in boost calculation
    if graph_context.available:
        mem_data = graph_context.memory_cache.get(memory_id)
        pagerank = mem_data.get("maxEntityPageRank", 0)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GraphContext:
    """
    Pre-fetched graph signals for batch reranking.

    Populated once before reranking loop to avoid per-memory queries.
    All data is normalized for scoring (0-1 range where applicable).

    Attributes:
        memory_cache: Map of memory_id to graph properties
            - similarityClusterSize: Count of OM_SIMILAR edges
            - maxEntityPageRank: Highest PageRank of connected entities
            - maxEntityDegree: Highest co-mention degree of connected entities
        entity_cache: Map of access-scoped entity key (accessEntity:name) to properties
            - pageRank: Pre-computed PageRank score
            - degree: Count of OM_CO_MENTIONED edges
        tag_pmi_cache: Map of (tag1, tag2) to PMI score
        available: Whether graph data was successfully fetched
        max_pagerank: Maximum PageRank value for normalization
        max_degree: Maximum entity degree for normalization
        max_cluster_size: Maximum cluster size for normalization
    """
    memory_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entity_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tag_pmi_cache: Dict[Tuple[str, str], float] = field(default_factory=dict)
    available: bool = False
    max_pagerank: float = 1.0
    max_degree: int = 1
    max_cluster_size: int = 1


def _is_neo4j_ready() -> bool:
    """Check if Neo4j is configured and healthy."""
    try:
        from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy
        return is_neo4j_configured() and is_neo4j_healthy()
    except ImportError:
        return False
    except Exception:
        return False


def fetch_graph_context(
    memory_ids: List[str],
    user_id: str,
    context_tags: Optional[List[str]] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> GraphContext:
    """
    Batch-fetch graph signals for all memories in the search pool.

    Called ONCE before reranking loop to avoid per-memory queries.
    This is a critical performance optimization - single batch query
    instead of N individual queries.

    Args:
        memory_ids: List of memory UUIDs to fetch signals for
        user_id: User ID for filtering
        context_tags: Optional tags to fetch PMI scores for

    Returns:
        GraphContext with cached data, or empty context if unavailable

    Performance:
        - Typical latency: 10-30ms for 100 memories
        - Single Cypher query combines memory and entity lookups
    """
    from app.graph.neo4j_client import is_neo4j_configured

    if not is_neo4j_configured() or not memory_ids:
        return GraphContext(available=False)

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            # Batch query for memory and entity signals
            # This combines multiple lookups into a single round-trip
            query = """
            UNWIND $memoryIds AS memId
            MATCH (m:OM_Memory {id: memId})
            WHERE (
              (m.accessEntity IS NOT NULL AND (
                m.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
              ))
              OR (m.accessEntity IS NULL AND m.userId = $userId)
            )
            OPTIONAL MATCH (m)-[:OM_ABOUT]->(e:OM_Entity)
            WITH m,
                 coalesce(m.similarityClusterSize, 0) AS clusterSize,
                 max(coalesce(e.pageRank, 0)) AS maxPageRank,
                 max(coalesce(e.degree, 0)) AS maxDegree,
                 collect(DISTINCT {name: e.name, accessEntity: coalesce(e.accessEntity, $legacyAccessEntity)}) AS entityRefs
            RETURN m.id AS memoryId,
                   clusterSize,
                   maxPageRank,
                   maxDegree,
                   entityRefs
            """

            access_entities = access_entities or [f"user:{user_id}"]
            access_entity_prefixes = access_entity_prefixes or []
            result = session.run(
                query,
                memoryIds=memory_ids,
                userId=user_id,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
                legacyAccessEntity=f"user:{user_id}",
            )

            memory_cache: Dict[str, Dict[str, Any]] = {}
            entity_refs: Set[Tuple[str, str]] = set()
            max_pagerank = 0.001  # Avoid division by zero
            max_degree = 1
            max_cluster_size = 1

            for record in result:
                mem_id = record["memoryId"]
                if mem_id is None:
                    continue

                memory_cache[mem_id] = {
                    "similarityClusterSize": record["clusterSize"] or 0,
                    "maxEntityPageRank": record["maxPageRank"] or 0,
                    "maxEntityDegree": record["maxDegree"] or 0,
                }
                max_pagerank = max(max_pagerank, record["maxPageRank"] or 0)
                max_degree = max(max_degree, record["maxDegree"] or 0)
                max_cluster_size = max(max_cluster_size, record["clusterSize"] or 0)

                # Collect entity names for potential further lookup
                if record["entityRefs"]:
                    for ref in record["entityRefs"]:
                        name = ref.get("name") if isinstance(ref, dict) else None
                        access_entity = ref.get("accessEntity") if isinstance(ref, dict) else None
                        if name and access_entity:
                            entity_refs.add((access_entity, name))

            # Fetch entity cache for detected entities
            entity_cache: Dict[str, Dict[str, Any]] = {}
            if entity_refs:
                entity_query = """
                UNWIND $entityRefs AS ref
                MATCH (e:OM_Entity {name: ref.name})
                WHERE coalesce(e.accessEntity, $legacyAccessEntity) = ref.accessEntity
                RETURN e.name AS name,
                       ref.accessEntity AS accessEntity,
                       coalesce(e.pageRank, 0) AS pageRank,
                       coalesce(e.degree, 0) AS degree
                """
                entity_result = session.run(
                    entity_query,
                    entityRefs=[{"name": name, "accessEntity": access_entity} for access_entity, name in entity_refs],
                    legacyAccessEntity=f"user:{user_id}",
                )
                for record in entity_result:
                    name = record.get("name")
                    access_entity = record.get("accessEntity")
                    if name and access_entity:
                        key = f"{access_entity}:{name}"
                        entity_cache[key] = {
                            "pageRank": record["pageRank"] or 0,
                            "degree": record["degree"] or 0,
                        }

            # Fetch tag PMI if context tags provided
            tag_pmi_cache: Dict[Tuple[str, str], float] = {}
            if context_tags:
                pmi_query = """
                UNWIND $tags AS tag1
                MATCH (t1:OM_Tag {key: tag1})-[r:OM_COOCCURS]-(t2:OM_Tag)
                WHERE (
                  (r.accessEntity IS NOT NULL AND (
                    r.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE r.accessEntity STARTS WITH prefix)
                  ))
                  OR (r.accessEntity IS NULL AND r.userId = $userId)
                )
                RETURN t1.key AS tag1, t2.key AS tag2, r.pmi AS pmi
                """
                pmi_result = session.run(
                    pmi_query,
                    tags=context_tags,
                    userId=user_id,
                    accessEntities=access_entities,
                    accessEntityPrefixes=access_entity_prefixes,
                )
                for record in pmi_result:
                    if record["tag1"] and record["tag2"]:
                        key = (record["tag1"], record["tag2"])
                        tag_pmi_cache[key] = record["pmi"] or 0.0

            return GraphContext(
                memory_cache=memory_cache,
                entity_cache=entity_cache,
                tag_pmi_cache=tag_pmi_cache,
                available=True,
                max_pagerank=max_pagerank,
                max_degree=max_degree,
                max_cluster_size=max_cluster_size,
            )

    except Exception as e:
        logger.warning(f"Failed to fetch graph context: {e}")
        return GraphContext(available=False)


def refresh_pagerank_cache(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Refresh PageRank values for all entities (batch operation).

    Should be called periodically (e.g., nightly) or after significant data changes.
    Uses Neo4j GDS PageRank algorithm if available.

    Args:
        user_id: User ID to refresh PageRank for (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Statistics about the refresh operation:
        - success: Whether the operation completed
        - entities_updated: Count of entities with updated PageRank
        - error: Error message if failed
    """
    try:
        from app.graph.gds_operations import entity_pagerank

        result = entity_pagerank(
            user_id=user_id,
            write_to_nodes=True,
            limit=10000,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )
        return {
            "success": True,
            "entities_updated": len(result) if result else 0,
        }
    except ImportError:
        logger.warning("GDS operations not available for PageRank refresh")
        return {"success": False, "error": "GDS not available"}
    except Exception as e:
        logger.error(f"PageRank refresh failed: {e}")
        return {"success": False, "error": str(e)}


def refresh_cluster_sizes(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Refresh similarityClusterSize on all OM_Memory nodes.

    Counts outbound OM_SIMILAR edges for each memory.
    This property is used to boost memories that are well-connected
    in the similarity graph.

    Args:
        user_id: User ID to refresh cluster sizes for (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Statistics about the refresh operation
    """
    from app.graph.neo4j_client import is_neo4j_configured

    if not is_neo4j_configured():
        return {"success": False, "error": "Neo4j not configured"}

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            query = """
            MATCH (m:OM_Memory)
            WHERE (
              (m.accessEntity IS NOT NULL AND (
                m.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
              ))
              OR (m.accessEntity IS NULL AND m.userId = $userId)
            )
            OPTIONAL MATCH (m)-[r:OM_SIMILAR]->()
            WHERE (
              (r.accessEntity IS NOT NULL AND (
                r.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE r.accessEntity STARTS WITH prefix)
              ))
              OR (r.accessEntity IS NULL AND r.userId = $userId)
            )
            WITH m, count(r) AS clusterSize
            SET m.similarityClusterSize = clusterSize
            RETURN count(m) AS updated
            """
            access_entities = access_entities or [f"user:{user_id}"]
            access_entity_prefixes = access_entity_prefixes or []
            result = session.run(
                query,
                userId=user_id,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
            )
            record = result.single()
            return {
                "success": True,
                "memories_updated": record["updated"] if record else 0
            }
    except Exception as e:
        logger.error(f"Cluster size refresh failed: {e}")
        return {"success": False, "error": str(e)}


def refresh_entity_degrees(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Refresh degree property on all OM_Entity nodes.

    Counts OM_CO_MENTIONED edges for each entity.
    This property indicates how central an entity is in the
    user's memory network.

    Args:
        user_id: User ID to refresh entity degrees for (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Statistics about the refresh operation
    """
    from app.graph.neo4j_client import is_neo4j_configured

    if not is_neo4j_configured():
        return {"success": False, "error": "Neo4j not configured"}

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            query = """
            MATCH (e:OM_Entity)
            WHERE (
              (e.accessEntity IS NOT NULL AND (
                e.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
              ))
              OR (e.accessEntity IS NULL AND e.userId = $userId)
            )
            OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-()
            WHERE (
              (r.accessEntity IS NOT NULL AND (
                r.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE r.accessEntity STARTS WITH prefix)
              ))
              OR (r.accessEntity IS NULL AND r.userId = $userId)
            )
            WITH e, count(r) AS degree
            SET e.degree = degree
            RETURN count(e) AS updated
            """
            access_entities = access_entities or [f"user:{user_id}"]
            access_entity_prefixes = access_entity_prefixes or []
            result = session.run(
                query,
                userId=user_id,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
            )
            record = result.single()
            return {
                "success": True,
                "entities_updated": record["updated"] if record else 0
            }
    except Exception as e:
        logger.error(f"Entity degree refresh failed: {e}")
        return {"success": False, "error": str(e)}


def refresh_all_graph_properties(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Refresh all cached graph properties for a user.

    Convenience function that runs all refresh operations.
    Use after major data changes or as a scheduled maintenance task.

    Args:
        user_id: User ID to refresh properties for (legacy fallback)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Combined results from all refresh operations
    """
    results = {
        "pagerank": refresh_pagerank_cache(
            user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        ),
        "cluster_sizes": refresh_cluster_sizes(
            user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        ),
        "entity_degrees": refresh_entity_degrees(
            user_id,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        ),
    }

    all_success = all(r.get("success", False) for r in results.values())
    results["all_success"] = all_success

    return results
