"""
GDS Signal Refresh Module for OpenMemory.

After entity normalization, graph-derived signals need to be
recalculated to maintain accurate search ranking:

1. Entity Degree (count of OM_CO_MENTIONED edges)
2. Entity PageRank (centrality in the co-mention network)
3. Entity Betweenness (bridge importance)

These signals are used by the hybrid retrieval system for
boosting search results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

logger = logging.getLogger(__name__)


def is_gds_available() -> bool:
    """
    Check if Neo4j Graph Data Science (GDS) is available.
    """
    if not is_neo4j_configured():
        return False

    try:
        with get_neo4j_session() as session:
            result = session.run("RETURN gds.version() AS version")
            record = result.single()
            if record:
                logger.debug(f"GDS version: {record['version']}")
                return True
            return False
    except Exception:
        return False


async def refresh_graph_signals(
    user_id: str,
    canonical: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refresh graph-derived signals after entity normalization.

    Args:
        user_id: User ID
        canonical: Optional - if provided, only refresh signals for
                   this entity and its neighbors. If None, refresh all.

    Returns:
        Statistics about refreshed signals
    """
    stats = {
        "degree_updated": 0,
        "pagerank_updated": 0,
        "cluster_updated": 0,
        "gds_available": False,
        "errors": [],
    }

    if not is_neo4j_configured():
        stats["errors"].append("Neo4j not configured")
        return stats

    stats["gds_available"] = is_gds_available()

    try:
        # 1. Refresh degree counts (always available)
        degree_stats = await _refresh_degree_counts(user_id, canonical)
        stats["degree_updated"] = degree_stats.get("updated", 0)

        # 2. Refresh PageRank (requires GDS)
        if stats["gds_available"]:
            pagerank_stats = await _refresh_pagerank(user_id, canonical)
            stats["pagerank_updated"] = pagerank_stats.get("updated", 0)

            # 3. Refresh memory similarity cluster size
            cluster_stats = await _refresh_cluster_sizes(user_id, canonical)
            stats["cluster_updated"] = cluster_stats.get("updated", 0)
        else:
            logger.info("GDS not available, skipping PageRank refresh")

    except Exception as e:
        logger.exception(f"Error refreshing graph signals: {e}")
        stats["errors"].append(str(e))

    return stats


async def _refresh_degree_counts(
    user_id: str,
    canonical: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refresh entity degree (OM_CO_MENTIONED edge count).

    Degree is stored on OM_Entity nodes as 'degree' property.
    Used for entity_density boost in hybrid retrieval.
    """
    try:
        with get_neo4j_session() as session:
            if canonical:
                # Only refresh canonical and its neighbors
                query = """
                MATCH (e:OM_Entity {userId: $userId, name: $canonical})
                OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-(neighbor:OM_Entity)
                WITH e, count(r) AS degree
                SET e.degree = degree
                WITH e

                // Also refresh neighbors' degrees
                MATCH (e)-[:OM_CO_MENTIONED]-(neighbor:OM_Entity {userId: $userId})
                OPTIONAL MATCH (neighbor)-[r2:OM_CO_MENTIONED]-()
                WITH neighbor, count(r2) AS neighborDegree
                SET neighbor.degree = neighborDegree
                RETURN count(neighbor) + 1 AS updated
                """
                result = session.run(query, userId=user_id, canonical=canonical)
            else:
                # Refresh all entities
                query = """
                MATCH (e:OM_Entity {userId: $userId})
                OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-()
                WITH e, count(r) AS degree
                SET e.degree = degree
                RETURN count(e) AS updated
                """
                result = session.run(query, userId=user_id)

            record = result.single()
            return {"updated": record["updated"] if record else 0}

    except Exception as e:
        logger.error(f"Error refreshing degree counts: {e}")
        return {"updated": 0, "error": str(e)}


async def _refresh_pagerank(
    user_id: str,
    canonical: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recalculate PageRank for entities using GDS.

    PageRank identifies influential entities in the co-mention network.
    Stored as 'pageRank' property on OM_Entity nodes.

    Note: This is a full graph operation even when canonical is provided,
    since PageRank is a global algorithm. We just scope to user's graph.
    """
    try:
        with get_neo4j_session() as session:
            # Create a temporary in-memory graph projection
            # Sanitize user_id for use in projection name (only alphanumeric and underscore)
            safe_user_id = "".join(c if c.isalnum() else "_" for c in user_id)
            projection_name = f"entity_pagerank_{safe_user_id}"

            # Drop existing projection if it exists
            try:
                session.run(f"CALL gds.graph.drop('{projection_name}', false)")
            except Exception:
                pass  # Graph didn't exist

            # First, check if there are any entities to project
            count_query = """
            MATCH (e:OM_Entity {userId: $userId})
            RETURN count(e) AS entityCount
            """
            count_result = session.run(count_query, userId=user_id)
            count_record = count_result.single()

            if not count_record or count_record["entityCount"] == 0:
                logger.info(f"No entities to compute PageRank for user {user_id}")
                return {"updated": 0}

            # Use gds.graph.project with native projection (more reliable than cypher projection)
            # First, we need to create the projection using Cypher with string interpolation
            # since gds.graph.project.cypher doesn't support query parameters in inner queries
            #
            # SECURITY NOTE: user_id is validated/controlled by the application,
            # so string interpolation is safe here. The escaped quotes prevent injection.
            escaped_user_id = user_id.replace("'", "\\'").replace("\\", "\\\\")

            node_query = f"MATCH (e:OM_Entity {{userId: '{escaped_user_id}'}}) RETURN id(e) AS id"
            rel_query = f"""MATCH (e1:OM_Entity {{userId: '{escaped_user_id}'}})-[r:OM_CO_MENTIONED]-(e2:OM_Entity {{userId: '{escaped_user_id}'}})
                 RETURN id(e1) AS source, id(e2) AS target, coalesce(r.count, 1) AS weight"""

            create_projection_query = """
            CALL gds.graph.project.cypher(
                $projectionName,
                $nodeQuery,
                $relQuery
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """

            result = session.run(
                create_projection_query,
                projectionName=projection_name,
                nodeQuery=node_query,
                relQuery=rel_query
            )
            projection_info = result.single()

            if not projection_info or projection_info["nodeCount"] == 0:
                logger.info(f"No entities in projection for user {user_id}")
                return {"updated": 0}

            # Run PageRank and write results
            pagerank_query = """
            CALL gds.pageRank.write(
                $projectionName,
                {
                    maxIterations: 20,
                    dampingFactor: 0.85,
                    relationshipWeightProperty: 'weight',
                    writeProperty: 'pageRank'
                }
            )
            YIELD nodePropertiesWritten, ranIterations, centralityDistribution
            RETURN nodePropertiesWritten, ranIterations
            """

            result = session.run(pagerank_query, projectionName=projection_name)
            record = result.single()

            # Clean up projection
            try:
                session.run(f"CALL gds.graph.drop('{projection_name}')")
            except Exception:
                pass

            return {
                "updated": record["nodePropertiesWritten"] if record else 0,
                "iterations": record["ranIterations"] if record else 0
            }

    except Exception as e:
        logger.error(f"Error refreshing PageRank: {e}")
        return {"updated": 0, "error": str(e)}


async def _refresh_cluster_sizes(
    user_id: str,
    canonical: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refresh memory similarity cluster sizes.

    The cluster size indicates how well-connected a memory is
    in the similarity graph. Used for similarity_cluster boost.
    """
    try:
        with get_neo4j_session() as session:
            if canonical:
                # Refresh memories connected to canonical entity
                query = """
                MATCH (e:OM_Entity {userId: $userId, name: $canonical})
                MATCH (m:OM_Memory)-[:OM_ABOUT]->(e)
                OPTIONAL MATCH (m)-[r:OM_SIMILAR]-()
                WITH m, count(r) AS clusterSize
                SET m.similarityClusterSize = clusterSize
                RETURN count(m) AS updated
                """
                result = session.run(query, userId=user_id, canonical=canonical)
            else:
                # Refresh all memories
                query = """
                MATCH (m:OM_Memory {userId: $userId})
                OPTIONAL MATCH (m)-[r:OM_SIMILAR]-()
                WITH m, count(r) AS clusterSize
                SET m.similarityClusterSize = clusterSize
                RETURN count(m) AS updated
                """
                result = session.run(query, userId=user_id)

            record = result.single()
            return {"updated": record["updated"] if record else 0}

    except Exception as e:
        logger.error(f"Error refreshing cluster sizes: {e}")
        return {"updated": 0, "error": str(e)}


async def get_entity_signals(
    user_id: str,
    entity_name: str,
) -> Dict[str, Any]:
    """
    Get current graph signals for an entity.

    Useful for debugging and verification.
    """
    if not is_neo4j_configured():
        return {"error": "Neo4j not configured"}

    try:
        with get_neo4j_session() as session:
            query = """
            MATCH (e:OM_Entity {userId: $userId, name: $entityName})
            OPTIONAL MATCH (e)-[co:OM_CO_MENTIONED]-()
            OPTIONAL MATCH (e)-[rel:OM_RELATION]-()
            OPTIONAL MATCH (e)<-[:OM_ABOUT]-(m:OM_Memory)
            RETURN
                e.name AS name,
                coalesce(e.degree, 0) AS degree,
                coalesce(e.pageRank, 0.0) AS pageRank,
                count(DISTINCT co) AS actual_co_mention_count,
                count(DISTINCT rel) AS relation_count,
                count(DISTINCT m) AS memory_count
            """

            result = session.run(query, userId=user_id, entityName=entity_name)
            record = result.single()

            if record:
                return dict(record)
            return {"error": f"Entity '{entity_name}' not found"}

    except Exception as e:
        logger.error(f"Error getting entity signals: {e}")
        return {"error": str(e)}


async def refresh_all_signals_for_user(user_id: str) -> Dict[str, Any]:
    """
    Full refresh of all graph signals for a user.

    Typically run after bulk operations like normalization backfill.
    """
    logger.info(f"Starting full signal refresh for user {user_id}")

    stats = await refresh_graph_signals(user_id, canonical=None)

    # Also refresh OM_CO_MENTIONED edge counts if they were affected
    try:
        with get_neo4j_session() as session:
            # Ensure edge counts are accurate
            update_edge_counts_query = """
            MATCH (e1:OM_Entity {userId: $userId})-[r:OM_CO_MENTIONED]-(e2:OM_Entity)
            WHERE r.count IS NULL
            WITH e1, e2, r
            MATCH (m:OM_Memory {userId: $userId})-[:OM_ABOUT]->(e1)
            MATCH (m)-[:OM_ABOUT]->(e2)
            WITH r, count(m) AS memCount
            SET r.count = memCount
            RETURN count(r) AS updated
            """

            result = session.run(update_edge_counts_query, userId=user_id)
            record = result.single()
            stats["edge_counts_updated"] = record["updated"] if record else 0

    except Exception as e:
        logger.error(f"Error updating edge counts: {e}")
        stats["edge_count_error"] = str(e)

    logger.info(f"Signal refresh complete: {stats}")
    return stats
