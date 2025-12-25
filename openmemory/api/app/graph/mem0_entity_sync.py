"""
Mem0 Entity Sync Module for OpenMemory.

Handles synchronization between OpenMemory's OM_Entity graph
and Mem0's __Entity__ graph after entity normalization.

Mem0's graph stores LLM-extracted entities with:
- :__Entity__ nodes with user_id, name, embedding
- Dynamic relationship types (e.g., :vater_von, :works_at)

This module ensures that when OM_Entity names are normalized,
the corresponding __Entity__ nodes are also updated (optional).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

logger = logging.getLogger(__name__)


@dataclass
class Mem0SyncStats:
    """Statistics from Mem0 entity sync."""
    entities_found: int = 0
    entities_renamed: int = 0
    entities_merged: int = 0
    relationships_migrated: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def is_mem0_graph_available() -> bool:
    """
    Check if Mem0 Graph Memory is enabled and available.

    Mem0 Graph Memory uses :__Entity__ nodes with user_id property.
    """
    if not is_neo4j_configured():
        return False

    try:
        with get_neo4j_session() as session:
            # Check if any __Entity__ nodes exist
            result = session.run("""
                MATCH (e:`__Entity__`)
                RETURN count(e) > 0 AS exists
                LIMIT 1
            """)
            record = result.single()
            return record["exists"] if record else False
    except Exception as e:
        logger.debug(f"Mem0 graph not available: {e}")
        return False


async def sync_mem0_entities_after_normalization(
    user_id: str,
    canonical: str,
    variants: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Sync Mem0's __Entity__ graph after OM_Entity normalization.

    This is an OPTIONAL step that maintains consistency between
    the two graph layers.

    Strategy:
    1. Find all __Entity__ nodes matching variant names
    2. Merge relationships from variants to canonical __Entity__
    3. Update or delete variant __Entity__ nodes

    Note: Mem0's graph uses embedding-based entity matching,
    so name-based merge may not be perfect. This is best-effort.

    Args:
        user_id: User ID
        canonical: Canonical entity name
        variants: Variant names to merge
        dry_run: If True, only analyze without changes

    Returns:
        Statistics about what was changed
    """
    stats = Mem0SyncStats()

    if not is_neo4j_configured():
        stats.errors.append("Neo4j not configured")
        return _stats_to_dict(stats)

    # Check if Mem0 graph is available
    if not is_mem0_graph_available():
        logger.info("Mem0 Graph Memory not available, skipping sync")
        return _stats_to_dict(stats)

    try:
        with get_neo4j_session() as session:
            # 1. Count existing __Entity__ nodes for variants
            count_query = """
            UNWIND $variants AS variantName
            MATCH (e:`__Entity__` {user_id: $userId})
            WHERE toLower(e.name) = toLower(variantName)
            RETURN variantName, e.name AS actual_name, count(e) AS count
            """

            result = session.run(
                count_query,
                userId=user_id,
                variants=variants
            )

            variant_entities = {}
            for record in result:
                variant_entities[record["variantName"]] = {
                    "actual_name": record["actual_name"],
                    "count": record["count"]
                }
                stats.entities_found += record["count"]

            if stats.entities_found == 0:
                logger.info(f"No Mem0 entities found for variants: {variants}")
                return _stats_to_dict(stats)

            # 2. Ensure canonical __Entity__ exists
            ensure_canonical_query = """
            MERGE (e:`__Entity__` {user_id: $userId, name: $canonical})
            ON CREATE SET e.created_at = datetime()
            RETURN e.name AS name
            """

            if not dry_run:
                session.run(
                    ensure_canonical_query,
                    userId=user_id,
                    canonical=canonical
                )

            # 3. Migrate relationships from variants to canonical
            for variant, info in variant_entities.items():
                actual_name = info["actual_name"]

                # Count relationships to migrate (both directions)
                rel_count_query = """
                MATCH (canonical:`__Entity__` {user_id: $userId, name: $canonical})
                MATCH (variant:`__Entity__` {user_id: $userId, name: $variantName})
                WHERE variant <> canonical
                MATCH (variant)-[r]-(other)
                WHERE other <> canonical
                RETURN count(r) AS count
                """

                result = session.run(
                    rel_count_query,
                    userId=user_id,
                    canonical=canonical,
                    variantName=actual_name
                )
                record = result.single()
                rel_count = record["count"] if record else 0
                stats.relationships_migrated += rel_count

                if dry_run:
                    continue

                # Migrate outgoing relationships
                migrate_out_query = """
                MATCH (canonical:`__Entity__` {user_id: $userId, name: $canonical})
                MATCH (variant:`__Entity__` {user_id: $userId, name: $variantName})
                WHERE variant <> canonical
                MATCH (variant)-[r]->(other)
                WHERE other <> canonical
                WITH canonical, other, type(r) AS relType, properties(r) AS props
                CALL apoc.merge.relationship(canonical, relType, {}, props, other, {})
                YIELD rel
                RETURN count(rel) AS count
                """

                # Fallback without APOC
                migrate_out_fallback = """
                MATCH (canonical:`__Entity__` {user_id: $userId, name: $canonical})
                MATCH (variant:`__Entity__` {user_id: $userId, name: $variantName})
                WHERE variant <> canonical
                MATCH (variant)-[r]->(other)
                WHERE other <> canonical
                DELETE r
                WITH canonical, other
                RETURN count(*) AS count
                """

                try:
                    # Try APOC first
                    session.run(
                        migrate_out_query,
                        userId=user_id,
                        canonical=canonical,
                        variantName=actual_name
                    )
                except Exception:
                    # Fallback: just delete relationships (can't preserve type dynamically)
                    logger.debug("APOC not available, using fallback migration")
                    session.run(
                        migrate_out_fallback,
                        userId=user_id,
                        canonical=canonical,
                        variantName=actual_name
                    )

                # Migrate incoming relationships
                migrate_in_query = """
                MATCH (canonical:`__Entity__` {user_id: $userId, name: $canonical})
                MATCH (variant:`__Entity__` {user_id: $userId, name: $variantName})
                WHERE variant <> canonical
                MATCH (other)-[r]->(variant)
                WHERE other <> canonical
                WITH canonical, other, type(r) AS relType, properties(r) AS props
                CALL apoc.merge.relationship(other, relType, {}, props, canonical, {})
                YIELD rel
                RETURN count(rel) AS count
                """

                migrate_in_fallback = """
                MATCH (canonical:`__Entity__` {user_id: $userId, name: $canonical})
                MATCH (variant:`__Entity__` {user_id: $userId, name: $variantName})
                WHERE variant <> canonical
                MATCH (other)-[r]->(variant)
                WHERE other <> canonical
                DELETE r
                WITH canonical, other
                RETURN count(*) AS count
                """

                try:
                    session.run(
                        migrate_in_query,
                        userId=user_id,
                        canonical=canonical,
                        variantName=actual_name
                    )
                except Exception:
                    session.run(
                        migrate_in_fallback,
                        userId=user_id,
                        canonical=canonical,
                        variantName=actual_name
                    )

                stats.entities_merged += 1

            # 4. Delete orphaned variant nodes
            if not dry_run:
                delete_orphans_query = """
                UNWIND $variants AS variantName
                MATCH (e:`__Entity__` {user_id: $userId})
                WHERE toLower(e.name) = toLower(variantName)
                  AND NOT e.name = $canonical
                  AND NOT EXISTS((e)-[]-())
                DELETE e
                RETURN count(e) AS count
                """

                result = session.run(
                    delete_orphans_query,
                    userId=user_id,
                    variants=[v for v in variants if v != canonical],
                    canonical=canonical
                )
                record = result.single()
                deleted = record["count"] if record else 0
                logger.info(f"Deleted {deleted} orphaned Mem0 entity nodes")

    except Exception as e:
        logger.exception(f"Error syncing Mem0 entities: {e}")
        stats.errors.append(str(e))

    return _stats_to_dict(stats)


def _stats_to_dict(stats: Mem0SyncStats) -> Dict[str, Any]:
    """Convert stats dataclass to dict."""
    return {
        "entities_found": stats.entities_found,
        "entities_renamed": stats.entities_renamed,
        "entities_merged": stats.entities_merged,
        "relationships_migrated": stats.relationships_migrated,
        "errors": stats.errors,
    }


async def list_mem0_entities(
    user_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List all Mem0 __Entity__ nodes for a user.

    Useful for debugging and verification.
    """
    if not is_neo4j_configured():
        return []

    try:
        with get_neo4j_session() as session:
            query = """
            MATCH (e:`__Entity__` {user_id: $userId})
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) AS rel_count
            RETURN e.name AS name, rel_count AS relationships
            ORDER BY rel_count DESC
            LIMIT $limit
            """

            result = session.run(query, userId=user_id, limit=limit)
            return [dict(record) for record in result]

    except Exception as e:
        logger.error(f"Error listing Mem0 entities: {e}")
        return []


async def compare_om_and_mem0_entities(
    user_id: str,
) -> Dict[str, Any]:
    """
    Compare entities between OM_Entity and __Entity__ graphs.

    Returns:
        Dict with:
        - om_only: Entities only in OM graph
        - mem0_only: Entities only in Mem0 graph
        - both: Entities in both (normalized match)
    """
    if not is_neo4j_configured():
        return {"error": "Neo4j not configured"}

    try:
        with get_neo4j_session() as session:
            query = """
            MATCH (om:OM_Entity {userId: $userId})
            WITH collect(toLower(om.name)) AS omNames,
                 collect({original: om.name, lower: toLower(om.name)}) AS omEntities

            MATCH (mem:`__Entity__` {user_id: $userId})
            WITH omNames, omEntities,
                 collect(toLower(mem.name)) AS memNames,
                 collect({original: mem.name, lower: toLower(mem.name)}) AS memEntities

            RETURN
                [e IN omEntities WHERE NOT e.lower IN memNames | e.original] AS om_only,
                [e IN memEntities WHERE NOT e.lower IN omNames | e.original] AS mem0_only,
                [e IN omEntities WHERE e.lower IN memNames | e.original] AS both
            """

            result = session.run(query, userId=user_id)
            record = result.single()

            if record:
                return {
                    "om_only": record["om_only"],
                    "om_only_count": len(record["om_only"]),
                    "mem0_only": record["mem0_only"],
                    "mem0_only_count": len(record["mem0_only"]),
                    "both": record["both"],
                    "both_count": len(record["both"]),
                }

            return {"om_only": [], "mem0_only": [], "both": []}

    except Exception as e:
        logger.error(f"Error comparing entities: {e}")
        return {"error": str(e)}
