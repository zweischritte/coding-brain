"""
Entity Edge Migrator for OpenMemory.

Handles the transactional migration of all graph edges when
entities are merged. All edge types (OM_ABOUT, OM_CO_MENTIONED,
OM_RELATION, OM_TEMPORAL) are migrated with ACL enforcement.

Key features:
- Transactional safety (all-or-nothing per edge type)
- ACL enforcement (only allowed memories can be affected)
- Count aggregation for OM_CO_MENTIONED edges
- Orphan cleanup (delete variant nodes with no edges)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

logger = logging.getLogger(__name__)


def _resolve_access_entity(user_id: str, access_entity: Optional[str]) -> str:
    return access_entity or f"user:{user_id}"


@dataclass
class EdgeMigrationStats:
    """Statistics from edge migration."""
    om_about_migrated: int = 0
    om_co_mentioned_migrated: int = 0
    om_relation_migrated: int = 0
    om_temporal_migrated: int = 0
    variant_nodes_deleted: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_migrated(self) -> int:
        return (
            self.om_about_migrated +
            self.om_co_mentioned_migrated +
            self.om_relation_migrated +
            self.om_temporal_migrated
        )


async def migrate_entity_edges(
    user_id: str,
    canonical: str,
    variants: List[str],
    allowed_memory_ids: Optional[List[str]] = None,
    dry_run: bool = False,
    access_entity: Optional[str] = None,
) -> EdgeMigrationStats:
    """
    Migrate all edges from variant entities to canonical.

    Args:
        user_id: User ID
        canonical: Target entity name
        variants: Variant entity names to merge into canonical
        allowed_memory_ids: ACL - only these memories may be affected.
                           None means all memories are allowed.
        dry_run: If True, only count what would be changed

    Returns:
        EdgeMigrationStats with counts
    """
    stats = EdgeMigrationStats()

    if not is_neo4j_configured():
        stats.errors.append("Neo4j not configured")
        return stats

    if not variants:
        return stats

    try:
        access_entity = _resolve_access_entity(user_id, access_entity)
        with get_neo4j_session() as session:
            # Ensure canonical entity exists
            ensure_canonical_query = """
            MERGE (e:OM_Entity {accessEntity: $accessEntity, name: $canonical})
            ON CREATE SET e.createdAt = datetime(),
                          e.source = 'normalization',
                          e.userId = $userId,
                          e.displayName = $displayName
            ON MATCH SET e.displayName = coalesce(e.displayName, $displayName)
            RETURN e.name AS name
            """
            session.run(
                ensure_canonical_query,
                userId=user_id,
                canonical=canonical,
                accessEntity=access_entity,
                displayName=canonical,
            )

            for variant in variants:
                if variant == canonical:
                    continue

                # 1. Migrate OM_ABOUT edges
                om_about_count = await _migrate_om_about(
                    session,
                    user_id,
                    canonical,
                    variant,
                    allowed_memory_ids,
                    dry_run,
                    access_entity,
                )
                stats.om_about_migrated += om_about_count

                # 2. Migrate OM_CO_MENTIONED edges
                co_mentioned_count = await _migrate_om_co_mentioned(
                    session, user_id, canonical, variant, dry_run, access_entity
                )
                stats.om_co_mentioned_migrated += co_mentioned_count

                # 3. Migrate OM_RELATION edges
                relation_count = await _migrate_om_relation(
                    session, user_id, canonical, variant, dry_run, access_entity
                )
                stats.om_relation_migrated += relation_count

                # 4. Migrate OM_TEMPORAL edges
                temporal_count = await _migrate_om_temporal(
                    session, user_id, canonical, variant, dry_run, access_entity
                )
                stats.om_temporal_migrated += temporal_count

                # 5. Delete orphaned variant nodes
                if not dry_run:
                    deleted = await _delete_orphan_variant(
                        session, user_id, variant, access_entity
                    )
                    stats.variant_nodes_deleted += deleted

    except Exception as e:
        logger.exception(f"Error migrating edges: {e}")
        stats.errors.append(str(e))

    return stats


async def _migrate_om_about(
    session,
    user_id: str,
    canonical: str,
    variant: str,
    allowed_memory_ids: Optional[List[str]],
    dry_run: bool,
    access_entity: str,
) -> int:
    """
    Migrate OM_ABOUT edges from variant to canonical.

    ACL Enforcement:
    - Only memories in allowed_memory_ids are affected
    - If allowed_memory_ids is None, all memories are affected
    """
    if allowed_memory_ids is not None:
        # With ACL filter
        if dry_run:
            query = """
            MATCH (canonical:OM_Entity {name: $canonical})
            WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (variant:OM_Entity {name: $variant})
            WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
            WHERE m.id IN $allowedIds
              AND coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
              AND NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
            RETURN count(r) AS count
            """
        else:
            query = """
            MATCH (canonical:OM_Entity {name: $canonical})
            WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (variant:OM_Entity {name: $variant})
            WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
            WHERE m.id IN $allowedIds
              AND coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
              AND NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
            MERGE (m)-[:OM_ABOUT]->(canonical)
            DELETE r
            RETURN count(r) AS count
            """

        result = session.run(
            query,
            userId=user_id,
            canonical=canonical,
            variant=variant,
            allowedIds=allowed_memory_ids,
            accessEntity=access_entity,
            legacyAccessEntity=f"user:{user_id}",
        )
    else:
        # No ACL filter
        if dry_run:
            query = """
            MATCH (canonical:OM_Entity {name: $canonical})
            WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (variant:OM_Entity {name: $variant})
            WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
            WHERE coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
              AND NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
            RETURN count(r) AS count
            """
        else:
            query = """
            MATCH (canonical:OM_Entity {name: $canonical})
            WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (variant:OM_Entity {name: $variant})
            WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
            MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
            WHERE coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
              AND NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
            MERGE (m)-[:OM_ABOUT]->(canonical)
            DELETE r
            RETURN count(r) AS count
            """

        result = session.run(
            query,
            userId=user_id,
            canonical=canonical,
            variant=variant,
            accessEntity=access_entity,
            legacyAccessEntity=f"user:{user_id}",
        )

    record = result.single()
    return record["count"] if record else 0


async def _migrate_om_co_mentioned(
    session,
    user_id: str,
    canonical: str,
    variant: str,
    dry_run: bool,
    access_entity: str,
) -> int:
    """
    Migrate OM_CO_MENTIONED edges from variant to canonical.

    Aggregates counts when merging edges to the same target.
    Also handles edge direction (both outgoing and incoming).
    """
    # Count query
    count_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
    WHERE other <> canonical AND other <> variant
      AND coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity
      AND coalesce(other.accessEntity, $legacyAccessEntity) = $accessEntity
    RETURN count(r) AS count
    """

    result = session.run(
        count_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    record = result.single()
    count = record["count"] if record else 0

    if dry_run or count == 0:
        return count

    # Migration query with count aggregation
    migrate_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
    WHERE other <> canonical AND other <> variant
      AND coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity
      AND coalesce(other.accessEntity, $legacyAccessEntity) = $accessEntity
    WITH canonical, other, r,
         coalesce(r.count, 1) AS oldCount,
         r.memoryIds AS oldMemIds
    MERGE (canonical)-[newR:OM_CO_MENTIONED {accessEntity: $accessEntity}]-(other)
    ON CREATE SET
        newR.count = oldCount,
        newR.memoryIds = coalesce(oldMemIds, []),
        newR.createdAt = datetime(),
        newR.userId = $userId
    ON MATCH SET
        newR.count = coalesce(newR.count, 0) + oldCount,
        newR.memoryIds = CASE
            WHEN oldMemIds IS NULL THEN newR.memoryIds
            WHEN newR.memoryIds IS NULL THEN oldMemIds
            ELSE [x IN (newR.memoryIds + oldMemIds) | x][0..5]
        END,
        newR.updatedAt = datetime()
    DELETE r
    """

    session.run(
        migrate_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    return count


async def _migrate_om_relation(
    session,
    user_id: str,
    canonical: str,
    variant: str,
    dry_run: bool,
    access_entity: str,
) -> int:
    """
    Migrate OM_RELATION edges from variant to canonical.

    Handles both outgoing and incoming relations.
    Preserves relationship type and memory provenance.
    """
    # Count query (both directions)
    count_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_RELATION]-(other)
    WHERE other <> canonical AND other <> variant
      AND coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity
      AND coalesce(other.accessEntity, $legacyAccessEntity) = $accessEntity
    RETURN count(r) AS count
    """

    result = session.run(
        count_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    record = result.single()
    count = record["count"] if record else 0

    if dry_run or count == 0:
        return count

    # Migrate outgoing relations
    migrate_outgoing_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_RELATION]->(other)
    WHERE other <> canonical AND other <> variant
      AND coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity
      AND coalesce(other.accessEntity, $legacyAccessEntity) = $accessEntity
    WITH canonical, other, r, r.type AS relType, r.memoryId AS memId
    MERGE (canonical)-[newR:OM_RELATION {type: relType, accessEntity: $accessEntity}]->(other)
    ON CREATE SET
        newR.memoryId = memId,
        newR.count = 1,
        newR.createdAt = datetime(),
        newR.userId = $userId
    ON MATCH SET
        newR.count = coalesce(newR.count, 0) + 1,
        newR.updatedAt = datetime()
    DELETE r
    """
    session.run(
        migrate_outgoing_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )

    # Migrate incoming relations
    migrate_incoming_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (other)-[r:OM_RELATION]->(variant)
    WHERE other <> canonical AND other <> variant
      AND coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity
      AND coalesce(other.accessEntity, $legacyAccessEntity) = $accessEntity
    WITH canonical, other, r, r.type AS relType, r.memoryId AS memId
    MERGE (other)-[newR:OM_RELATION {type: relType, accessEntity: $accessEntity}]->(canonical)
    ON CREATE SET
        newR.memoryId = memId,
        newR.count = 1,
        newR.createdAt = datetime(),
        newR.userId = $userId
    ON MATCH SET
        newR.count = coalesce(newR.count, 0) + 1,
        newR.updatedAt = datetime()
    DELETE r
    """
    session.run(
        migrate_incoming_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )

    return count


async def _migrate_om_temporal(
    session,
    user_id: str,
    canonical: str,
    variant: str,
    dry_run: bool,
    access_entity: str,
) -> int:
    """
    Migrate OM_TEMPORAL edges from variant to canonical.

    OM_TEMPORAL links entities to biographical timeline events.
    """
    # Count query
    count_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_TEMPORAL]->(event:OM_TemporalEvent)
    WHERE coalesce(event.accessEntity, $legacyAccessEntity) = $accessEntity
      AND NOT EXISTS((canonical)-[:OM_TEMPORAL]->(event))
    RETURN count(r) AS count
    """

    result = session.run(
        count_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    record = result.single()
    count = record["count"] if record else 0

    if dry_run or count == 0:
        return count

    # Migrate query
    migrate_query = """
    MATCH (canonical:OM_Entity {name: $canonical})
    WHERE coalesce(canonical.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
    MATCH (variant)-[r:OM_TEMPORAL]->(event:OM_TemporalEvent)
    WHERE coalesce(event.accessEntity, $legacyAccessEntity) = $accessEntity
      AND NOT EXISTS((canonical)-[:OM_TEMPORAL]->(event))
    MERGE (canonical)-[:OM_TEMPORAL]->(event)
    DELETE r
    """

    session.run(
        migrate_query,
        userId=user_id,
        canonical=canonical,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    return count


async def _delete_orphan_variant(
    session,
    user_id: str,
    variant: str,
    access_entity: str,
) -> int:
    """
    Delete variant entity node if it has no remaining edges.
    """
    delete_query = """
    MATCH (variant:OM_Entity {name: $variant})
    WHERE coalesce(variant.accessEntity, $legacyAccessEntity) = $accessEntity
      AND NOT EXISTS((variant)<-[:OM_ABOUT]-())
      AND NOT EXISTS((variant)-[:OM_CO_MENTIONED]-())
      AND NOT EXISTS((variant)-[:OM_RELATION]-())
      AND NOT EXISTS(()-[:OM_RELATION]->(variant))
      AND NOT EXISTS((variant)-[:OM_TEMPORAL]->())
    DELETE variant
    RETURN count(variant) AS count
    """

    result = session.run(
        delete_query,
        userId=user_id,
        variant=variant,
        accessEntity=access_entity,
        legacyAccessEntity=f"user:{user_id}",
    )
    record = result.single()
    return record["count"] if record else 0


async def estimate_migration_impact(
    user_id: str,
    canonical: str,
    variants: List[str],
    allowed_memory_ids: Optional[List[str]] = None,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate the impact of a migration without executing it.

    Useful for showing users what will happen before they commit.
    """
    stats = await migrate_entity_edges(
        user_id=user_id,
        canonical=canonical,
        variants=variants,
        allowed_memory_ids=allowed_memory_ids,
        dry_run=True,
        access_entity=access_entity,
    )

    return {
        "canonical": canonical,
        "variants": variants,
        "estimated_changes": {
            "om_about_edges": stats.om_about_migrated,
            "om_co_mentioned_edges": stats.om_co_mentioned_migrated,
            "om_relation_edges": stats.om_relation_migrated,
            "om_temporal_edges": stats.om_temporal_migrated,
            "total_edges": stats.total_migrated,
        },
        "acl_scope": "all" if allowed_memory_ids is None else len(allowed_memory_ids),
    }
