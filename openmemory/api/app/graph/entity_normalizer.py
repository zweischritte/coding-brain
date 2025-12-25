"""
Entity Normalization Layer für OpenMemory.

Konsolidiert fragmentierte Entity-Varianten zu kanonischen Formen:
- "Matthias", "matthias", "MATTHIAS" → "matthias"
- "Matthias Coers", "matthias_coers" → "matthias_coers"
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

logger = logging.getLogger(__name__)


@dataclass
class EntityVariant:
    """Eine Entity-Variante mit Statistiken."""
    name: str
    memory_count: int
    co_mention_count: int = 0
    relation_count: int = 0


@dataclass
class CanonicalEntity:
    """Eine kanonische Entity mit ihren Varianten."""
    canonical: str
    variants: List[EntityVariant] = field(default_factory=list)

    @property
    def total_memories(self) -> int:
        return sum(v.memory_count for v in self.variants)


def normalize_entity_name(name: str) -> str:
    """
    Normalisiert einen Entity-Namen zu einer kanonischen Form.

    Regeln:
    1. Lowercase
    2. Spaces → Underscores
    3. Mehrfache Underscores → Einzeln
    4. Trim
    """
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')
    return normalized


def find_entity_variants(user_id: str) -> Dict[str, List[EntityVariant]]:
    """
    Findet alle Entity-Varianten für einen User gruppiert nach normalisiertem Namen.

    Returns:
        Dict[normalized_name, List[EntityVariant]]
    """
    if not is_neo4j_configured():
        return {}

    query = """
    MATCH (e:OM_Entity {userId: $userId})
    OPTIONAL MATCH (e)<-[:OM_ABOUT]-(m:OM_Memory)
    WITH e.name AS name, count(DISTINCT m) AS memoryCount
    OPTIONAL MATCH (e2:OM_Entity {userId: $userId, name: name})-[:OM_CO_MENTIONED]-()
    WITH name, memoryCount, count(*) AS coMentionCount
    OPTIONAL MATCH (e3:OM_Entity {userId: $userId, name: name})-[:OM_RELATION]-()
    RETURN name, memoryCount, coMentionCount, count(*) AS relationCount
    ORDER BY memoryCount DESC
    """

    variants_by_normalized: Dict[str, List[EntityVariant]] = defaultdict(list)

    try:
        with get_neo4j_session() as session:
            result = session.run(query, userId=user_id)
            for record in result:
                name = record["name"]
                if not name:
                    continue
                normalized = normalize_entity_name(name)
                variant = EntityVariant(
                    name=name,
                    memory_count=record["memoryCount"] or 0,
                    co_mention_count=record["coMentionCount"] or 0,
                    relation_count=record["relationCount"] or 0,
                )
                variants_by_normalized[normalized].append(variant)
    except Exception as e:
        logger.error(f"Error finding entity variants: {e}")

    return variants_by_normalized


def identify_duplicates(user_id: str, min_variants: int = 2) -> List[CanonicalEntity]:
    """
    Identifiziert Entity-Duplikate die zusammengeführt werden sollten.

    Args:
        user_id: User ID
        min_variants: Minimale Anzahl Varianten um als Duplikat zu gelten

    Returns:
        Liste von CanonicalEntity mit Merge-Kandidaten
    """
    variants_by_normalized = find_entity_variants(user_id)

    duplicates = []
    for normalized, variants in variants_by_normalized.items():
        if len(variants) >= min_variants:
            # Wähle Variante mit meisten Memories als kanonisch
            variants.sort(key=lambda v: v.memory_count, reverse=True)
            canonical = CanonicalEntity(
                canonical=variants[0].name,
                variants=variants,
            )
            duplicates.append(canonical)

    # Sortiere nach Gesamtanzahl Memories
    duplicates.sort(key=lambda c: c.total_memories, reverse=True)
    return duplicates


def merge_entity_variants(
    user_id: str,
    canonical_name: str,
    variant_names: List[str],
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Führt Entity-Varianten zu einer kanonischen Entity zusammen.

    Schritte:
    1. Alle OM_ABOUT Kanten von Varianten auf canonical umleiten
    2. Alle OM_CO_MENTIONED Kanten auf canonical umleiten (Counts aggregieren)
    3. Alle OM_RELATION Kanten auf canonical umleiten
    4. Varianten-Nodes löschen

    Args:
        user_id: User ID
        canonical_name: Ziel-Entity Name
        variant_names: Liste der zu mergenden Varianten-Namen
        dry_run: Wenn True, nur simulieren

    Returns:
        Statistiken über durchgeführte Änderungen
    """
    if not is_neo4j_configured():
        return {"error": "Neo4j not configured"}

    stats = {
        "canonical": canonical_name,
        "variants_merged": [],
        "about_edges_migrated": 0,
        "co_mention_edges_migrated": 0,
        "relation_edges_migrated": 0,
        "nodes_deleted": 0,
        "dry_run": dry_run,
    }

    # Filtere canonical aus variants
    variants_to_merge = [v for v in variant_names if v != canonical_name]

    if not variants_to_merge:
        return stats

    try:
        with get_neo4j_session() as session:
            for variant in variants_to_merge:
                # 1. Count OM_ABOUT edges to migrate
                count_about_query = """
                MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
                WHERE NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
                RETURN count(r) AS count
                """

                result = session.run(
                    count_about_query,
                    userId=user_id,
                    canonicalName=canonical_name,
                    variantName=variant,
                )
                record = result.single()
                about_count = record["count"] if record else 0
                stats["about_edges_migrated"] += about_count

                # 2. Count OM_CO_MENTIONED edges to migrate
                count_co_mention_query = """
                MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                MATCH (variant)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
                WHERE other <> canonical AND other <> variant
                RETURN count(r) AS count
                """

                result = session.run(
                    count_co_mention_query,
                    userId=user_id,
                    canonicalName=canonical_name,
                    variantName=variant,
                )
                record = result.single()
                co_mention_count = record["count"] if record else 0
                stats["co_mention_edges_migrated"] += co_mention_count

                # 3. Count OM_RELATION edges to migrate
                count_relation_query = """
                MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                MATCH (variant)-[r:OM_RELATION]-(other)
                WHERE other <> canonical AND other <> variant
                RETURN count(r) AS count
                """

                result = session.run(
                    count_relation_query,
                    userId=user_id,
                    canonicalName=canonical_name,
                    variantName=variant,
                )
                record = result.single()
                relation_count = record["count"] if record else 0
                stats["relation_edges_migrated"] += relation_count

                # If not dry_run, actually perform the migration
                if not dry_run:
                    # Migrate OM_ABOUT edges
                    migrate_about_query = """
                    MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                    MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                    MATCH (m:OM_Memory)-[r:OM_ABOUT]->(variant)
                    WHERE NOT EXISTS((m)-[:OM_ABOUT]->(canonical))
                    MERGE (m)-[:OM_ABOUT]->(canonical)
                    DELETE r
                    """
                    session.run(
                        migrate_about_query,
                        userId=user_id,
                        canonicalName=canonical_name,
                        variantName=variant,
                    )

                    # Migrate OM_CO_MENTIONED edges (aggregate counts)
                    migrate_co_mention_query = """
                    MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                    MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                    MATCH (variant)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
                    WHERE other <> canonical AND other <> variant
                    WITH canonical, other, r, coalesce(r.count, 1) AS oldCount
                    MERGE (canonical)-[newR:OM_CO_MENTIONED {userId: $userId}]-(other)
                    ON CREATE SET newR.count = oldCount, newR.createdAt = datetime()
                    ON MATCH SET newR.count = coalesce(newR.count, 0) + oldCount
                    DELETE r
                    """
                    session.run(
                        migrate_co_mention_query,
                        userId=user_id,
                        canonicalName=canonical_name,
                        variantName=variant,
                    )

                    # Migrate OM_RELATION edges
                    migrate_relation_query = """
                    MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                    MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                    MATCH (variant)-[r:OM_RELATION]->(other)
                    WHERE other <> canonical AND other <> variant
                    WITH canonical, other, r, r.type AS relType, r.memoryId AS memId
                    MERGE (canonical)-[newR:OM_RELATION {type: relType, userId: $userId}]->(other)
                    ON CREATE SET newR.memoryId = memId, newR.createdAt = datetime()
                    ON MATCH SET newR.count = coalesce(newR.count, 0) + 1
                    DELETE r
                    """
                    session.run(
                        migrate_relation_query,
                        userId=user_id,
                        canonicalName=canonical_name,
                        variantName=variant,
                    )

                    # Also handle incoming OM_RELATION edges
                    migrate_relation_incoming_query = """
                    MATCH (canonical:OM_Entity {userId: $userId, name: $canonicalName})
                    MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                    MATCH (other)-[r:OM_RELATION]->(variant)
                    WHERE other <> canonical AND other <> variant
                    WITH canonical, other, r, r.type AS relType, r.memoryId AS memId
                    MERGE (other)-[newR:OM_RELATION {type: relType, userId: $userId}]->(canonical)
                    ON CREATE SET newR.memoryId = memId, newR.createdAt = datetime()
                    ON MATCH SET newR.count = coalesce(newR.count, 0) + 1
                    DELETE r
                    """
                    session.run(
                        migrate_relation_incoming_query,
                        userId=user_id,
                        canonicalName=canonical_name,
                        variantName=variant,
                    )

                    # Delete variant node if it has no more edges
                    delete_query = """
                    MATCH (variant:OM_Entity {userId: $userId, name: $variantName})
                    WHERE NOT EXISTS((variant)<-[:OM_ABOUT]-())
                      AND NOT EXISTS((variant)-[:OM_CO_MENTIONED]-())
                      AND NOT EXISTS((variant)-[:OM_RELATION]-())
                      AND NOT EXISTS(()-[:OM_RELATION]->(variant))
                    DELETE variant
                    RETURN count(variant) AS count
                    """
                    result = session.run(
                        delete_query,
                        userId=user_id,
                        variantName=variant,
                    )
                    record = result.single()
                    if record:
                        stats["nodes_deleted"] += record["count"]

                stats["variants_merged"].append(variant)

        return stats

    except Exception as e:
        logger.exception(f"Error merging entity variants: {e}")
        return {"error": str(e), **stats}


def auto_normalize_entities(
    user_id: str,
    min_variants: int = 2,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Automatische Normalisierung aller Entity-Duplikate für einen User.

    Args:
        user_id: User ID
        min_variants: Minimale Varianten für Merge
        dry_run: Wenn True, nur Report generieren

    Returns:
        Gesamt-Statistiken
    """
    duplicates = identify_duplicates(user_id, min_variants)

    total_stats = {
        "user_id": user_id,
        "duplicate_groups": len(duplicates),
        "total_about_migrated": 0,
        "total_co_mention_migrated": 0,
        "total_relation_migrated": 0,
        "total_nodes_deleted": 0,
        "merges": [],
        "dry_run": dry_run,
    }

    for dup in duplicates:
        variant_names = [v.name for v in dup.variants]
        stats = merge_entity_variants(
            user_id=user_id,
            canonical_name=dup.canonical,
            variant_names=variant_names,
            dry_run=dry_run,
        )

        total_stats["merges"].append({
            "canonical": dup.canonical,
            "variants": variant_names,
            "stats": stats,
        })
        total_stats["total_about_migrated"] += stats.get("about_edges_migrated", 0)
        total_stats["total_co_mention_migrated"] += stats.get("co_mention_edges_migrated", 0)
        total_stats["total_relation_migrated"] += stats.get("relation_edges_migrated", 0)
        total_stats["total_nodes_deleted"] += stats.get("nodes_deleted", 0)

    return total_stats
