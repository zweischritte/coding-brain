"""
Entity Bridge: Connects Mem0's LLM-extracted entities to OpenMemory's OM_* graph.

This module solves the "single entity per memory" limitation by:
1. Extracting entities from memory content using Mem0's entity extraction
2. Creating multiple OM_ABOUT edges from OM_Memory to OM_Entity nodes
3. Enabling OM_CO_MENTIONED edges between entities that appear in the same memory

The bridge runs after memory creation and queries Mem0's LLM to extract entities,
then projects them into the OpenMemory graph structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from memory content."""
    name: str
    entity_type: str


@dataclass
class ExtractedRelation:
    """A relationship between two entities."""
    source: str
    relationship: str
    destination: str


def extract_entities_from_content(
    content: str,
    user_id: str,
) -> Tuple[List[ExtractedEntity], List[ExtractedRelation]]:
    """
    Extract entities and relationships from memory content using Mem0's LLM extraction.

    This leverages the same extraction logic that Mem0 Graph Memory uses,
    but returns the results without writing to Neo4j.

    Args:
        content: The memory text content
        user_id: User ID for entity resolution (self-references become user_id)

    Returns:
        Tuple of (entities, relations)
    """
    try:
        from app.utils.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client is None or not hasattr(memory_client, 'graph') or memory_client.graph is None:
            logger.debug("Mem0 Graph Memory not configured, skipping entity extraction")
            return [], []

        graph = memory_client.graph
        filters = {"user_id": user_id}

        # Phase 1: Extract entities from content
        entity_type_map = graph._retrieve_nodes_from_data(content, filters)

        if not entity_type_map:
            logger.debug(f"No entities extracted from content: {content[:100]}...")
            return [], []

        entities = [
            ExtractedEntity(name=name, entity_type=entity_type)
            for name, entity_type in entity_type_map.items()
        ]

        # Phase 2: Extract relationships between entities
        relations_raw = graph._establish_nodes_relations_from_data(content, filters, entity_type_map)

        relations = [
            ExtractedRelation(
                source=r["source"],
                relationship=r["relationship"],
                destination=r["destination"],
            )
            for r in relations_raw
        ]

        logger.info(
            f"Extracted {len(entities)} entities and {len(relations)} relations from content"
        )

        return entities, relations

    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return [], []


def bridge_entities_to_om_graph(
    memory_id: str,
    user_id: str,
    content: str,
    existing_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bridge extracted entities to the OpenMemory OM_* graph.

    This function:
    1. Extracts entities from memory content using Mem0's LLM
    2. Creates OM_Entity nodes for each extracted entity
    3. Creates OM_ABOUT edges from OM_Memory to each OM_Entity
    4. Creates OM_RELATION edges between related entities (with provenance)
    5. Triggers OM_CO_MENTIONED edge updates

    Args:
        memory_id: UUID of the memory
        user_id: String user ID
        content: The memory text content
        existing_entity: Optional entity from metadata.re (will be included)

    Returns:
        Dict with statistics: entities_created, relations_created, co_mentions_updated
    """
    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session
    from app.graph.metadata_projector import get_projector

    result = {
        "memory_id": memory_id,
        "entities_bridged": 0,
        "relations_created": 0,
        "co_mentions_updated": False,
        "entities": [],
    }

    if not is_neo4j_configured():
        logger.debug("Neo4j not configured, skipping entity bridge")
        return result

    from app.graph.entity_normalizer import normalize_entity_name

    # Extract entities and relations from content
    entities, relations = extract_entities_from_content(content, user_id)

    # Include existing entity from metadata if provided (keep raw casing for displayName)
    if existing_entity:
        existing_name = existing_entity.strip()
        existing_normalized = normalize_entity_name(existing_name) if existing_name else ""
        if existing_normalized:
            if not any(
                normalize_entity_name(e.name) == existing_normalized
                for e in entities
            ):
                entities.append(ExtractedEntity(name=existing_name, entity_type="metadata_entity"))

    if not entities:
        logger.debug(f"No entities to bridge for memory {memory_id}")
        return result

    normalized_entities: Dict[str, str] = {}
    for entity in entities:
        normalized = normalize_entity_name(entity.name)
        if not normalized:
            continue
        display_name = entity.name.strip() if entity.name else normalized
        if normalized not in normalized_entities:
            normalized_entities[normalized] = display_name

    result["entities"] = list(normalized_entities.keys())

    try:
        with get_neo4j_session() as session:
            # Step 1: Create OM_Entity nodes and OM_ABOUT edges for each entity
            for normalized_name, display_name in normalized_entities.items():
                entity_created = _create_entity_and_link(
                    session, memory_id, user_id, normalized_name, display_name
                )
                if entity_created:
                    result["entities_bridged"] += 1

            # Step 2: Create OM_RELATION edges between entities (with typed relationships)
            for relation in relations:
                source_normalized = normalize_entity_name(relation.source)
                destination_normalized = normalize_entity_name(relation.destination)
                if not source_normalized or not destination_normalized:
                    continue
                relation_created = _create_typed_relation(
                    session,
                    memory_id,
                    user_id,
                    source_normalized,
                    destination_normalized,
                    relation.relationship,
                )
                if relation_created:
                    result["relations_created"] += 1

        # Step 3: Update OM_CO_MENTIONED edges
        # Now that we have multiple entities per memory, the co-mention logic will work
        projector = get_projector()
        if projector is not None:
            try:
                projector.update_entity_edges_on_add(memory_id, user_id)
                result["co_mentions_updated"] = True
            except Exception as e:
                logger.warning(f"Failed to update co-mention edges: {e}")

        logger.info(
            f"Bridged {result['entities_bridged']} entities for memory {memory_id}, "
            f"created {result['relations_created']} typed relations"
        )

    except Exception as e:
        logger.exception(f"Entity bridge failed for memory {memory_id}: {e}")

    return result


def _create_entity_and_link(
    session,
    memory_id: str,
    user_id: str,
    entity_name: str,
    display_name: Optional[str] = None,
) -> bool:
    """
    Create an OM_Entity node (if not exists) and link it to the OM_Memory.

    Returns True if successful.
    """
    now = datetime.now(timezone.utc).isoformat()

    query = """
    MATCH (m:OM_Memory {id: $memoryId})
    WITH m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
    MERGE (e:OM_Entity {accessEntity: accessEntity, name: $entityName})
    ON CREATE SET e.createdAt = $now,
                  e.userId = $userId,
                  e.displayName = $displayName
    ON MATCH SET e.updatedAt = $now,
                 e.displayName = coalesce(e.displayName, $displayName)
    MERGE (m)-[r:OM_ABOUT]->(e)
    ON CREATE SET r.createdAt = $now
    ON MATCH SET r.updatedAt = $now
    RETURN e.name AS entity, type(r) AS rel
    """

    try:
        result = session.run(
            query,
            memoryId=memory_id,
            userId=user_id,
            entityName=entity_name,
            displayName=display_name,
            now=now,
            legacyAccessEntity=f"user:{user_id}",
        )
        record = result.single()
        return record is not None
    except Exception as e:
        logger.warning(f"Failed to create entity link {entity_name}: {e}")
        return False


def _create_typed_relation(
    session,
    memory_id: str,
    user_id: str,
    source_name: str,
    destination_name: str,
    relation_type: str,
) -> bool:
    """
    Create an OM_RELATION edge between two entities with the relationship type as property.

    This preserves semantic relationship types (e.g., "vater_von", "works_at")
    with provenance back to the source memory.

    Returns True if successful.
    """
    now = datetime.now(timezone.utc).isoformat()

    query = """
    MATCH (m:OM_Memory {id: $memoryId})
    WITH m.userId AS uid, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
    MATCH (e1:OM_Entity {accessEntity: accessEntity, name: $source})
    MATCH (e2:OM_Entity {accessEntity: accessEntity, name: $destination})
    MERGE (e1)-[r:OM_RELATION {type: $relationType, accessEntity: accessEntity}]->(e2)
    ON CREATE SET
        r.memoryId = $memoryId,
        r.createdAt = $now,
        r.count = 1,
        r.userId = uid
    ON MATCH SET
        r.updatedAt = $now,
        r.count = coalesce(r.count, 0) + 1
    RETURN e1.name AS source, r.type AS relationType, e2.name AS destination
    """

    try:
        result = session.run(
            query,
            userId=user_id,
            source=source_name,
            destination=destination_name,
            relationType=relation_type,
            memoryId=memory_id,
            now=now,
            legacyAccessEntity=f"user:{user_id}",
        )
        record = result.single()
        return record is not None
    except Exception as e:
        logger.warning(
            f"Failed to create relation {source_name}-[{relation_type}]->{destination_name}: {e}"
        )
        return False


def bridge_entities_for_existing_memory(
    memory_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Bridge entities for an existing memory by fetching its content from the database.

    Used for backfilling existing memories.

    Args:
        memory_id: UUID of the memory
        user_id: String user ID

    Returns:
        Bridge result dict
    """
    from app.database import SessionLocal
    from app.models import Memory
    import uuid

    try:
        db = SessionLocal()
        try:
            memory = db.query(Memory).filter(Memory.id == uuid.UUID(memory_id)).first()
            if not memory:
                logger.warning(f"Memory {memory_id} not found for bridging")
                return {"error": "Memory not found", "memory_id": memory_id}

            existing_entity = None
            if memory.metadata_:
                existing_entity = memory.metadata_.get("re") or memory.metadata_.get("entity")

            return bridge_entities_to_om_graph(
                memory_id=memory_id,
                user_id=user_id,
                content=memory.content,
                existing_entity=existing_entity,
            )
        finally:
            db.close()
    except Exception as e:
        logger.exception(f"Failed to bridge entities for memory {memory_id}: {e}")
        return {"error": str(e), "memory_id": memory_id}
