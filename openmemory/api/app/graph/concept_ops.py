"""
Business Concept Operations for OpenMemory.

High-level operations for the business concepts system, integrating
concept extraction, graph projection, and convergence detection.

Usage:
    from app.graph.concept_ops import (
        extract_and_store_concepts,
        get_concept,
        list_concepts,
        detect_contradictions,
        get_concept_network,
    )

    # Extract concepts from a memory
    result = extract_and_store_concepts(
        memory_id="...",
        user_id="...",
        content="...",
    )

    # Get a specific concept
    concept = get_concept(user_id="...", name="...")

    # List all concepts
    concepts = list_concepts(user_id="...")

Based on implementation plan:
- /docs/implementation/business-concept-implementation-plan.md
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import BusinessConceptsConfig
from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured, is_neo4j_healthy

logger = logging.getLogger(__name__)


def _normalize_access_filters(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    return (
        access_entities or [f"user:{user_id}"],
        access_entity_prefixes or [],
    )


# =============================================================================
# Feature Flag Checks
# =============================================================================

def is_concepts_enabled() -> bool:
    """Check if business concepts system is enabled and available."""
    return (
        BusinessConceptsConfig.is_enabled()
        and is_neo4j_configured()
        and is_neo4j_healthy()
    )


def is_auto_extract_enabled() -> bool:
    """Check if auto-extraction on memory add is enabled."""
    return (
        is_concepts_enabled()
        and BusinessConceptsConfig.is_auto_extract_enabled()
    )


def is_contradiction_detection_enabled() -> bool:
    """Check if contradiction detection is enabled."""
    return (
        is_concepts_enabled()
        and BusinessConceptsConfig.is_contradiction_detection_enabled()
    )


# =============================================================================
# Concept Extraction and Storage
# =============================================================================

def extract_and_store_concepts(
    memory_id: str,
    user_id: str,
    content: str,
    category: Optional[str] = None,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract business concepts from memory content and store in graph.

    This is the main entry point for concept extraction. It:
    1. Uses the ConceptExtractor to extract entities and concepts
    2. Stores them in Neo4j via ConceptProjector
    3. Links them to the source memory
    4. Optionally runs contradiction detection

    Args:
        memory_id: UUID of the memory
        user_id: User ID for legacy fallback
        content: Memory content to extract from
        category: Optional category for concept scoping
        access_entity: Access entity scope (defaults to user)

    Returns:
        Dict with extraction results:
        - entities_extracted: Number of entities extracted
        - concepts_extracted: Number of concepts extracted
        - contradictions_found: Number of contradictions detected (if enabled)
        - error: Error message if extraction failed
    """
    if not is_concepts_enabled():
        return {"error": "Business concepts system is not enabled"}

    try:
        # Get extractors and projectors
        from app.utils.concept_extractor import ConceptExtractor
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return {"error": "Concept projector not available"}

        extractor = ConceptExtractor(
            model=BusinessConceptsConfig.get_extraction_model(),
            ollama_base_url=BusinessConceptsConfig.get_ollama_base_url() or None,
            max_tokens_per_chunk=BusinessConceptsConfig.get_max_tokens_per_chunk(),
        )

        # Extract concepts and entities
        extraction = extractor.extract_full(content)

        min_confidence = BusinessConceptsConfig.get_min_confidence()
        entities_stored = 0
        concepts_stored = 0

        access_entity = access_entity or f"user:{user_id}"
        # Store business entities
        for entity in extraction.entities:
            if entity.importance >= min_confidence:
                result = projector.upsert_bizentity(
                    user_id=user_id,
                    name=entity.entity,
                    entity_type=entity.type,
                    importance=entity.importance,
                    context=entity.context,
                    mention_count=entity.mention_count,
                    access_entity=access_entity,
                )
                if result:
                    # Link memory to entity
                    projector.link_memory_to_bizentity(
                        memory_id=memory_id,
                        user_id=user_id,
                        entity_name=entity.entity,
                        importance=entity.importance,
                        access_entity=access_entity,
                    )
                    entities_stored += 1

        # Store concepts
        for concept in extraction.concepts:
            if concept.confidence >= min_confidence:
                result = projector.upsert_concept(
                    user_id=user_id,
                    name=concept.concept,
                    concept_type=concept.type,
                    confidence=concept.confidence,
                    category=category,
                    summary=None,
                    source_type=concept.source_type,
                    evidence_count=len(concept.evidence),
                    access_entity=access_entity,
                )
                if result:
                    # Link memory to concept
                    projector.link_memory_to_concept(
                        memory_id=memory_id,
                        user_id=user_id,
                        concept_name=concept.concept,
                        confidence=concept.confidence,
                        access_entity=access_entity,
                    )
                    # Link concept to mentioned entities
                    for entity_name in concept.entities:
                        projector.link_concept_to_entity(
                            user_id=user_id,
                            concept_name=concept.concept,
                            entity_name=entity_name,
                            access_entity=access_entity,
                        )
                    concepts_stored += 1

        result = {
            "entities_extracted": entities_stored,
            "concepts_extracted": concepts_stored,
            "language": extraction.language,
            "summary": extraction.summary,
        }

        # Optionally run contradiction detection
        if is_contradiction_detection_enabled() and concepts_stored > 0:
            contradictions = detect_contradictions_for_memory(
                memory_id=memory_id,
                user_id=user_id,
            )
            result["contradictions_found"] = len(contradictions)

        logger.info(
            f"Extracted {entities_stored} entities, {concepts_stored} concepts "
            f"from memory {memory_id}"
        )

        return result

    except Exception as e:
        logger.error(f"Concept extraction failed for memory {memory_id}: {e}")
        return {"error": str(e)}


def extract_concepts_batch(
    memories: List[Dict[str, Any]],
    user_id: str,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract concepts from multiple memories in batch.

    Args:
        memories: List of dicts with 'id', 'content', and optional 'category'
        user_id: User ID for legacy fallback
        access_entity: Default access entity for batch (overridden per memory)

    Returns:
        Dict with batch results
    """
    if not is_concepts_enabled():
        return {"error": "Business concepts system is not enabled"}

    results = {
        "processed": 0,
        "entities_total": 0,
        "concepts_total": 0,
        "errors": 0,
    }

    for memory in memories:
        memory_id = memory.get("id")
        content = memory.get("content")
        category = memory.get("category")

        if not memory_id or not content:
            results["errors"] += 1
            continue

        extraction = extract_and_store_concepts(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            category=category,
            access_entity=memory.get("access_entity") or access_entity,
        )

        if "error" in extraction:
            results["errors"] += 1
        else:
            results["processed"] += 1
            results["entities_total"] += extraction.get("entities_extracted", 0)
            results["concepts_total"] += extraction.get("concepts_extracted", 0)

    logger.info(
        f"Batch extraction: processed {results['processed']}/{len(memories)} memories"
    )

    return results


# =============================================================================
# Concept CRUD Operations
# =============================================================================

def get_concept(
    user_id: str,
    name: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]] | List[Dict[str, Any]]:
    """
    Get a concept by name with its evidence.

    Args:
        user_id: User ID for legacy fallback
        name: Concept name
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Dict with concept data, list of matches, or None if not found
    """
    if not is_concepts_enabled():
        return None

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return None

        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        return projector.get_concept(
            user_id=user_id,
            name=name,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    except Exception as e:
        logger.error(f"Failed to get concept {name}: {e}")
        return None


def list_concepts(
    user_id: str,
    category: Optional[str] = None,
    concept_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List concepts for a user with optional filters.

    Args:
        user_id: User ID for legacy fallback
        category: Optional category filter
        concept_type: Optional type filter
        min_confidence: Optional minimum confidence filter
        limit: Maximum results (default 50)
        offset: Pagination offset
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of concept dicts
    """
    if not is_concepts_enabled():
        return []

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return []

        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        return projector.list_concepts(
            user_id=user_id,
            category=category,
            concept_type=concept_type,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    except Exception as e:
        logger.error(f"Failed to list concepts: {e}")
        return []


def delete_concept(
    user_id: str,
    name: str,
    access_entity: Optional[str] = None,
) -> bool:
    """
    Delete a concept and its relationships.

    Args:
        user_id: User ID for scoping
        name: Concept name

    Returns:
        True if deleted, False otherwise
    """
    if not is_concepts_enabled():
        return False

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return False

        return projector.delete_concept(user_id, name, access_entity=access_entity)

    except Exception as e:
        logger.error(f"Failed to delete concept {name}: {e}")
        return False


def update_concept_confidence(
    user_id: str,
    name: str,
    access_entity: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Recalculate concept confidence based on supporting memories.

    Args:
        user_id: User ID for legacy fallback
        name: Concept name
        access_entity: Access entity scope (defaults to user)

    Returns:
        Dict with new confidence and evidence count, or None on error
    """
    if not is_concepts_enabled():
        return None

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return None

        return projector.update_concept_confidence(
            user_id=user_id,
            name=name,
            access_entity=access_entity,
        )

    except Exception as e:
        logger.error(f"Failed to update concept confidence: {e}")
        return None


# =============================================================================
# Business Entity Operations
# =============================================================================

def list_business_entities(
    user_id: str,
    entity_type: Optional[str] = None,
    min_importance: Optional[float] = None,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List business entities for a user.

    Args:
        user_id: User ID for legacy fallback
        entity_type: Optional type filter
        min_importance: Optional minimum importance filter
        limit: Maximum results (default 50)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of entity dicts
    """
    if not is_concepts_enabled():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            MATCH (e:OM_BizEntity)
            WHERE (
              (e.accessEntity IS NOT NULL AND (
                e.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
              ))
              OR (e.accessEntity IS NULL AND e.userId = $userId)
            )
              AND ($type IS NULL OR e.type = $type)
              AND ($minImportance IS NULL OR e.importance >= $minImportance)
            RETURN
                e.id AS id,
                e.name AS name,
                e.type AS type,
                e.importance AS importance,
                e.context AS context,
                e.mentionCount AS mentionCount,
                coalesce(e.accessEntity, $legacyAccessEntity) AS accessEntity,
                e.createdAt AS createdAt
            ORDER BY e.importance DESC, e.mentionCount DESC
            LIMIT $limit
            """
            access_entities, access_entity_prefixes = _normalize_access_filters(
                user_id,
                access_entities,
                access_entity_prefixes,
            )
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "type": entity_type,
                    "minImportance": min_importance,
                    "limit": min(100, max(1, limit)),
                    "accessEntities": access_entities,
                    "accessEntityPrefixes": access_entity_prefixes,
                    "legacyAccessEntity": f"user:{user_id}",
                }
            )
            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "importance": record["importance"],
                    "context": record["context"],
                    "mentionCount": record["mentionCount"],
                    "accessEntity": record["accessEntity"],
                    "createdAt": record["createdAt"],
                })
            return entities

    except Exception as e:
        logger.error(f"Failed to list business entities: {e}")
        return []


# =============================================================================
# Contradiction Detection
# =============================================================================

def detect_contradictions_for_memory(
    memory_id: str,
    user_id: str,
    access_entity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect contradictions between concepts related to a memory.

    Args:
        memory_id: UUID of the memory
        user_id: User ID for legacy fallback
        access_entity: Access entity scope (defaults to memory access)

    Returns:
        List of detected contradictions
    """
    if not is_contradiction_detection_enabled():
        return []

    try:
        from app.graph.convergence_detector import detect_contradictions_for_concept

        # Get concepts linked to this memory
        with get_neo4j_session() as session:
            cypher = """
            MATCH (m:OM_Memory {id: $memoryId})
            WITH m, coalesce(m.accessEntity, $legacyAccessEntity) AS resolvedAccess
            MATCH (m)-[:SUPPORTS]->(c:OM_Concept)
            WHERE coalesce(c.accessEntity, $legacyAccessEntity) = coalesce($accessEntity, resolvedAccess)
            RETURN c.name AS name, coalesce(c.accessEntity, $legacyAccessEntity) AS accessEntity
            """
            result = session.run(
                cypher,
                {
                    "memoryId": memory_id,
                    "userId": user_id,
                    "legacyAccessEntity": f"user:{user_id}",
                    "accessEntity": access_entity,
                },
            )
            concept_rows = [(record["name"], record["accessEntity"]) for record in result]

        all_contradictions = []
        for concept_name, concept_access_entity in concept_rows:
            contradictions = detect_contradictions_for_concept(
                user_id=user_id,
                concept_name=concept_name,
                access_entity=concept_access_entity,
            )
            all_contradictions.extend(contradictions)

        return all_contradictions

    except Exception as e:
        logger.error(f"Failed to detect contradictions: {e}")
        return []


def find_contradictions(
    user_id: str,
    concept_name: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find unresolved contradictions for a concept.

    Args:
        user_id: User ID for legacy fallback
        concept_name: Name of the concept
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of contradiction dicts
    """
    if not is_concepts_enabled():
        return []

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return []

        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        return projector.find_contradictions(
            user_id=user_id,
            concept_name=concept_name,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    except Exception as e:
        logger.error(f"Failed to find contradictions: {e}")
        return []


def resolve_contradiction(
    user_id: str,
    concept_name1: str,
    concept_name2: str,
    resolution: str,
    access_entity: Optional[str] = None,
) -> bool:
    """
    Mark a contradiction as resolved.

    Args:
        user_id: User ID for legacy fallback
        concept_name1: Name of the first concept
        concept_name2: Name of the second concept
        resolution: Resolution explanation
        access_entity: Access entity scope (defaults to user)

    Returns:
        True if successful, False otherwise
    """
    if not is_concepts_enabled():
        return False

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return False

        return projector.resolve_contradiction(
            user_id=user_id,
            concept_name1=concept_name1,
            concept_name2=concept_name2,
            resolution=resolution,
            access_entity=access_entity,
        )

    except Exception as e:
        logger.error(f"Failed to resolve contradiction: {e}")
        return False


# =============================================================================
# Concept Network Operations
# =============================================================================

def get_concept_network(
    user_id: str,
    concept_name: Optional[str] = None,
    depth: int = 2,
    limit: int = 50,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get the concept network graph for visualization.

    Args:
        user_id: User ID for legacy fallback
        concept_name: Optional seed concept (if None, returns full network)
        depth: Traversal depth (1-3)
        limit: Maximum nodes to return
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        Dict with nodes and edges for visualization
    """
    if not is_concepts_enabled():
        return {"nodes": [], "edges": [], "error": "Concepts not enabled"}

    depth = max(1, min(3, depth))
    limit = max(1, min(200, limit))
    access_entities, access_entity_prefixes = _normalize_access_filters(
        user_id,
        access_entities,
        access_entity_prefixes,
    )

    def access_filter(alias: str) -> str:
        return (
            "("
            f"({alias}.accessEntity IS NOT NULL AND ("
            f"{alias}.accessEntity IN $accessEntities "
            f"OR any(prefix IN $accessEntityPrefixes WHERE {alias}.accessEntity STARTS WITH prefix)"
            ")) "
            f"OR ({alias}.accessEntity IS NULL AND {alias}.userId = $userId)"
            ")"
        )

    try:
        with get_neo4j_session() as session:
            if concept_name:
                # Get network around a specific concept
                cypher = f"""
                MATCH (seed:OM_Concept {{name: $conceptName}})
                WHERE {access_filter("seed")}
                CALL apoc.path.subgraphAll(seed, {{
                    maxLevel: {depth},
                    relationshipFilter: 'SUPPORTS|RELATES_TO|CONTRADICTS|INVOLVES|MENTIONS'
                }})
                YIELD nodes, relationships
                UNWIND nodes AS node
                WITH COLLECT(DISTINCT node) AS allNodes, relationships
                UNWIND allNodes AS n
                WITH n, relationships
                WHERE (n:OM_Concept OR n:OM_BizEntity OR n:OM_Memory)
                  AND {access_filter("n")}
                WITH COLLECT(DISTINCT n) AS scopedNodes, relationships
                WITH scopedNodes,
                     [n IN scopedNodes | {{
                    id: CASE
                        WHEN n:OM_Concept THEN 'concept:' + n.name
                        WHEN n:OM_BizEntity THEN 'entity:' + n.name
                        WHEN n:OM_Memory THEN 'memory:' + n.id
                    END,
                    label: CASE
                        WHEN n:OM_Concept THEN n.name
                        WHEN n:OM_BizEntity THEN n.name
                        WHEN n:OM_Memory THEN left(n.content, 50)
                    END,
                    type: CASE
                        WHEN n:OM_Concept THEN 'concept'
                        WHEN n:OM_BizEntity THEN 'entity'
                        WHEN n:OM_Memory THEN 'memory'
                    END,
                    data: properties(n)
                }}][0..$limit] AS nodes, relationships
                UNWIND relationships AS r
                WITH nodes, scopedNodes, r
                WHERE {access_filter("r")}
                  AND startNode(r) IN scopedNodes
                  AND endNode(r) IN scopedNodes
                WITH nodes, COLLECT(DISTINCT {{
                    source: CASE
                        WHEN startNode(r):OM_Concept THEN 'concept:' + startNode(r).name
                        WHEN startNode(r):OM_BizEntity THEN 'entity:' + startNode(r).name
                        WHEN startNode(r):OM_Memory THEN 'memory:' + startNode(r).id
                    END,
                    target: CASE
                        WHEN endNode(r):OM_Concept THEN 'concept:' + endNode(r).name
                        WHEN endNode(r):OM_BizEntity THEN 'entity:' + endNode(r).name
                        WHEN endNode(r):OM_Memory THEN 'memory:' + endNode(r).id
                    END,
                    type: type(r),
                    data: properties(r)
                }}) AS edges
                RETURN nodes, edges
                """
            else:
                # Get full concept network
                cypher = """
                MATCH (c:OM_Concept)
                WHERE (
                  (c.accessEntity IS NOT NULL AND (
                    c.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE c.accessEntity STARTS WITH prefix)
                  ))
                  OR (c.accessEntity IS NULL AND c.userId = $userId)
                )
                OPTIONAL MATCH (c)-[r:RELATES_TO|CONTRADICTS]-(other:OM_Concept)
                WHERE (
                  (other.accessEntity IS NOT NULL AND (
                    other.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE other.accessEntity STARTS WITH prefix)
                  ))
                  OR (other.accessEntity IS NULL AND other.userId = $userId)
                )
                OPTIONAL MATCH (c)-[i:INVOLVES]->(e:OM_BizEntity)
                WHERE (
                  (e.accessEntity IS NOT NULL AND (
                    e.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
                  ))
                  OR (e.accessEntity IS NULL AND e.userId = $userId)
                )
                WITH COLLECT(DISTINCT {
                    id: 'concept:' + c.name,
                    label: c.name,
                    type: 'concept',
                    data: {
                        confidence: c.confidence,
                        conceptType: c.type,
                        category: c.category
                    }
                })[0..$limit] AS conceptNodes,
                COLLECT(DISTINCT CASE WHEN other IS NOT NULL THEN {
                    source: 'concept:' + c.name,
                    target: 'concept:' + other.name,
                    type: type(r),
                    data: properties(r)
                } END) AS conceptEdges,
                COLLECT(DISTINCT CASE WHEN e IS NOT NULL THEN {
                    id: 'entity:' + e.name,
                    label: e.name,
                    type: 'entity',
                    data: {
                        entityType: e.type,
                        importance: e.importance
                    }
                } END) AS entityNodes,
                COLLECT(DISTINCT CASE WHEN e IS NOT NULL THEN {
                    source: 'concept:' + c.name,
                    target: 'entity:' + e.name,
                    type: 'INVOLVES',
                    data: {}
                } END) AS entityEdges
                RETURN
                    conceptNodes + [n IN entityNodes WHERE n IS NOT NULL] AS nodes,
                    [e IN conceptEdges WHERE e IS NOT NULL] + [e IN entityEdges WHERE e IS NOT NULL] AS edges
                """

            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "conceptName": concept_name,
                    "limit": limit,
                    "accessEntities": access_entities,
                    "accessEntityPrefixes": access_entity_prefixes,
                }
            )

            for record in result:
                return {
                    "nodes": record["nodes"] or [],
                    "edges": record["edges"] or [],
                }

            return {"nodes": [], "edges": []}

    except Exception as e:
        logger.error(f"Failed to get concept network: {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


def get_concept_evolution(
    user_id: str,
    concept_name: str,
    days: int = 90,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get the evolution of a concept's confidence over time.

    Args:
        user_id: User ID for legacy fallback
        concept_name: Name of the concept
        days: Number of days to look back (default 90)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of dicts with date and confidence snapshots
    """
    if not is_concepts_enabled():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            MATCH (c:OM_Concept {name: $conceptName})
            WHERE (
              (c.accessEntity IS NOT NULL AND (
                c.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE c.accessEntity STARTS WITH prefix)
              ))
              OR (c.accessEntity IS NULL AND c.userId = $userId)
            )
            MATCH (m:OM_Memory)-[s:SUPPORTS]->(c)
            WHERE coalesce(m.accessEntity, $legacyAccessEntity) = coalesce(c.accessEntity, $legacyAccessEntity)
              AND m.createdAt >= datetime() - duration({days: $days})
            WITH c, m, s
            ORDER BY m.createdAt ASC
            WITH c, collect({
                date: m.createdAt,
                confidence: s.confidence,
                memoryId: m.id
            }) AS timeline
            RETURN timeline
            """
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "conceptName": concept_name,
                    "days": max(1, min(365, days)),
                    "accessEntities": (access_entities or [f"user:{user_id}"]),
                    "accessEntityPrefixes": access_entity_prefixes or [],
                    "legacyAccessEntity": f"user:{user_id}",
                }
            )

            for record in result:
                return record["timeline"] or []

            return []

    except Exception as e:
        logger.error(f"Failed to get concept evolution: {e}")
        return []


# =============================================================================
# Search Operations
# =============================================================================

def search_concepts(
    user_id: str,
    query: str,
    limit: int = 20,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Full-text search across concepts.

    Args:
        user_id: User ID for legacy fallback
        query: Search query
        limit: Maximum results
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of matching concepts with scores
    """
    if not is_concepts_enabled():
        return []

    if not query or not query.strip():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            CALL db.index.fulltext.queryNodes('om_concept_fulltext', $query)
            YIELD node, score
            WHERE score IS NOT NULL AND (
              (node.accessEntity IS NOT NULL AND (
                node.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE node.accessEntity STARTS WITH prefix)
              ))
              OR (node.accessEntity IS NULL AND node.userId = $userId)
            )
            RETURN
                node.id AS id,
                node.name AS name,
                node.type AS type,
                node.confidence AS confidence,
                node.category AS category,
                node.summary AS summary,
                coalesce(node.accessEntity, $legacyAccessEntity) AS accessEntity,
                score AS searchScore
            ORDER BY score DESC
            LIMIT $limit
            """
            access_entities, access_entity_prefixes = _normalize_access_filters(
                user_id,
                access_entities,
                access_entity_prefixes,
            )
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "query": query.strip(),
                    "limit": min(100, max(1, limit)),
                    "accessEntities": access_entities,
                    "accessEntityPrefixes": access_entity_prefixes,
                    "legacyAccessEntity": f"user:{user_id}",
                }
            )

            concepts = []
            for record in result:
                concepts.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "confidence": record["confidence"],
                    "category": record["category"],
                    "summary": record["summary"],
                    "accessEntity": record["accessEntity"],
                    "searchScore": record["searchScore"],
                })
            return concepts

    except Exception as e:
        # Full-text index may not exist yet
        logger.warning(f"Concept search failed: {e}")
        return []


def search_business_entities(
    user_id: str,
    query: str,
    limit: int = 20,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Full-text search across business entities.

    Args:
        user_id: User ID for legacy fallback
        query: Search query
        limit: Maximum results
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of matching entities with scores
    """
    if not is_concepts_enabled():
        return []

    if not query or not query.strip():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            CALL db.index.fulltext.queryNodes('om_bizentity_fulltext', $query)
            YIELD node, score
            WHERE score IS NOT NULL AND (
              (node.accessEntity IS NOT NULL AND (
                node.accessEntity IN $accessEntities
                OR any(prefix IN $accessEntityPrefixes WHERE node.accessEntity STARTS WITH prefix)
              ))
              OR (node.accessEntity IS NULL AND node.userId = $userId)
            )
            RETURN
                node.id AS id,
                node.name AS name,
                node.type AS type,
                node.importance AS importance,
                node.context AS context,
                coalesce(node.accessEntity, $legacyAccessEntity) AS accessEntity,
                score AS searchScore
            ORDER BY score DESC
            LIMIT $limit
            """
            access_entities, access_entity_prefixes = _normalize_access_filters(
                user_id,
                access_entities,
                access_entity_prefixes,
            )
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "query": query.strip(),
                    "limit": min(100, max(1, limit)),
                    "accessEntities": access_entities,
                    "accessEntityPrefixes": access_entity_prefixes,
                    "legacyAccessEntity": f"user:{user_id}",
                }
            )

            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "importance": record["importance"],
                    "context": record["context"],
                    "accessEntity": record["accessEntity"],
                    "searchScore": record["searchScore"],
                })
            return entities

    except Exception as e:
        logger.warning(f"Entity search failed: {e}")
        return []


# =============================================================================
# Vector Similarity Search Operations
# =============================================================================

def is_vector_search_enabled() -> bool:
    """Check if vector similarity search is available."""
    try:
        from app.graph.concept_vector_store import is_concept_embeddings_enabled
        return is_concepts_enabled() and is_concept_embeddings_enabled()
    except ImportError:
        return False


def semantic_search_concepts(
    user_id: str,
    query: str,
    top_k: int = 10,
    min_score: float = 0.5,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search for concepts using vector embeddings.

    This provides better search results than full-text search by
    understanding the meaning of the query, not just keywords.

    Args:
        user_id: User ID for legacy fallback
        query: Search query text
        top_k: Number of results to return
        min_score: Minimum similarity score (0-1)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of matching concepts with similarity scores
    """
    if not is_vector_search_enabled():
        # Fall back to full-text search
        return search_concepts(
            user_id,
            query,
            top_k,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    try:
        from app.graph.concept_vector_store import get_concept_similarity_service

        service = get_concept_similarity_service()
        if not service:
            return search_concepts(
                user_id,
                query,
                top_k,
                access_entities=access_entities,
                access_entity_prefixes=access_entity_prefixes,
            )

        # Get vector results
        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        vector_results = service.search_concepts(
            query=query,
            user_id=user_id,
            top_k=top_k,
            min_score=min_score,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        if not vector_results:
            return []

        # Enrich with full concept data from Neo4j
        enriched_results = []
        for vr in vector_results:
            concept_id = vr.get("concept_id")
            name = vr.get("name")

            # Get full concept from graph
            from app.graph.concept_projector import get_projector
            projector = get_projector()
            if projector:
                access_entity = vr.get("access_entity")
                full_concepts = projector.get_concept(
                    user_id=user_id,
                    name=name,
                    access_entities=access_entities,
                    access_entity_prefixes=access_entity_prefixes,
                )
                if isinstance(full_concepts, list):
                    scoped = [
                        c for c in full_concepts
                        if not access_entity or c.get("accessEntity") == access_entity
                    ]
                    selected = scoped[0] if scoped else (full_concepts[0] if full_concepts else None)
                else:
                    selected = full_concepts
                if selected:
                    selected["similarityScore"] = vr.get("score", 0)
                    enriched_results.append(selected)
                    continue

            # Fall back to vector result if graph lookup fails
            enriched_results.append({
                "id": concept_id,
                "name": name,
                "type": vr.get("type"),
                "category": vr.get("category"),
                "similarityScore": vr.get("score", 0),
            })

        return enriched_results

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Fall back to full-text
        return search_concepts(
            user_id,
            query,
            top_k,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )


def find_similar_concepts(
    user_id: str,
    concept_name: str,
    top_k: int = 5,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find concepts semantically similar to a given concept.

    Useful for:
    - Discovering related concepts
    - Identifying potential duplicates
    - Building concept clusters

    Args:
        user_id: User ID for legacy fallback
        concept_name: Name of the seed concept
        top_k: Number of similar concepts to return
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of similar concepts with similarity scores
    """
    if not is_vector_search_enabled():
        return []

    try:
        from app.graph.concept_vector_store import get_concept_similarity_service

        service = get_concept_similarity_service()
        if not service:
            return []

        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        return service.find_similar_concepts(
            concept_name=concept_name,
            user_id=user_id,
            top_k=top_k,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

    except Exception as e:
        logger.error(f"Find similar concepts failed: {e}")
        return []


def find_concept_duplicates(
    user_id: str,
    concept_name: str,
    threshold: Optional[float] = None,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Find potential duplicate concepts for a given concept.

    Uses a higher similarity threshold than general search
    to identify concepts that likely represent the same thing.

    Args:
        user_id: User ID for legacy fallback
        concept_name: Name of the concept to check
        threshold: Similarity threshold (default: from config, typically 0.75)
        access_entities: Explicit access_entity matches
        access_entity_prefixes: Access_entity prefixes (without % wildcards)

    Returns:
        List of potential duplicates with similarity scores
    """
    if not is_vector_search_enabled():
        return []

    try:
        from app.graph.concept_vector_store import (
            get_concept_similarity_service,
            ConceptEmbeddingConfig,
        )

        service = get_concept_similarity_service()
        if not service:
            return []

        # Use config threshold if not specified
        if threshold is None:
            config = ConceptEmbeddingConfig.from_env()
            threshold = config.similarity_threshold

        # Get embedding for the concept name
        embedding = service.embedder.embed_text(concept_name)

        # Find similar with high threshold
        access_entities, access_entity_prefixes = _normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        results = service.store.search_similar(
            query_embedding=embedding,
            user_id=user_id,
            top_k=10,
            min_score=threshold,
            access_entities=access_entities,
            access_entity_prefixes=access_entity_prefixes,
        )

        # Filter out exact name matches
        return [r for r in results if r.get("name") != concept_name]

    except Exception as e:
        logger.error(f"Find duplicates failed: {e}")
        return []


def get_concept_embedding_stats(user_id: str) -> Dict[str, Any]:
    """
    Get statistics about concept embeddings for a user.

    Args:
        user_id: User ID for scoping

    Returns:
        Dict with embedding statistics
    """
    result = {
        "embedding_enabled": is_vector_search_enabled(),
        "user_id": user_id,
    }

    if not is_vector_search_enabled():
        return result

    try:
        from app.graph.concept_vector_store import get_concept_similarity_service

        service = get_concept_similarity_service()
        if service:
            stats = service.store.get_collection_stats()
            result.update(stats)

    except Exception as e:
        result["error"] = str(e)

    return result
