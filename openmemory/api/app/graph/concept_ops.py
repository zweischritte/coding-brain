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
    vault: Optional[str] = None,
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
        user_id: User ID for scoping
        content: Memory content to extract from
        vault: Optional vault for concept scoping

    Returns:
        Dict with extraction results:
        - entities_extracted: Number of entities extracted
        - concepts_extracted: Number of concepts extracted
        - contradictions_found: Number of contradictions detected (if enabled)
        - error: Error message if extraction failed
    """
    if not is_concepts_enabled():
        return {"error": "Business concepts system is not enabled"}

    api_key = BusinessConceptsConfig.get_openai_api_key()
    if not api_key:
        return {"error": "OpenAI API key not configured for concept extraction"}

    try:
        # Get extractors and projectors
        from app.utils.concept_extractor import ConceptExtractor, TranscriptExtraction
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return {"error": "Concept projector not available"}

        extractor = ConceptExtractor(
            api_key=api_key,
            model=BusinessConceptsConfig.get_extraction_model(),
            max_tokens_per_chunk=BusinessConceptsConfig.get_max_tokens_per_chunk(),
        )

        # Extract concepts and entities
        extraction = extractor.extract_full(content)

        min_confidence = BusinessConceptsConfig.get_min_confidence()
        entities_stored = 0
        concepts_stored = 0

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
                )
                if result:
                    # Link memory to entity
                    projector.link_memory_to_bizentity(
                        memory_id=memory_id,
                        user_id=user_id,
                        entity_name=entity.entity,
                        importance=entity.importance,
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
                    vault=vault,
                    summary=None,
                    source_type=concept.source_type,
                    evidence_count=len(concept.evidence),
                )
                if result:
                    # Link memory to concept
                    projector.link_memory_to_concept(
                        memory_id=memory_id,
                        user_id=user_id,
                        concept_name=concept.concept,
                        confidence=concept.confidence,
                    )
                    # Link concept to mentioned entities
                    for entity_name in concept.entities:
                        projector.link_concept_to_entity(
                            user_id=user_id,
                            concept_name=concept.concept,
                            entity_name=entity_name,
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
) -> Dict[str, Any]:
    """
    Extract concepts from multiple memories in batch.

    Args:
        memories: List of dicts with 'id', 'content', and optional 'vault'
        user_id: User ID for scoping

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
        vault = memory.get("vault")

        if not memory_id or not content:
            results["errors"] += 1
            continue

        extraction = extract_and_store_concepts(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            vault=vault,
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
) -> Optional[Dict[str, Any]]:
    """
    Get a concept by name with its evidence.

    Args:
        user_id: User ID for scoping
        name: Concept name

    Returns:
        Dict with concept data, or None if not found
    """
    if not is_concepts_enabled():
        return None

    try:
        from app.graph.concept_projector import get_projector

        projector = get_projector()
        if not projector:
            return None

        return projector.get_concept(user_id, name)

    except Exception as e:
        logger.error(f"Failed to get concept {name}: {e}")
        return None


def list_concepts(
    user_id: str,
    vault: Optional[str] = None,
    concept_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List concepts for a user with optional filters.

    Args:
        user_id: User ID for scoping
        vault: Optional vault filter
        concept_type: Optional type filter
        min_confidence: Optional minimum confidence filter
        limit: Maximum results (default 50)
        offset: Pagination offset

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

        return projector.list_concepts(
            user_id=user_id,
            vault=vault,
            concept_type=concept_type,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"Failed to list concepts: {e}")
        return []


def delete_concept(
    user_id: str,
    name: str,
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

        return projector.delete_concept(user_id, name)

    except Exception as e:
        logger.error(f"Failed to delete concept {name}: {e}")
        return False


def update_concept_confidence(
    user_id: str,
    name: str,
) -> Optional[Dict[str, Any]]:
    """
    Recalculate concept confidence based on supporting memories.

    Args:
        user_id: User ID for scoping
        name: Concept name

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

        return projector.update_concept_confidence(user_id, name)

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
) -> List[Dict[str, Any]]:
    """
    List business entities for a user.

    Args:
        user_id: User ID for scoping
        entity_type: Optional type filter
        min_importance: Optional minimum importance filter
        limit: Maximum results (default 50)

    Returns:
        List of entity dicts
    """
    if not is_concepts_enabled():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            MATCH (e:OM_BizEntity {userId: $userId})
            WHERE ($type IS NULL OR e.type = $type)
              AND ($minImportance IS NULL OR e.importance >= $minImportance)
            RETURN
                e.id AS id,
                e.name AS name,
                e.type AS type,
                e.importance AS importance,
                e.context AS context,
                e.mentionCount AS mentionCount,
                e.createdAt AS createdAt
            ORDER BY e.importance DESC, e.mentionCount DESC
            LIMIT $limit
            """
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "type": entity_type,
                    "minImportance": min_importance,
                    "limit": min(100, max(1, limit)),
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
) -> List[Dict[str, Any]]:
    """
    Detect contradictions between concepts related to a memory.

    Args:
        memory_id: UUID of the memory
        user_id: User ID for scoping

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
            MATCH (m:OM_Memory {id: $memoryId})-[:SUPPORTS]->(c:OM_Concept {userId: $userId})
            RETURN c.name AS name
            """
            result = session.run(cypher, {"memoryId": memory_id, "userId": user_id})
            concept_names = [record["name"] for record in result]

        all_contradictions = []
        for concept_name in concept_names:
            contradictions = detect_contradictions_for_concept(
                user_id=user_id,
                concept_name=concept_name,
            )
            all_contradictions.extend(contradictions)

        return all_contradictions

    except Exception as e:
        logger.error(f"Failed to detect contradictions: {e}")
        return []


def find_contradictions(
    user_id: str,
    concept_name: str,
) -> List[Dict[str, Any]]:
    """
    Find unresolved contradictions for a concept.

    Args:
        user_id: User ID for scoping
        concept_name: Name of the concept

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

        return projector.find_contradictions(user_id, concept_name)

    except Exception as e:
        logger.error(f"Failed to find contradictions: {e}")
        return []


def resolve_contradiction(
    user_id: str,
    concept_name1: str,
    concept_name2: str,
    resolution: str,
) -> bool:
    """
    Mark a contradiction as resolved.

    Args:
        user_id: User ID for scoping
        concept_name1: Name of the first concept
        concept_name2: Name of the second concept
        resolution: Resolution explanation

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
) -> Dict[str, Any]:
    """
    Get the concept network graph for visualization.

    Args:
        user_id: User ID for scoping
        concept_name: Optional seed concept (if None, returns full network)
        depth: Traversal depth (1-3)
        limit: Maximum nodes to return

    Returns:
        Dict with nodes and edges for visualization
    """
    if not is_concepts_enabled():
        return {"nodes": [], "edges": [], "error": "Concepts not enabled"}

    depth = max(1, min(3, depth))
    limit = max(1, min(200, limit))

    try:
        with get_neo4j_session() as session:
            if concept_name:
                # Get network around a specific concept
                cypher = f"""
                MATCH (seed:OM_Concept {{userId: $userId, name: $conceptName}})
                CALL apoc.path.subgraphAll(seed, {{
                    maxLevel: {depth},
                    relationshipFilter: 'SUPPORTS|RELATES_TO|CONTRADICTS|INVOLVES|MENTIONS'
                }})
                YIELD nodes, relationships
                UNWIND nodes AS node
                WITH COLLECT(DISTINCT node) AS allNodes, relationships
                UNWIND allNodes AS n
                WITH n, relationships
                WHERE n:OM_Concept OR n:OM_BizEntity OR n:OM_Memory
                WITH COLLECT(DISTINCT {{
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
                }})[0..$limit] AS nodes, relationships
                UNWIND relationships AS r
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
                MATCH (c:OM_Concept {userId: $userId})
                OPTIONAL MATCH (c)-[r:RELATES_TO|CONTRADICTS]-(other:OM_Concept {userId: $userId})
                OPTIONAL MATCH (c)-[i:INVOLVES]->(e:OM_BizEntity {userId: $userId})
                WITH COLLECT(DISTINCT {
                    id: 'concept:' + c.name,
                    label: c.name,
                    type: 'concept',
                    data: {
                        confidence: c.confidence,
                        conceptType: c.type,
                        vault: c.vault
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
) -> List[Dict[str, Any]]:
    """
    Get the evolution of a concept's confidence over time.

    Args:
        user_id: User ID for scoping
        concept_name: Name of the concept
        days: Number of days to look back (default 90)

    Returns:
        List of dicts with date and confidence snapshots
    """
    if not is_concepts_enabled():
        return []

    try:
        with get_neo4j_session() as session:
            cypher = """
            MATCH (c:OM_Concept {userId: $userId, name: $conceptName})
            MATCH (m:OM_Memory)-[s:SUPPORTS]->(c)
            WHERE m.createdAt >= datetime() - duration({days: $days})
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
) -> List[Dict[str, Any]]:
    """
    Full-text search across concepts.

    Args:
        user_id: User ID for scoping
        query: Search query
        limit: Maximum results

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
            WHERE node.userId = $userId
            RETURN
                node.id AS id,
                node.name AS name,
                node.type AS type,
                node.confidence AS confidence,
                node.vault AS vault,
                node.summary AS summary,
                score AS searchScore
            ORDER BY score DESC
            LIMIT $limit
            """
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "query": query.strip(),
                    "limit": min(100, max(1, limit)),
                }
            )

            concepts = []
            for record in result:
                concepts.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "confidence": record["confidence"],
                    "vault": record["vault"],
                    "summary": record["summary"],
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
) -> List[Dict[str, Any]]:
    """
    Full-text search across business entities.

    Args:
        user_id: User ID for scoping
        query: Search query
        limit: Maximum results

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
            WHERE node.userId = $userId
            RETURN
                node.id AS id,
                node.name AS name,
                node.type AS type,
                node.importance AS importance,
                node.context AS context,
                score AS searchScore
            ORDER BY score DESC
            LIMIT $limit
            """
            result = session.run(
                cypher,
                {
                    "userId": user_id,
                    "query": query.strip(),
                    "limit": min(100, max(1, limit)),
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
) -> List[Dict[str, Any]]:
    """
    Semantic search for concepts using vector embeddings.

    This provides better search results than full-text search by
    understanding the meaning of the query, not just keywords.

    Args:
        user_id: User ID for scoping
        query: Search query text
        top_k: Number of results to return
        min_score: Minimum similarity score (0-1)

    Returns:
        List of matching concepts with similarity scores
    """
    if not is_vector_search_enabled():
        # Fall back to full-text search
        return search_concepts(user_id, query, top_k)

    try:
        from app.graph.concept_vector_store import get_concept_similarity_service

        service = get_concept_similarity_service()
        if not service:
            return search_concepts(user_id, query, top_k)

        # Get vector results
        vector_results = service.search_concepts(
            query=query,
            user_id=user_id,
            top_k=top_k,
            min_score=min_score,
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
                full_concept = projector.get_concept(user_id, name)
                if full_concept:
                    full_concept["similarityScore"] = vr.get("score", 0)
                    enriched_results.append(full_concept)
                    continue

            # Fall back to vector result if graph lookup fails
            enriched_results.append({
                "id": concept_id,
                "name": name,
                "type": vr.get("type"),
                "vault": vr.get("vault"),
                "similarityScore": vr.get("score", 0),
            })

        return enriched_results

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        # Fall back to full-text
        return search_concepts(user_id, query, top_k)


def find_similar_concepts(
    user_id: str,
    concept_name: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find concepts semantically similar to a given concept.

    Useful for:
    - Discovering related concepts
    - Identifying potential duplicates
    - Building concept clusters

    Args:
        user_id: User ID for scoping
        concept_name: Name of the seed concept
        top_k: Number of similar concepts to return

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

        return service.find_similar_concepts(
            concept_name=concept_name,
            user_id=user_id,
            top_k=top_k,
        )

    except Exception as e:
        logger.error(f"Find similar concepts failed: {e}")
        return []


def find_concept_duplicates(
    user_id: str,
    concept_name: str,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Find potential duplicate concepts for a given concept.

    Uses a higher similarity threshold than general search
    to identify concepts that likely represent the same thing.

    Args:
        user_id: User ID for scoping
        concept_name: Name of the concept to check
        threshold: Similarity threshold (default: from config, typically 0.75)

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
        results = service.store.search_similar(
            query_embedding=embedding,
            user_id=user_id,
            top_k=10,
            min_score=threshold,
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
