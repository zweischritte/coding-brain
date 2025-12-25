"""
Business Concept Projector for Neo4j.

Creates and maintains the OM_Concept node type and its relationships with memories,
entities, and other concepts.

Schema:
- OM_Concept: Node type for business concepts
- OM_BIZ_ENTITY: Node type for business entities (companies, people, products, etc.)
- SUPPORTS: Memory supports a concept
- MENTIONS: Memory mentions a business entity
- RELATES_TO: Concept relates to another concept
- CONTRADICTS: Concept contradicts another concept

Based on implementation plan:
- /docs/implementation/business-concept-implementation-plan.md
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Valid concept types (from BusinessConcept schema)
CONCEPT_TYPES = frozenset([
    "causal",
    "pattern",
    "comparison",
    "trend",
    "contradiction",
    "hypothesis",
    "fact",
    # NEW
    "implementation",
    "product_architecture",
    "market_structure",
    "success_story",
    "mental_model",
    "pricing_insight",
])

# Valid entity types (from BusinessEntity schema)
ENTITY_TYPES = frozenset([
    "company",
    "person",
    "product",
    "market",
    "metric",
    "business_model",
    "technology",
    "strategy",
    # NEW
    "tactic",
    "case_study",
    "product_idea",
    "framework",
    "competitive_intel",
    "pricing",
    "tool_config",
])

# Valid source types
SOURCE_TYPES = frozenset([
    "stated_fact",
    "inference",
    "opinion",
])


class ConceptCypherBuilder:
    """Builds Cypher queries for concept projection."""

    @staticmethod
    def constraint_queries() -> List[str]:
        """
        Generate constraint creation queries for concept schema.

        Returns list of Cypher statements to create constraints IF NOT EXISTS.
        """
        return [
            # OM_Concept constraints
            "CREATE CONSTRAINT om_concept_id IF NOT EXISTS FOR (c:OM_Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT om_concept_user_name IF NOT EXISTS FOR (c:OM_Concept) REQUIRE (c.userId, c.name) IS UNIQUE",
            # OM_BizEntity constraints
            "CREATE CONSTRAINT om_bizentity_id IF NOT EXISTS FOR (e:OM_BizEntity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT om_bizentity_user_name IF NOT EXISTS FOR (e:OM_BizEntity) REQUIRE (e.userId, e.name) IS UNIQUE",
            # Indexes for efficient querying
            "CREATE INDEX om_concept_user_id IF NOT EXISTS FOR (c:OM_Concept) ON (c.userId)",
            "CREATE INDEX om_concept_vault IF NOT EXISTS FOR (c:OM_Concept) ON (c.vault)",
            "CREATE INDEX om_concept_type IF NOT EXISTS FOR (c:OM_Concept) ON (c.type)",
            "CREATE INDEX om_concept_confidence IF NOT EXISTS FOR (c:OM_Concept) ON (c.confidence)",
            "CREATE INDEX om_concept_created IF NOT EXISTS FOR (c:OM_Concept) ON (c.createdAt)",
            "CREATE INDEX om_bizentity_user_id IF NOT EXISTS FOR (e:OM_BizEntity) ON (e.userId)",
            "CREATE INDEX om_bizentity_type IF NOT EXISTS FOR (e:OM_BizEntity) ON (e.type)",
            "CREATE INDEX om_bizentity_importance IF NOT EXISTS FOR (e:OM_BizEntity) ON (e.importance)",
            # Relationship indexes
            "CREATE INDEX om_supports_confidence IF NOT EXISTS FOR ()-[r:SUPPORTS]-() ON (r.confidence)",
            "CREATE INDEX om_contradicts_user_id IF NOT EXISTS FOR ()-[r:CONTRADICTS]-() ON (r.userId)",
        ]

    @staticmethod
    def fulltext_index_queries() -> List[str]:
        """
        Generate full-text index creation queries for concept search.
        """
        return [
            # Full-text index on concept names and content
            "CREATE FULLTEXT INDEX om_concept_fulltext IF NOT EXISTS FOR (c:OM_Concept) ON EACH [c.name, c.summary]",
            # Full-text index on business entity names
            "CREATE FULLTEXT INDEX om_bizentity_fulltext IF NOT EXISTS FOR (e:OM_BizEntity) ON EACH [e.name, e.context]",
        ]

    @staticmethod
    def upsert_concept_query() -> str:
        """
        Generate the concept upsert query.

        Creates or updates an OM_Concept node.
        """
        return """
        MERGE (c:OM_Concept {userId: $userId, name: $name})
        ON CREATE SET
            c.id = $id,
            c.type = $type,
            c.confidence = $confidence,
            c.vault = $vault,
            c.summary = $summary,
            c.sourceType = $sourceType,
            c.evidenceCount = $evidenceCount,
            c.createdAt = datetime(),
            c.updatedAt = datetime(),
            c.projectedAt = datetime()
        ON MATCH SET
            c.type = COALESCE($type, c.type),
            c.confidence = CASE WHEN $confidence > c.confidence THEN $confidence ELSE c.confidence END,
            c.vault = COALESCE($vault, c.vault),
            c.summary = COALESCE($summary, c.summary),
            c.sourceType = COALESCE($sourceType, c.sourceType),
            c.evidenceCount = COALESCE($evidenceCount, c.evidenceCount),
            c.updatedAt = datetime(),
            c.projectedAt = datetime()
        RETURN c.id AS id, c.name AS name
        """

    @staticmethod
    def upsert_bizentity_query() -> str:
        """
        Generate the business entity upsert query.

        Creates or updates an OM_BizEntity node.
        """
        return """
        MERGE (e:OM_BizEntity {userId: $userId, name: $name})
        ON CREATE SET
            e.id = $id,
            e.type = $type,
            e.importance = $importance,
            e.context = $context,
            e.mentionCount = $mentionCount,
            e.createdAt = datetime(),
            e.updatedAt = datetime(),
            e.projectedAt = datetime()
        ON MATCH SET
            e.type = COALESCE($type, e.type),
            e.importance = CASE WHEN $importance > e.importance THEN $importance ELSE e.importance END,
            e.context = COALESCE($context, e.context),
            e.mentionCount = COALESCE(e.mentionCount, 0) + COALESCE($mentionCount, 1),
            e.updatedAt = datetime(),
            e.projectedAt = datetime()
        RETURN e.id AS id, e.name AS name
        """

    @staticmethod
    def link_memory_to_concept_query() -> str:
        """
        Generate query to link a memory to a concept via SUPPORTS relationship.
        """
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MATCH (c:OM_Concept {userId: $userId, name: $conceptName})
        MERGE (m)-[r:SUPPORTS]->(c)
        ON CREATE SET
            r.confidence = $confidence,
            r.createdAt = datetime()
        ON MATCH SET
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r.updatedAt = datetime()
        RETURN type(r) AS relType
        """

    @staticmethod
    def link_memory_to_bizentity_query() -> str:
        """
        Generate query to link a memory to a business entity via MENTIONS relationship.
        """
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MATCH (e:OM_BizEntity {userId: $userId, name: $entityName})
        MERGE (m)-[r:MENTIONS]->(e)
        ON CREATE SET
            r.importance = $importance,
            r.createdAt = datetime()
        ON MATCH SET
            r.importance = CASE WHEN $importance > r.importance THEN $importance ELSE r.importance END,
            r.updatedAt = datetime()
        RETURN type(r) AS relType
        """

    @staticmethod
    def link_concept_to_concept_query() -> str:
        """
        Generate query to link concepts via RELATES_TO relationship.
        """
        return """
        MATCH (c1:OM_Concept {userId: $userId, name: $conceptName1})
        MATCH (c2:OM_Concept {userId: $userId, name: $conceptName2})
        WHERE c1 <> c2
        MERGE (c1)-[r:RELATES_TO]->(c2)
        ON CREATE SET
            r.strength = $strength,
            r.createdAt = datetime()
        ON MATCH SET
            r.strength = CASE WHEN $strength > r.strength THEN $strength ELSE r.strength END,
            r.updatedAt = datetime()
        RETURN type(r) AS relType
        """

    @staticmethod
    def create_contradiction_query() -> str:
        """
        Generate query to create CONTRADICTS relationship between concepts.
        """
        return """
        MATCH (c1:OM_Concept {userId: $userId, name: $conceptName1})
        MATCH (c2:OM_Concept {userId: $userId, name: $conceptName2})
        WHERE c1 <> c2
        MERGE (c1)-[r:CONTRADICTS {userId: $userId}]->(c2)
        ON CREATE SET
            r.detectedAt = datetime(),
            r.severity = $severity,
            r.evidence = $evidence,
            r.resolved = false,
            r.createdAt = datetime()
        ON MATCH SET
            r.severity = CASE WHEN $severity > r.severity THEN $severity ELSE r.severity END,
            r.evidence = CASE WHEN size($evidence) > size(r.evidence) THEN $evidence ELSE r.evidence END,
            r.updatedAt = datetime()
        RETURN type(r) AS relType
        """

    @staticmethod
    def link_concept_to_entity_query() -> str:
        """
        Generate query to link a concept to a business entity.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId, name: $conceptName})
        MATCH (e:OM_BizEntity {userId: $userId, name: $entityName})
        MERGE (c)-[r:INVOLVES]->(e)
        ON CREATE SET
            r.createdAt = datetime()
        RETURN type(r) AS relType
        """

    @staticmethod
    def get_concept_query() -> str:
        """
        Generate query to get a concept by name.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId, name: $name})
        OPTIONAL MATCH (m:OM_Memory)-[s:SUPPORTS]->(c)
        WITH c, collect({id: m.id, content: m.content, confidence: s.confidence}) AS evidence
        RETURN
            c.id AS id,
            c.name AS name,
            c.type AS type,
            c.confidence AS confidence,
            c.vault AS vault,
            c.summary AS summary,
            c.sourceType AS sourceType,
            c.evidenceCount AS evidenceCount,
            c.createdAt AS createdAt,
            c.updatedAt AS updatedAt,
            evidence
        """

    @staticmethod
    def list_concepts_query() -> str:
        """
        Generate query to list concepts for a user.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId})
        WHERE ($vault IS NULL OR c.vault = $vault)
          AND ($type IS NULL OR c.type = $type)
          AND ($minConfidence IS NULL OR c.confidence >= $minConfidence)
        OPTIONAL MATCH (m:OM_Memory)-[:SUPPORTS]->(c)
        WITH c, count(m) AS supportingMemories
        RETURN
            c.id AS id,
            c.name AS name,
            c.type AS type,
            c.confidence AS confidence,
            c.vault AS vault,
            c.summary AS summary,
            c.sourceType AS sourceType,
            c.evidenceCount AS evidenceCount,
            supportingMemories,
            c.createdAt AS createdAt,
            c.updatedAt AS updatedAt
        ORDER BY c.confidence DESC, c.updatedAt DESC
        SKIP $offset
        LIMIT $limit
        """

    @staticmethod
    def find_contradictions_query() -> str:
        """
        Generate query to find contradictions for a concept.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId, name: $conceptName})
        OPTIONAL MATCH (c)-[r:CONTRADICTS]-(other:OM_Concept)
        WHERE r.resolved = false
        RETURN
            other.name AS contradictingConcept,
            other.confidence AS otherConfidence,
            r.severity AS severity,
            r.evidence AS evidence,
            r.detectedAt AS detectedAt
        ORDER BY r.severity DESC
        """

    @staticmethod
    def resolve_contradiction_query() -> str:
        """
        Generate query to mark a contradiction as resolved.
        """
        return """
        MATCH (c1:OM_Concept {userId: $userId, name: $conceptName1})
        MATCH (c2:OM_Concept {userId: $userId, name: $conceptName2})
        MATCH (c1)-[r:CONTRADICTS]-(c2)
        SET r.resolved = true,
            r.resolvedAt = datetime(),
            r.resolution = $resolution
        RETURN type(r) AS relType
        """

    @staticmethod
    def delete_concept_query() -> str:
        """
        Generate query to delete a concept and its relationships.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId, name: $name})
        DETACH DELETE c
        RETURN count(c) AS deleted
        """

    @staticmethod
    def update_concept_confidence_query() -> str:
        """
        Generate query to update concept confidence based on supporting memories.
        """
        return """
        MATCH (c:OM_Concept {userId: $userId, name: $name})
        OPTIONAL MATCH (m:OM_Memory)-[s:SUPPORTS]->(c)
        WITH c, count(m) AS memoryCount, avg(s.confidence) AS avgConfidence
        SET c.confidence = CASE
            WHEN memoryCount >= 5 THEN LEAST(1.0, avgConfidence + 0.1)
            WHEN memoryCount >= 3 THEN avgConfidence
            ELSE GREATEST(0.3, avgConfidence - 0.1)
        END,
            c.evidenceCount = memoryCount,
            c.updatedAt = datetime()
        RETURN c.confidence AS newConfidence, c.evidenceCount AS evidenceCount
        """


class ConceptProjector:
    """
    Projects business concepts into Neo4j as a concept graph.

    Manages OM_Concept and OM_BizEntity nodes, and their relationships
    with memories and each other.
    """

    def __init__(self, session_factory, config=None):
        """
        Initialize the projector.

        Args:
            session_factory: Callable that returns a Neo4j session context manager
            config: Optional projector configuration
        """
        self.session_factory = session_factory
        self.config = config or {}
        self._constraints_created = False

    def ensure_constraints(self) -> bool:
        """
        Ensure all required constraints and indexes exist.

        Returns:
            True if constraints were created/verified, False on error
        """
        if self._constraints_created:
            return True

        try:
            with self.session_factory() as session:
                # Create standard constraints and indexes
                for query in ConceptCypherBuilder.constraint_queries():
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.debug(f"Constraint query note: {e}")

                # Create full-text indexes
                for query in ConceptCypherBuilder.fulltext_index_queries():
                    try:
                        session.run(query)
                    except Exception as e:
                        logger.debug(f"Full-text index note: {e}")

                session.run("RETURN 1")  # Ensure transaction commits
            self._constraints_created = True
            logger.info("Neo4j constraints and indexes ensured for concept projection")
            return True
        except Exception as e:
            logger.error(f"Failed to create Neo4j concept constraints: {e}")
            return False

    def upsert_concept(
        self,
        user_id: str,
        name: str,
        concept_type: str,
        confidence: float,
        vault: Optional[str] = None,
        summary: Optional[str] = None,
        source_type: Optional[str] = None,
        evidence_count: int = 0,
        concept_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update a concept node.

        Also generates and stores vector embedding if embedding is enabled.

        Args:
            user_id: User ID for scoping
            name: Concept name (unique per user)
            concept_type: Type from CONCEPT_TYPES
            confidence: Confidence score 0.0-1.0
            vault: Optional vault (WLT, FRC, etc.)
            summary: Optional summary text
            source_type: Optional source type from SOURCE_TYPES
            evidence_count: Number of supporting evidence items
            concept_id: Optional UUID (generated if not provided)

        Returns:
            Dict with id, name, and potential_duplicates (if embeddings enabled)
        """
        if concept_type not in CONCEPT_TYPES:
            logger.warning(f"Invalid concept type: {concept_type}")
            return None

        if source_type and source_type not in SOURCE_TYPES:
            logger.warning(f"Invalid source type: {source_type}")
            source_type = None

        import uuid
        concept_id = concept_id or str(uuid.uuid4())

        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.upsert_concept_query(),
                    {
                        "userId": user_id,
                        "id": concept_id,
                        "name": name,
                        "type": concept_type,
                        "confidence": min(1.0, max(0.0, confidence)),
                        "vault": vault,
                        "summary": summary,
                        "sourceType": source_type,
                        "evidenceCount": evidence_count,
                    }
                )
                result_dict = None
                for record in result:
                    logger.debug(f"Upserted concept: {record['name']}")
                    result_dict = {"id": record["id"], "name": record["name"]}
                    break

                if result_dict is None:
                    return None

                # Generate and store embedding if enabled
                try:
                    from app.graph.concept_vector_store import (
                        get_concept_similarity_service,
                        is_concept_embeddings_enabled,
                    )

                    if is_concept_embeddings_enabled():
                        similarity_service = get_concept_similarity_service()
                        if similarity_service:
                            success, duplicates = similarity_service.embed_and_store(
                                concept_id=result_dict["id"],
                                user_id=user_id,
                                name=name,
                                concept_type=concept_type,
                                summary=summary,
                                vault=vault,
                                source_type=source_type,
                            )
                            if success:
                                logger.debug(f"Stored embedding for concept: {name}")
                            if duplicates:
                                result_dict["potential_duplicates"] = duplicates
                                logger.info(
                                    f"Found {len(duplicates)} potential duplicates for concept: {name}"
                                )
                except Exception as embed_error:
                    # Log but don't fail - embedding is optional
                    logger.warning(f"Failed to embed concept {name}: {embed_error}")

                return result_dict

        except Exception as e:
            logger.error(f"Failed to upsert concept {name}: {e}")
            return None

    def upsert_bizentity(
        self,
        user_id: str,
        name: str,
        entity_type: str,
        importance: float,
        context: Optional[str] = None,
        mention_count: int = 1,
        entity_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update a business entity node.

        Args:
            user_id: User ID for scoping
            name: Entity name (unique per user)
            entity_type: Type from ENTITY_TYPES
            importance: Importance score 0.0-1.0
            context: Optional context text
            mention_count: Number of mentions
            entity_id: Optional UUID (generated if not provided)

        Returns:
            Dict with id and name, or None on error
        """
        if entity_type not in ENTITY_TYPES:
            logger.warning(f"Invalid entity type: {entity_type}")
            return None

        import uuid
        entity_id = entity_id or str(uuid.uuid4())

        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.upsert_bizentity_query(),
                    {
                        "userId": user_id,
                        "id": entity_id,
                        "name": name,
                        "type": entity_type,
                        "importance": min(1.0, max(0.0, importance)),
                        "context": context,
                        "mentionCount": mention_count,
                    }
                )
                for record in result:
                    logger.debug(f"Upserted bizentity: {record['name']}")
                    return {"id": record["id"], "name": record["name"]}
                return None
        except Exception as e:
            logger.error(f"Failed to upsert bizentity {name}: {e}")
            return None

    def link_memory_to_concept(
        self,
        memory_id: str,
        user_id: str,
        concept_name: str,
        confidence: float = 0.5,
    ) -> bool:
        """
        Link a memory to a concept via SUPPORTS relationship.

        Args:
            memory_id: UUID of the memory
            user_id: User ID for scoping
            concept_name: Name of the concept
            confidence: Confidence score for the support relationship

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.link_memory_to_concept_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "conceptName": concept_name,
                        "confidence": min(1.0, max(0.0, confidence)),
                    }
                )
                for record in result:
                    logger.debug(f"Linked memory {memory_id} to concept {concept_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to link memory to concept: {e}")
            return False

    def link_memory_to_bizentity(
        self,
        memory_id: str,
        user_id: str,
        entity_name: str,
        importance: float = 0.5,
    ) -> bool:
        """
        Link a memory to a business entity via MENTIONS relationship.

        Args:
            memory_id: UUID of the memory
            user_id: User ID for scoping
            entity_name: Name of the business entity
            importance: Importance score for the mention

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.link_memory_to_bizentity_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "entityName": entity_name,
                        "importance": min(1.0, max(0.0, importance)),
                    }
                )
                for record in result:
                    logger.debug(f"Linked memory {memory_id} to entity {entity_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to link memory to bizentity: {e}")
            return False

    def link_concepts(
        self,
        user_id: str,
        concept_name1: str,
        concept_name2: str,
        strength: float = 0.5,
    ) -> bool:
        """
        Link two concepts via RELATES_TO relationship.

        Args:
            user_id: User ID for scoping
            concept_name1: Name of the first concept
            concept_name2: Name of the second concept
            strength: Relationship strength 0.0-1.0

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.link_concept_to_concept_query(),
                    {
                        "userId": user_id,
                        "conceptName1": concept_name1,
                        "conceptName2": concept_name2,
                        "strength": min(1.0, max(0.0, strength)),
                    }
                )
                for record in result:
                    logger.debug(f"Linked concepts: {concept_name1} -> {concept_name2}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to link concepts: {e}")
            return False

    def create_contradiction(
        self,
        user_id: str,
        concept_name1: str,
        concept_name2: str,
        severity: float = 0.5,
        evidence: Optional[List[str]] = None,
    ) -> bool:
        """
        Create a CONTRADICTS relationship between two concepts.

        Args:
            user_id: User ID for scoping
            concept_name1: Name of the first concept
            concept_name2: Name of the second concept
            severity: Severity of the contradiction 0.0-1.0
            evidence: List of evidence strings

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.create_contradiction_query(),
                    {
                        "userId": user_id,
                        "conceptName1": concept_name1,
                        "conceptName2": concept_name2,
                        "severity": min(1.0, max(0.0, severity)),
                        "evidence": evidence or [],
                    }
                )
                for record in result:
                    logger.info(f"Created contradiction: {concept_name1} <-> {concept_name2}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to create contradiction: {e}")
            return False

    def link_concept_to_entity(
        self,
        user_id: str,
        concept_name: str,
        entity_name: str,
    ) -> bool:
        """
        Link a concept to a business entity via INVOLVES relationship.

        Args:
            user_id: User ID for scoping
            concept_name: Name of the concept
            entity_name: Name of the business entity

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.link_concept_to_entity_query(),
                    {
                        "userId": user_id,
                        "conceptName": concept_name,
                        "entityName": entity_name,
                    }
                )
                for record in result:
                    logger.debug(f"Linked concept {concept_name} to entity {entity_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to link concept to entity: {e}")
            return False

    def get_concept(
        self,
        user_id: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a concept by name with its evidence.

        Args:
            user_id: User ID for scoping
            name: Concept name

        Returns:
            Dict with concept data and evidence, or None if not found
        """
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.get_concept_query(),
                    {"userId": user_id, "name": name}
                )
                for record in result:
                    return {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "confidence": record["confidence"],
                        "vault": record["vault"],
                        "summary": record["summary"],
                        "sourceType": record["sourceType"],
                        "evidenceCount": record["evidenceCount"],
                        "createdAt": record["createdAt"],
                        "updatedAt": record["updatedAt"],
                        "evidence": record["evidence"],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get concept {name}: {e}")
            return None

    def list_concepts(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.list_concepts_query(),
                    {
                        "userId": user_id,
                        "vault": vault,
                        "type": concept_type,
                        "minConfidence": min_confidence,
                        "limit": min(100, max(1, limit)),
                        "offset": max(0, offset),
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
                        "sourceType": record["sourceType"],
                        "evidenceCount": record["evidenceCount"],
                        "supportingMemories": record["supportingMemories"],
                        "createdAt": record["createdAt"],
                        "updatedAt": record["updatedAt"],
                    })
                return concepts
        except Exception as e:
            logger.error(f"Failed to list concepts: {e}")
            return []

    def find_contradictions(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.find_contradictions_query(),
                    {"userId": user_id, "conceptName": concept_name}
                )
                contradictions = []
                for record in result:
                    if record["contradictingConcept"]:
                        contradictions.append({
                            "concept": record["contradictingConcept"],
                            "confidence": record["otherConfidence"],
                            "severity": record["severity"],
                            "evidence": record["evidence"],
                            "detectedAt": record["detectedAt"],
                        })
                return contradictions
        except Exception as e:
            logger.error(f"Failed to find contradictions: {e}")
            return []

    def resolve_contradiction(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.resolve_contradiction_query(),
                    {
                        "userId": user_id,
                        "conceptName1": concept_name1,
                        "conceptName2": concept_name2,
                        "resolution": resolution,
                    }
                )
                for record in result:
                    logger.info(f"Resolved contradiction: {concept_name1} <-> {concept_name2}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to resolve contradiction: {e}")
            return False

    def delete_concept(
        self,
        user_id: str,
        name: str,
    ) -> bool:
        """
        Delete a concept and its relationships.

        Also deletes the vector embedding if embeddings are enabled.

        Args:
            user_id: User ID for scoping
            name: Concept name

        Returns:
            True if deleted, False otherwise
        """
        try:
            # First, get the concept ID for embedding deletion
            concept_id = None
            with self.session_factory() as session:
                id_result = session.run(
                    "MATCH (c:OM_Concept {userId: $userId, name: $name}) RETURN c.id AS id",
                    {"userId": user_id, "name": name}
                )
                for record in id_result:
                    concept_id = record["id"]

            # Delete from Neo4j
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.delete_concept_query(),
                    {"userId": user_id, "name": name}
                )
                deleted = False
                for record in result:
                    if record["deleted"] > 0:
                        deleted = True
                        logger.info(f"Deleted concept: {name}")

                # Also delete embedding if available
                if deleted and concept_id:
                    try:
                        from app.graph.concept_vector_store import (
                            get_concept_similarity_service,
                            is_concept_embeddings_enabled,
                        )

                        if is_concept_embeddings_enabled():
                            similarity_service = get_concept_similarity_service()
                            if similarity_service:
                                similarity_service.delete_concept(concept_id)
                                logger.debug(f"Deleted embedding for concept: {name}")
                    except Exception as embed_error:
                        logger.warning(f"Failed to delete embedding for {name}: {embed_error}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to delete concept {name}: {e}")
            return False

    def update_concept_confidence(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    ConceptCypherBuilder.update_concept_confidence_query(),
                    {"userId": user_id, "name": name}
                )
                for record in result:
                    return {
                        "confidence": record["newConfidence"],
                        "evidenceCount": record["evidenceCount"],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to update concept confidence: {e}")
            return None


# =============================================================================
# Factory Functions
# =============================================================================

def get_concept_projector():
    """
    Factory function to get a ConceptProjector instance.

    Returns:
        ConceptProjector instance, or None if Neo4j is not configured
    """
    try:
        from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured
        from app.config import BusinessConceptsConfig

        if not is_neo4j_configured():
            return None

        if not BusinessConceptsConfig.is_enabled():
            return None

        projector = ConceptProjector(get_neo4j_session)
        projector.ensure_constraints()
        return projector

    except Exception as e:
        logger.warning(f"Failed to create concept projector: {e}")
        return None


# Module-level projector instance (lazy initialization)
_concept_projector_instance = None


def get_projector() -> Optional[ConceptProjector]:
    """
    Get the singleton ConceptProjector instance.

    Returns:
        ConceptProjector if Neo4j and concepts are enabled, None otherwise
    """
    global _concept_projector_instance

    if _concept_projector_instance is None:
        _concept_projector_instance = get_concept_projector()

    return _concept_projector_instance


def reset_projector():
    """Reset the projector instance (for testing or config changes)."""
    global _concept_projector_instance
    _concept_projector_instance = None
