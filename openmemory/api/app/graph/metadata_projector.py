"""
Metadata Projector for OpenMemory.

Translates OpenMemory's vector-store metadata into deterministic Neo4j relations.
This creates a "metadata subgraph" that is separate from Mem0's Graph Memory
(which uses __Entity__ nodes for LLM-extracted relations).

Namespace: All labels and relationship types use OM_ prefix to avoid clashing
with Mem0's graph schema.

Metadata Keys Supported:
- category: decision, convention, architecture, dependency, workflow, testing, security, performance, runbook, glossary
- scope: session, user, team, project, org, enterprise
- artifact_type: repo, service, module, component, api, db, infra, file
- artifact_ref: string reference (path, service name, etc.)
- entity: primary entity reference
- evidence: Evidence items (string or list)
- tags: Dict with arbitrary keys and bool/int/str/list/dict values
- source: user, inference
- source_app, mcp_client: App metadata
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Max OM_SIMILAR relations per memory in response (sorted by score)
OM_SIMILAR_LIMIT = 5


def _limit_similar_relations(relations: List[Dict], limit: int = OM_SIMILAR_LIMIT) -> List[Dict]:
    """
    Limit OM_SIMILAR relations to top N by score.

    OM_SIMILAR edges can be numerous (40+). This function:
    1. Separates OM_SIMILAR from other relation types
    2. Sorts OM_SIMILAR by score descending
    3. Takes top N (default 5)
    4. Returns other relations + limited OM_SIMILAR

    Args:
        relations: List of relation dicts with 'type', 'target_value', optionally 'score'
        limit: Max OM_SIMILAR to keep (default: OM_SIMILAR_LIMIT)

    Returns:
        Filtered list with at most `limit` OM_SIMILAR relations
    """
    similar = [r for r in relations if r.get("type") == "OM_SIMILAR"]
    other = [r for r in relations if r.get("type") != "OM_SIMILAR"]

    # Sort by score descending, treat missing/None as 0
    similar_sorted = sorted(similar, key=lambda x: x.get("score") or 0, reverse=True)
    return other + similar_sorted[:limit]


DEFAULT_METADATA_RELATION_TYPES: List[str] = [
    "OM_ABOUT",
    "OM_IN_CATEGORY",
    "OM_IN_SCOPE",
    "OM_HAS_ARTIFACT_TYPE",
    "OM_REFERENCES_ARTIFACT",
    "OM_TAGGED",
    "OM_HAS_EVIDENCE",
    "OM_WRITTEN_VIA",
    "OM_RELATION",      # Entity-to-Entity typed relations (bruder_von, arbeitet_bei, etc.)
    "OM_CO_MENTIONED",  # Entity co-occurrence within same memory
]


@dataclass
class ProjectorConfig:
    """Configuration for the metadata projector."""

    # Whether to create dimension nodes (category, scope, artifact, etc.) or just store as properties
    create_dimension_nodes: bool = True

    # Whether to store tag values on relationships
    store_tag_values: bool = True

    # Maximum length for text properties (to avoid huge nodes)
    max_text_length: int = 10000


@dataclass
class MemoryMetadata:
    """Normalized metadata for a memory."""

    id: str
    user_id: str
    content: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    state: Optional[str] = None

    # Dev-assistant schema
    category: Optional[str] = None
    scope: Optional[str] = None
    artifact_type: Optional[str] = None
    artifact_ref: Optional[str] = None

    # Relations
    entity: Optional[str] = None
    evidence: List[str] = field(default_factory=list)

    # Source
    source: Optional[str] = None  # user or inference
    source_app: Optional[str] = None
    mcp_client: Optional[str] = None
    access_entity: Optional[str] = None

    # Tags
    tags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], memory_id: str, user_id: str) -> "MemoryMetadata":
        """
        Create MemoryMetadata from a raw metadata dictionary.

        Handles various edge cases:
        - Missing keys
        - evidence as string or list
        """
        metadata = data.get("metadata", data)

        # Handle evidence (can be string or list)
        evidence_raw = metadata.get("evidence")
        if evidence_raw is None:
            evidence = []
        elif isinstance(evidence_raw, str):
            evidence = [evidence_raw]
        elif isinstance(evidence_raw, list):
            evidence = [str(e) for e in evidence_raw if e]
        else:
            evidence = []

        # Handle tags (can be dict or list)
        tags_raw = metadata.get("tags")
        if isinstance(tags_raw, dict):
            tags = tags_raw
        elif isinstance(tags_raw, list):
            tags = {t: True for t in tags_raw if t}
        else:
            tags = {}

        access_entity = metadata.get("access_entity") or data.get("access_entity")
        if not access_entity and user_id:
            access_entity = f"user:{user_id}"

        return cls(
            id=memory_id,
            user_id=user_id,
            content=data.get("content") or metadata.get("data"),
            created_at=data.get("created_at") or metadata.get("created_at"),
            updated_at=data.get("updated_at") or metadata.get("updated_at"),
            state=data.get("state") or metadata.get("state"),
            category=metadata.get("category"),
            scope=metadata.get("scope"),
            artifact_type=metadata.get("artifact_type"),
            artifact_ref=metadata.get("artifact_ref"),
            entity=metadata.get("entity"),
            evidence=evidence,
            source=metadata.get("source") or metadata.get("src"),
            source_app=metadata.get("source_app"),
            mcp_client=metadata.get("mcp_client"),
            access_entity=access_entity,
            tags=tags,
        )


class CypherBuilder:
    """Builds Cypher queries for metadata projection."""

    @staticmethod
    def constraint_queries() -> List[str]:
        """
        Generate constraint creation queries.

        Returns list of Cypher statements to create constraints IF NOT EXISTS.

        Naming convention (per Neo4j best practices):
        - Node labels: CamelCase (OM_Memory, OM_Entity)
        - Relationship types: SNAKE_CASE (OM_ABOUT, OM_IN_CATEGORY)
        - Properties: camelCase (userId, createdAt, memoryIds)
        """
        return [
            # Unique constraint on OM_Memory.id
            "CREATE CONSTRAINT om_memory_id IF NOT EXISTS FOR (m:OM_Memory) REQUIRE m.id IS UNIQUE",
            # Composite unique on OM_Entity (accessEntity + name)
            "CREATE CONSTRAINT om_entity_access_name IF NOT EXISTS FOR (e:OM_Entity) REQUIRE (e.accessEntity, e.name) IS UNIQUE",
            # Unique constraints on dimension nodes
            "CREATE CONSTRAINT om_category_name IF NOT EXISTS FOR (c:OM_Category) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT om_scope_name IF NOT EXISTS FOR (s:OM_Scope) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT om_artifact_type_name IF NOT EXISTS FOR (a:OM_ArtifactType) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT om_artifact_ref_name IF NOT EXISTS FOR (a:OM_ArtifactRef) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT om_tag_key IF NOT EXISTS FOR (t:OM_Tag) REQUIRE t.key IS UNIQUE",
            "CREATE CONSTRAINT om_evidence_name IF NOT EXISTS FOR (e:OM_Evidence) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT om_app_name IF NOT EXISTS FOR (a:OM_App) REQUIRE a.name IS UNIQUE",
            # Index on accessEntity for efficient filtering (camelCase per best practices)
            "CREATE INDEX om_memory_access_entity IF NOT EXISTS FOR (m:OM_Memory) ON (m.accessEntity)",
            # Index on userId for audit queries
            "CREATE INDEX om_memory_user_id IF NOT EXISTS FOR (m:OM_Memory) ON (m.userId)",
            # Composite index for time-range queries (common pattern)
            "CREATE INDEX om_memory_access_created IF NOT EXISTS FOR (m:OM_Memory) ON (m.accessEntity, m.createdAt)",
            "CREATE INDEX om_memory_user_created IF NOT EXISTS FOR (m:OM_Memory) ON (m.userId, m.createdAt)",
            # Index on OM_Entity.accessEntity for scope lookups
            "CREATE INDEX om_entity_access_entity IF NOT EXISTS FOR (e:OM_Entity) ON (e.accessEntity)",
            # Index on OM_Entity.userId for audit queries
            "CREATE INDEX om_entity_user_id IF NOT EXISTS FOR (e:OM_Entity) ON (e.userId)",
            # Indexes for entity-to-entity edges (OM_CO_MENTIONED)
            "CREATE INDEX om_co_mentioned_access_entity IF NOT EXISTS FOR ()-[r:OM_CO_MENTIONED]-() ON (r.accessEntity)",
            "CREATE INDEX om_co_mentioned_user_id IF NOT EXISTS FOR ()-[r:OM_CO_MENTIONED]-() ON (r.userId)",
            # Indexes for tag-to-tag edges (OM_COOCCURS)
            "CREATE INDEX om_cooccurs_access_entity IF NOT EXISTS FOR ()-[r:OM_COOCCURS]-() ON (r.accessEntity)",
            "CREATE INDEX om_cooccurs_user_id IF NOT EXISTS FOR ()-[r:OM_COOCCURS]-() ON (r.userId)",
            # Indexes for memory-to-memory similarity edges (OM_SIMILAR)
            "CREATE INDEX om_similar_access_entity IF NOT EXISTS FOR ()-[r:OM_SIMILAR]-() ON (r.accessEntity)",
            "CREATE INDEX om_similar_user_id IF NOT EXISTS FOR ()-[r:OM_SIMILAR]-() ON (r.userId)",
            "CREATE INDEX om_similar_score IF NOT EXISTS FOR ()-[r:OM_SIMILAR]-() ON (r.score)",
            # Indexes for entity-to-entity typed relationship edges (OM_RELATION)
            "CREATE INDEX om_relation_access_entity IF NOT EXISTS FOR ()-[r:OM_RELATION]-() ON (r.accessEntity)",
            "CREATE INDEX om_relation_user_id IF NOT EXISTS FOR ()-[r:OM_RELATION]-() ON (r.userId)",
            "CREATE INDEX om_relation_type IF NOT EXISTS FOR ()-[r:OM_RELATION]-() ON (r.type)",
            # Index on memory state for filtering active/deleted
            "CREATE INDEX om_memory_state IF NOT EXISTS FOR (m:OM_Memory) ON (m.state)",
            # Index on category for aggregation queries
            "CREATE INDEX om_memory_category IF NOT EXISTS FOR (m:OM_Memory) ON (m.category)",
            # Index on scope for aggregation queries
            "CREATE INDEX om_memory_scope IF NOT EXISTS FOR (m:OM_Memory) ON (m.scope)",
            # Indexes for artifact filters
            "CREATE INDEX om_memory_artifact_type IF NOT EXISTS FOR (m:OM_Memory) ON (m.artifactType)",
            "CREATE INDEX om_memory_artifact_ref IF NOT EXISTS FOR (m:OM_Memory) ON (m.artifactRef)",
        ]

    @staticmethod
    def fulltext_index_queries() -> List[str]:
        """
        Generate full-text index creation queries for content search.

        Full-text indexes enable efficient text search within the graph
        without needing to go through Qdrant.
        """
        return [
            # Full-text index on memory content for text search
            "CREATE FULLTEXT INDEX om_memory_content IF NOT EXISTS FOR (m:OM_Memory) ON EACH [m.content]",
            # Full-text index on entity names for fuzzy entity search
            "CREATE FULLTEXT INDEX om_entity_name IF NOT EXISTS FOR (e:OM_Entity) ON EACH [e.name]",
        ]

    @staticmethod
    def upsert_memory_query() -> Tuple[str, List[str]]:
        """
        Generate the main memory upsert query.

        Returns:
            Tuple of (cypher_query, required_params)

        Note: Uses camelCase properties per Neo4j best practices.
        """
        query = """
        MERGE (m:OM_Memory {id: $id})
        SET m.userId = $userId,
            m.content = $content,
            m.createdAt = $createdAt,
            m.updatedAt = $updatedAt,
            m.state = $state,
            m.category = $category,
            m.scope = $scope,
            m.artifactType = $artifactType,
            m.artifactRef = $artifactRef,
            m.entity = $entity,
            m.source = $source,
            m.accessEntity = $accessEntity,
            m.projectedAt = datetime()
        RETURN m.id AS id
        """
        required_params = ["id", "userId"]
        return query, required_params

    @staticmethod
    def entity_relation_query() -> str:
        """Generate query to create entity relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        WITH m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        MERGE (e:OM_Entity {accessEntity: accessEntity, name: $entityName})
        ON CREATE SET e.userId = $userId,
                      e.createdAt = datetime(),
                      e.displayName = $entityDisplayName
        ON MATCH SET e.updatedAt = datetime(),
                     e.displayName = coalesce(e.displayName, $entityDisplayName)
        MERGE (m)-[r:OM_ABOUT]->(e)
        ON CREATE SET r.createdAt = datetime()
        ON MATCH SET r.updatedAt = datetime()
        """

    @staticmethod
    def category_relation_query() -> str:
        """Generate query to create category relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (c:OM_Category {name: $categoryName})
        MERGE (m)-[:OM_IN_CATEGORY]->(c)
        """

    @staticmethod
    def scope_relation_query() -> str:
        """Generate query to create scope relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (s:OM_Scope {name: $scopeName})
        MERGE (m)-[:OM_IN_SCOPE]->(s)
        """

    @staticmethod
    def artifact_type_relation_query() -> str:
        """Generate query to create artifact type relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (a:OM_ArtifactType {name: $artifactType})
        MERGE (m)-[:OM_HAS_ARTIFACT_TYPE]->(a)
        """

    @staticmethod
    def artifact_ref_relation_query() -> str:
        """Generate query to create artifact reference relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (a:OM_ArtifactRef {name: $artifactRef})
        MERGE (m)-[:OM_REFERENCES_ARTIFACT]->(a)
        """

    @staticmethod
    def tag_relation_query() -> str:
        """Generate query to create tag relation with value."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (t:OM_Tag {key: $tagKey})
        MERGE (m)-[r:OM_TAGGED]->(t)
        SET r.tagValue = $tagValue
        """

    @staticmethod
    def evidence_relation_query() -> str:
        """Generate query to create evidence relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (e:OM_Evidence {name: $evidenceName})
        MERGE (m)-[:OM_HAS_EVIDENCE]->(e)
        """

    @staticmethod
    def app_relation_query() -> str:
        """Generate query to create app relation."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        MERGE (a:OM_App {name: $appName})
        MERGE (m)-[:OM_WRITTEN_VIA]->(a)
        """

    @staticmethod
    def delete_memory_query() -> str:
        """Generate query to delete memory node and its relationships."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})
        DETACH DELETE m
        """

    @staticmethod
    def delete_all_user_memories_query() -> str:
        """Generate query to delete all memories for a user."""
        return """
        MATCH (m:OM_Memory {userId: $userId})
        DETACH DELETE m
        """

    # =========================================================================
    # Entity Co-Mention Edges (OM_CO_MENTIONED)
    # =========================================================================

    @staticmethod
    def update_co_mention_on_add_query() -> str:
        """
        Update OM_CO_MENTIONED edges when a memory is added.

        For each pair of entities referenced by this memory, create or update
        an edge between them with count and sample memory IDs.
        """
        return """
        MATCH (m:OM_Memory {id: $memoryId})-[:OM_ABOUT]->(e1:OM_Entity)
        MATCH (m)-[:OM_ABOUT]->(e2:OM_Entity)
        WITH m, e1, e2, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        WHERE e1.name < e2.name
          AND coalesce(e1.accessEntity, $legacyAccessEntity) = accessEntity
          AND coalesce(e2.accessEntity, $legacyAccessEntity) = accessEntity
        MERGE (e1)-[r:OM_CO_MENTIONED {accessEntity: accessEntity}]->(e2)
        ON CREATE SET r.count = 1,
                      r.memoryIds = [$memoryId],
                      r.createdAt = datetime(),
                      r.updatedAt = datetime(),
                      r.userId = $userId
        ON MATCH SET r.count = r.count + 1,
                     r.memoryIds = CASE
                       WHEN size(r.memoryIds) < 5 AND NOT $memoryId IN r.memoryIds
                       THEN r.memoryIds + $memoryId
                       ELSE r.memoryIds
                     END,
                     r.updatedAt = datetime()
        RETURN count(r) AS updated
        """

    @staticmethod
    def update_co_mention_on_delete_query() -> str:
        """
        Decrement OM_CO_MENTIONED edges when a memory is deleted.

        Removes edges with count <= 0.
        """
        return """
        MATCH (e1:OM_Entity)<-[:OM_ABOUT]-(m:OM_Memory {id: $memoryId})-[:OM_ABOUT]->(e2:OM_Entity)
        WITH e1, e2, m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        WHERE e1.name < e2.name
          AND coalesce(e1.accessEntity, $legacyAccessEntity) = accessEntity
          AND coalesce(e2.accessEntity, $legacyAccessEntity) = accessEntity
        MATCH (e1)-[r:OM_CO_MENTIONED {accessEntity: accessEntity}]->(e2)
        SET r.count = r.count - 1,
            r.memoryIds = [mid IN r.memoryIds WHERE mid <> $memoryId],
            r.updatedAt = datetime()
        WITH r
        WHERE r.count <= 0
        DELETE r
        RETURN count(r) AS deleted
        """

    @staticmethod
    def get_entity_connections_query() -> str:
        """Get all OM_CO_MENTIONED connections for an entity."""
        return """
        MATCH (e:OM_Entity {name: $entityName})
        WHERE (
          (e.accessEntity IS NOT NULL AND (
            e.accessEntity IN $accessEntities
            OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
          ))
          OR (e.accessEntity IS NULL AND e.userId = $userId)
        )
        OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
        WHERE r.count >= $minCount
          AND (
            (r.accessEntity IS NOT NULL AND r.accessEntity = e.accessEntity)
            OR (r.accessEntity IS NULL AND r.userId = $userId)
          )
          AND (
            (other.accessEntity IS NOT NULL AND other.accessEntity = e.accessEntity)
            OR (other.accessEntity IS NULL AND other.userId = $userId)
          )
        WITH e, other, r
        ORDER BY r.count DESC
        LIMIT $limit
        RETURN e.name AS entity,
               coalesce(e.displayName, e.name) AS entityDisplayName,
               collect({
                 entity: other.name,
                 displayName: coalesce(other.displayName, other.name),
                 count: r.count,
                 memoryIds: r.memoryIds
               }) AS connections
        """

    @staticmethod
    def backfill_co_mention_query() -> str:
        """
        Backfill all OM_CO_MENTIONED edges for a user.

        Creates edges between all entity pairs that appear together in memories.
        """
        return """
        MATCH (e1:OM_Entity)<-[:OM_ABOUT]-(m:OM_Memory)-[:OM_ABOUT]->(e2:OM_Entity)
        WITH e1, e2, m, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        WHERE e1.name < e2.name
          AND coalesce(e1.accessEntity, $legacyAccessEntity) = accessEntity
          AND coalesce(e2.accessEntity, $legacyAccessEntity) = accessEntity
          AND accessEntity = $accessEntity
        WITH e1, e2, accessEntity, count(m) AS cnt, collect(m.id)[0..5] AS sampleIds
        WHERE cnt >= $minCount
        MERGE (e1)-[r:OM_CO_MENTIONED {accessEntity: accessEntity}]->(e2)
        ON CREATE SET r.createdAt = datetime()
        SET r.count = cnt,
            r.memoryIds = sampleIds,
            r.updatedAt = datetime(),
            r.userId = coalesce(r.userId, $userId)
        RETURN count(r) AS edgesCreated
        """

    # =========================================================================
    # Tag Co-Occurrence Edges (OM_COOCCURS) with PMI
    # =========================================================================

    @staticmethod
    def backfill_tag_cooccurs_query() -> str:
        """
        Backfill all OM_COOCCURS edges for a user with PMI calculation.

        PMI (Pointwise Mutual Information) measures how much more likely
        two tags are to co-occur than if they were independent.
        """
        return """
        // First get total memories and tag frequencies
        MATCH (m:OM_Memory)
        WHERE coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
        WITH count(m) AS totalMemories

        // Get co-occurrences
        MATCH (m:OM_Memory)-[:OM_TAGGED]->(t1:OM_Tag)
        MATCH (m)-[:OM_TAGGED]->(t2:OM_Tag)
        WHERE coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
          AND t1.key < t2.key
        WITH totalMemories, t1, t2, count(m) AS cooccurCount
        WHERE cooccurCount >= $minCount

        // Get individual tag frequencies
        MATCH (m1:OM_Memory)-[:OM_TAGGED]->(t1)
        WHERE coalesce(m1.accessEntity, $legacyAccessEntity) = $accessEntity
        WITH totalMemories, t1, t2, cooccurCount, count(m1) AS countA
        MATCH (m2:OM_Memory)-[:OM_TAGGED]->(t2)
        WHERE coalesce(m2.accessEntity, $legacyAccessEntity) = $accessEntity
        WITH totalMemories, t1, t2, cooccurCount, countA, count(m2) AS countB

        // Calculate PMI: log2(P(a,b) / (P(a) * P(b)))
        WITH t1, t2, cooccurCount, totalMemories,
             toFloat(cooccurCount) / totalMemories AS pAb,
             toFloat(countA) / totalMemories AS pA,
             toFloat(countB) / totalMemories AS pB

        WITH t1, t2, cooccurCount,
             CASE WHEN pA * pB > 0 THEN log(pAb / (pA * pB)) / log(2) ELSE 0 END AS pmi,
             CASE WHEN pAb > 0 THEN -log(pAb) / log(2) ELSE 1 END AS hAb

        // Normalize PMI to [-1, 1]
        WITH t1, t2, cooccurCount, pmi,
             CASE WHEN hAb > 0 THEN pmi / hAb ELSE 0 END AS npmi

        WHERE npmi >= $minPmi

        MERGE (t1)-[r:OM_COOCCURS {accessEntity: $accessEntity}]->(t2)
        ON CREATE SET r.createdAt = datetime()
        SET r.count = cooccurCount,
            r.pmi = npmi,
            r.updatedAt = datetime(),
            r.userId = coalesce(r.userId, $userId)
        RETURN count(r) AS edgesCreated
        """

    @staticmethod
    def update_tag_cooccurs_for_memory_query() -> str:
        """
        Update OM_COOCCURS edges for tags in a specific memory.

        This is called after a memory is added to update tag pair edges.
        Note: This is a simplified version that doesn't recalculate PMI
        (full PMI recalculation should be done via backfill periodically).
        """
        return """
        MATCH (m:OM_Memory {id: $memoryId})-[:OM_TAGGED]->(t1:OM_Tag)
        MATCH (m)-[:OM_TAGGED]->(t2:OM_Tag)
        WITH m, t1, t2, coalesce(m.accessEntity, $legacyAccessEntity) AS accessEntity
        WHERE t1.key < t2.key
        MERGE (t1)-[r:OM_COOCCURS {accessEntity: accessEntity}]->(t2)
        ON CREATE SET r.count = 1,
                      r.pmi = 0.0,
                      r.createdAt = datetime(),
                      r.updatedAt = datetime(),
                      r.userId = $userId
        ON MATCH SET r.count = r.count + 1,
                     r.updatedAt = datetime()
        RETURN count(r) AS updated
        """

    @staticmethod
    def get_related_tags_query() -> str:
        """Get co-occurring tags for a given tag."""
        return """
        MATCH (t:OM_Tag {key: $tagKey})-[r:OM_COOCCURS]-(other:OM_Tag)
        WHERE r.count >= $minCount
          AND (
            (r.accessEntity IS NOT NULL AND (
              r.accessEntity IN $accessEntities
              OR any(prefix IN $accessEntityPrefixes WHERE r.accessEntity STARTS WITH prefix)
            ))
            OR (r.accessEntity IS NULL AND r.userId = $userId)
          )
        RETURN other.key AS tag,
               r.count AS count,
               r.pmi AS pmi
        ORDER BY r.pmi DESC, r.count DESC
        LIMIT $limit
        """

    # =========================================================================
    # Full-Text Search Queries
    # =========================================================================

    @staticmethod
    def fulltext_search_memories_query() -> str:
        """
        Full-text search across memory content.

        Uses Neo4j's full-text index for efficient text search.
        Supports Lucene query syntax (AND, OR, wildcards, fuzzy).
        """
        return """
        CALL db.index.fulltext.queryNodes('om_memory_content', $searchText)
        YIELD node, score
        WHERE ($allowedMemoryIds IS NULL OR node.id IN $allowedMemoryIds)
          AND node.state = 'active'
          AND (
            (node.accessEntity IS NOT NULL AND (
              node.accessEntity IN $accessEntities
              OR any(prefix IN $accessEntityPrefixes WHERE node.accessEntity STARTS WITH prefix)
            ))
            OR (node.accessEntity IS NULL AND node.userId = $userId)
          )
        RETURN node.id AS id,
               node.content AS content,
               node.category AS category,
               node.scope AS scope,
               node.artifactType AS artifactType,
               node.artifactRef AS artifactRef,
               node.createdAt AS createdAt,
               score AS searchScore
        ORDER BY score DESC
        LIMIT $limit
        """

    @staticmethod
    def fulltext_search_entities_query() -> str:
        """
        Full-text search across entity names.

        Useful for fuzzy entity matching and discovery.
        """
        return """
        CALL db.index.fulltext.queryNodes('om_entity_name', $searchText)
        YIELD node, score
        WHERE (
          (node.accessEntity IS NOT NULL AND (
            node.accessEntity IN $accessEntities
            OR any(prefix IN $accessEntityPrefixes WHERE node.accessEntity STARTS WITH prefix)
          ))
          OR (node.accessEntity IS NULL AND node.userId = $userId)
        )
        RETURN node.name AS name,
               coalesce(node.displayName, node.name) AS displayName,
               score AS searchScore
        ORDER BY score DESC
        LIMIT $limit
        """

    # =========================================================================
    # Graph Analytics Queries
    # =========================================================================

    @staticmethod
    def entity_centrality_query() -> str:
        """
        Calculate entity importance based on co-mention network.

        Uses degree centrality (number of co-mention connections).
        For PageRank, use Neo4j GDS library.
        """
        return """
        MATCH (e:OM_Entity)
        WHERE (
          (e.accessEntity IS NOT NULL AND (
            e.accessEntity IN $accessEntities
            OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
          ))
          OR (e.accessEntity IS NULL AND e.userId = $userId)
        )
        OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-(other:OM_Entity)
        WHERE (
          (r.accessEntity IS NOT NULL AND r.accessEntity = e.accessEntity)
          OR (r.accessEntity IS NULL AND r.userId = $userId)
        )
          AND (
            (other.accessEntity IS NOT NULL AND other.accessEntity = e.accessEntity)
            OR (other.accessEntity IS NULL AND other.userId = $userId)
          )
        WITH e, count(r) AS degree, sum(r.count) AS totalMentions
        RETURN e.name AS entity,
               coalesce(e.displayName, e.name) AS displayName,
               degree AS connections,
               totalMentions AS mentionCount
        ORDER BY totalMentions DESC, degree DESC
        LIMIT $limit
        """

    @staticmethod
    def memory_connectivity_query() -> str:
        """
        Get memory connectivity statistics.

        Shows how many shared dimensions each memory has with others.
        """
        return """
        MATCH (m:OM_Memory)
        WHERE ($allowedMemoryIds IS NULL OR m.id IN $allowedMemoryIds)
          AND (
            (m.accessEntity IS NOT NULL AND (
              m.accessEntity IN $accessEntities
              OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
            ))
            OR (m.accessEntity IS NULL AND m.userId = $userId)
          )
        OPTIONAL MATCH (m)-[r]->(:OM_Entity)
        WITH m, count(r) AS entityCount
        OPTIONAL MATCH (m)-[sim:OM_SIMILAR]->(similar:OM_Memory)
        WHERE (
          (sim.accessEntity IS NOT NULL AND sim.accessEntity = m.accessEntity)
          OR (sim.accessEntity IS NULL AND sim.userId = $userId)
        )
        WITH m, entityCount, count(similar) AS similarCount
        RETURN m.id AS id,
               m.content AS content,
               entityCount AS entities,
               similarCount AS similarMemories,
               entityCount + similarCount AS connectivity
        ORDER BY connectivity DESC
        LIMIT $limit
        """

    @staticmethod
    def clear_memory_relations_query() -> str:
        """Generate query to clear all relations from a memory (before re-projecting)."""
        return """
        MATCH (m:OM_Memory {id: $memoryId})-[r]->()
        DELETE r
        """

    @staticmethod
    def get_memory_relations_query() -> str:
        """Generate query to get all relations for given memory IDs.

        For OM_SIMILAR edges (to other OM_Memory nodes), also returns:
        - similarityScore: The r.score property from the edge
        - targetPreview: First 80 chars of target memory content
        """
        return """
        MATCH (m:OM_Memory)-[r]->(target)
        WHERE m.id IN $memoryIds
        RETURN m.id AS memoryId,
               type(r) AS relationType,
               labels(target)[0] AS targetLabel,
               CASE
                   WHEN target:OM_Memory THEN target.id
                   WHEN target:OM_Entity THEN target.name
                   WHEN target:OM_Category THEN target.name
                   WHEN target:OM_Scope THEN target.name
                   WHEN target:OM_ArtifactType THEN target.name
                   WHEN target:OM_ArtifactRef THEN target.name
                   WHEN target:OM_Tag THEN target.key
                   WHEN target:OM_Evidence THEN target.name
                   WHEN target:OM_App THEN target.name
                   ELSE null
               END AS targetValue,
               CASE
                   WHEN target:OM_Entity THEN coalesce(target.displayName, target.name)
                   ELSE null
               END AS targetDisplayName,
               r.tagValue AS relationValue,
               CASE
                   WHEN type(r) = 'OM_SIMILAR' THEN r.score
                   ELSE null
               END AS similarityScore,
               CASE
                   WHEN target:OM_Memory THEN left(target.content, 80)
                   ELSE null
               END AS targetPreview
        """


class MetadataProjector:
    """
    Projects OpenMemory metadata into Neo4j as a deterministic graph.

    This is separate from Mem0's Graph Memory which extracts entities
    from text using LLM. The metadata projector creates a 1:1 mapping
    of structured metadata to graph relations.
    """

    def __init__(self, session_factory, config: ProjectorConfig = None):
        """
        Initialize the projector.

        Args:
            session_factory: Callable that returns a Neo4j session context manager
            config: Optional projector configuration
        """
        self.session_factory = session_factory
        self.config = config or ProjectorConfig()
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
                for query in CypherBuilder.constraint_queries():
                    try:
                        session.run(query)
                    except Exception as e:
                        # Constraint may already exist with different name
                        logger.debug(f"Constraint query note: {e}")

                # Create full-text indexes for content search
                for query in CypherBuilder.fulltext_index_queries():
                    try:
                        session.run(query)
                    except Exception as e:
                        # Full-text index may already exist or not supported
                        logger.debug(f"Full-text index note: {e}")

                session.run("RETURN 1")  # Ensure transaction commits
            self._constraints_created = True
            logger.info("Neo4j constraints and indexes ensured for metadata projection")
            return True
        except Exception as e:
            logger.error(f"Failed to create Neo4j constraints: {e}")
            return False

    def _serialize_tag_value(self, value: Any) -> str:
        """Serialize tag value for storage on relationship."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _normalize_access_filters(
        self,
        user_id: str,
        access_entities: Optional[List[str]],
        access_entity_prefixes: Optional[List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Ensure access_entity filters are always populated with a safe default."""
        if not access_entities:
            access_entities = [f"user:{user_id}"] if user_id else []
        if access_entity_prefixes is None:
            access_entity_prefixes = []
        return access_entities, access_entity_prefixes

    def upsert_memory(self, metadata: MemoryMetadata) -> bool:
        """
        Upsert a memory and its metadata relations.

        Args:
            metadata: Normalized MemoryMetadata object

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                # First, clear existing relations (for clean update)
                session.run(
                    CypherBuilder.clear_memory_relations_query(),
                    {"memoryId": metadata.id}
                )

                # Upsert the memory node
                content = metadata.content
                if content and len(content) > self.config.max_text_length:
                    content = content[:self.config.max_text_length]

                session.run(
                    CypherBuilder.upsert_memory_query()[0],
                    {
                        "id": metadata.id,
                        "userId": metadata.user_id,
                        "content": content,
                        "createdAt": metadata.created_at,
                        "updatedAt": metadata.updated_at,
                        "state": metadata.state,
                        "category": metadata.category,
                        "scope": metadata.scope,
                        "artifactType": metadata.artifact_type,
                        "artifactRef": metadata.artifact_ref,
                        "entity": metadata.entity,
                        "source": metadata.source,
                        "accessEntity": metadata.access_entity,
                    }
                )

                # Create dimension relations if configured
                if self.config.create_dimension_nodes:
                    # Entity relation
                    if metadata.entity:
                        from app.graph.entity_normalizer import normalize_entity_name

                        normalized_entity = normalize_entity_name(metadata.entity)
                        if not normalized_entity:
                            logger.warning(
                                f"Skipping entity relation for memory {metadata.id}: invalid entity '{metadata.entity}'"
                            )
                            normalized_entity = None
                        if normalized_entity:
                            session.run(
                                CypherBuilder.entity_relation_query(),
                                {
                                    "memoryId": metadata.id,
                                    "userId": metadata.user_id,
                                    "entityName": normalized_entity,
                                    "entityDisplayName": metadata.entity,
                                    "legacyAccessEntity": f"user:{metadata.user_id}",
                                }
                            )

                    # Category relation
                    if metadata.category:
                        session.run(
                            CypherBuilder.category_relation_query(),
                            {"memoryId": metadata.id, "categoryName": metadata.category}
                        )

                    # Scope relation
                    if metadata.scope:
                        session.run(
                            CypherBuilder.scope_relation_query(),
                            {"memoryId": metadata.id, "scopeName": metadata.scope}
                        )

                    # Artifact type relation
                    if metadata.artifact_type:
                        session.run(
                            CypherBuilder.artifact_type_relation_query(),
                            {
                                "memoryId": metadata.id,
                                "artifactType": metadata.artifact_type,
                            }
                        )

                    # Artifact ref relation
                    if metadata.artifact_ref:
                        session.run(
                            CypherBuilder.artifact_ref_relation_query(),
                            {
                                "memoryId": metadata.id,
                                "artifactRef": metadata.artifact_ref,
                            }
                        )

                    # Evidence relations
                    for evidence in metadata.evidence:
                        if evidence:
                            session.run(
                                CypherBuilder.evidence_relation_query(),
                                {"memoryId": metadata.id, "evidenceName": evidence}
                            )

                    # App relations
                    if metadata.source_app:
                        session.run(
                            CypherBuilder.app_relation_query(),
                            {"memoryId": metadata.id, "appName": metadata.source_app}
                        )
                    if metadata.mcp_client and metadata.mcp_client != metadata.source_app:
                        session.run(
                            CypherBuilder.app_relation_query(),
                            {"memoryId": metadata.id, "appName": metadata.mcp_client}
                        )

                    # Tag relations
                    for tag_key, tag_value in metadata.tags.items():
                        if tag_key:
                            session.run(
                                CypherBuilder.tag_relation_query(),
                                {
                                    "memoryId": metadata.id,
                                    "tagKey": tag_key,
                                    "tagValue": self._serialize_tag_value(tag_value),
                                }
                            )

            logger.debug(f"Projected memory {metadata.id} to Neo4j")
            return True

        except Exception as e:
            logger.error(f"Failed to project memory {metadata.id}: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory node and its relations.

        Args:
            memory_id: UUID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                session.run(
                    CypherBuilder.delete_memory_query(),
                    {"memoryId": memory_id}
                )
            logger.debug(f"Deleted memory {memory_id} from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def delete_all_user_memories(self, user_id: str) -> bool:
        """
        Delete all memories for a user.

        Args:
            user_id: The user's string ID (e.g., "grischadallmer")

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                session.run(
                    CypherBuilder.delete_all_user_memories_query(),
                    {"userId": user_id}
                )
            logger.info(f"Deleted all memories for user {user_id} from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all memories for user {user_id}: {e}")
            return False

    def get_relations_for_memories(self, memory_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Get all metadata relations for a list of memory IDs.

        For OM_SIMILAR relations, includes score and preview fields.
        OM_SIMILAR relations are limited to top N by score (see OM_SIMILAR_LIMIT).

        Args:
            memory_ids: List of memory UUIDs

        Returns:
            Dict mapping memory_id -> list of relation dicts
        """
        if not memory_ids:
            return {}

        try:
            with self.session_factory() as session:
                result = session.run(
                    CypherBuilder.get_memory_relations_query(),
                    {"memoryIds": memory_ids}
                )

                relations_by_memory = {}
                for record in result:
                    memory_id = record["memoryId"]
                    if memory_id not in relations_by_memory:
                        relations_by_memory[memory_id] = []

                    relation = {
                        "type": record["relationType"],
                        "target_label": record["targetLabel"],
                        "target_value": record["targetValue"],
                    }
                    if record.get("targetDisplayName") is not None:
                        relation["target_display_name"] = record["targetDisplayName"]
                    if record["relationValue"] is not None:
                        relation["value"] = record["relationValue"]

                    # Add score and preview for OM_SIMILAR relations
                    if record["similarityScore"] is not None:
                        relation["score"] = round(record["similarityScore"], 2)
                    if record["targetPreview"] is not None:
                        relation["preview"] = record["targetPreview"]

                    relations_by_memory[memory_id].append(relation)

                # Apply OM_SIMILAR limit per memory
                for memory_id in relations_by_memory:
                    relations_by_memory[memory_id] = _limit_similar_relations(
                        relations_by_memory[memory_id]
                    )

                return relations_by_memory

        except Exception as e:
            logger.error(f"Failed to get relations for memories: {e}")
            return {}

    def get_memory_node(
        self,
        memory_id: str,
        user_id: str,
        allowed_memory_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single OM_Memory node's key properties.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID
            allowed_memory_ids: Optional allowlist of memory IDs (ACL)

        Returns:
            Dict of memory properties, or None if not found / not allowed
        """
        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                match = "MATCH (m:OM_Memory {id: $memoryId})"
                cypher = f"""
                {match}
                WHERE ($allowedMemoryIds IS NULL OR m.id IN $allowedMemoryIds)
                  AND (
                    (m.accessEntity IS NOT NULL AND (
                      m.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
                    ))
                    OR (m.accessEntity IS NULL AND m.userId = $userId)
                  )
                RETURN m.id AS id,
                       m.userId AS userId,
                       m.accessEntity AS accessEntity,
                       m.content AS content,
                       m.createdAt AS createdAt,
                       m.updatedAt AS updatedAt,
                       m.state AS state,
                       m.category AS category,
                       m.scope AS scope,
                       m.artifactType AS artifactType,
                       m.artifactRef AS artifactRef,
                       m.entity AS entity,
                       m.source AS source
                LIMIT 1
                """
                result = session.run(
                    cypher,
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )
                for record in result:
                    return {
                        "id": record["id"],
                        "userId": record["userId"],
                        "accessEntity": record["accessEntity"],
                        "content": record["content"],
                        "createdAt": record["createdAt"],
                        "updatedAt": record["updatedAt"],
                        "state": record["state"],
                        "category": record["category"],
                        "scope": record["scope"],
                        "artifactType": record["artifactType"],
                        "artifactRef": record["artifactRef"],
                        "entity": record["entity"],
                        "source": record["source"],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get memory node {memory_id}: {e}")
            return None

    def find_related_memories(
        self,
        memory_id: str,
        user_id: str,
        allowed_memory_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find related memories by traversing metadata dimension nodes.

        A memory is "related" if it shares a dimension node with the seed
        (e.g. same tag key, same entity, same category, etc.).

        Args:
            memory_id: Seed memory UUID
            user_id: String user ID
            allowed_memory_ids: Optional allowlist of memory IDs (ACL)
            rel_types: Optional list of relationship types to traverse
            limit: Max related memories to return

        Returns:
            List of related memory dicts, each including sharedRelations and sharedCount
        """
        rel_types = rel_types or DEFAULT_METADATA_RELATION_TYPES
        limit = max(1, min(int(limit or 10), 100))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                seed_match = "MATCH (seed:OM_Memory {id: $memoryId})"
                other_match = "MATCH (seed)-[r]->(dim)<-[r2]-(other:OM_Memory)"
                cypher = "\n".join(
                    [
                        seed_match,
                        other_match,
                        "WHERE other.id <> seed.id",
                        "  AND type(r) IN $relTypes",
                        "  AND type(r2) = type(r)",
                        "  AND ($allowedMemoryIds IS NULL OR other.id IN $allowedMemoryIds)",
                        "  AND ($allowedMemoryIds IS NULL OR seed.id IN $allowedMemoryIds)",
                        "  AND (",
                        "    (seed.accessEntity IS NOT NULL AND (",
                        "      seed.accessEntity IN $accessEntities",
                        "      OR any(prefix IN $accessEntityPrefixes WHERE seed.accessEntity STARTS WITH prefix)",
                        "    ))",
                        "    OR (seed.accessEntity IS NULL AND seed.userId = $userId)",
                        "  )",
                        "  AND (",
                        "    (other.accessEntity IS NOT NULL AND (",
                        "      other.accessEntity IN $accessEntities",
                        "      OR any(prefix IN $accessEntityPrefixes WHERE other.accessEntity STARTS WITH prefix)",
                        "    ))",
                        "    OR (other.accessEntity IS NULL AND other.userId = $userId)",
                        "  )",
                        "WITH other,",
                        "     collect(DISTINCT {",
                        "        type: type(r),",
                        "        targetLabel: labels(dim)[0],",
                        "        targetValue: CASE",
                        "            WHEN dim:OM_Entity THEN dim.name",
                        "            WHEN dim:OM_Category THEN dim.name",
                        "            WHEN dim:OM_Scope THEN dim.name",
                        "            WHEN dim:OM_ArtifactType THEN dim.name",
                        "            WHEN dim:OM_ArtifactRef THEN dim.name",
                        "            WHEN dim:OM_Tag THEN dim.key",
                        "            WHEN dim:OM_Evidence THEN dim.name",
                        "            WHEN dim:OM_App THEN dim.name",
                        "            ELSE null",
                        "        END,",
                        "        targetDisplayName: CASE",
                        "            WHEN dim:OM_Entity THEN coalesce(dim.displayName, dim.name)",
                        "            ELSE null",
                        "        END,",
                        "        seedValue: r.tagValue,",
                        "        otherValue: r2.tagValue",
                        "     }) AS sharedRelations,",
                        "     count(DISTINCT dim) AS sharedCount",
                        "RETURN other.id AS memoryId,",
                        "       other.content AS content,",
                        "       other.createdAt AS createdAt,",
                        "       other.updatedAt AS updatedAt,",
                        "       other.state AS state,",
                        "       other.category AS category,",
                        "       other.scope AS scope,",
                        "       other.artifactType AS artifactType,",
                        "       other.artifactRef AS artifactRef,",
                        "       other.entity AS entity,",
                        "       other.accessEntity AS accessEntity,",
                        "       other.source AS source,",
                        "       sharedRelations AS sharedRelations,",
                        "       sharedCount AS sharedCount",
                        "ORDER BY sharedCount DESC, other.createdAt DESC",
                        "LIMIT $limit",
                    ]
                )

                result = session.run(
                    cypher,
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "relTypes": rel_types,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )

                related: List[Dict[str, Any]] = []
                for record in result:
                    shared = record["sharedRelations"] or []
                    # Normalize relation maps: omit null values for compactness
                    normalized_shared: List[Dict[str, Any]] = []
                    for rel in shared:
                        if not rel:
                            continue
                        rel_out: Dict[str, Any] = {
                            "type": rel.get("type"),
                            "targetLabel": rel.get("targetLabel"),
                            "targetValue": rel.get("targetValue"),
                        }
                        if rel.get("targetDisplayName") is not None:
                            rel_out["targetDisplayName"] = rel.get("targetDisplayName")
                        if rel.get("seedValue") is not None:
                            rel_out["seedValue"] = rel.get("seedValue")
                        if rel.get("otherValue") is not None:
                            rel_out["otherValue"] = rel.get("otherValue")
                        normalized_shared.append(rel_out)

                    related.append(
                        {
                            "id": record["memoryId"],
                            "content": record["content"],
                            "createdAt": record["createdAt"],
                            "updatedAt": record["updatedAt"],
                            "state": record["state"],
                            "category": record["category"],
                            "scope": record["scope"],
                            "artifactType": record["artifactType"],
                            "artifactRef": record["artifactRef"],
                            "entity": record["entity"],
                            "accessEntity": record["accessEntity"],
                            "source": record["source"],
                            "sharedRelations": normalized_shared,
                            "sharedCount": record["sharedCount"],
                        }
                    )

                return related

        except Exception as e:
            logger.error(f"Failed to find related memories for {memory_id}: {e}")
            return []

    def aggregate_memories(
        self,
        user_id: str,
        group_by: str,
        allowed_memory_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate memories by a supported dimension.

        Args:
            user_id: String user ID
            group_by: One of category, scope, artifact_type, artifact_ref, tag, entity, app, evidence, source, state
            allowed_memory_ids: Optional allowlist of memory IDs (ACL)
            limit: Max buckets to return

        Returns:
            List of {"key": <str>, "count": <int>}
        """
        group_by = (group_by or "").strip().lower()
        limit = max(1, min(int(limit or 20), 200))
        access_entities, access_entity_prefixes = self._normalize_access_filters(
            user_id,
            access_entities,
            access_entity_prefixes,
        )
        memory_label = "m:OM_Memory"
        entity_label = "d:OM_Entity"

        # Cypher fragments per aggregation type
        if group_by == "category":
            match = f"MATCH ({memory_label})-[:OM_IN_CATEGORY]->(d:OM_Category)"
            key_expr = "d.name"
        elif group_by == "scope":
            match = f"MATCH ({memory_label})-[:OM_IN_SCOPE]->(d:OM_Scope)"
            key_expr = "d.name"
        elif group_by == "artifact_type":
            match = f"MATCH ({memory_label})-[:OM_HAS_ARTIFACT_TYPE]->(d:OM_ArtifactType)"
            key_expr = "d.name"
        elif group_by == "artifact_ref":
            match = f"MATCH ({memory_label})-[:OM_REFERENCES_ARTIFACT]->(d:OM_ArtifactRef)"
            key_expr = "d.name"
        elif group_by == "tag":
            match = f"MATCH ({memory_label})-[:OM_TAGGED]->(d:OM_Tag)"
            key_expr = "d.key"
        elif group_by == "entity":
            match = f"MATCH ({memory_label})-[:OM_ABOUT]->({entity_label})"
            key_expr = "d.name"
        elif group_by == "app":
            match = f"MATCH ({memory_label})-[:OM_WRITTEN_VIA]->(d:OM_App)"
            key_expr = "d.name"
        elif group_by == "evidence":
            match = f"MATCH ({memory_label})-[:OM_HAS_EVIDENCE]->(d:OM_Evidence)"
            key_expr = "d.name"
        elif group_by == "source":
            match = f"MATCH ({memory_label})"
            key_expr = "m.source"
        elif group_by == "state":
            match = f"MATCH ({memory_label})"
            key_expr = "m.state"
        else:
            raise ValueError(
                f"Unsupported group_by='{group_by}'. "
                "Use one of: category, scope, artifact_type, artifact_ref, tag, entity, app, evidence, source, state."
            )

        memory_access_filter = """
        (
          (m.accessEntity IS NOT NULL AND (
            m.accessEntity IN $accessEntities
            OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
          ))
          OR (m.accessEntity IS NULL AND m.userId = $userId)
        )
        """
        entity_access_filter = """
        (
          (d.accessEntity IS NOT NULL AND (
            d.accessEntity IN $accessEntities
            OR any(prefix IN $accessEntityPrefixes WHERE d.accessEntity STARTS WITH prefix)
          ))
          OR (d.accessEntity IS NULL AND d.userId = $userId)
        )
        """
        extra_entity_filter = f"AND {entity_access_filter}" if group_by == "entity" else ""
        display_expr = (
            "coalesce(d.displayName, d.name)"
            if group_by == "entity"
            else "null"
        )

        cypher = f"""
        {match}
        WHERE ($allowedMemoryIds IS NULL OR m.id IN $allowedMemoryIds)
          AND {memory_access_filter}
          {extra_entity_filter}
          AND {key_expr} IS NOT NULL
        RETURN {key_expr} AS key, count(DISTINCT m) AS count, {display_expr} AS displayName
        ORDER BY count DESC, key ASC
        LIMIT $limit
        """

        try:
            with self.session_factory() as session:
                result = session.run(
                    cypher,
                    {
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )
                buckets: List[Dict[str, Any]] = []
                for record in result:
                    bucket = {"key": record["key"], "count": record["count"]}
                    display_name = record.get("displayName")
                    if isinstance(display_name, str) and display_name.strip():
                        bucket["displayName"] = display_name
                    buckets.append(bucket)
                return buckets
        except Exception as e:
            logger.error(f"Failed to aggregate memories by {group_by}: {e}")
            return []

    def tag_cooccurrence(
        self,
        user_id: str,
        allowed_memory_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        limit: int = 20,
        min_count: int = 2,
        sample_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Compute co-occurrence of OM_Tag keys across memories.

        Args:
            user_id: String user ID
            allowed_memory_ids: Optional allowlist of memory IDs (ACL)
            limit: Max pairs to return
            min_count: Minimum co-occurrence count to include
            sample_size: Number of example memory IDs to attach (best-effort)

        Returns:
            List of {"tag1": str, "tag2": str, "count": int, "exampleMemoryIds": [..]}
        """
        limit = max(1, min(int(limit or 20), 200))
        min_count = max(1, int(min_count or 1))
        sample_size = max(0, min(int(sample_size or 0), 10))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                memory_match = "(m:OM_Memory)"
                cypher = f"""
                MATCH {memory_match}-[:OM_TAGGED]->(t1:OM_Tag)
                MATCH (m)-[:OM_TAGGED]->(t2:OM_Tag)
                WHERE t1.key < t2.key
                  AND ($allowedMemoryIds IS NULL OR m.id IN $allowedMemoryIds)
                  AND (
                    (m.accessEntity IS NOT NULL AND (
                      m.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
                    ))
                    OR (m.accessEntity IS NULL AND m.userId = $userId)
                  )
                WITH t1.key AS tag1, t2.key AS tag2, count(*) AS count, collect(m.id) AS memoryIds
                WHERE count >= $minCount
                RETURN tag1, tag2, count, memoryIds AS memoryIds
                ORDER BY count DESC, tag1 ASC, tag2 ASC
                LIMIT $limit
                """
                result = session.run(
                    cypher,
                    {
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "limit": limit,
                        "minCount": min_count,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )

                pairs: List[Dict[str, Any]] = []
                for record in result:
                    memory_ids = record["memoryIds"]
                    if not isinstance(memory_ids, list):
                        memory_ids = []
                    if sample_size:
                        memory_ids = memory_ids[:sample_size]
                    else:
                        memory_ids = []
                    pairs.append(
                        {
                            "tag1": record["tag1"],
                            "tag2": record["tag2"],
                            "count": record["count"],
                            "exampleMemoryIds": memory_ids,
                        }
                    )

                return pairs

        except Exception as e:
            logger.error(f"Failed to compute tag cooccurrence: {e}")
            return []

    # =========================================================================
    # Entity Co-Mention Operations
    # =========================================================================

    def update_entity_edges_on_add(self, memory_id: str, user_id: str) -> bool:
        """
        Update entity-to-entity edges when a memory is added.

        Creates or increments OM_CO_MENTIONED edges between entities
        that appear together in this memory.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                session.run(
                    CypherBuilder.update_co_mention_on_add_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )
            logger.debug(f"Updated entity edges for memory {memory_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update entity edges for memory {memory_id}: {e}")
            return False

    def update_entity_edges_on_delete(self, memory_id: str, user_id: str) -> bool:
        """
        Update entity-to-entity edges when a memory is deleted.

        Decrements OM_CO_MENTIONED edge counts and removes edges with count=0.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                session.run(
                    CypherBuilder.update_co_mention_on_delete_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )
            logger.debug(f"Updated entity edges on delete for memory {memory_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update entity edges on delete for {memory_id}: {e}")
            return False

    def get_entity_connections(
        self,
        entity_name: str,
        user_id: str,
        min_count: int = 1,
        limit: int = 50,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get all co-mentioned entities for a given entity.

        Args:
            entity_name: Name of the entity
            user_id: String user ID
            min_count: Minimum co-mention count
            limit: Maximum connections to return

        Returns:
            Dict with entity name and list of connections, or None if not found
        """
        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                result = session.run(
                    CypherBuilder.get_entity_connections_query(),
                    {
                        "entityName": entity_name,
                        "userId": user_id,
                        "minCount": max(1, int(min_count or 1)),
                        "limit": max(1, min(int(limit or 50), 200)),
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )
                for record in result:
                    # Filter out null entries from OPTIONAL MATCH
                    connections = [
                        c for c in record["connections"]
                        if c.get("entity") is not None
                    ]
                    return {
                        "entity": record["entity"],
                        "entityDisplayName": record.get("entityDisplayName") or record["entity"],
                        "connections": connections,
                        "total_connections": len(connections),
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get entity connections for {entity_name}: {e}")
            return None

    def backfill_entity_edges(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    CypherBuilder.backfill_co_mention_query(),
                    {
                        "userId": user_id,
                        "minCount": max(1, int(min_count or 1)),
                        "accessEntity": access_entity or f"user:{user_id}",
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )
                for record in result:
                    return record["edgesCreated"]
                return 0
        except Exception as e:
            logger.error(f"Failed to backfill entity edges for user {user_id}: {e}")
            return 0

    # =========================================================================
    # Tag Co-Occurrence Operations
    # =========================================================================

    def update_tag_edges_on_add(self, memory_id: str, user_id: str) -> bool:
        """
        Update tag-to-tag edges when a memory is added.

        Creates or increments OM_COOCCURS edges between tags
        that appear together in this memory.

        Args:
            memory_id: UUID of the memory
            user_id: String user ID

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_factory() as session:
                session.run(
                    CypherBuilder.update_tag_cooccurs_for_memory_query(),
                    {
                        "memoryId": memory_id,
                        "userId": user_id,
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )
            logger.debug(f"Updated tag edges for memory {memory_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to update tag edges for memory {memory_id}: {e}")
            return False

    def get_related_tags(
        self,
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
        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                result = session.run(
                    CypherBuilder.get_related_tags_query(),
                    {
                        "tagKey": tag_key,
                        "userId": user_id,
                        "minCount": max(1, int(min_count or 1)),
                        "limit": max(1, min(int(limit or 20), 100)),
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )
                tags = []
                for record in result:
                    tags.append({
                        "tag": record["tag"],
                        "count": record["count"],
                        "pmi": record["pmi"],
                    })
                return tags
        except Exception as e:
            logger.error(f"Failed to get related tags for {tag_key}: {e}")
            return []

    def backfill_tag_edges(
        self,
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
        try:
            with self.session_factory() as session:
                result = session.run(
                    CypherBuilder.backfill_tag_cooccurs_query(),
                    {
                        "userId": user_id,
                        "minCount": max(1, int(min_count or 2)),
                        "minPmi": float(min_pmi or 0.0),
                        "accessEntity": access_entity or f"user:{user_id}",
                        "legacyAccessEntity": f"user:{user_id}",
                    }
                )
                for record in result:
                    return record["edgesCreated"]
                return 0
        except Exception as e:
            logger.error(f"Failed to backfill tag edges for user {user_id}: {e}")
            return 0

    # =========================================================================
    # Path and Graph Traversal
    # =========================================================================

    def path_between_entities(
        self,
        user_id: str,
        entity_a: str,
        entity_b: str,
        allowed_memory_ids: Optional[List[str]] = None,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        max_hops: int = 6,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a shortest path between two OM_Entity nodes through the metadata subgraph.

        Important: many dimension nodes are global (no user_id). We enforce that any
        OM_Memory nodes in the path belong to the given user and pass ACL allowlists.

        Args:
            user_id: String user ID
            entity_a: Name of entity A (matches OM_Entity.name)
            entity_b: Name of entity B (matches OM_Entity.name)
            allowed_memory_ids: Optional allowlist of memory IDs (ACL)
            max_hops: Maximum relationship hops (cap for safety)

        Returns:
            Dict with nodes + relationships, or None if no path found.
        """
        max_hops = max(2, min(int(max_hops or 6), 12))
        rel_types = "|".join(DEFAULT_METADATA_RELATION_TYPES)

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                entity_a_match = "MATCH (a:OM_Entity {name: $entityA})"
                entity_b_match = "MATCH (b:OM_Entity {name: $entityB})"
                memory_acl_prefix = ""

                cypher = f"""
                {entity_a_match}
                {entity_b_match}
                MATCH p = shortestPath((a)-[:{rel_types}*..{max_hops}]-(b))
                WHERE all(n IN nodes(p) WHERE NOT n:OM_Memory OR (
                    ($allowedMemoryIds IS NULL OR n.id IN $allowedMemoryIds)
                    AND (
                      (n.accessEntity IS NOT NULL AND (
                        n.accessEntity IN $accessEntities
                        OR any(prefix IN $accessEntityPrefixes WHERE n.accessEntity STARTS WITH prefix)
                      ))
                      OR (n.accessEntity IS NULL AND n.userId = $userId)
                    )
                ))
                  AND (
                    (a.accessEntity IS NOT NULL AND (
                      a.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE a.accessEntity STARTS WITH prefix)
                    ))
                    OR (a.accessEntity IS NULL AND a.userId = $userId)
                  )
                  AND (
                    (b.accessEntity IS NOT NULL AND (
                      b.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE b.accessEntity STARTS WITH prefix)
                    ))
                    OR (b.accessEntity IS NULL AND b.userId = $userId)
                  )
                RETURN
                  [n IN nodes(p) | {{
                    label: labels(n)[0],
                    value: CASE
                      WHEN n:OM_Memory THEN n.id
                      WHEN n:OM_Entity THEN n.name
                      WHEN n:OM_Category THEN n.name
                      WHEN n:OM_Scope THEN n.name
                      WHEN n:OM_ArtifactType THEN n.name
                      WHEN n:OM_ArtifactRef THEN n.name
                      WHEN n:OM_Tag THEN n.key
                      WHEN n:OM_Evidence THEN n.name
                      WHEN n:OM_App THEN n.name
                      ELSE null
                    END,
                    displayName: CASE
                      WHEN n:OM_Entity THEN coalesce(n.displayName, n.name)
                      ELSE null
                    END,
                    memoryId: CASE WHEN n:OM_Memory THEN n.id ELSE null END,
                    content: CASE WHEN n:OM_Memory THEN n.content ELSE null END,
                    category: CASE WHEN n:OM_Memory THEN n.category ELSE null END,
                    scope: CASE WHEN n:OM_Memory THEN n.scope ELSE null END
                  }}] AS nodes,
                  [r IN relationships(p) | {{
                    type: type(r),
                    value: r.value
                  }}] AS relationships
                LIMIT 1
                """
                result = session.run(
                    cypher,
                    {
                        "userId": user_id,
                        "entityA": entity_a,
                        "entityB": entity_b,
                        "allowedMemoryIds": allowed_memory_ids,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )
                for record in result:
                    return {
                        "entity_a": entity_a,
                        "entity_b": entity_b,
                        "nodes": record["nodes"],
                        "relationships": record["relationships"],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to find path between entities '{entity_a}' and '{entity_b}': {e}")
            return None

    # =========================================================================
    # Full-Text Search Operations
    # =========================================================================

    def fulltext_search_memories(
        self,
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
        if not search_text or not search_text.strip():
            return []

        limit = max(1, min(int(limit or 20), 100))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                cypher = f"""
                CALL db.index.fulltext.queryNodes('om_memory_content', $searchText)
                YIELD node, score
                WHERE ($allowedMemoryIds IS NULL OR node.id IN $allowedMemoryIds)
                  AND node.state = 'active'
                  AND (
                    (node.accessEntity IS NOT NULL AND (
                      node.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE node.accessEntity STARTS WITH prefix)
                    ))
                    OR (node.accessEntity IS NULL AND node.userId = $userId)
                  )
                RETURN node.id AS id,
                       node.content AS content,
                       node.category AS category,
                       node.scope AS scope,
                       node.artifactType AS artifactType,
                       node.artifactRef AS artifactRef,
                       node.createdAt AS createdAt,
                       score AS searchScore
                ORDER BY score DESC
                LIMIT $limit
                """
                result = session.run(
                    cypher,
                    {
                        "searchText": search_text.strip(),
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )

                memories = []
                for record in result:
                    memories.append({
                        "id": record["id"],
                        "content": record["content"],
                        "category": record["category"],
                        "scope": record["scope"],
                        "artifactType": record["artifactType"],
                        "artifactRef": record["artifactRef"],
                        "createdAt": record["createdAt"],
                        "searchScore": record["searchScore"],
                    })
                return memories

        except Exception as e:
            # Full-text index may not exist yet
            logger.warning(f"Full-text search failed: {e}")
            return []

    def fulltext_search_entities(
        self,
        search_text: str,
        user_id: str,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across entity names.

        Useful for fuzzy entity matching and discovery.

        Args:
            search_text: Search query
            user_id: String user ID
            limit: Maximum results to return

        Returns:
            List of entity dicts with searchScore
        """
        if not search_text or not search_text.strip():
            return []

        limit = max(1, min(int(limit or 20), 100))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                result = session.run(
                    CypherBuilder.fulltext_search_entities_query(),
                    {
                        "searchText": search_text.strip(),
                        "userId": user_id,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )

                entities = []
                for record in result:
                    entities.append({
                        "name": record["name"],
                        "displayName": record.get("displayName") or record["name"],
                        "searchScore": record["searchScore"],
                    })
                return entities

        except Exception as e:
            logger.warning(f"Entity full-text search failed: {e}")
            return []

    # =========================================================================
    # Graph Analytics Operations
    # =========================================================================

    def get_entity_centrality(
        self,
        user_id: str,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get entity importance based on co-mention network.

        Uses degree centrality (number of connections) and total mention count.
        For more sophisticated PageRank, use Neo4j GDS library directly.

        Args:
            user_id: String user ID
            limit: Maximum entities to return

        Returns:
            List of entities with connections and mentionCount
        """
        limit = max(1, min(int(limit or 20), 100))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                result = session.run(
                    CypherBuilder.entity_centrality_query(),
                    {
                        "userId": user_id,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )

                entities = []
                for record in result:
                    entities.append({
                        "entity": record["entity"],
                        "displayName": record.get("displayName") or record["entity"],
                        "connections": record["connections"],
                        "mentionCount": record["mentionCount"] or 0,
                    })
                return entities

        except Exception as e:
            logger.error(f"Failed to get entity centrality: {e}")
            return []

    def get_memory_connectivity(
        self,
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
        limit = max(1, min(int(limit or 20), 100))

        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                memory_match = "MATCH (m:OM_Memory)"
                cypher = f"""
                {memory_match}
                WHERE $allowedMemoryIds IS NULL OR m.id IN $allowedMemoryIds
                  AND (
                    (m.accessEntity IS NOT NULL AND (
                      m.accessEntity IN $accessEntities
                      OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
                    ))
                    OR (m.accessEntity IS NULL AND m.userId = $userId)
                  )
                OPTIONAL MATCH (m)-[r]->(:OM_Entity)
                WITH m, count(r) AS entityCount
                OPTIONAL MATCH (m)-[sim:OM_SIMILAR]->(similar:OM_Memory)
                WHERE (
                  (sim.accessEntity IS NOT NULL AND sim.accessEntity = m.accessEntity)
                  OR (sim.accessEntity IS NULL AND sim.userId = $userId)
                )
                WITH m, entityCount, count(similar) AS similarCount
                RETURN m.id AS id,
                       m.content AS content,
                       entityCount AS entities,
                       similarCount AS similarMemories,
                       entityCount + similarCount AS connectivity
                ORDER BY connectivity DESC
                LIMIT $limit
                """
                result = session.run(
                    cypher,
                    {
                        "userId": user_id,
                        "allowedMemoryIds": allowed_memory_ids,
                        "limit": limit,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    }
                )

                memories = []
                for record in result:
                    memories.append({
                        "id": record["id"],
                        "content": record["content"],
                        "entities": record["entities"],
                        "similarMemories": record["similarMemories"],
                        "connectivity": record["connectivity"],
                    })
                return memories

        except Exception as e:
            logger.error(f"Failed to get memory connectivity: {e}")
            return []

    def get_graph_statistics(
        self,
        user_id: str,
        access_entities: Optional[List[str]] = None,
        access_entity_prefixes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics about the user's graph.

        Args:
            user_id: String user ID

        Returns:
            Dict with graph statistics
        """
        try:
            with self.session_factory() as session:
                access_entities, access_entity_prefixes = self._normalize_access_filters(
                    user_id,
                    access_entities,
                    access_entity_prefixes,
                )
                cypher = """
                MATCH (m:OM_Memory)
                WHERE (
                  (m.accessEntity IS NOT NULL AND (
                    m.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE m.accessEntity STARTS WITH prefix)
                  ))
                  OR (m.accessEntity IS NULL AND m.userId = $userId)
                )
                WITH count(m) AS memoryCount

                OPTIONAL MATCH (e:OM_Entity)
                WHERE (
                  (e.accessEntity IS NOT NULL AND (
                    e.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE e.accessEntity STARTS WITH prefix)
                  ))
                  OR (e.accessEntity IS NULL AND e.userId = $userId)
                )
                WITH memoryCount, count(e) AS entityCount

                OPTIONAL MATCH ()-[co:OM_CO_MENTIONED]->()
                WHERE (
                  (co.accessEntity IS NOT NULL AND (
                    co.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE co.accessEntity STARTS WITH prefix)
                  ))
                  OR (co.accessEntity IS NULL AND co.userId = $userId)
                )
                WITH memoryCount, entityCount, count(co) AS coMentionEdges

                OPTIONAL MATCH ()-[sim:OM_SIMILAR]->()
                WHERE (
                  (sim.accessEntity IS NOT NULL AND (
                    sim.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE sim.accessEntity STARTS WITH prefix)
                  ))
                  OR (sim.accessEntity IS NULL AND sim.userId = $userId)
                )
                WITH memoryCount, entityCount, coMentionEdges, count(sim) AS similarityEdges

                OPTIONAL MATCH ()-[tag:OM_COOCCURS]->()
                WHERE (
                  (tag.accessEntity IS NOT NULL AND (
                    tag.accessEntity IN $accessEntities
                    OR any(prefix IN $accessEntityPrefixes WHERE tag.accessEntity STARTS WITH prefix)
                  ))
                  OR (tag.accessEntity IS NULL AND tag.userId = $userId)
                )
                WITH memoryCount, entityCount, coMentionEdges, similarityEdges, count(tag) AS tagEdges

                RETURN memoryCount, entityCount, coMentionEdges, similarityEdges, tagEdges
                """
                result = session.run(
                    cypher,
                    {
                        "userId": user_id,
                        "accessEntities": access_entities,
                        "accessEntityPrefixes": access_entity_prefixes,
                    },
                )

                for record in result:
                    return {
                        "memoryCount": record["memoryCount"],
                        "entityCount": record["entityCount"],
                        "coMentionEdges": record["coMentionEdges"],
                        "similarityEdges": record["similarityEdges"],
                        "tagCooccurEdges": record["tagEdges"],
                        "totalEdges": (
                            record["coMentionEdges"] +
                            record["similarityEdges"] +
                            record["tagEdges"]
                        ),
                    }

                return {
                    "memoryCount": 0,
                    "entityCount": 0,
                    "coMentionEdges": 0,
                    "similarityEdges": 0,
                    "tagCooccurEdges": 0,
                    "totalEdges": 0,
                }

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}


def get_metadata_projector():
    """
    Factory function to get a MetadataProjector instance.

    Returns:
        MetadataProjector instance, or None if Neo4j is not configured
    """
    try:
        from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

        if not is_neo4j_configured():
            return None

        projector = MetadataProjector(get_neo4j_session)
        projector.ensure_constraints()
        return projector

    except Exception as e:
        logger.warning(f"Failed to create metadata projector: {e}")
        return None


# Module-level projector instance (lazy initialization)
_projector_instance = None


def get_projector() -> Optional[MetadataProjector]:
    """
    Get the singleton MetadataProjector instance.

    Returns:
        MetadataProjector if Neo4j is configured, None otherwise
    """
    global _projector_instance

    if _projector_instance is None:
        _projector_instance = get_metadata_projector()

    return _projector_instance


def reset_projector():
    """Reset the projector instance (for testing or config changes)."""
    global _projector_instance
    _projector_instance = None
