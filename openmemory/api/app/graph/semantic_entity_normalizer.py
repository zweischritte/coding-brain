"""
Semantic Entity Normalizer for OpenMemory.

Extends the basic case-based entity normalization with semantic
similarity detection using multiple phases:

1. String Similarity (Levenshtein/fuzzy matching)
2. Prefix/Suffix Matching
3. Domain Normalization (.community, .de, etc.)
4. Embedding Similarity (optional, API-based)

Each phase contributes to a confidence score that determines
whether entities should be merged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured, is_neo4j_healthy
from app.graph.similarity.string_similarity import find_string_similar_entities
from app.graph.similarity.prefix_matcher import find_prefix_matches
from app.graph.similarity.domain_normalizer import find_domain_matches
from app.graph.entity_normalizer import normalize_entity_name

logger = logging.getLogger(__name__)


def _normalize_access_filters(
    user_id: str,
    access_entities: Optional[List[str]],
    access_entity_prefixes: Optional[List[str]],
) -> tuple[List[str], List[str]]:
    if not access_entities:
        access_entities = [f"user:{user_id}"] if user_id else []
    if access_entity_prefixes is None:
        access_entity_prefixes = []
    return access_entities, access_entity_prefixes


def _access_filter_clause(alias: str) -> str:
    return (
        f"(({alias}.accessEntity IS NOT NULL AND ("
        f"{alias}.accessEntity IN $accessEntities "
        f"OR any(prefix IN $accessEntityPrefixes WHERE {alias}.accessEntity STARTS WITH prefix)"
        f")) OR ({alias}.accessEntity IS NULL AND {alias}.userId = $userId))"
    )


@dataclass
class MergeCandidate:
    """A merge candidate pair with confidence score."""
    entity_a: str
    entity_b: str
    confidence: float  # 0.0 - 1.0
    sources: List[str] = field(default_factory=list)  # Which phases matched


@dataclass
class SemanticCanonicalEntity:
    """A canonical entity with its semantic variants."""
    canonical: str
    variants: List[str]
    confidence: float
    merge_sources: Dict[str, List[str]]  # variant -> [sources]


class SemanticEntityNormalizer:
    """
    Hybrid Entity Normalizer with multi-phase detection.

    Combines multiple similarity methods to find entity variants
    that should be merged, then assigns confidence scores based
    on which methods matched.
    """

    # Confidence weights per phase
    PHASE_WEIGHTS = {
        "case_exact": 1.0,        # Phase 0: Exact case variants
        "string_similarity": 0.8,  # Phase 1: Levenshtein/fuzzy
        "prefix_match": 0.7,       # Phase 2: Prefix/suffix
        "domain_match": 0.9,       # Phase 3: Domain normalization
        "embedding": 0.85,         # Phase 4: Embedding similarity
    }

    # Thresholds
    STRING_SIMILARITY_THRESHOLD = 0.85
    PREFIX_MIN_LENGTH = 4
    PREFIX_MIN_OVERLAP = 0.5
    EMBEDDING_THRESHOLD = 0.90

    # Final merge threshold
    MERGE_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        enable_embeddings: bool = False,
    ):
        """
        Initialize the semantic normalizer.

        Args:
            embedding_model: Optional embedding client for phase 4
            enable_embeddings: Whether to use embedding similarity
        """
        self.embedding_model = embedding_model
        self.enable_embeddings = enable_embeddings and embedding_model is not None

    async def find_merge_candidates(
        self,
        entities: List[str],
    ) -> List[MergeCandidate]:
        """
        Find all merge candidates across all phases.

        Args:
            entities: List of all entity names for a user

        Returns:
            List of MergeCandidate with aggregated confidence
        """
        # Collect all matches with scores
        pair_scores: Dict[Tuple[str, str], MergeCandidate] = {}

        def add_match(a: str, b: str, score: float, source: str):
            """Add or update a match pair."""
            # Normalize pair order for deduplication
            pair = tuple(sorted([a, b]))
            if pair not in pair_scores:
                pair_scores[pair] = MergeCandidate(
                    entity_a=pair[0],
                    entity_b=pair[1],
                    confidence=0.0,
                    sources=[]
                )

            # Aggregate confidence (use max for robustness)
            weighted_score = score * self.PHASE_WEIGHTS.get(source, 0.5)
            pair_scores[pair].confidence = max(
                pair_scores[pair].confidence,
                weighted_score
            )
            if source not in pair_scores[pair].sources:
                pair_scores[pair].sources.append(source)

        # Phase 1: String Similarity
        logger.info(f"Phase 1: String Similarity for {len(entities)} entities")
        try:
            string_matches = find_string_similar_entities(
                entities,
                threshold=self.STRING_SIMILARITY_THRESHOLD
            )
            for match in string_matches:
                add_match(match.entity_a, match.entity_b, match.score, "string_similarity")
            logger.info(f"Phase 1: Found {len(string_matches)} string similarity matches")
        except Exception as e:
            logger.warning(f"Phase 1 failed: {e}")

        # Phase 2: Prefix/Suffix Matching
        logger.info("Phase 2: Prefix/Suffix Matching")
        try:
            prefix_matches = find_prefix_matches(
                entities,
                min_prefix_len=self.PREFIX_MIN_LENGTH,
                min_overlap_ratio=self.PREFIX_MIN_OVERLAP
            )
            for match in prefix_matches:
                # Higher confidence for higher overlap ratio
                add_match(match.shorter, match.longer, match.overlap_ratio, "prefix_match")
            logger.info(f"Phase 2: Found {len(prefix_matches)} prefix/suffix matches")
        except Exception as e:
            logger.warning(f"Phase 2 failed: {e}")

        # Phase 3: Domain Normalization
        logger.info("Phase 3: Domain Normalization")
        try:
            domain_matches = find_domain_matches(entities)
            for entity_a, entity_b, score in domain_matches:
                add_match(entity_a, entity_b, score, "domain_match")
            logger.info(f"Phase 3: Found {len(domain_matches)} domain matches")
        except Exception as e:
            logger.warning(f"Phase 3 failed: {e}")

        # Phase 4: Embedding Similarity (optional)
        if self.enable_embeddings:
            logger.info("Phase 4: Embedding Similarity")
            try:
                # Only for unmatched entities to save API costs
                matched_entities: Set[str] = set()
                for pair in pair_scores.keys():
                    matched_entities.add(pair[0])
                    matched_entities.add(pair[1])

                unmatched = [e for e in entities if e not in matched_entities]

                if unmatched and len(unmatched) > 1:
                    from app.graph.similarity.embedding_similarity import find_embedding_similar_entities

                    embedding_matches = await find_embedding_similar_entities(
                        unmatched,
                        self.embedding_model,
                        threshold=self.EMBEDDING_THRESHOLD
                    )
                    for match in embedding_matches:
                        add_match(
                            match.entity_a,
                            match.entity_b,
                            match.cosine_similarity,
                            "embedding"
                        )
                    logger.info(f"Phase 4: Found {len(embedding_matches)} embedding matches")
            except Exception as e:
                logger.warning(f"Phase 4 failed: {e}")

        # Filter by minimum confidence
        candidates = [
            c for c in pair_scores.values()
            if c.confidence >= self.MERGE_CONFIDENCE_THRESHOLD
        ]

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        logger.info(f"Total merge candidates: {len(candidates)}")
        return candidates

    def cluster_candidates(
        self,
        candidates: List[MergeCandidate],
    ) -> List[SemanticCanonicalEntity]:
        """
        Cluster merge candidates into groups with canonical entity.

        Uses Union-Find for transitive clustering.

        Args:
            candidates: List of merge candidates

        Returns:
            List of SemanticCanonicalEntity with canonical + variants
        """
        # Union-Find for clustering
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: str, y: str):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Store merge info for each pair
        merge_info: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        for candidate in candidates:
            union(candidate.entity_a, candidate.entity_b)
            merge_info[candidate.entity_a][candidate.entity_b] = candidate.sources
            merge_info[candidate.entity_b][candidate.entity_a] = candidate.sources

        # Group by root
        clusters: Dict[str, Set[str]] = defaultdict(set)
        for entity in parent.keys():
            root = find(entity)
            clusters[root].add(entity)

        # Create SemanticCanonicalEntity for each group
        result = []
        for root, members in clusters.items():
            if len(members) < 2:
                continue

            # Choose canonical entity
            canonical = self._choose_canonical(list(members))
            variants = [m for m in members if m != canonical]

            # Calculate average confidence
            relevant_candidates = [
                c for c in candidates
                if c.entity_a in members and c.entity_b in members
            ]
            if relevant_candidates:
                avg_confidence = sum(c.confidence for c in relevant_candidates) / len(relevant_candidates)
            else:
                avg_confidence = self.MERGE_CONFIDENCE_THRESHOLD

            result.append(SemanticCanonicalEntity(
                canonical=canonical,
                variants=variants,
                confidence=avg_confidence,
                merge_sources={
                    v: merge_info[canonical].get(v, merge_info[v].get(canonical, []))
                    for v in variants
                }
            ))

        return result

    def _choose_canonical(self, entities: List[str]) -> str:
        """
        Choose the best canonical form from a list of variants.

        Criteria (in order of priority):
        1. No domain suffixes (.community, .de, etc.)
        2. Shorter is better
        3. Fewer special characters
        4. Uppercase at start preferred
        """
        def score(entity: str) -> Tuple:
            has_domain = any(
                entity.lower().endswith(s)
                for s in ['.community', '.de', '.com', '.org', '.net', '.io', '-community', '_community']
            )
            special_chars = sum(1 for c in entity if c in '._-')
            starts_upper = entity[0].isupper() if entity else False

            return (
                has_domain,           # False (0) < True (1)
                len(entity),          # Shorter is better
                special_chars,        # Fewer special chars
                not starts_upper,     # Uppercase preferred
            )

        return min(entities, key=score)

    async def execute_merge(
        self,
        user_id: str,
        group: SemanticCanonicalEntity,
        allowed_memory_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        access_entity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an entity merge with ACL enforcement.

        Args:
            user_id: User ID
            group: Entity group to merge
            allowed_memory_ids: ACL - only these memories may be affected
            dry_run: If True, only simulate

        Returns:
            Merge statistics
        """
        from app.graph.entity_edge_migrator import migrate_entity_edges, EdgeMigrationStats
        from app.graph.mem0_entity_sync import sync_mem0_entities_after_normalization
        from app.graph.gds_signal_refresh import refresh_graph_signals

        result = {
            "canonical": group.canonical,
            "variants": group.variants,
            "confidence": group.confidence,
            "dry_run": dry_run,
            "edge_migration": None,
            "mem0_sync": None,
            "gds_refresh": None,
        }

        try:
            # 1. Edge Migration with ACL
            edge_stats = await migrate_entity_edges(
                user_id=user_id,
                canonical=group.canonical,
                variants=group.variants,
                allowed_memory_ids=allowed_memory_ids,
                dry_run=dry_run,
                access_entity=access_entity,
            )
            result["edge_migration"] = {
                "om_about_migrated": edge_stats.om_about_migrated,
                "om_co_mentioned_migrated": edge_stats.om_co_mentioned_migrated,
                "om_relation_migrated": edge_stats.om_relation_migrated,
                "om_temporal_migrated": edge_stats.om_temporal_migrated,
                "variant_nodes_deleted": edge_stats.variant_nodes_deleted,
                "total_migrated": edge_stats.total_migrated,
                "errors": edge_stats.errors,
            }

            # 2. Mem0 Graph Sync (if enabled)
            if not dry_run:
                mem0_stats = await sync_mem0_entities_after_normalization(
                    user_id=user_id,
                    canonical=group.canonical,
                    variants=group.variants,
                    dry_run=dry_run,
                )
                result["mem0_sync"] = mem0_stats

                # 3. GDS Signal Updates
                gds_stats = await refresh_graph_signals(
                    user_id=user_id,
                    canonical=group.canonical,
                    access_entities=[access_entity] if access_entity else None,
                )
                result["gds_refresh"] = gds_stats

        except Exception as e:
            logger.exception(f"Error executing merge: {e}")
            result["error"] = str(e)

        return result


async def get_all_user_entities(
    user_id: str,
    access_entities: Optional[List[str]] = None,
    access_entity_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Collect ALL entity names for a user from all sources.

    Sources:
    1. OM_Entity nodes (main storage)
    2. metadata.re in OM_Memory (legacy, single-entity)
    3. OM_ABOUT edges (multi-entity via Entity Bridge)

    Returns:
        Deduplicated list of all entity names
    """
    if not is_neo4j_configured():
        return []

    if not is_neo4j_healthy():
        logger.warning("Neo4j unhealthy, returning empty entity list")
        return []

    entities: Set[str] = set()
    access_entities, access_entity_prefixes = _normalize_access_filters(
        user_id,
        access_entities,
        access_entity_prefixes,
    )

    try:
        with get_neo4j_session() as session:
            # 1. From OM_Entity nodes
            query1 = f"""
            MATCH (e:OM_Entity)
            WHERE {_access_filter_clause("e")}
            RETURN DISTINCT e.name AS name
            """
            result1 = session.run(
                query1,
                userId=user_id,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
            )
            for record in result1:
                if record["name"]:
                    entities.add(record["name"])

            # 2. From metadata.re (single-entity legacy field)
            query2 = f"""
            MATCH (m:OM_Memory)
            WHERE {_access_filter_clause("m")}
              AND m.metadata_re IS NOT NULL AND m.metadata_re <> ''
            RETURN DISTINCT m.metadata_re AS name
            """
            result2 = session.run(
                query2,
                userId=user_id,
                accessEntities=access_entities,
                accessEntityPrefixes=access_entity_prefixes,
            )
            for record in result2:
                if record["name"]:
                    entities.add(record["name"])

            # Note: OM_ABOUT targets are already in query 1 since they point to OM_Entity

        logger.info(f"Found {len(entities)} unique entities for user {user_id}")

    except Exception as e:
        logger.exception(f"Error collecting entities: {e}")

    return list(entities)


def with_neo4j_fallback(fallback_value):
    """Decorator for graceful degradation when Neo4j is unavailable."""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_neo4j_healthy():
                logger.warning(
                    f"Neo4j unhealthy. Skipping {func.__name__}, "
                    f"returning fallback value."
                )
                return fallback_value

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator
