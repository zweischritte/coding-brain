"""
Convergence Detection System for Business Concept Development.

Detects when multiple independent sources point to the same insight,
enabling high-confidence concept synthesis.

Key Capabilities:
- Analyze evidence convergence for concepts
- Detect emerging consensus patterns
- Find cross-domain bridges
- Score evidence independence across temporal/domain/entity dimensions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import math

try:
    from app.graph.neo4j_client import get_neo4j_session, execute_with_retry
except ImportError:
    from openmemory.api.app.graph.neo4j_client import get_neo4j_session, execute_with_retry

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceEvidence:
    """Evidence item for convergence analysis."""

    memory_id: str
    content: str
    vault: str
    created_at: datetime
    entities: List[str]
    origin: Optional[str] = None
    confidence: float = 0.5

    def __hash__(self):
        return hash(self.memory_id)


@dataclass
class ConvergenceResult:
    """Result of convergence analysis."""

    concept_name: str
    evidence: List[ConvergenceEvidence]
    convergence_score: float
    temporal_spread_days: int
    vault_diversity: float
    source_diversity: float
    entity_path_diversity: float
    recommended_confidence: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_strong_convergence(self) -> bool:
        """
        Check if convergence meets high-confidence threshold.

        Strong convergence criteria:
        - Convergence score >= 0.7 (highly independent sources)
        - Temporal spread >= 14 days (not immediate echo)
        - Vault diversity >= 0.5 (crosses domains)
        - At least 3 pieces of evidence
        """
        return (
            self.convergence_score >= 0.7
            and self.temporal_spread_days >= 14
            and self.vault_diversity >= 0.5
            and len(self.evidence) >= 3
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "concept_name": self.concept_name,
            "evidence_count": len(self.evidence),
            "convergence_score": round(self.convergence_score, 3),
            "temporal_spread_days": self.temporal_spread_days,
            "vault_diversity": round(self.vault_diversity, 3),
            "source_diversity": round(self.source_diversity, 3),
            "entity_path_diversity": round(self.entity_path_diversity, 3),
            "recommended_confidence": round(self.recommended_confidence, 3),
            "is_strong": self.is_strong_convergence(),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
        }


class ConvergenceDetector:
    """
    Detect convergent evidence patterns in knowledge graph.

    Identifies when multiple independent sources support the same concept,
    enabling confidence boosting and cross-domain synthesis.

    Usage:
        detector = ConvergenceDetector(user_id="concepts")

        # Analyze specific concept
        result = detector.analyze_concept_convergence("BMG Revenue Model")
        if result and result.is_strong_convergence():
            print(f"Strong convergence: {result.convergence_score}")

        # Find emerging consensus
        emerging = detector.find_emerging_consensus(
            min_temporal_spread=30,
            min_vault_diversity=0.5
        )
    """

    def __init__(self, user_id: str):
        """
        Initialize convergence detector.

        Args:
            user_id: User ID for filtering concepts/memories
        """
        self.user_id = user_id

    def analyze_concept_convergence(
        self, concept_name: str, min_evidence: int = 3
    ) -> Optional[ConvergenceResult]:
        """
        Analyze convergence for a specific concept.

        Fetches supporting memories and calculates convergence metrics:
        - Temporal spread (days between earliest/latest evidence)
        - Vault diversity (how many different vaults)
        - Source diversity (how many different origins)
        - Entity path diversity (how different are the entity networks)

        Args:
            concept_name: Name of concept to analyze
            min_evidence: Minimum supporting memories required (default: 3)

        Returns:
            ConvergenceResult if sufficient evidence, None otherwise
        """
        # Fetch supporting memories with entity relationships
        cypher = """
        MATCH (concept:OM_Concept {userId: $userId, name: $conceptName})
        MATCH (concept)<-[:SUPPORTS]-(m:OM_Memory)

        OPTIONAL MATCH (m)-[:OM_ABOUT]->(e:OM_Entity)

        WITH concept, m, COLLECT(DISTINCT e.name) AS entities

        RETURN
            m.id AS memory_id,
            m.content AS content,
            m.vault AS vault,
            m.createdAt AS created_at,
            COALESCE(m.origin, 'unknown') AS origin,
            COALESCE(m.confidence, 0.5) AS confidence,
            entities
        ORDER BY m.createdAt ASC
        """

        try:
            results = execute_with_retry(
                cypher, {"userId": self.user_id, "conceptName": concept_name}
            )
        except Exception as e:
            logger.error(f"Failed to fetch concept evidence: {e}")
            return None

        if len(results) < min_evidence:
            logger.debug(
                f"Insufficient evidence for '{concept_name}': "
                f"{len(results)} < {min_evidence}"
            )
            return None

        # Build evidence list
        evidence = [
            ConvergenceEvidence(
                memory_id=r["memory_id"],
                content=r["content"],
                vault=r["vault"],
                created_at=r["created_at"],
                entities=r["entities"] or [],
                origin=r["origin"],
                confidence=r["confidence"],
            )
            for r in results
        ]

        # Calculate convergence metrics
        temporal_spread = self._calculate_temporal_spread(evidence)
        vault_diversity = self._calculate_vault_diversity(evidence)
        source_diversity = self._calculate_source_diversity(evidence)
        entity_path_diversity = self._calculate_entity_path_diversity(evidence)

        # Overall convergence score (weighted combination)
        # Weights: temporal=25%, vault=30%, entity=25%, source=20%
        convergence_score = (
            0.25 * min(1.0, temporal_spread / 90.0)
            + 0.30 * vault_diversity
            + 0.25 * entity_path_diversity
            + 0.20 * source_diversity
        )

        # Recommended confidence boost
        base_confidence = sum(e.confidence for e in evidence) / len(evidence)
        recommended_confidence = self._calculate_boosted_confidence(
            base_confidence, convergence_score, len(evidence), temporal_spread
        )

        logger.info(
            f"Convergence analysis for '{concept_name}': "
            f"score={convergence_score:.3f}, "
            f"temporal_spread={temporal_spread}d, "
            f"vault_diversity={vault_diversity:.3f}"
        )

        return ConvergenceResult(
            concept_name=concept_name,
            evidence=evidence,
            convergence_score=convergence_score,
            temporal_spread_days=temporal_spread,
            vault_diversity=vault_diversity,
            source_diversity=source_diversity,
            entity_path_diversity=entity_path_diversity,
            recommended_confidence=recommended_confidence,
        )

    def find_emerging_consensus(
        self,
        vault: Optional[str] = None,
        min_temporal_spread: int = 30,
        min_vault_diversity: float = 0.5,
        limit: int = 50,
    ) -> List[ConvergenceResult]:
        """
        Find concepts where consensus is emerging from independent sources.

        Identifies concepts with:
        - Multiple supporting memories
        - Evidence spread over time (not all at once)
        - Evidence from different vaults (cross-domain)

        Args:
            vault: Optional vault filter (e.g., 'WLT', 'FRC')
            min_temporal_spread: Minimum days between first and last evidence (default: 30)
            min_vault_diversity: Minimum vault diversity score 0.0-1.0 (default: 0.5)
            limit: Maximum results to return (default: 50)

        Returns:
            List of ConvergenceResult objects sorted by convergence score
        """
        cypher = """
        MATCH (concept:OM_Concept {userId: $userId})
        MATCH (concept)<-[:SUPPORTS]-(m:OM_Memory)

        // Filter by vault if specified
        WHERE $vault IS NULL OR concept.vault = $vault

        WITH concept,
             COLLECT({
                 id: m.id,
                 content: m.content,
                 vault: m.vault,
                 created: m.createdAt,
                 origin: COALESCE(m.origin, 'unknown'),
                 confidence: COALESCE(m.confidence, 0.5)
             }) AS memories

        WHERE SIZE(memories) >= 3

        // Calculate temporal spread
        WITH concept, memories,
             [m IN memories | m.created] AS timestamps
        WITH concept, memories,
             duration.inDays(MIN(timestamps), MAX(timestamps)).days AS temporal_spread

        WHERE temporal_spread >= $minTemporalSpread

        // Calculate vault diversity
        WITH concept, memories, temporal_spread,
             SIZE(memories) AS total,
             SIZE(COLLECT(DISTINCT [m IN memories | m.vault])) AS unique_vaults
        WITH concept, memories, temporal_spread,
             unique_vaults * 1.0 / total AS vault_diversity

        WHERE vault_diversity >= $minVaultDiversity

        RETURN
            concept.name AS concept_name,
            memories,
            temporal_spread,
            vault_diversity
        ORDER BY vault_diversity DESC, temporal_spread DESC
        LIMIT $limit
        """

        try:
            results = execute_with_retry(
                cypher,
                {
                    "userId": self.user_id,
                    "vault": vault,
                    "minTemporalSpread": min_temporal_spread,
                    "minVaultDiversity": min_vault_diversity,
                    "limit": limit,
                },
            )
        except Exception as e:
            logger.error(f"Failed to find emerging consensus: {e}")
            return []

        convergence_results = []
        for r in results:
            # Build evidence list
            evidence = [
                ConvergenceEvidence(
                    memory_id=m["id"],
                    content=m["content"],
                    vault=m["vault"],
                    created_at=m["created"],
                    entities=[],  # Not fetched in this query for performance
                    origin=m["origin"],
                    confidence=m["confidence"],
                )
                for m in r["memories"]
            ]

            # Calculate metrics
            source_diversity = self._calculate_source_diversity(evidence)
            entity_path_diversity = 0.5  # Approximation since we didn't fetch entities

            convergence_score = (
                0.25 * min(1.0, r["temporal_spread"] / 90.0)
                + 0.30 * r["vault_diversity"]
                + 0.25 * entity_path_diversity
                + 0.20 * source_diversity
            )

            base_confidence = sum(e.confidence for e in evidence) / len(evidence)
            recommended_confidence = self._calculate_boosted_confidence(
                base_confidence,
                convergence_score,
                len(evidence),
                r["temporal_spread"],
            )

            convergence_results.append(
                ConvergenceResult(
                    concept_name=r["concept_name"],
                    evidence=evidence,
                    convergence_score=convergence_score,
                    temporal_spread_days=r["temporal_spread"],
                    vault_diversity=r["vault_diversity"],
                    source_diversity=source_diversity,
                    entity_path_diversity=entity_path_diversity,
                    recommended_confidence=recommended_confidence,
                )
            )

        logger.info(
            f"Found {len(convergence_results)} concepts with emerging consensus "
            f"(vault={vault}, min_temporal_spread={min_temporal_spread}d)"
        )

        return sorted(
            convergence_results, key=lambda x: x.convergence_score, reverse=True
        )

    def detect_cross_domain_bridges(
        self, vault_a: str, vault_b: str, min_common_memories: int = 3
    ) -> List[Tuple[str, str, float, int]]:
        """
        Find concepts from different vaults with shared evidence.

        Identifies potential cross-domain synthesis opportunities using
        Adamic Adar scoring (weighted by rarity of shared connections).

        Args:
            vault_a: First vault (e.g., 'WLT' for business)
            vault_b: Second vault (e.g., 'FRC' for emotional)
            min_common_memories: Minimum shared supporting memories (default: 3)

        Returns:
            List of (concept_a, concept_b, adamic_adar_score, common_count) tuples
            sorted by adamic_adar_score descending
        """
        cypher = """
        // Find concepts from different vaults sharing supporting memories
        MATCH (c1:OM_Concept {userId: $userId, vault: $vaultA})
        MATCH (c2:OM_Concept {userId: $userId, vault: $vaultB})
        MATCH (c1)<-[:SUPPORTS]-(m:OM_Memory)-[:SUPPORTS]->(c2)

        WHERE c1 <> c2
          AND NOT EXISTS { (c1)-[:BRIDGES]-(c2) }

        WITH c1, c2, COLLECT(DISTINCT m) AS common_memories
        WHERE SIZE(common_memories) >= $minCommon

        // Calculate Adamic Adar score (weighted by memory rarity)
        UNWIND common_memories AS m
        MATCH (m)-[:SUPPORTS]->(c:OM_Concept)
        WITH c1, c2, SIZE(common_memories) AS common_count,
             m, COUNT(c) AS degree
        WITH c1, c2, common_count,
             SUM(1.0 / log(degree + 1)) AS adamic_adar

        RETURN
            c1.name AS concept_a,
            c2.name AS concept_b,
            adamic_adar,
            common_count
        ORDER BY adamic_adar DESC
        LIMIT 20
        """

        try:
            results = execute_with_retry(
                cypher,
                {
                    "userId": self.user_id,
                    "vaultA": vault_a,
                    "vaultB": vault_b,
                    "minCommon": min_common_memories,
                },
            )

            bridges = [
                (r["concept_a"], r["concept_b"], r["adamic_adar"], r["common_count"])
                for r in results
            ]

            logger.info(
                f"Found {len(bridges)} cross-domain bridges between "
                f"{vault_a} and {vault_b}"
            )

            return bridges

        except Exception as e:
            logger.error(f"Failed to detect cross-domain bridges: {e}")
            return []

    def calculate_concept_pagerank(
        self, projection_name: str = "concept-pagerank-graph", force_recreate: bool = False
    ) -> List[Tuple[str, float, int]]:
        """
        Calculate PageRank for concepts based on memory support network.

        Concepts supported by many high-quality memories (which themselves
        support other concepts) get higher PageRank scores.

        Args:
            projection_name: Name for GDS graph projection (default: 'concept-pagerank-graph')
            force_recreate: Force recreation of graph projection (default: False)

        Returns:
            List of (concept_name, pagerank_score, evidence_count) tuples
            sorted by PageRank descending
        """
        # Check if projection exists
        if not force_recreate:
            try:
                check_cypher = """
                CALL gds.graph.exists($projectionName) YIELD exists
                RETURN exists
                """
                result = execute_with_retry(
                    check_cypher, {"projectionName": projection_name}
                )
                if result and result[0]["exists"]:
                    logger.debug(f"Graph projection '{projection_name}' already exists")
                else:
                    force_recreate = True
            except Exception:
                force_recreate = True

        # Create graph projection if needed
        if force_recreate:
            try:
                # Drop existing projection if it exists
                drop_cypher = """
                CALL gds.graph.exists($projectionName) YIELD exists
                CALL apoc.when(
                    exists,
                    'CALL gds.graph.drop($projectionName) RETURN 1',
                    'RETURN 0',
                    {projectionName: $projectionName}
                ) YIELD value
                RETURN value
                """
                execute_with_retry(drop_cypher, {"projectionName": projection_name})

                # Create projection
                project_cypher = """
                CALL gds.graph.project(
                    $projectionName,
                    ['OM_Concept', 'OM_Memory'],
                    {
                        SUPPORTS: {orientation: 'UNDIRECTED'}
                    },
                    {
                        nodeProperties: ['confidence']
                    }
                )
                """
                execute_with_retry(project_cypher, {"projectionName": projection_name})
                logger.info(f"Created graph projection '{projection_name}'")

            except Exception as e:
                logger.error(f"Failed to create graph projection: {e}")
                return []

        # Run PageRank
        try:
            pagerank_cypher = """
            CALL gds.pageRank.stream($projectionName, {
                maxIterations: 20,
                dampingFactor: 0.85
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS node, score
            WHERE node:OM_Concept AND node.userId = $userId

            // Get evidence count
            MATCH (node)<-[:SUPPORTS]-(m:OM_Memory)
            WITH node, score, COUNT(m) AS evidence_count

            RETURN
                node.name AS concept_name,
                score AS pagerank,
                evidence_count
            ORDER BY pagerank DESC
            LIMIT 50
            """

            results = execute_with_retry(
                pagerank_cypher,
                {"projectionName": projection_name, "userId": self.user_id},
            )

            rankings = [
                (r["concept_name"], r["pagerank"], r["evidence_count"])
                for r in results
            ]

            logger.info(f"Calculated PageRank for {len(rankings)} concepts")

            return rankings

        except Exception as e:
            logger.error(f"Failed to calculate PageRank: {e}")
            return []

    # Private helper methods

    def _calculate_temporal_spread(self, evidence: List[ConvergenceEvidence]) -> int:
        """
        Calculate days between earliest and latest evidence.

        Returns:
            Number of days (0 if fewer than 2 evidence items)
        """
        if len(evidence) < 2:
            return 0

        timestamps = [e.created_at for e in evidence]
        delta = max(timestamps) - min(timestamps)
        return delta.days

    def _calculate_vault_diversity(self, evidence: List[ConvergenceEvidence]) -> float:
        """
        Calculate vault diversity score (0.0-1.0).

        Formula: unique_vaults / total_evidence

        Returns:
            Diversity score between 0.0 (all same vault) and 1.0 (all different vaults)
        """
        unique_vaults = len(set(e.vault for e in evidence))
        return unique_vaults / len(evidence)

    def _calculate_source_diversity(self, evidence: List[ConvergenceEvidence]) -> float:
        """
        Calculate source diversity score (0.0-1.0).

        Formula: unique_origins / total_evidence

        Returns:
            Diversity score between 0.0 and 1.0
        """
        origins = [e.origin for e in evidence if e.origin and e.origin != "unknown"]
        if not origins:
            return 0.0

        unique_origins = len(set(origins))
        return unique_origins / len(evidence)

    def _calculate_entity_path_diversity(
        self, evidence: List[ConvergenceEvidence]
    ) -> float:
        """
        Calculate entity path diversity using Jaccard distance.

        Measures how different the entity relationship chains are
        across evidence items.

        Returns:
            Average Jaccard distance (0.0 = identical entities, 1.0 = completely different)
        """
        entity_sets = [set(e.entities) for e in evidence]

        # Calculate pairwise Jaccard distances
        distances = []
        for i in range(len(entity_sets)):
            for j in range(i + 1, len(entity_sets)):
                intersection = len(entity_sets[i] & entity_sets[j])
                union = len(entity_sets[i] | entity_sets[j])
                if union > 0:
                    # Jaccard distance = 1 - Jaccard similarity
                    distances.append(1.0 - (intersection / union))

        return sum(distances) / len(distances) if distances else 0.0

    def _calculate_boosted_confidence(
        self,
        base_confidence: float,
        convergence_score: float,
        evidence_count: int,
        temporal_spread_days: int,
    ) -> float:
        """
        Calculate confidence with convergence boost.

        Formula:
        boosted = base + (convergence_score × boost_factor)

        Where boost_factor scales with:
        - Evidence count (more evidence = higher boost, caps at 10)
        - Temporal spread (longer = higher boost, caps at 90 days)
        - Convergence score (independent sources = higher boost)

        Args:
            base_confidence: Original concept confidence (0.0-1.0)
            convergence_score: Convergence score from analysis (0.0-1.0)
            evidence_count: Number of supporting memories
            temporal_spread_days: Days between first and last evidence

        Returns:
            Boosted confidence (0.0-1.0)
        """
        # Base boost from convergence (max 30%)
        convergence_boost = convergence_score * 0.3

        # Evidence count multiplier (caps at 10 memories)
        evidence_multiplier = min(1.0, evidence_count / 10.0)

        # Temporal spread multiplier (caps at 90 days)
        temporal_multiplier = min(1.0, temporal_spread_days / 90.0)

        # Combined boost
        total_boost = convergence_boost * evidence_multiplier * temporal_multiplier

        # Apply boost (cap at 1.0)
        boosted_confidence = min(1.0, base_confidence + total_boost)

        return boosted_confidence


# =============================================================================
# Contradiction Detection System
# =============================================================================


@dataclass
class ContradictionResult:
    """Result of contradiction detection between concepts."""

    concept_a: str
    concept_b: str
    similarity_score: float
    semantic_relation: str  # "contradicts", "partially_conflicts", "tension"
    severity: float  # 0.0-1.0
    evidence_a: List[str]
    evidence_b: List[str]
    detection_method: str
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "concept_a": self.concept_a,
            "concept_b": self.concept_b,
            "similarity_score": round(self.similarity_score, 3),
            "semantic_relation": self.semantic_relation,
            "severity": round(self.severity, 3),
            "evidence_a": self.evidence_a[:3],  # Limit evidence for response size
            "evidence_b": self.evidence_b[:3],
            "detection_method": self.detection_method,
            "detected_at": self.detected_at.isoformat(),
        }


class ContradictionDetector:
    """
    Detects contradictions and tensions between business concepts.

    Uses multiple detection strategies:
    1. Semantic opposition: Concepts with opposite meanings
    2. Temporal conflict: Same topic with conflicting conclusions over time
    3. Source disagreement: Same topic with conflicting sources
    4. Metric contradiction: Numerical claims that conflict

    Usage:
        detector = ContradictionDetector(user_id="concepts")

        # Detect contradictions for a specific concept
        contradictions = detector.detect_for_concept("Revenue is growing fast")

        # Find all contradictions in the system
        all_contradictions = detector.find_all_contradictions()
    """

    # Contradiction indicators for semantic detection
    CONTRADICTION_PATTERNS = [
        # Direct opposites
        ("increasing", "decreasing"),
        ("growing", "shrinking"),
        ("rising", "falling"),
        ("improving", "declining"),
        ("success", "failure"),
        ("profitable", "unprofitable"),
        ("strong", "weak"),
        ("high", "low"),
        ("fast", "slow"),
        ("good", "bad"),
        # German equivalents
        ("steigend", "sinkend"),
        ("wächst", "schrumpft"),
        ("erfolgreich", "gescheitert"),
        ("profitabel", "unprofitabel"),
        ("stark", "schwach"),
        ("hoch", "niedrig"),
        ("schnell", "langsam"),
        ("gut", "schlecht"),
        # Business-specific
        ("bullish", "bearish"),
        ("expansion", "contraction"),
        ("growth", "decline"),
        ("outperform", "underperform"),
        ("overvalued", "undervalued"),
    ]

    def __init__(self, user_id: str, similarity_threshold: float = 0.7):
        """
        Initialize contradiction detector.

        Args:
            user_id: User ID for filtering concepts
            similarity_threshold: Minimum similarity for concepts to be compared (0.0-1.0)
        """
        self.user_id = user_id
        self.similarity_threshold = similarity_threshold

    def detect_for_concept(
        self,
        concept_name: str,
        limit: int = 10,
    ) -> List[ContradictionResult]:
        """
        Detect contradictions for a specific concept.

        Finds other concepts that may contradict this one using:
        - Semantic similarity + opposition patterns
        - Shared entities with conflicting conclusions
        - Temporal contradictions (same topic, different conclusions)

        Args:
            concept_name: Name of the concept to analyze
            limit: Maximum contradictions to return

        Returns:
            List of ContradictionResult objects
        """
        contradictions = []

        # Method 1: Find semantically similar concepts with opposition patterns
        similar_contradictions = self._detect_semantic_contradictions(
            concept_name, limit
        )
        contradictions.extend(similar_contradictions)

        # Method 2: Find concepts with shared entities but opposite conclusions
        entity_contradictions = self._detect_entity_based_contradictions(
            concept_name, limit
        )
        contradictions.extend(entity_contradictions)

        # Deduplicate and sort by severity
        seen = set()
        unique_contradictions = []
        for c in contradictions:
            key = frozenset([c.concept_a, c.concept_b])
            if key not in seen:
                seen.add(key)
                unique_contradictions.append(c)

        return sorted(unique_contradictions, key=lambda x: x.severity, reverse=True)[
            :limit
        ]

    def find_all_contradictions(
        self,
        vault: Optional[str] = None,
        min_severity: float = 0.5,
        limit: int = 50,
    ) -> List[ContradictionResult]:
        """
        Find all contradictions in the concept graph.

        Args:
            vault: Optional vault filter
            min_severity: Minimum severity threshold (0.0-1.0)
            limit: Maximum results

        Returns:
            List of ContradictionResult objects
        """
        # Query existing CONTRADICTS relationships
        cypher = """
        MATCH (c1:OM_Concept {userId: $userId})-[r:CONTRADICTS]-(c2:OM_Concept {userId: $userId})
        WHERE ($vault IS NULL OR c1.vault = $vault OR c2.vault = $vault)
          AND r.severity >= $minSeverity
          AND r.resolved = false
        RETURN
            c1.name AS concept_a,
            c2.name AS concept_b,
            r.severity AS severity,
            r.evidence AS evidence,
            r.detectedAt AS detected_at
        ORDER BY r.severity DESC
        LIMIT $limit
        """

        try:
            results = execute_with_retry(
                cypher,
                {
                    "userId": self.user_id,
                    "vault": vault,
                    "minSeverity": min_severity,
                    "limit": limit,
                },
            )

            contradictions = []
            for r in results:
                contradictions.append(
                    ContradictionResult(
                        concept_a=r["concept_a"],
                        concept_b=r["concept_b"],
                        similarity_score=0.0,  # Not recalculated
                        semantic_relation="contradicts",
                        severity=r["severity"],
                        evidence_a=r["evidence"][:2] if r["evidence"] else [],
                        evidence_b=[],
                        detection_method="stored",
                        detected_at=r["detected_at"] or datetime.utcnow(),
                    )
                )

            return contradictions

        except Exception as e:
            logger.error(f"Failed to find all contradictions: {e}")
            return []

    def _detect_semantic_contradictions(
        self,
        concept_name: str,
        limit: int,
    ) -> List[ContradictionResult]:
        """
        Detect contradictions using semantic similarity and opposition patterns.

        Finds concepts that are semantically similar but contain opposing terms.
        """
        # Find concepts with high textual similarity
        cypher = """
        MATCH (target:OM_Concept {userId: $userId, name: $conceptName})
        MATCH (other:OM_Concept {userId: $userId})
        WHERE other <> target
          AND NOT EXISTS { (target)-[:CONTRADICTS]-(other) }

        // Calculate simple token overlap as similarity proxy
        WITH target, other,
             split(toLower(target.name), ' ') AS targetWords,
             split(toLower(other.name), ' ') AS otherWords

        WITH target, other,
             [w IN targetWords WHERE w IN otherWords] AS common,
             targetWords, otherWords

        WITH target, other,
             SIZE(common) * 2.0 / (SIZE(targetWords) + SIZE(otherWords)) AS similarity

        WHERE similarity > 0.2

        // Get evidence for both concepts
        OPTIONAL MATCH (m1:OM_Memory)-[:SUPPORTS]->(target)
        OPTIONAL MATCH (m2:OM_Memory)-[:SUPPORTS]->(other)

        RETURN
            target.name AS concept_a,
            other.name AS concept_b,
            similarity,
            COLLECT(DISTINCT m1.content)[0..3] AS evidence_a,
            COLLECT(DISTINCT m2.content)[0..3] AS evidence_b
        ORDER BY similarity DESC
        LIMIT $limit
        """

        try:
            results = execute_with_retry(
                cypher,
                {
                    "userId": self.user_id,
                    "conceptName": concept_name,
                    "limit": limit * 3,  # Fetch more, filter later
                },
            )

            contradictions = []
            for r in results:
                # Check for opposition patterns
                opposition = self._check_opposition_patterns(
                    r["concept_a"], r["concept_b"]
                )

                if opposition:
                    severity = min(1.0, r["similarity"] + 0.3)  # Boost for opposition
                    contradictions.append(
                        ContradictionResult(
                            concept_a=r["concept_a"],
                            concept_b=r["concept_b"],
                            similarity_score=r["similarity"],
                            semantic_relation=opposition,
                            severity=severity,
                            evidence_a=r["evidence_a"] or [],
                            evidence_b=r["evidence_b"] or [],
                            detection_method="semantic_opposition",
                        )
                    )

            return contradictions[:limit]

        except Exception as e:
            logger.error(f"Failed to detect semantic contradictions: {e}")
            return []

    def _detect_entity_based_contradictions(
        self,
        concept_name: str,
        limit: int,
    ) -> List[ContradictionResult]:
        """
        Detect contradictions based on shared entities with different conclusions.

        Finds concepts that discuss the same entities but reach opposite conclusions.
        """
        cypher = """
        MATCH (target:OM_Concept {userId: $userId, name: $conceptName})
        MATCH (target)-[:INVOLVES]->(e:OM_BizEntity)<-[:INVOLVES]-(other:OM_Concept {userId: $userId})
        WHERE other <> target
          AND NOT EXISTS { (target)-[:CONTRADICTS]-(other) }

        WITH target, other, COLLECT(DISTINCT e.name) AS shared_entities

        WHERE SIZE(shared_entities) >= 1

        // Get evidence
        OPTIONAL MATCH (m1:OM_Memory)-[:SUPPORTS]->(target)
        OPTIONAL MATCH (m2:OM_Memory)-[:SUPPORTS]->(other)

        RETURN
            target.name AS concept_a,
            other.name AS concept_b,
            shared_entities,
            SIZE(shared_entities) AS entity_overlap,
            COLLECT(DISTINCT m1.content)[0..3] AS evidence_a,
            COLLECT(DISTINCT m2.content)[0..3] AS evidence_b
        ORDER BY entity_overlap DESC
        LIMIT $limit
        """

        try:
            results = execute_with_retry(
                cypher,
                {
                    "userId": self.user_id,
                    "conceptName": concept_name,
                    "limit": limit * 2,
                },
            )

            contradictions = []
            for r in results:
                # Check for opposition patterns
                opposition = self._check_opposition_patterns(
                    r["concept_a"], r["concept_b"]
                )

                if opposition:
                    # Severity based on entity overlap
                    severity = min(1.0, 0.4 + r["entity_overlap"] * 0.15)
                    contradictions.append(
                        ContradictionResult(
                            concept_a=r["concept_a"],
                            concept_b=r["concept_b"],
                            similarity_score=r["entity_overlap"] / 5.0,
                            semantic_relation=opposition,
                            severity=severity,
                            evidence_a=r["evidence_a"] or [],
                            evidence_b=r["evidence_b"] or [],
                            detection_method="entity_based",
                        )
                    )

            return contradictions[:limit]

        except Exception as e:
            logger.error(f"Failed to detect entity-based contradictions: {e}")
            return []

    def _check_opposition_patterns(
        self,
        text_a: str,
        text_b: str,
    ) -> Optional[str]:
        """
        Check if two texts contain opposing patterns.

        Returns:
            Semantic relation type if opposition found, None otherwise
        """
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for pos, neg in self.CONTRADICTION_PATTERNS:
            if (pos in text_a_lower and neg in text_b_lower) or (
                neg in text_a_lower and pos in text_b_lower
            ):
                return "contradicts"

        # Check for "not" negation patterns
        if " not " in text_a_lower or " not " in text_b_lower:
            # Check if same core concept with negation
            words_a = set(text_a_lower.split())
            words_b = set(text_b_lower.split())
            overlap = words_a & words_b
            if len(overlap) >= 2:  # Significant overlap with negation
                return "partially_conflicts"

        return None

    def store_contradiction(
        self,
        concept_a: str,
        concept_b: str,
        severity: float,
        evidence: List[str],
    ) -> bool:
        """
        Store a detected contradiction in the graph.

        Args:
            concept_a: First concept name
            concept_b: Second concept name
            severity: Severity score (0.0-1.0)
            evidence: List of evidence strings

        Returns:
            True if stored successfully
        """
        try:
            from app.graph.concept_projector import get_projector

            projector = get_projector()
            if projector:
                return projector.create_contradiction(
                    user_id=self.user_id,
                    concept_name1=concept_a,
                    concept_name2=concept_b,
                    severity=severity,
                    evidence=evidence,
                )
            return False

        except Exception as e:
            logger.error(f"Failed to store contradiction: {e}")
            return False


# =============================================================================
# Module-level Functions
# =============================================================================


def detect_contradictions_for_concept(
    user_id: str,
    concept_name: str,
    store: bool = True,
) -> List[Dict]:
    """
    Detect and optionally store contradictions for a concept.

    Args:
        user_id: User ID for scoping
        concept_name: Name of the concept
        store: Whether to store detected contradictions in the graph

    Returns:
        List of contradiction dicts
    """
    detector = ContradictionDetector(user_id)
    contradictions = detector.detect_for_concept(concept_name)

    if store:
        for c in contradictions:
            if c.severity >= 0.5:  # Only store significant contradictions
                detector.store_contradiction(
                    concept_a=c.concept_a,
                    concept_b=c.concept_b,
                    severity=c.severity,
                    evidence=c.evidence_a[:2] + c.evidence_b[:2],
                )

    return [c.to_dict() for c in contradictions]


def find_all_contradictions(
    user_id: str,
    vault: Optional[str] = None,
    min_severity: float = 0.5,
) -> List[Dict]:
    """
    Find all stored contradictions for a user.

    Args:
        user_id: User ID for scoping
        vault: Optional vault filter
        min_severity: Minimum severity threshold

    Returns:
        List of contradiction dicts
    """
    detector = ContradictionDetector(user_id)
    contradictions = detector.find_all_contradictions(
        vault=vault,
        min_severity=min_severity,
    )
    return [c.to_dict() for c in contradictions]
