"""
Intelligent query routing for hybrid retrieval.

Analyzes queries to determine optimal search strategy without LLM calls.
Uses entity detection (Neo4j fulltext) and keyword pattern matching.

Routing Decision Matrix:
| Entity Count | Relationship Keywords | Route         |
|--------------|----------------------|---------------|
| 0            | No                   | VECTOR_ONLY   |
| 1            | No                   | HYBRID        |
| 1            | Yes                  | GRAPH_PRIMARY |
| 2+           | Any                  | GRAPH_PRIMARY |

Usage:
    from app.utils.query_router import analyze_query, RouteType

    analysis = analyze_query("How is Julia connected to CloudFactory?", user_id)
    if analysis.route == RouteType.GRAPH_PRIMARY:
        # Use lower alpha (0.4) for RRF to prefer graph results
        rrf_alpha = 0.4
"""

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """
    Search route strategy.

    VECTOR_ONLY: Pure vector similarity search
        - No entity mentions detected
        - Generic semantic queries
        - Best for "what are my goals", "find memories about work"

    HYBRID: Combined vector + graph with balanced weights
        - Single entity detected
        - No relationship keywords
        - Best for "memories about Julia", "CloudFactory updates"

    GRAPH_PRIMARY: Graph-preferred search (lower RRF alpha)
        - Multiple entities detected, OR
        - Relationship keywords present
        - Best for "how is Julia connected to Bob", "path between projects"
    """
    VECTOR_ONLY = "vector"
    HYBRID = "hybrid"
    GRAPH_PRIMARY = "graph"


@dataclass
class RoutingConfig:
    """
    Configuration for query routing.

    Attributes:
        enabled: Master switch for intelligent routing
        min_entity_score: Minimum fulltext match score for entity detection
        fallback_min_results: If route returns fewer results, try fallback

        relationship_keywords: Regex patterns indicating relationship queries
            - Includes English and German patterns
    """
    enabled: bool = True
    min_entity_score: float = 0.7
    fallback_min_results: int = 3

    # Relationship keywords (English + German)
    relationship_keywords: List[str] = field(default_factory=lambda: [
        r"\bconnected to\b",
        r"\brelated to\b",
        r"\bbetween\b",
        r"\bpath\b",
        r"\bnetwork\b",
        r"\brelationship\b",
        r"\bverbunden mit\b",
        r"\bbeziehung\b",
        r"\bzwischen\b",
        r"\bwho knows\b",
        r"\bwer kennt\b",
        r"\bhow does .+ relate\b",
        r"\bwie h.ngt .+ zusammen\b",
    ])


@dataclass
class QueryAnalysis:
    """
    Result of query analysis.

    Provides full transparency into routing decision.

    Attributes:
        detected_entities: List of (entity_name, score) tuples
        relationship_keywords: List of matched keyword patterns
        route: Determined route strategy
        confidence: Confidence in routing decision (0.0-1.0)
        analysis_time_ms: Time taken for analysis
    """
    detected_entities: List[Tuple[str, float]]
    relationship_keywords: List[str]
    route: RouteType
    confidence: float
    analysis_time_ms: float = 0.0


def detect_entities_in_query(
    query: str,
    user_id: str,
    min_score: float = 0.7,
    limit: int = 10,
) -> List[Tuple[str, float]]:
    """
    Detect entities in query using Neo4j fulltext search.

    This is fast (~5-20ms) because it uses a pre-built Lucene index
    on OM_Entity.name. No embedding computation required.

    Args:
        query: Search query text
        user_id: User ID for filtering
        min_score: Minimum fulltext score for entity match
        limit: Max entities to return

    Returns:
        List of (entity_name, score) tuples, sorted by score descending
    """
    from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy

    if not is_neo4j_configured():
        return []

    if not is_neo4j_healthy():
        logger.debug("Neo4j unhealthy, skipping entity detection")
        return []

    try:
        from app.graph.neo4j_client import get_neo4j_session

        with get_neo4j_session() as session:
            # Use fulltext search on OM_Entity.name
            # Requires: CREATE FULLTEXT INDEX om_entity_name_fulltext
            #           FOR (e:OM_Entity) ON EACH [e.name]
            cypher = """
            CALL db.index.fulltext.queryNodes('om_entity_name', $queryText)
            YIELD node, score
            WHERE node.userId = $userId AND score >= $minScore
            RETURN node.name AS name, score
            ORDER BY score DESC
            LIMIT $limit
            """

            result = session.run(
                cypher,
                queryText=query,
                userId=user_id,
                minScore=min_score,
                limit=limit,
            )

            return [(r["name"], r["score"]) for r in result]

    except Exception as e:
        # Fulltext index might not exist - this is expected for new setups
        if "om_entity_name_fulltext" in str(e):
            logger.debug("Entity fulltext index not found, entity detection unavailable")
        else:
            logger.warning(f"Entity detection failed: {e}")
        return []


def detect_relationship_keywords(
    query: str,
    config: RoutingConfig,
) -> List[str]:
    """
    Detect relationship-indicating keywords in query.

    Uses regex pattern matching (fast, <1ms).

    Args:
        query: Search query text
        config: Routing configuration with keyword patterns

    Returns:
        List of matched keyword patterns
    """
    matches = []
    query_lower = query.lower()

    for pattern in config.relationship_keywords:
        if re.search(pattern, query_lower, re.IGNORECASE):
            matches.append(pattern)

    return matches


def determine_route(
    entity_count: int,
    has_relationship_keywords: bool,
) -> Tuple[RouteType, float]:
    """
    Apply routing decision matrix.

    Decision Logic:
    - 0 entities, no keywords -> VECTOR_ONLY (0.9 confidence)
    - 0 entities, keywords -> HYBRID (0.6, might be generic relationship query)
    - 1 entity, no keywords -> HYBRID (0.8)
    - 1 entity, keywords -> GRAPH_PRIMARY (0.85)
    - 2+ entities -> GRAPH_PRIMARY (0.9)

    Args:
        entity_count: Number of detected entities
        has_relationship_keywords: Whether relationship keywords were found

    Returns:
        Tuple of (route, confidence)
    """
    if entity_count == 0:
        if has_relationship_keywords:
            # Keywords but no entities - might be generic relationship query
            return RouteType.HYBRID, 0.6
        return RouteType.VECTOR_ONLY, 0.9

    if entity_count == 1:
        if has_relationship_keywords:
            return RouteType.GRAPH_PRIMARY, 0.85
        return RouteType.HYBRID, 0.8

    # 2+ entities - likely asking about relationships between them
    return RouteType.GRAPH_PRIMARY, 0.9


def analyze_query(
    query: str,
    user_id: str,
    config: Optional[RoutingConfig] = None,
) -> QueryAnalysis:
    """
    Analyze query to determine optimal search route.

    Combines entity detection and keyword matching.
    Total latency: ~10-25ms (dominated by Neo4j fulltext query)

    Args:
        query: Search query text
        user_id: User ID for entity filtering
        config: Routing configuration

    Returns:
        QueryAnalysis with route decision and confidence

    Example:
        >>> analysis = analyze_query("How is Julia connected to Bob?", "user123")
        >>> print(f"Route: {analysis.route}, Entities: {analysis.detected_entities}")
        Route: RouteType.GRAPH_PRIMARY, Entities: [('julia', 0.95), ('bob', 0.92)]
    """
    start = time.perf_counter()

    config = config or RoutingConfig()

    if not config.enabled:
        # Routing disabled, default to HYBRID
        return QueryAnalysis(
            detected_entities=[],
            relationship_keywords=[],
            route=RouteType.HYBRID,
            confidence=1.0,
            analysis_time_ms=0.0,
        )

    # Detect entities (Neo4j fulltext, ~10-20ms)
    entities = detect_entities_in_query(
        query=query,
        user_id=user_id,
        min_score=config.min_entity_score,
    )

    # Detect keywords (regex, <1ms)
    keywords = detect_relationship_keywords(query, config)

    # Determine route
    route, confidence = determine_route(
        entity_count=len(entities),
        has_relationship_keywords=len(keywords) > 0,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryAnalysis(
        detected_entities=entities,
        relationship_keywords=keywords,
        route=route,
        confidence=confidence,
        analysis_time_ms=round(elapsed_ms, 2),
    )


def get_routing_config() -> RoutingConfig:
    """
    Load routing configuration from environment variables.

    Environment Variables:
        OM_ROUTING_ENABLED: "true" or "false" (default: true)
        OM_ROUTING_MIN_ENTITY_SCORE: 0.0-1.0 (default: 0.7)
        OM_ROUTING_FALLBACK_MIN_RESULTS: int (default: 3)

    Returns:
        RoutingConfig with values from environment
    """
    return RoutingConfig(
        enabled=os.getenv("OM_ROUTING_ENABLED", "true").lower() == "true",
        min_entity_score=float(os.getenv("OM_ROUTING_MIN_ENTITY_SCORE", "0.7")),
        fallback_min_results=int(os.getenv("OM_ROUTING_FALLBACK_MIN_RESULTS", "3")),
    )


def get_rrf_alpha_for_route(route: RouteType) -> float:
    """
    Get recommended RRF alpha value for a route.

    Alpha controls vector vs graph preference:
    - 0.6: Vector preference (default for HYBRID)
    - 0.4: Graph preference (for GRAPH_PRIMARY)
    - 1.0: Vector only (for VECTOR_ONLY, RRF effectively disabled)

    Args:
        route: The determined route type

    Returns:
        RRF alpha value (0.0-1.0)
    """
    if route == RouteType.VECTOR_ONLY:
        return 1.0  # Pure vector, no graph fusion
    elif route == RouteType.GRAPH_PRIMARY:
        return 0.4  # Prefer graph results
    else:  # HYBRID
        return 0.6  # Slight vector preference
