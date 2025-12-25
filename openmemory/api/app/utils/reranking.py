"""
AXIS 3.4 Memory Reranking Module

Provides metadata-based re-ranking for memory search results.
Implements boost scoring (soft ranking) and exclusion filtering (hard exclusion).

Design Principles:
- Boosts influence ranking without excluding results
- Filters provide hard exclusion for specific criteria
- Recency boost uses exponential decay with configurable halflife
- All scores are transparent and debuggable

Usage:
    from app.utils.reranking import compute_boost, should_exclude, parse_datetime
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, log1p
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.graph_cache import GraphContext


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class BoostConfig:
    """
    Immutable configuration for boost weights.

    Attributes:
        entity: Boost for matching metadata.re (highest precision)
        layer: Boost for matching metadata.layer (12 values)
        vault: Boost for matching metadata.vault (7 values)
        vector: Boost for matching metadata.vector (3 values)
        circuit: Boost for matching metadata.circuit (8 values)
        tag: Boost per matching tag
        max_tag_boost: Maximum total boost from tags
        max_recency_boost: Maximum boost from recency

        # Graph-based weights (Phase 1: Graph-Enhanced Reranking)
        entity_centrality: Boost based on entity PageRank
        similarity_cluster: Boost based on OM_SIMILAR edge count
        entity_density: Boost based on entity co-mention degree
        tag_pmi_relevance: Boost based on tag co-occurrence PMI
        max_graph_boost: Cap on total graph-based boost

        max_total_boost: Cap on total boost (prevents runaway scores)
    """
    # Existing metadata weights
    entity: float = 0.5
    layer: float = 0.3
    vault: float = 0.2
    vector: float = 0.15
    circuit: float = 0.1
    tag: float = 0.1
    max_tag_boost: float = 0.5
    max_recency_boost: float = 0.5

    # Graph-based weights
    entity_centrality: float = 0.25
    similarity_cluster: float = 0.20
    entity_density: float = 0.15
    tag_pmi_relevance: float = 0.10
    max_graph_boost: float = 0.50

    max_total_boost: float = 1.5


# Default configuration instance
DEFAULT_BOOST_CONFIG = BoostConfig()


@dataclass
class SearchContext:
    """
    Search context for boost calculation.

    All fields are optional - only provided values will be used for boosting.
    This is the "soft" context that influences ranking without excluding.
    """
    entity: Optional[str] = None
    layer: Optional[str] = None
    vault: Optional[str] = None
    vector: Optional[str] = None
    circuit: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    recency_weight: float = 0.0
    recency_halflife_days: int = 45


@dataclass
class ExclusionFilters:
    """
    Filters for hard exclusion from results.

    These are TRUE filters that exclude results, not just influence ranking.
    """
    exclude_states: List[str] = field(default_factory=lambda: ["deleted"])
    exclude_tags: List[str] = field(default_factory=list)
    boost_tags: List[str] = field(default_factory=list)  # Tags to NOT auto-exclude
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None


@dataclass
class ScoredResult:
    """
    Container for a search result with scoring breakdown.

    Provides full transparency into how the final score was calculated.
    """
    id: str
    memory: str
    semantic_score: float
    boost: float
    final_score: float
    metadata: Dict[str, Any]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def scores(self) -> Dict[str, float]:
        """Score breakdown for debugging and transparency."""
        return {
            "semantic": round(self.semantic_score, 4),
            "boost": round(self.boost, 4),
            "final": round(self.final_score, 4),
        }


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO datetime string to timezone-aware datetime.

    Handles various ISO formats including:
    - "2025-12-04T11:14:00Z" (Z suffix)
    - "2025-12-04T11:14:00+00:00" (explicit offset)
    - "2025-12-04T11:14:00" (naive - assumes UTC)

    Args:
        dt_str: ISO datetime string or None

    Returns:
        Timezone-aware datetime or None if parsing fails

    Example:
        >>> parse_datetime("2025-12-04T11:14:00Z")
        datetime(2025, 12, 4, 11, 14, tzinfo=timezone.utc)
    """
    if not dt_str:
        return None

    try:
        # Handle Z suffix (common in JSON)
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

        # Ensure timezone-aware (default to UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except (ValueError, TypeError) as e:
        # Log-friendly: return None rather than raising
        return None


def normalize_tags(tags: Any) -> Dict[str, Any]:
    """
    Normalize tags to dict format.

    Handles both list format ["tag1", "tag2"] and dict format {"tag1": True}.

    Args:
        tags: Tags in list or dict format

    Returns:
        Tags as dict with tag names as keys
    """
    if isinstance(tags, dict):
        return tags
    if isinstance(tags, list):
        return {t: True for t in tags}
    return {}


# =============================================================================
# BOOST CALCULATION
# =============================================================================

def compute_metadata_boost(
    metadata: Dict[str, Any],
    context: SearchContext,
    config: BoostConfig = DEFAULT_BOOST_CONFIG,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute boost from metadata field matches.

    Uses a DRY pattern for field matching - each match check follows
    the same structure with different field names and weights.

    Args:
        metadata: Memory metadata dict
        context: Search context with boost criteria
        config: Boost weight configuration

    Returns:
        Tuple of (total_boost, breakdown_dict) for debugging
    """
    boost = 0.0
    breakdown = {}

    # Define match checks: (context_attr, metadata_key, config_attr)
    # This DRY pattern allows easy addition of new boost fields
    string_matches = [
        ("entity", "re", "entity"),      # context.entity → metadata["re"]
        ("layer", "layer", "layer"),
        ("vault", "vault", "vault"),
        ("vector", "vector", "vector"),
    ]

    for ctx_attr, meta_key, config_attr in string_matches:
        ctx_value = getattr(context, ctx_attr)
        meta_value = metadata.get(meta_key)

        if ctx_value and meta_value and str(meta_value) == str(ctx_value):
            weight = getattr(config, config_attr)
            boost += weight
            breakdown[ctx_attr] = weight

    # Circuit match (int comparison)
    if context.circuit is not None:
        meta_circuit = metadata.get("circuit")
        if meta_circuit is not None:
            try:
                if int(meta_circuit) == int(context.circuit):
                    boost += config.circuit
                    breakdown["circuit"] = config.circuit
            except (ValueError, TypeError):
                pass  # Invalid circuit value, skip boost

    return boost, breakdown


def compute_tag_boost(
    stored_tags: Dict[str, Any],
    context_tags: List[str],
    config: BoostConfig = DEFAULT_BOOST_CONFIG,
) -> Tuple[float, int]:
    """
    Compute boost from tag matches.

    Each matching tag adds a boost, capped at max_tag_boost.

    Args:
        stored_tags: Memory's stored tags dict
        context_tags: List of tags to boost
        config: Boost weight configuration

    Returns:
        Tuple of (tag_boost, match_count)
    """
    if not context_tags:
        return 0.0, 0

    match_count = sum(1 for t in context_tags if stored_tags.get(t))
    raw_boost = match_count * config.tag
    capped_boost = min(raw_boost, config.max_tag_boost)

    return capped_boost, match_count


def compute_recency_boost(
    created_at_str: Optional[str],
    recency_weight: float,
    halflife_days: int = 45,
    config: BoostConfig = DEFAULT_BOOST_CONFIG,
) -> Tuple[float, Optional[int]]:
    """
    Compute boost based on memory age using exponential decay.

    Formula: boost = weight * e^(-age_days / decay_rate) * max_recency_boost

    At age=0: full boost
    At age=halflife: ~61% of full boost
    At age=2*halflife: ~37% of full boost

    Args:
        created_at_str: Memory creation timestamp (ISO string)
        recency_weight: 0.0 = off, 0.4 = moderate, 0.7 = strong
        halflife_days: Days until recency boost is ~61%
        config: Boost weight configuration

    Returns:
        Tuple of (recency_boost, age_days_or_none)
    """
    if recency_weight <= 0 or not created_at_str:
        return 0.0, None

    created_at = parse_datetime(created_at_str)
    if not created_at:
        return 0.0, None

    now = datetime.now(timezone.utc)
    age_days = max(0, (now - created_at).days)

    # Exponential decay formula
    # decay_rate = halflife * 2 gives halflife at ~61% (e^-0.5 ≈ 0.606)
    decay_rate = halflife_days * 2
    recency_factor = exp(-age_days / decay_rate) if decay_rate > 0 else 0

    boost = recency_weight * recency_factor * config.max_recency_boost

    return boost, age_days


def compute_graph_boost(
    memory_id: str,
    metadata: Dict[str, Any],
    graph_context: "GraphContext",
    config: BoostConfig = DEFAULT_BOOST_CONFIG,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute boost from graph signals.

    Uses pre-fetched data from graph_context to avoid per-memory queries.
    Falls back gracefully to 0.0 boost if data unavailable.

    Graph Boost Components:
    1. Entity Centrality: Higher PageRank entities = more important memories
    2. Similarity Cluster: Well-connected memories in the OM_SIMILAR graph
    3. Entity Density: Memories about highly-connected entities
    4. Tag PMI: Tags that meaningfully co-occur with query tags

    Args:
        memory_id: UUID of the memory
        metadata: Memory metadata dict
        graph_context: Pre-fetched graph signals from fetch_graph_context
        config: Boost weight configuration

    Returns:
        Tuple of (total_graph_boost, breakdown_dict)
    """
    if not graph_context.available:
        return 0.0, {}

    breakdown: Dict[str, float] = {}
    total_boost = 0.0

    # Get memory-level cache
    mem_cache = graph_context.memory_cache.get(memory_id, {})

    # 1. Entity Centrality Boost (PageRank)
    # High PageRank entities are central to the user's knowledge graph
    max_pagerank = mem_cache.get("maxEntityPageRank", 0)
    if max_pagerank > 0 and graph_context.max_pagerank > 0:
        normalized = max_pagerank / graph_context.max_pagerank
        boost = normalized * config.entity_centrality
        total_boost += boost
        breakdown["entity_centrality"] = round(boost, 4)

    # 2. Similarity Cluster Boost
    # Memories with many OM_SIMILAR edges are well-connected
    cluster_size = mem_cache.get("similarityClusterSize", 0)
    if cluster_size > 0 and graph_context.max_cluster_size > 0:
        # Log scale to handle high variance
        normalized = log1p(cluster_size) / log1p(graph_context.max_cluster_size)
        boost = normalized * config.similarity_cluster
        total_boost += boost
        breakdown["similarity_cluster"] = round(boost, 4)

    # 3. Entity Density Boost (co-mention degree)
    # Memories about entities with many connections are more relevant
    max_degree = mem_cache.get("maxEntityDegree", 0)
    if max_degree > 0 and graph_context.max_degree > 0:
        normalized = log1p(max_degree) / log1p(graph_context.max_degree)
        boost = normalized * config.entity_density
        total_boost += boost
        breakdown["entity_density"] = round(boost, 4)

    # 4. Tag PMI Relevance (if context tags provided)
    # Sum positive PMI values for memory tags that co-occur with search tags
    if graph_context.tag_pmi_cache:
        memory_tags = list(metadata.get("tags", {}).keys()) if isinstance(
            metadata.get("tags"), dict
        ) else []
        pmi_sum = 0.0
        for mem_tag in memory_tags:
            for (t1, t2), pmi in graph_context.tag_pmi_cache.items():
                if mem_tag in (t1, t2) and pmi > 0:
                    pmi_sum += pmi
        if pmi_sum > 0:
            # Normalize PMI (typical range -1 to +1, positive is meaningful)
            normalized = min(1.0, pmi_sum / 3.0)  # Cap at 3 cumulative PMI
            boost = normalized * config.tag_pmi_relevance
            total_boost += boost
            breakdown["tag_pmi_relevance"] = round(boost, 4)

    # Cap total graph boost
    capped = min(total_boost, config.max_graph_boost)
    if capped < total_boost:
        breakdown["capped"] = True

    return capped, breakdown


def compute_boost(
    metadata: Dict[str, Any],
    stored_tags: Dict[str, Any],
    context: SearchContext,
    created_at_str: Optional[str] = None,
    config: BoostConfig = DEFAULT_BOOST_CONFIG,
    graph_context: Optional["GraphContext"] = None,
    memory_id: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute total boost factor from all sources.

    Combines metadata matches, tag matches, recency boost, and optionally
    graph-based boosts into a single factor. The breakdown dict provides
    full transparency for debugging.

    Args:
        metadata: Memory metadata dict
        stored_tags: Memory tags dict (normalized)
        context: Search context with boost criteria
        created_at_str: Memory creation timestamp (ISO string)
        config: Boost weight configuration
        graph_context: Optional pre-fetched graph signals (Phase 1)
        memory_id: Optional memory ID for graph boost lookup

    Returns:
        Tuple of (total_boost, breakdown_dict)

    Example:
        >>> boost, breakdown = compute_boost(
        ...     metadata={"vault": "FRACTURE_LOG", "layer": "emotional"},
        ...     stored_tags={"trigger": True},
        ...     context=SearchContext(vault="FRACTURE_LOG", tags=["trigger"]),
        ... )
        >>> print(f"Total boost: {boost}, breakdown: {breakdown}")
        Total boost: 0.3, breakdown: {'vault': 0.2, 'tags': {'count': 1, 'boost': 0.1}}
    """
    breakdown: Dict[str, Any] = {}

    # Metadata field matches
    meta_boost, meta_breakdown = compute_metadata_boost(metadata, context, config)
    if meta_breakdown:
        breakdown["metadata"] = meta_breakdown

    # Tag matches
    tag_boost, tag_count = compute_tag_boost(stored_tags, context.tags, config)
    if tag_count > 0:
        breakdown["tags"] = {"count": tag_count, "boost": round(tag_boost, 4)}

    # Recency boost
    recency_boost, age_days = compute_recency_boost(
        created_at_str,
        context.recency_weight,
        context.recency_halflife_days,
        config,
    )
    if recency_boost > 0:
        breakdown["recency"] = {
            "age_days": age_days,
            "boost": round(recency_boost, 4),
        }

    # Graph boost (Phase 1: Graph-Enhanced Reranking)
    graph_boost = 0.0
    if graph_context is not None and memory_id is not None:
        graph_boost, graph_breakdown = compute_graph_boost(
            memory_id=memory_id,
            metadata=metadata,
            graph_context=graph_context,
            config=config,
        )
        if graph_breakdown:
            breakdown["graph"] = graph_breakdown

    total_boost = meta_boost + tag_boost + recency_boost + graph_boost
    capped_boost = min(total_boost, config.max_total_boost)

    if capped_boost < total_boost:
        breakdown["capped"] = True
        breakdown["uncapped_total"] = round(total_boost, 4)

    return capped_boost, breakdown


# =============================================================================
# EXCLUSION LOGIC
# =============================================================================

def should_exclude(
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    stored_tags: Dict[str, Any],
    filters: ExclusionFilters,
) -> Tuple[bool, Optional[str]]:
    """
    Check if memory should be excluded from results.

    Returns both the decision and the reason for debugging.

    Args:
        payload: Full memory payload from vector store
        metadata: Extracted metadata dict
        stored_tags: Memory tags dict (normalized)
        filters: Exclusion filter configuration

    Returns:
        Tuple of (should_exclude, reason_or_none)
    """
    # State exclusion
    memory_state = metadata.get("state", "active")
    if memory_state in filters.exclude_states:
        return True, f"state:{memory_state}"

    # Tag exclusion
    for tag in filters.exclude_tags:
        if stored_tags.get(tag):
            return True, f"excluded_tag:{tag}"

    # Silent tag default exclusion (unless explicitly boosted)
    if stored_tags.get("silent") and "silent" not in filters.boost_tags:
        return True, "silent_default"

    # Date range filters (hard exclusion)
    date_checks: List[Tuple[str, str, Callable[[datetime, datetime], bool]]] = [
        ("created_after", "created_at", lambda mem, filt: mem >= filt),
        ("created_before", "created_at", lambda mem, filt: mem <= filt),
        ("updated_after", "updated_at", lambda mem, filt: mem >= filt),
        ("updated_before", "updated_at", lambda mem, filt: mem <= filt),
    ]

    for filter_attr, payload_key, comparison in date_checks:
        filter_dt = getattr(filters, filter_attr)
        if filter_dt:
            mem_dt = parse_datetime(payload.get(payload_key))
            if mem_dt and not comparison(mem_dt, filter_dt):
                return True, f"date_filter:{filter_attr}"

    return False, None


# =============================================================================
# SCORING UTILITIES
# =============================================================================

def calculate_final_score(semantic_score: float, boost: float) -> float:
    """
    Calculate final score from semantic score and boost.

    Formula: final = semantic * (1 + boost)

    This multiplicative approach means:
    - boost=0: final = semantic (no change)
    - boost=0.5: final = 1.5 * semantic (50% increase)
    - boost=1.0: final = 2 * semantic (100% increase)
    - boost=1.5: final = 2.5 * semantic (150% increase, max)

    Args:
        semantic_score: Raw semantic similarity score
        boost: Computed boost factor (0.0 to MAX_TOTAL_BOOST)

    Returns:
        Final score for ranking
    """
    return semantic_score * (1 + boost)
