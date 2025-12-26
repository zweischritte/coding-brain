"""Tri-hybrid retrieval (lexical + semantic + graph).

This module provides tri-hybrid retrieval combining:
- Lexical search (BM25) from OpenSearch
- Semantic/Vector search (kNN) from OpenSearch
- Graph context (CODE_* relationships) from Neo4j

Key features:
- RRF and weighted fusion strategies
- Graph context fetching with depth control
- Score normalization across retrieval types
- Graceful fallback when graph unavailable
- Sub-100ms p95 latency target
"""

from __future__ import annotations

import logging
import statistics
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openmemory.api.retrieval import (
    LexicalSearchQuery,
    VectorSearchQuery,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RetrievalError(Exception):
    """Base exception for retrieval errors."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TriHybridConfig:
    """Configuration for tri-hybrid retrieval.

    Default weights from v9 plan:
    - vector: 0.40
    - lexical: 0.35
    - graph: 0.25
    """

    vector_weight: float = 0.40
    lexical_weight: float = 0.35
    graph_weight: float = 0.25

    rrf_rank_constant: int = 60
    rrf_window_size: int = 100

    graph_depth: int = 2
    graph_edge_types: list[str] = field(
        default_factory=lambda: ["CALLS", "IMPORTS", "CONTAINS", "DEFINES"]
    )

    @property
    def total_weight(self) -> float:
        """Calculate total weight."""
        return self.vector_weight + self.lexical_weight + self.graph_weight

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = self.total_weight
        if total > 0:
            self.vector_weight /= total
            self.lexical_weight /= total
            self.graph_weight /= total


# =============================================================================
# Query
# =============================================================================


@dataclass
class TriHybridQuery:
    """Query for tri-hybrid retrieval.

    Args:
        query_text: Text query for lexical search
        embedding: Vector embedding for semantic search
        seed_symbols: Initial symbols for graph expansion
        filters: Metadata filters to apply
        size: Number of results to return
        offset: Pagination offset
    """

    query_text: str = ""
    embedding: list[float] = field(default_factory=list)
    seed_symbols: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    size: int = 10
    offset: int = 0

    def __post_init__(self):
        """Validate query has at least text or embedding."""
        if not self.query_text and not self.embedding:
            raise ValueError("Query must have at least text or embedding")


# =============================================================================
# Graph Context
# =============================================================================


@dataclass
class GraphContext:
    """Context fetched from the code graph."""

    neighbor_ids: set[str] = field(default_factory=set)
    edges: list[dict[str, Any]] = field(default_factory=list)
    node_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class GraphContextFetcher:
    """Fetches graph context from Neo4j."""

    def __init__(self, driver: Any):
        """Initialize with Neo4j driver.

        Args:
            driver: Neo4j driver implementing get_outgoing_edges and get_node
        """
        self._driver = driver

    def fetch_context(
        self,
        symbol_ids: list[str],
        depth: int = 2,
        edge_types: Optional[list[str]] = None,
        max_neighbors: int = 100,
    ) -> GraphContext:
        """Fetch graph context for given symbols.

        Args:
            symbol_ids: Starting symbol IDs
            depth: How many hops to traverse
            edge_types: Optional filter for edge types
            max_neighbors: Maximum neighbors to return

        Returns:
            GraphContext with neighbors, edges, and properties
        """
        context = GraphContext()
        visited: set[str] = set()
        frontier = set(symbol_ids)

        for _ in range(depth):
            if not frontier:
                break

            next_frontier: set[str] = set()

            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                try:
                    edges = self._driver.get_outgoing_edges(node_id)

                    for edge in edges:
                        edge_type_value = (
                            edge.edge_type.value
                            if hasattr(edge.edge_type, "value")
                            else str(edge.edge_type)
                        )

                        # Filter by edge type if specified
                        if edge_types and edge_type_value not in edge_types:
                            continue

                        target_id = edge.target_id

                        # Add to context
                        context.neighbor_ids.add(target_id)
                        context.edges.append(
                            {
                                "source": node_id,
                                "target": target_id,
                                "type": edge_type_value,
                            }
                        )

                        # Get node properties
                        if target_id not in context.node_properties:
                            node = self._driver.get_node(target_id)
                            if node:
                                context.node_properties[target_id] = node.properties

                        # Add to next frontier if not visited
                        if target_id not in visited:
                            next_frontier.add(target_id)

                        # Check limit
                        if len(context.neighbor_ids) >= max_neighbors:
                            return context

                except Exception as e:
                    logger.warning(f"Error fetching edges for {node_id}: {e}")
                    context.errors.append(str(e))
                    continue

            frontier = next_frontier

        return context


# =============================================================================
# Score Normalization
# =============================================================================


class ScoreNormalizer:
    """Normalizes scores from different sources."""

    def __init__(self, method: str = "min_max"):
        """Initialize normalizer.

        Args:
            method: Normalization method ('min_max' or 'z_score')
        """
        self.method = method

    def normalize(self, scores: list[float]) -> list[float]:
        """Normalize a list of scores.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        if self.method == "min_max":
            return self._min_max_normalize(scores)
        elif self.method == "z_score":
            return self._z_score_normalize(scores)
        else:
            return scores

    def _min_max_normalize(self, scores: list[float]) -> list[float]:
        """Min-max normalization to 0-1 range."""
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _z_score_normalize(self, scores: list[float]) -> list[float]:
        """Z-score normalization."""
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores) if len(scores) > 1 else 1.0

        if stdev == 0:
            return [0.0] * len(scores)

        return [(s - mean) / stdev for s in scores]


# =============================================================================
# Result Fusion
# =============================================================================


class FusionMethod(Enum):
    """Result fusion methods."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted score combination


@dataclass
class RankedResult:
    """A result with ranking information."""

    id: str
    score: float
    rank: int
    source: Optional[dict[str, Any]] = None


class ResultFusion:
    """Fuses results from multiple retrieval sources."""

    def __init__(
        self,
        method: FusionMethod = FusionMethod.RRF,
        rrf_k: int = 60,
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize fusion.

        Args:
            method: Fusion method to use
            rrf_k: RRF rank constant (default 60)
            weights: Weights for weighted fusion
        """
        self.method = method
        self.rrf_k = rrf_k
        self.weights = weights or {
            "lexical": 0.35,
            "vector": 0.40,
            "graph": 0.25,
        }

    def fuse(
        self,
        lexical: list[RankedResult],
        vector: list[RankedResult],
        graph: list[RankedResult],
        limit: Optional[int] = None,
    ) -> list[RankedResult]:
        """Fuse results from all sources.

        Args:
            lexical: Results from lexical search
            vector: Results from vector search
            graph: Results from graph expansion
            limit: Maximum results to return

        Returns:
            Fused and re-ranked results
        """
        if self.method == FusionMethod.RRF:
            fused = self._rrf_fuse(lexical, vector, graph)
        else:
            fused = self._weighted_fuse(lexical, vector, graph)

        # Sort by score descending
        fused.sort(key=lambda x: x.score, reverse=True)

        if limit:
            fused = fused[:limit]

        return fused

    def _rrf_fuse(
        self,
        lexical: list[RankedResult],
        vector: list[RankedResult],
        graph: list[RankedResult],
    ) -> list[RankedResult]:
        """Reciprocal Rank Fusion.

        Formula: score = sum(1 / (k + rank)) for each list
        """
        # Collect all unique document IDs
        all_results: dict[str, RankedResult] = {}
        rrf_scores: dict[str, float] = {}

        # Process each list
        for results in [lexical, vector, graph]:
            for result in results:
                # Calculate RRF contribution
                rrf_score = 1.0 / (self.rrf_k + result.rank)
                rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + rrf_score

                # Store result data
                if result.id not in all_results:
                    all_results[result.id] = result

        # Create fused results
        fused = []
        for doc_id, score in rrf_scores.items():
            original = all_results[doc_id]
            fused.append(
                RankedResult(
                    id=doc_id,
                    score=score,
                    rank=0,  # Will be set after sorting
                    source=original.source,
                )
            )

        return fused

    def _weighted_fuse(
        self,
        lexical: list[RankedResult],
        vector: list[RankedResult],
        graph: list[RankedResult],
    ) -> list[RankedResult]:
        """Weighted score combination."""
        # Build score maps
        lexical_scores = {r.id: r.score for r in lexical}
        vector_scores = {r.id: r.score for r in vector}
        graph_scores = {r.id: r.score for r in graph}

        # Collect all results
        all_results: dict[str, RankedResult] = {}
        for results in [lexical, vector, graph]:
            for result in results:
                if result.id not in all_results:
                    all_results[result.id] = result

        # Calculate weighted scores
        fused = []
        for doc_id, result in all_results.items():
            weighted_score = (
                lexical_scores.get(doc_id, 0.0) * self.weights["lexical"]
                + vector_scores.get(doc_id, 0.0) * self.weights["vector"]
                + graph_scores.get(doc_id, 0.0) * self.weights["graph"]
            )

            fused.append(
                RankedResult(
                    id=doc_id,
                    score=weighted_score,
                    rank=0,
                    source=result.source,
                )
            )

        return fused


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TriHybridTiming:
    """Timing breakdown for tri-hybrid retrieval."""

    lexical_ms: float = 0.0
    vector_ms: float = 0.0
    graph_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class TriHybridHit:
    """A single hit from tri-hybrid retrieval."""

    id: str
    score: float
    source: dict[str, Any]
    sources: dict[str, float] = field(default_factory=dict)
    graph_context: Optional[dict[str, Any]] = None


@dataclass
class TriHybridResult:
    """Result from tri-hybrid retrieval."""

    hits: list[TriHybridHit]
    total: int
    timing: TriHybridTiming
    graph_available: bool = True
    graph_error: Optional[str] = None
    lexical_error: Optional[str] = None
    vector_error: Optional[str] = None


# =============================================================================
# Main Retriever
# =============================================================================


class TriHybridRetriever:
    """Main tri-hybrid retriever combining lexical, vector, and graph search."""

    def __init__(
        self,
        opensearch_client: Any,
        graph_driver: Optional[Any] = None,
        config: Optional[TriHybridConfig] = None,
    ):
        """Initialize retriever.

        Args:
            opensearch_client: OpenSearch client for lexical/vector search
            graph_driver: Optional Neo4j driver for graph context
            config: Configuration options
        """
        self._opensearch = opensearch_client
        self.graph_driver = graph_driver
        self.config = config or TriHybridConfig()

        if graph_driver:
            self._graph_fetcher = GraphContextFetcher(graph_driver)
        else:
            self._graph_fetcher = None

    def retrieve(
        self,
        query: TriHybridQuery,
        index_name: str,
    ) -> TriHybridResult:
        """Perform tri-hybrid retrieval.

        Args:
            query: The search query
            index_name: OpenSearch index to search

        Returns:
            TriHybridResult with fused results

        Raises:
            RetrievalError: If all backends fail
        """
        total_start = time.perf_counter()

        timing = TriHybridTiming()
        lexical_results: list[RankedResult] = []
        vector_results: list[RankedResult] = []
        graph_results: list[RankedResult] = []

        lexical_error: Optional[str] = None
        vector_error: Optional[str] = None
        graph_error: Optional[str] = None

        # Track which sources contributed to each hit
        source_scores: dict[str, dict[str, float]] = {}

        # Run lexical and vector searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # Only submit lexical search if we have query text
            if query.query_text:
                futures["lexical"] = executor.submit(
                    self._lexical_search,
                    query,
                    index_name,
                )

            # Only submit vector search if we have embedding
            if query.embedding:
                futures["vector"] = executor.submit(
                    self._vector_search,
                    query,
                    index_name,
                )

            # Collect results
            for name, future in futures.items():
                try:
                    results, elapsed_ms = future.result()
                    if name == "lexical":
                        lexical_results = results
                        timing.lexical_ms = elapsed_ms
                        for r in results:
                            source_scores.setdefault(r.id, {})["lexical"] = r.score
                    else:
                        vector_results = results
                        timing.vector_ms = elapsed_ms
                        for r in results:
                            source_scores.setdefault(r.id, {})["vector"] = r.score
                except Exception as e:
                    logger.warning(f"{name} search failed: {e}")
                    if name == "lexical":
                        lexical_error = str(e)
                    else:
                        vector_error = str(e)

        # Graph context fetch
        graph_available = self.graph_driver is not None
        if self._graph_fetcher and query.seed_symbols:
            graph_start = time.perf_counter()
            try:
                graph_context = self._graph_fetcher.fetch_context(
                    symbol_ids=query.seed_symbols,
                    depth=self.config.graph_depth,
                    edge_types=self.config.graph_edge_types,
                )
                # Convert graph neighbors to ranked results
                # Score based on proximity (closer = higher score)
                for i, neighbor_id in enumerate(graph_context.neighbor_ids):
                    score = 1.0 / (i + 1)  # Simple proximity score
                    graph_results.append(
                        RankedResult(
                            id=neighbor_id,
                            score=score,
                            rank=i + 1,
                        )
                    )
                    source_scores.setdefault(neighbor_id, {})["graph"] = score

                timing.graph_ms = (time.perf_counter() - graph_start) * 1000

                # Capture errors from graph traversal
                if graph_context.errors:
                    graph_error = "; ".join(graph_context.errors)
            except Exception as e:
                logger.warning(f"Graph fetch failed: {e}")
                graph_error = str(e)
                timing.graph_ms = (time.perf_counter() - graph_start) * 1000

        # Check if we have any results
        if not lexical_results and not vector_results:
            if lexical_error and vector_error:
                raise RetrievalError(
                    f"All backends failed. Lexical: {lexical_error}, Vector: {vector_error}"
                )

        # Fuse results
        fusion_start = time.perf_counter()
        fusion = ResultFusion(
            method=FusionMethod.RRF,
            rrf_k=self.config.rrf_rank_constant,
            weights={
                "lexical": self.config.lexical_weight,
                "vector": self.config.vector_weight,
                "graph": self.config.graph_weight,
            },
        )
        fused = fusion.fuse(
            lexical=lexical_results,
            vector=vector_results,
            graph=graph_results,
            limit=query.size,
        )
        timing.fusion_ms = (time.perf_counter() - fusion_start) * 1000

        # Build hits with source breakdown
        hits = []
        for result in fused:
            sources = source_scores.get(result.id, {})
            hit = TriHybridHit(
                id=result.id,
                score=result.score,
                source=result.source or {},
                sources=sources,
            )
            hits.append(hit)

        timing.total_ms = (time.perf_counter() - total_start) * 1000

        return TriHybridResult(
            hits=hits,
            total=len(hits),
            timing=timing,
            graph_available=graph_available,
            graph_error=graph_error,
            lexical_error=lexical_error,
            vector_error=vector_error,
        )

    def _lexical_search(
        self,
        query: TriHybridQuery,
        index_name: str,
    ) -> tuple[list[RankedResult], float]:
        """Perform lexical search.

        Returns:
            Tuple of (results, elapsed_ms)
        """
        start = time.perf_counter()

        lexical_query = LexicalSearchQuery(
            query_text=query.query_text,
            fields=["content", "symbol_name"],
            filters=query.filters,
            size=self.config.rrf_window_size,  # Get more for fusion
        )

        response = self._opensearch.lexical_search(index_name, lexical_query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, hit in enumerate(response.hits, start=1):
            results.append(
                RankedResult(
                    id=hit.id,
                    score=hit.score,
                    rank=rank,
                    source=hit.source,
                )
            )

        return results, elapsed_ms

    def _vector_search(
        self,
        query: TriHybridQuery,
        index_name: str,
    ) -> tuple[list[RankedResult], float]:
        """Perform vector search.

        Returns:
            Tuple of (results, elapsed_ms)
        """
        start = time.perf_counter()

        vector_query = VectorSearchQuery(
            embedding=query.embedding,
            k=self.config.rrf_window_size,  # Get more for fusion
            filters=query.filters,
        )

        response = self._opensearch.vector_search(index_name, vector_query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, hit in enumerate(response.hits, start=1):
            results.append(
                RankedResult(
                    id=hit.id,
                    score=hit.score,
                    rank=rank,
                    source=hit.source,
                )
            )

        return results, elapsed_ms


# =============================================================================
# Factory Function
# =============================================================================


def create_trihybrid_retriever(
    opensearch_client: Any,
    graph_driver: Optional[Any] = None,
    config: Optional[TriHybridConfig] = None,
) -> TriHybridRetriever:
    """Create a tri-hybrid retriever.

    Args:
        opensearch_client: OpenSearch client for lexical/vector search
        graph_driver: Optional Neo4j driver for graph context
        config: Optional configuration

    Returns:
        Configured TriHybridRetriever
    """
    return TriHybridRetriever(
        opensearch_client=opensearch_client,
        graph_driver=graph_driver,
        config=config,
    )
