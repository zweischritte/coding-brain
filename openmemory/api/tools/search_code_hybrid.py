"""Search Code Hybrid Tool.

This module provides the search_code_hybrid MCP tool for tri-hybrid code search:
- SearchCodeHybridConfig: Configuration for the tool
- SearchCodeHybridInput: Input parameters
- SearchCodeHybridResult: Search results with MCP schema compliance
- SearchCodeHybridTool: Main tool entry point

Integration points:
- openmemory.api.retrieval.trihybrid: Tri-hybrid retrieval
- Embedding service for semantic search
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class SearchCodeHybridError(Exception):
    """Base exception for search_code_hybrid tool errors."""

    pass


class InvalidQueryError(SearchCodeHybridError):
    """Raised when query is invalid or empty."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SearchCodeHybridConfig:
    """Configuration for search_code_hybrid tool.

    Args:
        index_name: OpenSearch index to search
        limit: Maximum results to return (default 10)
        offset: Pagination offset
        include_snippet: Include code snippets in results
        snippet_max_chars: Maximum characters to return for snippet (None for no truncation)
        include_source_breakdown: Include per-source scores
        embed_query: Whether to embed the query for vector search
    """

    index_name: str = "code"
    limit: int = 10
    offset: int = 0
    include_snippet: bool = True
    snippet_max_chars: Optional[int] = 400
    include_source_breakdown: bool = True
    embed_query: bool = True


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class SearchCodeHybridInput:
    """Input parameters for search_code_hybrid.

    Args:
        query: Text query for search
        repo_id: Optional repository filter
        language: Optional language filter
        limit: Maximum results
        offset: Pagination offset
        seed_symbols: Optional seed symbols for graph expansion
        include_snippet: Override snippet inclusion for this call
        snippet_max_chars: Override snippet length limit for this call
        include_generated: Include generated results without source preference
    """

    query: str
    repo_id: Optional[str] = None
    language: Optional[str] = None
    limit: int = 10
    offset: int = 0
    seed_symbols: list[str] = field(default_factory=list)
    include_snippet: Optional[bool] = None
    snippet_max_chars: Optional[int] = None
    include_generated: Optional[bool] = None


# =============================================================================
# Result Types (MCP Schema Compliant)
# =============================================================================


@dataclass
class CodeSymbol:
    """A code symbol in search results.

    Maps to CodeSymbol in MCP schema.
    """

    symbol_id: Optional[str] = None
    symbol_name: str = ""
    symbol_type: str = ""
    signature: Optional[str] = None
    language: Optional[str] = None
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class CodeHit:
    """A single search hit.

    Maps to CodeHit in MCP schema.
    """

    symbol: CodeSymbol
    score: float
    source: str = "hybrid"  # "vector", "lexical", "graph", "hybrid"
    snippet: Optional[str] = None
    repo_id: Optional[str] = None
    source_scores: dict[str, float] = field(default_factory=dict)
    is_generated: Optional[bool] = None
    source_tier: Optional[str] = None


@dataclass
class ResponseMeta:
    """Response metadata.

    Maps to ResponseMeta in MCP schema.
    """

    request_id: str
    degraded_mode: bool = False
    missing_sources: list[str] = field(default_factory=list)
    confidence_adjustment: Optional[float] = None
    next_cursor: Optional[str] = None


@dataclass
class SearchCodeHybridResult:
    """Result from search_code_hybrid.

    Maps to SearchOutput in MCP schema.
    """

    results: list[CodeHit]
    meta: ResponseMeta
    next_cursor: Optional[str] = None


# =============================================================================
# Main Tool
# =============================================================================


class SearchCodeHybridTool:
    """MCP tool for tri-hybrid code search.

    Combines lexical, semantic, and graph search for comprehensive
    code discovery.
    """

    def __init__(
        self,
        retriever: Any,
        embedding_service: Optional[Any] = None,
        config: Optional[SearchCodeHybridConfig] = None,
    ):
        """Initialize search_code_hybrid tool.

        Args:
            retriever: TriHybridRetriever instance
            embedding_service: Optional embedding service for vector search
            config: Optional configuration
        """
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.config = config or SearchCodeHybridConfig()

    def _source_tier_for_hit(self, source_data: dict[str, Any]) -> str:
        tier = source_data.get("source_tier")
        if tier:
            return str(tier)
        if source_data.get("is_generated") is True:
            return "generated"
        return "source"

    def _prefer_source_hits(
        self,
        hits: list[Any],
        include_generated: bool,
    ) -> list[Any]:
        if include_generated or not hits:
            return hits
        tiers = [self._source_tier_for_hit(hit.source or {}) for hit in hits]
        if not any(tier == "source" for tier in tiers):
            return hits
        priority = {"source": 0, "generated": 1, "vendor": 2}
        indexed = list(enumerate(hits))
        indexed.sort(
            key=lambda item: (priority.get(tiers[item[0]], 1), item[0])
        )
        return [hit for _, hit in indexed]

    def _hydrate_graph_source(self, hit: Any) -> dict[str, Any]:
        source_data = hit.source or {}
        if source_data.get("symbol_name") or source_data.get("file_path"):
            return source_data

        graph_driver = getattr(self.retriever, "graph_driver", None)
        if not graph_driver or not hasattr(graph_driver, "get_node"):
            return source_data

        try:
            node = graph_driver.get_node(hit.id)
        except Exception as exc:
            logger.debug(f"Graph hydration failed for {hit.id}: {exc}")
            return source_data

        if not node:
            return source_data

        props = getattr(node, "properties", {}) or {}
        node_type = getattr(node, "node_type", None)
        node_type_value = getattr(node_type, "value", None) if node_type else None

        hydrated = dict(source_data)
        symbol_type = hydrated.get("symbol_type") or props.get("kind")
        if not symbol_type and node_type_value:
            symbol_type = str(node_type_value).lower().replace("code_", "")
        if not symbol_type:
            symbol_type = props.get("schema_type")
        if symbol_type:
            hydrated["symbol_type"] = symbol_type

        symbol_name = hydrated.get("symbol_name") or props.get("name") or props.get("leaf")
        if not symbol_name:
            path_value = props.get("file_path") or props.get("path")
            if path_value:
                symbol_name = str(path_value).split("/")[-1]
        if symbol_name:
            hydrated["symbol_name"] = symbol_name

        file_path = hydrated.get("file_path") or props.get("file_path") or props.get("path")
        if file_path:
            hydrated["file_path"] = file_path

        if "line_start" not in hydrated and props.get("line_start") is not None:
            hydrated["line_start"] = props.get("line_start")
        if "line_end" not in hydrated and props.get("line_end") is not None:
            hydrated["line_end"] = props.get("line_end")
        if "language" not in hydrated and props.get("language"):
            hydrated["language"] = props.get("language")
        if "signature" not in hydrated and props.get("signature"):
            hydrated["signature"] = props.get("signature")
        if "repo_id" not in hydrated and props.get("repo_id"):
            hydrated["repo_id"] = props.get("repo_id")

        return hydrated

    def search(
        self,
        input_data: SearchCodeHybridInput,
        config: Optional[SearchCodeHybridConfig] = None,
    ) -> SearchCodeHybridResult:
        """Perform tri-hybrid code search.

        Args:
            input_data: Search input parameters
            config: Optional config override

        Returns:
            SearchCodeHybridResult with search hits

        Raises:
            InvalidQueryError: If query is empty
            SearchCodeHybridError: If search fails
        """
        cfg = config or self.config
        if input_data.include_snippet is not None or input_data.snippet_max_chars is not None:
            cfg = replace(
                cfg,
                include_snippet=(
                    input_data.include_snippet
                    if input_data.include_snippet is not None
                    else cfg.include_snippet
                ),
                snippet_max_chars=(
                    input_data.snippet_max_chars
                    if input_data.snippet_max_chars is not None
                    else cfg.snippet_max_chars
                ),
            )
        request_id = str(uuid.uuid4())

        # Validate query
        if not input_data.query or not input_data.query.strip():
            raise InvalidQueryError("Query cannot be empty")

        # Build query embedding
        embedding: list[float] = []
        if cfg.embed_query and self.embedding_service:
            try:
                result = self.embedding_service.embed(input_data.query)
                if isinstance(result, list):
                    embedding = result
                elif hasattr(result, "embedding"):
                    embedding = list(result.embedding)
                else:
                    embedding = list(result)
            except Exception as e:
                logger.warning(f"Embedding failed, falling back to lexical: {e}")
                # Fall back to lexical-only
        if embedding and not any(embedding):
            embedding = []

        # Build filters
        filters: dict[str, Any] = {}
        if input_data.repo_id:
            filters["repo_id"] = input_data.repo_id
        if input_data.language:
            filters["language"] = input_data.language

        # Build tri-hybrid query
        try:
            from retrieval.trihybrid import TriHybridQuery
        except ImportError:
            from openmemory.api.retrieval.trihybrid import TriHybridQuery

        query = TriHybridQuery(
            query_text=input_data.query,
            embedding=embedding,
            seed_symbols=input_data.seed_symbols,
            filters=filters,
            size=input_data.limit or cfg.limit,
            offset=input_data.offset or cfg.offset,
        )

        # Execute search
        try:
            result = self.retriever.retrieve(query, cfg.index_name)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise SearchCodeHybridError(f"Search failed: {e}") from e

        include_generated = input_data.include_generated is True
        hits_for_output = self._prefer_source_hits(
            result.hits,
            include_generated=include_generated,
        )

        # Convert to MCP schema format
        hits = []
        for hit in hits_for_output:
            source_data = self._hydrate_graph_source(hit)

            if not source_data.get("symbol_name") and not source_data.get("file_path"):
                continue

            symbol = CodeSymbol(
                symbol_id=hit.id,
                symbol_name=source_data.get("symbol_name", ""),
                symbol_type=source_data.get("symbol_type", ""),
                signature=source_data.get("signature"),
                language=source_data.get("language"),
                file_path=source_data.get("file_path"),
                line_start=source_data.get("line_start"),
                line_end=source_data.get("line_end"),
            )

            code_hit = CodeHit(
                symbol=symbol,
                score=hit.score,
                source="hybrid",
                repo_id=source_data.get("repo_id"),
                is_generated=source_data.get("is_generated"),
                source_tier=source_data.get("source_tier"),
            )

            # Include snippet if configured
            if cfg.include_snippet:
                snippet = source_data.get("content")
                if snippet is not None and cfg.snippet_max_chars is not None:
                    max_chars = max(0, cfg.snippet_max_chars)
                    if len(snippet) > max_chars:
                        snippet = snippet[:max_chars]
                        if max_chars > 0:
                            snippet += "..."
                code_hit.snippet = snippet

            # Include source breakdown if configured
            if cfg.include_source_breakdown and hasattr(hit, "sources"):
                code_hit.source_scores = hit.sources

            hits.append(code_hit)

        # Build response metadata
        missing_sources: list[str] = []
        if result.lexical_error:
            missing_sources.append("lexical")
        if result.vector_error:
            missing_sources.append("vector")
        if result.graph_error or not result.graph_available:
            missing_sources.append("graph")

        meta = ResponseMeta(
            request_id=request_id,
            degraded_mode=len(missing_sources) > 0,
            missing_sources=missing_sources,
        )

        return SearchCodeHybridResult(
            results=hits,
            meta=meta,
            next_cursor=None,  # Could implement cursor-based pagination
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_search_code_hybrid_tool(
    retriever: Any,
    embedding_service: Optional[Any] = None,
    config: Optional[SearchCodeHybridConfig] = None,
) -> SearchCodeHybridTool:
    """Create a search_code_hybrid tool.

    Args:
        retriever: TriHybridRetriever instance
        embedding_service: Optional embedding service
        config: Optional configuration

    Returns:
        Configured SearchCodeHybridTool
    """
    return SearchCodeHybridTool(
        retriever=retriever,
        embedding_service=embedding_service,
        config=config,
    )
