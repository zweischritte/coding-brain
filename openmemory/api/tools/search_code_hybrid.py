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
from dataclasses import dataclass, field
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
        include_source_breakdown: Include per-source scores
        embed_query: Whether to embed the query for vector search
    """

    index_name: str = "code"
    limit: int = 10
    offset: int = 0
    include_snippet: bool = True
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
    """

    query: str
    repo_id: Optional[str] = None
    language: Optional[str] = None
    limit: int = 10
    offset: int = 0
    seed_symbols: list[str] = field(default_factory=list)


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

        # Build filters
        filters: dict[str, Any] = {}
        if input_data.repo_id:
            filters["repo_id"] = input_data.repo_id
        if input_data.language:
            filters["language"] = input_data.language

        # Build tri-hybrid query
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

        # Convert to MCP schema format
        hits = []
        for hit in result.hits:
            source_data = hit.source or {}

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
            )

            # Include snippet if configured
            if cfg.include_snippet:
                code_hit.snippet = source_data.get("content")

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
