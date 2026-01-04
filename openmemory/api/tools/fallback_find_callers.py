"""Fallback Find Callers Tool.

This module provides the FallbackFindCallersTool that wraps FindCallersTool
with automatic fallback cascade for symbol not found errors.

Fallback Cascade:
1. Graph-based search (SCIP) - Primary
2. Grep-based pattern matching - First fallback
3. Semantic search (embeddings) - Second fallback
4. Structured error with suggestions - Final fallback

Integration points:
- openmemory.api.tools.call_graph: FindCallersTool
- openmemory.api.tools.search_code_hybrid: SearchCodeHybridTool
- openmemory.api.app.resilience.circuit_breaker: ServiceCircuitBreaker
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openmemory.api.tools.call_graph import (
    CallGraphConfig,
    CallGraphInput,
    FindCallersTool,
    GraphEdge,
    GraphNode,
    GraphOutput,
    ResponseMeta,
    SymbolNotFoundError,
    CallGraphError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Grep Tool Interface
# =============================================================================


@dataclass
class GrepMatch:
    """A single grep match result."""

    file: str
    line: int
    context: str = ""
    match_text: str = ""


class GrepTool:
    """Interface for grep-based code search.

    Can be implemented with actual grep, ripgrep, or OpenSearch lexical search.
    """

    def __init__(self, search_func: Optional[Callable] = None):
        """Initialize grep tool.

        Args:
            search_func: Optional custom search function
        """
        self._search_func = search_func

    def search(
        self,
        pattern: str,
        repo_id: str,
        include_patterns: Optional[list[str]] = None,
        max_results: int = 100,
    ) -> list[GrepMatch]:
        """Search for pattern in repository.

        Args:
            pattern: Search pattern
            repo_id: Repository ID
            include_patterns: File patterns to include (e.g., ["*.ts", "*.py"])
            max_results: Maximum results to return

        Returns:
            List of GrepMatch results
        """
        if self._search_func:
            try:
                results = self._search_func(
                    pattern=pattern,
                    repo_id=repo_id,
                    include_patterns=include_patterns,
                    max_results=max_results,
                )
                return [
                    GrepMatch(
                        file=r.get("file", ""),
                        line=r.get("line", 0),
                        context=r.get("context", ""),
                        match_text=r.get("match_text", ""),
                    )
                    for r in results
                ]
            except Exception as e:
                logger.warning(f"Grep search failed: {e}")
                return []
        return []


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FallbackConfig:
    """Configuration for fallback cascade.

    Args:
        stage_timeout_ms: Timeout per fallback stage in milliseconds
        total_timeout_ms: Total timeout for entire cascade
        grep_max_results: Maximum grep results before switching to semantic
        semantic_min_score: Minimum score threshold for semantic results
        circuit_breaker_threshold: Failures before circuit opens
        circuit_breaker_reset_s: Seconds before circuit recovery attempt
    """

    stage_timeout_ms: int = 150
    total_timeout_ms: int = 500
    grep_max_results: int = 50
    semantic_min_score: float = 0.5
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: int = 30


# =============================================================================
# Fallback Find Callers Tool
# =============================================================================


class FallbackFindCallersTool:
    """Find callers with automatic fallback cascade.

    Wraps FindCallersTool and provides automatic fallback when
    symbol is not found in the graph index.

    Fallback stages:
    1. Graph-based search (SCIP index)
    2. Grep-based pattern matching
    3. Semantic search (embeddings)
    4. Structured error with suggestions

    Usage:
        tool = FallbackFindCallersTool(
            graph_driver=driver,
            grep_tool=GrepTool(),
            search_tool=SearchCodeHybridTool(...),
        )
        result = tool.find(input_data)

        if tool.fallback_used:
            print(f"Used fallback stage {tool.fallback_stage}")
    """

    def __init__(
        self,
        graph_driver: Any,
        grep_tool: Optional[GrepTool] = None,
        search_tool: Optional[Any] = None,
        circuit_breakers: Optional[dict] = None,
        config: Optional[FallbackConfig] = None,
        call_graph_config: Optional[CallGraphConfig] = None,
    ):
        """Initialize fallback find callers tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            grep_tool: Optional grep tool for fallback stage 2
            search_tool: Optional SearchCodeHybridTool for fallback stage 3
            circuit_breakers: Optional dict of repo_id -> CircuitBreaker
            config: Optional fallback configuration
            call_graph_config: Optional call graph configuration
        """
        self.graph_driver = graph_driver
        self.grep_tool = grep_tool or GrepTool()
        self.search_tool = search_tool
        self.circuit_breakers = circuit_breakers or {}
        self.config = config or FallbackConfig()
        self.call_graph_config = call_graph_config or CallGraphConfig()

        # State tracking
        self.fallback_used = False
        self.fallback_stage: Optional[int] = None
        self._stage_timings: dict[int, float] = {}

    def _get_circuit_breaker(self, repo_id: str) -> Any:
        """Get or create circuit breaker for repository.

        Args:
            repo_id: Repository ID

        Returns:
            CircuitBreaker instance
        """
        if repo_id not in self.circuit_breakers:
            try:
                from openmemory.api.app.resilience.circuit_breaker import (
                    ServiceCircuitBreaker,
                )

                self.circuit_breakers[repo_id] = ServiceCircuitBreaker(
                    name=f"find_callers:{repo_id}",
                    failure_threshold=self.config.circuit_breaker_threshold,
                    recovery_timeout=self.config.circuit_breaker_reset_s,
                )
            except ImportError:
                # Return a dummy that always allows calls
                return None
        return self.circuit_breakers.get(repo_id)

    def find(
        self,
        input_data: CallGraphInput,
        config: Optional[CallGraphConfig] = None,
    ) -> GraphOutput:
        """Find callers with automatic fallback cascade.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            GraphOutput with caller nodes and edges
        """
        start_time = time.time()
        self.fallback_used = False
        self.fallback_stage = None
        self._stage_timings = {}

        cfg = config or self.call_graph_config
        breaker = self._get_circuit_breaker(input_data.repo_id)

        # Stage 1: Graph-based search
        stage_start = time.time()
        try:
            if breaker is None or breaker.state != "open":
                result = self._graph_search(input_data, cfg)
                if result.nodes:
                    self._stage_timings[1] = (time.time() - stage_start) * 1000
                    self._log_stage(1, start_time, success=True)
                    if breaker:
                        # Record success manually since we're not using context manager
                        breaker._failure_count = 0
                    return result
                # No results but no error - try fallbacks
                if breaker:
                    breaker._on_failure()
        except SymbolNotFoundError as e:
            logger.info(f"Stage 1 failed: {e}")
            if breaker:
                breaker._on_failure()
        except CallGraphError as e:
            logger.warning(f"Stage 1 error: {e}")
            if breaker:
                breaker._on_failure()

        self._stage_timings[1] = (time.time() - stage_start) * 1000
        self.fallback_used = True

        # Check total timeout
        if self._check_timeout(start_time):
            return self._build_timeout_response(input_data)

        # Stage 2: Grep-based fallback
        self.fallback_stage = 2
        stage_start = time.time()
        grep_result = self._grep_fallback(
            input_data.symbol_name or input_data.symbol_id or "",
            input_data.repo_id,
        )
        self._stage_timings[2] = (time.time() - stage_start) * 1000

        if grep_result and 0 < len(grep_result) <= self.config.grep_max_results:
            self._log_stage(2, start_time, success=True)
            return self._convert_grep_to_graph_output(grep_result, input_data)

        # Check total timeout
        if self._check_timeout(start_time):
            return self._build_timeout_response(input_data)

        # Stage 3: Semantic search fallback
        self.fallback_stage = 3
        stage_start = time.time()
        semantic_result = self._semantic_fallback(
            input_data.symbol_name or input_data.symbol_id or "",
            input_data.repo_id,
        )
        self._stage_timings[3] = (time.time() - stage_start) * 1000

        if semantic_result:
            self._log_stage(3, start_time, success=True)
            return self._convert_semantic_to_graph_output(semantic_result, input_data)

        # Stage 4: Structured error response
        self.fallback_stage = 4
        self._stage_timings[4] = 0  # Instant
        self._log_stage(4, start_time, success=False)
        return self._build_error_response(input_data)

    def _graph_search(
        self,
        input_data: CallGraphInput,
        config: CallGraphConfig,
    ) -> GraphOutput:
        """Stage 1: Graph-based search via SCIP index.

        Args:
            input_data: Input parameters
            config: Call graph configuration

        Returns:
            GraphOutput from FindCallersTool
        """
        tool = FindCallersTool(
            graph_driver=self.graph_driver,
            config=config,
        )
        return tool.find(input_data, config)

    def _grep_fallback(
        self,
        symbol_name: str,
        repo_id: str,
    ) -> Optional[list[GrepMatch]]:
        """Stage 2: Grep-based fallback search.

        Args:
            symbol_name: Symbol name to search for
            repo_id: Repository ID

        Returns:
            List of GrepMatch results or None
        """
        if not self.grep_tool:
            return None

        try:
            return self.grep_tool.search(
                pattern=symbol_name,
                repo_id=repo_id,
                include_patterns=["*.ts", "*.tsx", "*.py", "*.js", "*.jsx"],
                max_results=self.config.grep_max_results + 10,
            )
        except Exception as e:
            logger.warning(f"Grep fallback failed: {e}")
            return None

    def _semantic_fallback(
        self,
        symbol_name: str,
        repo_id: str,
    ) -> Optional[list[dict]]:
        """Stage 3: Semantic search fallback.

        Args:
            symbol_name: Symbol name to search for
            repo_id: Repository ID

        Returns:
            List of search results or None
        """
        if not self.search_tool:
            return None

        try:
            # Extract keywords from camelCase/snake_case
            keywords = self._extract_keywords(symbol_name)
            query = " ".join(keywords)

            # Import search input type
            try:
                from openmemory.api.tools.search_code_hybrid import (
                    SearchCodeHybridInput,
                )
            except ImportError:
                from tools.search_code_hybrid import SearchCodeHybridInput

            input_data = SearchCodeHybridInput(
                query=query,
                repo_id=repo_id,
                limit=20,
            )

            result = self.search_tool.search(input_data)

            # Filter by minimum score
            hits = []
            for hit in result.results:
                if hit.score >= self.config.semantic_min_score:
                    hits.append(
                        {
                            "symbol_id": hit.symbol.symbol_id,
                            "symbol_name": hit.symbol.symbol_name,
                            "file_path": hit.symbol.file_path,
                            "score": hit.score,
                        }
                    )

            return hits if hits else None
        except Exception as e:
            logger.warning(f"Semantic fallback failed: {e}")
            return None

    def _extract_keywords(self, symbol_name: str) -> list[str]:
        """Extract keywords from camelCase/snake_case symbol names.

        Args:
            symbol_name: Symbol name to split

        Returns:
            List of lowercase keywords
        """
        # Split by underscore or camelCase
        words = re.split(r"_|(?=[A-Z])", symbol_name)
        return [w.lower() for w in words if w]

    def _convert_grep_to_graph_output(
        self,
        grep_results: list[GrepMatch],
        input_data: CallGraphInput,
    ) -> GraphOutput:
        """Convert grep results to GraphOutput format.

        Args:
            grep_results: Grep match results
            input_data: Original input data

        Returns:
            GraphOutput with grep-based nodes
        """
        nodes = []
        for i, match in enumerate(grep_results[:20]):  # Limit to 20
            nodes.append(
                GraphNode(
                    id=f"grep:{match.file}:{match.line}",
                    type="grep_match",
                    properties={
                        "name": f"{match.file}:{match.line}",
                        "file_path": match.file,
                        "line_number": match.line,
                        "context": match.context,
                        "match_text": match.match_text,
                    },
                )
            )

        return GraphOutput(
            nodes=nodes,
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                fallback_stage=2,
                fallback_strategy="grep",
                warning="Results from grep fallback - may include false positives",
            ),
        )

    def _convert_semantic_to_graph_output(
        self,
        semantic_results: list[dict],
        input_data: CallGraphInput,
    ) -> GraphOutput:
        """Convert semantic search results to GraphOutput format.

        Args:
            semantic_results: Semantic search results
            input_data: Original input data

        Returns:
            GraphOutput with semantic-based nodes
        """
        nodes = []
        for result in semantic_results[:10]:  # Limit to 10
            nodes.append(
                GraphNode(
                    id=f"semantic:{result.get('symbol_id', 'unknown')}",
                    type="semantic_match",
                    properties={
                        "name": result.get("symbol_name", "unknown"),
                        "file_path": result.get("file_path"),
                        "score": result.get("score"),
                    },
                )
            )

        return GraphOutput(
            nodes=nodes,
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                fallback_stage=3,
                fallback_strategy="semantic_search",
                warning="Results from semantic search - verify relevance manually",
            ),
        )

    def _build_error_response(self, input_data: CallGraphInput) -> GraphOutput:
        """Stage 4: Build structured error response with suggestions.

        Args:
            input_data: Original input data

        Returns:
            GraphOutput with empty nodes but detailed suggestions
        """
        symbol_name = input_data.symbol_name or input_data.symbol_id or "unknown"

        suggestions = [
            f"Try: grep -r '{symbol_name}' --include='*.ts' --include='*.py'",
            f"Try: search_code_hybrid(query='{symbol_name}')",
            "Symbol may be called via decorator (@OnEvent, @Subscribe, @EventHandler)",
            "Symbol may be injected via DI (constructor injection)",
            "Index may be stale - consider re-indexing with index_codebase(reset=true)",
        ]

        return GraphOutput(
            nodes=[],
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                missing_sources=["graph_index", "grep", "semantic_search"],
                fallback_stage=4,
                fallback_strategy="structured_error",
                warning="Symbol not found in any source - see suggestions",
            ),
            suggestions=suggestions,
        )

    def _build_timeout_response(self, input_data: CallGraphInput) -> GraphOutput:
        """Build response when timeout is reached.

        Args:
            input_data: Original input data

        Returns:
            GraphOutput indicating timeout
        """
        symbol_name = input_data.symbol_name or input_data.symbol_id or "unknown"

        return GraphOutput(
            nodes=[],
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                missing_sources=["timeout"],
                fallback_stage=self.fallback_stage or 1,
                fallback_strategy="timeout",
                warning="Fallback cascade timed out",
            ),
            suggestions=[
                f"Try manual search: grep -r '{symbol_name}'",
                "Consider re-indexing the repository",
            ],
        )

    def _check_timeout(self, start_time: float) -> bool:
        """Check if total timeout has been exceeded.

        Args:
            start_time: When the cascade started

        Returns:
            True if timeout exceeded
        """
        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms >= self.config.total_timeout_ms

    def _log_stage(self, stage: int, start_time: float, success: bool) -> None:
        """Log fallback stage execution.

        Args:
            stage: Stage number (1-4)
            start_time: When the cascade started
            success: Whether the stage succeeded
        """
        elapsed_ms = (time.time() - start_time) * 1000
        stage_time_ms = self._stage_timings.get(stage, 0)

        logger.info(
            f"Fallback cascade stage={stage} success={success} "
            f"stage_ms={stage_time_ms:.2f} total_ms={elapsed_ms:.2f}"
        )

    def get_stage_timings(self) -> dict[int, float]:
        """Get timing information for each stage.

        Returns:
            Dict mapping stage number to milliseconds
        """
        return self._stage_timings.copy()


# =============================================================================
# Factory Function
# =============================================================================


def create_fallback_find_callers_tool(
    graph_driver: Any,
    grep_tool: Optional[GrepTool] = None,
    search_tool: Optional[Any] = None,
    config: Optional[FallbackConfig] = None,
    call_graph_config: Optional[CallGraphConfig] = None,
) -> FallbackFindCallersTool:
    """Create a fallback find_callers tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        grep_tool: Optional grep tool for fallback stage 2
        search_tool: Optional SearchCodeHybridTool for fallback stage 3
        config: Optional fallback configuration
        call_graph_config: Optional call graph configuration

    Returns:
        Configured FallbackFindCallersTool
    """
    return FallbackFindCallersTool(
        graph_driver=graph_driver,
        grep_tool=grep_tool,
        search_tool=search_tool,
        config=config,
        call_graph_config=call_graph_config,
    )
