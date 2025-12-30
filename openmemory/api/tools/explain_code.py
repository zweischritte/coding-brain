"""Explain Code Tool (FR-007).

This module provides the explain_code MCP tool for explaining code symbols:
- ExplainCodeConfig: Configuration for the tool
- SymbolExplanation: Structured explanation result
- SymbolLookup: SCIP ID lookup in graph
- CallGraphTraverser: Call graph traversal (callers/callees)
- DocumentationExtractor: Docstring extraction from AST
- CodeContextRetriever: Tri-hybrid context retrieval
- ExplanationFormatter: LLM-friendly output formatting
- ExplainCodeTool: Main tool entry point

Integration points:
- openmemory.api.indexing.scip_symbols: SCIP symbol ID parsing
- openmemory.api.indexing.ast_parser: Symbol and docstring extraction
- openmemory.api.indexing.graph_projection: CODE_* graph queries
- openmemory.api.retrieval.trihybrid: Context retrieval
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ExplainCodeError(Exception):
    """Base exception for explain_code tool errors."""

    pass


class SymbolNotFoundError(ExplainCodeError):
    """Raised when a symbol cannot be found in the graph."""

    pass


class InvalidSymbolIDError(ExplainCodeError):
    """Raised when a symbol ID is invalid or malformed."""

    pass


class SymbolLookupError(ExplainCodeError):
    """Raised when symbol lookup fails due to driver error."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExplainCodeConfig:
    """Configuration for explain_code tool.

    Args:
        depth: Call graph traversal depth (default 2)
        include_callers: Include functions that call this symbol
        include_callees: Include functions this symbol calls
        include_usages: Include usage examples from codebase
        max_usages: Maximum usage examples to return
        include_related: Include related symbols
        max_related: Maximum related symbols to return
        cache_ttl_seconds: Cache TTL for repeated queries
    """

    depth: int = 2
    include_callers: bool = True
    include_callees: bool = True
    include_usages: bool = True
    max_usages: int = 5
    include_related: bool = True
    max_related: int = 10
    cache_ttl_seconds: int = 300


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class SymbolExplanation:
    """Structured explanation of a code symbol.

    Contains all information about a symbol including its location,
    documentation, call graph, usage examples, and related symbols.
    """

    symbol_id: str
    name: str
    kind: str
    signature: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    callers: list[dict[str, Any]] = field(default_factory=list)
    callees: list[dict[str, Any]] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)
    related: list[dict[str, Any]] = field(default_factory=list)
    context: Optional[str] = None


# =============================================================================
# Symbol Lookup
# =============================================================================


class SymbolLookup:
    """Looks up symbols in the code graph by SCIP ID."""

    def __init__(self, graph_driver: Any):
        """Initialize with Neo4j graph driver.

        Args:
            graph_driver: Neo4j driver with get_node method
        """
        self._driver = graph_driver

    def lookup(self, symbol_id: str) -> dict[str, Any]:
        """Look up a symbol by its SCIP ID.

        Args:
            symbol_id: SCIP symbol ID (e.g., "scip-python myapp module/func.")

        Returns:
            Dictionary with symbol properties

        Raises:
            InvalidSymbolIDError: If symbol ID is empty or invalid
            SymbolNotFoundError: If symbol not found in graph
            SymbolLookupError: If driver operation fails
        """
        if not symbol_id or not symbol_id.strip():
            raise InvalidSymbolIDError("Symbol ID cannot be empty")

        try:
            node = self._driver.get_node(symbol_id)
        except Exception as e:
            raise SymbolLookupError(f"Failed to lookup symbol: {e}") from e

        if node is None:
            raise SymbolNotFoundError(f"Symbol not found: {symbol_id}")

        return dict(node.properties)


# =============================================================================
# Call Graph Traverser
# =============================================================================


class CallGraphTraverser:
    """Traverses the code call graph to find callers and callees."""

    def __init__(self, graph_driver: Any):
        """Initialize with Neo4j graph driver.

        Args:
            graph_driver: Neo4j driver with edge query methods
        """
        self._driver = graph_driver

    def get_callers(
        self,
        symbol_id: str,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get symbols that call this symbol.

        Args:
            symbol_id: SCIP symbol ID to find callers for
            depth: How many levels to traverse (default 1)

        Returns:
            List of caller information dicts
        """
        callers = []
        visited = set()

        self._traverse_callers(symbol_id, depth, callers, visited)

        return callers

    def _traverse_callers(
        self,
        symbol_id: str,
        depth: int,
        callers: list[dict[str, Any]],
        visited: set[str],
    ) -> None:
        """Recursively traverse callers."""
        if depth <= 0 or symbol_id in visited:
            return

        visited.add(symbol_id)

        try:
            # Get incoming CALLS edges
            if hasattr(self._driver, "get_incoming_edges"):
                edges = self._driver.get_incoming_edges(symbol_id)
            else:
                # Fallback: no incoming edges method available
                return

            for edge in edges:
                edge_type = (
                    edge.edge_type.value
                    if hasattr(edge.edge_type, "value")
                    else str(edge.edge_type)
                )

                if edge_type != "CALLS":
                    continue

                caller_id = edge.source_id

                if caller_id in visited:
                    continue

                # Get caller node info
                caller_node = self._driver.get_node(caller_id)
                if caller_node:
                    caller_info = {
                        "symbol_id": caller_id,
                        "name": caller_node.properties.get("name", ""),
                        "kind": caller_node.properties.get("kind", ""),
                        "file_path": caller_node.properties.get("file_path", ""),
                        "line_start": caller_node.properties.get("line_start", 0),
                        "depth": depth,
                    }

                    # Add call site info if available
                    if edge.properties:
                        caller_info["call_line"] = edge.properties.get("call_line")
                        caller_info["call_col"] = edge.properties.get("call_col")

                    callers.append(caller_info)

                # Recurse for transitive callers
                if depth > 1:
                    self._traverse_callers(caller_id, depth - 1, callers, visited)

        except Exception as e:
            logger.warning(f"Error traversing callers for {symbol_id}: {e}")

    def get_callees(
        self,
        symbol_id: str,
        depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get symbols that this symbol calls.

        Args:
            symbol_id: SCIP symbol ID to find callees for
            depth: How many levels to traverse (default 1)

        Returns:
            List of callee information dicts
        """
        callees = []
        visited = set()

        self._traverse_callees(symbol_id, depth, callees, visited)

        return callees

    def _traverse_callees(
        self,
        symbol_id: str,
        depth: int,
        callees: list[dict[str, Any]],
        visited: set[str],
    ) -> None:
        """Recursively traverse callees."""
        if depth <= 0 or symbol_id in visited:
            return

        visited.add(symbol_id)

        try:
            edges = self._driver.get_outgoing_edges(symbol_id)

            for edge in edges:
                edge_type = (
                    edge.edge_type.value
                    if hasattr(edge.edge_type, "value")
                    else str(edge.edge_type)
                )

                if edge_type != "CALLS":
                    continue

                callee_id = edge.target_id

                if callee_id in visited:
                    continue

                # Get callee node info
                callee_node = self._driver.get_node(callee_id)
                if callee_node:
                    callee_info = {
                        "symbol_id": callee_id,
                        "name": callee_node.properties.get("name", ""),
                        "kind": callee_node.properties.get("kind", ""),
                        "file_path": callee_node.properties.get("file_path", ""),
                        "line_start": callee_node.properties.get("line_start", 0),
                        "depth": depth,
                    }

                    # Add call site info if available
                    if edge.properties:
                        callee_info["call_line"] = edge.properties.get("call_line")
                        callee_info["call_col"] = edge.properties.get("call_col")

                    callees.append(callee_info)

                # Recurse for transitive callees
                if depth > 1:
                    self._traverse_callees(callee_id, depth - 1, callees, visited)

        except Exception as e:
            logger.warning(f"Error traversing callees for {symbol_id}: {e}")


# =============================================================================
# Documentation Extractor
# =============================================================================


class DocumentationExtractor:
    """Extracts docstrings from source files using AST parsing."""

    def __init__(self, parser: Any):
        """Initialize with AST parser.

        Args:
            parser: ASTParser instance for parsing source files
        """
        self._parser = parser

    def extract(
        self,
        file_path: Path,
        symbol_name: str,
        line_start: int,
    ) -> Optional[str]:
        """Extract docstring for a symbol.

        Args:
            file_path: Path to source file
            symbol_name: Name of the symbol to find
            line_start: Starting line of the symbol

        Returns:
            Docstring if found, None otherwise
        """
        try:
            result = self._parser.parse_file(file_path)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return None

        if not result.success:
            return None

        # Find matching symbol by name and line
        for symbol in result.symbols:
            if symbol.name == symbol_name:
                # Check line proximity (within 5 lines)
                symbol_line = getattr(symbol, "line_start", 0)
                if abs(symbol_line - line_start) <= 5:
                    return symbol.docstring

        return None


# =============================================================================
# Code Context Retriever
# =============================================================================


class CodeContextRetriever:
    """Retrieves code context using tri-hybrid retrieval."""

    def __init__(self, retriever: Any):
        """Initialize with tri-hybrid retriever.

        Args:
            retriever: TriHybridRetriever instance
        """
        self._retriever = retriever

    def retrieve(
        self,
        symbol_id: str,
        symbol_name: str,
        max_results: int = 5,
    ) -> Optional[str]:
        """Retrieve context for a symbol.

        Args:
            symbol_id: SCIP symbol ID
            symbol_name: Symbol name for query
            max_results: Maximum results to retrieve

        Returns:
            Context string or None if not available
        """
        try:
            # Build query for context
            try:
                from retrieval.trihybrid import TriHybridQuery
            except ImportError:
                from openmemory.api.retrieval.trihybrid import TriHybridQuery

            query = TriHybridQuery(
                query_text=symbol_name,
                seed_symbols=[symbol_id],
                size=max_results,
            )

            result = self._retriever.retrieve(query, index_name="code")

            if not result.hits:
                return None

            # Combine hit content into context
            context_parts = []
            for hit in result.hits:
                content = hit.source.get("content", "")
                if content:
                    context_parts.append(content)

            return "\n\n".join(context_parts) if context_parts else None

        except Exception as e:
            logger.warning(f"Error retrieving context for {symbol_name}: {e}")
            return None

    def retrieve_usages(
        self,
        symbol_name: str,
        max_usages: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve usage examples for a symbol.

        Args:
            symbol_name: Symbol name to search for usages
            max_usages: Maximum usages to return

        Returns:
            List of usage information dicts
        """
        try:
            try:
                from retrieval.trihybrid import TriHybridQuery
            except ImportError:
                from openmemory.api.retrieval.trihybrid import TriHybridQuery

            # Search for usages of the symbol
            query = TriHybridQuery(
                query_text=f"usage {symbol_name}",
                size=max_usages,
            )

            result = self._retriever.retrieve(query, index_name="code")

            usages = []
            for hit in result.hits:
                usages.append(
                    {
                        "code": hit.source.get("content", ""),
                        "file": hit.source.get("file_path", ""),
                        "score": hit.score,
                    }
                )

            return usages

        except Exception as e:
            logger.warning(f"Error retrieving usages for {symbol_name}: {e}")
            return []

    def retrieve_related(
        self,
        symbol_id: str,
        symbol_name: str,
        max_related: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve related symbols.

        Args:
            symbol_id: SCIP symbol ID
            symbol_name: Symbol name
            max_related: Maximum related symbols

        Returns:
            List of related symbol dicts
        """
        try:
            try:
                from retrieval.trihybrid import TriHybridQuery
            except ImportError:
                from openmemory.api.retrieval.trihybrid import TriHybridQuery

            query = TriHybridQuery(
                query_text=symbol_name,
                seed_symbols=[symbol_id],
                size=max_related,
            )

            result = self._retriever.retrieve(query, index_name="code")

            related = []
            for hit in result.hits:
                if hit.id != symbol_id:  # Exclude self
                    related.append(
                        {
                            "symbol_id": hit.id,
                            "name": hit.source.get("symbol_name", ""),
                            "kind": hit.source.get("kind", ""),
                            "file_path": hit.source.get("file_path", ""),
                            "score": hit.score,
                        }
                    )

            return related

        except Exception as e:
            logger.warning(f"Error retrieving related symbols for {symbol_name}: {e}")
            return []


# =============================================================================
# Explanation Formatter
# =============================================================================


class ExplanationFormatter:
    """Formats symbol explanations for different output formats."""

    def format_json(self, explanation: SymbolExplanation) -> str:
        """Format explanation as JSON.

        Args:
            explanation: SymbolExplanation to format

        Returns:
            JSON string representation
        """
        data = {
            "symbol_id": explanation.symbol_id,
            "name": explanation.name,
            "kind": explanation.kind,
            "signature": explanation.signature,
            "file_path": explanation.file_path,
            "line_start": explanation.line_start,
            "line_end": explanation.line_end,
            "docstring": explanation.docstring,
            "callers": explanation.callers,
            "callees": explanation.callees,
            "usages": explanation.usages,
            "related": explanation.related,
            "context": explanation.context,
        }

        return json.dumps(data, indent=2)

    def format_markdown(self, explanation: SymbolExplanation) -> str:
        """Format explanation as Markdown.

        Args:
            explanation: SymbolExplanation to format

        Returns:
            Markdown string representation
        """
        lines = []

        # Header
        lines.append(f"# {explanation.name}")
        lines.append("")
        lines.append(f"**Kind:** {explanation.kind}")
        lines.append(f"**Location:** `{explanation.file_path}:{explanation.line_start}-{explanation.line_end}`")
        lines.append("")

        # Signature
        lines.append("## Signature")
        lines.append("```")
        lines.append(explanation.signature)
        lines.append("```")
        lines.append("")

        # Docstring
        if explanation.docstring:
            lines.append("## Documentation")
            lines.append(explanation.docstring)
            lines.append("")

        # Callers
        if explanation.callers:
            lines.append("## Callers")
            for caller in explanation.callers:
                name = caller.get("name", "unknown")
                file_path = caller.get("file_path", "")
                line = caller.get("line_start", "")
                lines.append(f"- `{name}` ({file_path}:{line})")
            lines.append("")

        # Callees
        if explanation.callees:
            lines.append("## Callees")
            for callee in explanation.callees:
                name = callee.get("name", "unknown")
                file_path = callee.get("file_path", "")
                line = callee.get("line_start", "")
                lines.append(f"- `{name}` ({file_path}:{line})")
            lines.append("")

        # Usages
        if explanation.usages:
            lines.append("## Usage Examples")
            for usage in explanation.usages[:3]:  # Limit to 3 examples
                code = usage.get("code", "")
                if code:
                    lines.append("```")
                    lines.append(code[:200])  # Truncate long code
                    lines.append("```")
                    lines.append("")

        # Related
        if explanation.related:
            lines.append("## Related Symbols")
            for related in explanation.related[:5]:  # Limit to 5
                name = related.get("name", "")
                kind = related.get("kind", "")
                lines.append(f"- `{name}` ({kind})")
            lines.append("")

        return "\n".join(lines)

    def format_for_llm(self, explanation: SymbolExplanation) -> str:
        """Format explanation for LLM consumption.

        This format is optimized for use as context in LLM prompts.

        Args:
            explanation: SymbolExplanation to format

        Returns:
            LLM-friendly string representation
        """
        lines = []

        # Symbol info
        lines.append(f"Symbol: {explanation.name}")
        lines.append(f"Type: {explanation.kind}")
        lines.append(f"File: {explanation.file_path}")
        lines.append(f"Lines: {explanation.line_start}-{explanation.line_end}")
        lines.append("")
        lines.append(f"Signature: {explanation.signature}")
        lines.append("")

        # Documentation
        if explanation.docstring:
            lines.append("Documentation:")
            lines.append(explanation.docstring)
            lines.append("")

        # Call graph summary
        if explanation.callers:
            caller_names = [c.get("name", "?") for c in explanation.callers[:5]]
            lines.append(f"Called by: {', '.join(caller_names)}")

        if explanation.callees:
            callee_names = [c.get("name", "?") for c in explanation.callees[:5]]
            lines.append(f"Calls: {', '.join(callee_names)}")

        if explanation.callers or explanation.callees:
            lines.append("")

        # Context
        if explanation.context:
            lines.append("Additional context:")
            lines.append(explanation.context[:500])  # Truncate
            lines.append("")

        # Related symbols
        if explanation.related:
            related_names = [r.get("name", "?") for r in explanation.related[:5]]
            lines.append(f"Related symbols: {', '.join(related_names)}")

        return "\n".join(lines)


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """Cache entry for explanation results."""

    explanation: SymbolExplanation
    timestamp: float
    config_hash: str


# =============================================================================
# Main Tool
# =============================================================================


class ExplainCodeTool:
    """MCP tool for explaining code symbols.

    This tool combines symbol lookup, call graph traversal, documentation
    extraction, and context retrieval to provide comprehensive explanations
    of code symbols.
    """

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        parser: Any,
        config: Optional[ExplainCodeConfig] = None,
    ):
        """Initialize explain_code tool.

        Args:
            graph_driver: Neo4j driver for CODE_* graph
            retriever: TriHybridRetriever for context
            parser: ASTParser for docstring extraction
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.parser = parser
        self.config = config or ExplainCodeConfig()

        # Initialize components
        self._lookup = SymbolLookup(graph_driver)
        self._traverser = CallGraphTraverser(graph_driver)
        self._doc_extractor = DocumentationExtractor(parser)
        self._context_retriever = CodeContextRetriever(retriever)
        self._formatter = ExplanationFormatter()

        # Cache
        self._cache: dict[str, CacheEntry] = {}

    def explain(
        self,
        symbol_id: str,
        config: Optional[ExplainCodeConfig] = None,
    ) -> SymbolExplanation:
        """Explain a code symbol.

        Args:
            symbol_id: SCIP symbol ID (e.g., "scip-python myapp module/Class#method.")
            config: Optional config override

        Returns:
            SymbolExplanation with full context

        Raises:
            InvalidSymbolIDError: If symbol ID is invalid
            SymbolNotFoundError: If symbol not found
        """
        cfg = config or self.config

        # Check cache
        cache_key = self._make_cache_key(symbol_id, cfg)
        cached = self._get_cached(cache_key, cfg.cache_ttl_seconds)
        if cached:
            return cached

        # Lookup symbol
        symbol_props = self._lookup.lookup(symbol_id)

        # Build explanation
        explanation = SymbolExplanation(
            symbol_id=symbol_id,
            name=symbol_props.get("name", ""),
            kind=symbol_props.get("kind", "unknown"),
            signature=symbol_props.get("signature", ""),
            file_path=symbol_props.get("file_path", ""),
            line_start=symbol_props.get("line_start", 0),
            line_end=symbol_props.get("line_end", 0),
            docstring=symbol_props.get("docstring"),
        )

        # Get callers if configured
        if cfg.include_callers:
            try:
                explanation.callers = self._traverser.get_callers(
                    symbol_id, depth=cfg.depth
                )
            except Exception as e:
                logger.warning(f"Error getting callers: {e}")
                explanation.callers = []

        # Get callees if configured
        if cfg.include_callees:
            try:
                explanation.callees = self._traverser.get_callees(
                    symbol_id, depth=cfg.depth
                )
            except Exception as e:
                logger.warning(f"Error getting callees: {e}")
                explanation.callees = []

        # Get docstring from source if not in graph
        if not explanation.docstring and explanation.file_path:
            try:
                explanation.docstring = self._doc_extractor.extract(
                    file_path=Path(explanation.file_path),
                    symbol_name=explanation.name,
                    line_start=explanation.line_start,
                )
            except Exception as e:
                logger.warning(f"Error extracting docstring: {e}")

        # Get usages if configured
        if cfg.include_usages:
            try:
                explanation.usages = self._context_retriever.retrieve_usages(
                    symbol_name=explanation.name,
                    max_usages=cfg.max_usages,
                )
            except Exception as e:
                logger.warning(f"Error retrieving usages: {e}")
                explanation.usages = []

        # Get related symbols if configured
        if cfg.include_related:
            try:
                explanation.related = self._context_retriever.retrieve_related(
                    symbol_id=symbol_id,
                    symbol_name=explanation.name,
                    max_related=cfg.max_related,
                )
            except Exception as e:
                logger.warning(f"Error retrieving related: {e}")
                explanation.related = []

        # Get additional context
        try:
            explanation.context = self._context_retriever.retrieve(
                symbol_id=symbol_id,
                symbol_name=explanation.name,
                max_results=3,
            )
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")

        # Cache result
        self._cache_result(cache_key, explanation, cfg)

        return explanation

    def explain_formatted(
        self,
        symbol_id: str,
        format: str = "llm",
        config: Optional[ExplainCodeConfig] = None,
    ) -> str:
        """Explain a symbol and return formatted output.

        Args:
            symbol_id: SCIP symbol ID
            format: Output format ("json", "markdown", "llm")
            config: Optional config override

        Returns:
            Formatted explanation string
        """
        explanation = self.explain(symbol_id, config)

        if format == "json":
            return self._formatter.format_json(explanation)
        elif format == "markdown":
            return self._formatter.format_markdown(explanation)
        else:
            return self._formatter.format_for_llm(explanation)

    def format_for_llm(self, explanation: SymbolExplanation) -> str:
        """Format explanation for LLM consumption.

        Args:
            explanation: SymbolExplanation to format

        Returns:
            LLM-friendly string
        """
        return self._formatter.format_for_llm(explanation)

    def clear_cache(self) -> None:
        """Clear the explanation cache."""
        self._cache.clear()

    def _make_cache_key(self, symbol_id: str, config: ExplainCodeConfig) -> str:
        """Create cache key from symbol ID and config."""
        config_str = (
            f"{config.depth}:{config.include_callers}:{config.include_callees}:"
            f"{config.include_usages}:{config.max_usages}:"
            f"{config.include_related}:{config.max_related}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{symbol_id}:{config_hash}"

    def _get_cached(
        self,
        cache_key: str,
        ttl_seconds: int,
    ) -> Optional[SymbolExplanation]:
        """Get cached explanation if valid."""
        if cache_key not in self._cache:
            return None

        entry = self._cache[cache_key]
        age = time.time() - entry.timestamp

        if age > ttl_seconds:
            del self._cache[cache_key]
            return None

        return entry.explanation

    def _cache_result(
        self,
        cache_key: str,
        explanation: SymbolExplanation,
        config: ExplainCodeConfig,
    ) -> None:
        """Cache an explanation result."""
        config_str = (
            f"{config.depth}:{config.include_callers}:{config.include_callees}:"
            f"{config.include_usages}:{config.max_usages}:"
            f"{config.include_related}:{config.max_related}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        self._cache[cache_key] = CacheEntry(
            explanation=explanation,
            timestamp=time.time(),
            config_hash=config_hash,
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_explain_code_tool(
    graph_driver: Any,
    retriever: Any,
    parser: Any,
    config: Optional[ExplainCodeConfig] = None,
) -> ExplainCodeTool:
    """Create an explain_code tool.

    Args:
        graph_driver: Neo4j driver for CODE_* graph
        retriever: TriHybridRetriever for context
        parser: ASTParser for docstring extraction
        config: Optional configuration

    Returns:
        Configured ExplainCodeTool
    """
    return ExplainCodeTool(
        graph_driver=graph_driver,
        retriever=retriever,
        parser=parser,
        config=config,
    )
