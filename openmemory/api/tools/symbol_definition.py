"""Get Symbol Definition Tool.

This module provides the get_symbol_definition MCP tool:
- SymbolDefinitionConfig: Configuration for the tool
- SymbolLookupInput: Input parameters
- SymbolDefinitionOutput: Result structure
- GetSymbolDefinitionTool: Main tool entry point

Integration points:
- openmemory.api.indexing.graph_projection: CODE_* graph queries
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class SymbolDefinitionError(Exception):
    """Base exception for symbol definition tool errors."""

    pass


class SymbolNotFoundError(SymbolDefinitionError):
    """Raised when a symbol cannot be found."""

    pass


class InvalidInputError(SymbolDefinitionError):
    """Raised when input is invalid."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SymbolDefinitionConfig:
    """Configuration for get_symbol_definition tool.

    Args:
        include_snippet: Include code snippet in output
        snippet_context_lines: Lines of context around definition
        include_docstring: Include docstring in output
    """

    include_snippet: bool = True
    snippet_context_lines: int = 5
    include_docstring: bool = True


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class SymbolLookupInput:
    """Input parameters for get_symbol_definition.

    At least one of symbol_id or symbol_name must be provided.

    Args:
        symbol_id: SCIP symbol ID
        symbol_name: Alternative: symbol name for lookup
        file_path: Optional file path to narrow search
        repo_id: Optional repository ID
    """

    symbol_id: Optional[str] = None
    symbol_name: Optional[str] = None
    file_path: Optional[str] = None
    repo_id: Optional[str] = None


# =============================================================================
# Result Types (MCP Schema Compliant)
# =============================================================================


@dataclass
class CodeLocation:
    """Code location information."""

    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class CodeSymbol:
    """A code symbol.

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
class ResponseMeta:
    """Response metadata.

    Maps to ResponseMeta in MCP schema.
    """

    request_id: str
    degraded_mode: bool = False
    missing_sources: list[str] = field(default_factory=list)


@dataclass
class SymbolDefinitionOutput:
    """Result from get_symbol_definition.

    Maps to SymbolDefinitionOutput in MCP schema.
    """

    symbol: CodeSymbol
    meta: ResponseMeta
    snippet: Optional[str] = None


# =============================================================================
# Main Tool
# =============================================================================


class GetSymbolDefinitionTool:
    """MCP tool for getting symbol definitions.

    Retrieves symbol information including location, signature,
    and optionally code snippets.
    """

    def __init__(
        self,
        graph_driver: Any,
        parser: Optional[Any] = None,
        config: Optional[SymbolDefinitionConfig] = None,
    ):
        """Initialize get_symbol_definition tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            parser: Optional AST parser for snippet extraction
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.parser = parser
        self.config = config or SymbolDefinitionConfig()

    def get_definition(
        self,
        input_data: SymbolLookupInput,
        config: Optional[SymbolDefinitionConfig] = None,
    ) -> SymbolDefinitionOutput:
        """Get symbol definition.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            SymbolDefinitionOutput with symbol details

        Raises:
            InvalidInputError: If input is invalid
            SymbolNotFoundError: If symbol not found
            SymbolDefinitionError: If lookup fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Resolve symbol ID
        symbol_id = self._resolve_symbol_id(input_data)

        # Look up symbol in graph
        try:
            node = self.graph_driver.get_node(symbol_id)
        except Exception as e:
            logger.error(f"Failed to lookup symbol: {e}")
            raise SymbolDefinitionError(f"Lookup failed: {e}") from e

        if node is None:
            raise SymbolNotFoundError(f"Symbol not found: {symbol_id}")

        # Extract properties
        props = node.properties

        # Build CodeSymbol
        symbol = CodeSymbol(
            symbol_id=props.get("scip_id", symbol_id),
            symbol_name=props.get("name", ""),
            symbol_type=props.get("kind", ""),
            signature=props.get("signature"),
            language=props.get("language"),
            file_path=props.get("file_path"),
            line_start=props.get("line_start"),
            line_end=props.get("line_end"),
        )

        # Extract snippet if configured
        snippet = None
        if cfg.include_snippet and symbol.file_path and symbol.line_start:
            snippet = self._extract_snippet(
                file_path=symbol.file_path,
                line_start=symbol.line_start,
                line_end=symbol.line_end or symbol.line_start,
                context_lines=cfg.snippet_context_lines,
            )

        return SymbolDefinitionOutput(
            symbol=symbol,
            snippet=snippet,
            meta=ResponseMeta(request_id=request_id),
        )

    def _validate_input(self, input_data: SymbolLookupInput) -> None:
        """Validate input parameters."""
        if not input_data.symbol_id and not input_data.symbol_name:
            raise InvalidInputError("Either symbol_id or symbol_name is required")

        if input_data.symbol_id is not None and not input_data.symbol_id.strip():
            raise InvalidInputError("symbol_id cannot be empty")

    def _resolve_symbol_id(self, input_data: SymbolLookupInput) -> str:
        """Resolve symbol_id from input."""
        if input_data.symbol_id:
            return input_data.symbol_id

        # TODO: Implement symbol name lookup via graph query
        raise InvalidInputError("symbol_name lookup not yet implemented")

    def _extract_snippet(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        context_lines: int,
    ) -> Optional[str]:
        """Extract code snippet from file.

        Args:
            file_path: Path to source file
            line_start: Start line (1-indexed)
            line_end: End line (1-indexed)
            context_lines: Lines of context to include

        Returns:
            Code snippet string or None if file not found
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None

            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Calculate range with context
            start_idx = max(0, line_start - 1 - context_lines)
            end_idx = min(len(lines), line_end + context_lines)

            # Extract snippet
            snippet_lines = lines[start_idx:end_idx]
            return "".join(snippet_lines)

        except Exception as e:
            logger.warning(f"Failed to extract snippet from {file_path}: {e}")
            return None


# =============================================================================
# Factory Function
# =============================================================================


def create_get_symbol_definition_tool(
    graph_driver: Any,
    parser: Optional[Any] = None,
    config: Optional[SymbolDefinitionConfig] = None,
) -> GetSymbolDefinitionTool:
    """Create a get_symbol_definition tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        parser: Optional AST parser
        config: Optional configuration

    Returns:
        Configured GetSymbolDefinitionTool
    """
    return GetSymbolDefinitionTool(
        graph_driver=graph_driver,
        parser=parser,
        config=config,
    )
