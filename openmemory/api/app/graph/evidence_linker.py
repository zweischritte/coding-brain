"""
Evidence Linker for Memory-Code Links.

Simplified approach to linking memories to code:
- 2 detection strategies: explicit refs + entity name matching
- Tags-based storage (reuse existing code_refs infrastructure)
- No new graph edge types

This module provides:
- CodeLink dataclass for representing memory-code links
- find_code_links_for_memory() for detecting links
- search_code_symbols_by_name() for entity-to-symbol matching
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CodeLink:
    """
    Represents a link between a memory and source code.

    Attributes:
        file_path: Path to the source file
        line_start: Start line number (1-indexed)
        line_end: End line number (1-indexed, inclusive)
        symbol_id: SCIP-compatible symbol identifier
        symbol_name: Human-readable symbol name
        link_source: How this link was created: "explicit" | "entity_match"
    """

    file_path: str
    symbol_name: str
    link_source: str  # "explicit" | "entity_match"
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    symbol_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dict format for code_refs storage.

        Returns:
            Dict with file_path, line_start, line_end, symbol_id, link_source
            Only includes fields that have values.
        """
        result: dict[str, Any] = {
            "file_path": self.file_path,
            "link_source": self.link_source,
        }

        if self.line_start is not None:
            result["line_start"] = self.line_start
        if self.line_end is not None:
            result["line_end"] = self.line_end
        if self.symbol_id is not None:
            result["symbol_id"] = self.symbol_id

        return result


async def search_code_symbols_by_name(
    name: str,
    neo4j_driver,
    limit: int = 10,
) -> list[dict]:
    """
    Search CODE_SYMBOL nodes by name.

    Performs case-insensitive search for symbols matching the given name.
    Exact matches are ranked higher than partial matches.

    Args:
        name: Symbol name to search for
        neo4j_driver: Neo4j driver instance
        limit: Maximum number of results to return

    Returns:
        List of dicts with symbol_id, name, file_path, line_start, line_end
    """
    query = """
    MATCH (s:CODE_SYMBOL)
    WHERE toLower(s.name) CONTAINS toLower($name)
       OR toLower(s.qualifiedName) CONTAINS toLower($name)
    RETURN s.scip_id as symbol_id,
           s.name as name,
           s.filePath as file_path,
           s.line_start as line_start,
           s.line_end as line_end
    ORDER BY
        CASE WHEN toLower(s.name) = toLower($name) THEN 0 ELSE 1 END,
        s.name
    LIMIT $limit
    """

    with neo4j_driver.session() as session:
        result = session.run(query, name=name, limit=limit)
        return [dict(record) for record in result]


async def find_code_links_for_memory(
    memory: dict,
    neo4j_driver,
) -> list[CodeLink]:
    """
    Find code links for a memory using 2-strategy detection.

    Strategy 1: Return existing code_refs as explicit links
    Strategy 2: Match entity name to symbol names (if no explicit refs)

    Args:
        memory: Memory dict with id, text, metadata
        neo4j_driver: Neo4j driver instance

    Returns:
        List of CodeLink objects representing memory-code links
    """
    links: list[CodeLink] = []

    metadata = memory.get("metadata", {})
    if metadata is None:
        metadata = {}

    # Strategy 1: Explicit refs (already in memory)
    existing_refs = metadata.get("code_refs", [])
    if existing_refs is None:
        existing_refs = []

    for ref in existing_refs:
        links.append(CodeLink(
            file_path=ref.get("file_path", ""),
            line_start=ref.get("line_start"),
            line_end=ref.get("line_end"),
            symbol_id=ref.get("symbol_id"),
            symbol_name=ref.get("symbol_name", ""),
            link_source="explicit",
        ))

    # Strategy 2: Entity name matching (only if no explicit refs)
    entity = metadata.get("entity")
    if entity and not existing_refs:
        matches = await search_code_symbols_by_name(entity, neo4j_driver)
        for match in matches[:3]:  # Top 3 only
            links.append(CodeLink(
                file_path=match.get("file_path", ""),
                line_start=match.get("line_start"),
                line_end=match.get("line_end"),
                symbol_id=match.get("symbol_id"),
                symbol_name=match.get("name", ""),
                link_source="entity_match",
            ))

    return links
