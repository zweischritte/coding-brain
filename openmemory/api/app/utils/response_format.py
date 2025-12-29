"""
Lean Response Formatting for Memory API

Provides optimized response formats for LLM consumption with ~40% token reduction.
Removes debug/re-ranking metadata while preserving essential information.

Design Principles:
- Remove re-ranking debug info (semantic/boost scores, query echo, filter debug)
- Keep essential fields for LLM interpretation (id, memory, score, structured metadata)
- Conditional field inclusion (only include if present/not null)
- Timezone normalization to Europe/Berlin

Usage:
    from app.utils.response_format import format_search_results, format_memory_list
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo


# =============================================================================
# CONSTANTS
# =============================================================================

# Timezone for output
BERLIN_TZ = ZoneInfo("Europe/Berlin")

# =============================================================================
# TIMESTAMP FORMATTING
# =============================================================================

def format_timestamp(dt_str: Optional[str]) -> Optional[str]:
    """
    Format timestamp to lean Berlin-time format.

    Input:  "2025-12-05T01:39:18.187996-08:00"
    Output: "2025-12-05T10:39:18+01:00"

    - Removes microseconds
    - Converts to Europe/Berlin timezone
    - Preserves timezone offset (+01:00 or +02:00)

    Args:
        dt_str: ISO datetime string or None

    Returns:
        Formatted datetime string or None
    """
    if not dt_str:
        return None

    try:
        # Parse ISO format (handles Z suffix and various offsets)
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

        # Ensure timezone-aware (default to UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to Berlin timezone
        berlin_dt = dt.astimezone(BERLIN_TZ)

        # Format without microseconds
        # Using isoformat() and truncating microseconds
        formatted = berlin_dt.replace(microsecond=0).isoformat()

        return formatted
    except (ValueError, TypeError):
        return None


# =============================================================================
# MEMORY RESULT FORMATTING
# =============================================================================

def format_memory_result(
    result: Dict[str, Any],
    include_score: bool = True,
) -> Dict[str, Any]:
    """
    Format a single memory result for lean response.

    Removes:
    - Debug scores (semantic, boost)
    - Nested metadata structure

    Keeps:
    - id: Primary key for chaining (update_memory, delete_memories)
    - memory: Actual content
    - score: Final score (if search result, rounded to 2 decimals)
    - category, scope: Structured memory classification
    - artifact_type, artifact_ref: Codebase location identifiers (if present)
    - entity: Reference object (if present)
    - source, evidence: Provenance fields (if present)
    - tags: Qualitative info (if present and non-empty)
    - created_at: Timestamp in Berlin time
    - updated_at: Timestamp in Berlin time (if present)

    Args:
        result: Raw memory result dict
        include_score: Whether to include score (True for search, False for list)

    Returns:
        Lean formatted result dict
    """
    formatted = {
        "id": result.get("id"),
        "memory": result.get("memory"),
    }

    # Score (only for search results)
    if include_score:
        scores = result.get("scores", {})
        final_score = scores.get("final") if isinstance(scores, dict) else None
        if final_score is not None:
            formatted["score"] = round(final_score, 2)

    # Extract metadata (may be nested or flat)
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Category & scope
    category = metadata.get("category") or result.get("category")
    if category:
        formatted["category"] = category

    scope = metadata.get("scope") or result.get("scope")
    if scope:
        formatted["scope"] = scope

    # Artifact identifiers
    artifact_type = metadata.get("artifact_type") or result.get("artifact_type")
    if artifact_type:
        formatted["artifact_type"] = artifact_type

    artifact_ref = metadata.get("artifact_ref") or result.get("artifact_ref")
    if artifact_ref:
        formatted["artifact_ref"] = artifact_ref

    # Entity (conditional) - stored as "entity" in metadata
    entity = metadata.get("entity") or metadata.get("re") or result.get("entity")
    if entity:
        formatted["entity"] = entity

    # Source (conditional)
    source = metadata.get("source") or result.get("source")
    if source:
        formatted["source"] = source

    # Evidence (conditional)
    evidence = metadata.get("evidence") or result.get("evidence")
    if evidence:
        formatted["evidence"] = evidence

    # Tags (conditional - only if non-empty)
    tags = metadata.get("tags") or result.get("tags")
    if isinstance(tags, list):
        tags = {t: True for t in tags}
    if tags and isinstance(tags, dict) and len(tags) > 0:
        formatted["tags"] = tags

    # Timestamps (formatted to Berlin time)
    created_at = result.get("created_at")
    if created_at:
        formatted["created_at"] = format_timestamp(created_at)

    # Updated_at (conditional - only if present)
    updated_at = result.get("updated_at")
    if updated_at:
        formatted_updated = format_timestamp(updated_at)
        if formatted_updated:
            formatted["updated_at"] = formatted_updated

    return formatted


def format_search_results(
    results: List[Dict[str, Any]],
    verbose: bool = False,
    query: str = None,
    context_applied: Dict[str, Any] = None,
    filters_applied: Dict[str, Any] = None,
    total_candidates: int = None,
) -> Dict[str, Any]:
    """
    Format search results for lean response.

    Default (verbose=False):
        {
            "results": [<lean memory results>]
        }

    Verbose (verbose=True):
        {
            "results": [<full memory results with score breakdown>],
            "query": "...",
            "context_applied": {...},
            "filters_applied": {...},
            "total_candidates": N,
            "returned": N
        }

    Args:
        results: List of raw search results
        verbose: If True, include debug information
        query: Original search query (for verbose mode)
        context_applied: Boost context (for verbose mode)
        filters_applied: Filter settings (for verbose mode)
        total_candidates: Total candidates before filtering (for verbose mode)

    Returns:
        Formatted response dict
    """
    if verbose:
        # Full format with debug info
        return {
            "results": results,  # Keep original format
            "query": query,
            "context_applied": context_applied,
            "filters_applied": filters_applied,
            "total_candidates": total_candidates,
            "returned": len(results),
        }

    # Lean format
    formatted_results = [
        format_memory_result(r, include_score=True)
        for r in results
    ]

    return {"results": formatted_results}


def format_memory_list(
    memories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Format list_memories results for lean response.

    Removes unnecessary fields:
    - hash, user_id, role

    Args:
        memories: List of raw memory dicts

    Returns:
        List of lean formatted memory dicts
    """
    formatted = []

    for memory in memories:
        # Format the memory without score
        lean_memory = format_memory_result(memory, include_score=False)
        formatted.append(lean_memory)

    return formatted


# =============================================================================
# ADD MEMORY RESULT FORMATTING
# =============================================================================

def format_add_memory_result(
    result: Dict[str, Any],
    structured_metadata: Dict[str, Any],
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format a single add_memories result for lean response.

    Consolidates result and structured_metadata into a single flat object.
    Matches list_memories structure for consistency.

    Design principles:
    - No redundancy — each field appears exactly once
    - Consistent with list_memories — same structure, same field names
    - Flat where possible — only nest tags when present
    - Always return created_at — confirms successful storage
    - Entity as direct field

    Args:
        result: Raw result from memory_client.add()
        structured_metadata: Parsed structured metadata from input
        created_at: ISO timestamp string (optional, will use current time if not provided)

    Returns:
        Lean formatted result dict matching list_memories structure
    """
    from datetime import datetime, timezone

    formatted = {
        "id": result.get("id"),
        "memory": result.get("memory"),
    }

    # Category & scope
    category = structured_metadata.get("category")
    if category:
        formatted["category"] = category

    scope = structured_metadata.get("scope")
    if scope:
        formatted["scope"] = scope

    # Artifact identifiers
    artifact_type = structured_metadata.get("artifact_type")
    if artifact_type:
        formatted["artifact_type"] = artifact_type

    artifact_ref = structured_metadata.get("artifact_ref")
    if artifact_ref:
        formatted["artifact_ref"] = artifact_ref

    # Entity (direct field)
    entity = structured_metadata.get("entity")
    if entity:
        formatted["entity"] = entity

    # Tags (only if non-empty)
    tags = structured_metadata.get("tags")
    if tags and isinstance(tags, dict) and len(tags) > 0:
        formatted["tags"] = tags

    # Source and evidence
    source = structured_metadata.get("source")
    if source:
        formatted["source"] = source

    evidence = structured_metadata.get("evidence")
    if evidence:
        formatted["evidence"] = evidence

    # Created_at (always included to confirm successful storage)
    if created_at:
        formatted["created_at"] = format_timestamp(created_at)
    else:
        # Generate current timestamp in Berlin time
        now = datetime.now(timezone.utc)
        formatted["created_at"] = format_timestamp(now.isoformat())

    return formatted


def format_add_memories_response(
    results: List[Dict[str, Any]],
    structured_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Format add_memories response for lean output.

    Single memory:
        Returns the formatted result directly (flat object)

    Batch operations:
        Returns results array with summary

    Args:
        results: List of raw results from memory_client.add()
        structured_metadata: Parsed structured metadata from input

    Returns:
        Formatted response dict
    """
    from datetime import datetime, timezone

    if not results:
        # Handle empty results case (potential bug in client add flow)
        # Still return structured_metadata fields as a single result to indicate parsing worked
        return format_add_memory_result(
            result={"id": None, "memory": None},
            structured_metadata=structured_metadata,
        )

    # Format all results
    formatted_results = []
    for result in results:
        formatted = format_add_memory_result(
            result=result,
            structured_metadata=structured_metadata,
            created_at=None,  # Will use current time
        )
        formatted_results.append(formatted)

    # Single result: return flat object
    if len(formatted_results) == 1:
        return formatted_results[0]

    # Batch: return with summary
    return {
        "results": formatted_results,
        "summary": {
            "total": len(formatted_results),
            "success": len([r for r in formatted_results if r.get("id")]),
            "failed": len([r for r in formatted_results if not r.get("id")]),
        }
    }
