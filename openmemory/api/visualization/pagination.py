"""Cursor-based pagination for graph exports.

This module provides:
- Cursor encoding/decoding for stable pagination
- PageInfo for navigation
- PaginatedResult for paginated exports
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Exceptions
# =============================================================================


class PaginationError(Exception):
    """Raised when pagination fails."""

    pass


# =============================================================================
# Cursor
# =============================================================================


@dataclass
class Cursor:
    """Cursor for pagination state.

    Attributes:
        position: Current position in result set.
        sort_key: Key used for stable sorting.
        filters: Optional filter state to preserve.
    """

    position: int
    sort_key: str
    filters: dict[str, Any] = field(default_factory=dict)


class CursorEncoder:
    """Encode and decode cursors for pagination.

    Uses URL-safe base64 encoding of JSON for cursor strings.
    """

    def encode(self, cursor: Cursor) -> str:
        """Encode cursor to URL-safe string.

        Args:
            cursor: Cursor to encode.

        Returns:
            URL-safe base64 encoded string.
        """
        data = {
            "p": cursor.position,
            "k": cursor.sort_key,
        }
        if cursor.filters:
            data["f"] = cursor.filters

        json_str = json.dumps(data, separators=(",", ":"))
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
        return encoded

    def decode(self, encoded: str) -> Cursor:
        """Decode cursor from string.

        Args:
            encoded: Encoded cursor string.

        Returns:
            Decoded Cursor.

        Raises:
            PaginationError: If cursor is invalid.
        """
        try:
            # Add padding if needed
            padding = 4 - len(encoded) % 4
            if padding != 4:
                encoded += "=" * padding

            decoded = base64.urlsafe_b64decode(encoded).decode()
            data = json.loads(decoded)

            return Cursor(
                position=data["p"],
                sort_key=data["k"],
                filters=data.get("f", {}),
            )
        except Exception as e:
            raise PaginationError(f"Invalid cursor: {e}") from e


# =============================================================================
# Page Info
# =============================================================================


@dataclass
class PageInfo:
    """Information about a page of results.

    Attributes:
        has_next_page: Whether there are more results after this page.
        has_previous_page: Whether there are results before this page.
        start_cursor: Cursor to the first item in this page.
        end_cursor: Cursor to the last item in this page.
        total_count: Total number of items across all pages.
    """

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_next_page": self.has_next_page,
            "has_previous_page": self.has_previous_page,
            "start_cursor": self.start_cursor,
            "end_cursor": self.end_cursor,
            "total_count": self.total_count,
        }


# =============================================================================
# Page
# =============================================================================


@dataclass
class Page:
    """A page of results.

    Attributes:
        items: Items in this page.
        page_info: Page navigation info.
    """

    items: list[dict[str, Any]]
    page_info: PageInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": self.items,
            "page_info": self.page_info.to_dict(),
        }


# =============================================================================
# Paginated Result
# =============================================================================


@dataclass
class PaginatedResult:
    """Paginated graph export result.

    Attributes:
        nodes: Nodes in this page.
        edges: Edges in this page.
        page_info: Page navigation info.
    """

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    page_info: PageInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "page_info": self.page_info.to_dict(),
        }
