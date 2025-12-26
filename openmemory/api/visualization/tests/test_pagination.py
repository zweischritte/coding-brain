"""Tests for cursor-based pagination.

This module tests:
- Cursor encoding and decoding
- Page navigation (next/prev)
- Paginated graph export
- Edge cases (empty results, single page)
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
    MemoryGraphStore,
)
from openmemory.api.visualization.config import ExportConfig, PaginationConfig
from openmemory.api.visualization.pagination import (
    Cursor,
    CursorEncoder,
    Page,
    PageInfo,
    PaginatedResult,
    PaginationError,
)


class TestCursor:
    """Test Cursor dataclass."""

    def test_create_cursor(self):
        """Create cursor with position and sort key."""
        cursor = Cursor(position=10, sort_key="node_id_10")
        assert cursor.position == 10
        assert cursor.sort_key == "node_id_10"

    def test_cursor_equality(self):
        """Cursors with same values are equal."""
        cursor1 = Cursor(position=5, sort_key="key")
        cursor2 = Cursor(position=5, sort_key="key")
        assert cursor1 == cursor2

    def test_cursor_inequality(self):
        """Cursors with different values are not equal."""
        cursor1 = Cursor(position=5, sort_key="key1")
        cursor2 = Cursor(position=5, sort_key="key2")
        assert cursor1 != cursor2

    def test_cursor_with_filters(self):
        """Cursor can store filter state."""
        cursor = Cursor(
            position=0,
            sort_key="start",
            filters={"node_type": "CODE_FILE"},
        )
        assert cursor.filters["node_type"] == "CODE_FILE"


class TestCursorEncoder:
    """Test CursorEncoder for encoding/decoding."""

    def test_encode_cursor(self):
        """Encode cursor to string."""
        encoder = CursorEncoder()
        cursor = Cursor(position=10, sort_key="node_id")
        encoded = encoder.encode(cursor)

        assert isinstance(encoded, str)
        # Should be base64
        assert encoded.isascii()

    def test_decode_cursor(self):
        """Decode cursor from string."""
        encoder = CursorEncoder()
        original = Cursor(position=42, sort_key="test_key")
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        assert decoded.position == 42
        assert decoded.sort_key == "test_key"

    def test_encode_decode_roundtrip(self):
        """Cursor survives encode/decode roundtrip."""
        encoder = CursorEncoder()
        original = Cursor(
            position=100,
            sort_key="complex/key/with/slashes",
            filters={"type": "value"},
        )
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        assert decoded.position == original.position
        assert decoded.sort_key == original.sort_key
        assert decoded.filters == original.filters

    def test_decode_invalid_cursor_raises(self):
        """Decoding invalid cursor raises error."""
        encoder = CursorEncoder()
        with pytest.raises(PaginationError, match="Invalid cursor"):
            encoder.decode("not_a_valid_cursor")

    def test_decode_malformed_json_raises(self):
        """Decoding malformed JSON raises error."""
        encoder = CursorEncoder()
        # Base64 of invalid JSON
        invalid = base64.b64encode(b"not json").decode()
        with pytest.raises(PaginationError, match="Invalid cursor"):
            encoder.decode(invalid)

    def test_encoded_cursor_is_url_safe(self):
        """Encoded cursors are URL-safe."""
        encoder = CursorEncoder()
        cursor = Cursor(
            position=999,
            sort_key="path/to/file.py",
            filters={"key": "value with spaces"},
        )
        encoded = encoder.encode(cursor)

        # Should not contain URL-unsafe characters
        assert "+" not in encoded
        assert "/" not in encoded
        assert "=" not in encoded or encoded.endswith("=")


class TestPageInfo:
    """Test PageInfo dataclass."""

    def test_create_page_info(self):
        """Create page info with required fields."""
        info = PageInfo(
            has_next_page=True,
            has_previous_page=False,
            start_cursor="start",
            end_cursor="end",
            total_count=100,
        )
        assert info.has_next_page is True
        assert info.has_previous_page is False
        assert info.total_count == 100

    def test_page_info_optional_cursors(self):
        """Page info can have None cursors."""
        info = PageInfo(
            has_next_page=False,
            has_previous_page=False,
            start_cursor=None,
            end_cursor=None,
            total_count=0,
        )
        assert info.start_cursor is None
        assert info.end_cursor is None


class TestPage:
    """Test Page dataclass."""

    def test_create_page(self):
        """Create page with items and page info."""
        items = [{"id": "1"}, {"id": "2"}]
        page_info = PageInfo(
            has_next_page=True,
            has_previous_page=False,
            start_cursor="s",
            end_cursor="e",
            total_count=10,
        )
        page = Page(items=items, page_info=page_info)

        assert len(page.items) == 2
        assert page.page_info.has_next_page is True

    def test_page_to_dict(self):
        """Convert page to dictionary."""
        items = [{"id": "1"}]
        page_info = PageInfo(
            has_next_page=False,
            has_previous_page=False,
            start_cursor="c1",
            end_cursor="c2",
            total_count=1,
        )
        page = Page(items=items, page_info=page_info)
        result = page.to_dict()

        assert "items" in result
        assert "page_info" in result
        assert result["page_info"]["total_count"] == 1

    def test_empty_page(self):
        """Create empty page."""
        page_info = PageInfo(
            has_next_page=False,
            has_previous_page=False,
            start_cursor=None,
            end_cursor=None,
            total_count=0,
        )
        page = Page(items=[], page_info=page_info)

        assert len(page.items) == 0
        assert page.page_info.total_count == 0


class TestPaginatedResult:
    """Test PaginatedResult for paginated exports."""

    def test_create_paginated_result(self):
        """Create paginated result."""
        page_info = PageInfo(
            has_next_page=True,
            has_previous_page=False,
            start_cursor="s",
            end_cursor="e",
            total_count=100,
        )
        result = PaginatedResult(
            nodes=[],
            edges=[],
            page_info=page_info,
        )
        assert result.page_info.total_count == 100

    def test_paginated_result_to_dict(self):
        """Convert result to dictionary."""
        page_info = PageInfo(
            has_next_page=False,
            has_previous_page=False,
            start_cursor=None,
            end_cursor=None,
            total_count=0,
        )
        result = PaginatedResult(
            nodes=[{"id": "n1"}],
            edges=[{"source": "n1", "target": "n2"}],
            page_info=page_info,
        )
        data = result.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "page_info" in data


class TestPaginatedExport:
    """Test paginated graph export."""

    def test_export_first_page(self, large_graph_store: MemoryGraphStore):
        """Export first page of nodes."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20, default_page_size=20),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated(first=20)

        # Should return paginated result
        assert result.page_info.total_count == 600  # All nodes
        assert len(result.nodes) == 20
        assert result.page_info.has_next_page is True
        assert result.page_info.has_previous_page is False

    def test_export_with_cursor(self, large_graph_store: MemoryGraphStore):
        """Export with cursor for next page."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)

        # Get first page
        first_page = exporter.export_paginated()
        assert first_page.page_info.end_cursor is not None

        # Get second page
        second_page = exporter.export_paginated(
            after=first_page.page_info.end_cursor,
        )

        # Should have different items
        first_ids = {n["id"] for n in first_page.nodes}
        second_ids = {n["id"] for n in second_page.nodes}
        assert first_ids.isdisjoint(second_ids)

        # Second page should have previous
        assert second_page.page_info.has_previous_page is True

    def test_export_last_page(self, large_graph_store: MemoryGraphStore):
        """Export last page has no next."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=100, max_page_size=100, default_page_size=100),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)

        # Navigate to last page (600 items / 100 per page = 6 pages)
        cursor = None
        page = None
        for _ in range(10):  # Max iterations
            page = exporter.export_paginated(after=cursor, first=100)
            if not page.page_info.has_next_page:
                break
            cursor = page.page_info.end_cursor

        assert page is not None
        assert page.page_info.has_next_page is False

    def test_export_with_custom_page_size(self, large_graph_store: MemoryGraphStore):
        """Export with custom page size."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=50),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated(first=50)

        assert len(result.nodes) == 50

    def test_export_respects_max_page_size(self, large_graph_store: MemoryGraphStore):
        """Export respects maximum page size."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=50, max_page_size=100),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)

        # Request more than max
        result = exporter.export_paginated(first=200)

        # Should cap at max
        assert len(result.nodes) <= 100

    def test_export_empty_graph(self, memory_graph_store: MemoryGraphStore):
        """Export empty graph returns empty page."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=memory_graph_store, config=config)
        result = exporter.export_paginated()

        assert len(result.nodes) == 0
        assert result.page_info.total_count == 0
        assert result.page_info.has_next_page is False
        assert result.page_info.has_previous_page is False


class TestPaginatedExportWithFilters:
    """Test paginated export with filters applied."""

    def test_paginate_filtered_nodes(self, large_graph_store: MemoryGraphStore):
        """Pagination applies to filtered results."""
        from openmemory.api.visualization.config import FilterConfig
        from openmemory.api.visualization.graph_export import GraphExporter

        filter_config = FilterConfig(node_types={CodeNodeType.FILE})
        config = ExportConfig(
            filter=filter_config,
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated()

        # Only file nodes
        assert result.page_info.total_count == 100
        for node in result.nodes:
            assert node["type"] == "CODE_FILE"

    def test_cursor_preserves_filters(self, large_graph_store: MemoryGraphStore):
        """Cursor maintains filter state across pages."""
        from openmemory.api.visualization.config import FilterConfig
        from openmemory.api.visualization.graph_export import GraphExporter

        filter_config = FilterConfig(node_types={CodeNodeType.FILE})
        config = ExportConfig(
            filter=filter_config,
            pagination=PaginationConfig(page_size=10),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)

        # Get first page
        first_page = exporter.export_paginated()

        # Get second page
        second_page = exporter.export_paginated(
            after=first_page.page_info.end_cursor,
        )

        # All items should still be filtered
        for node in second_page.nodes:
            assert node["type"] == "CODE_FILE"


class TestPaginationEdgeCases:
    """Test edge cases in pagination."""

    def test_single_item_page(self):
        """Handle single item in results."""
        from openmemory.api.visualization.graph_export import GraphExporter

        store = MemoryGraphStore()
        store.add_node(CodeNode(
            node_type=CodeNodeType.FILE,
            id="only-file",
            properties={"name": "only.py"},
        ))

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=store, config=config)
        result = exporter.export_paginated()

        assert len(result.nodes) == 1
        assert result.page_info.total_count == 1
        assert result.page_info.has_next_page is False
        assert result.page_info.has_previous_page is False

    def test_exact_page_size_items(self):
        """Handle exactly page_size items."""
        from openmemory.api.visualization.graph_export import GraphExporter

        store = MemoryGraphStore()
        for i in range(20):
            store.add_node(CodeNode(
                node_type=CodeNodeType.FILE,
                id=f"file-{i}",
                properties={"name": f"file{i}.py"},
            ))

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=store, config=config)
        result = exporter.export_paginated()

        assert len(result.nodes) == 20
        assert result.page_info.total_count == 20
        assert result.page_info.has_next_page is False

    def test_invalid_cursor_raises(self, large_graph_store: MemoryGraphStore):
        """Invalid cursor raises error."""
        from openmemory.api.visualization.graph_export import GraphExporter

        exporter = GraphExporter(driver=large_graph_store)
        with pytest.raises(PaginationError):
            exporter.export_paginated(after="invalid_cursor")

    def test_page_size_one(self, large_graph_store: MemoryGraphStore):
        """Handle page size of 1."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=1, max_page_size=1, default_page_size=1),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated(first=1)

        assert len(result.nodes) == 1
        assert result.page_info.has_next_page is True


class TestPaginatedJSON:
    """Test paginated JSON output."""

    def test_paginated_json_structure(self, large_graph_store: MemoryGraphStore):
        """Paginated result has correct JSON structure."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated()
        data = result.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert "page_info" in data
        assert "has_next_page" in data["page_info"]
        assert "has_previous_page" in data["page_info"]
        assert "start_cursor" in data["page_info"]
        assert "end_cursor" in data["page_info"]
        assert "total_count" in data["page_info"]

    def test_paginated_json_serializable(self, large_graph_store: MemoryGraphStore):
        """Paginated result is JSON serializable."""
        from openmemory.api.visualization.graph_export import GraphExporter

        config = ExportConfig(
            pagination=PaginationConfig(page_size=20),
        )
        exporter = GraphExporter(driver=large_graph_store, config=config)
        result = exporter.export_paginated()
        data = result.to_dict()

        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["page_info"]["total_count"] == 600
