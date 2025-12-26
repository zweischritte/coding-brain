"""Graph Visualization Module.

This module provides graph export and visualization capabilities:
- Export formats: JSON, DOT (Graphviz), Mermaid
- Hierarchical code graph JSON schema
- Configurable depth and filters for traversal
- Cursor-based pagination for large graphs

Integration points:
- openmemory.api.indexing.graph_projection (CodeNode, CodeEdge, Neo4jDriver)
- openmemory.api.tools.call_graph (find_callers, find_callees)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmemory.api.visualization.config import (
        ExportConfig,
        ExportFormat,
        FilterConfig,
        PaginationConfig,
        StyleConfig,
    )
    from openmemory.api.visualization.formatters import (
        DOTFormatter,
        Formatter,
        JSONFormatter,
        MermaidFormatter,
    )
    from openmemory.api.visualization.graph_export import (
        ExportResult,
        GraphExporter,
        GraphExportError,
        InvalidFormatError,
        TraversalError,
    )
    from openmemory.api.visualization.pagination import (
        Cursor,
        CursorEncoder,
        Page,
        PageInfo,
        PaginatedResult,
        PaginationError,
    )
    from openmemory.api.visualization.schema import (
        HierarchicalGraph,
        HierarchicalNode,
        SchemaValidationError,
        SchemaValidator,
    )

__all__ = [
    # Config
    "ExportConfig",
    "ExportFormat",
    "FilterConfig",
    "PaginationConfig",
    "StyleConfig",
    # Formatters
    "DOTFormatter",
    "Formatter",
    "JSONFormatter",
    "MermaidFormatter",
    # Graph Export
    "ExportResult",
    "GraphExporter",
    "GraphExportError",
    "InvalidFormatError",
    "TraversalError",
    # Pagination
    "Cursor",
    "CursorEncoder",
    "Page",
    "PageInfo",
    "PaginatedResult",
    "PaginationError",
    # Schema
    "HierarchicalGraph",
    "HierarchicalNode",
    "SchemaValidationError",
    "SchemaValidator",
]

_MODULE_MAP = {
    # Config
    "ExportConfig": "config",
    "ExportFormat": "config",
    "FilterConfig": "config",
    "PaginationConfig": "config",
    "StyleConfig": "config",
    # Formatters
    "DOTFormatter": "formatters",
    "Formatter": "formatters",
    "JSONFormatter": "formatters",
    "MermaidFormatter": "formatters",
    # Graph Export
    "ExportResult": "graph_export",
    "GraphExporter": "graph_export",
    "GraphExportError": "graph_export",
    "InvalidFormatError": "graph_export",
    "TraversalError": "graph_export",
    # Pagination
    "Cursor": "pagination",
    "CursorEncoder": "pagination",
    "Page": "pagination",
    "PageInfo": "pagination",
    "PaginatedResult": "pagination",
    "PaginationError": "pagination",
    # Schema
    "HierarchicalGraph": "schema",
    "HierarchicalNode": "schema",
    "SchemaValidationError": "schema",
    "SchemaValidator": "schema",
}


def __getattr__(name: str):
    if name in _MODULE_MAP:
        module_name = _MODULE_MAP[name]
        module = importlib.import_module(f"openmemory.api.visualization.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
