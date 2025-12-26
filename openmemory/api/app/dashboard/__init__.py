"""Dashboard templates and components.

This module provides dashboard templates per section 16 (FR-013):
- Memory browser for exploring stored memories
- Graph explorer for visualizing relationships
- Metrics dashboard for system health and usage
"""

from .templates import (
    DashboardConfig,
    Widget,
    WidgetType,
    Dashboard,
    MemoryBrowserWidget,
    GraphExplorerWidget,
    MetricsWidget,
    TimeSeriesWidget,
    TableWidget,
    DashboardBuilder,
    create_memory_browser,
    create_graph_explorer,
    create_metrics_dashboard,
)

__all__ = [
    "DashboardConfig",
    "Widget",
    "WidgetType",
    "Dashboard",
    "MemoryBrowserWidget",
    "GraphExplorerWidget",
    "MetricsWidget",
    "TimeSeriesWidget",
    "TableWidget",
    "DashboardBuilder",
    "create_memory_browser",
    "create_graph_explorer",
    "create_metrics_dashboard",
]
