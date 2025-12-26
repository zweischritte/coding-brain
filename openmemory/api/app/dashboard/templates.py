"""Dashboard templates and components.

This module provides dashboard templates per section 16 (FR-013):
- Memory browser for exploring stored memories
- Graph explorer for visualizing relationships
- Metrics dashboard for system health and usage
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DashboardConfig:
    """Configuration for a dashboard."""

    title: str = "Dashboard"
    refresh_interval_s: int = 30
    theme: str = "light"  # light, dark
    grid_columns: int = 12
    grid_row_height: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "refresh_interval_s": self.refresh_interval_s,
            "theme": self.theme,
            "grid_columns": self.grid_columns,
            "grid_row_height": self.grid_row_height,
        }


# ============================================================================
# Widget Types
# ============================================================================


class WidgetType(str, Enum):
    """Types of dashboard widgets."""

    TABLE = "table"
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    METRICS = "metrics"
    MEMORY_BROWSER = "memory_browser"
    GRAPH_EXPLORER = "graph_explorer"


# ============================================================================
# Base Widget
# ============================================================================


@dataclass
class Widget:
    """Base class for dashboard widgets."""

    widget_id: str
    title: str
    widget_type: WidgetType
    row: int = 0
    col: int = 0
    width: int = 6
    height: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "widget_id": self.widget_id,
            "title": self.title,
            "widget_type": self.widget_type.value,
            "row": self.row,
            "col": self.col,
            "width": self.width,
            "height": self.height,
        }


# ============================================================================
# Specialized Widgets
# ============================================================================


@dataclass
class MemoryBrowserWidget(Widget):
    """Widget for browsing memories."""

    filters: dict[str, Any] = field(default_factory=dict)
    sort_by: str = "created_at"
    sort_order: str = "desc"
    page_size: int = 20
    show_pagination: bool = True
    show_search: bool = True
    show_filters: bool = True

    def __init__(
        self,
        widget_id: str,
        title: str,
        row: int = 0,
        col: int = 0,
        width: int = 12,
        height: int = 6,
        filters: dict[str, Any] | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        page_size: int = 20,
        show_pagination: bool = True,
        show_search: bool = True,
        show_filters: bool = True,
    ):
        super().__init__(
            widget_id=widget_id,
            title=title,
            widget_type=WidgetType.MEMORY_BROWSER,
            row=row,
            col=col,
            width=width,
            height=height,
        )
        self.filters = filters or {}
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.page_size = page_size
        self.show_pagination = show_pagination
        self.show_search = show_search
        self.show_filters = show_filters

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "filters": self.filters,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "page_size": self.page_size,
            "show_pagination": self.show_pagination,
            "show_search": self.show_search,
            "show_filters": self.show_filters,
        })
        return d


@dataclass
class GraphExplorerWidget(Widget):
    """Widget for exploring the knowledge graph."""

    layout: str = "force"  # force, circular, hierarchical, tree
    focus_node_id: str | None = None
    depth: int = 3
    node_color_by: str | None = None
    edge_color_by: str | None = None
    show_labels: bool = True
    show_legend: bool = True
    enable_zoom: bool = True
    enable_pan: bool = True

    def __init__(
        self,
        widget_id: str,
        title: str,
        row: int = 0,
        col: int = 0,
        width: int = 12,
        height: int = 8,
        layout: str = "force",
        focus_node_id: str | None = None,
        depth: int = 3,
        node_color_by: str | None = None,
        edge_color_by: str | None = None,
        show_labels: bool = True,
        show_legend: bool = True,
        enable_zoom: bool = True,
        enable_pan: bool = True,
    ):
        super().__init__(
            widget_id=widget_id,
            title=title,
            widget_type=WidgetType.GRAPH_EXPLORER,
            row=row,
            col=col,
            width=width,
            height=height,
        )
        self.layout = layout
        self.focus_node_id = focus_node_id
        self.depth = depth
        self.node_color_by = node_color_by
        self.edge_color_by = edge_color_by
        self.show_labels = show_labels
        self.show_legend = show_legend
        self.enable_zoom = enable_zoom
        self.enable_pan = enable_pan

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "layout": self.layout,
            "focus_node_id": self.focus_node_id,
            "depth": self.depth,
            "node_color_by": self.node_color_by,
            "edge_color_by": self.edge_color_by,
            "show_labels": self.show_labels,
            "show_legend": self.show_legend,
            "enable_zoom": self.enable_zoom,
            "enable_pan": self.enable_pan,
        })
        return d


@dataclass
class MetricsWidget(Widget):
    """Widget for displaying metrics."""

    metrics: list[str] = field(default_factory=list)
    format_spec: dict[str, str] = field(default_factory=dict)
    refresh_interval_s: int = 30
    show_trend: bool = False
    compact: bool = False

    def __init__(
        self,
        widget_id: str,
        title: str,
        metrics: list[str],
        row: int = 0,
        col: int = 0,
        width: int = 6,
        height: int = 2,
        format_spec: dict[str, str] | None = None,
        refresh_interval_s: int = 30,
        show_trend: bool = False,
        compact: bool = False,
    ):
        super().__init__(
            widget_id=widget_id,
            title=title,
            widget_type=WidgetType.METRICS,
            row=row,
            col=col,
            width=width,
            height=height,
        )
        self.metrics = metrics
        self.format_spec = format_spec or {}
        self.refresh_interval_s = refresh_interval_s
        self.show_trend = show_trend
        self.compact = compact

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "metrics": self.metrics,
            "format_spec": self.format_spec,
            "refresh_interval_s": self.refresh_interval_s,
            "show_trend": self.show_trend,
            "compact": self.compact,
        })
        return d


@dataclass
class TimeSeriesWidget(Widget):
    """Widget for displaying time series data."""

    metric: str = ""
    time_range_hours: int = 24
    aggregation: str = "5m"
    chart_type: str = "line"  # line, area, bar
    fill: bool = False
    show_legend: bool = True
    stacked: bool = False

    def __init__(
        self,
        widget_id: str,
        title: str,
        metric: str,
        row: int = 0,
        col: int = 0,
        width: int = 6,
        height: int = 4,
        time_range_hours: int = 24,
        aggregation: str = "5m",
        chart_type: str = "line",
        fill: bool = False,
        show_legend: bool = True,
        stacked: bool = False,
    ):
        super().__init__(
            widget_id=widget_id,
            title=title,
            widget_type=WidgetType.TIMESERIES,
            row=row,
            col=col,
            width=width,
            height=height,
        )
        self.metric = metric
        self.time_range_hours = time_range_hours
        self.aggregation = aggregation
        self.chart_type = chart_type
        self.fill = fill
        self.show_legend = show_legend
        self.stacked = stacked

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "metric": self.metric,
            "time_range_hours": self.time_range_hours,
            "aggregation": self.aggregation,
            "chart_type": self.chart_type,
            "fill": self.fill,
            "show_legend": self.show_legend,
            "stacked": self.stacked,
        })
        return d


@dataclass
class TableWidget(Widget):
    """Widget for displaying tabular data."""

    columns: list[str] = field(default_factory=list)
    column_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    page_size: int = 20
    show_pagination: bool = True
    sortable: bool = True
    filterable: bool = True

    def __init__(
        self,
        widget_id: str,
        title: str,
        columns: list[str],
        row: int = 0,
        col: int = 0,
        width: int = 12,
        height: int = 6,
        column_config: dict[str, dict[str, Any]] | None = None,
        page_size: int = 20,
        show_pagination: bool = True,
        sortable: bool = True,
        filterable: bool = True,
    ):
        super().__init__(
            widget_id=widget_id,
            title=title,
            widget_type=WidgetType.TABLE,
            row=row,
            col=col,
            width=width,
            height=height,
        )
        self.columns = columns
        self.column_config = column_config or {}
        self.page_size = page_size
        self.show_pagination = show_pagination
        self.sortable = sortable
        self.filterable = filterable

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d.update({
            "columns": self.columns,
            "column_config": self.column_config,
            "page_size": self.page_size,
            "show_pagination": self.show_pagination,
            "sortable": self.sortable,
            "filterable": self.filterable,
        })
        return d


# ============================================================================
# Dashboard
# ============================================================================


@dataclass
class Dashboard:
    """A dashboard containing multiple widgets."""

    dashboard_id: str
    config: DashboardConfig
    widgets: list[Widget] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_widget(self, widget: Widget) -> None:
        """Add a widget to the dashboard.

        Args:
            widget: The widget to add
        """
        self.widgets.append(widget)

    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard.

        Args:
            widget_id: The widget ID to remove

        Returns:
            True if removed, False if not found
        """
        for i, widget in enumerate(self.widgets):
            if widget.widget_id == widget_id:
                del self.widgets[i]
                return True
        return False

    def get_widget(self, widget_id: str) -> Widget | None:
        """Get a widget by ID.

        Args:
            widget_id: The widget ID

        Returns:
            Widget if found, None otherwise
        """
        for widget in self.widgets:
            if widget.widget_id == widget_id:
                return widget
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dashboard_id": self.dashboard_id,
            "config": self.config.to_dict(),
            "widgets": [w.to_dict() for w in self.widgets],
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Dashboard Builder
# ============================================================================


class DashboardBuilder:
    """Builder for creating dashboards with fluent interface."""

    def __init__(
        self,
        title: str = "Dashboard",
        refresh_interval_s: int = 30,
        theme: str = "light",
    ):
        """Initialize the builder.

        Args:
            title: Dashboard title
            refresh_interval_s: Refresh interval
            theme: Dashboard theme
        """
        self._config = DashboardConfig(
            title=title,
            refresh_interval_s=refresh_interval_s,
            theme=theme,
        )
        self._widgets: list[Widget] = []
        self._current_row = 0

    def add_widget(
        self,
        widget_type: WidgetType,
        title: str,
        row: int | None = None,
        col: int = 0,
        width: int = 6,
        height: int = 4,
        **kwargs: Any,
    ) -> "DashboardBuilder":
        """Add a widget to the dashboard.

        Args:
            widget_type: Type of widget
            title: Widget title
            row: Row position (auto-increment if None)
            col: Column position
            width: Widget width in grid units
            height: Widget height in grid units
            **kwargs: Additional widget-specific parameters

        Returns:
            Self for chaining
        """
        widget_id = str(uuid.uuid4())

        if row is None:
            row = self._current_row
            self._current_row += height

        if widget_type == WidgetType.TABLE:
            widget = TableWidget(
                widget_id=widget_id,
                title=title,
                columns=kwargs.get("columns", []),
                row=row,
                col=col,
                width=width,
                height=height,
                column_config=kwargs.get("column_config"),
                page_size=kwargs.get("page_size", 20),
                show_pagination=kwargs.get("show_pagination", True),
            )
        elif widget_type == WidgetType.METRICS:
            widget = MetricsWidget(
                widget_id=widget_id,
                title=title,
                metrics=kwargs.get("metrics", []),
                row=row,
                col=col,
                width=width,
                height=height,
                format_spec=kwargs.get("format_spec"),
                refresh_interval_s=kwargs.get("refresh_interval_s", 30),
            )
        elif widget_type == WidgetType.TIMESERIES:
            widget = TimeSeriesWidget(
                widget_id=widget_id,
                title=title,
                metric=kwargs.get("metric", ""),
                row=row,
                col=col,
                width=width,
                height=height,
                time_range_hours=kwargs.get("time_range_hours", 24),
                aggregation=kwargs.get("aggregation", "5m"),
                chart_type=kwargs.get("chart_type", "line"),
            )
        elif widget_type == WidgetType.MEMORY_BROWSER:
            widget = MemoryBrowserWidget(
                widget_id=widget_id,
                title=title,
                row=row,
                col=col,
                width=width,
                height=height,
                filters=kwargs.get("filters"),
                page_size=kwargs.get("page_size", 20),
            )
        elif widget_type == WidgetType.GRAPH_EXPLORER:
            widget = GraphExplorerWidget(
                widget_id=widget_id,
                title=title,
                row=row,
                col=col,
                width=width,
                height=height,
                layout=kwargs.get("layout", "force"),
                focus_node_id=kwargs.get("focus_node_id"),
                depth=kwargs.get("depth", 3),
            )
        else:
            widget = Widget(
                widget_id=widget_id,
                title=title,
                widget_type=widget_type,
                row=row,
                col=col,
                width=width,
                height=height,
            )

        self._widgets.append(widget)
        return self

    def add_memory_browser(
        self,
        title: str = "Memory Browser",
        row: int | None = None,
        col: int = 0,
        width: int = 12,
        height: int = 6,
        **kwargs: Any,
    ) -> "DashboardBuilder":
        """Add a memory browser widget.

        Args:
            title: Widget title
            row: Row position
            col: Column position
            width: Widget width
            height: Widget height
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        return self.add_widget(
            widget_type=WidgetType.MEMORY_BROWSER,
            title=title,
            row=row,
            col=col,
            width=width,
            height=height,
            **kwargs,
        )

    def add_graph_explorer(
        self,
        title: str = "Graph Explorer",
        row: int | None = None,
        col: int = 0,
        width: int = 12,
        height: int = 8,
        **kwargs: Any,
    ) -> "DashboardBuilder":
        """Add a graph explorer widget.

        Args:
            title: Widget title
            row: Row position
            col: Column position
            width: Widget width
            height: Widget height
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        return self.add_widget(
            widget_type=WidgetType.GRAPH_EXPLORER,
            title=title,
            row=row,
            col=col,
            width=width,
            height=height,
            **kwargs,
        )

    def add_metrics(
        self,
        title: str = "Metrics",
        metrics: list[str] | None = None,
        row: int | None = None,
        col: int = 0,
        width: int = 6,
        height: int = 2,
        **kwargs: Any,
    ) -> "DashboardBuilder":
        """Add a metrics widget.

        Args:
            title: Widget title
            metrics: List of metric names
            row: Row position
            col: Column position
            width: Widget width
            height: Widget height
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        return self.add_widget(
            widget_type=WidgetType.METRICS,
            title=title,
            metrics=metrics or [],
            row=row,
            col=col,
            width=width,
            height=height,
            **kwargs,
        )

    def add_timeseries(
        self,
        title: str = "Time Series",
        metric: str = "",
        row: int | None = None,
        col: int = 0,
        width: int = 6,
        height: int = 4,
        **kwargs: Any,
    ) -> "DashboardBuilder":
        """Add a time series widget.

        Args:
            title: Widget title
            metric: Metric name
            row: Row position
            col: Column position
            width: Widget width
            height: Widget height
            **kwargs: Additional parameters

        Returns:
            Self for chaining
        """
        return self.add_widget(
            widget_type=WidgetType.TIMESERIES,
            title=title,
            metric=metric,
            row=row,
            col=col,
            width=width,
            height=height,
            **kwargs,
        )

    def build(self) -> Dashboard:
        """Build the dashboard.

        Returns:
            Configured Dashboard
        """
        dashboard = Dashboard(
            dashboard_id=str(uuid.uuid4()),
            config=self._config,
            widgets=self._widgets.copy(),
        )
        return dashboard


# ============================================================================
# Factory Functions
# ============================================================================


def create_memory_browser(
    user_id: str,
    title: str = "Memory Browser",
) -> Dashboard:
    """Create a memory browser dashboard.

    Args:
        user_id: User ID
        title: Dashboard title

    Returns:
        Dashboard configured for memory browsing
    """
    return (
        DashboardBuilder(title=title)
        .add_memory_browser(
            title="Memories",
            filters={"user_id": user_id},
            page_size=25,
        )
        .build()
    )


def create_graph_explorer(
    user_id: str,
    title: str = "Knowledge Graph",
) -> Dashboard:
    """Create a graph explorer dashboard.

    Args:
        user_id: User ID
        title: Dashboard title

    Returns:
        Dashboard configured for graph exploration
    """
    return (
        DashboardBuilder(title=title)
        .add_graph_explorer(
            title="Knowledge Graph",
            layout="force",
            depth=3,
        )
        .add_metrics(
            title="Graph Stats",
            metrics=["node_count", "edge_count", "density"],
            col=0,
            width=12,
            height=2,
        )
        .build()
    )


def create_metrics_dashboard(
    user_id: str,
    title: str = "System Metrics",
) -> Dashboard:
    """Create a metrics dashboard.

    Args:
        user_id: User ID
        title: Dashboard title

    Returns:
        Dashboard configured for metrics display
    """
    return (
        DashboardBuilder(title=title)
        .add_metrics(
            title="Key Metrics",
            metrics=["total_memories", "queries_today", "active_users"],
            width=12,
            height=2,
        )
        .add_timeseries(
            title="Query Rate",
            metric="queries_per_minute",
            width=6,
            height=4,
        )
        .add_timeseries(
            title="Memory Additions",
            metric="memories_added",
            col=6,
            width=6,
            height=4,
        )
        .add_timeseries(
            title="Latency (p99)",
            metric="latency_p99",
            width=6,
            height=4,
        )
        .add_timeseries(
            title="Error Rate",
            metric="error_rate",
            col=6,
            width=6,
            height=4,
        )
        .build()
    )
