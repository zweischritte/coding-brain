"""Tests for Dashboard templates.

This module tests dashboard templates per section 16 (FR-013):
- Memory browser for exploring stored memories
- Graph explorer for visualizing relationships
- Metrics dashboard for system health and usage
"""

import pytest

from openmemory.api.app.dashboard.templates import (
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


# ============================================================================
# DashboardConfig Tests
# ============================================================================


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DashboardConfig()
        assert config.title == "Dashboard"
        assert config.refresh_interval_s == 30
        assert config.theme == "light"

    def test_custom_config(self):
        """Test custom configuration."""
        config = DashboardConfig(
            title="Custom Dashboard",
            refresh_interval_s=60,
            theme="dark",
        )
        assert config.title == "Custom Dashboard"
        assert config.refresh_interval_s == 60
        assert config.theme == "dark"

    def test_config_to_dict(self):
        """Test config serialization."""
        config = DashboardConfig(title="Test")
        d = config.to_dict()
        assert d["title"] == "Test"
        assert "refresh_interval_s" in d


# ============================================================================
# WidgetType Tests
# ============================================================================


class TestWidgetType:
    """Tests for WidgetType enum."""

    def test_widget_types(self):
        """Test widget type values."""
        assert WidgetType.TABLE.value == "table"
        assert WidgetType.GRAPH.value == "graph"
        assert WidgetType.TIMESERIES.value == "timeseries"
        assert WidgetType.METRICS.value == "metrics"
        assert WidgetType.MEMORY_BROWSER.value == "memory_browser"
        assert WidgetType.GRAPH_EXPLORER.value == "graph_explorer"


# ============================================================================
# Widget Base Tests
# ============================================================================


class TestWidget:
    """Tests for Widget base class."""

    def test_widget_creation(self):
        """Test creating a base widget."""
        widget = Widget(
            widget_id="widget-1",
            title="Test Widget",
            widget_type=WidgetType.TABLE,
        )
        assert widget.widget_id == "widget-1"
        assert widget.title == "Test Widget"
        assert widget.widget_type == WidgetType.TABLE

    def test_widget_with_position(self):
        """Test widget with position."""
        widget = Widget(
            widget_id="widget-2",
            title="Positioned",
            widget_type=WidgetType.GRAPH,
            row=1,
            col=2,
            width=4,
            height=3,
        )
        assert widget.row == 1
        assert widget.col == 2
        assert widget.width == 4
        assert widget.height == 3

    def test_widget_to_dict(self):
        """Test widget serialization."""
        widget = Widget(
            widget_id="widget-3",
            title="Serializable",
            widget_type=WidgetType.METRICS,
        )
        d = widget.to_dict()
        assert d["widget_id"] == "widget-3"
        assert d["title"] == "Serializable"
        assert d["widget_type"] == "metrics"


# ============================================================================
# MemoryBrowserWidget Tests
# ============================================================================


class TestMemoryBrowserWidget:
    """Tests for MemoryBrowserWidget."""

    def test_memory_browser_creation(self):
        """Test creating a memory browser widget."""
        widget = MemoryBrowserWidget(
            widget_id="mb-1",
            title="Memory Browser",
        )
        assert widget.widget_type == WidgetType.MEMORY_BROWSER
        assert widget.page_size == 20

    def test_memory_browser_with_filters(self):
        """Test memory browser with filters."""
        widget = MemoryBrowserWidget(
            widget_id="mb-2",
            title="Filtered Browser",
            filters={"source": "code", "user_id": "user-123"},
            sort_by="created_at",
            sort_order="desc",
        )
        assert widget.filters["source"] == "code"
        assert widget.sort_by == "created_at"
        assert widget.sort_order == "desc"

    def test_memory_browser_pagination(self):
        """Test memory browser pagination settings."""
        widget = MemoryBrowserWidget(
            widget_id="mb-3",
            title="Paginated Browser",
            page_size=50,
            show_pagination=True,
        )
        assert widget.page_size == 50
        assert widget.show_pagination is True

    def test_memory_browser_to_dict(self):
        """Test memory browser serialization."""
        widget = MemoryBrowserWidget(
            widget_id="mb-4",
            title="Serializable Browser",
        )
        d = widget.to_dict()
        assert d["widget_type"] == "memory_browser"
        assert "page_size" in d


# ============================================================================
# GraphExplorerWidget Tests
# ============================================================================


class TestGraphExplorerWidget:
    """Tests for GraphExplorerWidget."""

    def test_graph_explorer_creation(self):
        """Test creating a graph explorer widget."""
        widget = GraphExplorerWidget(
            widget_id="ge-1",
            title="Graph Explorer",
        )
        assert widget.widget_type == WidgetType.GRAPH_EXPLORER
        assert widget.layout == "force"

    def test_graph_explorer_layouts(self):
        """Test different graph layouts."""
        layouts = ["force", "circular", "hierarchical", "tree"]
        for layout in layouts:
            widget = GraphExplorerWidget(
                widget_id=f"ge-{layout}",
                title=f"{layout.capitalize()} Layout",
                layout=layout,
            )
            assert widget.layout == layout

    def test_graph_explorer_with_focus(self):
        """Test graph explorer with focus node."""
        widget = GraphExplorerWidget(
            widget_id="ge-focus",
            title="Focused Graph",
            focus_node_id="node-123",
            depth=2,
        )
        assert widget.focus_node_id == "node-123"
        assert widget.depth == 2

    def test_graph_explorer_styling(self):
        """Test graph explorer styling options."""
        widget = GraphExplorerWidget(
            widget_id="ge-styled",
            title="Styled Graph",
            node_color_by="type",
            edge_color_by="relationship",
            show_labels=True,
        )
        assert widget.node_color_by == "type"
        assert widget.edge_color_by == "relationship"
        assert widget.show_labels is True

    def test_graph_explorer_to_dict(self):
        """Test graph explorer serialization."""
        widget = GraphExplorerWidget(
            widget_id="ge-ser",
            title="Serializable Graph",
        )
        d = widget.to_dict()
        assert d["widget_type"] == "graph_explorer"
        assert "layout" in d


# ============================================================================
# MetricsWidget Tests
# ============================================================================


class TestMetricsWidget:
    """Tests for MetricsWidget."""

    def test_metrics_creation(self):
        """Test creating a metrics widget."""
        widget = MetricsWidget(
            widget_id="m-1",
            title="System Metrics",
            metrics=["memory_count", "query_rate", "latency_p99"],
        )
        assert widget.widget_type == WidgetType.METRICS
        assert len(widget.metrics) == 3

    def test_metrics_with_format(self):
        """Test metrics with format options."""
        widget = MetricsWidget(
            widget_id="m-2",
            title="Formatted Metrics",
            metrics=["cpu_usage"],
            format_spec={"cpu_usage": "{:.1%}"},
        )
        assert widget.format_spec["cpu_usage"] == "{:.1%}"

    def test_metrics_refresh(self):
        """Test metrics refresh interval."""
        widget = MetricsWidget(
            widget_id="m-3",
            title="Auto-refresh Metrics",
            metrics=["active_users"],
            refresh_interval_s=10,
        )
        assert widget.refresh_interval_s == 10

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        widget = MetricsWidget(
            widget_id="m-4",
            title="Serializable Metrics",
            metrics=["total_memories"],
        )
        d = widget.to_dict()
        assert d["widget_type"] == "metrics"
        assert "metrics" in d


# ============================================================================
# TimeSeriesWidget Tests
# ============================================================================


class TestTimeSeriesWidget:
    """Tests for TimeSeriesWidget."""

    def test_timeseries_creation(self):
        """Test creating a time series widget."""
        widget = TimeSeriesWidget(
            widget_id="ts-1",
            title="Query Rate",
            metric="queries_per_second",
        )
        assert widget.widget_type == WidgetType.TIMESERIES
        assert widget.metric == "queries_per_second"

    def test_timeseries_time_range(self):
        """Test time series with time range."""
        widget = TimeSeriesWidget(
            widget_id="ts-2",
            title="Hourly Metrics",
            metric="memory_additions",
            time_range_hours=24,
            aggregation="1h",
        )
        assert widget.time_range_hours == 24
        assert widget.aggregation == "1h"

    def test_timeseries_chart_options(self):
        """Test time series chart options."""
        widget = TimeSeriesWidget(
            widget_id="ts-3",
            title="Line Chart",
            metric="latency",
            chart_type="line",
            fill=True,
        )
        assert widget.chart_type == "line"
        assert widget.fill is True

    def test_timeseries_to_dict(self):
        """Test time series serialization."""
        widget = TimeSeriesWidget(
            widget_id="ts-4",
            title="Serializable",
            metric="requests",
        )
        d = widget.to_dict()
        assert d["widget_type"] == "timeseries"
        assert d["metric"] == "requests"


# ============================================================================
# TableWidget Tests
# ============================================================================


class TestTableWidget:
    """Tests for TableWidget."""

    def test_table_creation(self):
        """Test creating a table widget."""
        widget = TableWidget(
            widget_id="t-1",
            title="Recent Memories",
            columns=["id", "content", "created_at"],
        )
        assert widget.widget_type == WidgetType.TABLE
        assert len(widget.columns) == 3

    def test_table_with_column_config(self):
        """Test table with column configuration."""
        widget = TableWidget(
            widget_id="t-2",
            title="Configured Table",
            columns=["name", "value"],
            column_config={
                "name": {"width": 200, "sortable": True},
                "value": {"width": 100, "align": "right"},
            },
        )
        assert widget.column_config["name"]["width"] == 200
        assert widget.column_config["value"]["align"] == "right"

    def test_table_pagination(self):
        """Test table pagination."""
        widget = TableWidget(
            widget_id="t-3",
            title="Paginated Table",
            columns=["id", "data"],
            page_size=25,
            show_pagination=True,
        )
        assert widget.page_size == 25
        assert widget.show_pagination is True

    def test_table_to_dict(self):
        """Test table serialization."""
        widget = TableWidget(
            widget_id="t-4",
            title="Serializable Table",
            columns=["col1"],
        )
        d = widget.to_dict()
        assert d["widget_type"] == "table"
        assert "columns" in d


# ============================================================================
# Dashboard Tests
# ============================================================================


class TestDashboard:
    """Tests for Dashboard."""

    def test_dashboard_creation(self):
        """Test creating a dashboard."""
        config = DashboardConfig(title="Test Dashboard")
        dashboard = Dashboard(
            dashboard_id="dash-1",
            config=config,
        )
        assert dashboard.dashboard_id == "dash-1"
        assert dashboard.config.title == "Test Dashboard"
        assert len(dashboard.widgets) == 0

    def test_add_widget(self):
        """Test adding widgets to dashboard."""
        dashboard = Dashboard(
            dashboard_id="dash-2",
            config=DashboardConfig(title="Multi-widget"),
        )

        widget1 = Widget(widget_id="w1", title="W1", widget_type=WidgetType.TABLE)
        widget2 = Widget(widget_id="w2", title="W2", widget_type=WidgetType.METRICS)

        dashboard.add_widget(widget1)
        dashboard.add_widget(widget2)

        assert len(dashboard.widgets) == 2

    def test_remove_widget(self):
        """Test removing widgets from dashboard."""
        dashboard = Dashboard(
            dashboard_id="dash-3",
            config=DashboardConfig(title="Removable"),
        )

        widget = Widget(widget_id="w-remove", title="Removable", widget_type=WidgetType.TABLE)
        dashboard.add_widget(widget)
        assert len(dashboard.widgets) == 1

        dashboard.remove_widget("w-remove")
        assert len(dashboard.widgets) == 0

    def test_get_widget(self):
        """Test getting a widget by ID."""
        dashboard = Dashboard(
            dashboard_id="dash-4",
            config=DashboardConfig(title="Getable"),
        )

        widget = Widget(widget_id="w-get", title="Getable", widget_type=WidgetType.TABLE)
        dashboard.add_widget(widget)

        retrieved = dashboard.get_widget("w-get")
        assert retrieved is not None
        assert retrieved.title == "Getable"

    def test_get_nonexistent_widget(self):
        """Test getting a non-existent widget."""
        dashboard = Dashboard(
            dashboard_id="dash-5",
            config=DashboardConfig(title="Empty"),
        )
        retrieved = dashboard.get_widget("nonexistent")
        assert retrieved is None

    def test_dashboard_to_dict(self):
        """Test dashboard serialization."""
        dashboard = Dashboard(
            dashboard_id="dash-6",
            config=DashboardConfig(title="Serializable"),
        )
        widget = Widget(widget_id="w-ser", title="Widget", widget_type=WidgetType.TABLE)
        dashboard.add_widget(widget)

        d = dashboard.to_dict()
        assert d["dashboard_id"] == "dash-6"
        assert len(d["widgets"]) == 1
        assert "config" in d


# ============================================================================
# DashboardBuilder Tests
# ============================================================================


class TestDashboardBuilder:
    """Tests for DashboardBuilder."""

    def test_builder_creation(self):
        """Test creating a builder."""
        builder = DashboardBuilder(title="Built Dashboard")
        assert builder is not None

    def test_builder_add_widget(self):
        """Test adding widgets with builder."""
        builder = DashboardBuilder(title="Built")
        builder.add_widget(
            widget_type=WidgetType.TABLE,
            title="Table Widget",
            columns=["a", "b"],
        )
        dashboard = builder.build()
        assert len(dashboard.widgets) == 1

    def test_builder_add_memory_browser(self):
        """Test adding memory browser with builder."""
        builder = DashboardBuilder(title="Memory Dashboard")
        builder.add_memory_browser(
            title="Browser",
            page_size=50,
        )
        dashboard = builder.build()
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].widget_type == WidgetType.MEMORY_BROWSER

    def test_builder_add_graph_explorer(self):
        """Test adding graph explorer with builder."""
        builder = DashboardBuilder(title="Graph Dashboard")
        builder.add_graph_explorer(
            title="Explorer",
            layout="hierarchical",
        )
        dashboard = builder.build()
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].widget_type == WidgetType.GRAPH_EXPLORER

    def test_builder_add_metrics(self):
        """Test adding metrics with builder."""
        builder = DashboardBuilder(title="Metrics Dashboard")
        builder.add_metrics(
            title="Key Metrics",
            metrics=["total", "active", "rate"],
        )
        dashboard = builder.build()
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].widget_type == WidgetType.METRICS

    def test_builder_add_timeseries(self):
        """Test adding timeseries with builder."""
        builder = DashboardBuilder(title="Timeseries Dashboard")
        builder.add_timeseries(
            title="Request Rate",
            metric="requests_per_second",
        )
        dashboard = builder.build()
        assert len(dashboard.widgets) == 1
        assert dashboard.widgets[0].widget_type == WidgetType.TIMESERIES

    def test_builder_fluent_interface(self):
        """Test builder fluent interface."""
        dashboard = (
            DashboardBuilder(title="Fluent")
            .add_memory_browser(title="Browser")
            .add_graph_explorer(title="Explorer")
            .add_metrics(title="Metrics", metrics=["count"])
            .build()
        )
        assert len(dashboard.widgets) == 3

    def test_builder_with_layout(self):
        """Test builder with layout positioning."""
        builder = DashboardBuilder(title="Positioned")
        builder.add_widget(
            widget_type=WidgetType.TABLE,
            title="Top Left",
            columns=["a"],
            row=0,
            col=0,
            width=6,
            height=4,
        )
        builder.add_widget(
            widget_type=WidgetType.METRICS,
            title="Top Right",
            metrics=["count"],
            row=0,
            col=6,
            width=6,
            height=4,
        )
        dashboard = builder.build()
        assert dashboard.widgets[0].row == 0
        assert dashboard.widgets[0].col == 0
        assert dashboard.widgets[1].col == 6


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for dashboard factory functions."""

    def test_create_memory_browser(self):
        """Test creating a memory browser dashboard."""
        dashboard = create_memory_browser(
            user_id="user-123",
            title="My Memories",
        )
        assert dashboard is not None
        assert "Memory" in dashboard.config.title or dashboard.config.title == "My Memories"

    def test_create_graph_explorer(self):
        """Test creating a graph explorer dashboard."""
        dashboard = create_graph_explorer(
            user_id="user-123",
            title="Knowledge Graph",
        )
        assert dashboard is not None
        # Should have graph explorer widget
        has_graph = any(
            w.widget_type == WidgetType.GRAPH_EXPLORER for w in dashboard.widgets
        )
        assert has_graph

    def test_create_metrics_dashboard(self):
        """Test creating a metrics dashboard."""
        dashboard = create_metrics_dashboard(
            user_id="user-123",
            title="System Metrics",
        )
        assert dashboard is not None
        # Should have metrics widgets
        has_metrics = any(
            w.widget_type in (WidgetType.METRICS, WidgetType.TIMESERIES)
            for w in dashboard.widgets
        )
        assert has_metrics


# ============================================================================
# Integration Tests
# ============================================================================


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_full_dashboard_workflow(self):
        """Test complete dashboard creation workflow."""
        # Create dashboard with builder
        dashboard = (
            DashboardBuilder(title="Full Dashboard")
            .add_memory_browser(title="Memories", page_size=20)
            .add_graph_explorer(title="Graph", layout="force")
            .add_metrics(title="Stats", metrics=["total", "rate"])
            .add_timeseries(title="Activity", metric="queries")
            .build()
        )

        # Verify structure
        assert len(dashboard.widgets) == 4

        # Serialize and verify
        data = dashboard.to_dict()
        assert len(data["widgets"]) == 4
        assert data["config"]["title"] == "Full Dashboard"

    def test_dashboard_widget_access(self):
        """Test accessing widgets in dashboard."""
        builder = DashboardBuilder(title="Accessible")
        builder.add_memory_browser(title="Browser")
        builder.add_graph_explorer(title="Graph")
        dashboard = builder.build()

        # Get widgets by type
        browser_widgets = [
            w for w in dashboard.widgets if w.widget_type == WidgetType.MEMORY_BROWSER
        ]
        graph_widgets = [
            w for w in dashboard.widgets if w.widget_type == WidgetType.GRAPH_EXPLORER
        ]

        assert len(browser_widgets) == 1
        assert len(graph_widgets) == 1

    def test_dashboard_serialization_roundtrip(self):
        """Test dashboard serialization preserves data."""
        original = (
            DashboardBuilder(title="Roundtrip Test")
            .add_memory_browser(title="Browser", page_size=30)
            .add_metrics(title="Metrics", metrics=["a", "b", "c"])
            .build()
        )

        # Serialize
        data = original.to_dict()

        # Verify key data preserved
        assert data["config"]["title"] == "Roundtrip Test"
        assert len(data["widgets"]) == 2

        # Find memory browser widget
        browser_widget = next(
            w for w in data["widgets"] if w["widget_type"] == "memory_browser"
        )
        assert browser_widget["page_size"] == 30

        # Find metrics widget
        metrics_widget = next(
            w for w in data["widgets"] if w["widget_type"] == "metrics"
        )
        assert len(metrics_widget["metrics"]) == 3
