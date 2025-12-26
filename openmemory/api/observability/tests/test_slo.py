"""Tests for SLO definitions and tracking.

Tests cover Phase 0c requirements per implementation plan section 13:
- SLO tracking with alerts on P95/P99
- Error budget tracking and freeze policy (>50% burn)
- Burn rate calculation (fast burn 2x over 1h, slow burn 4x over 6h)
- SLO definitions for various services
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openmemory.api.observability.slo import (
    SLOTracker,
    SLODefinition,
    SLOBudget,
    SLOConfig,
    BurnRate,
    BurnRateWindow,
    Metric,
    MetricType,
    SLOAlert,
    AlertSeverity,
    create_slo_tracker,
    SLOError,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def slo_config() -> SLOConfig:
    """Default SLO configuration."""
    return SLOConfig(
        service_name="openmemory-api",
        enabled=True,
    )


@pytest.fixture
def basic_slo() -> SLODefinition:
    """Basic SLO definition for testing."""
    return SLODefinition(
        name="api_availability",
        description="API availability target",
        target=0.99,  # 99%
        window=timedelta(days=30),
    )


@pytest.fixture
def latency_slo() -> SLODefinition:
    """Latency SLO definition for testing."""
    return SLODefinition(
        name="retrieval_latency_p95",
        description="P95 retrieval latency under 120ms",
        target=0.95,  # 95% of requests under threshold
        window=timedelta(days=7),
        threshold_ms=120.0,
        percentile=95,
    )


@pytest.fixture
def slo_tracker(slo_config) -> SLOTracker:
    """Create an SLO tracker for testing."""
    return create_slo_tracker(slo_config)


# ============================================================================
# SLODefinition Tests
# ============================================================================


class TestSLODefinition:
    """Tests for SLODefinition dataclass."""

    def test_create_basic_definition(self):
        """SLODefinition can be created with required fields."""
        slo = SLODefinition(
            name="availability",
            description="Service availability",
            target=0.999,
            window=timedelta(days=30),
        )
        assert slo.name == "availability"
        assert slo.target == 0.999
        assert slo.window == timedelta(days=30)

    def test_slo_with_threshold(self):
        """SLODefinition can include latency threshold."""
        slo = SLODefinition(
            name="latency_p95",
            description="P95 latency under 200ms",
            target=0.95,
            window=timedelta(days=7),
            threshold_ms=200.0,
            percentile=95,
        )
        assert slo.threshold_ms == 200.0
        assert slo.percentile == 95

    def test_slo_with_labels(self):
        """SLODefinition can include labels."""
        slo = SLODefinition(
            name="api_latency",
            description="API latency",
            target=0.99,
            window=timedelta(days=30),
            labels={"service": "api", "tier": "production"},
        )
        assert slo.labels["service"] == "api"

    def test_error_budget_calculation(self):
        """SLODefinition calculates error budget."""
        slo = SLODefinition(
            name="availability",
            description="99.9% availability",
            target=0.999,
            window=timedelta(days=30),
        )
        # 0.1% error budget (use approx due to floating point)
        assert abs(slo.error_budget - 0.001) < 1e-9

    def test_target_validation(self):
        """SLODefinition validates target range."""
        with pytest.raises(ValueError, match="target"):
            SLODefinition(
                name="invalid",
                description="Invalid SLO",
                target=1.5,  # Invalid: > 1.0
                window=timedelta(days=30),
            )


# ============================================================================
# SLOBudget Tests
# ============================================================================


class TestSLOBudget:
    """Tests for SLOBudget dataclass."""

    def test_create_budget(self, basic_slo):
        """SLOBudget can be created."""
        budget = SLOBudget(
            slo=basic_slo,
            consumed=0.25,
            remaining=0.75,
        )
        assert budget.consumed == 0.25
        assert budget.remaining == 0.75

    def test_budget_not_frozen_under_50_percent(self, basic_slo):
        """Budget is not frozen under 50% burn."""
        budget = SLOBudget(
            slo=basic_slo,
            consumed=0.49,
            remaining=0.51,
        )
        assert budget.is_frozen is False

    def test_budget_frozen_at_50_percent(self, basic_slo):
        """Budget is frozen at 50% burn per error budget policy."""
        budget = SLOBudget(
            slo=basic_slo,
            consumed=0.51,
            remaining=0.49,
        )
        assert budget.is_frozen is True

    def test_budget_frozen_fully_consumed(self, basic_slo):
        """Budget is frozen when fully consumed."""
        budget = SLOBudget(
            slo=basic_slo,
            consumed=1.0,
            remaining=0.0,
        )
        assert budget.is_frozen is True

    def test_budget_exhausted_percent(self, basic_slo):
        """Budget calculates exhausted percentage."""
        budget = SLOBudget(
            slo=basic_slo,
            consumed=0.75,
            remaining=0.25,
        )
        assert budget.exhausted_percent == 75.0


# ============================================================================
# BurnRate Tests
# ============================================================================


class TestBurnRate:
    """Tests for BurnRate tracking."""

    def test_fast_burn_threshold(self):
        """Fast burn threshold is 2x over 1h."""
        burn = BurnRate(
            rate=2.5,  # 2.5x burn rate
            window=BurnRateWindow.FAST,
        )
        assert burn.threshold == 2.0  # Fast burn = 2x
        assert burn.is_burning is True

    def test_slow_burn_threshold(self):
        """Slow burn threshold is 4x over 6h."""
        burn = BurnRate(
            rate=4.5,  # 4.5x burn rate
            window=BurnRateWindow.SLOW,
        )
        assert burn.threshold == 4.0  # Slow burn = 4x
        assert burn.is_burning is True

    def test_normal_rate_not_burning(self):
        """Normal burn rate is not alerting."""
        burn = BurnRate(
            rate=1.5,
            window=BurnRateWindow.FAST,
        )
        assert burn.is_burning is False

    def test_burn_rate_window_durations(self):
        """Burn rate windows have correct durations."""
        assert BurnRateWindow.FAST.duration == timedelta(hours=1)
        assert BurnRateWindow.SLOW.duration == timedelta(hours=6)


# ============================================================================
# Metric Tests
# ============================================================================


class TestMetric:
    """Tests for Metric recording."""

    def test_create_counter_metric(self):
        """Counter metric can be created."""
        metric = Metric(
            name="requests_total",
            type=MetricType.COUNTER,
            value=100,
            timestamp=datetime.now(timezone.utc),
        )
        assert metric.type == MetricType.COUNTER
        assert metric.value == 100

    def test_create_gauge_metric(self):
        """Gauge metric can be created."""
        metric = Metric(
            name="active_connections",
            type=MetricType.GAUGE,
            value=42,
            timestamp=datetime.now(timezone.utc),
        )
        assert metric.type == MetricType.GAUGE

    def test_create_histogram_metric(self):
        """Histogram metric can be created."""
        metric = Metric(
            name="request_duration",
            type=MetricType.HISTOGRAM,
            value=45.5,
            timestamp=datetime.now(timezone.utc),
            labels={"endpoint": "/api/search"},
        )
        assert metric.type == MetricType.HISTOGRAM
        assert metric.labels["endpoint"] == "/api/search"


# ============================================================================
# SLOTracker Creation Tests
# ============================================================================


class TestSLOTrackerCreation:
    """Tests for SLO tracker creation."""

    def test_create_tracker_basic(self, slo_config):
        """create_slo_tracker creates a tracker."""
        tracker = create_slo_tracker(slo_config)
        assert tracker is not None
        assert tracker.service_name == "openmemory-api"

    def test_create_tracker_with_slos(self, slo_config, basic_slo, latency_slo):
        """Tracker can be created with initial SLOs."""
        tracker = create_slo_tracker(
            slo_config,
            slos=[basic_slo, latency_slo],
        )
        assert len(tracker.slos) == 2

    def test_create_tracker_disabled(self):
        """Disabled tracker doesn't track metrics."""
        config = SLOConfig(service_name="test", enabled=False)
        tracker = create_slo_tracker(config)
        tracker.record_success("availability")
        # Should not raise, just no-op


# ============================================================================
# SLOTracker Recording Tests
# ============================================================================


class TestSLOTrackerRecording:
    """Tests for recording metrics to SLO tracker."""

    def test_record_success(self, slo_tracker, basic_slo):
        """record_success increments good events."""
        slo_tracker.register_slo(basic_slo)
        slo_tracker.record_success("api_availability")

        budget = slo_tracker.get_budget("api_availability")
        # After one success, budget should be full
        assert budget.remaining > 0

    def test_record_failure(self, slo_tracker, basic_slo):
        """record_failure increments bad events."""
        slo_tracker.register_slo(basic_slo)
        slo_tracker.record_failure("api_availability")

        budget = slo_tracker.get_budget("api_availability")
        # After one failure, some budget consumed
        assert budget.consumed > 0

    def test_record_latency(self, slo_tracker, latency_slo):
        """record_latency tracks against threshold."""
        slo_tracker.register_slo(latency_slo)

        # Record some latencies under threshold
        for _ in range(90):
            slo_tracker.record_latency("retrieval_latency_p95", 50.0)

        # Record some over threshold
        for _ in range(10):
            slo_tracker.record_latency("retrieval_latency_p95", 200.0)

        budget = slo_tracker.get_budget("retrieval_latency_p95")
        assert budget is not None
        # 90% success (under P95 threshold of 95%)
        assert budget.consumed > 0

    def test_record_with_labels(self, slo_tracker, basic_slo):
        """record_success can include labels."""
        slo_tracker.register_slo(basic_slo)
        slo_tracker.record_success(
            "api_availability",
            labels={"endpoint": "/api/search"},
        )
        # Should not raise

    def test_record_to_unknown_slo(self, slo_tracker):
        """Recording to unknown SLO raises error."""
        with pytest.raises(SLOError, match="not found"):
            slo_tracker.record_success("unknown_slo")


# ============================================================================
# Budget Calculation Tests
# ============================================================================


class TestBudgetCalculation:
    """Tests for error budget calculation."""

    def test_budget_from_success_failures(self, slo_tracker, basic_slo):
        """Budget calculated from success/failure ratio."""
        slo_tracker.register_slo(basic_slo)

        # 99 successes, 1 failure = 1% errors = 100% of 1% budget consumed
        for _ in range(99):
            slo_tracker.record_success("api_availability")
        slo_tracker.record_failure("api_availability")

        budget = slo_tracker.get_budget("api_availability")
        # For 99% SLO, 1% errors means 100% budget consumed
        assert budget.consumed >= 0.99

    def test_budget_window_sliding(self, slo_tracker, basic_slo):
        """Budget uses sliding window."""
        slo_tracker.register_slo(basic_slo)

        # Record events
        for _ in range(100):
            slo_tracker.record_success("api_availability")

        budget1 = slo_tracker.get_budget("api_availability")

        # Record more failures
        for _ in range(10):
            slo_tracker.record_failure("api_availability")

        budget2 = slo_tracker.get_budget("api_availability")

        # Budget should have more consumed after failures
        assert budget2.consumed > budget1.consumed

    def test_get_budget_unknown_slo(self, slo_tracker):
        """Get budget for unknown SLO returns None."""
        budget = slo_tracker.get_budget("unknown_slo")
        assert budget is None


# ============================================================================
# Burn Rate Calculation Tests
# ============================================================================


class TestBurnRateCalculation:
    """Tests for burn rate calculation."""

    def test_calculate_fast_burn(self, slo_tracker, basic_slo):
        """Calculate fast burn rate (1h window)."""
        slo_tracker.register_slo(basic_slo)

        # Simulate high error rate
        for _ in range(90):
            slo_tracker.record_success("api_availability")
        for _ in range(10):
            slo_tracker.record_failure("api_availability")

        burn = slo_tracker.get_burn_rate("api_availability", BurnRateWindow.FAST)
        assert burn is not None
        assert burn.rate > 0

    def test_calculate_slow_burn(self, slo_tracker, basic_slo):
        """Calculate slow burn rate (6h window)."""
        slo_tracker.register_slo(basic_slo)

        for _ in range(95):
            slo_tracker.record_success("api_availability")
        for _ in range(5):
            slo_tracker.record_failure("api_availability")

        burn = slo_tracker.get_burn_rate("api_availability", BurnRateWindow.SLOW)
        assert burn is not None

    def test_burn_rate_triggers_alert(self, slo_tracker, basic_slo):
        """High burn rate triggers alert."""
        slo_tracker.register_slo(basic_slo)
        alerts = []
        slo_tracker.on_alert(lambda a: alerts.append(a))

        # Record enough failures to trigger fast burn
        for _ in range(50):
            slo_tracker.record_success("api_availability")
        for _ in range(50):
            slo_tracker.record_failure("api_availability")

        slo_tracker.check_alerts()

        # Should have at least one alert
        assert len(alerts) >= 1
        assert any(a.severity == AlertSeverity.WARNING for a in alerts)


# ============================================================================
# Alert Tests
# ============================================================================


class TestSLOAlerts:
    """Tests for SLO alerting."""

    def test_alert_creation(self, basic_slo):
        """SLOAlert can be created."""
        alert = SLOAlert(
            slo=basic_slo,
            severity=AlertSeverity.WARNING,
            message="Budget consumed > 50%",
            timestamp=datetime.now(timezone.utc),
        )
        assert alert.severity == AlertSeverity.WARNING
        assert "50%" in alert.message

    def test_alert_severities(self):
        """All alert severities exist."""
        assert AlertSeverity.INFO
        assert AlertSeverity.WARNING
        assert AlertSeverity.CRITICAL

    def test_alert_callback(self, slo_tracker, basic_slo):
        """Alert callback is invoked."""
        slo_tracker.register_slo(basic_slo)
        received_alerts = []

        def alert_handler(alert: SLOAlert):
            received_alerts.append(alert)

        slo_tracker.on_alert(alert_handler)

        # Trigger alert condition
        for _ in range(10):
            slo_tracker.record_failure("api_availability")

        slo_tracker.check_alerts()

        assert len(received_alerts) > 0

    def test_alert_deduplication(self, slo_tracker, basic_slo):
        """Same alert is not sent repeatedly within dedup window."""
        slo_tracker.register_slo(basic_slo)
        received_alerts = []
        slo_tracker.on_alert(lambda a: received_alerts.append(a))

        # Record failures to trigger alert
        for _ in range(10):
            slo_tracker.record_failure("api_availability")

        # Check alerts multiple times rapidly - should deduplicate
        initial_count = len(slo_tracker.check_alerts())
        second_count = len(slo_tracker.check_alerts())
        third_count = len(slo_tracker.check_alerts())

        # First call should trigger alerts, subsequent calls should be deduplicated
        assert initial_count > 0
        assert second_count == 0
        assert third_count == 0


# ============================================================================
# Standard SLO Definitions Tests
# ============================================================================


class TestStandardSLOs:
    """Tests for standard SLO definitions per implementation plan."""

    def test_retrieval_latency_slo(self):
        """Retrieval latency SLO per section 12.1."""
        slo = SLODefinition(
            name="memory_retrieval_p95",
            description="P95 memory retrieval latency <= 120ms",
            target=0.95,
            window=timedelta(days=7),
            threshold_ms=120.0,
            percentile=95,
        )
        assert slo.threshold_ms == 120.0

    def test_inline_completion_slo(self):
        """Inline completion SLO per section 12.1."""
        slo = SLODefinition(
            name="inline_completion_p95",
            description="P95 inline completion latency <= 300ms",
            target=0.95,
            window=timedelta(days=7),
            threshold_ms=300.0,
            percentile=95,
        )
        assert slo.threshold_ms == 300.0

    def test_graph_query_slo(self):
        """Graph query SLO per section 12.1."""
        slo = SLODefinition(
            name="graph_query_p95",
            description="P95 graph query latency <= 500ms",
            target=0.95,
            window=timedelta(days=7),
            threshold_ms=500.0,
            percentile=95,
        )
        assert slo.threshold_ms == 500.0


# ============================================================================
# SLO Summary Tests
# ============================================================================


class TestSLOSummary:
    """Tests for SLO summary and reporting."""

    def test_get_all_budgets(self, slo_tracker, basic_slo, latency_slo):
        """Get all SLO budgets."""
        slo_tracker.register_slo(basic_slo)
        slo_tracker.register_slo(latency_slo)

        # Record some data
        for _ in range(100):
            slo_tracker.record_success("api_availability")
            slo_tracker.record_latency("retrieval_latency_p95", 50.0)

        budgets = slo_tracker.get_all_budgets()
        assert len(budgets) == 2

    def test_get_summary(self, slo_tracker, basic_slo):
        """Get SLO summary for dashboard."""
        slo_tracker.register_slo(basic_slo)

        for _ in range(95):
            slo_tracker.record_success("api_availability")
        for _ in range(5):
            slo_tracker.record_failure("api_availability")

        summary = slo_tracker.get_summary()
        assert "api_availability" in summary
        assert "current_rate" in summary["api_availability"]
        assert "budget_remaining" in summary["api_availability"]
        assert "is_burning" in summary["api_availability"]


# ============================================================================
# Error Budget Policy Tests
# ============================================================================


class TestErrorBudgetPolicy:
    """Tests for error budget policy enforcement."""

    def test_freeze_at_50_percent_burn(self, slo_tracker, basic_slo):
        """Freeze deployments at 50% budget burn per section 13."""
        slo_tracker.register_slo(basic_slo)

        # Consume more than 50% of error budget
        # For 99% SLO with 1% error budget:
        # 100 requests with 2 failures = 2% errors = 200% of budget
        for _ in range(98):
            slo_tracker.record_success("api_availability")
        for _ in range(2):
            slo_tracker.record_failure("api_availability")

        budget = slo_tracker.get_budget("api_availability")
        assert budget.is_frozen is True

    def test_should_freeze_deployments(self, slo_tracker, basic_slo):
        """Check if deployments should be frozen."""
        slo_tracker.register_slo(basic_slo)

        # Initially not frozen
        assert slo_tracker.should_freeze_deployments() is False

        # After many failures
        for _ in range(10):
            slo_tracker.record_failure("api_availability")

        assert slo_tracker.should_freeze_deployments() is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestSLOIntegration:
    """Integration tests for SLO tracking."""

    def test_full_slo_workflow(self):
        """Simulate full SLO tracking workflow."""
        config = SLOConfig(service_name="openmemory-api", enabled=True)
        tracker = create_slo_tracker(config)

        # Register SLOs
        tracker.register_slo(
            SLODefinition(
                name="api_availability",
                description="API availability",
                target=0.99,
                window=timedelta(days=30),
            )
        )
        tracker.register_slo(
            SLODefinition(
                name="search_latency_p95",
                description="Search latency P95",
                target=0.95,
                window=timedelta(days=7),
                threshold_ms=200.0,
                percentile=95,
            )
        )

        # Set up alerts
        alerts = []
        tracker.on_alert(lambda a: alerts.append(a))

        # Simulate traffic - 99.5% success for 99% SLO = 50% of budget consumed
        for i in range(1000):
            if i % 200 < 199:  # 99.5% success
                tracker.record_success("api_availability")
            else:
                tracker.record_failure("api_availability")

            latency = 50.0 if i % 25 != 0 else 250.0  # 4% over threshold
            tracker.record_latency("search_latency_p95", latency)

        # Check budgets
        availability_budget = tracker.get_budget("api_availability")
        latency_budget = tracker.get_budget("search_latency_p95")

        assert availability_budget is not None
        assert latency_budget is not None
        # 99.5% success rate for 99% SLO means ~50% budget consumed
        assert availability_budget.consumed < 1.0

        # Check for alerts
        tracker.check_alerts()

        # Get summary
        summary = tracker.get_summary()
        assert "api_availability" in summary
        assert "search_latency_p95" in summary
