"""SLO definitions and tracking.

Phase 0c implementation per IMPLEMENTATION-PLAN-DEV-ASSISTANT v7.md section 13.

Features:
- SLO tracking with alerts on P95/P99
- Error budget tracking and freeze policy (>50% burn)
- Burn rate calculation (fast burn 2x over 1h, slow burn 4x over 6h)
- SLO definitions for various services
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable


class SLOError(Exception):
    """Base exception for SLO errors."""

    pass


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BurnRateWindow(Enum):
    """Burn rate window types per implementation plan section 13."""

    FAST = "fast"  # 2x over 1h
    SLOW = "slow"  # 4x over 6h

    @property
    def duration(self) -> timedelta:
        """Get the duration for this window type."""
        if self == BurnRateWindow.FAST:
            return timedelta(hours=1)
        return timedelta(hours=6)

    @property
    def threshold(self) -> float:
        """Get the burn rate threshold for this window type."""
        if self == BurnRateWindow.FAST:
            return 2.0
        return 4.0


@dataclass
class Metric:
    """A recorded metric data point."""

    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective.

    Attributes:
        name: Unique name for the SLO.
        description: Human-readable description.
        target: Target percentage (0.0 to 1.0, e.g., 0.99 for 99%).
        window: Time window for the SLO (e.g., 30 days).
        threshold_ms: Optional latency threshold in milliseconds.
        percentile: Optional percentile for latency SLOs (e.g., 95 for P95).
        labels: Optional labels for filtering.
    """

    name: str
    description: str
    target: float
    window: timedelta
    threshold_ms: float | None = None
    percentile: int | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate SLO definition."""
        if not 0.0 <= self.target <= 1.0:
            raise ValueError(f"target must be between 0.0 and 1.0, got {self.target}")

    @property
    def error_budget(self) -> float:
        """Calculate error budget (inverse of target)."""
        return 1.0 - self.target


@dataclass
class BurnRate:
    """Burn rate tracking per implementation plan section 13.

    Attributes:
        rate: Current burn rate multiplier.
        window: The window type (FAST or SLOW).
        threshold: The threshold for alerting.
    """

    rate: float
    window: BurnRateWindow
    threshold: float = field(init=False)

    def __post_init__(self):
        """Set threshold based on window type."""
        self.threshold = self.window.threshold

    @property
    def is_burning(self) -> bool:
        """Check if burn rate exceeds threshold."""
        return self.rate > self.threshold


@dataclass
class SLOBudget:
    """Error budget tracking.

    Attributes:
        slo: The SLO definition.
        consumed: Fraction of budget consumed (0.0 to 1.0+).
        remaining: Fraction of budget remaining (1.0 to 0.0).
    """

    slo: SLODefinition
    consumed: float = 0.0
    remaining: float = 1.0

    @property
    def is_frozen(self) -> bool:
        """Check if deployments should be frozen (>50% burn)."""
        return self.consumed > 0.5

    @property
    def exhausted_percent(self) -> float:
        """Get percentage of budget exhausted."""
        return min(self.consumed * 100, 100.0)


@dataclass
class SLOAlert:
    """An SLO alert.

    Attributes:
        slo: The SLO that triggered the alert.
        severity: Alert severity level.
        message: Human-readable message.
        timestamp: When the alert was generated.
    """

    slo: SLODefinition
    severity: AlertSeverity
    message: str
    timestamp: datetime


@dataclass
class SLOConfig:
    """Configuration for SLO tracking.

    Attributes:
        service_name: Name of the service.
        enabled: Whether SLO tracking is enabled.
    """

    service_name: str
    enabled: bool = True


@dataclass
class _SLOState:
    """Internal state for an SLO."""

    slo: SLODefinition
    # Record of (timestamp, success) tuples
    events: deque = field(default_factory=lambda: deque(maxlen=100000))
    # Record of latencies for latency SLOs
    latencies: deque = field(default_factory=lambda: deque(maxlen=100000))
    # Last alert sent for deduplication: alert_type -> timestamp
    last_alerts: dict = field(default_factory=dict)


AlertCallback = Callable[[SLOAlert], None]


class SLOTracker:
    """SLO tracker for monitoring service level objectives.

    Provides error budget tracking, burn rate calculation,
    and alerting for SLO violations.
    """

    def __init__(self, config: SLOConfig):
        """Initialize the SLO tracker.

        Args:
            config: SLO configuration.
        """
        self._config = config
        self._slos: dict[str, _SLOState] = {}
        self._alert_callbacks: list[AlertCallback] = []

    @property
    def service_name(self) -> str:
        """Get the service name."""
        return self._config.service_name

    @property
    def slos(self) -> list[SLODefinition]:
        """Get all registered SLO definitions."""
        return [state.slo for state in self._slos.values()]

    def register_slo(self, slo: SLODefinition) -> None:
        """Register an SLO definition.

        Args:
            slo: The SLO definition to register.
        """
        self._slos[slo.name] = _SLOState(slo=slo)

    def on_alert(self, callback: AlertCallback) -> None:
        """Register an alert callback.

        Args:
            callback: Function to call when an alert is triggered.
        """
        self._alert_callbacks.append(callback)

    def _get_state(self, slo_name: str) -> _SLOState:
        """Get state for an SLO, raising if not found."""
        if slo_name not in self._slos:
            raise SLOError(f"SLO '{slo_name}' not found")
        return self._slos[slo_name]

    def record_success(
        self,
        slo_name: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a successful operation.

        Args:
            slo_name: Name of the SLO.
            labels: Optional labels for the event.
        """
        if not self._config.enabled:
            return

        state = self._get_state(slo_name)
        state.events.append((datetime.now(timezone.utc), True))

    def record_failure(
        self,
        slo_name: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a failed operation.

        Args:
            slo_name: Name of the SLO.
            labels: Optional labels for the event.
        """
        if not self._config.enabled:
            return

        state = self._get_state(slo_name)
        state.events.append((datetime.now(timezone.utc), False))

    def record_latency(
        self,
        slo_name: str,
        latency_ms: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a latency measurement.

        Args:
            slo_name: Name of the SLO.
            latency_ms: Latency in milliseconds.
            labels: Optional labels for the event.
        """
        if not self._config.enabled:
            return

        state = self._get_state(slo_name)
        now = datetime.now(timezone.utc)

        # Record latency
        state.latencies.append((now, latency_ms))

        # For latency SLOs, also record success/failure based on threshold
        if state.slo.threshold_ms is not None:
            is_good = latency_ms <= state.slo.threshold_ms
            state.events.append((now, is_good))

    def _calculate_error_rate(
        self,
        state: _SLOState,
        window: timedelta | None = None,
    ) -> tuple[float, int, int]:
        """Calculate error rate for an SLO.

        Args:
            state: The SLO state.
            window: Optional time window (uses SLO window if not specified).

        Returns:
            Tuple of (error_rate, total_events, good_events).
        """
        now = datetime.now(timezone.utc)
        window = window or state.slo.window
        cutoff = now - window

        total = 0
        good = 0
        for timestamp, success in state.events:
            if timestamp >= cutoff:
                total += 1
                if success:
                    good += 1

        if total == 0:
            return 0.0, 0, 0

        success_rate = good / total
        error_rate = 1.0 - success_rate
        return error_rate, total, good

    def get_budget(self, slo_name: str) -> SLOBudget | None:
        """Get current error budget for an SLO.

        Args:
            slo_name: Name of the SLO.

        Returns:
            SLOBudget or None if SLO not found.
        """
        if slo_name not in self._slos:
            return None

        state = self._slos[slo_name]
        error_rate, total, _ = self._calculate_error_rate(state)

        if total == 0:
            return SLOBudget(slo=state.slo, consumed=0.0, remaining=1.0)

        # Calculate budget consumption
        error_budget = state.slo.error_budget
        if error_budget > 0:
            consumed = min(error_rate / error_budget, 2.0)  # Cap at 200%
        else:
            consumed = 0.0 if error_rate == 0 else 2.0

        remaining = max(0.0, 1.0 - consumed)

        return SLOBudget(
            slo=state.slo,
            consumed=consumed,
            remaining=remaining,
        )

    def get_burn_rate(
        self,
        slo_name: str,
        window: BurnRateWindow,
    ) -> BurnRate | None:
        """Get burn rate for an SLO.

        Args:
            slo_name: Name of the SLO.
            window: The burn rate window type.

        Returns:
            BurnRate or None if SLO not found.
        """
        if slo_name not in self._slos:
            return None

        state = self._slos[slo_name]
        error_rate, total, _ = self._calculate_error_rate(state, window.duration)

        if total == 0:
            return BurnRate(rate=0.0, window=window)

        # Calculate burn rate relative to error budget
        error_budget = state.slo.error_budget
        if error_budget > 0:
            # Burn rate = (actual error rate / allowed error rate)
            # Normalized to window duration vs SLO window
            window_fraction = window.duration / state.slo.window
            allowed_error_in_window = error_budget * window_fraction
            if allowed_error_in_window > 0:
                rate = error_rate / allowed_error_in_window
            else:
                rate = 0.0 if error_rate == 0 else 100.0
        else:
            rate = 0.0 if error_rate == 0 else 100.0

        return BurnRate(rate=rate, window=window)

    def get_all_budgets(self) -> dict[str, SLOBudget]:
        """Get all SLO budgets.

        Returns:
            Dict mapping SLO names to budgets.
        """
        budgets = {}
        for name in self._slos:
            budget = self.get_budget(name)
            if budget:
                budgets[name] = budget
        return budgets

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary for all SLOs (for dashboards).

        Returns:
            Dict mapping SLO names to summary data.
        """
        summary = {}
        for name, state in self._slos.items():
            error_rate, total, good = self._calculate_error_rate(state)
            budget = self.get_budget(name)
            fast_burn = self.get_burn_rate(name, BurnRateWindow.FAST)
            slow_burn = self.get_burn_rate(name, BurnRateWindow.SLOW)

            summary[name] = {
                "current_rate": 1.0 - error_rate if total > 0 else 1.0,
                "target": state.slo.target,
                "budget_remaining": budget.remaining if budget else 1.0,
                "budget_consumed_percent": budget.exhausted_percent if budget else 0.0,
                "is_frozen": budget.is_frozen if budget else False,
                "is_burning": (fast_burn.is_burning if fast_burn else False)
                or (slow_burn.is_burning if slow_burn else False),
                "total_events": total,
                "good_events": good,
            }

        return summary

    def should_freeze_deployments(self) -> bool:
        """Check if any SLO requires deployment freeze.

        Per error budget policy: at 50% burn, freeze non-critical deployments.

        Returns:
            True if deployments should be frozen.
        """
        for budget in self.get_all_budgets().values():
            if budget.is_frozen:
                return True
        return False

    def check_alerts(self) -> list[SLOAlert]:
        """Check for alert conditions and trigger callbacks.

        Returns:
            List of alerts that were triggered.
        """
        alerts = []
        now = datetime.now(timezone.utc)

        for name, state in self._slos.items():
            budget = self.get_budget(name)
            fast_burn = self.get_burn_rate(name, BurnRateWindow.FAST)
            slow_burn = self.get_burn_rate(name, BurnRateWindow.SLOW)

            # Check budget consumption
            if budget and budget.consumed > 0.5:
                message = f"SLO '{name}' budget consumed > 50% ({budget.exhausted_percent:.1f}%)"
                alert = self._create_alert_if_new(
                    state, AlertSeverity.WARNING, message, now, alert_key="budget"
                )
                if alert:
                    alerts.append(alert)

            # Check fast burn
            if fast_burn and fast_burn.is_burning:
                message = f"SLO '{name}' fast burn rate {fast_burn.rate:.1f}x exceeds threshold"
                alert = self._create_alert_if_new(
                    state, AlertSeverity.WARNING, message, now, alert_key="fast_burn"
                )
                if alert:
                    alerts.append(alert)

            # Check slow burn
            if slow_burn and slow_burn.is_burning:
                message = f"SLO '{name}' slow burn rate {slow_burn.rate:.1f}x exceeds threshold"
                alert = self._create_alert_if_new(
                    state, AlertSeverity.WARNING, message, now, alert_key="slow_burn"
                )
                if alert:
                    alerts.append(alert)

        # Invoke callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                callback(alert)

        return alerts

    def _create_alert_if_new(
        self,
        state: _SLOState,
        severity: AlertSeverity,
        message: str,
        timestamp: datetime,
        alert_key: str,
    ) -> SLOAlert | None:
        """Create an alert if it's not a duplicate.

        Args:
            state: The SLO state.
            severity: Alert severity.
            message: Alert message.
            timestamp: Alert timestamp.
            alert_key: Key for deduplication.

        Returns:
            SLOAlert if new, None if duplicate.
        """
        # Deduplicate: don't send same alert type within 5 minutes
        last_time = state.last_alerts.get(alert_key)
        if last_time and (timestamp - last_time) < timedelta(minutes=5):
            return None

        state.last_alerts[alert_key] = timestamp

        return SLOAlert(
            slo=state.slo,
            severity=severity,
            message=message,
            timestamp=timestamp,
        )


def create_slo_tracker(
    config: SLOConfig,
    slos: list[SLODefinition] | None = None,
) -> SLOTracker:
    """Create an SLO tracker.

    Args:
        config: SLO configuration.
        slos: Optional list of SLOs to register.

    Returns:
        SLOTracker instance.
    """
    tracker = SLOTracker(config)
    if slos:
        for slo in slos:
            tracker.register_slo(slo)
    return tracker
