"""
Unit tests for latency percentile tracking.

Tests written FIRST following TDD approach.
"""

import pytest

# Import will fail until implementation is written (TDD red phase)
from benchmarks.embeddings.metrics.latency import LatencyTracker, LatencyStats


class TestLatencyTrackerBasics:
    """Test basic latency tracking functionality."""

    def test_tracker_can_record_sample(self):
        """LatencyTracker should accept latency samples."""
        tracker = LatencyTracker()
        tracker.record(10.0)  # 10ms

        stats = tracker.get_stats()
        assert stats.count == 1

    def test_tracker_records_multiple_samples(self):
        """LatencyTracker should track count of samples."""
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.record(30.0)

        stats = tracker.get_stats()
        assert stats.count == 3

    def test_tracker_calculates_mean(self):
        """LatencyTracker should calculate mean latency."""
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.record(30.0)

        stats = tracker.get_stats()
        assert stats.mean == 20.0

    def test_tracker_calculates_min_max(self):
        """LatencyTracker should track min and max."""
        tracker = LatencyTracker()
        tracker.record(5.0)
        tracker.record(50.0)
        tracker.record(25.0)

        stats = tracker.get_stats()
        assert stats.min == 5.0
        assert stats.max == 50.0


class TestP95Calculation:
    """Test P95 (95th percentile) calculation."""

    def test_p95_with_100_samples(self, latency_samples):
        """P95 should return the 95th percentile value."""
        tracker = LatencyTracker()
        for sample in latency_samples:
            tracker.record(sample)

        stats = tracker.get_stats()

        # With 100 samples: 50@10ms, 30@20ms, 15@50ms, 4@100ms, 1@500ms
        # 95th percentile should be around 50-100ms
        assert 50.0 <= stats.p95 <= 100.0

    def test_p95_all_same_value(self):
        """P95 of identical values should be that value."""
        tracker = LatencyTracker()
        for _ in range(100):
            tracker.record(42.0)

        stats = tracker.get_stats()
        assert stats.p95 == 42.0

    def test_p95_two_values(self):
        """P95 with limited samples should work correctly."""
        tracker = LatencyTracker()
        # 96 samples at 10ms, 4 samples at 100ms
        for _ in range(96):
            tracker.record(10.0)
        for _ in range(4):
            tracker.record(100.0)

        stats = tracker.get_stats()
        # P95 = position 95 in sorted 100 samples = index 95 = 10.0
        # (indices 0-95 are 10ms, indices 96-99 are 100ms)
        assert stats.p95 == 10.0


class TestP99Calculation:
    """Test P99 (99th percentile) calculation."""

    def test_p99_with_100_samples(self, latency_samples):
        """P99 should return the 99th percentile value."""
        tracker = LatencyTracker()
        for sample in latency_samples:
            tracker.record(sample)

        stats = tracker.get_stats()

        # With our samples, P99 should be in the 100-500ms range
        assert 100.0 <= stats.p99 <= 500.0

    def test_p99_higher_than_p95(self):
        """P99 should be >= P95."""
        tracker = LatencyTracker()
        for i in range(100):
            tracker.record(float(i))

        stats = tracker.get_stats()
        assert stats.p99 >= stats.p95


class TestP50Calculation:
    """Test P50 (median) calculation."""

    def test_p50_is_median(self):
        """P50 should be the median value."""
        tracker = LatencyTracker()
        for i in range(1, 101):  # 1 to 100
            tracker.record(float(i))

        stats = tracker.get_stats()
        # Median of 1-100 is 50.5 (average of 50 and 51)
        assert 50.0 <= stats.p50 <= 51.0

    def test_p50_odd_samples(self):
        """P50 works with odd number of samples."""
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.record(20.0)
        tracker.record(30.0)

        stats = tracker.get_stats()
        assert stats.p50 == 20.0


class TestLatencyTrackerReset:
    """Test reset functionality."""

    def test_reset_clears_samples(self):
        """Reset should clear all recorded samples."""
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.record(20.0)

        tracker.reset()
        stats = tracker.get_stats()

        assert stats.count == 0

    def test_reset_allows_new_recording(self):
        """After reset, can record new samples."""
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.reset()
        tracker.record(100.0)

        stats = tracker.get_stats()
        assert stats.count == 1
        assert stats.mean == 100.0


class TestLatencyStatsDataclass:
    """Test LatencyStats dataclass properties."""

    def test_stats_has_all_fields(self):
        """LatencyStats should have p50, p95, p99, mean, min, max, count."""
        tracker = LatencyTracker()
        tracker.record(10.0)

        stats = tracker.get_stats()

        assert hasattr(stats, "p50")
        assert hasattr(stats, "p95")
        assert hasattr(stats, "p99")
        assert hasattr(stats, "mean")
        assert hasattr(stats, "min")
        assert hasattr(stats, "max")
        assert hasattr(stats, "count")

    def test_stats_values_are_floats(self):
        """All latency values should be floats."""
        tracker = LatencyTracker()
        tracker.record(10.0)

        stats = tracker.get_stats()

        assert isinstance(stats.p50, float)
        assert isinstance(stats.p95, float)
        assert isinstance(stats.p99, float)
        assert isinstance(stats.mean, float)
        assert isinstance(stats.min, float)
        assert isinstance(stats.max, float)
        assert isinstance(stats.count, int)


class TestLatencyTrackerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tracker_stats(self):
        """Empty tracker should return zero stats without crashing."""
        tracker = LatencyTracker()
        stats = tracker.get_stats()

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0

    def test_single_sample(self):
        """Single sample should set all percentiles to that value."""
        tracker = LatencyTracker()
        tracker.record(42.0)

        stats = tracker.get_stats()

        assert stats.p50 == 42.0
        assert stats.p95 == 42.0
        assert stats.p99 == 42.0
        assert stats.mean == 42.0

    def test_handles_zero_latency(self):
        """Should handle 0ms latency."""
        tracker = LatencyTracker()
        tracker.record(0.0)

        stats = tracker.get_stats()
        assert stats.mean == 0.0

    def test_handles_very_high_latency(self):
        """Should handle very high latency values."""
        tracker = LatencyTracker()
        tracker.record(10000.0)  # 10 seconds

        stats = tracker.get_stats()
        assert stats.max == 10000.0


class TestLatencyTrackerTimingContext:
    """Test timing context manager (if implemented)."""

    def test_context_manager_records_time(self):
        """Context manager should auto-record elapsed time."""
        import time

        tracker = LatencyTracker()

        with tracker.time():
            time.sleep(0.01)  # 10ms sleep

        stats = tracker.get_stats()
        assert stats.count == 1
        # Should be at least 10ms, accounting for overhead
        assert stats.mean >= 8.0  # Allow some timing variance
