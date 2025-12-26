"""
Latency percentile tracking for benchmark measurements.

Provides P50, P95, P99 percentile calculations along with
mean, min, max statistics.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, List


@dataclass
class LatencyStats:
    """Latency statistics."""
    p50: float   # 50th percentile (median) in milliseconds
    p95: float   # 95th percentile in milliseconds
    p99: float   # 99th percentile in milliseconds
    mean: float  # Mean latency in milliseconds
    min: float   # Minimum latency in milliseconds
    max: float   # Maximum latency in milliseconds
    count: int   # Number of samples


class LatencyTracker:
    """
    Track and calculate latency percentiles.

    Records latency samples (in milliseconds) and calculates
    percentile statistics.

    Usage:
        tracker = LatencyTracker()
        tracker.record(10.5)  # 10.5ms
        tracker.record(20.3)  # 20.3ms
        stats = tracker.get_stats()
        print(f"P95: {stats.p95}ms")

    Or with context manager:
        with tracker.time():
            # operation to measure
            do_something()
    """

    def __init__(self):
        self._samples: List[float] = []

    def record(self, latency_ms: float) -> None:
        """
        Record a single latency sample.

        Args:
            latency_ms: Latency in milliseconds.
        """
        self._samples.append(latency_ms)

    def get_stats(self) -> LatencyStats:
        """
        Calculate and return latency statistics.

        Returns:
            LatencyStats with percentiles and summary statistics.
        """
        if not self._samples:
            return LatencyStats(
                p50=0.0,
                p95=0.0,
                p99=0.0,
                mean=0.0,
                min=0.0,
                max=0.0,
                count=0
            )

        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)

        return LatencyStats(
            p50=self._percentile(sorted_samples, 50),
            p95=self._percentile(sorted_samples, 95),
            p99=self._percentile(sorted_samples, 99),
            mean=sum(sorted_samples) / n,
            min=sorted_samples[0],
            max=sorted_samples[-1],
            count=n
        )

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._samples.clear()

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        """
        Context manager to automatically record elapsed time.

        Usage:
            with tracker.time():
                # Code to measure
                result = expensive_operation()

        The elapsed time in milliseconds is automatically recorded.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(elapsed_ms)

    @staticmethod
    def _percentile(sorted_samples: List[float], p: float) -> float:
        """
        Calculate the p-th percentile from sorted samples.

        Uses the "nearest rank" method for simplicity.

        Args:
            sorted_samples: Sorted list of samples.
            p: Percentile to calculate (0-100).

        Returns:
            The p-th percentile value.
        """
        if not sorted_samples:
            return 0.0

        n = len(sorted_samples)
        if n == 1:
            return sorted_samples[0]

        # Calculate the rank (1-indexed)
        # Using the "exclusive" method: rank = (p/100) * n
        rank = (p / 100) * n

        # Use the ceiling for "nearest rank" method
        # This means P95 of [1,2,3,...,100] returns the 95th value
        index = min(int(rank), n - 1)

        return sorted_samples[index]
