"""Debug timing utilities for MCP tools.

Provides detailed timing breakdowns for debugging performance issues.
Only activated when debug=True is passed to tool calls.

Usage:
    from app.utils.debug_timing import DebugTimer

    timer = DebugTimer(enabled=debug)

    timer.start("validation")
    # ... validation code ...
    timer.stop("validation")

    timer.start("mem0_add")
    # ... mem0 code ...
    timer.stop("mem0_add")

    # Get timing breakdown
    timing = timer.get_timing()
    # Returns: {"total_ms": 523.4, "breakdown": {"validation_ms": 2.1, "mem0_add_ms": 312.5}}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TimingEntry:
    """Single timing entry."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DebugTimer:
    """Debug timer for tracking operation durations.

    Thread-safe timing tracker that only records when enabled.
    """

    def __init__(self, enabled: bool = False):
        """Initialize timer.

        Args:
            enabled: Whether timing is active. If False, all operations are no-ops.
        """
        self._enabled = enabled
        self._entries: dict[str, TimingEntry] = {}
        self._order: list[str] = []
        self._global_start: float = time.perf_counter() if enabled else 0.0

    @property
    def enabled(self) -> bool:
        """Check if timing is enabled."""
        return self._enabled

    def start(self, name: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Start timing an operation.

        Args:
            name: Operation name (e.g., "mem0_add", "embedding", "postgresql_write")
            metadata: Optional metadata to attach to this timing
        """
        if not self._enabled:
            return

        entry = TimingEntry(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )
        self._entries[name] = entry
        if name not in self._order:
            self._order.append(name)

    def stop(self, name: str, metadata: Optional[dict[str, Any]] = None) -> float:
        """Stop timing an operation.

        Args:
            name: Operation name (must match a previous start() call)
            metadata: Optional metadata to merge with existing metadata

        Returns:
            Duration in milliseconds (0.0 if not enabled or not found)
        """
        if not self._enabled:
            return 0.0

        entry = self._entries.get(name)
        if not entry:
            return 0.0

        entry.end_time = time.perf_counter()
        entry.duration_ms = (entry.end_time - entry.start_time) * 1000

        if metadata:
            entry.metadata.update(metadata)

        return entry.duration_ms

    def record(self, name: str, duration_ms: float, metadata: Optional[dict[str, Any]] = None) -> None:
        """Record a duration directly (for pre-calculated timings).

        Args:
            name: Operation name
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        if not self._enabled:
            return

        entry = TimingEntry(
            name=name,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._entries[name] = entry
        if name not in self._order:
            self._order.append(name)

    def get_duration(self, name: str) -> float:
        """Get duration for a specific operation.

        Args:
            name: Operation name

        Returns:
            Duration in milliseconds (0.0 if not found)
        """
        if not self._enabled:
            return 0.0

        entry = self._entries.get(name)
        return entry.duration_ms if entry else 0.0

    def get_timing(self) -> Optional[dict[str, Any]]:
        """Get complete timing breakdown.

        Returns:
            Dict with total_ms and breakdown, or None if not enabled.

        Example:
            {
                "total_ms": 523.4,
                "breakdown": {
                    "validation_ms": 2.1,
                    "mem0_add_ms": 312.5,
                    "embedding_ms": 245.2,
                    "postgresql_write_ms": 8.4,
                    "graph_projection_ms": 200.4
                },
                "details": {
                    "mem0_add": {"infer": false},
                    "graph_projection": {"functions": ["project_memory", "bridge_entities"]}
                }
            }
        """
        if not self._enabled:
            return None

        total_ms = (time.perf_counter() - self._global_start) * 1000

        breakdown = {}
        details = {}

        for name in self._order:
            entry = self._entries.get(name)
            if entry:
                key = f"{name}_ms"
                breakdown[key] = round(entry.duration_ms, 2)
                if entry.metadata:
                    details[name] = entry.metadata

        result = {
            "total_ms": round(total_ms, 2),
            "breakdown": breakdown,
        }

        if details:
            result["details"] = details

        return result

    def merge_timing(self, other: "DebugTimer") -> None:
        """Merge timing from another timer.

        Args:
            other: Another DebugTimer to merge entries from
        """
        if not self._enabled or not other._enabled:
            return

        for name in other._order:
            entry = other._entries.get(name)
            if entry:
                self._entries[name] = entry
                if name not in self._order:
                    self._order.append(name)


class TimingContext:
    """Context manager for timing a block of code.

    Usage:
        with TimingContext(timer, "my_operation"):
            # ... code to time ...
    """

    def __init__(self, timer: DebugTimer, name: str, metadata: Optional[dict[str, Any]] = None):
        self._timer = timer
        self._name = name
        self._metadata = metadata

    def __enter__(self) -> "TimingContext":
        self._timer.start(self._name, self._metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._timer.stop(self._name)
        return None


def timed(timer: DebugTimer, name: str, metadata: Optional[dict[str, Any]] = None):
    """Decorator for timing a function.

    Usage:
        @timed(timer, "my_function")
        def my_function():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer.start(name, metadata)
            try:
                return func(*args, **kwargs)
            finally:
                timer.stop(name)
        return wrapper
    return decorator
