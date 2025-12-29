"""Bootstrap State and Priority Queue for Code Indexing (FR-001).

This module provides:
- Bootstrap state persistence (indexed files, progress %)
- Priority queue for tiered indexing (hot files first)
- Bootstrap status API response format
- Resume from partial indexing state
- File priority scoring (recently modified, frequently accessed)
- Progress callbacks for UI updates
"""

from __future__ import annotations

import heapq
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class BootstrapError(Exception):
    """Base exception for bootstrap errors."""

    pass


class StateNotFoundError(BootstrapError):
    """Raised when bootstrap state is not found."""

    pass


# =============================================================================
# Enums
# =============================================================================


class PriorityTier(Enum):
    """Priority tier for file indexing."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"

    @property
    def priority_value(self) -> int:
        """Get numeric priority value (higher = more priority)."""
        return {"hot": 3, "warm": 2, "cold": 1}[self.value]


class IndexingPhase(Enum):
    """Phase of the bootstrap indexing process."""

    NOT_STARTED = "not_started"
    SCANNING = "scanning"
    INDEXING_HOT = "indexing_hot"
    INDEXING_WARM = "indexing_warm"
    INDEXING_COLD = "indexing_cold"
    COMPLETED = "completed"
    PAUSED = "paused"


# =============================================================================
# FilePriority
# =============================================================================


@dataclass
class FilePriority:
    """Priority information for a file to be indexed."""

    path: Path
    tier: PriorityTier
    score: float
    modified_time: float
    access_count: int
    depth: int

    def __lt__(self, other: "FilePriority") -> bool:
        """Compare for heap ordering (higher score = higher priority)."""
        # Negate score for max-heap behavior with heapq (which is min-heap)
        return self.score > other.score

    def __le__(self, other: "FilePriority") -> bool:
        """Compare for ordering."""
        return self.score >= other.score

    def __gt__(self, other: "FilePriority") -> bool:
        """Compare for ordering."""
        return self.score > other.score

    def __ge__(self, other: "FilePriority") -> bool:
        """Compare for ordering."""
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        """Equality based on path."""
        if not isinstance(other, FilePriority):
            return False
        return self.path == other.path

    def __hash__(self) -> int:
        """Hash based on path."""
        return hash(self.path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": str(self.path),
            "tier": self.tier.value,
            "score": self.score,
            "modified_time": self.modified_time,
            "access_count": self.access_count,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilePriority":
        """Deserialize from dictionary."""
        return cls(
            path=Path(data["path"]),
            tier=PriorityTier(data["tier"]),
            score=data["score"],
            modified_time=data["modified_time"],
            access_count=data["access_count"],
            depth=data["depth"],
        )


# =============================================================================
# FilePriorityScorer
# =============================================================================


@dataclass
class FilePriorityScorer:
    """Scorer for calculating file priority based on various factors."""

    recency_weight: float = 0.5
    access_weight: float = 0.3
    depth_weight: float = 0.2
    hot_threshold: float = 0.7
    warm_threshold: float = 0.4
    recency_days: int = 7

    def score_file(
        self,
        file_path: Path,
        base_path: Path,
        modified_time: Optional[float] = None,
        access_count: int = 0,
    ) -> FilePriority:
        """Score a file to determine its priority tier and score.

        Args:
            file_path: Path to the file
            base_path: Root path of the project
            modified_time: Override modification time (for testing)
            access_count: Number of times file was accessed

        Returns:
            FilePriority with tier and score
        """
        # Get modification time
        if modified_time is None:
            try:
                stat = file_path.stat()
                modified_time = stat.st_mtime
            except OSError:
                modified_time = 0.0

        # Calculate depth
        try:
            rel_path = file_path.relative_to(base_path)
            depth = len(rel_path.parts) - 1  # -1 because file itself is a part
        except ValueError:
            depth = 0

        # Calculate recency score (0-1)
        now = time.time()
        age_seconds = now - modified_time
        recency_threshold = self.recency_days * 24 * 60 * 60
        recency_score = max(0.0, 1.0 - (age_seconds / recency_threshold))

        # Calculate access score (0-1, with diminishing returns)
        if access_count > 0:
            access_score = min(1.0, access_count / 100.0)
        else:
            access_score = 0.0

        # Calculate depth score (0-1, shallower is better)
        max_depth = 20  # Reasonable max depth
        depth_score = max(0.0, 1.0 - (depth / max_depth))

        # Calculate weighted score
        score = (
            self.recency_weight * recency_score
            + self.access_weight * access_score
            + self.depth_weight * depth_score
        )

        # Determine tier
        if score >= self.hot_threshold:
            tier = PriorityTier.HOT
        elif score >= self.warm_threshold:
            tier = PriorityTier.WARM
        else:
            tier = PriorityTier.COLD

        return FilePriority(
            path=file_path,
            tier=tier,
            score=score,
            modified_time=modified_time,
            access_count=access_count,
            depth=depth,
        )


# =============================================================================
# PriorityQueue
# =============================================================================


class PriorityQueue:
    """Priority queue for file indexing with tiered priorities."""

    def __init__(self):
        self._heap: list[FilePriority] = []
        self._entries: dict[Path, FilePriority] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Get number of items in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[FilePriority]:
        """Iterate over items without removing."""
        # Return sorted copy
        return iter(sorted(self._entries.values(), reverse=True))

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0

    def push(self, item: FilePriority) -> None:
        """Push item onto queue."""
        with self._lock:
            if item.path in self._entries:
                # Update existing entry
                self._entries[item.path] = item
                # Rebuild heap
                self._heap = list(self._entries.values())
                heapq.heapify(self._heap)
            else:
                self._entries[item.path] = item
                heapq.heappush(self._heap, item)

    def push_all(self, items: list[FilePriority]) -> None:
        """Push multiple items onto queue."""
        with self._lock:
            for item in items:
                self._entries[item.path] = item
            self._heap = list(self._entries.values())
            heapq.heapify(self._heap)

    def pop(self) -> Optional[FilePriority]:
        """Pop highest priority item from queue."""
        with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                if item.path in self._entries and self._entries[item.path] == item:
                    del self._entries[item.path]
                    return item
            return None

    def peek(self) -> Optional[FilePriority]:
        """Peek at highest priority item without removing."""
        with self._lock:
            if self._heap:
                return self._heap[0]
            return None

    def pop_batch(self, count: int) -> list[FilePriority]:
        """Pop multiple items from queue."""
        items = []
        for _ in range(count):
            item = self.pop()
            if item is None:
                break
            items.append(item)
        return items

    def clear(self) -> None:
        """Clear all items from queue."""
        with self._lock:
            self._heap.clear()
            self._entries.clear()

    def filter_by_tier(self, tier: PriorityTier) -> list[FilePriority]:
        """Get all items of a specific tier."""
        with self._lock:
            return [item for item in self._entries.values() if item.tier == tier]

    def count_by_tier(self) -> dict[PriorityTier, int]:
        """Count items by tier."""
        with self._lock:
            counts = {PriorityTier.HOT: 0, PriorityTier.WARM: 0, PriorityTier.COLD: 0}
            for item in self._entries.values():
                counts[item.tier] += 1
            return counts

    def to_list(self) -> list[FilePriority]:
        """Convert queue to list."""
        with self._lock:
            return list(self._entries.values())

    @classmethod
    def from_list(cls, items: list[FilePriority]) -> "PriorityQueue":
        """Create queue from list of items."""
        queue = cls()
        queue.push_all(items)
        return queue


# =============================================================================
# BootstrapProgress
# =============================================================================


@dataclass
class BootstrapProgress:
    """Progress tracking for bootstrap indexing."""

    phase: IndexingPhase
    total_files: int
    indexed_files: int
    failed_files: int
    start_time: Optional[float] = None

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_files == 0:
            return 0.0 if self.phase == IndexingPhase.NOT_STARTED else 100.0
        processed = self.indexed_files + self.failed_files
        return min(100.0, (processed / self.total_files) * 100)

    @property
    def is_complete(self) -> bool:
        """Check if indexing is complete."""
        if self.total_files == 0:
            return self.phase == IndexingPhase.COMPLETED
        return (self.indexed_files + self.failed_files) >= self.total_files

    @property
    def files_per_second(self) -> Optional[float]:
        """Calculate indexing rate."""
        if self.start_time is None or self.indexed_files == 0:
            return None
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return None
        return self.indexed_files / elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        rate = self.files_per_second
        if rate is None or rate <= 0:
            return None
        remaining = self.total_files - self.indexed_files - self.failed_files
        return remaining / rate

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "phase": self.phase.value,
            "total_files": self.total_files,
            "indexed_files": self.indexed_files,
            "failed_files": self.failed_files,
            "percentage": self.percentage,
            "files_per_second": self.files_per_second,
            "eta_seconds": self.eta_seconds,
        }


# =============================================================================
# BootstrapState
# =============================================================================


@dataclass
class BootstrapState:
    """Persistent state for bootstrap indexing."""

    root_path: Path
    progress: BootstrapProgress
    indexed_files: Set[Path]
    failed_files: dict[Path, str]  # path -> error message
    queue: PriorityQueue

    @classmethod
    def create_initial(cls, root_path: Path) -> "BootstrapState":
        """Create initial bootstrap state."""
        return cls(
            root_path=root_path,
            progress=BootstrapProgress(
                phase=IndexingPhase.NOT_STARTED,
                total_files=0,
                indexed_files=0,
                failed_files=0,
            ),
            indexed_files=set(),
            failed_files={},
            queue=PriorityQueue(),
        )

    def mark_indexed(self, path: Path) -> None:
        """Mark a file as successfully indexed."""
        self.indexed_files.add(path)
        self.progress.indexed_files = len(self.indexed_files)

    def mark_failed(self, path: Path, error: str) -> None:
        """Mark a file as failed to index."""
        self.failed_files[path] = error
        self.progress.failed_files = len(self.failed_files)

    def update_phase(self, phase: IndexingPhase) -> None:
        """Update the current phase."""
        self.progress.phase = phase

    def set_total_files(self, count: int) -> None:
        """Set total file count for progress calculation."""
        self.progress.total_files = count

    def is_indexed(self, path: Path) -> bool:
        """Check if a file was already indexed."""
        return path in self.indexed_files

    def to_json(self) -> str:
        """Serialize state to JSON."""
        data = {
            "root_path": str(self.root_path),
            "progress": self.progress.to_dict(),
            "indexed_files": [str(p) for p in self.indexed_files],
            "failed_files": {str(p): e for p, e in self.failed_files.items()},
            "queue": [fp.to_dict() for fp in self.queue.to_list()],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "BootstrapState":
        """Deserialize state from JSON."""
        data = json.loads(json_str)

        progress = BootstrapProgress(
            phase=IndexingPhase(data["progress"]["phase"]),
            total_files=data["progress"]["total_files"],
            indexed_files=data["progress"]["indexed_files"],
            failed_files=data["progress"]["failed_files"],
        )

        indexed_files = {Path(p) for p in data["indexed_files"]}
        failed_files = {Path(p): e for p, e in data["failed_files"].items()}

        queue_items = [FilePriority.from_dict(fp) for fp in data.get("queue", [])]
        queue = PriorityQueue.from_list(queue_items)

        return cls(
            root_path=Path(data["root_path"]),
            progress=progress,
            indexed_files=indexed_files,
            failed_files=failed_files,
            queue=queue,
        )


# =============================================================================
# BootstrapStateStore
# =============================================================================


class BootstrapStateStore:
    """Abstract state store interface for bootstrap state."""

    def save(self, state: BootstrapState) -> None:
        """Save bootstrap state."""
        raise NotImplementedError

    def load(self) -> Optional[BootstrapState]:
        """Load bootstrap state."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear stored state."""
        raise NotImplementedError

    def exists(self) -> bool:
        """Check if state exists."""
        raise NotImplementedError


class MemoryBootstrapStateStore(BootstrapStateStore):
    """In-memory bootstrap state store."""

    def __init__(self):
        self._state: Optional[BootstrapState] = None

    def save(self, state: BootstrapState) -> None:
        """Save state to memory."""
        self._state = state

    def load(self) -> Optional[BootstrapState]:
        """Load state from memory."""
        return self._state

    def clear(self) -> None:
        """Clear stored state."""
        self._state = None

    def exists(self) -> bool:
        """Check if state exists."""
        return self._state is not None


class FileBootstrapStateStore(BootstrapStateStore):
    """File-based bootstrap state store."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, state: BootstrapState) -> None:
        """Save state to file."""
        self.path.write_text(state.to_json())

    def load(self) -> Optional[BootstrapState]:
        """Load state from file."""
        if not self.path.exists():
            return None

        try:
            json_str = self.path.read_text()
            return BootstrapState.from_json(json_str)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise BootstrapError(f"Invalid state file: {e}")

    def clear(self) -> None:
        """Clear stored state."""
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        """Check if state file exists."""
        return self.path.exists()


# =============================================================================
# BootstrapStatusResponse
# =============================================================================


@dataclass
class BootstrapStatusResponse:
    """API response format for bootstrap status."""

    phase: IndexingPhase
    percentage: float
    indexed_count: int
    total_count: int
    failed_count: int = 0
    eta_seconds: Optional[float] = None
    files_per_second: Optional[float] = None
    current_file: Optional[Path] = None
    hot_files_remaining: Optional[int] = None
    warm_files_remaining: Optional[int] = None
    cold_files_remaining: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "phase": self.phase.value,
            "percentage": self.percentage,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "failed_count": self.failed_count,
        }

        if self.eta_seconds is not None:
            result["eta_seconds"] = self.eta_seconds
        if self.files_per_second is not None:
            result["files_per_second"] = self.files_per_second
        if self.current_file is not None:
            result["current_file"] = str(self.current_file)
        if self.hot_files_remaining is not None:
            result["hot_files_remaining"] = self.hot_files_remaining
        if self.warm_files_remaining is not None:
            result["warm_files_remaining"] = self.warm_files_remaining
        if self.cold_files_remaining is not None:
            result["cold_files_remaining"] = self.cold_files_remaining

        return result

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())


# =============================================================================
# BootstrapStatus Service
# =============================================================================


class BootstrapStatus:
    """Service for getting bootstrap status."""

    def __init__(self, state: BootstrapState):
        self._state = state

    def get_status(self) -> BootstrapStatusResponse:
        """Get current bootstrap status."""
        tier_counts = self._state.queue.count_by_tier()

        return BootstrapStatusResponse(
            phase=self._state.progress.phase,
            percentage=self._state.progress.percentage,
            indexed_count=self._state.progress.indexed_files,
            total_count=self._state.progress.total_files,
            failed_count=self._state.progress.failed_files,
            eta_seconds=self._state.progress.eta_seconds,
            files_per_second=self._state.progress.files_per_second,
            hot_files_remaining=tier_counts.get(PriorityTier.HOT, 0),
            warm_files_remaining=tier_counts.get(PriorityTier.WARM, 0),
            cold_files_remaining=tier_counts.get(PriorityTier.COLD, 0),
        )


# =============================================================================
# FileProgress
# =============================================================================


@dataclass
class FileProgress:
    """Progress info for a single file."""

    path: Path
    success: bool
    error: Optional[str] = None


# =============================================================================
# Callback Types
# =============================================================================

ProgressCallback = Callable[[BootstrapProgress], None]
FileCompleteCallback = Callable[[Path, bool], None]
PhaseChangeCallback = Callable[[IndexingPhase, IndexingPhase], None]


# =============================================================================
# BootstrapConfig
# =============================================================================


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap manager."""

    hot_threshold: float = 0.7
    warm_threshold: float = 0.4
    batch_size: int = 50
    parallel_workers: int = 4
    recency_days: int = 7
    extensions: list[str] = field(
        default_factory=lambda: [".py", ".ts", ".tsx", ".java", ".js", ".jsx"]
    )
    ignore_patterns: list[str] = field(
        default_factory=lambda: ["__pycache__", "node_modules", ".git", "venv", ".venv"]
    )


# =============================================================================
# ScanResult
# =============================================================================


@dataclass
class ScanResult:
    """Result of scanning directory for files."""

    total_files: int
    queue: PriorityQueue


# =============================================================================
# IndexBatchResult
# =============================================================================


@dataclass
class IndexBatchResult:
    """Result of indexing a batch of files."""

    indexed_count: int
    failed_count: int
    indexed_files: list[Path] = field(default_factory=list)
    failed_files: list[tuple[Path, str]] = field(default_factory=list)


# =============================================================================
# BootstrapManager
# =============================================================================


class BootstrapManager:
    """Manager for bootstrap indexing process."""

    def __init__(
        self,
        root_path: Path,
        config: Optional[BootstrapConfig] = None,
        state_store: Optional[BootstrapStateStore] = None,
        scorer: Optional[FilePriorityScorer] = None,
        indexer: Optional[Any] = None,
    ):
        self.root_path = root_path
        self.config = config or BootstrapConfig()
        self.state_store = state_store or MemoryBootstrapStateStore()
        self.scorer = scorer or FilePriorityScorer(
            hot_threshold=self.config.hot_threshold,
            warm_threshold=self.config.warm_threshold,
            recency_days=self.config.recency_days,
        )
        self.indexer = indexer

        # Load existing state or create new
        self._state: Optional[BootstrapState] = None
        self._current_file: Optional[Path] = None
        self._lock = threading.Lock()

    def _get_state(self) -> BootstrapState:
        """Get or create state."""
        if self._state is None:
            self._state = self.state_store.load()
            if self._state is None:
                self._state = BootstrapState.create_initial(self.root_path)
        return self._state

    def scan(self) -> ScanResult:
        """Scan directory for files to index.

        Returns:
            ScanResult with file count and priority queue
        """
        state = self._get_state()
        state.update_phase(IndexingPhase.SCANNING)

        # Collect all files
        file_priorities: list[FilePriority] = []

        for file_path in self.root_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix not in self.config.extensions:
                continue

            # Check ignore patterns
            if any(pattern in str(file_path) for pattern in self.config.ignore_patterns):
                continue

            # Skip already indexed files
            if state.is_indexed(file_path):
                continue

            # Score the file
            priority = self.scorer.score_file(file_path, self.root_path)
            file_priorities.append(priority)

        # Update state
        state.queue.push_all(file_priorities)
        total = len(file_priorities) + len(state.indexed_files)
        state.set_total_files(total)

        # Update phase based on queue state
        if len(file_priorities) == 0:
            state.update_phase(IndexingPhase.COMPLETED)
        else:
            tier_counts = state.queue.count_by_tier()
            if tier_counts[PriorityTier.HOT] > 0:
                state.update_phase(IndexingPhase.INDEXING_HOT)
            elif tier_counts[PriorityTier.WARM] > 0:
                state.update_phase(IndexingPhase.INDEXING_WARM)
            else:
                state.update_phase(IndexingPhase.INDEXING_COLD)

        # Save state
        self.state_store.save(state)

        return ScanResult(total_files=total, queue=state.queue)

    def index_batch(
        self,
        batch_size: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_file_complete: Optional[FileCompleteCallback] = None,
        on_phase_change: Optional[PhaseChangeCallback] = None,
    ) -> IndexBatchResult:
        """Index a batch of files from the priority queue.

        Args:
            batch_size: Number of files to index (defaults to config)
            on_progress: Callback for progress updates
            on_file_complete: Callback when a file is indexed
            on_phase_change: Callback when phase changes

        Returns:
            IndexBatchResult with indexed and failed counts
        """
        state = self._get_state()
        batch_size = batch_size or self.config.batch_size

        # Start timing if not already started
        if state.progress.start_time is None:
            state.progress.start_time = time.time()

        indexed_files: list[Path] = []
        failed_files: list[tuple[Path, str]] = []

        # Get batch from queue
        batch = state.queue.pop_batch(batch_size)

        old_phase = state.progress.phase

        for fp in batch:
            self._current_file = fp.path

            try:
                # Simulate indexing (actual indexing would go here)
                self._index_file(fp.path)

                state.mark_indexed(fp.path)
                indexed_files.append(fp.path)

                if on_file_complete:
                    on_file_complete(fp.path, True)

            except Exception as e:
                error_msg = str(e)
                state.mark_failed(fp.path, error_msg)
                failed_files.append((fp.path, error_msg))

                if on_file_complete:
                    on_file_complete(fp.path, False)

            # Update progress callback
            if on_progress:
                on_progress(state.progress)

        self._current_file = None

        # Update phase based on remaining items
        tier_counts = state.queue.count_by_tier()
        if state.queue.is_empty:
            new_phase = IndexingPhase.COMPLETED
        elif tier_counts[PriorityTier.HOT] > 0:
            new_phase = IndexingPhase.INDEXING_HOT
        elif tier_counts[PriorityTier.WARM] > 0:
            new_phase = IndexingPhase.INDEXING_WARM
        else:
            new_phase = IndexingPhase.INDEXING_COLD

        if new_phase != old_phase:
            state.update_phase(new_phase)
            if on_phase_change:
                on_phase_change(old_phase, new_phase)

        # Save state
        self.state_store.save(state)

        return IndexBatchResult(
            indexed_count=len(indexed_files),
            failed_count=len(failed_files),
            indexed_files=indexed_files,
            failed_files=failed_files,
        )

    def _index_file(self, path: Path) -> None:
        """Index a single file.

        This is a placeholder for actual indexing logic.
        Override or extend for real implementation.
        """
        if self.indexer is not None:
            self.indexer.index_file(path)
            return

        # Read file to verify it's accessible
        _ = path.read_bytes()

    def get_status(self) -> BootstrapStatusResponse:
        """Get current bootstrap status."""
        state = self._get_state()
        status = BootstrapStatus(state)
        response = status.get_status()

        if self._current_file:
            response.current_file = self._current_file

        return response

    def pause(self) -> None:
        """Pause indexing."""
        state = self._get_state()
        state.update_phase(IndexingPhase.PAUSED)
        self.state_store.save(state)

    def resume(self) -> None:
        """Resume indexing from saved state."""
        state = self._get_state()

        if state.progress.phase == IndexingPhase.PAUSED:
            # Determine phase from queue
            tier_counts = state.queue.count_by_tier()
            if state.queue.is_empty:
                state.update_phase(IndexingPhase.COMPLETED)
            elif tier_counts[PriorityTier.HOT] > 0:
                state.update_phase(IndexingPhase.INDEXING_HOT)
            elif tier_counts[PriorityTier.WARM] > 0:
                state.update_phase(IndexingPhase.INDEXING_WARM)
            else:
                state.update_phase(IndexingPhase.INDEXING_COLD)

            self.state_store.save(state)

        elif state.progress.phase == IndexingPhase.NOT_STARTED:
            # If not started, trigger a scan
            self.scan()

    @property
    def is_complete(self) -> bool:
        """Check if indexing is complete."""
        state = self._get_state()
        return state.progress.phase == IndexingPhase.COMPLETED


# =============================================================================
# Factory Function
# =============================================================================


def create_bootstrap_manager(
    root_path: Path,
    config: Optional[BootstrapConfig] = None,
    state_store: Optional[BootstrapStateStore] = None,
) -> BootstrapManager:
    """Create a bootstrap manager.

    Args:
        root_path: Root path of the project to index
        config: Optional configuration
        state_store: Optional state store for persistence

    Returns:
        BootstrapManager instance
    """
    return BootstrapManager(
        root_path=root_path,
        config=config,
        state_store=state_store,
    )
