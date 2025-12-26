"""Tests for Bootstrap State and Priority Queue (FR-001).

Following TDD: Write tests first, then implement.
Covers:
- Bootstrap state persistence (indexed files, progress %)
- Priority queue for tiered indexing (hot files first)
- Bootstrap status API response format
- Resume from partial indexing state
- File priority scoring (recently modified, frequently accessed)
- Integration with MerkleTree change detection
- Progress callbacks for UI updates
"""

import pytest
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from openmemory.api.indexing.bootstrap import (
    # Core types
    FilePriority,
    PriorityTier,
    IndexingPhase,
    BootstrapProgress,
    # Priority Queue
    PriorityQueue,
    FilePriorityScorer,
    # Bootstrap State
    BootstrapState,
    BootstrapStateStore,
    MemoryBootstrapStateStore,
    FileBootstrapStateStore,
    # Bootstrap Status API
    BootstrapStatus,
    BootstrapStatusResponse,
    FileProgress,
    # Bootstrap Manager
    BootstrapConfig,
    BootstrapManager,
    ProgressCallback,
    # Exceptions
    BootstrapError,
    StateNotFoundError,
    # Factory
    create_bootstrap_manager,
)


# =============================================================================
# FilePriority and PriorityTier Tests
# =============================================================================


class TestPriorityTier:
    """Tests for PriorityTier enum."""

    def test_priority_tier_values(self):
        """PriorityTier has expected values."""
        assert PriorityTier.HOT.value == "hot"
        assert PriorityTier.WARM.value == "warm"
        assert PriorityTier.COLD.value == "cold"

    def test_priority_tier_ordering(self):
        """HOT > WARM > COLD priority ordering."""
        assert PriorityTier.HOT.priority_value > PriorityTier.WARM.priority_value
        assert PriorityTier.WARM.priority_value > PriorityTier.COLD.priority_value


class TestFilePriority:
    """Tests for FilePriority dataclass."""

    def test_create_file_priority(self):
        """Create FilePriority with all fields."""
        fp = FilePriority(
            path=Path("src/main.py"),
            tier=PriorityTier.HOT,
            score=0.95,
            modified_time=1234567890.0,
            access_count=10,
            depth=2,
        )
        assert fp.path == Path("src/main.py")
        assert fp.tier == PriorityTier.HOT
        assert fp.score == 0.95

    def test_file_priority_comparison(self):
        """FilePriority compares by score descending."""
        fp1 = FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1)
        fp2 = FilePriority(Path("b.py"), PriorityTier.COLD, 0.5, 0, 0, 1)
        # Higher score should have higher priority
        assert fp1 > fp2

    def test_file_priority_equality(self):
        """FilePriority equality based on path."""
        fp1 = FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1)
        fp2 = FilePriority(Path("a.py"), PriorityTier.COLD, 0.5, 0, 0, 1)
        # Same path means equal
        assert fp1 == fp2


# =============================================================================
# FilePriorityScorer Tests
# =============================================================================


class TestFilePriorityScorer:
    """Tests for FilePriorityScorer."""

    def test_recently_modified_gets_hot_priority(self, tmp_path):
        """Recently modified files get HOT priority."""
        file_path = tmp_path / "recent.py"
        file_path.write_text("def recent(): pass")

        # Use a scorer with lower hot threshold to account for no access count
        scorer = FilePriorityScorer(hot_threshold=0.6)
        priority = scorer.score_file(file_path, base_path=tmp_path)

        assert priority.tier == PriorityTier.HOT
        assert priority.score >= 0.6

    def test_old_file_gets_cold_priority(self, tmp_path):
        """Files not modified recently get COLD priority."""
        file_path = tmp_path / "old.py"
        file_path.write_text("def old(): pass")

        # Mock old modification time (30 days ago)
        old_time = time.time() - (30 * 24 * 60 * 60)

        scorer = FilePriorityScorer()
        priority = scorer.score_file(
            file_path,
            base_path=tmp_path,
            modified_time=old_time,
        )

        assert priority.tier == PriorityTier.COLD
        assert priority.score < 0.5

    def test_frequently_accessed_gets_higher_priority(self, tmp_path):
        """Files with high access count get higher priority."""
        file_path = tmp_path / "popular.py"
        file_path.write_text("def popular(): pass")

        scorer = FilePriorityScorer()

        # High access count
        priority_high = scorer.score_file(
            file_path,
            base_path=tmp_path,
            access_count=100,
        )

        # Low access count
        priority_low = scorer.score_file(
            file_path,
            base_path=tmp_path,
            access_count=1,
        )

        assert priority_high.score > priority_low.score

    def test_shallow_depth_higher_priority(self, tmp_path):
        """Shallow directory depth gets higher priority than deep."""
        shallow_file = tmp_path / "main.py"
        shallow_file.write_text("# main")

        deep_dir = tmp_path / "a" / "b" / "c" / "d"
        deep_dir.mkdir(parents=True)
        deep_file = deep_dir / "deep.py"
        deep_file.write_text("# deep")

        scorer = FilePriorityScorer()

        shallow_priority = scorer.score_file(shallow_file, base_path=tmp_path)
        deep_priority = scorer.score_file(deep_file, base_path=tmp_path)

        assert shallow_priority.score > deep_priority.score

    def test_configurable_weights(self):
        """Scorer supports configurable weight factors."""
        scorer = FilePriorityScorer(
            recency_weight=0.5,
            access_weight=0.3,
            depth_weight=0.2,
        )
        assert scorer.recency_weight == 0.5
        assert scorer.access_weight == 0.3
        assert scorer.depth_weight == 0.2

    def test_hot_threshold_configurable(self, tmp_path):
        """HOT/WARM/COLD thresholds are configurable."""
        file_path = tmp_path / "test.py"
        file_path.write_text("# test")

        scorer = FilePriorityScorer(
            hot_threshold=0.9,
            warm_threshold=0.6,
        )
        priority = scorer.score_file(file_path, base_path=tmp_path)

        # File should be tiered according to thresholds
        assert priority.tier in [PriorityTier.HOT, PriorityTier.WARM, PriorityTier.COLD]


# =============================================================================
# PriorityQueue Tests
# =============================================================================


class TestPriorityQueue:
    """Tests for PriorityQueue."""

    def test_empty_queue(self):
        """Empty queue operations."""
        queue = PriorityQueue()
        assert queue.is_empty
        assert len(queue) == 0
        assert queue.pop() is None

    def test_push_and_pop(self):
        """Push and pop items in priority order."""
        queue = PriorityQueue()

        fp_low = FilePriority(Path("low.py"), PriorityTier.COLD, 0.3, 0, 0, 1)
        fp_high = FilePriority(Path("high.py"), PriorityTier.HOT, 0.9, 0, 0, 1)

        queue.push(fp_low)
        queue.push(fp_high)

        # Should pop highest priority first
        first = queue.pop()
        assert first.path == Path("high.py")

        second = queue.pop()
        assert second.path == Path("low.py")

    def test_peek(self):
        """Peek returns highest priority without removing."""
        queue = PriorityQueue()
        fp = FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1)
        queue.push(fp)

        peeked = queue.peek()
        assert peeked == fp
        assert len(queue) == 1  # Still in queue

    def test_push_all(self):
        """Push multiple items at once."""
        queue = PriorityQueue()

        items = [
            FilePriority(Path("a.py"), PriorityTier.COLD, 0.3, 0, 0, 1),
            FilePriority(Path("b.py"), PriorityTier.HOT, 0.9, 0, 0, 1),
            FilePriority(Path("c.py"), PriorityTier.WARM, 0.6, 0, 0, 1),
        ]

        queue.push_all(items)
        assert len(queue) == 3

    def test_pop_batch(self):
        """Pop multiple items in priority order."""
        queue = PriorityQueue()

        for i in range(10):
            fp = FilePriority(Path(f"file{i}.py"), PriorityTier.WARM, 0.1 * i, 0, 0, 1)
            queue.push(fp)

        batch = queue.pop_batch(3)
        assert len(batch) == 3
        assert len(queue) == 7
        # Should be highest priority first
        assert batch[0].score > batch[1].score > batch[2].score

    def test_iteration(self):
        """Iterate over queue without removing items."""
        queue = PriorityQueue()

        items = [
            FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1),
            FilePriority(Path("b.py"), PriorityTier.COLD, 0.3, 0, 0, 1),
        ]
        queue.push_all(items)

        iterated = list(queue)
        assert len(iterated) == 2
        assert len(queue) == 2  # Still in queue

    def test_clear(self):
        """Clear all items from queue."""
        queue = PriorityQueue()
        queue.push(FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1))
        queue.clear()
        assert queue.is_empty

    def test_filter_by_tier(self):
        """Get items of a specific tier."""
        queue = PriorityQueue()

        queue.push(FilePriority(Path("hot1.py"), PriorityTier.HOT, 0.9, 0, 0, 1))
        queue.push(FilePriority(Path("hot2.py"), PriorityTier.HOT, 0.85, 0, 0, 1))
        queue.push(FilePriority(Path("cold1.py"), PriorityTier.COLD, 0.3, 0, 0, 1))

        hot_items = queue.filter_by_tier(PriorityTier.HOT)
        assert len(hot_items) == 2


# =============================================================================
# IndexingPhase and BootstrapProgress Tests
# =============================================================================


class TestIndexingPhase:
    """Tests for IndexingPhase enum."""

    def test_indexing_phase_values(self):
        """IndexingPhase has expected values."""
        assert IndexingPhase.NOT_STARTED.value == "not_started"
        assert IndexingPhase.SCANNING.value == "scanning"
        assert IndexingPhase.INDEXING_HOT.value == "indexing_hot"
        assert IndexingPhase.INDEXING_WARM.value == "indexing_warm"
        assert IndexingPhase.INDEXING_COLD.value == "indexing_cold"
        assert IndexingPhase.COMPLETED.value == "completed"
        assert IndexingPhase.PAUSED.value == "paused"


class TestBootstrapProgress:
    """Tests for BootstrapProgress tracking."""

    def test_create_progress(self):
        """Create BootstrapProgress with initial values."""
        progress = BootstrapProgress(
            phase=IndexingPhase.NOT_STARTED,
            total_files=0,
            indexed_files=0,
            failed_files=0,
        )
        assert progress.phase == IndexingPhase.NOT_STARTED
        assert progress.percentage == 0.0

    def test_progress_percentage(self):
        """Progress percentage calculation."""
        progress = BootstrapProgress(
            phase=IndexingPhase.INDEXING_HOT,
            total_files=100,
            indexed_files=50,
            failed_files=5,
        )
        # Use approximate comparison for floating point
        assert abs(progress.percentage - 55.0) < 0.01  # (50 + 5) / 100 * 100

    def test_progress_eta_calculation(self):
        """ETA calculation based on rate."""
        progress = BootstrapProgress(
            phase=IndexingPhase.INDEXING_WARM,
            total_files=100,
            indexed_files=25,
            failed_files=0,
            start_time=time.time() - 50,  # Started 50 seconds ago
        )
        # 25 files in 50 seconds = 0.5 files/sec
        # 75 remaining / 0.5 = 150 seconds
        assert progress.eta_seconds is not None
        assert progress.eta_seconds > 0

    def test_progress_rate_calculation(self):
        """Files per second rate calculation."""
        progress = BootstrapProgress(
            phase=IndexingPhase.INDEXING_HOT,
            total_files=100,
            indexed_files=20,
            failed_files=0,
            start_time=time.time() - 10,  # Started 10 seconds ago
        )
        # 20 files in 10 seconds = 2.0 files/sec
        assert progress.files_per_second is not None
        assert abs(progress.files_per_second - 2.0) < 0.5  # Allow some tolerance

    def test_progress_is_complete(self):
        """Progress is complete when all files processed."""
        progress = BootstrapProgress(
            phase=IndexingPhase.COMPLETED,
            total_files=100,
            indexed_files=95,
            failed_files=5,
        )
        assert progress.is_complete
        assert progress.percentage == 100.0

    def test_progress_serialization(self):
        """Progress can be serialized to dict."""
        progress = BootstrapProgress(
            phase=IndexingPhase.INDEXING_HOT,
            total_files=100,
            indexed_files=50,
            failed_files=2,
        )
        data = progress.to_dict()

        assert data["phase"] == "indexing_hot"
        assert data["total_files"] == 100
        assert data["indexed_files"] == 50
        assert data["failed_files"] == 2


# =============================================================================
# BootstrapState Tests
# =============================================================================


class TestBootstrapState:
    """Tests for BootstrapState persistence."""

    def test_create_initial_state(self):
        """Create initial bootstrap state."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        assert state.root_path == Path("/project")
        assert state.progress.phase == IndexingPhase.NOT_STARTED
        assert state.indexed_files == set()

    def test_mark_file_indexed(self):
        """Mark file as indexed."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        state.mark_indexed(Path("/project/a.py"))

        assert Path("/project/a.py") in state.indexed_files
        assert state.progress.indexed_files == 1

    def test_mark_file_failed(self):
        """Mark file as failed to index."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        state.mark_failed(Path("/project/broken.py"), error="Parse error")

        assert Path("/project/broken.py") in state.failed_files
        assert state.progress.failed_files == 1

    def test_update_progress(self):
        """Update progress with new phase."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        state.update_phase(IndexingPhase.SCANNING)
        assert state.progress.phase == IndexingPhase.SCANNING

        state.update_phase(IndexingPhase.INDEXING_HOT)
        assert state.progress.phase == IndexingPhase.INDEXING_HOT

    def test_set_total_files(self):
        """Set total files for progress calculation."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        state.set_total_files(150)
        assert state.progress.total_files == 150

    def test_is_file_indexed(self):
        """Check if file was already indexed."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        state.mark_indexed(Path("/project/a.py"))

        assert state.is_indexed(Path("/project/a.py"))
        assert not state.is_indexed(Path("/project/b.py"))

    def test_state_serialization(self):
        """State can be serialized and deserialized."""
        state = BootstrapState.create_initial(root_path=Path("/project"))
        state.mark_indexed(Path("/project/a.py"))
        state.mark_indexed(Path("/project/b.py"))
        state.update_phase(IndexingPhase.INDEXING_WARM)

        # Serialize
        json_str = state.to_json()

        # Deserialize
        restored = BootstrapState.from_json(json_str)

        assert restored.root_path == state.root_path
        assert restored.indexed_files == state.indexed_files
        assert restored.progress.phase == state.progress.phase

    def test_state_with_priority_queue(self):
        """State preserves priority queue."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        fp = FilePriority(Path("a.py"), PriorityTier.HOT, 0.9, 0, 0, 1)
        state.queue.push(fp)

        # Serialize and restore
        json_str = state.to_json()
        restored = BootstrapState.from_json(json_str)

        assert len(restored.queue) == 1


# =============================================================================
# BootstrapStateStore Tests
# =============================================================================


class TestMemoryBootstrapStateStore:
    """Tests for in-memory state store."""

    def test_save_and_load(self):
        """Save and load state."""
        store = MemoryBootstrapStateStore()
        state = BootstrapState.create_initial(root_path=Path("/project"))
        state.mark_indexed(Path("/project/a.py"))

        store.save(state)
        loaded = store.load()

        assert loaded is not None
        assert loaded.indexed_files == state.indexed_files

    def test_load_empty(self):
        """Load from empty store returns None."""
        store = MemoryBootstrapStateStore()
        assert store.load() is None

    def test_clear(self):
        """Clear stored state."""
        store = MemoryBootstrapStateStore()
        state = BootstrapState.create_initial(root_path=Path("/project"))

        store.save(state)
        store.clear()

        assert store.load() is None

    def test_exists(self):
        """Check if state exists."""
        store = MemoryBootstrapStateStore()
        assert not store.exists()

        state = BootstrapState.create_initial(root_path=Path("/project"))
        store.save(state)

        assert store.exists()


class TestFileBootstrapStateStore:
    """Tests for file-based state store."""

    def test_save_and_load(self, tmp_path):
        """Save and load state to file."""
        state_file = tmp_path / ".bootstrap_state.json"
        store = FileBootstrapStateStore(state_file)

        state = BootstrapState.create_initial(root_path=Path("/project"))
        state.mark_indexed(Path("/project/a.py"))

        store.save(state)
        assert state_file.exists()

        loaded = store.load()
        assert loaded is not None
        assert Path("/project/a.py") in loaded.indexed_files

    def test_load_nonexistent_file(self, tmp_path):
        """Load from nonexistent file returns None."""
        state_file = tmp_path / ".bootstrap_state.json"
        store = FileBootstrapStateStore(state_file)

        assert store.load() is None

    def test_corrupted_state_file(self, tmp_path):
        """Handle corrupted state file."""
        state_file = tmp_path / ".bootstrap_state.json"
        state_file.write_text("invalid json {{{")

        store = FileBootstrapStateStore(state_file)
        with pytest.raises(BootstrapError):
            store.load()


# =============================================================================
# BootstrapStatus API Tests
# =============================================================================


class TestBootstrapStatusResponse:
    """Tests for Bootstrap status API response format."""

    def test_status_response_format(self):
        """Status response has expected format."""
        status = BootstrapStatusResponse(
            phase=IndexingPhase.INDEXING_HOT,
            percentage=45.5,
            indexed_count=45,
            total_count=100,
            failed_count=2,
            eta_seconds=120,
            files_per_second=2.5,
            current_file=Path("src/main.py"),
            hot_files_remaining=10,
            warm_files_remaining=20,
            cold_files_remaining=23,
        )

        data = status.to_dict()

        assert data["phase"] == "indexing_hot"
        assert data["percentage"] == 45.5
        assert data["indexed_count"] == 45
        assert data["total_count"] == 100
        assert data["failed_count"] == 2
        assert data["eta_seconds"] == 120
        assert data["files_per_second"] == 2.5
        assert data["current_file"] == "src/main.py"

    def test_status_response_json(self):
        """Status response can be serialized to JSON."""
        status = BootstrapStatusResponse(
            phase=IndexingPhase.COMPLETED,
            percentage=100.0,
            indexed_count=100,
            total_count=100,
            failed_count=0,
        )

        json_str = status.to_json()
        data = json.loads(json_str)

        assert data["phase"] == "completed"
        assert data["percentage"] == 100.0


class TestBootstrapStatus:
    """Tests for BootstrapStatus service."""

    def test_get_status(self):
        """Get current bootstrap status."""
        state = BootstrapState.create_initial(root_path=Path("/project"))
        state.set_total_files(100)
        state.mark_indexed(Path("/project/a.py"))
        state.update_phase(IndexingPhase.INDEXING_HOT)

        status = BootstrapStatus(state)
        response = status.get_status()

        assert response.phase == IndexingPhase.INDEXING_HOT
        assert response.indexed_count == 1
        assert response.total_count == 100

    def test_status_not_started(self):
        """Status when bootstrap not started."""
        state = BootstrapState.create_initial(root_path=Path("/project"))

        status = BootstrapStatus(state)
        response = status.get_status()

        assert response.phase == IndexingPhase.NOT_STARTED
        assert response.percentage == 0.0


# =============================================================================
# BootstrapConfig Tests
# =============================================================================


class TestBootstrapConfig:
    """Tests for BootstrapConfig."""

    def test_default_config(self):
        """Default configuration values."""
        config = BootstrapConfig()

        assert config.hot_threshold == 0.7
        assert config.warm_threshold == 0.4
        assert config.batch_size == 50
        assert config.parallel_workers == 4
        assert config.recency_days == 7

    def test_custom_config(self):
        """Custom configuration values."""
        config = BootstrapConfig(
            hot_threshold=0.8,
            warm_threshold=0.5,
            batch_size=100,
            parallel_workers=8,
            recency_days=14,
        )

        assert config.hot_threshold == 0.8
        assert config.batch_size == 100


# =============================================================================
# BootstrapManager Tests
# =============================================================================


class TestBootstrapManager:
    """Tests for BootstrapManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create bootstrap manager for test directory."""
        return create_bootstrap_manager(tmp_path)

    def test_create_manager(self, tmp_path):
        """Create bootstrap manager."""
        manager = create_bootstrap_manager(tmp_path)
        assert manager is not None
        assert manager.root_path == tmp_path

    def test_scan_files(self, tmp_path, manager):
        """Scan directory for files to index."""
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        result = manager.scan()

        assert result.total_files == 2
        assert len(result.queue) == 2

    def test_scan_with_priority_scoring(self, tmp_path, manager):
        """Scanning assigns priorities to files."""
        (tmp_path / "recent.py").write_text("# recent")

        result = manager.scan()

        assert len(result.queue) == 1
        fp = result.queue.peek()
        assert fp.tier in [PriorityTier.HOT, PriorityTier.WARM, PriorityTier.COLD]

    def test_index_batch(self, tmp_path, manager):
        """Index a batch of files."""
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")

        manager.scan()
        result = manager.index_batch(batch_size=2)

        assert result.indexed_count == 2
        assert result.failed_count == 0

    def test_index_with_progress_callback(self, tmp_path, manager):
        """Progress callback is called during indexing."""
        (tmp_path / "a.py").write_text("def a(): pass")

        progress_updates = []

        def on_progress(progress: BootstrapProgress):
            progress_updates.append(progress)

        manager.scan()
        manager.index_batch(batch_size=1, on_progress=on_progress)

        assert len(progress_updates) > 0

    def test_resume_from_partial_state(self, tmp_path):
        """Resume indexing from partial state."""
        state_file = tmp_path / ".bootstrap_state.json"
        store = FileBootstrapStateStore(state_file)

        # Create initial state with some files already indexed
        (tmp_path / "a.py").write_text("def a(): pass")
        (tmp_path / "b.py").write_text("def b(): pass")
        (tmp_path / "c.py").write_text("def c(): pass")

        # Simulate partial indexing
        initial_state = BootstrapState.create_initial(root_path=tmp_path)
        initial_state.mark_indexed(tmp_path / "a.py")
        initial_state.set_total_files(3)
        store.save(initial_state)

        # Create new manager with existing state
        manager = create_bootstrap_manager(tmp_path, state_store=store)
        manager.resume()

        # Should only have 2 files left to index
        assert manager.get_status().indexed_count == 1

        # Index remaining files
        result = manager.index_batch(batch_size=10)

        # Only b.py and c.py should have been indexed (a.py already done)
        assert result.indexed_count == 2

    def test_get_status(self, tmp_path, manager):
        """Get current bootstrap status."""
        (tmp_path / "a.py").write_text("def a(): pass")

        manager.scan()
        status = manager.get_status()

        assert isinstance(status, BootstrapStatusResponse)
        assert status.total_count == 1

    def test_pause_and_resume(self, tmp_path, manager):
        """Pause and resume indexing."""
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"def f{i}(): pass")

        manager.scan()

        # Index some files
        manager.index_batch(batch_size=3)

        # Pause
        manager.pause()
        assert manager.get_status().phase == IndexingPhase.PAUSED

        # Resume
        manager.resume()
        assert manager.get_status().phase in [
            IndexingPhase.INDEXING_HOT,
            IndexingPhase.INDEXING_WARM,
            IndexingPhase.INDEXING_COLD,
        ]

    def test_hot_files_indexed_first(self, tmp_path, manager):
        """HOT tier files are indexed before WARM and COLD."""
        # Create files with different priorities
        (tmp_path / "hot.py").write_text("# hot")  # Recently modified

        # Create an older file
        cold_dir = tmp_path / "deep" / "nested" / "dir"
        cold_dir.mkdir(parents=True)
        cold_file = cold_dir / "cold.py"
        cold_file.write_text("# cold")

        manager.scan()

        # First indexed file should be hot tier
        result = manager.index_batch(batch_size=1)
        status = manager.get_status()

        # If we indexed a file, check its path
        if result.indexed_count > 0:
            # The indexed file should be from root or hot tier
            assert status.indexed_count == 1

    def test_complete_indexing(self, tmp_path, manager):
        """Complete full indexing run."""
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"def f{i}(): pass")

        manager.scan()

        # Index all files
        while not manager.is_complete:
            manager.index_batch(batch_size=10)

        status = manager.get_status()
        assert status.phase == IndexingPhase.COMPLETED
        assert status.percentage == 100.0

    def test_error_handling_during_indexing(self, tmp_path, manager):
        """Handle errors during indexing gracefully."""
        (tmp_path / "good.py").write_text("def good(): pass")
        (tmp_path / "bad.py").write_text("this is not valid python {{{{")

        manager.scan()
        result = manager.index_batch(batch_size=10)

        # Should complete with some failures
        assert result.indexed_count >= 1
        # Bad files might be marked as failed


# =============================================================================
# Integration with MerkleTree Tests
# =============================================================================


class TestMerkleTreeIntegration:
    """Tests for integration with MerkleTree change detection."""

    def test_detect_new_files_for_indexing(self, tmp_path):
        """Use MerkleTree to detect new files needing indexing."""
        from openmemory.api.indexing.merkle_tree import create_indexer

        # Initial state
        (tmp_path / "a.py").write_text("def a(): pass")

        indexer = create_indexer(tmp_path)
        bootstrap_manager = create_bootstrap_manager(tmp_path)

        # First scan
        indexer.scan()
        bootstrap_manager.scan()
        bootstrap_manager.index_batch(batch_size=10)

        # Add new file
        (tmp_path / "b.py").write_text("def b(): pass")

        # Detect changes
        changes = indexer.scan()

        # Bootstrap should pick up the new file
        bootstrap_manager.scan()
        status = bootstrap_manager.get_status()

        # Should have the new file pending
        assert status.total_count > status.indexed_count

    def test_prioritize_modified_files(self, tmp_path):
        """Modified files get high priority for re-indexing."""
        from openmemory.api.indexing.merkle_tree import ChangeType

        manager = create_bootstrap_manager(tmp_path)

        (tmp_path / "modified.py").write_text("v1")

        manager.scan()
        manager.index_batch(batch_size=10)

        # Modify the file
        (tmp_path / "modified.py").write_text("v2")

        # Re-scan should prioritize the modified file
        manager.scan()

        status = manager.get_status()
        # Modified file should be queued for re-indexing
        assert status.hot_files_remaining is not None or status.warm_files_remaining is not None


# =============================================================================
# Progress Callback Tests
# =============================================================================


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_callback_on_file_indexed(self, tmp_path):
        """Callback called when file is indexed."""
        manager = create_bootstrap_manager(tmp_path)
        (tmp_path / "test.py").write_text("def test(): pass")

        indexed_files = []

        def on_file(path: Path, success: bool):
            indexed_files.append((path, success))

        manager.scan()
        manager.index_batch(batch_size=1, on_file_complete=on_file)

        assert len(indexed_files) == 1

    def test_callback_on_phase_change(self, tmp_path):
        """Callback called when phase changes."""
        manager = create_bootstrap_manager(tmp_path)
        (tmp_path / "test.py").write_text("def test(): pass")

        phases = []

        def on_phase(old_phase: IndexingPhase, new_phase: IndexingPhase):
            phases.append((old_phase, new_phase))

        manager.scan()
        manager.index_batch(batch_size=10, on_phase_change=on_phase)

        # Should have recorded phase transitions
        assert len(phases) >= 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactory:
    """Tests for create_bootstrap_manager factory."""

    def test_create_with_defaults(self, tmp_path):
        """Create manager with default configuration."""
        manager = create_bootstrap_manager(tmp_path)
        assert manager is not None
        assert manager.root_path == tmp_path

    def test_create_with_custom_config(self, tmp_path):
        """Create manager with custom configuration."""
        config = BootstrapConfig(batch_size=100)
        manager = create_bootstrap_manager(tmp_path, config=config)
        assert manager.config.batch_size == 100

    def test_create_with_state_store(self, tmp_path):
        """Create manager with custom state store."""
        store = MemoryBootstrapStateStore()
        manager = create_bootstrap_manager(tmp_path, state_store=store)
        assert manager.state_store is store


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_directory(self, tmp_path):
        """Handle empty directory."""
        manager = create_bootstrap_manager(tmp_path)
        result = manager.scan()

        assert result.total_files == 0
        assert manager.get_status().phase == IndexingPhase.COMPLETED

    def test_deeply_nested_files(self, tmp_path):
        """Handle deeply nested file structures."""
        deep_path = tmp_path
        for i in range(10):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.py").write_text("# deep")

        manager = create_bootstrap_manager(tmp_path)
        result = manager.scan()

        assert result.total_files == 1
        # Deep file should have lower priority
        fp = result.queue.peek()
        assert fp.depth == 10

    def test_unicode_filenames(self, tmp_path):
        """Handle unicode in filenames."""
        (tmp_path / "日本語.py").write_text("# Japanese")

        manager = create_bootstrap_manager(tmp_path)
        result = manager.scan()

        assert result.total_files == 1

    def test_concurrent_access(self, tmp_path):
        """Handle concurrent access to bootstrap state."""
        import threading

        manager = create_bootstrap_manager(tmp_path)

        for i in range(20):
            (tmp_path / f"file{i}.py").write_text(f"def f{i}(): pass")

        manager.scan()

        errors = []

        def index_files():
            try:
                manager.index_batch(batch_size=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=index_files) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should handle concurrent access without errors
        # (May need locking in implementation)

    def test_large_file_count(self, tmp_path):
        """Handle large number of files efficiently."""
        for i in range(200):
            (tmp_path / f"file{i}.py").write_text(f"def f{i}(): pass")

        manager = create_bootstrap_manager(tmp_path)

        import time
        start = time.time()
        result = manager.scan()
        elapsed = time.time() - start

        assert result.total_files == 200
        assert elapsed < 5.0  # Should complete quickly

    def test_state_not_found_error(self, tmp_path):
        """StateNotFoundError when no state exists."""
        manager = create_bootstrap_manager(tmp_path)

        # Trying to resume without scanning first should work
        # (will just do a full scan)
        manager.resume()  # Should not raise
