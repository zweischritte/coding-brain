"""
Tests for IndexingWorker.

TDD: These tests are written FIRST before the worker implementation.
They define the expected behavior for the background indexing worker.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Generator, Optional, Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import CodeIndexJob, CodeIndexJobStatus


# Test constants
TEST_REPO_ID = "coding-brain"
TEST_ROOT_PATH = "/path/to/repo"
TEST_INDEX_NAME = "code"
TEST_USER_ID = "test-user-123"


class MockCodeIndexingService:
    """Mock CodeIndexingService for testing."""

    def __init__(
        self,
        should_fail: bool = False,
        fail_message: str = "Indexing failed",
        files_indexed: int = 100,
    ):
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.files_indexed = files_indexed
        self.index_repository_called = False
        self.progress_callbacks: list[Any] = []

    def index_repository(
        self,
        max_files: Optional[int] = None,
        reset: bool = False,
        progress_callback: Optional[callable] = None,
    ):
        """Mock index_repository that tracks calls and optionally fails."""
        self.index_repository_called = True

        if progress_callback:
            # Simulate some progress updates
            progress_callback({
                "files_scanned": 50,
                "files_indexed": 25,
                "files_failed": 0,
                "current_phase": "parse"
            })

        if self.should_fail:
            raise Exception(self.fail_message)

        # Return a mock summary
        return MagicMock(
            repo_id=TEST_REPO_ID,
            files_indexed=self.files_indexed,
            files_failed=0,
            symbols_indexed=500,
            documents_indexed=600,
            call_edges_indexed=200,
            duration_ms=5000.0,
        )


class MockValkeyClient:
    """Mock Valkey client for testing."""

    def __init__(self):
        self._lists: dict[str, list[bytes]] = {}

    def ping(self) -> bool:
        return True

    def lpush(self, key: str, *values: str) -> int:
        if key not in self._lists:
            self._lists[key] = []
        for v in values:
            self._lists[key].insert(0, v.encode() if isinstance(v, str) else v)
        return len(self._lists[key])

    def rpoplpush(self, src: str, dst: str) -> Optional[bytes]:
        if src not in self._lists or not self._lists[src]:
            return None
        value = self._lists[src].pop()
        if dst not in self._lists:
            self._lists[dst] = []
        self._lists[dst].insert(0, value)
        return value

    def lrem(self, key: str, count: int, value: str) -> int:
        if key not in self._lists:
            return 0
        value_bytes = value.encode() if isinstance(value, str) else value
        removed = 0
        while value_bytes in self._lists[key]:
            self._lists[key].remove(value_bytes)
            removed += 1
            if count > 0 and removed >= count:
                break
        return removed

    def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))


@pytest.fixture
def mock_valkey_client() -> MockValkeyClient:
    """Create a mock Valkey client."""
    return MockValkeyClient()


@pytest.fixture
def sqlite_test_db_for_worker() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for worker tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def queued_job(sqlite_test_db_for_worker: Session) -> CodeIndexJob:
    """Create a queued job for testing."""
    job = CodeIndexJob(
        repo_id=TEST_REPO_ID,
        root_path=TEST_ROOT_PATH,
        index_name=TEST_INDEX_NAME,
        requested_by=TEST_USER_ID,
        status=CodeIndexJobStatus.queued,
        request={"max_files": 100, "reset": True}
    )
    sqlite_test_db_for_worker.add(job)
    sqlite_test_db_for_worker.commit()
    return job


class TestIndexingWorkerProcessJob:
    """Test worker job processing."""

    def test_process_job_calls_indexer(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
        queued_job: CodeIndexJob,
    ):
        """Worker should call the indexer for a job."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        mock_indexer = MockCodeIndexingService()
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )

        # Add job to queue
        mock_valkey_client.lpush("index:queue", str(queued_job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        worker.process_one_job()

        assert mock_indexer.index_repository_called is True

    def test_process_job_updates_status_to_succeeded(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
        queued_job: CodeIndexJob,
    ):
        """Successful job should be marked as succeeded."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        mock_indexer = MockCodeIndexingService()
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )
        mock_valkey_client.lpush("index:queue", str(queued_job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        worker.process_one_job()

        sqlite_test_db_for_worker.refresh(queued_job)
        assert queued_job.status == CodeIndexJobStatus.succeeded
        assert queued_job.summary is not None

    def test_process_job_updates_status_to_failed_on_error(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
        queued_job: CodeIndexJob,
    ):
        """Failed job should be marked as failed with error."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        mock_indexer = MockCodeIndexingService(
            should_fail=True,
            fail_message="Neo4j connection failed"
        )
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )
        mock_valkey_client.lpush("index:queue", str(queued_job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        worker.process_one_job()

        sqlite_test_db_for_worker.refresh(queued_job)
        assert queued_job.status == CodeIndexJobStatus.failed
        assert "Neo4j connection failed" in queued_job.error


class TestIndexingWorkerCancellation:
    """Test worker cancellation handling."""

    def test_process_job_respects_cancel_requested(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
    ):
        """Worker should stop if cancel_requested is True."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        # Create a job with cancel_requested=True
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.queued,
            cancel_requested=True
        )
        sqlite_test_db_for_worker.add(job)
        sqlite_test_db_for_worker.commit()

        mock_indexer = MockCodeIndexingService()
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )
        mock_valkey_client.lpush("index:queue", str(job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        worker.process_one_job()

        sqlite_test_db_for_worker.refresh(job)
        assert job.status == CodeIndexJobStatus.canceled
        assert mock_indexer.index_repository_called is False


class TestIndexingWorkerHeartbeat:
    """Test worker heartbeat updates."""

    def test_process_job_sends_heartbeats(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
        queued_job: CodeIndexJob,
    ):
        """Worker should send heartbeats during processing."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        mock_indexer = MockCodeIndexingService()
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )
        mock_valkey_client.lpush("index:queue", str(queued_job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        worker.process_one_job()

        sqlite_test_db_for_worker.refresh(queued_job)
        assert queued_job.last_heartbeat is not None


class TestIndexingWorkerOrphanRecovery:
    """Test orphan job recovery."""

    def test_recover_orphan_jobs_requeues_stale_jobs(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
    ):
        """Orphaned jobs should be requeued for retry."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        # Create an orphaned job (running with stale heartbeat)
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(minutes=10)

        orphan_job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=now - timedelta(minutes=15),
            last_heartbeat=stale_time,
            attempts=1
        )
        sqlite_test_db_for_worker.add(orphan_job)
        sqlite_test_db_for_worker.commit()

        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: MockCodeIndexingService(),
            orphan_threshold_minutes=5,
            max_attempts=3
        )

        recovered_count = worker.recover_orphan_jobs()

        assert recovered_count == 1

        sqlite_test_db_for_worker.refresh(orphan_job)
        assert orphan_job.status == CodeIndexJobStatus.queued

    def test_recover_orphan_jobs_fails_after_max_attempts(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
    ):
        """Jobs exceeding max attempts should be marked as failed."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(minutes=10)

        orphan_job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=now - timedelta(minutes=15),
            last_heartbeat=stale_time,
            attempts=3  # Already at max
        )
        sqlite_test_db_for_worker.add(orphan_job)
        sqlite_test_db_for_worker.commit()

        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: MockCodeIndexingService(),
            orphan_threshold_minutes=5,
            max_attempts=3
        )

        worker.recover_orphan_jobs()

        sqlite_test_db_for_worker.refresh(orphan_job)
        assert orphan_job.status == CodeIndexJobStatus.failed
        assert "max attempts" in orphan_job.error.lower()


class TestIndexingWorkerRunLoop:
    """Test worker main run loop."""

    def test_run_once_processes_available_job(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
        queued_job: CodeIndexJob,
    ):
        """run_once should process one available job."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        mock_indexer = MockCodeIndexingService()
        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )
        mock_valkey_client.lpush("index:queue", str(queued_job.id))

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: mock_indexer
        )

        processed = worker.run_once()

        assert processed is True
        sqlite_test_db_for_worker.refresh(queued_job)
        assert queued_job.status == CodeIndexJobStatus.succeeded

    def test_run_once_returns_false_when_no_jobs(
        self,
        sqlite_test_db_for_worker: Session,
        mock_valkey_client: MockValkeyClient,
    ):
        """run_once should return False when no jobs available."""
        from app.workers.indexing_worker import IndexingWorker
        from app.services.job_queue_service import IndexingJobQueueService

        queue_service = IndexingJobQueueService(
            db=sqlite_test_db_for_worker,
            valkey_client=mock_valkey_client
        )

        worker = IndexingWorker(
            queue_service=queue_service,
            indexer_factory=lambda root_path, repo_id, **kwargs: MockCodeIndexingService()
        )

        processed = worker.run_once()

        assert processed is False
