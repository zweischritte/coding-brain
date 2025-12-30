"""
Tests for IndexingJobQueueService.

TDD: These tests are written FIRST before the service implementation.
They define the expected behavior for Valkey-backed job queue operations.
"""
import uuid
from datetime import datetime, timezone
from typing import Generator, Optional
from unittest.mock import MagicMock, patch

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


class MockValkeyClient:
    """Mock Valkey client for testing queue operations."""

    def __init__(self, fail_on_connect: bool = False):
        self.fail_on_connect = fail_on_connect
        self._data: dict[str, bytes] = {}
        self._lists: dict[str, list[bytes]] = {}
        self._connected = not fail_on_connect

    def ping(self) -> bool:
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
        return True

    def lpush(self, key: str, *values: str) -> int:
        """Push values to the left of a list."""
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
        if key not in self._lists:
            self._lists[key] = []
        for v in values:
            self._lists[key].insert(0, v.encode() if isinstance(v, str) else v)
        return len(self._lists[key])

    def rpoplpush(self, src: str, dst: str) -> Optional[bytes]:
        """Pop from right of src, push to left of dst atomically."""
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
        if src not in self._lists or not self._lists[src]:
            return None
        value = self._lists[src].pop()
        if dst not in self._lists:
            self._lists[dst] = []
        self._lists[dst].insert(0, value)
        return value

    def brpoplpush(self, src: str, dst: str, timeout: int = 0) -> Optional[bytes]:
        """Blocking pop from right of src, push to left of dst."""
        # For tests, just use non-blocking version
        return self.rpoplpush(src, dst)

    def lrem(self, key: str, count: int, value: str) -> int:
        """Remove elements from a list."""
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
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
        """Get length of a list."""
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
        return len(self._lists.get(key, []))

    def lrange(self, key: str, start: int, stop: int) -> list[bytes]:
        """Get range of elements from a list."""
        if self.fail_on_connect:
            raise ConnectionError("Connection refused")
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start:stop + 1]


@pytest.fixture
def mock_valkey_client() -> MockValkeyClient:
    """Create a mock Valkey client for testing."""
    return MockValkeyClient()


@pytest.fixture
def mock_valkey_client_failing() -> MockValkeyClient:
    """Create a mock Valkey client that fails on connection."""
    return MockValkeyClient(fail_on_connect=True)


@pytest.fixture
def sqlite_test_db_for_queue() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for queue service tests."""
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


class TestIndexingJobQueueServiceCreate:
    """Test job creation in the queue service."""

    def test_create_job_returns_job_id(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job should return a valid job_id."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )
        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            request={"max_files": 100, "reset": True}
        )

        assert job_id is not None
        assert isinstance(job_id, uuid.UUID)

    def test_create_job_persists_to_database(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job should persist it to the database."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )
        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        assert job is not None
        assert job.repo_id == TEST_REPO_ID
        assert job.status == CodeIndexJobStatus.queued

    def test_create_job_adds_to_valkey_queue(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job should add it to the Valkey queue."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )
        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Check the queue has the job
        queue_length = mock_valkey_client.llen("index:queue")
        assert queue_length == 1

    def test_create_job_returns_existing_when_active_job_exists(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job for a repo with active job should return existing job_id."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        # Create first job
        job_id_1 = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Try to create another job for same repo
        job_id_2 = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Should return the existing job
        assert job_id_1 == job_id_2

    def test_create_job_with_force_cancels_existing(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job with force=True should cancel existing active job."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        # Create first job
        job_id_1 = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Force create another job
        job_id_2 = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            force=True
        )

        # Should be different jobs
        assert job_id_1 != job_id_2

        # First job should be marked for cancellation
        old_job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id_1
        ).first()
        assert old_job.cancel_requested is True

    def test_create_job_fails_when_queue_full(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Creating a job should fail when queue is at MAX_QUEUED_JOBS."""
        from app.services.job_queue_service import IndexingJobQueueService, QueueFullError

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client,
            max_queued_jobs=2
        )

        # Create jobs up to limit
        service.create_job(
            repo_id="repo-1",
            root_path="/path/1",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        service.create_job(
            repo_id="repo-2",
            root_path="/path/2",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Third job should fail
        with pytest.raises(QueueFullError):
            service.create_job(
                repo_id="repo-3",
                root_path="/path/3",
                index_name=TEST_INDEX_NAME,
                requested_by=TEST_USER_ID
            )


class TestIndexingJobQueueServiceClaim:
    """Test job claiming (worker picking up jobs)."""

    def test_claim_job_returns_job_id(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Claiming a job should return the job_id."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        # Create a job
        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        # Claim it
        claimed_id = service.claim_job()

        assert claimed_id == job_id

    def test_claim_job_moves_to_processing_queue(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Claimed job should be moved to processing queue."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        service.claim_job()

        # Queue should be empty, processing should have 1
        assert mock_valkey_client.llen("index:queue") == 0
        assert mock_valkey_client.llen("index:processing") == 1

    def test_claim_job_updates_status_to_running(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Claimed job should have status updated to running."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        service.claim_job()

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job.status == CodeIndexJobStatus.running
        assert job.started_at is not None

    def test_claim_job_returns_none_when_queue_empty(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Claiming from empty queue should return None."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        claimed_id = service.claim_job()

        assert claimed_id is None


class TestIndexingJobQueueServiceComplete:
    """Test job completion."""

    def test_complete_job_updates_status_to_succeeded(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Completing a job should update status to succeeded."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        service.claim_job()

        summary = {"files_indexed": 100, "duration_ms": 5000.0}
        service.complete_job(job_id, summary=summary)

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job.status == CodeIndexJobStatus.succeeded
        assert job.finished_at is not None
        assert job.summary == summary

    def test_complete_job_removes_from_processing_queue(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Completed job should be removed from processing queue."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        service.claim_job()

        assert mock_valkey_client.llen("index:processing") == 1

        service.complete_job(job_id, summary={})

        assert mock_valkey_client.llen("index:processing") == 0


class TestIndexingJobQueueServiceFail:
    """Test job failure handling."""

    def test_fail_job_updates_status_to_failed(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Failing a job should update status to failed."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        service.claim_job()

        service.fail_job(job_id, error="Neo4j connection failed")

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job.status == CodeIndexJobStatus.failed
        assert job.error == "Neo4j connection failed"
        assert job.finished_at is not None


class TestIndexingJobQueueServiceCancel:
    """Test job cancellation."""

    def test_cancel_job_sets_cancel_requested(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Cancelling a job should set cancel_requested=True."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        service.cancel_job(job_id)

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job.cancel_requested is True


class TestIndexingJobQueueServiceHeartbeat:
    """Test job heartbeat for orphan detection."""

    def test_heartbeat_updates_last_heartbeat(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Heartbeat should update last_heartbeat timestamp."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        service.claim_job()

        service.heartbeat(job_id)

        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job.last_heartbeat is not None


class TestIndexingJobQueueServiceGetJob:
    """Test job retrieval."""

    def test_get_job_returns_job_dict(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Getting a job should return its dict representation."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        job_dict = service.get_job(job_id)

        assert job_dict is not None
        assert job_dict["repo_id"] == TEST_REPO_ID
        assert job_dict["status"] == "queued"

    def test_get_job_returns_none_for_invalid_id(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client: MockValkeyClient
    ):
        """Getting a non-existent job should return None."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client
        )

        job_dict = service.get_job(uuid.uuid4())

        assert job_dict is None


class TestIndexingJobQueueServiceFallback:
    """Test fallback behavior when Valkey is unavailable."""

    def test_create_job_works_without_valkey(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client_failing: MockValkeyClient
    ):
        """Creating a job should work when Valkey is unavailable."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client_failing
        )

        # Should not raise, just log warning
        job_id = service.create_job(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )

        assert job_id is not None

        # Job should still be in DB
        job = sqlite_test_db_for_queue.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        assert job is not None

    def test_claim_job_uses_db_polling_without_valkey(
        self,
        sqlite_test_db_for_queue: Session,
        mock_valkey_client_failing: MockValkeyClient
    ):
        """Claiming a job should use DB polling when Valkey is unavailable."""
        from app.services.job_queue_service import IndexingJobQueueService

        service = IndexingJobQueueService(
            db=sqlite_test_db_for_queue,
            valkey_client=mock_valkey_client_failing
        )

        # Create job directly in DB (simulating restart scenario)
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.queued
        )
        sqlite_test_db_for_queue.add(job)
        sqlite_test_db_for_queue.commit()

        # Should find it via DB polling
        claimed_id = service.claim_job()

        assert claimed_id == job.id
