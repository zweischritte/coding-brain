"""
Tests for CodeIndexJob model and store operations.

TDD: These tests are written FIRST before the model implementation.
They define the expected behavior for persistent indexing job storage.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base


# Test constants
TEST_JOB_ID = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
TEST_USER_ID = "test-user-123"
TEST_REPO_ID = "coding-brain"
TEST_ROOT_PATH = "/path/to/repo"
TEST_INDEX_NAME = "code"


@pytest.fixture
def sqlite_test_db_with_jobs() -> Generator[Session, None, None]:
    """
    Create an in-memory SQLite database with CodeIndexJob model.

    Note: Import is done inside to allow test to fail gracefully
    if model doesn't exist yet (TDD red phase).
    """
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


class TestCodeIndexJobStatusEnum:
    """Test the CodeIndexJobStatus enum definition."""

    def test_status_enum_has_queued(self):
        """Status enum must have 'queued' value."""
        from app.models import CodeIndexJobStatus
        assert CodeIndexJobStatus.queued.value == "queued"

    def test_status_enum_has_running(self):
        """Status enum must have 'running' value."""
        from app.models import CodeIndexJobStatus
        assert CodeIndexJobStatus.running.value == "running"

    def test_status_enum_has_succeeded(self):
        """Status enum must have 'succeeded' value."""
        from app.models import CodeIndexJobStatus
        assert CodeIndexJobStatus.succeeded.value == "succeeded"

    def test_status_enum_has_failed(self):
        """Status enum must have 'failed' value."""
        from app.models import CodeIndexJobStatus
        assert CodeIndexJobStatus.failed.value == "failed"

    def test_status_enum_has_canceled(self):
        """Status enum must have 'canceled' value for cancellation support."""
        from app.models import CodeIndexJobStatus
        assert CodeIndexJobStatus.canceled.value == "canceled"


class TestCodeIndexJobModel:
    """Test CodeIndexJob SQLAlchemy model structure."""

    def test_model_has_id_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have UUID primary key 'id'."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.id is not None
        assert isinstance(job.id, uuid.UUID)

    def test_model_has_repo_id_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have repo_id string field."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.repo_id == TEST_REPO_ID

    def test_model_has_root_path_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have root_path string field."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.root_path == TEST_ROOT_PATH

    def test_model_has_index_name_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have index_name string field."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.index_name == TEST_INDEX_NAME

    def test_model_has_status_field_with_default(self, sqlite_test_db_with_jobs: Session):
        """Model must have status field defaulting to 'queued'."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.queued

    def test_model_has_requested_by_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have requested_by field for tracking who initiated the job."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.requested_by == TEST_USER_ID

    def test_model_has_created_at_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have created_at timestamp with auto-default."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.created_at is not None
        assert isinstance(job.created_at, datetime)

    def test_model_has_started_at_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have started_at nullable timestamp."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Initially null
        assert job.started_at is None

        # Can be set
        job.started_at = datetime.now(timezone.utc)
        sqlite_test_db_with_jobs.commit()
        assert job.started_at is not None

    def test_model_has_finished_at_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have finished_at nullable timestamp."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Initially null
        assert job.finished_at is None

        # Can be set
        job.finished_at = datetime.now(timezone.utc)
        sqlite_test_db_with_jobs.commit()
        assert job.finished_at is not None

    def test_model_has_last_heartbeat_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have last_heartbeat for orphan detection."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Initially null
        assert job.last_heartbeat is None

        # Can be set
        job.last_heartbeat = datetime.now(timezone.utc)
        sqlite_test_db_with_jobs.commit()
        assert job.last_heartbeat is not None

    def test_model_has_attempts_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have attempts counter for retry tracking."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Default is 0
        assert job.attempts == 0

        # Can be incremented
        job.attempts = 1
        sqlite_test_db_with_jobs.commit()
        assert job.attempts == 1

    def test_model_has_cancel_requested_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have cancel_requested boolean for graceful cancellation."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Default is False
        assert job.cancel_requested is False

        # Can be set to True
        job.cancel_requested = True
        sqlite_test_db_with_jobs.commit()
        assert job.cancel_requested is True

    def test_model_has_request_json_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have request JSON field for storing job parameters."""
        from app.models import CodeIndexJob
        request_data = {
            "max_files": 100,
            "reset": True,
            "include_api_boundaries": True,
            "async_mode": True
        }
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            request=request_data
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.request == request_data
        assert job.request["max_files"] == 100

    def test_model_has_progress_json_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have progress JSON field for tracking indexing progress."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Can store progress
        progress_data = {
            "files_scanned": 50,
            "files_indexed": 45,
            "files_failed": 5,
            "current_file": "/path/to/file.py",
            "current_phase": "parse"
        }
        job.progress = progress_data
        sqlite_test_db_with_jobs.commit()

        assert job.progress == progress_data

    def test_model_has_summary_json_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have summary JSON field for final results."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Can store summary
        summary_data = {
            "repo_id": TEST_REPO_ID,
            "files_indexed": 100,
            "files_failed": 5,
            "symbols_indexed": 500,
            "documents_indexed": 600,
            "call_edges_indexed": 200,
            "duration_ms": 5000.0
        }
        job.summary = summary_data
        sqlite_test_db_with_jobs.commit()

        assert job.summary == summary_data

    def test_model_has_meta_json_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have meta JSON field for CodeResponseMeta."""
        from app.models import CodeIndexJob
        meta_data = {"degraded_mode": False, "missing_sources": []}
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            meta=meta_data
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.meta == meta_data

    def test_model_has_error_field(self, sqlite_test_db_with_jobs: Session):
        """Model must have error text field for failure messages."""
        from app.models import CodeIndexJob
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        # Initially null
        assert job.error is None

        # Can store error message
        job.error = "Connection timeout to Neo4j"
        sqlite_test_db_with_jobs.commit()
        assert job.error == "Connection timeout to Neo4j"


class TestCodeIndexJobStateTransitions:
    """Test job state transition logic."""

    def test_job_starts_as_queued(self, sqlite_test_db_with_jobs: Session):
        """New jobs must start in queued status."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.queued

    def test_job_transitions_to_running(self, sqlite_test_db_with_jobs: Session):
        """Job can transition from queued to running."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        job.status = CodeIndexJobStatus.running
        job.started_at = datetime.now(timezone.utc)
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.running
        assert job.started_at is not None

    def test_job_transitions_to_succeeded(self, sqlite_test_db_with_jobs: Session):
        """Job can transition from running to succeeded."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=datetime.now(timezone.utc)
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        job.status = CodeIndexJobStatus.succeeded
        job.finished_at = datetime.now(timezone.utc)
        job.summary = {"files_indexed": 100}
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.succeeded
        assert job.finished_at is not None

    def test_job_transitions_to_failed(self, sqlite_test_db_with_jobs: Session):
        """Job can transition from running to failed."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=datetime.now(timezone.utc)
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        job.status = CodeIndexJobStatus.failed
        job.finished_at = datetime.now(timezone.utc)
        job.error = "Neo4j connection failed"
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.failed
        assert job.error is not None

    def test_job_transitions_to_canceled(self, sqlite_test_db_with_jobs: Session):
        """Job can transition to canceled when cancel_requested is True."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=datetime.now(timezone.utc),
            cancel_requested=True
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        job.status = CodeIndexJobStatus.canceled
        job.finished_at = datetime.now(timezone.utc)
        sqlite_test_db_with_jobs.commit()

        assert job.status == CodeIndexJobStatus.canceled


class TestCodeIndexJobQueries:
    """Test query patterns for the CodeIndexJob model."""

    def test_query_jobs_by_repo_id(self, sqlite_test_db_with_jobs: Session):
        """Can query jobs by repo_id."""
        from app.models import CodeIndexJob
        # Create multiple jobs for different repos
        job1 = CodeIndexJob(
            repo_id="repo-a",
            root_path="/path/a",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        job2 = CodeIndexJob(
            repo_id="repo-b",
            root_path="/path/b",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID
        )
        sqlite_test_db_with_jobs.add_all([job1, job2])
        sqlite_test_db_with_jobs.commit()

        results = sqlite_test_db_with_jobs.query(CodeIndexJob).filter(
            CodeIndexJob.repo_id == "repo-a"
        ).all()

        assert len(results) == 1
        assert results[0].repo_id == "repo-a"

    def test_query_active_jobs_by_repo(self, sqlite_test_db_with_jobs: Session):
        """Can query active (queued/running) jobs by repo_id."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        # Create jobs with different statuses
        job1 = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.queued
        )
        job2 = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.succeeded
        )
        sqlite_test_db_with_jobs.add_all([job1, job2])
        sqlite_test_db_with_jobs.commit()

        active_statuses = [CodeIndexJobStatus.queued, CodeIndexJobStatus.running]
        results = sqlite_test_db_with_jobs.query(CodeIndexJob).filter(
            CodeIndexJob.repo_id == TEST_REPO_ID,
            CodeIndexJob.status.in_(active_statuses)
        ).all()

        assert len(results) == 1
        assert results[0].status == CodeIndexJobStatus.queued

    def test_query_jobs_by_requested_by(self, sqlite_test_db_with_jobs: Session):
        """Can query jobs by user who requested them."""
        from app.models import CodeIndexJob
        job1 = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by="user-a"
        )
        job2 = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by="user-b"
        )
        sqlite_test_db_with_jobs.add_all([job1, job2])
        sqlite_test_db_with_jobs.commit()

        results = sqlite_test_db_with_jobs.query(CodeIndexJob).filter(
            CodeIndexJob.requested_by == "user-a"
        ).all()

        assert len(results) == 1
        assert results[0].requested_by == "user-a"

    def test_query_orphaned_jobs(self, sqlite_test_db_with_jobs: Session):
        """Can query orphaned jobs (running with stale heartbeat)."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(minutes=10)  # 10 minutes ago

        # Job with recent heartbeat (not orphaned)
        job1 = CodeIndexJob(
            repo_id="repo-a",
            root_path="/path/a",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=now - timedelta(minutes=5),
            last_heartbeat=now - timedelta(seconds=30)  # 30 seconds ago
        )
        # Job with stale heartbeat (orphaned)
        job2 = CodeIndexJob(
            repo_id="repo-b",
            root_path="/path/b",
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            status=CodeIndexJobStatus.running,
            started_at=now - timedelta(minutes=15),
            last_heartbeat=stale_time  # 10 minutes ago
        )
        sqlite_test_db_with_jobs.add_all([job1, job2])
        sqlite_test_db_with_jobs.commit()

        orphan_threshold = now - timedelta(minutes=5)
        orphaned = sqlite_test_db_with_jobs.query(CodeIndexJob).filter(
            CodeIndexJob.status == CodeIndexJobStatus.running,
            CodeIndexJob.last_heartbeat < orphan_threshold
        ).all()

        assert len(orphaned) == 1
        assert orphaned[0].repo_id == "repo-b"


class TestCodeIndexJobToDict:
    """Test serialization of CodeIndexJob to dictionary."""

    def test_to_dict_includes_all_fields(self, sqlite_test_db_with_jobs: Session):
        """Job should be serializable to dict with all fields."""
        from app.models import CodeIndexJob, CodeIndexJobStatus
        job = CodeIndexJob(
            repo_id=TEST_REPO_ID,
            root_path=TEST_ROOT_PATH,
            index_name=TEST_INDEX_NAME,
            requested_by=TEST_USER_ID,
            request={"max_files": 100},
            meta={"degraded_mode": False}
        )
        sqlite_test_db_with_jobs.add(job)
        sqlite_test_db_with_jobs.commit()

        job_dict = job.to_dict()

        assert "id" in job_dict
        assert job_dict["repo_id"] == TEST_REPO_ID
        assert job_dict["root_path"] == TEST_ROOT_PATH
        assert job_dict["index_name"] == TEST_INDEX_NAME
        assert job_dict["status"] == "queued"
        assert job_dict["requested_by"] == TEST_USER_ID
        assert "created_at" in job_dict
        assert job_dict["request"] == {"max_files": 100}
        assert job_dict["meta"] == {"degraded_mode": False}
