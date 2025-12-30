"""
IndexingJobQueueService - Manages code indexing jobs with Valkey queue.

Provides:
- Job creation with idempotency (one active job per repo)
- Reliable job claiming with BRPOPLPUSH pattern
- Heartbeat for orphan detection
- Fallback to DB polling when Valkey unavailable
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import CodeIndexJob, CodeIndexJobStatus

logger = logging.getLogger(__name__)


# Queue key names
QUEUE_KEY = "index:queue"
PROCESSING_KEY = "index:processing"


class QueueFullError(Exception):
    """Raised when the job queue is at capacity."""
    pass


class ValkeyClientProtocol(Protocol):
    """Protocol for Valkey/Redis client operations."""

    def ping(self) -> bool: ...
    def lpush(self, key: str, *values: str) -> int: ...
    def rpoplpush(self, src: str, dst: str) -> Optional[bytes]: ...
    def brpoplpush(self, src: str, dst: str, timeout: int) -> Optional[bytes]: ...
    def lrem(self, key: str, count: int, value: str) -> int: ...
    def llen(self, key: str) -> int: ...
    def lrange(self, key: str, start: int, stop: int) -> list[bytes]: ...


class IndexingJobQueueService:
    """
    Service for managing code indexing jobs with Valkey-backed queue.

    Supports:
    - Creating jobs with idempotency (one active job per repo)
    - Force creating jobs (cancels existing)
    - Claiming jobs for processing (BRPOPLPUSH pattern)
    - Completing/failing jobs
    - Heartbeat updates for orphan detection
    - Fallback to DB polling when Valkey unavailable
    """

    def __init__(
        self,
        db: Session,
        valkey_client: Optional[ValkeyClientProtocol] = None,
        max_queued_jobs: int = 100,
    ):
        """
        Initialize the job queue service.

        Args:
            db: SQLAlchemy database session
            valkey_client: Optional Valkey/Redis client (falls back to DB polling if None)
            max_queued_jobs: Maximum jobs allowed in queue before rejecting new ones
        """
        self.db = db
        self.valkey_client = valkey_client
        self.max_queued_jobs = max_queued_jobs
        self._valkey_available = self._check_valkey_available()

    def _check_valkey_available(self) -> bool:
        """Check if Valkey is available."""
        if self.valkey_client is None:
            return False
        try:
            self.valkey_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Valkey unavailable, falling back to DB polling: {e}")
            return False

    def _get_active_job_for_repo(self, repo_id: str) -> Optional[CodeIndexJob]:
        """Get any active (queued/running) job for a repo."""
        active_statuses = [CodeIndexJobStatus.queued, CodeIndexJobStatus.running]
        return self.db.query(CodeIndexJob).filter(
            CodeIndexJob.repo_id == repo_id,
            CodeIndexJob.status.in_(active_statuses),
            CodeIndexJob.cancel_requested.is_(False),
        ).first()

    def _count_queued_jobs(self) -> int:
        """Count jobs in queued status."""
        return self.db.query(CodeIndexJob).filter(
            CodeIndexJob.status == CodeIndexJobStatus.queued,
            CodeIndexJob.cancel_requested.is_(False),
        ).count()

    def create_job(
        self,
        repo_id: str,
        root_path: str,
        index_name: str,
        requested_by: str,
        request: Optional[dict[str, Any]] = None,
        meta: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> uuid.UUID:
        """
        Create a new indexing job.

        Args:
            repo_id: Repository identifier
            root_path: Path to repository root
            index_name: Search index name
            requested_by: User ID who requested the job
            request: Job request parameters (max_files, reset, etc.)
            meta: CodeResponseMeta snapshot
            force: If True, cancel any existing active job for the repo

        Returns:
            Job UUID

        Raises:
            QueueFullError: If queue is at MAX_QUEUED_JOBS capacity
        """
        # Check for existing active job
        existing_job = self._get_active_job_for_repo(repo_id)

        if existing_job:
            if force:
                # Cancel the existing job
                self.cancel_job(existing_job.id)
                logger.info(f"Cancelled existing job {existing_job.id} for repo {repo_id}")
            else:
                # Return existing job (idempotent)
                logger.info(f"Returning existing active job {existing_job.id} for repo {repo_id}")
                return existing_job.id

        # Check queue capacity
        queued_count = self._count_queued_jobs()
        if queued_count >= self.max_queued_jobs:
            raise QueueFullError(
                f"Queue full: {queued_count}/{self.max_queued_jobs} jobs queued"
            )

        # Create new job
        job = CodeIndexJob(
            repo_id=repo_id,
            root_path=root_path,
            index_name=index_name,
            requested_by=requested_by,
            request=request or {},
            meta=meta or {},
            status=CodeIndexJobStatus.queued,
        )
        try:
            self.db.add(job)
            self.db.commit()
        except IntegrityError:
            self.db.rollback()
            existing_job = self._get_active_job_for_repo(repo_id)
            if existing_job:
                logger.info(
                    "Concurrent create detected; returning existing job %s for repo %s",
                    existing_job.id,
                    repo_id,
                )
                return existing_job.id
            raise

        # Add to Valkey queue if available
        if self._valkey_available:
            try:
                self.valkey_client.lpush(QUEUE_KEY, str(job.id))
                logger.info(f"Added job {job.id} to Valkey queue")
            except Exception as e:
                logger.warning(f"Failed to add job to Valkey queue: {e}")
                # Job is still in DB, worker can find it via polling

        logger.info(f"Created indexing job {job.id} for repo {repo_id}")
        return job.id

    def claim_job(self, timeout: int = 1) -> Optional[uuid.UUID]:
        """
        Claim a job from the queue for processing.

        Uses BRPOPLPUSH pattern for reliability when Valkey is available.
        Falls back to DB polling when Valkey is unavailable.

        Args:
            timeout: Blocking timeout in seconds (for Valkey)

        Returns:
            Job UUID if a job was claimed, None otherwise
        """
        job_id: Optional[uuid.UUID] = None

        if self._valkey_available:
            try:
                result = self.valkey_client.brpoplpush(QUEUE_KEY, PROCESSING_KEY, timeout)
                if result:
                    job_id = uuid.UUID(result.decode())
            except Exception as e:
                logger.warning(f"Valkey claim failed, falling back to DB: {e}")
                self._valkey_available = False

        # Fallback to DB polling
        if job_id is None:
            job = self.db.query(CodeIndexJob).filter(
                CodeIndexJob.status == CodeIndexJobStatus.queued
            ).order_by(CodeIndexJob.created_at.asc()).first()

            if job:
                job_id = job.id

        if job_id is None:
            return None

        # Update job status to running (guard against races)
        now = datetime.now(timezone.utc)
        updated = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id,
            CodeIndexJob.status == CodeIndexJobStatus.queued,
        ).update(
            {
                CodeIndexJob.status: CodeIndexJobStatus.running,
                CodeIndexJob.started_at: now,
                CodeIndexJob.last_heartbeat: now,
                CodeIndexJob.attempts: CodeIndexJob.attempts + 1,
            },
            synchronize_session=False,
        )

        if updated == 0:
            # Job was already claimed or canceled
            if self._valkey_available:
                try:
                    self.valkey_client.lrem(PROCESSING_KEY, 1, str(job_id))
                except Exception as e:
                    logger.warning(f"Failed to remove stale job from processing queue: {e}")
            return None

        self.db.commit()
        logger.info(f"Claimed job {job_id}")
        return job_id

    def complete_job(
        self,
        job_id: uuid.UUID,
        summary: dict[str, Any],
    ) -> None:
        """
        Mark a job as successfully completed.

        Args:
            job_id: Job UUID
            summary: CodeIndexSummary as dict
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            job.status = CodeIndexJobStatus.succeeded
            job.finished_at = datetime.now(timezone.utc)
            job.summary = summary
            self.db.commit()

            # Remove from processing queue
            if self._valkey_available:
                try:
                    self.valkey_client.lrem(PROCESSING_KEY, 1, str(job_id))
                except Exception as e:
                    logger.warning(f"Failed to remove job from processing queue: {e}")

            logger.info(f"Completed job {job_id}")

    def fail_job(
        self,
        job_id: uuid.UUID,
        error: str,
    ) -> None:
        """
        Mark a job as failed.

        Args:
            job_id: Job UUID
            error: Error message
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            job.status = CodeIndexJobStatus.failed
            job.finished_at = datetime.now(timezone.utc)
            job.error = error
            self.db.commit()

            # Remove from processing queue
            if self._valkey_available:
                try:
                    self.valkey_client.lrem(PROCESSING_KEY, 1, str(job_id))
                except Exception as e:
                    logger.warning(f"Failed to remove job from processing queue: {e}")

            logger.info(f"Failed job {job_id}: {error}")

    def cancel_job(self, job_id: uuid.UUID) -> bool:
        """
        Request cancellation of a job.

        Args:
            job_id: Job UUID

        Returns:
            True if cancellation was requested, False if job not found
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            job.cancel_requested = True
            if job.status == CodeIndexJobStatus.queued:
                job.status = CodeIndexJobStatus.canceled
                job.finished_at = datetime.now(timezone.utc)
                self.db.commit()
                if self._valkey_available:
                    try:
                        self.valkey_client.lrem(QUEUE_KEY, 0, str(job_id))
                    except Exception as e:
                        logger.warning(f"Failed to remove job from queue: {e}")
            else:
                self.db.commit()
            logger.info(f"Cancellation requested for job {job_id}")
            return True

        return False

    def heartbeat(self, job_id: uuid.UUID) -> None:
        """
        Update job heartbeat timestamp.

        Args:
            job_id: Job UUID
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            job.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()

    def update_progress(
        self,
        job_id: uuid.UUID,
        progress: dict[str, Any],
    ) -> None:
        """
        Update job progress.

        Args:
            job_id: Job UUID
            progress: Progress data (files_scanned, files_indexed, etc.)
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            job.progress = progress
            job.last_heartbeat = datetime.now(timezone.utc)
            self.db.commit()

    def get_job(self, job_id: uuid.UUID) -> Optional[dict[str, Any]]:
        """
        Get job details.

        Args:
            job_id: Job UUID

        Returns:
            Job as dict, or None if not found
        """
        job = self.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()

        if job:
            return job.to_dict()

        return None

    def list_jobs(
        self,
        repo_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        status: Optional[CodeIndexJobStatus] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List jobs with optional filters.

        Args:
            repo_id: Filter by repository
            requested_by: Filter by user
            status: Filter by status
            limit: Maximum jobs to return

        Returns:
            List of jobs as dicts
        """
        query = self.db.query(CodeIndexJob)

        if repo_id:
            query = query.filter(CodeIndexJob.repo_id == repo_id)
        if requested_by:
            query = query.filter(CodeIndexJob.requested_by == requested_by)
        if status:
            query = query.filter(CodeIndexJob.status == status)

        jobs = query.order_by(CodeIndexJob.created_at.desc()).limit(limit).all()
        return [job.to_dict() for job in jobs]
