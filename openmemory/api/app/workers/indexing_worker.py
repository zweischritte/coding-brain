"""
IndexingWorker - Background worker for processing code indexing jobs.

Provides:
- Job processing with progress callbacks
- Cancellation handling
- Heartbeat updates for orphan detection
- Orphan job recovery
- Configurable retry policy
"""
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional, Protocol

from app.models import CodeIndexJob, CodeIndexJobStatus
from app.services.job_queue_service import IndexingJobQueueService
from indexing.code_indexer import IndexingCancelled

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_ORPHAN_THRESHOLD_MINUTES = 5
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30


class CodeIndexingServiceProtocol(Protocol):
    """Protocol for the CodeIndexingService."""

    def index_repository(
        self,
        max_files: Optional[int] = None,
        reset: bool = False,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Any: ...


IndexerFactory = Callable[..., CodeIndexingServiceProtocol]


class IndexingWorker:
    """
    Background worker for processing code indexing jobs.

    Features:
    - Processes jobs from the queue
    - Sends heartbeat updates during processing
    - Handles cancellation requests
    - Recovers orphaned jobs (crashed workers)
    """

    def __init__(
        self,
        queue_service: IndexingJobQueueService,
        indexer_factory: IndexerFactory,
        orphan_threshold_minutes: int = DEFAULT_ORPHAN_THRESHOLD_MINUTES,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        orphan_recovery_interval_seconds: int = 60,
    ):
        """
        Initialize the indexing worker.

        Args:
            queue_service: Service for queue operations
            indexer_factory: Factory function to create CodeIndexingService instances
            orphan_threshold_minutes: Minutes after which a running job is considered orphaned
            max_attempts: Maximum retry attempts before failing a job
            heartbeat_interval_seconds: Seconds between heartbeat updates
            orphan_recovery_interval_seconds: Seconds between orphan recovery checks
        """
        self.queue_service = queue_service
        self.indexer_factory = indexer_factory
        self.orphan_threshold_minutes = orphan_threshold_minutes
        self.max_attempts = max_attempts
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.orphan_recovery_interval_seconds = orphan_recovery_interval_seconds
        self._shutdown_requested = False

    def run_once(self) -> bool:
        """
        Process one job if available.

        Returns:
            True if a job was processed, False otherwise
        """
        job_id = self.queue_service.claim_job()
        if job_id is None:
            return False

        self._process_job(job_id)
        return True

    def process_one_job(self) -> None:
        """
        Claim and process one job from the queue.

        This method:
        1. Claims a job from the queue
        2. Checks for cancellation request
        3. Runs the indexer with progress callbacks
        4. Updates job status on completion/failure
        """
        job_id = self.queue_service.claim_job()
        if job_id is None:
            return
        self._process_job(job_id)

    def _process_job(self, job_id: uuid.UUID) -> None:
        """
        Process a specific claimed job.

        Args:
            job_id: UUID of the job to process
        """
        try:

            # Get job details
            job_dict = self.queue_service.get_job(job_id)
            if job_dict is None:
                logger.warning(f"Job {job_id} not found after claiming")
                return

            # Check for cancellation before starting
            if job_dict.get("cancel_requested"):
                self._cancel_job(job_id)
                return

            # Create indexer
            root_path = job_dict["root_path"]
            repo_id = job_dict["repo_id"]
            request = job_dict.get("request", {})

            indexer = self.indexer_factory(
                root_path,
                repo_id,
                index_name=job_dict.get("index_name") or "code",
                include_api_boundaries=request.get("include_api_boundaries", True),
            )

            # Create progress callback that sends heartbeats
            last_heartbeat = [datetime.now(timezone.utc)]

            def progress_callback(progress: dict) -> None:
                # Update progress
                self.queue_service.update_progress(job_id, progress)

                # Send heartbeat if interval elapsed
                now = datetime.now(timezone.utc)
                if (now - last_heartbeat[0]).total_seconds() >= self.heartbeat_interval_seconds:
                    self.queue_service.heartbeat(job_id)
                    last_heartbeat[0] = now

                # Check for cancellation during processing
                current_job = self.queue_service.get_job(job_id)
                if current_job and current_job.get("cancel_requested"):
                    raise IndexingCancelled()

            # Run indexer
            try:
                summary = indexer.index_repository(
                    max_files=request.get("max_files"),
                    reset=request.get("reset", False),
                    progress_callback=progress_callback,
                )

                # Success - update job
                summary_dict = {
                    "repo_id": summary.repo_id,
                    "files_indexed": summary.files_indexed,
                    "files_failed": summary.files_failed,
                    "symbols_indexed": summary.symbols_indexed,
                    "documents_indexed": summary.documents_indexed,
                    "call_edges_indexed": summary.call_edges_indexed,
                    "duration_ms": summary.duration_ms,
                }
                self.queue_service.complete_job(job_id, summary=summary_dict)
                logger.info(f"Job {job_id} completed successfully")

            except IndexingCancelled:
                self._cancel_job(job_id)

        except Exception as e:
            if job_id:
                self.queue_service.fail_job(job_id, error=str(e))
                logger.error(f"Job {job_id} failed: {e}")
            else:
                logger.error(f"Worker error: {e}")

    def _cancel_job(self, job_id: uuid.UUID) -> None:
        """Mark a job as canceled."""
        job = self.queue_service.db.query(CodeIndexJob).filter(
            CodeIndexJob.id == job_id
        ).first()
        if job:
            job.status = CodeIndexJobStatus.canceled
            job.finished_at = datetime.now(timezone.utc)
            self.queue_service.db.commit()
            logger.info(f"Job {job_id} was canceled")

    def recover_orphan_jobs(self) -> int:
        """
        Find and recover orphaned jobs (running with stale heartbeat).

        Jobs are considered orphaned if:
        - Status is 'running'
        - last_heartbeat is older than orphan_threshold_minutes

        Orphaned jobs are either:
        - Requeued if attempts < max_attempts
        - Failed if attempts >= max_attempts

        Returns:
            Number of jobs recovered
        """
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(minutes=self.orphan_threshold_minutes)

        orphaned_jobs = self.queue_service.db.query(CodeIndexJob).filter(
            CodeIndexJob.status == CodeIndexJobStatus.running,
            CodeIndexJob.last_heartbeat < threshold
        ).all()

        recovered = 0
        for job in orphaned_jobs:
            if job.attempts >= self.max_attempts:
                # Fail the job
                job.status = CodeIndexJobStatus.failed
                job.finished_at = now
                job.error = f"Job failed after max attempts ({self.max_attempts})"
                logger.warning(f"Job {job.id} failed after {job.attempts} attempts")
            else:
                # Requeue the job
                job.status = CodeIndexJobStatus.queued
                job.started_at = None
                job.last_heartbeat = None
                logger.info(f"Requeued orphaned job {job.id} (attempt {job.attempts})")

            recovered += 1

        if orphaned_jobs:
            self.queue_service.db.commit()

        return recovered

    def run(self, poll_interval: float = 1.0) -> None:
        """
        Run the worker loop continuously.

        Args:
            poll_interval: Seconds to wait between polls when no jobs available
        """
        logger.info("Starting indexing worker...")
        self._shutdown_requested = False

        last_orphan_check = time.monotonic()
        while not self._shutdown_requested:
            try:
                # Recover any orphaned jobs periodically
                now = time.monotonic()
                if now - last_orphan_check >= self.orphan_recovery_interval_seconds:
                    self.recover_orphan_jobs()
                    last_orphan_check = now

                # Try to process a job
                if not self.run_once():
                    # No job available, wait before polling again
                    time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(poll_interval)

    def shutdown(self) -> None:
        """Request graceful shutdown of the worker."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True


def create_indexer_factory():
    """
    Create a factory function for CodeIndexingService instances.

    Uses the code toolkit to get shared dependencies (Neo4j, OpenSearch, etc).
    """
    from app.code_toolkit import get_code_toolkit
    from indexing.code_indexer import CodeIndexingService

    toolkit = get_code_toolkit()

    def factory(
        root_path: str,
        repo_id: str,
        index_name: str = "code",
        include_api_boundaries: bool = True,
    ) -> CodeIndexingService:
        return CodeIndexingService(
            root_path=root_path,
            repo_id=repo_id,
            graph_driver=toolkit.neo4j_driver,
            opensearch_client=toolkit.opensearch_client if toolkit.is_available("opensearch") else None,
            embedding_service=toolkit.embedding_service if toolkit.is_available("embedding") else None,
            index_name=index_name,
            include_api_boundaries=include_api_boundaries,
        )

    return factory


def get_valkey_client():
    """Get Valkey client if configured."""
    import os

    valkey_host = os.environ.get("VALKEY_HOST")
    valkey_port = os.environ.get("VALKEY_PORT", "6379")

    if not valkey_host:
        logger.info("VALKEY_HOST not set, using DB polling only")
        return None

    try:
        import valkey

        client = valkey.Valkey(host=valkey_host, port=int(valkey_port))
        client.ping()
        logger.info(f"Connected to Valkey at {valkey_host}:{valkey_port}")
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Valkey: {e}, using DB polling only")
        return None


def main():
    """
    Main entry point for the indexing worker.

    Reads configuration from environment variables:
    - WORKER_POLL_INTERVAL: Seconds between polls (default: 1.0)
    - WORKER_ORPHAN_RECOVERY_INTERVAL: Seconds between orphan recovery (default: 60.0)
    - WORKER_ORPHAN_TIMEOUT_MINUTES: Minutes before job is orphaned (default: 5)
    - WORKER_MAX_ATTEMPTS: Max retry attempts (default: 3)
    - WORKER_HEARTBEAT_INTERVAL_SECONDS: Seconds between heartbeat updates (default: 30.0)
    """
    import os
    import signal

    from app.database import SessionLocal

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Read configuration from environment
    poll_interval = float(os.environ.get("WORKER_POLL_INTERVAL", "1.0"))
    orphan_recovery_interval = float(os.environ.get("WORKER_ORPHAN_RECOVERY_INTERVAL", "60.0"))
    orphan_threshold = int(os.environ.get("WORKER_ORPHAN_TIMEOUT_MINUTES", "5"))
    max_attempts = int(os.environ.get("WORKER_MAX_ATTEMPTS", "3"))
    heartbeat_interval = float(os.environ.get("WORKER_HEARTBEAT_INTERVAL_SECONDS", "30.0"))

    logger.info(
        "Starting indexing worker with config: poll_interval=%ss, "
        "orphan_recovery_interval=%ss, orphan_threshold=%smin, "
        "max_attempts=%s, heartbeat_interval=%ss",
        poll_interval,
        orphan_recovery_interval,
        orphan_threshold,
        max_attempts,
        heartbeat_interval,
    )

    # Create database session
    db = SessionLocal()

    # Create Valkey client (optional)
    valkey_client = get_valkey_client()

    # Create queue service
    queue_service = IndexingJobQueueService(
        db=db,
        valkey_client=valkey_client
    )

    # Create indexer factory
    indexer_factory = create_indexer_factory()

    # Create worker
    worker = IndexingWorker(
        queue_service=queue_service,
        indexer_factory=indexer_factory,
        orphan_threshold_minutes=orphan_threshold,
        max_attempts=max_attempts,
        heartbeat_interval_seconds=heartbeat_interval,
        orphan_recovery_interval_seconds=orphan_recovery_interval,
    )

    # Handle graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        worker.shutdown()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        worker.run(poll_interval=poll_interval)
    finally:
        db.close()
        if valkey_client:
            valkey_client.close()
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()
