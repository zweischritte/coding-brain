"""In-memory job tracking for async memory and graph projection jobs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryJob:
    """Represents a background memory add job."""

    job_id: str
    requested_by: str
    job_type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    debug_timing: Optional[dict[str, Any]] = None


_LOCK = threading.Lock()
_JOBS: dict[str, MemoryJob] = {}


def create_memory_job(
    requested_by: str,
    summary: Optional[dict[str, Any]] = None,
    job_type: str = "memory_add",
) -> str:
    job_id = str(uuid.uuid4())
    job = MemoryJob(
        job_id=job_id,
        requested_by=requested_by,
        job_type=job_type,
        status="queued",
        created_at=_now(),
        summary=summary,
    )
    with _LOCK:
        _JOBS[job_id] = job
    return job_id


def update_memory_job(job_id: str, **updates: Any) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def get_memory_job(job_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return asdict(job)


def list_memory_jobs(
    requested_by: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List memory jobs with optional filtering.

    Args:
        requested_by: Filter by user who requested the job
        status_filter: Filter by status. Options:
            - "running": queued or running jobs
            - "completed": succeeded jobs
            - "failed": failed jobs
            - "all": all jobs (default if None)
            - specific status: "queued", "running", "succeeded", "failed"
        limit: Maximum number of jobs to return (default 50)

    Returns:
        List of job dicts, sorted by created_at descending (newest first)
    """
    with _LOCK:
        jobs = list(_JOBS.values())

    # Filter by requested_by
    if requested_by:
        jobs = [j for j in jobs if j.requested_by == requested_by]

    # Filter by status
    if status_filter:
        if status_filter == "running":
            jobs = [j for j in jobs if j.status in ("queued", "running")]
        elif status_filter == "completed":
            jobs = [j for j in jobs if j.status == "succeeded"]
        elif status_filter == "failed":
            jobs = [j for j in jobs if j.status == "failed"]
        elif status_filter != "all":
            # Specific status filter
            jobs = [j for j in jobs if j.status == status_filter]

    # Sort by created_at descending (newest first)
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Apply limit
    jobs = jobs[:limit]

    return [asdict(j) for j in jobs]
