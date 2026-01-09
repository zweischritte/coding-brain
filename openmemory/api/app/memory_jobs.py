"""In-memory job tracking for async add_memories."""

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
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


_LOCK = threading.Lock()
_JOBS: dict[str, MemoryJob] = {}


def create_memory_job(
    requested_by: str,
    summary: Optional[dict[str, Any]] = None,
) -> str:
    job_id = str(uuid.uuid4())
    job = MemoryJob(
        job_id=job_id,
        requested_by=requested_by,
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
