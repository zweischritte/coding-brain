"""In-memory index job tracking for code indexing."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IndexJob:
    """Represents a background code indexing job."""

    job_id: str
    repo_id: str
    root_path: str
    index_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


_LOCK = threading.Lock()
_JOBS: dict[str, IndexJob] = {}


def create_index_job(
    repo_id: str,
    root_path: str,
    index_name: str,
    meta: Optional[dict[str, Any]] = None,
) -> str:
    job_id = str(uuid.uuid4())
    job = IndexJob(
        job_id=job_id,
        repo_id=repo_id,
        root_path=root_path,
        index_name=index_name,
        status="queued",
        created_at=_now(),
        meta=meta,
    )
    with _LOCK:
        _JOBS[job_id] = job
    return job_id


def update_index_job(job_id: str, **updates: Any) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def get_index_job(job_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return asdict(job)


def run_index_job(
    job_id: str,
    indexer: Any,
    *,
    max_files: Optional[int],
    reset: bool,
    missing_sources: list[str],
    meta: dict[str, Any],
) -> None:
    update_index_job(job_id, status="running", started_at=_now())
    try:
        summary = indexer.index_repository(
            max_files=max_files,
            reset=reset,
        )
        update_index_job(
            job_id,
            status="succeeded",
            finished_at=_now(),
            summary={
                "repo_id": summary.repo_id,
                "files_indexed": summary.files_indexed,
                "files_failed": summary.files_failed,
                "symbols_indexed": summary.symbols_indexed,
                "documents_indexed": summary.documents_indexed,
                "call_edges_indexed": summary.call_edges_indexed,
                "duration_ms": summary.duration_ms,
                "meta": meta,
            },
        )
    except Exception as exc:
        update_index_job(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc),
            meta={
                **meta,
                "degraded_mode": True,
                "missing_sources": missing_sources,
                "error": str(exc),
            },
        )
