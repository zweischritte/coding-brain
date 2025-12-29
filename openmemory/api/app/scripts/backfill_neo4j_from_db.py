#!/usr/bin/env python3
"""
Backfill Neo4j OM_* metadata graph from the OpenMemory SQL database.

This projects the *current* OpenMemory memory records (by default: ACTIVE only)
into Neo4j using the deterministic metadata projector (OM_* namespace).

Why SQL (SQLite/Postgres) and not the vector DB?
- SQL is OpenMemory's source of truth for memory lifecycle (state) + metadata edits.
  In this repo, `update_memory` updates SQL metadata but does not update vector-store
  metadata, so Qdrant payload metadata can be stale.
- SQL also defines what OpenMemory considers "current" (e.g., excludes deleted).

Usage (inside `openmemory/api`):
    python -m app.scripts.backfill_neo4j_from_db

Options:
    --user-id USER_ID         Only backfill for a specific string user_id (e.g. "grischadallmer")
    --include-non-active      Include paused/archived/deleted memories (default: active only)
    --limit N                 Process at most N memories (global)
    --dry-run                 Validate + log without writing to Neo4j
    --log-file PATH           Write logs to PATH (default: logs/neo4j_backfill_from_db_<ts>.log)
    --verbose                 Enable debug logging

Environment variables:
    - DATABASE_URL (optional): defaults to sqlite:///./openmemory.db
    - NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger(__name__)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _configure_logging(*, log_file: str, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


@dataclass
class BackfillStats:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"neo4j_backfill_from_db_{ts}.log")


def _iter_users(db, user_id: Optional[str]):
    from app.models import User

    q = db.query(User)
    if user_id:
        q = q.filter(User.user_id == user_id)
    return q.order_by(User.user_id.asc()).all()


def _iter_memories(db, *, user_pk, include_non_active: bool) -> Iterable:
    from app.models import Memory, MemoryState

    q = db.query(Memory).filter(Memory.user_id == user_pk)
    if not include_non_active:
        q = q.filter(Memory.state == MemoryState.active)
    return q.order_by(Memory.created_at.asc())


def backfill_from_db(
    *,
    user_id: Optional[str] = None,
    include_non_active: bool = False,
    limit: Optional[int] = None,
    dry_run: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> tuple[BackfillStats, str]:
    from app.database import SessionLocal
    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session
    from app.graph.metadata_projector import MetadataProjector, MemoryMetadata

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting Neo4j backfill from SQL DB")
    logger.info("DATABASE_URL=%s", os.environ.get("DATABASE_URL", "sqlite:///./openmemory.db"))
    logger.info("dry_run=%s include_non_active=%s user_id=%s limit=%s", dry_run, include_non_active, user_id, limit)
    logger.info("log_file=%s", chosen_log_file)

    if not is_neo4j_configured():
        logger.error(
            "Neo4j is not configured. Set NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD (and optionally NEO4J_DATABASE)."
        )
        raise SystemExit(1)

    projector = MetadataProjector(get_neo4j_session)
    if not dry_run:
        logger.info("Ensuring Neo4j constraints...")
        if not projector.ensure_constraints():
            logger.error("Failed to ensure Neo4j constraints")
            raise SystemExit(1)

    stats = BackfillStats()

    db = SessionLocal()
    try:
        users = _iter_users(db, user_id)
        if not users:
            logger.error("No users found for user_id=%s", user_id)
            raise SystemExit(1)

        for u in users:
            logger.info("User: %s (%s)", u.user_id, u.id)

            mem_iter = list(_iter_memories(db, user_pk=u.id, include_non_active=include_non_active))
            if limit is not None:
                mem_iter = mem_iter[: max(0, int(limit))]

            logger.info("Memories to process for user '%s': %d", u.user_id, len(mem_iter))

            for idx, m in enumerate(mem_iter, 1):
                stats.total += 1
                memory_id = str(m.id)

                try:
                    metadata = dict(m.metadata_ or {})

                    data = {
                        "content": m.content,
                        "metadata": metadata,
                        "created_at": _iso(m.created_at),
                        "updated_at": _iso(m.updated_at),
                        "state": m.state.value if m.state else "active",
                    }

                    mm = MemoryMetadata.from_dict(data=data, memory_id=memory_id, user_id=u.user_id)

                    if dry_run:
                        stats.skipped += 1
                        logger.info(
                            "[DRY RUN] %s user=%s state=%s category=%s scope=%s tags=%d",
                            memory_id,
                            u.user_id,
                            data["state"],
                            mm.category,
                            mm.scope,
                            len(mm.tags or {}),
                        )
                        continue

                    ok = projector.upsert_memory(mm)
                    if ok:
                        stats.processed += 1
                        logger.info(
                            "OK %s user=%s state=%s category=%s scope=%s tags=%d",
                            memory_id,
                            u.user_id,
                            data["state"],
                            mm.category,
                            mm.scope,
                            len(mm.tags or {}),
                        )
                    else:
                        stats.failed += 1
                        logger.warning("FAIL %s user=%s (projector returned False)", memory_id, u.user_id)

                    if stats.total % 50 == 0:
                        logger.info(
                            "Progress: total=%d processed=%d skipped=%d failed=%d",
                            stats.total,
                            stats.processed,
                            stats.skipped,
                            stats.failed,
                        )

                except Exception as e:  # noqa: BLE001
                    stats.failed += 1
                    logger.exception("Error processing memory %s user=%s: %s", memory_id, u.user_id, e)

        logger.info("Backfill complete: total=%d processed=%d skipped=%d failed=%d", stats.total, stats.processed, stats.skipped, stats.failed)
        return stats, chosen_log_file
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Neo4j OM_* graph from OpenMemory SQL DB")
    parser.add_argument("--user-id", default=None, help="Backfill only this user_id (string)")
    parser.add_argument(
        "--include-non-active",
        action="store_true",
        help="Include paused/archived/deleted memories (default: active only)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N memories (global)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Neo4j")
    parser.add_argument("--log-file", default=None, help="Log file path (default: logs/neo4j_backfill_from_db_<ts>.log)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    stats, log_file = backfill_from_db(
        user_id=args.user_id,
        include_non_active=args.include_non_active,
        limit=args.limit,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
    )

    print(
        f"Backfill complete. total={stats.total} processed={stats.processed} skipped={stats.skipped} failed={stats.failed} log_file={log_file}"
    )

    if stats.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
