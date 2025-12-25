#!/usr/bin/env python3
"""
Backfill Mem0 Graph Memory (__Entity__ graph) from the OpenMemory SQL database.

This uses Mem0's LLM-based graph extraction to create entity-to-entity relationships
in Neo4j (separate from the deterministic OM_* metadata projection).

WARNING:
- This will make LLM + embedding calls and can be slow/costly depending on your model.
- Mem0 graph ingestion can also delete/merge existing relationships as it goes.

Usage (inside `openmemory/api` container):
    python -m app.scripts.backfill_mem0_graph_from_db --user-id grischadallmer --log-file logs/mem0_graph_backfill.log

Options:
    --user-id USER_ID         Only backfill for a specific string user_id
    --include-non-active      Include paused/archived/deleted memories (default: active only)
    --limit N                 Process at most N memories (per user)
    --memory-ids CSV          Process only these memory UUIDs (comma-separated)
    --retry-failed-from-log   Retry only FAIL memory_ids from a previous log file
    --dry-run                 Do not call LLM/Neo4j, only log what would be processed
    --log-file PATH           Log file path (default: logs/mem0_graph_backfill_<ts>.log)
    --verbose                 Enable debug logging
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"mem0_graph_backfill_{ts}.log")


def _configure_logging(*, log_file: str, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)


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


def _parse_memory_ids(value: Optional[str]) -> Optional[set[uuid.UUID]]:
    if not value:
        return None
    ids: set[uuid.UUID] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        ids.add(uuid.UUID(token))
    return ids


def _memory_ids_from_log(log_file: str) -> set[uuid.UUID]:
    ids: set[uuid.UUID] = set()
    pattern = re.compile(r"FAIL memory_id=([0-9a-fA-F-]{36})\b")
    for line in Path(log_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        try:
            ids.add(uuid.UUID(m.group(1)))
        except Exception:
            continue
    return ids


def backfill_mem0_graph_from_db(
    *,
    user_id: Optional[str] = None,
    include_non_active: bool = False,
    limit: Optional[int] = None,
    memory_ids: Optional[set[uuid.UUID]] = None,
    dry_run: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> tuple[BackfillStats, str]:
    from app.database import SessionLocal
    from app.utils.memory import reset_memory_client, get_memory_client

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting Mem0 Graph Memory backfill from SQL DB")
    logger.info("dry_run=%s include_non_active=%s user_id=%s limit=%s", dry_run, include_non_active, user_id, limit)
    if memory_ids:
        logger.info("memory_ids=%d", len(memory_ids))
    logger.info("log_file=%s", chosen_log_file)

    reset_memory_client()
    memory_client = get_memory_client()
    if memory_client is None:
        logger.error("Failed to initialize Mem0 memory client")
        raise SystemExit(1)
    if not hasattr(memory_client, "graph") or memory_client.graph is None:
        logger.error("Mem0 Graph Memory is not enabled. Set mem0.graph_store in config and restart/reload.")
        raise SystemExit(1)

    stats = BackfillStats()

    db = SessionLocal()
    try:
        users = _iter_users(db, user_id)
        if not users:
            logger.error("No users found for user_id=%s", user_id)
            raise SystemExit(1)

        for u in users:
            mem_list = list(_iter_memories(db, user_pk=u.id, include_non_active=include_non_active))
            if memory_ids:
                mem_list = [m for m in mem_list if m.id in memory_ids]
            if limit is not None:
                mem_list = mem_list[: max(0, int(limit))]

            logger.info("User '%s': processing %d memories", u.user_id, len(mem_list))

            for idx, m in enumerate(mem_list, 1):
                stats.total += 1

                if not m.content:
                    stats.skipped += 1
                    logger.debug("SKIP empty content memory_id=%s user=%s", m.id, u.user_id)
                    continue

                snippet = m.content.replace("\n", " ")[:120]

                if dry_run:
                    stats.skipped += 1
                    logger.info("[DRY RUN] memory_id=%s user=%s \"%s\"", m.id, u.user_id, snippet)
                    continue

                try:
                    result = memory_client.graph.add(m.content, {"user_id": u.user_id})
                    added = result.get("added_entities") if isinstance(result, dict) else None
                    deleted = result.get("deleted_entities") if isinstance(result, dict) else None
                    stats.processed += 1
                    logger.info(
                        "OK memory_id=%s user=%s added=%s deleted=%s \"%s\"",
                        m.id,
                        u.user_id,
                        len(added) if isinstance(added, list) else None,
                        len(deleted) if isinstance(deleted, list) else None,
                        snippet,
                    )
                except Exception as e:  # noqa: BLE001
                    stats.failed += 1
                    logger.exception("FAIL memory_id=%s user=%s \"%s\": %s", m.id, u.user_id, snippet, e)

                if stats.total % 20 == 0:
                    logger.info(
                        "Progress: total=%d processed=%d skipped=%d failed=%d",
                        stats.total,
                        stats.processed,
                        stats.skipped,
                        stats.failed,
                    )

        logger.info(
            "Backfill complete: total=%d processed=%d skipped=%d failed=%d",
            stats.total,
            stats.processed,
            stats.skipped,
            stats.failed,
        )
        return stats, chosen_log_file
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Mem0 Graph Memory from OpenMemory SQL DB")
    parser.add_argument("--user-id", default=None, help="Backfill only this user_id (string)")
    parser.add_argument("--include-non-active", action="store_true", help="Include paused/archived/deleted memories")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N memories (per user)")
    parser.add_argument(
        "--memory-ids",
        default=None,
        help="Comma-separated list of memory UUIDs to process (useful for retries).",
    )
    parser.add_argument(
        "--retry-failed-from-log",
        default=None,
        help="Parse a previous backfill log file and retry only FAIL memory_ids.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM/Neo4j, just log")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    memory_ids = _parse_memory_ids(args.memory_ids)
    if args.retry_failed_from_log:
        retry_ids = _memory_ids_from_log(args.retry_failed_from_log)
        memory_ids = (memory_ids or set()) | retry_ids

    stats, log_file = backfill_mem0_graph_from_db(
        user_id=args.user_id,
        include_non_active=args.include_non_active,
        limit=args.limit,
        memory_ids=memory_ids,
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
