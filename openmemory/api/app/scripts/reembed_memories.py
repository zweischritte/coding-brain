#!/usr/bin/env python3
"""
Re-embed memories from the SQL database into the current vector store.

This is intended for embedding model swaps (e.g., 1536 -> 1024 dims).

Usage:
  python -m app.scripts.reembed_memories --user-id grischadallmer
  python -m app.scripts.reembed_memories --all-users
  python -m app.scripts.reembed_memories --user-id grischadallmer --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Memory, MemoryState, User
from app.utils.memory import get_memory_client


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _iso(dt):
    if not dt:
        return None
    try:
        return dt.astimezone(UTC).isoformat()
    except Exception:
        return dt.replace(tzinfo=UTC).isoformat()


def _iter_memories(
    db: Session,
    *,
    user: User,
    include_archived: bool,
    include_deleted: bool,
    limit: Optional[int],
    batch_size: int,
) -> Iterable[Memory]:
    query = db.query(Memory).filter(Memory.user_id == user.id)

    if not include_deleted:
        query = query.filter(Memory.state != MemoryState.deleted)
    if not include_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    query = query.order_by(Memory.created_at.asc())

    if limit:
        query = query.limit(limit)

    return query.yield_per(batch_size)


def _count_memories(
    db: Session,
    *,
    user: User,
    include_archived: bool,
    include_deleted: bool,
    limit: Optional[int],
) -> int:
    query = db.query(Memory).filter(Memory.user_id == user.id)

    if not include_deleted:
        query = query.filter(Memory.state != MemoryState.deleted)
    if not include_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    total = query.count()
    if limit:
        total = min(total, limit)
    return total


def _reembed_user(
    db: Session,
    *,
    user: User,
    include_archived: bool,
    include_deleted: bool,
    limit: Optional[int],
    batch_size: int,
    log_every: int,
    dry_run: bool,
) -> dict:
    memory_client = get_memory_client()
    if not memory_client or not getattr(memory_client, "vector_store", None):
        raise RuntimeError("Memory client or vector store unavailable")

    total = _count_memories(
        db,
        user=user,
        include_archived=include_archived,
        include_deleted=include_deleted,
        limit=limit,
    )
    logger.info("User=%s total=%s", user.user_id, total)

    if dry_run:
        logger.info("[DRY RUN] No vectors will be written.")

    processed = 0
    failed = 0

    for memory in _iter_memories(
        db,
        user=user,
        include_archived=include_archived,
        include_deleted=include_deleted,
        limit=limit,
        batch_size=batch_size,
    ):
        processed += 1
        content = memory.content or ""
        if not content.strip():
            failed += 1
            logger.warning("Skipping empty content id=%s", memory.id)
            continue

        payload = dict(memory.metadata_ or {})
        payload["data"] = content
        payload["user_id"] = user.user_id
        payload.setdefault("source_app", "openmemory")

        created_at = _iso(memory.created_at)
        updated_at = _iso(memory.updated_at)
        if created_at:
            payload["created_at"] = created_at
        if updated_at:
            payload["updated_at"] = updated_at

        if not dry_run:
            try:
                embedding = memory_client.embedding_model.embed(content, "add")
                memory_client.vector_store.insert(
                    vectors=[embedding],
                    payloads=[payload],
                    ids=[str(memory.id)],
                )
            except Exception as exc:
                failed += 1
                logger.error("Failed re-embedding id=%s: %s", memory.id, exc)
                continue

        if processed % log_every == 0:
            logger.info("Progress: %s/%s (failed=%s)", processed, total, failed)

    logger.info("Done user=%s processed=%s failed=%s", user.user_id, processed, failed)
    return {"processed": processed, "failed": failed, "total": total}


def _get_user(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def _get_all_users(db: Session) -> list[User]:
    return db.query(User).order_by(User.user_id.asc()).all()


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed memories into the current vector store.")
    parser.add_argument("--user-id", help="User ID to re-embed (e.g., grischadallmer)")
    parser.add_argument("--all-users", action="store_true", help="Re-embed memories for all users")
    parser.add_argument("--include-archived", action="store_true", help="Include archived memories")
    parser.add_argument("--include-deleted", action="store_true", help="Include deleted memories")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of memories per user")
    parser.add_argument("--batch-size", type=int, default=200, help="DB fetch batch size")
    parser.add_argument("--log-every", type=int, default=50, help="Progress log interval")
    parser.add_argument("--dry-run", action="store_true", help="Do not write vectors")
    args = parser.parse_args()

    if not args.user_id and not args.all_users:
        parser.error("Must provide --user-id or --all-users")

    db = SessionLocal()
    try:
        users: list[User] = []
        if args.all_users:
            users = _get_all_users(db)
        else:
            user = _get_user(db, args.user_id)
            if not user:
                logger.error("User not found: %s", args.user_id)
                return 1
            users = [user]

        overall_failed = 0
        overall_processed = 0
        overall_total = 0

        for user in users:
            stats = _reembed_user(
                db,
                user=user,
                include_archived=args.include_archived,
                include_deleted=args.include_deleted,
                limit=args.limit,
                batch_size=args.batch_size,
                log_every=args.log_every,
                dry_run=args.dry_run,
            )
            overall_processed += stats["processed"]
            overall_failed += stats["failed"]
            overall_total += stats["total"]

        logger.info("Summary: total=%s processed=%s failed=%s", overall_total, overall_processed, overall_failed)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
