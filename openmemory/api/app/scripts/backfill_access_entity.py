"""
Backfill access_entity for personal scopes in SQL metadata.

Usage:
    poetry run python -m app.scripts.backfill_access_entity --limit 500 --dry-run

Notes:
- Only updates memories with scope=user/session and missing access_entity.
- Uses Memory.user.user_id to derive user:<sub>.
- Does not attempt to set org_id (not stored on User).
- Re-sync vector/graph stores after backfill if needed.
"""

import argparse
import json
import logging
from typing import Dict

from sqlalchemy import cast, String, or_
from sqlalchemy.orm import selectinload

from app.database import SessionLocal
from app.models import Memory


def backfill_access_entity(*, limit: int | None = None, batch_size: int = 500, dry_run: bool = False) -> Dict[str, int]:
    db = SessionLocal()
    updated = skipped = missing_user = errors = 0

    try:
        access_entity_col = cast(Memory.metadata_["access_entity"], String)
        query = db.query(Memory).options(selectinload(Memory.user)).filter(
            or_(Memory.metadata_.is_(None), access_entity_col.is_(None))
        )
        if limit:
            query = query.limit(limit)

        for memory in query.yield_per(batch_size):
            metadata = memory.metadata_ or {}
            if metadata.get("access_entity"):
                skipped += 1
                continue

            scope = metadata.get("scope")
            if scope not in ("user", "session"):
                skipped += 1
                continue

            if not memory.user or not memory.user.user_id:
                missing_user += 1
                continue

            metadata["access_entity"] = f"user:{memory.user.user_id}"
            if not dry_run:
                memory.metadata_ = metadata
            updated += 1

        if not dry_run and updated:
            db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logging.exception("Backfill access_entity failed: %s", exc)
        errors += 1
    finally:
        db.close()

    return {
        "updated": updated,
        "skipped": skipped,
        "missing_user": missing_user,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill access_entity for personal scopes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of memories to process")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for iteration")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, just report")
    args = parser.parse_args()

    stats = backfill_access_entity(
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
