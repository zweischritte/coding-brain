"""
Backfill structured metadata into the SQL database from the vector store.

Usage:
    poetry run python -m app.scripts.backfill_metadata [--limit 500] [--dry-run]

Notes:
- Only memories with empty/None metadata_ are updated.
- Metadata is pulled from the vector store via memory_client.get(memory_id).
"""

import argparse
import json
import logging
from typing import Dict

from app.database import SessionLocal
from app.models import Memory
from app.utils.memory import get_memory_client


def backfill_metadata(*, limit: int | None = None, dry_run: bool = False) -> Dict[str, int]:
    db = SessionLocal()
    memory_client = get_memory_client()

    updated = skipped_existing = missing_remote = errors = 0

    try:
        query = db.query(Memory).order_by(Memory.created_at.asc())
        if limit:
            query = query.limit(limit)

        for memory in query:
            if memory.metadata_:
                skipped_existing += 1
                continue

            try:
                remote = memory_client.get(str(memory.id))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to fetch memory %s from vector store: %s", memory.id, exc)
                errors += 1
                continue

            meta = remote.get("metadata") if isinstance(remote, dict) else None
            if not meta:
                missing_remote += 1
                continue

            if not dry_run:
                memory.metadata_ = meta

            updated += 1

        if not dry_run and updated:
            db.commit()
    finally:
        db.close()

    return {
        "updated": updated,
        "skipped_existing": skipped_existing,
        "missing_remote": missing_remote,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill structured metadata into SQL from vector store")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of memories to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes, just report")
    args = parser.parse_args()

    stats = backfill_metadata(limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
