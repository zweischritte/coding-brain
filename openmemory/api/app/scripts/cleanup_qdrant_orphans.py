#!/usr/bin/env python3
"""
Remove orphaned Qdrant points (no matching DB memory) for a user.

Usage:
  python -m app.scripts.cleanup_qdrant_orphans --user-id grischadallmer
  python -m app.scripts.cleanup_qdrant_orphans --user-id grischadallmer --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx

from app.database import SessionLocal
from app.models import Memory, MemoryState, User
from app.utils.memory import get_memory_client


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _get_collection_name() -> str:
    memory_client = get_memory_client()
    vector_store = getattr(memory_client, "vector_store", None) if memory_client else None
    if vector_store and getattr(vector_store, "collection_name", None):
        return vector_store.collection_name
    return os.environ.get("QDRANT_COLLECTION", "openmemory_bge_m3")


def _get_qdrant_url() -> str:
    host = os.environ.get("QDRANT_HOST", "qdrant")
    port = os.environ.get("QDRANT_PORT", "6333")
    if host.startswith("http://") or host.startswith("https://"):
        return f"{host}:{port}" if ":" not in host.split("//", 1)[1] else host
    return f"http://{host}:{port}"


def _get_user(db, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def _db_ids_and_states(db, user: User) -> Tuple[Set[str], Dict[str, str]]:
    rows = db.query(Memory.id, Memory.state).filter(Memory.user_id == user.id).all()
    ids = set()
    states: Dict[str, str] = {}
    for mem_id, state in rows:
        key = str(mem_id)
        ids.add(key)
        states[key] = state.value if isinstance(state, MemoryState) else str(state)
    return ids, states


def _scroll_qdrant_ids(client: httpx.Client, *, collection: str, user_id: str) -> List[str]:
    ids: List[str] = []
    offset = None
    while True:
        payload = {
            "limit": 200,
            "with_payload": ["user_id"],
            "filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]},
        }
        if offset:
            payload["offset"] = offset

        resp = client.post(f"/collections/{collection}/points/scroll", json=payload)
        resp.raise_for_status()
        data = resp.json()
        points = data.get("result", {}).get("points", [])
        for point in points:
            ids.append(str(point.get("id")))
        offset = data.get("result", {}).get("next_page_offset")
        if not offset or not points:
            break
    return ids


def _delete_points(client: httpx.Client, *, collection: str, point_ids: Iterable[str]) -> None:
    payload = {"points": list(point_ids)}
    resp = client.post(f"/collections/{collection}/points/delete?wait=true", json=payload)
    resp.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete orphaned Qdrant points for a user.")
    parser.add_argument("--user-id", required=True, help="User ID (e.g., grischadallmer)")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete points")
    parser.add_argument("--batch-size", type=int, default=200, help="Delete batch size")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        user = _get_user(db, args.user_id)
        if not user:
            logger.error("User not found: %s", args.user_id)
            return 1

        db_ids, db_states = _db_ids_and_states(db, user)

    finally:
        db.close()

    collection = _get_collection_name()
    base_url = _get_qdrant_url()
    logger.info("Using collection=%s url=%s", collection, base_url)

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        qdrant_ids = _scroll_qdrant_ids(client, collection=collection, user_id=args.user_id)

        qdrant_set = set(qdrant_ids)
        orphans = sorted(qdrant_set - db_ids)

        stale_deleted = [
            mid for mid in qdrant_set
            if mid in db_states and db_states[mid] in {"deleted", "archived"}
        ]

        logger.info("Qdrant points (user)=%s", len(qdrant_set))
        logger.info("DB memories (all states)=%s", len(db_ids))
        logger.info("Orphans (not in DB)=%s", len(orphans))
        logger.info("Stale deleted/archived (in DB)=%s", len(stale_deleted))

        if args.dry_run:
            logger.info("[DRY RUN] No deletions performed.")
            return 0

        if not orphans:
            logger.info("No orphans to delete.")
            return 0

        batch_size = max(1, args.batch_size)
        for i in range(0, len(orphans), batch_size):
            batch = orphans[i:i + batch_size]
            _delete_points(client, collection=collection, point_ids=batch)
            logger.info("Deleted %s/%s orphan points", min(i + batch_size, len(orphans)), len(orphans))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
