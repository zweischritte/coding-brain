#!/usr/bin/env python3
"""
Sync missing memories from SQLite to Qdrant.

This script finds memories that exist in SQLite but not in Qdrant,
and creates embeddings + upserts them to the vector store.

Usage:
    docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
      python -m app.scripts.sync_qdrant_from_db --user-id grischadallmer

    # Dry run (show what would be synced):
    docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
      python -m app.scripts.sync_qdrant_from_db --user-id grischadallmer --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional, Set
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_uuid(uuid_str: str) -> str:
    """Normalize UUID to format without dashes for comparison."""
    return str(uuid_str).replace("-", "").lower()


def get_qdrant_memory_ids(user_id: str) -> Set[str]:
    """Get all memory IDs currently in Qdrant for a user (normalized without dashes)."""
    import httpx

    memory_ids = set()
    offset = None

    with httpx.Client(timeout=60.0) as client:
        while True:
            payload = {
                "limit": 100,
                "with_payload": ["user_id"],
                "filter": {
                    "must": [{"key": "user_id", "match": {"value": user_id}}]
                }
            }
            if offset:
                payload["offset"] = offset

            try:
                resp = client.post(
                    "http://mem0_store:6333/collections/openmemory/points/scroll",
                    json=payload
                )
                data = resp.json()
                points = data.get("result", {}).get("points", [])

                for p in points:
                    # Normalize to no-dash format for comparison
                    memory_ids.add(normalize_uuid(p["id"]))

                offset = data.get("result", {}).get("next_page_offset")
                if not offset or not points:
                    break
            except Exception as e:
                logger.error(f"Error scrolling Qdrant: {e}")
                break

    return memory_ids


def get_sqlite_memories(user_uuid: str):
    """Get all active memories from SQLite."""
    import sqlite3
    import json

    conn = sqlite3.connect('/usr/src/openmemory/openmemory.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, content, metadata, created_at, vault, layer, axis_vector
        FROM memories
        WHERE user_id = ? AND state = 'active'
    """, (user_uuid,))

    memories = []
    for row in cursor.fetchall():
        metadata = {}
        if row[2]:
            try:
                metadata = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            except:
                pass

        memories.append({
            "id": normalize_uuid(row[0]),  # Normalized for comparison
            "id_original": str(row[0]),
            "content": row[1],
            "metadata": metadata,
            "created_at": row[3],
            "vault": row[4],
            "layer": row[5],
            "axis_vector": row[6]
        })

    conn.close()
    return memories


def get_user_uuid(user_id: str) -> Optional[str]:
    """Map user_id string to internal UUID."""
    import sqlite3

    try:
        conn = sqlite3.connect('/usr/src/openmemory/openmemory.db')
        cursor = conn.cursor()
        # user_id column contains the string identifier (e.g., "grischadallmer")
        # id column contains the internal UUID
        cursor.execute("SELECT id FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        return str(row[0]) if row else None
    except Exception as e:
        logger.error(f"Error looking up user: {e}")
        return None


def sync_memory_to_qdrant(memory: dict, user_id: str, dry_run: bool = False) -> bool:
    """Sync a single memory to Qdrant."""
    from app.utils.memory import get_memory_client

    if dry_run:
        return True

    try:
        client = get_memory_client()

        # Build metadata from stored fields
        metadata = memory.get("metadata", {}) or {}
        if memory.get("vault"):
            metadata["vault"] = memory["vault"]
        if memory.get("layer"):
            metadata["layer"] = memory["layer"]
        if memory.get("axis_vector"):
            metadata["axis_vector"] = memory["axis_vector"]

        # Use mem0's add method which handles embedding
        result = client.add(
            memory["content"],
            user_id=user_id,
            metadata=metadata
        )

        return True
    except Exception as e:
        logger.error(f"Failed to sync memory {memory['id']}: {e}")
        return False


def run_sync(user_id: str, dry_run: bool = False, limit: Optional[int] = None):
    """Run the sync process."""

    # Get user UUID
    user_uuid = get_user_uuid(user_id)
    if not user_uuid:
        logger.error(f"User '{user_id}' not found in database")
        return

    logger.info(f"User '{user_id}' -> UUID: {user_uuid}")

    # Get memories from both stores
    logger.info("Fetching memory IDs from Qdrant...")
    qdrant_ids = get_qdrant_memory_ids(user_id)
    logger.info(f"Found {len(qdrant_ids)} memories in Qdrant")

    logger.info("Fetching memories from SQLite...")
    sqlite_memories = get_sqlite_memories(user_uuid)
    logger.info(f"Found {len(sqlite_memories)} active memories in SQLite")

    # Find missing memories
    sqlite_ids = {m["id"] for m in sqlite_memories}
    missing_ids = sqlite_ids - qdrant_ids

    logger.info(f"Missing in Qdrant: {len(missing_ids)} memories")

    if not missing_ids:
        logger.info("All memories are synced!")
        return

    # Get missing memory objects
    missing_memories = [m for m in sqlite_memories if m["id"] in missing_ids]

    if limit:
        missing_memories = missing_memories[:limit]
        logger.info(f"Limited to {limit} memories")

    # Sync missing memories
    success_count = 0
    fail_count = 0

    for i, memory in enumerate(missing_memories):
        if dry_run:
            logger.info(f"[DRY RUN] Would sync: {memory['id'][:8]}... - {memory['content'][:50]}...")
            success_count += 1
        else:
            logger.info(f"[{i+1}/{len(missing_memories)}] Syncing: {memory['id'][:8]}... - {memory['content'][:50]}...")
            if sync_memory_to_qdrant(memory, user_id, dry_run):
                success_count += 1
            else:
                fail_count += 1

        # Progress every 50
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(missing_memories)} ({success_count} success, {fail_count} failed)")

    # Summary
    logger.info("=" * 60)
    logger.info("Sync Summary:")
    logger.info(f"  Total missing: {len(missing_ids)}")
    logger.info(f"  Processed: {len(missing_memories)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    if dry_run:
        logger.info("  (DRY RUN - no changes made)")


def main():
    parser = argparse.ArgumentParser(
        description="Sync missing memories from SQLite to Qdrant"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to sync"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of memories to sync"
    )

    args = parser.parse_args()

    run_sync(
        user_id=args.user_id,
        dry_run=args.dry_run,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
