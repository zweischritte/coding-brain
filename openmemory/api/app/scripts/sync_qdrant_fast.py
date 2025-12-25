#!/usr/bin/env python3
"""
Fast sync: Directly embed and insert missing memories to Qdrant.

This bypasses mem0's deduplication logic and directly:
1. Computes embeddings via OpenAI
2. Upserts to Qdrant with proper payload

Usage:
    docker exec openmemory-openmemory-mcp-1 \
      python -m app.scripts.sync_qdrant_fast --user-id grischadallmer
"""

import argparse
import logging
import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, Set, List, Dict, Any
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

QDRANT_URL = "http://mem0_store:6333"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 20  # Process this many at once


def normalize_uuid(uuid_str: str) -> str:
    """Normalize UUID to format without dashes."""
    return str(uuid_str).replace("-", "").lower()


def to_qdrant_uuid(uuid_str: str) -> str:
    """Convert to Qdrant UUID format (with dashes)."""
    s = normalize_uuid(uuid_str)
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def get_qdrant_memory_ids(user_id: str) -> Set[str]:
    """Get all memory IDs currently in Qdrant for a user."""
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

            resp = client.post(f"{QDRANT_URL}/collections/openmemory/points/scroll", json=payload)
            data = resp.json()
            points = data.get("result", {}).get("points", [])

            for p in points:
                memory_ids.add(normalize_uuid(p["id"]))

            offset = data.get("result", {}).get("next_page_offset")
            if not offset or not points:
                break

    return memory_ids


def get_user_uuid(user_id: str) -> Optional[str]:
    """Map user_id string to internal UUID."""
    conn = sqlite3.connect('/usr/src/openmemory/openmemory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return str(row[0]) if row else None


def get_sqlite_memories(user_uuid: str) -> List[Dict[str, Any]]:
    """Get all active memories from SQLite."""
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
            "id": normalize_uuid(row[0]),
            "id_qdrant": to_qdrant_uuid(row[0]),
            "content": row[1],
            "metadata": metadata,
            "created_at": row[3],
            "vault": row[4],
            "layer": row[5],
            "axis_vector": row[6]
        })

    conn.close()
    return memories


def compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Compute embeddings for a batch of texts using OpenAI."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": texts
            }
        )
        data = resp.json()

        if "error" in data:
            raise Exception(f"OpenAI API error: {data['error']}")

        # Sort by index to maintain order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]


def upsert_to_qdrant(points: List[Dict]) -> bool:
    """Upsert points to Qdrant."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.put(
            f"{QDRANT_URL}/collections/openmemory/points?wait=true",
            json={"points": points}
        )
        return resp.status_code == 200


def build_payload(memory: Dict, user_id: str) -> Dict:
    """Build Qdrant payload from memory."""
    metadata = memory.get("metadata", {}) or {}

    payload = {
        "user_id": user_id,
        "data": memory["content"],
        "hash": "",  # Will be computed by mem0 on next access
        "created_at": memory.get("created_at") or datetime.now().isoformat(),
    }

    # Copy metadata fields
    for key in ["source_app", "mcp_client", "source", "vault", "layer",
                "axis_category", "src", "re", "tags", "circuit"]:
        if key in metadata:
            payload[key] = metadata[key]

    # Override with top-level fields if present
    if memory.get("vault"):
        payload["vault"] = memory["vault"]
    if memory.get("layer"):
        payload["layer"] = memory["layer"]
    if memory.get("axis_vector"):
        payload["axis_vector"] = memory["axis_vector"]

    return payload


def run_sync(user_id: str, dry_run: bool = False, limit: Optional[int] = None):
    """Run the fast sync process."""

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        return

    # Get user UUID
    user_uuid = get_user_uuid(user_id)
    if not user_uuid:
        logger.error(f"User '{user_id}' not found")
        return

    logger.info(f"User '{user_id}' -> UUID: {user_uuid}")

    # Get current state
    logger.info("Fetching memory IDs from Qdrant...")
    qdrant_ids = get_qdrant_memory_ids(user_id)
    logger.info(f"Found {len(qdrant_ids)} memories in Qdrant")

    logger.info("Fetching memories from SQLite...")
    sqlite_memories = get_sqlite_memories(user_uuid)
    logger.info(f"Found {len(sqlite_memories)} active memories in SQLite")

    # Find missing
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

    if dry_run:
        logger.info(f"[DRY RUN] Would sync {len(missing_memories)} memories")
        for m in missing_memories[:10]:
            logger.info(f"  {m['id'][:8]}... - {m['content'][:50]}...")
        return

    # Process in batches
    success_count = 0
    fail_count = 0

    for batch_start in range(0, len(missing_memories), BATCH_SIZE):
        batch = missing_memories[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(missing_memories) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} memories)...")

        try:
            # Compute embeddings for batch
            texts = [m["content"] for m in batch]
            embeddings = compute_embeddings(texts)

            # Build points
            points = []
            for i, memory in enumerate(batch):
                points.append({
                    "id": memory["id_qdrant"],
                    "vector": embeddings[i],
                    "payload": build_payload(memory, user_id)
                })

            # Upsert to Qdrant
            if upsert_to_qdrant(points):
                success_count += len(batch)
                logger.info(f"  ✓ Inserted {len(batch)} memories")
            else:
                fail_count += len(batch)
                logger.error(f"  ✗ Failed to insert batch")

        except Exception as e:
            fail_count += len(batch)
            logger.error(f"  ✗ Error processing batch: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("Sync Summary:")
    logger.info(f"  Total missing: {len(missing_ids)}")
    logger.info(f"  Processed: {len(missing_memories)}")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description="Fast sync memories to Qdrant")
    parser.add_argument("--user-id", required=True, help="User ID to sync")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--limit", type=int, help="Limit number of memories")

    args = parser.parse_args()
    run_sync(user_id=args.user_id, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
