#!/usr/bin/env python3
"""
Backfill Qdrant metadata from PostgreSQL.

This script syncs metadata from PostgreSQL to Qdrant for memories that have
stale or missing metadata in the vector store. It uses set_payload() to
update only the payload without re-computing embeddings.

This is the backfill script for PRD-QDRANT-METADATA-SYNC.

Usage:
    # Dry run (show what would be updated):
    docker compose exec openmemory-mcp python -m app.scripts.backfill_qdrant_metadata --dry-run

    # Run for all users:
    docker compose exec openmemory-mcp python -m app.scripts.backfill_qdrant_metadata

    # Run for specific user:
    docker compose exec openmemory-mcp python -m app.scripts.backfill_qdrant_metadata --user-id grischadallmer

    # Limit batch size:
    docker compose exec openmemory-mcp python -m app.scripts.backfill_qdrant_metadata --limit 100
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_memory_client():
    """Get the memory client for Qdrant operations."""
    from app.utils.memory import get_memory_client as _get_memory_client
    return _get_memory_client()


def get_db_session():
    """Get a database session."""
    from app.database import SessionLocal
    return SessionLocal()


def get_all_active_memories(db, user_id: Optional[str] = None, limit: Optional[int] = None):
    """Get all active memories from PostgreSQL."""
    from app.models import Memory, MemoryState, User

    query = db.query(Memory).filter(Memory.state == MemoryState.active)

    if user_id:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            logger.error(f"User '{user_id}' not found")
            return []
        query = query.filter(Memory.user_id == user.id)

    query = query.order_by(Memory.created_at.desc())

    if limit:
        query = query.limit(limit)

    return query.all()


def get_qdrant_point(memory_client, memory_id: str) -> Optional[Dict[str, Any]]:
    """Get current Qdrant point for a memory."""
    try:
        vs = memory_client.vector_store
        result = vs.get(vector_id=memory_id)
        return result
    except Exception as e:
        logger.warning(f"Failed to get Qdrant point {memory_id}: {e}")
        return None


def build_payload_from_memory(memory) -> Dict[str, Any]:
    """Build Qdrant payload from memory ORM object."""
    import hashlib

    content = memory.content or ""
    metadata = memory.metadata_ or {}

    payload = {
        "data": content,
        "user_id": memory.user.user_id if memory.user else None,
        "hash": hashlib.md5(content.encode()).hexdigest(),
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        # Structured metadata
        "category": metadata.get("category"),
        "scope": metadata.get("scope"),
        "artifact_type": metadata.get("artifact_type"),
        "artifact_ref": metadata.get("artifact_ref"),
        "entity": metadata.get("entity"),
        "source": metadata.get("source"),
        "access_entity": metadata.get("access_entity"),
        "evidence": metadata.get("evidence"),
        "tags": metadata.get("tags", {}),
    }

    # Remove None values
    return {k: v for k, v in payload.items() if v is not None}


def metadata_needs_sync(pg_metadata: Dict[str, Any], qdrant_payload: Dict[str, Any]) -> bool:
    """Check if Qdrant payload is missing metadata that exists in PostgreSQL."""
    # Fields we care about for search ranking
    important_fields = ["entity", "category", "scope", "artifact_type", "artifact_ref", "access_entity", "tags"]

    for field in important_fields:
        pg_value = pg_metadata.get(field)
        qdrant_value = qdrant_payload.get(field)

        # If PostgreSQL has a value but Qdrant doesn't (or has different value)
        if pg_value is not None and pg_value != qdrant_value:
            return True

    return False


def sync_payload_to_qdrant(memory_client, memory_id: str, payload: Dict[str, Any], dry_run: bool = False) -> bool:
    """Sync payload to Qdrant using set_payload."""
    if dry_run:
        return True

    try:
        vs = memory_client.vector_store

        # Use set_payload to preserve the vector
        if hasattr(vs, "set_payload"):
            vs.set_payload(vector_id=memory_id, payload=payload)
        elif hasattr(vs, "client") and hasattr(vs, "collection_name"):
            vs.client.set_payload(
                collection_name=vs.collection_name,
                payload=payload,
                points=[memory_id],
            )
        else:
            logger.warning(f"Vector store doesn't support set_payload for {memory_id}")
            return False

        return True
    except Exception as e:
        logger.error(f"Failed to sync payload for {memory_id}: {e}")
        return False


def run_backfill(user_id: Optional[str] = None, dry_run: bool = False, limit: Optional[int] = None):
    """Run the metadata backfill process."""
    logger.info("=" * 60)
    logger.info("Qdrant Metadata Backfill")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN - no changes will be made")

    # Get memory client
    try:
        memory_client = get_memory_client()
        if not memory_client:
            logger.error("Failed to get memory client")
            return
    except Exception as e:
        logger.error(f"Failed to initialize memory client: {e}")
        return

    # Get database session
    db = get_db_session()

    try:
        # Get all active memories
        logger.info(f"Fetching memories from PostgreSQL{f' for user {user_id}' if user_id else ''}...")
        memories = get_all_active_memories(db, user_id=user_id, limit=limit)
        logger.info(f"Found {len(memories)} active memories")

        if not memories:
            logger.info("No memories to process")
            return

        # Process each memory
        stats = {
            "total": len(memories),
            "needs_sync": 0,
            "synced": 0,
            "failed": 0,
            "skipped_not_in_qdrant": 0,
            "up_to_date": 0,
        }

        for i, memory in enumerate(memories):
            memory_id = str(memory.id)
            pg_metadata = memory.metadata_ or {}

            # Get current Qdrant payload
            qdrant_point = get_qdrant_point(memory_client, memory_id)

            if not qdrant_point:
                stats["skipped_not_in_qdrant"] += 1
                if (i + 1) % 100 == 0 or i == 0:
                    logger.debug(f"Memory {memory_id[:8]} not in Qdrant, skipping")
                continue

            qdrant_payload = getattr(qdrant_point, "payload", {}) or {}

            # Check if sync is needed
            if not metadata_needs_sync(pg_metadata, qdrant_payload):
                stats["up_to_date"] += 1
                continue

            stats["needs_sync"] += 1

            # Build new payload
            new_payload = build_payload_from_memory(memory)

            # Log what would be updated
            if dry_run or logger.level <= logging.DEBUG:
                entity = pg_metadata.get("entity", "N/A")
                category = pg_metadata.get("category", "N/A")
                scope = pg_metadata.get("scope", "N/A")
                content_preview = (memory.content or "")[:40]
                logger.info(
                    f"[{i+1}/{len(memories)}] {'Would sync' if dry_run else 'Syncing'}: "
                    f"{memory_id[:8]}... entity={entity}, category={category}, scope={scope} "
                    f"- {content_preview}..."
                )

            # Sync to Qdrant
            if sync_payload_to_qdrant(memory_client, memory_id, new_payload, dry_run=dry_run):
                stats["synced"] += 1
            else:
                stats["failed"] += 1

            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(memories)} "
                    f"({stats['synced']} synced, {stats['failed']} failed, {stats['up_to_date']} up-to-date)"
                )

        # Summary
        logger.info("=" * 60)
        logger.info("Backfill Summary:")
        logger.info(f"  Total memories: {stats['total']}")
        logger.info(f"  Already up-to-date: {stats['up_to_date']}")
        logger.info(f"  Needed sync: {stats['needs_sync']}")
        logger.info(f"  Successfully synced: {stats['synced']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Not in Qdrant (skipped): {stats['skipped_not_in_qdrant']}")
        if dry_run:
            logger.info("  (DRY RUN - no changes made)")
        logger.info("=" * 60)

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Qdrant metadata from PostgreSQL"
    )
    parser.add_argument(
        "--user-id",
        help="Only process memories for this user ID"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of memories to process"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_backfill(
        user_id=args.user_id,
        dry_run=args.dry_run,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
