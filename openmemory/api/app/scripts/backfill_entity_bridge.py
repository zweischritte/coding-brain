#!/usr/bin/env python3
"""
Backfill entity bridge for existing memories.

This script processes existing memories and:
1. Extracts entities from memory content using Mem0's LLM
2. Creates OM_ABOUT edges for each extracted entity (multi-entity per memory)
3. Creates OM_RELATION edges with typed relationships
4. Updates OM_CO_MENTIONED edges

This solves the "single entity per memory" limitation that prevented
entity co-mention analysis from working.

Usage (inside `openmemory/api`):
    python -m app.scripts.backfill_entity_bridge --user-id grischadallmer

Options:
    --user-id USER_ID         Required: User ID to backfill
    --limit N                 Process at most N memories (default: all)
    --batch-size N            Batch size for progress logging (default: 10)
    --dry-run                 Extract entities but don't write to Neo4j
    --verbose                 Enable debug logging
    --memory-ids IDS          Comma-separated list of specific memory IDs to process

Example:
    python -m app.scripts.backfill_entity_bridge --user-id grischadallmer --limit 50 --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)


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


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"backfill_entity_bridge_{ts}.log")


@dataclass
class BackfillStats:
    memories_processed: int = 0
    memories_skipped: int = 0
    memories_failed: int = 0
    total_entities_bridged: int = 0
    total_relations_created: int = 0
    failed_memory_ids: List[str] = field(default_factory=list)


def backfill_entity_bridge(
    *,
    user_id: str,
    limit: Optional[int] = None,
    batch_size: int = 10,
    dry_run: bool = False,
    log_file: Optional[str] = None,
    verbose: bool = False,
    memory_ids: Optional[List[str]] = None,
) -> tuple[BackfillStats, str]:
    """
    Backfill entity bridge for existing memories.

    Args:
        user_id: String user ID (required)
        limit: Max memories to process
        batch_size: Progress log interval
        dry_run: If True, extract but don't write
        log_file: Path to log file
        verbose: Enable debug logging
        memory_ids: Specific memory IDs to process

    Returns:
        Tuple of (stats, log_file_path)
    """
    from app.database import SessionLocal
    from app.models import Memory, MemoryState, User
    from app.graph.neo4j_client import is_neo4j_configured
    from app.graph.entity_bridge import (
        bridge_entities_to_om_graph,
        extract_entities_from_content,
    )
    from app.graph.graph_ops import is_mem0_graph_enabled

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting entity bridge backfill")
    logger.info(f"user_id={user_id} limit={limit} batch_size={batch_size} dry_run={dry_run}")
    logger.info(f"log_file={chosen_log_file}")

    if not is_neo4j_configured():
        logger.error("Neo4j is not configured")
        raise SystemExit(1)

    if not is_mem0_graph_enabled():
        logger.error("Mem0 Graph Memory is not enabled. Run enable_mem0_graph_memory.py first.")
        raise SystemExit(1)

    stats = BackfillStats()

    db = SessionLocal()
    try:
        # Get user
        user = db.query(User).filter(User.name == user_id).first()
        if not user:
            logger.error(f"User {user_id} not found")
            raise SystemExit(1)

        # Build query for memories
        query = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state == MemoryState.active,
        )

        if memory_ids:
            import uuid
            uuids = [uuid.UUID(mid) for mid in memory_ids]
            query = query.filter(Memory.id.in_(uuids))

        query = query.order_by(Memory.created_at.asc())

        if limit:
            query = query.limit(limit)

        memories = query.all()
        total_count = len(memories)
        logger.info(f"Found {total_count} memories to process")

        start_time = time.time()

        for i, memory in enumerate(memories, 1):
            memory_id = str(memory.id)

            try:
                content = memory.content
                if not content or len(content.strip()) < 10:
                    logger.debug(f"Skipping memory {memory_id}: content too short")
                    stats.memories_skipped += 1
                    continue

                existing_entity = None
                if memory.metadata_:
                    existing_entity = memory.metadata_.get("re") or memory.metadata_.get("entity")

                if dry_run:
                    # Just extract, don't write
                    entities, relations = extract_entities_from_content(content, user_id)
                    logger.info(
                        f"[DRY RUN] Memory {memory_id}: "
                        f"extracted {len(entities)} entities, {len(relations)} relations"
                    )
                    if entities:
                        logger.debug(f"  Entities: {[e.name for e in entities]}")
                    if relations:
                        logger.debug(f"  Relations: {[(r.source, r.relationship, r.destination) for r in relations]}")
                    stats.memories_processed += 1
                    stats.total_entities_bridged += len(entities)
                    stats.total_relations_created += len(relations)
                else:
                    # Actually bridge
                    result = bridge_entities_to_om_graph(
                        memory_id=memory_id,
                        user_id=user_id,
                        content=content,
                        existing_entity=existing_entity,
                    )

                    if result.get("error"):
                        logger.warning(f"FAIL memory_id={memory_id}: {result['error']}")
                        stats.memories_failed += 1
                        stats.failed_memory_ids.append(memory_id)
                    else:
                        stats.memories_processed += 1
                        stats.total_entities_bridged += result.get("entities_bridged", 0)
                        stats.total_relations_created += result.get("relations_created", 0)

                        if result.get("entities_bridged", 0) > 0:
                            logger.info(
                                f"OK memory_id={memory_id}: "
                                f"{result['entities_bridged']} entities, "
                                f"{result['relations_created']} relations"
                            )

            except Exception as e:
                logger.exception(f"FAIL memory_id={memory_id}: {e}")
                stats.memories_failed += 1
                stats.failed_memory_ids.append(memory_id)

            # Progress logging
            if i % batch_size == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i}/{total_count} memories "
                    f"({100*i/total_count:.1f}%), "
                    f"{rate:.1f} mem/s"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"Backfill complete in {elapsed:.1f}s: "
            f"processed={stats.memories_processed}, "
            f"skipped={stats.memories_skipped}, "
            f"failed={stats.memories_failed}, "
            f"entities={stats.total_entities_bridged}, "
            f"relations={stats.total_relations_created}"
        )

        if stats.failed_memory_ids:
            logger.info(f"Failed memory IDs: {','.join(stats.failed_memory_ids[:20])}")
            if len(stats.failed_memory_ids) > 20:
                logger.info(f"  ... and {len(stats.failed_memory_ids) - 20} more")

    finally:
        db.close()

    return stats, chosen_log_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill entity bridge for existing memories")
    parser.add_argument("--user-id", required=True, help="User ID to backfill")
    parser.add_argument("--limit", type=int, default=None, help="Max memories to process")
    parser.add_argument("--batch-size", type=int, default=10, help="Progress log interval")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't write")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--memory-ids", default=None, help="Comma-separated memory IDs")
    args = parser.parse_args()

    memory_ids = None
    if args.memory_ids:
        memory_ids = [mid.strip() for mid in args.memory_ids.split(",")]

    stats, log_file = backfill_entity_bridge(
        user_id=args.user_id,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
        memory_ids=memory_ids,
    )

    print(
        f"Entity bridge backfill complete. "
        f"processed={stats.memories_processed}, "
        f"entities={stats.total_entities_bridged}, "
        f"relations={stats.total_relations_created}, "
        f"log_file={log_file}"
    )


if __name__ == "__main__":
    main()
