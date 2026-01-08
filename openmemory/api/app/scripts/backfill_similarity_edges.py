#!/usr/bin/env python3
"""
Backfill memory-to-memory similarity edges (OM_SIMILAR) in Neo4j.

This creates edges between OM_Memory nodes based on semantic similarity
from Qdrant embeddings. Slower than entity/tag backfills because it
queries Qdrant for each memory.

Configuration via environment variables:
- OM_SIMILARITY_K: Number of nearest neighbors per memory (default: 20)
- OM_SIMILARITY_THRESHOLD: Minimum cosine similarity score (default: 0.6)
- OM_SIMILARITY_MAX_EDGES: Maximum edges per memory (default: 30)

Usage (inside `openmemory/api`):
    python -m app.scripts.backfill_similarity_edges

Options:
    --user-id USER_ID         Only backfill for a specific user (required)
    --limit N                 Process at most N memories
    --batch-size N            Memories per batch (default: 50)
    --dry-run                 Log what would be done without writing
    --verbose                 Enable debug logging

Example:
    python -m app.scripts.backfill_similarity_edges --user-id grischadallmer
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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
    return str(Path("logs") / f"backfill_similarity_edges_{ts}.log")


@dataclass
class BackfillStats:
    total_memories: int = 0
    processed: int = 0
    edges_created: int = 0
    skipped: int = 0
    failed: int = 0


def backfill_similarity_edges(
    *,
    user_id: str,
    limit: Optional[int] = None,
    batch_size: int = 50,
    dry_run: bool = False,
    access_entity: str | None = None,
    log_file: str | None = None,
    verbose: bool = False,
) -> tuple[BackfillStats, str]:
    """
    Backfill similarity edges for a user's memories.

    Args:
        user_id: String user ID (required)
        limit: Maximum memories to process
        batch_size: Memories per progress log
        dry_run: If True, only log what would be done
        log_file: Path to log file
        verbose: Enable debug logging

    Returns:
        Tuple of (stats, log_file_path)
    """
    from app.database import SessionLocal
    from app.models import Memory, MemoryState
    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session
    from app.graph.similarity_projector import get_similarity_projector, SimilarityConfig

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    config = SimilarityConfig.from_env()
    logger.info("Starting similarity edge backfill")
    access_entity = access_entity or f"user:{user_id}"
    logger.info(
        f"user_id={user_id} access_entity={access_entity} limit={limit} "
        f"batch_size={batch_size} dry_run={dry_run}"
    )
    logger.info(f"config: k={config.k_neighbors} threshold={config.min_similarity_threshold} max_edges={config.max_edges_per_memory}")
    logger.info(f"log_file={chosen_log_file}")

    if not is_neo4j_configured():
        logger.error("Neo4j is not configured")
        raise SystemExit(1)

    projector = get_similarity_projector()
    if projector is None:
        logger.error("Failed to initialize similarity projector (Neo4j or Qdrant not available)")
        raise SystemExit(1)

    stats = BackfillStats()

    # Get memories from database
    db = SessionLocal()
    try:
        from app.models import User

        # Find the user by user_id string (for legacy scope fallback)
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            logger.error(f"User not found: {user_id}")
            raise SystemExit(1)

        # Get active memories scoped by access_entity
        if access_entity.startswith("user:"):
            query = db.query(Memory).filter(
                Memory.user_id == user.id,
                Memory.state == MemoryState.active,
            )
        else:
            access_entity_col = Memory.metadata_["access_entity"].as_string()
            query = db.query(Memory).filter(
                Memory.state == MemoryState.active,
                access_entity_col == access_entity,
            )
        query = query.order_by(Memory.created_at.asc())

        if limit:
            query = query.limit(limit)

        memories = query.all()
        stats.total_memories = len(memories)
        logger.info(f"Found {stats.total_memories} active memories for user '{user_id}'")

        if dry_run:
            logger.info(f"[DRY RUN] Would process {stats.total_memories} memories")
            logger.info(f"[DRY RUN] Expected ~{stats.total_memories * config.k_neighbors} similarity edges")
            stats.skipped = stats.total_memories
            return stats, chosen_log_file

        # Process each memory
        for idx, memory in enumerate(memories, 1):
            memory_id = str(memory.id)

            try:
                edges = projector.project_similarity_edges(memory_id, user_id)
                stats.edges_created += edges
                stats.processed += 1

                if verbose:
                    logger.debug(f"OK {memory_id}: created {edges} similarity edges")

            except Exception as e:
                stats.failed += 1
                logger.warning(f"FAIL {memory_id}: {e}")

            # Progress logging
            if idx % batch_size == 0:
                logger.info(f"Progress: {idx}/{stats.total_memories} processed, {stats.edges_created} edges created")

        logger.info(
            f"Backfill complete: total={stats.total_memories} processed={stats.processed} "
            f"edges={stats.edges_created} failed={stats.failed}"
        )

        # Log edge count per user
        try:
            edge_count = projector.count_similarity_edges(user_id, access_entity=access_entity)
            logger.info(f"Total OM_SIMILAR edges for user '{user_id}': {edge_count}")
        except Exception:
            pass

        return stats, chosen_log_file

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill memory similarity edges")
    parser.add_argument("--user-id", required=True, help="User ID to backfill")
    parser.add_argument("--limit", type=int, default=None, help="Max memories to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Progress log interval")
    parser.add_argument("--access-entity", default=None, help="Access entity scope (default: user scope)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Neo4j")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    stats, log_file = backfill_similarity_edges(
        user_id=args.user_id,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        access_entity=args.access_entity,
        log_file=args.log_file,
        verbose=args.verbose,
    )

    print(
        f"Similarity edge backfill complete. "
        f"memories={stats.total_memories} processed={stats.processed} "
        f"edges={stats.edges_created} failed={stats.failed} "
        f"log_file={log_file}"
    )

    if stats.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
