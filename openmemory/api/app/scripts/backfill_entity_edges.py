#!/usr/bin/env python3
"""
Backfill entity co-mention edges (OM_CO_MENTIONED) in Neo4j.

This creates edges between OM_Entity nodes that appear together in the same memories.
Fast operation - uses pure Cypher, no vector DB queries.

Usage (inside `openmemory/api`):
    python -m app.scripts.backfill_entity_edges

Options:
    --user-id USER_ID         Only backfill for a specific user (required)
    --min-count N             Min co-mentions to create edge (default: 1)
    --dry-run                 Log what would be done without writing
    --verbose                 Enable debug logging

Example:
    python -m app.scripts.backfill_entity_edges --user-id grischadallmer
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
    return str(Path("logs") / f"backfill_entity_edges_{ts}.log")


@dataclass
class BackfillStats:
    edges_created: int = 0
    entity_pairs: int = 0


def backfill_entity_edges(
    *,
    user_id: str,
    min_count: int = 1,
    dry_run: bool = False,
    access_entity: str | None = None,
    log_file: str | None = None,
    verbose: bool = False,
) -> tuple[BackfillStats, str]:
    """
    Backfill entity co-mention edges for a user.

    Args:
        user_id: String user ID (required)
        min_count: Minimum co-mentions to create edge
        dry_run: If True, only log what would be done
        log_file: Path to log file
        verbose: Enable debug logging

    Returns:
        Tuple of (stats, log_file_path)
    """
    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session
    from app.graph.metadata_projector import get_projector

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting entity edge backfill")
    access_entity = access_entity or f"user:{user_id}"
    logger.info(f"user_id={user_id} access_entity={access_entity} min_count={min_count} dry_run={dry_run}")
    logger.info(f"log_file={chosen_log_file}")

    if not is_neo4j_configured():
        logger.error("Neo4j is not configured")
        raise SystemExit(1)

    projector = get_projector()
    if projector is None:
        logger.error("Failed to get metadata projector")
        raise SystemExit(1)

    stats = BackfillStats()

    if dry_run:
        # Count entity pairs that would get edges
        try:
            with get_neo4j_session() as session:
                result = session.run("""
                    MATCH (e1:OM_Entity)<-[:OM_ABOUT]-(m:OM_Memory)-[:OM_ABOUT]->(e2:OM_Entity)
                    WHERE coalesce(e1.accessEntity, $legacyAccessEntity) = $accessEntity
                      AND coalesce(e2.accessEntity, $legacyAccessEntity) = $accessEntity
                      AND coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity
                      AND e1.name < e2.name
                    WITH e1, e2, count(m) AS cnt
                    WHERE cnt >= $minCount
                    RETURN count(*) AS pairCount, sum(cnt) AS totalMentions
                """, {
                    "userId": user_id,
                    "accessEntity": access_entity,
                    "legacyAccessEntity": f"user:{user_id}",
                    "minCount": min_count,
                })
                record = result.single()
                if record:
                    stats.entity_pairs = record["pairCount"]
                    logger.info(f"[DRY RUN] Would create edges for {stats.entity_pairs} entity pairs")
        except Exception as e:
            logger.exception(f"Error during dry run: {e}")
            raise SystemExit(1)
    else:
        # Actually create the edges
        try:
            edges_created = projector.backfill_entity_edges(user_id, min_count, access_entity=access_entity)
            stats.edges_created = edges_created
            logger.info(f"Created {edges_created} OM_CO_MENTIONED edges")
        except Exception as e:
            logger.exception(f"Error backfilling entity edges: {e}")
            raise SystemExit(1)

    logger.info(f"Backfill complete: edges_created={stats.edges_created}")
    return stats, chosen_log_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill entity co-mention edges")
    parser.add_argument("--user-id", required=True, help="User ID to backfill")
    parser.add_argument("--min-count", type=int, default=1, help="Min co-mentions for edge")
    parser.add_argument("--access-entity", default=None, help="Access entity scope (default: user scope)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Neo4j")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    stats, log_file = backfill_entity_edges(
        user_id=args.user_id,
        min_count=args.min_count,
        dry_run=args.dry_run,
        access_entity=args.access_entity,
        log_file=args.log_file,
        verbose=args.verbose,
    )

    print(f"Entity edge backfill complete. edges={stats.edges_created} log_file={log_file}")


if __name__ == "__main__":
    main()
