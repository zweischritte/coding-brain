#!/usr/bin/env python3
"""
Backfill Neo4j metadata graph from OpenMemory export.

This script reads an OpenMemory export file (memories.json) and projects
all memories into Neo4j as a deterministic metadata graph.

Usage:
    python -m app.scripts.backfill_neo4j_metadata /path/to/memories.json

    Options:
        --include-deleted    Include deleted memories (default: skip)
        --dry-run           Print what would be done without executing
        --user-id USER_ID   Override user_id from export (use string ID)
        --limit N           Only process first N memories
        --verbose           Show detailed progress

Environment Variables Required:
    NEO4J_URL         Neo4j bolt URL (e.g., bolt://localhost:7687)
    NEO4J_USERNAME    Neo4j username
    NEO4J_PASSWORD    Neo4j password
    NEO4J_DATABASE    Neo4j database name (optional, default: neo4j)

Example:
    export NEO4J_URL=bolt://localhost:7687
    export NEO4J_USERNAME=neo4j
    export NEO4J_PASSWORD=your-password

    python -m app.scripts.backfill_neo4j_metadata \\
        "/Users/grischadallmer/Downloads/memories_export (19)/memories.json"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
script_dir = Path(__file__).parent
api_dir = script_dir.parent.parent
sys.path.insert(0, str(api_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_export_file(filepath: str) -> dict:
    """Load and parse the OpenMemory export JSON file."""
    logger.info(f"Loading export file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Found {len(data.get('memories', []))} memories in export")
    return data


def get_user_id_from_export(data: dict) -> Optional[str]:
    """Extract the string user_id from the export."""
    user = data.get('user', {})
    return user.get('user_id')  # This is the string ID like "grischadallmer"


def should_process_memory(memory: dict, include_deleted: bool) -> bool:
    """Check if a memory should be processed."""
    state = memory.get('state', 'active')
    if state == 'deleted' and not include_deleted:
        return False
    return True


def backfill_memories(
    data: dict,
    user_id_override: Optional[str] = None,
    include_deleted: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Backfill Neo4j with memories from export.

    Returns:
        Stats dict with counts of processed/skipped/failed memories
    """
    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session
    from app.graph.metadata_projector import MetadataProjector, MemoryMetadata

    if not is_neo4j_configured():
        logger.error("Neo4j is not configured. Set NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD environment variables.")
        sys.exit(1)

    # Determine user_id
    user_id = user_id_override or get_user_id_from_export(data)
    if not user_id:
        logger.error("Could not determine user_id. Use --user-id option.")
        sys.exit(1)

    logger.info(f"Using user_id: {user_id}")

    # Get memories, sorted by created_at for consistent ordering
    memories = data.get('memories', [])
    memories = sorted(memories, key=lambda m: m.get('created_at', ''))

    if limit:
        memories = memories[:limit]
        logger.info(f"Limited to {limit} memories")

    stats = {
        'total': len(memories),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
    }

    if dry_run:
        logger.info("DRY RUN mode - no changes will be made")

    # Initialize projector
    projector = MetadataProjector(get_neo4j_session)

    if not dry_run:
        logger.info("Ensuring Neo4j constraints...")
        if not projector.ensure_constraints():
            logger.error("Failed to create Neo4j constraints")
            sys.exit(1)

    logger.info(f"Processing {len(memories)} memories...")

    for i, memory in enumerate(memories, 1):
        memory_id = memory.get('id')

        if not memory_id:
            logger.warning(f"Memory at index {i-1} has no ID, skipping")
            stats['skipped'] += 1
            continue

        if not should_process_memory(memory, include_deleted):
            if verbose:
                logger.debug(f"Skipping deleted memory {memory_id}")
            stats['skipped'] += 1
            continue

        try:
            # Build metadata from export format
            memory_metadata = MemoryMetadata.from_dict(
                data=memory,
                memory_id=memory_id,
                user_id=user_id,
            )

            if dry_run:
                if verbose:
                    logger.info(f"[DRY RUN] Would project memory {memory_id}")
                    logger.info(f"  vault={memory_metadata.vault}, layer={memory_metadata.layer}")
                    logger.info(f"  entity={memory_metadata.entity}, tags={list(memory_metadata.tags.keys())}")
            else:
                success = projector.upsert_memory(memory_metadata)
                if success:
                    stats['processed'] += 1
                else:
                    logger.warning(f"Failed to project memory {memory_id}")
                    stats['failed'] += 1

            if not dry_run and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(memories)} memories processed")

        except Exception as e:
            logger.error(f"Error processing memory {memory_id}: {e}")
            stats['failed'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Neo4j metadata graph from OpenMemory export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "export_file",
        help="Path to memories.json export file",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include deleted memories (default: skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--user-id",
        help="Override user_id from export",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process first N memories",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check export file exists
    if not os.path.exists(args.export_file):
        logger.error(f"Export file not found: {args.export_file}")
        sys.exit(1)

    # Load export
    data = load_export_file(args.export_file)

    # Run backfill
    stats = backfill_memories(
        data=data,
        user_id_override=args.user_id,
        include_deleted=args.include_deleted,
        dry_run=args.dry_run,
        limit=args.limit,
        verbose=args.verbose,
    )

    # Print summary
    logger.info("=" * 50)
    logger.info("Backfill complete!")
    logger.info(f"  Total:     {stats['total']}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Skipped:   {stats['skipped']}")
    logger.info(f"  Failed:    {stats['failed']}")

    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
