#!/usr/bin/env python3
"""
Batch link memories to code via entity name matching.

This script finds memories that have an entity but no code_refs,
then attempts to link them to CODE_SYMBOL nodes in Neo4j.

Usage:
    python -m app.scripts.batch_link_memories [--limit 100] [--dry-run]

Environment Variables Required:
    NEO4J_URL         Neo4j bolt URL (e.g., bolt://localhost:7687)
    NEO4J_USERNAME    Neo4j username
    NEO4J_PASSWORD    Neo4j password

Schedule via cron for nightly execution:
    0 2 * * * cd /app && python -m app.scripts.batch_link_memories --limit 100
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent directories to path for imports
script_dir = Path(__file__).parent
api_dir = script_dir.parent.parent
sys.path.insert(0, str(api_dir))

from app.database import SessionLocal
from app.models import Memory, MemoryState
from app.graph.neo4j_client import get_neo4j_driver, is_neo4j_configured
from app.graph.evidence_linker import find_code_links_for_memory, CodeLink

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_memories_without_code_refs(db, limit: int = 100) -> list[dict]:
    """
    Find memories that have an entity but no code_refs.

    Args:
        db: SQLAlchemy session
        limit: Maximum number of memories to return

    Returns:
        List of memory dicts with id, text, metadata
    """
    # Query memories where:
    # 1. State is active
    # 2. metadata_ has an entity
    # 3. metadata_ does not have code_refs (or code_refs is empty)
    query = db.query(Memory).filter(
        Memory.state == MemoryState.active,
        Memory.metadata_.isnot(None),
    ).limit(limit * 10)  # Fetch more to filter in Python

    memories = []
    for memory in query:
        if memory.metadata_ is None:
            continue

        entity = memory.metadata_.get("entity")
        code_refs = memory.metadata_.get("code_refs", [])

        # Include if has entity but no code_refs
        if entity and not code_refs:
            memories.append({
                "id": str(memory.id),
                "text": memory.memory,
                "metadata": memory.metadata_,
            })

            if len(memories) >= limit:
                break

    return memories


async def batch_link_memories(
    limit: int = 100,
    dry_run: bool = False,
) -> int:
    """
    Main batch linking job.

    Finds memories without code_refs and attempts to link them
    to CODE_SYMBOL nodes via entity name matching.

    Args:
        limit: Maximum number of memories to process
        dry_run: If True, only log what would be linked

    Returns:
        Number of memories successfully linked
    """
    if not is_neo4j_configured():
        logger.error("Neo4j is not configured. Set NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD.")
        return 0

    neo4j_driver = get_neo4j_driver()
    if neo4j_driver is None:
        logger.error("Failed to get Neo4j driver")
        return 0

    db = SessionLocal()
    linked_count = 0

    try:
        memories = await get_memories_without_code_refs(db, limit)
        logger.info(f"Found {len(memories)} memories without code_refs")

        for memory in memories:
            entity = memory.get("metadata", {}).get("entity")
            if not entity:
                continue

            # Find code links for this memory
            links = await find_code_links_for_memory(memory, neo4j_driver)

            # Filter to only entity_match links (explicit already stored)
            new_links = [link for link in links if link.link_source == "entity_match"]

            if new_links:
                if dry_run:
                    symbol_names = [link.symbol_name for link in new_links]
                    logger.info(
                        f"[DRY RUN] Would link memory {memory['id']} "
                        f"to {len(new_links)} symbols: {symbol_names}"
                    )
                else:
                    # Convert CodeLink objects to code_refs format
                    code_refs = [link.to_dict() for link in new_links]

                    # Update memory in database
                    memory_obj = db.query(Memory).filter(
                        Memory.id == memory["id"]
                    ).first()

                    if memory_obj and memory_obj.metadata_:
                        # Merge new code_refs with existing metadata
                        updated_metadata = dict(memory_obj.metadata_)
                        updated_metadata["code_refs"] = code_refs
                        memory_obj.metadata_ = updated_metadata

                        logger.info(
                            f"Linked memory {memory['id']} to {len(new_links)} symbols"
                        )
                        linked_count += 1

        if not dry_run and linked_count > 0:
            db.commit()
            logger.info(f"Committed {linked_count} memory updates")

    except Exception as e:
        logger.error(f"Error during batch linking: {e}")
        db.rollback()
        raise
    finally:
        db.close()

    logger.info(f"Batch linking complete: {linked_count}/{len(memories)} memories linked")
    return linked_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch link memories to code via entity name matching"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of memories to process (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only log what would be linked, don't update database"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(batch_link_memories(limit=args.limit, dry_run=args.dry_run))

    print(json.dumps({
        "linked": result,
        "limit": args.limit,
        "dry_run": args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
