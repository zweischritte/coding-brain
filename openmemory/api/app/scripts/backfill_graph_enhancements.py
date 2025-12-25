"""
Backfill Script für Graph-Erweiterungen.

Führt aus:
1. Entity Normalization (Duplikate zusammenführen)
2. Relation Extraction aus existierenden Memories
3. Temporal Event Creation aus date-tagged Memories

Usage:
    python -m app.scripts.backfill_graph_enhancements --user-id grischadallmer --dry-run
    python -m app.scripts.backfill_graph_enhancements --user-id grischadallmer --execute
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def normalize_entities(user_id: str, dry_run: bool) -> Dict[str, Any]:
    """Phase 1: Entity Normalization."""
    logger.info("=== Phase 1: Entity Normalization ===")

    from app.graph.entity_normalizer import auto_normalize_entities, identify_duplicates

    # First show what we found
    duplicates = identify_duplicates(user_id)
    logger.info(f"Found {len(duplicates)} duplicate groups")

    for dup in duplicates[:10]:  # Show first 10
        variants_str = ", ".join(f"{v.name}({v.memory_count})" for v in dup.variants)
        logger.info(f"  → {dup.canonical} ← [{variants_str}]")

    if len(duplicates) > 10:
        logger.info(f"  ... and {len(duplicates) - 10} more")

    # Perform normalization
    stats = auto_normalize_entities(user_id, dry_run=dry_run)

    logger.info(f"Normalization results:")
    logger.info(f"  - Groups merged: {len(stats.get('merges', []))}")
    logger.info(f"  - OM_ABOUT migrated: {stats.get('total_about_migrated', 0)}")
    logger.info(f"  - OM_CO_MENTIONED migrated: {stats.get('total_co_mention_migrated', 0)}")
    logger.info(f"  - OM_RELATION migrated: {stats.get('total_relation_migrated', 0)}")
    logger.info(f"  - Nodes deleted: {stats.get('total_nodes_deleted', 0)}")
    logger.info(f"  - Dry run: {dry_run}")

    return stats


async def extract_relations(user_id: str, dry_run: bool) -> Dict[str, Any]:
    """Phase 2: Extract typed relations from existing memories."""
    logger.info("=== Phase 2: Relation Extraction ===")

    from app.database import SessionLocal
    from app.models import Memory, MemoryState

    db = SessionLocal()
    stats = {
        "memories_processed": 0,
        "entities_bridged": 0,
        "relations_created": 0,
        "errors": 0,
    }

    try:
        # Get user by string ID
        from app.models import User
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            logger.error(f"User {user_id} not found")
            return stats

        # Get all active memories
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state == MemoryState.active,
        ).all()

        logger.info(f"Processing {len(memories)} memories...")

        # Try to import entity_bridge
        try:
            from app.graph.entity_bridge import bridge_entities_for_existing_memory
            has_entity_bridge = True
        except ImportError:
            logger.warning("entity_bridge module not available, skipping relation extraction")
            has_entity_bridge = False

        if has_entity_bridge:
            for i, memory in enumerate(memories):
                if i > 0 and i % 50 == 0:
                    logger.info(f"  Progress: {i}/{len(memories)}")

                if dry_run:
                    stats["memories_processed"] += 1
                    continue

                try:
                    result = bridge_entities_for_existing_memory(
                        memory_id=str(memory.id),
                        user_id=user_id,
                    )

                    stats["memories_processed"] += 1
                    stats["entities_bridged"] += result.get("entities_bridged", 0)
                    stats["relations_created"] += result.get("relations_created", 0)

                except Exception as e:
                    logger.warning(f"Error processing memory {memory.id}: {e}")
                    stats["errors"] += 1
        else:
            stats["memories_processed"] = len(memories)
            logger.info("Skipping relation extraction (entity_bridge not available)")

        logger.info(f"Relation extraction results:")
        logger.info(f"  - Memories processed: {stats['memories_processed']}")
        logger.info(f"  - Entities bridged: {stats['entities_bridged']}")
        logger.info(f"  - Relations created: {stats['relations_created']}")
        logger.info(f"  - Errors: {stats['errors']}")
        logger.info(f"  - Dry run: {dry_run}")

    finally:
        db.close()

    return stats


async def extract_temporal_events(user_id: str, dry_run: bool) -> Dict[str, Any]:
    """Phase 3: Extract temporal events from memories with dates."""
    logger.info("=== Phase 3: Temporal Event Extraction ===")

    from app.database import SessionLocal
    from app.models import Memory, MemoryState
    from app.graph.temporal_events import (
        TemporalEvent, EventType, create_temporal_event, parse_date_from_text
    )

    db = SessionLocal()
    stats = {
        "memories_scanned": 0,
        "events_found": 0,
        "events_created": 0,
        "errors": 0,
    }

    try:
        from app.models import User
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            logger.error(f"User {user_id} not found")
            return stats

        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state == MemoryState.active,
        ).all()

        logger.info(f"Scanning {len(memories)} memories for temporal data...")

        # Keywords that suggest temporal events
        event_patterns = {
            EventType.RESIDENCE: ["wohnt", "wohnte", "lebt", "lebte", "zog nach", "umgezogen"],
            EventType.WORK: ["arbeitet", "arbeitete", "job", "firma", "beschäftigt"],
            EventType.PROJECT: ["film", "projekt", "produziert", "veröffentlicht", "premiere"],
            EventType.MILESTONE: ["geboren", "geburt", "hochzeit", "heirat", "gestorben"],
            EventType.TRAVEL: ["reise", "besuch", "aufenthalt", "war in"],
            EventType.EDUCATION: ["schule", "studium", "studiert", "abschluss", "ausbildung"],
        }

        for memory in memories:
            stats["memories_scanned"] += 1
            content = memory.content.lower() if memory.content else ""

            # Check for date pattern
            dates = parse_date_from_text(memory.content or "")
            if not dates:
                continue

            start_date, end_date = dates

            # Determine event type
            event_type = None
            for etype, keywords in event_patterns.items():
                if any(kw in content for kw in keywords):
                    event_type = etype
                    break

            if not event_type:
                event_type = EventType.MILESTONE

            # Create event name from content
            words = (memory.content or "").split()[:5]
            event_name = "_".join(words).lower()[:50]
            # Clean event name
            event_name = "".join(c if c.isalnum() or c == "_" else "_" for c in event_name)
            event_name = event_name.strip("_")

            if not event_name:
                continue

            stats["events_found"] += 1

            if not dry_run:
                entity = None
                if memory.metadata_:
                    entity = memory.metadata_.get("re")

                event = TemporalEvent(
                    name=event_name,
                    event_type=event_type,
                    start_date=start_date,
                    end_date=end_date,
                    description=memory.content[:200] if memory.content else None,
                    entity=entity,
                    memory_ids=[str(memory.id)],
                )

                try:
                    if create_temporal_event(user_id, event):
                        stats["events_created"] += 1
                except Exception as e:
                    logger.warning(f"Error creating event: {e}")
                    stats["errors"] += 1

        logger.info(f"Temporal extraction results:")
        logger.info(f"  - Memories scanned: {stats['memories_scanned']}")
        logger.info(f"  - Events found: {stats['events_found']}")
        logger.info(f"  - Events created: {stats['events_created']}")
        logger.info(f"  - Errors: {stats['errors']}")
        logger.info(f"  - Dry run: {dry_run}")

    finally:
        db.close()

    return stats


async def main(user_id: str, dry_run: bool, phases: list):
    """Main backfill execution."""
    start_time = datetime.now()
    logger.info(f"Starting graph enhancement backfill for user: {user_id}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Phases: {phases}")
    logger.info("=" * 60)

    all_stats = {}

    if "normalize" in phases or "all" in phases:
        all_stats["normalize"] = await normalize_entities(user_id, dry_run)
        logger.info("")

    if "relations" in phases or "all" in phases:
        all_stats["relations"] = await extract_relations(user_id, dry_run)
        logger.info("")

    if "temporal" in phases or "all" in phases:
        all_stats["temporal"] = await extract_temporal_events(user_id, dry_run)
        logger.info("")

    duration = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info(f"Backfill completed in {duration}")
    logger.info(f"Summary: {all_stats}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill graph enhancements")
    parser.add_argument("--user-id", required=True, help="User ID to process")
    parser.add_argument("--dry-run", action="store_true", help="Only simulate, don't make changes")
    parser.add_argument("--execute", action="store_true", help="Actually execute changes")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["all"],
        choices=["normalize", "relations", "temporal", "all"],
        help="Which phases to run"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: Must specify either --dry-run or --execute")
        sys.exit(1)

    dry_run = args.dry_run or not args.execute

    asyncio.run(main(args.user_id, dry_run, args.phases))
