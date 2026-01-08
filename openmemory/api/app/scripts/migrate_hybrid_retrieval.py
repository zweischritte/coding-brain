#!/usr/bin/env python3
"""
Migration script for Hybrid Retrieval Neo4j properties.

This script adds the required schema properties and indexes for
hybrid vector+graph retrieval:

1. OM_Memory.similarityClusterSize - Count of OM_SIMILAR edges
2. OM_Entity.pageRank - PageRank score from GDS
3. OM_Entity.degree - Count of OM_CO_MENTIONED edges
4. Fulltext index on OM_Entity.name for entity detection

Usage:
    # From the openmemory-api container:
    python -m app.scripts.migrate_hybrid_retrieval --user-id YOUR_USER_ID

    # Or run without user filter (all users):
    python -m app.scripts.migrate_hybrid_retrieval --all-users

    # Dry run (show what would be done):
    python -m app.scripts.migrate_hybrid_retrieval --dry-run
"""

import argparse
import logging
import sys
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_neo4j_connection() -> bool:
    """Verify Neo4j is available and connected."""
    try:
        from app.graph.neo4j_client import (
            is_neo4j_configured, get_neo4j_driver, is_neo4j_healthy
        )

        if not is_neo4j_configured():
            logger.error("Neo4j not configured. Set NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD")
            return False

        if not is_neo4j_healthy():
            logger.error("Neo4j not healthy. Check connection.")
            return False

        driver = get_neo4j_driver()
        if driver is None:
            logger.error("Failed to get Neo4j driver")
            return False

        logger.info("Neo4j connection verified")
        return True

    except Exception as e:
        logger.error(f"Neo4j connection check failed: {e}")
        return False


def create_fulltext_index(dry_run: bool = False) -> Dict[str, Any]:
    """
    Create fulltext index on OM_Entity.name for entity detection.

    This index enables fast (~5-20ms) entity detection in queries.
    """
    from app.graph.neo4j_client import get_neo4j_session

    index_name = "om_entity_name_fulltext"

    try:
        with get_neo4j_session() as session:
            # Check if index exists
            # Note: Neo4j Community uses SHOW INDEXES YIELD name, then filter
            check_query = """
            SHOW INDEXES YIELD name
            WHERE name = $indexName
            RETURN name
            """
            result = session.run(check_query, indexName=index_name)
            exists = result.single() is not None

            if exists:
                logger.info(f"Index '{index_name}' already exists")
                return {"success": True, "action": "already_exists"}

            if dry_run:
                logger.info(f"[DRY RUN] Would create index: {index_name}")
                return {"success": True, "action": "would_create", "dry_run": True}

            # Create fulltext index
            create_query = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (e:OM_Entity) ON EACH [e.name]
            """
            session.run(create_query)
            logger.info(f"Created fulltext index: {index_name}")
            return {"success": True, "action": "created"}

    except Exception as e:
        logger.error(f"Failed to create fulltext index: {e}")
        return {"success": False, "error": str(e)}


def refresh_cluster_sizes(
    user_id: Optional[str],
    dry_run: bool = False,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refresh similarityClusterSize on OM_Memory nodes.

    Counts outbound OM_SIMILAR edges for each memory.
    """
    from app.graph.neo4j_client import get_neo4j_session

    try:
        with get_neo4j_session() as session:
            # First, check how many need updating
            count_query = """
            MATCH (m:OM_Memory)
            WHERE ($accessEntity IS NULL OR coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity)
            RETURN count(m) AS total
            """
            result = session.run(
                count_query,
                userId=user_id,
                accessEntity=access_entity,
                legacyAccessEntity=f"user:{user_id}" if user_id else None,
            )
            record = result.single()
            total = record["total"] if record else 0

            if dry_run:
                logger.info(f"[DRY RUN] Would update similarityClusterSize on {total} memories")
                return {"success": True, "would_update": total, "dry_run": True}

            # Update cluster sizes
            update_query = """
            MATCH (m:OM_Memory)
            WHERE ($accessEntity IS NULL OR coalesce(m.accessEntity, $legacyAccessEntity) = $accessEntity)
            OPTIONAL MATCH (m)-[r:OM_SIMILAR]->()
            WITH m, count(r) AS clusterSize
            SET m.similarityClusterSize = clusterSize
            RETURN count(m) AS updated
            """
            result = session.run(
                update_query,
                userId=user_id,
                accessEntity=access_entity,
                legacyAccessEntity=f"user:{user_id}" if user_id else None,
            )
            record = result.single()
            updated = record["updated"] if record else 0

            logger.info(f"Updated similarityClusterSize on {updated} memories")
            return {"success": True, "memories_updated": updated}

    except Exception as e:
        logger.error(f"Failed to refresh cluster sizes: {e}")
        return {"success": False, "error": str(e)}


def refresh_entity_degrees(
    user_id: Optional[str],
    dry_run: bool = False,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Refresh degree property on OM_Entity nodes.

    Counts OM_CO_MENTIONED edges for each entity.
    """
    from app.graph.neo4j_client import get_neo4j_session

    try:
        with get_neo4j_session() as session:
            # Count entities
            count_query = """
            MATCH (e:OM_Entity)
            WHERE ($accessEntity IS NULL OR coalesce(e.accessEntity, $legacyAccessEntity) = $accessEntity)
            RETURN count(e) AS total
            """
            result = session.run(
                count_query,
                userId=user_id,
                accessEntity=access_entity,
                legacyAccessEntity=f"user:{user_id}" if user_id else None,
            )
            record = result.single()
            total = record["total"] if record else 0

            if dry_run:
                logger.info(f"[DRY RUN] Would update degree on {total} entities")
                return {"success": True, "would_update": total, "dry_run": True}

            # Update degrees
            update_query = """
            MATCH (e:OM_Entity)
            WHERE ($accessEntity IS NULL OR coalesce(e.accessEntity, $legacyAccessEntity) = $accessEntity)
            OPTIONAL MATCH (e)-[r:OM_CO_MENTIONED]-()
            WHERE ($accessEntity IS NULL OR coalesce(r.accessEntity, $legacyAccessEntity) = $accessEntity)
            WITH e, count(r) AS degree
            SET e.degree = degree
            RETURN count(e) AS updated
            """
            result = session.run(
                update_query,
                userId=user_id,
                accessEntity=access_entity,
                legacyAccessEntity=f"user:{user_id}" if user_id else None,
            )
            record = result.single()
            updated = record["updated"] if record else 0

            logger.info(f"Updated degree on {updated} entities")
            return {"success": True, "entities_updated": updated}

    except Exception as e:
        logger.error(f"Failed to refresh entity degrees: {e}")
        return {"success": False, "error": str(e)}


def refresh_pagerank(
    user_id: Optional[str],
    dry_run: bool = False,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute and store PageRank on OM_Entity nodes.

    Uses Neo4j GDS if available, falls back to degree centrality.
    """
    try:
        from app.graph.gds_operations import entity_pagerank

        if dry_run:
            logger.info("[DRY RUN] Would compute PageRank for entities")
            return {"success": True, "action": "would_compute", "dry_run": True}

        access_entities = [access_entity] if access_entity else None
        result = entity_pagerank(
            user_id=user_id,
            write_to_nodes=True,
            limit=10000,
            access_entities=access_entities,
        )
        updated = len(result) if result else 0

        logger.info(f"Computed PageRank for {updated} entities")
        return {"success": True, "entities_updated": updated}

    except ImportError:
        logger.warning("GDS operations not available, skipping PageRank")
        return {"success": True, "action": "skipped", "reason": "GDS not available"}
    except Exception as e:
        logger.error(f"Failed to compute PageRank: {e}")
        return {"success": False, "error": str(e)}


def run_migration(
    user_id: Optional[str] = None,
    dry_run: bool = False,
    access_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete hybrid retrieval migration.

    Steps:
    1. Create fulltext index for entity detection
    2. Refresh similarityClusterSize on memories
    3. Refresh degree on entities
    4. Compute PageRank on entities (if GDS available)
    """
    access_entity = access_entity or (f"user:{user_id}" if user_id else None)
    results = {
        "user_id": user_id or "all",
        "access_entity": access_entity,
        "dry_run": dry_run,
        "steps": {}
    }

    # Step 1: Create fulltext index
    logger.info("Step 1/4: Creating fulltext index for entity detection...")
    results["steps"]["fulltext_index"] = create_fulltext_index(dry_run)

    # Step 2: Refresh cluster sizes
    logger.info("Step 2/4: Refreshing memory similarity cluster sizes...")
    results["steps"]["cluster_sizes"] = refresh_cluster_sizes(
        user_id,
        dry_run,
        access_entity=access_entity,
    )

    # Step 3: Refresh entity degrees
    logger.info("Step 3/4: Refreshing entity degrees...")
    results["steps"]["entity_degrees"] = refresh_entity_degrees(
        user_id,
        dry_run,
        access_entity=access_entity,
    )

    # Step 4: Compute PageRank
    logger.info("Step 4/4: Computing entity PageRank...")
    results["steps"]["pagerank"] = refresh_pagerank(
        user_id,
        dry_run,
        access_entity=access_entity,
    )

    # Summary
    all_success = all(
        step.get("success", False)
        for step in results["steps"].values()
    )
    results["all_success"] = all_success

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Neo4j schema for hybrid retrieval"
    )
    parser.add_argument(
        "--user-id",
        help="User ID to migrate (optional, defaults to all users)"
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Migrate all users (explicit flag)"
    )
    parser.add_argument(
        "--access-entity",
        default=None,
        help="Access entity scope (default: user scope when --user-id is set)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.user_id and args.all_users:
        logger.error("Cannot specify both --user-id and --all-users")
        sys.exit(1)
    if args.access_entity and args.all_users:
        logger.error("Cannot specify --access-entity with --all-users")
        sys.exit(1)
    if args.access_entity and not args.user_id:
        logger.error("--access-entity requires --user-id for legacy fallback")
        sys.exit(1)

    user_id = args.user_id if args.user_id else None

    if not args.user_id and not args.all_users:
        logger.warning("No --user-id specified and --all-users not set. Use --all-users to confirm.")
        sys.exit(1)

    # Check connection
    if not check_neo4j_connection():
        sys.exit(1)

    # Run migration
    logger.info(f"Starting hybrid retrieval migration...")
    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")

    results = run_migration(
        user_id=user_id,
        dry_run=args.dry_run,
        access_entity=args.access_entity,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Migration Summary:")
    for step_name, step_result in results["steps"].items():
        status = "✓" if step_result.get("success") else "✗"
        logger.info(f"  {status} {step_name}: {step_result}")

    if results["all_success"]:
        logger.info("Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("Migration completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
