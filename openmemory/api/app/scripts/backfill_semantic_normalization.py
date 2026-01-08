#!/usr/bin/env python
"""
Semantic Entity Normalization Backfill Script.

Runs the full semantic entity normalization pipeline:
1. Collect all entities for a user
2. Detect semantic duplicates using multiple methods
3. Cluster duplicates and choose canonical forms
4. Migrate edges and sync graphs

Usage:
    # Dry run (preview only)
    python -m app.scripts.backfill_semantic_normalization --user-id USER_ID --dry-run

    # Execute normalization
    python -m app.scripts.backfill_semantic_normalization --user-id USER_ID --execute

    # With specific phases
    python -m app.scripts.backfill_semantic_normalization --user-id USER_ID --execute --phases string prefix domain

    # Skip Mem0 sync
    python -m app.scripts.backfill_semantic_normalization --user-id USER_ID --execute --skip-mem0-sync

    # Lower threshold for more matches
    python -m app.scripts.backfill_semantic_normalization --user-id USER_ID --execute --threshold 0.75
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.graph.neo4j_client import is_neo4j_configured, is_neo4j_healthy
from app.graph.semantic_entity_normalizer import (
    SemanticEntityNormalizer,
    get_all_user_entities,
)
from app.graph.gds_signal_refresh import refresh_all_signals_for_user


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("semantic_normalization")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


async def run_normalization(
    user_id: str,
    dry_run: bool = True,
    phases: Optional[List[str]] = None,
    threshold: float = 0.7,
    access_entity: Optional[str] = None,
    skip_mem0_sync: bool = False,
    skip_gds_refresh: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Run the semantic normalization pipeline.

    Args:
        user_id: User ID to normalize
        dry_run: If True, only preview changes
        phases: Which phases to run (string, prefix, domain, embedding)
        threshold: Minimum confidence for merge (0.0-1.0)
        skip_mem0_sync: Skip Mem0 __Entity__ sync
        skip_gds_refresh: Skip GDS signal refresh
        logger: Optional logger instance

    Returns:
        Summary statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Check Neo4j availability
    if not is_neo4j_configured():
        logger.error("Neo4j is not configured")
        return {"error": "Neo4j not configured"}

    if not is_neo4j_healthy():
        logger.error("Neo4j is not healthy")
        return {"error": "Neo4j unhealthy"}

    access_entity = access_entity or f"user:{user_id}"
    logger.info(f"Starting semantic normalization for user: {user_id}")
    logger.info(f"Access entity: {access_entity}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    logger.info(f"Threshold: {threshold}")

    # 1. Collect all entities
    logger.info("Phase 0: Collecting entities...")
    entities = await get_all_user_entities(
        user_id,
        access_entities=[access_entity],
        access_entity_prefixes=None,
    )
    logger.info(f"Found {len(entities)} unique entities")

    if len(entities) < 2:
        logger.info("Less than 2 entities, nothing to normalize")
        return {"entities": len(entities), "merges": 0}

    # 2. Initialize normalizer with optional phases
    normalizer = SemanticEntityNormalizer(
        embedding_model=None,  # Embeddings disabled by default
        enable_embeddings=False,
    )

    # Adjust threshold
    normalizer.MERGE_CONFIDENCE_THRESHOLD = threshold

    # 3. Find merge candidates
    logger.info("Finding merge candidates...")
    candidates = await normalizer.find_merge_candidates(entities)
    logger.info(f"Found {len(candidates)} merge candidates")

    if not candidates:
        logger.info("No merge candidates found")
        return {"entities": len(entities), "candidates": 0, "merges": 0}

    # 4. Cluster candidates
    logger.info("Clustering candidates...")
    groups = normalizer.cluster_candidates(candidates)
    logger.info(f"Created {len(groups)} merge groups")

    # 5. Preview or execute merges
    results = {
        "entities_scanned": len(entities),
        "candidates_found": len(candidates),
        "merge_groups": len(groups),
        "dry_run": dry_run,
        "merges": [],
        "total_edges_migrated": 0,
        "total_variants_merged": 0,
    }

    for i, group in enumerate(groups, 1):
        logger.info(f"\n=== Merge Group {i}/{len(groups)} ===")
        logger.info(f"Canonical: {group.canonical}")
        logger.info(f"Variants: {group.variants}")
        logger.info(f"Confidence: {group.confidence:.2f}")
        logger.info(f"Sources: {group.merge_sources}")

        if dry_run:
            # Preview only
            from app.graph.entity_edge_migrator import estimate_migration_impact

            impact = await estimate_migration_impact(
                user_id=user_id,
                canonical=group.canonical,
                variants=group.variants,
                access_entity=access_entity,
            )

            results["merges"].append({
                "canonical": group.canonical,
                "variants": group.variants,
                "confidence": group.confidence,
                "estimated_impact": impact["estimated_changes"],
            })

            results["total_edges_migrated"] += impact["estimated_changes"]["total_edges"]
            results["total_variants_merged"] += len(group.variants)

            logger.info(f"Estimated impact: {impact['estimated_changes']}")
        else:
            # Execute merge
            merge_result = await normalizer.execute_merge(
                user_id=user_id,
                group=group,
                allowed_memory_ids=None,  # No ACL restriction
                dry_run=False,
                access_entity=access_entity,
            )

            results["merges"].append(merge_result)

            if merge_result.get("edge_migration"):
                results["total_edges_migrated"] += merge_result["edge_migration"]["total_migrated"]
            results["total_variants_merged"] += len(group.variants)

            logger.info(f"Merge result: {merge_result}")

    # 6. Post-normalization tasks (only if not dry run)
    if not dry_run:
        # Refresh GDS signals
        if not skip_gds_refresh:
            logger.info("\n=== Refreshing Graph Signals ===")
            if access_entity.startswith("user:"):
                gds_stats = await refresh_all_signals_for_user(user_id)
            else:
                from app.graph.gds_signal_refresh import refresh_graph_signals
                gds_stats = await refresh_graph_signals(
                    user_id,
                    access_entities=[access_entity],
                )
            results["gds_refresh"] = gds_stats
            logger.info(f"GDS refresh: {gds_stats}")

    # Summary
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Entities scanned: {results['entities_scanned']}")
    logger.info(f"Candidates found: {results['candidates_found']}")
    logger.info(f"Merge groups: {results['merge_groups']}")
    logger.info(f"Total variants merged: {results['total_variants_merged']}")
    logger.info(f"Total edges migrated: {results['total_edges_migrated']}")
    logger.info(f"Dry run: {results['dry_run']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Entity Normalization Backfill"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to normalize entities for"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute normalization (required if not dry-run)"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["string", "prefix", "domain", "embedding"],
        help="Which detection phases to run"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for merge (default: 0.7)"
    )
    parser.add_argument(
        "--access-entity",
        default=None,
        help="Access entity scope (default: user scope)"
    )
    parser.add_argument(
        "--skip-mem0-sync",
        action="store_true",
        help="Skip Mem0 __Entity__ synchronization"
    )
    parser.add_argument(
        "--skip-gds-refresh",
        action="store_true",
        help="Skip GDS signal refresh"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )

    args = parser.parse_args()

    # Validate args
    if not args.dry_run and not args.execute:
        parser.error("Must specify either --dry-run or --execute")

    if args.dry_run and args.execute:
        parser.error("Cannot specify both --dry-run and --execute")

    # Set up logging
    if not args.log_file:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = os.path.join(
            log_dir,
            f"semantic_normalization_{timestamp}.log"
        )

    logger = setup_logging(args.log_file)
    logger.info(f"Log file: {args.log_file}")

    # Run normalization
    try:
        results = asyncio.run(run_normalization(
            user_id=args.user_id,
            dry_run=args.dry_run,
            phases=args.phases,
            threshold=args.threshold,
            access_entity=args.access_entity,
            skip_mem0_sync=args.skip_mem0_sync,
            skip_gds_refresh=args.skip_gds_refresh,
            logger=logger,
        ))

        # Print results summary
        print("\n" + "=" * 60)
        print("SEMANTIC ENTITY NORMALIZATION RESULTS")
        print("=" * 60)
        print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTED'}")
        print(f"User: {args.user_id}")
        print(f"Entities scanned: {results.get('entities_scanned', 0)}")
        print(f"Merge groups: {results.get('merge_groups', 0)}")
        print(f"Variants merged: {results.get('total_variants_merged', 0)}")
        print(f"Edges migrated: {results.get('total_edges_migrated', 0)}")

        if not args.dry_run:
            print("\nChanges have been applied to the graph.")
        else:
            print("\nThis was a DRY RUN. Use --execute to apply changes.")

    except Exception as e:
        logger.exception(f"Error during normalization: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
