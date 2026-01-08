#!/usr/bin/env python3
"""
Backfill entity bridge per access_entity with resume support.

This script processes existing memories and:
1. Extracts entities from memory content using Mem0's LLM
2. Creates OM_ABOUT edges for each extracted entity (multi-entity per memory)
3. Creates OM_RELATION edges with typed relationships
4. Updates OM_CO_MENTIONED edges

It runs per access_entity and persists a JSON state file so the process can
resume after interruption without double-counting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import UUID


logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"backfill_entity_bridge_{ts}.log")


def _default_state_file() -> str:
    return str(Path("logs") / "backfill_entity_bridge_state.json")


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


def _load_state(path: str, *, reset: bool) -> Dict[str, Any]:
    if not reset and Path(path).exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "version": 1,
        "started_at": _utcnow(),
        "updated_at": _utcnow(),
        "access_entities": {},
    }


def _save_state(state: Dict[str, Any], path: str) -> None:
    state["updated_at"] = _utcnow()
    tmp_path = f"{path}.tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _apply_host_ports_if_requested(use_host_ports: bool) -> None:
    if not use_host_ports:
        return

    postgres_user = os.getenv("POSTGRES_USER") or "codingbrain"
    postgres_password = os.getenv("POSTGRES_PASSWORD") or ""
    postgres_db = os.getenv("POSTGRES_DB") or "codingbrain"

    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5532"
    os.environ["DATABASE_URL"] = (
        f"postgresql://{postgres_user}:{postgres_password}@localhost:5532/{postgres_db}"
    )

    os.environ["NEO4J_URL"] = "bolt://localhost:7787"
    os.environ["QDRANT_HOST"] = "localhost"
    os.environ["QDRANT_PORT"] = "6433"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _resolve_access_entities(db, raw_access_entities: Optional[str]) -> list[str]:
    from app.models import Memory, User

    if raw_access_entities:
        return [value.strip() for value in raw_access_entities.split(",") if value.strip()]

    access_entity_col = Memory.metadata_["access_entity"].as_string()
    rows = (
        db.query(access_entity_col)
        .distinct()
        .order_by(access_entity_col.asc())
        .all()
    )

    access_entities = {row[0] for row in rows if row[0]}

    missing_user_rows = (
        db.query(User.user_id)
        .join(Memory, Memory.user_id == User.id)
        .filter((access_entity_col.is_(None)) | (access_entity_col == ""))
        .distinct()
        .all()
    )
    for (user_id,) in missing_user_rows:
        access_entities.add(f"user:{user_id}")

    return sorted(access_entities)


def _iter_memories(
    db,
    *,
    access_entity: str,
    last_created_at: Optional[str],
    last_id: Optional[str],
    limit: Optional[int],
) -> Iterable[Tuple[Any, str]]:
    from sqlalchemy import and_, or_
    from app.models import Memory, MemoryState, User

    access_entity_col = Memory.metadata_["access_entity"].as_string()

    query = db.query(Memory, User.user_id).join(User, Memory.user_id == User.id)
    query = query.filter(Memory.state == MemoryState.active)

    if access_entity.startswith("user:"):
        user_id_value = access_entity.split("user:", 1)[1]
        user = db.query(User).filter(User.user_id == user_id_value).first()
        if not user:
            logger.warning(f"Skipping {access_entity}: user not found")
            return []
        query = query.filter(
            Memory.user_id == user.id,
            or_(
                access_entity_col == access_entity,
                access_entity_col.is_(None),
                access_entity_col == "",
            ),
        )
    else:
        query = query.filter(access_entity_col == access_entity)

    query = query.order_by(Memory.created_at.asc(), Memory.id.asc())

    if last_created_at and last_id:
        last_dt = datetime.fromisoformat(last_created_at)
        last_uuid = UUID(last_id)
        query = query.filter(
            or_(
                Memory.created_at > last_dt,
                and_(Memory.created_at == last_dt, Memory.id > last_uuid),
            )
        )

    if limit:
        query = query.limit(limit)

    return query


def backfill_entity_bridge(
    *,
    user_id: str,
    access_entities: Optional[str],
    limit_per_access: Optional[int],
    batch_size: int,
    dry_run: bool,
    log_file: Optional[str],
    verbose: bool,
    state_file: Optional[str],
    reset_state: bool,
    use_host_ports: bool,
) -> tuple[Dict[str, Any], str, str]:
    _load_dotenv()
    _apply_host_ports_if_requested(use_host_ports)

    from app.database import SessionLocal
    from app.graph.entity_bridge import (
        bridge_entities_to_om_graph,
        extract_entities_from_content,
    )
    from app.graph.graph_ops import is_mem0_graph_enabled
    from app.graph.neo4j_client import is_neo4j_configured

    chosen_log_file = log_file or _default_log_file()
    chosen_state_file = state_file or _default_state_file()

    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting entity bridge backfill (per access_entity)")
    logger.info(
        f"user_id={user_id} limit_per_access={limit_per_access} "
        f"batch_size={batch_size} dry_run={dry_run}"
    )
    logger.info(f"log_file={chosen_log_file}")
    logger.info(f"state_file={chosen_state_file}")

    if not is_neo4j_configured():
        raise SystemExit("Neo4j is not configured")
    if not is_mem0_graph_enabled():
        raise SystemExit("Mem0 Graph Memory is not enabled")

    state = _load_state(chosen_state_file, reset=reset_state)

    db = SessionLocal()
    try:
        resolved_access_entities = _resolve_access_entities(db, access_entities)
        if not resolved_access_entities:
            logger.warning("No access_entity values found")
            return state, chosen_log_file, chosen_state_file

        for access_entity in resolved_access_entities:
            entry = state["access_entities"].setdefault(
                access_entity,
                {
                    "completed": False,
                    "last_created_at": None,
                    "last_id": None,
                    "processed": 0,
                    "skipped": 0,
                    "failed": 0,
                    "failed_ids": [],
                },
            )
            if entry.get("completed"):
                logger.info(f"Skipping {access_entity}: already completed")
                continue

            last_created_at = entry.get("last_created_at")
            last_id = entry.get("last_id")

            query = _iter_memories(
                db,
                access_entity=access_entity,
                last_created_at=last_created_at,
                last_id=last_id,
                limit=limit_per_access,
            )
            total_count = None
            try:
                total_count = query.count()
            except Exception:
                total_count = None

            logger.info(
                f"Processing access_entity={access_entity} "
                f"remaining={total_count if total_count is not None else 'unknown'}"
            )

            start_time = time.time()
            processed_in_access = 0

            for idx, (memory, creator_user_id) in enumerate(query, 1):
                memory_id = str(memory.id)
                effective_user_id = creator_user_id or user_id

                try:
                    content = memory.content or ""
                    if len(content.strip()) < 10:
                        entry["skipped"] += 1
                    else:
                        if dry_run:
                            entities, relations = extract_entities_from_content(content, effective_user_id)
                            entry["processed"] += 1
                            if entities or relations:
                                logger.info(
                                    f"[DRY RUN] {memory_id}: "
                                    f"{len(entities)} entities, {len(relations)} relations"
                                )
                        else:
                            existing_entity = None
                            if memory.metadata_:
                                existing_entity = memory.metadata_.get("re") or memory.metadata_.get("entity")

                            result = bridge_entities_to_om_graph(
                                memory_id=memory_id,
                                user_id=effective_user_id,
                                content=content,
                                existing_entity=existing_entity,
                            )

                            if result.get("error"):
                                entry["failed"] += 1
                                entry["failed_ids"].append(memory_id)
                                logger.warning(f"FAIL memory_id={memory_id}: {result['error']}")
                            else:
                                entry["processed"] += 1

                except Exception as exc:
                    entry["failed"] += 1
                    entry["failed_ids"].append(memory_id)
                    logger.exception(f"FAIL memory_id={memory_id}: {exc}")

                processed_in_access += 1
                entry["last_created_at"] = (
                    memory.created_at.isoformat() if memory.created_at else entry.get("last_created_at")
                )
                entry["last_id"] = memory_id

                if processed_in_access % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = processed_in_access / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress {access_entity}: "
                        f"{processed_in_access} "
                        f"{'(of ' + str(total_count) + ')' if total_count is not None else ''} "
                        f"{rate:.1f} mem/s"
                    )
                    _save_state(state, chosen_state_file)

            entry["completed"] = True
            _save_state(state, chosen_state_file)

    finally:
        db.close()

    return state, chosen_log_file, chosen_state_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill entity bridge per access_entity with resume support"
    )
    parser.add_argument("--user-id", required=True, help="User ID to backfill")
    parser.add_argument("--access-entities", default=None, help="Comma-separated access_entity list")
    parser.add_argument("--limit-per-access", type=int, default=None, help="Limit per access_entity")
    parser.add_argument("--batch-size", type=int, default=10, help="Progress log interval")
    parser.add_argument("--dry-run", action="store_true", help="Extract but don't write")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--state-file", default=None, help="State file path")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing state file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--use-host-ports",
        action="store_true",
        help="Use localhost-mapped ports for Postgres/Neo4j/Qdrant",
    )
    args = parser.parse_args()

    state, log_file, state_file = backfill_entity_bridge(
        user_id=args.user_id,
        access_entities=args.access_entities,
        limit_per_access=args.limit_per_access,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        log_file=args.log_file,
        verbose=args.verbose,
        state_file=args.state_file,
        reset_state=args.reset_state,
        use_host_ports=args.use_host_ports,
    )

    print(
        f"Entity bridge backfill complete. "
        f"log_file={log_file} "
        f"state_file={state_file}"
    )


if __name__ == "__main__":
    main()
