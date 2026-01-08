#!/usr/bin/env python3
"""
Hybrid graph backfill with resume support.

Steps:
1) Re-project metadata into Neo4j (OM_Memory + metadata relations).
2) Update OM_RELATION edges with accessEntity based on source memory.
3) Rebuild OM_CO_MENTIONED edges per access_entity.
4) Rebuild OM_COOCCURS edges per access_entity.
5) Rebuild OM_SIMILAR edges per access_entity (checkpointed per memory).

Resume state is written to a JSON file so the process can continue after interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import UUID


logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"backfill_graph_access_entity_hybrid_{ts}.log")


def _default_state_file() -> str:
    return str(Path("logs") / "backfill_graph_access_entity_hybrid_state.json")


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


def _load_state(path: str) -> Dict[str, Any]:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "version": 1,
        "started_at": _utcnow(),
        "updated_at": _utcnow(),
        "steps": {},
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


def _iter_memories(
    db,
    *,
    access_entity: Optional[str],
    user_id: str,
    last_created_at: Optional[str],
    last_id: Optional[str],
) -> Iterable:
    from sqlalchemy import and_, or_
    from app.models import Memory, MemoryState, User

    access_entity_col = Memory.metadata_["access_entity"].as_string()

    query = db.query(Memory)

    if access_entity and access_entity.startswith("user:"):
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise SystemExit(f"User not found: {user_id}")
        query = query.filter(
            Memory.user_id == user.id,
            Memory.state == MemoryState.active,
            or_(
                access_entity_col == access_entity,
                access_entity_col.is_(None),
                access_entity_col == "",
            ),
        )
    elif access_entity:
        query = query.filter(
            Memory.state == MemoryState.active,
            access_entity_col == access_entity,
        )
    else:
        query = query.filter(Memory.state == MemoryState.active)

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

    return query


def _resolve_access_entities(db, user_id: str, raw_access_entities: Optional[str]) -> list[str]:
    from app.models import Memory

    if raw_access_entities:
        return [value.strip() for value in raw_access_entities.split(",") if value.strip()]

    access_entity_col = Memory.metadata_["access_entity"].as_string()
    rows = (
        db.query(access_entity_col)
        .distinct()
        .order_by(access_entity_col.asc())
        .all()
    )

    access_entities = [row[0] for row in rows if row[0]]

    missing_count = (
        db.query(Memory)
        .filter((access_entity_col.is_(None)) | (access_entity_col == ""))
        .count()
    )
    if missing_count and f"user:{user_id}" not in access_entities:
        access_entities.append(f"user:{user_id}")

    return access_entities


def _backfill_metadata(
    *,
    db,
    projector,
    state: Dict[str, Any],
    state_file: str,
    checkpoint_every: int,
) -> None:
    from app.models import Memory, MemoryState, User
    from app.graph.metadata_projector import MemoryMetadata
    from sqlalchemy import and_, or_

    step_state = state.setdefault("steps", {}).setdefault("metadata_projection", {})
    if step_state.get("done"):
        logger.info("Skip metadata projection (already done).")
        return

    last_created_at = step_state.get("last_created_at")
    last_id = step_state.get("last_id")
    processed = step_state.get("processed", 0)
    failed = step_state.get("failed", 0)

    query = (
        db.query(Memory, User.user_id)
        .join(User, Memory.user_id == User.id)
        .filter(Memory.state == MemoryState.active)
        .order_by(Memory.created_at.asc(), Memory.id.asc())
    )

    if last_created_at and last_id:
        last_dt = datetime.fromisoformat(last_created_at)
        last_uuid = UUID(last_id)
        query = query.filter(
            or_(
                Memory.created_at > last_dt,
                and_(Memory.created_at == last_dt, Memory.id > last_uuid),
            )
        )

    total_remaining = query.count()
    logger.info("Metadata projection: %d memories remaining", total_remaining)

    for memory, user_id in query.yield_per(100):
        memory_id = str(memory.id)

        try:
            metadata = dict(memory.metadata_ or {})
            data = {
                "content": memory.content,
                "metadata": metadata,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "state": memory.state.value if memory.state else "active",
            }
            mm = MemoryMetadata.from_dict(data=data, memory_id=memory_id, user_id=user_id)
            ok = projector.upsert_memory(mm)
            if not ok:
                failed += 1
                logger.warning("Metadata projection failed: %s", memory_id)
            else:
                processed += 1

        except Exception as exc:
            failed += 1
            logger.warning("Metadata projection error for %s: %s", memory_id, exc)

        last_created_at = memory.created_at.isoformat() if memory.created_at else last_created_at
        last_id = memory_id

        if processed % checkpoint_every == 0:
            step_state.update(
                {
                    "last_created_at": last_created_at,
                    "last_id": last_id,
                    "processed": processed,
                    "failed": failed,
                }
            )
            _save_state(state, state_file)
            logger.info("Metadata projection checkpoint: processed=%d failed=%d", processed, failed)

    step_state.update(
        {
            "last_created_at": last_created_at,
            "last_id": last_id,
            "processed": processed,
            "failed": failed,
            "done": True,
            "done_at": _utcnow(),
        }
    )
    _save_state(state, state_file)
    logger.info("Metadata projection complete: processed=%d failed=%d", processed, failed)


def _drop_legacy_entity_constraint(*, state: Dict[str, Any], state_file: str) -> None:
    from app.graph.neo4j_client import get_neo4j_session

    step_state = state.setdefault("steps", {}).setdefault("drop_legacy_entity_constraint", {})
    if step_state.get("done"):
        return

    with get_neo4j_session() as session:
        constraints = session.run("SHOW CONSTRAINTS").data()
        if any(c.get("name") == "om_entity_user_name" for c in constraints):
            session.run("DROP CONSTRAINT om_entity_user_name IF EXISTS")
            logger.info("Dropped legacy constraint om_entity_user_name")
        else:
            logger.info("Legacy constraint om_entity_user_name not present")

    step_state.update({"done": True, "done_at": _utcnow()})
    _save_state(state, state_file)


def _update_relation_access_entity(
    *,
    state: Dict[str, Any],
    state_file: str,
) -> None:
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

    step_state = state.setdefault("steps", {}).setdefault("relation_access_entity_update", {})
    if step_state.get("done"):
        logger.info("Skip OM_RELATION accessEntity update (already done).")
        return

    if not is_neo4j_configured():
        raise SystemExit("Neo4j is not configured")

    query = """
    MATCH ()-[r:OM_RELATION]->()
    WHERE r.accessEntity IS NULL AND r.memoryId IS NOT NULL
    MATCH (m:OM_Memory {id: r.memoryId})
    WITH r, m,
         CASE
           WHEN m.accessEntity IS NOT NULL THEN m.accessEntity
           WHEN m.userId IS NOT NULL THEN 'user:' + m.userId
           ELSE NULL
         END AS resolvedAccess
    WHERE resolvedAccess IS NOT NULL
    SET r.accessEntity = resolvedAccess,
        r.userId = coalesce(r.userId, m.userId)
    RETURN count(r) AS updated
    """

    with get_neo4j_session() as session:
        record = session.run(query).single()
        updated = record["updated"] if record else 0

    step_state.update(
        {
            "updated": updated,
            "done": True,
            "done_at": _utcnow(),
        }
    )
    _save_state(state, state_file)
    logger.info("OM_RELATION accessEntity update complete: updated=%d", updated)


def _backfill_co_mentions(
    *,
    projector,
    access_entity: str,
    user_id: str,
    state: Dict[str, Any],
    state_file: str,
    min_count: int,
) -> None:
    step_state = state.setdefault("steps", {}).setdefault("co_mentions", {})
    entry = step_state.get(access_entity)
    if entry and entry.get("done"):
        logger.info("Skip co-mentions for %s (already done).", access_entity)
        return

    edges_created = projector.backfill_entity_edges(user_id, min_count, access_entity=access_entity)
    step_state[access_entity] = {
        "edges_created": edges_created,
        "done": True,
        "done_at": _utcnow(),
    }
    _save_state(state, state_file)
    logger.info("Co-mentions backfill complete for %s: edges=%d", access_entity, edges_created)


def _backfill_tag_cooccurs(
    *,
    projector,
    access_entity: str,
    user_id: str,
    state: Dict[str, Any],
    state_file: str,
    min_count: int,
    min_pmi: float,
) -> None:
    step_state = state.setdefault("steps", {}).setdefault("tag_cooccurs", {})
    entry = step_state.get(access_entity)
    if entry and entry.get("done"):
        logger.info("Skip tag co-occurs for %s (already done).", access_entity)
        return

    edges_created = projector.backfill_tag_edges(user_id, min_count, min_pmi, access_entity=access_entity)
    step_state[access_entity] = {
        "edges_created": edges_created,
        "done": True,
        "done_at": _utcnow(),
    }
    _save_state(state, state_file)
    logger.info("Tag co-occurs backfill complete for %s: edges=%d", access_entity, edges_created)


def _backfill_similarity(
    *,
    db,
    similarity_projector,
    access_entity: str,
    user_id: str,
    state: Dict[str, Any],
    state_file: str,
    checkpoint_every: int,
) -> None:
    step_state = state.setdefault("steps", {}).setdefault("similarity", {})
    entry = step_state.setdefault(access_entity, {})

    if entry.get("done"):
        logger.info("Skip similarity for %s (already done).", access_entity)
        return

    last_created_at = entry.get("last_created_at")
    last_id = entry.get("last_id")
    processed = entry.get("processed", 0)
    edges_created = entry.get("edges_created", 0)
    failed = entry.get("failed", 0)

    query = _iter_memories(
        db,
        access_entity=access_entity,
        user_id=user_id,
        last_created_at=last_created_at,
        last_id=last_id,
    )

    total_remaining = query.count()
    logger.info("Similarity backfill: %s has %d memories remaining", access_entity, total_remaining)

    for memory in query.yield_per(50):
        memory_id = str(memory.id)
        try:
            created = similarity_projector.project_similarity_edges(memory_id, user_id)
            edges_created += created
            processed += 1
        except Exception as exc:
            failed += 1
            logger.warning("Similarity failed for %s: %s", memory_id, exc)

        last_created_at = memory.created_at.isoformat() if memory.created_at else last_created_at
        last_id = memory_id

        if processed % checkpoint_every == 0:
            entry.update(
                {
                    "last_created_at": last_created_at,
                    "last_id": last_id,
                    "processed": processed,
                    "edges_created": edges_created,
                    "failed": failed,
                }
            )
            _save_state(state, state_file)
            logger.info(
                "Similarity checkpoint for %s: processed=%d edges=%d failed=%d",
                access_entity,
                processed,
                edges_created,
                failed,
            )

    entry.update(
        {
            "last_created_at": last_created_at,
            "last_id": last_id,
            "processed": processed,
            "edges_created": edges_created,
            "failed": failed,
            "done": True,
            "done_at": _utcnow(),
        }
    )
    _save_state(state, state_file)
    logger.info(
        "Similarity backfill complete for %s: processed=%d edges=%d failed=%d",
        access_entity,
        processed,
        edges_created,
        failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid graph backfill with resume support")
    parser.add_argument("--user-id", required=True, help="User ID for legacy scope fallback")
    parser.add_argument("--access-entities", default=None, help="CSV list of access_entity scopes to process")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--state-file", default=None, help="State file path")
    parser.add_argument("--co-mention-min-count", type=int, default=1, help="Min co-mentions for edge")
    parser.add_argument("--tag-min-count", type=int, default=2, help="Min tag co-occurrences for edge")
    parser.add_argument("--min-pmi", type=float, default=0.0, help="Min PMI for tag co-occurs")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Checkpoint interval for resume")
    parser.add_argument("--use-host-ports", action="store_true", help="Use host-mapped Docker ports")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    _load_dotenv()
    _apply_host_ports_if_requested(args.use_host_ports)

    log_file = args.log_file or _default_log_file()
    state_file = args.state_file or _default_state_file()
    _configure_logging(log_file=log_file, verbose=args.verbose)

    logger.info("Starting hybrid graph backfill")
    logger.info("log_file=%s state_file=%s", log_file, state_file)

    state = _load_state(state_file)
    _save_state(state, state_file)

    from app.database import SessionLocal
    from app.graph.metadata_projector import MetadataProjector, get_projector
    from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured
    from app.graph.similarity_projector import get_similarity_projector

    if not is_neo4j_configured():
        raise SystemExit("Neo4j is not configured")

    db = SessionLocal()
    try:
        projector = MetadataProjector(get_neo4j_session)
        projector.ensure_constraints()

        _drop_legacy_entity_constraint(state=state, state_file=state_file)

        _backfill_metadata(
            db=db,
            projector=projector,
            state=state,
            state_file=state_file,
            checkpoint_every=args.checkpoint_every,
        )

        _update_relation_access_entity(state=state, state_file=state_file)

        access_entities = _resolve_access_entities(db, args.user_id, args.access_entities)
        if not access_entities:
            logger.info("No access_entity values found. Exiting.")
            return

        logger.info("Access entities: %s", ", ".join(access_entities))

        scoped_projector = get_projector()
        if scoped_projector is None:
            raise SystemExit("Failed to initialize metadata projector")

        similarity_projector = get_similarity_projector()
        if similarity_projector is None:
            raise SystemExit("Failed to initialize similarity projector")

        for access_entity in access_entities:
            logger.info("Processing access_entity=%s", access_entity)
            _backfill_co_mentions(
                projector=scoped_projector,
                access_entity=access_entity,
                user_id=args.user_id,
                state=state,
                state_file=state_file,
                min_count=args.co_mention_min_count,
            )
            _backfill_tag_cooccurs(
                projector=scoped_projector,
                access_entity=access_entity,
                user_id=args.user_id,
                state=state,
                state_file=state_file,
                min_count=args.tag_min_count,
                min_pmi=args.min_pmi,
            )
            _backfill_similarity(
                db=db,
                similarity_projector=similarity_projector,
                access_entity=access_entity,
                user_id=args.user_id,
                state=state,
                state_file=state_file,
                checkpoint_every=args.checkpoint_every,
            )

        logger.info("Hybrid graph backfill complete.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
