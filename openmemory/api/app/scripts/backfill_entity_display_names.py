#!/usr/bin/env python3
"""
Backfill OM_Entity.displayName from repo sources.

Scans source repos for entity-like tokens and quoted strings, then selects
best-cased candidates for existing OM_Entity nodes. Defaults to filling only
missing displayName values (use --overwrite to force updates).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".next",
    ".turbo",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "logs",
    "venv",
    ".venv",
}

DEFAULT_IGNORED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".webm",
    ".wasm",
    ".lock",
    ".bin",
}

TOKEN_RE = re.compile(r"\\b[A-Za-z][A-Za-z0-9_-]{1,80}\\b")
QUOTED_RE = re.compile(r"(['\"`])([^\\r\\n]{2,160})\\1")
ALLOWED_QUOTED_RE = re.compile(r"[A-Za-z0-9 _-]{2,80}")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("logs") / f"backfill_entity_display_names_{ts}.log")


def _configure_logging(*, log_file: str, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _is_ignored_path(path: Path, ignored_dirs: set) -> bool:
    return any(part in ignored_dirs for part in path.parts)


def _iter_text_files(
    repo_root: Path,
    ignored_dirs: set,
    ignored_exts: set,
    max_size_bytes: int,
) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        if _is_ignored_path(path, ignored_dirs):
            continue
        if path.suffix.lower() in ignored_exts:
            continue
        try:
            if path.stat().st_size > max_size_bytes:
                continue
        except OSError:
            continue
        yield path


def _candidate_quality(candidate: str) -> float:
    score = 0.0
    if candidate != candidate.lower():
        score += 2.0
    if candidate.isupper():
        score += 1.0
    if " " in candidate:
        score += 1.5
    if "-" in candidate:
        score += 1.0
    if "_" in candidate:
        score += 0.5
    return score


def _gather_candidates(
    repo_paths: List[Path],
    target_names: set,
    ignored_dirs: set,
    ignored_exts: set,
    max_size_bytes: int,
) -> Dict[str, Dict[str, int]]:
    from app.graph.entity_normalizer import normalize_entity_name

    candidates: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for repo_root in repo_paths:
        if not repo_root.exists():
            logger.warning("Repo path not found: %s", repo_root)
            continue

        logger.info("Scanning repo: %s", repo_root)
        for path in _iter_text_files(repo_root, ignored_dirs, ignored_exts, max_size_bytes):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for token in TOKEN_RE.findall(text):
                normalized = normalize_entity_name(token)
                if normalized and normalized in target_names:
                    candidates[normalized][token] += 1

            for _, quoted in QUOTED_RE.findall(text):
                candidate = quoted.strip()
                if not candidate or not ALLOWED_QUOTED_RE.fullmatch(candidate):
                    continue
                normalized = normalize_entity_name(candidate)
                if normalized and normalized in target_names:
                    candidates[normalized][candidate] += 1

    return candidates


def _pick_best_display_name(candidates: Dict[str, int]) -> Optional[str]:
    best = None
    best_score = -1.0
    best_count = 0
    for candidate, count in candidates.items():
        score = count + _candidate_quality(candidate)
        if score > best_score or (score == best_score and count > best_count):
            best = candidate
            best_score = score
            best_count = count
    return best


@dataclass
class EntityRow:
    name: str
    display_name: Optional[str]
    access_entity: Optional[str]
    user_id: Optional[str]


@dataclass
class BackfillStats:
    total_entities: int = 0
    candidates_found: int = 0
    updated: int = 0
    skipped: int = 0
    unchanged: int = 0


def backfill_entity_display_names(
    *,
    repo_paths: List[Path],
    overwrite: bool,
    dry_run: bool,
    limit: Optional[int],
    log_file: Optional[str],
    verbose: bool,
    ignored_dirs: set,
    ignored_exts: set,
    max_size_bytes: int,
) -> Tuple[BackfillStats, str]:
    _load_dotenv()

    from app.graph.neo4j_client import is_neo4j_configured, get_neo4j_session

    chosen_log_file = log_file or _default_log_file()
    _configure_logging(log_file=chosen_log_file, verbose=verbose)

    logger.info("Starting entity displayName backfill")
    logger.info("repos=%s overwrite=%s dry_run=%s limit=%s", repo_paths, overwrite, dry_run, limit)
    logger.info("log_file=%s", chosen_log_file)

    if not is_neo4j_configured():
        raise SystemExit("Neo4j is not configured")

    entities: List[EntityRow] = []
    with get_neo4j_session() as session:
        result = session.run(
            """
            MATCH (e:OM_Entity)
            RETURN e.name AS name,
                   e.displayName AS displayName,
                   e.accessEntity AS accessEntity,
                   e.userId AS userId
            """
        )
        for record in result:
            name = record.get("name")
            if not name:
                continue
            entities.append(
                EntityRow(
                    name=name,
                    display_name=record.get("displayName"),
                    access_entity=record.get("accessEntity"),
                    user_id=record.get("userId"),
                )
            )

    stats = BackfillStats(total_entities=len(entities))
    if not entities:
        logger.info("No entities found")
        return stats, chosen_log_file

    target_rows: List[EntityRow] = []
    for row in entities:
        if row.display_name and row.display_name.strip() and not overwrite:
            normalized_display = row.display_name.strip()
            if normalized_display == row.name and normalized_display == normalized_display.lower():
                target_rows.append(row)
            else:
                stats.skipped += 1
            continue
        target_rows.append(row)

    if not target_rows:
        logger.info("No entities eligible for backfill")
        return stats, chosen_log_file

    if limit is not None:
        target_rows = target_rows[: max(1, int(limit))]

    target_names = {row.name for row in target_rows}
    candidate_map = _gather_candidates(
        repo_paths=repo_paths,
        target_names=target_names,
        ignored_dirs=ignored_dirs,
        ignored_exts=ignored_exts,
        max_size_bytes=max_size_bytes,
    )

    stats.candidates_found = sum(1 for name in target_names if candidate_map.get(name))

    updates: List[Dict[str, str]] = []
    for row in target_rows:
        display_name = None
        if row.name and any(c.isupper() for c in row.name):
            display_name = row.name
        else:
            display_name = _pick_best_display_name(candidate_map.get(row.name, {}))
        if not display_name:
            display_name = row.name

        if row.display_name and row.display_name.strip() == display_name.strip():
            stats.unchanged += 1
            continue

        access_entity = row.access_entity or (f"user:{row.user_id}" if row.user_id else None)
        if not access_entity:
            stats.skipped += 1
            continue

        updates.append(
            {
                "name": row.name,
                "displayName": display_name,
                "accessEntity": access_entity,
                "legacyAccessEntity": f"user:{row.user_id}" if row.user_id else access_entity,
            }
        )

    if dry_run:
        logger.info("[DRY RUN] Would update %s entities", len(updates))
        stats.updated = 0
        return stats, chosen_log_file

    if not updates:
        logger.info("No updates to apply")
        return stats, chosen_log_file

    with get_neo4j_session() as session:
        chunk_size = 200
        for i in range(0, len(updates), chunk_size):
            chunk = updates[i : i + chunk_size]
            result = session.run(
                """
                UNWIND $rows AS row
                MATCH (e:OM_Entity {name: row.name})
                WHERE coalesce(e.accessEntity, row.legacyAccessEntity) = row.accessEntity
                SET e.displayName = row.displayName,
                    e.updatedAt = datetime()
                RETURN count(e) AS updated
                """,
                rows=chunk,
            )
            record = result.single()
            if record:
                stats.updated += record["updated"] or 0

    logger.info("Backfill complete. updated=%s skipped=%s unchanged=%s", stats.updated, stats.skipped, stats.unchanged)
    return stats, chosen_log_file


def _default_repo_paths() -> List[Path]:
    paths: List[Path] = []
    try:
        coding_brain_root = Path(__file__).resolve().parents[4]
        paths.append(coding_brain_root)
    except Exception:
        pass

    vgbk_repo = Path("/Users/grischadallmer/git/vg-bild-kunst-mitgliederportal")
    if vgbk_repo.exists():
        paths.append(vgbk_repo)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OM_Entity.displayName from repo sources")
    parser.add_argument("--repo", action="append", default=None, help="Repo path to scan (can be used multiple times)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing displayName values")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Neo4j")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entities to update")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-file-bytes", type=int, default=800_000, help="Max file size to scan")
    args = parser.parse_args()

    repo_paths = [Path(p).expanduser() for p in args.repo] if args.repo else _default_repo_paths()
    if not repo_paths:
        raise SystemExit("No repo paths found. Use --repo to specify one or more paths.")

    stats, log_file = backfill_entity_display_names(
        repo_paths=repo_paths,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        limit=args.limit,
        log_file=args.log_file,
        verbose=args.verbose,
        ignored_dirs=set(DEFAULT_IGNORED_DIRS),
        ignored_exts=set(DEFAULT_IGNORED_EXTS),
        max_size_bytes=max(10_000, int(args.max_file_bytes)),
    )

    print(
        "Entity displayName backfill complete. "
        f"updated={stats.updated} skipped={stats.skipped} unchanged={stats.unchanged} "
        f"log_file={log_file}"
    )


if __name__ == "__main__":
    main()
