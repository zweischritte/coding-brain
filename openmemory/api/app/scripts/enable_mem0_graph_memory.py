#!/usr/bin/env python3
"""
Enable Mem0 Graph Memory (LLM-extracted entity relations) for OpenMemory.

This updates the OpenMemory config stored in the SQL database (configs.key='main')
to include a Mem0 `graph_store` configuration pointing at Neo4j.

After this, new `add_memories` calls will automatically write entity-to-entity
relationships into Neo4j (Mem0's __Entity__ graph), and `search_memory` can enrich
results with `relations` (Mem0 Graph Memory search output).

Notes:
- This does NOT backfill existing memories into Mem0 Graph Memory. Use a backfill
  script/job if you want relationships for historical memories.
- Values are stored as env references (env:NEO4J_URL etc.) so Docker/host envs work.

Usage:
    python -m app.scripts.enable_mem0_graph_memory

Options:
    --threshold 0.75         Embedding match threshold (default: 0.75)
    --base-label             Use base label __Entity__ for all nodes (recommended)
    --custom-prompt TEXT     Optional additional extraction guidance
    --database NAME          Override Neo4j database name (default: env:NEO4J_DATABASE)
"""

from __future__ import annotations

import argparse
import json
import os
import copy
from typing import Any, Dict, Optional

from app.database import SessionLocal
from app.models import Config as ConfigModel


def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(source.get(key), dict):
            source[key] = _deep_update(source[key], value)
        else:
            source[key] = value
    return source


def enable_mem0_graph_memory(
    *,
    threshold: float = 0.75,
    base_label: bool = True,
    custom_prompt: Optional[str] = None,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist graph_store configuration into OpenMemory's config table.

    Returns the updated config dict.
    """
    db = SessionLocal()
    try:
        row = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
        if row is None:
            row = ConfigModel(key="main", value={"openmemory": {}, "mem0": {}})
            db.add(row)
            db.commit()
            db.refresh(row)

        # IMPORTANT: deep-copy to avoid in-place mutation of SQLAlchemy JSON dicts
        # (in-place changes are not tracked unless using MutableDict).
        current = copy.deepcopy(row.value or {})

        graph_store: Dict[str, Any] = {
            "provider": "neo4j",
            "config": {
                "url": "env:NEO4J_URL",
                "username": "env:NEO4J_USERNAME",
                "password": "env:NEO4J_PASSWORD",
                "database": database or "env:NEO4J_DATABASE",
                "base_label": bool(base_label),
            },
            "threshold": float(threshold),
        }
        if custom_prompt:
            graph_store["custom_prompt"] = custom_prompt

        updated = _deep_update(
            current,
            {
                "mem0": {
                    "graph_store": graph_store,
                }
            },
        )

        row.value = updated
        db.commit()
        db.refresh(row)
        return row.value
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enable Mem0 Graph Memory (Neo4j) in OpenMemory config")
    parser.add_argument("--threshold", type=float, default=0.75, help="Embedding similarity threshold (0.0-1.0)")
    parser.add_argument("--base-label", action="store_true", help="Use base __Entity__ label for all nodes")
    parser.add_argument("--custom-prompt", default=None, help="Optional extra extraction guidance line")
    parser.add_argument("--database", default=None, help="Override Neo4j database name (default: env:NEO4J_DATABASE)")
    args = parser.parse_args()

    # Quick sanity: warn if env vars are missing (but keep env: references in DB)
    missing = [k for k in ("NEO4J_URL", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not os.environ.get(k)]
    if missing:
        print(f"Warning: missing env vars {missing}. Mem0 Graph Memory will fail to connect until they are set.")

    updated = enable_mem0_graph_memory(
        threshold=args.threshold,
        base_label=args.base_label,
        custom_prompt=args.custom_prompt,
        database=args.database,
    )

    # Print just the relevant part to avoid leaking secrets (values are env: refs)
    print(json.dumps({"mem0": {"graph_store": updated.get("mem0", {}).get("graph_store")}}, indent=2))


if __name__ == "__main__":
    main()
