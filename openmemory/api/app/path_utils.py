"""Path resolution helpers for containerized environments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_repo_root(root_path: str, repo_id: Optional[str] = None) -> Path:
    """Resolve a repository path, handling container mounts."""
    root = Path(root_path)
    if root.exists() and root.is_dir():
        return root

    candidates: list[Path] = []
    env_root = os.environ.get("OPENMEMORY_REPO_ROOT") or os.environ.get(
        "OPENMEMORY_WORKSPACE_ROOT"
    )
    if env_root:
        env_path = Path(env_root)
        candidates.append(env_path)
        if repo_id:
            if f"/{repo_id}/" in root_path:
                suffix = root_path.split(f"/{repo_id}/", 1)[1]
                candidates.append(env_path / repo_id / suffix)
            candidates.append(env_path / repo_id)

    basename = root.name
    if basename:
        candidates.append(Path("/usr/src") / basename)
    if repo_id:
        if f"/{repo_id}/" in root_path:
            suffix = root_path.split(f"/{repo_id}/", 1)[1]
            candidates.append(Path("/usr/src") / repo_id / suffix)
        candidates.append(Path("/usr/src") / repo_id)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return root
