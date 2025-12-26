"""Shared fixtures for cross-repository tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest


@pytest.fixture
def sample_repo_metadata() -> dict[str, Any]:
    """Create sample repository metadata."""
    return {
        "name": "my-org/my-repo",
        "description": "A sample repository for testing",
        "default_branch": "main",
        "languages": ["python", "typescript"],
        "owner": "my-org",
        "visibility": "private",
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 6, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_repo_metadata_2() -> dict[str, Any]:
    """Create a second sample repository metadata."""
    return {
        "name": "my-org/shared-lib",
        "description": "Shared library used across projects",
        "default_branch": "main",
        "languages": ["python"],
        "owner": "my-org",
        "visibility": "private",
        "created_at": datetime(2023, 6, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 7, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def sample_repo_metadata_3() -> dict[str, Any]:
    """Create a third sample repository metadata."""
    return {
        "name": "my-org/frontend",
        "description": "Frontend application",
        "default_branch": "develop",
        "languages": ["typescript", "javascript"],
        "owner": "my-org",
        "visibility": "private",
        "created_at": datetime(2024, 3, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 8, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def external_repo_metadata() -> dict[str, Any]:
    """Create metadata for an external repository."""
    return {
        "name": "external-org/common-utils",
        "description": "External common utilities library",
        "default_branch": "main",
        "languages": ["python"],
        "owner": "external-org",
        "visibility": "public",
        "created_at": datetime(2022, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 5, 1, tzinfo=timezone.utc),
    }
