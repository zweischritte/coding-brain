"""Repository registry and discovery.

This module provides repository registration, discovery, and metadata management
for cross-repository intelligence features per implementation plan v9.

Per section 6.6 (Multi-Repository Graph):
- New nodes: CODE_Repository, CODE_APISpec
- New edges: CODE_DEPENDS_ON, CODE_PUBLISHES_API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# =============================================================================
# Exceptions
# =============================================================================


class RepositoryRegistryError(Exception):
    """Base exception for repository registry errors."""

    pass


class RepositoryNotFoundError(RepositoryRegistryError):
    """Raised when a repository is not found."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        super().__init__(f"Repository not found: {repo_id}")


class RepositoryAlreadyExistsError(RepositoryRegistryError):
    """Raised when a repository already exists."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        super().__init__(f"Repository already exists: {repo_id}")


# =============================================================================
# Enums
# =============================================================================


class RepositoryStatus(str, Enum):
    """Status of a repository in the registry."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    INDEXING = "indexing"
    ERROR = "error"
    PENDING = "pending"

    @property
    def is_operational(self) -> bool:
        """Check if the repository is in an operational state."""
        return self in (RepositoryStatus.ACTIVE, RepositoryStatus.INDEXING)

    @property
    def allows_queries(self) -> bool:
        """Check if the repository status allows queries."""
        return self in (RepositoryStatus.ACTIVE, RepositoryStatus.INDEXING)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RepositoryMetadata:
    """Metadata for a repository."""

    description: str = ""
    default_branch: str = "main"
    languages: list[str] = field(default_factory=list)
    owner: str = ""
    visibility: str = "private"
    tags: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "description": self.description,
            "default_branch": self.default_branch,
            "languages": self.languages,
            "owner": self.owner,
            "visibility": self.visibility,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepositoryMetadata":
        """Create metadata from dictionary."""
        return cls(
            description=data.get("description", ""),
            default_branch=data.get("default_branch", "main"),
            languages=data.get("languages", []),
            owner=data.get("owner", ""),
            visibility=data.get("visibility", "private"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            extra=data.get("extra", {}),
        )


@dataclass
class Repository:
    """A repository in the registry."""

    repo_id: str
    name: str
    url: str = ""
    status: RepositoryStatus = RepositoryStatus.PENDING
    metadata: RepositoryMetadata = field(default_factory=RepositoryMetadata)
    index_progress: float = 0.0
    last_indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Repository):
            return False
        return self.repo_id == other.repo_id

    def __hash__(self) -> int:
        return hash(self.repo_id)

    @property
    def is_queryable(self) -> bool:
        """Check if the repository can be queried."""
        return self.status.allows_queries

    def to_dict(self) -> dict[str, Any]:
        """Convert repository to dictionary."""
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "url": self.url,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "index_progress": self.index_progress,
            "last_indexed_at": self.last_indexed_at.isoformat()
            if self.last_indexed_at
            else None,
            "error_message": self.error_message,
            "registered_at": self.registered_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Repository":
        """Create repository from dictionary."""
        return cls(
            repo_id=data["repo_id"],
            name=data["name"],
            url=data.get("url", ""),
            status=RepositoryStatus(data.get("status", "pending")),
            metadata=RepositoryMetadata.from_dict(data.get("metadata", {})),
            index_progress=data.get("index_progress", 0.0),
            last_indexed_at=datetime.fromisoformat(data["last_indexed_at"])
            if data.get("last_indexed_at")
            else None,
            error_message=data.get("error_message"),
            registered_at=datetime.fromisoformat(data["registered_at"])
            if data.get("registered_at")
            else datetime.now(timezone.utc),
        )


@dataclass
class RepositoryConfig:
    """Configuration for the repository registry."""

    max_repositories: int = 1000
    auto_discover: bool = False
    discovery_interval_seconds: int = 3600
    allow_external_repos: bool = True
    require_authentication: bool = True


# =============================================================================
# Repository Registry Interface
# =============================================================================


class RepositoryRegistry(ABC):
    """Abstract interface for repository registry operations."""

    @abstractmethod
    def register(
        self,
        repo_id: str,
        name: str,
        url: str = "",
        metadata: Optional[RepositoryMetadata] = None,
    ) -> Repository:
        """Register a new repository."""
        pass

    @abstractmethod
    def unregister(self, repo_id: str) -> None:
        """Unregister a repository."""
        pass

    @abstractmethod
    def get(self, repo_id: str) -> Optional[Repository]:
        """Get a repository by ID."""
        pass

    @abstractmethod
    def get_or_raise(self, repo_id: str) -> Repository:
        """Get a repository by ID or raise if not found."""
        pass

    @abstractmethod
    def update(
        self,
        repo_id: str,
        metadata: Optional[RepositoryMetadata] = None,
        url: Optional[str] = None,
    ) -> Repository:
        """Update repository metadata."""
        pass

    @abstractmethod
    def update_status(
        self,
        repo_id: str,
        status: RepositoryStatus,
        index_progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Repository:
        """Update repository status."""
        pass

    @abstractmethod
    def exists(self, repo_id: str) -> bool:
        """Check if a repository exists."""
        pass

    @abstractmethod
    def list_all(self) -> list[Repository]:
        """List all repositories."""
        pass

    @abstractmethod
    def list_by_status(self, status: RepositoryStatus) -> list[Repository]:
        """List repositories by status."""
        pass

    @abstractmethod
    def list_by_owner(self, owner: str) -> list[Repository]:
        """List repositories by owner."""
        pass

    @abstractmethod
    def list_queryable(self) -> list[Repository]:
        """List all queryable repositories."""
        pass

    @abstractmethod
    def search_by_language(self, language: str) -> list[Repository]:
        """Search repositories by language."""
        pass

    @abstractmethod
    def search_by_tag(self, tag: str) -> list[Repository]:
        """Search repositories by tag."""
        pass

    @abstractmethod
    def search_by_name(self, pattern: str) -> list[Repository]:
        """Search repositories by name pattern."""
        pass

    @abstractmethod
    def discover_related(
        self, repo_id: str, by: str = "owner"
    ) -> list[Repository]:
        """Discover related repositories."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get repository statistics."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all repositories."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get total number of repositories."""
        pass

    @property
    @abstractmethod
    def config(self) -> RepositoryConfig:
        """Get registry configuration."""
        pass


# =============================================================================
# In-Memory Implementation
# =============================================================================


class InMemoryRepositoryRegistry(RepositoryRegistry):
    """In-memory repository registry for testing and development."""

    def __init__(self, config: Optional[RepositoryConfig] = None):
        self._config = config or RepositoryConfig()
        self._repositories: dict[str, Repository] = {}

    @property
    def config(self) -> RepositoryConfig:
        """Get registry configuration."""
        return self._config

    @property
    def count(self) -> int:
        """Get total number of repositories."""
        return len(self._repositories)

    def register(
        self,
        repo_id: str,
        name: str,
        url: str = "",
        metadata: Optional[RepositoryMetadata] = None,
    ) -> Repository:
        """Register a new repository."""
        if repo_id in self._repositories:
            raise RepositoryAlreadyExistsError(repo_id)

        if len(self._repositories) >= self._config.max_repositories:
            raise RepositoryRegistryError(
                f"Maximum number of repositories ({self._config.max_repositories}) reached"
            )

        repo = Repository(
            repo_id=repo_id,
            name=name,
            url=url,
            metadata=metadata or RepositoryMetadata(),
        )
        self._repositories[repo_id] = repo
        return repo

    def unregister(self, repo_id: str) -> None:
        """Unregister a repository."""
        if repo_id not in self._repositories:
            raise RepositoryNotFoundError(repo_id)
        del self._repositories[repo_id]

    def get(self, repo_id: str) -> Optional[Repository]:
        """Get a repository by ID."""
        return self._repositories.get(repo_id)

    def get_or_raise(self, repo_id: str) -> Repository:
        """Get a repository by ID or raise if not found."""
        repo = self.get(repo_id)
        if repo is None:
            raise RepositoryNotFoundError(repo_id)
        return repo

    def update(
        self,
        repo_id: str,
        metadata: Optional[RepositoryMetadata] = None,
        url: Optional[str] = None,
    ) -> Repository:
        """Update repository metadata."""
        repo = self.get_or_raise(repo_id)

        if metadata is not None:
            repo.metadata = metadata
        if url is not None:
            repo.url = url

        return repo

    def update_status(
        self,
        repo_id: str,
        status: RepositoryStatus,
        index_progress: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Repository:
        """Update repository status."""
        repo = self.get_or_raise(repo_id)

        repo.status = status

        if index_progress is not None:
            repo.index_progress = index_progress

        if error_message is not None:
            repo.error_message = error_message

        if status == RepositoryStatus.ACTIVE and index_progress == 1.0:
            repo.last_indexed_at = datetime.now(timezone.utc)

        return repo

    def exists(self, repo_id: str) -> bool:
        """Check if a repository exists."""
        return repo_id in self._repositories

    def list_all(self) -> list[Repository]:
        """List all repositories."""
        return list(self._repositories.values())

    def list_by_status(self, status: RepositoryStatus) -> list[Repository]:
        """List repositories by status."""
        return [r for r in self._repositories.values() if r.status == status]

    def list_by_owner(self, owner: str) -> list[Repository]:
        """List repositories by owner."""
        return [r for r in self._repositories.values() if r.metadata.owner == owner]

    def list_queryable(self) -> list[Repository]:
        """List all queryable repositories."""
        return [r for r in self._repositories.values() if r.is_queryable]

    def search_by_language(self, language: str) -> list[Repository]:
        """Search repositories by language."""
        return [
            r
            for r in self._repositories.values()
            if language in r.metadata.languages
        ]

    def search_by_tag(self, tag: str) -> list[Repository]:
        """Search repositories by tag."""
        return [r for r in self._repositories.values() if tag in r.metadata.tags]

    def search_by_name(self, pattern: str) -> list[Repository]:
        """Search repositories by name pattern."""
        pattern_lower = pattern.lower()
        return [
            r for r in self._repositories.values() if pattern_lower in r.name.lower()
        ]

    def discover_related(
        self, repo_id: str, by: str = "owner"
    ) -> list[Repository]:
        """Discover related repositories."""
        repo = self.get_or_raise(repo_id)

        if by == "owner":
            return [
                r
                for r in self._repositories.values()
                if r.metadata.owner == repo.metadata.owner and r.repo_id != repo_id
            ]
        elif by == "language":
            repo_languages = set(repo.metadata.languages)
            return [
                r
                for r in self._repositories.values()
                if r.repo_id != repo_id
                and bool(set(r.metadata.languages) & repo_languages)
            ]
        elif by == "tag":
            repo_tags = set(repo.metadata.tags)
            return [
                r
                for r in self._repositories.values()
                if r.repo_id != repo_id and bool(set(r.metadata.tags) & repo_tags)
            ]
        else:
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get repository statistics."""
        by_status: dict[str, int] = {}
        by_language: dict[str, int] = {}

        for repo in self._repositories.values():
            # Count by status
            status_key = repo.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            # Count by language
            for lang in repo.metadata.languages:
                by_language[lang] = by_language.get(lang, 0) + 1

        return {
            "total": len(self._repositories),
            "by_status": by_status,
            "by_language": by_language,
        }

    def clear(self) -> None:
        """Clear all repositories."""
        self._repositories.clear()
