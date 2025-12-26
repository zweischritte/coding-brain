"""Tests for repository registry and discovery.

This module tests repository registration, discovery, and metadata management
for cross-repository intelligence features.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmemory.api.cross_repo.registry import (
    InMemoryRepositoryRegistry,
    Repository,
    RepositoryAlreadyExistsError,
    RepositoryConfig,
    RepositoryMetadata,
    RepositoryNotFoundError,
    RepositoryRegistry,
    RepositoryRegistryError,
    RepositoryStatus,
)


# =============================================================================
# RepositoryStatus Tests
# =============================================================================


class TestRepositoryStatus:
    """Tests for RepositoryStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert RepositoryStatus.ACTIVE.value == "active"
        assert RepositoryStatus.INACTIVE.value == "inactive"
        assert RepositoryStatus.INDEXING.value == "indexing"
        assert RepositoryStatus.ERROR.value == "error"
        assert RepositoryStatus.PENDING.value == "pending"

    def test_status_is_operational(self):
        """Test is_operational property for different statuses."""
        assert RepositoryStatus.ACTIVE.is_operational is True
        assert RepositoryStatus.INDEXING.is_operational is True
        assert RepositoryStatus.INACTIVE.is_operational is False
        assert RepositoryStatus.ERROR.is_operational is False
        assert RepositoryStatus.PENDING.is_operational is False

    def test_status_allows_queries(self):
        """Test allows_queries property for different statuses."""
        assert RepositoryStatus.ACTIVE.allows_queries is True
        assert RepositoryStatus.INDEXING.allows_queries is True  # Partial queries allowed
        assert RepositoryStatus.INACTIVE.allows_queries is False
        assert RepositoryStatus.ERROR.allows_queries is False
        assert RepositoryStatus.PENDING.allows_queries is False


# =============================================================================
# RepositoryMetadata Tests
# =============================================================================


class TestRepositoryMetadata:
    """Tests for RepositoryMetadata dataclass."""

    def test_create_metadata(self, sample_repo_metadata: dict[str, Any]):
        """Test creating repository metadata."""
        metadata = RepositoryMetadata(
            description=sample_repo_metadata["description"],
            default_branch=sample_repo_metadata["default_branch"],
            languages=sample_repo_metadata["languages"],
            owner=sample_repo_metadata["owner"],
            visibility=sample_repo_metadata["visibility"],
            created_at=sample_repo_metadata["created_at"],
            updated_at=sample_repo_metadata["updated_at"],
        )
        assert metadata.description == "A sample repository for testing"
        assert metadata.default_branch == "main"
        assert metadata.languages == ["python", "typescript"]
        assert metadata.owner == "my-org"
        assert metadata.visibility == "private"

    def test_metadata_defaults(self):
        """Test default values for optional metadata fields."""
        metadata = RepositoryMetadata()
        assert metadata.description == ""
        assert metadata.default_branch == "main"
        assert metadata.languages == []
        assert metadata.owner == ""
        assert metadata.visibility == "private"
        assert metadata.tags == []
        assert metadata.extra == {}

    def test_metadata_with_tags(self):
        """Test metadata with tags."""
        metadata = RepositoryMetadata(
            tags=["backend", "api", "python"],
        )
        assert metadata.tags == ["backend", "api", "python"]

    def test_metadata_with_extra(self):
        """Test metadata with extra properties."""
        metadata = RepositoryMetadata(
            extra={"stars": 100, "forks": 20, "license": "MIT"},
        )
        assert metadata.extra["stars"] == 100
        assert metadata.extra["forks"] == 20
        assert metadata.extra["license"] == "MIT"

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        now = datetime.now(timezone.utc)
        metadata = RepositoryMetadata(
            description="Test repo",
            default_branch="develop",
            languages=["python"],
            owner="test-org",
            visibility="public",
            created_at=now,
            updated_at=now,
        )
        data = metadata.to_dict()
        assert data["description"] == "Test repo"
        assert data["default_branch"] == "develop"
        assert data["languages"] == ["python"]
        assert data["owner"] == "test-org"
        assert data["visibility"] == "public"

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "description": "From dict",
            "default_branch": "main",
            "languages": ["typescript"],
            "owner": "org",
            "visibility": "private",
        }
        metadata = RepositoryMetadata.from_dict(data)
        assert metadata.description == "From dict"
        assert metadata.languages == ["typescript"]


# =============================================================================
# Repository Tests
# =============================================================================


class TestRepository:
    """Tests for Repository dataclass."""

    def test_create_repository(self, sample_repo_metadata: dict[str, Any]):
        """Test creating a repository."""
        metadata = RepositoryMetadata(
            description=sample_repo_metadata["description"],
            languages=sample_repo_metadata["languages"],
            owner=sample_repo_metadata["owner"],
        )
        repo = Repository(
            repo_id="my-org/my-repo",
            name="my-repo",
            url="https://github.com/my-org/my-repo",
            metadata=metadata,
        )
        assert repo.repo_id == "my-org/my-repo"
        assert repo.name == "my-repo"
        assert repo.url == "https://github.com/my-org/my-repo"
        assert repo.status == RepositoryStatus.PENDING
        assert repo.metadata.description == sample_repo_metadata["description"]

    def test_repository_defaults(self):
        """Test default values for repository."""
        repo = Repository(
            repo_id="org/repo",
            name="repo",
        )
        assert repo.url == ""
        assert repo.status == RepositoryStatus.PENDING
        assert repo.metadata == RepositoryMetadata()
        assert repo.index_progress == 0.0
        assert repo.last_indexed_at is None
        assert repo.error_message is None

    def test_repository_equality(self):
        """Test repository equality based on repo_id."""
        repo1 = Repository(repo_id="org/repo", name="repo")
        repo2 = Repository(repo_id="org/repo", name="repo", url="https://example.com")
        repo3 = Repository(repo_id="org/other", name="other")

        assert repo1 == repo2  # Same repo_id
        assert repo1 != repo3  # Different repo_id

    def test_repository_hash(self):
        """Test repository is hashable."""
        repo1 = Repository(repo_id="org/repo", name="repo")
        repo2 = Repository(repo_id="org/repo", name="repo")

        # Same repo_id should have same hash
        assert hash(repo1) == hash(repo2)

        # Can use in sets
        repo_set = {repo1, repo2}
        assert len(repo_set) == 1

    def test_repository_is_queryable(self):
        """Test is_queryable property."""
        repo = Repository(repo_id="org/repo", name="repo")
        repo.status = RepositoryStatus.ACTIVE
        assert repo.is_queryable is True

        repo.status = RepositoryStatus.INDEXING
        assert repo.is_queryable is True

        repo.status = RepositoryStatus.ERROR
        assert repo.is_queryable is False

    def test_repository_to_dict(self):
        """Test converting repository to dictionary."""
        repo = Repository(
            repo_id="org/repo",
            name="repo",
            url="https://github.com/org/repo",
            status=RepositoryStatus.ACTIVE,
            index_progress=1.0,
        )
        data = repo.to_dict()
        assert data["repo_id"] == "org/repo"
        assert data["name"] == "repo"
        assert data["url"] == "https://github.com/org/repo"
        assert data["status"] == "active"
        assert data["index_progress"] == 1.0

    def test_repository_from_dict(self):
        """Test creating repository from dictionary."""
        data = {
            "repo_id": "org/repo",
            "name": "repo",
            "url": "https://github.com/org/repo",
            "status": "active",
            "index_progress": 0.75,
            "metadata": {"description": "Test", "languages": ["python"]},
        }
        repo = Repository.from_dict(data)
        assert repo.repo_id == "org/repo"
        assert repo.status == RepositoryStatus.ACTIVE
        assert repo.index_progress == 0.75
        assert repo.metadata.description == "Test"


# =============================================================================
# RepositoryConfig Tests
# =============================================================================


class TestRepositoryConfig:
    """Tests for RepositoryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RepositoryConfig()
        assert config.max_repositories == 1000
        assert config.auto_discover is False
        assert config.discovery_interval_seconds == 3600
        assert config.allow_external_repos is True
        assert config.require_authentication is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RepositoryConfig(
            max_repositories=500,
            auto_discover=True,
            discovery_interval_seconds=1800,
            allow_external_repos=False,
            require_authentication=False,
        )
        assert config.max_repositories == 500
        assert config.auto_discover is True
        assert config.discovery_interval_seconds == 1800
        assert config.allow_external_repos is False
        assert config.require_authentication is False


# =============================================================================
# RepositoryRegistry Interface Tests
# =============================================================================


class TestRepositoryRegistryInterface:
    """Tests for RepositoryRegistry abstract interface."""

    def test_registry_is_abstract(self):
        """Test that RepositoryRegistry cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RepositoryRegistry()  # type: ignore


# =============================================================================
# InMemoryRepositoryRegistry Tests
# =============================================================================


class TestInMemoryRepositoryRegistry:
    """Tests for InMemoryRepositoryRegistry implementation."""

    def test_create_registry(self):
        """Test creating an empty registry."""
        registry = InMemoryRepositoryRegistry()
        assert registry.count == 0
        assert registry.list_all() == []

    def test_create_registry_with_config(self):
        """Test creating registry with custom config."""
        config = RepositoryConfig(max_repositories=10)
        registry = InMemoryRepositoryRegistry(config=config)
        assert registry.config.max_repositories == 10

    def test_register_repository(self, sample_repo_metadata: dict[str, Any]):
        """Test registering a new repository."""
        registry = InMemoryRepositoryRegistry()
        metadata = RepositoryMetadata(
            description=sample_repo_metadata["description"],
            languages=sample_repo_metadata["languages"],
        )
        repo = registry.register(
            repo_id="my-org/my-repo",
            name="my-repo",
            url="https://github.com/my-org/my-repo",
            metadata=metadata,
        )
        assert repo.repo_id == "my-org/my-repo"
        assert repo.name == "my-repo"
        assert repo.status == RepositoryStatus.PENDING
        assert registry.count == 1

    def test_register_duplicate_raises_error(self):
        """Test that registering a duplicate repo raises an error."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        with pytest.raises(RepositoryAlreadyExistsError) as exc_info:
            registry.register(repo_id="org/repo", name="repo")

        assert "org/repo" in str(exc_info.value)

    def test_register_exceeds_max_raises_error(self):
        """Test that exceeding max repositories raises an error."""
        config = RepositoryConfig(max_repositories=2)
        registry = InMemoryRepositoryRegistry(config=config)

        registry.register(repo_id="org/repo1", name="repo1")
        registry.register(repo_id="org/repo2", name="repo2")

        with pytest.raises(RepositoryRegistryError) as exc_info:
            registry.register(repo_id="org/repo3", name="repo3")

        assert "maximum" in str(exc_info.value).lower()

    def test_get_repository(self):
        """Test getting a repository by ID."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        repo = registry.get("org/repo")
        assert repo is not None
        assert repo.repo_id == "org/repo"
        assert repo.name == "repo"

    def test_get_nonexistent_returns_none(self):
        """Test getting a nonexistent repository returns None."""
        registry = InMemoryRepositoryRegistry()
        assert registry.get("nonexistent/repo") is None

    def test_get_or_raise(self):
        """Test get_or_raise method."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        repo = registry.get_or_raise("org/repo")
        assert repo.repo_id == "org/repo"

    def test_get_or_raise_nonexistent(self):
        """Test get_or_raise raises for nonexistent repository."""
        registry = InMemoryRepositoryRegistry()

        with pytest.raises(RepositoryNotFoundError) as exc_info:
            registry.get_or_raise("nonexistent/repo")

        assert "nonexistent/repo" in str(exc_info.value)

    def test_unregister_repository(self):
        """Test unregistering a repository."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")
        assert registry.count == 1

        registry.unregister("org/repo")
        assert registry.count == 0
        assert registry.get("org/repo") is None

    def test_unregister_nonexistent_raises_error(self):
        """Test unregistering a nonexistent repo raises error."""
        registry = InMemoryRepositoryRegistry()

        with pytest.raises(RepositoryNotFoundError):
            registry.unregister("nonexistent/repo")

    def test_update_repository(self):
        """Test updating repository metadata."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/repo",
            name="repo",
            metadata=RepositoryMetadata(description="Original"),
        )

        registry.update(
            "org/repo",
            metadata=RepositoryMetadata(description="Updated"),
        )

        repo = registry.get("org/repo")
        assert repo is not None
        assert repo.metadata.description == "Updated"

    def test_update_status(self):
        """Test updating repository status."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        registry.update_status("org/repo", RepositoryStatus.INDEXING)
        repo = registry.get("org/repo")
        assert repo is not None
        assert repo.status == RepositoryStatus.INDEXING

        registry.update_status("org/repo", RepositoryStatus.ACTIVE, index_progress=1.0)
        repo = registry.get("org/repo")
        assert repo is not None
        assert repo.status == RepositoryStatus.ACTIVE
        assert repo.index_progress == 1.0

    def test_update_status_with_error(self):
        """Test updating repository status with error message."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        registry.update_status(
            "org/repo",
            RepositoryStatus.ERROR,
            error_message="Indexing failed: connection timeout",
        )

        repo = registry.get("org/repo")
        assert repo is not None
        assert repo.status == RepositoryStatus.ERROR
        assert repo.error_message == "Indexing failed: connection timeout"

    def test_update_nonexistent_raises_error(self):
        """Test updating a nonexistent repo raises error."""
        registry = InMemoryRepositoryRegistry()

        with pytest.raises(RepositoryNotFoundError):
            registry.update("nonexistent/repo", metadata=RepositoryMetadata())

    def test_exists(self):
        """Test checking if repository exists."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo", name="repo")

        assert registry.exists("org/repo") is True
        assert registry.exists("nonexistent/repo") is False

    def test_list_all(self):
        """Test listing all repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo1", name="repo1")
        registry.register(repo_id="org/repo2", name="repo2")
        registry.register(repo_id="org/repo3", name="repo3")

        repos = registry.list_all()
        assert len(repos) == 3
        repo_ids = [r.repo_id for r in repos]
        assert "org/repo1" in repo_ids
        assert "org/repo2" in repo_ids
        assert "org/repo3" in repo_ids

    def test_list_by_status(self):
        """Test listing repositories by status."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo1", name="repo1")
        registry.register(repo_id="org/repo2", name="repo2")
        registry.register(repo_id="org/repo3", name="repo3")

        registry.update_status("org/repo1", RepositoryStatus.ACTIVE)
        registry.update_status("org/repo2", RepositoryStatus.INDEXING)
        # repo3 stays PENDING

        active = registry.list_by_status(RepositoryStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].repo_id == "org/repo1"

        indexing = registry.list_by_status(RepositoryStatus.INDEXING)
        assert len(indexing) == 1
        assert indexing[0].repo_id == "org/repo2"

        pending = registry.list_by_status(RepositoryStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].repo_id == "org/repo3"

    def test_list_by_owner(self):
        """Test listing repositories by owner."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org1/repo1",
            name="repo1",
            metadata=RepositoryMetadata(owner="org1"),
        )
        registry.register(
            repo_id="org1/repo2",
            name="repo2",
            metadata=RepositoryMetadata(owner="org1"),
        )
        registry.register(
            repo_id="org2/repo3",
            name="repo3",
            metadata=RepositoryMetadata(owner="org2"),
        )

        org1_repos = registry.list_by_owner("org1")
        assert len(org1_repos) == 2

        org2_repos = registry.list_by_owner("org2")
        assert len(org2_repos) == 1
        assert org2_repos[0].repo_id == "org2/repo3"

    def test_list_queryable(self):
        """Test listing queryable repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo1", name="repo1")
        registry.register(repo_id="org/repo2", name="repo2")
        registry.register(repo_id="org/repo3", name="repo3")

        registry.update_status("org/repo1", RepositoryStatus.ACTIVE)
        registry.update_status("org/repo2", RepositoryStatus.INDEXING)
        registry.update_status("org/repo3", RepositoryStatus.ERROR)

        queryable = registry.list_queryable()
        assert len(queryable) == 2
        repo_ids = [r.repo_id for r in queryable]
        assert "org/repo1" in repo_ids
        assert "org/repo2" in repo_ids
        assert "org/repo3" not in repo_ids

    def test_search_by_language(self):
        """Test searching repositories by language."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/python-repo",
            name="python-repo",
            metadata=RepositoryMetadata(languages=["python"]),
        )
        registry.register(
            repo_id="org/ts-repo",
            name="ts-repo",
            metadata=RepositoryMetadata(languages=["typescript"]),
        )
        registry.register(
            repo_id="org/multi-repo",
            name="multi-repo",
            metadata=RepositoryMetadata(languages=["python", "typescript"]),
        )

        python_repos = registry.search_by_language("python")
        assert len(python_repos) == 2
        repo_ids = [r.repo_id for r in python_repos]
        assert "org/python-repo" in repo_ids
        assert "org/multi-repo" in repo_ids

        ts_repos = registry.search_by_language("typescript")
        assert len(ts_repos) == 2

        java_repos = registry.search_by_language("java")
        assert len(java_repos) == 0

    def test_search_by_tag(self):
        """Test searching repositories by tag."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/backend",
            name="backend",
            metadata=RepositoryMetadata(tags=["backend", "api"]),
        )
        registry.register(
            repo_id="org/frontend",
            name="frontend",
            metadata=RepositoryMetadata(tags=["frontend", "web"]),
        )
        registry.register(
            repo_id="org/shared",
            name="shared",
            metadata=RepositoryMetadata(tags=["backend", "frontend", "shared"]),
        )

        backend_repos = registry.search_by_tag("backend")
        assert len(backend_repos) == 2

        web_repos = registry.search_by_tag("web")
        assert len(web_repos) == 1
        assert web_repos[0].repo_id == "org/frontend"

    def test_search_by_name(self):
        """Test searching repositories by name pattern."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/api-service", name="api-service")
        registry.register(repo_id="org/api-gateway", name="api-gateway")
        registry.register(repo_id="org/frontend", name="frontend")

        api_repos = registry.search_by_name("api")
        assert len(api_repos) == 2
        repo_ids = [r.repo_id for r in api_repos]
        assert "org/api-service" in repo_ids
        assert "org/api-gateway" in repo_ids

        front_repos = registry.search_by_name("front")
        assert len(front_repos) == 1

    def test_clear(self):
        """Test clearing all repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(repo_id="org/repo1", name="repo1")
        registry.register(repo_id="org/repo2", name="repo2")
        assert registry.count == 2

        registry.clear()
        assert registry.count == 0
        assert registry.list_all() == []


# =============================================================================
# Repository Discovery Tests
# =============================================================================


class TestRepositoryDiscovery:
    """Tests for repository discovery features."""

    def test_discover_related_repos(self):
        """Test discovering related repositories."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/shared-lib",
            name="shared-lib",
            metadata=RepositoryMetadata(
                owner="org",
                tags=["shared", "library"],
                languages=["python"],
            ),
        )
        registry.register(
            repo_id="org/api-service",
            name="api-service",
            metadata=RepositoryMetadata(
                owner="org",
                tags=["api", "service"],
                languages=["python"],
            ),
        )
        registry.register(
            repo_id="other-org/unrelated",
            name="unrelated",
            metadata=RepositoryMetadata(
                owner="other-org",
                tags=["unrelated"],
                languages=["java"],
            ),
        )

        # Discover related by owner
        related = registry.discover_related("org/shared-lib", by="owner")
        assert len(related) == 1
        assert related[0].repo_id == "org/api-service"

        # Discover related by language
        related = registry.discover_related("org/shared-lib", by="language")
        assert len(related) == 1
        assert related[0].repo_id == "org/api-service"

    def test_get_repo_stats(self):
        """Test getting repository statistics."""
        registry = InMemoryRepositoryRegistry()
        registry.register(
            repo_id="org/repo1",
            name="repo1",
            metadata=RepositoryMetadata(languages=["python"]),
        )
        registry.register(
            repo_id="org/repo2",
            name="repo2",
            metadata=RepositoryMetadata(languages=["python", "typescript"]),
        )

        registry.update_status("org/repo1", RepositoryStatus.ACTIVE)
        registry.update_status("org/repo2", RepositoryStatus.INDEXING)

        stats = registry.get_stats()
        assert stats["total"] == 2
        assert stats["by_status"]["active"] == 1
        assert stats["by_status"]["indexing"] == 1
        assert stats["by_language"]["python"] == 2
        assert stats["by_language"]["typescript"] == 1


# =============================================================================
# Exception Tests
# =============================================================================


class TestRegistryExceptions:
    """Tests for registry exceptions."""

    def test_repository_registry_error(self):
        """Test base registry error."""
        error = RepositoryRegistryError("Test error")
        assert str(error) == "Test error"

    def test_repository_not_found_error(self):
        """Test repository not found error."""
        error = RepositoryNotFoundError("org/repo")
        assert "org/repo" in str(error)
        assert error.repo_id == "org/repo"

    def test_repository_already_exists_error(self):
        """Test repository already exists error."""
        error = RepositoryAlreadyExistsError("org/repo")
        assert "org/repo" in str(error)
        assert error.repo_id == "org/repo"
