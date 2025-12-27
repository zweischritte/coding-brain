"""
Tests for Phase 0.5: Infrastructure Prerequisites - Docker Compose Configuration.

TDD: These tests are written first and should fail until implementation is complete.
"""
import os
import re
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def docker_compose_path():
    """Path to the main docker-compose file."""
    # The docker-compose.yml is in the openmemory directory
    return Path(__file__).parent.parent.parent.parent / "docker-compose.yml"


@pytest.fixture
def docker_compose_config(docker_compose_path):
    """Load and parse docker-compose.yml."""
    with open(docker_compose_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def env_example_path():
    """Path to the .env.example file for openmemory."""
    # The .env.example should be in the openmemory directory for infrastructure secrets
    return Path(__file__).parent.parent.parent.parent / ".env.example"


class TestRequiredServices:
    """Test that required services are present in docker-compose."""

    def test_postgresql_service_exists(self, docker_compose_config):
        """PostgreSQL 16+ service must be defined."""
        services = docker_compose_config.get("services", {})
        # Accept 'postgres' or 'postgresql' as service name
        assert any(
            "postgres" in name.lower() for name in services.keys()
        ), "PostgreSQL service not found in docker-compose"

    def test_valkey_service_exists(self, docker_compose_config):
        """Valkey 8+ service must be defined for cache/session store."""
        services = docker_compose_config.get("services", {})
        assert any(
            "valkey" in name.lower() for name in services.keys()
        ), "Valkey service not found in docker-compose"

    def test_neo4j_service_exists(self, docker_compose_config):
        """Neo4j service must be defined for graph store."""
        services = docker_compose_config.get("services", {})
        assert any(
            "neo4j" in name.lower() for name in services.keys()
        ), "Neo4j service not found in docker-compose"

    def test_opensearch_service_exists(self, docker_compose_config):
        """OpenSearch service must be defined for full-text search."""
        services = docker_compose_config.get("services", {})
        assert any(
            "opensearch" in name.lower() for name in services.keys()
        ), "OpenSearch service not found in docker-compose"

    def test_qdrant_service_exists(self, docker_compose_config):
        """Qdrant service must be defined for vector store."""
        services = docker_compose_config.get("services", {})
        assert any(
            "qdrant" in name.lower() for name in services.keys()
        ), "Qdrant service not found in docker-compose"


class TestHealthChecks:
    """Test that all critical services have health checks."""

    def _get_service(self, config, name_pattern):
        """Get service config by name pattern."""
        services = config.get("services", {})
        for name, svc in services.items():
            if name_pattern.lower() in name.lower():
                return svc
        return None

    def test_postgresql_has_healthcheck(self, docker_compose_config):
        """PostgreSQL must have a health check defined."""
        svc = self._get_service(docker_compose_config, "postgres")
        assert svc is not None, "PostgreSQL service not found"
        assert "healthcheck" in svc, "PostgreSQL service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"

    def test_valkey_has_healthcheck(self, docker_compose_config):
        """Valkey must have a health check defined."""
        svc = self._get_service(docker_compose_config, "valkey")
        assert svc is not None, "Valkey service not found"
        assert "healthcheck" in svc, "Valkey service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"

    def test_neo4j_has_healthcheck(self, docker_compose_config):
        """Neo4j must have a health check defined."""
        svc = self._get_service(docker_compose_config, "neo4j")
        assert svc is not None, "Neo4j service not found"
        assert "healthcheck" in svc, "Neo4j service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"

    def test_opensearch_has_healthcheck(self, docker_compose_config):
        """OpenSearch must have a health check defined."""
        svc = self._get_service(docker_compose_config, "opensearch")
        assert svc is not None, "OpenSearch service not found"
        assert "healthcheck" in svc, "OpenSearch service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"

    def test_qdrant_has_healthcheck(self, docker_compose_config):
        """Qdrant must have a health check defined."""
        svc = self._get_service(docker_compose_config, "qdrant")
        assert svc is not None, "Qdrant service not found"
        assert "healthcheck" in svc, "Qdrant service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"

    def test_api_has_healthcheck(self, docker_compose_config):
        """API/MCP service must have a health check defined."""
        svc = self._get_service(docker_compose_config, "mcp")
        assert svc is not None, "MCP/API service not found"
        assert "healthcheck" in svc, "MCP/API service must have a healthcheck"
        assert "test" in svc["healthcheck"], "Healthcheck must have a test command"


class TestNoHardcodedSecrets:
    """Test that no secrets are hardcoded in docker-compose."""

    # Patterns that indicate hardcoded secrets
    SECRET_PATTERNS = [
        r"password\s*[:=]\s*['\"]?[a-zA-Z0-9!@#$%^&*()_+=\-]+['\"]?",  # password=xxx
        r"api_key\s*[:=]\s*['\"]?sk-[a-zA-Z0-9]+['\"]?",  # api_key=sk-xxx
        r"secret\s*[:=]\s*['\"]?[a-zA-Z0-9!@#$%^&*()_+=\-]+['\"]?",  # secret=xxx
    ]

    # Allowed patterns (env variable references)
    ALLOWED_PATTERNS = [
        r"\$\{[A-Z_]+\}",  # ${VAR}
        r"\$[A-Z_]+",  # $VAR
    ]

    def test_no_hardcoded_postgres_password(self, docker_compose_path):
        """PostgreSQL password must use environment variable substitution."""
        with open(docker_compose_path) as f:
            content = f.read()

        # Check for POSTGRES_PASSWORD entries
        postgres_pw_matches = re.findall(
            r"POSTGRES_PASSWORD\s*[:=]\s*(.+)", content, re.IGNORECASE
        )
        for match in postgres_pw_matches:
            match = match.strip()
            # Must be an env var reference, not a literal value
            assert re.match(
                r"^\$\{?[A-Z_]+\}?$", match
            ), f"POSTGRES_PASSWORD must use env substitution, found: {match}"

    def test_no_hardcoded_neo4j_auth(self, docker_compose_path):
        """Neo4j auth must use environment variable substitution."""
        with open(docker_compose_path) as f:
            content = f.read()

        # Check for NEO4J_AUTH entries
        neo4j_auth_matches = re.findall(r"NEO4J_AUTH\s*[:=]\s*(.+)", content, re.IGNORECASE)
        for match in neo4j_auth_matches:
            match = match.strip()
            # Must contain env var references
            assert (
                "$" in match
            ), f"NEO4J_AUTH must use env substitution, found: {match}"

    def test_no_hardcoded_opensearch_password(self, docker_compose_path):
        """OpenSearch admin password must use environment variable substitution."""
        with open(docker_compose_path) as f:
            content = f.read()

        # Check for OPENSEARCH_INITIAL_ADMIN_PASSWORD entries
        opensearch_pw_matches = re.findall(
            r"OPENSEARCH_INITIAL_ADMIN_PASSWORD\s*[:=]\s*(.+)", content, re.IGNORECASE
        )
        for match in opensearch_pw_matches:
            match = match.strip()
            # Must be an env var reference
            assert re.match(
                r"^\$\{?[A-Z_]+\}?$", match
            ), f"OPENSEARCH_INITIAL_ADMIN_PASSWORD must use env substitution, found: {match}"


class TestPinnedVersions:
    """Test that container images have pinned versions."""

    def _get_service(self, config, name_pattern):
        """Get service config by name pattern."""
        services = config.get("services", {})
        for name, svc in services.items():
            if name_pattern.lower() in name.lower():
                return svc
        return None

    def _assert_pinned_version(self, image_ref, service_name):
        """Assert that an image reference has a pinned version (not 'latest' or untagged)."""
        if image_ref is None:
            return  # Built from Dockerfile, not an image reference

        # Check it's not 'latest'
        assert not image_ref.endswith(
            ":latest"
        ), f"{service_name} uses 'latest' tag which is not pinned"

        # Check it has a version tag
        if ":" in image_ref:
            tag = image_ref.split(":")[-1]
            # Version should contain digits (e.g., 16, 5.26.4, v0.5.1)
            assert any(
                c.isdigit() for c in tag
            ), f"{service_name} tag '{tag}' does not appear to be a pinned version"
        else:
            pytest.fail(f"{service_name} image '{image_ref}' has no version tag")

    def test_postgresql_version_pinned(self, docker_compose_config):
        """PostgreSQL image must have a pinned version."""
        svc = self._get_service(docker_compose_config, "postgres")
        assert svc is not None, "PostgreSQL service not found"
        if "image" in svc:
            self._assert_pinned_version(svc["image"], "PostgreSQL")

    def test_valkey_version_pinned(self, docker_compose_config):
        """Valkey image must have a pinned version."""
        svc = self._get_service(docker_compose_config, "valkey")
        assert svc is not None, "Valkey service not found"
        if "image" in svc:
            self._assert_pinned_version(svc["image"], "Valkey")

    def test_neo4j_version_pinned(self, docker_compose_config):
        """Neo4j image must have a pinned version."""
        svc = self._get_service(docker_compose_config, "neo4j")
        assert svc is not None, "Neo4j service not found"
        if "image" in svc:
            self._assert_pinned_version(svc["image"], "Neo4j")

    def test_opensearch_version_pinned(self, docker_compose_config):
        """OpenSearch image must have a pinned version."""
        svc = self._get_service(docker_compose_config, "opensearch")
        assert svc is not None, "OpenSearch service not found"
        if "image" in svc:
            self._assert_pinned_version(svc["image"], "OpenSearch")

    def test_qdrant_version_pinned(self, docker_compose_config):
        """Qdrant image must have a pinned version."""
        svc = self._get_service(docker_compose_config, "qdrant")
        assert svc is not None, "Qdrant service not found"
        if "image" in svc:
            self._assert_pinned_version(svc["image"], "Qdrant")


class TestEnvExample:
    """Test that .env.example file has all required placeholders."""

    REQUIRED_VARS = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "OPENSEARCH_INITIAL_ADMIN_PASSWORD",
        "OPENAI_API_KEY",
    ]

    def test_env_example_exists(self, env_example_path):
        """A .env.example file must exist."""
        assert env_example_path.exists(), f".env.example not found at {env_example_path}"

    def test_env_example_has_required_vars(self, env_example_path):
        """The .env.example must contain all required variable placeholders."""
        if not env_example_path.exists():
            pytest.skip(".env.example does not exist yet")

        with open(env_example_path) as f:
            content = f.read()

        for var in self.REQUIRED_VARS:
            assert (
                var in content
            ), f"Required variable {var} not found in .env.example"

    def test_env_example_has_no_real_secrets(self, env_example_path):
        """The .env.example must not contain real secret values."""
        if not env_example_path.exists():
            pytest.skip(".env.example does not exist yet")

        with open(env_example_path) as f:
            content = f.read()

        # Check for patterns that look like real API keys or passwords
        dangerous_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API key pattern
            r"password\s*=\s*[a-zA-Z0-9!@#$%^&*]{8,}",  # Real password
        ]

        for pattern in dangerous_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            assert not matches, f"Possible real secret found in .env.example: {matches}"


class TestServiceDependencies:
    """Test that services have proper dependency declarations."""

    def _get_service(self, config, name_pattern):
        """Get service config by name pattern."""
        services = config.get("services", {})
        for name, svc in services.items():
            if name_pattern.lower() in name.lower():
                return name, svc
        return None, None

    def test_api_depends_on_postgres(self, docker_compose_config):
        """API/MCP service must depend on PostgreSQL with health condition."""
        _, api_svc = self._get_service(docker_compose_config, "mcp")
        assert api_svc is not None, "MCP/API service not found"

        depends_on = api_svc.get("depends_on", {})
        # Find postgres dependency
        postgres_dep = None
        for dep_name, dep_config in depends_on.items() if isinstance(depends_on, dict) else []:
            if "postgres" in dep_name.lower():
                postgres_dep = dep_config
                break

        assert postgres_dep is not None, "API must depend on PostgreSQL"
        if isinstance(postgres_dep, dict):
            assert (
                postgres_dep.get("condition") == "service_healthy"
            ), "API should wait for PostgreSQL to be healthy"

    def test_api_depends_on_valkey(self, docker_compose_config):
        """API/MCP service must depend on Valkey with health condition."""
        _, api_svc = self._get_service(docker_compose_config, "mcp")
        assert api_svc is not None, "MCP/API service not found"

        depends_on = api_svc.get("depends_on", {})
        # Find valkey dependency
        valkey_dep = None
        for dep_name, dep_config in depends_on.items() if isinstance(depends_on, dict) else []:
            if "valkey" in dep_name.lower():
                valkey_dep = dep_config
                break

        assert valkey_dep is not None, "API must depend on Valkey"
        if isinstance(valkey_dep, dict):
            assert (
                valkey_dep.get("condition") == "service_healthy"
            ), "API should wait for Valkey to be healthy"
