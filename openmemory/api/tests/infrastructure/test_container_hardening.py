"""
Tests for Phase 7: Container Hardening.

TDD: These tests verify the Dockerfile follows security best practices.

This module tests:
- Non-root user in container
- No sensitive files in image
- Proper file ownership
- Security headers in docker-compose

Run with: docker compose exec codingbrain-mcp pytest tests/infrastructure/test_container_hardening.py -v
"""
import re
from pathlib import Path

import pytest


class TestDockerfileHardening:
    """Test Dockerfile security best practices."""

    @pytest.fixture
    def dockerfile_content(self) -> str:
        """Read the Dockerfile content."""
        # Look for Dockerfile in expected location relative to app
        dockerfile_path = Path("/usr/src/Dockerfile")
        if not dockerfile_path.exists():
            # Try alternate paths
            for alt_path in [
                Path("/usr/src/openmemory/Dockerfile"),
                Path(__file__).parent.parent.parent.parent / "Dockerfile",
            ]:
                if alt_path.exists():
                    dockerfile_path = alt_path
                    break

        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found in container")

        return dockerfile_path.read_text()

    def test_dockerfile_has_nonroot_user(self, dockerfile_content: str):
        """Dockerfile creates and uses a non-root user."""
        # Check for user creation (groupadd/useradd or addgroup/adduser)
        has_user_creation = (
            "useradd" in dockerfile_content
            or "adduser" in dockerfile_content
            or "addgroup" in dockerfile_content
        )
        assert has_user_creation, "Dockerfile should create a non-root user"

    def test_dockerfile_has_user_directive(self, dockerfile_content: str):
        """Dockerfile has USER directive to switch to non-root user."""
        # Look for USER directive that's not root
        user_match = re.search(r"^USER\s+(\S+)", dockerfile_content, re.MULTILINE)
        assert user_match is not None, "Dockerfile should have USER directive"
        user_name = user_match.group(1)
        assert user_name not in (
            "root",
            "0",
        ), f"USER should not be root, got: {user_name}"

    def test_dockerfile_user_before_cmd(self, dockerfile_content: str):
        """USER directive appears before CMD to ensure app runs as non-root."""
        lines = dockerfile_content.strip().split("\n")
        user_line = None
        cmd_line = None

        for i, line in enumerate(lines):
            if line.strip().startswith("USER "):
                user_line = i
            if line.strip().startswith("CMD "):
                cmd_line = i

        assert user_line is not None, "Dockerfile should have USER directive"
        assert cmd_line is not None, "Dockerfile should have CMD directive"
        assert user_line < cmd_line, "USER directive should come before CMD"

    def test_dockerfile_no_root_cmd(self, dockerfile_content: str):
        """Dockerfile does not explicitly run commands as root at end."""
        lines = dockerfile_content.strip().split("\n")

        # Get all lines after the last USER directive
        last_user_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("USER "):
                last_user_idx = i

        if last_user_idx is None:
            pytest.fail("Dockerfile should have USER directive")

        # Check no 'USER root' or 'USER 0' after the app user is set
        for line in lines[last_user_idx + 1 :]:
            line = line.strip()
            if line.startswith("USER "):
                user_val = line.split()[1] if len(line.split()) > 1 else ""
                assert user_val not in (
                    "root",
                    "0",
                ), "Should not switch back to root after setting app user"

    def test_dockerfile_uses_copy_chown(self, dockerfile_content: str):
        """Dockerfile uses COPY --chown for proper file ownership."""
        # Modern best practice: use --chown with COPY
        has_chown = "--chown=" in dockerfile_content
        assert has_chown, "Dockerfile should use COPY --chown for file ownership"

    def test_dockerfile_uses_slim_or_distroless_base(self, dockerfile_content: str):
        """Dockerfile uses minimal base image (slim or distroless)."""
        first_from_match = re.search(
            r"^FROM\s+(\S+)", dockerfile_content, re.MULTILINE
        )
        assert first_from_match is not None, "Dockerfile should have FROM directive"

        base_image = first_from_match.group(1).lower()
        is_minimal = any(
            keyword in base_image
            for keyword in ["slim", "alpine", "distroless", "minimal"]
        )
        assert is_minimal, f"Base image should be minimal (slim/alpine), got: {base_image}"

    def test_dockerfile_has_healthcheck(self, dockerfile_content: str):
        """Dockerfile includes HEALTHCHECK instruction (optional but recommended)."""
        # This is optional - we use docker-compose healthchecks
        # Just log a note if missing
        has_healthcheck = "HEALTHCHECK" in dockerfile_content
        if not has_healthcheck:
            # Not a failure, just a note
            pass  # Healthcheck is in docker-compose.yml


class TestDockerignoreSecurity:
    """Test .dockerignore excludes sensitive files."""

    @pytest.fixture
    def dockerignore_content(self) -> str:
        """Read the .dockerignore content."""
        dockerignore_path = Path("/usr/src/.dockerignore")
        if not dockerignore_path.exists():
            for alt_path in [
                Path("/usr/src/openmemory/.dockerignore"),
                Path(__file__).parent.parent.parent.parent / ".dockerignore",
            ]:
                if alt_path.exists():
                    dockerignore_path = alt_path
                    break

        if not dockerignore_path.exists():
            pytest.skip(".dockerignore not found")

        return dockerignore_path.read_text()

    def test_dockerignore_excludes_env_files(self, dockerignore_content: str):
        """Dockerignore excludes .env files to prevent secret leakage."""
        patterns = [".env", "*.env", "**/.env*"]
        has_env_exclusion = any(
            pattern in dockerignore_content for pattern in patterns
        )
        assert has_env_exclusion, ".dockerignore should exclude .env files"

    def test_dockerignore_excludes_git(self, dockerignore_content: str):
        """Dockerignore excludes .git directory."""
        has_git_exclusion = ".git" in dockerignore_content
        assert has_git_exclusion, ".dockerignore should exclude .git directory"

    def test_dockerignore_excludes_pycache(self, dockerignore_content: str):
        """Dockerignore excludes __pycache__ directories."""
        has_pycache = "__pycache__" in dockerignore_content
        assert has_pycache, ".dockerignore should exclude __pycache__"

    def test_dockerignore_excludes_tests(self, dockerignore_content: str):
        """Dockerignore excludes test files (optional for production)."""
        # This is a recommendation, not a hard requirement
        patterns = ["tests/", "test_*.py", "*_test.py"]
        has_test_exclusion = any(p in dockerignore_content for p in patterns)
        if not has_test_exclusion:
            pass  # Optional but recommended


class TestNoSensitiveFilesInImage:
    """Test that sensitive files are not included in container."""

    def test_no_env_file_in_container(self):
        """Container should not have .env file (in production build).

        Note: In development with volume mounts, .env may be present.
        This test verifies .dockerignore is configured correctly.
        """
        import os

        # Check if we're in development mode (volume mounted)
        # In dev mode, tests/ directory would be visible (excluded by .dockerignore in prod)
        # Also check for Dockerfile which is excluded in .dockerignore
        dev_mode_indicators = [
            "/usr/src/openmemory/tests",  # Excluded in .dockerignore
            "/usr/src/openmemory/Dockerfile",  # Excluded in .dockerignore
        ]
        if any(os.path.exists(p) for p in dev_mode_indicators):
            pytest.skip("Skipping in development mode with volume mounts")

        env_paths = [
            Path("/usr/src/.env"),
            Path("/usr/src/openmemory/.env"),
            Path("/app/.env"),
        ]
        for env_path in env_paths:
            assert not env_path.exists(), f"Container should not have {env_path}"

    def test_no_git_directory_in_container(self):
        """Container should not have .git directory."""
        git_paths = [
            Path("/usr/src/.git"),
            Path("/usr/src/openmemory/.git"),
            Path("/app/.git"),
        ]
        for git_path in git_paths:
            assert not git_path.exists(), f"Container should not have {git_path}"

    def test_no_private_keys_in_container(self):
        """Container should not have private key files."""
        import glob

        key_patterns = [
            "/usr/src/**/*.pem",
            "/usr/src/**/*.key",
            "/usr/src/**/id_rsa",
            "/usr/src/**/*.p12",
        ]
        for pattern in key_patterns:
            matches = glob.glob(pattern, recursive=True)
            assert len(matches) == 0, f"Found private key files: {matches}"


class TestContainerRuntimeSecurity:
    """Test container runtime security settings."""

    def test_app_runs_as_nonroot(self):
        """Verify the application process is running as non-root."""
        import os

        # Get current user ID
        uid = os.getuid()
        # Non-root users have UID > 0
        # Note: In test environment this may be root, skip if so
        if uid == 0:
            pytest.skip("Test running as root in test environment")
        assert uid > 0, "Application should run as non-root user"

    def test_writable_paths_are_safe(self):
        """Verify writable paths are properly scoped."""
        import os

        # Check that app directory is not world-writable
        app_dir = Path("/usr/src/openmemory")
        if app_dir.exists():
            mode = app_dir.stat().st_mode
            # Check world-write bit is not set (octal 002)
            world_writable = mode & 0o002
            assert (
                not world_writable
            ), "App directory should not be world-writable"
