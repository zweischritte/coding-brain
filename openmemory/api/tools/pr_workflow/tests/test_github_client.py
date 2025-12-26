"""Tests for GitHub client integration.

This module tests the GitHub MCP integration with TDD approach:
- GitHubConfig: Configuration for GitHub client
- GitHubClient: Client for GitHub operations via gh CLI
- GitHubPR: PR data representation
- GitHubComment: Comment data representation
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_pr_json() -> dict:
    """Return sample PR data as returned by gh CLI."""
    return {
        "number": 123,
        "title": "Add new feature",
        "body": "This PR adds a new feature to the codebase.",
        "state": "OPEN",
        "author": {"login": "testuser"},
        "baseRefName": "main",
        "headRefName": "feature/new-feature",
        "additions": 150,
        "deletions": 50,
        "changedFiles": 5,
        "url": "https://github.com/owner/repo/pull/123",
        "createdAt": "2025-12-20T10:00:00Z",
        "updatedAt": "2025-12-25T15:30:00Z",
    }


@pytest.fixture
def sample_diff() -> str:
    """Return a sample PR diff."""
    return """diff --git a/src/main.py b/src/main.py
index abc1234..def5678 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,5 +10,7 @@ def main():
     print("Hello")
+    process_data()
     print("World")
"""


@pytest.fixture
def sample_comments_json() -> list[dict]:
    """Return sample PR comments as returned by gh CLI."""
    return [
        {
            "id": 1,
            "author": {"login": "reviewer1"},
            "body": "LGTM!",
            "createdAt": "2025-12-21T12:00:00Z",
            "path": None,
            "line": None,
        },
        {
            "id": 2,
            "author": {"login": "reviewer2"},
            "body": "Can you add tests for this?",
            "createdAt": "2025-12-22T14:00:00Z",
            "path": "src/main.py",
            "line": 12,
        },
    ]


# =============================================================================
# GitHubConfig Tests
# =============================================================================


class TestGitHubConfig:
    """Tests for GitHubConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubConfig

        config = GitHubConfig()

        assert config.owner is None
        assert config.repo is None
        assert config.timeout == 30

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubConfig

        config = GitHubConfig(
            owner="myorg",
            repo="myrepo",
            timeout=60,
        )

        assert config.owner == "myorg"
        assert config.repo == "myrepo"
        assert config.timeout == 60

    def test_repo_from_string(self):
        """Test parsing repo from owner/repo string."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubConfig

        config = GitHubConfig.from_repo_string("myorg/myrepo")

        assert config.owner == "myorg"
        assert config.repo == "myrepo"


# =============================================================================
# GitHubPR Tests
# =============================================================================


class TestGitHubPR:
    """Tests for GitHubPR dataclass."""

    def test_pr_creation(self, sample_pr_json):
        """Test creating a GitHubPR from JSON."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubPR

        pr = GitHubPR.from_json(sample_pr_json)

        assert pr.number == 123
        assert pr.title == "Add new feature"
        assert pr.state == "OPEN"
        assert pr.author == "testuser"
        assert pr.base_branch == "main"
        assert pr.head_branch == "feature/new-feature"
        assert pr.additions == 150
        assert pr.deletions == 50
        assert pr.files_changed == 5

    def test_pr_url(self, sample_pr_json):
        """Test PR URL property."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubPR

        pr = GitHubPR.from_json(sample_pr_json)

        assert pr.url == "https://github.com/owner/repo/pull/123"

    def test_pr_is_open(self, sample_pr_json):
        """Test PR open state check."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubPR

        pr = GitHubPR.from_json(sample_pr_json)
        assert pr.is_open is True

        sample_pr_json["state"] = "CLOSED"
        pr_closed = GitHubPR.from_json(sample_pr_json)
        assert pr_closed.is_open is False


# =============================================================================
# GitHubComment Tests
# =============================================================================


class TestGitHubComment:
    """Tests for GitHubComment dataclass."""

    def test_general_comment(self, sample_comments_json):
        """Test creating a general comment."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubComment

        comment = GitHubComment.from_json(sample_comments_json[0])

        assert comment.id == 1
        assert comment.author == "reviewer1"
        assert comment.body == "LGTM!"
        assert comment.path is None
        assert comment.line is None

    def test_inline_comment(self, sample_comments_json):
        """Test creating an inline comment."""
        from openmemory.api.tools.pr_workflow.github_client import GitHubComment

        comment = GitHubComment.from_json(sample_comments_json[1])

        assert comment.id == 2
        assert comment.author == "reviewer2"
        assert comment.path == "src/main.py"
        assert comment.line == 12
        assert comment.is_inline is True


# =============================================================================
# GitHubClient Tests
# =============================================================================


class TestGitHubClient:
    """Tests for GitHubClient."""

    def test_get_pr_info(self, sample_pr_json):
        """Test fetching PR information."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(sample_pr_json),
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            pr = client.get_pr(123)

            assert pr.number == 123
            assert pr.title == "Add new feature"
            mock_run.assert_called_once()

    def test_get_pr_diff(self, sample_diff):
        """Test fetching PR diff."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=sample_diff,
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            diff = client.get_pr_diff(123)

            assert "diff --git" in diff
            assert "src/main.py" in diff

    def test_get_pr_comments(self, sample_comments_json):
        """Test fetching PR comments."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(sample_comments_json),
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            comments = client.get_pr_comments(123)

            assert len(comments) == 2
            assert comments[0].body == "LGTM!"

    def test_post_pr_comment(self):
        """Test posting a PR comment."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            result = client.post_comment(123, "Great work!")

            assert result is True
            mock_run.assert_called_once()

    def test_post_inline_comment(self):
        """Test posting an inline comment on a specific line."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            result = client.post_review_comment(
                pr_number=123,
                body="Consider adding validation here.",
                path="src/main.py",
                line=15,
            )

            assert result is True

    def test_post_review(self):
        """Test posting a review."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            result = client.post_review(
                pr_number=123,
                body="LGTM with minor suggestions.",
                event="APPROVE",
            )

            assert result is True

    def test_request_changes(self):
        """Test requesting changes on a PR."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="",
                stderr="",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            result = client.post_review(
                pr_number=123,
                body="Please address the security issues.",
                event="REQUEST_CHANGES",
            )

            assert result is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_gh_cli_not_found(self):
        """Test error when gh CLI is not installed."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
            GitHubError,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gh not found")

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            with pytest.raises(GitHubError) as exc_info:
                client.get_pr(123)

            assert "gh CLI not found" in str(exc_info.value)

    def test_authentication_error(self):
        """Test error when not authenticated."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
            GitHubError,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="error: not logged in",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            with pytest.raises(GitHubError) as exc_info:
                client.get_pr(123)

            assert "error" in str(exc_info.value).lower()

    def test_pr_not_found(self):
        """Test error when PR doesn't exist."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
            GitHubError,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Could not resolve to a PullRequest",
            )

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            with pytest.raises(GitHubError) as exc_info:
                client.get_pr(99999)

            assert "not found" in str(exc_info.value).lower() or "PullRequest" in str(exc_info.value)

    def test_timeout_handling(self):
        """Test timeout handling."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
            GitHubError,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="gh", timeout=30)

            config = GitHubConfig(owner="owner", repo="repo", timeout=30)
            client = GitHubClient(config)

            with pytest.raises(GitHubError) as exc_info:
                client.get_pr(123)

            assert "timeout" in str(exc_info.value).lower()


# =============================================================================
# Repository Detection Tests
# =============================================================================


class TestRepoDetection:
    """Tests for automatic repository detection."""

    def test_detect_repo_from_git(self):
        """Test detecting repo from git remote."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="git@github.com:owner/repo.git\n",
                stderr="",
            )

            config = GitHubConfig()  # No owner/repo specified
            client = GitHubClient(config)

            owner, repo = client._detect_repo()

            assert owner == "owner"
            assert repo == "repo"

    def test_detect_repo_https(self):
        """Test detecting repo from HTTPS remote."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="https://github.com/owner/repo.git\n",
                stderr="",
            )

            config = GitHubConfig()
            client = GitHubClient(config)

            owner, repo = client._detect_repo()

            assert owner == "owner"
            assert repo == "repo"


# =============================================================================
# Command Building Tests
# =============================================================================


class TestCommandBuilding:
    """Tests for gh CLI command building."""

    def test_build_pr_view_command(self):
        """Test building pr view command."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        config = GitHubConfig(owner="owner", repo="repo")
        client = GitHubClient(config)

        cmd = client._build_pr_view_cmd(123)

        assert "gh" in cmd
        assert "pr" in cmd
        assert "view" in cmd
        assert "123" in cmd
        assert "--repo" in cmd or "-R" in cmd

    def test_build_pr_diff_command(self):
        """Test building pr diff command."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        config = GitHubConfig(owner="owner", repo="repo")
        client = GitHubClient(config)

        cmd = client._build_pr_diff_cmd(123)

        assert "gh" in cmd
        assert "pr" in cmd
        assert "diff" in cmd
        assert "123" in cmd


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_github_client(self):
        """Test factory function creates client correctly."""
        from openmemory.api.tools.pr_workflow.github_client import (
            create_github_client,
            GitHubConfig,
        )

        client = create_github_client(owner="owner", repo="repo")

        assert client is not None
        assert client.config.owner == "owner"
        assert client.config.repo == "repo"

    def test_create_from_repo_string(self):
        """Test creating client from owner/repo string."""
        from openmemory.api.tools.pr_workflow.github_client import create_github_client

        client = create_github_client(repo_string="myorg/myrepo")

        assert client.config.owner == "myorg"
        assert client.config.repo == "myrepo"


# =============================================================================
# Integration with Analysis Tools Tests
# =============================================================================


class TestIntegrationWithAnalysis:
    """Tests for integration with PR analysis tools."""

    def test_fetch_and_analyze(self, sample_pr_json, sample_diff):
        """Test fetching PR data and preparing for analysis."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )

        with patch("subprocess.run") as mock_run:
            # First call returns PR info, second returns diff
            mock_run.side_effect = [
                Mock(returncode=0, stdout=json.dumps(sample_pr_json), stderr=""),
                Mock(returncode=0, stdout=sample_diff, stderr=""),
            ]

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            pr = client.get_pr(123)
            diff = client.get_pr_diff(123)

            assert pr.number == 123
            assert "diff --git" in diff

    def test_post_analysis_results(self):
        """Test posting analysis results as review."""
        from openmemory.api.tools.pr_workflow.github_client import (
            GitHubClient,
            GitHubConfig,
        )
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewSuggestion,
            ReviewComment,
            CommentType,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            config = GitHubConfig(owner="owner", repo="repo")
            client = GitHubClient(config)

            # Create analysis results
            suggestion = ReviewSuggestion(
                comments=[
                    ReviewComment(
                        file_path="src/main.py",
                        line_number=15,
                        comment_type=CommentType.SUGGESTION,
                        body="Consider adding error handling here.",
                        severity="info",
                    )
                ],
                overall_assessment="LGTM with minor suggestions.",
                request_changes=False,
            )

            # Post results
            success = client.post_review(
                pr_number=123,
                body=suggestion.overall_assessment,
                event="APPROVE" if not suggestion.request_changes else "REQUEST_CHANGES",
            )

            assert success is True
