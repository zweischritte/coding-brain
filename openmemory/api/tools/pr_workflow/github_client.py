"""GitHub Client for PR Workflow.

This module provides GitHub integration via the gh CLI:
- GitHubConfig: Configuration for GitHub client
- GitHubClient: Client for GitHub operations
- GitHubPR: PR data representation
- GitHubComment: Comment data representation
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class GitHubError(Exception):
    """Base exception for GitHub client errors."""

    pass


class GitHubAuthError(GitHubError):
    """Raised when authentication fails."""

    pass


class GitHubNotFoundError(GitHubError):
    """Raised when a resource is not found."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GitHubConfig:
    """Configuration for GitHub client.

    Args:
        owner: Repository owner/organization
        repo: Repository name
        timeout: Command timeout in seconds
    """

    owner: Optional[str] = None
    repo: Optional[str] = None
    timeout: int = 30

    @classmethod
    def from_repo_string(cls, repo_string: str, timeout: int = 30) -> "GitHubConfig":
        """Create config from owner/repo string.

        Args:
            repo_string: String in format "owner/repo"
            timeout: Command timeout in seconds

        Returns:
            GitHubConfig instance
        """
        if "/" not in repo_string:
            raise ValueError(f"Invalid repo string: {repo_string}. Expected format: owner/repo")

        owner, repo = repo_string.split("/", 1)
        return cls(owner=owner, repo=repo, timeout=timeout)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GitHubPR:
    """Representation of a GitHub PR.

    Args:
        number: PR number
        title: PR title
        body: PR description
        state: PR state (OPEN, CLOSED, MERGED)
        author: Author username
        base_branch: Base branch name
        head_branch: Head branch name
        additions: Total additions
        deletions: Total deletions
        files_changed: Number of files changed
        url: PR URL
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    number: int
    title: str
    body: str
    state: str
    author: str
    base_branch: str
    head_branch: str
    additions: int
    deletions: int
    files_changed: int
    url: str
    created_at: str
    updated_at: str

    @property
    def is_open(self) -> bool:
        """Check if PR is open."""
        return self.state == "OPEN"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GitHubPR":
        """Create from gh CLI JSON output."""
        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body", ""),
            state=data.get("state", "UNKNOWN"),
            author=data.get("author", {}).get("login", "") if isinstance(data.get("author"), dict) else "",
            base_branch=data.get("baseRefName", ""),
            head_branch=data.get("headRefName", ""),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            files_changed=data.get("changedFiles", 0),
            url=data.get("url", ""),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", ""),
        )


@dataclass
class GitHubComment:
    """Representation of a GitHub comment.

    Args:
        id: Comment ID
        author: Author username
        body: Comment body
        created_at: Creation timestamp
        path: File path for inline comments
        line: Line number for inline comments
    """

    id: int
    author: str
    body: str
    created_at: str
    path: Optional[str] = None
    line: Optional[int] = None

    @property
    def is_inline(self) -> bool:
        """Check if this is an inline comment."""
        return self.path is not None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "GitHubComment":
        """Create from gh CLI JSON output."""
        return cls(
            id=data.get("id", 0),
            author=data.get("author", {}).get("login", "") if isinstance(data.get("author"), dict) else "",
            body=data.get("body", ""),
            created_at=data.get("createdAt", ""),
            path=data.get("path"),
            line=data.get("line"),
        )


# =============================================================================
# Main Client
# =============================================================================


class GitHubClient:
    """Client for GitHub operations via gh CLI.

    Provides methods for:
    - Fetching PR information
    - Fetching PR diffs
    - Posting comments and reviews
    """

    # Patterns for extracting repo info from git remote
    SSH_REMOTE_RE = re.compile(r"git@github\.com:([^/]+)/([^.]+)(?:\.git)?")
    HTTPS_REMOTE_RE = re.compile(r"https://github\.com/([^/]+)/([^/.]+)(?:\.git)?")

    def __init__(self, config: Optional[GitHubConfig] = None):
        """Initialize GitHub client.

        Args:
            config: Optional configuration
        """
        self.config = config or GitHubConfig()

    def get_pr(self, pr_number: int) -> GitHubPR:
        """Fetch PR information.

        Args:
            pr_number: PR number

        Returns:
            GitHubPR instance

        Raises:
            GitHubError: If fetch fails
        """
        cmd = self._build_pr_view_cmd(pr_number)
        result = self._run_command(cmd)

        try:
            data = json.loads(result)
            return GitHubPR.from_json(data)
        except json.JSONDecodeError as e:
            raise GitHubError(f"Failed to parse PR data: {e}")

    def get_pr_diff(self, pr_number: int) -> str:
        """Fetch PR diff.

        Args:
            pr_number: PR number

        Returns:
            Diff text

        Raises:
            GitHubError: If fetch fails
        """
        cmd = self._build_pr_diff_cmd(pr_number)
        return self._run_command(cmd)

    def get_pr_comments(self, pr_number: int) -> list[GitHubComment]:
        """Fetch PR comments.

        Args:
            pr_number: PR number

        Returns:
            List of GitHubComment instances

        Raises:
            GitHubError: If fetch fails
        """
        cmd = self._build_pr_comments_cmd(pr_number)
        result = self._run_command(cmd)

        try:
            data = json.loads(result)
            return [GitHubComment.from_json(c) for c in data]
        except json.JSONDecodeError as e:
            raise GitHubError(f"Failed to parse comments: {e}")

    def post_comment(self, pr_number: int, body: str) -> bool:
        """Post a general comment on a PR.

        Args:
            pr_number: PR number
            body: Comment body

        Returns:
            True if successful

        Raises:
            GitHubError: If post fails
        """
        cmd = self._build_comment_cmd(pr_number, body)
        self._run_command(cmd)
        return True

    def post_review_comment(
        self,
        pr_number: int,
        body: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> bool:
        """Post an inline review comment.

        Args:
            pr_number: PR number
            body: Comment body
            path: File path
            line: Line number
            side: Side of diff (LEFT or RIGHT)

        Returns:
            True if successful

        Raises:
            GitHubError: If post fails
        """
        # Use gh api for inline comments
        owner, repo = self._get_repo_info()
        cmd = [
            "gh", "api",
            "-X", "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            "-f", f"body={body}",
            "-f", f"path={path}",
            "-F", f"line={line}",
            "-f", f"side={side}",
            "-f", "commit_id=$(gh pr view {pr_number} --json headRefOid -q .headRefOid)",
        ]
        self._run_command(cmd)
        return True

    def post_review(
        self,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
    ) -> bool:
        """Post a review on a PR.

        Args:
            pr_number: PR number
            body: Review body
            event: Review event (APPROVE, REQUEST_CHANGES, COMMENT)

        Returns:
            True if successful

        Raises:
            GitHubError: If post fails
        """
        cmd = self._build_review_cmd(pr_number, body, event)
        self._run_command(cmd)
        return True

    def _get_repo_info(self) -> tuple[str, str]:
        """Get owner and repo, detecting from git if needed."""
        if self.config.owner and self.config.repo:
            return self.config.owner, self.config.repo

        return self._detect_repo()

    def _detect_repo(self) -> tuple[str, str]:
        """Detect repo from git remote.

        Returns:
            Tuple of (owner, repo)

        Raises:
            GitHubError: If detection fails
        """
        cmd = ["git", "remote", "get-url", "origin"]
        result = self._run_command(cmd)

        # Try SSH format
        match = self.SSH_REMOTE_RE.match(result.strip())
        if match:
            return match.group(1), match.group(2)

        # Try HTTPS format
        match = self.HTTPS_REMOTE_RE.match(result.strip())
        if match:
            return match.group(1), match.group(2)

        raise GitHubError(f"Could not parse git remote: {result}")

    def _build_pr_view_cmd(self, pr_number: int) -> list[str]:
        """Build gh pr view command."""
        cmd = ["gh", "pr", "view", str(pr_number), "--json",
               "number,title,body,state,author,baseRefName,headRefName,"
               "additions,deletions,changedFiles,url,createdAt,updatedAt"]

        owner, repo = self._get_repo_info()
        cmd.extend(["-R", f"{owner}/{repo}"])

        return cmd

    def _build_pr_diff_cmd(self, pr_number: int) -> list[str]:
        """Build gh pr diff command."""
        cmd = ["gh", "pr", "diff", str(pr_number)]

        owner, repo = self._get_repo_info()
        cmd.extend(["-R", f"{owner}/{repo}"])

        return cmd

    def _build_pr_comments_cmd(self, pr_number: int) -> list[str]:
        """Build command to fetch PR comments."""
        owner, repo = self._get_repo_info()
        return [
            "gh", "api",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            "--jq", "[.[] | {id, author: {login: .user.login}, body, createdAt: .created_at, path, line}]",
        ]

    def _build_comment_cmd(self, pr_number: int, body: str) -> list[str]:
        """Build gh pr comment command."""
        cmd = ["gh", "pr", "comment", str(pr_number), "-b", body]

        owner, repo = self._get_repo_info()
        cmd.extend(["-R", f"{owner}/{repo}"])

        return cmd

    def _build_review_cmd(self, pr_number: int, body: str, event: str) -> list[str]:
        """Build gh pr review command."""
        cmd = ["gh", "pr", "review", str(pr_number), "-b", body]

        if event == "APPROVE":
            cmd.append("--approve")
        elif event == "REQUEST_CHANGES":
            cmd.append("--request-changes")
        else:
            cmd.append("--comment")

        owner, repo = self._get_repo_info()
        cmd.extend(["-R", f"{owner}/{repo}"])

        return cmd

    def _run_command(self, cmd: list[str]) -> str:
        """Run a command and return output.

        Args:
            cmd: Command to run

        Returns:
            Command stdout

        Raises:
            GitHubError: If command fails
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.lower()
                if "not logged in" in stderr or "authentication" in stderr:
                    raise GitHubAuthError(f"Authentication failed: {result.stderr}")
                if "could not resolve" in stderr.lower() or "not found" in stderr:
                    raise GitHubNotFoundError(f"Resource not found: {result.stderr}")
                raise GitHubError(f"Command failed: {result.stderr}")

            return result.stdout

        except FileNotFoundError:
            raise GitHubError("gh CLI not found. Please install: https://cli.github.com/")
        except subprocess.TimeoutExpired:
            raise GitHubError(f"Command timeout after {self.config.timeout}s")


# =============================================================================
# Factory Function
# =============================================================================


def create_github_client(
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    repo_string: Optional[str] = None,
    timeout: int = 30,
) -> GitHubClient:
    """Create a GitHub client.

    Args:
        owner: Repository owner
        repo: Repository name
        repo_string: Alternative owner/repo string
        timeout: Command timeout

    Returns:
        Configured GitHubClient
    """
    if repo_string:
        config = GitHubConfig.from_repo_string(repo_string, timeout)
    else:
        config = GitHubConfig(owner=owner, repo=repo, timeout=timeout)

    return GitHubClient(config)
