"""Review Comment Suggestions Tool.

This module provides the suggest_review_comments MCP tool:
- ReviewCommentConfig: Configuration for the tool
- ReviewCommentInput: Input parameters
- ReviewComment: A single review comment
- CommentType: Types of comments
- ReviewSuggestion: Suggested review with comments
- ReviewCommentTool: Main tool entry point
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from tools.pr_workflow.pr_parser import DiffParser, PRDiff, PRFile, PRHunk
except ImportError:
    from openmemory.api.tools.pr_workflow.pr_parser import DiffParser, PRDiff, PRFile, PRHunk

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ReviewCommentError(Exception):
    """Base exception for review comment tool errors."""

    pass


class InvalidReviewInputError(ReviewCommentError):
    """Raised when input is invalid."""

    pass


# =============================================================================
# Configuration
# =============================================================================


SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


@dataclass
class ReviewCommentConfig:
    """Configuration for suggest_review_comments tool.

    Args:
        check_todos: Check for TODO/FIXME/HACK comments
        check_security: Check for security issues
        check_style: Check for style issues
        check_conventions: Check for convention violations
        max_comments: Maximum number of comments to return
        min_severity: Minimum severity level to include
    """

    check_todos: bool = True
    check_security: bool = True
    check_style: bool = True
    check_conventions: bool = True
    max_comments: int = 20
    min_severity: str = "info"


# =============================================================================
# Enums
# =============================================================================


class CommentType(Enum):
    """Types of review comments."""

    SUGGESTION = "suggestion"
    QUESTION = "question"
    ISSUE = "issue"
    PRAISE = "praise"
    NIT = "nit"


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class ReviewCommentInput:
    """Input parameters for suggest_review_comments.

    Args:
        repo_id: Repository ID (required)
        diff: The diff text to review
        title: PR title for context
        description: PR description for context
        target_files: Optional list of files to focus on
    """

    repo_id: str = ""
    diff: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    target_files: list[str] = field(default_factory=list)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ReviewComment:
    """A single review comment.

    Args:
        file_path: Path to the file
        line_number: Line number in the file
        comment_type: Type of comment
        body: Comment body text
        severity: Severity level (info, warning, error, critical)
        end_line_number: Optional end line for multi-line comments
        suggested_change: Optional code suggestion
    """

    file_path: str
    line_number: Optional[int]
    comment_type: CommentType
    body: str
    severity: str = "info"
    end_line_number: Optional[int] = None
    suggested_change: Optional[str] = None


@dataclass
class ReviewSuggestion:
    """Complete review suggestion.

    Args:
        comments: List of review comments
        overall_assessment: Overall assessment of the PR
        request_changes: Whether to request changes
        request_id: Unique request ID
    """

    comments: list[ReviewComment] = field(default_factory=list)
    overall_assessment: str = ""
    request_changes: bool = False
    request_id: str = ""


# =============================================================================
# Detection Patterns
# =============================================================================

TODO_PATTERNS = [
    {
        "pattern": re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)[:!]?\s*(.*)$", re.IGNORECASE),
        "severity": "info",
        "message_template": "Found {type} comment: {text}",
    },
]

SECURITY_PATTERNS = [
    {
        "pattern": re.compile(r'f["\'].*SELECT.*\{', re.IGNORECASE),
        "message": "Potential SQL injection: using f-string in SQL query",
        "severity": "error",
        "suggestion": "Use parameterized queries instead of string formatting",
    },
    {
        "pattern": re.compile(r'\.execute\s*\(\s*f["\']', re.IGNORECASE),
        "message": "Potential SQL injection: execute with f-string",
        "severity": "error",
        "suggestion": "Use execute(query, params) with parameter binding",
    },
    {
        "pattern": re.compile(r'eval\s*\('),
        "message": "Use of eval() is dangerous - can execute arbitrary code",
        "severity": "error",
        "suggestion": "Consider using ast.literal_eval() or safer alternatives",
    },
    {
        "pattern": re.compile(r'exec\s*\('),
        "message": "Use of exec() is dangerous - can execute arbitrary code",
        "severity": "error",
        "suggestion": "Avoid dynamic code execution if possible",
    },
    {
        "pattern": re.compile(r'subprocess\..*shell\s*=\s*True'),
        "message": "shell=True in subprocess is a security risk",
        "severity": "warning",
        "suggestion": "Use shell=False and pass command as a list",
    },
    {
        "pattern": re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
        "message": "Possible hardcoded password",
        "severity": "error",
        "suggestion": "Use environment variables or a secrets manager",
    },
]

STYLE_PATTERNS = [
    {
        "max_line_length": 120,
        "message": "Line exceeds {max} characters ({actual} chars)",
        "severity": "info",
    },
]


# =============================================================================
# Main Tool
# =============================================================================


class ReviewCommentTool:
    """MCP tool for suggesting review comments.

    Analyzes PR diff to suggest:
    - Code quality improvements
    - Security issue warnings
    - Style and convention violations
    - TODO/FIXME tracking
    """

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        config: Optional[ReviewCommentConfig] = None,
    ):
        """Initialize review comment tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            retriever: Retriever for code search
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.config = config or ReviewCommentConfig()
        self.diff_parser = DiffParser()

    def suggest(
        self,
        input_data: ReviewCommentInput,
        config: Optional[ReviewCommentConfig] = None,
    ) -> ReviewSuggestion:
        """Suggest review comments for a PR.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            ReviewSuggestion with comments and assessment

        Raises:
            ReviewCommentError: If suggestion fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Parse the diff
        diff_text = input_data.diff or ""
        if not diff_text:
            return ReviewSuggestion(
                comments=[],
                overall_assessment="No changes to review.",
                request_changes=False,
                request_id=request_id,
            )

        parsed_diff = self.diff_parser.parse(diff_text)

        # Collect comments
        comments: list[ReviewComment] = []

        # Check for TODOs
        if cfg.check_todos:
            todo_comments = self._check_todos(parsed_diff)
            comments.extend(todo_comments)

        # Check for security issues
        if cfg.check_security:
            security_comments = self._check_security(parsed_diff)
            comments.extend(security_comments)

        # Check for style issues
        if cfg.check_style:
            style_comments = self._check_style(parsed_diff)
            comments.extend(style_comments)

        # Filter by severity
        min_severity = SEVERITY_ORDER.get(cfg.min_severity, 0)
        comments = [
            c for c in comments
            if SEVERITY_ORDER.get(c.severity, 0) >= min_severity
        ]

        # Sort by severity (highest first) then by file and line
        comments.sort(
            key=lambda c: (
                -SEVERITY_ORDER.get(c.severity, 0),
                c.file_path,
                c.line_number or 0,
            )
        )

        # Limit comments
        comments = comments[: cfg.max_comments]

        # Determine if changes should be requested
        has_errors = any(c.severity in ("error", "critical") for c in comments)
        request_changes = has_errors

        # Generate overall assessment
        assessment = self._generate_assessment(comments, parsed_diff, request_changes)

        return ReviewSuggestion(
            comments=comments,
            overall_assessment=assessment,
            request_changes=request_changes,
            request_id=request_id,
        )

    def _validate_input(self, input_data: ReviewCommentInput) -> None:
        """Validate input parameters."""
        if input_data.diff is None:
            raise ReviewCommentError("diff is required")

    def _check_todos(self, diff: PRDiff) -> list[ReviewComment]:
        """Check for TODO/FIXME/HACK comments."""
        comments: list[ReviewComment] = []

        for file in diff.files:
            for hunk in file.hunks:
                for line in hunk.lines:
                    if line.line_type != "addition":
                        continue

                    for pattern_def in TODO_PATTERNS:
                        match = pattern_def["pattern"].search(line.content)
                        if match:
                            todo_type = match.group(1).upper()
                            todo_text = match.group(2).strip() if match.lastindex >= 2 else ""
                            message = pattern_def["message_template"].format(
                                type=todo_type,
                                text=todo_text or "(no description)",
                            )
                            comments.append(
                                ReviewComment(
                                    file_path=file.path,
                                    line_number=line.new_line_number,
                                    comment_type=CommentType.NIT,
                                    body=message,
                                    severity=pattern_def["severity"],
                                )
                            )

        return comments

    def _check_security(self, diff: PRDiff) -> list[ReviewComment]:
        """Check for security issues."""
        comments: list[ReviewComment] = []

        for file in diff.files:
            for hunk in file.hunks:
                for line in hunk.lines:
                    if line.line_type != "addition":
                        continue

                    for pattern_def in SECURITY_PATTERNS:
                        if pattern_def["pattern"].search(line.content):
                            comments.append(
                                ReviewComment(
                                    file_path=file.path,
                                    line_number=line.new_line_number,
                                    comment_type=CommentType.ISSUE,
                                    body=pattern_def["message"],
                                    severity=pattern_def["severity"],
                                    suggested_change=pattern_def.get("suggestion"),
                                )
                            )

        return comments

    def _check_style(self, diff: PRDiff) -> list[ReviewComment]:
        """Check for style issues."""
        comments: list[ReviewComment] = []

        for file in diff.files:
            for hunk in file.hunks:
                for line in hunk.lines:
                    if line.line_type != "addition":
                        continue

                    # Check line length (accounting for the + prefix)
                    content = line.content[1:] if line.content.startswith("+") else line.content
                    max_length = 120

                    if len(content) > max_length:
                        comments.append(
                            ReviewComment(
                                file_path=file.path,
                                line_number=line.new_line_number,
                                comment_type=CommentType.NIT,
                                body=f"Line exceeds {max_length} characters ({len(content)} chars)",
                                severity="info",
                            )
                        )

        return comments

    def _generate_assessment(
        self,
        comments: list[ReviewComment],
        diff: PRDiff,
        request_changes: bool,
    ) -> str:
        """Generate overall assessment."""
        if not comments:
            return "LGTM! No issues found in this PR."

        error_count = len([c for c in comments if c.severity in ("error", "critical")])
        warning_count = len([c for c in comments if c.severity == "warning"])
        info_count = len([c for c in comments if c.severity == "info"])

        parts = []

        if error_count:
            parts.append(f"{error_count} error(s) that should be addressed")
        if warning_count:
            parts.append(f"{warning_count} warning(s)")
        if info_count:
            parts.append(f"{info_count} suggestion(s)")

        summary = ", ".join(parts) if parts else "No significant issues"

        if request_changes:
            return f"Changes requested: {summary}. Please address the errors before merging."
        else:
            return f"Overall looks good. {summary}."

    def get_mcp_schema(self) -> dict[str, Any]:
        """Get MCP tool schema."""
        return {
            "name": "suggest_review_comments",
            "description": "Suggest review comments for a pull request based on code analysis",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": "Repository ID",
                    },
                    "diff": {
                        "type": "string",
                        "description": "The diff text to review",
                    },
                    "title": {
                        "type": "string",
                        "description": "PR title for context",
                    },
                    "description": {
                        "type": "string",
                        "description": "PR description for context",
                    },
                    "check_todos": {
                        "type": "boolean",
                        "description": "Check for TODO/FIXME comments",
                        "default": True,
                    },
                    "check_security": {
                        "type": "boolean",
                        "description": "Check for security issues",
                        "default": True,
                    },
                    "max_comments": {
                        "type": "integer",
                        "description": "Maximum number of comments to return",
                        "default": 20,
                    },
                },
                "required": ["repo_id", "diff"],
            },
        }

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute tool via MCP interface."""
        # Build config from input
        config = ReviewCommentConfig(
            check_todos=input_data.get("check_todos", True),
            check_security=input_data.get("check_security", True),
            check_style=input_data.get("check_style", True),
            check_conventions=input_data.get("check_conventions", True),
            max_comments=input_data.get("max_comments", 20),
            min_severity=input_data.get("min_severity", "info"),
        )

        # Build input
        review_input = ReviewCommentInput(
            repo_id=input_data.get("repo_id", ""),
            diff=input_data.get("diff"),
            title=input_data.get("title"),
            description=input_data.get("description"),
        )

        result = self.suggest(review_input, config)

        # Convert to dict
        return {
            "comments": [
                {
                    "file_path": c.file_path,
                    "line_number": c.line_number,
                    "end_line_number": c.end_line_number,
                    "comment_type": c.comment_type.value,
                    "body": c.body,
                    "severity": c.severity,
                    "suggested_change": c.suggested_change,
                }
                for c in result.comments
            ],
            "overall_assessment": result.overall_assessment,
            "request_changes": result.request_changes,
            "request_id": result.request_id,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_review_comment_tool(
    graph_driver: Any,
    retriever: Any,
    config: Optional[ReviewCommentConfig] = None,
) -> ReviewCommentTool:
    """Create a review comment tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        retriever: Retriever for code search
        config: Optional configuration

    Returns:
        Configured ReviewCommentTool
    """
    return ReviewCommentTool(
        graph_driver=graph_driver,
        retriever=retriever,
        config=config,
    )
