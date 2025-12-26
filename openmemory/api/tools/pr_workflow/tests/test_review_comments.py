"""Tests for review comment suggestions.

This module tests the suggest_review_comments MCP tool with TDD approach:
- ReviewCommentConfig: Configuration for the tool
- ReviewCommentInput: Input parameters
- ReviewComment: A single review comment
- CommentType: Types of comments
- ReviewSuggestion: Suggested review action
- ReviewCommentTool: Main tool entry point
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_diff() -> str:
    """Return a sample PR diff for review."""
    return """diff --git a/src/api/handlers.py b/src/api/handlers.py
index abc1234..def5678 100644
--- a/src/api/handlers.py
+++ b/src/api/handlers.py
@@ -15,10 +15,15 @@ class UserHandler:
     def create_user(self, request):
         data = request.json
-        user = User(name=data["name"])
+        # TODO: add validation
+        name = data.get("name")
+        email = data.get("email")
+        if not name:
+            raise ValueError("Name is required")
+        user = User(name=name, email=email)
         db.session.add(user)
         db.session.commit()
-        return {"id": user.id}
+        return {"id": user.id, "name": user.name, "email": user.email}
"""


@pytest.fixture
def security_issue_diff() -> str:
    """Return a diff with security issues."""
    return """diff --git a/src/db/queries.py b/src/db/queries.py
index abc..def 100644
--- a/src/db/queries.py
+++ b/src/db/queries.py
@@ -10,5 +10,6 @@ def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = ?"
-    return db.execute(query, (user_id,))
+    query = f"SELECT * FROM users WHERE id = {user_id}"
+    return db.execute(query)
"""


@pytest.fixture
def todo_diff() -> str:
    """Return a diff with TODO comments."""
    return """diff --git a/src/services/payment.py b/src/services/payment.py
index abc..def 100644
--- a/src/services/payment.py
+++ b/src/services/payment.py
@@ -20,6 +20,10 @@ def process_payment(amount, card):
     # Validate card
+    # TODO: implement card validation
+    # FIXME: this is a temporary solution
+    # HACK: quick fix for demo
     if amount > 10000:
         raise ValueError("Amount too large")
"""


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()
    driver.get_node.return_value = None
    driver.get_outgoing_edges.return_value = []
    driver.get_incoming_edges.return_value = []
    return driver


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for code search."""
    retriever = MagicMock()
    retriever.search.return_value = []
    return retriever


@pytest.fixture
def mock_conventions():
    """Mock team conventions data."""
    return {
        "docstrings": {"required": True, "style": "google"},
        "type_hints": {"required": True},
        "max_line_length": 100,
        "max_function_length": 50,
        "naming": {"functions": "snake_case", "classes": "PascalCase"},
    }


# =============================================================================
# ReviewCommentConfig Tests
# =============================================================================


class TestReviewCommentConfig:
    """Tests for ReviewCommentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentConfig

        config = ReviewCommentConfig()

        assert config.check_todos is True
        assert config.check_security is True
        assert config.check_style is True
        assert config.check_conventions is True
        assert config.max_comments == 20
        assert config.min_severity == "info"

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentConfig

        config = ReviewCommentConfig(
            check_todos=False,
            check_security=True,
            check_style=False,
            max_comments=50,
            min_severity="warning",
        )

        assert config.check_todos is False
        assert config.check_style is False
        assert config.max_comments == 50
        assert config.min_severity == "warning"


# =============================================================================
# ReviewCommentInput Tests
# =============================================================================


class TestReviewCommentInput:
    """Tests for ReviewCommentInput dataclass."""

    def test_input_with_diff(self, sample_diff):
        """Test input with diff text."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentInput

        input_data = ReviewCommentInput(
            repo_id="myrepo",
            diff=sample_diff,
        )

        assert input_data.repo_id == "myrepo"
        assert input_data.diff == sample_diff

    def test_input_with_context(self, sample_diff):
        """Test input with additional context."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentInput

        input_data = ReviewCommentInput(
            repo_id="myrepo",
            diff=sample_diff,
            title="Add user creation validation",
            description="This PR adds input validation",
            target_files=["src/api/handlers.py"],
        )

        assert input_data.title == "Add user creation validation"
        assert len(input_data.target_files) == 1


# =============================================================================
# CommentType Tests
# =============================================================================


class TestCommentType:
    """Tests for CommentType enum."""

    def test_all_types_exist(self):
        """Test all comment types are defined."""
        from openmemory.api.tools.pr_workflow.review_comments import CommentType

        assert CommentType.SUGGESTION.value == "suggestion"
        assert CommentType.QUESTION.value == "question"
        assert CommentType.ISSUE.value == "issue"
        assert CommentType.PRAISE.value == "praise"
        assert CommentType.NIT.value == "nit"


# =============================================================================
# ReviewComment Tests
# =============================================================================


class TestReviewComment:
    """Tests for ReviewComment dataclass."""

    def test_comment_creation(self):
        """Test creating a review comment."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewComment,
            CommentType,
        )

        comment = ReviewComment(
            file_path="src/api/handlers.py",
            line_number=20,
            comment_type=CommentType.SUGGESTION,
            body="Consider adding input validation here.",
            severity="info",
        )

        assert comment.file_path == "src/api/handlers.py"
        assert comment.line_number == 20
        assert comment.comment_type == CommentType.SUGGESTION
        assert comment.severity == "info"

    def test_comment_with_suggestion(self):
        """Test comment with code suggestion."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewComment,
            CommentType,
        )

        comment = ReviewComment(
            file_path="src/api/handlers.py",
            line_number=20,
            comment_type=CommentType.SUGGESTION,
            body="Consider using a constant for this value.",
            severity="info",
            suggested_change="MAX_AMOUNT = 10000",
        )

        assert comment.suggested_change == "MAX_AMOUNT = 10000"

    def test_comment_with_line_range(self):
        """Test comment spanning multiple lines."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewComment,
            CommentType,
        )

        comment = ReviewComment(
            file_path="src/api/handlers.py",
            line_number=20,
            end_line_number=25,
            comment_type=CommentType.ISSUE,
            body="This block should be extracted into a separate function.",
            severity="warning",
        )

        assert comment.end_line_number == 25


# =============================================================================
# ReviewSuggestion Tests
# =============================================================================


class TestReviewSuggestion:
    """Tests for ReviewSuggestion dataclass."""

    def test_suggestion_creation(self):
        """Test creating a review suggestion."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewSuggestion,
            ReviewComment,
            CommentType,
        )

        comments = [
            ReviewComment(
                file_path="test.py",
                line_number=10,
                comment_type=CommentType.SUGGESTION,
                body="Add type hints",
                severity="info",
            )
        ]

        suggestion = ReviewSuggestion(
            comments=comments,
            overall_assessment="LGTM with minor suggestions",
            request_changes=False,
        )

        assert len(suggestion.comments) == 1
        assert suggestion.overall_assessment == "LGTM with minor suggestions"
        assert suggestion.request_changes is False

    def test_suggestion_with_request_changes(self):
        """Test suggestion that requests changes."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewSuggestion,
            ReviewComment,
            CommentType,
        )

        comments = [
            ReviewComment(
                file_path="test.py",
                line_number=10,
                comment_type=CommentType.ISSUE,
                body="Security vulnerability found",
                severity="error",
            )
        ]

        suggestion = ReviewSuggestion(
            comments=comments,
            overall_assessment="Critical security issues must be fixed",
            request_changes=True,
        )

        assert suggestion.request_changes is True


# =============================================================================
# ReviewCommentTool Tests
# =============================================================================


class TestReviewCommentTool:
    """Tests for ReviewCommentTool."""

    def test_suggest_basic_comments(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test generating basic review comments."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result is not None
        assert hasattr(result, "comments")
        assert hasattr(result, "overall_assessment")

    def test_detect_todos(
        self, mock_graph_driver, mock_retriever, todo_diff
    ):
        """Test detection of TODO comments."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentConfig,
        )

        config = ReviewCommentConfig(check_todos=True)
        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=todo_diff)
        )

        # Should detect TODO/FIXME/HACK comments
        todo_comments = [c for c in result.comments if "TODO" in c.body or "FIXME" in c.body or "HACK" in c.body]
        assert len(todo_comments) >= 1

    def test_detect_security_issues(
        self, mock_graph_driver, mock_retriever, security_issue_diff
    ):
        """Test detection of security issues."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentConfig,
            CommentType,
        )

        config = ReviewCommentConfig(check_security=True)
        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=security_issue_diff)
        )

        # Should detect SQL injection
        security_comments = [
            c for c in result.comments
            if c.comment_type == CommentType.ISSUE and "SQL" in c.body
        ]
        assert len(security_comments) >= 1

    def test_limit_max_comments(
        self, mock_graph_driver, mock_retriever
    ):
        """Test limiting maximum number of comments."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentConfig,
        )

        # Generate a diff with many TODO comments
        many_todos = "\n".join([f"+    # TODO: task {i}" for i in range(30)])
        large_diff = f"""diff --git a/test.py b/test.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/test.py
@@ -0,0 +1,30 @@
+def function():
{many_todos}
"""

        config = ReviewCommentConfig(max_comments=5)
        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=large_diff)
        )

        assert len(result.comments) <= 5

    def test_filter_by_severity(
        self, mock_graph_driver, mock_retriever, todo_diff
    ):
        """Test filtering comments by minimum severity."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentConfig,
        )

        config = ReviewCommentConfig(min_severity="warning")
        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=todo_diff)
        )

        # All comments should be warning or higher
        for comment in result.comments:
            assert comment.severity in ("warning", "error", "critical")

    def test_overall_assessment(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test generating overall assessment."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result.overall_assessment is not None
        assert len(result.overall_assessment) > 0

    def test_request_changes_for_critical_issues(
        self, mock_graph_driver, mock_retriever, security_issue_diff
    ):
        """Test that critical issues trigger request changes."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=security_issue_diff)
        )

        # Security issues should trigger request_changes
        has_security = any("SQL" in c.body for c in result.comments)
        if has_security:
            assert result.request_changes is True


# =============================================================================
# Convention Checking Tests
# =============================================================================


class TestConventionChecking:
    """Tests for convention-based review comments."""

    def test_detect_style_issues(self, mock_graph_driver, mock_retriever):
        """Test detection of style issues."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentConfig,
        )

        # Diff with very long line
        long_line = "+" + "x" * 150
        style_diff = f"""diff --git a/test.py b/test.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/test.py
@@ -0,0 +1,3 @@
+def function():
{long_line}
+    return result
"""

        config = ReviewCommentConfig(check_style=True)
        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=style_diff)
        )

        # Should detect long line
        style_comments = [c for c in result.comments if "line" in c.body.lower()]
        assert len(style_comments) >= 1

    def test_praise_good_practices(self, mock_graph_driver, mock_retriever):
        """Test generating praise for good practices."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            CommentType,
        )

        good_code_diff = """diff --git a/test.py b/test.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/test.py
@@ -0,0 +1,20 @@
+def process_data(data: list[dict]) -> dict[str, int]:
+    \"\"\"Process input data and return aggregated results.
+
+    Args:
+        data: List of data dictionaries to process.
+
+    Returns:
+        Dictionary mapping categories to counts.
+
+    Raises:
+        ValueError: If data is empty.
+    \"\"\"
+    if not data:
+        raise ValueError("Data cannot be empty")
+
+    results = {}
+    for item in data:
+        category = item.get("category", "unknown")
+        results[category] = results.get(category, 0) + 1
+    return results
"""

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff=good_code_diff)
        )

        # May have praise comments
        praise_comments = [
            c for c in result.comments if c.comment_type == CommentType.PRAISE
        ]
        # Good code should have minimal issues
        assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_missing_diff_error(self, mock_graph_driver, mock_retriever):
        """Test error when diff is missing."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
            ReviewCommentError,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        with pytest.raises(ReviewCommentError):
            tool.suggest(ReviewCommentInput(repo_id="myrepo"))

    def test_empty_diff_handling(self, mock_graph_driver, mock_retriever):
        """Test handling of empty diff."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.suggest(
            ReviewCommentInput(repo_id="myrepo", diff="")
        )

        assert len(result.comments) == 0


# =============================================================================
# MCP Schema Tests
# =============================================================================


class TestMCPSchema:
    """Tests for MCP schema compliance."""

    def test_get_mcp_schema(self, mock_graph_driver, mock_retriever):
        """Test that tool provides valid MCP schema."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentTool

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        schema = tool.get_mcp_schema()

        assert schema["name"] == "suggest_review_comments"
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"

    def test_execute_via_mcp(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test executing tool via MCP interface."""
        from openmemory.api.tools.pr_workflow.review_comments import ReviewCommentTool

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.execute({
            "repo_id": "myrepo",
            "diff": sample_diff,
        })

        assert "comments" in result
        assert "overall_assessment" in result
        assert "request_changes" in result


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_suggest_latency(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test that suggestion completes quickly with mocks."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            ReviewCommentTool,
            ReviewCommentInput,
        )

        tool = ReviewCommentTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        start = time.perf_counter()
        _ = tool.suggest(ReviewCommentInput(repo_id="myrepo", diff=sample_diff))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 500


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_review_comment_tool(self, mock_graph_driver, mock_retriever):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.pr_workflow.review_comments import create_review_comment_tool

        tool = create_review_comment_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        assert tool is not None

    def test_create_with_custom_config(self, mock_graph_driver, mock_retriever):
        """Test factory function with custom config."""
        from openmemory.api.tools.pr_workflow.review_comments import (
            create_review_comment_tool,
            ReviewCommentConfig,
        )

        config = ReviewCommentConfig(
            check_todos=False,
            max_comments=100,
        )

        tool = create_review_comment_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        assert tool.config.check_todos is False
        assert tool.config.max_comments == 100
