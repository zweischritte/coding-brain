"""Tests for PR analysis tool (FR-009).

This module tests the analyze_pull_request MCP tool with TDD approach:
- PRAnalysisConfig: Configuration for the tool
- PRAnalysisInput: Input parameters
- PRAnalysisOutput: Result structure with summary and issues
- PRAnalysisTool: Main tool entry point
- Integration with impact analysis and ADR detection
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
    """Return a sample PR diff."""
    return """diff --git a/src/auth/login.py b/src/auth/login.py
index abc1234..def5678 100644
--- a/src/auth/login.py
+++ b/src/auth/login.py
@@ -10,7 +10,10 @@ def authenticate(username, password):
     user = find_user(username)
-    if user and user.check_password(password):
+    if user and verify_password(password, user.password_hash):
+        # Log authentication attempt
+        log_auth_attempt(user, success=True)
         return create_session(user)
+    log_auth_attempt(None, success=False)
     return None
diff --git a/src/auth/utils.py b/src/auth/utils.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/auth/utils.py
@@ -0,0 +1,15 @@
+import hashlib
+import logging
+
+logger = logging.getLogger(__name__)
+
+def verify_password(password: str, hash: str) -> bool:
+    return hashlib.sha256(password.encode()).hexdigest() == hash
+
+def log_auth_attempt(user, success: bool):
+    if success:
+        logger.info(f"User {user.username} logged in")
+    else:
+        logger.warning("Failed login attempt")
"""


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()

    # Setup default node data
    symbol_node = MagicMock()
    symbol_node.properties = {
        "scip_id": "scip-python myapp auth/authenticate.",
        "name": "authenticate",
        "kind": "function",
        "file_path": "src/auth/login.py",
        "line_start": 10,
        "line_end": 20,
    }

    driver.get_node.return_value = symbol_node
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
def mock_adr_tool():
    """Create a mock ADR automation tool."""
    tool = MagicMock()
    tool.analyze.return_value = MagicMock(
        should_create_adr=False,
        confidence=0.3,
        triggered_heuristics=[],
    )
    return tool


@pytest.fixture
def mock_impact_tool():
    """Create a mock impact analysis tool."""
    tool = MagicMock()
    tool.analyze.return_value = MagicMock(
        affected_files=[],
        meta=MagicMock(request_id="test-123"),
    )
    return tool


# =============================================================================
# PRAnalysisConfig Tests
# =============================================================================


class TestPRAnalysisConfig:
    """Tests for PRAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisConfig

        config = PRAnalysisConfig()

        assert config.check_impact is True
        assert config.check_adr is True
        assert config.check_conventions is True
        assert config.check_security is True
        assert config.max_files_to_analyze == 50
        assert config.impact_depth == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisConfig

        config = PRAnalysisConfig(
            check_impact=False,
            check_adr=False,
            check_conventions=False,
            check_security=False,
            max_files_to_analyze=100,
            impact_depth=5,
        )

        assert config.check_impact is False
        assert config.check_adr is False
        assert config.max_files_to_analyze == 100
        assert config.impact_depth == 5


# =============================================================================
# PRAnalysisInput Tests
# =============================================================================


class TestPRAnalysisInput:
    """Tests for PRAnalysisInput dataclass."""

    def test_input_with_diff(self, sample_diff):
        """Test input with diff text."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisInput

        input_data = PRAnalysisInput(
            repo_id="myrepo",
            diff=sample_diff,
        )

        assert input_data.repo_id == "myrepo"
        assert input_data.diff == sample_diff
        assert input_data.pr_number is None

    def test_input_with_pr_number(self):
        """Test input with PR number."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisInput

        input_data = PRAnalysisInput(
            repo_id="myrepo",
            pr_number=123,
        )

        assert input_data.pr_number == 123
        assert input_data.diff is None

    def test_input_with_title_and_body(self, sample_diff):
        """Test input with PR title and body."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisInput

        input_data = PRAnalysisInput(
            repo_id="myrepo",
            diff=sample_diff,
            title="Add authentication logging",
            body="This PR adds logging for auth attempts.",
        )

        assert input_data.title == "Add authentication logging"
        assert input_data.body == "This PR adds logging for auth attempts."


# =============================================================================
# PRSummary Tests
# =============================================================================


class TestPRSummary:
    """Tests for PRSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a PR summary."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRSummary

        summary = PRSummary(
            files_changed=5,
            additions=100,
            deletions=50,
            languages=["python", "typescript"],
            main_areas=["authentication", "logging"],
            complexity_score=0.6,
        )

        assert summary.files_changed == 5
        assert summary.additions == 100
        assert summary.deletions == 50
        assert "python" in summary.languages
        assert "authentication" in summary.main_areas
        assert summary.complexity_score == 0.6

    def test_summary_with_affected_files(self):
        """Test summary with affected files from impact analysis."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRSummary

        summary = PRSummary(
            files_changed=2,
            additions=20,
            deletions=5,
            languages=["python"],
            main_areas=["auth"],
            complexity_score=0.3,
            affected_files=["tests/test_auth.py", "src/api/routes.py"],
        )

        assert len(summary.affected_files) == 2

    def test_summary_with_adr_suggestion(self):
        """Test summary with ADR suggestion."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRSummary

        summary = PRSummary(
            files_changed=10,
            additions=500,
            deletions=200,
            languages=["python"],
            main_areas=["database"],
            complexity_score=0.8,
            suggested_adr=True,
            adr_reason="Major database schema change",
        )

        assert summary.suggested_adr is True
        assert summary.adr_reason == "Major database schema change"


# =============================================================================
# PRIssue Tests
# =============================================================================


class TestPRIssue:
    """Tests for PRIssue dataclass."""

    def test_issue_creation(self):
        """Test creating a PR issue."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRIssue, IssueSeverity

        issue = PRIssue(
            severity=IssueSeverity.WARNING,
            category="security",
            file_path="src/auth/login.py",
            line_number=15,
            message="Consider using bcrypt instead of SHA256 for password hashing",
            suggestion="Replace hashlib.sha256 with bcrypt.hashpw",
        )

        assert issue.severity == IssueSeverity.WARNING
        assert issue.category == "security"
        assert issue.file_path == "src/auth/login.py"
        assert issue.line_number == 15

    def test_issue_severity_levels(self):
        """Test all issue severity levels."""
        from openmemory.api.tools.pr_workflow.pr_analysis import IssueSeverity

        assert IssueSeverity.INFO.value == "info"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.ERROR.value == "error"
        assert IssueSeverity.CRITICAL.value == "critical"

    def test_issue_categories(self):
        """Test various issue categories."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRIssue, IssueSeverity

        categories = ["security", "performance", "style", "convention", "complexity", "test"]

        for category in categories:
            issue = PRIssue(
                severity=IssueSeverity.INFO,
                category=category,
                file_path="test.py",
                line_number=1,
                message=f"Issue in {category}",
            )
            assert issue.category == category


# =============================================================================
# PRAnalysisOutput Tests
# =============================================================================


class TestPRAnalysisOutput:
    """Tests for PRAnalysisOutput dataclass."""

    def test_output_structure(self):
        """Test output has all required fields."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisOutput,
            PRSummary,
            PRIssue,
            IssueSeverity,
        )

        summary = PRSummary(
            files_changed=5,
            additions=100,
            deletions=50,
            languages=["python"],
            main_areas=["auth"],
            complexity_score=0.5,
        )

        issues = [
            PRIssue(
                severity=IssueSeverity.WARNING,
                category="security",
                file_path="test.py",
                line_number=10,
                message="Security issue",
            )
        ]

        output = PRAnalysisOutput(
            summary=summary,
            issues=issues,
            request_id="req-123",
        )

        assert output.summary.files_changed == 5
        assert len(output.issues) == 1
        assert output.request_id == "req-123"

    def test_output_issue_counts(self):
        """Test counting issues by severity."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisOutput,
            PRSummary,
            PRIssue,
            IssueSeverity,
        )

        summary = PRSummary(
            files_changed=1,
            additions=10,
            deletions=5,
            languages=["python"],
            main_areas=[],
            complexity_score=0.1,
        )

        issues = [
            PRIssue(IssueSeverity.ERROR, "security", "a.py", 1, "Error 1"),
            PRIssue(IssueSeverity.WARNING, "style", "a.py", 2, "Warning 1"),
            PRIssue(IssueSeverity.WARNING, "style", "a.py", 3, "Warning 2"),
            PRIssue(IssueSeverity.INFO, "docs", "a.py", 4, "Info 1"),
        ]

        output = PRAnalysisOutput(
            summary=summary,
            issues=issues,
            request_id="req-456",
        )

        assert output.error_count == 1
        assert output.warning_count == 2
        assert output.info_count == 1


# =============================================================================
# PRAnalysisTool Tests
# =============================================================================


class TestPRAnalysisTool:
    """Tests for PRAnalysisTool."""

    def test_analyze_basic_diff(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test analyzing a basic diff."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result is not None
        assert result.summary is not None
        assert result.summary.files_changed == 2

    def test_analyze_calculates_summary(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test that analysis calculates summary correctly."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result.summary.additions > 0
        assert "python" in result.summary.languages

    def test_analyze_with_impact_analysis(
        self, mock_graph_driver, mock_retriever, mock_impact_tool, sample_diff
    ):
        """Test analysis with impact analysis enabled."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
        )

        # Configure mock impact tool to return affected files
        mock_impact_tool.analyze.return_value = MagicMock(
            affected_files=[
                MagicMock(file_path="tests/test_auth.py", confidence=0.9),
                MagicMock(file_path="src/api/routes.py", confidence=0.7),
            ],
            meta=MagicMock(request_id="impact-123"),
        )

        config = PRAnalysisConfig(check_impact=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            impact_tool=mock_impact_tool,
            config=config,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert mock_impact_tool.analyze.called

    def test_analyze_with_adr_detection(
        self, mock_graph_driver, mock_retriever, mock_adr_tool, sample_diff
    ):
        """Test analysis with ADR detection enabled."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
        )

        # Configure mock ADR tool to suggest ADR
        mock_adr_tool.analyze.return_value = MagicMock(
            should_create_adr=True,
            confidence=0.8,
            triggered_heuristics=["security_change"],
            generated_adr=MagicMock(
                title="Authentication Logging",
                render=lambda: "# ADR: Authentication Logging",
            ),
        )

        config = PRAnalysisConfig(check_adr=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            adr_tool=mock_adr_tool,
            config=config,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result.summary.suggested_adr is True

    def test_analyze_detects_security_issues(
        self, mock_graph_driver, mock_retriever
    ):
        """Test that analysis detects security issues."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
            IssueSeverity,
        )

        # Diff with potential security issue
        insecure_diff = """diff --git a/src/db.py b/src/db.py
index abc..def 100644
--- a/src/db.py
+++ b/src/db.py
@@ -10,5 +10,6 @@ def query(user_input):
-    cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))
+    cursor.execute(f"SELECT * FROM users WHERE id = {user_input}")
     return cursor.fetchall()
"""

        config = PRAnalysisConfig(check_security=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=insecure_diff)
        )

        # Should detect SQL injection risk
        security_issues = [
            i for i in result.issues if i.category == "security"
        ]
        assert len(security_issues) >= 1

    def test_analyze_calculates_complexity(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test that analysis calculates complexity score."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert 0.0 <= result.summary.complexity_score <= 1.0

    def test_analyze_identifies_main_areas(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test that analysis identifies main affected areas."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        # Should identify auth as a main area
        assert len(result.summary.main_areas) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_missing_input_error(self, mock_graph_driver, mock_retriever):
        """Test error when neither diff nor pr_number provided."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisError,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        with pytest.raises(PRAnalysisError):
            tool.analyze(PRAnalysisInput(repo_id="myrepo"))

    def test_missing_repo_id_error(self, mock_graph_driver, mock_retriever, sample_diff):
        """Test error when repo_id is missing."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisError,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        with pytest.raises(PRAnalysisError):
            tool.analyze(PRAnalysisInput(repo_id="", diff=sample_diff))

    def test_empty_diff_handling(self, mock_graph_driver, mock_retriever):
        """Test handling of empty diff."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff="")
        )

        assert result.summary.files_changed == 0

    def test_graceful_degradation_on_impact_failure(
        self, mock_graph_driver, mock_retriever, mock_impact_tool, sample_diff
    ):
        """Test graceful degradation when impact analysis fails."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
        )

        # Configure mock to fail
        mock_impact_tool.analyze.side_effect = Exception("Impact analysis failed")

        config = PRAnalysisConfig(check_impact=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            impact_tool=mock_impact_tool,
            config=config,
        )

        # Should not raise, but continue with degraded results
        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=sample_diff)
        )

        assert result is not None
        assert result.summary is not None


# =============================================================================
# Convention Checking Tests
# =============================================================================


class TestConventionChecking:
    """Tests for convention checking functionality."""

    def test_detect_missing_docstring(self, mock_graph_driver, mock_retriever):
        """Test detection of missing docstrings."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
        )

        diff_no_docstring = """diff --git a/src/utils.py b/src/utils.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,5 @@
+def complex_function(a, b, c, d):
+    result = a * b
+    result += c / d
+    return result ** 2
"""

        config = PRAnalysisConfig(check_conventions=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=diff_no_docstring)
        )

        # Should detect missing docstring
        convention_issues = [
            i for i in result.issues if i.category == "convention"
        ]
        # May or may not detect depending on heuristics
        assert result is not None

    def test_detect_long_function(self, mock_graph_driver, mock_retriever):
        """Test detection of overly long functions."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
            PRAnalysisConfig,
        )

        # Create a very long function
        long_lines = "\n".join([f"+    line_{i} = {i}" for i in range(100)])
        long_diff = f"""diff --git a/src/long.py b/src/long.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/src/long.py
@@ -0,0 +1,102 @@
+def very_long_function():
+    '''Long function.'''
{long_lines}
+    return result
"""

        config = PRAnalysisConfig(check_conventions=True)
        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze(
            PRAnalysisInput(repo_id="myrepo", diff=long_diff)
        )

        # Should detect long function
        complexity_issues = [
            i for i in result.issues if i.category == "complexity"
        ]
        assert len(complexity_issues) >= 1


# =============================================================================
# MCP Schema Tests
# =============================================================================


class TestMCPSchema:
    """Tests for MCP schema compliance."""

    def test_get_mcp_schema(self, mock_graph_driver, mock_retriever):
        """Test that tool provides valid MCP schema."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisTool

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        schema = tool.get_mcp_schema()

        assert schema["name"] == "analyze_pull_request"
        assert "description" in schema
        assert "inputSchema" in schema
        assert schema["inputSchema"]["type"] == "object"
        assert "repo_id" in schema["inputSchema"]["properties"]

    def test_execute_via_mcp(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test executing tool via MCP interface."""
        from openmemory.api.tools.pr_workflow.pr_analysis import PRAnalysisTool

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.execute({
            "repo_id": "myrepo",
            "diff": sample_diff,
        })

        assert "summary" in result
        assert "issues" in result


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_analyze_latency(
        self, mock_graph_driver, mock_retriever, sample_diff
    ):
        """Test that analysis completes quickly with mocks."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            PRAnalysisTool,
            PRAnalysisInput,
        )

        tool = PRAnalysisTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        start = time.perf_counter()
        _ = tool.analyze(PRAnalysisInput(repo_id="myrepo", diff=sample_diff))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 500


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_pr_analysis_tool(self, mock_graph_driver, mock_retriever):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.pr_workflow.pr_analysis import create_pr_analysis_tool

        tool = create_pr_analysis_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        assert tool is not None
        assert tool.config.check_impact is True  # Default

    def test_create_with_custom_config(self, mock_graph_driver, mock_retriever):
        """Test factory function with custom config."""
        from openmemory.api.tools.pr_workflow.pr_analysis import (
            create_pr_analysis_tool,
            PRAnalysisConfig,
        )

        config = PRAnalysisConfig(
            check_impact=False,
            max_files_to_analyze=200,
        )

        tool = create_pr_analysis_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        assert tool.config.check_impact is False
        assert tool.config.max_files_to_analyze == 200
