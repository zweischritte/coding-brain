"""PR Analysis Tool (FR-009).

This module provides the analyze_pull_request MCP tool:
- PRAnalysisConfig: Configuration for the tool
- PRAnalysisInput: Input parameters
- PRAnalysisOutput: Result structure with summary and issues
- PRAnalysisTool: Main tool entry point

Integration points:
- openmemory.api.indexing.graph_projection: CODE_* graph queries
- openmemory.api.tools.impact_analysis: Impact analysis
- openmemory.api.tools.adr_automation: ADR detection
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openmemory.api.tools.pr_workflow.pr_parser import DiffParser, PRDiff, PRFile

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class PRAnalysisError(Exception):
    """Base exception for PR analysis tool errors."""

    pass


class InvalidPRInputError(PRAnalysisError):
    """Raised when input is invalid."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PRAnalysisConfig:
    """Configuration for analyze_pull_request tool.

    Args:
        check_impact: Run impact analysis on changes
        check_adr: Check if changes warrant an ADR
        check_conventions: Check for code convention violations
        check_security: Check for security issues
        max_files_to_analyze: Maximum files to analyze in detail
        impact_depth: Depth for impact analysis traversal
    """

    check_impact: bool = True
    check_adr: bool = True
    check_conventions: bool = True
    check_security: bool = True
    max_files_to_analyze: int = 50
    impact_depth: int = 3


# =============================================================================
# Enums
# =============================================================================


class IssueSeverity(Enum):
    """Severity levels for PR issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Input Types
# =============================================================================


@dataclass
class PRAnalysisInput:
    """Input parameters for analyze_pull_request.

    Either diff or pr_number must be provided.

    Args:
        repo_id: Repository ID (required)
        diff: The diff text to analyze
        pr_number: PR number to fetch from GitHub
        title: PR title for context
        body: PR body/description for context
        base_branch: Base branch name
        head_branch: Head branch name
    """

    repo_id: str = ""
    diff: Optional[str] = None
    pr_number: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    base_branch: Optional[str] = None
    head_branch: Optional[str] = None


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class PRSummary:
    """Summary of PR analysis.

    Args:
        files_changed: Number of files changed
        additions: Total lines added
        deletions: Total lines removed
        languages: Languages detected in changes
        main_areas: Main code areas affected
        complexity_score: Estimated review complexity (0-1)
        affected_files: Files affected by changes (from impact analysis)
        suggested_adr: Whether an ADR is suggested
        adr_reason: Reason for ADR suggestion
    """

    files_changed: int
    additions: int
    deletions: int
    languages: list[str] = field(default_factory=list)
    main_areas: list[str] = field(default_factory=list)
    complexity_score: float = 0.0
    affected_files: list[str] = field(default_factory=list)
    suggested_adr: bool = False
    adr_reason: Optional[str] = None


@dataclass
class PRIssue:
    """An issue found during PR analysis.

    Args:
        severity: Issue severity level
        category: Issue category (security, performance, style, etc.)
        file_path: Path to the file with the issue
        line_number: Line number in the file
        message: Description of the issue
        suggestion: Optional suggestion for fixing
    """

    severity: IssueSeverity
    category: str
    file_path: str
    line_number: Optional[int]
    message: str
    suggestion: Optional[str] = None


@dataclass
class PRAnalysisOutput:
    """Result from analyze_pull_request.

    Args:
        summary: Summary of the PR
        issues: List of issues found
        request_id: Unique request ID
    """

    summary: PRSummary
    issues: list[PRIssue] = field(default_factory=list)
    request_id: str = ""

    @property
    def error_count(self) -> int:
        """Count error-level issues."""
        return len([i for i in self.issues if i.severity == IssueSeverity.ERROR])

    @property
    def warning_count(self) -> int:
        """Count warning-level issues."""
        return len([i for i in self.issues if i.severity == IssueSeverity.WARNING])

    @property
    def info_count(self) -> int:
        """Count info-level issues."""
        return len([i for i in self.issues if i.severity == IssueSeverity.INFO])


# =============================================================================
# Security Patterns
# =============================================================================

SECURITY_PATTERNS = [
    {
        "pattern": re.compile(r'f["\'].*SELECT.*\{', re.IGNORECASE),
        "message": "Possible SQL injection vulnerability: using f-string in SQL query",
        "suggestion": "Use parameterized queries instead of string formatting",
    },
    {
        "pattern": re.compile(r'execute\s*\(\s*f["\']', re.IGNORECASE),
        "message": "Possible SQL injection: execute with f-string",
        "suggestion": "Use execute(query, params) with parameter binding",
    },
    {
        "pattern": re.compile(r'\.format\s*\(.*\).*SELECT', re.IGNORECASE),
        "message": "Possible SQL injection: .format() in SQL query",
        "suggestion": "Use parameterized queries instead of .format()",
    },
    {
        "pattern": re.compile(r'eval\s*\('),
        "message": "Use of eval() is dangerous and should be avoided",
        "suggestion": "Consider using ast.literal_eval() or safer alternatives",
    },
    {
        "pattern": re.compile(r'exec\s*\('),
        "message": "Use of exec() is dangerous and should be avoided",
        "suggestion": "Avoid dynamic code execution if possible",
    },
    {
        "pattern": re.compile(r'subprocess\..*shell\s*=\s*True'),
        "message": "shell=True in subprocess is a security risk",
        "suggestion": "Use shell=False and pass command as a list",
    },
    {
        "pattern": re.compile(r'pickle\.load'),
        "message": "Unpickling data can execute arbitrary code",
        "suggestion": "Only unpickle data from trusted sources",
    },
    {
        "pattern": re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
        "message": "Possible hardcoded password",
        "suggestion": "Use environment variables or a secrets manager",
    },
    {
        "pattern": re.compile(r'(api[_-]?key|secret[_-]?key|access[_-]?token)\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
        "message": "Possible hardcoded secret or API key",
        "suggestion": "Use environment variables or a secrets manager",
    },
]


# =============================================================================
# Main Tool
# =============================================================================


class PRAnalysisTool:
    """MCP tool for analyzing pull requests.

    Analyzes PR diff to provide:
    - Summary of changes
    - Impact analysis
    - ADR suggestions
    - Security issues
    - Convention violations
    """

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        impact_tool: Optional[Any] = None,
        adr_tool: Optional[Any] = None,
        config: Optional[PRAnalysisConfig] = None,
    ):
        """Initialize PR analysis tool.

        Args:
            graph_driver: Neo4j driver for graph queries
            retriever: Retriever for code search
            impact_tool: Optional impact analysis tool
            adr_tool: Optional ADR automation tool
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.impact_tool = impact_tool
        self.adr_tool = adr_tool
        self.config = config or PRAnalysisConfig()
        self.diff_parser = DiffParser()

    def analyze(
        self,
        input_data: PRAnalysisInput,
        config: Optional[PRAnalysisConfig] = None,
    ) -> PRAnalysisOutput:
        """Analyze a pull request.

        Args:
            input_data: Input parameters
            config: Optional config override

        Returns:
            PRAnalysisOutput with summary and issues

        Raises:
            PRAnalysisError: If analysis fails
        """
        cfg = config or self.config
        request_id = str(uuid.uuid4())

        # Validate input
        self._validate_input(input_data)

        # Parse the diff
        diff_text = input_data.diff or ""
        parsed_diff = self.diff_parser.parse(diff_text)

        # Build summary
        summary = self._build_summary(parsed_diff)

        # Collect issues
        issues: list[PRIssue] = []

        # Run security checks
        if cfg.check_security:
            security_issues = self._check_security(parsed_diff)
            issues.extend(security_issues)

        # Run convention checks
        if cfg.check_conventions:
            convention_issues = self._check_conventions(parsed_diff)
            issues.extend(convention_issues)

        # Run impact analysis if enabled and tool available
        if cfg.check_impact and self.impact_tool:
            try:
                affected = self._run_impact_analysis(
                    parsed_diff, input_data.repo_id, cfg.impact_depth
                )
                summary.affected_files = affected
            except Exception as e:
                logger.warning(f"Impact analysis failed: {e}")

        # Check for ADR if enabled and tool available
        if cfg.check_adr and self.adr_tool:
            try:
                adr_result = self._check_adr(parsed_diff)
                if adr_result:
                    summary.suggested_adr = True
                    summary.adr_reason = adr_result
            except Exception as e:
                logger.warning(f"ADR check failed: {e}")

        return PRAnalysisOutput(
            summary=summary,
            issues=issues,
            request_id=request_id,
        )

    def _validate_input(self, input_data: PRAnalysisInput) -> None:
        """Validate input parameters."""
        if not input_data.repo_id or not input_data.repo_id.strip():
            raise PRAnalysisError("repo_id is required")

        if input_data.diff is None and not input_data.pr_number:
            raise PRAnalysisError("Either diff or pr_number is required")

    def _build_summary(self, diff: PRDiff) -> PRSummary:
        """Build summary from parsed diff."""
        # Collect languages
        languages = list(set(f.language for f in diff.files if f.language != "unknown"))

        # Identify main areas from file paths
        main_areas = self._identify_main_areas(diff.files)

        # Calculate complexity score
        complexity = self._calculate_complexity(diff)

        return PRSummary(
            files_changed=diff.files_changed,
            additions=diff.total_additions,
            deletions=diff.total_deletions,
            languages=languages,
            main_areas=main_areas,
            complexity_score=complexity,
        )

    def _identify_main_areas(self, files: list[PRFile]) -> list[str]:
        """Identify main code areas from file paths."""
        areas: dict[str, int] = {}

        for f in files:
            path = f.path
            if not path:
                continue

            # Extract directory parts
            parts = path.split("/")
            if len(parts) >= 2:
                # Use first non-trivial directory as area
                area = parts[0] if parts[0] not in ("src", "lib", "app") else parts[1] if len(parts) > 1 else parts[0]
                areas[area] = areas.get(area, 0) + 1

        # Return top 3 areas
        sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        return [area for area, _ in sorted_areas[:3]]

    def _calculate_complexity(self, diff: PRDiff) -> float:
        """Calculate review complexity score (0-1)."""
        # Factors:
        # - Number of files
        # - Total changes
        # - Number of languages
        # - Presence of binary files

        files = diff.files_changed
        changes = diff.total_additions + diff.total_deletions
        languages = len(set(f.language for f in diff.files if f.language != "unknown"))
        binary_files = len([f for f in diff.files if f.is_binary])

        # Normalize each factor
        file_score = min(files / 20, 1.0)  # 20+ files = max
        change_score = min(changes / 500, 1.0)  # 500+ changes = max
        lang_score = min(languages / 3, 1.0)  # 3+ languages = max
        binary_score = min(binary_files / 5, 1.0)  # 5+ binary = max

        # Weighted average
        complexity = (
            file_score * 0.3
            + change_score * 0.4
            + lang_score * 0.2
            + binary_score * 0.1
        )

        return round(complexity, 2)

    def _check_security(self, diff: PRDiff) -> list[PRIssue]:
        """Check for security issues in the diff."""
        issues: list[PRIssue] = []

        for file in diff.files:
            for hunk in file.hunks:
                for line in hunk.lines:
                    if line.line_type != "addition":
                        continue

                    for pattern_def in SECURITY_PATTERNS:
                        if pattern_def["pattern"].search(line.content):
                            issues.append(
                                PRIssue(
                                    severity=IssueSeverity.WARNING,
                                    category="security",
                                    file_path=file.path,
                                    line_number=line.new_line_number,
                                    message=pattern_def["message"],
                                    suggestion=pattern_def["suggestion"],
                                )
                            )

        return issues

    def _check_conventions(self, diff: PRDiff) -> list[PRIssue]:
        """Check for convention violations."""
        issues: list[PRIssue] = []

        for file in diff.files:
            # Check for very long functions (heuristic: hunks with many consecutive additions)
            for hunk in file.hunks:
                consecutive_additions = 0
                hunk_start_line = None

                for line in hunk.lines:
                    if line.line_type == "addition":
                        if consecutive_additions == 0:
                            hunk_start_line = line.new_line_number
                        consecutive_additions += 1
                    else:
                        if consecutive_additions > 50:
                            issues.append(
                                PRIssue(
                                    severity=IssueSeverity.WARNING,
                                    category="complexity",
                                    file_path=file.path,
                                    line_number=hunk_start_line,
                                    message=f"Large block of {consecutive_additions} new lines - consider breaking into smaller functions",
                                    suggestion="Break down into smaller, focused functions",
                                )
                            )
                        consecutive_additions = 0

                # Check at end of hunk
                if consecutive_additions > 50:
                    issues.append(
                        PRIssue(
                            severity=IssueSeverity.WARNING,
                            category="complexity",
                            file_path=file.path,
                            line_number=hunk_start_line,
                            message=f"Large block of {consecutive_additions} new lines - consider breaking into smaller functions",
                            suggestion="Break down into smaller, focused functions",
                        )
                    )

        return issues

    def _run_impact_analysis(
        self, diff: PRDiff, repo_id: str, depth: int
    ) -> list[str]:
        """Run impact analysis on changed files."""
        if not self.impact_tool:
            return []

        changed_files = [f.path for f in diff.files if f.path]

        try:
            # Create impact input - using a simple dict since we may not have the import
            result = self.impact_tool.analyze({
                "repo_id": repo_id,
                "changed_files": changed_files,
                "max_depth": depth,
            })

            if hasattr(result, "affected_files"):
                return [af.file_path for af in result.affected_files]
            return []
        except Exception as e:
            logger.warning(f"Impact analysis failed: {e}")
            return []

    def _check_adr(self, diff: PRDiff) -> Optional[str]:
        """Check if changes warrant an ADR."""
        if not self.adr_tool:
            return None

        try:
            # Convert diff to changes format expected by ADR tool
            changes = []
            for file in diff.files:
                changes.append({
                    "file_path": file.path,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                })

            result = self.adr_tool.analyze(changes)

            if hasattr(result, "should_create_adr") and result.should_create_adr:
                if hasattr(result, "triggered_heuristics"):
                    return f"Triggered by: {', '.join(result.triggered_heuristics)}"
                return "ADR recommended based on change patterns"
            return None
        except Exception as e:
            logger.warning(f"ADR check failed: {e}")
            return None

    def get_mcp_schema(self) -> dict[str, Any]:
        """Get MCP tool schema."""
        return {
            "name": "analyze_pull_request",
            "description": "Analyze a pull request for impact, security issues, and ADR suggestions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": "Repository ID",
                    },
                    "diff": {
                        "type": "string",
                        "description": "The diff text to analyze",
                    },
                    "pr_number": {
                        "type": "integer",
                        "description": "PR number to fetch from GitHub",
                    },
                    "title": {
                        "type": "string",
                        "description": "PR title for context",
                    },
                    "body": {
                        "type": "string",
                        "description": "PR body/description for context",
                    },
                    "check_impact": {
                        "type": "boolean",
                        "description": "Run impact analysis",
                        "default": True,
                    },
                    "check_adr": {
                        "type": "boolean",
                        "description": "Check for ADR suggestions",
                        "default": True,
                    },
                    "check_security": {
                        "type": "boolean",
                        "description": "Check for security issues",
                        "default": True,
                    },
                },
                "required": ["repo_id"],
            },
        }

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute tool via MCP interface."""
        # Build config from input
        config = PRAnalysisConfig(
            check_impact=input_data.get("check_impact", True),
            check_adr=input_data.get("check_adr", True),
            check_security=input_data.get("check_security", True),
            check_conventions=input_data.get("check_conventions", True),
        )

        # Build input
        pr_input = PRAnalysisInput(
            repo_id=input_data.get("repo_id", ""),
            diff=input_data.get("diff"),
            pr_number=input_data.get("pr_number"),
            title=input_data.get("title"),
            body=input_data.get("body"),
        )

        result = self.analyze(pr_input, config)

        # Convert to dict
        return {
            "summary": {
                "files_changed": result.summary.files_changed,
                "additions": result.summary.additions,
                "deletions": result.summary.deletions,
                "languages": result.summary.languages,
                "main_areas": result.summary.main_areas,
                "complexity_score": result.summary.complexity_score,
                "affected_files": result.summary.affected_files,
                "suggested_adr": result.summary.suggested_adr,
                "adr_reason": result.summary.adr_reason,
            },
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in result.issues
            ],
            "request_id": result.request_id,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_pr_analysis_tool(
    graph_driver: Any,
    retriever: Any,
    impact_tool: Optional[Any] = None,
    adr_tool: Optional[Any] = None,
    config: Optional[PRAnalysisConfig] = None,
) -> PRAnalysisTool:
    """Create a PR analysis tool.

    Args:
        graph_driver: Neo4j driver for graph queries
        retriever: Retriever for code search
        impact_tool: Optional impact analysis tool
        adr_tool: Optional ADR automation tool
        config: Optional configuration

    Returns:
        Configured PRAnalysisTool
    """
    return PRAnalysisTool(
        graph_driver=graph_driver,
        retriever=retriever,
        impact_tool=impact_tool,
        adr_tool=adr_tool,
        config=config,
    )
