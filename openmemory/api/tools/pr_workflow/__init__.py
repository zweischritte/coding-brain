"""PR Workflow Tools for code review and analysis (FR-009).

This module provides MCP tools for PR workflow:
- analyze_pull_request: Analyze PR changes with impact and ADR detection
- suggest_review_comments: Generate review comments based on code changes

Integration points:
- openmemory.api.indexing.graph_projection: CODE_* graph queries
- openmemory.api.tools.adr_automation: ADR detection for changes
- openmemory.api.tools.impact_analysis: Impact analysis for changes
- GitHub CLI (gh): PR data fetching and comment posting
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    # PR parser
    "PRDiff",
    "PRFile",
    "PRHunk",
    "PRLine",
    "DiffParser",
    "DiffParseError",
    # PR analysis tool
    "PRAnalysisConfig",
    "PRAnalysisInput",
    "PRAnalysisOutput",
    "PRSummary",
    "PRIssue",
    "IssueSeverity",
    "PRAnalysisTool",
    "PRAnalysisError",
    "create_pr_analysis_tool",
    # Review comment suggestions
    "ReviewCommentConfig",
    "ReviewCommentInput",
    "ReviewComment",
    "CommentType",
    "ReviewSuggestion",
    "ReviewCommentTool",
    "ReviewCommentError",
    "create_review_comment_tool",
    # GitHub integration
    "GitHubConfig",
    "GitHubClient",
    "GitHubPR",
    "GitHubComment",
    "GitHubError",
    "create_github_client",
]

# Module mappings for lazy imports
_MODULE_MAP = {
    # pr_parser
    "PRDiff": "pr_parser",
    "PRFile": "pr_parser",
    "PRHunk": "pr_parser",
    "PRLine": "pr_parser",
    "DiffParser": "pr_parser",
    "DiffParseError": "pr_parser",
    # pr_analysis
    "PRAnalysisConfig": "pr_analysis",
    "PRAnalysisInput": "pr_analysis",
    "PRAnalysisOutput": "pr_analysis",
    "PRSummary": "pr_analysis",
    "PRIssue": "pr_analysis",
    "IssueSeverity": "pr_analysis",
    "PRAnalysisTool": "pr_analysis",
    "PRAnalysisError": "pr_analysis",
    "create_pr_analysis_tool": "pr_analysis",
    # review_comments
    "ReviewCommentConfig": "review_comments",
    "ReviewCommentInput": "review_comments",
    "ReviewComment": "review_comments",
    "CommentType": "review_comments",
    "ReviewSuggestion": "review_comments",
    "ReviewCommentTool": "review_comments",
    "ReviewCommentError": "review_comments",
    "create_review_comment_tool": "review_comments",
    # github_client
    "GitHubConfig": "github_client",
    "GitHubClient": "github_client",
    "GitHubPR": "github_client",
    "GitHubComment": "github_client",
    "GitHubError": "github_client",
    "create_github_client": "github_client",
}


def __getattr__(name):
    """Lazy import attributes."""
    if name in _MODULE_MAP:
        import importlib

        module_name = _MODULE_MAP[name]
        module = importlib.import_module(
            f"openmemory.api.tools.pr_workflow.{module_name}"
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
