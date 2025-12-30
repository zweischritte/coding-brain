"""
Code Intelligence REST API Router.

Exposes code-intel tools via REST endpoints under /api/v1/code/*.
All endpoints require appropriate OAuth scopes.

Scope mapping (from PRD):
- search_code_hybrid: search:read
- find_callers/callees: graph:read
- impact_analysis: graph:read
- explain_code: search:read + graph:read
- adr_automation: search:read + graph:read
- test_generation: search:read + graph:read
- pr_analysis: search:read + graph:read
- index_codebase: graph:write
"""

import dataclasses
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from app.schemas import (
    # Requests
    CodeSearchRequest,
    ExplainCodeRequest,
    CallGraphRequest,
    ImpactAnalysisRequest,
    ADRRequest,
    ADRDetection,
    TestGenerationRequest,
    PRAnalysisRequest,
    CodeIndexRequest,
    # Responses
    CodeSearchResponse,
    ExplainCodeResponse,
    CallGraphResponse,
    ImpactAnalysisResponse,
    ADRResponse,
    TestGenerationResponse,
    PRAnalysisResponse,
    CodeIndexResponse,
    # Common
    CodeResponseMeta,
)
from app.security.dependencies import require_scopes
from app.security.types import Scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/code", tags=["code-intelligence"])


def get_code_toolkit():
    """Lazy-load code toolkit to avoid import-time failures.

    Returns:
        CodeToolkit instance with available tools and dependencies.
    """
    from app.code_toolkit import get_code_toolkit as _get_toolkit
    return _get_toolkit()


def _create_degraded_meta(
    missing_sources: list[str],
    error: str = None,
) -> CodeResponseMeta:
    """Create a degraded mode response meta."""
    return CodeResponseMeta(
        request_id=str(uuid.uuid4()),
        degraded_mode=True,
        missing_sources=missing_sources,
        error=error,
    )


def _create_meta() -> CodeResponseMeta:
    """Create a normal response meta."""
    return CodeResponseMeta(
        request_id=str(uuid.uuid4()),
        degraded_mode=False,
        missing_sources=[],
    )


def _safe_asdict(obj: Any) -> Dict[str, Any]:
    """Safely convert dataclass or object to dict."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


# =============================================================================
# Code Search (search:read)
# =============================================================================


@router.post(
    "/search",
    response_model=CodeSearchResponse,
    summary="Search code using tri-hybrid retrieval",
    description="Combines lexical, semantic, and graph-based search for code.",
)
async def search_code(
    request: CodeSearchRequest,
    _: None = Depends(require_scopes(Scope.SEARCH_READ)),
) -> CodeSearchResponse:
    """Search code using tri-hybrid retrieval (lexical + semantic + graph)."""
    toolkit = get_code_toolkit()

    # Check if OpenSearch is available
    if not toolkit.is_available("opensearch") and not toolkit.is_available("trihybrid"):
        return CodeSearchResponse(
            results=[],
            meta=_create_degraded_meta(
                missing_sources=["opensearch"],
                error="Search backend unavailable",
            ),
        )

    # Check if search tool is available
    if not toolkit.search_tool:
        return CodeSearchResponse(
            results=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("search_tool"),
            ),
        )

    try:
        from tools.search_code_hybrid import SearchCodeHybridInput

        input_data = SearchCodeHybridInput(
            query=request.query,
            repo_id=request.repo_id,
            language=request.language,
            limit=request.limit,
            offset=request.offset,
            seed_symbols=request.seed_symbols or [],
        )

        result = toolkit.search_tool.search(input_data)
        result_dict = _safe_asdict(result)

        return CodeSearchResponse(
            results=result_dict.get("results", []),
            meta=CodeResponseMeta(
                request_id=result_dict.get("meta", {}).get("request_id", str(uuid.uuid4())),
                degraded_mode=result_dict.get("meta", {}).get("degraded_mode", False),
                missing_sources=result_dict.get("meta", {}).get("missing_sources", []),
            ),
            next_cursor=result_dict.get("next_cursor"),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return CodeSearchResponse(
            results=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Explain Code (search:read + graph:read)
# =============================================================================


@router.post(
    "/explain",
    response_model=ExplainCodeResponse,
    summary="Explain a code symbol",
    description="Get detailed explanation of a code symbol including call graph.",
)
async def explain_code(
    request: ExplainCodeRequest,
    _: None = Depends(require_scopes(Scope.SEARCH_READ, Scope.GRAPH_READ)),
) -> ExplainCodeResponse:
    """Get detailed explanation of a code symbol with context."""
    toolkit = get_code_toolkit()

    # Check if Neo4j is available
    if not toolkit.is_available("neo4j"):
        return ExplainCodeResponse(
            explanation=None,
            meta=_create_degraded_meta(
                missing_sources=["neo4j"],
                error="Graph backend unavailable",
            ),
        )

    # Check if explain tool is available
    if not toolkit.explain_tool:
        return ExplainCodeResponse(
            explanation=None,
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("explain_tool"),
            ),
        )

    try:
        from tools.explain_code import ExplainCodeConfig

        config = ExplainCodeConfig(
            depth=request.depth,
            include_callers=request.include_callers,
            include_callees=request.include_callees,
            include_usages=request.include_usages,
            max_usages=request.max_usages,
        )

        result = toolkit.explain_tool.explain(request.symbol_id, config)
        result_dict = _safe_asdict(result)

        return ExplainCodeResponse(
            explanation=result_dict,
            meta=_create_meta(),
        )

    except Exception as e:
        logger.error(f"Explain failed: {e}")
        return ExplainCodeResponse(
            explanation=None,
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Find Callers (graph:read)
# =============================================================================


@router.post(
    "/callers",
    response_model=CallGraphResponse,
    summary="Find functions that call a symbol",
    description="Find all functions/methods that call the specified symbol.",
)
async def find_callers(
    request: CallGraphRequest,
    _: None = Depends(require_scopes(Scope.GRAPH_READ)),
) -> CallGraphResponse:
    """Find functions that call a given symbol."""
    toolkit = get_code_toolkit()

    # Check if Neo4j is available
    if not toolkit.is_available("neo4j"):
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=["neo4j"],
                error="Graph backend unavailable",
            ),
        )

    # Check if callers tool is available
    if not toolkit.callers_tool:
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("callers_tool"),
            ),
        )

    try:
        from tools.call_graph import CallGraphInput

        input_data = CallGraphInput(
            repo_id=request.repo_id,
            symbol_id=request.symbol_id,
            symbol_name=request.symbol_name,
            depth=request.depth,
        )

        result = toolkit.callers_tool.find(input_data)
        result_dict = _safe_asdict(result)

        return CallGraphResponse(
            nodes=result_dict.get("nodes", []),
            edges=result_dict.get("edges", []),
            meta=CodeResponseMeta(
                request_id=result_dict.get("meta", {}).get("request_id", str(uuid.uuid4())),
                degraded_mode=result_dict.get("meta", {}).get("degraded_mode", False),
                missing_sources=result_dict.get("meta", {}).get("missing_sources", []),
            ),
            next_cursor=result_dict.get("next_cursor"),
        )

    except Exception as e:
        logger.error(f"Find callers failed: {e}")
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Find Callees (graph:read)
# =============================================================================


@router.post(
    "/callees",
    response_model=CallGraphResponse,
    summary="Find functions called by a symbol",
    description="Find all functions/methods that are called by the specified symbol.",
)
async def find_callees(
    request: CallGraphRequest,
    _: None = Depends(require_scopes(Scope.GRAPH_READ)),
) -> CallGraphResponse:
    """Find functions called by a given symbol."""
    toolkit = get_code_toolkit()

    # Check if Neo4j is available
    if not toolkit.is_available("neo4j"):
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=["neo4j"],
                error="Graph backend unavailable",
            ),
        )

    # Check if callees tool is available
    if not toolkit.callees_tool:
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("callees_tool"),
            ),
        )

    try:
        from tools.call_graph import CallGraphInput

        input_data = CallGraphInput(
            repo_id=request.repo_id,
            symbol_id=request.symbol_id,
            symbol_name=request.symbol_name,
            depth=request.depth,
        )

        result = toolkit.callees_tool.find(input_data)
        result_dict = _safe_asdict(result)

        return CallGraphResponse(
            nodes=result_dict.get("nodes", []),
            edges=result_dict.get("edges", []),
            meta=CodeResponseMeta(
                request_id=result_dict.get("meta", {}).get("request_id", str(uuid.uuid4())),
                degraded_mode=result_dict.get("meta", {}).get("degraded_mode", False),
                missing_sources=result_dict.get("meta", {}).get("missing_sources", []),
            ),
            next_cursor=result_dict.get("next_cursor"),
        )

    except Exception as e:
        logger.error(f"Find callees failed: {e}")
        return CallGraphResponse(
            nodes=[],
            edges=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Impact Analysis (graph:read)
# =============================================================================


@router.post(
    "/impact",
    response_model=ImpactAnalysisResponse,
    summary="Analyze impact of code changes",
    description="Analyze which files and symbols are affected by code changes.",
)
async def impact_analysis(
    request: ImpactAnalysisRequest,
    _: None = Depends(require_scopes(Scope.GRAPH_READ)),
) -> ImpactAnalysisResponse:
    """Analyze the impact of code changes."""
    toolkit = get_code_toolkit()

    # Check if Neo4j is available
    if not toolkit.is_available("neo4j"):
        return ImpactAnalysisResponse(
            affected_files=[],
            meta=_create_degraded_meta(
                missing_sources=["neo4j"],
                error="Graph backend unavailable",
            ),
        )

    # Check if impact tool is available
    if not toolkit.impact_tool:
        return ImpactAnalysisResponse(
            affected_files=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("impact_tool"),
            ),
        )

    try:
        from tools.impact_analysis import ImpactInput

        input_data = ImpactInput(
            repo_id=request.repo_id,
            changed_files=request.changed_files or [],
            symbol_id=request.symbol_id,
            include_cross_language=request.include_cross_language,
            max_depth=request.max_depth,
            confidence_threshold=str(request.confidence_threshold),
        )

        result = toolkit.impact_tool.analyze(input_data)
        result_dict = _safe_asdict(result)

        return ImpactAnalysisResponse(
            affected_files=result_dict.get("affected_files", []),
            meta=CodeResponseMeta(
                request_id=result_dict.get("meta", {}).get("request_id", str(uuid.uuid4())),
                degraded_mode=result_dict.get("meta", {}).get("degraded_mode", False),
                missing_sources=result_dict.get("meta", {}).get("missing_sources", []),
            ),
        )

    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        return ImpactAnalysisResponse(
            affected_files=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# ADR Automation (search:read + graph:read)
# =============================================================================


@router.post(
    "/adr",
    response_model=ADRResponse,
    summary="Detect and generate ADRs",
    description="Automatically detect architectural decisions and generate ADRs.",
)
async def adr_automation(
    request: ADRRequest,
    _: None = Depends(require_scopes(Scope.SEARCH_READ, Scope.GRAPH_READ)),
) -> ADRResponse:
    """Detect architectural decisions and generate ADRs."""
    toolkit = get_code_toolkit()

    changes = request.changes
    if changes is None and request.diff:
        from tools.adr_automation import ChangeAnalyzer
        analyzer = ChangeAnalyzer()
        changes = analyzer.parse_diff(request.diff)
        if not changes:
            added_lines = []
            removed_lines = []
            for line in request.diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    removed_lines.append(line[1:])
            if added_lines or removed_lines:
                changes = [
                    {
                        "file_path": "unknown",
                        "change_type": "modified",
                        "diff": request.diff,
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                    }
                ]

    if changes is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'diff' or 'changes' is required for ADR analysis",
        )

    # Check if ADR tool is available
    if not toolkit.adr_tool:
        return ADRResponse(
            detections=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("adr_tool") or "ADR tool not available",
            ),
        )

    try:
        change_payload = []
        for change in changes:
            if hasattr(change, "model_dump"):
                change_payload.append(change.model_dump())
            else:
                change_payload.append(change)

        result = toolkit.adr_tool.execute(
            {
                "changes": change_payload,
                "min_confidence": request.min_confidence,
            }
        )

        detections = []
        if result.get("should_create_adr"):
            reasons = result.get("reasons") or []
            triggered = result.get("triggered_heuristics") or []
            generated_adr = result.get("generated_adr") or {}
            detections.append(
                ADRDetection(
                    detected=True,
                    confidence=result.get("confidence", 0.0),
                    reason="; ".join(reasons) if reasons else "ADR recommended",
                    category=triggered[0] if triggered else "general",
                    suggested_title=generated_adr.get("title"),
                    suggested_content=generated_adr.get("markdown"),
                )
            )

        return ADRResponse(
            detections=detections,
            meta=_create_meta(),
        )

    except Exception as e:
        logger.error(f"ADR automation failed: {e}")
        return ADRResponse(
            detections=[],
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Test Generation (search:read + graph:read)
# =============================================================================


@router.post(
    "/test-generation",
    response_model=TestGenerationResponse,
    summary="Generate test cases",
    description="Automatically generate test cases for a symbol or file.",
)
async def test_generation(
    request: TestGenerationRequest,
    _: None = Depends(require_scopes(Scope.SEARCH_READ, Scope.GRAPH_READ)),
) -> TestGenerationResponse:
    """Generate test cases for a symbol or file."""
    toolkit = get_code_toolkit()

    # Check if test generation tool is available
    if not toolkit.test_gen_tool:
        return TestGenerationResponse(
            test_cases=[],
            imports=[],
            fixtures={},
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=toolkit.get_error("test_gen_tool") or "Test generation tool not available",
            ),
        )

    try:
        result = toolkit.test_gen_tool.generate(
            symbol_id=request.symbol_id,
            file_path=request.file_path,
            repo_id=request.repo_id,
            test_framework=request.test_framework,
            include_fixtures=request.include_fixtures,
            include_mocks=request.include_mocks,
            include_edge_cases=request.include_edge_cases,
            max_tests=request.max_tests,
        )
        result_dict = _safe_asdict(result) if result else {}

        return TestGenerationResponse(
            test_cases=result_dict.get("test_cases", []),
            imports=result_dict.get("imports", []),
            fixtures=result_dict.get("fixtures", {}),
            meta=_create_meta(),
        )

    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return TestGenerationResponse(
            test_cases=[],
            imports=[],
            fixtures={},
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# PR Analysis (search:read + graph:read)
# =============================================================================


@router.post(
    "/pr-analysis",
    response_model=PRAnalysisResponse,
    summary="Analyze a pull request",
    description="Analyze a pull request for risks, complexity, and reviewers.",
)
async def pr_analysis(
    request: PRAnalysisRequest,
    _: None = Depends(require_scopes(Scope.SEARCH_READ, Scope.GRAPH_READ)),
) -> PRAnalysisResponse:
    """Analyze a pull request."""
    toolkit = get_code_toolkit()

    try:
        from tools.pr_workflow.pr_analysis import (
            PRAnalysisInput,
            create_pr_analysis_tool,
        )

        tool = create_pr_analysis_tool(
            graph_driver=toolkit.neo4j_driver,
            retriever=toolkit.trihybrid_retriever,
            impact_tool=toolkit.impact_tool,
            adr_tool=toolkit.adr_tool,
        )

        result = tool.analyze(
            PRAnalysisInput(
                repo_id=request.repo_id,
                diff=request.diff,
                pr_number=request.pr_number,
                title=request.title,
                body=request.body,
                base_branch=request.base_branch,
                head_branch=request.head_branch,
            )
        )

        summary_parts = [
            f"{result.summary.files_changed} files changed (+{result.summary.additions}/-{result.summary.deletions})",
        ]
        if result.summary.languages:
            summary_parts.append(f"languages: {', '.join(result.summary.languages)}")
        if result.summary.main_areas:
            summary_parts.append(f"areas: {', '.join(result.summary.main_areas)}")
        if result.summary.affected_files:
            summary_parts.append(f"affected: {', '.join(result.summary.affected_files)}")
        if result.summary.suggested_adr:
            summary_parts.append("ADR suggested")

        risks = [
            {
                "category": issue.category,
                "severity": issue.severity.value,
                "description": issue.message,
                "affected_files": [issue.file_path] if issue.file_path else [],
                "suggestion": issue.suggestion,
            }
            for issue in result.issues
        ]

        missing_sources = []
        if not toolkit.is_available("neo4j"):
            missing_sources.append("neo4j")
        if not toolkit.is_available("opensearch"):
            missing_sources.append("opensearch")

        meta = _create_meta()
        if missing_sources:
            meta = _create_degraded_meta(
                missing_sources=missing_sources,
                error="Some backends unavailable",
            )

        return PRAnalysisResponse(
            summary=" | ".join(summary_parts),
            risks=risks,
            suggested_reviewers=[],
            complexity_score=result.summary.complexity_score,
            meta=meta,
        )

    except ImportError:
        # PR analyzer not available
        return PRAnalysisResponse(
            summary="",
            risks=[],
            suggested_reviewers=[],
            complexity_score=0.0,
            meta=_create_degraded_meta(
                missing_sources=["pr_analysis"],
                error="PR analysis module not available",
            ),
        )
    except Exception as e:
        logger.error(f"PR analysis failed: {e}")
        return PRAnalysisResponse(
            summary="",
            risks=[],
            suggested_reviewers=[],
            complexity_score=0.0,
            meta=_create_degraded_meta(
                missing_sources=toolkit.get_missing_sources(),
                error=str(e),
            ),
        )


# =============================================================================
# Code Indexing (graph:write)
# =============================================================================


@router.post(
    "/index",
    response_model=CodeIndexResponse,
    summary="Index a code repository",
    description="Parse a local repository and populate CODE_* graph + OpenSearch.",
)
async def index_codebase(
    request: CodeIndexRequest,
    _: None = Depends(require_scopes(Scope.GRAPH_WRITE)),
) -> CodeIndexResponse:
    """Index a repository into the code graph and search index."""
    toolkit = get_code_toolkit()

    root_path = Path(request.root_path)
    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(status_code=404, detail="root_path not found")

    if not toolkit.is_available("neo4j"):
        return CodeIndexResponse(
            repo_id=request.repo_id,
            files_indexed=0,
            files_failed=0,
            symbols_indexed=0,
            documents_indexed=0,
            call_edges_indexed=0,
            duration_ms=0.0,
            meta=_create_degraded_meta(
                missing_sources=["neo4j"],
                error="Graph backend unavailable",
            ),
        )

    missing_sources = []
    if not toolkit.is_available("opensearch"):
        missing_sources.append("opensearch")
    if not toolkit.is_available("embedding"):
        missing_sources.append("embedding")

    from indexing.code_indexer import CodeIndexingService

    indexer = CodeIndexingService(
        root_path=root_path,
        repo_id=request.repo_id,
        graph_driver=toolkit.neo4j_driver,
        opensearch_client=toolkit.opensearch_client if toolkit.is_available("opensearch") else None,
        embedding_service=toolkit.embedding_service if toolkit.is_available("embedding") else None,
        index_name=request.index_name or "code",
        include_api_boundaries=request.include_api_boundaries,
    )

    summary = indexer.index_repository(
        max_files=request.max_files,
        reset=request.reset,
    )

    meta = _create_meta()
    if missing_sources:
        meta = _create_degraded_meta(
            missing_sources=missing_sources,
            error="Some backends unavailable",
        )

    return CodeIndexResponse(
        repo_id=summary.repo_id,
        files_indexed=summary.files_indexed,
        files_failed=summary.files_failed,
        symbols_indexed=summary.symbols_indexed,
        documents_indexed=summary.documents_indexed,
        call_edges_indexed=summary.call_edges_indexed,
        duration_ms=summary.duration_ms,
        meta=meta,
    )
