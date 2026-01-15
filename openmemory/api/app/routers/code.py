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

import asyncio
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
from app.security.types import Principal, Scope
from app.indexing_jobs import create_index_job, get_index_job, run_index_job
from app.path_utils import resolve_repo_root
from app.database import get_db
from app.services.job_queue_service import IndexingJobQueueService, QueueFullError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/code", tags=["code-intelligence"])


def _get_valkey_client():
    """Get Valkey client if available."""
    import os
    try:
        import redis
        host = os.getenv("VALKEY_HOST", "valkey")
        port = int(os.getenv("VALKEY_PORT", "6379"))
        client = redis.Redis(host=host, port=port, socket_timeout=5)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Valkey unavailable: {e}")
        return None


def get_queue_service(db: Session) -> IndexingJobQueueService:
    """Create a queue service with database session."""
    import os
    max_queued_jobs = int(os.getenv("MAX_QUEUED_JOBS", "100"))
    return IndexingJobQueueService(
        db=db,
        valkey_client=_get_valkey_client(),
        max_queued_jobs=max_queued_jobs,
    )


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


def _can_access_job(job: Dict[str, Any], principal: Principal, admin_scope: Scope) -> bool:
    """Check whether the principal can access a job."""
    if principal.has_scope(admin_scope):
        return True
    requested_by = job.get("requested_by")
    if not requested_by:
        return True
    return requested_by == principal.user_id


def _meta_to_dict(meta: CodeResponseMeta) -> Dict[str, Any]:
    """Normalize CodeResponseMeta to a plain dict."""
    if hasattr(meta, "model_dump"):
        return meta.model_dump()
    if hasattr(meta, "dict"):
        return meta.dict()
    return _safe_asdict(meta)


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
            include_generated=request.include_generated,
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
            include_inferred_edges=request.include_inferred_edges,
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
                warnings=result_dict.get("meta", {}).get("warnings", []),
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
            include_inferred_edges=request.include_inferred_edges,
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
            symbol_name=request.symbol_name,
            parent_name=request.parent_name,
            symbol_kind=request.symbol_kind,
            file_path=request.file_path,
            include_cross_language=request.include_cross_language,
            max_depth=request.max_depth,
            confidence_threshold=str(request.confidence_threshold),
            include_inferred_edges=request.include_inferred_edges,
            include_field_edges=request.include_field_edges,
            include_schema_edges=request.include_schema_edges,
            include_path_edges=request.include_path_edges,
        )

        result = toolkit.impact_tool.analyze(input_data)
        result_dict = _safe_asdict(result)

        return ImpactAnalysisResponse(
            affected_files=result_dict.get("affected_files", []),
            required_files=result_dict.get("required_files", []),
            coverage_summary=result_dict.get("coverage_summary", {}),
            coverage_low=result_dict.get("coverage_low", False),
            action_required=result_dict.get("action_required"),
            action_message=result_dict.get("action_message"),
            meta=CodeResponseMeta(
                request_id=result_dict.get("meta", {}).get("request_id", str(uuid.uuid4())),
                degraded_mode=result_dict.get("meta", {}).get("degraded_mode", False),
                missing_sources=result_dict.get("meta", {}).get("missing_sources", []),
                warnings=result_dict.get("meta", {}).get("warnings", []),
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
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_scopes(Scope.GRAPH_WRITE)),
) -> CodeIndexResponse:
    """Index a repository into the code graph and search index."""
    toolkit = get_code_toolkit()

    root_path = resolve_repo_root(request.root_path, repo_id=request.repo_id)
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

    meta = _create_meta()
    if missing_sources:
        meta = _create_degraded_meta(
            missing_sources=missing_sources,
            error="Some backends unavailable",
        )

    indexer = CodeIndexingService(
        root_path=root_path,
        repo_id=request.repo_id,
        graph_driver=toolkit.neo4j_driver,
        opensearch_client=toolkit.opensearch_client if toolkit.is_available("opensearch") else None,
        embedding_service=toolkit.embedding_service if toolkit.is_available("embedding") else None,
        index_name=request.index_name or "code",
        include_api_boundaries=request.include_api_boundaries,
        enable_zod_schema_aliases=request.enable_zod_schema_aliases,
        ignore_patterns=request.ignore_patterns,
        allow_patterns=request.allow_patterns,
    )

    if request.async_mode:
        # Use persistent job queue for async mode
        queue_service = get_queue_service(db)
        try:
            job_id = queue_service.create_job(
                repo_id=request.repo_id,
                root_path=str(root_path),
                index_name=request.index_name or "code",
                requested_by=principal.user_id,
                request={
                    "max_files": request.max_files,
                    "reset": request.reset,
                    "include_api_boundaries": request.include_api_boundaries,
                    "enable_zod_schema_aliases": request.enable_zod_schema_aliases,
                    "ignore_patterns": request.ignore_patterns,
                    "allow_patterns": request.allow_patterns,
                    "async_mode": True,
                },
                meta=_meta_to_dict(meta),
                force=request.force,
            )
            return CodeIndexResponse(
                repo_id=request.repo_id,
                files_indexed=0,
                files_failed=0,
                symbols_indexed=0,
                documents_indexed=0,
                call_edges_indexed=0,
                duration_ms=0.0,
                meta=meta,
                job_id=str(job_id),
                status="queued",
            )
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e))

    summary = indexer.index_repository(
        max_files=request.max_files,
        reset=request.reset,
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


@router.get(
    "/index/status/{job_id}",
    summary="Get index job status",
    description="Fetch status and results for a background indexing job.",
)
async def index_codebase_status(
    job_id: str,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
) -> Dict[str, Any]:
    """Get status of a background indexing job."""
    # Try persistent store first
    try:
        job_uuid = uuid.UUID(job_id)
        queue_service = get_queue_service(db)
        job = queue_service.get_job(job_uuid)
        if job:
            if not _can_access_job(job, principal, Scope.ADMIN_READ):
                raise HTTPException(status_code=403, detail="Access denied")
            return job
    except ValueError:
        pass  # Not a valid UUID, try legacy store

    # Fallback to legacy in-memory store for backwards compatibility
    job = get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job


@router.post(
    "/index/cancel/{job_id}",
    summary="Cancel an indexing job",
    description="Request cancellation of a running or queued indexing job.",
)
async def index_codebase_cancel(
    job_id: str,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_scopes(Scope.GRAPH_WRITE)),
) -> Dict[str, Any]:
    """Request cancellation of an indexing job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format")

    queue_service = get_queue_service(db)
    job = queue_service.get_job(job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if not _can_access_job(job, principal, Scope.ADMIN_WRITE):
        raise HTTPException(status_code=403, detail="Access denied")

    success = queue_service.cancel_job(job_uuid)
    if success:
        return {"job_id": job_id, "status": "cancel_requested"}
    else:
        raise HTTPException(status_code=500, detail="Failed to request cancellation")


@router.get(
    "/index/jobs",
    summary="List indexing jobs",
    description="List recent indexing jobs with optional filters.",
)
async def list_index_jobs(
    repo_id: str = None,
    status: str = None,
    requested_by: str = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    principal: Principal = Depends(require_scopes(Scope.GRAPH_READ)),
) -> Dict[str, Any]:
    """List recent indexing jobs."""
    from app.models import CodeIndexJobStatus

    queue_service = get_queue_service(db)

    status_enum = None
    if status:
        try:
            status_enum = CodeIndexJobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {[s.value for s in CodeIndexJobStatus]}"
            )

    if not principal.has_scope(Scope.ADMIN_READ):
        requested_by = principal.user_id

    jobs = queue_service.list_jobs(
        repo_id=repo_id,
        requested_by=requested_by,
        status=status_enum,
        limit=min(limit, 100),
    )
    return {"jobs": jobs, "count": len(jobs)}
