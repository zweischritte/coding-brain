from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, validator


class MemoryBase(BaseModel):
    content: str
    metadata_: Optional[dict] = Field(default_factory=dict)

class MemoryCreate(MemoryBase):
    user_id: UUID
    app_id: UUID


class Category(BaseModel):
    name: str


class App(BaseModel):
    id: UUID
    name: str


class Memory(MemoryBase):
    id: UUID
    user_id: UUID
    app_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    state: str
    categories: Optional[List[Category]] = None
    app: App

    model_config = ConfigDict(from_attributes=True)

class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    metadata_: Optional[dict] = None
    state: Optional[str] = None


class MemoryResponse(BaseModel):
    id: UUID
    content: str
    created_at: int
    state: str
    app_id: UUID
    app_name: str
    categories: List[str]
    metadata_: Optional[dict] = None

    @validator('created_at', pre=True)
    def convert_to_epoch(cls, v):
        if isinstance(v, datetime):
            return int(v.timestamp())
        return v

class PaginatedMemoryResponse(BaseModel):
    items: List[MemoryResponse]
    total: int
    page: int
    size: int
    pages: int


# =============================================================================
# Code Intelligence Schemas
# =============================================================================


from typing import Any, Dict
from pydantic import model_validator


class CodeResponseMeta(BaseModel):
    """Metadata included in all code-intel responses."""

    request_id: str = Field(..., description="Unique request identifier")
    degraded_mode: bool = Field(
        default=False,
        description="True if some data sources were unavailable"
    )
    missing_sources: List[str] = Field(
        default_factory=list,
        description="List of unavailable data sources"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if operation partially failed"
    )


class CodeSymbol(BaseModel):
    """Represents a code symbol (function, class, etc.)."""

    symbol_id: Optional[str] = Field(default=None)
    symbol_name: str = Field(default="")
    symbol_type: str = Field(default="")
    signature: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)
    line_start: Optional[int] = Field(default=None)
    line_end: Optional[int] = Field(default=None)


class GraphNode(BaseModel):
    """Represents a node in a call graph."""

    id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Represents an edge in a call graph."""

    from_id: str
    to_id: str
    type: str
    confidence: str = Field(default="definite")
    properties: Dict[str, Any] = Field(default_factory=dict)


class CodeSearchRequest(BaseModel):
    """Request body for code search endpoint."""

    query: str = Field(..., min_length=1)
    repo_id: Optional[str] = None
    language: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    seed_symbols: Optional[List[str]] = None


class CodeHit(BaseModel):
    """A single code search result."""

    symbol: CodeSymbol
    score: float
    source: str = Field(default="hybrid")
    snippet: Optional[str] = None
    repo_id: Optional[str] = None
    source_scores: Dict[str, float] = Field(default_factory=dict)


class CodeSearchResponse(BaseModel):
    """Response body for code search endpoint."""

    results: List[CodeHit] = Field(default_factory=list)
    meta: CodeResponseMeta
    next_cursor: Optional[str] = None


class ExplainCodeRequest(BaseModel):
    """Request body for code explain endpoint."""

    symbol_id: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)
    include_callers: bool = True
    include_callees: bool = True
    include_usages: bool = True
    max_usages: int = Field(default=5, ge=0, le=20)


class SymbolExplanation(BaseModel):
    """Detailed explanation of a code symbol."""

    symbol_id: str
    name: str
    kind: str
    signature: str
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    callers: List[Dict[str, Any]] = Field(default_factory=list)
    callees: List[Dict[str, Any]] = Field(default_factory=list)
    usages: List[Dict[str, Any]] = Field(default_factory=list)
    related: List[Dict[str, Any]] = Field(default_factory=list)
    context: Optional[str] = None


class ExplainCodeResponse(BaseModel):
    """Response body for code explain endpoint."""

    explanation: Optional[SymbolExplanation] = None
    meta: CodeResponseMeta


class CallGraphRequest(BaseModel):
    """Request body for call graph endpoints (callers/callees)."""

    repo_id: str
    symbol_id: Optional[str] = None
    symbol_name: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def require_symbol_identifier(self) -> "CallGraphRequest":
        """Require either symbol_id or symbol_name."""
        if not self.symbol_id and not self.symbol_name:
            raise ValueError("Either symbol_id or symbol_name is required")
        return self


class CallGraphResponse(BaseModel):
    """Response body for call graph endpoints."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    meta: CodeResponseMeta
    next_cursor: Optional[str] = None


class ImpactAnalysisRequest(BaseModel):
    """Request body for impact analysis endpoint."""

    repo_id: str
    changed_files: Optional[List[str]] = None
    symbol_id: Optional[str] = None
    include_cross_language: bool = False
    max_depth: int = Field(default=3, ge=1, le=10)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AffectedFile(BaseModel):
    """A file affected by code changes."""

    file_path: str
    impact_score: float = 0.0
    affected_symbols: List[str] = Field(default_factory=list)
    reason: str = ""


class ImpactAnalysisResponse(BaseModel):
    """Response body for impact analysis endpoint."""

    affected_files: List[AffectedFile] = Field(default_factory=list)
    meta: CodeResponseMeta


class ADRRequest(BaseModel):
    """Request body for ADR automation endpoint."""

    diff: Optional[str] = None
    commit_messages: Optional[List[str]] = None
    repo_id: Optional[str] = None
    pr_number: Optional[int] = None
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class ADRDetection(BaseModel):
    """An ADR detection result."""

    detected: bool
    confidence: float
    reason: str
    category: str = "general"
    suggested_title: Optional[str] = None
    suggested_content: Optional[str] = None


class ADRResponse(BaseModel):
    """Response body for ADR automation endpoint."""

    detections: List[ADRDetection] = Field(default_factory=list)
    meta: CodeResponseMeta


class TestGenerationRequest(BaseModel):
    """Request body for test generation endpoint."""

    symbol_id: Optional[str] = None
    file_path: Optional[str] = None
    repo_id: Optional[str] = None
    test_framework: str = "pytest"
    include_fixtures: bool = True
    include_mocks: bool = True
    include_edge_cases: bool = True
    max_tests: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def require_target(self) -> "TestGenerationRequest":
        """Require either symbol_id or file_path."""
        if not self.symbol_id and not self.file_path:
            raise ValueError("Either symbol_id or file_path is required")
        return self


class TestCase(BaseModel):
    """A generated test case."""

    name: str
    description: str
    code: str
    category: str = "general"


class TestGenerationResponse(BaseModel):
    """Response body for test generation endpoint."""

    test_cases: List[TestCase] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    fixtures: Dict[str, Any] = Field(default_factory=dict)
    meta: CodeResponseMeta


class PRAnalysisRequest(BaseModel):
    """Request body for PR analysis endpoint."""

    repo_id: str
    diff: Optional[str] = None
    pr_number: Optional[int] = None
    title: Optional[str] = None
    body: Optional[str] = None
    base_branch: Optional[str] = None
    head_branch: Optional[str] = None


class PRRisk(BaseModel):
    """A risk identified in a PR."""

    category: str
    severity: str
    description: str
    affected_files: List[str] = Field(default_factory=list)
    suggestion: Optional[str] = None


class PRAnalysisResponse(BaseModel):
    """Response body for PR analysis endpoint."""

    summary: str = ""
    risks: List[PRRisk] = Field(default_factory=list)
    suggested_reviewers: List[str] = Field(default_factory=list)
    complexity_score: float = 0.0
    meta: CodeResponseMeta
