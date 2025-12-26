"""Query Playground router.

This module implements the query playground per section 16 (FR-013):
- Parameter tuning interface
- Side-by-side retrieval comparison
- Query execution with configurable parameters
- Result comparison and analysis
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PlaygroundConfig:
    """Configuration for query playground."""

    max_results: int = 20
    enable_caching: bool = True
    default_search_mode: str = "hybrid"
    cache_ttl_seconds: int = 300


# ============================================================================
# Query Parameters
# ============================================================================


@dataclass
class QueryParams:
    """Parameters for a query execution."""

    query: str
    mode: str = "hybrid"  # semantic, lexical, hybrid
    limit: int = 10
    threshold: float = 0.0
    semantic_weight: float = 0.5
    lexical_weight: float = 0.5
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "mode": self.mode,
            "limit": self.limit,
            "threshold": self.threshold,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "filters": self.filters,
        }


# ============================================================================
# Results
# ============================================================================


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of comparing two query executions."""

    left_params: QueryParams
    right_params: QueryParams
    left_results: list[RetrievalResult]
    right_results: list[RetrievalResult]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def calculate_overlap(self) -> dict[str, Any]:
        """Calculate overlap between results.

        Returns:
            Dictionary with common, left-only, and right-only IDs
        """
        left_ids = {r.id for r in self.left_results}
        right_ids = {r.id for r in self.right_results}

        return {
            "common_ids": left_ids & right_ids,
            "left_only_ids": left_ids - right_ids,
            "right_only_ids": right_ids - left_ids,
            "overlap_ratio": (
                len(left_ids & right_ids) / len(left_ids | right_ids)
                if left_ids | right_ids
                else 0.0
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get comparison statistics.

        Returns:
            Dictionary with statistics
        """
        left_scores = [r.score for r in self.left_results]
        right_scores = [r.score for r in self.right_results]

        return {
            "left_count": len(self.left_results),
            "right_count": len(self.right_results),
            "left_avg_score": sum(left_scores) / len(left_scores) if left_scores else 0.0,
            "right_avg_score": sum(right_scores) / len(right_scores) if right_scores else 0.0,
            "left_max_score": max(left_scores) if left_scores else 0.0,
            "right_max_score": max(right_scores) if right_scores else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "left_params": self.left_params.to_dict(),
            "right_params": self.right_params.to_dict(),
            "left_results": [r.to_dict() for r in self.left_results],
            "right_results": [r.to_dict() for r in self.right_results],
            "statistics": self.get_statistics(),
            "overlap": self.calculate_overlap(),
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Query Execution
# ============================================================================


@dataclass
class QueryExecution:
    """Record of a query execution."""

    execution_id: str
    params: QueryParams
    results: list[RetrievalResult]
    execution_time_ms: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "params": self.params.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Session
# ============================================================================


@dataclass
class PlaygroundSession:
    """A playground session for a user."""

    session_id: str
    user_id: str
    executions: list[QueryExecution] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_execution(self, execution: QueryExecution) -> None:
        """Add an execution to the session.

        Args:
            execution: The execution to add
        """
        self.executions.append(execution)

    def add_comparison(self, comparison: ComparisonResult) -> None:
        """Add a comparison to the session.

        Args:
            comparison: The comparison to add
        """
        self.comparisons.append(comparison)

    def get_history(self) -> list[dict[str, Any]]:
        """Get execution history.

        Returns:
            List of execution dictionaries
        """
        return [e.to_dict() for e in self.executions]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "executions": [e.to_dict() for e in self.executions],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Service
# ============================================================================


class PlaygroundService:
    """Service for query playground operations."""

    # Parameter presets
    PRESETS = {
        "semantic": {
            "mode": "semantic",
            "limit": 10,
            "threshold": 0.5,
            "semantic_weight": 1.0,
            "lexical_weight": 0.0,
        },
        "lexical": {
            "mode": "lexical",
            "limit": 10,
            "threshold": 0.0,
            "semantic_weight": 0.0,
            "lexical_weight": 1.0,
        },
        "hybrid": {
            "mode": "hybrid",
            "limit": 10,
            "threshold": 0.3,
            "semantic_weight": 0.5,
            "lexical_weight": 0.5,
        },
        "hybrid_semantic_heavy": {
            "mode": "hybrid",
            "limit": 10,
            "threshold": 0.3,
            "semantic_weight": 0.7,
            "lexical_weight": 0.3,
        },
        "hybrid_lexical_heavy": {
            "mode": "hybrid",
            "limit": 10,
            "threshold": 0.3,
            "semantic_weight": 0.3,
            "lexical_weight": 0.7,
        },
    }

    def __init__(self, config: PlaygroundConfig | None = None):
        """Initialize the service.

        Args:
            config: Optional configuration
        """
        self._config = config or PlaygroundConfig()
        self._sessions: dict[str, PlaygroundSession] = {}

    def create_session(self, user_id: str) -> PlaygroundSession:
        """Create a new playground session.

        Args:
            user_id: The user ID

        Returns:
            New session
        """
        session_id = str(uuid.uuid4())
        session = PlaygroundSession(
            session_id=session_id,
            user_id=user_id,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> PlaygroundSession | None:
        """Get a session by ID.

        Args:
            session_id: The session ID

        Returns:
            Session if found, None otherwise
        """
        return self._sessions.get(session_id)

    def list_sessions(self, user_id: str) -> list[PlaygroundSession]:
        """List sessions for a user.

        Args:
            user_id: The user ID

        Returns:
            List of sessions
        """
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def execute_query(
        self,
        session_id: str,
        params: QueryParams,
    ) -> QueryExecution | None:
        """Execute a query and record it in the session.

        Args:
            session_id: The session ID
            params: Query parameters

        Returns:
            Query execution record, or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Execute the search
        start_time = time.perf_counter()
        raw_results = self._do_search(params)
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert to RetrievalResults
        results = []
        for r in raw_results:
            results.append(
                RetrievalResult(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    source=r.get("source", "unknown"),
                    metadata=r.get("metadata", {}),
                )
            )

        # Create execution record
        execution = QueryExecution(
            execution_id=str(uuid.uuid4()),
            params=params,
            results=results,
            execution_time_ms=execution_time_ms,
        )

        session.add_execution(execution)
        return execution

    def compare_queries(
        self,
        session_id: str,
        left_params: QueryParams,
        right_params: QueryParams,
    ) -> ComparisonResult | None:
        """Compare two query executions.

        Args:
            session_id: The session ID
            left_params: Parameters for left query
            right_params: Parameters for right query

        Returns:
            Comparison result, or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Execute both searches
        left_raw = self._do_search(left_params)
        right_raw = self._do_search(right_params)

        # Convert to RetrievalResults
        left_results = [
            RetrievalResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                source=r.get("source", "unknown"),
                metadata=r.get("metadata", {}),
            )
            for r in left_raw
        ]

        right_results = [
            RetrievalResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                source=r.get("source", "unknown"),
                metadata=r.get("metadata", {}),
            )
            for r in right_raw
        ]

        # Create comparison
        comparison = ComparisonResult(
            left_params=left_params,
            right_params=right_params,
            left_results=left_results,
            right_results=right_results,
        )

        session.add_comparison(comparison)
        return comparison

    def get_presets(self) -> dict[str, dict[str, Any]]:
        """Get available parameter presets.

        Returns:
            Dictionary of preset names to parameter dicts
        """
        return self.PRESETS.copy()

    def apply_preset(self, preset_name: str, query: str) -> QueryParams:
        """Apply a preset to create query parameters.

        Args:
            preset_name: Name of the preset
            query: The query string

        Returns:
            QueryParams with preset values
        """
        preset = self.PRESETS.get(preset_name, self.PRESETS["hybrid"])

        return QueryParams(
            query=query,
            mode=preset["mode"],
            limit=preset["limit"],
            threshold=preset["threshold"],
            semantic_weight=preset["semantic_weight"],
            lexical_weight=preset["lexical_weight"],
        )

    def _do_search(self, params: QueryParams) -> list[dict[str, Any]]:
        """Perform the actual search (stub for testing).

        In production, this would call the actual search backend.

        Args:
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        # Stub implementation - in production would call actual search
        return []


# ============================================================================
# Factory
# ============================================================================


def create_playground_service(
    config: PlaygroundConfig | None = None,
) -> PlaygroundService:
    """Create a playground service.

    Args:
        config: Optional configuration

    Returns:
        PlaygroundService instance
    """
    return PlaygroundService(config=config)
