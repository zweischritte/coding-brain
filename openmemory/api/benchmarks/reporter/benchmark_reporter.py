"""Benchmark reporter for generating comparison reports.

Generates structured reports from benchmark results including:
- Model comparison tables
- Lexical backend comparison
- Winner selection with rationale
- Multiple output formats (JSON, Markdown, Console)
"""

import json
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any

from openmemory.api.benchmarks.runner.results import (
    BenchmarkResult,
    ModelResult,
    LexicalBackendResult,
)


class ReportFormat(Enum):
    """Available report output formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    CONSOLE = "console"


class BenchmarkReporter:
    """Generates comparison reports from benchmark results.

    Supports multiple output formats and includes winner selection
    based on production readiness thresholds.

    Attributes:
        result: The benchmark result to report on
        mrr_threshold: Minimum MRR score for production readiness (default 0.75)
        ndcg_threshold: Minimum NDCG score for production readiness (default 0.80)
    """

    def __init__(
        self,
        result: BenchmarkResult,
        mrr_threshold: float = 0.75,
        ndcg_threshold: float = 0.80,
    ):
        """Initialize the reporter with benchmark results.

        Args:
            result: BenchmarkResult to generate report from
            mrr_threshold: Minimum MRR score for production readiness (per v7 plan)
            ndcg_threshold: Minimum NDCG score for production readiness (per v7 plan)
        """
        self.result = result
        self.mrr_threshold = mrr_threshold
        self.ndcg_threshold = ndcg_threshold

    def generate_report(self, format: ReportFormat = ReportFormat.JSON) -> str:
        """Generate a benchmark report in the specified format.

        Args:
            format: Output format (JSON, MARKDOWN, or CONSOLE)

        Returns:
            Report string in the specified format
        """
        if format == ReportFormat.JSON:
            return self.to_json()
        elif format == ReportFormat.MARKDOWN:
            return self.to_markdown()
        elif format == ReportFormat.CONSOLE:
            return self.to_console()
        else:
            raise ValueError(f"Unknown format: {format}")

    def to_dict(self) -> Dict[str, Any]:
        """Generate report as a dictionary for programmatic access.

        Returns:
            Dictionary containing the complete report structure
        """
        return {
            "metadata": self._build_metadata(),
            "thresholds": {
                "mrr": self.mrr_threshold,
                "ndcg": self.ndcg_threshold,
            },
            "summary": self._build_summary(),
            "model_comparison": self._build_model_comparison(),
            "lexical_comparison": self._build_lexical_comparison(),
            "winner": self._build_winner_section(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Generate report as JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON-formatted report string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate report as Markdown string.

        Returns:
            Markdown-formatted report string
        """
        lines = []
        report_dict = self.to_dict()

        # Title
        lines.append("# Benchmark Report")
        lines.append("")

        # Metadata
        metadata = report_dict["metadata"]
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Dataset**: {metadata['dataset']}")
        lines.append(f"- **Language**: {metadata.get('language', 'python')}")
        lines.append(f"- **Timestamp**: {metadata['timestamp']}")
        lines.append(f"- **Duration**: {metadata['duration_seconds']:.2f} seconds")
        lines.append("")

        # Thresholds
        lines.append("## Production Thresholds")
        lines.append("")
        lines.append(f"- **MRR**: >= {report_dict['thresholds']['mrr']}")
        lines.append(f"- **NDCG**: >= {report_dict['thresholds']['ndcg']}")
        lines.append("")

        # Summary
        summary = report_dict["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Total models evaluated: {summary['total_models']}")
        lines.append(f"- Production-ready models: {summary['production_ready_count']}")
        lines.append(f"- Total lexical backends: {summary['total_backends']}")
        lines.append("")

        # Model Comparison Table
        models = report_dict["model_comparison"]["models"]
        if models:
            lines.append("## Embedding Model Comparison")
            lines.append("")
            lines.append(
                "| Model | Provider | MRR | NDCG | Latency P95 (ms) | Production Ready |"
            )
            lines.append("|-------|----------|-----|------|------------------|------------------|")
            for model in models:
                ready = "Yes" if model["production_ready"] else "No"
                lines.append(
                    f"| {model['model_name']} | {model['provider']} | "
                    f"{model['mrr_score']:.3f} | {model['ndcg_score']:.3f} | "
                    f"{model['latency_p95']:.1f} | {ready} |"
                )
            lines.append("")

        # Lexical Backend Comparison
        backends = report_dict["lexical_comparison"]["backends"]
        if backends:
            lines.append("## Lexical Backend Comparison")
            lines.append("")
            lines.append(
                "| Backend | Weighted Total | Latency P95 (ms) | Latency | Ops Complexity | Scalability | Features |"
            )
            lines.append(
                "|---------|----------------|------------------|---------|----------------|-------------|----------|"
            )
            for backend in backends:
                scores = backend["criterion_scores"]
                lines.append(
                    f"| {backend['backend_name']} | {backend['weighted_total']:.3f} | "
                    f"{backend['latency_p95']:.1f} | {scores.get('latency', 0):.2f} | "
                    f"{scores.get('ops_complexity', 0):.2f} | "
                    f"{scores.get('scalability', 0):.2f} | "
                    f"{scores.get('feature_support', 0):.2f} |"
                )
            lines.append("")

        # Winner Section
        winner = report_dict["winner"]
        lines.append("## Recommendation")
        lines.append("")

        if winner["embedding_model"]:
            model = winner["embedding_model"]
            lines.append(f"### Winner: Embedding Model")
            lines.append("")
            lines.append(f"**{model['model_name']}** ({model['provider']})")
            lines.append(f"- MRR: {model['mrr_score']:.3f}")
            lines.append(f"- NDCG: {model['ndcg_score']:.3f}")
            lines.append(f"- Latency P95: {model['latency_p95']:.1f} ms")
            lines.append("")
        else:
            lines.append(
                "### No embedding model meets production thresholds (MRR >= 0.75, NDCG >= 0.80)"
            )
            lines.append("")

        if winner["lexical_backend"]:
            backend = winner["lexical_backend"]
            lines.append(f"### Winner: Lexical Backend")
            lines.append("")
            lines.append(f"**{backend['backend_name']}**")
            lines.append(f"- Weighted Total: {backend['weighted_total']:.3f}")
            lines.append(f"- Latency P95: {backend['latency_p95']:.1f} ms")
            lines.append("")

        if winner["rationale"]:
            lines.append("### Rationale")
            lines.append("")
            lines.append(winner["rationale"])
            lines.append("")

        return "\n".join(lines)

    def to_console(self) -> str:
        """Generate compact console-friendly report.

        Returns:
            Console-formatted report string
        """
        lines = []
        report_dict = self.to_dict()

        lines.append("=== Benchmark Results ===")
        lines.append("")

        # Summary
        summary = report_dict["summary"]
        lines.append(
            f"Models: {summary['production_ready_count']}/{summary['total_models']} production-ready"
        )
        lines.append(f"Backends: {summary['total_backends']} evaluated")
        lines.append("")

        # Winner
        winner = report_dict["winner"]

        if winner["embedding_model"]:
            model = winner["embedding_model"]
            lines.append(f"Best Model: {model['model_name']}")
            lines.append(
                f"  MRR: {model['mrr_score']:.3f}  NDCG: {model['ndcg_score']:.3f}  P95: {model['latency_p95']:.1f}ms"
            )
        else:
            lines.append("Best Model: None (no models meet production thresholds)")

        if winner["lexical_backend"]:
            backend = winner["lexical_backend"]
            lines.append(f"Best Backend: {backend['backend_name']}")
            lines.append(
                f"  Score: {backend['weighted_total']:.3f}  P95: {backend['latency_p95']:.1f}ms"
            )

        lines.append("")
        lines.append(
            f"Thresholds: MRR >= {report_dict['thresholds']['mrr']}, NDCG >= {report_dict['thresholds']['ndcg']}"
        )

        return "\n".join(lines)

    def validate_thresholds(self) -> Dict[str, Any]:
        """Validate models against production thresholds.

        Returns:
            Dictionary with threshold validation summary
        """
        models_passing_mrr = sum(
            1
            for m in self.result.model_results
            if m.meets_mrr_threshold(self.mrr_threshold)
        )
        models_passing_ndcg = sum(
            1
            for m in self.result.model_results
            if m.meets_ndcg_threshold(self.ndcg_threshold)
        )
        models_production_ready = sum(
            1
            for m in self.result.model_results
            if m.is_production_ready(self.mrr_threshold, self.ndcg_threshold)
        )

        return {
            "mrr_threshold": self.mrr_threshold,
            "ndcg_threshold": self.ndcg_threshold,
            "total_models": len(self.result.model_results),
            "models_passing_mrr": models_passing_mrr,
            "models_passing_ndcg": models_passing_ndcg,
            "models_production_ready": models_production_ready,
        }

    def _build_metadata(self) -> Dict[str, Any]:
        """Build metadata section of the report."""
        return {
            "timestamp": self.result.timestamp.isoformat(),
            "duration_seconds": self.result.total_duration_seconds,
            "dataset": self.result.config.dataset_name,
            "language": self.result.config.dataset_language,
            "split": self.result.config.dataset_split,
            "sample_limit": self.result.config.sample_limit,
        }

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary statistics section."""
        production_ready = self.result.get_production_ready_models(
            self.mrr_threshold, self.ndcg_threshold
        )

        return {
            "total_models": len(self.result.model_results),
            "production_ready_count": len(production_ready),
            "total_backends": len(self.result.lexical_results),
        }

    def _build_model_comparison(self) -> Dict[str, Any]:
        """Build model comparison table data."""
        # Sort by MRR descending
        sorted_models = sorted(
            self.result.model_results,
            key=lambda m: m.mrr_score,
            reverse=True,
        )

        models = []
        for model in sorted_models:
            models.append(
                {
                    "model_name": model.model_name,
                    "provider": model.provider,
                    "mrr_score": model.mrr_score,
                    "ndcg_score": model.ndcg_score,
                    "latency_p95": model.latency_stats.p95,
                    "latency_mean": model.latency_stats.mean,
                    "num_queries": model.num_queries,
                    "embedding_dimensions": model.embedding_dimensions,
                    "production_ready": model.is_production_ready(
                        self.mrr_threshold, self.ndcg_threshold
                    ),
                }
            )

        return {"models": models}

    def _build_lexical_comparison(self) -> Dict[str, Any]:
        """Build lexical backend comparison table data."""
        # Sort by weighted_total descending
        sorted_backends = sorted(
            self.result.lexical_results,
            key=lambda b: b.weighted_total,
            reverse=True,
        )

        backends = []
        for backend in sorted_backends:
            # Convert CriterionName keys to string values
            criterion_scores = {
                k.value: v for k, v in backend.criterion_scores.items()
            }

            backends.append(
                {
                    "backend_name": backend.backend_name,
                    "weighted_total": backend.weighted_total,
                    "latency_p95": backend.latency_stats.p95,
                    "latency_mean": backend.latency_stats.mean,
                    "document_count": backend.document_count,
                    "criterion_scores": criterion_scores,
                }
            )

        return {"backends": backends}

    def _build_winner_section(self) -> Dict[str, Any]:
        """Build winner selection section with rationale."""
        # Get production-ready models sorted by MRR
        production_ready = sorted(
            self.result.get_production_ready_models(
                self.mrr_threshold, self.ndcg_threshold
            ),
            key=lambda m: m.mrr_score,
            reverse=True,
        )

        # Select winner embedding model (best production-ready by MRR)
        embedding_winner = None
        if production_ready:
            winner = production_ready[0]
            embedding_winner = {
                "model_name": winner.model_name,
                "provider": winner.provider,
                "mrr_score": winner.mrr_score,
                "ndcg_score": winner.ndcg_score,
                "latency_p95": winner.latency_stats.p95,
            }

        # Select winner lexical backend (highest weighted_total)
        lexical_winner = None
        best_backend = self.result.get_best_lexical_backend()
        if best_backend:
            lexical_winner = {
                "backend_name": best_backend.backend_name,
                "weighted_total": best_backend.weighted_total,
                "latency_p95": best_backend.latency_stats.p95,
            }

        # Build rationale
        rationale = self._build_rationale(embedding_winner, lexical_winner)

        return {
            "embedding_model": embedding_winner,
            "lexical_backend": lexical_winner,
            "rationale": rationale,
        }

    def _build_rationale(
        self,
        embedding_winner: Optional[Dict],
        lexical_winner: Optional[Dict],
    ) -> str:
        """Build rationale text explaining winner selections."""
        parts = []

        if embedding_winner:
            parts.append(
                f"{embedding_winner['model_name']} was selected as the best embedding model "
                f"with MRR={embedding_winner['mrr_score']:.3f} and NDCG={embedding_winner['ndcg_score']:.3f}, "
                f"meeting production thresholds (MRR >= {self.mrr_threshold}, NDCG >= {self.ndcg_threshold})."
            )
        else:
            parts.append(
                f"No embedding model met the production thresholds "
                f"(MRR >= {self.mrr_threshold}, NDCG >= {self.ndcg_threshold})."
            )

        if lexical_winner:
            parts.append(
                f"{lexical_winner['backend_name']} was selected as the best lexical backend "
                f"with a weighted decision matrix score of {lexical_winner['weighted_total']:.3f}."
            )

        return " ".join(parts)
