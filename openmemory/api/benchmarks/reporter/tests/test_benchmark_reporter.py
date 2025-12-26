"""Tests for benchmark reporter.

TDD: Tests written first to define the contract for benchmark report generation.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock

from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyStats
from openmemory.api.benchmarks.lexical.decision_matrix.criteria import CriterionName
from openmemory.api.benchmarks.runner.results import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelResult,
    LexicalBackendResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_latency_stats():
    """Create sample latency stats for testing."""
    return LatencyStats(
        p50=10.0,
        p95=25.0,
        p99=50.0,
        mean=12.0,
        min=5.0,
        max=100.0,
        count=100,
    )


@pytest.fixture
def sample_lexical_latency_stats():
    """Create sample latency stats for lexical backends."""
    return LatencyStats(
        p50=5.0,
        p95=15.0,
        p99=30.0,
        mean=7.0,
        min=2.0,
        max=50.0,
        count=100,
    )


@pytest.fixture
def production_ready_model(sample_latency_stats):
    """Create a model that meets production thresholds."""
    return ModelResult(
        model_name="qwen3-8b",
        provider="ollama",
        mrr_score=0.85,
        ndcg_score=0.90,
        latency_stats=sample_latency_stats,
        num_queries=100,
        embedding_dimensions=4096,
    )


@pytest.fixture
def non_production_ready_model(sample_latency_stats):
    """Create a model that fails production thresholds."""
    return ModelResult(
        model_name="weak-model",
        provider="local",
        mrr_score=0.50,
        ndcg_score=0.60,
        latency_stats=sample_latency_stats,
        num_queries=100,
        embedding_dimensions=768,
    )


@pytest.fixture
def borderline_model(sample_latency_stats):
    """Create a model at exactly the threshold."""
    return ModelResult(
        model_name="borderline-model",
        provider="local",
        mrr_score=0.75,  # Exactly at threshold
        ndcg_score=0.80,  # Exactly at threshold
        latency_stats=sample_latency_stats,
        num_queries=100,
        embedding_dimensions=1024,
    )


@pytest.fixture
def sample_criterion_scores():
    """Create sample criterion scores for lexical backends."""
    return {
        CriterionName.LATENCY: 0.9,
        CriterionName.OPS_COMPLEXITY: 0.95,
        CriterionName.SCALABILITY: 0.6,
        CriterionName.FEATURE_SUPPORT: 0.7,
    }


@pytest.fixture
def tantivy_result(sample_lexical_latency_stats, sample_criterion_scores):
    """Create a Tantivy backend result."""
    return LexicalBackendResult(
        backend_name="tantivy",
        criterion_scores=sample_criterion_scores,
        weighted_total=0.82,
        latency_stats=sample_lexical_latency_stats,
        document_count=1000,
    )


@pytest.fixture
def opensearch_result(sample_lexical_latency_stats):
    """Create an OpenSearch backend result."""
    return LexicalBackendResult(
        backend_name="opensearch",
        criterion_scores={
            CriterionName.LATENCY: 0.7,
            CriterionName.OPS_COMPLEXITY: 0.5,
            CriterionName.SCALABILITY: 0.9,
            CriterionName.FEATURE_SUPPORT: 0.95,
        },
        weighted_total=0.74,
        latency_stats=sample_lexical_latency_stats,
        document_count=1000,
    )


@pytest.fixture
def benchmark_config():
    """Create a sample benchmark configuration."""
    return BenchmarkConfig(
        dataset_name="codesearchnet",
        dataset_language="python",
        dataset_split="test",
        sample_limit=100,
        embedding_models=["qwen3-8b", "nomic", "weak-model"],
        lexical_backends=["tantivy", "opensearch"],
        mrr_k=10,
        ndcg_k=10,
    )


@pytest.fixture
def full_benchmark_result(
    benchmark_config,
    production_ready_model,
    non_production_ready_model,
    tantivy_result,
    opensearch_result,
    sample_latency_stats,
):
    """Create a complete benchmark result with multiple models and backends."""
    # Add a second production-ready model
    nomic_model = ModelResult(
        model_name="nomic",
        provider="local",
        mrr_score=0.78,
        ndcg_score=0.82,
        latency_stats=sample_latency_stats,
        num_queries=100,
        embedding_dimensions=768,
    )

    return BenchmarkResult(
        config=benchmark_config,
        model_results=[production_ready_model, nomic_model, non_production_ready_model],
        lexical_results=[tantivy_result, opensearch_result],
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        total_duration_seconds=120.5,
    )


@pytest.fixture
def empty_benchmark_result(benchmark_config):
    """Create a benchmark result with no results."""
    return BenchmarkResult(
        config=benchmark_config,
        model_results=[],
        lexical_results=[],
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        total_duration_seconds=0.0,
    )


# ============================================================================
# Test BenchmarkReporter Class
# ============================================================================


class TestBenchmarkReporterClass:
    """Tests for BenchmarkReporter class import and creation."""

    def test_benchmark_reporter_import(self):
        """BenchmarkReporter should be importable from reporter module."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        assert BenchmarkReporter is not None

    def test_benchmark_reporter_creation(self, full_benchmark_result):
        """BenchmarkReporter should be creatable with BenchmarkResult."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert reporter is not None
        assert reporter.result == full_benchmark_result

    def test_benchmark_reporter_custom_thresholds(self, full_benchmark_result):
        """BenchmarkReporter should accept custom thresholds."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(
            full_benchmark_result,
            mrr_threshold=0.80,
            ndcg_threshold=0.85,
        )

        assert reporter.mrr_threshold == 0.80
        assert reporter.ndcg_threshold == 0.85

    def test_benchmark_reporter_default_thresholds(self, full_benchmark_result):
        """BenchmarkReporter should use v7 plan default thresholds."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)

        # Per v7 plan: MRR >= 0.75, NDCG >= 0.80
        assert reporter.mrr_threshold == 0.75
        assert reporter.ndcg_threshold == 0.80


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_report_format_import(self):
        """ReportFormat should be importable from reporter module."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import ReportFormat

        assert ReportFormat is not None

    def test_report_format_has_json(self):
        """ReportFormat should have JSON format."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import ReportFormat

        assert hasattr(ReportFormat, "JSON")
        assert ReportFormat.JSON.value == "json"

    def test_report_format_has_markdown(self):
        """ReportFormat should have Markdown format."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import ReportFormat

        assert hasattr(ReportFormat, "MARKDOWN")
        assert ReportFormat.MARKDOWN.value == "markdown"

    def test_report_format_has_console(self):
        """ReportFormat should have Console format."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import ReportFormat

        assert hasattr(ReportFormat, "CONSOLE")
        assert ReportFormat.CONSOLE.value == "console"


# ============================================================================
# Test Report Generation
# ============================================================================


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_exists(self, full_benchmark_result):
        """BenchmarkReporter should have generate_report method."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "generate_report")
        assert callable(reporter.generate_report)

    def test_generate_report_default_format_is_json(self, full_benchmark_result):
        """generate_report should default to JSON format."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()

        # Should be valid JSON
        parsed = json.loads(report)
        assert isinstance(parsed, dict)

    def test_generate_report_accepts_format_parameter(self, full_benchmark_result):
        """generate_report should accept format parameter."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)

        json_report = reporter.generate_report(format=ReportFormat.JSON)
        md_report = reporter.generate_report(format=ReportFormat.MARKDOWN)
        console_report = reporter.generate_report(format=ReportFormat.CONSOLE)

        # All should return strings
        assert isinstance(json_report, str)
        assert isinstance(md_report, str)
        assert isinstance(console_report, str)

    def test_generate_report_empty_result(self, empty_benchmark_result):
        """generate_report should handle empty results gracefully."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(empty_benchmark_result)
        report = reporter.generate_report()

        parsed = json.loads(report)
        assert parsed["model_comparison"]["models"] == []
        assert parsed["lexical_comparison"]["backends"] == []
        assert parsed["winner"]["embedding_model"] is None
        assert parsed["winner"]["lexical_backend"] is None


# ============================================================================
# Test Model Comparison Table
# ============================================================================


class TestModelComparisonTable:
    """Tests for model comparison table generation."""

    def test_model_comparison_in_json(self, full_benchmark_result):
        """JSON report should include model comparison table."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert "model_comparison" in parsed
        assert "models" in parsed["model_comparison"]
        assert len(parsed["model_comparison"]["models"]) == 3

    def test_model_comparison_fields(self, full_benchmark_result):
        """Model comparison should include required fields."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        model = parsed["model_comparison"]["models"][0]
        assert "model_name" in model
        assert "provider" in model
        assert "mrr_score" in model
        assert "ndcg_score" in model
        assert "latency_p95" in model
        assert "production_ready" in model

    def test_model_comparison_sorted_by_mrr(self, full_benchmark_result):
        """Model comparison should be sorted by MRR score descending."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        models = parsed["model_comparison"]["models"]
        mrr_scores = [m["mrr_score"] for m in models]

        # Should be in descending order
        assert mrr_scores == sorted(mrr_scores, reverse=True)

    def test_model_comparison_production_ready_flag(
        self, full_benchmark_result, production_ready_model, non_production_ready_model
    ):
        """Model comparison should correctly flag production-ready models."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        models = {m["model_name"]: m for m in parsed["model_comparison"]["models"]}

        assert models["qwen3-8b"]["production_ready"] is True
        assert models["nomic"]["production_ready"] is True
        assert models["weak-model"]["production_ready"] is False


# ============================================================================
# Test Lexical Backend Comparison
# ============================================================================


class TestLexicalBackendComparison:
    """Tests for lexical backend comparison."""

    def test_lexical_comparison_in_json(self, full_benchmark_result):
        """JSON report should include lexical backend comparison."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert "lexical_comparison" in parsed
        assert "backends" in parsed["lexical_comparison"]
        assert len(parsed["lexical_comparison"]["backends"]) == 2

    def test_lexical_comparison_fields(self, full_benchmark_result):
        """Lexical comparison should include required fields."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        backend = parsed["lexical_comparison"]["backends"][0]
        assert "backend_name" in backend
        assert "weighted_total" in backend
        assert "latency_p95" in backend
        assert "criterion_scores" in backend

    def test_lexical_comparison_sorted_by_weighted_total(self, full_benchmark_result):
        """Lexical comparison should be sorted by weighted_total descending."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        backends = parsed["lexical_comparison"]["backends"]
        scores = [b["weighted_total"] for b in backends]

        # Should be in descending order
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# Test Winner Selection
# ============================================================================


class TestWinnerSelection:
    """Tests for winner selection logic."""

    def test_winner_section_in_json(self, full_benchmark_result):
        """JSON report should include winner section."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert "winner" in parsed
        assert "embedding_model" in parsed["winner"]
        assert "lexical_backend" in parsed["winner"]

    def test_winner_embedding_model_selection(self, full_benchmark_result):
        """Winner should be the best production-ready model by MRR."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        winner = parsed["winner"]["embedding_model"]
        assert winner is not None
        assert winner["model_name"] == "qwen3-8b"  # Best MRR among production-ready

    def test_winner_lexical_backend_selection(self, full_benchmark_result):
        """Winner should be the backend with highest weighted_total."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        winner = parsed["winner"]["lexical_backend"]
        assert winner is not None
        assert winner["backend_name"] == "tantivy"  # Highest weighted_total

    def test_winner_includes_rationale(self, full_benchmark_result):
        """Winner section should include rationale for selections."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert "rationale" in parsed["winner"]
        assert isinstance(parsed["winner"]["rationale"], str)
        assert len(parsed["winner"]["rationale"]) > 0

    def test_winner_no_production_ready_models(
        self, benchmark_config, non_production_ready_model, tantivy_result
    ):
        """Winner should be None if no models meet production thresholds."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        result = BenchmarkResult(
            config=benchmark_config,
            model_results=[non_production_ready_model],
            lexical_results=[tantivy_result],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=60.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert parsed["winner"]["embedding_model"] is None

    def test_winner_borderline_model_included(
        self, benchmark_config, borderline_model, tantivy_result
    ):
        """Model at exactly threshold should be considered production-ready."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        result = BenchmarkResult(
            config=benchmark_config,
            model_results=[borderline_model],
            lexical_results=[tantivy_result],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=60.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert parsed["winner"]["embedding_model"] is not None
        assert parsed["winner"]["embedding_model"]["model_name"] == "borderline-model"


# ============================================================================
# Test Threshold Validation
# ============================================================================


class TestThresholdValidation:
    """Tests for threshold validation."""

    def test_validate_thresholds_method_exists(self, full_benchmark_result):
        """BenchmarkReporter should have validate_thresholds method."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "validate_thresholds")
        assert callable(reporter.validate_thresholds)

    def test_validate_thresholds_returns_summary(self, full_benchmark_result):
        """validate_thresholds should return validation summary."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        validation = reporter.validate_thresholds()

        assert isinstance(validation, dict)
        assert "mrr_threshold" in validation
        assert "ndcg_threshold" in validation
        assert "models_passing_mrr" in validation
        assert "models_passing_ndcg" in validation
        assert "models_production_ready" in validation
        assert "total_models" in validation

    def test_validate_thresholds_counts(self, full_benchmark_result):
        """validate_thresholds should correctly count passing models."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        validation = reporter.validate_thresholds()

        assert validation["total_models"] == 3
        # qwen3-8b (0.85) and nomic (0.78) pass MRR >= 0.75
        assert validation["models_passing_mrr"] == 2
        # qwen3-8b (0.90) and nomic (0.82) pass NDCG >= 0.80
        assert validation["models_passing_ndcg"] == 2
        # Both qwen3-8b and nomic are production-ready
        assert validation["models_production_ready"] == 2

    def test_validate_thresholds_custom_values(self, full_benchmark_result):
        """validate_thresholds should use custom threshold values."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        # Use stricter thresholds
        reporter = BenchmarkReporter(
            full_benchmark_result,
            mrr_threshold=0.80,
            ndcg_threshold=0.85,
        )
        validation = reporter.validate_thresholds()

        # Only qwen3-8b (MRR=0.85, NDCG=0.90) passes stricter thresholds
        assert validation["models_production_ready"] == 1


# ============================================================================
# Test JSON Output Format
# ============================================================================


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_json_output_is_valid_json(self, full_benchmark_result):
        """JSON output should be valid JSON."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.JSON)

        # Should not raise
        parsed = json.loads(report)
        assert isinstance(parsed, dict)

    def test_json_output_includes_metadata(self, full_benchmark_result):
        """JSON output should include metadata."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.JSON)
        parsed = json.loads(report)

        assert "metadata" in parsed
        assert "timestamp" in parsed["metadata"]
        assert "duration_seconds" in parsed["metadata"]
        assert "dataset" in parsed["metadata"]

    def test_json_output_includes_thresholds(self, full_benchmark_result):
        """JSON output should include threshold values used."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.JSON)
        parsed = json.loads(report)

        assert "thresholds" in parsed
        assert parsed["thresholds"]["mrr"] == 0.75
        assert parsed["thresholds"]["ndcg"] == 0.80

    def test_json_output_includes_summary(self, full_benchmark_result):
        """JSON output should include summary statistics."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.JSON)
        parsed = json.loads(report)

        assert "summary" in parsed
        assert "total_models" in parsed["summary"]
        assert "production_ready_count" in parsed["summary"]
        assert "total_backends" in parsed["summary"]


# ============================================================================
# Test Markdown Output Format
# ============================================================================


class TestMarkdownOutput:
    """Tests for Markdown output format."""

    def test_markdown_output_is_string(self, full_benchmark_result):
        """Markdown output should be a string."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        assert isinstance(report, str)

    def test_markdown_output_has_title(self, full_benchmark_result):
        """Markdown output should have a title."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        assert "# Benchmark Report" in report

    def test_markdown_output_has_model_table(self, full_benchmark_result):
        """Markdown output should include model comparison table."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        # Should have markdown table with headers
        assert "| Model |" in report or "| model_name |" in report.lower()
        assert "MRR" in report
        assert "NDCG" in report

    def test_markdown_output_has_backend_table(self, full_benchmark_result):
        """Markdown output should include lexical backend table."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        assert "tantivy" in report.lower()
        assert "opensearch" in report.lower()

    def test_markdown_output_has_winner_section(self, full_benchmark_result):
        """Markdown output should include winner section."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        assert "Winner" in report or "Recommendation" in report

    def test_markdown_output_has_thresholds(self, full_benchmark_result):
        """Markdown output should show threshold values."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        assert "0.75" in report  # MRR threshold
        assert "0.8" in report  # NDCG threshold (may be formatted as 0.8 or 0.80)


# ============================================================================
# Test Console Output Format
# ============================================================================


class TestConsoleOutput:
    """Tests for console-friendly output format."""

    def test_console_output_is_string(self, full_benchmark_result):
        """Console output should be a string."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.CONSOLE)

        assert isinstance(report, str)

    def test_console_output_is_compact(self, full_benchmark_result):
        """Console output should be compact (not overly verbose)."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        console_report = reporter.generate_report(format=ReportFormat.CONSOLE)
        markdown_report = reporter.generate_report(format=ReportFormat.MARKDOWN)

        # Console should be more compact than markdown
        assert len(console_report) <= len(markdown_report)

    def test_console_output_shows_winner(self, full_benchmark_result):
        """Console output should clearly show winner."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.CONSOLE)

        # Should mention the winning model
        assert "qwen3-8b" in report

    def test_console_output_shows_production_status(self, full_benchmark_result):
        """Console output should indicate production readiness."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.CONSOLE)

        # Should indicate production readiness somehow
        assert "production" in report.lower() or "ready" in report.lower()

    def test_console_output_no_escape_sequences_by_default(self, full_benchmark_result):
        """Console output should not include ANSI escape codes by default."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
            ReportFormat,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        report = reporter.generate_report(format=ReportFormat.CONSOLE)

        # ANSI escape codes start with \x1b[
        assert "\x1b[" not in report


# ============================================================================
# Test Export Methods
# ============================================================================


class TestExportMethods:
    """Tests for export convenience methods."""

    def test_to_json_method(self, full_benchmark_result):
        """BenchmarkReporter should have to_json method."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "to_json")

        json_str = reporter.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_markdown_method(self, full_benchmark_result):
        """BenchmarkReporter should have to_markdown method."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "to_markdown")

        md = reporter.to_markdown()
        assert "# Benchmark Report" in md

    def test_to_console_method(self, full_benchmark_result):
        """BenchmarkReporter should have to_console method."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "to_console")

        console = reporter.to_console()
        assert isinstance(console, str)

    def test_to_dict_method(self, full_benchmark_result):
        """BenchmarkReporter should have to_dict method for programmatic access."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        reporter = BenchmarkReporter(full_benchmark_result)
        assert hasattr(reporter, "to_dict")

        result = reporter.to_dict()
        assert isinstance(result, dict)
        assert "model_comparison" in result
        assert "lexical_comparison" in result
        assert "winner" in result


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_model_result(
        self, benchmark_config, production_ready_model, tantivy_result
    ):
        """Reporter should handle single model result."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        result = BenchmarkResult(
            config=benchmark_config,
            model_results=[production_ready_model],
            lexical_results=[tantivy_result],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=30.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert len(parsed["model_comparison"]["models"]) == 1
        assert parsed["winner"]["embedding_model"]["model_name"] == "qwen3-8b"

    def test_only_lexical_results(
        self, benchmark_config, tantivy_result, opensearch_result
    ):
        """Reporter should handle no embedding models."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=[],
            lexical_backends=["tantivy", "opensearch"],
        )

        result = BenchmarkResult(
            config=config,
            model_results=[],
            lexical_results=[tantivy_result, opensearch_result],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=30.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        assert parsed["model_comparison"]["models"] == []
        assert parsed["winner"]["embedding_model"] is None
        assert parsed["winner"]["lexical_backend"]["backend_name"] == "tantivy"

    def test_models_with_identical_scores(self, benchmark_config, sample_latency_stats):
        """Reporter should handle models with identical scores deterministically."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        model1 = ModelResult(
            model_name="model-a",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.85,
            latency_stats=sample_latency_stats,
            num_queries=100,
            embedding_dimensions=1024,
        )

        model2 = ModelResult(
            model_name="model-b",
            provider="local",
            mrr_score=0.80,  # Same score
            ndcg_score=0.85,  # Same score
            latency_stats=sample_latency_stats,
            num_queries=100,
            embedding_dimensions=1024,
        )

        result = BenchmarkResult(
            config=benchmark_config,
            model_results=[model1, model2],
            lexical_results=[],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=30.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        # Should pick one deterministically (first one in case of tie)
        assert parsed["winner"]["embedding_model"] is not None

    def test_very_low_scores(self, benchmark_config, sample_latency_stats):
        """Reporter should handle very low scores gracefully."""
        from openmemory.api.benchmarks.reporter.benchmark_reporter import (
            BenchmarkReporter,
        )

        model = ModelResult(
            model_name="bad-model",
            provider="local",
            mrr_score=0.01,
            ndcg_score=0.02,
            latency_stats=sample_latency_stats,
            num_queries=100,
            embedding_dimensions=1024,
        )

        result = BenchmarkResult(
            config=benchmark_config,
            model_results=[model],
            lexical_results=[],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=30.0,
        )

        reporter = BenchmarkReporter(result)
        report = reporter.generate_report()
        parsed = json.loads(report)

        # Model should be in comparison but not production-ready
        assert len(parsed["model_comparison"]["models"]) == 1
        assert parsed["model_comparison"]["models"][0]["production_ready"] is False
        assert parsed["winner"]["embedding_model"] is None
