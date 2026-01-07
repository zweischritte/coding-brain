"""
Tests for the Simplified Evaluation Framework (PRD-08-SIMPLE).

TDD: These tests are written BEFORE the implementation.
Tests cover:
- Golden dataset loading
- Context precision metric
- Answer relevancy metric
- Single sample evaluation
- Full evaluation run
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile


# ============================================================================
# SECTION 1: Golden Dataset Loading Tests
# ============================================================================


class TestLoadGoldenDataset:
    """Tests for loading the golden dataset from JSON file."""

    @pytest.fixture
    def sample_golden_data(self):
        """Sample golden dataset JSON structure."""
        return {
            "version": "1.0",
            "samples": [
                {
                    "id": "local_001",
                    "query": "What database does the AuthService use?",
                    "query_type": "local",
                    "expected_memory_ids": ["mem_abc123"],
                    "expected_answer_keywords": ["PostgreSQL", "database", "auth"],
                },
                {
                    "id": "global_001",
                    "query": "What are the main architectural patterns?",
                    "query_type": "global",
                    "expected_memory_ids": [],
                    "expected_answer_keywords": ["architecture", "pattern"],
                },
                {
                    "id": "code_001",
                    "query": "What functions call the authenticate method?",
                    "query_type": "code",
                    "expected_memory_ids": ["mem_xyz789"],
                    "expected_answer_keywords": ["authenticate", "caller"],
                },
            ],
        }

    def test_load_golden_dataset_returns_list_of_golden_samples(self, sample_golden_data):
        """Load function returns list of GoldenSample dataclasses."""
        from app.evaluation.simple_eval import load_golden_dataset, GoldenSample

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_golden_data, f)
            f.flush()

            samples = load_golden_dataset(f.name)

            assert isinstance(samples, list)
            assert len(samples) == 3
            assert all(isinstance(s, GoldenSample) for s in samples)

    def test_golden_sample_has_correct_fields(self, sample_golden_data):
        """GoldenSample has all required fields populated correctly."""
        from app.evaluation.simple_eval import load_golden_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_golden_data, f)
            f.flush()

            samples = load_golden_dataset(f.name)
            local_sample = samples[0]

            assert local_sample.id == "local_001"
            assert local_sample.query == "What database does the AuthService use?"
            assert local_sample.query_type == "local"
            assert local_sample.expected_memory_ids == ["mem_abc123"]
            assert local_sample.expected_answer_keywords == ["PostgreSQL", "database", "auth"]

    def test_load_handles_empty_expected_fields(self, sample_golden_data):
        """Load handles samples with empty expected_memory_ids gracefully."""
        from app.evaluation.simple_eval import load_golden_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_golden_data, f)
            f.flush()

            samples = load_golden_dataset(f.name)
            global_sample = samples[1]

            assert global_sample.expected_memory_ids == []

    def test_load_raises_on_missing_file(self):
        """Load raises FileNotFoundError for non-existent file."""
        from app.evaluation.simple_eval import load_golden_dataset

        with pytest.raises(FileNotFoundError):
            load_golden_dataset("/nonexistent/path/golden.json")

    def test_load_raises_on_invalid_json(self):
        """Load raises JSONDecodeError for invalid JSON."""
        from app.evaluation.simple_eval import load_golden_dataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()

            with pytest.raises(json.JSONDecodeError):
                load_golden_dataset(f.name)


# ============================================================================
# SECTION 2: Context Precision Metric Tests
# ============================================================================


class TestEvaluateContextPrecision:
    """Tests for the context precision metric."""

    def test_perfect_precision_when_all_retrieved_are_expected(self):
        """Precision is 1.0 when all retrieved IDs match expected."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = ["mem_1", "mem_2", "mem_3"]
        expected = ["mem_1", "mem_2", "mem_3"]

        score = evaluate_context_precision(retrieved, expected)
        assert score == 1.0

    def test_zero_precision_when_no_matches(self):
        """Precision is 0.0 when no retrieved IDs match expected."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = ["mem_1", "mem_2"]
        expected = ["mem_x", "mem_y"]

        score = evaluate_context_precision(retrieved, expected)
        assert score == 0.0

    def test_partial_precision(self):
        """Precision is proportional when some IDs match."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = ["mem_1", "mem_2", "mem_3", "mem_4"]  # 4 retrieved
        expected = ["mem_1", "mem_2"]  # 2 match

        score = evaluate_context_precision(retrieved, expected)
        assert score == 0.5  # 2/4 = 0.5

    def test_precision_with_empty_retrieved(self):
        """Precision is 0.0 when nothing retrieved."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = []
        expected = ["mem_1", "mem_2"]

        score = evaluate_context_precision(retrieved, expected)
        assert score == 0.0

    def test_neutral_score_when_no_expected_ids(self):
        """Returns 0.5 baseline when no ground truth expected IDs."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = ["mem_1", "mem_2"]
        expected = []  # No ground truth

        score = evaluate_context_precision(retrieved, expected)
        assert score == 0.5

    def test_precision_handles_duplicate_retrieved_ids(self):
        """Precision treats duplicates correctly (set intersection)."""
        from app.evaluation.simple_eval import evaluate_context_precision

        retrieved = ["mem_1", "mem_1", "mem_2"]  # 3 items but only 2 unique
        expected = ["mem_1", "mem_2"]

        # Retrieved has 3 items, 2 unique match expected
        # Interpretation: len(intersection) / len(retrieved) = 2/3
        score = evaluate_context_precision(retrieved, expected)
        assert score == pytest.approx(2 / 3, rel=0.01)


# ============================================================================
# SECTION 3: Answer Relevancy Metric Tests
# ============================================================================


class TestEvaluateAnswerRelevancy:
    """Tests for the answer relevancy metric (keyword-based)."""

    def test_perfect_relevancy_all_keywords_present(self):
        """Relevancy is 1.0 when all keywords found in answer."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "We use PostgreSQL as our database for the auth service."
        keywords = ["PostgreSQL", "database", "auth"]

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 1.0

    def test_zero_relevancy_no_keywords_present(self):
        """Relevancy is 0.0 when no keywords found."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "I don't know the answer to that question."
        keywords = ["PostgreSQL", "database", "auth"]

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 0.0

    def test_partial_relevancy(self):
        """Relevancy is proportional to keyword hits."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "We use PostgreSQL for storage."  # Only 1 of 3 keywords
        keywords = ["PostgreSQL", "database", "auth"]

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == pytest.approx(1 / 3, rel=0.01)

    def test_case_insensitive_matching(self):
        """Keyword matching is case-insensitive."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "POSTGRESQL is our DATABASE for AUTH."
        keywords = ["PostgreSQL", "database", "auth"]

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 1.0

    def test_neutral_score_when_no_keywords(self):
        """Returns 0.5 when no expected keywords provided."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "Some answer text."
        keywords = []

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 0.5

    def test_neutral_score_when_empty_answer(self):
        """Returns 0.5 when answer is empty."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = ""
        keywords = ["PostgreSQL", "database"]

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 0.5

    def test_partial_word_matching(self):
        """Keywords can match as substrings."""
        from app.evaluation.simple_eval import evaluate_answer_relevancy

        answer = "The authentication flow uses tokens."
        keywords = ["auth"]  # 'auth' is in 'authentication'

        score = evaluate_answer_relevancy(answer, keywords)
        assert score == 1.0


# ============================================================================
# SECTION 4: EvalResult Dataclass Tests
# ============================================================================


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_eval_result_creation(self):
        """EvalResult can be created with all required fields."""
        from app.evaluation.simple_eval import EvalResult

        result = EvalResult(
            sample_id="local_001",
            query_type="local",
            context_precision=0.75,
            answer_relevancy=0.85,
            latency_ms=150.5,
        )

        assert result.sample_id == "local_001"
        assert result.query_type == "local"
        assert result.context_precision == 0.75
        assert result.answer_relevancy == 0.85
        assert result.latency_ms == 150.5


# ============================================================================
# SECTION 5: Evaluate Sample Async Function Tests
# ============================================================================


class TestEvaluateSample:
    """Tests for the evaluate_sample async function."""

    @pytest.fixture
    def sample(self):
        """Sample GoldenSample for testing."""
        from app.evaluation.simple_eval import GoldenSample

        return GoldenSample(
            id="local_001",
            query="What database does the AuthService use?",
            query_type="local",
            expected_memory_ids=["mem_1", "mem_2"],
            expected_answer_keywords=["PostgreSQL", "database"],
        )

    @pytest.mark.asyncio
    async def test_evaluate_sample_returns_eval_result(self, sample):
        """evaluate_sample returns an EvalResult with correct sample_id."""
        from app.evaluation.simple_eval import evaluate_sample, EvalResult

        async def mock_search(query, limit):
            return {
                "results": [
                    {"id": "mem_1", "memory": "We use PostgreSQL database."},
                    {"id": "mem_2", "memory": "Auth service config."},
                ]
            }

        result = await evaluate_sample(sample, mock_search)

        assert isinstance(result, EvalResult)
        assert result.sample_id == "local_001"
        assert result.query_type == "local"

    @pytest.mark.asyncio
    async def test_evaluate_sample_measures_latency(self, sample):
        """evaluate_sample measures latency in milliseconds."""
        from app.evaluation.simple_eval import evaluate_sample
        import asyncio

        async def slow_search(query, limit):
            await asyncio.sleep(0.05)  # 50ms delay
            return {"results": []}

        result = await evaluate_sample(sample, slow_search)

        assert result.latency_ms >= 50  # At least 50ms

    @pytest.mark.asyncio
    async def test_evaluate_sample_calculates_precision(self, sample):
        """evaluate_sample correctly calculates context precision."""
        from app.evaluation.simple_eval import evaluate_sample

        async def mock_search(query, limit):
            return {
                "results": [
                    {"id": "mem_1"},  # Match
                    {"id": "mem_3"},  # No match
                ]
            }

        result = await evaluate_sample(sample, mock_search)

        # 1 of 2 retrieved matches expected, precision = 0.5
        assert result.context_precision == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_sample_calculates_relevancy_from_results(self, sample):
        """evaluate_sample calculates relevancy from concatenated memory content."""
        from app.evaluation.simple_eval import evaluate_sample

        async def mock_search(query, limit):
            return {
                "results": [
                    {"id": "mem_1", "memory": "We use PostgreSQL."},  # 1 keyword
                    {"id": "mem_2", "memory": "The database is configured."},  # 1 keyword
                ]
            }

        result = await evaluate_sample(sample, mock_search)

        # Both keywords found across results, relevancy = 1.0
        assert result.answer_relevancy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_sample_handles_empty_results(self, sample):
        """evaluate_sample handles empty search results gracefully."""
        from app.evaluation.simple_eval import evaluate_sample

        async def mock_search(query, limit):
            return {"results": []}

        result = await evaluate_sample(sample, mock_search)

        assert result.context_precision == 0.0
        assert result.answer_relevancy == 0.5  # Neutral when no answer


# ============================================================================
# SECTION 6: Run Evaluation Tests
# ============================================================================


class TestRunEvaluation:
    """Tests for the run_evaluation async function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a temporary golden dataset file."""
        data = {
            "version": "1.0",
            "samples": [
                {
                    "id": "local_001",
                    "query": "What database?",
                    "query_type": "local",
                    "expected_memory_ids": ["mem_1"],
                    "expected_answer_keywords": ["PostgreSQL"],
                },
                {
                    "id": "global_001",
                    "query": "Architecture?",
                    "query_type": "global",
                    "expected_memory_ids": [],
                    "expected_answer_keywords": ["pattern"],
                },
                {
                    "id": "code_001",
                    "query": "Who calls authenticate?",
                    "query_type": "code",
                    "expected_memory_ids": ["mem_2"],
                    "expected_answer_keywords": ["caller"],
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            return f.name

    @pytest.mark.asyncio
    async def test_run_evaluation_returns_summary(self, sample_dataset):
        """run_evaluation returns summary dict with expected keys."""
        from app.evaluation.simple_eval import run_evaluation

        async def mock_search(query, limit):
            return {
                "results": [
                    {"id": "mem_1", "memory": "PostgreSQL database."},
                ]
            }

        summary = await run_evaluation(mock_search, sample_dataset)

        assert "total_samples" in summary
        assert "overall" in summary
        assert "by_type" in summary

    @pytest.mark.asyncio
    async def test_run_evaluation_counts_samples(self, sample_dataset):
        """run_evaluation correctly counts total samples."""
        from app.evaluation.simple_eval import run_evaluation

        async def mock_search(query, limit):
            return {"results": []}

        summary = await run_evaluation(mock_search, sample_dataset)

        assert summary["total_samples"] == 3

    @pytest.mark.asyncio
    async def test_run_evaluation_computes_overall_averages(self, sample_dataset):
        """run_evaluation computes overall average metrics."""
        from app.evaluation.simple_eval import run_evaluation

        async def mock_search(query, limit):
            return {
                "results": [{"id": "mem_1", "memory": "PostgreSQL pattern caller"}]
            }

        summary = await run_evaluation(mock_search, sample_dataset)

        overall = summary["overall"]
        assert "avg_precision" in overall
        assert "avg_relevancy" in overall
        assert "avg_latency_ms" in overall
        assert 0.0 <= overall["avg_precision"] <= 1.0
        assert 0.0 <= overall["avg_relevancy"] <= 1.0

    @pytest.mark.asyncio
    async def test_run_evaluation_groups_by_query_type(self, sample_dataset):
        """run_evaluation groups results by query type."""
        from app.evaluation.simple_eval import run_evaluation

        async def mock_search(query, limit):
            return {"results": []}

        summary = await run_evaluation(mock_search, sample_dataset)

        by_type = summary["by_type"]
        assert "local" in by_type
        assert "global" in by_type
        assert "code" in by_type
        assert by_type["local"]["count"] == 1
        assert by_type["global"]["count"] == 1
        assert by_type["code"]["count"] == 1

    @pytest.mark.asyncio
    async def test_run_evaluation_handles_sample_failures(self, sample_dataset):
        """run_evaluation continues after sample failures."""
        from app.evaluation.simple_eval import run_evaluation

        call_count = 0

        async def failing_search(query, limit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated failure")
            return {"results": []}

        summary = await run_evaluation(mock_search=failing_search, dataset_path=sample_dataset)

        # Should have processed 2 samples (1 failed)
        assert summary["total_samples"] == 2


# ============================================================================
# SECTION 7: Print Summary Tests
# ============================================================================


class TestPrintSummary:
    """Tests for the print_summary function."""

    def test_print_summary_outputs_to_console(self, capsys):
        """print_summary outputs formatted text to console."""
        from app.evaluation.simple_eval import print_summary

        summary = {
            "total_samples": 30,
            "overall": {
                "avg_precision": 0.72,
                "avg_relevancy": 0.81,
                "avg_latency_ms": 450.0,
            },
            "by_type": {
                "local": {
                    "count": 10,
                    "avg_precision": 0.78,
                    "avg_relevancy": 0.85,
                    "avg_latency_ms": 320.0,
                },
                "global": {
                    "count": 10,
                    "avg_precision": 0.65,
                    "avg_relevancy": 0.74,
                    "avg_latency_ms": 580.0,
                },
                "code": {
                    "count": 10,
                    "avg_precision": 0.73,
                    "avg_relevancy": 0.84,
                    "avg_latency_ms": 450.0,
                },
            },
        }

        print_summary(summary)
        captured = capsys.readouterr()

        assert "EVALUATION SUMMARY" in captured.out
        assert "Total samples: 30" in captured.out
        assert "Context Precision:" in captured.out
        assert "Answer Relevancy:" in captured.out
        assert "LOCAL" in captured.out
        assert "GLOBAL" in captured.out
        assert "CODE" in captured.out


# ============================================================================
# SECTION 8: Integration Tests with Mocked search_memory
# ============================================================================


class TestGoldenDatasetFile:
    """Tests for the actual golden_30.json dataset file."""

    def test_golden_30_file_exists_and_loads(self):
        """The actual golden_30.json file exists and loads correctly."""
        from app.evaluation.simple_eval import load_golden_dataset
        import os

        # Path relative to openmemory/api directory
        dataset_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "golden_30.json"
        )

        samples = load_golden_dataset(dataset_path)

        assert len(samples) == 30
        assert sum(1 for s in samples if s.query_type == "local") == 10
        assert sum(1 for s in samples if s.query_type == "global") == 10
        assert sum(1 for s in samples if s.query_type == "code") == 10

    def test_golden_30_all_samples_have_keywords(self):
        """All samples in golden_30.json have expected_answer_keywords."""
        from app.evaluation.simple_eval import load_golden_dataset
        import os

        dataset_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "golden_30.json"
        )

        samples = load_golden_dataset(dataset_path)

        for sample in samples:
            assert len(sample.expected_answer_keywords) > 0, (
                f"Sample {sample.id} has no keywords"
            )

    def test_golden_30_sample_ids_are_unique(self):
        """All sample IDs in golden_30.json are unique."""
        from app.evaluation.simple_eval import load_golden_dataset
        import os

        dataset_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "golden_30.json"
        )

        samples = load_golden_dataset(dataset_path)
        ids = [s.id for s in samples]

        assert len(ids) == len(set(ids)), "Duplicate sample IDs found"


class TestIntegrationWithSearchMemory:
    """Integration tests simulating real search_memory behavior."""

    @pytest.fixture
    def realistic_golden_sample(self):
        """A realistic golden sample matching the PRD."""
        from app.evaluation.simple_eval import GoldenSample

        return GoldenSample(
            id="local_003",
            query="What is our convention for error handling in TypeScript?",
            query_type="local",
            expected_memory_ids=[],  # Empty as per PRD
            expected_answer_keywords=["try", "catch", "error", "throw"],
        )

    @pytest.mark.asyncio
    async def test_realistic_search_response_format(self, realistic_golden_sample):
        """Test with realistic search_memory JSON response format."""
        from app.evaluation.simple_eval import evaluate_sample

        # Simulate actual search_memory response structure
        async def mock_search(query, limit):
            return {
                "results": [
                    {
                        "id": "mem-uuid-1",
                        "memory": "In TypeScript, we use try-catch blocks for error handling. "
                        "Always throw custom Error types with meaningful messages.",
                        "score": 0.92,
                        "category": "convention",
                        "scope": "project",
                        "entity": "ErrorHandling",
                        "created_at": "2025-12-05T10:39:18+01:00",
                    },
                    {
                        "id": "mem-uuid-2",
                        "memory": "Use Result types for recoverable errors, throw for unrecoverable.",
                        "score": 0.85,
                        "category": "convention",
                        "scope": "project",
                        "entity": "ErrorHandling",
                        "created_at": "2025-12-04T14:22:00+01:00",
                    },
                ]
            }

        result = await evaluate_sample(realistic_golden_sample, mock_search)

        # All 4 keywords should be found: try, catch, error, throw
        assert result.answer_relevancy == 1.0
        # Precision is neutral (0.5) since expected_memory_ids is empty
        assert result.context_precision == 0.5
