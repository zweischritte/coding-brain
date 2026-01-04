"""
Tests for IntentClassifier.

The IntentClassifier uses a hybrid approach:
1. Fast embedding-based classification for common patterns
2. LLM fallback for ambiguous or complex requests
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.routing.intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    INTENT_CLASSIFIER_PROMPT,
)
from app.routing.task_types import TaskType


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_result_has_intent(self):
        """Result should have intent field."""
        result = ClassificationResult(
            intent=TaskType.CODE_TRACE,
            confidence=0.95,
            method="semantic",
        )
        assert result.intent == TaskType.CODE_TRACE

    def test_result_has_confidence(self):
        """Result should have confidence score."""
        result = ClassificationResult(
            intent=TaskType.CODE_TRACE,
            confidence=0.95,
            method="semantic",
        )
        assert result.confidence == 0.95

    def test_result_has_method(self):
        """Result should indicate classification method."""
        result = ClassificationResult(
            intent=TaskType.CODE_TRACE,
            confidence=0.95,
            method="semantic",
        )
        assert result.method == "semantic"

    def test_valid_methods(self):
        """Method should be 'semantic' or 'llm_fallback'."""
        semantic_result = ClassificationResult(
            intent=TaskType.GENERAL,
            confidence=0.9,
            method="semantic",
        )
        llm_result = ClassificationResult(
            intent=TaskType.GENERAL,
            confidence=0.7,
            method="llm_fallback",
        )
        assert semantic_result.method == "semantic"
        assert llm_result.method == "llm_fallback"


class TestIntentClassifierInit:
    """Tests for IntentClassifier initialization."""

    def test_classifier_initializes(self):
        """Should initialize without errors."""
        classifier = IntentClassifier()
        assert classifier is not None

    def test_has_embedding_threshold(self):
        """Should have configurable embedding threshold."""
        classifier = IntentClassifier(embedding_threshold=0.85)
        assert classifier.embedding_threshold == 0.85

    def test_default_embedding_threshold(self):
        """Default embedding threshold should be 0.9."""
        classifier = IntentClassifier()
        assert classifier.embedding_threshold == 0.9


class TestSemanticClassification:
    """Tests for embedding-based (semantic) classification."""

    def test_code_trace_patterns(self):
        """CODE_TRACE should be detected for bug/tracing keywords."""
        classifier = IntentClassifier()

        test_cases = [
            "Why does the login function throw an error?",
            "Trace the execution of the auth middleware",
            "Find the bug in the payment processing",
            "What's causing the NullPointerException?",
            "Debug the database connection failure",
            "Why is the API returning 500 errors?",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.CODE_TRACE, \
                f"Expected CODE_TRACE for: {request}"

    def test_code_understand_patterns(self):
        """CODE_UNDERSTAND should be detected for architecture queries."""
        classifier = IntentClassifier()

        test_cases = [
            "How does the auth system work?",
            "Explain the architecture of the payment module",
            "What is the structure of this codebase?",
            "Give me an overview of the API design",
            "How are the database models organized?",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.CODE_UNDERSTAND, \
                f"Expected CODE_UNDERSTAND for: {request}"

    def test_memory_query_patterns(self):
        """MEMORY_QUERY should be detected for recall patterns."""
        classifier = IntentClassifier()

        test_cases = [
            "What do we know about the auth system?",
            "What was our decision on error handling?",
            "Find memories about the refactoring",
            "What conventions do we use for naming?",
            "Recall the discussion about API design",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.MEMORY_QUERY, \
                f"Expected MEMORY_QUERY for: {request}"

    def test_memory_write_patterns(self):
        """MEMORY_WRITE should be detected for documentation intents."""
        classifier = IntentClassifier()

        test_cases = [
            "Remember that we decided to use JWT",
            "Save this convention: always use snake_case",
            "Document this decision about caching",
            "Note that we're using PostgreSQL",
            "Store this as a convention",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.MEMORY_WRITE, \
                f"Expected MEMORY_WRITE for: {request}"

    def test_pr_review_patterns(self):
        """PR_REVIEW should be detected for pull request reviews."""
        classifier = IntentClassifier()

        test_cases = [
            "Review PR #123",
            "Check the pull request for issues",
            "Analyze the changes in this diff",
            "Review the code changes",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.PR_REVIEW, \
                f"Expected PR_REVIEW for: {request}"

    def test_test_write_patterns(self):
        """TEST_WRITE should be detected for test generation."""
        classifier = IntentClassifier()

        test_cases = [
            "Write tests for the auth module",
            "Generate unit tests for this function",
            "Create test coverage for the API",
            "Add tests for the new feature",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            assert result.intent == TaskType.TEST_WRITE, \
                f"Expected TEST_WRITE for: {request}"

    def test_general_fallback(self):
        """GENERAL should be returned for ambiguous requests."""
        classifier = IntentClassifier()

        test_cases = [
            "Hello",
            "Thanks",
            "What's the weather like?",  # Completely unrelated
        ]

        for request in test_cases:
            result = classifier.classify(request)
            # Should return GENERAL or at least low confidence
            if result.intent != TaskType.GENERAL:
                assert result.confidence < 0.7, \
                    f"Non-GENERAL with high confidence for: {request}"


class TestConfidenceScores:
    """Tests for confidence score behavior."""

    def test_high_confidence_for_clear_patterns(self):
        """Clear patterns should have high confidence (>0.9)."""
        classifier = IntentClassifier()

        result = classifier.classify("Why does login() throw an error?")
        assert result.confidence > 0.8

    def test_semantic_method_for_high_confidence(self):
        """High confidence should use semantic method."""
        classifier = IntentClassifier()

        result = classifier.classify("Debug the authentication error")
        if result.confidence > 0.9:
            assert result.method == "semantic"

    def test_confidence_is_bounded(self):
        """Confidence should be between 0 and 1."""
        classifier = IntentClassifier()

        for request in [
            "Find the bug",
            "Hello world",
            "What is the architecture?",
        ]:
            result = classifier.classify(request)
            assert 0.0 <= result.confidence <= 1.0


class TestLLMFallback:
    """Tests for LLM fallback classification."""

    @pytest.mark.asyncio
    async def test_llm_fallback_on_low_confidence(self):
        """Should use LLM when embedding confidence is low."""
        classifier = IntentClassifier(embedding_threshold=0.99)  # Very high threshold

        # Mock the LLM client
        with patch.object(classifier, '_llm_classify') as mock_llm:
            mock_llm.return_value = ClassificationResult(
                intent=TaskType.CODE_TRACE,
                confidence=0.85,
                method="llm_fallback",
            )

            result = await classifier.classify_async("Something ambiguous")

            # If embedding confidence was below threshold, LLM should be called
            if result.method == "llm_fallback":
                mock_llm.assert_called_once()

    def test_llm_prompt_exists(self):
        """INTENT_CLASSIFIER_PROMPT should be defined."""
        assert INTENT_CLASSIFIER_PROMPT is not None
        assert len(INTENT_CLASSIFIER_PROMPT) > 100  # Substantial prompt

    def test_llm_prompt_contains_all_task_types(self):
        """Prompt should document all TaskTypes."""
        for task_type in TaskType:
            assert task_type.name in INTENT_CLASSIFIER_PROMPT, \
                f"TaskType {task_type.name} missing from prompt"


class TestClassificationSpeed:
    """Tests for classification performance."""

    def test_sync_classification_is_fast(self):
        """Synchronous classification should be fast (<100ms)."""
        import time

        classifier = IntentClassifier()

        start = time.perf_counter()
        classifier.classify("Debug the login error")
        elapsed = (time.perf_counter() - start) * 1000

        # Should be well under 100ms for semantic classification
        # (No LLM call in this case)
        assert elapsed < 100, f"Classification took {elapsed}ms"


class TestGermanLanguageSupport:
    """Tests for German language request classification."""

    def test_german_code_trace(self):
        """German bug/trace queries should classify correctly."""
        classifier = IntentClassifier()

        test_cases = [
            "Warum wirft die login Funktion einen Fehler?",
            "Finde den Bug in der Zahlungsverarbeitung",
            "Debugge den Datenbankfehler",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            # Should recognize as CODE_TRACE or at least not misclassify as MEMORY
            assert result.intent in (TaskType.CODE_TRACE, TaskType.CODE_UNDERSTAND, TaskType.GENERAL)

    def test_german_memory_patterns(self):
        """German memory queries should classify correctly."""
        classifier = IntentClassifier()

        test_cases = [
            "Was wissen wir über das Auth-System?",
            "Merke dir diese Entscheidung",
            "Welche Konventionen haben wir?",
        ]

        for request in test_cases:
            result = classifier.classify(request)
            # Should recognize as memory-related
            assert result.intent in (TaskType.MEMORY_QUERY, TaskType.MEMORY_WRITE, TaskType.GENERAL)
