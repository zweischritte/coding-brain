"""
Intent Classifier for Task-based Routing.

Uses a hybrid approach for intent classification:
1. Fast pattern matching for common, clear patterns
2. Optional LLM fallback for ambiguous requests

The classifier aims for:
- >95% accuracy on clear patterns
- <100ms latency for synchronous classification
- Graceful degradation without LLM

Usage:
    from app.routing import classify_intent, TaskType

    result = classify_intent("Debug the login error")
    assert result.intent == TaskType.CODE_TRACE
    assert result.confidence > 0.9
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from .task_types import TaskType

logger = logging.getLogger(__name__)


# LLM Prompt for fallback classification
INTENT_CLASSIFIER_PROMPT = """
Classify the user request into ONE of these task types:

- CODE_TRACE: Finding bugs, tracing execution, understanding call chains,
              debugging errors. Keywords: "why does", "bug", "error", "trace",
              "call", "stack", "exception", "debug", "failing"

- CODE_UNDERSTAND: Architecture analysis, code exploration, learning codebase.
                   Keywords: "how does", "architecture", "structure", "overview",
                   "explain", "what is", "how is ... organized"

- MEMORY_QUERY: Retrieving past decisions, conventions, context.
                Keywords: "what do we know", "recall", "find memory",
                "what was decided", "convention", "what did we"

- MEMORY_WRITE: Documenting decisions, conventions, learnings.
                Keywords: "remember", "save", "document", "note", "store",
                "convention", "decision", "record that"

- PR_REVIEW: Reviewing code changes, pull requests.
             Keywords: "review PR", "pull request", "diff", "changes",
             "review #", "check the PR"

- TEST_WRITE: Writing or generating tests.
              Keywords: "write test", "generate test", "test for", "coverage",
              "add tests", "unit test", "integration test"

- GENERAL: Anything that doesn't clearly fit above categories.
           Use this for greetings, general questions, or unclear requests.

User request: {request}

Respond with ONLY the task type name (e.g., "CODE_TRACE").
Task type:
"""


@dataclass
class ClassificationResult:
    """
    Result of intent classification.

    Attributes:
        intent: Classified TaskType
        confidence: Confidence score (0.0-1.0)
        method: Classification method ("semantic" or "llm_fallback")
        latency_ms: Classification time in milliseconds
    """

    intent: TaskType
    confidence: float
    method: str
    latency_ms: float = 0.0


# Pattern definitions for each task type
# Format: (compiled_regex, confidence_boost)
@dataclass
class PatternRule:
    """A classification pattern rule."""

    pattern: Pattern[str]
    task_type: TaskType
    confidence: float


class IntentClassifier:
    """
    Hybrid intent classifier using patterns and optional LLM fallback.

    Primary classification is pattern-based for speed (<10ms).
    LLM fallback is used when confidence is below threshold.

    Attributes:
        embedding_threshold: Minimum confidence to skip LLM fallback
        patterns: Compiled regex patterns for classification
    """

    def __init__(
        self,
        embedding_threshold: float = 0.9,
        enable_llm_fallback: bool = True,
    ):
        """
        Initialize classifier.

        Args:
            embedding_threshold: Minimum confidence to trust pattern match
            enable_llm_fallback: Whether to use LLM for ambiguous cases
        """
        self.embedding_threshold = embedding_threshold
        self.enable_llm_fallback = enable_llm_fallback
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> List[PatternRule]:
        """
        Build compiled regex patterns for classification.

        Returns:
            List of PatternRule objects
        """
        patterns = []

        # MEMORY_QUERY patterns - MUST be checked before CODE_TRACE
        # because phrases like "decision on error handling" contain "error"
        # but should be classified as MEMORY_QUERY
        memory_query_patterns = [
            (r"\bwhat\s+do\s+we\s+know\b", 0.98),
            (r"\bwas\s+wissen\s+wir\b", 0.98),
            (r"\b(recall|erinnere)\b", 0.95),
            (r"\bfind\s+(memory|memories)\b", 0.98),
            (r"\bwhat\s+(was|were)\s+(decided|our\s+decision)\b", 0.98),
            (r"\bour\s+decision\s+on\b", 0.98),  # "our decision on X"
            (r"\bdecision\s+on\s+\w+\s+handling\b", 0.98),  # "decision on error handling"
            (r"\bconvention(s)?\s+(for|about|do\s+we)\b", 0.95),
            (r"\bwhat\s+convention(s)?\s+do\s+we\b", 0.98),  # "what conventions do we use"
            (r"\bkonvention(en)?\s+(für|über)\b", 0.95),
            (r"\bwhat\s+did\s+we\s+(decide|agree)\b", 0.98),
            (r"\bprevious\s+(decision|discussion)\b", 0.95),
            (r"\bsearch\s+memor(y|ies)\b", 0.98),
            (r"\bfind\s+memories\s+about\b", 0.98),  # "find memories about"
        ]

        for regex, conf in memory_query_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.MEMORY_QUERY,
                confidence=conf,
            ))

        # CODE_TRACE patterns - for debugging keywords
        # These have lower confidence than MEMORY_QUERY to avoid conflicts
        code_trace_patterns = [
            (r"\b(bug|exception|fehler)\b", 0.93),  # Lower than MEMORY patterns
            (r"\berror\b(?!.*\bdecision\b)", 0.93),  # "error" but not "decision on error"
            (r"\b(debug|debugge|debuggen)\b", 0.95),
            (r"\bwhy\s+(does|is|did|doesn't|isn't)\b", 0.90),
            (r"\bwarum\s+(wirft|gibt|ist|funktioniert)\b", 0.90),
            (r"\b(trace|tracing|traceback)\b", 0.95),
            (r"\b(stack\s*trace|stacktrace|call\s*stack)\b", 0.95),
            (r"\b(fails?|failing|failure)\b", 0.90),
            (r"\b(crash|crashes|crashing)\b", 0.95),
            (r"\bnullpointer|null\s*pointer\b", 0.95),
            (r"\b(500|404|403|401)\s*(error|response|status)\b", 0.90),
            (r"\b(find\s+the\s+bug|find\s+bug)\b", 0.95),
            (r"\bfinde\s+(den\s+)?(bug|fehler)\b", 0.95),
        ]

        for regex, conf in code_trace_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.CODE_TRACE,
                confidence=conf,
            ))

        # CODE_UNDERSTAND patterns
        code_understand_patterns = [
            (r"\bhow\s+does\s+.+\s+work\b", 0.95),
            (r"\bwie\s+funktioniert\b", 0.95),
            (r"\b(architecture|architektur)\b", 0.90),
            (r"\b(structure|struktur)\s+of\b", 0.90),
            (r"\boverview\s+of\b", 0.90),
            (r"\bexplain\s+(the\s+)?(code|system|module|architecture)\b", 0.90),
            (r"\bwhat\s+is\s+the\s+(purpose|design)\b", 0.85),
            (r"\bhow\s+(is|are)\s+.+\s+organized\b", 0.95),  # "how is/are X organized"
            (r"\bcodebase\s+(overview|structure)\b", 0.95),
            (r"\bgive\s+me\s+an?\s+overview\b", 0.90),  # "give me an overview"
            (r"\bwhat\s+is\s+the\s+structure\b", 0.90),  # "what is the structure"
        ]

        for regex, conf in code_understand_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.CODE_UNDERSTAND,
                confidence=conf,
            ))

        # MEMORY_WRITE patterns
        memory_write_patterns = [
            (r"\b(remember|merke)\s+(that|dir|this)\b", 0.95),
            (r"\b(save|speichere)\s+(this|diese)\b", 0.90),
            (r"\bdocument\s+(this|that|the)\b", 0.90),
            (r"\bdokumentiere\b", 0.90),
            (r"\b(note|notiere)\s+(that|this)\b", 0.90),
            (r"\bstore\s+(this|as)\b", 0.90),
            (r"\brecord\s+(that|this)\b", 0.90),
            (r"\badd\s+(to\s+)?memor(y|ies)\b", 0.95),
            (r"\b(new\s+)?convention:\b", 0.95),
            (r"\bdecision:\s+\b", 0.90),
        ]

        for regex, conf in memory_write_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.MEMORY_WRITE,
                confidence=conf,
            ))

        # PR_REVIEW patterns
        pr_review_patterns = [
            (r"\breview\s+(PR|pull\s*request)\b", 0.95),
            (r"\bPR\s*#?\d+\b", 0.90),
            (r"\bpull\s*request\s*#?\d+\b", 0.95),
            (r"\bcheck\s+(the\s+)?(diff|changes|PR|pull\s*request)\b", 0.90),  # Added pull request
            (r"\bcode\s+review\b", 0.90),
            (r"\breview\s+(the\s+)?(code\s+)?changes\b", 0.85),
            (r"\banalyze\s+(the\s+)?diff\b", 0.90),
            (r"\bpull\s*request\s+for\s+issues\b", 0.90),  # "pull request for issues"
            (r"\bchanges\s+in\s+.+\s+diff\b", 0.90),  # "changes in this diff"
            (r"\banalyze\s+(the\s+)?changes\b", 0.85),  # "analyze the changes"
        ]

        for regex, conf in pr_review_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.PR_REVIEW,
                confidence=conf,
            ))

        # TEST_WRITE patterns
        test_write_patterns = [
            (r"\b(write|generate|create)\s+test(s)?\b", 0.95),
            (r"\btest(s)?\s+for\s+\b", 0.90),
            (r"\bunit\s+test(s)?\b", 0.90),
            (r"\bintegration\s+test(s)?\b", 0.90),
            (r"\b(add|write)\s+test\s+coverage\b", 0.95),
            (r"\btest\s+generation\b", 0.95),
            (r"\bschreibe?\s+tests?\b", 0.90),
            (r"\bgeneriere?\s+tests?\b", 0.90),
        ]

        for regex, conf in test_write_patterns:
            patterns.append(PatternRule(
                pattern=re.compile(regex, re.IGNORECASE),
                task_type=TaskType.TEST_WRITE,
                confidence=conf,
            ))

        return patterns

    def classify(self, request: str) -> ClassificationResult:
        """
        Classify a request synchronously.

        Uses pattern matching for fast classification.
        Does NOT use LLM fallback (use classify_async for that).

        Args:
            request: User request text

        Returns:
            ClassificationResult with intent and confidence
        """
        start = time.perf_counter()

        # Pattern-based classification
        result = self._pattern_classify(request)

        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    async def classify_async(self, request: str) -> ClassificationResult:
        """
        Classify a request with optional LLM fallback.

        Uses pattern matching first, then LLM if confidence is low.

        Args:
            request: User request text

        Returns:
            ClassificationResult with intent and confidence
        """
        start = time.perf_counter()

        # Try pattern classification first
        result = self._pattern_classify(request)

        # LLM fallback if confidence is low
        if (
            self.enable_llm_fallback
            and result.confidence < self.embedding_threshold
        ):
            llm_result = await self._llm_classify(request)
            if llm_result is not None:
                result = llm_result

        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _pattern_classify(self, request: str) -> ClassificationResult:
        """
        Classify using regex patterns.

        Args:
            request: User request text

        Returns:
            ClassificationResult from pattern matching
        """
        best_match: Optional[Tuple[TaskType, float]] = None

        for rule in self._patterns:
            if rule.pattern.search(request):
                if best_match is None or rule.confidence > best_match[1]:
                    best_match = (rule.task_type, rule.confidence)

        if best_match:
            return ClassificationResult(
                intent=best_match[0],
                confidence=best_match[1],
                method="semantic",
            )

        # No pattern matched - return GENERAL with low confidence
        return ClassificationResult(
            intent=TaskType.GENERAL,
            confidence=0.5,
            method="semantic",
        )

    async def _llm_classify(self, request: str) -> Optional[ClassificationResult]:
        """
        Classify using LLM fallback.

        Args:
            request: User request text

        Returns:
            ClassificationResult from LLM, or None if unavailable
        """
        try:
            # Try to import and use local LLM
            from app.utils.llm_client import get_llm_client

            client = get_llm_client()
            if client is None:
                logger.debug("No LLM client available for fallback")
                return None

            prompt = INTENT_CLASSIFIER_PROMPT.format(request=request)

            response = await client.complete_async(prompt, max_tokens=20)
            response_text = response.strip().upper()

            # Parse response to TaskType
            for task_type in TaskType:
                if task_type.name in response_text:
                    return ClassificationResult(
                        intent=task_type,
                        confidence=0.85,  # LLM confidence is slightly lower
                        method="llm_fallback",
                    )

            logger.warning(f"LLM returned unparseable response: {response_text}")
            return None

        except ImportError:
            logger.debug("LLM client not available")
            return None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None


# Convenience function for simple usage
def classify_intent(
    request: str,
    return_confidence: bool = False,
) -> ClassificationResult:
    """
    Classify a request intent.

    Convenience function that creates a classifier and classifies.
    For repeated use, create an IntentClassifier instance instead.

    Args:
        request: User request text
        return_confidence: If True, return full result (always True now)

    Returns:
        ClassificationResult with intent and confidence

    Example:
        >>> result = classify_intent("Debug the login error")
        >>> result.intent
        TaskType.CODE_TRACE
    """
    classifier = IntentClassifier()
    return classifier.classify(request)
