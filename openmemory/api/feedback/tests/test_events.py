"""Tests for FeedbackEvent and related types.

TDD tests for feedback event dataclasses following the v9 plan spec:
- FeedbackEvent with outcome, timing, weights, reranker info
- FeedbackType (implicit vs explicit)
- FeedbackOutcome (accepted, modified, rejected, ignored)
"""

import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from openmemory.api.feedback.events import (
    FeedbackEvent,
    FeedbackOutcome,
    FeedbackType,
)


class TestFeedbackOutcome:
    """Tests for FeedbackOutcome enum."""

    def test_has_accepted_value(self):
        """Outcome has accepted value."""
        assert FeedbackOutcome.ACCEPTED.value == "accepted"

    def test_has_modified_value(self):
        """Outcome has modified value."""
        assert FeedbackOutcome.MODIFIED.value == "modified"

    def test_has_rejected_value(self):
        """Outcome has rejected value."""
        assert FeedbackOutcome.REJECTED.value == "rejected"

    def test_has_ignored_value(self):
        """Outcome has ignored value."""
        assert FeedbackOutcome.IGNORED.value == "ignored"

    def test_all_outcomes_are_strings(self):
        """All outcome values are strings."""
        for outcome in FeedbackOutcome:
            assert isinstance(outcome.value, str)


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_has_implicit_value(self):
        """Type has implicit value (click, copy, use)."""
        assert FeedbackType.IMPLICIT.value == "implicit"

    def test_has_explicit_value(self):
        """Type has explicit value (user feedback via MCP tool)."""
        assert FeedbackType.EXPLICIT.value == "explicit"


class TestFeedbackEvent:
    """Tests for FeedbackEvent dataclass."""

    def test_create_minimal_event(self):
        """Can create event with minimal required fields."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        assert event.query_id == "q123"
        assert event.user_id == "u456"
        assert event.org_id == "org789"
        assert event.tool_name == "search_code_hybrid"
        assert event.outcome == FeedbackOutcome.ACCEPTED

    def test_event_has_auto_generated_id(self):
        """Event has auto-generated UUID if not provided."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        assert event.event_id is not None
        # Verify it's a valid UUID
        UUID(event.event_id)

    def test_event_has_auto_timestamp(self):
        """Event has auto-generated timestamp if not provided."""
        before = datetime.now(timezone.utc)
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_event_with_session_id(self):
        """Event can include session_id."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            session_id="sess_abc",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        assert event.session_id == "sess_abc"

    def test_event_with_decision_time_ms(self):
        """Event can include decision time in milliseconds."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            decision_time_ms=1500,
        )
        assert event.decision_time_ms == 1500

    def test_event_with_rrf_weights(self):
        """Event can include RRF weights used for the query."""
        weights = {"vector": 0.40, "lexical": 0.35, "graph": 0.25}
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            rrf_weights=weights,
        )
        assert event.rrf_weights == weights

    def test_event_with_reranker_used(self):
        """Event can indicate if reranker was used."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            reranker_used=True,
        )
        assert event.reranker_used is True

    def test_event_with_feedback_type(self):
        """Event can specify feedback type (implicit/explicit)."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            feedback_type=FeedbackType.EXPLICIT,
        )
        assert event.feedback_type == FeedbackType.EXPLICIT

    def test_event_defaults_to_implicit_type(self):
        """Event defaults to implicit feedback type."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        assert event.feedback_type == FeedbackType.IMPLICIT

    def test_event_with_result_index(self):
        """Event can include which result index was interacted with."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            result_index=2,
        )
        assert event.result_index == 2

    def test_event_with_result_id(self):
        """Event can include the specific result ID."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            result_id="doc_xyz",
        )
        assert event.result_id == "doc_xyz"

    def test_event_with_experiment_id(self):
        """Event can include A/B test experiment ID."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            experiment_id="exp_001",
        )
        assert event.experiment_id == "exp_001"

    def test_event_with_variant_id(self):
        """Event can include A/B test variant ID."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            variant_id="variant_a",
        )
        assert event.variant_id == "variant_a"

    def test_event_with_metadata(self):
        """Event can include arbitrary metadata."""
        metadata = {"source": "ide", "action": "copy"}
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            metadata=metadata,
        )
        assert event.metadata == metadata

    def test_event_to_dict(self):
        """Event can be serialized to dictionary."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            decision_time_ms=1500,
        )
        d = event.to_dict()
        assert d["query_id"] == "q123"
        assert d["user_id"] == "u456"
        assert d["org_id"] == "org789"
        assert d["tool_name"] == "search_code_hybrid"
        assert d["outcome"] == "accepted"
        assert d["decision_time_ms"] == 1500

    def test_event_from_dict(self):
        """Event can be deserialized from dictionary."""
        data = {
            "event_id": str(uuid4()),
            "query_id": "q123",
            "user_id": "u456",
            "org_id": "org789",
            "tool_name": "search_code_hybrid",
            "outcome": "accepted",
            "feedback_type": "explicit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        event = FeedbackEvent.from_dict(data)
        assert event.query_id == "q123"
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.feedback_type == FeedbackType.EXPLICIT

    def test_event_immutable_by_default(self):
        """Event should be frozen (immutable)."""
        event = FeedbackEvent(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
        )
        with pytest.raises(AttributeError):
            event.query_id = "changed"


class TestImplicitFeedbackEvents:
    """Tests for implicit feedback event creation helpers."""

    def test_create_click_event(self):
        """Can create click event (result was clicked)."""
        event = FeedbackEvent.create_click(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            result_index=0,
            result_id="doc_xyz",
        )
        assert event.feedback_type == FeedbackType.IMPLICIT
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.result_index == 0
        assert event.result_id == "doc_xyz"
        assert event.metadata.get("action") == "click"

    def test_create_copy_event(self):
        """Can create copy event (result was copied)."""
        event = FeedbackEvent.create_copy(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            result_index=1,
            result_id="doc_abc",
        )
        assert event.feedback_type == FeedbackType.IMPLICIT
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.metadata.get("action") == "copy"

    def test_create_use_event(self):
        """Can create use event (result was used in code)."""
        event = FeedbackEvent.create_use(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            result_index=0,
            result_id="doc_xyz",
        )
        assert event.feedback_type == FeedbackType.IMPLICIT
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.metadata.get("action") == "use"

    def test_create_skip_event(self):
        """Can create skip event (result was skipped/scrolled past)."""
        event = FeedbackEvent.create_skip(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            result_index=0,
            result_id="doc_xyz",
        )
        assert event.feedback_type == FeedbackType.IMPLICIT
        assert event.outcome == FeedbackOutcome.IGNORED
        assert event.metadata.get("action") == "skip"

    def test_create_dwell_event(self):
        """Can create dwell event (user spent time viewing result)."""
        event = FeedbackEvent.create_dwell(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            result_index=0,
            result_id="doc_xyz",
            dwell_time_ms=5000,
        )
        assert event.feedback_type == FeedbackType.IMPLICIT
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.metadata.get("action") == "dwell"
        assert event.metadata.get("dwell_time_ms") == 5000


class TestExplicitFeedbackEvents:
    """Tests for explicit feedback event creation helpers."""

    def test_create_explicit_accept(self):
        """Can create explicit accept feedback."""
        event = FeedbackEvent.create_explicit(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.ACCEPTED,
            result_index=0,
            result_id="doc_xyz",
            comment="Great result!",
        )
        assert event.feedback_type == FeedbackType.EXPLICIT
        assert event.outcome == FeedbackOutcome.ACCEPTED
        assert event.metadata.get("comment") == "Great result!"

    def test_create_explicit_reject(self):
        """Can create explicit reject feedback."""
        event = FeedbackEvent.create_explicit(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.REJECTED,
            result_index=0,
            result_id="doc_xyz",
            comment="Not relevant",
        )
        assert event.feedback_type == FeedbackType.EXPLICIT
        assert event.outcome == FeedbackOutcome.REJECTED

    def test_create_explicit_modified(self):
        """Can create explicit modified feedback (user used but changed it)."""
        event = FeedbackEvent.create_explicit(
            query_id="q123",
            user_id="u456",
            org_id="org789",
            tool_name="search_code_hybrid",
            outcome=FeedbackOutcome.MODIFIED,
            result_index=0,
            result_id="doc_xyz",
        )
        assert event.feedback_type == FeedbackType.EXPLICIT
        assert event.outcome == FeedbackOutcome.MODIFIED
