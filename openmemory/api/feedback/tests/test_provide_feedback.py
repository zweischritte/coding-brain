"""Tests for provide_feedback MCP tool.

TDD tests for the explicit feedback MCP tool (FR-002) following v9 plan:
- Explicit user feedback via MCP
- Input validation and schema
- Feedback storage integration
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from openmemory.api.feedback.events import FeedbackEvent, FeedbackOutcome, FeedbackType
from openmemory.api.feedback.provide_feedback import (
    ProvideFeedbackConfig,
    ProvideFeedbackError,
    ProvideFeedbackInput,
    ProvideFeedbackOutput,
    ProvideFeedbackTool,
    create_provide_feedback_tool,
)
from openmemory.api.feedback.store import InMemoryFeedbackStore


class TestProvideFeedbackInput:
    """Tests for ProvideFeedbackInput dataclass."""

    def test_create_minimal_input(self):
        """Can create input with minimal required fields."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
        )
        assert input_data.query_id == "q123"
        assert input_data.outcome == "accepted"

    def test_input_with_result_index(self):
        """Input can include result index."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            result_index=0,
        )
        assert input_data.result_index == 0

    def test_input_with_result_id(self):
        """Input can include result ID."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            result_id="doc_xyz",
        )
        assert input_data.result_id == "doc_xyz"

    def test_input_with_comment(self):
        """Input can include user comment."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="rejected",
            comment="Not relevant to my query",
        )
        assert input_data.comment == "Not relevant to my query"

    def test_input_with_tool_name(self):
        """Input can include tool name."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            tool_name="search_code_hybrid",
        )
        assert input_data.tool_name == "search_code_hybrid"

    def test_input_validates_outcome_values(self):
        """Input validates outcome is one of allowed values."""
        valid_outcomes = ["accepted", "modified", "rejected", "ignored"]
        for outcome in valid_outcomes:
            input_data = ProvideFeedbackInput(query_id="q123", outcome=outcome)
            assert input_data.outcome == outcome

    def test_input_rejects_invalid_outcome(self):
        """Input raises error for invalid outcome."""
        with pytest.raises(ValueError, match="Invalid outcome"):
            ProvideFeedbackInput(query_id="q123", outcome="invalid")

    def test_input_to_dict(self):
        """Input can be serialized to dictionary."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            result_index=0,
            comment="Great result!",
        )
        d = input_data.to_dict()
        assert d["query_id"] == "q123"
        assert d["outcome"] == "accepted"
        assert d["result_index"] == 0
        assert d["comment"] == "Great result!"

    def test_input_from_dict(self):
        """Input can be deserialized from dictionary."""
        data = {
            "query_id": "q123",
            "outcome": "rejected",
            "comment": "Not helpful",
        }
        input_data = ProvideFeedbackInput.from_dict(data)
        assert input_data.query_id == "q123"
        assert input_data.outcome == "rejected"
        assert input_data.comment == "Not helpful"


class TestProvideFeedbackOutput:
    """Tests for ProvideFeedbackOutput dataclass."""

    def test_create_success_output(self):
        """Can create success output."""
        output = ProvideFeedbackOutput(
            success=True,
            event_id="evt_123",
            message="Feedback recorded successfully",
        )
        assert output.success is True
        assert output.event_id == "evt_123"
        assert output.message == "Feedback recorded successfully"

    def test_create_error_output(self):
        """Can create error output."""
        output = ProvideFeedbackOutput(
            success=False,
            event_id=None,
            message="Failed to record feedback",
            error_code="STORE_ERROR",
        )
        assert output.success is False
        assert output.event_id is None
        assert output.error_code == "STORE_ERROR"

    def test_output_to_dict(self):
        """Output can be serialized to dictionary."""
        output = ProvideFeedbackOutput(
            success=True,
            event_id="evt_123",
            message="OK",
        )
        d = output.to_dict()
        assert d["success"] is True
        assert d["event_id"] == "evt_123"


class TestProvideFeedbackConfig:
    """Tests for ProvideFeedbackConfig."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = ProvideFeedbackConfig()
        assert config.default_tool_name == "unknown"

    def test_custom_config(self):
        """Can customize config values."""
        config = ProvideFeedbackConfig(default_tool_name="search_code_hybrid")
        assert config.default_tool_name == "search_code_hybrid"


class TestProvideFeedbackTool:
    """Tests for ProvideFeedbackTool class."""

    @pytest.fixture
    def store(self) -> InMemoryFeedbackStore:
        """Create a feedback store."""
        return InMemoryFeedbackStore()

    @pytest.fixture
    def tool(self, store: InMemoryFeedbackStore) -> ProvideFeedbackTool:
        """Create a provide_feedback tool."""
        return ProvideFeedbackTool(store=store)

    def test_tool_has_name(self, tool: ProvideFeedbackTool):
        """Tool has correct name."""
        assert tool.name == "provide_feedback"

    def test_tool_has_description(self, tool: ProvideFeedbackTool):
        """Tool has description."""
        assert tool.description is not None
        assert len(tool.description) > 0

    def test_tool_has_schema(self, tool: ProvideFeedbackTool):
        """Tool has JSON schema for input."""
        schema = tool.get_input_schema()
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query_id" in schema["properties"]
        assert "outcome" in schema["properties"]

    def test_provide_feedback_accepted(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Can provide accepted feedback."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            result_index=0,
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        assert output.event_id is not None

        # Verify event was stored
        events = store.query_by_user("u456", "org789")
        assert len(events) == 1
        assert events[0].outcome == FeedbackOutcome.ACCEPTED
        assert events[0].feedback_type == FeedbackType.EXPLICIT

    def test_provide_feedback_rejected(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Can provide rejected feedback."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="rejected",
            comment="Not relevant",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert events[0].outcome == FeedbackOutcome.REJECTED
        assert events[0].metadata.get("comment") == "Not relevant"

    def test_provide_feedback_modified(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Can provide modified feedback."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="modified",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert events[0].outcome == FeedbackOutcome.MODIFIED

    def test_provide_feedback_with_session(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Feedback includes session ID when provided."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
            session_id="sess_abc",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert events[0].session_id == "sess_abc"

    def test_provide_feedback_with_tool_name(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Feedback includes tool name."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            tool_name="search_code_hybrid",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert events[0].tool_name == "search_code_hybrid"

    def test_provide_feedback_with_experiment(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Feedback can include experiment data."""
        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
            experiment_id="exp_001",
            variant_id="treatment",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert events[0].experiment_id == "exp_001"
        assert events[0].variant_id == "treatment"

    def test_provide_feedback_from_raw_dict(
        self, tool: ProvideFeedbackTool, store: InMemoryFeedbackStore
    ):
        """Can provide feedback from raw dictionary input."""
        raw_input = {
            "query_id": "q123",
            "outcome": "accepted",
            "result_index": 0,
        }

        output = tool.provide_feedback_raw(
            input_dict=raw_input,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is True
        events = store.query_by_user("u456", "org789")
        assert len(events) == 1


class TestProvideFeedbackToolErrors:
    """Tests for error handling in ProvideFeedbackTool."""

    @pytest.fixture
    def store(self) -> InMemoryFeedbackStore:
        return InMemoryFeedbackStore()

    @pytest.fixture
    def tool(self, store: InMemoryFeedbackStore) -> ProvideFeedbackTool:
        return ProvideFeedbackTool(store=store)

    def test_missing_query_id(self, tool: ProvideFeedbackTool):
        """Raises error for missing query_id."""
        with pytest.raises(ValueError, match="query_id"):
            ProvideFeedbackInput(query_id="", outcome="accepted")

    def test_missing_outcome(self, tool: ProvideFeedbackTool):
        """Raises error for missing outcome."""
        with pytest.raises(ValueError, match="outcome"):
            ProvideFeedbackInput(query_id="q123", outcome="")

    def test_store_error_handling(self, tool: ProvideFeedbackTool):
        """Handles store errors gracefully."""
        # Mock store to raise an error
        tool._store.append = Mock(side_effect=Exception("Database error"))

        input_data = ProvideFeedbackInput(
            query_id="q123",
            outcome="accepted",
        )

        output = tool.provide_feedback(
            input_data=input_data,
            user_id="u456",
            org_id="org789",
        )

        assert output.success is False
        assert "error" in output.message.lower() or output.error_code is not None


class TestCreateProvideFeedbackTool:
    """Tests for factory function."""

    def test_create_with_store(self):
        """Can create tool with custom store."""
        store = InMemoryFeedbackStore()
        tool = create_provide_feedback_tool(store=store)
        assert tool._store is store

    def test_create_with_config(self):
        """Can create tool with custom config."""
        store = InMemoryFeedbackStore()
        config = ProvideFeedbackConfig(default_tool_name="custom_tool")
        tool = create_provide_feedback_tool(store=store, config=config)
        assert tool._config.default_tool_name == "custom_tool"


class TestProvideFeedbackMCPSchema:
    """Tests for MCP schema compliance."""

    @pytest.fixture
    def tool(self) -> ProvideFeedbackTool:
        return ProvideFeedbackTool(store=InMemoryFeedbackStore())

    def test_schema_has_required_properties(self, tool: ProvideFeedbackTool):
        """Schema specifies required properties."""
        schema = tool.get_input_schema()
        assert "required" in schema
        assert "query_id" in schema["required"]
        assert "outcome" in schema["required"]

    def test_schema_outcome_enum(self, tool: ProvideFeedbackTool):
        """Schema specifies valid outcome values."""
        schema = tool.get_input_schema()
        outcome_prop = schema["properties"]["outcome"]
        assert "enum" in outcome_prop
        assert "accepted" in outcome_prop["enum"]
        assert "modified" in outcome_prop["enum"]
        assert "rejected" in outcome_prop["enum"]
        assert "ignored" in outcome_prop["enum"]

    def test_schema_has_descriptions(self, tool: ProvideFeedbackTool):
        """Schema properties have descriptions."""
        schema = tool.get_input_schema()
        for prop_name, prop_schema in schema["properties"].items():
            assert "description" in prop_schema, f"{prop_name} missing description"

    def test_output_schema(self, tool: ProvideFeedbackTool):
        """Tool has output schema."""
        schema = tool.get_output_schema()
        assert "type" in schema
        assert "properties" in schema
        assert "success" in schema["properties"]
