"""provide_feedback MCP tool for explicit user feedback.

Implements FR-002: Feedback Integration - explicit feedback via MCP tool.
Users can provide feedback on retrieval results through this tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from openmemory.api.feedback.events import FeedbackEvent, FeedbackOutcome, FeedbackType
from openmemory.api.feedback.store import FeedbackStore

logger = logging.getLogger(__name__)

# Valid outcome values
VALID_OUTCOMES = {"accepted", "modified", "rejected", "ignored"}


class ProvideFeedbackError(Exception):
    """Error raised by provide_feedback tool."""

    pass


@dataclass
class ProvideFeedbackInput:
    """Input for provide_feedback tool.

    Attributes:
        query_id: ID of the query this feedback is for
        outcome: User's assessment (accepted, modified, rejected, ignored)
        result_index: Optional index of the result being rated (0-based)
        result_id: Optional ID of the specific result
        comment: Optional user comment explaining the feedback
        tool_name: Optional name of the tool that produced the result
        experiment_id: Optional A/B test experiment ID
        variant_id: Optional A/B test variant ID
    """

    query_id: str
    outcome: str
    result_index: Optional[int] = None
    result_id: Optional[str] = None
    comment: Optional[str] = None
    tool_name: Optional[str] = None
    experiment_id: Optional[str] = None
    variant_id: Optional[str] = None

    def __post_init__(self):
        """Validate input."""
        if not self.query_id:
            raise ValueError("query_id is required")
        if not self.outcome:
            raise ValueError("outcome is required")
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome: {self.outcome}. "
                f"Must be one of: {', '.join(VALID_OUTCOMES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "outcome": self.outcome,
            "result_index": self.result_index,
            "result_id": self.result_id,
            "comment": self.comment,
            "tool_name": self.tool_name,
            "experiment_id": self.experiment_id,
            "variant_id": self.variant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvideFeedbackInput:
        """Deserialize from dictionary."""
        return cls(
            query_id=data.get("query_id", ""),
            outcome=data.get("outcome", ""),
            result_index=data.get("result_index"),
            result_id=data.get("result_id"),
            comment=data.get("comment"),
            tool_name=data.get("tool_name"),
            experiment_id=data.get("experiment_id"),
            variant_id=data.get("variant_id"),
        )


@dataclass
class ProvideFeedbackOutput:
    """Output from provide_feedback tool.

    Attributes:
        success: Whether feedback was recorded successfully
        event_id: ID of the created feedback event
        message: Human-readable status message
        error_code: Error code if failed (VALIDATION_ERROR, STORE_ERROR, etc.)
    """

    success: bool
    event_id: Optional[str]
    message: str
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "event_id": self.event_id,
            "message": self.message,
            "error_code": self.error_code,
        }


@dataclass
class ProvideFeedbackConfig:
    """Configuration for provide_feedback tool.

    Attributes:
        default_tool_name: Default tool name when not specified
    """

    default_tool_name: str = "unknown"


class ProvideFeedbackTool:
    """MCP tool for providing explicit feedback on retrieval results.

    This tool allows users to rate retrieval results and provide feedback
    that can be used to improve retrieval quality over time.

    Per v9 plan Section 11.4:
    - provide_feedback (FR-002)
    """

    name = "provide_feedback"
    description = (
        "Provide feedback on a retrieval result. "
        "Use this to indicate whether a result was helpful (accepted), "
        "partially helpful (modified), not helpful (rejected), or irrelevant (ignored)."
    )

    def __init__(
        self,
        store: FeedbackStore,
        config: Optional[ProvideFeedbackConfig] = None,
    ):
        """Initialize the tool.

        Args:
            store: Feedback store for persisting events
            config: Optional configuration
        """
        self._store = store
        self._config = config or ProvideFeedbackConfig()

    def get_input_schema(self) -> dict[str, Any]:
        """Get JSON schema for tool input."""
        return {
            "type": "object",
            "properties": {
                "query_id": {
                    "type": "string",
                    "description": "ID of the query this feedback is for",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["accepted", "modified", "rejected", "ignored"],
                    "description": (
                        "User's assessment: "
                        "accepted (result was used as-is), "
                        "modified (result was used but changed), "
                        "rejected (result was not useful), "
                        "ignored (result was skipped)"
                    ),
                },
                "result_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Index of the result being rated (0-based)",
                },
                "result_id": {
                    "type": "string",
                    "description": "ID of the specific result being rated",
                },
                "comment": {
                    "type": "string",
                    "description": "Optional user comment explaining the feedback",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool that produced the result",
                },
                "experiment_id": {
                    "type": "string",
                    "description": "A/B test experiment ID if applicable",
                },
                "variant_id": {
                    "type": "string",
                    "description": "A/B test variant ID if applicable",
                },
            },
            "required": ["query_id", "outcome"],
        }

    def get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for tool output."""
        return {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether feedback was recorded successfully",
                },
                "event_id": {
                    "type": "string",
                    "description": "ID of the created feedback event",
                },
                "message": {
                    "type": "string",
                    "description": "Human-readable status message",
                },
                "error_code": {
                    "type": "string",
                    "description": "Error code if failed",
                },
            },
            "required": ["success", "message"],
        }

    def provide_feedback(
        self,
        input_data: ProvideFeedbackInput,
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
    ) -> ProvideFeedbackOutput:
        """Provide feedback on a retrieval result.

        Args:
            input_data: The feedback input data
            user_id: User providing the feedback
            org_id: Organization the user belongs to
            session_id: Optional session ID

        Returns:
            ProvideFeedbackOutput with success status and event ID
        """
        try:
            # Map outcome string to enum
            outcome = FeedbackOutcome(input_data.outcome)

            # Build metadata
            metadata: dict[str, Any] = {}
            if input_data.comment:
                metadata["comment"] = input_data.comment

            # Create feedback event
            event = FeedbackEvent.create_explicit(
                query_id=input_data.query_id,
                user_id=user_id,
                org_id=org_id,
                tool_name=input_data.tool_name or self._config.default_tool_name,
                outcome=outcome,
                result_index=input_data.result_index,
                result_id=input_data.result_id,
                session_id=session_id,
                experiment_id=input_data.experiment_id,
                variant_id=input_data.variant_id,
                metadata=metadata,
            )

            # Store the event
            event_id = self._store.append(event)
            logger.info(
                f"Recorded feedback event {event_id} for query {input_data.query_id}"
            )

            return ProvideFeedbackOutput(
                success=True,
                event_id=event_id,
                message="Feedback recorded successfully",
            )

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return ProvideFeedbackOutput(
                success=False,
                event_id=None,
                message=f"Failed to record feedback: {str(e)}",
                error_code="STORE_ERROR",
            )

    def provide_feedback_raw(
        self,
        input_dict: dict[str, Any],
        user_id: str,
        org_id: str,
        session_id: Optional[str] = None,
    ) -> ProvideFeedbackOutput:
        """Provide feedback from raw dictionary input.

        Convenience method for MCP integration where input comes as dict.

        Args:
            input_dict: Raw dictionary input
            user_id: User providing the feedback
            org_id: Organization the user belongs to
            session_id: Optional session ID

        Returns:
            ProvideFeedbackOutput with success status and event ID
        """
        try:
            input_data = ProvideFeedbackInput.from_dict(input_dict)
            return self.provide_feedback(input_data, user_id, org_id, session_id)
        except ValueError as e:
            return ProvideFeedbackOutput(
                success=False,
                event_id=None,
                message=str(e),
                error_code="VALIDATION_ERROR",
            )


def create_provide_feedback_tool(
    store: FeedbackStore,
    config: Optional[ProvideFeedbackConfig] = None,
) -> ProvideFeedbackTool:
    """Factory function to create provide_feedback tool.

    Args:
        store: Feedback store for persisting events
        config: Optional configuration

    Returns:
        Configured ProvideFeedbackTool
    """
    return ProvideFeedbackTool(store=store, config=config)
