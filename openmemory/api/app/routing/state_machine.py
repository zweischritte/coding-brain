"""
Task State Machine for Tool Availability.

Manages the current task state and determines which tools are available.
State transitions are triggered by intent classification.

Key design decisions:
1. No mid-iteration tool changes (avoids KV-cache invalidation)
2. Session-scoped state (one state per conversation)
3. Explicit transitions only (predictable behavior)

Usage:
    machine = TaskStateMachine()
    assert machine.state == TaskState.INITIAL

    # Transition based on intent
    machine.transition(TaskType.CODE_TRACE)
    assert machine.state == TaskState.CODE_ANALYSIS
    assert "find_callers" in machine.available_tools
    assert "search_memory" not in machine.available_tools
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .task_types import (
    TaskType,
    TaskState,
    TOOL_AVAILABILITY,
    STATE_TRANSITIONS,
    CORE_TOOLS,
)

logger = logging.getLogger(__name__)


@dataclass
class StateTransitionEvent:
    """Record of a state transition."""

    from_state: TaskState
    to_state: TaskState
    intent: TaskType


class TaskStateMachine:
    """
    State machine for task-based tool routing.

    Manages transitions between states based on intent classification
    and determines tool availability for the current state.

    Attributes:
        state: Current TaskState
        previous_state: Previous TaskState (before last transition)
        history: List of StateTransitionEvent records
        available_tools: Set of currently available tool names

    Example:
        >>> machine = TaskStateMachine()
        >>> machine.transition(TaskType.CODE_TRACE)
        TaskState.INITIAL  # Returns previous state
        >>> machine.state
        TaskState.CODE_ANALYSIS
        >>> "find_callers" in machine.available_tools
        True
    """

    def __init__(self, initial_state: TaskState = TaskState.INITIAL):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state (default: INITIAL)
        """
        self._state = initial_state
        self._previous_state: Optional[TaskState] = None
        self._history: List[StateTransitionEvent] = []
        self._available_tools: Optional[Set[str]] = None  # Lazy evaluation

    @property
    def state(self) -> TaskState:
        """Current state."""
        return self._state

    @property
    def previous_state(self) -> Optional[TaskState]:
        """Previous state (before last transition)."""
        return self._previous_state

    @property
    def history(self) -> List[StateTransitionEvent]:
        """List of all state transitions."""
        return self._history.copy()

    @property
    def available_tools(self) -> Set[str]:
        """
        Set of currently available tool names.

        Lazily computed based on current state.
        Includes core tools plus state-specific tools.
        """
        if self._available_tools is None:
            self._available_tools = self._compute_available_tools()
        return self._available_tools

    def transition(self, intent: TaskType) -> TaskState:
        """
        Attempt to transition to a new state based on intent.

        If a valid transition exists for the current state and intent,
        the state is updated. Otherwise, the state remains unchanged.

        Args:
            intent: The classified TaskType

        Returns:
            The previous state (before transition)

        Raises:
            TypeError: If intent is not a TaskType
        """
        if not isinstance(intent, TaskType):
            raise TypeError(f"Expected TaskType, got {type(intent)}")

        old_state = self._state

        # Look up transition
        transitions = STATE_TRANSITIONS.get(self._state, {})
        new_state = transitions.get(intent)

        if new_state is not None:
            self._previous_state = old_state
            self._state = new_state
            self._available_tools = None  # Invalidate cache

            # Record transition
            event = StateTransitionEvent(
                from_state=old_state,
                to_state=new_state,
                intent=intent,
            )
            self._history.append(event)

            logger.debug(
                f"State transition: {old_state.name} -> {new_state.name} "
                f"(intent: {intent.name})"
            )
        else:
            logger.debug(
                f"No transition from {self._state.name} for intent {intent.name}"
            )

        return old_state

    def reset(self) -> None:
        """
        Reset to INITIAL state.

        Clears history and resets tool availability.
        """
        self._previous_state = self._state
        self._state = TaskState.INITIAL
        self._available_tools = None
        # Note: We don't clear history to allow debugging

        logger.debug("State machine reset to INITIAL")

    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a specific tool is available in current state.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is available, False otherwise
        """
        return tool_name in self.available_tools

    def _compute_available_tools(self) -> Set[str]:
        """
        Compute the set of available tools for current state.

        Starts with core tools, adds state-specific tools,
        then removes hidden tools (including wildcard patterns).

        Returns:
            Set of available tool names
        """
        availability = TOOL_AVAILABILITY.get(self._state)
        if availability is None:
            logger.warning(f"No tool availability defined for {self._state}")
            return CORE_TOOLS.copy()

        # Start with explicitly available tools
        available = set(availability.get("available", []))

        # Always include core tools
        available.update(CORE_TOOLS)

        # Get hidden patterns
        hidden_patterns = availability.get("hidden", [])

        # Filter out hidden tools
        result = set()
        for tool in available:
            if not self._matches_hidden_pattern(tool, hidden_patterns):
                result.add(tool)

        return result

    def _matches_hidden_pattern(
        self,
        tool_name: str,
        hidden_patterns: List[str]
    ) -> bool:
        """
        Check if a tool matches any hidden pattern.

        Supports:
        - Exact match: "search_memory"
        - Wildcard prefix: "delete_*"
        - All tools: "*"

        Args:
            tool_name: Name of tool to check
            hidden_patterns: List of patterns to match against

        Returns:
            True if tool matches any hidden pattern
        """
        for pattern in hidden_patterns:
            if pattern == "*":
                # Wildcard matches all (except core tools handled separately)
                if tool_name not in CORE_TOOLS:
                    return True
            elif pattern.endswith("*"):
                # Prefix match
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    return True
            elif fnmatch.fnmatch(tool_name, pattern):
                # Full fnmatch for complex patterns
                return True
            elif tool_name == pattern:
                # Exact match
                return True

        return False

    def get_filtered_tools(self, all_tools: List[str]) -> List[str]:
        """
        Filter a list of tools based on current state availability.

        Useful for filtering the full tool list before exposing to LLM.

        Args:
            all_tools: List of all possible tool names

        Returns:
            List of tools that are available in current state
        """
        available = self.available_tools
        return [tool for tool in all_tools if tool in available]

    def __repr__(self) -> str:
        return (
            f"TaskStateMachine(state={self._state.name}, "
            f"tools={len(self.available_tools)})"
        )
