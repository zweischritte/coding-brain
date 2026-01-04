"""
Tests for TaskType and TaskState enums.

These tests verify the core data structures for task-based routing:
- TaskType: Classification of user requests into task categories
- TaskState: State machine states for tool availability
- Tool availability mappings per state
"""
import pytest
from app.routing.task_types import (
    TaskType,
    TaskState,
    TOOL_AVAILABILITY,
    STATE_TRANSITIONS,
    CORE_TOOLS,
)


class TestTaskTypeEnum:
    """Tests for TaskType enum values and completeness."""

    def test_has_code_trace(self):
        """CODE_TRACE exists for bug finding, execution tracing tasks."""
        assert TaskType.CODE_TRACE is not None
        assert TaskType.CODE_TRACE.value

    def test_has_code_understand(self):
        """CODE_UNDERSTAND exists for architecture analysis tasks."""
        assert TaskType.CODE_UNDERSTAND is not None
        assert TaskType.CODE_UNDERSTAND.value

    def test_has_memory_query(self):
        """MEMORY_QUERY exists for retrieving past decisions."""
        assert TaskType.MEMORY_QUERY is not None
        assert TaskType.MEMORY_QUERY.value

    def test_has_memory_write(self):
        """MEMORY_WRITE exists for documenting decisions."""
        assert TaskType.MEMORY_WRITE is not None
        assert TaskType.MEMORY_WRITE.value

    def test_has_pr_review(self):
        """PR_REVIEW exists for pull request analysis."""
        assert TaskType.PR_REVIEW is not None
        assert TaskType.PR_REVIEW.value

    def test_has_test_write(self):
        """TEST_WRITE exists for test generation tasks."""
        assert TaskType.TEST_WRITE is not None
        assert TaskType.TEST_WRITE.value

    def test_has_general(self):
        """GENERAL exists as fallback for unclassified requests."""
        assert TaskType.GENERAL is not None
        assert TaskType.GENERAL.value

    def test_all_task_types_count(self):
        """Should have exactly 7 task types as per PRD."""
        assert len(TaskType) == 7


class TestTaskStateEnum:
    """Tests for TaskState enum values and transitions."""

    def test_has_initial(self):
        """INITIAL state exists for before intent classification."""
        assert TaskState.INITIAL is not None
        assert TaskState.INITIAL.value

    def test_has_code_analysis(self):
        """CODE_ANALYSIS state for code-related tasks."""
        assert TaskState.CODE_ANALYSIS is not None
        assert TaskState.CODE_ANALYSIS.value

    def test_has_memory_ops(self):
        """MEMORY_OPS state for memory-related tasks."""
        assert TaskState.MEMORY_OPS is not None
        assert TaskState.MEMORY_OPS.value

    def test_has_pr_review(self):
        """PR_REVIEW state for pull request analysis."""
        assert TaskState.PR_REVIEW is not None
        assert TaskState.PR_REVIEW.value

    def test_has_testing(self):
        """TESTING state for test generation."""
        assert TaskState.TESTING is not None
        assert TaskState.TESTING.value

    def test_all_task_states_count(self):
        """Should have exactly 5 task states as per PRD."""
        assert len(TaskState) == 5


class TestCoreTools:
    """Tests for the core tools that are always available."""

    def test_core_tools_defined(self):
        """Core tools list should be defined."""
        assert CORE_TOOLS is not None
        assert isinstance(CORE_TOOLS, (list, tuple, set))

    def test_core_tools_has_read(self):
        """Read tool should always be available."""
        assert "Read" in CORE_TOOLS

    def test_core_tools_has_grep(self):
        """Grep tool should always be available."""
        assert "Grep" in CORE_TOOLS

    def test_core_tools_has_glob(self):
        """Glob tool should always be available."""
        assert "Glob" in CORE_TOOLS

    def test_core_tools_has_bash(self):
        """Bash tool should always be available."""
        assert "Bash" in CORE_TOOLS


class TestToolAvailability:
    """Tests for tool availability per TaskState."""

    def test_tool_availability_defined_for_all_states(self):
        """Every TaskState should have tool availability defined."""
        for state in TaskState:
            assert state in TOOL_AVAILABILITY, f"Missing TOOL_AVAILABILITY for {state}"

    def test_initial_state_has_core_tools(self):
        """INITIAL state should have core tools available."""
        available = TOOL_AVAILABILITY[TaskState.INITIAL]["available"]
        for tool in CORE_TOOLS:
            assert tool in available, f"Core tool {tool} missing from INITIAL state"

    def test_code_analysis_has_code_tools(self):
        """CODE_ANALYSIS state should have code-specific tools."""
        available = TOOL_AVAILABILITY[TaskState.CODE_ANALYSIS]["available"]
        assert "find_callers" in available
        assert "find_callees" in available
        assert "explain_code" in available
        assert "search_code_hybrid" in available

    def test_code_analysis_hides_memory_tools(self):
        """CODE_ANALYSIS should hide memory tools to reduce confusion."""
        hidden = TOOL_AVAILABILITY[TaskState.CODE_ANALYSIS]["hidden"]
        # Hidden patterns should include memory tools
        assert any("memory" in h.lower() or h == "search_memory" for h in hidden)

    def test_memory_ops_has_memory_tools(self):
        """MEMORY_OPS state should have memory-specific tools."""
        available = TOOL_AVAILABILITY[TaskState.MEMORY_OPS]["available"]
        assert "search_memory" in available
        assert "add_memories" in available
        assert "list_memories" in available
        assert "get_memory" in available

    def test_memory_ops_hides_code_tools(self):
        """MEMORY_OPS should hide code-specific tools."""
        hidden = TOOL_AVAILABILITY[TaskState.MEMORY_OPS]["hidden"]
        # Hidden patterns should include code tools
        assert any("code" in h.lower() or h == "find_callers" for h in hidden)

    def test_pr_review_has_analysis_tools(self):
        """PR_REVIEW state should have PR analysis tools."""
        available = TOOL_AVAILABILITY[TaskState.PR_REVIEW]["available"]
        assert "pr_analysis" in available
        assert "impact_analysis" in available
        assert "Read" in available
        assert "Grep" in available

    def test_testing_has_test_tools(self):
        """TESTING state should have test generation tools."""
        available = TOOL_AVAILABILITY[TaskState.TESTING]["available"]
        assert "test_generation" in available
        assert "explain_code" in available
        assert "Read" in available

    def test_each_state_has_available_key(self):
        """Each state config should have 'available' key."""
        for state in TaskState:
            assert "available" in TOOL_AVAILABILITY[state], f"{state} missing 'available'"

    def test_each_state_has_hidden_key(self):
        """Each state config should have 'hidden' key."""
        for state in TaskState:
            assert "hidden" in TOOL_AVAILABILITY[state], f"{state} missing 'hidden'"


class TestStateTransitions:
    """Tests for valid state transitions."""

    def test_transitions_defined_for_initial(self):
        """INITIAL state should have transitions to other states."""
        assert TaskState.INITIAL in STATE_TRANSITIONS
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert len(transitions) > 0

    def test_code_trace_transitions_to_code_analysis(self):
        """CODE_TRACE intent should transition to CODE_ANALYSIS state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.CODE_TRACE in transitions
        assert transitions[TaskType.CODE_TRACE] == TaskState.CODE_ANALYSIS

    def test_code_understand_transitions_to_code_analysis(self):
        """CODE_UNDERSTAND intent should transition to CODE_ANALYSIS state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.CODE_UNDERSTAND in transitions
        assert transitions[TaskType.CODE_UNDERSTAND] == TaskState.CODE_ANALYSIS

    def test_memory_query_transitions_to_memory_ops(self):
        """MEMORY_QUERY intent should transition to MEMORY_OPS state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.MEMORY_QUERY in transitions
        assert transitions[TaskType.MEMORY_QUERY] == TaskState.MEMORY_OPS

    def test_memory_write_transitions_to_memory_ops(self):
        """MEMORY_WRITE intent should transition to MEMORY_OPS state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.MEMORY_WRITE in transitions
        assert transitions[TaskType.MEMORY_WRITE] == TaskState.MEMORY_OPS

    def test_pr_review_transitions_to_pr_review(self):
        """PR_REVIEW intent should transition to PR_REVIEW state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.PR_REVIEW in transitions
        assert transitions[TaskType.PR_REVIEW] == TaskState.PR_REVIEW

    def test_test_write_transitions_to_testing(self):
        """TEST_WRITE intent should transition to TESTING state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.TEST_WRITE in transitions
        assert transitions[TaskType.TEST_WRITE] == TaskState.TESTING

    def test_general_stays_in_initial(self):
        """GENERAL intent should stay in INITIAL state."""
        transitions = STATE_TRANSITIONS[TaskState.INITIAL]
        assert TaskType.GENERAL in transitions
        assert transitions[TaskType.GENERAL] == TaskState.INITIAL

    def test_code_analysis_can_transition_to_memory(self):
        """CODE_ANALYSIS should allow transition to MEMORY_OPS (document findings)."""
        if TaskState.CODE_ANALYSIS in STATE_TRANSITIONS:
            transitions = STATE_TRANSITIONS[TaskState.CODE_ANALYSIS]
            if TaskType.MEMORY_WRITE in transitions:
                assert transitions[TaskType.MEMORY_WRITE] == TaskState.MEMORY_OPS
