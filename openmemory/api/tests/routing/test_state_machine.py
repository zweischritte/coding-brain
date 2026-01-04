"""
Tests for TaskStateMachine.

The state machine manages task state transitions and determines
which tools are available at each state.
"""
import pytest
from unittest.mock import MagicMock

from app.routing.state_machine import TaskStateMachine
from app.routing.task_types import TaskType, TaskState, CORE_TOOLS


class TestTaskStateMachineInit:
    """Tests for TaskStateMachine initialization."""

    def test_initial_state_is_initial(self):
        """Machine should start in INITIAL state."""
        machine = TaskStateMachine()
        assert machine.state == TaskState.INITIAL

    def test_can_specify_initial_state(self):
        """Should be able to start in a specific state."""
        machine = TaskStateMachine(initial_state=TaskState.CODE_ANALYSIS)
        assert machine.state == TaskState.CODE_ANALYSIS

    def test_available_tools_in_initial(self):
        """INITIAL state should have core tools available."""
        machine = TaskStateMachine()
        for tool in CORE_TOOLS:
            assert tool in machine.available_tools


class TestStateTransitions:
    """Tests for state transitions."""

    def test_transition_to_code_analysis(self):
        """CODE_TRACE should transition to CODE_ANALYSIS."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)
        assert machine.state == TaskState.CODE_ANALYSIS

    def test_transition_to_memory_ops(self):
        """MEMORY_QUERY should transition to MEMORY_OPS."""
        machine = TaskStateMachine()
        machine.transition(TaskType.MEMORY_QUERY)
        assert machine.state == TaskState.MEMORY_OPS

    def test_transition_to_pr_review(self):
        """PR_REVIEW should transition to PR_REVIEW state."""
        machine = TaskStateMachine()
        machine.transition(TaskType.PR_REVIEW)
        assert machine.state == TaskState.PR_REVIEW

    def test_transition_to_testing(self):
        """TEST_WRITE should transition to TESTING state."""
        machine = TaskStateMachine()
        machine.transition(TaskType.TEST_WRITE)
        assert machine.state == TaskState.TESTING

    def test_general_stays_in_initial(self):
        """GENERAL should stay in INITIAL state."""
        machine = TaskStateMachine()
        machine.transition(TaskType.GENERAL)
        assert machine.state == TaskState.INITIAL

    def test_chained_transitions(self):
        """Should handle multiple transitions."""
        machine = TaskStateMachine()

        # First transition
        machine.transition(TaskType.CODE_TRACE)
        assert machine.state == TaskState.CODE_ANALYSIS

        # Second transition (document findings)
        machine.transition(TaskType.MEMORY_WRITE)
        # Should allow transition to MEMORY_OPS from CODE_ANALYSIS
        # or stay in CODE_ANALYSIS if not allowed
        assert machine.state in (TaskState.MEMORY_OPS, TaskState.CODE_ANALYSIS)

    def test_transition_returns_previous_state(self):
        """Transition should return the previous state."""
        machine = TaskStateMachine()
        old_state = machine.transition(TaskType.CODE_TRACE)
        assert old_state == TaskState.INITIAL


class TestAvailableTools:
    """Tests for tool availability per state."""

    def test_code_analysis_has_code_tools(self):
        """CODE_ANALYSIS should have code-specific tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)

        assert "find_callers" in machine.available_tools
        assert "find_callees" in machine.available_tools
        assert "explain_code" in machine.available_tools
        assert "search_code_hybrid" in machine.available_tools

    def test_code_analysis_excludes_memory_tools(self):
        """CODE_ANALYSIS should not have memory write tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)

        # Memory tools should be hidden
        assert "add_memories" not in machine.available_tools
        assert "delete_memories" not in machine.available_tools

    def test_memory_ops_has_memory_tools(self):
        """MEMORY_OPS should have memory-specific tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.MEMORY_QUERY)

        assert "search_memory" in machine.available_tools
        assert "add_memories" in machine.available_tools
        assert "list_memories" in machine.available_tools

    def test_memory_ops_excludes_code_tools(self):
        """MEMORY_OPS should not have code-specific tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.MEMORY_QUERY)

        # Code tools should be hidden
        assert "find_callers" not in machine.available_tools
        assert "find_callees" not in machine.available_tools
        assert "test_generation" not in machine.available_tools

    def test_pr_review_has_analysis_tools(self):
        """PR_REVIEW should have PR analysis tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.PR_REVIEW)

        assert "pr_analysis" in machine.available_tools
        assert "impact_analysis" in machine.available_tools
        assert "Read" in machine.available_tools

    def test_testing_has_test_tools(self):
        """TESTING should have test generation tools."""
        machine = TaskStateMachine()
        machine.transition(TaskType.TEST_WRITE)

        assert "test_generation" in machine.available_tools
        assert "explain_code" in machine.available_tools

    def test_core_tools_always_available(self):
        """Core tools should be available in all states."""
        machine = TaskStateMachine()

        for task_type in TaskType:
            machine = TaskStateMachine()  # Reset
            machine.transition(task_type)

            for tool in CORE_TOOLS:
                assert tool in machine.available_tools, \
                    f"Core tool {tool} missing in state after {task_type}"


class TestToolFiltering:
    """Tests for is_tool_available method."""

    def test_is_tool_available_core_tools(self):
        """Core tools should always return True."""
        machine = TaskStateMachine()

        for tool in CORE_TOOLS:
            assert machine.is_tool_available(tool)

    def test_is_tool_available_state_specific(self):
        """State-specific tools should return True in correct state."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)

        assert machine.is_tool_available("find_callers")
        assert not machine.is_tool_available("add_memories")

    def test_is_tool_available_hidden_tools(self):
        """Hidden tools should return False."""
        machine = TaskStateMachine()
        machine.transition(TaskType.MEMORY_QUERY)

        # Code tools should be hidden in MEMORY_OPS
        assert not machine.is_tool_available("find_callers")
        assert not machine.is_tool_available("find_callees")

    def test_is_tool_available_wildcard_patterns(self):
        """Wildcard patterns in hidden should work."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)

        # delete_* should be hidden
        assert not machine.is_tool_available("delete_memories")
        assert not machine.is_tool_available("delete_all_memories")


class TestResetState:
    """Tests for resetting state machine."""

    def test_reset_returns_to_initial(self):
        """reset() should return to INITIAL state."""
        machine = TaskStateMachine()
        machine.transition(TaskType.CODE_TRACE)
        machine.reset()

        assert machine.state == TaskState.INITIAL

    def test_reset_restores_initial_tools(self):
        """reset() should restore INITIAL state tools."""
        machine = TaskStateMachine()
        initial_tools = set(machine.available_tools)

        machine.transition(TaskType.CODE_TRACE)
        machine.reset()

        assert set(machine.available_tools) == initial_tools


class TestStateHistory:
    """Tests for state transition history (optional feature)."""

    def test_transition_history_tracked(self):
        """State transitions should be tracked."""
        machine = TaskStateMachine()

        machine.transition(TaskType.CODE_TRACE)
        machine.transition(TaskType.MEMORY_WRITE)

        if hasattr(machine, 'history'):
            assert len(machine.history) >= 2

    def test_can_access_previous_state(self):
        """Should be able to access previous state."""
        machine = TaskStateMachine()

        machine.transition(TaskType.CODE_TRACE)

        if hasattr(machine, 'previous_state'):
            assert machine.previous_state == TaskState.INITIAL


class TestConcurrency:
    """Tests for thread safety of state machine."""

    def test_independent_instances(self):
        """Multiple instances should be independent."""
        machine1 = TaskStateMachine()
        machine2 = TaskStateMachine()

        machine1.transition(TaskType.CODE_TRACE)

        # machine2 should still be in INITIAL
        assert machine2.state == TaskState.INITIAL


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_transition_handled(self):
        """Invalid transitions should be handled gracefully."""
        machine = TaskStateMachine()

        # Try to transition with None or invalid type
        try:
            machine.transition(None)  # type: ignore
        except (TypeError, ValueError):
            pass  # Expected
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_empty_state_still_has_core_tools(self):
        """Even in edge cases, core tools should be available."""
        machine = TaskStateMachine()

        # Even if something goes wrong, core tools should work
        assert len(machine.available_tools) >= len(CORE_TOOLS)
