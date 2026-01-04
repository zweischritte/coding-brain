"""
Task Types and State Definitions for Task-based Tool Routing.

This module defines the core data structures for task classification:
- TaskType: What kind of task is the user trying to do?
- TaskState: What tools should be available in this state?
- TOOL_AVAILABILITY: Mapping of states to available/hidden tools
- STATE_TRANSITIONS: Valid transitions between states

Design rationale from PRD-05-TASK-ROUTER.md:
> "Curating a minimal viable set of tools for the agent can lead to more
>  reliable maintenance and pruning of context over long interactions...
>  Too many tools or overlapping tools can distract agents from pursuing
>  efficient strategies." (Anthropic)
"""

from enum import Enum, auto
from typing import Dict, List, Set


class TaskType(Enum):
    """
    Classification of user requests into task categories.

    Each task type triggers a different tool availability set:
    - CODE_TRACE: Bug finding, execution tracing, call chain analysis
    - CODE_UNDERSTAND: Architecture analysis, codebase exploration
    - MEMORY_QUERY: Retrieving past decisions, conventions, context
    - MEMORY_WRITE: Documenting decisions, conventions, learnings
    - PR_REVIEW: Pull request analysis and code review
    - TEST_WRITE: Test generation and coverage
    - GENERAL: Fallback for unclear or general requests
    """

    CODE_TRACE = auto()
    """Bug finding, execution tracing, call chains, debugging errors."""

    CODE_UNDERSTAND = auto()
    """Architecture analysis, code exploration, learning codebase."""

    MEMORY_QUERY = auto()
    """Retrieving past decisions, conventions, context."""

    MEMORY_WRITE = auto()
    """Documenting decisions, conventions, learnings."""

    PR_REVIEW = auto()
    """Pull request analysis and code review."""

    TEST_WRITE = auto()
    """Test generation and coverage improvement."""

    GENERAL = auto()
    """Fallback for unclear or general requests."""


class TaskState(Enum):
    """
    State machine states for tool availability.

    Each state determines which tools are visible to the agent.
    Transitions are triggered by TaskType classification.
    """

    INITIAL = auto()
    """Before intent classification. Only core tools available."""

    CODE_ANALYSIS = auto()
    """Code-related tasks. Code tools visible, memory tools hidden."""

    MEMORY_OPS = auto()
    """Memory-related tasks. Memory tools visible, code tools hidden."""

    PR_REVIEW = auto()
    """PR review tasks. Analysis tools visible."""

    TESTING = auto()
    """Test generation tasks. Test tools visible."""


# Core tools that are always available regardless of state
CORE_TOOLS: Set[str] = {
    "Read",
    "Grep",
    "Glob",
    "Bash",
}


# Tool availability per state
# Each state has:
# - "available": List of tools that are exposed
# - "hidden": List of tools/patterns that are hidden
TOOL_AVAILABILITY: Dict[TaskState, Dict[str, List[str]]] = {
    TaskState.INITIAL: {
        "available": list(CORE_TOOLS),
        "hidden": ["*"],  # All others hidden until intent is clear
    },
    TaskState.CODE_ANALYSIS: {
        "available": [
            # Core tools
            "Read", "Grep", "Glob", "Bash",
            # Code analysis tools
            "find_callers",
            "find_callees",
            "explain_code",
            "search_code_hybrid",
            "impact_analysis",
        ],
        "hidden": [
            # Memory tools hidden to prevent distraction
            "search_memory",
            "add_memories",
            "update_memory",
            "list_memories",
            "get_memory",
            # Graph tools hidden (except code graph)
            "graph_related_memories",
            "graph_entity_network",
            "graph_similar_memories",
            "graph_aggregate",
            "graph_tag_cooccurrence",
            "graph_path_between_entities",
            "graph_subgraph",
            "graph_normalize_entities",
            "graph_normalize_entities_semantic",
            "graph_entity_relations",
            "graph_biography_timeline",
            "graph_related_tags",
            # Delete operations hidden
            "delete_*",
            # Test generation hidden (use TEST_WRITE for that)
            "test_generation",
            # ADR and PR hidden (use specific states)
            "adr_automation",
            "pr_analysis",
            # Indexing hidden
            "index_*",
        ],
    },
    TaskState.MEMORY_OPS: {
        "available": [
            # Core tools
            "Read", "Grep", "Glob", "Bash",
            # Memory tools
            "search_memory",
            "add_memories",
            "update_memory",
            "list_memories",
            "get_memory",
            # Graph tools for memory exploration
            "graph_related_memories",
            "graph_entity_network",
            "graph_similar_memories",
            "graph_aggregate",
            "graph_entity_relations",
        ],
        "hidden": [
            # Code tools hidden
            "find_callers",
            "find_callees",
            "search_code_hybrid",
            "explain_code",
            "impact_analysis",
            "test_generation",
            "pr_analysis",
            "adr_automation",
            # Delete hidden
            "delete_*",
            # Indexing hidden
            "index_*",
        ],
    },
    TaskState.PR_REVIEW: {
        "available": [
            # Core tools
            "Read", "Grep", "Glob", "Bash",
            # PR analysis tools
            "pr_analysis",
            "impact_analysis",
            "search_code_hybrid",
            # Memory for conventions lookup
            "search_memory",
        ],
        "hidden": [
            # Memory write hidden
            "add_memories",
            "update_memory",
            # Delete hidden
            "delete_*",
            # Test generation hidden
            "test_generation",
            # Indexing hidden
            "index_*",
            # ADR hidden
            "adr_automation",
        ],
    },
    TaskState.TESTING: {
        "available": [
            # Core tools
            "Read", "Grep", "Glob", "Bash",
            # Test tools
            "test_generation",
            "explain_code",
            "search_code_hybrid",
            "find_callers",
            "find_callees",
        ],
        "hidden": [
            # Memory tools hidden
            "search_memory",
            "add_memories",
            "update_memory",
            "list_memories",
            "get_memory",
            # Graph tools hidden
            "graph_*",
            # Delete hidden
            "delete_*",
            # PR review hidden
            "pr_analysis",
            # ADR hidden
            "adr_automation",
        ],
    },
}


# State transition rules
# Maps (current_state, intent) -> new_state
STATE_TRANSITIONS: Dict[TaskState, Dict[TaskType, TaskState]] = {
    TaskState.INITIAL: {
        TaskType.CODE_TRACE: TaskState.CODE_ANALYSIS,
        TaskType.CODE_UNDERSTAND: TaskState.CODE_ANALYSIS,
        TaskType.MEMORY_QUERY: TaskState.MEMORY_OPS,
        TaskType.MEMORY_WRITE: TaskState.MEMORY_OPS,
        TaskType.PR_REVIEW: TaskState.PR_REVIEW,
        TaskType.TEST_WRITE: TaskState.TESTING,
        TaskType.GENERAL: TaskState.INITIAL,  # Stay in initial
    },
    TaskState.CODE_ANALYSIS: {
        # Allow transition to MEMORY_OPS to document findings
        TaskType.MEMORY_WRITE: TaskState.MEMORY_OPS,
        # Allow staying in CODE_ANALYSIS for related tasks
        TaskType.CODE_TRACE: TaskState.CODE_ANALYSIS,
        TaskType.CODE_UNDERSTAND: TaskState.CODE_ANALYSIS,
        # Allow transition to testing
        TaskType.TEST_WRITE: TaskState.TESTING,
    },
    TaskState.MEMORY_OPS: {
        # Allow staying in MEMORY_OPS
        TaskType.MEMORY_QUERY: TaskState.MEMORY_OPS,
        TaskType.MEMORY_WRITE: TaskState.MEMORY_OPS,
        # Allow transition to code analysis
        TaskType.CODE_TRACE: TaskState.CODE_ANALYSIS,
        TaskType.CODE_UNDERSTAND: TaskState.CODE_ANALYSIS,
    },
    TaskState.PR_REVIEW: {
        # Allow transition to MEMORY_OPS to save review findings
        TaskType.MEMORY_WRITE: TaskState.MEMORY_OPS,
        # Allow staying in PR_REVIEW
        TaskType.PR_REVIEW: TaskState.PR_REVIEW,
        # Allow transition to code for deep analysis
        TaskType.CODE_TRACE: TaskState.CODE_ANALYSIS,
    },
    TaskState.TESTING: {
        # Allow transition to code for understanding
        TaskType.CODE_TRACE: TaskState.CODE_ANALYSIS,
        TaskType.CODE_UNDERSTAND: TaskState.CODE_ANALYSIS,
        # Allow staying in TESTING
        TaskType.TEST_WRITE: TaskState.TESTING,
        # Allow documenting test decisions
        TaskType.MEMORY_WRITE: TaskState.MEMORY_OPS,
    },
}
