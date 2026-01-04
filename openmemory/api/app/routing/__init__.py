"""
Task-based Tool Routing for Coding Brain.

This module provides intelligent task routing to reduce tool overload
and improve agent accuracy. Key components:

- TaskType: Classification of user requests (CODE_TRACE, MEMORY_QUERY, etc.)
- TaskState: State machine states (INITIAL, CODE_ANALYSIS, MEMORY_OPS, etc.)
- IntentClassifier: Hybrid embedding + LLM classification
- TaskStateMachine: State management for tool availability
- ToolGroupConfig: YAML-based configuration for tool groups

Usage:
    from app.routing import classify_intent, TaskType, TaskStateMachine

    # Classify user intent
    result = classify_intent("Debug the login error")
    assert result.intent == TaskType.CODE_TRACE

    # Use state machine for tool filtering
    machine = TaskStateMachine()
    machine.transition(result.intent)
    tools = machine.available_tools  # Only code-relevant tools

    # Load custom tool configuration
    from app.routing import get_tool_config
    config = get_tool_config()
"""

from .task_types import (
    TaskType,
    TaskState,
    TOOL_AVAILABILITY,
    STATE_TRANSITIONS,
    CORE_TOOLS,
)

from .state_machine import TaskStateMachine

from .intent_classifier import (
    IntentClassifier,
    ClassificationResult,
    classify_intent,
    INTENT_CLASSIFIER_PROMPT,
)

from .config import (
    ToolGroupConfig,
    StateConfig,
    load_tool_config,
    get_tool_config,
    reload_tool_config,
)

__all__ = [
    # Enums
    "TaskType",
    "TaskState",
    # Configuration
    "TOOL_AVAILABILITY",
    "STATE_TRANSITIONS",
    "CORE_TOOLS",
    # YAML Config
    "ToolGroupConfig",
    "StateConfig",
    "load_tool_config",
    "get_tool_config",
    "reload_tool_config",
    # State machine
    "TaskStateMachine",
    # Intent classification
    "IntentClassifier",
    "ClassificationResult",
    "classify_intent",
    "INTENT_CLASSIFIER_PROMPT",
]
