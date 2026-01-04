"""
Tests for YAML-based tool group configuration.
"""
import pytest
import tempfile
from pathlib import Path

from app.routing.config import (
    ToolGroupConfig,
    StateConfig,
    load_tool_config,
    _parse_config,
    _get_default_config,
)
from app.routing.task_types import CORE_TOOLS


class TestToolGroupConfig:
    """Tests for ToolGroupConfig dataclass."""

    def test_default_core_tools(self):
        """Config should have core tools by default."""
        config = ToolGroupConfig()
        assert config.core_tools == CORE_TOOLS

    def test_get_available_tools_unknown_state(self):
        """Unknown state should return core tools."""
        config = ToolGroupConfig()
        tools = config.get_available_tools("UNKNOWN_STATE")
        assert set(tools) == CORE_TOOLS

    def test_get_hidden_patterns_unknown_state(self):
        """Unknown state should hide everything."""
        config = ToolGroupConfig()
        patterns = config.get_hidden_patterns("UNKNOWN_STATE")
        assert "*" in patterns


class TestStateConfig:
    """Tests for StateConfig dataclass."""

    def test_state_config_creation(self):
        """StateConfig should store all fields."""
        config = StateConfig(
            description="Test state",
            available=["Read", "Grep"],
            hidden=["search_memory"],
        )
        assert config.description == "Test state"
        assert "Read" in config.available
        assert "search_memory" in config.hidden


class TestLoadToolConfig:
    """Tests for loading configuration from YAML."""

    def test_load_default_config_file(self):
        """Should load the default config file."""
        config = load_tool_config()
        assert config is not None
        assert config.version == "1.0"

    def test_loaded_config_has_states(self):
        """Loaded config should have all required states."""
        config = load_tool_config()
        required_states = ["INITIAL", "CODE_ANALYSIS", "MEMORY_OPS", "PR_REVIEW", "TESTING"]
        for state in required_states:
            assert state in config.states, f"Missing state: {state}"

    def test_loaded_config_has_transitions(self):
        """Loaded config should have transition rules."""
        config = load_tool_config()
        assert "INITIAL" in config.transitions
        assert len(config.transitions["INITIAL"]) > 0

    def test_load_from_custom_path(self):
        """Should load from a custom path."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
version: "2.0"
core_tools:
  - Read
  - Grep
states:
  TEST_STATE:
    description: Test
    available:
      - Read
    hidden:
      - "*"
transitions: {}
""")
            f.flush()
            config = load_tool_config(Path(f.name))
            assert config.version == "2.0"
            assert "TEST_STATE" in config.states

    def test_load_fallback_on_missing_file(self):
        """Should use defaults if file is missing."""
        config = load_tool_config(Path("/nonexistent/path.yaml"))
        assert config is not None
        assert len(config.states) > 0


class TestParseConfig:
    """Tests for config parsing."""

    def test_parse_minimal_config(self):
        """Should parse a minimal config."""
        data = {
            "version": "1.0",
            "core_tools": ["Read"],
            "states": {},
            "transitions": {},
        }
        config = _parse_config(data)
        assert config.version == "1.0"
        assert "Read" in config.core_tools

    def test_parse_full_config(self):
        """Should parse a complete config."""
        data = {
            "version": "1.0",
            "core_tools": ["Read", "Grep", "Glob", "Bash"],
            "states": {
                "CODE_ANALYSIS": {
                    "description": "Code analysis tools",
                    "available": ["find_callers", "find_callees"],
                    "hidden": ["search_memory"],
                },
            },
            "transitions": {
                "INITIAL": {
                    "CODE_TRACE": "CODE_ANALYSIS",
                },
            },
        }
        config = _parse_config(data)
        assert "CODE_ANALYSIS" in config.states
        assert "find_callers" in config.states["CODE_ANALYSIS"].available
        assert config.get_next_state("INITIAL", "CODE_TRACE") == "CODE_ANALYSIS"


class TestDefaultConfig:
    """Tests for default configuration."""

    def test_default_config_has_all_states(self):
        """Default config should have all TaskStates."""
        config = _get_default_config()
        from app.routing.task_types import TaskState
        for state in TaskState:
            assert state.name in config.states

    def test_default_config_has_transitions(self):
        """Default config should have all transitions."""
        config = _get_default_config()
        assert "INITIAL" in config.transitions
        assert "CODE_TRACE" in config.transitions["INITIAL"]


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_config_matches_task_types(self):
        """Config states should match TaskType names."""
        config = load_tool_config()
        from app.routing.task_types import TaskState

        for state in TaskState:
            assert state.name in config.states, \
                f"TaskState {state.name} not in config"

    def test_core_tools_in_all_states(self):
        """Core tools should be available in all state configs."""
        config = load_tool_config()

        for state_name, state_config in config.states.items():
            for core_tool in CORE_TOOLS:
                # Core tools should either be explicitly available
                # or not explicitly hidden
                assert core_tool in state_config.available, \
                    f"Core tool {core_tool} missing from {state_name}.available"
