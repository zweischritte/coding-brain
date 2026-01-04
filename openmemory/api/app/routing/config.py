"""
Configuration loader for task-based tool routing.

Loads tool group configurations from YAML or uses built-in defaults.
Allows runtime reconfiguration without code changes.

Usage:
    from app.routing.config import load_tool_config, ToolGroupConfig

    config = load_tool_config()  # Load from YAML or use defaults
    print(config.get_available_tools("CODE_ANALYSIS"))
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from .task_types import (
    TaskType,
    TaskState,
    TOOL_AVAILABILITY,
    STATE_TRANSITIONS,
    CORE_TOOLS,
)

logger = logging.getLogger(__name__)

# Default config file path (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "tool_groups.yaml"


@dataclass
class StateConfig:
    """Configuration for a single task state."""

    description: str
    available: List[str]
    hidden: List[str]


@dataclass
class ToolGroupConfig:
    """
    Complete tool group configuration.

    Attributes:
        version: Config file version
        core_tools: Tools that are always available
        states: Configuration per TaskState
        transitions: State transition rules
    """

    version: str = "1.0"
    core_tools: Set[str] = field(default_factory=lambda: set(CORE_TOOLS))
    states: Dict[str, StateConfig] = field(default_factory=dict)
    transitions: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def get_available_tools(self, state: str) -> List[str]:
        """Get list of available tools for a state."""
        if state in self.states:
            return self.states[state].available
        return list(self.core_tools)

    def get_hidden_patterns(self, state: str) -> List[str]:
        """Get list of hidden tool patterns for a state."""
        if state in self.states:
            return self.states[state].hidden
        return ["*"]

    def get_next_state(
        self,
        current_state: str,
        intent: str,
    ) -> Optional[str]:
        """Get next state for a given current state and intent."""
        if current_state in self.transitions:
            return self.transitions[current_state].get(intent)
        return None


def load_tool_config(config_path: Optional[Path] = None) -> ToolGroupConfig:
    """
    Load tool configuration from YAML file.

    Falls back to built-in defaults if file not found or invalid.

    Args:
        config_path: Path to YAML config file

    Returns:
        ToolGroupConfig with loaded or default settings
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            return _parse_config(data)

        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using built-in defaults")

    return _get_default_config()


def _parse_config(data: dict) -> ToolGroupConfig:
    """Parse YAML data into ToolGroupConfig."""
    config = ToolGroupConfig(
        version=data.get("version", "1.0"),
        core_tools=set(data.get("core_tools", CORE_TOOLS)),
    )

    # Parse states
    states_data = data.get("states", {})
    for state_name, state_config in states_data.items():
        config.states[state_name] = StateConfig(
            description=state_config.get("description", ""),
            available=state_config.get("available", []),
            hidden=state_config.get("hidden", []),
        )

    # Parse transitions
    config.transitions = data.get("transitions", {})

    return config


def _get_default_config() -> ToolGroupConfig:
    """Get built-in default configuration."""
    config = ToolGroupConfig()

    # Convert TOOL_AVAILABILITY to StateConfig
    for state, availability in TOOL_AVAILABILITY.items():
        config.states[state.name] = StateConfig(
            description=f"Default config for {state.name}",
            available=availability.get("available", []),
            hidden=availability.get("hidden", []),
        )

    # Convert STATE_TRANSITIONS
    for from_state, transitions in STATE_TRANSITIONS.items():
        config.transitions[from_state.name] = {
            intent.name: to_state.name
            for intent, to_state in transitions.items()
        }

    return config


def save_tool_config(
    config: ToolGroupConfig,
    config_path: Optional[Path] = None,
) -> None:
    """
    Save tool configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to save to
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    data = {
        "version": config.version,
        "core_tools": list(config.core_tools),
        "states": {},
        "transitions": config.transitions,
    }

    for state_name, state_config in config.states.items():
        data["states"][state_name] = {
            "description": state_config.description,
            "available": state_config.available,
            "hidden": state_config.hidden,
        }

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {config_path}")


# Singleton config instance
_config: Optional[ToolGroupConfig] = None


def get_tool_config() -> ToolGroupConfig:
    """
    Get the global tool configuration (cached).

    Returns:
        ToolGroupConfig singleton
    """
    global _config
    if _config is None:
        _config = load_tool_config()
    return _config


def reload_tool_config() -> ToolGroupConfig:
    """
    Reload tool configuration from file.

    Returns:
        Newly loaded ToolGroupConfig
    """
    global _config
    _config = load_tool_config()
    return _config
