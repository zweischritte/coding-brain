"""Settings module for Coding Brain API."""

from .settings import Settings, get_settings, reset_settings

__all__ = ["Settings", "get_settings", "reset_settings"]

# Module-level settings instance for cache clearing in tests
_settings = None
