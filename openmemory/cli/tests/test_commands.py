"""Tests for CLI commands.

This module tests the developer CLI per section 16 (FR-013):
- search: Search code and memories
- memory: Memory management
- index: Indexing operations
- graph: Graph queries
- health: System health checks
- debug: Debugging utilities
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from openmemory.cli.commands import (
    CLIConfig,
    CLIContext,
    SearchCommand,
    MemoryCommand,
    IndexCommand,
    GraphCommand,
    HealthCommand,
    DebugCommand,
    CommandResult,
    run_cli,
)


# ============================================================================
# CLIConfig Tests
# ============================================================================


class TestCLIConfig:
    """Tests for CLIConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CLIConfig()
        assert config.api_url == "http://localhost:8000"
        assert config.timeout_s == 30
        assert config.output_format == "text"

    def test_custom_config(self):
        """Test custom configuration."""
        config = CLIConfig(
            api_url="http://custom:8080",
            timeout_s=60,
            output_format="json",
        )
        assert config.api_url == "http://custom:8080"
        assert config.timeout_s == 60
        assert config.output_format == "json"


# ============================================================================
# CLIContext Tests
# ============================================================================


class TestCLIContext:
    """Tests for CLIContext."""

    def test_context_creation(self):
        """Test creating a CLI context."""
        config = CLIConfig()
        ctx = CLIContext(config=config)
        assert ctx.config == config

    def test_context_with_output(self):
        """Test context with custom output."""
        config = CLIConfig()
        output = StringIO()
        ctx = CLIContext(config=config, output=output)
        ctx.print("Hello, CLI!")
        assert "Hello, CLI!" in output.getvalue()

    def test_context_print_json(self):
        """Test JSON output mode."""
        config = CLIConfig(output_format="json")
        output = StringIO()
        ctx = CLIContext(config=config, output=output)
        ctx.print_json({"status": "ok"})
        assert '"status"' in output.getvalue()
        assert '"ok"' in output.getvalue()


# ============================================================================
# CommandResult Tests
# ============================================================================


class TestCommandResult:
    """Tests for CommandResult."""

    def test_success_result(self):
        """Test successful command result."""
        result = CommandResult(
            success=True,
            message="Operation completed",
            data={"count": 5},
        )
        assert result.success is True
        assert result.message == "Operation completed"
        assert result.data["count"] == 5

    def test_failure_result(self):
        """Test failed command result."""
        result = CommandResult(
            success=False,
            message="Operation failed",
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = CommandResult(
            success=True,
            message="Done",
            data={"items": [1, 2, 3]},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["message"] == "Done"


# ============================================================================
# SearchCommand Tests
# ============================================================================


class TestSearchCommand:
    """Tests for SearchCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_search_command_creation(self):
        """Test creating a search command."""
        cmd = SearchCommand()
        assert cmd.name == "search"
        assert cmd.description is not None

    def test_search_help(self, ctx):
        """Test search command help."""
        cmd = SearchCommand()
        result = cmd.execute(ctx, ["--help"])
        # Help should not fail
        assert result.success is True or "help" in ctx.output.getvalue().lower()

    def test_search_semantic(self, ctx):
        """Test semantic search."""
        cmd = SearchCommand()
        with patch.object(cmd, "_do_search", return_value=[]):
            result = cmd.execute(ctx, ["semantic", "find user authentication"])
            assert result.success is True

    def test_search_lexical(self, ctx):
        """Test lexical search."""
        cmd = SearchCommand()
        with patch.object(cmd, "_do_search", return_value=[]):
            result = cmd.execute(ctx, ["lexical", "class UserAuth"])
            assert result.success is True

    def test_search_hybrid(self, ctx):
        """Test hybrid search."""
        cmd = SearchCommand()
        with patch.object(cmd, "_do_search", return_value=[]):
            result = cmd.execute(ctx, ["hybrid", "authentication middleware"])
            assert result.success is True


# ============================================================================
# MemoryCommand Tests
# ============================================================================


class TestMemoryCommand:
    """Tests for MemoryCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_memory_command_creation(self):
        """Test creating a memory command."""
        cmd = MemoryCommand()
        assert cmd.name == "memory"

    def test_memory_list(self, ctx):
        """Test listing memories."""
        cmd = MemoryCommand()
        with patch.object(cmd, "_list_memories", return_value=[]):
            result = cmd.execute(ctx, ["list"])
            assert result.success is True

    def test_memory_add(self, ctx):
        """Test adding a memory."""
        cmd = MemoryCommand()
        with patch.object(cmd, "_add_memory", return_value={"id": "mem-123"}):
            result = cmd.execute(ctx, ["add", "Important note"])
            assert result.success is True

    def test_memory_search(self, ctx):
        """Test searching memories."""
        cmd = MemoryCommand()
        with patch.object(cmd, "_search_memories", return_value=[]):
            result = cmd.execute(ctx, ["search", "project config"])
            assert result.success is True


# ============================================================================
# IndexCommand Tests
# ============================================================================


class TestIndexCommand:
    """Tests for IndexCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_index_command_creation(self):
        """Test creating an index command."""
        cmd = IndexCommand()
        assert cmd.name == "index"

    def test_index_status(self, ctx):
        """Test getting index status."""
        cmd = IndexCommand()
        with patch.object(cmd, "_get_status", return_value={"status": "ready"}):
            result = cmd.execute(ctx, ["status"])
            assert result.success is True

    def test_index_trigger(self, ctx):
        """Test triggering reindex."""
        cmd = IndexCommand()
        with patch.object(cmd, "_trigger_reindex", return_value=True):
            result = cmd.execute(ctx, ["trigger"])
            assert result.success is True

    def test_index_bootstrap(self, ctx):
        """Test bootstrap status."""
        cmd = IndexCommand()
        with patch.object(cmd, "_get_bootstrap", return_value={"progress": 0.75}):
            result = cmd.execute(ctx, ["bootstrap"])
            assert result.success is True


# ============================================================================
# GraphCommand Tests
# ============================================================================


class TestGraphCommand:
    """Tests for GraphCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_graph_command_creation(self):
        """Test creating a graph command."""
        cmd = GraphCommand()
        assert cmd.name == "graph"

    def test_graph_query(self, ctx):
        """Test graph query."""
        cmd = GraphCommand()
        with patch.object(cmd, "_execute_query", return_value={"nodes": [], "edges": []}):
            result = cmd.execute(ctx, ["query", "MATCH (n) RETURN n LIMIT 10"])
            assert result.success is True

    def test_graph_callers(self, ctx):
        """Test finding callers."""
        cmd = GraphCommand()
        with patch.object(cmd, "_find_callers", return_value=[]):
            result = cmd.execute(ctx, ["callers", "UserService.authenticate"])
            assert result.success is True

    def test_graph_callees(self, ctx):
        """Test finding callees."""
        cmd = GraphCommand()
        with patch.object(cmd, "_find_callees", return_value=[]):
            result = cmd.execute(ctx, ["callees", "main"])
            assert result.success is True


# ============================================================================
# HealthCommand Tests
# ============================================================================


class TestHealthCommand:
    """Tests for HealthCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_health_command_creation(self):
        """Test creating a health command."""
        cmd = HealthCommand()
        assert cmd.name == "health"

    def test_health_check(self, ctx):
        """Test health check."""
        cmd = HealthCommand()
        with patch.object(cmd, "_check_health", return_value={"status": "healthy"}):
            result = cmd.execute(ctx, [])
            assert result.success is True

    def test_health_verbose(self, ctx):
        """Test verbose health check."""
        cmd = HealthCommand()
        with patch.object(
            cmd,
            "_check_health",
            return_value={
                "status": "healthy",
                "services": {"api": "up", "qdrant": "up", "neo4j": "up"},
            },
        ):
            result = cmd.execute(ctx, ["--verbose"])
            assert result.success is True


# ============================================================================
# DebugCommand Tests
# ============================================================================


class TestDebugCommand:
    """Tests for DebugCommand."""

    @pytest.fixture
    def ctx(self):
        """Create a context."""
        config = CLIConfig()
        output = StringIO()
        return CLIContext(config=config, output=output)

    def test_debug_command_creation(self):
        """Test creating a debug command."""
        cmd = DebugCommand()
        assert cmd.name == "debug"

    def test_debug_config(self, ctx):
        """Test showing config."""
        cmd = DebugCommand()
        result = cmd.execute(ctx, ["config"])
        assert result.success is True

    def test_debug_trace(self, ctx):
        """Test tracing a request."""
        cmd = DebugCommand()
        with patch.object(cmd, "_trace_request", return_value={"trace_id": "abc123"}):
            result = cmd.execute(ctx, ["trace", "request-123"])
            assert result.success is True

    def test_debug_audit(self, ctx):
        """Test viewing audit logs."""
        cmd = DebugCommand()
        with patch.object(cmd, "_get_audit_events", return_value=[]):
            result = cmd.execute(ctx, ["audit", "--user", "user-123"])
            assert result.success is True


# ============================================================================
# CLI Runner Tests
# ============================================================================


class TestCLIRunner:
    """Tests for CLI runner."""

    def test_run_cli_with_help(self):
        """Test running CLI with help."""
        output = StringIO()
        exit_code = run_cli(["--help"], output=output)
        assert exit_code == 0
        assert "usage" in output.getvalue().lower() or "commands" in output.getvalue().lower()

    def test_run_cli_unknown_command(self):
        """Test running CLI with unknown command."""
        output = StringIO()
        exit_code = run_cli(["unknown-cmd"], output=output)
        assert exit_code != 0 or "unknown" in output.getvalue().lower()

    def test_run_cli_health(self):
        """Test running health command."""
        output = StringIO()
        with patch("openmemory.cli.commands.HealthCommand._check_health", return_value={"status": "ok"}):
            exit_code = run_cli(["health"], output=output)
            # Should not crash
            assert exit_code in (0, 1)


# ============================================================================
# Integration Tests
# ============================================================================


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_search_workflow(self):
        """Test complete search workflow."""
        output = StringIO()
        config = CLIConfig(output_format="json")
        ctx = CLIContext(config=config, output=output)

        cmd = SearchCommand()
        with patch.object(
            cmd,
            "_do_search",
            return_value=[
                {"file": "/src/auth.py", "score": 0.95},
                {"file": "/src/user.py", "score": 0.87},
            ],
        ):
            result = cmd.execute(ctx, ["semantic", "authentication"])
            assert result.success is True
            assert result.data is not None
            assert len(result.data) >= 2

    def test_json_output_format(self):
        """Test JSON output across commands."""
        output = StringIO()
        config = CLIConfig(output_format="json")
        ctx = CLIContext(config=config, output=output)

        cmd = HealthCommand()
        with patch.object(cmd, "_check_health", return_value={"status": "healthy"}):
            result = cmd.execute(ctx, [])

        output_text = output.getvalue()
        # Should be valid JSON (command already printed JSON)
        import json

        data = json.loads(output_text)
        assert "status" in data
