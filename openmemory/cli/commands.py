"""CLI commands for OpenMemory.

This module implements developer CLI commands per section 16 (FR-013):
- search: Search code and memories
- memory: Memory management
- index: Indexing operations
- graph: Graph queries
- health: System health checks
- debug: Debugging utilities
"""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, TextIO


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CLIConfig:
    """Configuration for CLI."""

    api_url: str = "http://localhost:8000"
    timeout_s: int = 30
    output_format: str = "text"  # text or json
    verbose: bool = False


# ============================================================================
# Context
# ============================================================================


class CLIContext:
    """Context for CLI command execution."""

    def __init__(
        self,
        config: CLIConfig,
        output: TextIO | None = None,
    ):
        """Initialize the context.

        Args:
            config: CLI configuration
            output: Output stream (defaults to stdout)
        """
        self.config = config
        self.output = output or sys.stdout

    def print(self, message: str) -> None:
        """Print a message to output.

        Args:
            message: The message to print
        """
        print(message, file=self.output)

    def print_json(self, data: dict[str, Any]) -> None:
        """Print JSON data to output.

        Args:
            data: The data to print as JSON
        """
        print(json.dumps(data, indent=2, default=str), file=self.output)

    def print_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Print a table to output.

        Args:
            headers: Table column headers
            rows: Table rows
        """
        if not rows:
            self.print("No data")
            return

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        self.print(header_line)
        self.print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(c).ljust(w) for c, w in zip(row, widths))
            self.print(row_line)


# ============================================================================
# Command Result
# ============================================================================


@dataclass
class CommandResult:
    """Result from a CLI command."""

    success: bool
    message: str = ""
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }


# ============================================================================
# Base Command
# ============================================================================


class Command(ABC):
    """Base class for CLI commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get command name."""
        pass

    @property
    def description(self) -> str:
        """Get command description."""
        return ""

    @abstractmethod
    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute the command.

        Args:
            ctx: CLI context
            args: Command arguments

        Returns:
            Command result
        """
        pass


# ============================================================================
# Search Command
# ============================================================================


class SearchCommand(Command):
    """Search code and memories."""

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search code and memories (semantic, lexical, or hybrid)"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute search command."""
        if not args or args[0] == "--help":
            ctx.print("Usage: search <mode> <query>")
            ctx.print("Modes: semantic, lexical, hybrid")
            ctx.print("Example: search semantic 'user authentication'")
            return CommandResult(success=True, message="Help displayed")

        mode = args[0] if args else "hybrid"
        query = " ".join(args[1:]) if len(args) > 1 else ""

        if not query:
            return CommandResult(
                success=False,
                message="Query is required",
                error="No query provided",
            )

        results = self._do_search(mode, query, ctx.config)

        if ctx.config.output_format == "json":
            ctx.print_json({"mode": mode, "query": query, "results": results})
        else:
            ctx.print(f"Search ({mode}): {query}")
            ctx.print(f"Found {len(results)} results")
            for r in results[:10]:
                ctx.print(f"  - {r.get('file', 'unknown')}: {r.get('score', 0):.2f}")

        return CommandResult(success=True, message="Search completed", data=results)

    def _do_search(
        self,
        mode: str,
        query: str,
        config: CLIConfig,
    ) -> list[dict[str, Any]]:
        """Perform the search (stub for testing).

        In production, this would call the API.
        """
        # Stub implementation
        return []


# ============================================================================
# Memory Command
# ============================================================================


class MemoryCommand(Command):
    """Memory management commands."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Manage memories (list, add, search, delete)"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute memory command."""
        if not args or args[0] == "--help":
            ctx.print("Usage: memory <action> [args]")
            ctx.print("Actions: list, add, search, delete")
            return CommandResult(success=True, message="Help displayed")

        action = args[0]
        action_args = args[1:]

        if action == "list":
            memories = self._list_memories(ctx.config)
            ctx.print(f"Found {len(memories)} memories")
            return CommandResult(success=True, data=memories)

        elif action == "add":
            content = " ".join(action_args)
            result = self._add_memory(content, ctx.config)
            ctx.print(f"Added memory: {result.get('id', 'unknown')}")
            return CommandResult(success=True, data=result)

        elif action == "search":
            query = " ".join(action_args)
            results = self._search_memories(query, ctx.config)
            ctx.print(f"Found {len(results)} matching memories")
            return CommandResult(success=True, data=results)

        else:
            return CommandResult(
                success=False,
                message=f"Unknown action: {action}",
                error=f"Unknown action: {action}",
            )

    def _list_memories(self, config: CLIConfig) -> list[dict]:
        """List memories (stub)."""
        return []

    def _add_memory(self, content: str, config: CLIConfig) -> dict:
        """Add a memory (stub)."""
        return {"id": "mem-new"}

    def _search_memories(self, query: str, config: CLIConfig) -> list[dict]:
        """Search memories (stub)."""
        return []


# ============================================================================
# Index Command
# ============================================================================


class IndexCommand(Command):
    """Indexing operations."""

    @property
    def name(self) -> str:
        return "index"

    @property
    def description(self) -> str:
        return "Manage code indexing (status, trigger, bootstrap)"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute index command."""
        if not args or args[0] == "--help":
            ctx.print("Usage: index <action>")
            ctx.print("Actions: status, trigger, bootstrap")
            return CommandResult(success=True, message="Help displayed")

        action = args[0]

        if action == "status":
            status = self._get_status(ctx.config)
            if ctx.config.output_format == "json":
                ctx.print_json(status)
            else:
                ctx.print(f"Index status: {status.get('status', 'unknown')}")
            return CommandResult(success=True, data=status)

        elif action == "trigger":
            success = self._trigger_reindex(ctx.config)
            msg = "Reindex triggered" if success else "Failed to trigger reindex"
            ctx.print(msg)
            return CommandResult(success=success, message=msg)

        elif action == "bootstrap":
            bootstrap = self._get_bootstrap(ctx.config)
            if ctx.config.output_format == "json":
                ctx.print_json(bootstrap)
            else:
                progress = bootstrap.get("progress", 0) * 100
                ctx.print(f"Bootstrap progress: {progress:.1f}%")
            return CommandResult(success=True, data=bootstrap)

        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}",
            )

    def _get_status(self, config: CLIConfig) -> dict:
        """Get index status (stub)."""
        return {"status": "ready"}

    def _trigger_reindex(self, config: CLIConfig) -> bool:
        """Trigger reindex (stub)."""
        return True

    def _get_bootstrap(self, config: CLIConfig) -> dict:
        """Get bootstrap status (stub)."""
        return {"progress": 0.0, "eta_seconds": None}


# ============================================================================
# Graph Command
# ============================================================================


class GraphCommand(Command):
    """Graph query commands."""

    @property
    def name(self) -> str:
        return "graph"

    @property
    def description(self) -> str:
        return "Query the code graph (query, callers, callees)"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute graph command."""
        if not args or args[0] == "--help":
            ctx.print("Usage: graph <action> [args]")
            ctx.print("Actions: query, callers, callees")
            return CommandResult(success=True, message="Help displayed")

        action = args[0]
        action_args = args[1:]

        if action == "query":
            cypher = " ".join(action_args)
            result = self._execute_query(cypher, ctx.config)
            if ctx.config.output_format == "json":
                ctx.print_json(result)
            else:
                ctx.print(f"Nodes: {len(result.get('nodes', []))}")
                ctx.print(f"Edges: {len(result.get('edges', []))}")
            return CommandResult(success=True, data=result)

        elif action == "callers":
            symbol = action_args[0] if action_args else ""
            callers = self._find_callers(symbol, ctx.config)
            ctx.print(f"Found {len(callers)} callers for {symbol}")
            return CommandResult(success=True, data=callers)

        elif action == "callees":
            symbol = action_args[0] if action_args else ""
            callees = self._find_callees(symbol, ctx.config)
            ctx.print(f"Found {len(callees)} callees for {symbol}")
            return CommandResult(success=True, data=callees)

        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}",
            )

    def _execute_query(self, cypher: str, config: CLIConfig) -> dict:
        """Execute Cypher query (stub)."""
        return {"nodes": [], "edges": []}

    def _find_callers(self, symbol: str, config: CLIConfig) -> list[dict]:
        """Find callers (stub)."""
        return []

    def _find_callees(self, symbol: str, config: CLIConfig) -> list[dict]:
        """Find callees (stub)."""
        return []


# ============================================================================
# Health Command
# ============================================================================


class HealthCommand(Command):
    """System health check."""

    @property
    def name(self) -> str:
        return "health"

    @property
    def description(self) -> str:
        return "Check system health"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute health command."""
        verbose = "--verbose" in args or "-v" in args

        health = self._check_health(ctx.config)

        if ctx.config.output_format == "json":
            ctx.print_json(health)
        else:
            status = health.get("status", "unknown")
            ctx.print(f"System Health: {status.upper()}")

            if verbose and "services" in health:
                ctx.print("\nServices:")
                for name, state in health["services"].items():
                    ctx.print(f"  {name}: {state}")

        success = health.get("status") in ("healthy", "ok")
        return CommandResult(success=success, data=health)

    def _check_health(self, config: CLIConfig) -> dict:
        """Check health (stub)."""
        return {
            "status": "healthy",
            "services": {
                "api": "up",
                "qdrant": "up",
                "neo4j": "up",
                "opensearch": "up",
            },
        }


# ============================================================================
# Debug Command
# ============================================================================


class DebugCommand(Command):
    """Debugging utilities."""

    @property
    def name(self) -> str:
        return "debug"

    @property
    def description(self) -> str:
        return "Debugging utilities (config, trace, audit)"

    def execute(self, ctx: CLIContext, args: list[str]) -> CommandResult:
        """Execute debug command."""
        if not args or args[0] == "--help":
            ctx.print("Usage: debug <action> [args]")
            ctx.print("Actions: config, trace, audit")
            return CommandResult(success=True, message="Help displayed")

        action = args[0]
        action_args = args[1:]

        if action == "config":
            config_data = {
                "api_url": ctx.config.api_url,
                "timeout_s": ctx.config.timeout_s,
                "output_format": ctx.config.output_format,
            }
            if ctx.config.output_format == "json":
                ctx.print_json(config_data)
            else:
                ctx.print("Current Configuration:")
                for k, v in config_data.items():
                    ctx.print(f"  {k}: {v}")
            return CommandResult(success=True, data=config_data)

        elif action == "trace":
            trace_id = action_args[0] if action_args else ""
            trace_data = self._trace_request(trace_id, ctx.config)
            if ctx.config.output_format == "json":
                ctx.print_json(trace_data)
            else:
                ctx.print(f"Trace ID: {trace_data.get('trace_id', 'unknown')}")
            return CommandResult(success=True, data=trace_data)

        elif action == "audit":
            # Parse options
            user_id = None
            for i, arg in enumerate(action_args):
                if arg == "--user" and i + 1 < len(action_args):
                    user_id = action_args[i + 1]

            events = self._get_audit_events(user_id, ctx.config)
            ctx.print(f"Found {len(events)} audit events")
            return CommandResult(success=True, data=events)

        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}",
            )

    def _trace_request(self, trace_id: str, config: CLIConfig) -> dict:
        """Get trace data (stub)."""
        return {"trace_id": trace_id, "spans": []}

    def _get_audit_events(self, user_id: str | None, config: CLIConfig) -> list[dict]:
        """Get audit events (stub)."""
        return []


# ============================================================================
# CLI Runner
# ============================================================================


COMMANDS = {
    "search": SearchCommand,
    "memory": MemoryCommand,
    "index": IndexCommand,
    "graph": GraphCommand,
    "health": HealthCommand,
    "debug": DebugCommand,
}


def run_cli(
    args: list[str],
    output: TextIO | None = None,
) -> int:
    """Run the CLI.

    Args:
        args: Command-line arguments
        output: Output stream

    Returns:
        Exit code
    """
    output = output or sys.stdout
    config = CLIConfig()

    # Parse global options
    remaining_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            config.output_format = "json"
        elif arg == "--api-url" and i + 1 < len(args):
            config.api_url = args[i + 1]
            i += 1
        elif arg == "--verbose" or arg == "-v":
            config.verbose = True
        elif arg == "--help" or arg == "-h":
            print_help(output)
            return 0
        else:
            remaining_args.append(arg)
        i += 1

    if not remaining_args:
        print_help(output)
        return 0

    # Get command
    command_name = remaining_args[0]
    command_args = remaining_args[1:]

    if command_name not in COMMANDS:
        print(f"Unknown command: {command_name}", file=output)
        print(f"Available commands: {', '.join(COMMANDS.keys())}", file=output)
        return 1

    # Execute command
    ctx = CLIContext(config=config, output=output)
    command = COMMANDS[command_name]()
    result = command.execute(ctx, command_args)

    return 0 if result.success else 1


def print_help(output: TextIO) -> None:
    """Print CLI help."""
    print("OpenMemory CLI", file=output)
    print("", file=output)
    print("Usage: openmemory <command> [options]", file=output)
    print("", file=output)
    print("Commands:", file=output)
    for name, cmd_class in COMMANDS.items():
        cmd = cmd_class()
        print(f"  {name:12} {cmd.description}", file=output)
    print("", file=output)
    print("Global Options:", file=output)
    print("  --json         Output in JSON format", file=output)
    print("  --api-url URL  API server URL", file=output)
    print("  --verbose, -v  Verbose output", file=output)
    print("  --help, -h     Show this help", file=output)
