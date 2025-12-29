"""
Guidance MCP Server

Serves on-demand guidance documents over MCP so clients can request only the
relevant material instead of loading all guidance into the system prompt.

Exposes a dedicated MCP SSE endpoint under /guidance/...
"""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# Initialize MCP
mcp = FastMCP("guidance")

# Pfad zu den Anleitungsdateien
GUIDANCE_DIR = Path(__file__).parent / "guidance"

# Create a router for Guidance endpoints
guidance_router = APIRouter(prefix="/guidance")

# Initialize SSE transport
sse = SseServerTransport("/guidance/messages/")

# Mapping: short name -> file + description
GUIDES = {
    "risks": {
        "file": "risk_contradictions.md",
        "description": "Contradictions, risks, and assumptions capture",
        "when": "When tracking conflicts or risks in decisions"
    },
    "memory": {
        "file": "memory_examples.md",
        "description": "Structured memory examples and metadata patterns",
        "when": "When capturing durable facts, conventions, and decisions"
    },
    "maintenance": {
        "file": "memory_maintenance.md",
        "description": "Memory review, deprecation, and upkeep practices",
        "when": "When reviewing or updating existing memories"
    },
    "tasks": {
        "file": "task_capture.md",
        "description": "Task capture guidance for durable workflow items",
        "when": "When deciding whether a task should become memory"
    },
    "calendar": {
        "file": "calendar.md",
        "description": "Recurring meetings and time-based workflow memories",
        "when": "When capturing schedules or recurring processes"
    },
    "messages": {
        "file": "messages.md",
        "description": "Message capture patterns for decisions and conventions",
        "when": "When translating conversations into structured memories"
    },
    "reference": {
        "file": "quick_reference.md",
        "description": "Compact reference for memory and graph tools",
        "when": "When you need a quick schema or tool reminder"
    },
    "auto_detection": {
        "file": "auto_detection.md",
        "description": "Heuristics for extracting structured metadata",
        "when": "When classifying category/scope from free-form text"
    },
    "graph": {
        "file": "graph.md",
        "description": "Neo4j graph operations and metadata relations",
        "when": "When using graph tools or entity networks"
    },
}


@mcp.tool(description="""Load a guidance document on demand.

Available guides:
- risks: contradictions and risk capture
- memory: structured memory examples
- maintenance: memory review and updates
- tasks: task capture guidance
- calendar: recurring schedule memories
- messages: message-to-memory patterns
- reference: quick tool reference
- auto_detection: metadata extraction heuristics
- graph: graph operations and entity networks

Example: get_guidance("memory") -> loads memory_examples.md
""")
async def get_guidance(guide_name: str) -> str:
    """Load a specific guidance document."""
    guide_name = guide_name.lower().strip()

    if guide_name not in GUIDES:
        available = ", ".join(GUIDES.keys())
        return f"Unknown guide: '{guide_name}'. Available: {available}"

    guide = GUIDES[guide_name]
    file_path = GUIDANCE_DIR / guide["file"]

    if not file_path.exists():
        return f"Guide file not found: {guide['file']}"

    content = file_path.read_text(encoding="utf-8")
    return f"# Guidance: {guide_name}\n\n{content}"


@mcp.tool(description="List all available guidance documents with descriptions and usage hints.")
async def list_guides() -> str:
    """Return an overview of all available guides."""
    lines = ["# Available Guidance\n"]
    for name, info in GUIDES.items():
        lines.append(f"## {name}")
        lines.append(f"- **File:** {info['file']}")
        lines.append(f"- **Content:** {info['description']}")
        lines.append(f"- **When:** {info['when']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(description="Search guidance documents for a keyword and return context.")
async def search_guidance(query: str) -> str:
    """Search guidance documents for a term."""
    results = []
    query_lower = query.lower()

    for name, info in GUIDES.items():
        file_path = GUIDANCE_DIR / info["file"]
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8")
        if query_lower in content.lower():
            # Finde Zeilen mit Treffer
            lines = content.split("\n")
            matches = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Kontext: 1 Zeile davor/danach
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])
                    matches.append(f"  Zeile {i+1}:\n{context}")

            if matches:
                results.append(f"## {name} ({info['file']})\n" + "\n---\n".join(matches[:3]))

    if not results:
        return f"No matches for '{query}' in guidance documents."

    return f"# Results for '{query}'\n\n" + "\n\n".join(results)


@guidance_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for guidance."""
    try:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    except Exception:
        pass


@guidance_router.post("/messages/")
async def handle_messages_root(request: Request):
    """Handle POST messages for SSE (root path)."""
    return await _handle_post_message(request)


@guidance_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_messages_user(request: Request):
    """Handle POST messages for SSE (user path)."""
    return await _handle_post_message(request)


async def _handle_post_message(request: Request):
    """Handle POST messages for SSE."""
    try:
        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            return {}

        await sse.handle_post_message(request.scope, receive, send)
        return {"status": "ok"}
    except Exception:
        return {"status": "error"}


def setup_guidance_server(app: FastAPI):
    """Setup Guidance MCP server with the FastAPI application."""
    mcp._mcp_server.name = "guidance"
    app.include_router(guidance_router)


# For standalone testing
if __name__ == "__main__":
    mcp.run()
