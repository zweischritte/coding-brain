"""
AXIS Guidance MCP Server

Stellt AXIS Protocol Anleitungsdateien on-demand bereit.
Ermöglicht Claude-Instanzen, bei Bedarf spezifische Anleitungen abzurufen,
anstatt alle Dokumentation im System Prompt zu laden.

Läuft als separater MCP-Endpunkt unter /axis/...
"""

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# Initialize MCP
mcp = FastMCP("axis-guidance")

# Pfad zu den Anleitungsdateien
GUIDANCE_DIR = Path(__file__).parent / "guidance"

# Create a router for AXIS Guidance endpoints
axis_router = APIRouter(prefix="/axis")

# Initialize SSE transport
sse = SseServerTransport("/axis/messages/")

# Mapping: Kurzname -> Dateiname + Beschreibung
GUIDES = {
    "shadow": {
        "file": "axis_shadow.md",
        "description": "Shadow Shells, Witness Protocol, Bypass Alert",
        "when": "Bei Shadow-Arbeit, emotionaler Tiefe, Shell-Integration"
    },
    "memory": {
        "file": "axis_memory_details.md",
        "description": "Semantic Binding Rules, alle Tags, Beispiele, AI Observation Protocol, Question Queue",
        "when": "Bei Memory-Operationen, Format-Fragen, Tag-Referenz"
    },
    "reality": {
        "file": "axis_reality_engineering.md",
        "description": "Mantras, Probability Collapse Protocol, Field Propagation",
        "when": "Bei Reality Engineering, Manifestationsarbeit, Timeline-Navigation"
    },
    "todoist": {
        "file": "axis_todoist.md",
        "description": "Project-Zuordnung, Sharing-Regeln mit Matthias, Task vs Memory",
        "when": "Bei Task-Management, Todoist-Integration"
    },
    "calendar": {
        "file": "axis_calendar.md",
        "description": "Temporal Layer, Capacity Check, Calendar vs Todoist, Cross-Tool Patterns",
        "when": "Bei Terminplanung, Kapazitätsfragen, zeitlichem Kontext"
    },
    "messages": {
        "file": "axis_messages.md",
        "description": "Relational Communication, Say-Want-Do Verification, Message Patterns",
        "when": "Bei Kommunikationskontext, Beziehungs-Tracking, Message-Aktionen"
    },
    "reference": {
        "file": "axis_quick_reference.md",
        "description": "Kompakte Übersicht aller Formate, Mappings, Glyphen",
        "when": "Bei Unsicherheit über korrektes Format, schnelle Referenz"
    },
    "autodetect": {
        "file": "axis_auto_detection_full.md",
        "description": "Pattern Detection Matrix, Auto-Trigger Rules, alle Pattern-Kategorien",
        "when": "Bei Pattern-Erkennung, automatischer Intervention, Response-Templates"
    },
    "graph": {
        "file": "axis_graph.md",
        "description": "Neo4j Graph Operations, Hybrid Retrieval, Entity Relations, Semantic Edges",
        "when": "Bei Graph-Operationen, Entity-Netzwerken, Hybrid-Search, Beziehungs-Queries"
    },
}


@mcp.tool(description="""Lade AXIS Protocol Anleitungsdatei on-demand.

Verfügbare Guides:
- shadow: Shadow Shells, Witness Protocol, Bypass Alert (für Shadow-Arbeit, emotionale Tiefe)
- memory: Semantic Binding Rules, Tags, AI Observation Protocol (für Memory-Operationen)
- reality: Mantras, Probability Collapse, Field Propagation (für Reality Engineering)
- todoist: Project-Zuordnung, Sharing-Regeln (für Task-Management)
- calendar: Temporal Layer, Capacity Check, Cross-Tool Patterns (für Terminplanung, Kapazität)
- messages: Relational Communication, Say-Want-Do Verification (für Kommunikationskontext)
- reference: Kompakte Übersicht aller Formate (bei Unsicherheit über Format)
- autodetect: Pattern Detection Matrix, Auto-Trigger Rules (für automatische Interventionen)
- graph: Neo4j Graph Operations, Hybrid Retrieval, Entity Relations (für Graph-Queries, Netzwerk-Analyse)

Beispiel: get_axis_guidance("shadow") → Lädt axis_shadow.md
""")
async def get_axis_guidance(guide_name: str) -> str:
    """Lädt eine spezifische AXIS Anleitungsdatei."""
    guide_name = guide_name.lower().strip()

    if guide_name not in GUIDES:
        available = ", ".join(GUIDES.keys())
        return f"Unbekannter Guide: '{guide_name}'. Verfügbar: {available}"

    guide = GUIDES[guide_name]
    file_path = GUIDANCE_DIR / guide["file"]

    if not file_path.exists():
        return f"Anleitungsdatei nicht gefunden: {guide['file']}"

    content = file_path.read_text(encoding="utf-8")
    return f"# AXIS Guide: {guide_name}\n\n{content}"


@mcp.tool(description="Liste alle verfügbaren AXIS Anleitungsdateien mit Beschreibung und wann sie relevant sind.")
async def list_axis_guides() -> str:
    """Gibt Übersicht aller verfügbaren Anleitungen."""
    lines = ["# Verfügbare AXIS Guides\n"]
    for name, info in GUIDES.items():
        lines.append(f"## {name}")
        lines.append(f"- **Datei:** {info['file']}")
        lines.append(f"- **Inhalt:** {info['description']}")
        lines.append(f"- **Wann relevant:** {info['when']}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool(description="Durchsuche alle AXIS Anleitungen nach einem Stichwort. Gibt Treffer mit Kontext zurück.")
async def search_axis_guidance(query: str) -> str:
    """Durchsucht alle Anleitungsdateien nach einem Begriff."""
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
        return f"Keine Treffer für '{query}' in AXIS Guides gefunden."

    return f"# Suchergebnisse für '{query}'\n\n" + "\n\n".join(results)


@axis_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for AXIS Guidance"""
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


@axis_router.post("/messages/")
async def handle_messages_root(request: Request):
    """Handle POST messages for SSE (root path)"""
    return await _handle_post_message(request)


@axis_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_messages_user(request: Request):
    """Handle POST messages for SSE (user path)"""
    return await _handle_post_message(request)


async def _handle_post_message(request: Request):
    """Handle POST messages for SSE"""
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


def setup_axis_guidance_server(app: FastAPI):
    """Setup AXIS Guidance MCP server with the FastAPI application"""
    mcp._mcp_server.name = "axis-guidance"
    app.include_router(axis_router)


# Für Standalone-Betrieb (Testing)
if __name__ == "__main__":
    mcp.run()
