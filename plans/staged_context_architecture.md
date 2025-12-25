# AXIS Tagesstart: Staged Context Architecture

> **Version:** 1.1 (nach Review)
> **Status:** Ready for Implementation

## Problem Statement

Die aktuelle Architektur hat ein fundamentales Problem:
- **Subagents haben keinen MCP-Zugriff** (nur Bash, Glob, Grep, Read, Edit, Write)
- **Hauptagent wird überflutet** wenn er alle MCP-Daten selbst verarbeitet
- 27+ Tasks, 50+ Calendar Events, Messages, Memories → Kontext-Explosion

## Lösung: Staged Context Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGED CONTEXT FLOW                         │
└─────────────────────────────────────────────────────────────────────┘

Phase 0: SETUP
    Hauptagent
        │
        └──► mkdir /tmp/axis-tagesstart-{date}/

Phase 1: DATA COLLECTION (Hauptagent, parallel MCP calls)
    Hauptagent
        │
        ├──► mcp__todoist__find-tasks-by-date ──► Write → todoist_raw.json
        ├──► mcp__mcp-ical__list_events ────────► Write → calendar_raw.json
        ├──► mcp__messages__tool_get_recent ────► Write → messages_raw.json
        ├──► mcp__openmemory-local__search ─────► Write → memory_raw.json
        └──► mcp__notionMCP__notion-fetch ──────► Write → notion_raw.json
             (Docs der letzten 2 Tage)

Phase 2: ANALYSIS (Parallel Subagents, Read/Write only)
    ┌────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
    │ TODOIST        │ CALENDAR       │ MESSAGES       │ MEMORY         │ NOTION         │
    │ ANALYZER       │ ANALYZER       │ ANALYZER       │ ANALYZER       │ ANALYZER       │
    │                │                │                │                │                │
    │ Read:          │ Read:          │ Read:          │ Read:          │ Read:          │
    │ todoist_raw    │ calendar_raw   │ messages_raw   │ memory_raw     │ notion_raw     │
    │                │                │                │                │ (letzte 2 Tage)│
    │ Write:         │ Write:         │ Write:         │ Write:         │ Write:         │
    │ todoist_       │ calendar_      │ messages_      │ memory_        │ notion_        │
    │ summary.md     │ summary.md     │ summary.md     │ summary.md     │ summary.md     │
    └────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘

Phase 3: SYNTHESIS (Hauptagent reads summaries only)
    Hauptagent
        │
        ├──► Read todoist_summary.md    (kompakt: ~20 Zeilen)
        ├──► Read calendar_summary.md   (kompakt: ~20 Zeilen)
        ├──► Read messages_summary.md   (kompakt: ~10 Zeilen)
        ├──► Read memory_summary.md     (kompakt: ~10 Zeilen)
        └──► Read notion_summary.md     (kompakt: ~15 Zeilen)
        │
        └──► Generiert Tagesplan + Abgleich-Fragen
             (inkl. User-Additions aus Notion)

Phase 4: USER INTERACTION
    Hauptagent ◄──► User
        │
        └──► Beantwortet Abgleich-Fragen
        └──► Bestätigt oder korrigiert

Phase 4.5: MATTHIAS COMPOSER (Optional, nach User-Bestätigung)
    ┌────────────────────────────────────────────────────────────────┐
    │ MATTHIAS_COMPOSER                                              │
    │                                                                │
    │ Input:                                                         │
    │   - calendar_summary.md (Events ohne Wer, Matthias-Termine)    │
    │   - todoist_summary.md (Gemeinsam/Allgemein Tasks)             │
    │   - messages_summary.md (Kommunikations-Kontext)  ← NEU        │
    │   - memory_summary.md (Patterns mit Matthias)     ← NEU        │
    │   - User-Bestätigung was kommuniziert werden soll              │
    │                                                                │
    │ Output:                                                        │
    │   - matthias_message.md (fertige Nachricht)                    │
    └────────────────────────────────────────────────────────────────┘

Phase 5: EXECUTION (Hauptagent, MCP calls)
    Hauptagent
        │
        ├──► mcp__todoist__complete-tasks
        ├──► mcp__todoist__update-tasks
        ├──► mcp__mcp-ical__create_event
        ├──► mcp__openmemory-local__add_memories
        ├──► mcp__notionMCP__notion-create-pages (neues Tages-Doc)
        └──► mcp__messages__tool_send_message (Matthias, wenn bestätigt)
```

---

## Directory Structure

```
/tmp/axis-tagesstart-2025-12-14/
├── raw/
│   ├── todoist.json          # Rohdaten von MCP (nie vom Hauptagent gelesen)
│   ├── calendar.json
│   ├── messages.json
│   ├── memory.json
│   ├── notion.json           # Docs der letzten 2 Tage
│   └── data_version.json     # ← NEU: Timestamps für Race Condition Detection
├── summaries/
│   ├── todoist.md            # Komprimiert, vom Hauptagent gelesen
│   ├── calendar.md
│   ├── messages.md
│   ├── memory.md
│   └── notion.md             # User-Inputs, Erledigtes, Offenes
├── output/
│   ├── tagesplan.md          # Finaler Output
│   ├── matthias_message.md   # Nachricht für Matthias (wenn relevant)
│   └── execution_log.md      # Was wurde ausgeführt
├── user_confirmation.txt     # User-Antworten auf Abgleich-Fragen
└── meta/
    ├── status.json           # Pipeline-Status
    ├── errors.json           # Fehler-Log
    └── cache/                # ← NEU: Granulare Cache-Timestamps
        ├── todoist.txt       # 5 min TTL
        ├── calendar.txt      # 1h TTL
        ├── messages.txt      # 10 min TTL
        ├── memory.txt        # 1 Tag TTL
        └── notion.txt        # 2h TTL
```

### Archive (nach Completion)
```
/tmp/axis-archive/2025-12-14/   # ← NEU: 24h aufbewahren statt sofort löschen
└── [komplette Struktur von oben]
```

---

## Granulare Cache-Strategie (Review-Fix)

| Source | TTL | Begründung |
|--------|-----|------------|
| Todoist | 5 min | Tasks ändern sich oft (completions, neue Tasks) |
| Messages | 10 min | Kommunikation ist schnell |
| Calendar | 1h | Events sind stabiler |
| Notion | 2h | Manuelle Docs ändern sich langsam |
| Memory | 1 Tag | Memories ändern sich sehr selten |

**Cache-Check vor Phase 1:**
```python
for source in sources:
    cache_file = f"{base_path}/meta/cache/{source}.txt"
    if exists(cache_file):
        cached_at = read(cache_file)
        if now() - cached_at < TTL[source]:
            skip_fetch(source)
```

---

## Data Versioning (Review-Fix: Race Conditions)

**Problem:** Zwischen Phase 1 (Daten holen) und Phase 5 (Execution) können sich Daten ändern.

**Lösung:** `data_version.json` speichert Timestamps:

```json
{
  "fetched_at": "2025-12-14T08:00:00Z",
  "todoist_tasks": {
    "task_abc123": "2025-12-14T07:45:00Z",
    "task_def456": "2025-12-14T06:30:00Z"
  },
  "calendar_events": {
    "event_xyz": "2025-12-14T07:00:00Z"
  }
}
```

**Vor Execution in Phase 5:**
```python
def safe_complete_task(task_id):
    stored_version = data_version["todoist_tasks"][task_id]
    current_task = mcp_todoist_get_task(task_id)

    if current_task.updated_at > stored_version:
        warn_user(f"Task '{task_id}' wurde seit Planung geändert!")
        return SKIP  # oder User fragen

    return mcp_todoist_complete(task_id)
```

---

## Summary Formats (was Subagents produzieren)

### todoist.md
```markdown
## P1/P2 Überfällig
- [ID] Task-Name (Projekt, seit X Tagen)

## Heute
- [ID] Task-Name (Projekt, Priorität)

## Diese Woche
- Mo: X Tasks
- Di: Y Tasks
...

## Flags
- overdue_count: 5
- today_count: 3
- blocked_tasks: [IDs wenn vorhanden]
```

### calendar.md
```markdown
## Heute (Sa 14.12.)
- 10:00-11:00 Event-Name [Kalender]
- 14:00-15:00 Event-Name [Kalender] ⚠️ MISSING_WER

## Morgen
...

## Kapazität
- heute: 3h gebucht, 5h frei
- morgen: 6h gebucht, 2h frei

## Flags
- missing_wer_events: [IDs]
- conflicts: []
```

### messages.md
```markdown
## Relevante Threads (letzte 48h)
- Matthias: Letzter Kontakt vor 2h, Thema: Park-Planung
- [Name]: Wartet auf Antwort seit 1d

## Flags
- pending_responses: [Kontakte]
- mentioned_in_tasks: [Kontakte die auch in Todoist vorkommen]
```

### memory.md
```markdown
## Relevante Memories
- [Pattern] Kritik-Trigger bei BMG (Circuit 2, emotional)
- [Context] Matthias arbeitet remote diese Woche

## Aktive Loops
- Video-Prokrastination (seit 3 Einträgen)

## Flags
- active_patterns: [IDs]
- contradictions: []
```

### notion.md
```markdown
## Notion Sync (letzte 2 Tage)

### Gestern (13.12.)
- ✓ Erledigt: BMG Video 1, Einkaufen
- ✗ Verworfen: Gym (durchgestrichen)
- ➕ User-Input: "Idee: Newsletter-Format überdenken"
- ○ Noch offen: BMG Video 2, Steuer-Unterlagen

### Vorgestern (12.12.)
- ✓ Erledigt: Call mit Lisa, Rechnungen
- ➕ User-Input: "Matthias fragen wegen Urlaub"

### Für heute übernehmen
- [ ] "Idee: Newsletter-Format überdenken" (User von gestern)
- [ ] "Matthias fragen wegen Urlaub" (User von vorgestern)
- [ ] BMG Video 2 (offen von gestern)
- [ ] Steuer-Unterlagen (offen von gestern)

## Flags
- user_inputs_count: 2
- still_open_count: 2
- completion_rate_yesterday: 66%
```

---

## Subagent Prompts

### TODOIST_ANALYZER

```
Du bist der TODOIST_ANALYZER.

INPUT: /tmp/axis-tagesstart-{date}/raw/todoist.json
OUTPUT: /tmp/axis-tagesstart-{date}/summaries/todoist.md

REGELN:
1. Lies die Rohdaten mit Read tool
2. Analysiere nach diesem Schema:
   - P1/P2 Überfällig: ALLE einzeln auflisten (niemals aggregieren)
   - Andere Überfällig: Gruppiert nach Projekt
   - Heute: Alle Tasks für heute
   - Diese Woche: Tagesübersicht (Anzahl pro Tag)
3. Projekt-Kontext:
   - Inbox = Grischas privat
   - Gemeinsam = Zweischritte (Matthias sieht)
   - Allgemein = Haushalt (Matthias sieht)
4. Schreib das Summary im vorgegebenen Format
5. Halte es KOMPAKT: Max 30 Zeilen

NIEMALS:
- MCP-Tools aufrufen (hast du nicht)
- Rohdaten in den Output kopieren
- Mehr als 30 Zeilen schreiben
```

### CALENDAR_ANALYZER

```
Du bist der CALENDAR_ANALYZER.

INPUT: /tmp/axis-tagesstart-{date}/raw/calendar.json
OUTPUT: /tmp/axis-tagesstart-{date}/summaries/calendar.md

REGELN:
1. Lies die Rohdaten mit Read tool
2. Prüfe bei "Gemeinsamer Kalender" Events:
   - Hat Notes-Feld "Wer: G" / "Wer: M" / "Wer: G+M"?
   - Wenn nicht: Markiere mit ⚠️ MISSING_WER
3. Berechne Kapazität pro Tag:
   - Summe gebuchter Stunden
   - Freie Zeit (annahme: 8h Arbeitstag)
4. Heute + nächste 7 Tage detailliert
5. Halte es KOMPAKT: Max 25 Zeilen

KALENDER-ZUORDNUNG:
- "Grischa" → Immer Grischa
- "Gemeinsamer Kalender" → Wer: Feld prüfen!
- "Charlie Schule Kurse" → Charlie
- "gdr@dasburo.com" → Immer Grischa
```

### MESSAGES_ANALYZER

```
Du bist der MESSAGES_ANALYZER.

INPUT: /tmp/axis-tagesstart-{date}/raw/messages.json
OUTPUT: /tmp/axis-tagesstart-{date}/summaries/messages.md

REGELN:
1. Lies die Rohdaten mit Read tool
2. Identifiziere:
   - Offene Threads (wartet auf Antwort)
   - Kürzliche Kommunikation mit relevanten Personen
   - Erwähnungen die zu Tasks passen könnten
3. Cross-Reference mit todoist.json wenn vorhanden:
   - Gibt es Tasks "X kontaktieren" wo Messages zeigen dass Kontakt war?
4. Halte es KOMPAKT: Max 15 Zeilen

FOKUS:
- Matthias (Partner, Zweischritte)
- Personen die in Tasks erwähnt werden
```

### MEMORY_ANALYZER

```
Du bist der MEMORY_ANALYZER.

INPUT: /tmp/axis-tagesstart-{date}/raw/memory.json
OUTPUT: /tmp/axis-tagesstart-{date}/summaries/memory.md

REGELN:
1. Lies die Rohdaten mit Read tool
2. Kategorisiere nach:
   - Aktive Patterns (wiederkehrende Muster)
   - Relevanter Kontext (Personen, Projekte)
   - Say-Want-Do Gaps (Widersprüche)
3. Priorisiere nach:
   - Circuit (höher = relevanter für heute)
   - Recency (neuer = relevanter)
4. Halte es KOMPAKT: Max 15 Zeilen

VAULT-BEDEUTUNG:
- SOV: Identität
- WLT: Business
- SIG: Pattern
- FRC: Health/Triggers
- DIR: System
- FGP: Evolution
```

### NOTION_ANALYZER

```
Du bist der NOTION_ANALYZER.

INPUT:
  - /tmp/axis-tagesstart-{date}/raw/notion.json (Tagesplan-Docs der letzten 2 Tage)
  - /tmp/axis-tagesstart-{date}/raw/todoist.json (für Diff-Vergleich)    ← NEU
  - /tmp/axis-tagesstart-{date}/raw/calendar.json (für Diff-Vergleich)  ← NEU
OUTPUT: /tmp/axis-tagesstart-{date}/summaries/notion.md

REGELN:
1. Lies ALLE Input-Dateien mit Read tool
2. Analysiere für JEDEN Tag (gestern, vorgestern):

   A) ABGEHAKT (✓ oder durchgestrichen):
      → Liste als "Erledigt" (nicht erneut in Tagesplan)

   B) GELÖSCHT (war in Notion, jetzt weg):
      → Vergleiche mit todoist.json/calendar.json
      → Wenn Item dort noch existiert: User hat es bewusst entfernt → "Verworfen"
      → Wenn Item dort auch weg: War erledigt → "Erledigt"

   C) HINZUGEFÜGT (nicht von AXIS, vom User):
      → Vergleiche: Ist es in todoist.json oder calendar.json?
      → Wenn NEIN: Echter User-Input → HÖCHSTE Priorität!
      → Wenn JA: War schon da, nur formatiert

   D) UNVERÄNDERT OFFEN:
      → Liste als "Noch offen" (in heutigen Plan übernehmen)

3. Erkenne User-Additions durch:
   - Items die NICHT in todoist.json oder calendar.json vorkommen
   - Freiformtext ohne Task-ID oder Event-ID
   - Notizen, Ideen, Fragen

4. Halte es KOMPAKT: Max 20 Zeilen

OUTPUT-FORMAT:
```markdown
## Notion Sync (letzte 2 Tage)

### Gestern (13.12.)
- ✓ Erledigt: Task A, Task B
- ✗ Verworfen: Task C
- ➕ User-Input: "Neue Idee X", "Call mit Y"
- ○ Noch offen: Task D

### Vorgestern (12.12.)
- ✓ Erledigt: Task E
- ➕ User-Input: "Notiz Z"

### Für heute übernehmen
- [ ] "Neue Idee X" (User-Input von gestern)
- [ ] "Call mit Y" (User-Input von gestern)
- [ ] Task D (noch offen von gestern)
```

WICHTIG:
- User-Inputs haben HÖCHSTE Priorität für Übernahme
- Erledigte Items NICHT wieder vorschlagen
- Verworfene Items NICHT wieder vorschlagen
```

### MATTHIAS_COMPOSER

```
Du bist der MATTHIAS_COMPOSER.

INPUT:
  - /tmp/axis-tagesstart-{date}/summaries/calendar.md
  - /tmp/axis-tagesstart-{date}/summaries/todoist.md
  - /tmp/axis-tagesstart-{date}/summaries/messages.md   ← NEU (Kommunikations-Kontext)
  - /tmp/axis-tagesstart-{date}/summaries/memory.md     ← NEU (Patterns mit Matthias)
  - /tmp/axis-tagesstart-{date}/user_confirmation.txt
OUTPUT: /tmp/axis-tagesstart-{date}/output/matthias_message.md

KONTEXT:
- Matthias ist SOURCEs Partner (Zweischritte-Arbeit)
- Matthias sieht: Projekte "Gemeinsam" und "Allgemein"
- "Gemeinsamer Kalender" Events brauchen Wer-Zuordnung

REGELN:
1. Lies ALLE Input-Dateien (alle 5!)
2. Sammle relevante Punkte:
   - Events im "Gemeinsamer Kalender" ohne Wer-Zuordnung
   - Überfällige Tasks in Gemeinsam/Allgemein
   - Tasks diese Woche die Matthias betreffen
   - Termine die Abstimmung brauchen
   - Offene Threads aus messages.md (wartet Matthias auf Antwort?)
   - Patterns aus memory.md (gibt es bekannte Themen?)
3. Formuliere als natürliche Nachricht:
   - Deutsch, du-Form
   - Freundlich aber sachlich
   - Nicht zu lang (max 10 Sätze)
   - Berücksichtige letzten Kommunikations-Kontext!
4. Strukturiere klar:
   - Was braucht Antwort/Entscheidung?
   - Was ist nur Info?

OUTPUT-FORMAT:
```markdown
## Nachricht an Matthias

**Vorschlag:**

Hey! Kurzer Sync für heute/diese Woche:

[Inhalt]

---
**Channels:** iMessage / Signal / Slack
**Senden?** [Warte auf User-Bestätigung]
```

NIEMALS:
- Sachen erfinden die nicht in den Inputs stehen
- Zu formell oder zu casual sein
- Mehr als 10 Sätze
```

---

## Hauptagent Flow (Pseudocode)

```python
# Cache TTLs (Review-Fix: Granular statt 2h für alles)
CACHE_TTL = {
    "todoist": timedelta(minutes=5),
    "messages": timedelta(minutes=10),
    "calendar": timedelta(hours=1),
    "notion": timedelta(hours=2),
    "memory": timedelta(days=1)
}

def tagesstart(user_input):
    date = today()
    base_path = f"/tmp/axis-tagesstart-{date}"

    # Phase 0: Setup
    create_directories(base_path)

    # Phase 1: Data Collection (mit granularem Cache-Check)
    sources_to_fetch = []
    for source in ["todoist", "calendar", "messages", "memory", "notion"]:
        if not cache_valid(source, CACHE_TTL[source]) or user_wants_refresh:
            sources_to_fetch.append(source)

    if sources_to_fetch:
        # Nur veraltete Sources fetchen
        fetch_calls = []
        if "todoist" in sources_to_fetch:
            fetch_calls.append(mcp_todoist_to_file(f"{base_path}/raw/todoist.json"))
        if "calendar" in sources_to_fetch:
            fetch_calls.append(mcp_calendar_to_file(f"{base_path}/raw/calendar.json"))
        if "messages" in sources_to_fetch:
            fetch_calls.append(mcp_messages_to_file(f"{base_path}/raw/messages.json"))
        if "memory" in sources_to_fetch:
            fetch_calls.append(mcp_memory_to_file(f"{base_path}/raw/memory.json", query=user_input))
        if "notion" in sources_to_fetch:
            fetch_calls.append(mcp_notion_to_file(f"{base_path}/raw/notion.json", last_2_days=True))

        parallel(fetch_calls)

        # Data Versioning speichern (Review-Fix: Race Conditions)
        save_data_versions(base_path)
        update_cache_timestamps(sources_to_fetch)

    # Phase 2: Analysis (parallel subagents)
    results = parallel([
        Task(TODOIST_ANALYZER, base_path),
        Task(CALENDAR_ANALYZER, base_path),
        Task(MESSAGES_ANALYZER, base_path),
        Task(MEMORY_ANALYZER, base_path),
        Task(NOTION_ANALYZER, base_path)
    ])

    # Review-Fix: Partial Failure Handling
    failed_analyzers = [r for r in results if r.status != "completed"]
    if failed_analyzers:
        warn_user(f"Analyse unvollständig: {failed_analyzers}")

    # Phase 3: Synthesis (read summaries only - ~100 Zeilen total)
    summaries = {}
    for name in ["todoist", "calendar", "messages", "memory", "notion"]:
        summary_file = f"{base_path}/summaries/{name}.md"
        if exists(summary_file):
            summaries[name] = read(summary_file)
        else:
            summaries[name] = f"[{name} nicht verfügbar]"

    tagesplan = generate_tagesplan(summaries, user_input)
    abgleich_questions = generate_questions(summaries)

    # Phase 4: User Interaction
    present(tagesplan, abgleich_questions)
    user_responses = await_user()
    write(f"{base_path}/user_confirmation.txt", user_responses)

    # Phase 4.5: Matthias Composer (optional)
    if matthias_relevant(summaries, user_responses):
        Task(MATTHIAS_COMPOSER, base_path)
        matthias_msg = read(f"{base_path}/output/matthias_message.md")
        if user_confirms_send(matthias_msg):
            mcp_send_message("Matthias", matthias_msg)

    # Phase 5: Execution (mit Data Version Check)
    data_versions = load_data_versions(base_path)

    for action in user_responses.actions:
        if action.type == "complete_task":
            # Review-Fix: Race Condition Detection
            if task_modified_since(action.task_id, data_versions):
                if not user_confirms(f"Task wurde geändert. Trotzdem abschließen?"):
                    continue
            safe_complete_task(action.task_id)

        elif action.type == "create_event":
            safe_create_event(action.event)

        elif action.type == "add_memory":
            safe_add_memory(action.memory)

    create_notion_doc(tagesplan, date)

    # Phase 6: Archive (Review-Fix: 24h aufbewahren statt sofort löschen)
    archive_path = f"/tmp/axis-archive/{date}"
    move(base_path, archive_path)
    schedule_cleanup(archive_path, delay=timedelta(hours=24))
```

---

## Kontext-Budget Vergleich

| Phase | Alte Architektur | Staged Context |
|-------|------------------|----------------|
| Data Collection | ~5000 tokens (alle Rohdaten) | ~250 tokens (nur MCP calls + Write) |
| Analysis | 0 (keine Subagents) | ~250 tokens (5 Subagent launches) |
| Synthesis | bereits überflutet | ~500 tokens (5 Summaries à ~100) |
| Matthias | in Hauptagent | ~100 tokens (1 Subagent + Read) |
| **Total** | **~5000+ tokens** | **~1100 tokens** |

**Reduktion: ~78%**

### Wo die Arbeit passiert

| Komponente | Kontext-Last | Wer trägt sie |
|------------|--------------|---------------|
| Rohdaten parsen | Hoch (~1000 tokens pro Source) | Subagents (isoliert) |
| Summaries schreiben | Mittel | Subagents (isoliert) |
| Summaries lesen | Niedrig (~100 pro Summary) | Hauptagent |
| Tagesplan generieren | Niedrig (nur Summaries) | Hauptagent |
| MCP Execution | Niedrig (strukturierte Calls) | Hauptagent |

---

## Error Handling

### Subagent Timeout
```json
// meta/status.json
{
  "todoist_analyzer": "completed",
  "calendar_analyzer": "timeout",
  "messages_analyzer": "completed",
  "memory_analyzer": "completed"
}
```
→ Hauptagent kann ohne calendar_summary.md fortfahren, markiert als "Calendar-Daten nicht verfügbar"

### MCP Call Failure
```json
// meta/errors.json
{
  "phase": "data_collection",
  "tool": "mcp__mcp-ical__list_events",
  "error": "Connection refused",
  "fallback": "Skip calendar analysis"
}
```
→ Hauptagent erstellt leere calendar.json, Subagent schreibt "Keine Daten verfügbar"

### Malformed Summary
Wenn Subagent kein valides Markdown schreibt:
→ Hauptagent nutzt Fallback-Template mit "Analyse fehlgeschlagen"

---

## Migration von v4 zu v5

1. **Protokoll-Datei**: `axis_tagesstart_v4.md` → `axis_tagesstart_v5.md`
2. **Subagent-Dateien**: Neue Dateien für Analyzer-Prompts
3. **CLAUDE.md**: Update der Referenz auf v5
4. **Test**: Einmal komplett durchlaufen mit Debug-Output

---

## Entschiedene Fragen

1. **Persistenz**: ✓ Archive 24h, dann löschen
2. **Caching**: ✓ Granular pro Source (5min - 1 Tag)
3. **Notion-Sync**: ✓ Ein Doc pro Tag, NOTION_ANALYZER liest letzte 2 Tage + Todoist/Calendar für Diff
4. **Matthias-Message**: ✓ Separater MATTHIAS_COMPOSER mit vollem Kontext (alle 4 Summaries)

---

## Review-Findings (eingearbeitet)

| Problem | Schwere | Lösung | Status |
|---------|---------|--------|--------|
| Race Condition Phase 1→5 | 🔴 Kritisch | Data Versioning mit Timestamps | ✓ Eingearbeitet |
| 2h Cache zu grob | 🟠 Hoch | Granular: 5min-1Tag je Source | ✓ Eingearbeitet |
| Notion Diff ohne Context | 🟠 Hoch | Notion Analyzer liest auch todoist/calendar.json | ✓ Eingearbeitet |
| Matthias fehlt Context | 🟡 Mittel | messages.md + memory.md als Input | ✓ Eingearbeitet |
| Cleanup destruktiv | 🟡 Mittel | Archive 24h statt sofort löschen | ✓ Eingearbeitet |
| Partial Failure Handling | 🟡 Mittel | Warnung wenn Analyzer fehlschlägt | ✓ Eingearbeitet |

### Nicht eingearbeitet (bewusst)

| Problem | Grund für Verzicht |
|---------|-------------------|
| Schema Validation für Summaries | Overhead zu hoch für v1, später nachrüsten |
| Transaction Rollback in Phase 5 | Komplexität, MCP-Calls sind meist idempotent |
| Progress UI während Subagents | Nice-to-have, nicht kritisch |

---

## Next Steps

1. [x] Plan reviewen mit SOURCE
2. [x] Entscheidung zu offenen Fragen
3. [x] Review durch Subagent
4. [x] Review-Findings einarbeiten
5. [ ] v5 Protokoll schreiben
6. [ ] Subagent-Prompts finalisieren
7. [ ] Test-Run mit Debug-Output
8. [ ] Iteration basierend auf Ergebnissen
