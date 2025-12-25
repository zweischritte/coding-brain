# Todoist Protocol

## Project Logic

| Project | Content | Shared with Matthias |
|---------|---------|----------------------|
| **Gemeinsam** | Zweischritte work | ✓ |
| **Allgemein** | Household, family, daily life | ✓ |
| **Inbox** | SOURCE's personal items | ✗ |

---

## When to Read

- **Planning/capacity questions:** Today + overdue
- **Project context:** Overview of relevant section
- **NOT during:** Emotional depth work, shadow integration

**Tools:**
- `find-tasks-by-date` for today + overdue
- `get-overview` for project-specific context

---

## When to Write

- Zweischritte work → Gemeinsam (+ matching section)
- Household/family/daily → Allgemein
- Personal → Inbox

**⚠️ Matthias sees Gemeinsam + Allgemein** — phrase accordingly

---

## Todoist vs Memory vs Both

| Condition | Action |
|-----------|--------|
| Concrete action + clear goal | → Todoist only |
| Pattern, emotion, insight, relational dynamic | → Memory only |
| Action + AXIS-relevant context (resistance, pattern, strategic) | → Both |

### Examples

**Todoist only:**
- "BMG-Feedback bis Freitag einarbeiten"
- "Leica zur Reparatur bringen"
- "Steuerberater anrufen"

**Memory only:**
- "Kritik triggert bei BMG-Projekt"
- "Matthias gibt Energie"
- "90% Pattern bei Projekten"

**Both:**
- Todoist: "eljuego.community fertigstellen"
- Memory: `add_memories(text="90% Pattern bei eljuego", vault="FGP", layer="goals", entity="eljuego", tags={"abandoned": True, "project": True})`

---

## Pattern Signals

| Signal | Response |
|--------|----------|
| >5 overdue | "Aktueller Druck. Was hat echte Priorität?" |
| Old tasks without date | "Lebt das noch?" |
| Section tasks all stagnant | Check project blocker |
| Same task rescheduled 3x+ | Pattern detection: "Was blockiert hier wirklich?" |

---

## Section Structure (Gemeinsam)

Sections follow project-based organization:
- Each client/project gets own section
- Thematic grouping, not time-based
- Archive completed projects, don't delete

---

## Task Phrasing

**Good task:**
- Starts with verb
- Clear outcome
- Reasonable scope

**Examples:**
- ✓ "BMG-Startseite Feedback einarbeiten"
- ✗ "BMG" (too vague)
- ✓ "Matthias Re: kitev Radio-Konzept schicken"
- ✗ "kitev besprechen" (no clear action)

---

## Priority Mapping

| Todoist Priority | AXIS Meaning |
|------------------|--------------|
| p1 (red) | Actual deadline, external commitment |
| p2 (orange) | Important, this week |
| p3 (blue) | Would be good, flexible |
| p4 (none) | Someday/maybe, capture only |

---

## Integration with Memory

When task reveals pattern:
1. Complete/update in Todoist
2. Store insight in Memory

When memory reveals needed action:
1. Store pattern in Memory
2. Create concrete task in Todoist
