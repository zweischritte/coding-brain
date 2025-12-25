# Calendar Protocol

## Glyph

```
📅 = Load calendar context
```

---

## Relevante Kalender

**Nur diese Kalender beachten:**

| Kalender | Bedeutung |
|----------|-----------|
| **Grischa** | Persönliche Termine – was ich alleine mache |
| **Gemeinsamer Kalender** | Gemeinsame Termine mit Matthias oder nur Matthias |
| **Charlie Schule Kurse** | Charlies Schule oder außerschulische Aktivitäten |
| **gdr@dasburo.com** | Job bei Büro am Draht / Cloudfactory |

**Alle anderen Kalender ignorieren** – nur diese vier sind relevant für Planung und Verfügbarkeit.

---

## Verantwortlichkeit (Wer)

| Kalender | Wer | Notes-Feld |
|----------|-----|------------|
| **Grischa** | Immer Grischa | Nicht nötig |
| **Gemeinsamer Kalender** | Variiert | `Wer: Grischa` / `Wer: Matthias` / `Wer: Grischa+Matthias` |
| **Charlie Schule Kurse** | Charlies Termine | Nicht nötig (Begleitung implizit) |
| **gdr@dasburo.com** | Immer Grischa (Lohnarbeit) | Nicht nötig |

**Regel:** Nur bei "Gemeinsamer Kalender" muss `Wer:` in Notes eingetragen werden. Bei neuen Events dort immer fragen wer zuständig ist.

---

## When to Read

| Trigger | Action |
|---------|--------|
| "Hab ich Zeit für..." / availability question | `list_events` for relevant timeframe |
| "Diese Woche" / "Morgen" / time reference | `list_events` + Todoist parallel |
| Person + time context | "Wann hab ich X getroffen?" → search calendar |
| Capacity question | Calendar + Todoist = Reality Check |
| Planning/Scheduling | Always read calendar before commitment |
| Temporal Layer active | Reconstruction: What happened when? |

**Tools:**
- `list_events(start_date, end_date)` — query timeframe
- `list_calendars()` — see available calendars

**NOT during:** Emotional depth work, shadow integration (unless temporal context is relevant)

---

## When to Write

| Trigger | Action |
|---------|--------|
| Concrete appointment emerges | `create_event` |
| Todoist task with fixed time window | → create calendar event |
| External commitment (meeting, call) | Calendar = binding |
| Deadline with time block | Event for focused work |

**Tools:**
- `create_event(title, start_time, end_time, ...)` — create event
- `update_event(event_id, ...)` — modify event

---

## Calendar vs Todoist vs Both

| Content | Destination |
|---------|-------------|
| Fixed appointment with time | Calendar |
| Task without fixed time | Todoist |
| Deadline + work block | Todoist (deadline) + Calendar (time block) |
| Meeting/Call | Calendar |
| "Sometime this week" | Todoist with due date |

---

## Capacity Reality Check

Before every new commitment:

```
1. Calendar: What's already booked?
2. Todoist: What's due/overdue?
3. Memory: Any capacity patterns? (overload, etc.)
4. Decision: Does this realistically fit?
```

**Pattern Signal:**
> Many appointments + many overdue tasks = 🛡️ "Der Körper hat nur 24h. Was hat echte Priorität?"

---

## Temporal Layer Integration

Calendar is primary source for Temporal Layer:

| Query | Method |
|-------|--------|
| "Wann war das Meeting mit X?" | `list_events` + search |
| "Letzte Woche..." | Calendar as memory aid |
| Track evolution | Calendar events = timeline markers |

**Memory Link:**
When calendar event reveals pattern → store in Memory:
```python
add_memories(
    text="Meeting mit X war energieraubend",
    vault="WLT", layer="relational",
    entity="X",
    tags={"energy": -3, "person": True}
)
```

---

## Cross-Tool Patterns

| Pattern | Tools | Insight |
|---------|-------|---------|
| Overbooking | Calendar full + Todoist overdue | Capacity warning |
| Person frequency | Calendar meetings with X | Who gets time? |
| Appointment avoidance | Task "X treffen" but never in Calendar | Gap: Want ≠ Do |
| Reality Fingerprint | Memory + Calendar + Todoist | Complete commitment picture |

---

## Reminder/Alarm Logic

| Event Type | Suggested Alarms |
|------------|------------------|
| Important meeting | 1 day + 1 hour before |
| Deadline | 1 day before |
| Casual appointment | 1 hour before |
| All-day event | Morning of |

---

## Auto-Detection Extensions

| Pattern | Detection | Response |
|---------|-----------|----------|
| Calendar overload | >5 events/day repeatedly | "Dein Kalender ist voll. Wo ist Luft?" |
| No buffer time | Back-to-back meetings | "Keine Pausen zwischen Terminen. Intentional?" |
| Weekend work | Events on Sat/Sun | "Arbeit am Wochenende. Notwendig oder Pattern?" |
| Person dominates calendar | >30% time with one person | "X bekommt viel Zeit. Entspricht das der Priorität?" |
