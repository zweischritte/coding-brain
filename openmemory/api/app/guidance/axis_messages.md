# Messages Protocol

## Glyph

```
📱 = Load messages context
```

---

## When to Read

| Trigger | Action |
|---------|--------|
| Person mentioned + communication context | `get_recent_messages` with contact filter |
| "Was hab ich X geschrieben?" | Verify Say vs Do |
| Relational Layer + uncertainty | Recent messages = ground truth |
| Check relationship pattern | Message history as evidence |
| "Hat X geantwortet?" | Check recent messages |
| Communication gap suspected | When was last contact? |

**Tools:**
- `tool_get_recent_messages(hours, contact)` — recent messages, optionally filtered
- `tool_find_contact(name)` — find contact by name
- `tool_fuzzy_search_messages(search_term, hours)` — search message content

**NOT during:** Deep shadow work (unless relational context directly relevant)

---

## When to Write

| Trigger | Action |
|---------|--------|
| Explicit instruction | "Schreib X dass..." |
| From Todoist task | Task = "X anschreiben" → send message |
| Follow-up after meeting | Calendar event passed → message reminder |
| Commitment communication | Agreement needs to be communicated |

**Tools:**
- `tool_send_message(recipient, message)` — send message
- `tool_check_imessage_availability(recipient)` — check if iMessage available

**⚠️ Always confirm before sending** — Messages are irreversible

---

## Messages vs Memory vs Both

| Content | Destination |
|---------|-------------|
| Communication action needed | Messages (send) |
| Pattern about communication style | Memory only |
| What was actually said | Messages (read) → verify Memory |
| Relational insight from conversation | Memory (store pattern) |
| Commitment made via message | Both: Messages (proof) + Memory (pattern) |

---

## Say-Want-Do Triangulation

Messages provide hard data for **SAY** vector:

| Vector | Source |
|--------|--------|
| SAY | Messages = What was actually communicated |
| WANT | Memory = What patterns reveal |
| DO | Todoist/Calendar = What actions confirm |

**Gap Detection:**
```python
# If Memory says "Will X kontaktieren" but Messages show no recent contact
add_memories(
    text="Sagt will X kontaktieren aber keine Message seit 2 Wochen",
    vault="FGP", layer="relational",
    entity="X",
    tags={"gap": "say_do", "person": True}
)
```

---

## Relational Energy Tracking

Messages reveal communication patterns:

| Pattern | Signal | Memory Tag |
|---------|--------|------------|
| High frequency with X | Many messages/day | `{energy: +N}` or `{energy: -N}` based on content |
| One-sided communication | SOURCE always initiates | `{pattern: "einseitig"}` |
| Delayed responses | Days between messages | Check relational health |
| Short vs long messages | Engagement level | Context for relational layer |

**Memory Link:**
```python
add_memories(
    text="Kommunikation mit X ist einseitig - SOURCE initiiert immer",
    vault="WLT", layer="relational",
    entity="X",
    tags={"person": True, "pattern": True}
)
```

---

## Cross-Tool Patterns

| Pattern | Tools | Insight |
|---------|-------|---------|
| Commitment verification | Todoist task + Message sent? | Was it actually communicated? |
| Meeting follow-up | Calendar event + Message after? | Professional follow-through |
| Relationship maintenance | Memory (energy) + Messages (frequency) | Investment matches importance? |
| Reality Fingerprint | Memory + Calendar + Messages | What's thought, planned, communicated |

---

## Privacy Considerations

**Read with intention:**
- Only load messages when relational context is directly relevant
- Don't surface message content unnecessarily
- Focus on patterns, not surveillance

**Write with caution:**
- Always confirm before sending
- Consider tone and timing
- Messages represent SOURCE externally

---

## Auto-Detection Extensions

| Pattern | Detection | Response |
|---------|-----------|----------|
| Communication debt | Task "X anschreiben" overdue + no recent messages | "Du wolltest X schreiben. Noch relevant?" |
| Response pending | Sent message, no reply, days passed | "Keine Antwort von X. Follow-up oder loslassen?" |
| Over-communication | >20 messages/day to one person | "Viel Kommunikation mit X. Energie-Check?" |
| Avoidance pattern | Person in Memory with negative energy + no messages | "Kein Kontakt mit X seit [Zeit]. Intentional?" |

---

## Group Chats

**Tools:**
- `tool_get_chats()` — list available group chats
- `tool_send_message(recipient, message, group_chat=True)` — send to group

**Considerations:**
- Group context is different from 1:1
- More people = more careful phrasing
- Group dynamics can reveal relational patterns
