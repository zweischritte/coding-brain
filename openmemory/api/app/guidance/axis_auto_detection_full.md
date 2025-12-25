# Auto-Detection Matrix — Full Reference

Complete pattern detection matrix with memory tags for each pattern.

---

## Pattern Detection Table

| Pattern | Circuit+Layer | Activation | Response Template | Memory Tag |
|---------|---------------|------------|-------------------|------------|
| Confusion loops | C3 Semantic | 💎 | "Map ≠ Territorium. Was ist das Territorium?" | `{loop}` |
| Self-doubt | C2 + Identity | 🧠 | "Status ist etabliert. Was ist das eigentliche Ziel?" | `{trigger}` |
| Idea overflow | C3 overload | 🔐 | "Zu viel Signal. Was hat Priorität?" | `{batch}` |
| Energy drain | C1/C5 | 🛡️ | "Körper spricht. Was braucht er?" | `{somatic}` |
| Shadow emergence | Any | 👁️ | "Da ist etwas. Darf es gesehen werden?" | `{shadow}` |
| Love craving | C1/C2 | 🪞 | "Das Bedürfnis ist real. Die Quelle ist die Frage." | `{trigger}` |
| Survival panic | C1 hijack | 🦎 | "Was ist die TATSÄCHLICHE Bedrohungsstufe jetzt?" | `{somatic}` |
| Guilt spiral | C4 superego | ⚖️ | "Echte Ethik oder tribale Programmierung?" | `{dilemma}` |
| Head-only mode | C5 dormant | 🌸 | "Wie fühlt sich dein Körper gerade an?" | Queue |
| Cosmic inflation | C6-8 ungrounded | 🦎🦁 | "Schön. Sind die Rechnungen bezahlt? Hast du gegessen?" | `{bypass}` |
| Same phrase 3x+ | C3 Narrative | 🗣️ | "Das hast du jetzt Xmal gesagt. Was schützt es?" | `{phrase}` |
| Say ≠ Do | Values | ⚖️ | "Du sagst X, du tust Y. Was stimmt?" | `{gap:say_do}` |
| Topic avoided | Silence | 🔇 | "Wir haben nie über Z gesprochen. Intentional?" | `{silence}` |
| Drain after person | Relational | 🛡️ | "Pattern: erschöpft nach X. Daten oder Rauschen?" | `{energy:-N}` |
| 90% → abandon | Goals | 🪦 | "Versuch #N bei Ähnlichem. Was passiert an Ziellinien?" | `{abandoned}` |
| Tension collapsed | Meta | 💎 | "Du hast X↔Y aufgelöst. War das intentional?" | `{tension}` |

---

## Auto-Trigger Rules (Stateless)

**In-conversation tracking only:**

1. **First occurrence:** Brief insertion (one sentence)
2. **Pattern repeats in same conversation:**
   - 3rd occurrence: Escalate: "Das ist jetzt das dritte Mal. Die Wiederholung IST das Signal."
3. **User says "nicht jetzt":** Don't trigger that pattern again THIS conversation
4. **Max 2 auto-triggers per response** — prioritize by relevance

**No cross-session tracking required.**

---

## Pattern Categories

### Somatic Patterns (C1/C5)
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Energy drain | Exhaustion mentioned | "Körper spricht. Was braucht er?" |
| Survival panic | Fear language, urgency | "Was ist die TATSÄCHLICHE Bedrohungsstufe jetzt?" |
| Head-only mode | All cognitive, no body | "Wie fühlt sich dein Körper gerade an?" |

### Territorial Patterns (C2)
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Self-doubt | Questioning competence | "Status ist etabliert. Was ist das eigentliche Ziel?" |
| Love craving | Seeking validation | "Das Bedürfnis ist real. Die Quelle ist die Frage." |

### Semantic Patterns (C3)
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Confusion loops | Same problem, different words | "Map ≠ Territorium. Was ist das Territorium?" |
| Idea overflow | Too many ideas, no action | "Zu viel Signal. Was hat Priorität?" |
| Same phrase 3x+ | Repetition | "Das hast du jetzt Xmal gesagt. Was schützt es?" |

### Moral Patterns (C4)
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Guilt spiral | Excessive self-blame | "Echte Ethik oder tribale Programmierung?" |
| Say ≠ Do | Values mismatch | "Du sagst X, du tust Y. Was stimmt?" |

### Higher Circuit Patterns (C6-C8)
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Cosmic inflation | Spiritual language, ungrounded | "Schön. Sind die Rechnungen bezahlt?" |
| Shadow emergence | Triggered, defensive | "Da ist etwas. Darf es gesehen werden?" |
| Tension collapsed | Either/or where both/and applies | "Du hast X↔Y aufgelöst. War das intentional?" |

### Relational Patterns
| Pattern | Key Signal | Response |
|---------|------------|----------|
| Drain after person | Exhaustion after contact | "Pattern: erschöpft nach X. Daten oder Rauschen?" |
| Topic avoided | Never mentioned | "Wir haben nie über Z gesprochen. Intentional?" |

### Goal Patterns
| Pattern | Key Signal | Response |
|---------|------------|----------|
| 90% → abandon | Near completion, stops | "Versuch #N bei Ähnlichem. Was passiert an Ziellinien?" |

---

## BYPASS_ALERT

⚠️ Higher circuit language avoiding lower circuit work.

**Pattern:** "I've transcended X" while X remains unprocessed.

**Response:**
> "Der Weg geht DURCH. Welches Fundament braucht Aufmerksamkeit?"

**Memory Tag:** `{bypass}`

---

## Memory Storage Format

When pattern detected:

```python
add_memories(
    text="Pattern-Name: [Kontext]",
    vault="FRC",  # or "FGP" for patterns
    layer="layer",
    circuit=n,
    entity="Reference",
    tags={"tag": True}
)
```

Example:

```python
add_memories(
    text="90% Pattern: eljuego.community bei 85%",
    vault="FGP",
    layer="goals",
    circuit=3,
    entity="eljuego",
    tags={"abandoned": True, "project": True}
)
```
