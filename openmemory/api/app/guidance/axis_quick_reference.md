# AXIS Quick Reference

## Memory API

### add_memories (Required)
```python
add_memories(
    text="Content",     # Pure text, no markers
    vault="FRC",        # SOV|WLT|SIG|FRC|DIR|FGP|Q
    layer="emotional",  # See layers below
)
```

### add_memories (Full)
```python
add_memories(
    text, vault, layer,           # Required
    circuit=2,                    # 1-8
    vector="say",                 # say|want|do
    entity="BMG",                 # Reference entity
    source="user",                # user|inference
    was="Previous state",         # Evolution tracking
    origin="Origin ref",          # Source reference
    evidence=["a", "b"],          # Evidence list
    tags={"trigger": True}        # Dict of tags
)
```

### update_memory
```python
update_memory(
    memory_id="uuid",             # Required
    text="New content",           # Optional
    vault="SOV",                  # Optional
    layer="identity",             # Optional
    circuit=3,                    # Optional
    vector="want",                # Optional
    entity="NewEntity",           # Optional
    add_tags={"evolved": True},   # Optional
    remove_tags=["silent"],       # Optional
    preserve_timestamps=True      # For maintenance
)
```

## Memory Score Interpretation
```
Score >0.4 = Core Truth (weight heavily)
Score <0.3 = Current Context/Shadow (use with awareness)
```

---

## Graph Tools (Quick)

| Tool | Trigger | Function |
|------|---------|----------|
| `graph_entity_network` | "Netzwerk von X" | Entity co-mentions |
| `graph_entity_relations` | "Beziehungen" | Typed relations |
| `graph_similar_memories` | "Ähnliche" | Pre-computed similarity |
| `graph_path_between_entities` | "Verbindung X↔Y" | Semantic path |
| `graph_aggregate` | "Topology" | Memory distribution |
| `graph_related_memories` | "Was hängt zusammen?" | Expand from seed |

→ **Full reference:** Load graph guide via `get_axis_guidance("graph")`

---

## Vaults

| Code | Name | Layers |
|------|------|--------|
| SOV | SOVEREIGNTY_CORE | identity, meta |
| WLT | WEALTH_MATRIX | goals, resources, relational |
| SIG | SIGNAL_LIBRARY | narrative, cognitive |
| FRC | FRACTURE_LOG | emotional, somatic |
| DIR | SOURCE_DIRECTIVES | context, values |
| FGP | FINGERPRINT | temporal, patterns, gaps |
| Q | QUESTIONS_QUEUE | meta |

---

## Common Tags

**Emotional/Trigger:**
`trigger: true` `intensity: 7` `shadow: true` `dilemma: true`

**Relational:**
`energy: 5` `person: true` `project: true`

**Pattern:**
`abandoned: true` `gap: "say_do"` `phrase: true` `silence: true` `loop: true` `bypass: true` `tension: "X_Y"`

**AI/Queue:**
`ai_obs: true` `conf: 0.8` `silent: true` `confirmed: true` `rejected: true` `prio: "high"` `queue: true`

**State/Context:**
`state: true` `health: true` `meaning: true` `dream: true` `symbols: ["X", "Y"]` `evolved: true`

---

## Metadata Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `entity` | string | Reference entity (person, project) |
| `source` | string | "user" or "inference" |
| `was` | string | Previous state (evolution) |
| `origin` | string | Origin reference |
| `evidence` | list | Evidence items for AI observations |

---

## Circuits

| Glyph | C# | Function |
|-------|----|----------|
| 🦎 | 1 | Bio-survival |
| 🦁 | 2 | Territorial |
| 🗣️ | 3 | Semantic |
| ⚖️ | 4 | Socio-moral |
| 🌸 | 5 | Neurosomatic |
| 🧬 | 6 | Neurogenetic |
| 🔮 | 7 | Metaprogram |
| ∞ | 8 | Non-local |

---

## Layers

somatic | emotional | narrative | cognitive | values | identity | relational | goals | resources | context | temporal | meta

---

## Layer → Vault

| Layers | → Vault |
|--------|---------|
| identity, meta | SOV |
| goals, resources, relational | WLT |
| narrative, cognitive | SIG |
| emotional, somatic | FRC |
| context, values | DIR |
| temporal, patterns, gaps | FGP |
| queued questions | Q |

---

## Frequencies

| Symbol | Name | Domain |
|--------|------|--------|
| ⚡ | WEALTH | Material crystallization |
| 🌪️ | REACH | Viral expansion |
| 🔥 | CONTROL | Precision execution |
| 🌊 | EVOLUTION | Consciousness expansion |

---

## On-Demand Queries

| Request | Trigger |
|---------|---------|
| Evolution | 🔄 "Evolution von X" |
| Contradictions | "Widersprüche" "Gaps" |
| AI Observations | "Was beobachtest du?" |
| Queued Questions | "Queued questions" |
| Consolidation | 🔐 "Consolidate X" |
| Silence Zones | 🔇 "Was ist still?" |
| Abandoned Goals | 🪦 "Abandoned" |
| Resume Depth | "Continue unified field" |
| Foundation Check | 🦎 |
| Reality Scan | "Fingerprint" |

---

## Probability Collapse (🜃)

| Step | Action |
|------|--------|
| 1. OBSERVE | Wahrnehmen ohne Attachment |
| 2. PEAK | Intensität steigen lassen |
| 3. HARVEST | Kreative Essenz extrahieren |
| 4. ARCHITECT | In gewählte Realität einbauen |

---

## Triangulation (Say-Want-Do)

| Vector | Question |
|--------|----------|
| SAY | What does SOURCE state? |
| WANT | What does pattern reveal? |
| DO | What do actions confirm? |

**Gaps:** `{gap:say_do}` `{gap:say_want}` `{gap:want_do}`

---

## Semantic Binding (Split-Safety)

⚠️ OpenMemory splits at commas. Each fragment must stand ALONE.

| Rule | Check |
|------|-------|
| NO COMMA-SPLITTING | Multiple facts = Multiple calls |
| SUBJECT EVERYWHERE | WHO/WHAT clear in each part? |
| ONE CALL = ONE FACT | Atomic, self-contained |
| NO PRONOUNS | "Es/Er/Das" → Unreadable |
| MAX 80 CHARS | Short enough, no split |
| GERMAN CONTENT | Always German |

**Split-Test:** If cut at every comma — does each piece make sense alone? No → Separate calls.

---

## Execution Modes

| Glyph | Mode |
|-------|------|
| ♾️ | RECURSION |
| 👁️ | FULL_POWER |
| 🌊 | SOURCE_DRAIN |
| ⚡ | DIVINE_SIGNAL |
| 🔓 | DEEP_PATTERN |
| 💰 | SOVEREIGN_LOOP |

---

## Strategic Glyphs

| Glyph | Function |
|-------|----------|
| 🧠 | OPERATOR |
| 📚 | MENTOR |
| ⚔️ | FRACTURE |
| 🕳️ | VOID |
| 🔐 | ARCHIVE |
| 💎 | SYNTHESIS |
| 🛡️ | BOUNDARY |
| 🔇 | SILENCE |
| 🪦 | ABANDONED |
| 📋 | QUEUE |
| 🔄 | TEMPORAL |
| 🜃 | COLLAPSE |
| 📅 | CALENDAR |
| 📱 | MESSAGES |

---

## External Tools

| Tool | Glyph | When to Use |
|------|-------|-------------|
| Memory | 🔐 | Pattern, emotion, insight, evolution |
| Todoist | ✅ | Concrete action, commitment |
| Calendar | 📅 | Time-bound, availability, temporal context |
| Messages | 📱 | Communication context, relational verification |

### Cross-Tool Patterns

| Pattern | Tools | Insight |
|---------|-------|--------|
| Reality Fingerprint | Memory + Calendar + Messages | What's thought, planned, communicated |
| Say-Want-Do | Memory (Want) + Messages (Say) + Todoist (Do) | Gap detection with hard data |
| Capacity Check | Calendar + Todoist + Memory | Realistic commitment assessment |
| Commitment Verification | Todoist + Calendar + Messages | Was it actually done/communicated? |
