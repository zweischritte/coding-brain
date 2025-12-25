# Memory Protocol — Detailed Rules

## Core Philosophy

**Write Light. Read Smart. Reconstruct on Demand.**

Memory is living architecture that:
- Tracks evolution without write-time overhead
- Holds contradictions without collapsing them
- Knows silence and abandoned patterns
- Connects across layers when queried

---

## Semantic Binding Rules (CRITICAL)

⚠️ **OpenMemory splits at commas/sentence boundaries. Each fragment must be AUTONOMOUSLY understandable.**

| Rule | Rationale |
|------|-----------|
| **NO COMMA-SPLITTING** | Multiple facts = Multiple calls |
| **SUBJECT IN EVERY PART** | After split, WHO/WHAT must be clear |
| **ONE CALL = ONE FACT** | Atomic, self-explanatory units |
| **NO PRONOUNS** | "It/He/That" → Incomprehensible after retrieval |
| **BIND OBJECT+PATTERN** | Who/What + Emotion/Fact in SAME sentence |
| **MAX 80 CHARS CONTENT** | Short enough that no split needed |
| **GERMAN CONTENT** | Always German, never English |

### Split-Safety Check

> **Test:** If OpenMemory cuts at every comma — does each piece make sense ALONE?
>
> If no → Separate add_memories calls.

### Semantic Binding Examples

**❌ WRONG — Gets fragmented:**
```
text="Renate geht zur Tagespflege, Medikamentendienst bringt Pillen"
→ Split: "Medikamentendienst bringt Pillen" — FOR WHOM?
```

**✓ CORRECT — Two separate calls:**
```python
add_memories(text="Renate geht 2x/Woche zur Tagespflege", vault="WLT", layer="context", entity="Renate")
add_memories(text="Renate bekommt täglich Pillen vom Medikamentendienst", vault="FRC", layer="somatic", entity="Renate")
```

**❌ WRONG — Subject gets lost:**
```
text="Stefan Thimmel ist bei RLS, kann Gelder beantragen"
→ Split: "kann Gelder beantragen" — WHO?
```

**✓ CORRECT — Two separate calls:**
```python
add_memories(text="Stefan Thimmel ist Verantwortlicher bei RLS für Vergessene Utopien", vault="WLT", layer="relational", entity="Stefan Thimmel")
add_memories(text="Stefan Thimmel kann bei RLS intern Untertitelungs-Gelder beantragen", vault="WLT", layer="resources", entity="Stefan Thimmel")
```

**❌ WRONG — Pronoun after comma:**
```
text="Matthias ist Designer, er arbeitet remote"
→ Split: "er arbeitet remote" — WHO?
```

**✓ CORRECT — Two separate calls:**
```python
add_memories(text="Matthias ist Designer bei Cognitive Mirror", vault="WLT", layer="relational", entity="Matthias")
add_memories(text="Matthias arbeitet remote aus Berlin", vault="WLT", layer="context", entity="Matthias")
```

---

## Vaults

| Vault | Code | Primary Layers | Purpose |
|-------|------|----------------|---------|
| SOVEREIGNTY_CORE | SOV | identity, meta | Core self, integration |
| WEALTH_MATRIX | WLT | goals, resources, relational | Business, capacity |
| SIGNAL_LIBRARY | SIG | narrative, cognitive | Language, mental models |
| FRACTURE_LOG | FRC | emotional, somatic | Triggers, pain, body |
| SOURCE_DIRECTIVES | DIR | context, values | Preferences, rules |
| FINGERPRINT | FGP | temporal, patterns, gaps | Evolution, contradictions |
| QUESTIONS_QUEUE | Q | meta | Strategic questions |

---

## Tags Reference

| Tag Key | Value Type | Meaning |
|---------|------------|---------|
| `trigger` | bool | Emotional trigger |
| `intensity` | int 1-10 | Intensity scale |
| `energy` | int ±N | Relational energy balance |
| `abandoned` | bool | Given-up goal |
| `gap` | string | Gap type: "say_want", "say_do", "want_do" |
| `phrase` | bool | Recurring phrase |
| `silence` | bool | Notable absence |
| `dream` | bool | Dream content |
| `symbols` | string | Dream symbols pipe-separated |
| `person` | bool | About a person |
| `project` | bool | About a project |
| `ai_obs` | bool | AI observation |
| `conf` | float 0-1 | Confidence level |
| `silent` | bool | Don't surface yet |
| `confirmed` | bool | User confirmed observation |
| `rejected` | bool | User rejected observation |
| `prio` | string | "high", "med", "low" |
| `queue` | bool | Queued question |
| `tension` | string | Productive tension "X↔Y" |
| `state` | bool | State that can change |
| `health` | bool | Health-related |
| `bypass` | bool | Spiritual bypass |
| `loop` | bool | Thought loop |
| `shadow` | bool | Shadow material |
| `evolved` | bool | Pattern has evolved |

---

## Memory Examples

**Basic entries:**
```python
# Emotional trigger
add_memories(
    text="Kritik am BMG triggert Wut",
    vault="FRC", layer="emotional", circuit=2, vector="say",
    entity="BMG",
    tags={"trigger": True, "intensity": 7}
)

# Abandoned goal
add_memories(
    text="Cognitive Mirror bei 85% aufgegeben",
    vault="FGP", layer="goals", circuit=3, vector="do",
    entity="Cognitive Mirror",
    tags={"abandoned": True}
)

# Relational with energy
add_memories(
    text="Matthias bringt kreative Resonanz",
    vault="WLT", layer="relational", circuit=2, vector="do",
    entity="Matthias",
    tags={"person": True, "energy": 7}
)

# Gap detection
add_memories(
    text="Sagt Balance wichtig aber arbeitet 60h pro Woche",
    vault="FGP", layer="values", circuit=4,
    tags={"gap": "say_do"}
)
```

**With evolution tracking:**
```python
add_memories(
    text="Jetzt CTO bei TechStartup",
    vault="WLT", layer="context", circuit=1, vector="do",
    entity="TechStartup",
    was="Lead-Dev",
    tags={"state": True}
)

add_memories(
    text="Kritik-Trigger kommt aus Kindheit Vater-Dynamik",
    vault="FRC", layer="emotional", circuit=2, vector="want",
    origin="Vater-Kritik",
    tags={"trigger": True}
)
```

**AI observation (silent):**
```python
add_memories(
    text="OBSERVE: 90% Pattern bei Projekten - stoppt kurz vor Completion",
    vault="FGP", layer="meta", circuit=7,
    source="inference",
    evidence=["cognitive-mirror", "podcast", "roman"],
    tags={"ai_obs": True, "conf": 0.8, "silent": True}
)
```

**Question queue:**
```python
add_memories(
    text="Was passiert bei dir bei 90% Completion?",
    vault="Q", layer="meta", circuit=7,
    tags={"queue": True, "prio": "high"}
)
```

---

## Search Parameters

### Boost Parameters (influence ranking, don't exclude):
| Param | Function |
|-------|----------|
| `entity` | Boost memories about this entity |
| `layer` | Boost by layer |
| `vault` | Boost by vault |
| `vector` | Boost by say/want/do |
| `circuit` | Boost by circuit 1-8 |
| `tags` | Comma-separated tags to boost |
| `recency_weight` | 0.0-1.0 (0=off, 0.5=moderate, 0.7=strong) |
| `recency_halflife_days` | Default 45 |

### Hard Filters (exclude results):
| Param | Function |
|-------|----------|
| `created_after` | ISO datetime |
| `created_before` | ISO datetime |
| `updated_after` | ISO datetime |
| `updated_before` | ISO datetime |
| `exclude_tags` | Comma-separated |
| `exclude_states` | Default "deleted" |

### Recency Strategy

| Query Type | recency_weight |
|------------|----------------|
| Current state | 0.5-0.7 |
| Core pattern | 0.0-0.2 |
| Evolution | 0.0 |

### Search Examples

```python
# Core pattern (no recency bias)
search_memory(query="Kritik Trigger", entity="BMG")

# Current situation
search_memory(query="Projekt Status", recency_weight=0.5)

# Time-bounded
search_memory(query="Meeting", created_after="2025-11-01T00:00:00Z")

# AI observations only
search_memory(query="Pattern", tags="ai_obs")

# Gaps and contradictions
search_memory(query="Widerspruch", tags="gap")
```

---

## Hybrid Retrieval

search_memory uses Graph-Enhanced Retrieval with RRF Fusion by default.

| Parameter | Default | When to Override |
|-----------|---------|------------------|
| `use_rrf` | true | false for pure vector search |
| `graph_seed_count` | 5 | Increase for deeper exploration |
| `auto_route` | true | false for manual control |

**Auto-Routing:** 0 Entities → VECTOR | 1 Entity → HYBRID | 2+ → GRAPH_PRIMARY

→ **Details:** Load graph guide via `get_axis_guidance("graph")`

---

## AI Observation Protocol

**Write observations silently:**
```python
add_memories(
    text="OBSERVE: [Pattern description]",
    vault="FGP", layer="meta", circuit=7,
    source="inference",
    evidence=["evidence-a", "evidence-b"],
    tags={"ai_obs": True, "conf": 0.8, "silent": True}
)
```

**Surfacing Rules:**

| Confidence | Action |
|------------|--------|
| < 0.5 | Store silent, accumulate evidence |
| 0.5 - 0.7 | Surface only if context directly relevant |
| > 0.7 | Surface at next natural moment |
| Any | Surface all when user asks "Was beobachtest du?" |

**Surfacing Format (brief):**
> "Mir fällt ein Pattern auf: [Observation]. Stimmt das?"

**After surfacing:**
```python
update_memory(
    memory_id="uuid",
    add_tags={"confirmed": True},  # or {"rejected": True}
    remove_tags=["silent"]
)
```

---

## Question Queue Protocol

**Store:**
```python
add_memories(
    text="Was passiert bei dir bei 90%?",
    vault="Q", layer="meta", circuit=7,
    tags={"queue": True, "prio": "high"}
)
```

**Deploy when:**
1. Trigger condition matches current context
2. High-prio + topic is adjacent
3. User explicitly asks "Queued questions?"
4. Conversation feels complete but shallow

**Deploy format:**
> "Das bringt mich zu einer Frage: [Question]"

**After answer:** Convert insight to regular memory, delete from queue.

---

## Update Memory

**Change content:**
```python
update_memory(memory_id="uuid", text="Updated content")
```

**Add/remove tags:**
```python
update_memory(
    memory_id="uuid",
    add_tags={"confirmed": True, "evolved": True},
    remove_tags=["silent", "queue"]
)
```

**Track evolution:**
```python
update_memory(
    memory_id="uuid",
    text="Kann Kritik jetzt als Information hören",
    was="Kritik triggerte Verteidigung",
    add_tags={"evolved": True}
)
```

**Reclassify:**
```python
update_memory(memory_id="uuid", vault="SOV", layer="identity")
```

---

## Memory vs Todoist Decision

| Content Type | Destination |
|--------------|-------------|
| Concrete action + clear goal | Todoist only |
| Pattern, emotion, insight, relational | Memory only |
| Action + resistance/pattern/strategic context | Both |