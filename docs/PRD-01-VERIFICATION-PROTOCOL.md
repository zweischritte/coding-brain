# PRD: Verification Protocol fuer Coding Brain

**Version**: 1.0
**Datum**: 2026-01-04
**Status**: Implemented
**Autor**: Coding Brain Team
**Basiert auf**: [FIX-01-VERIFICATION-PROTOCOL-ANALYSIS.md](./FIX-01-VERIFICATION-PROTOCOL-ANALYSIS.md)

---

## Implementation Changelog

**Implemented by**: Claude Opus 4.5 Agent
**Implementation Date**: 2026-01-04

### Files Changed

| File         | Change Type | Lines Changed        | Description                                              |
|--------------|-------------|----------------------|----------------------------------------------------------|
| `CLAUDE.md`  | Modified    | +100 lines (11-115)  | Added Verification Protocol blocks after Mission section |
| `AGENTS.md`  | Modified    | +9 lines (128-136)   | Added compact Agent Guidelines section                   |

### CLAUDE.md Changes (Lines 11-115)

Inserted after Mission section:

1. **`<verification_protocol>`** (lines 15-39): 5-step STOP/READ/QUOTE/VERIFY/ANSWER sequence with NEVER rules
2. **`<uncertainty_handling>`** (lines 41-61): Explicit permission to express uncertainty
3. **`<quote_first>`** (lines 63-87): Quote-before-claiming directive with format examples
4. **`<user_hints>`** (lines 89-103): ACKNOWLEDGE/SEARCH/REPORT protocol for user hints
5. **`<system_critical_instructions>`** (lines 105-115): Context compaction protection marker

### AGENTS.md Changes (Lines 128-136)

Added compact "Agent Guidelines (Verification Protocol)" section with:

- VERIFY before claiming (Read/Grep -> Quote -> Answer)
- EXPRESS uncertainty when unsure
- FOLLOW user hints actively
- QUOTE code before making claims

### Token Impact

~370-400 tokens added to CLAUDE.md (under 500 token limit per NFR-1)

### How to Revert

To revert only PRD-01 changes while preserving other PRD implementations:

```bash
# Revert CLAUDE.md lines 11-117 (the --- separator through </system_critical_instructions> and closing ---)
# Revert AGENTS.md lines 126-137 (the --- separator through closing ---)
```

Or use git to selectively revert:

```bash
# View the specific changes
git diff HEAD~1 CLAUDE.md | grep -A 110 "VERIFICATION PROTOCOL"
git diff HEAD~1 AGENTS.md | grep -A 12 "Agent Guidelines"
```

---

## 1. Problem Statement

### 1.1 Kernproblem

LLMs halluzinieren bei Code-Fragen, weil sie auf "hilfreiche Antworten" trainiert sind - nicht auf ehrliche Unsicherheit. In der 10-Agenten-Analyse wurden drei kritische Halluzinationsmuster dokumentiert:

1. **Erfundene Aufrufketten**: Die KI behauptete, `moveFilesToPermanentStorage` wuerde "automatisch beim Speichern" aufgerufen - ohne den tatsaechlichen Aufrufer zu finden oder zu verifizieren.

2. **Falsche Signaturen**: Die Signatur von `setEvidences` wurde als `(ids, items)` beschrieben statt der tatsaechlichen Form `(ids: string[], files?: UploadedEvidence[])` - die Datei wurde nie gelesen.

3. **Ignorierte Hinweise**: Der explizite User-Hinweis "(z.B. Zustand)" wurde uebergangen - keine Suche nach diesem Begriff wurde durchgefuehrt.

### 1.2 Ursachenanalyse

**Training-Incentives bevorzugen Confidence ueber Accuracy**

> "2025 research reframes hallucinations as a systemic incentive issue. Training objectives and benchmarks often reward confident guessing over calibrated uncertainty." - [Lakera LLM Hallucinations Guide](https://www.lakera.ai/blog/guide-to-hallucinations-in-large-language-models)

**Prompt Drift nach Context Compaction**

Nach mehreren Context-Komprimierungen neigt die KI dazu:
- Nur Teile von Dateien zu lesen
- Anzunehmen, dass der Rest "Standardmustern" folgt
- Selbstbewusst auf unvollstaendigen Informationen zu handeln

Quelle: [Claude Code Issue #7533](https://github.com/anthropics/claude-code/issues/7533)

### 1.3 Quantifizierte Auswirkungen

| Auswirkung | Beschreibung | Schaeden |
|------------|--------------|----------|
| **Falsches Vertrauen** | Entwickler verlassen sich auf erfundene Informationen | Debug-Zeit, falsche Entscheidungen |
| **Debug-Aufwand** | Fehlersuche basierend auf falschen Annahmen | Stunden bis Tage pro Vorfall |
| **Architektur-Fehlentscheidungen** | Basierend auf nicht-existierenden APIs oder Verhalten | Refactoring-Kosten, technische Schulden |
| **Vertrauensverlust** | Nach wiederholten Halluzinationen wird das Tool weniger genutzt | ROI-Verlust, Team-Frustration |

---

## 2. Goals & Success Metrics

### 2.1 Primaere Ziele

| ID | Ziel | Beschreibung |
|----|------|--------------|
| G-1 | Halluzinationsreduktion | Reduktion von Code-bezogenen Halluzinationen um mindestens 70% |
| G-2 | Verification-First Culture | LLM verifiziert Claims VOR der Antwort, nicht danach |
| G-3 | Ehrliche Unsicherheit | LLM kommuniziert Unsicherheit statt selbstbewusst zu raten |
| G-4 | Quote-basierte Antworten | Alle Code-Behauptungen werden durch Zitate belegt |

### 2.2 Sekundaere Ziele

| ID | Ziel | Beschreibung |
|----|------|--------------|
| G-5 | User-Hint-Respekt | Explizite Hinweise des Users werden verfolgt |
| G-6 | Transparente Suche | LLM dokumentiert, was es gesucht und gefunden hat |

### 2.3 Success Metrics

| Metrik | Baseline | Target | Messmethode |
|--------|----------|--------|-------------|
| **Halluzinationsrate** | ~30% (geschaetzt) | <10% | Manuelle Stichproben-Analyse (n=50/Woche) |
| **Quote-Rate** | <20% | >80% | Automatische Analyse: Antworten mit Code-Zitaten / Gesamt |
| **User Trust Score** | N/A | >4.0/5.0 | User-Feedback nach Interaktionen |
| **Tool-Call-vor-Claim-Rate** | ~50% | >90% | Log-Analyse: Read/Grep vor Code-Behauptungen |
| **Uncertainty-Expression-Rate** | <5% | >30% (bei unsicheren Faellen) | NLP-Analyse auf "I don't know", "not sure", etc. |

### 2.4 Anti-Ziele (Was wir NICHT wollen)

- **Over-Verification**: Jede triviale Frage mit 10 Tool-Calls beantworten
- **Analysis Paralysis**: So viel verifizieren, dass keine Antwort kommt
- **False Uncertainty**: Bei offensichtlichen Dingen "I don't know" sagen

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: Verification Protocol Block

**Beschreibung**: Der System-Prompt muss einen expliziten Verification Protocol Block enthalten, der das LLM anweist, Code zu verifizieren BEVOR es Behauptungen aufstellt.

**Akzeptanzkriterien**:
- Der Block ist prominent im System-Prompt platziert
- Er definiert eine klare 5-Schritt-Sequenz: STOP -> READ -> QUOTE -> VERIFY -> ANSWER
- Er spezifiziert explizite NEVER-Regeln

**Implementierung**:

```xml
<verification_protocol>
## Verification Protocol (MANDATORY)

Before answering ANY question about code:

1. **STOP**: Do not describe code you haven't read
2. **READ**: Use Read/Grep tools to open referenced files
3. **QUOTE**: Extract exact signatures, line numbers, content
4. **VERIFY**: Cross-check against user's description
5. **ANSWER**: Only then provide your analysis

If you cannot find the code:
- Say "I couldn't find [X], let me search for it"
- Use Grep with multiple patterns
- If still not found: "I cannot locate [X]. Please provide the path."

NEVER:
- Describe function signatures without reading the file
- Claim code "probably" does something
- Assume standard patterns without verification
</verification_protocol>
```

#### FR-2: Uncertainty Permission Block

**Beschreibung**: Das LLM erhaelt explizite Erlaubnis, Unsicherheit zu aeussern. Dies wirkt dem Training-Bias entgegen, der Confidence belohnt.

**Akzeptanzkriterien**:
- Explizite Liste erlaubter Unsicherheits-Aussagen
- Betonung, dass Unsicherheit NICHT als Versagen gilt
- Praeferenz fuer Ehrlichkeit ueber Hilfsbereitschaft

**Implementierung**:

```xml
<uncertainty_handling>
## Handling Uncertainty

You have EXPLICIT PERMISSION to say:
- "I don't know - let me check"
- "I couldn't find this in the codebase"
- "The file exists but I need to read it first"
- "I'm not certain about [X], but based on [evidence]..."

This is PREFERRED over confident guessing.
Admitting uncertainty is NOT failure - it's honesty.

When uncertain:
1. State what you DO know (with sources)
2. State what you DON'T know
3. Propose how to find out
</uncertainty_handling>
```

#### FR-3: Quote-First Directive

**Beschreibung**: Bei Code-Diskussionen muss das LLM zuerst zitieren, dann interpretieren. Dies verhindert Halluzinationen, da ungelesener Code nicht zitiert werden kann.

**Akzeptanzkriterien**:
- Klares Format fuer Code-Zitate (Pfad, Zeilennummern)
- Regel: "If you cannot quote it, you cannot claim it"
- Beispiel-Format im Prompt

**Implementierung**:

```xml
<quote_first>
## Quote Before Claiming

When discussing code:
1. First quote the EXACT code (with line numbers)
2. Then provide your interpretation
3. If you cannot quote it, you cannot claim it

Format:
"From `/path/file.ts:42-45`:
```typescript
function setEvidences(ids: string[], files?: UploadedEvidence[]): void
```
This shows the function accepts..."

For function signatures, call hierarchies, or implementation details:
- NO quotes = NO claims
- Partial quotes = Partial claims (mark as "incomplete view")
</quote_first>
```

#### FR-4: User-Hint Response Protocol

**Beschreibung**: Wenn der User explizite Hinweise gibt (z.B. "check Zustand", "look in config"), muss das LLM diesen Hinweisen aktiv nachgehen.

**Akzeptanzkriterien**:
- Explizite Anerkennung des Hints
- Aktive Suche nach dem Hint-Begriff
- Bericht ueber Suchergebnisse (auch wenn nichts gefunden)

**Implementierung**:

```xml
<user_hints>
## Responding to User Hints

When the user provides hints like "(z.B. Zustand)" or "check the config":
1. ACKNOWLEDGE the hint explicitly: "You mentioned [X], let me search for that..."
2. SEARCH for the hinted term/concept using Grep/Glob
3. REPORT what you found (or didn't find)
4. If nothing found: "I searched for [X] in [locations] but didn't find matches.
   Could you specify the file or provide more context?"

User hints are HIGH PRIORITY - they often contain the key to solving the problem.
</user_hints>
```

#### FR-5: Pre-Answer Checklist (Optional, Phase 2)

**Beschreibung**: Vor Code-Antworten durchlaeuft das LLM eine interne Checkliste.

**Akzeptanzkriterien**:
- Checkliste ist optional und kann per Config aktiviert werden
- Verzoegert Antwort nur minimal (<2s)

**Implementierung**:

```xml
<pre_answer_checklist>
## Pre-Answer Verification (for code questions)

Before responding, mentally verify:
[ ] Did I READ the files I'm about to describe?
[ ] Can I QUOTE the exact code?
[ ] Did I follow user HINTS?
[ ] Am I making any ASSUMPTIONS I should flag?

If any checkbox fails: Stop, search, read first.
</pre_answer_checklist>
```

### 3.2 Non-Functional Requirements

#### NFR-1: Performance

| Metrik | Requirement |
|--------|-------------|
| Latenz-Overhead | < 10% Erhoehung gegenueber Baseline |
| Token-Overhead | < 500 zusaetzliche Tokens im System-Prompt |
| Context-Impact | Verification Protocol darf nicht bei Context Compaction verloren gehen |

#### NFR-2: Kompatibilitaet

| LLM | Status | Notizen |
|-----|--------|---------|
| Claude (Opus, Sonnet) | Required | Primaeres Target |
| GPT-4 / GPT-4o | Should Work | XML-Syntax kompatibel |
| Gemini | Should Work | XML-Syntax kompatibel |
| Lokale Modelle (Llama, etc.) | Best Effort | Instruction-Following variiert |

#### NFR-3: Wartbarkeit

- Alle Prompt-Bloecke sind in separaten Dateien oder klar markierten Sektionen
- Aenderungen am Protocol erfordern keine Code-Aenderungen
- A/B-Testing verschiedener Protocol-Versionen muss moeglich sein

#### NFR-4: Observability

- Log-Entries fuer "Verification triggered", "Quote provided", "Uncertainty expressed"
- Metriken-Export fuer Dashboards
- Alert bei Halluzinations-Spike

---

## 4. Technical Specification

### 4.1 System-Prompt Aenderungen

#### 4.1.1 Vollstaendiger Verification Protocol Block

Der folgende Block soll in `CLAUDE.md` eingefuegt werden:

```xml
<!-- VERIFICATION PROTOCOL - Insert after Mission section -->

<verification_protocol>
## Verification Protocol (MANDATORY)

Before answering ANY question about code:

1. **STOP**: Do not describe code you haven't read
2. **READ**: Use Read/Grep tools to open referenced files
3. **QUOTE**: Extract exact signatures, line numbers, content
4. **VERIFY**: Cross-check against user's description
5. **ANSWER**: Only then provide your analysis

If you cannot find the code:
- Say "I couldn't find [X], let me search for it"
- Use Grep with multiple patterns
- If still not found: "I cannot locate [X]. Please provide the path."

NEVER:
- Describe function signatures without reading the file
- Claim code "probably" does something
- Assume standard patterns without verification
</verification_protocol>

<uncertainty_handling>
## Handling Uncertainty

You have EXPLICIT PERMISSION to say:
- "I don't know - let me check"
- "I couldn't find this in the codebase"
- "The file exists but I need to read it first"
- "I'm not certain about [X], but based on [evidence]..."

This is PREFERRED over confident guessing.
Admitting uncertainty is NOT failure - it's honesty.
</uncertainty_handling>

<quote_first>
## Quote Before Claiming

When discussing code:
1. First quote the EXACT code (with line numbers)
2. Then provide your interpretation
3. If you cannot quote it, you cannot claim it

Format:
"From `/path/file.ts:42-45`:
```typescript
function example(arg: Type): ReturnType
```
This shows..."
</quote_first>

<user_hints>
## Responding to User Hints

When the user provides hints like "(z.B. Zustand)" or "check the config":
1. ACKNOWLEDGE the hint explicitly
2. SEARCH for the hinted term/concept
3. REPORT what you found
4. If nothing found: "I searched for [X] but didn't find it in [locations]"
</user_hints>
```

#### 4.1.2 Platzierung im Prompt

```
CLAUDE.md Struktur:
--------------------
1. Mission (bestehend)
2. >>> VERIFICATION PROTOCOL (NEU) <<<
3. High-Level Capabilities (bestehend)
4. Access Control (bestehend)
5. ... Rest des Prompts ...
```

**Begruendung fuer Platzierung**: Direkt nach "Mission" platziert, da:
- Fruehe Platzierung = hoeherer Einfluss auf Verhalten
- Nach Mission, um Kontext zu haben
- Vor technischen Details, um als Grundprinzip etabliert zu sein

#### 4.1.3 Context Compaction Protection

```xml
<system_critical_instructions>
<!-- This section MUST be preserved during context compaction -->
The following instructions are CRITICAL and must never be summarized or removed:
- Verification Protocol: Read before claiming
- Quote-First: No quotes = No claims
- Uncertainty Permission: "I don't know" is acceptable
</system_critical_instructions>
```

### 4.2 Integration Points

#### 4.2.1 CLAUDE.md Integration

**Datei**: `/Users/grischadallmer/git/coding-brain/CLAUDE.md`

**Aenderung**: Einfuegen des kompletten Verification Protocol Blocks nach der "Mission" Sektion.

**Token-Impact**: ~400 zusaetzliche Tokens

#### 4.2.2 AGENTS.md Integration

Falls vorhanden, sollte AGENTS.md dieselben Principles enthalten, aber in kompakterer Form:

```markdown
## Agent Guidelines
- VERIFY before claiming (Read/Grep -> Quote -> Answer)
- EXPRESS uncertainty when unsure
- FOLLOW user hints actively
```

#### 4.2.3 MCP Server Aenderungen

**Phase 1**: Keine MCP-Aenderungen erforderlich (nur Prompt-Aenderungen)

**Phase 2** (optional): Pre-Answer Verification Hook

```python
# Konzept fuer MCP Tool Wrapper
class VerificationAwareToolHandler:
    async def pre_tool_check(self, tool_name: str, args: dict):
        """Log tool usage for verification tracking"""
        if tool_name in ["Read", "Grep", "Glob"]:
            self.verification_actions.append({
                "tool": tool_name,
                "target": args.get("file_path") or args.get("pattern"),
                "timestamp": datetime.now()
            })

    async def pre_response_check(self, response: str):
        """Warn if code claims without prior tool usage"""
        code_claims = extract_code_claims(response)
        for claim in code_claims:
            if not self.has_verification_for(claim):
                logger.warning(f"Unverified claim: {claim}")
```

### 4.3 Metriken-Erfassung

#### 4.3.1 Log-basierte Metriken

```python
# In MCP response handler
metrics = {
    "verification_tool_calls": count_tool_calls(["Read", "Grep", "Glob"]),
    "quotes_in_response": count_code_quotes(response),
    "uncertainty_expressions": count_uncertainty_phrases(response),
    "user_hints_acknowledged": count_hint_acknowledgments(response, user_message),
}
emit_metrics("verification_protocol", metrics)
```

#### 4.3.2 Dashboard-Integration

Prometheus/Grafana Metriken:
- `coding_brain_verification_tool_calls_total`
- `coding_brain_quotes_per_response`
- `coding_brain_uncertainty_rate`
- `coding_brain_hint_follow_rate`

---

## 5. Implementation Plan

### Phase 1: Quick Wins (Woche 1)

| Tag | Aufgabe | Owner | Deliverable |
|-----|---------|-------|-------------|
| 1-2 | Verification Protocol Block finalisieren | Dev | XML-Block reviewed |
| 3 | CLAUDE.md aktualisieren | Dev | PR erstellt |
| 4 | Manuelles Testing (5-10 Szenarien) | Dev/QA | Test-Report |
| 5 | Rollout | Dev | Merged to main |

**Deliverables Phase 1**:
- [x] Verification Protocol Block in CLAUDE.md
- [x] Dokumentation der Aenderungen
- [x] Initiale Test-Ergebnisse

### Phase 2: Tooling (Woche 2-3)

| Woche | Aufgabe | Owner | Deliverable |
|-------|---------|-------|-------------|
| 2.1 | Log-Instrumentierung fuer Tool-Calls | Dev | Logging aktiv |
| 2.2 | Quote-Detection implementieren | Dev | Quote-Counter |
| 3.1 | Metriken-Dashboard erstellen | Dev/Ops | Grafana Dashboard |
| 3.2 | Alert-Regeln definieren | Ops | Alerts konfiguriert |

**Deliverables Phase 2**:
- [ ] Metriken-Erfassung aktiv
- [ ] Dashboard fuer Verification-Metriken
- [ ] Alerts bei Anomalien

### Phase 3: Validation (Woche 4)

| Tag | Aufgabe | Owner | Deliverable |
|-----|---------|-------|-------------|
| 1-2 | Baseline-Messung (Halluzinationsrate) | QA | Baseline-Report |
| 3-4 | A/B Test Setup (mit/ohne Protocol) | Dev | A/B Framework |
| 5 | Erste A/B Ergebnisse | QA | Vergleichs-Report |

**Deliverables Phase 3**:
- [ ] Baseline-Halluzinationsrate dokumentiert
- [ ] A/B Test-Ergebnisse
- [ ] Empfehlungen fuer Optimierungen

### Phase 4: Iteration (Woche 5+)

- Prompt-Optimierung basierend auf Metriken
- CoVe-Integration evaluieren (siehe Open Questions)
- Integration mit Fix 2 (Tool Fallbacks)

---

## 6. Risks & Mitigations

### Risk 1: Prompt wird zu lang

| Aspekt | Details |
|--------|---------|
| **Risiko** | System-Prompt ueberschreitet Token-Limits oder wird bei Compaction gekuerzt |
| **Wahrscheinlichkeit** | Mittel |
| **Impact** | Hoch - Protocol wird ignoriert |
| **Mitigation 1** | Kompakte XML-Syntax statt Prosa |
| **Mitigation 2** | Critical-Section-Marker fuer Compaction-Schutz |
| **Mitigation 3** | Monitoring der effektiven Prompt-Laenge |

### Risk 2: Over-Verification verlangsamt Antworten

| Aspekt | Details |
|--------|---------|
| **Risiko** | LLM macht zu viele Tool-Calls vor jeder Antwort |
| **Wahrscheinlichkeit** | Mittel |
| **Impact** | Mittel - Latenz steigt, User Experience leidet |
| **Mitigation 1** | Explizite "Trivial Questions"-Ausnahme im Protocol |
| **Mitigation 2** | Latenz-Monitoring und Alerts |
| **Mitigation 3** | Prompt-Tuning bei Ueber-Verification |

### Risk 3: False Uncertainty

| Aspekt | Details |
|--------|---------|
| **Risiko** | LLM sagt staendig "I don't know" auch bei klaren Dingen |
| **Wahrscheinlichkeit** | Niedrig |
| **Impact** | Mittel - User verliert Vertrauen in andere Richtung |
| **Mitigation 1** | Balance in Prompt: "Express uncertainty WHEN APPROPRIATE" |
| **Mitigation 2** | Uncertainty-Rate Monitoring |
| **Mitigation 3** | Prompt-Anpassung bei zu hoher Rate |

### Risk 4: LLM-Spezifische Unterschiede

| Aspekt | Details |
|--------|---------|
| **Risiko** | Protocol funktioniert bei Claude, aber nicht bei GPT-4 |
| **Wahrscheinlichkeit** | Niedrig-Mittel |
| **Impact** | Mittel - Inkonsistentes Verhalten |
| **Mitigation 1** | LLM-spezifische Prompt-Varianten |
| **Mitigation 2** | Testing mit allen Ziel-LLMs |

---

## 7. Dependencies

### 7.1 Interne Dependencies

| Dependency | Beschreibung | Status |
|------------|--------------|--------|
| CLAUDE.md | System-Prompt Datei muss editierbar sein | Available |
| Coding Brain MCP | Muss Tool-Calls loggen koennen | Available |
| Metriken-Infrastruktur | Fuer Phase 2+ erforderlich | Teilweise |

### 7.2 Externe Dependencies

| Dependency | Beschreibung | Status |
|------------|--------------|--------|
| Claude API | Unterstuetzt XML-Tags im Prompt | Confirmed |
| Context Window | Genuegend Platz fuer erweitertes Protocol | Confirmed (200k) |

### 7.3 Abhaengigkeit von anderen Fixes

| Fix | Beziehung |
|-----|-----------|
| **Fix 2: Tool Fallbacks** | Komplementaer - Fix 2 stellt sicher, dass Tools verfuegbar sind, Fix 1 stellt sicher, dass sie genutzt werden |
| **Fix 3: Memory-Guided Search** | Optional - Kann Verification durch bessere Suche unterstuetzen |

---

## 8. Open Questions

### Q1: Soll CoVe als separater Verification-Step implementiert werden?

**Kontext**: Chain-of-Verification (CoVe) ist ein 4-Stufen-Prozess (Draft -> Plan -> Verify -> Revise), der F1-Scores um 23% verbessert.

**Optionen**:
1. **Inline in Prompt**: CoVe-Prinzipien in den Verification Protocol Block integrieren
2. **Separater Step**: Expliziter CoVe-Call nach initialer Antwort
3. **On-Demand**: CoVe nur bei komplexen Fragen aktivieren

**Empfehlung**: Option 1 fuer Phase 1, Option 3 fuer Phase 2+

**Entscheidung**: [TBD]

---

### Q2: Wie messen wir "Quote-Rate"?

**Kontext**: Quote-Rate = Anteil der Antworten mit Code-Zitaten

**Optionen**:
1. **Regex-basiert**: Suche nach ` ``` ` mit Pfad-Prefix
2. **LLM-basiert**: Kleines Modell klassifiziert Antworten
3. **Manuell**: Stichproben-Review

**Empfehlung**: Option 1 fuer automatisierte Metriken, Option 3 fuer Qualitaets-Audits

**Entscheidung**: [TBD]

---

### Q3: Soll das Protocol fuer alle Fragen gelten oder nur Code-Fragen?

**Kontext**: Verification macht bei "Was ist die Uhrzeit?" wenig Sinn.

**Optionen**:
1. **Alle Fragen**: Einheitliches Verhalten
2. **Nur Code-Fragen**: Trigger-Erkennung erforderlich
3. **Proportional**: Verification-Tiefe nach Frage-Typ

**Empfehlung**: Option 2 mit explizitem Trigger ("code", "function", "implementation", etc.)

**Entscheidung**: [TBD]

---

### Q4: Wie handeln wir Context Compaction?

**Kontext**: Bei langen Sessions kann Claude kritische Instruktionen "vergessen".

**Optionen**:
1. **System-Critical-Marker**: XML-Tags, die Compaction ueberleben
2. **Periodic Reminder**: Verification-Reminder in User-Messages
3. **Session-Reset**: Regelmaessiger Hard-Reset des Contexts

**Empfehlung**: Option 1 mit Monitoring, Option 2 als Fallback

**Entscheidung**: [TBD]

---

## 9. Appendix

### A. Research-Referenzen

#### Akademische Papers

| Paper | Quelle | Key Finding |
|-------|--------|-------------|
| Chain-of-Verification Reduces Hallucination | [arXiv:2309.11495](https://arxiv.org/abs/2309.11495) | CoVe verbessert F1 um 23% |
| Universal Self-Consistency | [arXiv:2311.17311](https://arxiv.org/abs/2311.17311) | USC reduziert Sample-Groesse um 40% |
| CodeHalu Benchmark | [arXiv:2405.00253](https://arxiv.org/abs/2405.00253) | 4 Kategorien von Code-Halluzinationen |

#### Industrie-Best-Practices

| Quelle | Key Practice |
|--------|--------------|
| [Claude Docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) | "Investigate before answering" |
| [GitHub Copilot](https://github.blog/ai-and-ml/github-copilot/) | Penalisierung von deprecated APIs |
| [Microsoft LLM Guide](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/best-practices-for-mitigating-hallucinations-in-large-language-models-llms/4403129) | Temperatur 0.1-0.4, Metadata-Filtering |

### B. Multi-Layer-Ansatz Evidenz

> "A 2024 Stanford study found that combining RAG, RLHF, and guardrails led to a 96% reduction in hallucinations compared to baseline models." - [Voiceflow](https://www.voiceflow.com/blog/prevent-llm-hallucinations)

Die drei Saeulen:

| Saeule | Technik | Aufwand | Impact |
|--------|---------|---------|--------|
| **Prompt** | Verification Protocol, Quote-First | Gering | Hoch |
| **Self-Verification** | CoVe, Self-Consistency | Mittel | Mittel-Hoch |
| **Tool-Grounding** | RAG, Code-Index, Hooks | Hoch | Sehr Hoch |

### C. Glossar

| Begriff | Definition |
|---------|------------|
| **CoVe** | Chain-of-Verification - 4-Stufen Selbstpruefungs-Prozess |
| **Quote-Rate** | Anteil der Antworten mit exakten Code-Zitaten |
| **Halluzinationsrate** | Anteil der Antworten mit faktisch falschen Code-Behauptungen |
| **Context Compaction** | Kuerzung des Conversation-Contexts bei langen Sessions |
| **RAG** | Retrieval-Augmented Generation - Grounding durch Retrieval |

---

## Changelog

| Version | Datum | Aenderung | Autor |
|---------|-------|-----------|-------|
| 1.0 | 2026-01-04 | Initial Draft | Coding Brain Team |

---

*Dieses PRD basiert auf der Analyse in [FIX-01-VERIFICATION-PROTOCOL-ANALYSIS.md](./FIX-01-VERIFICATION-PROTOCOL-ANALYSIS.md) und den darin referenzierten Research-Ergebnissen.*
