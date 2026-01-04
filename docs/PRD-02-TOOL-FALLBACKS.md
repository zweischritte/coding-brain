# PRD: Tool Fallback-Kaskaden fuer Coding Brain

## Dokumentinformationen

| Feld | Wert |
|------|------|
| PRD-ID | PRD-02 |
| Titel | Tool Fallback-Kaskaden |
| Status | Implemented |
| Autor | System |
| Erstellt | 2026-01-04 |
| Implementiert | 2026-01-04 |
| Basiert auf | FIX-02-TOOL-FALLBACKS-ANALYSIS.md |

---

## Implementation Changelog

This section documents all file changes made during implementation for tracking and potential rollback.

### Files Created

| File | Description |
|------|-------------|
| `openmemory/api/tools/fallback_find_callers.py` | FallbackFindCallersTool with 4-stage cascade (Graph → Grep → Semantic → Error), GrepTool, GrepMatch, FallbackConfig classes |
| `openmemory/api/tools/tests/test_fallback_find_callers.py` | Unit tests for FallbackFindCallersTool, keyword extraction, config defaults |
| `openmemory/api/tests/mcp/test_fallback_find_callers_mcp.py` | MCP integration tests for find_callers with fallback |

### Files Modified

| File | Changes |
|------|---------|
| `CLAUDE.md` | Added "Tool Failure Protocol" section with explicit instructions to never hallucinate on tool errors, example of correct vs incorrect behavior, fallback cascade documentation |
| `openmemory/api/tools/call_graph.py` | Enhanced `SymbolNotFoundError` with `suggestions` array, `to_dict()` method, `_build_suggestions()` and `_extract_symbol_from_message()` helpers. Added `fallback_stage`, `fallback_strategy`, `warning` fields to `ResponseMeta`. Added `suggestions` field to `GraphOutput` |
| `openmemory/api/app/mcp_server.py` | Updated `find_callers()` function with `use_fallback: bool = True` parameter, FallbackFindCallersTool integration, enhanced SymbolNotFoundError handling with structured JSON response |
| `openmemory/api/tools/__init__.py` | Added exports: `FallbackConfig`, `FallbackFindCallersTool`, `GrepTool`, `GrepMatch`, `create_fallback_find_callers_tool` |

### Key Implementation Details

1. **FallbackFindCallersTool** (`fallback_find_callers.py`):
   - 4-stage cascade: Graph → Grep → Semantic Search → Structured Error
   - Configurable timeouts (150ms per stage, 500ms total)
   - Integrates with existing `ServiceCircuitBreaker` from `resilience/circuit_breaker.py`
   - Keyword extraction from camelCase/snake_case for semantic search

2. **Enhanced SymbolNotFoundError** (`call_graph.py`):
   - `suggestions` array with actionable alternatives
   - `to_dict()` method returning structured JSON with `error`, `symbol`, `suggestions`, `next_actions`
   - Extracts symbol name from error message if not provided

3. **MCP Integration** (`mcp_server.py`):
   - `find_callers(use_fallback=True)` parameter controls fallback behavior
   - Returns `_fallback_info` when fallback is used
   - Structured error response on SymbolNotFoundError

4. **Reused Components**:
   - Circuit Breaker from `resilience/circuit_breaker.py` (already existed)
   - No new circuit breaker implementation needed

### Test Results

- 138 existing MCP tests: **PASSED**
- 2 new fallback tests: **PASSED**
- 11 tests skipped due to pre-existing import infrastructure issues

---

## 1. Problem Statement

### 1.1 Kernproblem

Wenn Code-Intelligence-Tools wie `find_callers()` ein Symbol nicht finden, geben sie eine einfache Fehlermeldung zurueck. Die KI interpretiert dies nicht als "Information fehlt", sondern halluziniert eine Antwort - mit potenziell fatalen Konsequenzen fuer die Code-Analyse.

### 1.2 Konkretes Beispiel aus der Analyse

**Szenario:** Analyse des Evidence-Upload-Flows in einer NestJS/React-Anwendung.

**Tool-Aufruf:**
```
find_callers(symbol_name="moveFilesToPermanentStorage")
```

**Tool-Antwort:**
```
Symbol not found: moveFilesToPermanentStorage
```

**KI-Reaktion (Halluzination):**
> "Die Methode wird automatisch beim Speichern des Reports aufgerufen."

**Realitaet:** Die Methode wird von einem Event-Consumer (`FeesReportConsumerController`) aufgerufen, der auf Dapr Pub/Sub Events reagiert - ein asynchroner, Event-Driven Flow, kein synchroner Resolver-Call.

### 1.3 Technische Ursachen

1. **AST Parser ignoriert Decorators**: Event-Handler-Registrierungen (`@OnEvent('report.created')`) werden nicht als Call-Edges erkannt
2. **Call-Edge-Inferenz basiert auf Regex**: Keine Aufloesung von Dependency Injection oder Event-Subscriptions
3. **Kein Fallback bei "Symbol not found"**: Die Exception wird ohne Alternativvorschlaege geworfen

### 1.4 Auswirkungen

| Auswirkung | Beschreibung |
|------------|--------------|
| Falsche Architektur-Behauptungen | KI behauptet synchronen Flow bei asynchronem Event-Flow |
| Vertrauensverlust | Coding Brain verlor gegen Gemini ohne Memory bei Code-Tracing |
| Cascading Errors | Alle Folgeschlussfolgerungen basieren auf falscher Annahme |

---

## 2. Goals & Success Metrics

### 2.1 Primaeres Ziel

**Kein Tool-Fehler fuehrt zu Halluzination.** Stattdessen wird eine strukturierte Fallback-Kaskade durchlaufen, die entweder ein alternatives Ergebnis oder eine explizite Fehlermeldung mit Handlungsempfehlungen liefert.

### 2.2 Success Metrics

| Metrik | Baseline | Ziel | Messmethode |
|--------|----------|------|-------------|
| Halluzinationsrate nach Tool-Fehler | ~80% | <5% | Manuelles Review von Tool-Failure-Sessions |
| Fallback-Nutzungsrate | 0% | 100% bei Tool-Fehlern | Logging der Fallback-Kaskade |
| Erfolgsrate nach Fallback | N/A | >60% | Anteil der Fallbacks mit verwertbarem Ergebnis |
| Fallback-Kaskade Latenz | N/A | <500ms gesamt | Performance Monitoring |
| User-Satisfaction nach Tool-Fehler | Niedrig | Neutral/Positiv | User-Feedback |

### 2.3 Non-Goals

- Vollstaendige Decorator-Indexierung (siehe PRD-04)
- Event-Registry fuer SUBSCRIBES/PUBLISHES Edges (separates PRD)
- Cross-Repository-Navigation

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: FallbackFindCallersTool Klasse

Eine Wrapper-Klasse um die bestehenden Code-Intelligence-Tools, die automatisch eine Fallback-Kaskade durchlaeuft.

**Akzeptanzkriterien:**
- [ ] Implementiert 4-stufige Fallback-Kaskade (Graph -> Grep -> Semantic -> Error)
- [ ] Setzt `fallback_used` Flag bei Nutzung einer Fallback-Stufe
- [ ] Liefert strukturierte `suggestions` bei finalem Fehlschlag
- [ ] Logging jeder Fallback-Stufe fuer Metriken

#### FR-2: Circuit Breaker Pattern pro Repository

Verhindert kaskadierende Fehler bei wiederholten Tool-Fehlern fuer ein Repository.

**Akzeptanzkriterien:**
- [ ] Circuit Breaker mit States: CLOSED, OPEN, HALF_OPEN
- [ ] Konfigurierbare Thresholds (failure_threshold, reset_timeout)
- [ ] Pro Repository ein separater Breaker
- [ ] Automatischer Wechsel zu Fallback bei OPEN State

#### FR-3: Strukturierte Fehlermeldungen mit Suggestions

Jede `SymbolNotFoundError` enthaelt konkrete Handlungsempfehlungen.

**Akzeptanzkriterien:**
- [ ] `suggestions` Array mit mindestens 3 konkreten Alternativen
- [ ] Jede Suggestion ist ausfuehrbar (Tool-Name + Query)
- [ ] Erklaerung warum das Symbol fehlen koennte

#### FR-4: MCP-Tool Response Enhancement

MCP-Tools geben bei Fehlern strukturierte Fallback-Informationen zurueck.

**Akzeptanzkriterien:**
- [ ] Response-Format mit `error`, `fallback_strategy`, `next_actions`
- [ ] `next_actions` als Array von Tool-Aufrufen
- [ ] `explanation` mit kontextbezogener Fehlerbeschreibung

#### FR-5: Prompt-Ergaenzung fuer Tool Failure Protocol

CLAUDE.md und andere Prompt-Files enthalten klare Anweisungen zum Umgang mit Tool-Fehlern.

**Akzeptanzkriterien:**
- [ ] "Tool Failure Protocol" Section in CLAUDE.md
- [ ] Explizites Verbot von Halluzinationen bei Tool-Fehlern
- [ ] Beispiele fuer korrektes vs. falsches Verhalten

### 3.2 Non-Functional Requirements

#### NFR-1: Performance

| Anforderung | Spezifikation |
|-------------|---------------|
| Fallback-Kaskade Gesamtlatenz | <500ms |
| Einzelne Fallback-Stufe | <150ms |
| Circuit Breaker Check | <1ms |

#### NFR-2: Reliability

| Anforderung | Spezifikation |
|-------------|---------------|
| Tool-Verfuegbarkeit | 99.9% - Tool darf nie ohne Antwort enden |
| Graceful Degradation | Immer mindestens strukturierte Fehlermeldung |
| Recovery nach Circuit Breaker Open | Automatisch nach reset_timeout |

#### NFR-3: Observability

| Anforderung | Spezifikation |
|-------------|---------------|
| Logging | Jede Fallback-Stufe geloggt mit Timing |
| Metriken | fallback_used, fallback_stage, success_rate |
| Alerting | Bei >20% Fallback-Rate pro Repository |

---

## 4. Technical Specification

### 4.1 Fallback-Kaskade

```
Stufe 1: Graph-basierte Suche (SCIP)
   |
   v Fehler: "Symbol not found"
   |
Stufe 2: Grep nach Symbolname
   -> grep -r "symbolName" --include="*.ts"
   |
   v Keine Treffer oder zu viele (>50)
   |
Stufe 3: Semantische Suche
   -> search_code_hybrid(query="symbol context keywords")
   |
   v Keine relevanten Treffer
   |
Stufe 4: Strukturierte Fehlermeldung
   -> Suggestions + Explanation + degraded_mode=true
```

### 4.2 Circuit Breaker Implementation

```python
class CircuitBreaker:
    """Prevents cascading failures in tool calls."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Fail fast, use fallback
    HALF_OPEN = "half_open" # Testing recovery

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = self.CLOSED
        self.last_failure_time: Optional[float] = None

    def execute(self, primary_func: Callable, fallback_func: Callable) -> Any:
        if self.state == self.OPEN:
            if self._should_attempt_reset():
                self.state = self.HALF_OPEN
            else:
                return fallback_func()

        try:
            result = primary_func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            return fallback_func()

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.reset_timeout

    def _record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN

    def _record_success(self) -> None:
        self.failure_count = 0
        self.state = self.CLOSED
```

**Parameter:**
| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| failure_threshold | 5 | Fehler bis Circuit oeffnet |
| reset_timeout | 30s | Zeit bis HALF_OPEN Versuch |

**State Transitions:**
- CLOSED -> OPEN: Nach `failure_threshold` Fehlern
- OPEN -> HALF_OPEN: Nach `reset_timeout` Sekunden
- HALF_OPEN -> CLOSED: Bei Erfolg
- HALF_OPEN -> OPEN: Bei erneutem Fehler

### 4.3 FallbackFindCallersTool Implementation

```python
from typing import List, Optional
import uuid
import time
import logging

from openmemory.api.app.routers.code.schemas import (
    CallGraphInput,
    GraphOutput,
    ResponseMeta,
)
from openmemory.api.app.routers.code.tools.find_callers import FindCallersTool
from openmemory.api.app.routers.code.exceptions import SymbolNotFoundError

logger = logging.getLogger(__name__)


class FallbackFindCallersTool:
    """Find callers with automatic fallback cascade."""

    def __init__(
        self,
        graph_driver,
        grep_tool,
        search_tool,
        circuit_breakers: Optional[dict] = None,
    ):
        self.graph_driver = graph_driver
        self.grep_tool = grep_tool
        self.search_tool = search_tool
        self.circuit_breakers = circuit_breakers or {}
        self.fallback_used = False
        self.fallback_stage: Optional[int] = None

    def _get_breaker(self, repo_id: str) -> CircuitBreaker:
        if repo_id not in self.circuit_breakers:
            self.circuit_breakers[repo_id] = CircuitBreaker()
        return self.circuit_breakers[repo_id]

    def find(self, input_data: CallGraphInput) -> GraphOutput:
        start_time = time.time()
        self.fallback_used = False
        self.fallback_stage = None

        breaker = self._get_breaker(input_data.repo_id)

        # Stufe 1: Graph-basierte Suche (mit Circuit Breaker)
        try:
            if breaker.state != CircuitBreaker.OPEN:
                result = self._graph_search(input_data)
                if result.nodes:
                    breaker._record_success()
                    self._log_stage(1, start_time, success=True)
                    return result
                breaker._record_failure()
        except SymbolNotFoundError:
            breaker._record_failure()
            logger.info(f"Stage 1 failed: Symbol not found in graph")

        self.fallback_used = True

        # Stufe 2: Grep-Fallback
        self.fallback_stage = 2
        grep_result = self._grep_fallback(input_data.symbol_name, input_data.repo_id)
        if grep_result and 0 < len(grep_result) <= 50:
            self._log_stage(2, start_time, success=True)
            return self._convert_grep_to_graph_output(grep_result, input_data)

        # Stufe 3: Semantische Suche
        self.fallback_stage = 3
        semantic_result = self._semantic_fallback(
            input_data.symbol_name, input_data.repo_id
        )
        if semantic_result:
            self._log_stage(3, start_time, success=True)
            return self._convert_semantic_to_graph_output(semantic_result, input_data)

        # Stufe 4: Strukturierte Fehlermeldung
        self.fallback_stage = 4
        self._log_stage(4, start_time, success=False)
        return self._build_error_response(input_data)

    def _graph_search(self, input_data: CallGraphInput) -> GraphOutput:
        """Stufe 1: Graph-basierte Suche ueber SCIP-Index."""
        tool = FindCallersTool(graph_driver=self.graph_driver)
        return tool.find(input_data)

    def _grep_fallback(
        self, symbol_name: str, repo_id: str
    ) -> Optional[List[dict]]:
        """Stufe 2: Grep-basierte Suche nach Symbolname."""
        try:
            return self.grep_tool.search(
                pattern=symbol_name,
                repo_id=repo_id,
                include_patterns=["*.ts", "*.tsx", "*.py", "*.js", "*.jsx"],
                max_results=100,
            )
        except Exception as e:
            logger.warning(f"Grep fallback failed: {e}")
            return None

    def _semantic_fallback(
        self, symbol_name: str, repo_id: str
    ) -> Optional[List[dict]]:
        """Stufe 3: Semantische Suche mit Embeddings."""
        try:
            # Extrahiere Keywords aus camelCase/snake_case
            keywords = self._extract_keywords(symbol_name)
            query = " ".join(keywords)
            return self.search_tool.search_hybrid(
                query=query,
                repo_id=repo_id,
                limit=20,
            )
        except Exception as e:
            logger.warning(f"Semantic fallback failed: {e}")
            return None

    def _extract_keywords(self, symbol_name: str) -> List[str]:
        """Extrahiert Keywords aus camelCase/snake_case."""
        import re
        # Split by underscore or camelCase
        words = re.split(r'_|(?=[A-Z])', symbol_name)
        return [w.lower() for w in words if w]

    def _convert_grep_to_graph_output(
        self, grep_result: List[dict], input_data: CallGraphInput
    ) -> GraphOutput:
        """Konvertiert Grep-Ergebnisse in GraphOutput."""
        nodes = []
        for match in grep_result[:20]:  # Limit
            nodes.append({
                "id": f"grep:{match['file']}:{match['line']}",
                "name": f"{match['file']}:{match['line']}",
                "type": "grep_match",
                "file_path": match["file"],
                "line_number": match["line"],
                "context": match.get("context", ""),
            })

        return GraphOutput(
            nodes=nodes,
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                fallback_stage=2,
                fallback_strategy="grep",
            ),
        )

    def _convert_semantic_to_graph_output(
        self, semantic_result: List[dict], input_data: CallGraphInput
    ) -> GraphOutput:
        """Konvertiert semantische Suchergebnisse in GraphOutput."""
        nodes = []
        for match in semantic_result[:10]:
            nodes.append({
                "id": f"semantic:{match['symbol_id']}",
                "name": match.get("symbol_name", "unknown"),
                "type": "semantic_match",
                "file_path": match.get("file_path"),
                "score": match.get("score"),
            })

        return GraphOutput(
            nodes=nodes,
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                fallback_stage=3,
                fallback_strategy="semantic_search",
            ),
        )

    def _build_error_response(self, input_data: CallGraphInput) -> GraphOutput:
        """Stufe 4: Strukturierte Fehlermeldung mit Suggestions."""
        symbol_name = input_data.symbol_name or input_data.symbol_id

        suggestions = [
            f"Try: grep -r '{symbol_name}' --include='*.ts' --include='*.py'",
            f"Try: search_code_hybrid(query='{symbol_name}')",
            "Symbol may be called via decorator (@OnEvent, @Subscribe, @EventHandler)",
            "Symbol may be injected via DI (constructor injection)",
            "Index may be stale - consider re-indexing with index_codebase(reset=true)",
        ]

        return GraphOutput(
            nodes=[],
            edges=[],
            meta=ResponseMeta(
                request_id=str(uuid.uuid4()),
                degraded_mode=True,
                fallback_stage=4,
                fallback_strategy="structured_error",
                missing_sources=["graph_index", "grep", "semantic_search"],
            ),
            suggestions=suggestions,
        )

    def _log_stage(self, stage: int, start_time: float, success: bool) -> None:
        """Logging fuer Metriken."""
        elapsed = time.time() - start_time
        logger.info(
            f"Fallback cascade stage={stage} success={success} elapsed_ms={elapsed*1000:.2f}"
        )
```

### 4.4 API Changes

#### 4.4.1 Enhanced SymbolNotFoundError

```python
class SymbolNotFoundError(CallGraphError):
    """Error with structured suggestions for fallback."""

    def __init__(self, symbol_name: str, repo_id: str = None):
        self.symbol_name = symbol_name
        self.repo_id = repo_id
        self.suggestions = [
            f"Try: grep -r '{symbol_name}' --include='*.ts'",
            f"Try: search_code_hybrid(query='{symbol_name}')",
            "Symbol may be called via decorator (@OnEvent, @Subscribe)",
            "Symbol may be injected via DI (constructor injection)",
        ]
        super().__init__(
            f"Symbol not found: {symbol_name}. "
            f"Suggestions: {'; '.join(self.suggestions[:2])}"
        )

    def to_dict(self) -> dict:
        return {
            "error": "SYMBOL_NOT_FOUND",
            "symbol": self.symbol_name,
            "repo_id": self.repo_id,
            "suggestions": self.suggestions,
        }
```

#### 4.4.2 MCP Response Format

```python
# Bei Fehler:
{
    "error": "SYMBOL_NOT_FOUND",
    "symbol": "moveFilesToPermanentStorage",
    "fallback_strategy": "RECOMMENDED",
    "next_actions": [
        {"tool": "grep", "query": "moveFilesToPermanentStorage", "include": "*.ts"},
        {"tool": "search_code_hybrid", "query": "move files permanent storage"},
    ],
    "explanation": (
        "The symbol was not found in the indexed graph. "
        "This may happen with decorators, event handlers, or dependency injection. "
        "Use the suggested fallback tools to locate the symbol."
    )
}

# Bei Fallback-Erfolg:
{
    "nodes": [...],
    "edges": [...],
    "meta": {
        "request_id": "...",
        "degraded_mode": true,
        "fallback_stage": 2,
        "fallback_strategy": "grep",
        "warning": "Results from grep fallback - may include false positives"
    }
}
```

### 4.5 Prompt-Ergaenzung (CLAUDE.md)

```markdown
## Tool Failure Protocol

Wenn ein Tool "Symbol not found" oder aehnliche Fehler zurueckgibt:

1. **NIEMALS** die Information erraten oder halluzinieren
2. **SOFORT** Fallback-Suche durchfuehren:
   - Nutze `search_code_hybrid(query="<symbol_name>")`
   - Oder manuelle Grep-Suche via Bash
3. Wenn alle Fallbacks scheitern: **EXPLIZIT** kommunizieren

### Beispiel: Korrektes Verhalten

**Tool-Antwort:**
```
Symbol not found: moveFilesToPermanentStorage
```

**FALSCH:**
> "Die Methode wird automatisch aufgerufen."

**RICHTIG:**
> "Ich konnte den Aufrufer nicht im Graph finden. Das kann passieren bei:
> - Event-basierten Aufrufen (z.B. @OnEvent Decorator)
> - Dependency Injection
> - Dynamischen Aufrufen
>
> Ich fuehre eine Grep-Suche durch..."
> [Fuehrt search_code_hybrid oder Grep aus]
```

---

## 5. Implementation Plan

### Phase 1: Sofort (Woche 1)

| Task | Beschreibung | Owner | Status |
|------|--------------|-------|--------|
| P1-1 | Prompt-Ergaenzung in CLAUDE.md | - | TODO |
| P1-2 | Enhanced SymbolNotFoundError mit suggestions | - | TODO |
| P1-3 | MCP-Tool Response mit fallback_strategy | - | TODO |

**Deliverables:**
- Aktualisierte CLAUDE.md mit Tool Failure Protocol
- SymbolNotFoundError mit suggestions Array
- MCP find_callers mit next_actions bei Fehler

### Phase 2: Fallback-Kaskade (Woche 2-4)

| Task | Beschreibung | Owner | Status |
|------|--------------|-------|--------|
| P2-1 | CircuitBreaker Klasse implementieren | - | TODO |
| P2-2 | FallbackFindCallersTool Klasse | - | TODO |
| P2-3 | Grep-Tool Integration | - | TODO |
| P2-4 | Semantic Search Integration | - | TODO |
| P2-5 | Unit Tests fuer alle Fallback-Stufen | - | TODO |
| P2-6 | Integration Tests End-to-End | - | TODO |

**Deliverables:**
- Vollstaendige FallbackFindCallersTool Implementation
- CircuitBreaker pro Repository
- Test Coverage >80%

### Phase 3: Monitoring & Alerting (Woche 5)

| Task | Beschreibung | Owner | Status |
|------|--------------|-------|--------|
| P3-1 | Fallback-Metriken Logging | - | TODO |
| P3-2 | Prometheus/Grafana Dashboard | - | TODO |
| P3-3 | Alerting bei hoher Fallback-Rate | - | TODO |
| P3-4 | Documentation Update | - | TODO |

**Deliverables:**
- Dashboard mit Fallback-Metriken
- Alert bei >20% Fallback-Rate pro Repo
- Aktualisierte Dokumentation

---

## 6. Code Examples

### 6.1 MCP Server Integration

```python
# In openmemory/api/app/mcp/mcp_server.py

@mcp.tool()
async def find_callers(
    repo_id: str,
    symbol_name: str = None,
    symbol_id: str = None,
    depth: int = 2,
) -> dict:
    """Find functions that call a given symbol with automatic fallback."""
    try:
        input_data = CallGraphInput(
            repo_id=repo_id,
            symbol_name=symbol_name,
            symbol_id=symbol_id,
            depth=depth,
        )

        tool = FallbackFindCallersTool(
            graph_driver=get_graph_driver(),
            grep_tool=get_grep_tool(),
            search_tool=get_search_tool(),
            circuit_breakers=app_state.circuit_breakers,
        )

        result = tool.find(input_data)

        response = result.to_dict()
        if tool.fallback_used:
            response["_fallback_info"] = {
                "fallback_used": True,
                "fallback_stage": tool.fallback_stage,
                "warning": "Results from fallback strategy - verify accuracy",
            }

        return response

    except SymbolNotFoundError as e:
        return {
            "error": "SYMBOL_NOT_FOUND",
            "symbol": symbol_name or symbol_id,
            "fallback_strategy": "RECOMMENDED",
            "next_actions": [
                {"tool": "grep", "query": symbol_name},
                {"tool": "search_code_hybrid", "query": symbol_name},
            ],
            "explanation": str(e),
            "suggestions": e.suggestions,
        }
```

### 6.2 Repository-spezifischer Circuit Breaker

```python
# In openmemory/api/app/state.py

from openmemory.api.app.routers.code.circuit_breaker import CircuitBreaker

class AppState:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(self, repo_id: str) -> CircuitBreaker:
        if repo_id not in self.circuit_breakers:
            self.circuit_breakers[repo_id] = CircuitBreaker(
                failure_threshold=5,
                reset_timeout=30,
            )
        return self.circuit_breakers[repo_id]

    def reset_breaker(self, repo_id: str) -> None:
        """Reset breaker after successful re-indexing."""
        if repo_id in self.circuit_breakers:
            self.circuit_breakers[repo_id] = CircuitBreaker()


app_state = AppState()
```

---

## 7. Risks & Mitigations

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Fallback-Kaskade zu langsam | Mittel | Hoch | Timeout pro Stufe (150ms), parallele Ausfuehrung moeglich |
| Grep-Fallback liefert zu viele Ergebnisse | Hoch | Mittel | Max 50 Treffer, danach semantische Suche |
| False Positives durch Grep | Hoch | Mittel | Warning in Response, degraded_mode Flag |
| Circuit Breaker oeffnet zu frueh | Niedrig | Mittel | Konfigurierbare Thresholds, Monitoring |
| Semantische Suche findet falsche Symbole | Mittel | Mittel | Score-Threshold, manuelle Verifikation empfohlen |

### 7.1 Mitigation Details

**Fallback-Kaskade Latenz:**
- Timeout pro Stufe: 150ms
- Gesamttimeout: 500ms
- Bei Timeout: Naechste Stufe
- Option: Parallele Ausfuehrung von Stufe 2+3

**Grep Overflow:**
```python
grep_result = self.grep_tool.search(...)
if len(grep_result) > 50:
    # Zu viele Treffer - weiter zu semantischer Suche
    logger.info(f"Grep returned {len(grep_result)} results, switching to semantic")
    return None
```

---

## 8. Dependencies

### 8.1 Interne Abhaengigkeiten

| Abhaengigkeit | PRD | Status | Beschreibung |
|---------------|-----|--------|--------------|
| Decorator Indexing | PRD-04 | Geplant | Vollstaendige Graph-Abdeckung fuer Event-Handler |
| Event Registry | - | Nicht geplant | SUBSCRIBES/PUBLISHES Edges |

### 8.2 Externe Abhaengigkeiten

| Komponente | Erforderlich fuer | Status |
|------------|-------------------|--------|
| OpenSearch | Semantische Suche | Vorhanden |
| Neo4j | Graph-Suche | Vorhanden |
| Grep Tool | Fallback Stufe 2 | Zu implementieren |

### 8.3 Beziehung zu PRD-04

PRD-04 (Decorator Indexing) reduziert die Haeufigkeit von "Symbol not found" Fehlern, indem Event-Handler und DI-Patterns korrekt indexiert werden. PRD-02 (dieses Dokument) stellt sicher, dass auch bei unvollstaendigem Index eine nutzbare Antwort geliefert wird.

**Empfohlene Reihenfolge:**
1. PRD-02 (Fallbacks) - Sofortige Verbesserung der User Experience
2. PRD-04 (Decorator Indexing) - Reduziert Fallback-Notwendigkeit langfristig

---

## 9. Open Questions

| Frage | Kontext | Entscheidung |
|-------|---------|--------------|
| Sollen Fallback-Ergebnisse gecached werden? | Wiederholte Anfragen | TBD |
| Wie werden Cross-Repo Symbole behandelt? | Multi-Repo Projekte | Out of Scope |
| Soll der User ueber Fallback informiert werden? | Transparency vs. UX | Ja, via degraded_mode |

---

## 10. Appendix

### 10.1 Referenzen

- [Sourcegraph: Precise vs. Search-Based Navigation](https://sourcegraph.com/docs/code-navigation/precise-code-navigation)
- [LangChain: How to handle tool errors](https://python.langchain.com/docs/how_to/tools_error/)
- [Microservices.io: Circuit Breaker Pattern](https://microservices.io/patterns/reliability/circuit-breaker.html)
- [Microsoft: Implementing Circuit Breaker Pattern](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/implement-resilient-applications/implement-circuit-breaker-pattern)

### 10.2 Verwandte Dokumente

- `/docs/FIX-02-TOOL-FALLBACKS-ANALYSIS.md` - Analyse-Dokument
- `/docs/PRD-04-DECORATOR-INDEXING.md` - Decorator Indexing PRD (geplant)

---

*Erstellt: 2026-01-04*
*Basierend auf: FIX-02-TOOL-FALLBACKS-ANALYSIS.md*
