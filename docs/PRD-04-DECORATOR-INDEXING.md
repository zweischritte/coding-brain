# PRD: NestJS Decorator Indexing fuer Coding Brain

**Version:** 1.0
**Date:** 2026-01-04
**Status:** ✅ IMPLEMENTED
**Author:** Generated from FIX-04-DECORATOR-INDEXING-ANALYSIS.md

---

## Implementation Summary (2026-01-04)

### Files Modified

#### 1. `openmemory/api/indexing/ast_parser.py`

**New Data Structures (lines 92-122):**
- Added `DecoratorInfo` dataclass with fields: `name`, `decorator_type`, `arguments`, `raw_text`
- Added `decorators: list[DecoratorInfo]` field to `Symbol` dataclass

**KNOWN_DECORATORS Dictionary (lines 433-505):**
- Added 60+ NestJS/Angular decorator type mappings including:
  - Event handlers: `OnEvent`, `OnQueueEvent`, `Process`
  - HTTP handlers: `Get`, `Post`, `Put`, `Patch`, `Delete`, `Head`, `Options`, `All`
  - Controllers: `Controller`, `Resolver`
  - Microservices: `EventPattern`, `MessagePattern`, `GrpcMethod`
  - DI: `Injectable`, `Inject`, `Optional`, `Global`
  - Middleware: `UseGuards`, `UseInterceptors`, `UsePipes`, `UseFilters`
  - Scheduling: `Cron`, `Interval`, `Timeout`
  - WebSockets: `WebSocketGateway`, `SubscribeMessage`
  - Angular: `Component`, `Directive`, `Pipe`, `NgModule`, etc.

**TypeScriptPlugin Methods (lines 511-813):**
- Modified `_extract_class()` to call `_extract_decorators_from_node()` and populate `decorators` field
- Modified `_extract_method()` to call `_extract_decorators_from_node()` and populate `decorators` field
- Added `_extract_decorator_info()` method for parsing single decorator nodes
- Added `_extract_call_arguments()` method for extracting decorator arguments
- Added `_extract_decorators_from_node()` method handling three AST patterns:
  1. Method decorators in class body (siblings)
  2. Exported decorated classes (siblings inside export_statement)
  3. Non-exported decorated classes (children before 'class' keyword)

**Event Registry Infrastructure (lines 1089-1274):**
- Added `EventEdge` dataclass for publisher->subscriber connections
- Added `EventRegistry` class with `register_publisher()`, `register_subscriber()`, `get_edges()` methods
- Added `_EMIT_PATTERN` regex for detecting `.emit('event.name')` calls
- Added `discover_event_publishers()` function for finding emit() calls in source
- Added `discover_event_subscribers()` function for collecting @OnEvent handlers
- Added `build_event_registry()` function for building registry from parsed symbols
- Added `generate_event_edges()` function for Neo4j-compatible edge generation

#### 2. `openmemory/api/indexing/graph_projection.py`

**CodeEdgeType Enum (line 79):**
- Added `TRIGGERS_EVENT = "TRIGGERS_EVENT"` edge type for event-based call edges

#### 3. `openmemory/api/indexing/tests/test_ast_parser.py`

**New Imports (lines 44-50):**
- Added imports for `DecoratorInfo`, `EventEdge`, `EventRegistry`, `KNOWN_DECORATORS`
- Added imports for `discover_event_publishers`, `discover_event_subscribers`, `build_event_registry`, `generate_event_edges`

**New Test Classes:**
- `TestTypeScriptDecorators` (lines 641-877): 11 tests for decorator extraction
- `TestDecoratorInfo` (lines 880-902): 2 tests for DecoratorInfo dataclass
- `TestKnownDecorators` (lines 905-933): 4 tests for KNOWN_DECORATORS mapping
- `TestEventEdge` (lines 941-964): 2 tests for EventEdge dataclass
- `TestEventRegistry` (lines 967-1041): 9 tests for EventRegistry class
- `TestDiscoverEventPublishers` (lines 1044-1118): 4 tests for emit() detection
- `TestDiscoverEventSubscribers` (lines 1121-1201): 3 tests for @OnEvent detection
- `TestBuildEventRegistry` (lines 1204-1253): 1 integration test
- `TestGenerateEventEdges` (lines 1256-1274): 1 test for Neo4j edge generation

### Test Results
- **110 tests pass** in test_ast_parser.py (36 new tests added)
- **66 tests pass** in test_graph_projection.py (no regressions)

### Reverting This Implementation

To revert this implementation while preserving other changes:

1. In `ast_parser.py`:
   - Remove `DecoratorInfo` dataclass (lines 92-106)
   - Remove `decorators` field from `Symbol` dataclass (line 122)
   - Remove `KNOWN_DECORATORS` dict (lines 433-505)
   - Revert `_extract_class()` to not call `_extract_decorators_from_node()`
   - Revert `_extract_method()` to not call `_extract_decorators_from_node()`
   - Remove methods: `_extract_decorator_info()`, `_extract_call_arguments()`, `_extract_decorators_from_node()`
   - Remove Event Registry section (lines 1089-1274)

2. In `graph_projection.py`:
   - Remove `TRIGGERS_EVENT` from `CodeEdgeType` enum (line 79)

3. In `test_ast_parser.py`:
   - Remove new imports (lines 44-50)
   - Remove test classes: `TestTypeScriptDecorators`, `TestDecoratorInfo`, `TestKnownDecorators`, `TestEventEdge`, `TestEventRegistry`, `TestDiscoverEventPublishers`, `TestDiscoverEventSubscribers`, `TestBuildEventRegistry`, `TestGenerateEventEdges`

---

## 1. Problem Statement

### 1.1 Aktuelles Verhalten

Der Coding Brain Indexer (`openmemory/api/indexing/ast_parser.py`) erkennt NestJS-Decorators nicht. Die `TypeScriptPlugin._extract_from_node()` Methode verarbeitet nur syntaktische Node-Typen wie `function_declaration`, `class_declaration`, `method_definition`, aber ignoriert Decorator-Nodes vollstaendig.

**Konkretes Beispiel:**

```typescript
// Das sieht der Indexer:
class FileProcessingService {
  moveFilesToPermanentStorage() { ... }
}

// Das versteht er NICHT:
@OnEvent('file.uploaded')
handleFileUploaded(payload: FileUploadedEvent) {
  this.moveFilesToPermanentStorage();  // Kein CALLS-Edge zu Event-Emitter
}
```

### 1.2 Technische Ursachen

1. **Decorator-Blindheit**: Der `decorator` Node-Typ wird in `_extract_from_node()` nie behandelt
2. **Nur explizite Methoden**: `method_definition` wird erkannt, aber Decorator-Argumente werden nicht extrahiert
3. **Keine DI-Aufloesung**: Call-Edge-Inferenz nutzt Regex, Constructor-Injection wird nicht erkannt

### 1.3 Betroffene NestJS Patterns

| Pattern | Decorator | Problem |
|---------|-----------|---------|
| Event-Handler | `@OnEvent('event.name')` | Event-Emitter -> Handler unsichtbar |
| Controller-Endpunkte | `@Get()`, `@Post()`, `@Controller()` | HTTP-Route -> Handler unsichtbar |
| Dependency Injection | `@Injectable()`, `@Inject()` | Service-Abhaengigkeiten unsichtbar |
| Microservice-Kommunikation | `@EventPattern()`, `@MessagePattern()` | Inter-Service-Kommunikation unsichtbar |
| Guards & Interceptors | `@UseGuards()`, `@UseInterceptors()` | Middleware-Ketten unsichtbar |
| Scheduled Tasks | `@Cron()`, `@Interval()` | Zeitgesteuerte Aufrufe unsichtbar |

### 1.4 Auswirkungen

- `find_callers()` findet keine Event-basierten Aufrufe
- `impact_analysis()` unterschaetzt betroffene Komponenten
- Call-Graph ist unvollstaendig fuer NestJS/Angular-Projekte
- Code-Intelligence liefert falsche Negative

---

## 2. Goals & Success Metrics

### 2.1 Primaerziele

1. **Decorator-Extraktion**: Alle NestJS-Decorators werden als Metadaten im Symbol-Objekt gespeichert
2. **Event-Edge-Inferenz**: `@OnEvent('x')` erzeugt Call-Edge von `emit('x')` zum Handler
3. **HTTP-Endpoint-Mapping**: `@Get('/users')` erzeugt Route-Metadaten
4. **DI-Graph-Extraktion**: Constructor-Injection wird zu Service-Abhaengigkeiten aufgeloest

### 2.2 Success Metrics

| Metrik | Baseline | Target | Messung |
|--------|----------|--------|---------|
| Decorator Coverage | 0% | >95% | `count(symbols.decorators) / count(decorators_in_source)` |
| Event-Edge-Count | 0 | >80% der Events | `count(SUBSCRIBES edges) / count(@OnEvent decorators)` |
| find_callers Success Rate | ~60% | >90% | Manuelle Validierung an 50 NestJS-Methoden |
| Indexing-Zeit Overhead | 0% | <20% | Benchmark vor/nach Implementation |

### 2.3 Non-Goals

- Dynamische Module (`forRoot()`, `register()`) - zu komplex fuer Phase 1
- Custom Decorator Resolution - erfordert Type-Analyse
- Cross-Repository Event-Tracking

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: Decorator-Feld im Symbol-Objekt

Die `Symbol` Dataclass muss um ein `decorators` Feld erweitert werden.

**Aktuell (Zeile 92-106 in ast_parser.py):**
```python
@dataclass
class Symbol:
    name: str
    symbol_type: SymbolType
    line_start: int
    line_end: int
    language: Language
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_name: Optional[str] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
```

**Neu:**
```python
@dataclass
class DecoratorInfo:
    """Information about a decorator."""
    name: str
    decorator_type: Optional[str] = None  # event_handler, http_handler, di_provider, etc.
    arguments: list[str] = field(default_factory=list)
    raw_text: Optional[str] = None

@dataclass
class Symbol:
    name: str
    symbol_type: SymbolType
    line_start: int
    line_end: int
    language: Language
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_name: Optional[str] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    decorators: list[DecoratorInfo] = field(default_factory=list)  # NEU
```

#### FR-2: NestJSDecoratorPlugin fuer ast_parser.py

Erweitere `TypeScriptPlugin` um Decorator-Extraktion mit bekannten NestJS-Mappings.

```python
class NestJSDecoratorPlugin:
    """Plugin fuer NestJS Decorator-Erkennung."""

    KNOWN_DECORATORS = {
        # Event-Handler
        "OnEvent": "event_handler",
        "OnQueueEvent": "event_handler",
        # HTTP
        "Get": "http_handler",
        "Post": "http_handler",
        "Put": "http_handler",
        "Patch": "http_handler",
        "Delete": "http_handler",
        "Head": "http_handler",
        "Options": "http_handler",
        "All": "http_handler",
        "Controller": "controller",
        # Microservices
        "EventPattern": "message_handler",
        "MessagePattern": "message_handler",
        # DI
        "Injectable": "di_provider",
        "Inject": "di_injection",
        "Optional": "di_optional",
        # Guards & Interceptors
        "UseGuards": "guard_chain",
        "UseInterceptors": "interceptor_chain",
        "UsePipes": "pipe_chain",
        "UseFilters": "filter_chain",
        # Scheduling
        "Cron": "scheduled_task",
        "Interval": "scheduled_task",
        "Timeout": "scheduled_task",
        # WebSockets
        "WebSocketGateway": "websocket_gateway",
        "SubscribeMessage": "websocket_handler",
    }

    def extract_decorator_info(self, node: Node, source: bytes) -> Optional[DecoratorInfo]:
        """Extrahiere Decorator-Name und Argumente aus AST-Node."""
        if node.type != "decorator":
            return None

        # Finde call_expression oder identifier
        call_expr = self._find_child_by_type(node, "call_expression")
        if call_expr:
            # @OnEvent('file.uploaded')
            func_node = call_expr.child_by_field_name("function")
            args_node = call_expr.child_by_field_name("arguments")
            if func_node:
                decorator_name = source[func_node.start_byte:func_node.end_byte].decode("utf-8")
                arguments = self._extract_arguments(args_node, source) if args_node else []
                return DecoratorInfo(
                    name=decorator_name,
                    decorator_type=self.KNOWN_DECORATORS.get(decorator_name),
                    arguments=arguments,
                    raw_text=source[node.start_byte:node.end_byte].decode("utf-8"),
                )
        else:
            # @Injectable (ohne Klammern)
            for child in node.children:
                if child.type == "identifier":
                    decorator_name = source[child.start_byte:child.end_byte].decode("utf-8")
                    return DecoratorInfo(
                        name=decorator_name,
                        decorator_type=self.KNOWN_DECORATORS.get(decorator_name),
                        arguments=[],
                        raw_text=source[node.start_byte:node.end_byte].decode("utf-8"),
                    )
        return None
```

#### FR-3: Event-Name-basierte Call-Edge-Inferenz

Neue Funktion zur Generierung von SUBSCRIBES/PUBLISHES Edges basierend auf Event-Namen.

```python
def infer_event_call_edges(symbols: list[Symbol]) -> list[tuple[Symbol, Symbol, str]]:
    """
    Inferiere Call-Edges zwischen Event-Emittern und Handlern.

    Returns:
        List of (emitter_symbol, handler_symbol, event_name) tuples
    """
    event_handlers: dict[str, Symbol] = {}

    # Phase 1: Sammle alle Event-Handler
    for sym in symbols:
        for dec in sym.decorators:
            if dec.decorator_type == "event_handler" and dec.arguments:
                # Entferne Quotes: "'file.uploaded'" -> "file.uploaded"
                event_name = dec.arguments[0].strip("'\"")
                event_handlers[event_name] = sym

    edges = []

    # Phase 2: Finde emit() Aufrufe
    for sym in symbols:
        # Suche in Symbol-Body nach emit('event.name') Aufrufen
        emit_calls = extract_emit_calls(sym)  # Separate Funktion
        for event_name in emit_calls:
            if event_name in event_handlers:
                edges.append((sym, event_handlers[event_name], event_name))

    return edges
```

#### FR-4: Event-Registry fuer SUBSCRIBES/PUBLISHES Edges

Neue Edge-Typen im Neo4j Graph-Schema.

```cypher
// Neue Relationship-Typen
(emitter:Symbol)-[:PUBLISHES {event_name: 'file.uploaded'}]->(event:Event)
(handler:Symbol)-[:SUBSCRIBES {event_name: 'file.uploaded'}]->(event:Event)

// Alternativ: Direkter Edge
(emitter:Symbol)-[:TRIGGERS_EVENT {event_name: 'file.uploaded'}]->(handler:Symbol)
```

**Python-Integration:**
```python
@dataclass
class EventEdge:
    """Repraesentiert eine Event-basierte Verbindung."""
    publisher_symbol_id: str
    subscriber_symbol_id: str
    event_name: str
    edge_type: str = "TRIGGERS_EVENT"
```

#### FR-5: DI-Graph-Extraktion (Constructor Injection)

Extraktion von Dependency-Injection-Beziehungen aus Constructor-Parametern.

```python
def extract_constructor_dependencies(class_symbol: Symbol, source: bytes) -> list[str]:
    """
    Extrahiere injizierte Services aus Constructor.

    Beispiel:
        constructor(private readonly storageService: StorageService) {}
        -> ["StorageService"]
    """
    # Finde constructor method_definition
    # Extrahiere Parameter mit Typ-Annotationen
    # Mappe zu Service-Typen
    pass
```

### 3.2 Non-Functional Requirements

#### NFR-1: Performance

| Anforderung | Spezifikation |
|-------------|---------------|
| Indexing-Zeit Overhead | max. +20% gegenueber aktuellem Stand |
| Memory Overhead | max. +10% pro Symbol (Decorator-Daten) |
| Parse-Zeit pro Datei | max. 100ms fuer 1000-Zeilen TypeScript |

#### NFR-2: Kompatibilitaet

| Framework | Versionen | Support-Level |
|-----------|-----------|---------------|
| NestJS | 9.x, 10.x | Full |
| Angular | 15+, 16+, 17+ | Decorators only |
| TypeScript | 4.7+ | Required |

#### NFR-3: Erweiterbarkeit

- Plugin-Architektur fuer Framework-spezifische Decorator-Mappings
- Custom Decorator-Listen konfigurierbar via Config-Datei
- Clear Interface fuer ts-morph Integration in Phase 3

---

## 4. Technical Specification

### 4.1 Decorator Extraktion (Tree-sitter)

#### 4.1.1 AST-Struktur von Decorators

Tree-sitter parst Decorators als eigene Nodes:

```
method_definition
  decorator[]  <- "multiple": true, "required": false
  name: identifier
  parameters: formal_parameters
  body: statement_block
```

**Beispiel-AST fuer `@OnEvent('file.uploaded')`:**
```
decorator {
  call_expression {
    function: identifier "OnEvent"
    arguments: arguments {
      string: "'file.uploaded'"
    }
  }
}
```

#### 4.1.2 _extract_decorator_info Methode

```python
def _extract_decorator_info(self, node: Node, source: bytes) -> Optional[DecoratorInfo]:
    """
    Extrahiere Decorator-Informationen aus einem decorator Node.

    Args:
        node: Tree-sitter Node vom Typ "decorator"
        source: Quellcode als bytes

    Returns:
        DecoratorInfo oder None wenn kein gueltiger Decorator
    """
    if node.type != "decorator":
        return None

    # Case 1: @OnEvent('x') - call_expression
    for child in node.children:
        if child.type == "call_expression":
            func_node = child.child_by_field_name("function")
            args_node = child.child_by_field_name("arguments")

            if func_node:
                name = source[func_node.start_byte:func_node.end_byte].decode("utf-8")
                args = self._extract_call_arguments(args_node, source) if args_node else []

                return DecoratorInfo(
                    name=name,
                    decorator_type=KNOWN_DECORATORS.get(name),
                    arguments=args,
                    raw_text=source[node.start_byte:node.end_byte].decode("utf-8"),
                )

        # Case 2: @Injectable - identifier only
        elif child.type == "identifier":
            name = source[child.start_byte:child.end_byte].decode("utf-8")
            return DecoratorInfo(
                name=name,
                decorator_type=KNOWN_DECORATORS.get(name),
                arguments=[],
                raw_text=source[node.start_byte:node.end_byte].decode("utf-8"),
            )

    return None

def _extract_call_arguments(self, args_node: Node, source: bytes) -> list[str]:
    """Extrahiere Argumente aus arguments Node."""
    args = []
    for child in args_node.children:
        if child.type in ("string", "template_string", "number", "identifier"):
            args.append(source[child.start_byte:child.end_byte].decode("utf-8"))
        elif child.type == "object":
            # Fuer @Controller({ path: 'users' })
            args.append(source[child.start_byte:child.end_byte].decode("utf-8"))
    return args
```

#### 4.1.3 KNOWN_DECORATORS Mapping

```python
KNOWN_DECORATORS: dict[str, str] = {
    # === Event Handling ===
    "OnEvent": "event_handler",
    "OnQueueEvent": "event_handler",
    "Process": "queue_processor",

    # === HTTP Handlers ===
    "Get": "http_handler",
    "Post": "http_handler",
    "Put": "http_handler",
    "Patch": "http_handler",
    "Delete": "http_handler",
    "Head": "http_handler",
    "Options": "http_handler",
    "All": "http_handler",

    # === Controllers ===
    "Controller": "controller",
    "Resolver": "graphql_resolver",

    # === Microservices ===
    "EventPattern": "message_handler",
    "MessagePattern": "message_handler",
    "GrpcMethod": "grpc_handler",
    "GrpcStreamMethod": "grpc_stream_handler",

    # === Dependency Injection ===
    "Injectable": "di_provider",
    "Inject": "di_injection",
    "Optional": "di_optional",
    "Global": "di_global",

    # === Middleware ===
    "UseGuards": "guard_chain",
    "UseInterceptors": "interceptor_chain",
    "UsePipes": "pipe_chain",
    "UseFilters": "filter_chain",

    # === Scheduling ===
    "Cron": "scheduled_task",
    "Interval": "scheduled_task",
    "Timeout": "scheduled_task",

    # === WebSockets ===
    "WebSocketGateway": "websocket_gateway",
    "SubscribeMessage": "websocket_handler",
    "ConnectedSocket": "websocket_param",
    "MessageBody": "websocket_param",

    # === Validation ===
    "Body": "param_decorator",
    "Query": "param_decorator",
    "Param": "param_decorator",
    "Headers": "param_decorator",

    # === GraphQL ===
    "Query": "graphql_query",
    "Mutation": "graphql_mutation",
    "Subscription": "graphql_subscription",
    "ResolveField": "graphql_field_resolver",
}
```

### 4.2 Event-Edge-Inferenz

#### 4.2.1 EventRegistry Klasse

```python
@dataclass
class EventRegistry:
    """Registry fuer Event-Publisher und Subscriber."""

    publishers: dict[str, list[str]] = field(default_factory=dict)   # event_name -> [symbol_ids]
    subscribers: dict[str, list[str]] = field(default_factory=dict)  # event_name -> [symbol_ids]

    def register_publisher(self, event_name: str, symbol_id: str) -> None:
        """Registriere einen Event-Publisher."""
        if event_name not in self.publishers:
            self.publishers[event_name] = []
        self.publishers[event_name].append(symbol_id)

    def register_subscriber(self, event_name: str, symbol_id: str) -> None:
        """Registriere einen Event-Subscriber (@OnEvent Handler)."""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(symbol_id)

    def get_edges(self) -> list[tuple[str, str, str]]:
        """
        Generiere Call-Edges zwischen Publishern und Subscribern.

        Returns:
            List of (publisher_id, subscriber_id, event_name)
        """
        edges = []
        for event_name, subscriber_ids in self.subscribers.items():
            publisher_ids = self.publishers.get(event_name, [])
            for pub_id in publisher_ids:
                for sub_id in subscriber_ids:
                    edges.append((pub_id, sub_id, event_name))
        return edges
```

#### 4.2.2 Publisher/Subscriber Discovery

```python
def discover_event_publishers(symbol: Symbol, source: bytes) -> list[str]:
    """
    Finde emit() Aufrufe im Symbol-Body.

    Patterns:
        - this.eventEmitter.emit('event.name', payload)
        - eventEmitter.emit('event.name')
        - this.emit('event.name')

    Returns:
        Liste von Event-Namen
    """
    # Regex-basierte Erkennung (Quick Win)
    emit_pattern = re.compile(
        r"""
        \.emit\s*\(\s*          # .emit(
        ['"]([^'"]+)['"]        # 'event.name' oder "event.name"
        """,
        re.VERBOSE
    )

    # Extrahiere Body aus Source basierend auf Symbol-Position
    body = source[symbol.col_start:symbol.col_end]  # Vereinfacht

    matches = emit_pattern.findall(body.decode("utf-8"))
    return matches

def discover_event_subscribers(symbols: list[Symbol]) -> dict[str, list[Symbol]]:
    """
    Sammle alle @OnEvent Handler.

    Returns:
        Dict: event_name -> [handler_symbols]
    """
    subscribers = {}
    for sym in symbols:
        for dec in sym.decorators:
            if dec.decorator_type == "event_handler" and dec.arguments:
                event_name = dec.arguments[0].strip("'\"")
                if event_name not in subscribers:
                    subscribers[event_name] = []
                subscribers[event_name].append(sym)
    return subscribers
```

#### 4.2.3 Call-Edge Generierung

```python
def generate_event_edges(
    registry: EventRegistry,
    repo_id: str,
) -> list[dict]:
    """
    Generiere Neo4j-kompatible Edge-Daten.

    Returns:
        Liste von Edge-Dicts fuer Neo4j Import
    """
    edges = []
    for pub_id, sub_id, event_name in registry.get_edges():
        edges.append({
            "source_id": pub_id,
            "target_id": sub_id,
            "relationship": "TRIGGERS_EVENT",
            "properties": {
                "event_name": event_name,
                "repo_id": repo_id,
                "inferred": True,
            }
        })
    return edges
```

### 4.3 ts-morph Integration (Phase 2)

#### 4.3.1 Wann ts-morph vs. Tree-sitter

| Anwendungsfall | Tool | Begruendung |
|----------------|------|-------------|
| Decorator-Syntax-Extraktion | Tree-sitter | Schnell, kein tsconfig noetig |
| Decorator-Argumente | Tree-sitter | Literale direkt lesbar |
| Type-Resolution | ts-morph | TypeChecker benoetigt |
| Constructor-Injection | ts-morph | Typ-Aufloesung fuer Services |
| Cross-File References | ts-morph | Full Project Analysis |
| Import-Resolution | ts-morph | Module-Aufloesung |

#### 4.3.2 Type-Resolution fuer DI

```typescript
// ts-morph Beispiel (Node.js Worker)
import { Project, SyntaxKind } from "ts-morph";

interface DIEdge {
  consumer: string;   // Symbol-ID der Klasse
  provider: string;   // Symbol-ID des injizierten Service
  paramName: string;  // Name des Constructor-Parameters
}

function extractDIGraph(tsConfigPath: string): DIEdge[] {
  const project = new Project({ tsConfigFilePath: tsConfigPath });
  const edges: DIEdge[] = [];

  project.getSourceFiles().forEach(sourceFile => {
    sourceFile.getClasses().forEach(classDecl => {
      const constructor = classDecl.getConstructors()[0];
      if (!constructor) return;

      constructor.getParameters().forEach(param => {
        const typeNode = param.getTypeNode();
        if (typeNode) {
          const type = typeNode.getType();
          const symbol = type.getSymbol();

          if (symbol) {
            edges.push({
              consumer: classDecl.getName() || "Anonymous",
              provider: symbol.getName(),
              paramName: param.getName(),
            });
          }
        }
      });
    });
  });

  return edges;
}
```

#### 4.3.3 Cross-File References

```typescript
// Resolve @Inject() Tokens
function resolveInjectToken(decorator: Decorator): string | null {
  const args = decorator.getArguments();
  if (args.length === 0) return null;

  const arg = args[0];
  const type = arg.getType();

  // Handle InjectionToken, string, or class reference
  if (type.isString()) {
    return arg.getText().replace(/['"]/g, '');
  } else if (type.isClassOrInterface()) {
    return type.getSymbol()?.getName() || null;
  }

  return null;
}
```

---

## 5. Implementation Plan

### Phase 1: Tree-sitter Extension (Woche 1-3)

#### Woche 1: Symbol.decorators Feld

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| DecoratorInfo Dataclass hinzufuegen | 2h | - |
| Symbol.decorators Feld hinzufuegen | 1h | DecoratorInfo |
| Unit Tests fuer DecoratorInfo | 2h | DecoratorInfo |
| Bestehende Tests anpassen | 2h | Symbol-Aenderung |

**Deliverable:** `Symbol` Dataclass mit `decorators: list[DecoratorInfo]`

#### Woche 2: _parse_decorator Methode

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| _extract_decorator_info() implementieren | 4h | DecoratorInfo |
| _extract_call_arguments() implementieren | 2h | - |
| TypeScriptPlugin._extract_method() erweitern | 3h | _extract_decorator_info |
| TypeScriptPlugin._extract_class() erweitern | 2h | _extract_decorator_info |
| Integration Tests | 4h | Alle Methoden |

**Deliverable:** Decorator-Extraktion fuer Methods und Classes

#### Woche 3: NestJS Decorator Mappings

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| KNOWN_DECORATORS Dict erstellen | 2h | - |
| Decorator-Type-Mapping implementieren | 2h | KNOWN_DECORATORS |
| NestJS-spezifische Tests | 4h | Mapping |
| Angular-spezifische Tests | 4h | Mapping |
| Dokumentation | 2h | Alle |

**Deliverable:** Vollstaendige Decorator-Erkennung fuer NestJS/Angular

### Phase 2: Event-Registry (Woche 4-5)

#### Woche 4: EventRegistry Klasse

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| EventRegistry Dataclass | 3h | - |
| Publisher Discovery (emit() Pattern) | 4h | - |
| Subscriber Discovery (@OnEvent) | 2h | Decorator-Extraktion |
| Unit Tests | 4h | EventRegistry |

**Deliverable:** EventRegistry mit Publisher/Subscriber Discovery

#### Woche 5: Edge Generation

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| generate_event_edges() implementieren | 3h | EventRegistry |
| Neo4j TRIGGERS_EVENT Edge-Typ | 2h | Schema-Erweiterung |
| Integration mit index_codebase | 4h | Alle |
| End-to-End Tests | 4h | Integration |

**Deliverable:** Event-Edges im Call-Graph

### Phase 3: ts-morph (Woche 6-10)

#### Woche 6-7: Node.js Bridge

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| ts-morph Worker Setup (Node.js) | 8h | - |
| JSON-RPC Protokoll definieren | 4h | - |
| Python Subprocess Wrapper | 4h | Worker |
| Error Handling | 4h | Wrapper |

**Deliverable:** Python -> Node.js Bridge fuer ts-morph

#### Woche 8-9: DI-Graph Extraktion

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| extractDIGraph() in TypeScript | 6h | ts-morph Worker |
| @Inject() Token Resolution | 4h | extractDIGraph |
| Constructor Injection Mapping | 4h | extractDIGraph |
| Neo4j INJECTS Edge-Typ | 2h | Schema |

**Deliverable:** DI-Abhaengigkeiten im Graph

#### Woche 10: Constructor Injection Aufloesung

| Task | Aufwand | Abhaengigkeiten |
|------|---------|-----------------|
| Service-zu-Symbol Mapping | 4h | DI-Graph |
| CALLS-Edges fuer injizierte Methoden | 6h | Service-Mapping |
| Integration Tests | 4h | Alle |
| Performance-Optimierung | 4h | Benchmarks |

**Deliverable:** Vollstaendige DI-Aufloesung

---

## 6. Decorator Mapping Table

### 6.1 Event Handling

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@OnEvent('x')` | EVENT_HANDLER | `emit('x')` -> handler | event_name |
| `@OnQueueEvent('x')` | EVENT_HANDLER | queue.add() -> handler | event_name |
| `@Process('x')` | QUEUE_PROCESSOR | queue.add('x') -> handler | job_name |

### 6.2 HTTP Handlers

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@Get('/users')` | HTTP_HANDLER | route -> controller.method | path |
| `@Post('/users')` | HTTP_HANDLER | route -> controller.method | path |
| `@Put('/users/:id')` | HTTP_HANDLER | route -> controller.method | path |
| `@Delete('/users/:id')` | HTTP_HANDLER | route -> controller.method | path |
| `@Controller('users')` | CONTROLLER | module -> controller | prefix |

### 6.3 Dependency Injection

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@Injectable()` | DI_PROVIDER | consumer.ctor -> provider | scope? |
| `@Inject('TOKEN')` | DI_INJECTION | param -> token provider | token |
| `@Optional()` | DI_OPTIONAL | (marks optional) | - |

### 6.4 Microservices

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@EventPattern('x')` | MESSAGE_HANDLER | send('x') -> handler | pattern |
| `@MessagePattern('x')` | MESSAGE_HANDLER | send('x') -> handler | pattern |

### 6.5 WebSockets

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@WebSocketGateway()` | WEBSOCKET_GATEWAY | client.emit -> gateway | port?, namespace? |
| `@SubscribeMessage('x')` | WEBSOCKET_HANDLER | emit('x') -> handler | message |

### 6.6 Scheduling

| Decorator | Type | Call-Edge | Arguments |
|-----------|------|-----------|-----------|
| `@Cron('0 * * * *')` | SCHEDULED_TASK | scheduler -> method | cron_expression |
| `@Interval(5000)` | SCHEDULED_TASK | scheduler -> method | ms |
| `@Timeout(3000)` | SCHEDULED_TASK | scheduler -> method | ms |

---

## 7. Test Cases

### 7.1 Unit Tests

#### test_on_event_decorator_extracted

```python
def test_on_event_decorator_extracted():
    """@OnEvent Decorator wird korrekt extrahiert."""
    code = """
    import { OnEvent } from '@nestjs/event-emitter';

    class FileService {
      @OnEvent('file.uploaded')
      handleFileUploaded(payload: FileUploadedEvent) {}
    }
    """
    result = parser.parse_string(code, Language.TYPESCRIPT)

    # Finde handleFileUploaded Method
    method = next(s for s in result.symbols if s.name == "handleFileUploaded")

    assert len(method.decorators) == 1
    dec = method.decorators[0]
    assert dec.name == "OnEvent"
    assert dec.decorator_type == "event_handler"
    assert dec.arguments == ["'file.uploaded'"]
```

#### test_multiple_decorators_extracted

```python
def test_multiple_decorators_extracted():
    """Mehrere Decorators an einer Methode werden alle extrahiert."""
    code = """
    class UserController {
      @UseGuards(AuthGuard)
      @Post('/users')
      createUser(@Body() dto: CreateUserDto) {}
    }
    """
    result = parser.parse_string(code, Language.TYPESCRIPT)

    method = next(s for s in result.symbols if s.name == "createUser")

    assert len(method.decorators) == 2
    decorator_names = [d.name for d in method.decorators]
    assert "UseGuards" in decorator_names
    assert "Post" in decorator_names
```

#### test_controller_decorator_with_prefix

```python
def test_controller_decorator_with_prefix():
    """@Controller mit Route-Prefix wird extrahiert."""
    code = """
    @Controller('users')
    export class UserController {}
    """
    result = parser.parse_string(code, Language.TYPESCRIPT)

    cls = next(s for s in result.symbols if s.name == "UserController")

    assert len(cls.decorators) == 1
    dec = cls.decorators[0]
    assert dec.name == "Controller"
    assert dec.decorator_type == "controller"
    assert dec.arguments == ["'users'"]
```

### 7.2 Integration Tests

#### test_event_edge_generated

```python
def test_event_edge_generated():
    """Event-basierte Call-Edges werden generiert."""
    publisher_code = """
    class UploadService {
      constructor(private eventEmitter: EventEmitter2) {}

      async uploadFile(file: File) {
        await this.saveFile(file);
        this.eventEmitter.emit('file.uploaded', { fileId: file.id });
      }
    }
    """

    handler_code = """
    class ProcessingService {
      @OnEvent('file.uploaded')
      handleUpload(payload: { fileId: string }) {
        // process file
      }
    }
    """

    symbols = parser.parse_files([publisher_code, handler_code])
    registry = EventRegistry()

    # Populate registry
    for sym in symbols:
        publishers = discover_event_publishers(sym, code)
        for event_name in publishers:
            registry.register_publisher(event_name, sym.id)

        for dec in sym.decorators:
            if dec.decorator_type == "event_handler":
                event_name = dec.arguments[0].strip("'\"")
                registry.register_subscriber(event_name, sym.id)

    edges = registry.get_edges()

    assert len(edges) == 1
    pub_id, sub_id, event_name = edges[0]
    assert event_name == "file.uploaded"
```

#### test_di_injection_resolved

```python
def test_di_injection_resolved():
    """Constructor Injection wird zu DI-Edges aufgeloest."""
    code = """
    @Injectable()
    class StorageService {
      saveFile(file: File) {}
    }

    @Injectable()
    class UploadService {
      constructor(
        private readonly storageService: StorageService,
        private readonly logger: LoggerService,
      ) {}

      async upload(file: File) {
        await this.storageService.saveFile(file);
      }
    }
    """

    # ts-morph basierte Analyse
    di_edges = extract_di_graph(code)

    assert len(di_edges) == 2
    storage_edge = next(e for e in di_edges if e.provider == "StorageService")
    assert storage_edge.consumer == "UploadService"
    assert storage_edge.paramName == "storageService"
```

### 7.3 End-to-End Tests

#### test_find_callers_with_event_edges

```python
def test_find_callers_with_event_edges():
    """find_callers() findet Event-basierte Aufrufer."""
    # Index a NestJS project
    index_result = index_codebase(
        repo_id="test-nestjs",
        root_path="/path/to/nestjs/project",
    )

    # Find callers of handleFileUploaded (Event Handler)
    result = find_callers(
        repo_id="test-nestjs",
        symbol_name="handleFileUploaded",
    )

    # Should find uploadFile() as caller via emit('file.uploaded')
    caller_names = [n.name for n in result.nodes]
    assert "uploadFile" in caller_names
```

---

## 8. Risks & Mitigations

### 8.1 Performance-Risiken

#### Risk 1: Tree-sitter Performance bei grossen Codebases

**Wahrscheinlichkeit:** Mittel
**Impact:** Hoch

**Beschreibung:** Die zusaetzliche Decorator-Extraktion koennte bei Projekten mit >10.000 TypeScript-Dateien zu spuerbaren Verzoegerungen fuehren.

**Mitigations:**
- Lazy Decorator Parsing: Nur bei aktiviertem `include_decorators` Flag
- Caching: Decorator-Infos im Symbol-Cache speichern
- Incremental Parsing: Nur geaenderte Dateien neu parsen
- Benchmark-Suite: CI-Pipeline mit Performance-Regression-Tests

#### Risk 2: ts-morph Memory-Verbrauch

**Wahrscheinlichkeit:** Hoch
**Impact:** Mittel

**Beschreibung:** ts-morph laedt das gesamte TypeScript-Projekt in den Speicher, was bei grossen Monorepos zu OOM fuehren kann.

**Mitigations:**
- Batch Processing: Dateien in Gruppen von 100 verarbeiten
- Worker Isolation: ts-morph in separatem Node.js Prozess mit Memory-Limit
- Selective Analysis: Nur Dateien mit relevanten Decorators analysieren
- Fallback: Bei OOM auf Tree-sitter-only zurueckfallen

### 8.2 Kompatibilitaets-Risiken

#### Risk 3: Breaking Changes in NestJS Decorator API

**Wahrscheinlichkeit:** Niedrig
**Impact:** Mittel

**Beschreibung:** NestJS koennte Decorator-Signaturen oder -Namen aendern.

**Mitigations:**
- Version-spezifische Mappings: `KNOWN_DECORATORS_V9`, `KNOWN_DECORATORS_V10`
- Fallback-Erkennung: Unbekannte Decorators als `unknown` taggen
- Update-Workflow: Dokumentierter Prozess fuer Mapping-Updates

#### Risk 4: TC39 Decorators vs. Legacy Decorators

**Wahrscheinlichkeit:** Mittel
**Impact:** Hoch

**Beschreibung:** TypeScript 5.0+ unterstuetzt TC39 Stage 3 Decorators, die sich syntaktisch von experimentalDecorators unterscheiden.

**Mitigations:**
- Detection: tsconfig.json auf `experimentalDecorators` pruefen
- Dual-Parser: Beide Decorator-Syntaxen unterstuetzen
- Warnung: Bei TC39 Decorators ohne volle Unterstuetzung warnen

### 8.3 Architektur-Risiken

#### Risk 5: Event-Name Collisions

**Wahrscheinlichkeit:** Niedrig
**Impact:** Niedrig

**Beschreibung:** Unterschiedliche Module koennten den gleichen Event-Namen verwenden.

**Mitigations:**
- Namespace-Prefix: Event-Namen mit Modul-Prefix qualifizieren
- Scope-Tracking: Events auf Modul-Ebene tracken
- Warnung: Bei potentiellen Collisions im Index-Log warnen

---

## 9. Dependencies

### 9.1 Runtime Dependencies

| Dependency | Version | Zweck |
|------------|---------|-------|
| Python | 3.10+ | tree-sitter 0.21+ Kompatibilitaet |
| tree-sitter | 0.21+ | AST Parsing |
| tree-sitter-typescript | latest | TypeScript Grammar |
| Node.js | 18+ (optional) | ts-morph Worker |
| ts-morph | 21+ (optional) | Type-Resolution |

### 9.2 Development Dependencies

| Dependency | Zweck |
|------------|-------|
| pytest | Unit Tests |
| pytest-benchmark | Performance Tests |
| mypy | Type Checking |
| ruff | Linting |

### 9.3 Infrastructure Dependencies

| Component | Aenderung |
|-----------|-----------|
| Neo4j | Neue Edge-Typen: `TRIGGERS_EVENT`, `INJECTS` |
| OpenSearch | Decorator-Felder im Document Schema |
| MCP API | Neue Felder in Symbol-Responses |

---

## 10. Appendix

### A. Tree-sitter Query fuer Decorators

```scheme
;; Query fuer TypeScript Decorator-Nodes
(method_definition
  (decorator
    (call_expression
      function: (identifier) @decorator.name
      arguments: (arguments) @decorator.args
    )
  )
  name: (property_identifier) @method.name
) @method

;; Query fuer Class-Level Decorators
(class_declaration
  (decorator
    (call_expression
      function: (identifier) @decorator.name
      arguments: (arguments) @decorator.args
    )
  )
  name: (type_identifier) @class.name
) @class
```

### B. Neo4j Schema Erweiterung

```cypher
// Neue Node-Labels
CREATE CONSTRAINT event_name IF NOT EXISTS
FOR (e:Event) REQUIRE e.name IS UNIQUE;

// Neue Relationship-Typen
// (publisher:Symbol)-[:PUBLISHES]->(event:Event)
// (event:Event)-[:HANDLED_BY]->(subscriber:Symbol)
// (consumer:Symbol)-[:INJECTS]->(provider:Symbol)

// Indexes fuer Performance
CREATE INDEX event_name_idx IF NOT EXISTS FOR (e:Event) ON (e.name);
CREATE INDEX decorator_type_idx IF NOT EXISTS FOR (s:Symbol) ON (s.decorator_type);
```

### C. OpenSearch Document Schema Update

```json
{
  "mappings": {
    "properties": {
      "decorators": {
        "type": "nested",
        "properties": {
          "name": { "type": "keyword" },
          "decorator_type": { "type": "keyword" },
          "arguments": { "type": "keyword" },
          "raw_text": { "type": "text" }
        }
      },
      "is_event_handler": { "type": "boolean" },
      "event_names": { "type": "keyword" },
      "is_http_handler": { "type": "boolean" },
      "http_method": { "type": "keyword" },
      "http_path": { "type": "keyword" }
    }
  }
}
```

### D. Referenzen

- [FIX-04-DECORATOR-INDEXING-ANALYSIS.md](/docs/FIX-04-DECORATOR-INDEXING-ANALYSIS.md)
- [ast_parser.py](/openmemory/api/indexing/ast_parser.py)
- [ts-morph Documentation](https://ts-morph.com/)
- [Tree-sitter TypeScript Grammar](https://github.com/tree-sitter/tree-sitter-typescript)
- [NestJS Decorators Documentation](https://docs.nestjs.com/custom-decorators)
- [TC39 Decorators Proposal](https://github.com/tc39/proposal-decorators)
