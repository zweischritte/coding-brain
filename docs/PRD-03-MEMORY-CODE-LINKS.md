# PRD: Code-Linked Memories fuer Coding Brain

**Version:** 1.1
**Erstellt:** 2026-01-04
**Autor:** Claude Code (Opus 4.5)
**Status:** Phase 1 Implemented
**Basiert auf:** FIX-03-MEMORY-CODE-LINKS-ANALYSIS.md

---

## Implementation Log (Phase 1)

**Implemented:** 2026-01-04
**Commits:** `82ea7251`, `df04ce3b`

### Files Created

| File | Purpose |
|------|---------|
| `openmemory/api/app/utils/code_reference.py` | Core module with LineRange, CodeReference dataclasses, tags serialization, validation, and response formatting |
| `openmemory/api/tests/test_code_reference.py` | Unit tests for LineRange, CodeReference, serialization, validation |
| `openmemory/api/tests/mcp/test_add_memories_code_refs.py` | Integration tests for add_memories with code_refs parameter |

### Files Modified

| File | Lines Changed | Changes |
|------|---------------|---------|
| `openmemory/api/app/mcp_server.py` | 39-47, 638-662, 728-743 | Added code_reference imports; added code_refs parameter to add_memories; added code_refs validation and serialization |
| `openmemory/api/app/utils/response_format.py` | 21-25, 168-181 | Added code_reference imports; added code_refs extraction from tags in format_memory_result() |

### Key Implementation Details

- **Storage Strategy:** Phase 1 uses tags-based storage (no schema migration)
- **Tag Format:** `code_ref_count`, `code_ref_0_path`, `code_ref_0_lines`, `code_ref_0_symbol`, etc.
- **URI Format:** `file:/path/to/file.ts#L42-L87`
- **Backward Compatible:** Existing memories without code_refs continue to work

### To Revert This Implementation

```bash
# Revert commits (in reverse order)
git revert df04ce3b  # mcp_server.py changes
git revert 82ea7251  # new files and response_format.py

# Or manually remove:
# 1. Delete: openmemory/api/app/utils/code_reference.py
# 2. Delete: openmemory/api/tests/test_code_reference.py
# 3. Delete: openmemory/api/tests/mcp/test_add_memories_code_refs.py
# 4. Remove code_reference imports from mcp_server.py (lines 39-47)
# 5. Remove code_refs parameter from add_memories function
# 6. Remove code_refs validation block from add_memories
# 7. Remove code_reference imports from response_format.py (lines 21-25)
# 8. Restore original tags handling in format_memory_result()
```

---

## 1. Problem Statement

### 1.1 Kernproblem

Memories in Coding Brain fungieren derzeit als **"Wegweiser ohne Koordinaten"** - sie beschreiben Wissen ueber Code, aber ohne praezise Lokalisierung. Wenn eine Memory `createFileUploads` erwaehnt, fehlen:

- **Dateipfad**: Wo genau liegt die Funktion?
- **Zeilennummern**: Welche Zeilen sind relevant?
- **Staleness-Erkennung**: Ist der referenzierte Code noch aktuell?

### 1.2 Beobachtetes Verhalten

In der 10-Agenten-Analyse wurde folgendes Problem identifiziert:

1. **Fehlende Dateipfade**: Eine Memory erwaehnt `createFileUploads`, aber ohne konkreten Pfad wie `/apps/merlin/src/storage/storage.service.ts:42`
2. **Unverlinkte Consumer**: Memory beschreibt Consumer, die `moveFilesToPermanentStorage` aufrufen, listet aber nicht die konkreten Consumer-Dateien
3. **Keine Kreuzreferenzen**: Kein Memory-Eintrag verbindet Consumer mit ihren jeweiligen Dateipfaden

**Konsequenz:** Der Agent konnte nicht direkt zu den relevanten Code-Stellen navigieren und musste stattdessen global suchen - Zeitverlust und reduzierte Praezision.

### 1.3 Aktuelles Schema (Defizite)

Das aktuelle `StructuredMemoryInput` Schema hat folgende Limitierungen:

```python
@dataclass
class StructuredMemoryInput:
    text: str
    category: str
    scope: str
    artifact_type: Optional[str] = None   # repo, service, module, file, etc.
    artifact_ref: Optional[str] = None    # Freier String ohne Struktur
    entity: Optional[str] = None
    access_entity: Optional[str] = None
    source: str = "user"
    evidence: Optional[List[str]] = None
    tags: Optional[Dict[str, Any]] = None
```

**Fehlende Felder:**

| Feld | Beschreibung | Status |
|------|-------------|--------|
| `file_path` | Absoluter Pfad zur Quelldatei | Fehlt |
| `line_range` | Start- und Endzeile | Fehlt |
| `symbol_id` | SCIP-kompatible Symbol-ID | Fehlt |
| `code_hash` | SHA256 des referenzierten Code-Blocks | Fehlt |
| `confidence_score` | Wie sicher ist das Wissen? | Fehlt |
| `last_verified_at` | Wann wurde die Code-Referenz zuletzt verifiziert? | Fehlt |
| `stale_since` | Wann wurde der referenzierte Code geaendert? | Fehlt |
| `git_commit` | Commit-SHA zum Zeitpunkt der Memory-Erstellung | Fehlt |

---

## 2. Goals & Success Metrics

### 2.1 Primaerziel

**Jede Code-bezogene Memory hat praezise Lokalisierung** - Dateipfad, Zeilennummern, und Staleness-Status sind verfuegbar.

### 2.2 Sekundaerziele

1. **Bidirektionale Navigation**: Von Memory zu Code und von Code zu Memory
2. **Automatische Staleness-Erkennung**: Aenderungen im Code invalidieren verknuepfte Memories
3. **Confidence-basiertes Ranking**: Aktuelle, verifizierte Memories werden bevorzugt

### 2.3 Success Metrics

| Metrik | Baseline | Ziel (Phase 3) | Messmethode |
|--------|----------|----------------|-------------|
| **Code-Link Coverage** | 0% | 80% aller Code-Memories | Anteil Memories mit mind. 1 `code_ref` |
| **Staleness Detection Rate** | 0% | 95% | Anteil geaenderter Code-Referenzen, die korrekt als stale markiert werden |
| **Navigation Success Rate** | N/A | 90% | Anteil Code-Links, die zu gueltigem Code fuehren |
| **Search Relevance (MRR)** | Baseline | +15% | Mean Reciprocal Rank mit Staleness-Reranking |
| **Agent Task Completion Time** | Baseline | -20% | Zeit bis Agent relevanten Code findet |

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: CodeReference Dataclass

**Beschreibung:** Strukturierte Repraesentation einer Code-Referenz mit Lokalisierung und Versionierung.

**Schema:**

```python
@dataclass
class LineRange:
    """Start- und Endzeile einer Code-Referenz."""
    start: int          # 1-indexed, inklusive
    end: int            # 1-indexed, inklusive

    def __post_init__(self):
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"Invalid line range: {self.start}-{self.end}")


@dataclass
class CodeReference:
    """Strukturierte Code-Referenz fuer eine Memory."""

    # Lokalisierung (mindestens file_path empfohlen)
    file_path: Optional[str] = None        # Absoluter Pfad
    line_range: Optional[LineRange] = None # Start/End Zeilen
    symbol_id: Optional[str] = None        # SCIP-kompatible Symbol-ID

    # Versionierung
    git_commit: Optional[str] = None       # Commit-SHA bei Erstellung
    code_hash: Optional[str] = None        # SHA256 des Code-Blocks

    # Temporal Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = None
    stale_since: Optional[datetime] = None

    # Confidence
    confidence_score: float = 0.8          # 0.0-1.0
    confidence_reason: Optional[str] = None
```

**Akzeptanzkriterien:**
- [ ] LineRange validiert start >= 1 und end >= start
- [ ] file_path akzeptiert absolute und repo-relative Pfade
- [ ] symbol_id folgt SCIP-Format: `scip-<lang> <package> <path>/<Symbol>#<method>()`
- [ ] code_hash verwendet SHA256 mit Praefix `sha256:`

---

#### FR-2: Git-basierte Versionierung

**Beschreibung:** Jede Code-Referenz speichert den Git-Commit und einen Content-Hash zur Aenderungserkennung.

**Implementierung:**

```python
def compute_code_hash(file_path: str, start_line: int, end_line: int) -> str:
    """Berechne SHA256 des Code-Blocks."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    block = ''.join(lines[start_line-1:end_line])
    return f"sha256:{hashlib.sha256(block.encode()).hexdigest()}"
```

**Akzeptanzkriterien:**
- [ ] git_commit wird automatisch bei Memory-Erstellung gesetzt (aktueller HEAD)
- [ ] code_hash wird aus dem spezifizierten Zeilenbereich berechnet
- [ ] Hash-Berechnung beruecksichtigt nur den Content, nicht Whitespace-Aenderungen (optional: normalisiert)

---

#### FR-3: Staleness Detection Pipeline

**Beschreibung:** Automatische Erkennung, wenn referenzierter Code geaendert wurde.

**Implementierung:**

```python
class StalenessPipeline:
    """Pipeline zur Staleness-Erkennung bei Code-Aenderungen."""

    def check_memory_staleness(self, memory: CodeLinkedMemory) -> CodeLinkedMemory:
        """Pruefe ob Code-Referenzen noch aktuell sind."""
        if not memory.code_refs:
            return memory

        any_stale = False
        stale_reasons = []

        for ref in memory.code_refs:
            if not ref.file_path or not ref.line_range or not ref.code_hash:
                continue

            try:
                current_hash = self.compute_code_hash(
                    ref.file_path,
                    ref.line_range.start,
                    ref.line_range.end
                )

                if current_hash != ref.code_hash:
                    ref.stale_since = datetime.utcnow()
                    any_stale = True
                    stale_reasons.append(f"{ref.file_path}:{ref.line_range.start}")
                else:
                    ref.last_verified_at = datetime.utcnow()
                    ref.stale_since = None

            except FileNotFoundError:
                ref.stale_since = datetime.utcnow()
                any_stale = True
                stale_reasons.append(f"{ref.file_path} (deleted)")

        memory.is_stale = any_stale
        if any_stale:
            memory.stale_reason = f"code_changed: {', '.join(stale_reasons)}"

        return memory

    def on_git_push_hook(self, changed_files: List[str]):
        """Git Post-Push Hook: Alle Memories mit geaenderten Dateien pruefen."""
        # 1. Finde alle Memories mit code_refs in changed_files
        # 2. Berechne neue Hashes
        # 3. Markiere geaenderte als stale
        # 4. Update confidence_score (reduzieren bei Staleness)
        pass
```

**Akzeptanzkriterien:**
- [ ] Staleness wird bei jedem Memory-Abruf geprueft (lazy) ODER via Git Hook (eager)
- [ ] Geloeschte Dateien werden als stale markiert mit Reason "deleted"
- [ ] Stale Memories erhalten reduzierte Confidence Scores

---

#### FR-4: Confidence Score Berechnung

**Beschreibung:** Gewichteter Score basierend auf Verifikation, Evidence, Hash-Aktualitaet und Quelle.

**Formel:**

```python
class ConfidenceCalculator:
    WEIGHTS = {
        'verification_recency': 0.3,   # Wie kuerzlich verifiziert?
        'evidence_count': 0.2,          # Wieviele ADRs/PRs?
        'code_hash_freshness': 0.3,     # Ist Hash aktuell?
        'source_reliability': 0.2,      # User vs Inference
    }

    def calculate(
        self,
        last_verified_at: datetime = None,
        evidence: List[str] = None,
        is_code_hash_current: bool = True,
        source: str = "user"
    ) -> float:
        # Verification Recency (decay ueber 30 Tage)
        if last_verified_at:
            days_since = (datetime.utcnow() - last_verified_at).days
            verification_score = max(0.0, 1.0 - (days_since / 30))
        else:
            verification_score = 0.5

        # Evidence Count (mehr = hoeher, max bei 3+)
        evidence_count = len(evidence) if evidence else 0
        evidence_score = min(1.0, evidence_count / 3)

        # Code Hash Freshness
        hash_score = 1.0 if is_code_hash_current else 0.2

        # Source Reliability
        source_score = 1.0 if source == "user" else 0.7

        return (
            self.WEIGHTS['verification_recency'] * verification_score +
            self.WEIGHTS['evidence_count'] * evidence_score +
            self.WEIGHTS['code_hash_freshness'] * hash_score +
            self.WEIGHTS['source_reliability'] * source_score
        )
```

**Akzeptanzkriterien:**
- [ ] Score liegt zwischen 0.0 und 1.0
- [ ] Stale Memories haben max. 0.5 Score (wg. hash_score = 0.2)
- [ ] Neu erstellte User-Memories starten mit Score >= 0.8

---

#### FR-5: Integration mit bestehendem Code-Index

**Beschreibung:** Automatische Verknuepfung von Memories mit SCIP-Symbolen aus dem Code-Index.

**Implementierung:**

```python
class MemoryCodeLinker:
    """Verknuepfe Memories automatisch mit Code-Symbolen."""

    def __init__(self, code_indexer):
        self.indexer = code_indexer

    def enrich_memory_with_code_refs(self, memory_text: str, repo_id: str) -> List[CodeReference]:
        """Extrahiere Code-Referenzen aus Memory-Text via Code-Index."""

        # 1. Identifiziere potentielle Symbole im Text
        symbols = self.extract_potential_symbols(memory_text)

        # 2. Suche im Code-Index
        code_refs = []
        for symbol in symbols:
            results = self.indexer.search(symbol, repo_id=repo_id, limit=1)
            if results:
                hit = results[0]
                code_refs.append(CodeReference(
                    file_path=hit.file_path,
                    line_range=LineRange(start=hit.start_line, end=hit.end_line),
                    symbol_id=hit.scip_symbol_id,
                    git_commit=hit.indexed_at_commit,
                    code_hash=self.compute_code_hash(hit.file_path, hit.start_line, hit.end_line),
                    confidence_score=0.8,
                    confidence_reason="inferred_from_code_index"
                ))

        return code_refs

    def extract_potential_symbols(self, text: str) -> List[str]:
        """Extrahiere potentielle Code-Symbole aus Text."""
        import re
        patterns = [
            r'\b[A-Z][a-zA-Z0-9]*(?:Service|Controller|Repository|Handler)\b',
            r'\b[a-z][a-zA-Z0-9]*\(\)',
            r'\b[a-z_][a-z0-9_]+\b',
        ]

        symbols = []
        for pattern in patterns:
            symbols.extend(re.findall(pattern, text))

        return list(set(symbols))
```

**Akzeptanzkriterien:**
- [ ] Symbole werden via Regex aus Memory-Text extrahiert
- [ ] Gefundene Symbole werden gegen Code-Index gematcht
- [ ] Automatisch erstellte Code-Refs haben confidence_reason "inferred_from_code_index"

---

### 3.2 Non-Functional Requirements

#### NFR-1: Performance

| Operation | Ziel | Max |
|-----------|------|-----|
| Hash-Berechnung pro Block | < 5ms | 10ms |
| Staleness-Check pro Memory | < 20ms | 50ms |
| Symbol-Extraktion aus Text | < 10ms | 30ms |
| Code-Index Lookup | < 50ms | 100ms |

#### NFR-2: Storage

| Metrik | Limit |
|--------|-------|
| Zusaetzliche Bytes pro Memory (durchschnittlich) | < 500 Bytes |
| Zusaetzliche Bytes pro Memory (maximum) | < 1 KB |
| Code-Reference Collection Overhead | < 10% der Memory-Collection |

#### NFR-3: Reliability

- Staleness Detection muss idempotent sein
- Bei Git-Hook-Fehler: Memory bleibt unveraendert (kein Datenverlust)
- Hash-Kollisionen: SHA256 ist ausreichend (praktisch keine Kollisionen)

#### NFR-4: Backwards Compatibility

- Bestehende Memories ohne `code_refs` bleiben gueltig
- `artifact_ref` bleibt als Legacy-Feld erhalten
- Migration kann inkrementell erfolgen

---

## 4. Technical Specification

### 4.1 Neues Datenmodell

#### LineRange Dataclass

```python
@dataclass
class LineRange:
    """Start- und Endzeile einer Code-Referenz."""
    start: int          # 1-indexed, inklusive
    end: int            # 1-indexed, inklusive

    def __post_init__(self):
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"Invalid line range: {self.start}-{self.end}")

    def to_fragment(self) -> str:
        """Konvertiere zu URI-Fragment: #L42-L87"""
        return f"#L{self.start}-L{self.end}"

    @classmethod
    def from_fragment(cls, fragment: str) -> "LineRange":
        """Parse URI-Fragment: #L42-L87 -> LineRange(42, 87)"""
        import re
        match = re.match(r'#L(\d+)-L(\d+)', fragment)
        if not match:
            raise ValueError(f"Invalid fragment: {fragment}")
        return cls(start=int(match.group(1)), end=int(match.group(2)))
```

#### CodeReference Dataclass

```python
@dataclass
class CodeReference:
    """Strukturierte Code-Referenz fuer eine Memory."""

    # Lokalisierung
    file_path: Optional[str] = None
    line_range: Optional[LineRange] = None
    symbol_id: Optional[str] = None

    # Versionierung
    git_commit: Optional[str] = None
    code_hash: Optional[str] = None

    # Temporal Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = None
    stale_since: Optional[datetime] = None

    # Confidence
    confidence_score: float = 0.8
    confidence_reason: Optional[str] = None

    def to_uri(self) -> str:
        """Konvertiere zu file: URI mit Fragment."""
        uri = f"file:{self.file_path}"
        if self.line_range:
            uri += self.line_range.to_fragment()
        return uri

    @classmethod
    def from_uri(cls, uri: str) -> "CodeReference":
        """Parse file: URI: file:/path/to/file.ts#L42-L87"""
        import re
        match = re.match(r'file:([^#]+)(#L\d+-L\d+)?', uri)
        if not match:
            raise ValueError(f"Invalid URI: {uri}")

        file_path = match.group(1)
        line_range = None
        if match.group(2):
            line_range = LineRange.from_fragment(match.group(2))

        return cls(file_path=file_path, line_range=line_range)

    def is_stale(self) -> bool:
        """Pruefe ob diese Referenz als stale markiert ist."""
        return self.stale_since is not None

    def to_dict(self) -> dict:
        """Serialisiere fuer JSON/Tags-Speicherung."""
        return {
            "file_path": self.file_path,
            "line_start": self.line_range.start if self.line_range else None,
            "line_end": self.line_range.end if self.line_range else None,
            "symbol_id": self.symbol_id,
            "git_commit": self.git_commit,
            "code_hash": self.code_hash,
            "confidence_score": self.confidence_score,
            "confidence_reason": self.confidence_reason,
            "stale_since": self.stale_since.isoformat() if self.stale_since else None,
        }
```

#### CodeLinkedMemory erweitert StructuredMemoryInput

```python
@dataclass
class CodeLinkedMemory:
    """Memory mit praezisen Code-Links."""

    # Core Memory (bestehend)
    text: str
    category: str
    scope: str

    # Existing fields (bestehend)
    entity: Optional[str] = None
    artifact_type: Optional[str] = None
    artifact_ref: Optional[str] = None      # Legacy-Kompatibilitaet
    access_entity: Optional[str] = None
    source: str = "user"
    evidence: Optional[List[str]] = None
    tags: Optional[Dict[str, Any]] = None

    # NEU: Strukturierte Code-Referenzen
    code_refs: Optional[List[CodeReference]] = None

    # NEU: Globale Staleness-Info
    is_stale: bool = False
    stale_reason: Optional[str] = None
```

### 4.2 Staleness Pipeline

#### StalenessPipeline Klasse

```python
class StalenessPipeline:
    """Pipeline zur Staleness-Erkennung bei Code-Aenderungen."""

    def __init__(self, memory_store, git_client=None):
        self.memory_store = memory_store
        self.git_client = git_client or GitClient()

    def compute_code_hash(self, file_path: str, start_line: int, end_line: int) -> str:
        """Berechne SHA256 des Code-Blocks."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        block = ''.join(lines[start_line-1:end_line])
        return f"sha256:{hashlib.sha256(block.encode()).hexdigest()}"

    def check_memory_staleness(self, memory: CodeLinkedMemory) -> CodeLinkedMemory:
        """Pruefe ob Code-Referenzen noch aktuell sind."""
        if not memory.code_refs:
            return memory

        any_stale = False
        stale_reasons = []

        for ref in memory.code_refs:
            if not ref.file_path or not ref.line_range or not ref.code_hash:
                continue

            try:
                current_hash = self.compute_code_hash(
                    ref.file_path,
                    ref.line_range.start,
                    ref.line_range.end
                )

                if current_hash != ref.code_hash:
                    ref.stale_since = datetime.utcnow()
                    any_stale = True
                    stale_reasons.append(f"{ref.file_path}:{ref.line_range.start}")
                else:
                    ref.last_verified_at = datetime.utcnow()
                    ref.stale_since = None

            except FileNotFoundError:
                ref.stale_since = datetime.utcnow()
                any_stale = True
                stale_reasons.append(f"{ref.file_path} (deleted)")

        memory.is_stale = any_stale
        if any_stale:
            memory.stale_reason = f"code_changed: {', '.join(stale_reasons)}"

        return memory

    def batch_check_staleness(self, memory_ids: List[str]) -> dict:
        """Batch-Pruefung mehrerer Memories."""
        results = {"checked": 0, "stale": 0, "fresh": 0, "errors": 0}

        for memory_id in memory_ids:
            try:
                memory = self.memory_store.get(memory_id)
                updated = self.check_memory_staleness(memory)
                self.memory_store.update(updated)

                results["checked"] += 1
                if updated.is_stale:
                    results["stale"] += 1
                else:
                    results["fresh"] += 1
            except Exception as e:
                results["errors"] += 1

        return results
```

#### Git Hook Integration (post-commit)

```bash
#!/bin/bash
# .git/hooks/post-commit

# Hole geaenderte Dateien
CHANGED_FILES=$(git diff-tree --name-only -r HEAD)

# Rufe Staleness-Checker auf
python -m coding_brain.staleness_checker --changed-files $CHANGED_FILES
```

```python
# coding_brain/staleness_checker.py
import argparse
from staleness_pipeline import StalenessPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--changed-files', nargs='+', required=True)
    args = parser.parse_args()

    pipeline = StalenessPipeline()

    # Finde alle Memories mit code_refs in geaenderten Dateien
    affected_memories = pipeline.find_memories_by_files(args.changed_files)

    # Pruefe und update Staleness
    results = pipeline.batch_check_staleness(affected_memories)

    print(f"Staleness check complete: {results}")

if __name__ == "__main__":
    main()
```

#### Confidence Score Formel

```
confidence_score =
    0.3 * verification_recency +
    0.2 * evidence_score +
    0.3 * code_hash_freshness +
    0.2 * source_reliability

where:
    verification_recency = max(0, 1 - (days_since_verification / 30))
    evidence_score = min(1, evidence_count / 3)
    code_hash_freshness = 1.0 if hash_current else 0.2
    source_reliability = 1.0 if source == "user" else 0.7
```

### 4.3 Schema Migration

#### Phase 1: Tags-basierte Implementierung (keine Schema-Aenderung)

In Phase 1 werden Code-Referenzen in den bestehenden `tags` gespeichert:

```python
# Speicherung in Tags (Phase 1)
tags = {
    # Erweitertes artifact_ref Format
    "code_link": "file:/apps/merlin/src/storage.ts#L42-L87",

    # Strukturierte Code-Referenz als JSON
    "code_ref_0_path": "/apps/merlin/src/storage.ts",
    "code_ref_0_lines": "42-87",
    "code_ref_0_symbol": "StorageService#createFileUploads",
    "code_ref_0_hash": "sha256:e3b0c44298fc1c149afbf4c...",
    "code_ref_0_commit": "abc123def",

    # Git-Info
    "git_commit": "abc123def",
    "git_branch": "main",

    # Staleness (manuell gesetzt)
    "stale": False,
    "stale_reason": None,
    "stale_since": None,
}
```

**Vorteile:**
- Keine Schema-Migration erforderlich
- Sofort nutzbar
- Rueckwaertskompatibel

**Nachteile:**
- Keine typisierte Validierung
- Keine effiziente Indexierung nach code_refs
- Manuelle Serialisierung/Deserialisierung

#### Phase 2: Eigene CodeReference Collection

```python
# Qdrant Collection fuer Code-Referenzen (Phase 2)
code_references_schema = {
    "memory_id": "uuid",           # Foreign Key zur Memory
    "file_path": "string",         # Indexed
    "line_start": "integer",
    "line_end": "integer",
    "symbol_id": "string",         # Indexed
    "git_commit": "string",
    "code_hash": "string",         # Indexed
    "confidence_score": "float",
    "confidence_reason": "string",
    "created_at": "datetime",
    "last_verified_at": "datetime",
    "stale_since": "datetime",     # Indexed (null = fresh)
}
```

**Migration:**

```python
def migrate_tags_to_code_refs():
    """Migriere code_ref Tags zu eigener Collection."""

    memories = memory_store.list_all()

    for memory in memories:
        if not memory.tags:
            continue

        # Extrahiere code_ref_* Tags
        code_refs = extract_code_refs_from_tags(memory.tags)

        if code_refs:
            # Speichere in neuer Collection
            for ref in code_refs:
                code_ref_store.create(
                    memory_id=memory.id,
                    **ref.to_dict()
                )

            # Entferne alte Tags
            clean_tags = {k: v for k, v in memory.tags.items()
                         if not k.startswith("code_ref_")}
            memory_store.update(memory.id, tags=clean_tags)
```

---

## 5. Implementation Plan

### Phase 1: Quick Wins (Woche 1-2)

**Ziel:** Funktionsfaehige Code-Links ohne Schema-Aenderung

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 1.1 | Erweitertes `artifact_ref` Format implementieren | 2h |
| 1.2 | `code_refs` als JSON in Tags speichern | 4h |
| 1.3 | Parser fuer `file:/path#L42-L87` Format | 2h |
| 1.4 | `git_commit` automatisch bei Memory-Erstellung | 2h |
| 1.5 | MCP-Tool Update: `add_memories` mit `code_refs` Parameter | 4h |
| 1.6 | MCP-Tool Update: `search_memory` gibt Code-Links zurueck | 4h |
| 1.7 | Tests fuer Phase 1 | 4h |

**Deliverables:**
- `add_memories` akzeptiert `code_refs` Parameter
- Gespeicherte Memories enthalten Code-Links in Tags
- Search-Ergebnisse zeigen Code-Links

### Phase 2: Schema & Staleness (Woche 3-6)

**Ziel:** Typisiertes Schema und automatische Staleness-Erkennung

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 2.1 | `CodeReference` Dataclass implementieren | 4h |
| 2.2 | Qdrant Collection fuer Code-Referenzen | 8h |
| 2.3 | Migration Tags -> Collection | 4h |
| 2.4 | `StalenessPipeline` implementieren | 8h |
| 2.5 | `ConfidenceCalculator` implementieren | 4h |
| 2.6 | Git Hook (post-commit) einrichten | 4h |
| 2.7 | Staleness-basiertes Reranking in Search | 4h |
| 2.8 | REST API Endpoints fuer Code-Refs | 8h |
| 2.9 | Tests fuer Phase 2 | 8h |

**Deliverables:**
- Typisiertes `CodeReference` Modell
- Automatische Staleness-Erkennung via Git Hook
- Search bevorzugt frische Memories

### Phase 3: Integration (Woche 7-8)

**Ziel:** Volle SCIP-Integration und bidirektionale Navigation

| Task | Beschreibung | Aufwand |
|------|--------------|---------|
| 3.1 | SCIP Symbol-Extraktion bei `add_memories` | 8h |
| 3.2 | Automatische Code-Link Suggestion | 8h |
| 3.3 | Code-zu-Memory Navigation (Reverse Lookup) | 8h |
| 3.4 | Memory-zu-Code Navigation (IDE-Integration) | 8h |
| 3.5 | Staleness Dashboard/Visualisierung | 8h |
| 3.6 | End-to-End Tests | 8h |
| 3.7 | Dokumentation | 4h |

**Deliverables:**
- Automatische Symbol-Erkennung
- "Did you mean to link to X?" Suggestions
- Bidirektionale Navigation

---

## 6. API Examples

### 6.1 add_memories mit code_refs

```python
# MCP Tool Call
add_memories(
    text="createFileUploads verwendet S3 MultipartUpload fuer grosse Dateien",
    category="architecture",
    scope="project",
    entity="FileUploadService",
    artifact_type="file",
    artifact_ref="storage.service.ts",  # Legacy
    code_refs=[
        {
            "file_path": "/apps/merlin/src/storage/storage.service.ts",
            "line_start": 42,
            "line_end": 87,
            "symbol_id": "scip-typescript npm merlin storage.service.ts/StorageService#createFileUploads().",
        },
        {
            "file_path": "/apps/merlin/src/storage/storage.service.ts",
            "line_start": 120,
            "line_end": 145,
            "symbol_id": "scip-typescript npm merlin storage.service.ts/StorageService#moveFilesToPermanentStorage().",
        }
    ],
    access_entity="project:default_org/merlin"
)
```

### 6.2 search_memory mit Staleness-Filter

```python
# MCP Tool Call - nur frische Memories
search_memory(
    query="file upload S3",
    exclude_tags="stale",  # Phase 1: Tag-basiert
    limit=10
)

# Phase 2: Dedizierter Parameter
search_memory(
    query="file upload S3",
    exclude_stale=True,
    min_confidence=0.7,
    limit=10
)
```

### 6.3 Memory-zu-Code Navigation

```python
# Response Format mit Code-Links
{
    "id": "uuid-123",
    "memory": "createFileUploads verwendet S3 MultipartUpload...",
    "category": "architecture",
    "scope": "project",
    "code_refs": [
        {
            "file_path": "/apps/merlin/src/storage/storage.service.ts",
            "line_range": {"start": 42, "end": 87},
            "symbol_id": "scip-typescript npm merlin storage.service.ts/StorageService#createFileUploads().",
            "code_link": "file:/apps/merlin/src/storage/storage.service.ts#L42-L87",
            "is_stale": False,
            "confidence_score": 0.95
        }
    ],
    "is_stale": False
}
```

### 6.4 Code-zu-Memory Navigation (Reverse Lookup)

```python
# REST API
GET /api/v1/code/memories?file_path=/apps/merlin/src/storage/storage.service.ts&line=50

# Response
{
    "memories": [
        {
            "id": "uuid-123",
            "memory": "createFileUploads verwendet S3 MultipartUpload...",
            "relevance": "direct",  # Line 50 ist innerhalb der Referenz
            "code_ref": {
                "line_range": {"start": 42, "end": 87}
            }
        }
    ]
}
```

---

## 7. Risks & Mitigations

### 7.1 Performance Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Hash-Berechnung bei grossen Dateien langsam | Mittel | Mittel | Caching, Lazy Evaluation, nur bei Bedarf berechnen |
| Staleness-Check bei vielen Memories langsam | Mittel | Hoch | Batch-Processing, Index auf file_path |
| Git Hook verlangsamt Commits | Niedrig | Mittel | Async Processing, Rate Limiting |

### 7.2 Datenintegritaet Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Staleness-Cascade bei grossen Refactorings | Hoch | Mittel | Bulk-Update Mechanismus, "Reset Staleness" Funktion |
| Verwaiste Code-Refs (geloeschte Dateien) | Mittel | Niedrig | Cleanup-Job, "deleted" Status |
| Hash-Kollisionen | Sehr niedrig | Hoch | SHA256 ist praktisch kollisionsfrei |

### 7.3 UX Risiken

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Zu viele stale Warnings nerven User | Mittel | Mittel | Threshold fuer Staleness-Anzeige, Aggregation |
| Automatische Code-Links falsch | Mittel | Mittel | Confidence-basierte Suggestions, User-Confirmation |

---

## 8. Dependencies

### 8.1 Interne Dependencies

| Dependency | Status | Beschreibung |
|------------|--------|--------------|
| SCIP-Index | Existiert | Wird fuer Symbol-Lookup verwendet |
| Code-Indexer | Existiert | `index_codebase` MCP Tool |
| Qdrant | Existiert | Vector Store fuer Memories |
| Neo4j | Existiert | Graph Store fuer Relationen |

### 8.2 Externe Dependencies

| Dependency | Version | Beschreibung |
|------------|---------|--------------|
| Git | >= 2.0 | Fuer Commit-SHA und Hooks |
| hashlib (Python) | stdlib | Fuer SHA256 Hash-Berechnung |

### 8.3 Neue Infrastruktur

| Komponente | Beschreibung | Phase |
|------------|--------------|-------|
| Git Hook Script | post-commit Hook fuer Staleness | Phase 2 |
| Staleness Cron Job | Periodische Batch-Pruefung | Phase 2 |
| Code-Ref Collection | Qdrant Collection (optional) | Phase 2 |

---

## 9. Open Questions

### 9.1 Design Decisions

1. **Wie tief soll die automatische Symbol-Extraktion gehen?**
   - Option A: Nur explizite Funktionsnamen/Klassen
   - Option B: Auch Variablen und Imports
   - **Empfehlung:** Start mit Option A, iterativ erweitern

2. **Soll Staleness UI-seitig visualisiert werden?**
   - Option A: Nur in API-Response
   - Option B: Dediziertes Staleness-Dashboard
   - **Empfehlung:** Phase 1 nur API, Phase 3 Dashboard

3. **Staleness-Schwellwert: Wann gilt eine Memory als "zu stale"?**
   - Option A: Sofort bei jeder Aenderung
   - Option B: Nur bei signifikanten Aenderungen (>10% Content)
   - **Empfehlung:** Start mit Option A, spaeter Tuning

### 9.2 Technische Fragen

1. **Wie sollen Zeilennummern-Verschiebungen behandelt werden?**
   - Wenn Code oberhalb eingefuegt wird, verschieben sich Zeilen
   - Moegliche Loesung: Symbol-ID als primaerer Identifier, Zeilen als Fallback

2. **Git-Hook vs Polling: Welcher Ansatz ist robuster?**
   - Hook: Echtzeit, aber erfordert Setup pro Repo
   - Polling: Universell, aber Latenz
   - **Empfehlung:** Hook fuer lokale Repos, Polling fuer Remote-Aenderungen

3. **Wie sollen Multi-Repo-Setups behandelt werden?**
   - Code-Refs koennen auf verschiedene Repos zeigen
   - Benoetigt `repo_id` in `CodeReference`

---

## 10. Appendix

### A. SCIP Symbol-ID Format

```
scip-<language> <package-manager> <package-name> <relative-path>/<SymbolName>#<method>(<params>).
```

Beispiele:
```
scip-typescript npm merlin storage.service.ts/StorageService#createFileUploads().
scip-python pip myapp src/utils.py/helper_function().
scip-go gomod github.com/org/repo pkg/service.go/Service.Handle().
```

### B. file: URI Format

```
file:<absolute-path>#L<start>-L<end>
```

Beispiele:
```
file:/apps/merlin/src/storage/storage.service.ts#L42-L87
file:/Users/dev/project/src/main.py#L100-L150
```

### C. Tags-basierte Speicherung (Phase 1)

```python
# Beispiel: Memory mit 2 Code-Referenzen
tags = {
    "code_ref_count": 2,

    "code_ref_0_path": "/apps/merlin/src/storage.ts",
    "code_ref_0_lines": "42-87",
    "code_ref_0_symbol": "StorageService#createFileUploads",
    "code_ref_0_hash": "sha256:e3b0c44...",
    "code_ref_0_commit": "abc123def",
    "code_ref_0_confidence": 0.95,

    "code_ref_1_path": "/apps/merlin/src/storage.ts",
    "code_ref_1_lines": "120-145",
    "code_ref_1_symbol": "StorageService#moveFilesToPermanentStorage",
    "code_ref_1_hash": "sha256:d7a8fbb...",
    "code_ref_1_commit": "abc123def",
    "code_ref_1_confidence": 0.95,

    "git_commit": "abc123def",
    "git_branch": "main",
    "stale": False,
}
```

---

## Changelog

| Version | Datum | Aenderungen |
|---------|-------|-------------|
| 1.0 | 2026-01-04 | Initial Draft basierend auf FIX-03 Analyse |

---

*Erstellt: 2026-01-04*
*Autor: Claude Code (Opus 4.5)*
*Kontext: 10-Agenten-Analyse Fix 3*
