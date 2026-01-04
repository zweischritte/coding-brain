# PRD: meta_relations Output Optimization

## Reduce Token Waste in MCP Search Memory Responses

**Version:** 1.2
**Date:** 2026-01-04
**Status:** IMPLEMENTED ✅
**Author:** Generated from LLM feedback analysis
**Reviewers:** Plan Agent, Explore Agent (2x)
**Implementation:** Claude Opus 4.5

---

## Executive Summary

This PRD addresses the verbosity of `meta_relations` in `search_memory` responses. Analysis shows that while the graph metadata is valuable for navigation, the current JSON structure wastes tokens on:

1. Irrelevant metadata (OM_WRITTEN_VIA, OM_IN_CATEGORY, OM_IN_SCOPE)
2. Verbose JSON labels ("target_label": "OM_Entity")
3. Precision noise (exact similarity scores like 0.72)

The goal is to reduce `meta_relations` token usage by ~60-80% while preserving navigation utility.

---

## Review Findings (Critical Updates)

### Architecture Review Recommendations

Based on sub-agent code exploration and architecture review:

1. **Field Name Collision Risk**: `relations` already exists as a separate field for Mem0 Graph Memory (mcp_server.py line 1534).
   - **Resolution**: Keep field name as `meta_relations` but use compact internal structure

2. **Missing Relation Type**: `OM_HAS_EVIDENCE` not included in original PRD but valuable for tracing decisions to ADRs/PRs.
   - **Resolution**: Add `evidence` field to compact format

3. **Score Preservation**: Original PRD omitted similarity scores from compact format, but scores help LLM prioritize.
   - **Resolution**: Include scores in `similar` as objects: `[{"id": "...", "score": 0.92}]`

4. **Formatting Layer**: Apply compact formatting only at MCP response layer, NOT in underlying `get_relations_for_memories()`.
   - **Resolution**: Add `format_compact_relations()` in response_format.py, call from mcp_server.py

### Code Locations Identified

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| meta_relations added | `mcp_server.py` | 1529-1532 | Integration point for changes |
| Relation builder | `graph_ops.py` | 218-241 | Wrapper (DO NOT MODIFY) |
| Core implementation | `metadata_projector.py` | 912-967 | Returns verbose format (DO NOT MODIFY) |
| Test fixtures | `test_mcp_search_graph_enrichment.py` | 57-69 | Need to update for new format |

### Tests Requiring Updates

- `test_mcp_search_graph_enrichment.py::TestSearchEnrichment::test_enrichment_adds_meta_relations`
- `test_om_similar_enhancement.py` - All tests if changing OM_SIMILAR output shape
- `test_neo4j_metadata_projector.py::test_get_relations_for_memories` - Internal format unchanged

---

## Background & Problem Statement

### Current Behavior

When `search_memory` returns results with graph enrichment, each memory includes a `meta_relations` block:

```json
{
  "meta_relations": {
    "memory-123": [
      {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "def-456", "score": 0.72, "preview": "..."},
      {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "ghi-789", "score": 0.68, "preview": "..."},
      {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "apps/merlin/src/reports/abstract-report.service.ts"},
      {"type": "OM_HAS_ARTIFACT_TYPE", "target_label": "OM_ArtifactType", "target_value": "file"},
      {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "architecture"},
      {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
      {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "claude-code"},
      {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "area", "value": "search"},
      {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "ReportsModule"},
      {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "Redis"}
    ]
  }
}
```

### Problems Identified

| Issue | Impact | Example |
|-------|--------|---------|
| **Redundant metadata** | ~30% token waste | OM_WRITTEN_VIA, OM_IN_CATEGORY, OM_IN_SCOPE are rarely useful |
| **Verbose JSON structure** | ~25% token waste | `"target_label": "OM_Entity"` vs flat list |
| **Precision noise** | Minor waste | Exact scores (0.72) vs tier ("high"/"medium") |
| **Duplicate info** | ~10% waste | category/scope already in main result |

### What LLM Agents Actually Need

From real-world usage analysis:

| Relation Type | Value | Use Case |
|---------------|-------|----------|
| `OM_REFERENCES_ARTIFACT` | **Critical** | Jump to code location |
| `OM_SIMILAR` (IDs only) | **High** | Impact analysis, find related code |
| `OM_TAGGED` | **Medium** | Refine searches |
| `OM_ABOUT` (entities) | **Medium** | Navigate to related concepts |
| `OM_IN_CATEGORY` | **Low** | Already in main result |
| `OM_IN_SCOPE` | **Low** | Already in main result |
| `OM_WRITTEN_VIA` | **None** | Never relevant for code tasks |

---

## Proposed Solution

### Option A: Compact Structured Output (Recommended - Updated per Review)

Replace verbose `meta_relations` internals with a compact, purpose-driven structure.
Keep field name as `meta_relations` to avoid collision with Mem0's `relations` field.

```json
{
  "meta_relations": {
    "memory-123": {
      "artifact": "apps/merlin/src/reports/abstract-report.service.ts",
      "similar": [
        {"id": "def-456", "score": 0.92, "preview": "BooksReportService extends..."},
        {"id": "ghi-789", "score": 0.87, "preview": "MoviesReportService extends..."}
      ],
      "entities": ["ReportsModule", "Redis"],
      "tags": {"area": "search", "importance": "high"},
      "evidence": ["ADR-014", "PR-123"]
    }
  }
}
```

**Key changes from original proposal:**
- Keep `meta_relations` field name (avoid collision with Mem0 `relations`)
- Include scores in `similar` objects (actionable for LLM prioritization)
- Add `evidence` field for ADR/PR references
- Keep `preview` in similar (eliminates follow-up calls)

**Token reduction:** ~55-60% (slightly less due to scores/evidence, but more useful)

### Option B: Tiered Detail Levels

Add a `relation_detail` parameter with levels:

| Level | Content | Use Case |
|-------|---------|----------|
| `none` | No relations | Pure search, minimal tokens |
| `minimal` | artifact + similar IDs | Navigation (default) |
| `standard` | + entities + tags | Discovery |
| `full` | Current behavior | Debugging |

### Option C: Exclude-List Parameter

Add `exclude_relations` parameter to filter specific types:

```
search_memory(query="pagination", exclude_relations=["OM_WRITTEN_VIA", "OM_IN_CATEGORY", "OM_IN_SCOPE"])
```

**Recommendation:** Implement **Option A** as the new default, with **Option B** for explicit control.

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Define Success Criteria

1. [ ] Token reduction of 60%+ for typical `meta_relations` output
2. [ ] All navigation-critical relations preserved (artifact, similar, entities, tags)
3. [ ] New compact format backward compatible (old clients see JSON)
4. [ ] New parameter `relation_detail` controls output verbosity
5. [ ] Existing tests continue to pass
6. [ ] Performance unchanged or improved

### 1.2 Define Edge Cases

1. **Memory with no relations**: Should return empty `relations: {}`, not null
2. **Memory with only OM_SIMILAR**: `relations: {similar: [...]}`
3. **Multiple artifact refs**: Flatten to array `artifacts: [...]`
4. **Tags with complex values**: Preserve dict structure
5. **relation_detail=full**: Return current verbose format for debugging

### 1.3 Test Suite Structure

```
openmemory/api/tests/
├── test_meta_relations_compact.py     # NEW - compact format tests
├── test_relation_detail_levels.py     # NEW - detail level parameter tests
└── test_mcp_search_graph_enrichment.py  # UPDATE - verify no regression
```

### 1.4 Test Specifications

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| Compact artifact | Unit | Single artifact ref | `relations.artifact = "path/to/file.ts"` |
| Compact similar | Unit | 3 similar memories | `relations.similar = ["id1", "id2", "id3"]` |
| Compact entities | Unit | 2 entity refs | `relations.entities = ["E1", "E2"]` |
| Compact tags | Unit | Tags dict | `relations.tags = {"key": "value"}` |
| Empty relations | Unit | No graph relations | `relations = {}` |
| Exclude OM_WRITTEN_VIA | Unit | Default excludes | Field not present |
| relation_detail=none | Integration | No relations | No `relations` key |
| relation_detail=minimal | Integration | Artifact + similar only | Only those fields |
| relation_detail=full | Integration | Full verbose | Current format |
| Token reduction | Integration | Compare output sizes | >60% reduction |

---

## PHASE 2: Feature Specifications

### Feature 1: Compact Relations Formatter

**Description**: New function to convert verbose relations to compact format.

**File**: `openmemory/api/app/utils/response_format.py`

**Dependencies**: None

**Test Cases**:

- [ ] Unit test: `test_format_compact_relations_artifact`
- [ ] Unit test: `test_format_compact_relations_similar_with_scores`
- [ ] Unit test: `test_format_compact_relations_entities`
- [ ] Unit test: `test_format_compact_relations_tags`
- [ ] Unit test: `test_format_compact_relations_evidence`
- [ ] Unit test: `test_format_compact_relations_empty`
- [ ] Unit test: `test_format_compact_excludes_noise_relations`

**Implementation Approach** (Updated per Review):

```python
def format_compact_relations(relations: List[Dict]) -> Dict[str, Any]:
    """
    Convert verbose meta_relations to compact navigation format.

    Extracts:
    - artifact: File path from OM_REFERENCES_ARTIFACT
    - similar: List of {id, score, preview} from OM_SIMILAR
    - entities: List of entity names from OM_ABOUT
    - tags: Dict from OM_TAGGED
    - evidence: List of evidence refs from OM_HAS_EVIDENCE

    Excludes (noise):
    - OM_WRITTEN_VIA (never useful)
    - OM_IN_CATEGORY (in main result)
    - OM_IN_SCOPE (in main result)
    - OM_HAS_ARTIFACT_TYPE (redundant with artifact)
    """
    compact = {}

    # Extract artifact path
    artifact_refs = [r["target_value"] for r in relations
                     if r["type"] == "OM_REFERENCES_ARTIFACT" and r.get("target_value")]
    if len(artifact_refs) == 1:
        compact["artifact"] = artifact_refs[0]
    elif artifact_refs:
        compact["artifacts"] = artifact_refs

    # Extract similar memories with scores and previews (already limited to top 5)
    similar = []
    for r in relations:
        if r["type"] == "OM_SIMILAR" and r.get("target_value"):
            entry = {"id": r["target_value"]}
            if r.get("score") is not None:
                entry["score"] = r["score"]
            if r.get("preview"):
                entry["preview"] = r["preview"]
            similar.append(entry)
    if similar:
        compact["similar"] = similar

    # Extract entity names
    entities = [r["target_value"] for r in relations
                if r["type"] == "OM_ABOUT" and r.get("target_value")]
    if entities:
        compact["entities"] = entities

    # Extract tags as dict
    tags = {}
    for r in relations:
        if r["type"] == "OM_TAGGED" and r.get("target_value"):
            tag_key = r["target_value"]
            tag_value = r.get("value", True)
            tags[tag_key] = tag_value
    if tags:
        compact["tags"] = tags

    # Extract evidence refs (ADRs, PRs, issues)
    evidence_refs = [r["target_value"] for r in relations
                     if r["type"] == "OM_HAS_EVIDENCE" and r.get("target_value")]
    if evidence_refs:
        compact["evidence"] = evidence_refs

    return compact
```

**Git Commit Message**: `feat(response): add compact relations formatter`

---

### Feature 2: Add relation_detail Parameter to search_memory

**Description**: New parameter to control relation output verbosity.

**File**: `openmemory/api/app/mcp_server.py`

**Dependencies**: Feature 1

**Test Cases**:
- [ ] Integration test: `test_search_relation_detail_none`
- [ ] Integration test: `test_search_relation_detail_minimal`
- [ ] Integration test: `test_search_relation_detail_standard`
- [ ] Integration test: `test_search_relation_detail_full`

**Implementation Approach**:

```python
# In search_memory function signature:
async def search_memory(
    query: str,
    # ... existing params ...
    relation_detail: str = "standard",  # none | minimal | standard | full
) -> str:
```

Detail levels:
- `none`: No relations block at all
- `minimal`: Only `artifact` and `similar` IDs
- `standard`: artifact + similar + entities + tags (default)
- `full`: Current verbose format (for debugging)

**Git Commit Message**: `feat(search): add relation_detail parameter for output control`

---

### Feature 3: Update Default Output Format

**Description**: Change default `search_memory` output to use compact format.

**File**: `openmemory/api/app/mcp_server.py`

**Dependencies**: Features 1, 2

**Test Cases**:
- [ ] Integration test: `test_search_default_uses_compact_relations`
- [ ] Integration test: `test_token_reduction_vs_full_format`

**Implementation Approach**:

In `search_memory`, after building `meta_relations`:

```python
# Apply relation detail level
if relation_detail == "none":
    # Remove relations entirely
    pass
elif relation_detail == "full":
    # Keep current verbose format
    result_entry["meta_relations"] = raw_relations
else:
    # Compact format (minimal or standard)
    compact = format_compact_relations(raw_relations)
    if relation_detail == "minimal":
        # Only artifact + similar
        compact = {k: v for k, v in compact.items() if k in ("artifact", "artifacts", "similar")}
    result_entry["relations"] = compact
```

**Git Commit Message**: `feat(search): use compact relations format by default`

---

### Feature 4: Update MCP Tool Docstring

**Description**: Document the new relation_detail parameter and compact format.

**File**: `openmemory/api/app/mcp_server.py`

**Dependencies**: Features 1-3

**Test Cases**:
- [ ] Manual: Verify docstring appears in MCP tool listing

**Implementation Approach**:

Add to search_memory docstring:

```
- relation_detail: Output verbosity for graph relations (default: "standard")
  - "none": No relations, minimal tokens
  - "minimal": artifact path + similar memory IDs only
  - "standard": + entities + tags (recommended)
  - "full": Verbose format with all metadata (debugging)

Compact relations format (standard):
- relations.artifact: File/artifact path for code navigation
- relations.similar: Top 5 related memory IDs
- relations.entities: Related entity names
- relations.tags: Tag key-value pairs
```

**Git Commit Message**: `docs(mcp): document relation_detail parameter`

---

## PHASE 3: Development Protocol

### File Modification Summary

| File | Changes |
|------|---------|
| `openmemory/api/app/utils/response_format.py` | Add `format_compact_relations()` |
| `openmemory/api/app/mcp_server.py` | Add `relation_detail` param, use compact format |
| `openmemory/api/tests/test_meta_relations_compact.py` | NEW: Compact format unit tests |
| `openmemory/api/tests/test_relation_detail_levels.py` | NEW: Parameter integration tests |

### Implementation Order

1. **Write tests first** (TDD)
2. Implement Feature 1 (compact formatter)
3. Run tests, commit on green
4. Implement Feature 2 (relation_detail param)
5. Run tests, commit on green
6. Implement Feature 3 (default format change)
7. Run tests, commit on green
8. Implement Feature 4 (docstring)
9. Run full regression, commit on green
10. Measure token reduction, tag milestone

### Test Commands

```bash
# Run specific test file
cd /Users/grischadallmer/git/coding-brain/openmemory
docker compose exec codingbrain-mcp python -m pytest tests/test_meta_relations_compact.py -v

# Run all MCP tests
docker compose exec codingbrain-mcp python -m pytest tests/test_mcp*.py -v

# Run with coverage
docker compose exec codingbrain-mcp python -m pytest tests/ --cov=app --cov-report=term-missing
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2026-01-04
**Current Phase**: COMPLETE
**Last Action**: All features implemented and tested

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | format_compact_relations | [x] 25 tests | [x] | [x] | b176130e |
| 2 | relation_detail parameter | [x] 19 tests | [x] | [x] | f86c3221 |
| 3 | Default compact format | [x] (in #2) | [x] | [x] | f86c3221 |
| 4 | Update docstring | [x] N/A | [x] | [x] | f86c3221 |

### Decisions Made

1. **Decision**: Use flat compact structure, not nested JSON
   - **Rationale**: Minimizes tokens while preserving usability
   - **Alternatives Considered**: Nested structure (more verbose), CSV-like (less flexible)

2. **Decision**: Default to "standard" detail level
   - **Rationale**: Balances token savings with navigation utility
   - **Alternatives Considered**: "minimal" (too sparse), "full" (no improvement)

3. **Decision**: Exclude OM_WRITTEN_VIA, OM_IN_CATEGORY, OM_IN_SCOPE by default
   - **Rationale**: Never useful for code navigation, already in main result
   - **Alternatives Considered**: Keep all (token waste), configurable exclude (complexity)

### Known Issues & Blockers

- [x] Issue: Need to verify existing clients handle new format
  - Status: RESOLVED - relation_detail=full provides backward compatibility

### Notes for Next Session

> Review checklist:

- [x] Validate token reduction estimate (60%+) - ACHIEVED 65.1%
- [x] Verify compact format has all needed navigation info - YES
- [x] Check if any downstream tools depend on verbose format - relation_detail=full available
- [x] Consider adding "preview" to similar in compact format - INCLUDED

### Test Results Log

```
=== Token Reduction Analysis ===
Verbose JSON length: 1308 chars
Compact JSON length: 457 chars
Reduction: 65.1%

=== Test Summary ===
test_meta_relations_compact.py: 25 passed
test_relation_detail_levels.py: 19 passed
test_mcp_search_graph_enrichment.py: 21 passed
test_om_similar_enhancement.py: 20 passed
test_neo4j_metadata_projector.py: 35 passed
TOTAL: 120 tests passed
```

---

## Appendix A: Token Comparison Example

### Current Verbose Format (~450 tokens)

```json
{
  "meta_relations": {
    "memory-123": [
      {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "def-456", "score": 0.72, "preview": "BooksReportService extends AbstractReportService..."},
      {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "ghi-789", "score": 0.68, "preview": "MoviesReportService extends AbstractReportService..."},
      {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "jkl-012", "score": 0.65, "preview": "Report pagination uses Redis cursor..."},
      {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "apps/merlin/src/reports/abstract-report.service.ts"},
      {"type": "OM_HAS_ARTIFACT_TYPE", "target_label": "OM_ArtifactType", "target_value": "file"},
      {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "architecture"},
      {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
      {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "claude-code"},
      {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "area", "value": "search"},
      {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "importance", "value": "high"},
      {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "ReportsModule"},
      {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "Redis"}
    ]
  }
}
```

### New Compact Format (~130 tokens)

```json
{
  "relations": {
    "artifact": "apps/merlin/src/reports/abstract-report.service.ts",
    "similar": ["def-456", "ghi-789", "jkl-012"],
    "entities": ["ReportsModule", "Redis"],
    "tags": {"area": "search", "importance": "high"}
  }
}
```

**Reduction: ~71%**

---

## Appendix B: Migration Path

### Phase 1: Soft Launch
- Add `relation_detail` parameter with default="standard"
- Both formats available via parameter
- Monitor for issues

### Phase 2: Deprecation Warning
- Log warning when clients use verbose format without explicit `relation_detail=full`
- Document migration in changelog

### Phase 3: Remove Verbose Default
- Change default to compact
- Keep `relation_detail=full` for debugging

---

## Execution Checklist

When executing this PRD, follow this order:

- [ ] 1. Read Agent Scratchpad for prior context
- [ ] 2. Review success criteria
- [ ] 3. Write test files first (TDD)
- [ ] 4. Implement Feature 1 (compact formatter)
- [ ] 5. Run tests, commit on green
- [ ] 6. Implement Feature 2 (relation_detail param)
- [ ] 7. Run tests, commit on green
- [ ] 8. Implement Feature 3 (default format)
- [ ] 9. Run tests, commit on green
- [ ] 10. Implement Feature 4 (docstring)
- [ ] 11. Run full regression
- [ ] 12. Measure token reduction
- [ ] 13. Commit and tag milestone

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
