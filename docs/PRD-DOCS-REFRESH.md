# PRD: Docs Refresh for Access Entity, Graph Visibility, and Code Edge Controls

Owner: Coding Brain
Status: Draft
Last Updated: 2026-01-08

---

## 1) Background
The codebase has shifted to access_entity-first visibility, expanded graph backfill tooling, added entity displayName support, and introduced deterministic vs inferred edge controls in code intelligence. The current docs still describe scope-based access and omit several new scripts and behaviors, which risks incorrect integrations and operational drift.

---

## 2) Goals
- Align all listed docs with the access_entity-only access model (scope is legacy metadata only and can be derived from access_entity when omitted).
- Document entity name normalization and displayName usage for UI/output.
- Document graph visibility rules: accessEntity drives visibility; userId is audit only.
- Add and explain new backfill scripts and when to run them.
- Document deterministic vs inferred code edges and the include_inferred_edges control.
- Refresh MCP tool guidance and test plans to reflect current behavior and outputs.

---

## 3) Non-Goals
- No API changes or new features.
- No modifications to database schemas beyond what already exists.
- No updates to docs outside the explicitly requested files.

---

## 4) In-Scope Docs
- docs/README-CODING-BRAIN.md
- docs/TECHNICAL-ARCHITECTURE.md
- docs/SYSTEM-CONTEXT.md
- docs/RUNBOOK-DEPLOYMENT.md
- docs/OPERATIONS-READINESS.md
- docs/MCP-TOOL-TEST-PLAN.md
- docs/LLM-USAGE-IDEAS.md
- docs/LLM-REPO-INTEGRATION-GUIDE.md
- docs/LLM-REPO-INTEGRATION-GUIDE-GENERIC.md
- docs/FINAL-SYSTEM-GOALS.md

---

## 5) Deprecations to Reflect
- Scope-based access control. Scope is kept as optional metadata only and is not used for visibility or permissions.
- Any guidance that implies scope selection drives access control.

---

## 6) Additions to Document
- Access model: access_entity controls visibility for memories, entities, and graph relations. userId is audit-only.
- Entity naming: OM_Entity.name is normalized (lowercase + underscores) for matching; displayName preserves original casing for UI.
- Graph response changes: entity endpoints and relation_detail outputs surface displayName (entityDisplayNames or targetDisplayName).
- Backfill scripts:
  - app/scripts/backfill_graph_access_entity_hybrid.py (metadata + edges per access_entity with resume).
  - app/scripts/backfill_entity_bridge_access_entity.py (LLM entity bridge per access_entity with resume).
  - app/scripts/backfill_entity_display_names.py (set displayName from repo sources).
- Code intelligence edge controls:
  - include_inferred_edges parameter on call graph tools.
  - CODE_INTEL_INCLUDE_INFERRED_EDGES env default.
  - Deterministic edge extraction module and how to disable inferred edges for strict results.

---

## 7) Success Criteria
- All in-scope docs explicitly state access_entity drives access control and scope is legacy metadata (optional and derived from access_entity when omitted).
- All relevant examples use access_entity and omit scope unless calling out legacy behavior (scope can be derived from access_entity).
- New backfill scripts and displayName behavior are documented with usage notes.
- MCP tool test plan includes displayName outputs and include_inferred_edges coverage.

---

## 8) Risks / Open Questions
- Some clients may still rely on scope fields; docs should call out compatibility without encouraging new usage.
- Access_entity defaults (auto resolution) should be described carefully to avoid ambiguous grants.

---

## 9) Rollout / Validation
- Update docs in one pass.
- Spot-check MCP outputs (search_memory relation_detail, graph entity endpoints) for displayName fields.
- Verify runbook updates include new backfill scripts and sequencing.
