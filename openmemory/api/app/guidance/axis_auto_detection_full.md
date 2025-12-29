# Auto-Detection Guide

Use these heuristics when extracting structured metadata from free-form text.

## Category hints
- decision: decided, approved, chose, trade-off
- convention: standard, should, always, default
- architecture: design, schema, service layout, boundaries
- dependency: vendor, library, version, upgrade
- workflow: process, checklist, cadence, handoff
- testing: tests, coverage, regression, QA
- security: auth, threat, compliance, secrets
- performance: latency, throughput, scaling, benchmarks
- runbook: steps, incident, rollback, recovery
- glossary: definition, term, meaning

## Scope hints
- session: short-lived conversation context
- user: personal preferences or habits
- team: team-level agreements
- project: project-specific choices
- org: company-wide practices
- enterprise: multi-org standards

## Evidence and tags
- evidence: cite docs, tickets, or meetings
- tags: add flags like {"priority": "high"} or {"review": True}
