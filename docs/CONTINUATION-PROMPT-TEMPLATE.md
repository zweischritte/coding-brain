# [Phase Name] Continuation Prompt

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md` for overall plan
1. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress and daily log
1. Check Section 4 (Next Tasks) below for what to work on
1. Continue from where the last session left off

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update task status tables for completed work
   - Add entry to Daily Log with date, work completed, and notes
1. Update Section 4 (Next Tasks) with remaining work
1. Commit BOTH files together:

```bash
git add docs/CONTINUATION-PROMPT-[PHASE].md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "docs: update session progress and implementation tracker"
```

---

## 1. Current Gaps (per Implementation Plan)

<!-- List incomplete items from previous phases and current phase -->

**Phase X Gap**: [Description of what's incomplete]
**Phase Y Gap**: [Description of what's incomplete]

---

## 2. Command Reference

```bash
# [Relevant commands for this phase]
docker compose exec codingbrain-mcp [command]
```

---

## 3. Architecture Patterns

<!-- Include only patterns relevant to current phase work -->
<!-- Keep code examples minimal - just enough to show the pattern -->

### [Pattern Name]

```python
# Minimal code example showing the pattern
```

---

## 4. Next Tasks

<!-- Organize by phase/category -->
<!-- Use checkboxes for tracking -->

### [Category 1]

- [ ] Task 1
- [ ] Task 2

### [Category 2]

- [ ] Task 1
- [ ] Task 2

---

## 5. Known Issues

<!-- List blockers and issues that affect current work -->

1. **[Issue title]**: [Brief description]
2. **[Issue title]**: [Brief description]

---

## 6. Last Session Summary ([DATE])

**Completed**: [Brief description]

- [Item 1]
- [Item 2]

**Result**: [Test counts, key metrics]

---

## 7. Commit Template

```bash
git add docs/CONTINUATION-PROMPT-[PHASE].md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "$(cat <<'EOF'
docs: update session progress

Session: YYYY-MM-DD
- [Brief summary]
- [Test count if changed]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Template Usage Notes

<!--
REMOVE THIS SECTION WHEN CREATING A NEW CONTINUATION PROMPT

Guidelines for creating a new continuation prompt:

1. Keep it MINIMAL - the Plan and Progress files have the details
2. Only include information that is:
   - Actionable (next tasks, commands)
   - Not in Plan/Progress files (patterns, known issues)
   - Quick reference (current gaps summary)
3. Target size: ~150-200 lines max
4. Update the filename to match the phase: CONTINUATION-PROMPT-PHASE[X]-[NAME].md
5. Remove sections that don't apply to the current phase
6. Keep only the most recent session summary (not full history)
-->
