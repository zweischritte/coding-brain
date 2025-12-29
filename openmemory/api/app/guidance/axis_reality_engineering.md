# Memory Review and Maintenance

Use periodic reviews to keep structured memories accurate and actionable.

## Review cadence
- Weekly: decisions, workflow updates, open risks
- Monthly: architecture and dependency notes
- Quarterly: security and performance assumptions

## Update workflow
```python
update_memory(
    memory_id="uuid",
    text="Updated runbook for incident response.",
    category="runbook",
    scope="org",
    add_tags={"reviewed": True},
    preserve_timestamps=True,
)
```

## Deprecation
If a memory is no longer valid, update it with context and tags like:
{"deprecated": True, "superseded_by": "ADR-202"}
