# Calendar Memory Guide

Capture recurring meetings, decisions, and follow-ups that should persist
beyond a single session.

## Examples
```python
add_memories(
    text="Weekly platform sync happens every Monday at 10:00.",
    category="workflow",
    scope="team",
    entity="platform-team",
    tags={"recurring": True},
)
```

```python
add_memories(
    text="Q2 roadmap review is scheduled for 2025-04-15.",
    category="workflow",
    scope="project",
    artifact_type="repo",
    artifact_ref="coding-brain",
)
```
