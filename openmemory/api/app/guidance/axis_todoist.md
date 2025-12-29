# Task Capture Guide

Store tasks that represent shared workflow knowledge or durable decisions.
Ephemeral personal reminders can stay in the task system only.

## Example
```python
add_memories(
    text="Run dependency audit before the next release.",
    category="workflow",
    scope="project",
    artifact_type="repo",
    artifact_ref="coding-brain",
    tags={"priority": "high"},
)
```

```python
add_memories(
    text="Rotate API keys every quarter.",
    category="security",
    scope="org",
    entity="security-team",
    tags={"recurring": True},
)
```
