# Risk and Contradiction Guide

Capture conflicting statements, assumptions, and risks so they can be resolved
in future reviews.

## Contradictions
```python
add_memories(
    text="Team agreed to ship weekly, but release process still requires biweekly approvals.",
    category="decision",
    scope="team",
    tags={"contradiction": True},
)
```

## Risks
```python
add_memories(
    text="Current database schema cannot handle projected Q3 traffic.",
    category="performance",
    scope="project",
    artifact_type="db",
    artifact_ref="primary-postgres",
    evidence=["Load test report 2025-01-20"],
)
```
