# Structured Memory Examples

Use these examples to capture durable facts, decisions, and workflows with
structured metadata.

## People and roles
```python
add_memories(
    text="Alex is the platform on-call lead this quarter.",
    category="glossary",
    scope="team",
    entity="Alex",
    tags={"role": "oncall-lead"},
)
```

```python
add_memories(
    text="Security reviews are owned by the AppSec team.",
    category="glossary",
    scope="org",
    entity="AppSec",
    evidence=["Security handbook v2"],
)
```

## Project context
```python
add_memories(
    text="Project documentation lives in the /docs directory.",
    category="convention",
    scope="project",
    artifact_type="repo",
    artifact_ref="coding-brain",
)
```

## Decisions
```python
add_memories(
    text="Adopt feature flags for risky deploys.",
    category="decision",
    scope="org",
    artifact_type="service",
    artifact_ref="deploy-service",
    entity="platform-team",
    evidence=["ADR-101"],
    tags={"rollout": True},
)
```

## Operations
```python
add_memories(
    text="Weekly release checklist is required before production deploys.",
    category="runbook",
    scope="team",
    entity="release-engineering",
)
```
