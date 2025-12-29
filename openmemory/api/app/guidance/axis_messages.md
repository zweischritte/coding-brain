# Message Capture Guide

Use structured memory for messages that represent decisions, conventions, or
durable workflow signals. Short-lived chatter can stay ephemeral.

## Common patterns
- Decisions: category="decision", scope="team" or "project"
- Standards: category="convention", scope="project"
- Status updates: category="workflow", scope="session"
- Risks and mitigations: category="security" or "performance"

## Example
```python
add_memories(
    text="Deploys should be paused during incident response.",
    category="convention",
    scope="org",
    entity="incident-response",
    tags={"priority": "high"},
)
```
