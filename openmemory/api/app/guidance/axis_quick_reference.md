# Structured Memory Quick Reference

## add_memories (required)
```python
add_memories(
    text="Capture key decision context",
    category="decision",
    scope="project",
)
```

## add_memories (full)
```python
add_memories(
    text="Adopt feature flags for risky deploys",
    category="decision",
    scope="org",
    artifact_type="service",
    artifact_ref="deploy-service",
    entity="platform-team",
    source="user",
    evidence=["ADR-101"],
    tags={"rollout": True},
)
```

## update_memory
```python
update_memory(
    memory_id="uuid",
    text="Updated decision note",
    category="decision",
    scope="org",
    entity="platform-team",
    add_tags={"revised": True},
    remove_tags=["draft"],
    preserve_timestamps=True,
)
```

## Categories
- decision: choices and trade-offs
- convention: standards, naming, defaults
- architecture: system design and structure
- dependency: third-party or internal dependencies
- workflow: process and operational flow
- testing: test strategy, coverage, quality
- security: threats, mitigations, policy
- performance: latency, scaling, tuning
- runbook: operational steps and procedures
- glossary: shared definitions and terms

## Scopes
session, user, team, project, org, enterprise

## Optional fields
artifact_type, artifact_ref, entity, source, evidence, tags

## Graph tools (quick)
- graph_entity_network: co-mentions and clusters
- graph_entity_relations: typed relations
- graph_similar_memories: similarity expansion
- graph_path_between_entities: semantic paths
- graph_aggregate: distribution by category/scope/etc
- graph_related_memories: expand from seeds
