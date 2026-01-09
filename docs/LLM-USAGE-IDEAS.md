# Creative LLM Usage Ideas for Coding Brain

This is an idea bank for using the existing Coding Brain system (no new features). It focuses on how a developer or user can get maximum value from the current memory, graph, search, and code intelligence tools.

## Research-informed techniques to borrow

These external sources highlight techniques you can apply with the current stack:

- RAG grounding and evaluation sets (Pinecone) to reduce hallucinations and measure retrieval quality.
  - https://www.pinecone.io/learn/retrieval-augmented-generation/
- Hybrid search to combine keyword and vector retrieval (Weaviate) when queries include domain terms and natural language.
  - https://weaviate.io/developers/weaviate/search/hybrid
- Reranking to improve top-k relevance (Cohere) after initial retrieval.
  - https://docs.cohere.com/docs/rerank-overview
- Graph-based retrieval (GraphRAG) to answer multi-hop, relationship-heavy questions.
  - https://microsoft.github.io/graphrag/
- RAG evaluation metrics (Ragas) like context precision, context recall, and faithfulness.
  - https://docs.ragas.io/en/stable/

How these map to Coding Brain now:
- Hybrid retrieval is already supported (OpenSearch lexical + vector search + graph fusion in memory and code search).
- Reranking exists in memory search (metadata-based) and code search (tri-hybrid retrieval pipeline).
- Graph queries can be used for multi-hop reasoning (entity networks, related memories, tag co-occurrence).
- Evaluation can be done by building query sets and logging feedback and experiments already supported by the API.

## Core usage patterns

### 1) Memory capture as structured knowledge
- Store decisions and conventions with categories and evidence links (PRs, ADRs, issues).
- Use entity labels to create a navigable knowledge graph (components, teams, services, projects).
- Capture short-lived learnings (debug notes, incident notes) with `access_entity=user:<id>`.
- Tag memories by lifecycle stage (draft, validated, deprecated) and update, not delete, when they evolve.
- Save prompts that work well as memory entries so they can be reused by the team.

### 2) Access control as a usage feature
- Keep personal workflows private with `access_entity=user:<id>` while sharing runbooks and decisions with project/team/org access entities.
- Use access_entity to share just enough: project-level for repo knowledge, team-level for broader practices.
- Scope is legacy metadata only; it can be omitted and derived from access_entity. Do not rely on it for visibility.
- When a question spans teams, use `access_entity=org:<org>` for common standards and policies.

### 3) Retrieval patterns that improve answers
- Use hybrid search for ambiguous terms (lexical + vector) to capture both jargon and meaning.
- Use recency weighting for "current state" questions (deploy status, latest decision).
- Use tag and entity boosts to focus retrieval on specific components or teams.
- When results are noisy, rerank by narrowing the metadata or splitting the query into sub-questions.
- Run related-memory graph queries to expand context beyond the top-k hits.

### 4) Graph-first reasoning
- Use entity networks to surface hidden relationships between components and teams.
- Use tag co-occurrence to discover implicit clusters (e.g., tags that always appear together).
- Use path queries to answer "how does X relate to Y" across memories.
- Use related memories to build multi-hop answers instead of relying on a single chunk.

### 5) Code intelligence as a daily tool
- Index the repo after large refactors to refresh the code graph and search index.
- Use code search to answer "where is this behavior defined" and "what uses this symbol".
- Use callers/callees to navigate change impact and avoid hidden dependencies.
- Use impact analysis to estimate blast radius before modifying critical modules.
- Use test generation to quickly draft test coverage for risky changes.
- Use PR analysis to surface risks, conventions, and security issues.
- Use ADR automation when architectural changes are detected, then store the ADR as a memory.

## Creative idea bank (no new features)

### Individual developer workflows
- Maintain a personal "debug diary" with each issue, reproduction steps, root cause, and fix.
- Store daily notes about code areas touched and tag them by component for later retrieval.
- Build a personal glossary of acronyms and internal terms with `access_entity=user:<id>`.
- Ask the LLM to explain a symbol, then store the explanation as memory with evidence links.
- Use the LLM to draft a change plan and link it to relevant code call graphs.

### Team knowledge workflows
- Create a shared runbook memory for each on-call alert, updated after incidents.
- Capture architecture decisions as memories with category=decision and evidence=ADR.
- Maintain a team "known issues" memory list and link each to logs or incidents.
- Store onboarding checklists and link to key entities (services, owners, deploy steps).
- Use the UI to browse memory networks during incident retrospectives.

### Feature delivery workflows
- Start with a feature prompt and retrieve relevant decisions, tags, and prior incidents.
- Use code search to locate existing patterns, then store the pattern as a reusable memory.
- Use impact analysis to predict which tests must be updated.
- Use PR analysis to generate a risk checklist and store it as evidence.
- Summarize the release notes using memories tagged with the feature name.

### Debugging and incident response
- Query by error signature and graph-expand to related incidents and fixes.
- Use entity paths to find the responsible component and owner quickly.
- Store a "timeline memory" during an incident as a single shared record.
- Use search with recency bias to find the latest mitigation steps.
- After resolution, update the memory with lessons learned and a future guardrail.

### Documentation and knowledge base
- Turn repeated Slack answers into memories with tags=faq and entity=team.
- Build a living docs index by storing summaries and linking to canonical docs.
- Use graph queries to find missing documentation for high-centrality entities.
- Track deprecated APIs by tagging memories and boosting them in retrieval.
- Create a rolling "tech debt list" with `access_entity=project:<org>/<repo>` and update it monthly.

### Architecture and design
- Capture constraints and tradeoffs as structured decisions, not freeform notes.
- Use code graphs to validate that new designs match existing module boundaries.
- Store post-mortems as memories and link to impacted components.
- Use path queries to visualize dependencies between services before changes.
- Use test generation to create guardrails for architectural boundaries.

### Quality and testing
- Store flaky test patterns as memories with tags=flaky and entity=tests.
- Use PR analysis to spot missing test coverage and store the outcome.
- Build a small eval set of query->expected answer pairs to test retrieval changes.
- Track retrieval quality over time using feedback and experiments endpoints.
- Use RAG evaluation metrics (context precision, faithfulness) to guide tuning.

### Security and compliance
- Store security decisions with evidence links to audits or tickets.
- Tag memories that contain sensitive handling rules and boost them during retrieval.
- Use scoped access_entity to keep sensitive runbooks inside the correct team or project.
- Use GDPR endpoints for export or deletion workflows when required.
- Use audit logging to track edits to critical knowledge records.

### Guidance and business concepts
- Use guidance MCP to deliver company-specific behavioral guidance on demand.
- Use business concepts graph to keep product vocabulary consistent across teams.
- Link concept entities to code components for better cross-team alignment.
- Build onboarding flows that pull both concepts and code context into one answer.

### Cross-repo and multi-team usage
- Index multiple repos and use cross-repo insights to detect duplicate work.
- Store integration contracts as shared memories across projects.
- Use org-wide tags for standard patterns (auth, logging, tracing).
- Use graph paths to identify cross-team dependencies before big changes.

### UI and human workflows
- Use the UI to browse memory lineage when writing retrospectives.
- Use the UI to validate what a new team member can access with their grants.
- Use the UI to spot entity clusters that indicate under-documented areas.

## End-to-end usage recipes

### A) "Investigate a production error" flow
1. Search memories by error signature (recency-weighted).
2. Expand with related memories and entity network queries.
3. Use code search to locate call sites and configuration references.
4. Use callers/callees to identify upstream and downstream effects.
5. Store the fix as a runbook memory with evidence.

### B) "Design a new feature" flow
1. Search for prior decisions and conventions tagged with the component.
2. Use code search to find existing patterns and modules.
3. Use impact analysis to estimate changes needed.
4. Draft a plan and store it as a decision or workflow memory.

### C) "Onboard a new engineer" flow
1. Provide a curated list of top entities and memories (team, components, runbooks).
2. Use graph queries to surface a short map of dependencies.
3. Use code search to answer "where is X implemented" on day one.
4. Use guidance MCP to keep onboarding answers consistent.

### D) "Improve retrieval quality" flow
1. Build a small set of representative queries (5-20).
2. Compare search results with and without recency and tag boosts.
3. Use feedback endpoints to log good and bad results.
4. Run experiments and keep a simple evaluation log as memories.

## Notes on working within current capabilities

- Code tools require indexing; refresh after major refactors.
- Memory search is stronger when entities and tags are consistent.
- Graph queries are powerful for multi-hop questions that plain search misses.
- Access_entity is a feature: use it to guarantee correct visibility.

## Sources consulted (web research)
- Pinecone RAG overview: https://www.pinecone.io/learn/retrieval-augmented-generation/
- Weaviate hybrid search: https://weaviate.io/developers/weaviate/search/hybrid
- Cohere rerank overview: https://docs.cohere.com/docs/rerank-overview
- Microsoft GraphRAG docs: https://microsoft.github.io/graphrag/
- Ragas evaluation docs: https://docs.ragas.io/en/stable/
