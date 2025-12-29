"""
Business Concept Extractor for Transcripts

Auto-extracts business entities and concepts from video transcripts,
documents, and mixed German/English content.

Usage:
    from app.utils.concept_extractor import ConceptExtractor

    extractor = ConceptExtractor(api_key=os.getenv("OPENAI_API_KEY"))
    result = extractor.extract_full(transcript_text)

    for entity in result.entities:
        print(f"{entity.entity} ({entity.type}) - importance: {entity.importance}")

    for concept in result.concepts:
        print(f"{concept.concept} - confidence: {concept.confidence}")

Based on research:
- LLM-IE: https://pmc.ncbi.nlm.nih.gov/articles/PMC11901043/
- Structured Extraction: https://simonwillison.net/2025/Feb/28/llm-schemas/
- PromptNER: https://arxiv.org/abs/2305.15444

See: /docs/AUTO-EXTRACT-BUSINESS-CONCEPTS-RESEARCH.md
"""

import json
import re
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field
import tiktoken

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "openai package is required. Install with: pip install openai"
    )

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

class BusinessEntity(BaseModel):
    """Extracted business entity with metadata"""
    entity: str = Field(description="The extracted entity text (preserve original language)")
    type: Literal[
        # Existing types
        "company", "person", "product", "market",
        "metric", "business_model", "technology", "strategy",
        # NEW types
        "tactic",           # Specific how-to knowledge
        "case_study",       # Person + outcome with numbers
        "product_idea",     # Feature/architecture concepts
        "framework",        # Mental models, philosophies
        "competitive_intel", # Market structure, competitor info
        "pricing",          # Pricing tiers, billing structures
        "tool_config",      # Specific tool settings/configurations
    ]
    importance: float = Field(
        ge=0.0, le=1.0,
        description="Importance score: 0.9-1.0=core, 0.7-0.9=important, 0.5-0.7=supporting, 0.0-0.5=mentioned"
    )
    context: str = Field(description="Surrounding context (1-2 sentences)")
    mention_count: int = Field(default=1, description="How many times mentioned in text")


class BusinessConcept(BaseModel):
    """Extracted business concept or insight"""
    concept: str = Field(description="The business concept or insight statement")
    type: Literal[
        # Existing types
        "causal", "pattern", "comparison", "trend",
        "contradiction", "hypothesis", "fact",
        # NEW types
        "implementation",    # Step-by-step how-to
        "product_architecture", # System design concepts
        "market_structure",  # Competitive landscape
        "success_story",     # Case study with outcome
        "mental_model",      # Framework for thinking
        "pricing_insight",   # Pricing/billing patterns
    ]
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence: 0.9-1.0=validated, 0.7-0.9=likely, 0.5-0.7=hypothesis, 0.0-0.5=speculative"
    )
    evidence: List[str] = Field(description="Supporting quotes from transcript (verbatim)")
    entities: List[str] = Field(description="Related entities mentioned in concept")
    source_type: Literal["stated_fact", "inference", "opinion"] = Field(
        description="stated_fact=explicitly said, inference=derived, opinion=subjective view"
    )


class TranscriptExtraction(BaseModel):
    """Complete extraction result from transcript"""
    entities: List[BusinessEntity]
    concepts: List[BusinessConcept]
    summary: str = Field(description="1-2 sentence summary of main business topics")
    language: Literal["en", "de", "mixed"] = Field(description="Detected language")


# ============================================================================
# PROMPTS
# ============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a business intelligence analyst extracting structured information from transcripts.

ENTITY TYPES AND DEFINITIONS:

=== CORE BUSINESS ENTITIES ===

1. COMPANY: Any business, startup, organization, or company name
   Example: "Stripe", "Google", "that Berlin-based startup", "OpenRouter", "GoHighLevel"

2. PERSON: Founders, investors, employees, customers, advisors (include role if mentioned)
   Example: "Sarah (CEO)", "John from Sequoia", "Carlos Mac (mentorship student)"

3. PRODUCT: Products, services, features, platforms, tools
   Example: "AI-powered CRM", "Cursor IDE", "Claude Code", "Limitless pendant"

4. MARKET: Customer segments, industries, geographies, target markets
   Example: "B2B SaaS companies", "German SMEs", "healthcare sector"

5. METRIC: Revenue, users, growth rates, KPIs (always include units/values)
   Example: "€500k ARR", "10,000 users", "30% MoM growth", "$60/hour"

6. BUSINESS_MODEL: Pricing strategies, revenue streams, GTM approaches
   Example: "freemium with enterprise upsell", "usage-based pricing", "white-label SaaS"

7. TECHNOLOGY: Technical platforms, programming languages, infrastructure, AI tools
   Example: "React", "AWS", "Neo4j graph memory", "multi-agent architecture"

8. STRATEGY: Goals, initiatives, plans, strategic decisions
   Example: "expand to US market", "hire sales team", "pivot to B2B"

=== NEW: ACTIONABLE KNOWLEDGE ENTITIES ===

9. TACTIC: Specific how-to knowledge, implementation steps, configurations
   - Must be actionable and specific
   - Include tool names and settings
   Example: "Set credit limits per API key in OpenRouter with monthly reset"
   Example: "Use Jarvis + Fireflies integration for automatic meeting context"
   Example: "Create separate API key per user for usage tracking"

10. CASE_STUDY: Person/company + specific outcome with numbers
    - Must include: WHO + WHAT THEY DID + MEASURABLE RESULT
    - Include timeframes when mentioned
    Example: "Carlos Mac: $5k mentorship investment → $10k revenue in 1 month"
    Example: "Parth Bhangale: months of grinding on education-as-service → now killing it"

11. PRODUCT_IDEA: Feature concepts, architecture ideas, product visions
    - Specific enough to implement
    - Include the "what" and "why"
    Example: "Quantum Mirror as separate tab in Jarvis for psychological coaching"
    Example: "Third-party AI agent filter to validate memory inferences"
    Example: "USB drive of myself - portable personal AI identity"

12. FRAMEWORK: Mental models, philosophies, principles for decision-making
    - Named frameworks or clear principles
    - Include the core insight
    Example: "FAFO (fuck around and find out) - learn by experimentation"
    Example: "Build systems around processes, not models - AI improves, system improves"
    Example: "Authenticity as AI defense - be so yourself that AI can't replicate"

13. COMPETITIVE_INTEL: Market structure, competitor comparisons, positioning
    - Market players and their relationships
    - Competitive advantages/disadvantages
    Example: "Only two players in external memory market: Open Memory vs Super Memory"
    Example: "Perplexity's wrapper model - bulk API pricing passed to users with 8.5% fee"

14. PRICING: Pricing tiers, billing structures, cost breakdowns
    - Specific numbers and what you get
    - Traps or gotchas in pricing
    Example: "$297 vs $497 GoHighLevel plan - SMS billing trap on lower tier"
    Example: "Open Memory: $250/month for graph features, or self-host (complicated)"
    Example: "OpenRouter: 8.5% service fee on top of model costs"

15. TOOL_CONFIG: Specific tool settings, configurations, integrations
    - Tool name + specific setting + what it does
    - Include gotchas or non-obvious features
    Example: "OpenRouter API keys: can set credit limit, monthly reset, expiration date"
    Example: "Limitless pendant: breaks audio into 10-min chunks, loses full context"

IMPORTANCE SCORING (0.0-1.0):
- 0.9-1.0: Core to discussion, directly actionable, mentioned multiple times
- 0.7-0.9: Important context, useful for decisions
- 0.5-0.7: Supporting detail, good to know
- 0.0-0.5: Briefly mentioned, tangential

RULES:
1. Preserve original language (don't translate German ↔ English)
2. For mixed content, mark language as "mixed"
3. Include context: 1-2 sentences surrounding the entity
4. Normalize similar entities (e.g., "10k users" and "10,000 users" → same entity)
5. PRIORITIZE actionable knowledge (tactics, configs, case studies) over abstract entities
6. For case studies, ALWAYS include the measurable outcome
7. For pricing, ALWAYS include specific numbers
8. For tool configs, ALWAYS include the tool name and specific setting

EXTRACTION PRIORITY (extract these first):
1. Tactics and tool configurations (most actionable)
2. Case studies with outcomes (social proof)
3. Pricing and competitive intel (decision-making)
4. Product ideas (innovation pipeline)
5. Frameworks (mental models)
6. Then standard entities (companies, people, metrics, etc.)

Now extract entities from the following transcript."""

CONCEPT_EXTRACTION_SYSTEM_PROMPT = """You are a business analyst extracting insights and concepts from transcripts.

CONCEPT TYPES:

=== EXISTING CONCEPT TYPES ===

1. CAUSAL: X causes Y, X leads to Y, X is the reason for Y
   Example: "High churn is caused by poor onboarding"

2. PATTERN: When X, then Y; Whenever X happens, Y follows
   Example: "When we launch on ProductHunt, we get 100+ signups in the first 24h"

3. COMPARISON: X vs Y, X is better/worse than Y, X differs from Y
   Example: "Direct sales works better than PLG for enterprise customers"

4. TREND: X is increasing/decreasing/changing, X is growing/declining
   Example: "Customer acquisition costs have doubled in the last 6 months"

5. CONTRADICTION: X but Y, X however Y, despite X, Y
   Example: "Product is strong but distribution is weak"

6. HYPOTHESIS: We think X, We believe Y, Our hypothesis is Z
   Example: "We believe the German market is ready for AI-powered legal tools"

7. FACT: Concrete, verifiable statement with specifics
   Example: "We raised €500k in seed funding in Q2 2024"

=== NEW CONCEPT TYPES ===

8. IMPLEMENTATION: Step-by-step how-to, specific implementation approach
   - Must be actionable
   - Should include enough detail to execute
   Example: "To track per-user API usage: create API key per user in OpenRouter, set credit limit, enable monthly reset"
   Example: "For AI memory validation: create third-party agent that double-checks inferences before storing"

9. PRODUCT_ARCHITECTURE: System design, feature architecture, technical approach
   - How components fit together
   - Design decisions and their rationale
   Example: "Multi-agent with persona switching: daily assistant mode + psychological coach mode, same graph memory"
   Example: "Jarvis architecture: OpenMemory for graph, Fireflies for call transcripts, auto-scheduling from conversation context"

10. MARKET_STRUCTURE: Competitive landscape, market dynamics, player positioning
    - Who the players are
    - How they compete/differentiate
    Example: "External AI memory market is a duopoly: Open Memory (preferred, $250/mo for graph) vs Super Memory"
    Example: "API aggregators like OpenRouter and Perplexity use wrapper model - bulk pricing + service fee"

11. SUCCESS_STORY: Case study with clear before/after and measurable outcome
    - WHO did WHAT and got WHAT RESULT
    - Include timeframe
    Example: "Carlos Mac: 1 year watching videos → $5k mentorship → $10k revenue in first month"
    Example: "Eight-figure company paying $60/hr for engineers not using AI IDEs → $10-20k education consulting opportunity"

12. MENTAL_MODEL: Framework for thinking, decision-making principle
    - Named or nameable framework
    - Core insight that can be applied elsewhere
    Example: "FAFO principle: Learn by experimentation rather than excessive research"
    Example: "System over model thinking: Build around processes, not specific AI models - as AI improves, system improves"
    Example: "Authenticity as moat: Be so uniquely yourself that AI can't replicate your style"

13. PRICING_INSIGHT: Pricing strategy observation, billing gotcha, cost structure
    - Specific numbers
    - Non-obvious implications
    Example: "GoHighLevel $297 plan trap: SMS costs billed directly to your account, $497 plan includes them"
    Example: "OpenRouter credit system allows per-key limits - solves SaaS abuse problem without custom code"

SOURCE TYPES:
- stated_fact: Explicitly stated in the transcript
- inference: Implied or derived from multiple statements
- opinion: Subjective view or belief

CONFIDENCE SCORING (0.0-1.0):
- 0.9-1.0: Validated with data/evidence, specific numbers, multiple supporting quotes
- 0.7-0.9: Clearly stated, credible source, consistent with context
- 0.5-0.7: Plausible hypothesis, limited evidence
- 0.0-0.5: Speculative, contradictory info

EVIDENCE RULES:
1. Extract verbatim quotes (don't paraphrase)
2. Include 2-3 supporting quotes per concept
3. Keep quotes concise (1-2 sentences max)
4. For implementation/architecture concepts, include the specific details mentioned

EXTRACTION PRIORITY (extract these first):
1. Implementation concepts (most actionable)
2. Success stories (social proof with numbers)
3. Pricing insights (decision-making intel)
4. Market structure (competitive landscape)
5. Mental models (reusable frameworks)
6. Product architecture (innovation ideas)
7. Then standard concepts (causal, pattern, trend, etc.)

IMPORTANT: Don't extract vague observations. Every concept should be:
- Specific enough to act on, OR
- Specific enough to make a decision, OR
- Specific enough to remember and apply later

BAD: "AI is useful for businesses" (too vague)
GOOD: "Eight-figure companies don't know about Cursor/Claude Code - $10-20k education consulting opportunity"

Now extract concepts from the following transcript."""


# ============================================================================
# CONCEPT EXTRACTOR CLASS
# ============================================================================

class ConceptExtractor:
    """
    Extracts business entities and concepts from transcripts using LLMs.

    Features:
    - Structured extraction with Pydantic schemas
    - Confidence scoring
    - German + English support
    - Token counting for cost estimation
    - Batch processing for long transcripts

    Example:
        >>> extractor = ConceptExtractor(api_key=os.getenv("OPENAI_API_KEY"))
        >>> result = extractor.extract_full("Your transcript text here...")
        >>> print(f"Found {len(result.entities)} entities and {len(result.concepts)} concepts")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens_per_chunk: int = 8000
    ):
        """
        Initialize extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini recommended for cost/performance)
            max_tokens_per_chunk: Max tokens per extraction call
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens_per_chunk = max_tokens_per_chunk
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Split text into chunks that fit within token limit.
        Tries to split on paragraph boundaries.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def extract_entities(self, text: str) -> List[BusinessEntity]:
        """
        Extract business entities from text.

        Returns:
            List of BusinessEntity objects
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format=TranscriptExtraction,
                temperature=0.1  # Low temperature for consistent extraction
            )

            extraction = completion.choices[0].message.parsed
            return extraction.entities

        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    def extract_concepts(self, text: str) -> List[BusinessConcept]:
        """
        Extract business concepts and insights from text.

        Returns:
            List of BusinessConcept objects
        """
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format=TranscriptExtraction,
                temperature=0.2  # Slightly higher for concept synthesis
            )

            extraction = completion.choices[0].message.parsed
            return extraction.concepts

        except Exception as e:
            print(f"Concept extraction error: {e}")
            return []

    def extract_full(self, text: str) -> TranscriptExtraction:
        """
        Full extraction: entities + concepts + summary.

        For long texts, chunks and merges results.

        Args:
            text: Transcript or document text to extract from

        Returns:
            TranscriptExtraction object with entities, concepts, summary, and language
        """
        tokens = self.count_tokens(text)

        # If text is short enough, extract in one call
        if tokens <= self.max_tokens_per_chunk:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"{ENTITY_EXTRACTION_SYSTEM_PROMPT}\n\n{CONCEPT_EXTRACTION_SYSTEM_PROMPT}"
                        },
                        {"role": "user", "content": text}
                    ],
                    response_format=TranscriptExtraction,
                    temperature=0.1
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                print(f"Full extraction error: {e}")
                return TranscriptExtraction(
                    entities=[], concepts=[], summary="Extraction failed", language="en"
                )

        # For long texts, chunk and merge
        chunks = self.chunk_text(text, max_tokens=self.max_tokens_per_chunk)
        all_entities = []
        all_concepts = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            entities = self.extract_entities(chunk)
            concepts = self.extract_concepts(chunk)
            all_entities.extend(entities)
            all_concepts.extend(concepts)

        # Deduplicate entities (merge similar ones)
        all_entities = self._deduplicate_entities(all_entities)

        # Generate summary from all concepts
        summary = self._generate_summary(all_concepts)

        # Detect language
        language = self._detect_language(text)

        return TranscriptExtraction(
            entities=all_entities,
            concepts=all_concepts,
            summary=summary,
            language=language
        )

    def _deduplicate_entities(self, entities: List[BusinessEntity]) -> List[BusinessEntity]:
        """
        Merge similar entities and update mention counts.

        Simple deduplication by exact match (lowercased).
        In production, consider using fuzzy matching (Levenshtein distance).
        """
        entity_map = {}

        for entity in entities:
            key = entity.entity.lower().strip()
            if key in entity_map:
                # Merge: keep higher importance, sum mention counts
                existing = entity_map[key]
                existing.importance = max(existing.importance, entity.importance)
                existing.mention_count += entity.mention_count
            else:
                entity_map[key] = entity

        return list(entity_map.values())

    def _generate_summary(self, concepts: List[BusinessConcept]) -> str:
        """Generate 1-2 sentence summary from concepts"""
        if not concepts:
            return "No concepts extracted."

        # Simple heuristic: pick top 2 highest-confidence concepts
        top_concepts = sorted(concepts, key=lambda c: c.confidence, reverse=True)[:2]
        summary_parts = [c.concept for c in top_concepts]
        return " ".join(summary_parts)

    def _detect_language(self, text: str) -> Literal["en", "de", "mixed"]:
        """Simple language detection based on common words"""
        # Count common German words
        german_indicators = ["der", "die", "das", "und", "ist", "wir", "haben", "werden", "sein"]
        german_count = sum(1 for word in german_indicators if f" {word.lower()} " in f" {text.lower()} ")

        # Count common English words
        english_indicators = ["the", "and", "is", "we", "have", "are", "will", "be"]
        english_count = sum(1 for word in english_indicators if f" {word.lower()} " in f" {text.lower()} ")

        if german_count > 3 and english_count > 3:
            return "mixed"
        elif german_count > english_count:
            return "de"
        else:
            return "en"

    def estimate_cost(self, text: str) -> dict:
        """
        Estimate extraction cost for given text.

        Returns:
            {
                "input_tokens": int,
                "estimated_output_tokens": int,
                "estimated_cost_usd": float
            }
        """
        input_tokens = self.count_tokens(text)

        # Rough estimate: output is ~20% of input for extraction tasks
        estimated_output_tokens = int(input_tokens * 0.2)

        # GPT-4o-mini pricing (as of Dec 2024)
        # Input: $0.150 per 1M tokens
        # Output: $0.600 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 0.150
        output_cost = (estimated_output_tokens / 1_000_000) * 0.600

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ============================================================================
# INTEGRATION WITH MEMORY SYSTEM
# ============================================================================


def extract_from_memory(
    memory_id: str,
    user_id: str,
    content: str,
    category: str = None,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    max_tokens_per_chunk: int = 8000,
    min_confidence: float = 0.5,
    store_in_graph: bool = True,
) -> Dict:
    """
    Extract and store business concepts from a memory.

    This is the main integration point with the OpenMemory system.
    It extracts concepts and entities, then stores them in Neo4j.

    Args:
        memory_id: UUID of the source memory
        user_id: User ID for scoping
        content: Memory content to extract from
        category: Optional category for concept scoping
        api_key: OpenAI API key (uses config if not provided)
        model: Model for extraction (default: gpt-4o-mini)
        max_tokens_per_chunk: Max tokens per chunk
        min_confidence: Minimum confidence for storage
        store_in_graph: Whether to store in Neo4j graph

    Returns:
        Dict with extraction results:
        - entities: List of extracted entities
        - concepts: List of extracted concepts
        - summary: Extraction summary
        - language: Detected language
        - stored_entities: Count of entities stored (if store_in_graph)
        - stored_concepts: Count of concepts stored (if store_in_graph)
        - error: Error message if extraction failed
    """
    import os
    from typing import Dict

    # Get API key from config if not provided
    if not api_key:
        try:
            from app.config import BusinessConceptsConfig
            api_key = BusinessConceptsConfig.get_openai_api_key()
            model = BusinessConceptsConfig.get_extraction_model()
            max_tokens_per_chunk = BusinessConceptsConfig.get_max_tokens_per_chunk()
            min_confidence = BusinessConceptsConfig.get_min_confidence()
        except ImportError:
            api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        return {"error": "OpenAI API key not configured"}

    try:
        # Perform extraction
        extractor = ConceptExtractor(
            api_key=api_key,
            model=model,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )

        extraction = extractor.extract_full(content)

        result = {
            "entities": [e.model_dump() for e in extraction.entities],
            "concepts": [c.model_dump() for c in extraction.concepts],
            "summary": extraction.summary,
            "language": extraction.language,
        }

        # Store in graph if enabled
        if store_in_graph:
            try:
                from app.graph.concept_projector import get_projector

                projector = get_projector()
                if projector:
                    stored_entities = 0
                    stored_concepts = 0

                    # Store entities
                    for entity in extraction.entities:
                        if entity.importance >= min_confidence:
                            entity_result = projector.upsert_bizentity(
                                user_id=user_id,
                                name=entity.entity,
                                entity_type=entity.type,
                                importance=entity.importance,
                                context=entity.context,
                                mention_count=entity.mention_count,
                            )
                            if entity_result:
                                projector.link_memory_to_bizentity(
                                    memory_id=memory_id,
                                    user_id=user_id,
                                    entity_name=entity.entity,
                                    importance=entity.importance,
                                )
                                stored_entities += 1

                    # Store concepts
                    for concept in extraction.concepts:
                        if concept.confidence >= min_confidence:
                            concept_result = projector.upsert_concept(
                                user_id=user_id,
                                name=concept.concept,
                                concept_type=concept.type,
                                confidence=concept.confidence,
                                category=category,
                                source_type=concept.source_type,
                                evidence_count=len(concept.evidence),
                            )
                            if concept_result:
                                projector.link_memory_to_concept(
                                    memory_id=memory_id,
                                    user_id=user_id,
                                    concept_name=concept.concept,
                                    confidence=concept.confidence,
                                )
                                # Link to entities
                                for entity_name in concept.entities:
                                    projector.link_concept_to_entity(
                                        user_id=user_id,
                                        concept_name=concept.concept,
                                        entity_name=entity_name,
                                    )
                                stored_concepts += 1

                    result["stored_entities"] = stored_entities
                    result["stored_concepts"] = stored_concepts
                else:
                    result["warning"] = "Graph projector not available"

            except Exception as graph_error:
                result["graph_error"] = str(graph_error)

        return result

    except Exception as e:
        return {"error": str(e)}


def batch_extract_from_memories(
    memories: List[Dict],
    user_id: str,
    api_key: str = None,
    store_in_graph: bool = True,
) -> Dict:
    """
    Extract concepts from multiple memories in batch.

    Args:
        memories: List of dicts with 'id', 'content', and optional 'category'
        user_id: User ID for scoping
        api_key: OpenAI API key
        store_in_graph: Whether to store in Neo4j graph

    Returns:
        Dict with batch results
    """
    results = {
        "processed": 0,
        "total_entities": 0,
        "total_concepts": 0,
        "errors": 0,
        "details": [],
    }

    for memory in memories:
        memory_id = memory.get("id")
        content = memory.get("content")
        category = memory.get("category")

        if not memory_id or not content:
            results["errors"] += 1
            continue

        extraction = extract_from_memory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            category=category,
            api_key=api_key,
            store_in_graph=store_in_graph,
        )

        if "error" in extraction:
            results["errors"] += 1
            results["details"].append({
                "memory_id": memory_id,
                "error": extraction["error"],
            })
        else:
            results["processed"] += 1
            results["total_entities"] += len(extraction.get("entities", []))
            results["total_concepts"] += len(extraction.get("concepts", []))

    return results


def calculate_extraction_confidence(
    entity: str,
    extraction_context: str,
    mention_count: int,
    has_numeric_value: bool = False,
    source_similarity: float = 0.0
) -> float:
    """
    Calculate confidence score for an extracted entity.

    Args:
        entity: The extracted entity text
        extraction_context: Surrounding context
        mention_count: How many times entity appears in source
        has_numeric_value: Whether entity contains numbers (e.g., metrics)
        source_similarity: Similarity to known-good patterns (0.0-1.0)

    Returns:
        Confidence score 0.0-1.0

    Factors:
    1. Mention frequency (more mentions = more confident)
    2. Numeric values (concrete metrics more reliable)
    3. Source similarity (matches known patterns)
    4. Length (very short entities often noise)
    """
    confidence = 0.5  # baseline

    # Mention frequency boost
    if mention_count >= 5:
        confidence += 0.2
    elif mention_count >= 3:
        confidence += 0.1
    elif mention_count == 1:
        confidence -= 0.15  # penalize single mentions

    # Concrete value boost
    if has_numeric_value:
        confidence += 0.15

    # Similarity boost
    confidence += source_similarity * 0.15

    # Length penalty (very short = often noise)
    if len(entity) <= 2:
        confidence -= 0.2

    return max(0.0, min(1.0, confidence))
