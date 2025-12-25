"""
Output Format Generator for Business Concept System.

This module generates business artifacts (BMC, Lean Canvas, Strategy Docs, etc.)
from knowledge stored in OpenMemory + Neo4j graph.

Design principles:
1. Query-driven: Each section queries specific concept patterns
2. Evidence-based: All claims link back to source memories
3. Confidence-aware: Show gaps and certainty levels
4. Minimal viable format: Useful, not perfect
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    BMC = "business_model_canvas"
    LEAN_CANVAS = "lean_canvas"
    STRATEGY_ONE_PAGER = "strategy_one_pager"
    INVESTMENT_MEMO = "investment_memo"
    WEEKLY_DIGEST = "weekly_digest"
    CONCEPT_MAP = "concept_map"


class ConfidenceLevel(Enum):
    """Confidence levels for synthesized content."""
    UNKNOWN = 0
    LOW = 1      # <50% - single source, hypothesis
    MEDIUM = 2   # 50-70% - multiple sources, some validation
    HIGH = 3     # 70-90% - strong evidence, validated
    VERIFIED = 4 # >90% - multiple validated sources, tested


@dataclass
class EvidenceChain:
    """Evidence supporting a synthesized claim."""
    memory_id: str
    content_snippet: str
    source_type: str  # video, document, conversation, etc.
    entity: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: float = 0.0

    def to_citation(self, index: int) -> str:
        """Format as markdown citation."""
        source = self.source_type.title()
        if self.entity:
            source = f"{source}: {self.entity}"
        if self.timestamp:
            source = f"{source} ({self.timestamp})"
        return f"[{index}] {source}"


@dataclass
class SectionContent:
    """Content for a single section with evidence."""
    title: str
    content: str
    confidence: ConfidenceLevel
    evidence: List[EvidenceChain] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)  # What's missing
    contradictions: List[str] = field(default_factory=list)  # Unresolved tensions

    def has_content(self) -> bool:
        """Check if section has actual content."""
        return bool(self.content and self.content.strip())

    def to_markdown(self, include_evidence: bool = True, include_metadata: bool = True) -> str:
        """Render as markdown with optional evidence citations."""
        lines = [f"## {self.title}\n"]

        if not self.has_content():
            lines.append("_[No information available]_\n")
            if self.gaps:
                lines.append("\n**Information Gaps:**")
                for gap in self.gaps:
                    lines.append(f"- {gap}")
            return "\n".join(lines)

        # Main content
        lines.append(self.content)
        lines.append("")

        if include_metadata:
            # Confidence indicator
            confidence_emoji = {
                ConfidenceLevel.UNKNOWN: "❓",
                ConfidenceLevel.LOW: "🟡",
                ConfidenceLevel.MEDIUM: "🟠",
                ConfidenceLevel.HIGH: "🟢",
                ConfidenceLevel.VERIFIED: "✅"
            }
            emoji = confidence_emoji.get(self.confidence, "❓")
            lines.append(f"**Confidence**: {emoji} {self.confidence.name}")
            lines.append("")

        # Evidence citations
        if include_evidence and self.evidence:
            lines.append("**Evidence:**")
            for i, ev in enumerate(self.evidence, 1):
                lines.append(f"{i}. {ev.to_citation(i)}")
                if ev.content_snippet:
                    # Truncate long snippets
                    snippet = ev.content_snippet[:150]
                    if len(ev.content_snippet) > 150:
                        snippet += "..."
                    lines.append(f"   _{snippet}_")
            lines.append("")

        # Gaps
        if self.gaps:
            lines.append("**Information Gaps:**")
            for gap in self.gaps:
                lines.append(f"- {gap}")
            lines.append("")

        # Contradictions
        if self.contradictions:
            lines.append("**Unresolved Tensions:**")
            for contradiction in self.contradictions:
                lines.append(f"- {contradiction}")
            lines.append("")

        return "\n".join(lines)


class OutputGenerator(ABC):
    """Base class for output format generators."""

    def __init__(self, search_fn, graph_query_fn=None):
        """
        Initialize with search and graph query functions.

        Args:
            search_fn: Async function to search memories
                       signature: (query, entity, vault, layer, limit) -> List[memory_dicts]
            graph_query_fn: Optional async function to run graph queries
                           signature: (cypher_query, params) -> List[result_dicts]
        """
        self.search = search_fn
        self.graph_query = graph_query_fn

    @abstractmethod
    async def generate(self, entity: str, **kwargs) -> Dict[str, Any]:
        """Generate output for the given entity/concept."""
        pass

    async def _query_section(
        self,
        query: str,
        entity: str,
        vault: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Query memories for a section and calculate confidence.

        Returns:
            Tuple of (memories, confidence_score)
        """
        try:
            results = await self.search(
                query=query,
                entity=entity,
                vault=vault,
                layer=layer,
                limit=limit
            )

            if not results:
                return [], 0.0

            # Calculate confidence based on:
            # - Number of results (more is better)
            # - Source diversity (multiple sources better than one)
            # - Recency (newer is more relevant)

            num_results = len(results)
            sources = set(r.get("tags", {}).get("source_type", "unknown") for r in results)
            source_diversity = len(sources)

            # Simple confidence formula
            confidence = min(
                (num_results / 5.0) * 0.4 +  # Up to 0.4 for quantity
                (source_diversity / 3.0) * 0.4 +  # Up to 0.4 for diversity
                0.2,  # Base score
                1.0
            )

            return results, confidence

        except Exception as e:
            logger.error(f"Error querying section: {e}")
            return [], 0.0

    def _extract_evidence(self, memories: List[Dict]) -> List[EvidenceChain]:
        """Convert memory results to evidence chains."""
        evidence = []
        for mem in memories[:5]:  # Limit to top 5
            evidence.append(EvidenceChain(
                memory_id=mem.get("id", ""),
                content_snippet=mem.get("memory", "")[:200],
                source_type=mem.get("tags", {}).get("source_type", "unknown"),
                entity=mem.get("entity"),
                timestamp=mem.get("created_at"),
                confidence=mem.get("score", 0.0)
            ))
        return evidence

    def _confidence_level(self, score: float) -> ConfidenceLevel:
        """Map confidence score to level."""
        if score >= 0.9:
            return ConfidenceLevel.VERIFIED
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score > 0:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN


class BusinessModelCanvasGenerator(OutputGenerator):
    """Generate Business Model Canvas from stored concepts."""

    BMC_SECTIONS = {
        "value_propositions": {
            "query": "value proposition customer benefit problem solving unique",
            "vault": "WLT",
            "layer": "goals"
        },
        "customer_segments": {
            "query": "customer segment target audience persona user type",
            "vault": "WLT",
            "layer": "relational"
        },
        "channels": {
            "query": "channel distribution sales marketing reach customer acquisition",
            "vault": "WLT",
            "layer": "resources"
        },
        "customer_relationships": {
            "query": "customer relationship retention engagement support interaction",
            "vault": "WLT",
            "layer": "relational"
        },
        "revenue_streams": {
            "query": "revenue pricing monetization income business model payment",
            "vault": "WLT",
            "layer": "resources"
        },
        "key_resources": {
            "query": "key resource asset capability infrastructure technology team",
            "vault": "WLT",
            "layer": "resources"
        },
        "key_activities": {
            "query": "key activity operation process core function execution",
            "vault": "WLT",
            "layer": "goals"
        },
        "key_partnerships": {
            "query": "partner partnership strategic alliance vendor supplier ecosystem",
            "vault": "WLT",
            "layer": "relational"
        },
        "cost_structure": {
            "query": "cost expense structure budget burn rate economics",
            "vault": "WLT",
            "layer": "resources"
        }
    }

    async def generate(self, entity: str, **kwargs) -> Dict[str, Any]:
        """Generate BMC for an entity."""
        include_evidence = kwargs.get("include_evidence", True)
        include_metadata = kwargs.get("include_metadata", True)

        sections = {}
        overall_confidence = []

        for section_name, config in self.BMC_SECTIONS.items():
            memories, confidence = await self._query_section(
                query=config["query"],
                entity=entity,
                vault=config.get("vault"),
                layer=config.get("layer"),
                limit=10
            )

            overall_confidence.append(confidence)

            # Synthesize content from memories
            if memories:
                # Simple synthesis: combine top insights
                content_parts = []
                for mem in memories[:3]:
                    content = mem.get("memory", "")
                    if len(content) < 200:
                        content_parts.append(f"- {content}")
                    else:
                        content_parts.append(f"- {content[:150]}...")

                content = "\n".join(content_parts) if content_parts else ""
            else:
                content = ""

            # Identify gaps
            gaps = []
            if confidence < 0.3:
                gaps.append(f"Limited information available (only {len(memories)} sources)")
            if len(set(m.get("tags", {}).get("source_type") for m in memories)) < 2:
                gaps.append("Single source type - needs cross-validation")

            section = SectionContent(
                title=section_name.replace("_", " ").title(),
                content=content,
                confidence=self._confidence_level(confidence),
                evidence=self._extract_evidence(memories),
                gaps=gaps
            )

            sections[section_name] = section

        # Generate markdown
        markdown = self._render_bmc_markdown(
            entity=entity,
            sections=sections,
            overall_confidence=sum(overall_confidence) / len(overall_confidence),
            include_evidence=include_evidence,
            include_metadata=include_metadata
        )

        return {
            "entity": entity,
            "format": OutputFormat.BMC.value,
            "generated_at": datetime.now().isoformat(),
            "overall_confidence": sum(overall_confidence) / len(overall_confidence),
            "sections": {k: v.__dict__ for k, v in sections.items()},
            "markdown": markdown
        }

    def _render_bmc_markdown(
        self,
        entity: str,
        sections: Dict[str, SectionContent],
        overall_confidence: float,
        include_evidence: bool,
        include_metadata: bool
    ) -> str:
        """Render BMC as markdown."""
        lines = [
            f"# Business Model Canvas: {entity}",
            f"",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            f"",
            f"**Overall Confidence**: {overall_confidence:.0%}",
            f"",
            "---",
            "",
        ]

        # Left side (Key Partners, Activities, Resources)
        lines.append("## Left Side: Operations")
        lines.append("")
        for section in ["key_partnerships", "key_activities", "key_resources"]:
            if section in sections:
                lines.append(sections[section].to_markdown(include_evidence, include_metadata))

        lines.append("---")
        lines.append("")

        # Center (Value Propositions)
        lines.append("## Center: Value")
        lines.append("")
        if "value_propositions" in sections:
            lines.append(sections["value_propositions"].to_markdown(include_evidence, include_metadata))

        lines.append("---")
        lines.append("")

        # Right side (Customer Relationships, Channels, Segments)
        lines.append("## Right Side: Customers")
        lines.append("")
        for section in ["customer_relationships", "channels", "customer_segments"]:
            if section in sections:
                lines.append(sections[section].to_markdown(include_evidence, include_metadata))

        lines.append("---")
        lines.append("")

        # Bottom (Costs and Revenue)
        lines.append("## Bottom: Financials")
        lines.append("")
        for section in ["cost_structure", "revenue_streams"]:
            if section in sections:
                lines.append(sections[section].to_markdown(include_evidence, include_metadata))

        return "\n".join(lines)


class LeanCanvasGenerator(OutputGenerator):
    """Generate Lean Canvas from stored concepts."""

    LEAN_SECTIONS = {
        "problem": {
            "query": "problem pain point challenge issue customer struggle",
            "vault": "WLT",
            "layer": "cognitive"
        },
        "solution": {
            "query": "solution approach product feature capability answer",
            "vault": "WLT",
            "layer": "goals"
        },
        "unique_value_proposition": {
            "query": "unique value proposition differentiation competitive advantage why us",
            "vault": "WLT",
            "layer": "identity"
        },
        "unfair_advantage": {
            "query": "unfair advantage moat barrier to entry secret sauce unique asset",
            "vault": "WLT",
            "layer": "resources"
        },
        "customer_segments": {
            "query": "customer segment early adopter target market persona",
            "vault": "WLT",
            "layer": "relational"
        },
        "existing_alternatives": {
            "query": "alternative competitor current solution existing approach",
            "vault": "WLT",
            "layer": "context"
        },
        "key_metrics": {
            "query": "metric KPI measure success indicator performance",
            "vault": "WLT",
            "layer": "goals"
        },
        "channels": {
            "query": "channel distribution go-to-market reach customer acquisition",
            "vault": "WLT",
            "layer": "resources"
        },
        "cost_structure": {
            "query": "cost expense fixed variable CAC burn rate",
            "vault": "WLT",
            "layer": "resources"
        },
        "revenue_streams": {
            "query": "revenue pricing LTV monetization business model",
            "vault": "WLT",
            "layer": "resources"
        }
    }

    async def generate(self, entity: str, **kwargs) -> Dict[str, Any]:
        """Generate Lean Canvas."""
        include_evidence = kwargs.get("include_evidence", True)

        sections = {}

        for section_name, config in self.LEAN_SECTIONS.items():
            memories, confidence = await self._query_section(
                query=config["query"],
                entity=entity,
                vault=config.get("vault"),
                layer=config.get("layer")
            )

            # For Lean Canvas, we want concise bullet points
            content_parts = []
            for mem in memories[:3]:
                text = mem.get("memory", "")
                # Extract first sentence or first 100 chars
                first_sentence = text.split(".")[0] if "." in text else text[:100]
                content_parts.append(f"- {first_sentence.strip()}")

            section = SectionContent(
                title=section_name.replace("_", " ").title(),
                content="\n".join(content_parts),
                confidence=self._confidence_level(confidence),
                evidence=self._extract_evidence(memories)
            )

            sections[section_name] = section

        markdown = self._render_lean_canvas_markdown(entity, sections, include_evidence)

        return {
            "entity": entity,
            "format": OutputFormat.LEAN_CANVAS.value,
            "generated_at": datetime.now().isoformat(),
            "sections": {k: v.__dict__ for k, v in sections.items()},
            "markdown": markdown
        }

    def _render_lean_canvas_markdown(
        self,
        entity: str,
        sections: Dict[str, SectionContent],
        include_evidence: bool
    ) -> str:
        """Render Lean Canvas as markdown."""
        lines = [
            f"# Lean Canvas: {entity}",
            "",
            f"_Generated: {datetime.now().strftime('%Y-%m-%d')}_",
            "",
        ]

        # Top row
        lines.append("## Problem | Solution | Unique Value Proposition | Unfair Advantage | Customer Segments")
        lines.append("")
        for section in ["problem", "solution", "unique_value_proposition", "unfair_advantage", "customer_segments"]:
            if section in sections:
                lines.append(f"### {sections[section].title}")
                lines.append(sections[section].content)
                if include_evidence and sections[section].evidence:
                    lines.append(f"_({len(sections[section].evidence)} sources)_")
                lines.append("")

        lines.append("---")
        lines.append("")

        # Bottom row
        lines.append("## Existing Alternatives | Key Metrics | Channels | Cost Structure | Revenue Streams")
        lines.append("")
        for section in ["existing_alternatives", "key_metrics", "channels", "cost_structure", "revenue_streams"]:
            if section in sections:
                lines.append(f"### {sections[section].title}")
                lines.append(sections[section].content)
                if include_evidence and sections[section].evidence:
                    lines.append(f"_({len(sections[section].evidence)} sources)_")
                lines.append("")

        return "\n".join(lines)


class StrategyOnePagerGenerator(OutputGenerator):
    """Generate one-page strategy summary."""

    async def generate(self, entity: str, **kwargs) -> Dict[str, Any]:
        """Generate strategy one-pager."""

        # Core insight (highest confidence memories)
        core_memories, core_conf = await self._query_section(
            "core insight key learning main takeaway",
            entity, vault="WLT", layer="cognitive", limit=5
        )

        # Problem/opportunity
        problem_memories, prob_conf = await self._query_section(
            "problem opportunity gap need market",
            entity, vault="WLT", layer="cognitive", limit=5
        )

        # Solution approach
        solution_memories, sol_conf = await self._query_section(
            "solution approach strategy how execute",
            entity, vault="WLT", layer="goals", limit=5
        )

        # Why now (market timing)
        timing_memories, timing_conf = await self._query_section(
            "timing market trend catalyst why now momentum",
            entity, vault="WLT", layer="context", limit=5
        )

        # Risks and mitigations
        risk_memories, risk_conf = await self._query_section(
            "risk challenge concern mitigation contingency",
            entity, vault="Q", limit=5
        )

        # Next steps
        action_memories, action_conf = await self._query_section(
            "next step action priority immediate focus",
            entity, vault="WLT", layer="goals", limit=5
        )

        # Build sections
        sections = {
            "core_insight": SectionContent(
                "Core Insight",
                self._synthesize_bullets(core_memories, 1),
                self._confidence_level(core_conf),
                self._extract_evidence(core_memories)
            ),
            "problem": SectionContent(
                "Problem/Opportunity",
                self._synthesize_bullets(problem_memories, 2),
                self._confidence_level(prob_conf),
                self._extract_evidence(problem_memories)
            ),
            "solution": SectionContent(
                "Our Approach",
                self._synthesize_bullets(solution_memories, 3),
                self._confidence_level(sol_conf),
                self._extract_evidence(solution_memories)
            ),
            "why_now": SectionContent(
                "Why Now",
                self._synthesize_bullets(timing_memories, 2),
                self._confidence_level(timing_conf),
                self._extract_evidence(timing_memories)
            ),
            "risks": SectionContent(
                "Key Risks & Mitigations",
                self._synthesize_bullets(risk_memories, 3),
                self._confidence_level(risk_conf),
                self._extract_evidence(risk_memories)
            ),
            "next_steps": SectionContent(
                "Next Steps",
                self._synthesize_bullets(action_memories, 3),
                self._confidence_level(action_conf),
                self._extract_evidence(action_memories)
            )
        }

        markdown = self._render_strategy_markdown(entity, sections)

        return {
            "entity": entity,
            "format": OutputFormat.STRATEGY_ONE_PAGER.value,
            "generated_at": datetime.now().isoformat(),
            "sections": {k: v.__dict__ for k, v in sections.items()},
            "markdown": markdown
        }

    def _synthesize_bullets(self, memories: List[Dict], count: int) -> str:
        """Extract top N bullet points from memories."""
        bullets = []
        for mem in memories[:count]:
            text = mem.get("memory", "")
            # Get first sentence or first 100 chars
            bullet = text.split(".")[0] if "." in text else text[:100]
            bullets.append(f"- {bullet.strip()}")
        return "\n".join(bullets) if bullets else "_[No information available]_"

    def _render_strategy_markdown(self, entity: str, sections: Dict[str, SectionContent]) -> str:
        """Render strategy doc as markdown."""
        lines = [
            f"# Strategy: {entity}",
            "",
            f"_One-Page Summary | {datetime.now().strftime('%Y-%m-%d')}_",
            "",
            "---",
            ""
        ]

        for section in sections.values():
            lines.append(section.to_markdown(include_evidence=True, include_metadata=False))

        return "\n".join(lines)


class WeeklyDigestGenerator(OutputGenerator):
    """Generate weekly concept digest."""

    async def generate(self, entity: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate weekly digest.

        Args:
            entity: Optional focus entity, if None generates cross-entity digest
            **kwargs: Can include 'days_back' (default 7)
        """
        days_back = kwargs.get("days_back", 7)

        # This would use date-filtered search in production
        # For now, using recency_weight to favor recent memories

        # New concepts (high connectivity, recent)
        new_concepts, new_conf = await self._query_section(
            "concept pattern insight discovery learning",
            entity or "",
            limit=10
        )

        # Evolved concepts (has 'was' field or version tag)
        # Would need special query to find updated memories

        # Contradictions (from Q vault or contradiction tags)
        contradictions, contra_conf = await self._query_section(
            "contradiction tension conflict different",
            entity or "",
            vault="Q",
            limit=5
        )

        # High-impact insights (high confidence + multiple sources)
        insights, insight_conf = await self._query_section(
            "important key critical significant breakthrough",
            entity or "",
            limit=5
        )

        sections = {
            "new_concepts": SectionContent(
                "New Concepts This Week",
                self._synthesize_bullets(new_concepts, 5),
                self._confidence_level(new_conf),
                self._extract_evidence(new_concepts)
            ),
            "contradictions": SectionContent(
                "Contradictions to Explore",
                self._synthesize_bullets(contradictions, 3),
                self._confidence_level(contra_conf),
                self._extract_evidence(contradictions)
            ),
            "insights": SectionContent(
                "Key Insights",
                self._synthesize_bullets(insights, 3),
                self._confidence_level(insight_conf),
                self._extract_evidence(insights)
            )
        }

        markdown = self._render_digest_markdown(entity, sections, days_back)

        return {
            "entity": entity or "All Concepts",
            "format": OutputFormat.WEEKLY_DIGEST.value,
            "generated_at": datetime.now().isoformat(),
            "period": f"{days_back} days",
            "sections": {k: v.__dict__ for k, v in sections.items()},
            "markdown": markdown
        }

    def _synthesize_bullets(self, memories: List[Dict], count: int) -> str:
        """Extract top N bullet points from memories."""
        bullets = []
        for mem in memories[:count]:
            text = mem.get("memory", "")
            bullet = text.split(".")[0] if "." in text else text[:100]
            bullets.append(f"- {bullet.strip()}")
        return "\n".join(bullets) if bullets else "_[No new items this week]_"

    def _render_digest_markdown(
        self,
        entity: Optional[str],
        sections: Dict[str, SectionContent],
        days_back: int
    ) -> str:
        """Render weekly digest."""
        entity_str = f" - {entity}" if entity else ""
        lines = [
            f"# Weekly Digest{entity_str}",
            "",
            f"_Period: Last {days_back} days | Generated: {datetime.now().strftime('%Y-%m-%d')}_",
            "",
            "---",
            ""
        ]

        for section in sections.values():
            lines.append(section.to_markdown(include_evidence=False, include_metadata=False))

        return "\n".join(lines)


# LLM Synthesis Prompts
# These would be used if you want LLM-generated synthesis instead of simple concatenation

BMC_SYNTHESIS_PROMPT = """You are synthesizing a Business Model Canvas section from stored memories.

Section: {section_name}
Entity: {entity}

Memories:
{memories}

Task:
1. Synthesize the key points from these memories into 2-3 concise bullet points
2. Focus on actionable, specific information
3. Note any contradictions or gaps
4. Keep it under 150 words

Output format:
- Bullet point 1
- Bullet point 2
- Bullet point 3 (if applicable)

Gaps: [list any information gaps you notice]
"""

LEAN_CANVAS_SYNTHESIS_PROMPT = """Synthesize Lean Canvas section from memories.

Section: {section_name}
Memories:
{memories}

Requirements:
- 1-3 bullet points maximum
- Each bullet under 20 words
- Actionable and specific
- No fluff

Output as markdown bullets.
"""

STRATEGY_SYNTHESIS_PROMPT = """Synthesize strategy section from stored concepts.

Section: {section_name}
Entity: {entity}
Memories:
{memories}

Create a brief narrative (2-3 sentences) that:
1. Captures the core insight
2. Shows the logic/reasoning
3. Indicates confidence level

Keep under 100 words.
"""

# Template for evidence chain visualization
EVIDENCE_CHAIN_TEMPLATE = """
## {claim}

**Confidence**: {confidence} ({confidence_level})
**Evidence Chain**:
{evidence_items}

**Evolution**:
{evolution_history}
"""
