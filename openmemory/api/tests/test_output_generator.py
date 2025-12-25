"""
Tests for Output Generator.

These tests demonstrate usage patterns and validate the output format generators.
"""

import pytest
from datetime import datetime
from typing import Dict, List

from app.utils.output_generator import (
    BusinessModelCanvasGenerator,
    LeanCanvasGenerator,
    StrategyOnePagerGenerator,
    WeeklyDigestGenerator,
    OutputFormat,
    ConfidenceLevel,
    SectionContent,
    EvidenceChain
)


# Mock search function for testing
async def mock_search(
    query: str,
    entity: str = None,
    vault: str = None,
    layer: str = None,
    limit: int = 10
) -> List[Dict]:
    """Mock search function that returns sample memories based on query."""

    # Simulate different results based on query keywords
    if "value proposition" in query.lower():
        return [
            {
                "id": "mem-vp-1",
                "memory": "AI-powered decision support for technical founders through behavioral pattern analysis. Helps identify blind spots in real-time.",
                "score": 0.85,
                "entity": entity,
                "tags": {"source_type": "video", "confidence": 0.8},
                "created_at": "2025-01-20T10:30:00Z"
            },
            {
                "id": "mem-vp-2",
                "memory": "Core value is transforming implicit decision patterns into explicit, reviewable frameworks.",
                "score": 0.78,
                "entity": entity,
                "tags": {"source_type": "document", "confidence": 0.7},
                "created_at": "2025-01-19T14:20:00Z"
            },
            {
                "id": "mem-vp-3",
                "memory": "Unlike traditional consulting, system provides 24/7 insight without human bottleneck.",
                "score": 0.72,
                "entity": entity,
                "tags": {"source_type": "conversation", "confidence": 0.6},
                "created_at": "2025-01-18T09:15:00Z"
            }
        ]

    elif "customer segment" in query.lower():
        return [
            {
                "id": "mem-cs-1",
                "memory": "Target customer: Technical founders at seed-stage startups needing structured decision-making support.",
                "score": 0.82,
                "entity": entity,
                "tags": {"source_type": "video", "confidence": 0.75},
                "created_at": "2025-01-20T11:00:00Z"
            },
            {
                "id": "mem-cs-2",
                "memory": "Secondary segment: Solopreneurs who need thinking partner but can't afford consultants.",
                "score": 0.75,
                "entity": entity,
                "tags": {"source_type": "document", "confidence": 0.7},
                "created_at": "2025-01-19T15:30:00Z"
            }
        ]

    elif "revenue" in query.lower():
        return [
            {
                "id": "mem-rev-1",
                "memory": "Revenue model: Freemium with enterprise upsell. Free tier for individuals, $10/user/month for teams.",
                "score": 0.80,
                "entity": entity,
                "tags": {"source_type": "video", "confidence": 0.75},
                "created_at": "2025-01-21T10:00:00Z"
            },
            {
                "id": "mem-rev-2",
                "memory": "Enterprise tier: Custom pricing starting at $1000/month for organizations >50 users.",
                "score": 0.72,
                "entity": entity,
                "tags": {"source_type": "document", "confidence": 0.65},
                "created_at": "2025-01-20T16:20:00Z"
            }
        ]

    elif "problem" in query.lower():
        return [
            {
                "id": "mem-prob-1",
                "memory": "Technical founders struggle with decision-making under uncertainty. No structured framework for reviewing choices.",
                "score": 0.85,
                "entity": entity,
                "tags": {"source_type": "video", "confidence": 0.8},
                "created_at": "2025-01-22T09:00:00Z"
            },
            {
                "id": "mem-prob-2",
                "memory": "Current alternatives (consultants, coaches) are expensive and don't scale with founder's pace.",
                "score": 0.78,
                "entity": entity,
                "tags": {"source_type": "conversation", "confidence": 0.7},
                "created_at": "2025-01-21T14:30:00Z"
            }
        ]

    elif "solution" in query.lower():
        return [
            {
                "id": "mem-sol-1",
                "memory": "AI-powered decision tracker that synthesizes patterns from meetings, documents, and conversations.",
                "score": 0.82,
                "entity": entity,
                "tags": {"source_type": "video", "confidence": 0.75},
                "created_at": "2025-01-22T10:30:00Z"
            }
        ]

    elif "risk" in query.lower() or "challenge" in query.lower():
        return [
            {
                "id": "mem-risk-1",
                "memory": "Risk: AI accuracy concerns. Mitigation: Human-in-loop review, confidence scoring, evidence trails.",
                "score": 0.75,
                "entity": entity,
                "tags": {"source_type": "document", "confidence": 0.7},
                "created_at": "2025-01-20T11:00:00Z"
            }
        ]

    # Default: return empty for unmapped queries (simulates gaps)
    return []


class TestEvidenceChain:
    """Test EvidenceChain functionality."""

    def test_evidence_chain_creation(self):
        """Test creating evidence chain."""
        ev = EvidenceChain(
            memory_id="mem-123",
            content_snippet="Revenue model is freemium with enterprise upsell",
            source_type="video",
            entity="BMG",
            timestamp="2025-01-20T10:00:00Z",
            confidence=0.8
        )

        assert ev.memory_id == "mem-123"
        assert ev.source_type == "video"
        assert ev.confidence == 0.8

    def test_citation_format(self):
        """Test citation formatting."""
        ev = EvidenceChain(
            memory_id="mem-123",
            content_snippet="Test content",
            source_type="video",
            entity="BMG",
            timestamp="2025-01-20",
            confidence=0.8
        )

        citation = ev.to_citation(1)
        assert "[1]" in citation
        assert "Video" in citation
        assert "BMG" in citation


class TestSectionContent:
    """Test SectionContent functionality."""

    def test_section_with_content(self):
        """Test section with content."""
        section = SectionContent(
            title="Value Propositions",
            content="AI-powered decision support for founders",
            confidence=ConfidenceLevel.HIGH,
            evidence=[
                EvidenceChain(
                    memory_id="mem-1",
                    content_snippet="AI decision support",
                    source_type="video",
                    confidence=0.8
                )
            ],
            gaps=[]
        )

        assert section.has_content()
        assert "Value Propositions" in section.title
        assert section.confidence == ConfidenceLevel.HIGH

    def test_section_without_content(self):
        """Test section with no content (gap)."""
        section = SectionContent(
            title="Key Partnerships",
            content="",
            confidence=ConfidenceLevel.UNKNOWN,
            gaps=["No partnership information available"]
        )

        assert not section.has_content()
        assert len(section.gaps) == 1

    def test_markdown_rendering(self):
        """Test markdown rendering."""
        section = SectionContent(
            title="Test Section",
            content="Test content here",
            confidence=ConfidenceLevel.MEDIUM,
            evidence=[],
            gaps=["Some gap"]
        )

        markdown = section.to_markdown(include_evidence=True, include_metadata=True)

        assert "## Test Section" in markdown
        assert "Test content" in markdown
        assert "MEDIUM" in markdown
        assert "Some gap" in markdown


class TestBusinessModelCanvasGenerator:
    """Test BMC generation."""

    @pytest.mark.asyncio
    async def test_bmc_generation(self):
        """Test generating BMC."""
        gen = BusinessModelCanvasGenerator(search_fn=mock_search)

        result = await gen.generate(
            entity="BMG",
            include_evidence=True,
            include_metadata=True
        )

        # Check structure
        assert result["entity"] == "BMG"
        assert result["format"] == OutputFormat.BMC.value
        assert "markdown" in result
        assert "sections" in result
        assert "overall_confidence" in result

        # Check sections exist
        sections = result["sections"]
        assert "value_propositions" in sections
        assert "customer_segments" in sections
        assert "revenue_streams" in sections

        # Check markdown output
        markdown = result["markdown"]
        assert "Business Model Canvas: BMG" in markdown
        assert "Value Propositions" in markdown

    @pytest.mark.asyncio
    async def test_bmc_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        gen = BusinessModelCanvasGenerator(search_fn=mock_search)
        result = await gen.generate(entity="BMG")

        # Value propositions should have high confidence (3 sources)
        vp_section = result["sections"]["value_propositions"]
        assert vp_section["confidence"] in [ConfidenceLevel.MEDIUM.name, ConfidenceLevel.HIGH.name]

        # Some sections should have low/unknown confidence (no data)
        # (depending on mock_search implementation)

    @pytest.mark.asyncio
    async def test_bmc_gap_detection(self):
        """Test gap detection for missing sections."""
        gen = BusinessModelCanvasGenerator(search_fn=mock_search)
        result = await gen.generate(entity="BMG")

        # Check that sections with no data have gaps identified
        for section_name, section_data in result["sections"].items():
            if not section_data["content"]:
                assert len(section_data["gaps"]) > 0


class TestLeanCanvasGenerator:
    """Test Lean Canvas generation."""

    @pytest.mark.asyncio
    async def test_lean_canvas_generation(self):
        """Test generating Lean Canvas."""
        gen = LeanCanvasGenerator(search_fn=mock_search)

        result = await gen.generate(entity="BMG", include_evidence=True)

        assert result["entity"] == "BMG"
        assert result["format"] == OutputFormat.LEAN_CANVAS.value
        assert "sections" in result

        # Check lean-specific sections
        sections = result["sections"]
        assert "problem" in sections
        assert "solution" in sections
        assert "unique_value_proposition" in sections
        assert "unfair_advantage" in sections

    @pytest.mark.asyncio
    async def test_lean_canvas_conciseness(self):
        """Test that Lean Canvas is more concise than BMC."""
        gen = LeanCanvasGenerator(search_fn=mock_search)
        result = await gen.generate(entity="BMG")

        # Check that content is bullet-pointed and concise
        for section_data in result["sections"].values():
            if section_data["content"]:
                # Should be bullet points
                assert section_data["content"].startswith("-") or "[No information" in section_data["content"]


class TestStrategyOnePagerGenerator:
    """Test Strategy One-Pager generation."""

    @pytest.mark.asyncio
    async def test_strategy_generation(self):
        """Test generating strategy doc."""
        gen = StrategyOnePagerGenerator(search_fn=mock_search)

        result = await gen.generate(entity="BMG")

        assert result["entity"] == "BMG"
        assert result["format"] == OutputFormat.STRATEGY_ONE_PAGER.value

        # Check sections
        sections = result["sections"]
        assert "core_insight" in sections
        assert "problem" in sections
        assert "solution" in sections
        assert "why_now" in sections
        assert "risks" in sections
        assert "next_steps" in sections

    @pytest.mark.asyncio
    async def test_strategy_markdown_structure(self):
        """Test strategy doc markdown structure."""
        gen = StrategyOnePagerGenerator(search_fn=mock_search)
        result = await gen.generate(entity="BMG")

        markdown = result["markdown"]

        # Should have key headings
        assert "# Strategy: BMG" in markdown
        assert "Core Insight" in markdown
        assert "Problem/Opportunity" in markdown
        assert "Next Steps" in markdown


class TestWeeklyDigestGenerator:
    """Test Weekly Digest generation."""

    @pytest.mark.asyncio
    async def test_weekly_digest_generation(self):
        """Test generating weekly digest."""
        gen = WeeklyDigestGenerator(search_fn=mock_search)

        result = await gen.generate(entity="BMG", days_back=7)

        assert result["format"] == OutputFormat.WEEKLY_DIGEST.value
        assert result["period"] == "7 days"

        sections = result["sections"]
        assert "new_concepts" in sections
        assert "contradictions" in sections
        assert "insights" in sections

    @pytest.mark.asyncio
    async def test_weekly_digest_no_entity(self):
        """Test digest without specific entity (all concepts)."""
        gen = WeeklyDigestGenerator(search_fn=mock_search)

        result = await gen.generate(days_back=7)

        assert result["entity"] == "All Concepts"


class TestIntegration:
    """Integration tests for common workflows."""

    @pytest.mark.asyncio
    async def test_generate_all_formats(self):
        """Test generating all formats for same entity."""
        entity = "BMG"

        bmc_gen = BusinessModelCanvasGenerator(search_fn=mock_search)
        lean_gen = LeanCanvasGenerator(search_fn=mock_search)
        strategy_gen = StrategyOnePagerGenerator(search_fn=mock_search)

        bmc = await bmc_gen.generate(entity)
        lean = await lean_gen.generate(entity)
        strategy = await strategy_gen.generate(entity)

        # All should succeed
        assert bmc["entity"] == entity
        assert lean["entity"] == entity
        assert strategy["entity"] == entity

        # Should have different formats
        assert bmc["format"] != lean["format"]
        assert lean["format"] != strategy["format"]

    @pytest.mark.asyncio
    async def test_evidence_chain_across_formats(self):
        """Test that evidence is consistently tracked across formats."""
        entity = "BMG"

        bmc_gen = BusinessModelCanvasGenerator(search_fn=mock_search)
        lean_gen = LeanCanvasGenerator(search_fn=mock_search)

        bmc = await bmc_gen.generate(entity)
        lean = await lean_gen.generate(entity)

        # Both should have evidence for value proposition
        bmc_vp_evidence = bmc["sections"]["value_propositions"]["evidence"]
        lean_vp_evidence = lean["sections"]["unique_value_proposition"]["evidence"]

        # Should have some evidence (exact match depends on query results)
        assert len(bmc_vp_evidence) > 0 or len(lean_vp_evidence) > 0


# Example usage test
@pytest.mark.asyncio
async def test_example_usage():
    """Demonstrate example usage pattern."""

    # Initialize generator
    gen = BusinessModelCanvasGenerator(search_fn=mock_search)

    # Generate BMC
    result = await gen.generate(
        entity="BMG",
        include_evidence=True,
        include_metadata=True
    )

    # Extract markdown for display
    markdown = result["markdown"]
    print("\n" + "="*60)
    print("GENERATED BUSINESS MODEL CANVAS")
    print("="*60)
    print(markdown)

    # Extract gaps for action items
    gaps = {}
    for section_name, section_data in result["sections"].items():
        if section_data["gaps"]:
            gaps[section_name] = section_data["gaps"]

    if gaps:
        print("\n" + "="*60)
        print("IDENTIFIED GAPS")
        print("="*60)
        for section, gap_list in gaps.items():
            print(f"\n{section}:")
            for gap in gap_list:
                print(f"  - {gap}")

    # Check overall confidence
    print(f"\nOverall Confidence: {result['overall_confidence']:.0%}")

    assert "markdown" in result
    assert result["overall_confidence"] >= 0
