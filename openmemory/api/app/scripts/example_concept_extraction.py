#!/usr/bin/env python3
"""
Example: Business Concept Extraction from Transcript

This script demonstrates how to use the ConceptExtractor to extract
business entities and concepts from a video transcript or document.

Usage:
    python example_concept_extraction.py

Requirements:
    - openai package: pip install openai
    - OpenAI API key in environment variable OPENAI_API_KEY
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.concept_extractor import ConceptExtractor

# Example German + English mixed business transcript
EXAMPLE_TRANSCRIPT = """
Wir haben letztes Jahr ein AI-powered document processing tool für B2B SaaS companies gebaut.
Currently we're at €500k ARR with about 50 enterprise customers, mostly in the DACH region.

What we noticed is that customers who integrate our API within the first week have 3x higher
retention than those who take longer. Das zeigt uns, dass time-to-value der wichtigste retention
driver ist. We've analyzed the data and it's very clear: fast onboarding = low churn.

Our CEO Sarah believes we should expand to the US market next quarter, aber ich denke wir sollten
erst unsere product-market fit in Germany solidify. There's a contradiction here: strong demand
signals from US (we get 10-15 inbound leads per week) but limited resources to serve two markets well.

Key metrics from last quarter:
- €500k ARR (up 30% MoM)
- 50 enterprise customers
- 8% monthly churn rate
- Average deal size: €10k/year
- Time to value: currently 3-4 days, target is <24 hours

We're using React for the frontend, FastAPI for the backend, and PostgreSQL for the database.
The whole stack runs on AWS. We're considering moving to a microservices architecture but
honestly it feels like overkill for our current stage.

Main strategic question: Should we focus on reducing churn (by improving onboarding) or
expanding to new markets (US)? Our burn rate is €50k/month and we have 18 months runway.
"""


def main():
    """Run concept extraction example"""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("=" * 80)
    print("BUSINESS CONCEPT EXTRACTION EXAMPLE")
    print("=" * 80)
    print()

    # Initialize extractor
    print("Initializing ConceptExtractor...")
    extractor = ConceptExtractor(
        api_key=api_key,
        model="gpt-4o-mini"  # Cost-effective choice
    )
    print(f"✓ Using model: {extractor.model}")
    print()

    # Estimate cost
    print("Estimating extraction cost...")
    cost_estimate = extractor.estimate_cost(EXAMPLE_TRANSCRIPT)
    print(f"✓ Input tokens: {cost_estimate['input_tokens']:,}")
    print(f"✓ Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
    print(f"✓ Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")
    print()

    # Extract
    print("Extracting entities and concepts...")
    print("(This will take 5-10 seconds...)")
    print()

    result = extractor.extract_full(EXAMPLE_TRANSCRIPT)

    # Display results
    print("=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)
    print()

    print(f"Language detected: {result.language.upper()}")
    print()

    print("SUMMARY")
    print("-" * 80)
    print(result.summary)
    print()

    print("ENTITIES EXTRACTED")
    print("-" * 80)
    print(f"Found {len(result.entities)} entities\n")

    # Group entities by type
    entities_by_type = {}
    for entity in result.entities:
        if entity.type not in entities_by_type:
            entities_by_type[entity.type] = []
        entities_by_type[entity.type].append(entity)

    for entity_type, entities in sorted(entities_by_type.items()):
        print(f"\n{entity_type.upper().replace('_', ' ')}:")
        for entity in sorted(entities, key=lambda e: e.importance, reverse=True):
            importance_bar = "█" * int(entity.importance * 10)
            print(f"  [{importance_bar:<10}] {entity.importance:.2f} - {entity.entity}")
            if entity.mention_count > 1:
                print(f"              (mentioned {entity.mention_count}x)")

    print()
    print("CONCEPTS EXTRACTED")
    print("-" * 80)
    print(f"Found {len(result.concepts)} concepts\n")

    for i, concept in enumerate(result.concepts, 1):
        confidence_bar = "█" * int(concept.confidence * 10)
        print(f"{i}. [{concept.type.upper()}] {concept.concept}")
        print(f"   Confidence: [{confidence_bar:<10}] {concept.confidence:.2f}")
        print(f"   Source: {concept.source_type}")
        print(f"   Evidence: \"{concept.evidence[0][:100]}...\"")
        print(f"   Related entities: {', '.join(concept.entities[:3])}")
        print()

    # Recommendations based on extraction
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    high_confidence_concepts = [c for c in result.concepts if c.confidence >= 0.7]
    low_confidence_concepts = [c for c in result.concepts if 0.5 <= c.confidence < 0.7]

    print(f"✓ {len(high_confidence_concepts)} high-confidence concepts (auto-accept: category=glossary, scope=project)")
    print(f"⚠ {len(low_confidence_concepts)} medium-confidence concepts (review needed, tag for review)")

    contradictions = [c for c in result.concepts if c.type == "contradiction"]
    if contradictions:
        print(f"\n⚡ {len(contradictions)} contradiction(s) detected - requires strategic resolution:")
        for c in contradictions:
            print(f"   - {c.concept}")

    print()
    print("Next steps:")
    print("1. Store high-confidence concepts in OpenMemory (category=glossary, scope=project)")
    print("2. Flag contradictions for weekly review")
    print("3. Track concept evolution over time")
    print("4. Use entity network to build knowledge graph")
    print()

    # Cost summary
    print("=" * 80)
    print("COST SUMMARY")
    print("=" * 80)
    print(f"This extraction cost: ${cost_estimate['estimated_cost_usd']:.4f}")
    print(f"Estimated annual cost (52 weekly videos): ${cost_estimate['estimated_cost_usd'] * 52:.2f}")
    print()


if __name__ == "__main__":
    main()
