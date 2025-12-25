"""
Example usage and integration patterns for Output Generator.

Shows how to integrate with existing MCP server and generate various outputs.
"""

import asyncio
from typing import Dict, List

from app.utils.output_generator import (
    BusinessModelCanvasGenerator,
    LeanCanvasGenerator,
    StrategyOnePagerGenerator,
    WeeklyDigestGenerator,
    OutputFormat
)


# Example 1: Integration with existing search
async def example_search_function(query: str, entity: str = None, vault: str = None,
                                   layer: str = None, limit: int = 10) -> List[Dict]:
    """
    Mock search function - replace with actual MCP search_memory call.

    In production, this would be:
        from app.mcp_server import search_memory
        return await search_memory(query=query, entity=entity, vault=vault, ...)
    """
    # Simulate search results
    return [
        {
            "id": "mem-123",
            "memory": "BMG's revenue model is freemium with enterprise upsell. Free tier for individuals, paid for teams.",
            "score": 0.85,
            "entity": entity,
            "tags": {"source_type": "video", "confidence": 0.8},
            "created_at": "2025-01-15T10:30:00Z"
        },
        {
            "id": "mem-456",
            "memory": "Target customer segment: Technical founders at seed-stage startups needing structured decision-making tools.",
            "score": 0.78,
            "entity": entity,
            "tags": {"source_type": "document", "confidence": 0.7},
            "created_at": "2025-01-14T14:20:00Z"
        },
        {
            "id": "mem-789",
            "memory": "Key partnership opportunity with CloudFactory for enterprise distribution channel.",
            "score": 0.72,
            "entity": entity,
            "tags": {"source_type": "conversation", "confidence": 0.6},
            "created_at": "2025-01-13T09:15:00Z"
        }
    ]


async def example_graph_query(cypher: str, params: Dict) -> List[Dict]:
    """Mock graph query - replace with actual Neo4j query."""
    return []


# Example 2: Generate Business Model Canvas
async def generate_bmc_example():
    """Example: Generate BMC for an entity."""
    # Initialize generator with search function
    bmc_gen = BusinessModelCanvasGenerator(
        search_fn=example_search_function,
        graph_query_fn=example_graph_query
    )

    # Generate canvas
    result = await bmc_gen.generate(
        entity="BMG",
        include_evidence=True,
        include_metadata=True
    )

    print("=== BUSINESS MODEL CANVAS ===")
    print(result["markdown"])
    print("\n")
    print(f"Overall Confidence: {result['overall_confidence']:.0%}")
    print(f"Generated: {result['generated_at']}")

    return result


# Example 3: Generate Lean Canvas
async def generate_lean_canvas_example():
    """Example: Generate Lean Canvas."""
    lean_gen = LeanCanvasGenerator(
        search_fn=example_search_function
    )

    result = await lean_gen.generate(
        entity="CloudFactory",
        include_evidence=True
    )

    print("=== LEAN CANVAS ===")
    print(result["markdown"])

    return result


# Example 4: Generate Strategy One-Pager
async def generate_strategy_example():
    """Example: Generate strategy doc."""
    strategy_gen = StrategyOnePagerGenerator(
        search_fn=example_search_function
    )

    result = await strategy_gen.generate(entity="BMG")

    print("=== STRATEGY ONE-PAGER ===")
    print(result["markdown"])

    return result


# Example 5: Generate Weekly Digest
async def generate_weekly_digest_example():
    """Example: Generate weekly digest."""
    digest_gen = WeeklyDigestGenerator(
        search_fn=example_search_function
    )

    # Entity-specific digest
    result = await digest_gen.generate(
        entity="BMG",
        days_back=7
    )

    print("=== WEEKLY DIGEST: BMG ===")
    print(result["markdown"])

    # Cross-entity digest
    all_digest = await digest_gen.generate(days_back=7)
    print("\n=== WEEKLY DIGEST: ALL CONCEPTS ===")
    print(all_digest["markdown"])

    return result


# Example 6: MCP Tool Integration
def add_output_generation_tools_to_mcp(mcp, search_memory_fn):
    """
    Add output generation tools to existing MCP server.

    Usage in mcp_server.py:
        from app.utils.output_examples import add_output_generation_tools_to_mcp
        add_output_generation_tools_to_mcp(mcp, search_memory)
    """

    @mcp.tool(description="""Generate Business Model Canvas from stored concepts.

    Creates a 9-block Business Model Canvas with:
    - Evidence citations for each section
    - Confidence levels
    - Gap identification
    - Source attribution

    Args:
        entity: The business/project entity to analyze (e.g., "BMG", "CloudFactory")
        include_evidence: Whether to include evidence citations (default: true)
        include_metadata: Whether to include confidence indicators (default: true)

    Returns JSON with:
        - markdown: Full BMC as formatted markdown
        - sections: Individual section data with evidence
        - overall_confidence: Aggregate confidence score
    """)
    async def generate_business_model_canvas(
        entity: str,
        include_evidence: bool = True,
        include_metadata: bool = True
    ) -> str:
        """Generate Business Model Canvas."""
        try:
            gen = BusinessModelCanvasGenerator(search_fn=search_memory_fn)
            result = await gen.generate(
                entity=entity,
                include_evidence=include_evidence,
                include_metadata=include_metadata
            )
            return result["markdown"]
        except Exception as e:
            return f"Error generating BMC: {str(e)}"

    @mcp.tool(description="""Generate Lean Canvas from stored concepts.

    Creates a Lean Canvas with:
    - Problem, Solution, UVP
    - Unfair Advantage
    - Customer Segments
    - Existing Alternatives
    - Key Metrics, Channels
    - Cost Structure, Revenue Streams

    Args:
        entity: The business/project entity
        include_evidence: Include source citations (default: true)
    """)
    async def generate_lean_canvas(
        entity: str,
        include_evidence: bool = True
    ) -> str:
        """Generate Lean Canvas."""
        try:
            gen = LeanCanvasGenerator(search_fn=search_memory_fn)
            result = await gen.generate(entity=entity, include_evidence=include_evidence)
            return result["markdown"]
        except Exception as e:
            return f"Error generating Lean Canvas: {str(e)}"

    @mcp.tool(description="""Generate one-page strategy summary.

    Creates a concise strategy document with:
    - Core Insight
    - Problem/Opportunity
    - Our Approach
    - Why Now
    - Key Risks & Mitigations
    - Next Steps

    Args:
        entity: The business/project entity
    """)
    async def generate_strategy_one_pager(entity: str) -> str:
        """Generate strategy one-pager."""
        try:
            gen = StrategyOnePagerGenerator(search_fn=search_memory_fn)
            result = await gen.generate(entity=entity)
            return result["markdown"]
        except Exception as e:
            return f"Error generating strategy doc: {str(e)}"

    @mcp.tool(description="""Generate weekly concept digest.

    Summarizes recent activity:
    - New concepts discovered
    - Contradictions to explore
    - Key insights

    Args:
        entity: Optional focus entity (if omitted, covers all concepts)
        days_back: Number of days to look back (default: 7)
    """)
    async def generate_weekly_digest(
        entity: str = None,
        days_back: int = 7
    ) -> str:
        """Generate weekly digest."""
        try:
            gen = WeeklyDigestGenerator(search_fn=search_memory_fn)
            result = await gen.generate(entity=entity, days_back=days_back)
            return result["markdown"]
        except Exception as e:
            return f"Error generating digest: {str(e)}"


# Example 7: Notion Integration
async def sync_bmc_to_notion(entity: str, notion_page_id: str):
    """
    Example: Generate BMC and sync to Notion.

    Requires: notion-client library
    """
    # Generate BMC
    gen = BusinessModelCanvasGenerator(search_fn=example_search_function)
    result = await gen.generate(entity=entity)

    # Convert to Notion blocks
    # (This would use notion-client library to create blocks)
    notion_blocks = [
        {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": f"Business Model Canvas: {entity}"}}]
            }
        },
        {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": result["markdown"]}}]
            }
        }
    ]

    # In production, would call Notion API:
    # notion.blocks.children.append(page_id=notion_page_id, children=notion_blocks)

    print(f"Would sync {len(notion_blocks)} blocks to Notion page {notion_page_id}")


# Example 8: n8n Workflow Integration
N8N_WORKFLOW_WEEKLY_DIGEST = {
    "name": "Weekly Business Concept Digest",
    "nodes": [
        {
            "name": "Schedule Trigger",
            "type": "n8n-nodes-base.cron",
            "position": [250, 300],
            "parameters": {
                "rule": {
                    "interval": [{"field": "cronExpression", "expression": "0 8 * * 1"}]  # Monday 8am
                }
            }
        },
        {
            "name": "Generate Digest",
            "type": "n8n-nodes-base.httpRequest",
            "position": [450, 300],
            "parameters": {
                "url": "http://localhost:8000/mcp/generate_weekly_digest",
                "method": "POST",
                "bodyParameters": {
                    "entity": None,
                    "days_back": 7
                }
            }
        },
        {
            "name": "Create Notion Page",
            "type": "n8n-nodes-base.notion",
            "position": [650, 300],
            "parameters": {
                "resource": "page",
                "operation": "create",
                "pageId": "{{$json[\"notion_database_id\"]}}",
                "title": "Weekly Digest - {{$now.format('YYYY-MM-DD')}}",
                "markdown": "{{$json[\"markdown\"]}}"
            }
        },
        {
            "name": "Send Email",
            "type": "n8n-nodes-base.emailSend",
            "position": [850, 300],
            "parameters": {
                "fromEmail": "concepts@yourdomain.com",
                "toEmail": "you@yourdomain.com",
                "subject": "Weekly Concept Digest",
                "text": "{{$json[\"markdown\"]}}"
            }
        }
    ],
    "connections": {
        "Schedule Trigger": {"main": [[{"node": "Generate Digest", "type": "main", "index": 0}]]},
        "Generate Digest": {"main": [[
            {"node": "Create Notion Page", "type": "main", "index": 0},
            {"node": "Send Email", "type": "main", "index": 0}
        ]]}
    }
}


# Example 9: Gap Detection
async def detect_bmc_gaps(entity: str) -> Dict[str, List[str]]:
    """
    Analyze BMC to find information gaps.

    Returns dict of section -> list of gaps.
    """
    gen = BusinessModelCanvasGenerator(search_fn=example_search_function)
    result = await gen.generate(entity=entity)

    gaps = {}
    for section_name, section_data in result["sections"].items():
        if section_data["gaps"]:
            gaps[section_name] = section_data["gaps"]

    return gaps


# Example 10: Confidence Evolution Tracking
async def track_concept_confidence_over_time(entity: str, days: int = 90):
    """
    Track how concept confidence has evolved.

    This would query memories with timestamps and track confidence scores.
    """
    # Pseudocode:
    # 1. Query memories for entity over time
    # 2. Group by time periods (weekly)
    # 3. Calculate confidence for each period
    # 4. Return time series data

    return {
        "entity": entity,
        "period_days": days,
        "confidence_evolution": [
            {"date": "2025-01-01", "confidence": 0.3},
            {"date": "2025-01-08", "confidence": 0.45},
            {"date": "2025-01-15", "confidence": 0.62},
            {"date": "2025-01-22", "confidence": 0.78}
        ]
    }


# Run examples
async def main():
    """Run all examples."""
    print("=" * 60)
    print("OUTPUT GENERATOR EXAMPLES")
    print("=" * 60)
    print()

    # await generate_bmc_example()
    # await generate_lean_canvas_example()
    # await generate_strategy_example()
    # await generate_weekly_digest_example()

    # Gap detection
    print("\n=== GAP DETECTION ===")
    gaps = await detect_bmc_gaps("BMG")
    for section, gap_list in gaps.items():
        print(f"\n{section}:")
        for gap in gap_list:
            print(f"  - {gap}")


if __name__ == "__main__":
    asyncio.run(main())
