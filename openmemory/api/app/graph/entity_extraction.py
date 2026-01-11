from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """An entity extracted from memory content."""
    name: str
    entity_type: str


@dataclass
class ExtractedRelation:
    """A relationship between two entities."""
    source: str
    relationship: str
    destination: str


@dataclass
class ExtractionResult:
    entity_type_map: Dict[str, str]
    relations_raw: List[Dict[str, str]]
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


def extract_entities_and_relations(
    content: str,
    user_id: str,
    filters: Optional[Dict[str, str]] = None,
) -> ExtractionResult:
    """
    Extract entities and relations from content using Mem0's graph extraction.

    Returns extraction output without writing to any graph.
    """
    try:
        from app.utils.memory import get_memory_client

        memory_client = get_memory_client()
        if memory_client is None or not hasattr(memory_client, "graph") or memory_client.graph is None:
            logger.debug("Mem0 graph not configured, skipping entity extraction")
            return ExtractionResult({}, [], [], [])

        graph = memory_client.graph
        effective_filters: Dict[str, str] = {"user_id": user_id}
        if filters:
            effective_filters.update(filters)

        if hasattr(graph, "extract_entities_and_relations"):
            entity_type_map, relations_raw = graph.extract_entities_and_relations(
                content, effective_filters
            )
        else:
            entity_type_map = graph._retrieve_nodes_from_data(content, effective_filters)
            if not entity_type_map:
                return ExtractionResult({}, [], [], [])
            relations_raw = graph._establish_nodes_relations_from_data(
                content, effective_filters, entity_type_map
            )

        if not entity_type_map:
            return ExtractionResult({}, [], [], [])

        entities = [
            ExtractedEntity(name=name, entity_type=entity_type)
            for name, entity_type in entity_type_map.items()
        ]
        relations = [
            ExtractedRelation(
                source=r["source"],
                relationship=r["relationship"],
                destination=r["destination"],
            )
            for r in relations_raw
        ]

        return ExtractionResult(entity_type_map, relations_raw, entities, relations)
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return ExtractionResult({}, [], [], [])


def write_mem0_graph_from_extraction(
    content: str,
    user_id: str,
    extraction: ExtractionResult,
    filters: Optional[Dict[str, str]] = None,
) -> Dict[str, List]:
    """
    Write Mem0 graph entities/relations using precomputed extraction.
    """
    from app.utils.memory import get_memory_client

    memory_client = get_memory_client()
    if memory_client is None or not hasattr(memory_client, "graph") or memory_client.graph is None:
        return {"deleted_entities": [], "added_entities": []}

    effective_filters: Dict[str, str] = {"user_id": user_id}
    if filters:
        effective_filters.update(filters)

    graph = memory_client.graph
    if not extraction.entity_type_map:
        return {"deleted_entities": [], "added_entities": []}

    if hasattr(graph, "add_from_extraction"):
        return graph.add_from_extraction(
            content,
            effective_filters,
            extraction.entity_type_map,
            extraction.relations_raw,
        )

    # Fallback: no precomputed write path available
    return graph.add(content, effective_filters)
