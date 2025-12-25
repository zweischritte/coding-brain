"""
Biografische Timeline für OpenMemory.

Ermöglicht temporale Queries wie:
- "Was passierte vor X?"
- "Zeige Ereignisse zwischen 2014-2018"
- "Welche Projekte überlappten?"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.graph.neo4j_client import get_neo4j_session, is_neo4j_configured

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    RESIDENCE = "residence"      # Wohnort
    EDUCATION = "education"      # Ausbildung
    WORK = "work"               # Beruf
    PROJECT = "project"         # Projekt (Film, etc.)
    RELATIONSHIP = "relationship"  # Beziehungsereignis
    HEALTH = "health"           # Gesundheit
    TRAVEL = "travel"           # Reise
    MILESTONE = "milestone"     # Lebensereignis (Geburt, etc.)


class TemporalRelation(str, Enum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    OVERLAPS = "OVERLAPS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"


@dataclass
class TemporalEvent:
    """Ein temporales Ereignis in der Biografie."""
    name: str
    event_type: EventType
    start_date: Optional[str] = None  # YYYY, YYYY-MM, or YYYY-MM-DD
    end_date: Optional[str] = None
    description: Optional[str] = None
    entity: Optional[str] = None      # Zugehörige Entity (Person, Ort, etc.)
    memory_ids: List[str] = None      # Quell-Memories

    def __post_init__(self):
        if self.memory_ids is None:
            self.memory_ids = []


def ensure_temporal_constraints():
    """Erstellt Neo4j Constraints für TemporalEvent Nodes."""
    if not is_neo4j_configured():
        return

    constraints = [
        """
        CREATE CONSTRAINT om_temporal_event_unique IF NOT EXISTS
        FOR (t:OM_TemporalEvent) REQUIRE (t.userId, t.name) IS UNIQUE
        """,
        """
        CREATE INDEX om_temporal_event_date IF NOT EXISTS
        FOR (t:OM_TemporalEvent) ON (t.startDate)
        """,
    ]

    try:
        with get_neo4j_session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception:
                    pass  # Constraint already exists
    except Exception as e:
        logger.warning(f"Failed to create temporal constraints: {e}")


def parse_date_from_text(text: str) -> Optional[Tuple[str, str]]:
    """
    Extrahiert Start- und End-Datum aus Text.

    Unterstützte Formate:
    - "2014" → ("2014", None)
    - "2014-2018" → ("2014", "2018")
    - "1986-1988" → ("1986", "1988")
    - "seit 2020" → ("2020", None)
    - "bis 2019" → (None, "2019")

    Returns:
        Tuple (start_date, end_date) oder None
    """
    # Jahr-Range: "2014-2018" oder "1986 - 1988"
    range_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', text)
    if range_match:
        return (range_match.group(1), range_match.group(2))

    # "seit YYYY"
    seit_match = re.search(r'seit\s+(\d{4})', text, re.IGNORECASE)
    if seit_match:
        return (seit_match.group(1), None)

    # "bis YYYY"
    bis_match = re.search(r'bis\s+(\d{4})', text, re.IGNORECASE)
    if bis_match:
        return (None, bis_match.group(1))

    # Einzelnes Jahr
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if year_match:
        return (year_match.group(1), None)

    return None


def create_temporal_event(
    user_id: str,
    event: TemporalEvent,
) -> bool:
    """
    Erstellt einen OM_TemporalEvent Node.

    Wenn event.entity angegeben, wird auch eine OM_TEMPORAL Kante
    zur Entity erstellt.
    """
    if not is_neo4j_configured():
        return False

    query = """
    MERGE (t:OM_TemporalEvent {userId: $userId, name: $name})
    ON CREATE SET
        t.eventType = $eventType,
        t.startDate = $startDate,
        t.endDate = $endDate,
        t.description = $description,
        t.memoryIds = $memoryIds,
        t.createdAt = datetime()
    ON MATCH SET
        t.updatedAt = datetime(),
        t.memoryIds = t.memoryIds + [x IN $memoryIds WHERE NOT x IN t.memoryIds]
    RETURN t.name AS name
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                name=event.name,
                eventType=event.event_type.value,
                startDate=event.start_date,
                endDate=event.end_date,
                description=event.description,
                memoryIds=event.memory_ids,
            )
            record = result.single()

            # Link to entity if specified
            if record and event.entity:
                link_query = """
                MATCH (t:OM_TemporalEvent {userId: $userId, name: $eventName})
                MATCH (e:OM_Entity {userId: $userId, name: $entityName})
                MERGE (e)-[:OM_TEMPORAL]->(t)
                """
                session.run(
                    link_query,
                    userId=user_id,
                    eventName=event.name,
                    entityName=event.entity,
                )

            return record is not None
    except Exception as e:
        logger.exception(f"Error creating temporal event: {e}")
        return False


def get_biography_timeline(
    user_id: str,
    entity_name: Optional[str] = None,
    event_types: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Holt die biografische Timeline für eine Entity oder den gesamten User.

    Args:
        user_id: User ID
        entity_name: Optional - beschränkt auf Events einer Entity
        event_types: Optional - Filter nach Event-Typen
        start_year: Optional - Nur Events ab diesem Jahr
        end_year: Optional - Nur Events bis zu diesem Jahr
        limit: Max Events

    Returns:
        Chronologisch sortierte Events
    """
    if not is_neo4j_configured():
        return []

    # Build query based on whether entity is specified
    if entity_name:
        match_clause = """
        MATCH (e:OM_Entity {userId: $userId, name: $entityName})-[:OM_TEMPORAL]->(t:OM_TemporalEvent)
        """
    else:
        match_clause = """
        MATCH (t:OM_TemporalEvent {userId: $userId})
        """

    where_clauses = []
    if event_types:
        where_clauses.append("t.eventType IN $eventTypes")
    if start_year:
        where_clauses.append(f"toInteger(substring(t.startDate, 0, 4)) >= {start_year}")
    if end_year:
        where_clauses.append(f"toInteger(substring(t.startDate, 0, 4)) <= {end_year}")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    query = f"""
    {match_clause}
    {where_clause}
    OPTIONAL MATCH (t)<-[:OM_TEMPORAL]-(related:OM_Entity)
    RETURN
        t.name AS name,
        t.eventType AS eventType,
        t.startDate AS startDate,
        t.endDate AS endDate,
        t.description AS description,
        t.memoryIds AS memoryIds,
        collect(DISTINCT related.name) AS relatedEntities
    ORDER BY t.startDate ASC
    LIMIT $limit
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                entityName=entity_name,
                eventTypes=event_types,
                limit=limit,
            )

            events = []
            for record in result:
                events.append({
                    "name": record["name"],
                    "event_type": record["eventType"],
                    "start_date": record["startDate"],
                    "end_date": record["endDate"],
                    "description": record["description"],
                    "memory_ids": record["memoryIds"] or [],
                    "related_entities": record["relatedEntities"] or [],
                })

            return events
    except Exception as e:
        logger.exception(f"Error getting biography timeline: {e}")
        return []


def delete_temporal_event(user_id: str, event_name: str) -> bool:
    """
    Löscht einen TemporalEvent Node und alle zugehörigen Kanten.

    Args:
        user_id: User ID
        event_name: Name des zu löschenden Events

    Returns:
        True wenn erfolgreich gelöscht
    """
    if not is_neo4j_configured():
        return False

    query = """
    MATCH (t:OM_TemporalEvent {userId: $userId, name: $eventName})
    DETACH DELETE t
    RETURN count(*) AS deleted
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                eventName=event_name,
            )
            record = result.single()
            return record and record["deleted"] > 0
    except Exception as e:
        logger.exception(f"Error deleting temporal event: {e}")
        return False


def find_overlapping_events(
    user_id: str,
    event_name: str,
) -> List[Dict[str, Any]]:
    """
    Findet Events die zeitlich mit dem gegebenen Event überlappen.

    Args:
        user_id: User ID
        event_name: Name des Referenz-Events

    Returns:
        Liste von überlappenden Events
    """
    if not is_neo4j_configured():
        return []

    query = """
    MATCH (ref:OM_TemporalEvent {userId: $userId, name: $eventName})
    MATCH (other:OM_TemporalEvent {userId: $userId})
    WHERE other.name <> ref.name
      AND ref.startDate IS NOT NULL
      AND other.startDate IS NOT NULL
      AND (
        // other starts during ref
        (other.startDate >= ref.startDate AND
         (ref.endDate IS NULL OR other.startDate <= ref.endDate))
        OR
        // ref starts during other
        (ref.startDate >= other.startDate AND
         (other.endDate IS NULL OR ref.startDate <= other.endDate))
      )
    RETURN
        other.name AS name,
        other.eventType AS eventType,
        other.startDate AS startDate,
        other.endDate AS endDate,
        other.description AS description
    ORDER BY other.startDate
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                eventName=event_name,
            )

            events = []
            for record in result:
                events.append({
                    "name": record["name"],
                    "event_type": record["eventType"],
                    "start_date": record["startDate"],
                    "end_date": record["endDate"],
                    "description": record["description"],
                })

            return events
    except Exception as e:
        logger.exception(f"Error finding overlapping events: {e}")
        return []


def get_events_in_year_range(
    user_id: str,
    start_year: int,
    end_year: int,
    entity_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Holt alle Events in einem Jahreszeitraum.

    Args:
        user_id: User ID
        start_year: Startjahr (inklusive)
        end_year: Endjahr (inklusive)
        entity_name: Optional - beschränkt auf eine Entity

    Returns:
        Events im Zeitraum
    """
    return get_biography_timeline(
        user_id=user_id,
        entity_name=entity_name,
        start_year=start_year,
        end_year=end_year,
        limit=200,
    )


def link_event_to_memory(
    user_id: str,
    event_name: str,
    memory_id: str,
) -> bool:
    """
    Verknüpft ein Event mit einem Memory (fügt memory_id zur Liste hinzu).

    Args:
        user_id: User ID
        event_name: Event Name
        memory_id: Memory ID

    Returns:
        True wenn erfolgreich
    """
    if not is_neo4j_configured():
        return False

    query = """
    MATCH (t:OM_TemporalEvent {userId: $userId, name: $eventName})
    WHERE NOT $memoryId IN t.memoryIds
    SET t.memoryIds = coalesce(t.memoryIds, []) + [$memoryId],
        t.updatedAt = datetime()
    RETURN t.name AS name
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                eventName=event_name,
                memoryId=memory_id,
            )
            return result.single() is not None
    except Exception as e:
        logger.exception(f"Error linking event to memory: {e}")
        return False


def link_event_to_entity(
    user_id: str,
    event_name: str,
    entity_name: str,
) -> bool:
    """
    Erstellt eine OM_TEMPORAL Kante zwischen Entity und Event.

    Args:
        user_id: User ID
        event_name: Event Name
        entity_name: Entity Name

    Returns:
        True wenn erfolgreich
    """
    if not is_neo4j_configured():
        return False

    query = """
    MATCH (t:OM_TemporalEvent {userId: $userId, name: $eventName})
    MATCH (e:OM_Entity {userId: $userId, name: $entityName})
    MERGE (e)-[:OM_TEMPORAL]->(t)
    RETURN t.name AS name
    """

    try:
        with get_neo4j_session() as session:
            result = session.run(
                query,
                userId=user_id,
                eventName=event_name,
                entityName=entity_name,
            )
            return result.single() is not None
    except Exception as e:
        logger.exception(f"Error linking event to entity: {e}")
        return False
