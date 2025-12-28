"""Subject Access Request (SAR) Export for GDPR Compliance.

This module implements the SAR export functionality that retrieves all
user data from all data stores in the system. This is required for
GDPR Article 15: Right of access by the data subject.

The exporter connects to all available stores and retrieves:
- PostgreSQL: users, memories, apps, feedback, experiment assignments
- Neo4j: user graph nodes and relationships
- Qdrant: embedding vectors and payloads
- OpenSearch: indexed documents
- Valkey: session and episodic memory data
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.gdpr.schemas import SARResponse

logger = logging.getLogger(__name__)


class SARExporter:
    """Subject Access Request exporter for all stores.

    This class orchestrates the export of all user data across all data
    stores in the system. It handles partial failures gracefully and
    returns as much data as possible.

    Attributes:
        EXPORT_TIMEOUT_SECONDS: Maximum time for a complete export
    """

    EXPORT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        db: Session,
        neo4j_driver: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        opensearch_client: Optional[Any] = None,
        valkey_client: Optional[Any] = None,
    ):
        """Initialize the SAR exporter with store clients.

        Args:
            db: SQLAlchemy database session for PostgreSQL
            neo4j_driver: Neo4j driver instance (optional)
            qdrant_client: Qdrant client instance (optional)
            opensearch_client: OpenSearch client instance (optional)
            valkey_client: Valkey/Redis client instance (optional)
        """
        self._db = db
        self._neo4j = neo4j_driver
        self._qdrant = qdrant_client
        self._opensearch = opensearch_client
        self._valkey = valkey_client

    async def export_user_data(self, user_id: str) -> SARResponse:
        """Export all PII for a user across all stores.

        Args:
            user_id: The external user ID to export data for

        Returns:
            SARResponse containing all user data from all stores
        """
        start = datetime.now(timezone.utc)
        errors: List[str] = []

        # Export from each store with error handling
        postgres_data = self._export_postgres(user_id)

        neo4j_data: Dict[str, Any] = {}
        if self._neo4j:
            try:
                neo4j_data = self._export_neo4j(user_id)
            except Exception as e:
                logger.error(f"Neo4j export failed for user {user_id}: {e}")
                errors.append(f"Neo4j export failed: {e}")

        qdrant_data: Dict[str, Any] = {}
        if self._qdrant:
            try:
                qdrant_data = self._export_qdrant(user_id)
            except Exception as e:
                logger.error(f"Qdrant export failed for user {user_id}: {e}")
                errors.append(f"Qdrant export failed: {e}")

        opensearch_data: Dict[str, Any] = {}
        if self._opensearch:
            try:
                opensearch_data = self._export_opensearch(user_id)
            except Exception as e:
                logger.error(f"OpenSearch export failed for user {user_id}: {e}")
                errors.append(f"OpenSearch export failed: {e}")

        valkey_data: Dict[str, Any] = {}
        if self._valkey:
            try:
                valkey_data = self._export_valkey(user_id)
            except Exception as e:
                logger.error(f"Valkey export failed for user {user_id}: {e}")
                errors.append(f"Valkey export failed: {e}")

        duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SARResponse(
            user_id=user_id,
            export_date=datetime.now(timezone.utc),
            postgres=postgres_data,
            neo4j=neo4j_data,
            qdrant=qdrant_data,
            opensearch=opensearch_data,
            valkey=valkey_data,
            export_duration_ms=int(duration),
            partial=len(errors) > 0,
            errors=errors,
        )

    def _export_postgres(self, user_id: str) -> Dict[str, Any]:
        """Export user data from PostgreSQL tables.

        Args:
            user_id: The external user ID

        Returns:
            Dictionary containing user, memories, apps, feedback, experiments
        """
        from app.models import User, Memory, App

        # Get user record
        user = self._db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return {
                "user": None,
                "memories": [],
                "apps": [],
                "feedback": [],
                "experiments": [],
            }

        # Get all memories
        memories = self._db.query(Memory).filter(Memory.user_id == user.id).all()

        # Get all apps
        apps = self._db.query(App).filter(App.owner_id == user.id).all()

        # Get feedback events (try to import, may not exist)
        feedback = []
        try:
            from app.stores.feedback_store import FeedbackEventModel
            feedback = self._db.query(FeedbackEventModel).filter(
                FeedbackEventModel.user_id == user_id
            ).all()
        except ImportError:
            pass

        # Get experiment assignments (try to import, may not exist)
        experiments = []
        try:
            from app.stores.experiment_store import VariantAssignmentModel
            experiments = self._db.query(VariantAssignmentModel).filter(
                VariantAssignmentModel.user_id == user_id
            ).all()
        except ImportError:
            pass

        return {
            "user": self._serialize_user(user),
            "memories": [self._serialize_memory(m) for m in memories],
            "apps": [self._serialize_app(a) for a in apps],
            "feedback": [self._serialize_feedback(f) for f in feedback],
            "experiments": [self._serialize_assignment(a) for a in experiments],
        }

    def _export_neo4j(self, user_id: str) -> Dict[str, Any]:
        """Export user data from Neo4j graph.

        Args:
            user_id: The external user ID

        Returns:
            Dictionary containing nodes and relationships
        """
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        with self._neo4j.session() as session:
            # Get all nodes with user_id property
            result = session.run(
                """
                MATCH (n {user_id: $user_id})
                RETURN n, labels(n) as labels
                """,
                user_id=user_id,
            )
            for record in result:
                node = dict(record["n"])
                node["labels"] = record["labels"]
                nodes.append(node)

            # Get all relationships involving user's nodes
            result = session.run(
                """
                MATCH (n {user_id: $user_id})-[r]-(m)
                RETURN type(r) as type, properties(r) as props,
                       id(startNode(r)) as start_id, id(endNode(r)) as end_id
                """,
                user_id=user_id,
            )
            for record in result:
                relationships.append({
                    "type": record["type"],
                    "properties": record["props"],
                    "start_id": record["start_id"],
                    "end_id": record["end_id"],
                })

        return {
            "nodes": nodes,
            "relationships": relationships,
        }

    def _export_qdrant(self, user_id: str) -> Dict[str, Any]:
        """Export user data from Qdrant vector store.

        Args:
            user_id: The external user ID

        Returns:
            Dictionary containing embeddings by collection
        """
        from qdrant_client import models

        embeddings: Dict[str, List[Dict[str, Any]]] = {}

        # Get all collections
        collections = self._qdrant.get_collections().collections
        for collection in collections:
            if not collection.name.startswith("embeddings_"):
                continue

            # Scroll through all points with user_id filter
            points_data: List[Dict[str, Any]] = []
            offset = None
            while True:
                points, next_offset = self._qdrant.scroll(
                    collection_name=collection.name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Don't include raw vectors in export
                )

                for point in points:
                    points_data.append({
                        "id": str(point.id),
                        "payload": point.payload,
                    })

                if next_offset is None:
                    break
                offset = next_offset

            if points_data:
                embeddings[collection.name] = points_data

        return {"embeddings": embeddings}

    def _export_opensearch(self, user_id: str) -> Dict[str, Any]:
        """Export user data from OpenSearch index.

        Args:
            user_id: The external user ID

        Returns:
            Dictionary containing indexed documents
        """
        documents: List[Dict[str, Any]] = []

        # Search for all documents with user_id
        query = {
            "query": {
                "term": {"user_id": user_id}
            },
            "size": 10000,  # Max batch size
        }

        result = self._opensearch.search(index="memories", body=query)

        for hit in result.get("hits", {}).get("hits", []):
            documents.append({
                "id": hit["_id"],
                "source": hit["_source"],
            })

        return {"documents": documents}

    def _export_valkey(self, user_id: str) -> Dict[str, Any]:
        """Export user data from Valkey cache.

        Args:
            user_id: The external user ID

        Returns:
            Dictionary containing episodic session data
        """
        sessions: List[Dict[str, Any]] = []

        # Get all episodic memory keys for user
        pattern = f"episodic:{user_id}:*"
        keys = self._valkey.keys(pattern)

        for key in keys:
            value = self._valkey.get(key)
            if value:
                try:
                    data = json.loads(value) if isinstance(value, (str, bytes)) else value
                    sessions.append({
                        "key": key.decode() if isinstance(key, bytes) else key,
                        "data": data,
                    })
                except (json.JSONDecodeError, TypeError):
                    sessions.append({
                        "key": key.decode() if isinstance(key, bytes) else key,
                        "data": str(value),
                    })

        return {"episodic_sessions": sessions}

    def _serialize_user(self, user: Any) -> Dict[str, Any]:
        """Serialize a User model to dictionary."""
        return {
            "id": str(user.id),
            "user_id": user.user_id,
            "name": getattr(user, 'name', None),
            "email": getattr(user, 'email', None),
            "metadata": getattr(user, 'metadata_', None),
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        }

    def _serialize_memory(self, memory: Any) -> Dict[str, Any]:
        """Serialize a Memory model to dictionary."""
        return {
            "id": str(memory.id),
            "content": memory.content,
            "state": memory.state.value if hasattr(memory.state, 'value') else str(memory.state),
            "metadata": getattr(memory, 'metadata_', None),
            "vault": getattr(memory, 'vault', None),
            "layer": getattr(memory, 'layer', None),
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        }

    def _serialize_app(self, app: Any) -> Dict[str, Any]:
        """Serialize an App model to dictionary."""
        return {
            "id": str(app.id),
            "name": app.name,
            "description": getattr(app, 'description', None),
            "metadata": getattr(app, 'metadata_', None),
            "created_at": app.created_at.isoformat() if app.created_at else None,
            "updated_at": app.updated_at.isoformat() if app.updated_at else None,
        }

    def _serialize_feedback(self, feedback: Any) -> Dict[str, Any]:
        """Serialize a FeedbackEvent model to dictionary."""
        return {
            "event_id": feedback.event_id,
            "query_id": getattr(feedback, 'query_id', None),
            "action": getattr(feedback, 'action', None),
            "metadata": getattr(feedback, 'metadata_', None),
            "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
        }

    def _serialize_assignment(self, assignment: Any) -> Dict[str, Any]:
        """Serialize a VariantAssignment model to dictionary."""
        return {
            "id": str(assignment.id),
            "experiment_id": str(getattr(assignment, 'experiment_id', None)),
            "variant_config": getattr(assignment, 'variant_config', None),
            "created_at": assignment.created_at.isoformat() if assignment.created_at else None,
        }
