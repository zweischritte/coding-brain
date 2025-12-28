"""Cascading User Deletion for GDPR Compliance.

This module implements the Right to Erasure (Right to be Forgotten) as
required by GDPR Article 17. It orchestrates the deletion of all user
data across all data stores in the correct dependency order.

Deletion Order (dependencies first):
1. Valkey - Session/cache data (no dependencies, safe to delete first)
2. OpenSearch - Search indices (can be rebuilt from primary data)
3. Qdrant - Embeddings (can be rebuilt from primary data)
4. Neo4j - Graph relationships (references primary data)
5. PostgreSQL - Primary data (last, as other stores may reference it)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.gdpr.schemas import DeletionResult

logger = logging.getLogger(__name__)


class UserDeletionOrchestrator:
    """Orchestrate user deletion across all stores with audit trail.

    This class handles the complete deletion of a user's data from all
    stores in the system. It processes stores in dependency order and
    continues even if individual stores fail (with appropriate logging).

    Attributes:
        DELETION_ORDER: Order in which stores are processed
    """

    # Deletion order: dependencies first, primary data last
    DELETION_ORDER = [
        "valkey",      # Session/cache data (no dependencies)
        "opensearch",  # Search indices (can be rebuilt)
        "qdrant",      # Embeddings (can be rebuilt)
        "neo4j",       # Graph relationships
        "postgres",    # Primary data (last, FK constraints)
    ]

    def __init__(
        self,
        db: Session,
        neo4j_driver: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        opensearch_client: Optional[Any] = None,
        valkey_client: Optional[Any] = None,
    ):
        """Initialize the deletion orchestrator with store clients.

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

    async def delete_user(
        self,
        user_id: str,
        audit_reason: str,
        requestor_id: Optional[str] = None,
    ) -> DeletionResult:
        """Delete all user data with audit trail.

        Args:
            user_id: The external user ID to delete
            audit_reason: Reason for deletion (for audit trail)
            requestor_id: ID of the user/admin requesting deletion

        Returns:
            DeletionResult containing status of each store's deletion
        """
        audit_id = str(uuid4())
        results: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        logger.info(
            f"Starting user deletion: user_id={user_id}, "
            f"audit_id={audit_id}, requestor={requestor_id}"
        )

        for store in self.DELETION_ORDER:
            try:
                count = await self._delete_from_store(store, user_id)
                results[store] = {"status": "deleted", "count": count}
                logger.info(f"Deleted {count} records from {store} for user {user_id}")
            except Exception as e:
                error_msg = f"{store}: {e}"
                results[store] = {"status": "failed", "error": str(e)}
                errors.append(error_msg)
                logger.error(f"Failed to delete from {store} for user {user_id}: {e}")

        success = len(errors) == 0

        logger.info(
            f"Completed user deletion: user_id={user_id}, "
            f"success={success}, errors={len(errors)}"
        )

        return DeletionResult(
            audit_id=audit_id,
            user_id=user_id,
            timestamp=datetime.now(timezone.utc),
            results=results,
            success=success,
            errors=errors,
        )

    async def _delete_from_store(self, store: str, user_id: str) -> int:
        """Delete user data from a specific store.

        Args:
            store: Name of the store to delete from
            user_id: The external user ID

        Returns:
            Number of records deleted

        Raises:
            ValueError: If store name is unknown
        """
        if store == "valkey":
            return await self._delete_valkey(user_id)
        elif store == "opensearch":
            return await self._delete_opensearch(user_id)
        elif store == "qdrant":
            return await self._delete_qdrant(user_id)
        elif store == "neo4j":
            return await self._delete_neo4j(user_id)
        elif store == "postgres":
            return await self._delete_postgres(user_id)
        else:
            raise ValueError(f"Unknown store: {store}")

    async def _delete_valkey(self, user_id: str) -> int:
        """Delete all Valkey keys for user.

        Args:
            user_id: The external user ID

        Returns:
            Number of keys deleted
        """
        if not self._valkey:
            return 0

        # Delete episodic memory keys
        pattern = f"episodic:{user_id}:*"
        keys = self._valkey.keys(pattern)
        if keys:
            self._valkey.delete(*keys)
        return len(keys) if keys else 0

    async def _delete_opensearch(self, user_id: str) -> int:
        """Delete all OpenSearch documents for user.

        Args:
            user_id: The external user ID

        Returns:
            Number of documents deleted
        """
        if not self._opensearch:
            return 0

        try:
            response = self._opensearch.delete_by_query(
                index="memories",
                body={"query": {"term": {"user_id": user_id}}},
            )
            return response.get("deleted", 0)
        except Exception as e:
            # Index may not exist
            if "index_not_found" in str(e).lower():
                return 0
            raise

    async def _delete_qdrant(self, user_id: str) -> int:
        """Delete all Qdrant points for user.

        Args:
            user_id: The external user ID

        Returns:
            Number of points deleted (approximate)
        """
        if not self._qdrant:
            return 0

        from qdrant_client import models

        # Delete from all embedding collections
        collections = self._qdrant.get_collections().collections
        count = 0
        for coll in collections:
            if coll.name.startswith("embeddings_"):
                try:
                    self._qdrant.delete(
                        collection_name=coll.name,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="user_id",
                                        match=models.MatchValue(value=user_id),
                                    )
                                ]
                            )
                        ),
                    )
                    count += 1  # Count collections processed, not exact points
                except Exception as e:
                    logger.warning(f"Failed to delete from {coll.name}: {e}")
        return count

    async def _delete_neo4j(self, user_id: str) -> int:
        """Delete all Neo4j nodes for user.

        Uses DETACH DELETE to also remove all relationships involving
        the deleted nodes.

        Args:
            user_id: The external user ID

        Returns:
            Number of nodes deleted
        """
        if not self._neo4j:
            return 0

        with self._neo4j.session() as session:
            result = session.run(
                """
                MATCH (n {user_id: $user_id})
                DETACH DELETE n
                RETURN count(n) as deleted
                """,
                user_id=user_id,
            )
            record = result.single()
            return record["deleted"] if record else 0

    async def _delete_postgres(self, user_id: str) -> int:
        """Delete all PostgreSQL data for user.

        Deletes in correct order to respect foreign key constraints:
        1. Feedback events (references user_id string)
        2. Variant assignments (references user_id string)
        3. Memories (references user.id)
        4. Apps (references user.id as owner_id)
        5. User record

        Args:
            user_id: The external user ID

        Returns:
            Total number of records deleted
        """
        from app.models import User, Memory, App

        # Get user
        user = self._db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return 0

        count = 0

        # Delete feedback events
        try:
            from app.stores.feedback_store import FeedbackEventModel
            count += self._db.query(FeedbackEventModel).filter(
                FeedbackEventModel.user_id == user_id
            ).delete()
        except ImportError:
            pass

        # Delete variant assignments
        try:
            from app.stores.experiment_store import VariantAssignmentModel
            count += self._db.query(VariantAssignmentModel).filter(
                VariantAssignmentModel.user_id == user_id
            ).delete()
        except ImportError:
            pass

        # Delete memories (cascade will handle memory_categories, status_history, access_logs)
        count += self._db.query(Memory).filter(Memory.user_id == user.id).delete()

        # Delete apps
        count += self._db.query(App).filter(App.owner_id == user.id).delete()

        # Delete user
        self._db.delete(user)
        count += 1

        self._db.commit()
        return count
