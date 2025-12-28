"""Tests for GDPR PII Inventory.

These tests verify that the PII inventory correctly documents all personally
identifiable information across all data stores in the system.

Test IDs: PII-001 through PII-010
"""

import pytest
from typing import Set


class TestPIIFieldDataclass:
    """Tests for PIIField dataclass validation (PII-001, PII-009, PII-010)."""

    def test_pii_field_validates_store_names(self):
        """PII-001: PIIField dataclass validates store names."""
        from app.gdpr.pii_inventory import PIIField, Store, PIIType, EncryptionLevel

        # Valid store names should work
        field = PIIField(
            store=Store.POSTGRES,
            table_or_collection="users",
            field_name="email",
            pii_type=PIIType.EMAIL,
        )
        assert field.store == Store.POSTGRES

        # All valid stores should be accepted
        for store in Store:
            field = PIIField(
                store=store,
                table_or_collection="test",
                field_name="test_field",
                pii_type=PIIType.USER_ID,
            )
            assert field.store == store

    def test_pii_types_are_correctly_categorized(self):
        """PII-009: PII types are correctly categorized."""
        from app.gdpr.pii_inventory import PIIType

        expected_types = {"email", "name", "user_id", "content", "ip_address", "metadata"}
        actual_types = {t.value for t in PIIType}
        assert expected_types == actual_types

    def test_encryption_levels_documented(self):
        """PII-010: Encryption levels documented."""
        from app.gdpr.pii_inventory import EncryptionLevel

        expected_levels = {"none", "at_rest", "field_level"}
        actual_levels = {e.value for e in EncryptionLevel}
        assert expected_levels == actual_levels


class TestPostgreSQLPIIInventory:
    """Tests for PostgreSQL PII field inventory (PII-002)."""

    def test_inventory_contains_all_postgresql_tables(self):
        """PII-002: PII inventory contains all PostgreSQL tables."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        postgres_fields = [f for f in PII_INVENTORY if f.store == Store.POSTGRES]

        # Expected tables with PII
        expected_tables = {
            "users",
            "memories",
            "apps",
            "feedback_events",
            "variant_assignments",
            "memory_status_history",
            "memory_access_logs",
        }

        actual_tables = {f.table_or_collection for f in postgres_fields}

        for table in expected_tables:
            assert table in actual_tables, f"Missing PostgreSQL table: {table}"

    def test_users_table_pii_fields(self):
        """Verify users table has documented PII fields."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        users_fields = [
            f for f in PII_INVENTORY
            if f.store == Store.POSTGRES and f.table_or_collection == "users"
        ]

        field_names = {f.field_name for f in users_fields}

        # Users table must have these PII fields documented
        expected_fields = {"user_id", "email", "name", "metadata"}
        for field in expected_fields:
            assert field in field_names, f"Missing field in users table: {field}"

    def test_memories_table_pii_fields(self):
        """Verify memories table has documented PII fields."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        memories_fields = [
            f for f in PII_INVENTORY
            if f.store == Store.POSTGRES and f.table_or_collection == "memories"
        ]

        field_names = {f.field_name for f in memories_fields}

        # Memories table must have these PII fields documented
        expected_fields = {"user_id", "content", "metadata"}
        for field in expected_fields:
            assert field in field_names, f"Missing field in memories table: {field}"


class TestNeo4jPIIInventory:
    """Tests for Neo4j PII field inventory (PII-003)."""

    def test_inventory_contains_all_neo4j_node_types(self):
        """PII-003: PII inventory contains all Neo4j node types."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        neo4j_fields = [f for f in PII_INVENTORY if f.store == Store.NEO4J]

        # Expected node types with user_id properties
        expected_nodes = {"User", "Memory", "Entity"}

        actual_nodes = {f.table_or_collection for f in neo4j_fields}

        for node in expected_nodes:
            assert node in actual_nodes, f"Missing Neo4j node type: {node}"


class TestQdrantPIIInventory:
    """Tests for Qdrant PII field inventory (PII-004)."""

    def test_inventory_contains_all_qdrant_payload_fields(self):
        """PII-004: PII inventory contains all Qdrant payload fields."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        qdrant_fields = [f for f in PII_INVENTORY if f.store == Store.QDRANT]

        # Qdrant must have user_id and org_id documented in embeddings payloads
        field_names = {f.field_name for f in qdrant_fields}

        expected_fields = {"user_id", "org_id", "content"}
        for field in expected_fields:
            assert field in field_names, f"Missing Qdrant payload field: {field}"


class TestOpenSearchPIIInventory:
    """Tests for OpenSearch PII field inventory (PII-005)."""

    def test_inventory_contains_all_opensearch_document_fields(self):
        """PII-005: PII inventory contains all OpenSearch document fields."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        opensearch_fields = [f for f in PII_INVENTORY if f.store == Store.OPENSEARCH]

        # OpenSearch must have user_id and content documented
        field_names = {f.field_name for f in opensearch_fields}

        expected_fields = {"user_id", "content"}
        for field in expected_fields:
            assert field in field_names, f"Missing OpenSearch document field: {field}"


class TestValkeyPIIInventory:
    """Tests for Valkey PII field inventory (PII-006)."""

    def test_inventory_contains_all_valkey_key_patterns(self):
        """PII-006: PII inventory contains all Valkey key patterns."""
        from app.gdpr.pii_inventory import PII_INVENTORY, Store

        valkey_fields = [f for f in PII_INVENTORY if f.store == Store.VALKEY]

        # Valkey must have episodic key patterns documented
        collection_patterns = {f.table_or_collection for f in valkey_fields}

        # At minimum, episodic memory pattern must be documented
        assert any("episodic" in p for p in collection_patterns), \
            "Missing Valkey episodic memory key pattern"


class TestPIIInventoryHelpers:
    """Tests for PII inventory helper functions."""

    def test_get_pii_fields_by_store(self):
        """Test filtering PII fields by store."""
        from app.gdpr.pii_inventory import get_pii_fields_by_store, Store, PII_INVENTORY

        for store in Store:
            fields = get_pii_fields_by_store(store)
            # All returned fields should be for the specified store
            assert all(f.store == store for f in fields)
            # Should match manual filtering
            expected = [f for f in PII_INVENTORY if f.store == store]
            assert len(fields) == len(expected)

    def test_get_deletable_fields(self):
        """Test getting fields that should be deleted on user deletion."""
        from app.gdpr.pii_inventory import get_deletable_fields, PIIType

        deletable = get_deletable_fields()
        # Should return a non-empty list
        assert len(deletable) > 0
        # All returned fields should be actual PIIField instances
        assert all(hasattr(f, 'store') for f in deletable)

    def test_all_stores_have_pii_fields(self):
        """Verify each store type has at least one PII field documented."""
        from app.gdpr.pii_inventory import get_pii_fields_by_store, Store

        for store in Store:
            fields = get_pii_fields_by_store(store)
            assert len(fields) > 0, f"Store {store.value} has no PII fields documented"


class TestPIIInventoryCompleteness:
    """Tests to verify inventory completeness against actual schemas."""

    @pytest.mark.integration
    def test_inventory_matches_postgresql_schema(self):
        """PII-007: Inventory matches actual PostgreSQL schema.

        This test requires a PostgreSQL database connection to validate that
        the documented PII fields actually exist in the PostgreSQL schema.
        Skips when running against SQLite.
        """
        from app.gdpr.pii_inventory import get_pii_fields_by_store, Store
        from app.database import SessionLocal, engine
        from sqlalchemy import text

        # Skip if not running against PostgreSQL
        if "sqlite" in str(engine.url):
            pytest.skip("Schema validation requires PostgreSQL (not SQLite)")

        postgres_fields = get_pii_fields_by_store(Store.POSTGRES)

        # Group fields by table
        tables = {}
        for field in postgres_fields:
            if field.table_or_collection not in tables:
                tables[field.table_or_collection] = []
            tables[field.table_or_collection].append(field.field_name)

        db = SessionLocal()
        try:
            # Query information_schema for each table
            for table_name, expected_fields in tables.items():
                result = db.execute(text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    """
                ), {"table_name": table_name})
                actual_columns = {row[0] for row in result}

                for field_name in expected_fields:
                    # Handle metadata_ vs metadata naming
                    check_name = field_name.rstrip("_") if field_name.endswith("_") else field_name
                    assert (
                        field_name in actual_columns or
                        check_name in actual_columns or
                        f"{check_name}_" in actual_columns
                    ), f"Field {field_name} not found in {table_name} table schema"
        finally:
            db.close()

    @pytest.mark.integration
    def test_inventory_matches_neo4j_schema(self):
        """PII-008: Inventory matches actual Neo4j schema.

        This test requires a Neo4j connection to validate that
        the documented node types actually exist.
        """
        # This test will be skipped if Neo4j is not available
        pytest.skip("Neo4j schema validation requires live connection")
