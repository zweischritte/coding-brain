"""add_access_entity_index

Revision ID: add_access_entity_index
Revises: add_rls_policies
Create Date: 2025-02-10

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_access_entity_index"
down_revision = "add_rls_policies"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_access_entity "
        "ON memories ((metadata->>'access_entity'))"
    )


def downgrade():
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS idx_memories_access_entity")
