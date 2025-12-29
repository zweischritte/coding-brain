"""disable_rls_for_shared_access

Revision ID: disable_rls_for_shared_access
Revises: add_access_entity_index
Create Date: 2025-02-10

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "disable_rls_for_shared_access"
down_revision = "add_access_entity_index"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    # Disable RLS so access_entity sharing is not blocked at the DB layer.
    op.execute("DROP POLICY IF EXISTS memories_tenant_select ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_insert ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_update ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_delete ON memories")
    op.execute("ALTER TABLE memories DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS apps_tenant_select ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_insert ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_update ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_delete ON apps")
    op.execute("ALTER TABLE apps DISABLE ROW LEVEL SECURITY")


def downgrade():
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    # Restore the original per-user RLS policies.
    op.execute("ALTER TABLE memories ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY memories_tenant_select ON memories
        FOR SELECT
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY memories_tenant_insert ON memories
        FOR INSERT
        WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY memories_tenant_update ON memories
        FOR UPDATE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY memories_tenant_delete ON memories
        FOR DELETE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)

    op.execute("ALTER TABLE apps ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY apps_tenant_select ON apps
        FOR SELECT
        USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY apps_tenant_insert ON apps
        FOR INSERT
        WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY apps_tenant_update ON apps
        FOR UPDATE
        USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        WITH CHECK (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
    op.execute("""
        CREATE POLICY apps_tenant_delete ON apps
        FOR DELETE
        USING (owner_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)
