"""Enable Row Level Security on tenant-scoped tables

This migration enables PostgreSQL Row Level Security (RLS) on the memories
table to enforce tenant isolation at the database level.

RLS policies filter rows based on the 'app.current_user_id' session variable,
which is set by the tenant_session() context manager in application code.

Security model:
- All operations (SELECT, INSERT, UPDATE, DELETE) are restricted to rows
  where user_id matches the session variable
- Queries without the session variable set return empty results (fail closed)
- Superusers bypass RLS for administrative operations

Revision ID: add_rls_policies
Revises: structured_memory_cleanup
Create Date: 2025-12-27

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_rls_policies'
down_revision = 'structured_memory_cleanup'
branch_labels = None
depends_on = None


def upgrade():
    """Enable RLS and create tenant isolation policies."""
    # Only apply RLS on PostgreSQL
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        return

    # Enable RLS on memories table
    op.execute("ALTER TABLE memories ENABLE ROW LEVEL SECURITY")

    # Create policy for SELECT - users can only see their own memories
    op.execute("""
        CREATE POLICY memories_tenant_select ON memories
        FOR SELECT
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)

    # Create policy for INSERT - users can only insert memories for themselves
    op.execute("""
        CREATE POLICY memories_tenant_insert ON memories
        FOR INSERT
        WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)

    # Create policy for UPDATE - users can only update their own memories
    op.execute("""
        CREATE POLICY memories_tenant_update ON memories
        FOR UPDATE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        WITH CHECK (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)

    # Create policy for DELETE - users can only delete their own memories
    op.execute("""
        CREATE POLICY memories_tenant_delete ON memories
        FOR DELETE
        USING (user_id = NULLIF(current_setting('app.current_user_id', true), '')::uuid)
    """)

    # Enable RLS on apps table
    op.execute("ALTER TABLE apps ENABLE ROW LEVEL SECURITY")

    # Create policies for apps table (owned by owner_id)
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


def downgrade():
    """Remove RLS policies and disable RLS."""
    conn = op.get_bind()
    if conn.dialect.name != 'postgresql':
        return

    # Drop apps policies
    op.execute("DROP POLICY IF EXISTS apps_tenant_select ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_insert ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_update ON apps")
    op.execute("DROP POLICY IF EXISTS apps_tenant_delete ON apps")
    op.execute("ALTER TABLE apps DISABLE ROW LEVEL SECURITY")

    # Drop memories policies
    op.execute("DROP POLICY IF EXISTS memories_tenant_select ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_insert ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_update ON memories")
    op.execute("DROP POLICY IF EXISTS memories_tenant_delete ON memories")
    op.execute("ALTER TABLE memories DISABLE ROW LEVEL SECURITY")
