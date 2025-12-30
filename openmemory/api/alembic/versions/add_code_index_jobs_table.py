"""Add code_index_jobs table

Revision ID: add_code_index_jobs_table
Revises: disable_rls_for_shared_access
Create Date: 2025-12-30

Adds persistent storage for code indexing jobs with:
- UUID primary key
- Status enum (queued, running, succeeded, failed, canceled)
- Progress and summary JSON fields
- Heartbeat for orphan detection
- Composite indexes for common query patterns
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_code_index_jobs_table"
down_revision: Union[str, None] = "disable_rls_for_shared_access"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create code_index_jobs table with all indexes."""
    conn = op.get_bind()

    # Create the status enum type for PostgreSQL
    if conn.dialect.name == "postgresql":
        op.execute("""
            DO $$ BEGIN
                CREATE TYPE codeindexjobstatus AS ENUM (
                    'queued', 'running', 'succeeded', 'failed', 'canceled'
                );
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)

    # Create the table
    status_type = (
        sa.Enum('queued', 'running', 'succeeded', 'failed', 'canceled',
                name='codeindexjobstatus', create_type=False)
        if conn.dialect.name == "postgresql"
        else sa.String()
    )

    op.create_table(
        'code_index_jobs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('repo_id', sa.String(), nullable=False),
        sa.Column('root_path', sa.String(), nullable=False),
        sa.Column('index_name', sa.String(), nullable=False, server_default='code'),
        sa.Column('status', status_type, nullable=False, server_default='queued'),
        sa.Column('requested_by', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cancel_requested', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('request', sa.JSON(), nullable=True),
        sa.Column('progress', sa.JSON(), nullable=True),
        sa.Column('summary', sa.JSON(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create single-column indexes
    op.create_index('ix_code_index_jobs_repo_id', 'code_index_jobs', ['repo_id'])
    op.create_index('ix_code_index_jobs_status', 'code_index_jobs', ['status'])
    op.create_index('ix_code_index_jobs_requested_by', 'code_index_jobs', ['requested_by'])
    op.create_index('ix_code_index_jobs_created_at', 'code_index_jobs', ['created_at'])
    op.create_index('ix_code_index_jobs_started_at', 'code_index_jobs', ['started_at'])
    op.create_index('ix_code_index_jobs_finished_at', 'code_index_jobs', ['finished_at'])
    op.create_index('ix_code_index_jobs_last_heartbeat', 'code_index_jobs', ['last_heartbeat'])

    # Create composite indexes for common query patterns
    op.create_index('idx_code_job_repo_status', 'code_index_jobs', ['repo_id', 'status'])
    op.create_index('idx_code_job_requested_created', 'code_index_jobs', ['requested_by', 'created_at'])
    op.create_index('idx_code_job_status_heartbeat', 'code_index_jobs', ['status', 'last_heartbeat'])


def downgrade() -> None:
    """Drop code_index_jobs table and enum type."""
    conn = op.get_bind()

    # Drop composite indexes first
    op.drop_index('idx_code_job_status_heartbeat', table_name='code_index_jobs')
    op.drop_index('idx_code_job_requested_created', table_name='code_index_jobs')
    op.drop_index('idx_code_job_repo_status', table_name='code_index_jobs')

    # Drop single-column indexes
    op.drop_index('ix_code_index_jobs_last_heartbeat', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_finished_at', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_started_at', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_created_at', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_requested_by', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_status', table_name='code_index_jobs')
    op.drop_index('ix_code_index_jobs_repo_id', table_name='code_index_jobs')

    # Drop the table
    op.drop_table('code_index_jobs')

    # Drop the enum type for PostgreSQL
    if conn.dialect.name == "postgresql":
        op.execute("DROP TYPE IF EXISTS codeindexjobstatus")
