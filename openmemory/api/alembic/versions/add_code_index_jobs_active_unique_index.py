"""add_code_index_jobs_active_unique_index

Revision ID: add_code_index_jobs_active_unique_index
Revises: add_code_index_jobs_table
Create Date: 2025-12-30
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_code_index_jobs_active_unique_index"
down_revision: Union[str, None] = "add_code_index_jobs_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add partial unique index for active jobs per repo (PostgreSQL only)."""
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_code_index_jobs_repo_active "
        "ON code_index_jobs (repo_id) "
        "WHERE status IN ('queued', 'running') AND cancel_requested = false"
    )


def downgrade() -> None:
    """Drop partial unique index for active jobs per repo."""
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS idx_code_index_jobs_repo_active")
