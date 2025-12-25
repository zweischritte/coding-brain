"""Add AXIS 3.4 indexed fields to memories table

Revision ID: axis_34_fields
Revises: afd00efbd06b
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'axis_34_fields'
down_revision = 'afd00efbd06b'
branch_labels = None
depends_on = None


def upgrade():
    """Add indexed columns for AXIS 3.4 filtering."""
    # Add vault column with index
    op.add_column('memories', sa.Column('vault', sa.String(50), nullable=True))
    op.create_index('ix_memories_vault', 'memories', ['vault'])

    # Add layer column with index
    op.add_column('memories', sa.Column('layer', sa.String(20), nullable=True))
    op.create_index('ix_memories_layer', 'memories', ['layer'])

    # Add axis_vector column with index (say/want/do)
    op.add_column('memories', sa.Column('axis_vector', sa.String(10), nullable=True))
    op.create_index('ix_memories_axis_vector', 'memories', ['axis_vector'])


def downgrade():
    """Remove AXIS 3.4 columns."""
    op.drop_index('ix_memories_axis_vector', 'memories')
    op.drop_column('memories', 'axis_vector')

    op.drop_index('ix_memories_layer', 'memories')
    op.drop_column('memories', 'layer')

    op.drop_index('ix_memories_vault', 'memories')
    op.drop_column('memories', 'vault')
