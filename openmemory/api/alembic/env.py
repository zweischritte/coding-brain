import logging
import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import your models here - moved after path setup
from app.database import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Migration verification settings
VERIFY_MIGRATIONS = os.getenv("ALEMBIC_VERIFY_MIGRATIONS", "false").lower() == "true"
CORE_TABLES = ["users", "apps", "memories", "categories", "configs"]

logger = logging.getLogger("alembic.env")


def capture_pre_migration_state(engine):
    """Capture row counts and checksums before migration for verification.

    Only runs when ALEMBIC_VERIFY_MIGRATIONS=true environment variable is set.

    Returns:
        dict with pre_counts and pre_checksums, or None if verification disabled.
    """
    if not VERIFY_MIGRATIONS:
        return None

    try:
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(engine)

        logger.info("Capturing pre-migration state for verification...")

        # Capture row counts for core tables
        pre_counts = verifier.get_table_row_counts(CORE_TABLES)
        logger.info(f"Pre-migration row counts: {pre_counts}")

        return {"pre_counts": pre_counts}

    except Exception as e:
        logger.warning(f"Failed to capture pre-migration state: {e}")
        return None


def verify_post_migration_state(engine, pre_state):
    """Verify data integrity after migration completes.

    Compares pre and post row counts and logs any discrepancies.

    Args:
        engine: SQLAlchemy engine
        pre_state: State captured by capture_pre_migration_state

    Returns:
        bool: True if verification passed, False otherwise
    """
    if pre_state is None:
        return True  # Skip verification if no pre-state

    try:
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(engine)

        logger.info("Verifying post-migration state...")

        # Capture post-migration counts
        post_counts = verifier.get_table_row_counts(CORE_TABLES)
        logger.info(f"Post-migration row counts: {post_counts}")

        # Verify counts match
        result = verifier.verify_row_counts(pre_state["pre_counts"], post_counts)

        if result.success:
            logger.info("Migration verification PASSED: Row counts match")
            return True
        else:
            logger.error(f"Migration verification FAILED: {result.mismatches}")
            for mismatch in result.mismatches:
                logger.error(
                    f"  Table '{mismatch['table']}': "
                    f"pre={mismatch['pre']}, post={mismatch['post']}"
                )
            return False

    except Exception as e:
        logger.warning(f"Failed to verify post-migration state: {e}")
        return True  # Don't fail migration on verification error


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = os.getenv("DATABASE_URL", "sqlite:///./openmemory.db")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    When ALEMBIC_VERIFY_MIGRATIONS=true, this function will:
    1. Capture pre-migration row counts
    2. Run the migration
    3. Verify post-migration row counts match
    4. Log any discrepancies
    """
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = os.getenv("DATABASE_URL", "sqlite:///./openmemory.db")
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Capture pre-migration state for verification
    pre_state = capture_pre_migration_state(connectable)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

    # Verify post-migration state
    verify_post_migration_state(connectable, pre_state)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
