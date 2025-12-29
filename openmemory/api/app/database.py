import logging
import os
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Union

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

# load .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Alembic imports (lazy loaded to avoid circular imports)
try:
    from alembic import command as alembic_command
    from alembic.config import Config as alembic_Config
    ALEMBIC_AVAILABLE = True
except ImportError:
    alembic_command = None
    alembic_Config = None
    ALEMBIC_AVAILABLE = False


def get_database_url() -> str:
    """
    Get database URL from Settings or environment.

    Priority:
    1. Settings.database_url (PostgreSQL from Pydantic Settings)
    2. DATABASE_URL environment variable (legacy support)
    3. SQLite fallback for development only
    """
    # Try to use Settings (preferred)
    try:
        from app.settings import get_settings
        settings = get_settings()
        return settings.database_url
    except Exception:
        pass

    # Legacy fallback: DATABASE_URL environment variable
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Development fallback only - should not be used in production
    return "sqlite:///./openmemory.db"


def create_db_engine(database_url: str):
    """
    Create SQLAlchemy engine with appropriate settings for the database type.
    """
    if database_url.startswith("sqlite"):
        # SQLite requires check_same_thread=False for FastAPI
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False}
        )
    else:
        # PostgreSQL with connection pooling
        return create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
        )


# Create engine with appropriate configuration
DATABASE_URL = get_database_url()
engine = create_db_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def is_postgres_database(database_url: Optional[str] = None) -> bool:
    """
    Check if the database URL is for PostgreSQL.

    Args:
        database_url: Database URL to check. If None, uses the current DATABASE_URL.

    Returns:
        True if PostgreSQL, False otherwise (SQLite, etc.)
    """
    if database_url is None:
        database_url = DATABASE_URL

    if not database_url:
        return False

    return database_url.startswith("postgresql")


def should_auto_migrate() -> bool:
    """
    Check if AUTO_MIGRATE is enabled via environment variable.

    Returns:
        True if AUTO_MIGRATE is set to a truthy value.
    """
    auto_migrate = os.getenv("AUTO_MIGRATE", "false").lower()
    return auto_migrate in ("true", "1", "yes")


def run_alembic_upgrade() -> None:
    """
    Run Alembic migrations to upgrade database to head.

    This function is safe to call - it logs errors without crashing
    the application.
    """
    if not ALEMBIC_AVAILABLE:
        logger.warning("Alembic is not available - skipping migrations")
        return

    try:
        # Find alembic.ini relative to this file
        # This file is at: openmemory/api/app/database.py
        # alembic.ini is at: openmemory/api/alembic.ini
        current_dir = Path(__file__).resolve().parent.parent
        alembic_ini_path = current_dir / "alembic.ini"

        if not alembic_ini_path.exists():
            logger.error(f"alembic.ini not found at {alembic_ini_path}")
            return

        logger.info(f"Running Alembic migrations from {alembic_ini_path}")
        config = alembic_Config(str(alembic_ini_path))

        # Set the script location relative to the config file
        config.set_main_option("script_location", str(current_dir / "alembic"))

        alembic_command.upgrade(config, "head")
        logger.info("Alembic migrations completed successfully")

    except Exception as e:
        logger.error(f"Alembic migration failed: {e}")


def auto_migrate_on_startup() -> None:
    """
    Run database migrations on startup if AUTO_MIGRATE is enabled.

    This function:
    - Only runs if AUTO_MIGRATE=true
    - Only runs for PostgreSQL databases (not SQLite)
    - Logs the action and any errors
    """
    if not should_auto_migrate():
        logger.debug("AUTO_MIGRATE is disabled - skipping migrations")
        return

    if not is_postgres_database():
        logger.info("AUTO_MIGRATE skipped: SQLite does not support Alembic migrations")
        return

    logger.info("AUTO_MIGRATE enabled - running database migrations")
    run_alembic_upgrade()


@contextmanager
def tenant_session(
    db: Session,
    user_id: Union[uuid.UUID, str]
) -> Generator[Session, None, None]:
    """
    Context manager that sets PostgreSQL session variable for RLS.

    Sets the 'app.current_user_id' session variable used by Row Level Security
    policies to filter data by tenant. The variable is guaranteed to be reset
    on exit, even if an exception occurs.

    Args:
        db: SQLAlchemy session to use
        user_id: UUID of the current user (tenant identifier)

    Yields:
        The same session with tenant context set

    Raises:
        ValueError: If user_id is not a valid UUID
        TypeError: If user_id is None or empty

    Example:
        with tenant_session(db, principal.user_id) as session:
            memories = session.query(Memory).all()  # RLS filters automatically
    """
    # Validate user_id
    if user_id is None:
        raise TypeError("user_id cannot be None")

    if isinstance(user_id, str):
        if not user_id.strip():
            raise ValueError("user_id cannot be empty")
        try:
            user_id = uuid.UUID(user_id)
        except ValueError as e:
            raise ValueError(f"Invalid UUID format: {user_id}") from e

    if not isinstance(user_id, uuid.UUID):
        raise TypeError(f"user_id must be UUID or string, got {type(user_id)}")

    try:
        # Set the PostgreSQL session variable for RLS
        db.execute(
            text("SET app.current_user_id = :user_id"),
            {"user_id": str(user_id)}
        )
        yield db
    finally:
        # Always reset the session variable, even on exception
        db.execute(text("RESET app.current_user_id"))
