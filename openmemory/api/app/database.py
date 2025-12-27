import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

# load .env file
load_dotenv()


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
