"""
Shared fixtures for store tests.

Provides:
- Mock database sessions
- Test users with different tenant contexts
- Test memories for isolation testing
- PostgreSQL-specific fixtures for RLS testing
"""
import uuid
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import User, Memory, App, MemoryState


# Test user UUIDs for tenant isolation testing
TEST_USER_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_USER_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
TEST_ORG_A_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
TEST_ORG_B_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def test_user_a_id() -> uuid.UUID:
    """Return test user A's UUID."""
    return TEST_USER_A_ID


@pytest.fixture
def test_user_b_id() -> uuid.UUID:
    """Return test user B's UUID."""
    return TEST_USER_B_ID


@pytest.fixture
def test_org_a_id() -> uuid.UUID:
    """Return test org A's UUID."""
    return TEST_ORG_A_ID


@pytest.fixture
def test_org_b_id() -> uuid.UUID:
    """Return test org B's UUID."""
    return TEST_ORG_B_ID


@pytest.fixture
def mock_session() -> MagicMock:
    """
    Create a mock database session for unit tests.

    Configured to work as a context manager and support
    basic session operations.
    """
    session = MagicMock(spec=Session)
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)

    # Track executed statements for verification
    session._executed_statements = []

    def track_execute(stmt, params=None):
        session._executed_statements.append((str(stmt), params))
        return MagicMock()

    session.execute = MagicMock(side_effect=track_execute)

    return session


@pytest.fixture
def mock_engine(mock_session: MagicMock) -> MagicMock:
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    engine.connect.return_value = mock_session
    return engine


@pytest.fixture
def sqlite_test_db() -> Generator[Session, None, None]:
    """
    Create an in-memory SQLite database for unit tests.

    Note: SQLite does not support RLS, so this is only for
    testing store logic that doesn't require RLS.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_user_a(sqlite_test_db: Session) -> User:
    """Create and return test user A in the database."""
    user = User(
        id=TEST_USER_A_ID,
        user_id=str(TEST_USER_A_ID),  # Required string field
        email="user_a@test.com",
        name="Test User A"
    )
    sqlite_test_db.add(user)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(user)
    return user


@pytest.fixture
def sample_user_b(sqlite_test_db: Session) -> User:
    """Create and return test user B in the database."""
    user = User(
        id=TEST_USER_B_ID,
        user_id=str(TEST_USER_B_ID),  # Required string field
        email="user_b@test.com",
        name="Test User B"
    )
    sqlite_test_db.add(user)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(user)
    return user


@pytest.fixture
def sample_app_a(sqlite_test_db: Session, sample_user_a: User) -> App:
    """Create and return a test app owned by user A."""
    app = App(
        id=uuid.uuid4(),
        name="Test App A",
        owner_id=sample_user_a.id
    )
    sqlite_test_db.add(app)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(app)
    return app


@pytest.fixture
def sample_app_b(sqlite_test_db: Session, sample_user_b: User) -> App:
    """Create and return a test app owned by user B."""
    app = App(
        id=uuid.uuid4(),
        name="Test App B",
        owner_id=sample_user_b.id
    )
    sqlite_test_db.add(app)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(app)
    return app


@pytest.fixture
def sample_memory_a(
    sqlite_test_db: Session,
    sample_user_a: User,
    sample_app_a: App
) -> Memory:
    """Create and return a test memory owned by user A."""
    memory = Memory(
        id=uuid.uuid4(),
        user_id=sample_user_a.id,
        app_id=sample_app_a.id,
        content="Test memory content for user A",
        state=MemoryState.active
    )
    sqlite_test_db.add(memory)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(memory)
    return memory


@pytest.fixture
def sample_memory_b(
    sqlite_test_db: Session,
    sample_user_b: User,
    sample_app_b: App
) -> Memory:
    """Create and return a test memory owned by user B."""
    memory = Memory(
        id=uuid.uuid4(),
        user_id=sample_user_b.id,
        app_id=sample_app_b.id,
        content="Test memory content for user B",
        state=MemoryState.active
    )
    sqlite_test_db.add(memory)
    sqlite_test_db.commit()
    sqlite_test_db.refresh(memory)
    return memory


# PostgreSQL-specific fixtures (for RLS integration tests)

@pytest.fixture
def postgres_test_db() -> Generator[Session, None, None]:
    """
    Create a PostgreSQL test database session.

    Requires a running PostgreSQL instance with DATABASE_URL set.
    This fixture is used for RLS integration tests.
    """
    import os

    database_url = os.getenv("DATABASE_URL")
    if not database_url or not database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL not available for RLS tests")

    engine = create_engine(database_url)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def rls_enabled_session(postgres_test_db: Session) -> Session:
    """
    Return a PostgreSQL session with RLS enabled.

    This fixture verifies that RLS is enabled before yielding.
    """
    # Verify RLS is enabled on memories table
    result = postgres_test_db.execute(
        text("""
            SELECT relrowsecurity FROM pg_class
            WHERE relname = 'memories'
        """)
    )
    row = result.fetchone()
    if not row or not row[0]:
        pytest.skip("RLS not enabled on memories table")

    return postgres_test_db
