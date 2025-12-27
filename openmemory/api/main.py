import datetime
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from app.config import DEFAULT_APP_ID, USER_ID
from app.database import Base, SessionLocal, engine
from app.mcp_server import setup_mcp_server
from app.axis_guidance_server import setup_axis_guidance_server
from app.models import App, User
from app.routers import apps_router, backup_router, config_router, entities_router, graph_router, health_router, memories_router, stats_router
from app.security.middleware import SecurityHeadersMiddleware
from app.security.exception_handlers import register_security_exception_handlers
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Validate settings
    try:
        from app.settings import get_settings
        settings = get_settings()
        logger.info("Settings validated successfully")
        logger.info(f"JWT issuer: {settings.jwt_issuer}")
        logger.info(f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    except Exception as e:
        logger.critical(f"Settings validation failed: {e}")
        # In production, we'd want to fail fast
        # For now, log the error but continue for backward compatibility
        logger.warning("Continuing with environment-based configuration")

    yield  # Application runs

    # Shutdown: cleanup if needed
    logger.info("Application shutting down")


app = FastAPI(title="OpenMemory API", lifespan=lifespan)

# Security headers middleware (applied to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware - restrict origins in production
allowed_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register security exception handlers (401/403 responses)
register_security_exception_handlers(app)

# Create all tables
Base.metadata.create_all(bind=engine)

# Check for USER_ID and create default user if needed
def create_default_user():
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            # Create default user
            user = User(
                id=uuid4(),
                user_id=USER_ID,
                name="Default User",
                created_at=datetime.datetime.now(datetime.UTC)
            )
            db.add(user)
            db.commit()
    finally:
        db.close()


def create_default_app():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            return

        # Check if app already exists
        existing_app = db.query(App).filter(
            App.name == DEFAULT_APP_ID,
            App.owner_id == user.id
        ).first()

        if existing_app:
            return

        app = App(
            id=uuid4(),
            name=DEFAULT_APP_ID,
            owner_id=user.id,
            created_at=datetime.datetime.now(datetime.UTC),
            updated_at=datetime.datetime.now(datetime.UTC),
        )
        db.add(app)
        db.commit()
    finally:
        db.close()

# Create default user on startup
create_default_user()
create_default_app()

# Setup MCP servers
setup_mcp_server(app)
setup_axis_guidance_server(app)

# Include routers
app.include_router(health_router)  # Health checks first for quick probe responses
app.include_router(memories_router)
app.include_router(apps_router)
app.include_router(stats_router)
app.include_router(config_router)
app.include_router(backup_router)
app.include_router(entities_router)
app.include_router(graph_router)

# Add pagination support
add_pagination(app)
