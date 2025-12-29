import datetime
import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

from app.config import DEFAULT_APP_ID, USER_ID
from app.database import Base, SessionLocal, auto_migrate_on_startup, engine
from app.mcp_server import setup_mcp_server
from app.guidance_server import setup_guidance_server
from app.models import App, User
from app.routers import apps_router, backup_router, code_router, config_router, entities_router, experiments_router, feedback_router, gdpr_router, graph_router, health_router, memories_router, search_router, stats_router
from app.security.middleware import SecurityHeadersMiddleware
from app.security.exception_handlers import register_security_exception_handlers
from app.observability.metrics import MetricsMiddleware
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

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

# Metrics middleware for Prometheus (request count, duration)
app.add_middleware(MetricsMiddleware)

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

# Run auto-migration if enabled (PostgreSQL only)
auto_migrate_on_startup()

# Create all tables (fallback for SQLite, no-op if tables already exist from migrations)
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
setup_guidance_server(app)


# Prometheus metrics endpoint (no auth required)
@app.get("/metrics", tags=["observability"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    No authentication required.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


# Include routers
app.include_router(health_router)  # Health checks first for quick probe responses
app.include_router(memories_router)
app.include_router(apps_router)
app.include_router(stats_router)
app.include_router(config_router)
app.include_router(backup_router)
app.include_router(entities_router)
app.include_router(graph_router)
app.include_router(feedback_router)
app.include_router(experiments_router)
app.include_router(search_router)
app.include_router(gdpr_router)
app.include_router(code_router)  # Code intelligence endpoints

# Add pagination support
add_pagination(app)
