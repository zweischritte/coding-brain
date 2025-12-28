# Production-ready Dockerfile for coding-brain MCP API
# Phase 7: Security Hardening Applied
#
# Security features:
# - Non-root user (appuser:1000)
# - Minimal base image (python:3.12-slim)
# - Proper file ownership with COPY --chown
# - No secrets in image
# - Clean pip cache

FROM python:3.12-slim

LABEL org.opencontainers.image.name="codingbrain/mcp"
LABEL org.opencontainers.image.description="Coding Brain MCP API Server"
LABEL org.opencontainers.image.vendor="Coding Brain"

# Create non-root user and group
# Using UID/GID 1000 which is common convention for app users
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /usr/src/openmemory

# Install dependencies first (better layer caching)
# Install as root, then change ownership
COPY --chown=appuser:appgroup requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with proper ownership
COPY --chown=appuser:appgroup config.json .
COPY --chown=appuser:appgroup . .

# Remove any test files from production image (optional, controlled by .dockerignore)
# RUN rm -rf tests/ __pycache__/ .pytest_cache/

# Ensure data directories exist and are owned by appuser
RUN mkdir -p /data && chown appuser:appgroup /data

# Switch to non-root user before running the application
# This is critical for security - running as root is a vulnerability
USER appuser

# Expose the application port
EXPOSE 8765

# Health check (optional, also configured in docker-compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health/live')" || exit 1

# Run the application
# Using exec form for proper signal handling
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
