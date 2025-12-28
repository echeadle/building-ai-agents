# Dockerfile for AI Agent API
# 
# Chapter 40: Deployment Patterns
#
# Build:   docker build -t agent-api .
# Run:     docker run -p 8000:8000 --env-file .env agent-api
# 
# For development with hot-reload:
#   docker run -p 8000:8000 -v $(pwd):/app --env-file .env agent-api

# =============================================================================
# Stage 1: Base image with Python
# =============================================================================
FROM python:3.11-slim AS base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS dependencies

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM dependencies AS production

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Health check - verify the service is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

# Default command - production settings
CMD ["uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 4: Development image (optional, for hot-reload)
# =============================================================================
FROM dependencies AS development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio httpx

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Development command with hot-reload
CMD ["uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
