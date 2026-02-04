# syntax=docker/dockerfile:1
# Multi-stage build for Python Telegram Bot

# -----------------------------------------------------------------------------
# Stage 1: Builder (installs dependencies)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install package with dependencies (editable not needed for runtime)
# Use pip cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install ".[images]"

# -----------------------------------------------------------------------------
# Stage 2: Runtime (minimal image)
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/

# Set Python path for src/ layout
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER botuser

# Health check (adjust if bot exposes HTTP endpoint, otherwise remove)
# For polling bots, we can use a simple Python check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import bot; print('OK')" || exit 1

# Run the bot
CMD ["python", "-m", "bot"]
