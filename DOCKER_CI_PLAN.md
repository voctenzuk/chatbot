# Docker Compose + CI Plan

This document describes the containerization and CI/CD setup for the Telegram Bot.

## Overview

| Service | Purpose | External/Internal |
|---------|---------|-------------------|
| **bot** | Telegram bot application | Docker |
| **redis** | Session state, rate limiting | Docker |
| **qdrant** | (Optional) Local vector database | Docker |
| **supabase** | User data, payments | **External** |
| **mem0** | Vector memory API | **External** (cloud) |

---

## Files Added/Modified

### 1. `Dockerfile` (NEW)

Multi-stage build for optimized image size:
- **Builder stage**: Installs compile dependencies, builds Python wheels
- **Runtime stage**: Minimal image with only runtime dependencies
- Uses non-root user (`botuser`) for security
- Health check included

**Key features:**
- Python 3.12 slim base
- Virtual environment for isolation
- `PYTHONPATH=/app/src` for src/ layout
- Multi-stage reduces final image size (~150MB vs ~500MB)

### 2. `docker-compose.yml` (UPDATED)

Services defined:

```yaml
services:
  redis:        # Session state, caching
  qdrant:       # (Optional) Vector DB - use --profile qdrant to enable
  bot:          # Main application
```

**Usage:**
```bash
# Start bot + redis (default)
docker compose up -d

# Start with Qdrant
docker compose --profile qdrant up -d

# View logs
docker compose logs -f bot

# Restart after code changes
docker compose build bot && docker compose up -d bot
```

**Networking:**
- All services on `bot_network` bridge network
- Redis accessible at `redis://redis:6379/0` from bot
- Qdrant accessible at `http://qdrant:6333` from bot

### 3. `.env.example` (UPDATED)

Added sections:
- Local Qdrant configuration
- Development/testing flags (`MOCK_SERVICES`, `LOG_LEVEL`)
- Better organization with headers
- Clearer comments about Docker vs local URLs

**Required for Docker:**
```bash
# Copy and edit
cp .env.example .env
# Edit .env with your actual API keys
```

### 4. `.github/workflows/ci.yml` (UPDATED)

**Changes:**
- Added Redis service container for integration tests
- Added Docker build smoke test (PRs only)
- Added coverage artifact upload
- Better caching with pip cache
- Environment variables for test isolation

**CI Pipeline:**
```
1. Checkout code
2. Setup Python 3.12 + pip cache
3. Install package: pip install -e ".[dev,images]"
4. Run ruff format check
5. Run ruff lint check
6. Run pyright type check
7. Run pytest with coverage
8. Upload coverage report
9. (PR only) Build Docker image
```

---

## LangChain/LangGraph Runtime Notes

**Current Status:**
- Dependencies are listed in `pyproject.toml` but **not actively used** in source code
- The bot currently uses direct `httpx` calls to OpenRouter API
- `mem0` is conditionally imported (graceful fallback if not installed)

**Runtime Requirements:**
- No special runtime needs for LangChain/LangGraph
- Standard Python asyncio event loop is sufficient
- If LangGraph is used in future, ensure `langgraph` package is installed (already in deps)

**MCP Docs Confirmation:**
- LangChain core requires only standard Python packages
- LangGraph requires `langgraph` + `langchain-core` (both in deps)
- No additional system dependencies needed

---

## Local Development Workflow

### Option 1: Pure Docker (Recommended for consistency)

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker compose up -d

# 3. View logs
docker compose logs -f bot

# 4. Stop
docker compose down
```

### Option 2: Hybrid (Local Python + Docker Redis)

```bash
# 1. Start only Redis
docker compose up -d redis

# 2. Setup local Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,images]"

# 3. Run bot locally (uses localhost:6379)
python -m bot
```

### Option 3: Full Local (no Docker)

```bash
# 1. Install Redis locally (e.g., brew install redis)
redis-server

# 2. Setup Python
pip install -e ".[dev,images]"

# 3. Run
python -m bot
```

---

## CI Blockers and Notes

### Current Status: ✅ CI Ready

The CI workflow should work without blockers. Key points:

1. **Build System**: ✅ `setuptools` configured in `pyproject.toml`
2. **Editable Install**: ✅ `pip install -e ".[dev]"` works
3. **Tests**: ✅ Pass with mock services (`MOCK_SERVICES=true`)
4. **Type Checking**: ✅ Pyright configured with venv path
5. **Linting**: ✅ Ruff configured

### Potential Issues and Solutions

| Issue | Solution |
|-------|----------|
| Pyright import errors for optional deps | Add `# type: ignore[import]` or install all extras |
| Missing env vars in CI | Use `env:` section or GitHub Secrets |
| Redis connection in tests | Use mock or Redis service container |
| Slow Docker builds | Enable BuildKit cache (`cache-from: type=gha`) |

---

## Security Considerations

1. **Non-root container**: Bot runs as `botuser` (not root)
2. **No secrets in image**: All config via env vars / `.env` file
3. **Health checks**: Both Redis and bot have health checks
4. **Resource limits**: Memory/CPU limits set in compose

---

## Production Deployment Tips

1. **Use specific image tags** instead of `latest`
2. **Enable Watchtower** (commented in compose) for auto-updates
3. **Use Docker secrets** or external secret management
4. **Enable Redis persistence** (already configured with `appendonly yes`)
5. **Set up log aggregation** (bot logs to stdout/stderr)
6. **Monitor with Prometheus/Grafana** if needed

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Dockerfile | ✅ Added | Multi-stage, secure, optimized |
| Docker Compose | ✅ Updated | Bot + Redis + optional Qdrant |
| .env.example | ✅ Updated | Complete with all options |
| CI Workflow | ✅ Updated | Lint, type-check, test, build |
| LangChain deps | ✅ Verified | Listed but not actively used |
| CI Blockers | ✅ None | Ready to run |
