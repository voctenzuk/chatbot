# Telegram AI Companion Bot

Russian-language AI companion chatbot built with **aiogram 3** (async, long-polling), **LangChain** for LLM orchestration, and **mem0** for long-term memory with automatic fact extraction. Package manager: **uv**.

## Features

- Conversational AI persona with consistent personality (Russian language)
- **Short-term memory** — sliding window of recent messages per episode
- **Long-term memory** — mem0 with automatic fact extraction, dedup, and conflict resolution
- **Automatic episode management** — new episodes on 8h time gap or topic shift (cosine similarity < 0.7)
- **Episode summarization** — running/chunk/final summaries for context compression
- **Image generation** — LLM decides when to send photos via tool calling (OpenAI Images API)
- **Proactive messaging** — bot initiates conversations based on context and schedule
- **Monetization** — Telegram Stars payments, subscription tiers, usage tracking
- **Graceful degradation** — mem0, Supabase, Redis are all optional; bot works without them

## Quick start

```bash
# 1. Clone and install
git clone <repo-url> && cd chatbot
uv sync

# 2. Configure
cp .env.example .env  # or create .env manually (see Environment below)

# 3. Run
uv run bot
```

## Environment

All configuration via `.env` file (loaded by pydantic-settings).

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Yes** | — | Bot crashes at startup without it |
| `LLM_BASE_URL` | No | OpenAI | LLM provider endpoint |
| `LLM_API_KEY` | No | — | API key for LLM provider |
| `LLM_MODEL` | No | `kimi-k2p5` | Model name |
| `LLM_TEMPERATURE` | No | `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | No | `1024` | Max completion tokens |
| `MEM0_SUPABASE_CONNECTION_STRING` | No | — | Supabase pgvector connection for mem0 |
| `EMBEDDER_MODEL` | No | `text-embedding-3-small` | Embedding model for mem0 |
| `REDIS_URL` | No | — | Redis for rate-limiting; falls back gracefully |
| `IMAGE_BASE_URL` | No | OpenAI | Image generation API endpoint |
| `IMAGE_API_KEY` | No | — | Image generation API key |
| `IMAGE_MODEL` | No | `gpt-image-1` | Image model name |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse observability |
| `LANGFUSE_BASE_URL` | No | `https://cloud.langfuse.com` | Langfuse host |
| `SUPABASE_URL` | No | — | Supabase project URL (for DB persistence) |
| `SUPABASE_SERVICE_KEY` | No | — | Supabase service role key |

## Development

```bash
uv run bot                    # run the bot
uv sync                       # install/sync dependencies
uvx ruff format .              # format code
uvx ruff check .               # lint
uv run pyright                 # type check
uv run pytest                  # run all tests
```

CI gate (all must pass): `ruff format --check .` → `ruff check .` → `pyright` → `pytest`

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation, service table, and critical rules.

```
src/bot/
    __main__.py          # Entry point
    app.py               # asyncio bootstrap, Dispatcher setup
    config.py            # Pydantic Settings
    wiring.py            # Composition root: AppContext + build_app_context()
    chat_pipeline.py     # Framework-agnostic chat orchestration
    handlers.py          # Thin aiogram handlers (receive deps via kwargs)
    ports.py             # Protocol definitions (LLMPort, MemoryPort, MessageDeliveryPort)
    conversation/        # Episode management, context building, summarization
    memory/              # mem0 service, memory models, cleanup
    llm/                 # LLM service wrapper
    media/               # Image generation, artifacts, storage
    infra/               # Database client, Langfuse observability
    adapters/            # Telegram delivery, proactive scheduler
```

## Notes

- Per-user isolation via `telegram_user_id` across all services
- All mem0 operations use `user_id=f"tg_user_{user_id}"` for data isolation
- All service methods are async; LangChain uses `.ainvoke()` only
