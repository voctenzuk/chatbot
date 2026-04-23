# Telegram AI Companion Bot

Russian-language AI companion chatbot built with **aiogram 3** (async, long-polling), **LangChain** for LLM orchestration, and **mem0** for long-term memory with automatic fact extraction. Package manager: **uv**.

## Features

- **Character system** — defined personality, voice style, and few-shot examples via `CharacterConfig`
- **Consistent appearance** — image generation prepends character description for visual consistency across photos
- **Short-term memory** — sliding window of recent messages per episode
- **Long-term memory** — mem0 with automatic fact extraction, dedup, and conflict resolution
- **Automatic episode management** — new episodes on 8h time gap or topic shift (cosine similarity < 0.7)
- **Episode summarization** — running/chunk/final summaries for context compression
- **Image generation** — LLM decides when to send photos via tool calling (OpenRouter, `chat.completions.create` with image modality)
- **Sprite system** — pre-generated emotion sprites with Supabase Storage caching
- **Proactive messaging** — bot initiates conversations based on context and schedule
- **Monetization** — Telegram Stars payments, tiered subscriptions (Free/Plus), per-message cost tracking
- **Photo paywall** — daily photo limits per tier with atomic DB-based rate limiting
- **Onboarding** — new user detection, character greeting + free photo on first `/start`
- **Usage dashboard** — `/stats` command shows messages, photos, plan, days together
- **Graceful degradation** — mem0, Supabase, Redis are all optional; bot works without them

## Quick start

```bash
# 1. Clone and install
git clone <repo-url> && cd chatbot
uv sync

# 2. Configure
cp .env.example .env   # edit with your API keys (see Environment below)

# 3. (Optional) Import a character card from SillyTavern
uv run python -m bot.tools.import_card path/to/card.png
# → copy output into src/bot/character.py (see docs/CHARACTER_SETUP.md)

# 4. Run
uv run bot
```

## Launch checklist

Минимум для запуска:

- [ ] `TELEGRAM_BOT_TOKEN` — от [@BotFather](https://t.me/BotFather)
- [ ] `LLM_API_KEY` — ключ [OpenRouter](https://openrouter.ai/) (или другого OpenAI-совместимого провайдера)
- [ ] `uv sync` — установить зависимости
- [ ] `uv run bot` — бот запускается и отвечает на `/start`

Для полного функционала:

- [ ] `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` — для сохранения сообщений, подписок, usage tracking
- [ ] `IMAGE_API_KEY` — для генерации фото (OpenRouter, модель SeeDream 4.5)
- [ ] `MEM0_API_KEY` — для долгосрочной памяти (бот запоминает факты о пользователе)
- [ ] Применить миграции в Supabase (`supabase_migrations/001..010`)
- [ ] Настроить персонажа — см. [docs/CHARACTER_SETUP.md](docs/CHARACTER_SETUP.md)

## Environment

All configuration via `.env` file (loaded by pydantic-settings).

| Variable | Required | Default | Notes |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | **Yes** | — | Bot crashes at startup without it |
| `LLM_BASE_URL` | No | OpenAI | LLM provider endpoint |
| `LLM_API_KEY` | **Yes** | — | API key for LLM provider (bot exits at startup without it) |
| `LLM_MODEL` | No | `kimi-k2p5` | Model name |
| `LLM_TEMPERATURE` | No | `0.7` | Generation temperature |
| `LLM_MAX_TOKENS` | No | `1024` | Max completion tokens |
| `MEM0_SUPABASE_CONNECTION_STRING` | No | — | Supabase pgvector connection for mem0 |
| `EMBEDDER_MODEL` | No | `text-embedding-3-small` | Embedding model for mem0 |
| `REDIS_URL` | No | — | Redis for rate-limiting; falls back gracefully |
| `IMAGE_BASE_URL` | No | OpenAI | Image generation API endpoint |
| `IMAGE_API_KEY` | No | — | Image generation API key |
| `IMAGE_MODEL` | No | `bytedance/seedream-4.5` | Image model name (OpenRouter) |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse observability |
| `LANGFUSE_BASE_URL` | No | `https://cloud.langfuse.com` | Langfuse host |
| `LANGFUSE_ENABLED` | No | `true` | Set to `false` to disable Langfuse init |
| `SUPABASE_URL` | No | — | Supabase project URL (for DB persistence) |
| `SUPABASE_SERVICE_KEY` | No | — | Supabase service role key |

## Deployment

```bash
# Docker (local or VPS)
docker compose up -d --build                    # bot only
docker compose --profile with-redis up -d       # bot + Redis
```

CI/CD: GitHub Actions CI (`.github/workflows/ci.yml`) runs lint + type-check + tests on push/PR to `main`. On CI success, `.github/workflows/deploy.yml` SSHs to VPS and redeploys via `docker compose`. Supabase migrations (`supabase_migrations/`) must be applied manually before deploy.

## Development

```bash
uv run bot                    # run the bot
uv sync                       # install/sync dependencies
uvx ruff format .              # format code
uvx ruff check .               # lint
uvx pyright                    # type check
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
    character.py         # CharacterConfig dataclass + DEFAULT_CHARACTER
    wiring.py            # Composition root: AppContext + build_app_context()
    chat_pipeline.py     # Framework-agnostic chat orchestration
    handlers.py          # Thin aiogram handlers (/start, /upgrade, /stats, chat)
    ports.py             # Protocol definitions (LLMPort, MemoryPort, MessageDeliveryPort)
    conversation/        # Episode management, context building, system prompt
    memory/              # mem0 service, memory models, cleanup
    llm/                 # LLM service wrapper
    media/               # Image generation (with appearance prefix), artifacts, storage
    infra/               # Database client, Langfuse observability
    adapters/            # Telegram delivery, proactive scheduler
    tools/               # ST Card V2 parser (import_card.py)
```

## Character setup

Бот использует `CharacterConfig` dataclass для определения личности, внешности и стиля общения. По умолчанию — "Алиса", 24 года, дизайнер из Москвы.

Можно изменить персонажа двумя способами:
1. **Напрямую** — редактировать `DEFAULT_CHARACTER` в `src/bot/character.py`
2. **Импорт ST Card** — скачать карту с [CharaVault](https://charavault.net/), [AI Character Cards](https://aicharactercards.com/) или [CharacterCard.com](https://charactercard.com/download) и импортировать:
   ```bash
   uv run python -m bot.tools.import_card path/to/card.png
   ```

Подробнее: **[docs/CHARACTER_SETUP.md](docs/CHARACTER_SETUP.md)** — маппинг полей, визуальная идентичность, спрайты.

## Notes

- Per-user isolation via `telegram_user_id` across all services
- All mem0 operations use `user_id=f"tg_user_{user_id}"` for data isolation
- All service methods are async; LangChain uses `.ainvoke()` only
