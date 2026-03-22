# CLAUDE.md

Telegram AI companion chatbot: aiogram 3 (async, long-polling), LangChain for LLM orchestration, mem0 for long-term memory. Russian-language bot personality. Package manager: **uv**.

## Commands

```bash
uv run bot                                    # run the bot
uv sync                                       # install/sync dependencies
uvx ruff check --fix . && uvx ruff format .   # lint + format
uvx pyright                                   # type check
uv run pytest                                 # run all tests
uv run pytest tests/test_foo.py::test_bar -v  # run a single test
```

CI gate (all must pass): `uvx ruff format --check .` → `uvx ruff check .` → `uvx pyright` → `uv run pytest`

## Architecture

**Entry point:** `src/bot/__main__.py` → `app.run()` → `Dispatcher.start_polling()`

**Configuration:** `src/bot/config.py` — `Settings(BaseSettings)`, reads `.env`. Import as `from bot.config import settings`.

**Character:** `src/bot/character.py` — `CharacterConfig` frozen dataclass + `DEFAULT_CHARACTER`. Personality (Russian), appearance (English for image models), voice style, greeting, few-shot examples, `SPRITE_EMOTIONS` tuple. Loaded in `wiring.py`, passed to `ChatPipeline` + `ImageService`.

**Request flow** (per message in `handlers.py`):
1. `EpisodeManager.process_user_message()` — persist + auto-switch episodes (8h gap or topic shift)
2. `check_rate_limit()` — daily message limit per subscription tier (fail-open)
3. `Mem0MemoryService.search()` — vector similarity search over extracted facts (best-effort)
4. `get_system_prompt(character=...)` — build prompt from `CharacterConfig` (personality + voice + examples)
5. `ContextBuilder.assemble_for_llm()` — merge history + semantic memories + system prompt
6. `LLMService.generate()` — LLM call via LangChain, returns `LLMResponse`
7. Tool calls: `send_photo` → `try_consume_photo()` (atomic DB check) → `ImageService.generate()` via OpenRouter `chat.completions.create(modalities=["image"])`; `send_sprite` → `ImageService.get_sprite()` (cached emotion sprites from Supabase Storage)
8. `increment_usage(cost_cents=...)` — track tokens + cost per message/photo
9. Response persisted to episode, sent to user

**Commands:** `/start` (onboarding: new user greeting + photo vs returning user welcome), `/upgrade` (Telegram Stars), `/stats` (daily usage dashboard).

**Wiring:** Composition root in `src/bot/wiring.py`. `build_app_context()` → `AppContext` dataclass. `ChatPipeline` receives deps via constructor. Handlers get `pipeline` + `db_client` via `dp.workflow_data`.

**Graceful degradation:** mem0, Supabase, Redis are optional — bot falls back to in-memory operation. Photo generation is fail-closed (no DB = no photos).

## Critical Rules

<important if="writing async code or service methods">
All service methods MUST be async. Use `ainvoke()` / `astream()` for LangChain, never sync `.invoke()`.
</important>

<important if="writing handlers or touching handlers.py">
Catch-all `@router.message()` with NO filter MUST be registered LAST. Insert new handlers ABOVE it. Always try/except with Russian fallback text.
</important>

<important if="working with mem0 or memory">
All mem0 operations MUST use `user_id=f"tg_user_{user_id}"` for per-user data isolation. Memory writes go through `Mem0MemoryService.write_factual()` which auto-extracts facts via LLM.
</important>

<important if="writing imports from LangChain">
Split packages only: `langchain_core`, `langchain_openai`. Never `from langchain.` or `from langchain.schema`.
</important>

<important if="writing type annotations">
Use `str | None` not `Optional[str]`. Never blanket `# type: ignore` — use `# pyright: ignore[reportSpecificRule]` with comment.
</important>

## Code Style

- Python 3.13+, type hints on public functions, 100-char lines
- Async-first: all service methods are `async`
- Logging: `loguru.logger` with `{}` placeholders
- Commits: present tense, prefixed (`feat:`, `fix:`, `chore:`)

Detailed rules: `.claude/rules/python-style.md` (ruff, pyright, imports, patterns)

## Architecture (target)

Domain packages: `bot.llm`, `bot.memory`, `bot.conversation`, `bot.media`, `bot.infra`, `bot.adapters.telegram`
Protocols: `LLMPort`, `MemoryPort`, `MessageDeliveryPort` — only where swapping is realistic.

## Monetization

Two tiers (Free/Plus), Pro seeded but YAGNI for MVP. Photo limits: Free=3/day, Plus=10/day.
Cost tracking: `COST_PER_1M_INPUT=0.15`, `COST_PER_1M_OUTPUT=0.60` in `chat_pipeline.py`. Image cost via `ImageResult.cost_cents` (fallback `DEFAULT_IMAGE_COST_CENTS=4.0` in `image_service.py`).
Photo rate limit: atomic `try_consume_photo` RPC (race-safe, plan-tier-aware).

## Branching

`main` — production. Feature: `feature/<name>`, bugfix: `fix/<name>`. Squash merge preferred.

## Domain Rules

Loaded via `.claude/rules/` with path-based globs (only when touching relevant files):
- `testing.md` — pytest conventions, fixtures, mocking, markers (`tests/**`)
- `python-style.md` — ruff, pyright, imports, async patterns (`**/*.py`)
- `aiogram.md` — handler conventions, filters, FSM, middleware (`handlers.py`, `app.py`)
- `langchain.md` — imports, async invocation, LLMService pattern (`bot.llm/**`)
- `memory-system.md` — mem0, episodes, context builder (`bot.memory/**`, `bot.conversation/**`)
