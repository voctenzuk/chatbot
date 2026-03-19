# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Telegram AI companion chatbot built with aiogram 3 (async, long-polling), LangChain for LLM orchestration, and Cognee for knowledge-graph memory. Bot personality is Russian-language. Package manager is **uv**.

## Commands

```bash
uv run bot                                    # run the bot
uv sync                                       # install/sync dependencies
uvx ruff format .                             # format code
uvx ruff check .                              # lint
uv run pyright                                # type check
uv run pytest                                 # run all tests
uv run pytest tests/test_foo.py::test_bar -v  # run a single test
```

CI gate (all must pass before merge): `ruff format --check .`, `ruff check .`, `pyright`, `pytest`

## Architecture

**Entry point:** `src/bot/__main__.py` → `app.run()` → `Dispatcher.start_polling()`

**Configuration:** `src/bot/config.py` — single `Settings(BaseSettings)` instance, reads `.env` file. Imported as `from bot.config import settings`.

**Request flow** (per user message in `handlers.py`):
1. `EpisodeManager.process_user_message()` — persists message, auto-switches episodes on 8h gap or topic shift (cosine similarity < 0.7)
2. `CogneeMemoryService.search()` — vector similarity search over knowledge graph (best-effort, skipped if unavailable)
3. `ContextBuilder.assemble_for_llm()` — merges short-term history (last N messages) + semantic memories + system prompt into an LLM message list
4. `LLMService.generate()` — calls LLM via LangChain `ChatOpenAI` wrapper, returns `LLMResponse` with content + token counts
5. Response persisted to episode, sent back to user

**Service wiring:** module-level singletons via `get_*()` / `set_*()` factory functions (e.g. `get_llm_service()`, `get_episode_manager()`).

**Graceful degradation:** Cognee memory, Supabase DB, and Redis are all optional — the bot falls back to in-memory-only operation when any is missing. Handlers catch service-level exceptions and reply with Russian fallback text.

**Key services** (all in `src/bot/services/`):
| Service | Role |
|---|---|
| `llm_service.py` | LangChain ChatOpenAI wrapper, token tracking |
| `cognee_memory_service.py` | Long-term memory via Cognee knowledge graph + vector search |
| `context_builder.py` | Dual-memory (short-term + semantic) prompt assembly |
| `episode_manager.py` | Episode lifecycle, DB persistence, delegates to `episode_switcher` |
| `db_client.py` | Supabase client for threads/episodes/messages |
| `system_prompt.py` | Bot persona system prompt with user-name personalization |
| `memory_models.py` | Memory category/type enums + `MemoryUnit` data models |

## Critical Rules

<important if="writing async code or service methods">
All service methods MUST be async. Use `ainvoke()` / `astream()` for LangChain, never sync `.invoke()`.
</important>

<important if="writing handlers or touching handlers.py">
The catch-all `@router.message()` handler with NO filter MUST be registered LAST. Insert new handlers ABOVE it. Always include try/except with Russian fallback text.
</important>

<important if="working with Cognee or memory">
All `cognee.add()` calls MUST include `dataset_name=f"tg_user_{user_id}"` for user isolation. Never call `prune_data()`.
</important>

<important if="writing imports from LangChain">
Use split packages only: `langchain_core`, `langchain_openai`. Never `from langchain.` or `from langchain.schema`.
</important>

## Environment

Required env vars (loaded via `.env` by pydantic-settings):

| Variable | Required | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot crashes on startup without it |
| `LLM_BASE_URL` | No | LLM provider endpoint (defaults to OpenAI) |
| `LLM_API_KEY` | No | API key for LLM provider |
| `REDIS_URL` | No | Redis for rate-limiting; falls back gracefully |
| `COGNEE_VECTOR_DB_PROVIDER` | No | Default `lancedb` |
| `COGNEE_GRAPH_DB_PROVIDER` | No | Default `kuzu` |

## Code Style

- Python 3.12+ features encouraged
- Type hints required on public functions
- 100-char line length (ruff config)
- Async-first: all service methods are `async`
- Commits: present tense, concise ("Add feature" not "Added feature")

## Branching

- `main` — production. `develop` — integration. Feature branches: `feature/<name>`, bugfix: `fix/<name>`
- Squash or rebase merge preferred

## Claude Code Tooling

Commands: `/implement`, `/ship`, `/add-feature`, `/check`, `/review`
Agents: `planner`, `memory-specialist`, `llm-pipeline`, `telegram-handler`, `tester`, `reviewer`, `security-reviewer`, `prompt-engineer`
Skills: `add-migration`, `add-service`, `add-handler`, `prompt-review`, `diagnose`
