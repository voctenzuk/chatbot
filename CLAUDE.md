# CLAUDE.md

Telegram AI companion chatbot: aiogram 3 (async, long-polling), LangChain for LLM orchestration, Cognee for knowledge-graph memory. Russian-language bot personality. Package manager: **uv**.

## Commands

```bash
uv run bot                                    # run the bot
uv sync                                       # install/sync dependencies
uvx ruff check --fix . && uvx ruff format .   # lint + format
uv run pyright                                # type check
uv run pytest                                 # run all tests
uv run pytest tests/test_foo.py::test_bar -v  # run a single test
```

CI gate (all must pass): `ruff format --check .` → `ruff check .` → `pyright` → `pytest`

## Architecture

**Entry point:** `src/bot/__main__.py` → `app.run()` → `Dispatcher.start_polling()`

**Configuration:** `src/bot/config.py` — `Settings(BaseSettings)`, reads `.env`. Import as `from bot.config import settings`.

**Request flow** (per message in `handlers.py`):
1. `EpisodeManager.process_user_message()` — persist + auto-switch episodes (8h gap or topic shift)
2. `CogneeMemoryService.search()` — vector search over knowledge graph (best-effort)
3. `ContextBuilder.assemble_for_llm()` — merge history + semantic memories + system prompt
4. `LLMService.generate()` — LLM call via LangChain, returns `LLMResponse`
5. Response persisted to episode, sent to user

**Wiring:** Composition root in `src/bot/wiring.py`. `build_app_context()` → `AppContext` dataclass. `ChatPipeline` receives deps via constructor. Handlers get `pipeline` + `db_client` via `dp.workflow_data`.

**Graceful degradation:** Cognee, Supabase, Redis are optional — bot falls back to in-memory operation.

## Critical Rules

<important if="writing async code or service methods">
All service methods MUST be async. Use `ainvoke()` / `astream()` for LangChain, never sync `.invoke()`.
</important>

<important if="writing handlers or touching handlers.py">
Catch-all `@router.message()` with NO filter MUST be registered LAST. Insert new handlers ABOVE it. Always try/except with Russian fallback text.
</important>

<important if="working with Cognee or memory">
All `cognee.add()` calls MUST include `dataset_name=f"tg_user_{user_id}"` for user isolation. Never call `prune_data()`.
</important>

<important if="writing imports from LangChain">
Split packages only: `langchain_core`, `langchain_openai`. Never `from langchain.` or `from langchain.schema`.
</important>

<important if="writing type annotations">
Use `str | None` not `Optional[str]`. Never blanket `# type: ignore` — use `# pyright: ignore[reportSpecificRule]` with comment.
</important>

## Code Style

- Python 3.12+, type hints on public functions, 100-char lines
- Async-first: all service methods are `async`
- Logging: `loguru.logger` with `{}` placeholders
- Commits: present tense, prefixed (`feat:`, `fix:`, `chore:`)

Detailed rules: `.claude/rules/python-style.md` (ruff, pyright, imports, patterns)

## Architecture (target)

Refactoring from flat `services/` to domain packages with selective hexagonal ports. See memory for full plan.

Domain packages: `bot.llm`, `bot.memory`, `bot.conversation`, `bot.media`, `bot.infra`, `bot.adapters.telegram`
Protocols: `LLMPort`, `MemoryPort`, `MessageDeliveryPort` — only where swapping is realistic.

## Branching

`main` — production. Feature: `feature/<name>`, bugfix: `fix/<name>`. Squash merge preferred.

## Domain Rules

Loaded via `.claude/rules/` with path-based globs (only when touching relevant files):
- `testing.md` — pytest conventions, fixtures, mocking, markers (`tests/**`)
- `python-style.md` — ruff, pyright, imports, async patterns (`**/*.py`)
- `aiogram.md` — handler conventions, filters, FSM, middleware (`handlers.py`, `app.py`)
- `langchain.md` — imports, async invocation, LLMService pattern (`bot.llm/**`)
- `memory-system.md` — Cognee, episodes, context builder (`bot.memory/**`, `bot.conversation/**`)
