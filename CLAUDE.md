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

**Service wiring:** Composition root in `src/bot/wiring.py`. `build_app_context()` creates all services once at startup, returns `AppContext` dataclass. `ChatPipeline` receives all deps via constructor (keyword-only args). Handlers receive `pipeline` and `db_client` via aiogram `dp.workflow_data` kwargs injection. Individual services still have `get_*()`/`set_*()` for backward compat (P3 TODO to remove).

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

## Testing Rules

<important if="writing or modifying tests">

### Structure
- Flat `tests/test_<module>.py`, one file per module. Group with `class TestFeatureName`
- Shared fixtures in `tests/conftest.py` (mock_db, mock_llm, mock_delivery). File-local fixtures for module-specific needs only
- `asyncio_mode = "auto"` — `@pytest.mark.asyncio` decorators optional but allowed for clarity

### Fixtures
- **Factory functions** (`_make_pipeline(**overrides)`) for objects with variations — preferred pattern
- **`@pytest.fixture(autouse=True)`** for singleton resets. Do NOT use `setup_method/teardown_method`
- `yield` for teardown, never `addfinalizer`. Scope defaults to `function`, widen only if setup is genuinely expensive
- Max fixture chain depth: 2 levels

### Mocking
- Mock at boundaries (LLM API, Supabase, Redis, Cognee), NOT internal methods between own classes
- `AsyncMock` for all async methods, never `MagicMock` for `await`-ed calls
- `spec=RealClass` on mocks to catch attribute typos
- `patch()` at import site, not definition site
- Assert outcomes, NOT call order. `assert_has_calls(..., any_order=False)` is an anti-pattern
- If `mock.a.b.c.return_value` — test is over-coupled, refactor

### Parametrize
- Use when 3+ tests have identical structure with different data. `pytest.param(..., id="name")` for readable output
- Do NOT parametrize if cases need different setup/assertion logic

### Markers
- `@pytest.mark.slow` — tests >2s, skip with `pytest -m "not slow"`
- `@pytest.mark.integration` — tests hitting real external services

### Coverage
- CI gate: 80% line coverage with branch coverage (`pytest --cov=bot --cov-branch`)
- Exclude: `__main__.py`, `if TYPE_CHECKING:` blocks
- Do NOT chase 100% — diminishing returns after 90%

### Anti-patterns — do NOT
- Test private methods (`_internal()`) — test through public API
- Use `asyncio.sleep()` for sync — `await` tasks explicitly or use `asyncio.Event`
- Write tests longer than 30 lines without extracting helpers
- Share mutable state between tests
- Assert broadly (`assert "error" in str(result)`) — assert specific types/messages
</important>

## Branching

- `main` — production. `develop` — integration. Feature branches: `feature/<name>`, bugfix: `fix/<name>`
- Squash or rebase merge preferred

## Target Architecture (Selective Composite)

The project is being refactored from flat `services/` to domain-grouped packages with selective hexagonal ports. See memory for full rationale.

**Target structure:**
```
src/bot/
    ports.py                    # 3 Protocols: LLMPort, MemoryPort, MessageDeliveryPort
    chat_pipeline.py            # Core chat logic extracted from handlers.py
    adapters/telegram/          # Thin aiogram handlers + proactive scheduler
    conversation/               # episode_manager, episode_switcher, context_builder, summarizer, system_prompt
    memory/                     # cognee_service, models, cleanup
    llm/                        # service, models
    media/                      # image_service, artifact_service, storage_backend
    infra/                      # db_client
```

**Principles:**
- Protocol only where swapping is realistic (LLM, Memory, MessageDelivery) — NOT for every service
- Domain grouping for organization, not architectural purity
- Backward-compatible shims during migration (old imports keep working)
- Each refactoring phase = separate branch + PR

## gstack

Use `/browse` from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

### Development Workflow

| Phase | Command | What it does |
|---|---|---|
| **Think** | `/office-hours` | Brainstorm feature, generate design doc |
| **Plan (strategy)** | `/plan-ceo-review` | Challenge scope, find 10-star product |
| **Plan (architecture)** | `/plan-eng-review` | Lock architecture, data flow, edge cases |
| **Plan (design)** | `/plan-design-review` | Rate design dimensions, fix gaps |
| **Build** | `/implement`, `/add-feature` | Orchestrate with domain agents |
| **Refactor** | `/refactor-phase` | One architectural phase per PR |
| **Review** | `/review` | Pre-landing diff review (gstack) |
| **Second opinion** | `/codex` | Independent review via OpenAI Codex |
| **CI gate** | `/check` | ruff format → ruff check → pyright → pytest |
| **Debug** | `/investigate` | Systematic root-cause debugging |
| **Diagnose (bot)** | `diagnose` skill | Bot-specific: config, wiring, handlers, memory |
| **Ship** | `/ship` | Merge base, tests, review, version bump, PR |
| **Docs** | `/document-release` | Sync README/ARCHITECTURE/CHANGELOG post-ship |
| **Retro** | `/retro` | Weekly retrospective with trend tracking |

### Safety Tools

| Command | Use when |
|---|---|
| `/careful` | Touching prod, destructive commands |
| `/freeze` | Restrict edits to one directory |
| `/guard` | `/careful` + `/freeze` combined |
| `/unfreeze` | Remove edit restrictions |

### Project Commands & Skills

Commands: `/implement`, `/add-feature`, `/check`, `/refactor-phase`
Agents: `planner`, `memory-specialist`, `llm-pipeline`, `telegram-handler`, `tester`, `reviewer`, `security-reviewer`, `prompt-engineer`, `refactor-mover`
Skills: `add-migration`, `add-service`, `add-handler`, `prompt-review`, `diagnose`, `gen-test`, `verify-imports`
