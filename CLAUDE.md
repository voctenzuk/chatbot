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
| `episode_switcher.py` | Time-gap and topic-shift detection (numpy cosine similarity) |
| `db_client.py` | Supabase client for threads/episodes/messages |
| `summarizer.py` | Episode summary generation for long-term memory extraction |
| `artifact_service.py` | File uploads + text surrogate management |

## Code Style

- Python 3.12+ features encouraged
- Type hints required on public functions
- 100-char line length (ruff config)
- Async-first: all service methods are `async`
- Commits: present tense, concise ("Add feature" not "Added feature")

## Branching

- `main` — production. `develop` — integration. Feature branches: `feature/<name>`, bugfix: `fix/<name>`
- Squash or rebase merge preferred
