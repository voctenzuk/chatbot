# tg-virtual-girlfriend

Telegram-бот (aiogram 3.x) с LLM-интеграцией, cognee knowledge graph memory и Supabase.

## Stack

- **Python**: 3.12+
- **Bot framework**: aiogram 3.x
- **LLM**: LangChain-core + LangGraph (модель через OpenAI-compatible API)
- **Memory**: cognee (lancedb vectors + kuzu graph)
- **Database**: Supabase (episodes, messages)
- **Cache**: Redis
- **Config**: pydantic-settings (`.env`)

## Python Toolchain (MANDATORY)

- **ALWAYS** use `uv` — NEVER `pip`, `pip install`, `python -m pip`, or bare `python`
- `uv sync` — install deps from pyproject.toml / uv.lock
- `uv add <pkg>` — add dependency (NOT `pip install`, NOT `uv pip install`)
- `uv run <script>` — run Python scripts
- `uvx <tool>` — one-off CLI tools (ruff, pytest, mypy)
- Applies everywhere: local dev, CI, scripts, agents

## Build & Test

```bash
uv sync                       # install dependencies
uv run pytest                 # run tests
uv run pytest --cov           # tests + coverage
uvx ruff check .              # lint
uvx ruff format .             # format
uvx pyright                   # type check
uv run bot                    # start bot (entry: bot.__main__:main)
```

## Project Structure

```
src/bot/
├── app.py              # Bot + Dispatcher setup
├── config.py           # pydantic-settings (Settings)
├── handlers.py         # aiogram Router: /start + chat
├── webhooks.py         # Optional: FastAPI webhooks (stripe, paddle)
├── __main__.py         # Entry point
└── services/
    ├── llm_service.py          # LLM client (OpenAI-compatible)
    ├── cognee_memory_service.py # Cognee knowledge graph
    ├── context_builder.py       # Assemble system + history + memories
    ├── episode_manager.py       # Episode lifecycle + message persistence
    ├── episode_switcher.py      # Episode switching logic
    ├── db_client.py             # Supabase client
    ├── storage_backend.py       # Abstracted storage
    ├── summarizer.py            # Conversation summarization
    ├── system_prompt.py         # System prompt builder
    ├── artifact_service.py      # Artifacts pipeline
    └── memory_models.py         # Pydantic models for memory
tests/                  # pytest + pytest-asyncio (asyncio_mode=auto)
supabase_migrations/    # SQL migrations
prompts/                # Prompt templates
```

## Behavioral Rules

- Do what was asked; nothing more, nothing less
- NEVER create files unless absolutely necessary — prefer editing existing
- NEVER save files to root — use `src/`, `tests/`, `docs/`, `scripts/`
- ALWAYS read a file before editing it
- NEVER commit secrets, credentials, or .env files
- ALWAYS run `uv run pytest` after code changes
- ALWAYS use Context7 MCP to look up aiogram, langchain, cognee, supabase docs before writing code

## Documentation Lookup

- First: `mcp__context7__resolve-library-id` (e.g., "aiogram", "langchain", "cogneeai", "supabase")
- Then: `mcp__context7__query-docs` with resolved ID
- Priority libs: aiogram 3.x, LangChain/LangGraph, cognee, Supabase Python, Pydantic v2

## Security

- NEVER hardcode API keys or credentials
- NEVER commit `.env`
- Validate user input at system boundaries
- Sanitize file paths

## Git Workflow

- Main branch: `memory/issue-1-schema`
- Feature branches off main
- Pre-commit: ruff (lint + format), trailing whitespace, YAML check, merge conflict check
